"""HuggingFace dataset sharing for activation datasets.

Two-tier architecture: Parquet index + safetensors tensor store.

**Parquet index** (``index/train-00000-of-00001.parquet``): small, queryable
prompt metadata (text, labels, shard refs).  Works with ``load_dataset()``
and the HF Dataset Viewer.

**Safetensors tensor store** (``tensors/``): large activation tensors stored
as raw contiguous bytes.  Plays well with Xet's content-defined chunking
for byte-level dedup.  Hidden layers use per-layer sharding (v1.1): each
file contains a single layer across a batch of prompts.  Logits shards
are unchanged (no layer axis).

Everything lives in a single HF Dataset repo::

    repo/
      README.md
      lmprobe_info.json                      # provenance + tensor descriptors
      index/
        train-00000-of-00001.parquet         # queryable prompt metadata
      tensors/
        hidden_layer000_shard000.safetensors  # layer 0 for shard 0
        hidden_layer000_shard001.safetensors  # layer 0 for shard 1
        hidden_layer001_shard000.safetensors  # layer 1 for shard 0
        logits_topk_000.safetensors           # topk logits for shard 0
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .cache import (
    CachedPromptInfo,
    _hash_string,
    discover_cached,
    load_prompt_activations,
    load_prompt_logits,
    load_prompt_pooled_activations,
    save_prompt_activations,
    save_prompt_pooled_activations,
    write_shard_registry,
)

logger = logging.getLogger(__name__)

FORMAT_VERSION = "1.2"
DEFAULT_SHARD_BYTES = 1_073_741_824  # 1 GB
INFO_FILENAME = "lmprobe_info.json"
PARQUET_PATH = "index/train-00000-of-00001.parquet"


# =============================================================================
# Dependency checks
# =============================================================================


def _check_hub_deps() -> None:
    """Check that huggingface_hub is installed with sufficient version."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        raise ImportError(
            "Dataset sharing requires huggingface_hub. "
            "Install with: pip install 'huggingface_hub>=0.25.0'"
        )
    from packaging.version import Version

    if Version(huggingface_hub.__version__) < Version("0.25.0"):
        raise ImportError(
            f"Dataset sharing requires huggingface_hub >= 0.25.0, "
            f"found {huggingface_hub.__version__}. "
            f"Upgrade with: pip install --upgrade huggingface_hub"
        )


def _check_pyarrow() -> None:
    """Runtime check for pyarrow (transitive dep of transformers/datasets)."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise ImportError(
            "Parquet index generation requires pyarrow. "
            "Install with: pip install pyarrow"
        )



# =============================================================================
# Discovery helpers (unchanged from v1 — these are already correct)
# =============================================================================


def _discover_prompts(
    model_name: str,
    prompts: list[str],
    *,
    skip_missing: bool = True,
) -> tuple[list[int], list[CachedPromptInfo]]:
    """Discover cache state for all prompts.

    Uses a spot-check optimization: discovers the first prompt fully, then
    spot-checks ~10 evenly-spaced prompts.  The full validation happens during
    consolidation (load failures are caught there).

    Returns
    -------
    kept_indices : list[int]
        Indices into *prompts* for prompts that have cached data.
    infos : list[CachedPromptInfo]
        Corresponding CachedPromptInfo for each kept prompt.
    """
    if not prompts:
        raise ValueError("No prompts provided")

    first_info = discover_cached(model_name, prompts[0])
    if first_info is None:
        if not skip_missing:
            raise FileNotFoundError(
                f"No cached data for first prompt: {prompts[0]!r}"
            )

    # Spot-check ~10 evenly-spaced prompts
    n = len(prompts)
    spot_indices = set()
    if n > 2:
        step = max(1, n // 10)
        spot_indices = {i for i in range(0, n, step)}
        spot_indices.add(n - 1)
    spot_indices.add(0)

    spot_results: dict[int, CachedPromptInfo | None] = {}
    for i in spot_indices:
        spot_results[i] = discover_cached(model_name, prompts[i])

    # Log non-uniformity
    spot_infos = [v for v in spot_results.values() if v is not None]
    if spot_infos and first_info is not None:
        ref_raw = set(first_info.raw_layers)
        ref_pooled_keys = set(first_info.pooled.keys())
        ref_logits = first_info.has_logits
        for si in spot_infos:
            if (
                set(si.raw_layers) != ref_raw
                or set(si.pooled.keys()) != ref_pooled_keys
                or si.has_logits != ref_logits
            ):
                logger.info(
                    "[SHARING] Spot-check found non-uniform cache state "
                    "across prompts. Full discovery will happen during "
                    "consolidation."
                )
                break

    kept_indices = []
    infos = []

    for i, prompt in enumerate(prompts):
        if i in spot_results:
            info = spot_results[i]
        else:
            info = first_info
        if info is None:
            if not skip_missing:
                raise FileNotFoundError(
                    f"No cached data for prompt index {i}: {prompt!r}"
                )
            logger.debug(f"[SHARING] Skipping uncached prompt index {i}")
            continue
        kept_indices.append(i)
        infos.append(info)

    if not kept_indices:
        raise ValueError(
            "No prompts have cached data. Extract activations first."
        )

    return kept_indices, infos


def _compute_tensor_intersection(
    infos: list[CachedPromptInfo],
) -> dict[str, Any]:
    """Compute the intersection of available tensor types across infos.

    Returns a dict with keys:
      - "raw_layers": sorted list of layer indices present in ALL infos
      - "pooled": dict[strategy, sorted layer list] present in ALL infos
      - "has_logits": True only if ALL have logits
      - "logits_top_k": int or None (consistent across all, else None)
      - "has_perplexity": True only if ALL have perplexity
    """
    if not infos:
        return {
            "raw_layers": [],
            "pooled": {},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }

    raw_sets = [set(i.raw_layers) for i in infos]
    raw_intersection = sorted(raw_sets[0].intersection(*raw_sets[1:]))

    all_strategies = [set(i.pooled.keys()) for i in infos]
    common_strategies = all_strategies[0].intersection(*all_strategies[1:])
    pooled: dict[str, list[int]] = {}
    for strategy in common_strategies:
        layer_sets = [set(i.pooled[strategy]) for i in infos]
        common_layers = sorted(layer_sets[0].intersection(*layer_sets[1:]))
        if common_layers:
            pooled[strategy] = common_layers

    # Logits: prefer topk over full
    all_have_topk = all(i.logits_top_k is not None for i in infos)
    all_have_full_logits = all(i.has_logits for i in infos)

    logits_top_k_values = {
        i.logits_top_k for i in infos if i.logits_top_k is not None
    }
    consistent_topk = len(logits_top_k_values) == 1

    if all_have_topk and consistent_topk:
        has_logits = True
        logits_top_k = logits_top_k_values.pop()
    elif all_have_topk and not consistent_topk:
        warnings.warn(
            f"All prompts have topk logits but with different k values: "
            f"{sorted(logits_top_k_values)}. Logits will not be included in "
            f"the push. Re-cache with a consistent top_k to include logits.",
            stacklevel=3,
        )
        has_logits = False
        logits_top_k = None
    elif all_have_full_logits:
        has_logits = True
        logits_top_k = None
    else:
        has_logits = False
        logits_top_k = None

    has_perplexity = all(i.has_perplexity for i in infos)

    return {
        "raw_layers": raw_intersection,
        "pooled": pooled,
        "has_logits": has_logits,
        "logits_top_k": logits_top_k,
        "has_perplexity": has_perplexity,
    }


def _filter_tensor_types(
    available: dict[str, Any],
    tensors_filter: list[str] | None,
) -> dict[str, Any]:
    """Apply user's tensor filter to available types.

    The new filter uses type-level keys: ``"hidden_layers"``, ``"logits_topk"``.
    """
    if tensors_filter is None:
        return available

    result = {
        "raw_layers": [],
        "pooled": {},
        "has_logits": False,
        "logits_top_k": available.get("logits_top_k"),
        "has_perplexity": False,
    }

    for key in tensors_filter:
        if key == "hidden_layers":
            # Include all available pooled layers (co-located)
            result["pooled"] = dict(available.get("pooled", {}))
            result["raw_layers"] = list(available.get("raw_layers", []))
            continue

        if key == "logits_topk" and available["has_logits"]:
            result["has_logits"] = True
            continue

        if key == "perplexity" and available["has_perplexity"]:
            result["has_perplexity"] = True
            continue

        logger.warning(
            f"[SHARING] Unknown tensor filter key: {key!r}, ignoring"
        )

    return result


# =============================================================================
# Deterministic prompt shuffling
# =============================================================================


def _deterministic_seed(repo_id: str) -> int:
    """Derive a deterministic seed from repo_id for prompt shuffling."""
    h = hashlib.sha256(repo_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _shuffle_indices(n: int, seed: int) -> list[int]:
    """Return a deterministic permutation of range(n)."""
    import random

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return indices


# =============================================================================
# Strategy dict for consolidation
# =============================================================================


def _load_hidden_for_prompt(
    model_name: str,
    prompt: str,
    layers: list[int],
    pooling_strategy: str,
) -> dict[str, torch.Tensor]:
    """Load pooled hidden states for a prompt, returning per-layer tensors.

    Returns dict like ``{"hidden.layer_0": tensor(1, dim), ...}``.
    """
    result = {}
    for layer in layers:
        # load_prompt_pooled_activations returns (1, hidden_dim) for a single layer
        pooled = load_prompt_pooled_activations(
            model_name, prompt, [layer], pooling_strategy
        )
        # pooled shape: (1, hidden_dim) — one layer
        result[f"hidden.layer_{layer}"] = pooled.reshape(1, -1)
    return result


def _load_logits_for_prompt(
    model_name: str,
    prompt: str,
    top_k: int,
) -> dict[str, torch.Tensor]:
    """Load topk logits for a prompt.

    Returns dict like ``{"logits_topk.values": (1, k), "logits_topk.indices": (1, k)}``.
    """
    values, indices = load_prompt_logits(model_name, prompt, top_k=top_k)
    # values: (1, positions, K), indices: (1, positions, K)
    # For pooled (last_token), we expect shape (1, K) or (1, 1, K)
    return {
        "logits_topk.values": values.reshape(1, -1),
        "logits_topk.indices": indices.reshape(1, -1),
    }


def _load_hidden_raw_for_prompt(
    model_name: str,
    prompt: str,
    layers: list[int],
) -> tuple[dict[str, torch.Tensor], int]:
    """Load raw (full-sequence) hidden states for a prompt.

    Returns
    -------
    tensors : dict[str, Tensor]
        ``{"hidden.layer_0": (num_tokens, dim), ...}``
    num_tokens : int
        Number of non-padding tokens.
    """
    # load_prompt_activations returns (1, seq_len, hidden_dim*n_layers), mask
    activations, mask = load_prompt_activations(model_name, prompt, layers)
    # mask: (1, seq_len)  activations: (1, seq_len, hidden_dim * n_layers)
    n_layers = len(layers)
    total_dim = activations.shape[-1]
    hidden_dim = total_dim // n_layers

    # Remove batch dim and mask out padding
    act = activations.squeeze(0)  # (seq_len, total_dim)
    m = mask.squeeze(0).bool()  # (seq_len,)
    act = act[m]  # (num_tokens, total_dim)
    num_tokens = act.shape[0]

    result = {}
    for i, layer in enumerate(layers):
        start = i * hidden_dim
        end = (i + 1) * hidden_dim
        result[f"hidden.layer_{layer}"] = act[:, start:end]  # (num_tokens, dim)

    return result, num_tokens


# =============================================================================
# Consolidation engine
# =============================================================================


def _compute_shard_boundaries_variable(
    per_prompt_bytes: list[int],
    shard_max_bytes: int,
) -> list[int]:
    """Compute shard boundaries for variable-size prompts.

    Returns a list of prompt counts per shard (same format as
    ``_compute_shard_boundaries``).
    """
    boundaries: list[int] = []
    current_bytes = 0
    current_count = 0

    for pb in per_prompt_bytes:
        if current_count > 0 and current_bytes + pb > shard_max_bytes:
            boundaries.append(current_count)
            current_bytes = 0
            current_count = 0
        current_bytes += pb
        current_count += 1

    if current_count > 0:
        boundaries.append(current_count)

    return boundaries


def _compute_shard_boundaries(
    num_prompts: int,
    row_bytes: int,
    shard_max_bytes: int,
) -> list[int]:
    """Compute how many prompts go in each shard.

    Returns a list of prompt counts per shard.
    """
    if row_bytes <= 0:
        return [num_prompts]

    prompts_per_shard = max(1, shard_max_bytes // row_bytes)
    boundaries = []
    remaining = num_prompts
    while remaining > 0:
        chunk = min(prompts_per_shard, remaining)
        boundaries.append(chunk)
        remaining -= chunk
    return boundaries


def _consolidate_and_shard(
    model_name: str,
    prompts: list[str],
    kept_indices: list[int],
    tensor_types: dict[str, Any],
    labels: list[int | str | None] | None,
    shard_max_bytes: int,
    repo_id: str,
    metadata: list[dict] | None = None,
) -> tuple[Path, dict, list[dict]]:
    """Consolidate cached tensors into sharded safetensors files.

    Layers are co-located within each hidden_layers shard.  Each tensor type
    gets independent shard boundaries (v1.2), so small logits data isn't
    needlessly split across many shards.

    Uses streaming consolidation: only one shard's worth of tensors is held
    in memory at a time, so peak memory is bounded by ``shard_max_bytes``
    regardless of total dataset size.

    Returns
    -------
    tmpdir : Path
        Temporary directory containing all output files.
    tensor_descriptors : dict
        The "tensors" section of lmprobe_info.json.
    prompt_metadata : list[dict]
        Per-prompt metadata for the Parquet index.
    """
    from safetensors.torch import save_file

    tmpdir = Path(tempfile.mkdtemp(prefix="lmprobe_sharing_"))
    (tmpdir / "tensors").mkdir()
    (tmpdir / "index").mkdir()

    pooled = tensor_types["pooled"]
    has_logits = tensor_types["has_logits"]
    logits_top_k = tensor_types["logits_top_k"]

    # Auto-detect: if raw_layers available, store full-sequence
    raw_layers = tensor_types.get("raw_layers", [])
    use_raw = bool(raw_layers)

    # Determine which pooling strategy and layers we have
    # For hidden_layers, we co-locate all layers from the first available
    # pooling strategy
    hidden_strategy = None
    hidden_layers: list[int] = []
    if not use_raw and pooled:
        hidden_strategy = next(iter(pooled))
        hidden_layers = pooled[hidden_strategy]
    elif use_raw:
        hidden_layers = raw_layers

    # --- Phase 1: Metadata-only pass (no tensor loading) ---
    from tqdm import tqdm

    prompt_metadata: list[dict] = []
    valid_prompts: list[str] = []  # prompts that exist in cache
    per_prompt_tokens: list[int] = []  # only used when use_raw
    skipped_count = 0

    for idx in tqdm(kept_indices, desc="Scanning cache metadata", unit="prompt"):
        prompt = prompts[idx]
        label = labels[idx] if labels is not None else None
        extra_meta: dict[str, Any] = {}
        if metadata is not None:
            extra_meta = dict(metadata[idx])

        info = discover_cached(model_name, prompt)
        if info is None:
            skipped_count += 1
            logger.debug(f"[SHARING] Skipping prompt index {idx}: not in cache")
            continue

        num_tokens = info.num_tokens

        # For raw mode, we need num_tokens to compute shard boundaries
        if use_raw and hidden_layers:
            if num_tokens is None:
                skipped_count += 1
                logger.debug(
                    f"[SHARING] Skipping prompt index {idx}: "
                    f"no num_tokens metadata for raw mode"
                )
                continue
            per_prompt_tokens.append(num_tokens)

        valid_prompts.append(prompt)
        prompt_metadata.append({
            "text": prompt,
            "label": label,
            "num_tokens": num_tokens,
            **extra_meta,
        })

    if skipped_count > 0:
        logger.warning(
            f"[SHARING] Skipped {skipped_count} prompts due to missing or "
            f"corrupt cache entries"
        )

    if not prompt_metadata:
        raise ValueError(
            "No prompts could be loaded from cache. Check that activations "
            "have been extracted."
        )

    # --- Phase 2: Shuffle prompts (deterministic) ---
    seed = _deterministic_seed(repo_id)
    n = len(valid_prompts)
    perm = _shuffle_indices(n, seed)
    valid_prompts = [valid_prompts[i] for i in perm]
    prompt_metadata = [prompt_metadata[i] for i in perm]
    if per_prompt_tokens:
        per_prompt_tokens = [per_prompt_tokens[i] for i in perm]

    # --- Phase 3: Compute shard boundaries ---
    # Probe one prompt for hidden_dim (loads and immediately frees one tensor)
    hidden_dim = 0
    if hidden_layers and valid_prompts:
        sample_prompt = valid_prompts[0]
        if use_raw:
            sample_tensors, _ = _load_hidden_raw_for_prompt(
                model_name, sample_prompt, hidden_layers[:1],
            )
        elif hidden_strategy:
            sample_tensors = _load_hidden_for_prompt(
                model_name, sample_prompt, hidden_layers[:1], hidden_strategy,
            )
        else:
            sample_tensors = {}
        sample_key = f"hidden.layer_{hidden_layers[0]}"
        if sample_key in sample_tensors:
            hidden_dim = sample_tensors[sample_key].shape[-1]
        del sample_tensors

    logits_row_bytes = 0
    if has_logits and logits_top_k is not None:
        # values (float32) + indices (int64)
        logits_row_bytes = logits_top_k * 4 + logits_top_k * 8

    # Compute independent shard boundaries per tensor type (v1.2)
    has_hidden = bool(hidden_layers and (hidden_strategy or use_raw))
    want_logits = bool(has_logits and logits_top_k is not None)

    if use_raw:
        # Variable-size rows: bytes depend on per-prompt token count
        per_prompt_bytes = [
            tok * hidden_dim * 4
            for tok in per_prompt_tokens
        ]
        hidden_boundaries = _compute_shard_boundaries_variable(
            per_prompt_bytes, shard_max_bytes,
        )
    else:
        hidden_row_bytes = hidden_dim * 4  # float32, single layer
        hidden_boundaries = _compute_shard_boundaries(
            n, hidden_row_bytes, shard_max_bytes,
        ) if has_hidden else []

    if want_logits:
        logits_boundaries = _compute_shard_boundaries(
            n, max(logits_row_bytes, 1), shard_max_bytes,
        )
    else:
        logits_boundaries = []

    # --- Phase 4: Stream shards per tensor type ---
    tensor_descriptors: dict[str, dict] = {}

    # --- Hidden pass ---
    if has_hidden and hidden_boundaries:
        # Phase 4a: Assign shard metadata (no data loading yet)
        offset = 0
        token_offset_acc = 0
        for shard_idx, shard_size in enumerate(hidden_boundaries):
            for local_row in range(shard_size):
                global_row = offset + local_row
                if global_row < len(prompt_metadata):
                    prompt_metadata[global_row]["shard_index_hidden"] = shard_idx
                    prompt_metadata[global_row]["row_offset_hidden"] = local_row
                    # Legacy aliases (point to hidden by default)
                    prompt_metadata[global_row]["shard_index"] = shard_idx
                    prompt_metadata[global_row]["row_offset"] = local_row
                    if use_raw:
                        prompt_metadata[global_row]["token_offset_hidden"] = token_offset_acc
                        prompt_metadata[global_row]["token_offset"] = token_offset_acc
                        token_offset_acc += per_prompt_tokens[global_row]
            offset += shard_size

        # Phase 4b: Write per-layer shard files one layer at a time.
        # Loop order: layer → shard, so peak RAM = one shard's worth of one layer
        # (not all layers × all prompts as the old order required).
        total_layer_shards = len(hidden_layers) * len(hidden_boundaries)
        with tqdm(
            total=total_layer_shards, desc="Writing hidden shards", unit="shard"
        ) as pbar:
            for layer in hidden_layers:
                key = f"hidden.layer_{layer}"
                offset = 0
                for shard_idx, shard_size in enumerate(hidden_boundaries):
                    shard_prompts_text = valid_prompts[offset : offset + shard_size]
                    rows: list[torch.Tensor] = []
                    for prompt in shard_prompts_text:
                        try:
                            if use_raw:
                                layer_tensors, _ = _load_hidden_raw_for_prompt(
                                    model_name, prompt, [layer],
                                )
                            elif hidden_strategy:
                                layer_tensors = _load_hidden_for_prompt(
                                    model_name, prompt, [layer], hidden_strategy,
                                )
                            else:
                                layer_tensors = {}
                            if key in layer_tensors:
                                rows.append(layer_tensors[key])
                        except (FileNotFoundError, KeyError, OSError) as e:
                            raise OSError(
                                f"Prompt passed metadata scan but failed to load during "
                                f"shard write (cache may have been modified concurrently): "
                                f"{e}"
                            ) from e
                    if rows:
                        layer_tensor = {key: torch.cat(rows, dim=0)}
                        fname = (
                            f"tensors/hidden_layer{layer:03d}"
                            f"_shard{shard_idx:03d}.safetensors"
                        )
                        save_file(layer_tensor, str(tmpdir / fname))
                        del layer_tensor
                    del rows
                    offset += shard_size
                    pbar.update(1)

    # --- Logits pass ---
    if want_logits and logits_boundaries:
        offset = 0
        for shard_idx, shard_size in enumerate(tqdm(
            logits_boundaries, desc="Writing logits shards", unit="shard",
        )):
            shard_prompts_text = valid_prompts[offset : offset + shard_size]

            # Assign per-type shard metadata for logits
            for local_row in range(shard_size):
                global_row = offset + local_row
                if global_row < len(prompt_metadata):
                    prompt_metadata[global_row]["shard_index_logits"] = shard_idx
                    prompt_metadata[global_row]["row_offset_logits"] = local_row
                    # If no hidden, legacy columns point to logits
                    if not has_hidden:
                        prompt_metadata[global_row]["shard_index"] = shard_idx
                        prompt_metadata[global_row]["row_offset"] = local_row

            # Load and write logits tensors for this shard
            shard_data_logits: list[dict] = []
            for prompt in shard_prompts_text:
                try:
                    loaded_l: dict[str, torch.Tensor] = {}
                    if logits_top_k is not None:
                        logit_tensors = _load_logits_for_prompt(
                            model_name, prompt, logits_top_k,
                        )
                        loaded_l.update(logit_tensors)
                    shard_data_logits.append(loaded_l)
                except (FileNotFoundError, KeyError, OSError) as e:
                    raise OSError(
                        f"Prompt passed metadata scan but failed to load during "
                        f"shard write (cache may have been modified concurrently): "
                        f"{e}"
                    ) from e

            vals = [
                p["logits_topk.values"]
                for p in shard_data_logits
                if "logits_topk.values" in p
            ]
            idxs = [
                p["logits_topk.indices"]
                for p in shard_data_logits
                if "logits_topk.indices" in p
            ]
            if vals and idxs:
                logits_tensors_out = {
                    "logits_topk.values": torch.cat(vals, dim=0),
                    "logits_topk.indices": torch.cat(idxs, dim=0),
                }
                fname = f"tensors/logits_topk_{shard_idx:03d}.safetensors"
                save_file(logits_tensors_out, str(tmpdir / fname))
                del logits_tensors_out

            del shard_data_logits
            offset += shard_size

    # Build tensor descriptors for lmprobe_info.json
    if has_hidden and hidden_boundaries:
        hidden_shards = []
        off = 0
        for si, sz in enumerate(hidden_boundaries):
            actual = min(sz, n - off)
            shard_desc: dict[str, Any] = {
                "num_prompts": actual,
            }
            if use_raw:
                shard_desc["num_tokens"] = sum(
                    per_prompt_tokens[off : off + actual]
                )
            hidden_shards.append(shard_desc)
            off += actual

        hidden_desc: dict[str, Any] = {
            "type": "hidden",
            "layers": hidden_layers,
            "dim": hidden_dim,
            "dtype": "float32",
            "layout": "per_layer",
            "shards": hidden_shards,
        }
        if use_raw:
            hidden_desc["storage"] = "full_sequence"
        else:
            hidden_desc["storage"] = "pooled"
            hidden_desc["pooling"] = hidden_strategy
            hidden_desc["row_bytes"] = hidden_dim * 4

        tensor_descriptors["hidden_layers"] = hidden_desc

    if want_logits and logits_boundaries:
        logits_shards = []
        off = 0
        for si, sz in enumerate(logits_boundaries):
            actual = min(sz, n - off)
            logits_shards.append({
                "file": f"tensors/logits_topk_{si:03d}.safetensors",
                "num_prompts": actual,
            })
            off += actual

        tensor_descriptors["logits_topk"] = {
            "type": "logits_topk",
            "k": logits_top_k,
            "dtype": "float32",
            "pooling": "last_token",
            "row_bytes": logits_row_bytes,
            "shards": logits_shards,
        }

    return tmpdir, tensor_descriptors, prompt_metadata


# =============================================================================
# Parquet index
# =============================================================================


def _write_parquet_index(
    tmpdir: Path,
    prompt_metadata: list[dict],
) -> None:
    """Write the Parquet index from prompt metadata.

    Fixed columns: text, label, num_tokens, shard_index, row_offset (or
    token_offset for full-sequence storage).  Any additional keys in
    prompt_metadata are written as extra columns with auto-inferred types.
    """
    _check_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    texts = [p["text"] for p in prompt_metadata]
    labels = [p.get("label") for p in prompt_metadata]
    num_tokens = [p.get("num_tokens") for p in prompt_metadata]
    shard_indices = [p.get("shard_index", 0) for p in prompt_metadata]
    row_offsets = [p.get("row_offset", 0) for p in prompt_metadata]

    # Determine label type
    if all(isinstance(val, int) for val in labels if val is not None):
        label_array = pa.array(labels, type=pa.int32())
    else:
        label_array = pa.array(
            [str(val) if val is not None else None for val in labels],
            type=pa.string(),
        )

    columns: dict[str, pa.Array] = {
        "text": pa.array(texts, type=pa.string()),
        "label": label_array,
        "num_tokens": pa.array(num_tokens, type=pa.int32()),
        "shard_index": pa.array(shard_indices, type=pa.int32()),
        "row_offset": pa.array(row_offsets, type=pa.int32()),
    }

    # Per-type shard columns (v1.2)
    per_type_keys = {
        "shard_index_hidden", "row_offset_hidden", "token_offset_hidden",
        "shard_index_logits", "row_offset_logits",
    }
    for ptk in per_type_keys:
        if any(ptk in p for p in prompt_metadata):
            vals = [p.get(ptk, 0) for p in prompt_metadata]
            if "token_offset" in ptk:
                columns[ptk] = pa.array(vals, type=pa.int64())
            else:
                columns[ptk] = pa.array(vals, type=pa.int32())

    # token_offset is a fixed column (int64 for large shards) when present
    fixed_keys = {
        "text", "label", "num_tokens", "shard_index", "row_offset",
        "token_offset",
    } | per_type_keys
    if any("token_offset" in p for p in prompt_metadata):
        token_offsets = [p.get("token_offset", 0) for p in prompt_metadata]
        columns["token_offset"] = pa.array(token_offsets, type=pa.int64())
    if prompt_metadata:
        extra_keys = sorted(
            set(prompt_metadata[0].keys()) - fixed_keys
        )
        for ek in extra_keys:
            values = [p.get(ek) for p in prompt_metadata]
            # Infer pyarrow type from first non-None value
            sample = next((v for v in values if v is not None), None)
            if sample is None:
                pa_type = pa.string()
            elif isinstance(sample, int):
                pa_type = pa.int32()
            elif isinstance(sample, float):
                pa_type = pa.float64()
            else:
                pa_type = pa.string()
                values = [
                    str(v) if v is not None else None for v in values
                ]
            columns[ek] = pa.array(values, type=pa_type)

    table = pa.table(columns)
    pq.write_table(table, str(tmpdir / PARQUET_PATH))


# =============================================================================
# Metadata builders
# =============================================================================


def _build_lmprobe_info(
    model_name: str,
    num_prompts: int,
    tensor_descriptors: dict,
) -> dict:
    """Build lmprobe_info.json contents."""
    import platform

    import torch
    import transformers

    from . import __version__

    revision = None
    try:
        from huggingface_hub import model_info

        info = model_info(model_name)
        revision = info.sha
    except Exception:
        pass

    nnsight_version = None
    try:
        import nnsight

        nnsight_version = nnsight.__version__
    except Exception:
        pass

    return {
        "format_version": FORMAT_VERSION,
        "model": {
            "name": model_name,
            "revision": revision,
        },
        "num_prompts": num_prompts,
        "prompt_ordering": "random",
        "tensors": tensor_descriptors,
        "provenance": {
            "lmprobe_version": __version__,
            "extraction_backend": "local",
            "nnsight_version": nnsight_version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "python_version": platform.python_version(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def _build_readme(
    model_name: str,
    lmprobe_info: dict,
    num_prompts: int,
    repo_id: str,
    description: str | None = None,
    license: str = "cc-by-4.0",
) -> str:
    """Generate README.md with YAML frontmatter for the dataset repo."""
    revision = lmprobe_info["model"].get("revision") or "unknown"
    format_version = lmprobe_info["format_version"]
    provenance = lmprobe_info.get("provenance", {})
    tensor_descriptors = lmprobe_info.get("tensors", {})

    model_slug = model_name.replace("/", "-").lower()

    # Build tensor table
    table_rows = []
    for key, info in sorted(tensor_descriptors.items()):
        if key == "hidden_layers":
            layers = info.get("layers", [])
            layers_str = (
                f"{layers[0]}-{layers[-1]}" if len(layers) > 1
                else str(layers[0]) if layers else "-"
            )
            dim = info.get("dim", "-")
            pooling = info.get("pooling", "-")
            n_shards = len(info.get("shards", []))
            row_bytes = info.get("row_bytes", "-")
            table_rows.append(
                f"| {key} | {layers_str} | {dim} | {pooling} "
                f"| {n_shards} | {row_bytes} |"
            )
        elif key == "logits_topk":
            k = info.get("k", "-")
            pooling = info.get("pooling", "-")
            n_shards = len(info.get("shards", []))
            row_bytes = info.get("row_bytes", "-")
            table_rows.append(
                f"| {key} | - | k={k} | {pooling} "
                f"| {n_shards} | {row_bytes} |"
            )

    tensor_table = (
        "\n".join(table_rows) if table_rows
        else "| (none) | - | - | - | - | - |"
    )

    desc_section = f"\n{description}\n" if description else ""
    model_url = f"https://huggingface.co/{model_name}"

    # Find example shard and detect storage mode
    example_shard = "tensors/hidden_layer000_shard000.safetensors"
    is_full_sequence = False
    for info in tensor_descriptors.values():
        shards = info.get("shards", [])
        layout = info.get("layout")
        if layout == "per_layer" and shards:
            # Derive filename from convention
            layers = info.get("layers", [0])
            example_shard = (
                f"tensors/hidden_layer{layers[0]:03d}_shard000.safetensors"
            )
        elif shards and "file" in shards[0]:
            example_shard = shards[0]["file"]
        if info.get("storage") == "full_sequence":
            is_full_sequence = True
        if shards:
            break

    if is_full_sequence:
        standalone_slice_example = (
            'tok_off, num_tok = row["token_offset"], row["num_tokens"]\n'
            "# Slice full-sequence activations for this prompt\n"
            'prompt_acts = layer_0[tok_off : tok_off + num_tok]  '
            "# (num_tokens, hidden_dim)"
        )
    else:
        standalone_slice_example = (
            'shard_idx, row_offset = row["shard_index"], row["row_offset"]'
        )

    yaml_header = f"""---
tags:
- lmprobe
- activations
- interpretability
- {model_slug}
task_categories:
- feature-extraction
language:
- en
license: {license}
---"""

    readme = f"""{yaml_header}

# {model_name} — Activation Dataset

Cached activations extracted from \
[`{model_name}`]({model_url}) \
(revision `{revision}`).
{desc_section}
## Contents

| Tensor | Layers | Dim | Pooling | Shards | Row Bytes |
|--------|--------|-----|---------|--------|-----------|
{tensor_table}

- **Prompts:** {num_prompts}
- **Format version:** {format_version}

## Load with lmprobe

```python
from lmprobe import pull_dataset, load_activation_dataset

# Option 1: Pull into local cache (enables probe training without \
re-extraction)
pull_dataset("{repo_id}")

# Option 2: Load tensors directly
tensors, info = load_activation_dataset("{repo_id}")
# tensors["hidden.layer_16"].shape => (N, hidden_dim)
```

## Load without lmprobe (standalone)

```python
import json
import pyarrow.parquet as pq
from safetensors import safe_open

# 1. Read the Parquet index
index = pq.read_table("index/train-00000-of-00001.parquet").to_pandas()
print(index.columns)  # text, label, num_tokens, shard_index, row_offset

# 2. Read tensor metadata
with open("{INFO_FILENAME}") as f:
    info = json.load(f)
print(list(info["tensors"].keys()))  # e.g. ["hidden_layers", "logits_topk"]

# 3. Load a shard — per-layer files: hidden_layer{{L:03d}}_shard{{S:03d}}.safetensors
with safe_open("{example_shard}", framework="pt") as f:
    print(f.keys())  # e.g. ["hidden.layer_0"]
    layer_0 = f.get_tensor("hidden.layer_0")

# 4. Map prompt index -> shard row
row = index.iloc[42]
{standalone_slice_example}
```

## Load with HF Datasets

```python
from datasets import load_dataset

# Shows prompt text + labels in Dataset Viewer
ds = load_dataset("{repo_id}")
print(ds["train"][0])  # {{"text": "...", "label": ..., ...}}
```

## Provenance

- **lmprobe version:** \
{provenance.get('lmprobe_version', 'unknown')}
- **Extraction backend:** \
{provenance.get('extraction_backend', 'unknown')}
- **Created:** {provenance.get('created_at', 'unknown')}
- **PyTorch:** {provenance.get('torch_version', 'unknown')}
- **Transformers:** \
{provenance.get('transformers_version', 'unknown')}
"""
    return readme


# =============================================================================
# Public API
# =============================================================================


def push_dataset(
    repo_id: str,
    model_name: str,
    prompts: list[str],
    *,
    labels: list[int | str | None] | None = None,
    metadata: list[dict] | None = None,
    tensors: list[str] | None = None,
    shard_max_bytes: int = DEFAULT_SHARD_BYTES,
    private: bool = False,
    exist_ok: bool = False,
    skip_missing: bool = True,
    description: str | None = None,
    license: str = "cc-by-4.0",
    token: str | None = None,
) -> str:
    """Push cached activations to a HuggingFace Dataset repo.

    Parameters
    ----------
    repo_id : str
        HuggingFace repo ID (e.g. ``"username/my-activations"``).
    model_name : str
        The model whose activations are cached.
    prompts : list[str]
        Prompts to push (must have cached activations).
    labels : list[int | str | None] | None
        Per-prompt labels, positionally aligned with *prompts*.
    metadata : list[dict] | None
        Per-prompt metadata dicts, positionally aligned with *prompts*.
        All dicts must have the same keys.  Values appear as extra columns
        in the Parquet index.
    tensors : list[str] | None
        Filter: only push these tensor types (``["hidden_layers"]``,
        ``["logits_topk"]``).  None pushes all available types.
    shard_max_bytes : int
        Max bytes per shard file. Default 1 GB.
    private : bool
        Create a private repository.
    exist_ok : bool
        If False (default), raise if the repo already exists.
    skip_missing : bool
        If True (default), skip prompts missing from cache.
    description : str | None
        Description for the auto-generated README.
    license : str
        License identifier for the dataset card. Default ``"cc-by-4.0"``.
    token : str | None
        HuggingFace API token.

    Returns
    -------
    str
        URL of the dataset.
    """
    _check_hub_deps()
    _check_pyarrow()

    if labels is not None and len(labels) != len(prompts):
        raise ValueError(
            f"labels length ({len(labels)}) != prompts length ({len(prompts)})"
        )

    if metadata is not None:
        if len(metadata) != len(prompts):
            raise ValueError(
                f"metadata length ({len(metadata)}) != "
                f"prompts length ({len(prompts)})"
            )
        if metadata:
            ref_keys = set(metadata[0].keys())
            for i, m in enumerate(metadata[1:], 1):
                if set(m.keys()) != ref_keys:
                    raise ValueError(
                        f"metadata[{i}] has keys {sorted(m.keys())}, "
                        f"expected {sorted(ref_keys)}"
                    )

    # Step 1: Discover cache state
    logger.info(
        f"[SHARING] Discovering cache for {len(prompts)} prompts..."
    )
    kept_indices, infos = _discover_prompts(
        model_name, prompts, skip_missing=skip_missing,
    )

    if len(kept_indices) < len(prompts):
        logger.warning(
            f"[SHARING] {len(prompts) - len(kept_indices)} prompts missing "
            f"from cache, {len(kept_indices)} will be pushed"
        )

    # Step 2: Compute intersection of tensor types
    available = _compute_tensor_intersection(infos)

    # Step 3: Apply user filter
    tensor_types = _filter_tensor_types(available, tensors)

    has_any = (
        tensor_types["raw_layers"]
        or tensor_types["pooled"]
        or tensor_types["has_logits"]
    )
    if not has_any:
        raise ValueError(
            "No tensor types available to push after filtering. "
            f"Available: raw_layers={available['raw_layers']}, "
            f"pooled={list(available['pooled'].keys())}, "
            f"logits={available['has_logits']}"
        )

    # Step 4: Consolidate and shard
    logger.info("[SHARING] Consolidating cached tensors into shards...")
    tmpdir, tensor_descriptors, prompt_metadata = _consolidate_and_shard(
        model_name=model_name,
        prompts=prompts,
        kept_indices=kept_indices,
        tensor_types=tensor_types,
        labels=labels,
        shard_max_bytes=shard_max_bytes,
        repo_id=repo_id,
        metadata=metadata,
    )

    # Step 5: Write Parquet index
    _write_parquet_index(tmpdir, prompt_metadata)

    # Step 6: Write metadata
    num_prompts = len(prompt_metadata)
    lmprobe_info = _build_lmprobe_info(
        model_name, num_prompts, tensor_descriptors,
    )

    with open(tmpdir / INFO_FILENAME, "w") as f:
        json.dump(lmprobe_info, f, indent=2)

    readme = _build_readme(
        model_name=model_name,
        lmprobe_info=lmprobe_info,
        num_prompts=num_prompts,
        repo_id=repo_id,
        description=description,
        license=license,
    )
    with open(tmpdir / "README.md", "w") as f:
        f.write(readme)

    # Step 7: Upload
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id, exist_ok=exist_ok, private=private, repo_type="dataset",
    )
    total_size = sum(f.stat().st_size for f in tmpdir.rglob("*") if f.is_file())
    logger.info(
        "[SHARING] Uploading dataset (%.2f GB) via upload_large_folder",
        total_size / 1e9,
    )
    api.upload_large_folder(
        repo_id=repo_id,
        folder_path=str(tmpdir),
        repo_type="dataset",
    )
    url = f"https://huggingface.co/datasets/{repo_id}"

    # Cleanup
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)

    logger.info(f"[SHARING] Pushed {num_prompts} prompts to {url}")
    return url


@dataclass
class DatasetMetadata:
    """Metadata from a remote activation dataset.

    Returned by :func:`fetch_dataset_metadata`.  Contains just enough
    information to resolve layers, validate model compatibility, and
    check prompt availability without downloading any tensor data.
    """

    model_name: str
    available_layers: list[int]
    num_prompts: int
    format_version: str
    tensor_descriptors: dict
    prompts: list[str] = field(default_factory=list)


def fetch_dataset_metadata(
    repo_id: str,
    *,
    token: str | None = None,
) -> DatasetMetadata:
    """Fetch lightweight metadata from a remote activation dataset.

    Downloads only ``lmprobe_info.json`` (~KB) and the Parquet index
    (~KB–MB) — no tensor data is transferred.

    Parameters
    ----------
    repo_id : str
        HuggingFace Dataset repo ID (e.g. ``"user/my-activations"``).
    token : str | None
        HuggingFace API token.

    Returns
    -------
    DatasetMetadata
        Parsed metadata including model name, available layers, prompt
        list, and tensor descriptors.
    """
    _check_hub_deps()
    from huggingface_hub import hf_hub_download

    info_path = hf_hub_download(
        repo_id, INFO_FILENAME, repo_type="dataset", token=token,
    )
    parquet_path = hf_hub_download(
        repo_id, PARQUET_PATH, repo_type="dataset", token=token,
    )

    with open(info_path) as f:
        lmprobe_info = json.load(f)

    model_name = lmprobe_info["model"]["name"]
    format_version = lmprobe_info.get("format_version", "1.0")
    tensor_descriptors = lmprobe_info.get("tensors", {})

    # Extract available layers from hidden_layers descriptor
    available_layers: list[int] = []
    hidden_info = tensor_descriptors.get("hidden_layers", {})
    if hidden_info.get("layout") == "per_layer":
        available_layers = hidden_info.get("layers", [])
    else:
        # v1.0 co-located: all layers are available but not enumerated
        # Try to infer from shard keys if possible
        available_layers = hidden_info.get("layers", [])

    # Read prompts from Parquet index
    _check_pyarrow()
    import pyarrow.parquet as pq

    index_table = pq.read_table(parquet_path)
    index = index_table.to_pydict()
    prompts = index.get("text", [])

    return DatasetMetadata(
        model_name=model_name,
        available_layers=available_layers,
        num_prompts=len(prompts),
        format_version=format_version,
        tensor_descriptors=tensor_descriptors,
        prompts=prompts,
    )


def pull_dataset(
    repo_id: str,
    *,
    tensors: list[str] | None = None,
    layers: list[int] | None = None,
    target_prompts: list[str] | None = None,
    overwrite: bool = False,
    token: str | None = None,
    materialize: bool = False,
    num_workers: int = 0,
) -> int:
    """Pull activations from a HuggingFace Dataset repo into local cache.

    By default (``materialize=False``), downloads shard files and builds a
    shard registry so activations can be served lazily on-the-fly without
    unpacking per-prompt files.  Set ``materialize=True`` to unpack per-prompt
    safetensors files (the pre-0.8 behavior).

    Parameters
    ----------
    repo_id : str
        HuggingFace repo ID.
    tensors : list[str] | None
        Only pull these tensor types (``["hidden_layers"]``,
        ``["logits_topk"]``).  None pulls all.
    layers : list[int] | None
        Only download these hidden layers (e.g. ``[0, 1]``).  Requires
        per-layer layout (v1.1).  On v1.0 co-located datasets, this
        parameter is ignored with a warning.  None downloads all layers.
    target_prompts : list[str] | None
        Only pull these prompts.  None pulls all.
    overwrite : bool
        If False (default), skip prompts already in local cache.
    token : str | None
        HuggingFace API token.
    materialize : bool
        If True, unpack per-prompt safetensors files into the cache (old
        behavior).  If False (default), only build the shard registry for
        lazy on-the-fly loading.
    num_workers : int
        Number of parallel workers for materialization.  0 (default) runs
        in the main process.  Only used when ``materialize=True``.

    Returns
    -------
    int
        Number of prompts available (lazy) or unpacked (materialize).
    """
    _check_hub_deps()
    from huggingface_hub import hf_hub_download

    # Download metadata
    info_path = hf_hub_download(
        repo_id, INFO_FILENAME, repo_type="dataset", token=token,
    )
    parquet_path = hf_hub_download(
        repo_id, PARQUET_PATH, repo_type="dataset", token=token,
    )

    with open(info_path) as f:
        lmprobe_info = json.load(f)

    # Version check
    remote_version = lmprobe_info.get("format_version", "1.0")
    remote_major = int(remote_version.split(".")[0])
    local_major = int(FORMAT_VERSION.split(".")[0])
    if remote_major != local_major:
        raise ValueError(
            f"Incompatible format version: remote has {remote_version}, "
            f"lmprobe supports {FORMAT_VERSION}. "
            f"Please upgrade lmprobe: pip install --upgrade lmprobe"
        )
    remote_minor = int(remote_version.split(".")[1])
    local_minor = int(FORMAT_VERSION.split(".")[1])
    if remote_minor > local_minor:
        warnings.warn(
            f"Remote format {remote_version} is newer than supported "
            f"{FORMAT_VERSION}. Some tensor types may be skipped.",
            stacklevel=2,
        )

    model_name = lmprobe_info["model"]["name"]
    tensor_descriptors = lmprobe_info.get("tensors", {})

    # Read Parquet index
    _check_pyarrow()
    import pyarrow.parquet as pq

    index_table = pq.read_table(parquet_path)
    index = index_table.to_pydict()
    n_prompts = len(index["text"])

    # Filter prompts
    if target_prompts is not None:
        target_set = set(target_prompts)
        prompt_indices = [
            i for i in range(n_prompts) if index["text"][i] in target_set
        ]
    else:
        prompt_indices = list(range(n_prompts))

    # Dedup: skip already-cached
    if not overwrite:
        new_indices = []
        for i in prompt_indices:
            existing = discover_cached(model_name, index["text"][i])
            if existing is None:
                new_indices.append(i)
        skipped = len(prompt_indices) - len(new_indices)
        if skipped > 0:
            logger.info(
                f"[SHARING] Skipping {skipped} prompts already in local cache"
            )
        prompt_indices = new_indices

    if not prompt_indices:
        logger.info("[SHARING] All prompts already cached locally")
        return 0

    # Determine tensor types to pull
    if tensors is not None:
        pull_types = [k for k in tensor_descriptors if k in tensors]
    else:
        pull_types = list(tensor_descriptors.keys())

    # Figure out which shards we need (v1.2: per-type shard indices)
    needed_shards: dict[str, set[int]] = {}
    for i in prompt_indices:
        for t_type in pull_types:
            if t_type == "logits_topk" and "shard_index_logits" in index:
                si = index["shard_index_logits"][i]
            elif t_type == "hidden_layers" and "shard_index_hidden" in index:
                si = index["shard_index_hidden"][i]
            else:
                # Legacy v1.1 fallback: single shard_index for all types
                si = index["shard_index"][i]
            needed_shards.setdefault(t_type, set()).add(si)

    # Download shard files and record local paths
    # For per-layer layout (v1.1), we download per-layer files
    # For co-located layout (v1.0), we download the single shard file
    shard_local_paths: dict[str, dict[int, str]] = {}  # t_type -> shard_idx -> path
    # Per-layer: t_type -> shard_idx -> {layer: path}
    per_layer_paths: dict[str, dict[int, dict[int, str]]] = {}

    for t_type in pull_types:
        t_info = tensor_descriptors[t_type]
        shards = t_info["shards"]
        layout = t_info.get("layout")

        if layout == "per_layer":
            # v1.1 per-layer layout: derive filenames
            all_layers = t_info.get("layers", [])
            download_layers = all_layers
            if layers is not None:
                download_layers = [ly for ly in all_layers if ly in layers]

            per_layer_paths[t_type] = {}
            for shard_idx in sorted(needed_shards.get(t_type, [])):
                if shard_idx >= len(shards):
                    continue
                per_layer_paths[t_type][shard_idx] = {}
                for layer in download_layers:
                    fname = (
                        f"tensors/hidden_layer{layer:03d}"
                        f"_shard{shard_idx:03d}.safetensors"
                    )
                    shard_path = hf_hub_download(
                        repo_id, fname,
                        repo_type="dataset", token=token,
                    )
                    per_layer_paths[t_type][shard_idx][layer] = shard_path
        else:
            # v1.0 co-located layout
            if layers is not None:
                warnings.warn(
                    "layers parameter is ignored for v1.0 co-located datasets "
                    "(all layers are in the same file)",
                    stacklevel=2,
                )
            shard_local_paths[t_type] = {}
            for shard_idx in sorted(needed_shards.get(t_type, [])):
                if shard_idx >= len(shards):
                    continue
                shard = shards[shard_idx]
                shard_path = hf_hub_download(
                    repo_id, shard["file"],
                    repo_type="dataset", token=token,
                )
                shard_local_paths[t_type][shard_idx] = shard_path

    # ---- Build shard registry (manifest + index) ----
    # Manifest: tensor descriptors with local shard paths
    manifest_tensors = {}
    for t_type in pull_types:
        t_info = tensor_descriptors[t_type]
        layout = t_info.get("layout")
        shards_with_paths = []
        for si, shard_meta in enumerate(t_info["shards"]):
            entry = dict(shard_meta)
            if layout == "per_layer":
                # Store per-layer local paths (use string keys for JSON compat)
                if si in per_layer_paths.get(t_type, {}):
                    entry["per_layer_paths"] = {
                        str(k): v
                        for k, v in per_layer_paths[t_type][si].items()
                    }
            else:
                if si in shard_local_paths.get(t_type, {}):
                    entry["local_path"] = shard_local_paths[t_type][si]
            shards_with_paths.append(entry)
        manifest_tensors[t_type] = {
            **t_info,
            "shards": shards_with_paths,
        }

    manifest = {
        "model_name": model_name,
        "tensors": manifest_tensors,
    }

    # Shard index: prompt_hash -> offset info
    # Load existing shard index to merge (supports multiple repos)
    from .cache import _load_shard_index
    existing_index = _load_shard_index(model_name) or {}

    shard_index = dict(existing_index)
    for i in prompt_indices:
        prompt_text = index["text"][i]
        prompt_hash = _hash_string(prompt_text)
        entry: dict[str, Any] = {
            "shard_index": index["shard_index"][i],
            "row_offset": index["row_offset"][i],
            "num_tokens": index["num_tokens"][i],
        }
        if "token_offset" in index:
            entry["token_offset"] = index["token_offset"][i]
        # Per-type fields (v1.2)
        for col in (
            "shard_index_hidden", "row_offset_hidden", "token_offset_hidden",
            "shard_index_logits", "row_offset_logits",
        ):
            if col in index:
                entry[col] = index[col][i]
        shard_index[prompt_hash] = entry

    write_shard_registry(model_name, manifest, shard_index)

    total_prompts = len(prompt_indices)

    if not materialize:
        logger.info(
            f"[SHARING] Registered {total_prompts} prompts in shard registry "
            f"(lazy mode, no per-prompt files)"
        )
        return total_prompts

    # ---- Materialize: unpack per-prompt files (old behavior) ----
    _materialize_prompts(
        model_name=model_name,
        tensor_descriptors=tensor_descriptors,
        pull_types=pull_types,
        index=index,
        prompt_indices=prompt_indices,
        needed_shards=needed_shards,
        shard_local_paths=shard_local_paths,
        per_layer_paths=per_layer_paths,
        num_workers=num_workers,
    )

    logger.info(f"[SHARING] Unpacked {total_prompts} prompts into local cache")
    return total_prompts


def _unpack_shard_prompts(
    model_name: str,
    t_type: str,
    t_info: dict,
    shard_path: str | dict[int, str],
    shard_prompts: list[int],
    index: dict,
) -> None:
    """Unpack per-prompt files from a single shard.

    This is the core materialization logic, extracted so it can be called
    from the main process or from a worker process.

    Parameters
    ----------
    shard_path : str | dict[int, str]
        For v1.0 co-located layout: path to the single shard file.
        For v1.1 per-layer layout: dict mapping layer index to file path.
    """
    from safetensors import safe_open
    from tqdm import tqdm

    if isinstance(shard_path, dict):
        # Per-layer layout: load each per-layer file
        tensors_data: dict[str, torch.Tensor] = {}
        for layer, layer_path in shard_path.items():
            with safe_open(layer_path, framework="pt") as f:
                for k in f.keys():
                    tensors_data[k] = f.get_tensor(k)
        sf_keys = list(tensors_data.keys())
    else:
        with safe_open(shard_path, framework="pt") as f:
            sf_keys = list(f.keys())
            tensors_data = {k: f.get_tensor(k) for k in sf_keys}

    storage = t_info.get("storage", "pooled")

    for pi in tqdm(
        shard_prompts,
        desc=f"Unpacking {t_type}",
        unit="prompt",
        leave=False,
    ):
        prompt_text = index["text"][pi]

        if t_type == "hidden_layers" and storage == "full_sequence":
            tok_off = index["token_offset"][pi]
            num_tok = index["num_tokens"][pi]
            hidden_dim = t_info.get("dim", 0)

            layer_slices = []
            for sf_key in sf_keys:
                match = re.match(r"^hidden\.layer_(\d+)$", sf_key)
                if not match:
                    continue
                layer = int(match.group(1))
                chunk = tensors_data[sf_key][tok_off : tok_off + num_tok]
                layer_slices.append((layer, chunk))

            layer_slices.sort(key=lambda x: x[0])
            sorted_layers = [ls[0] for ls in layer_slices]
            raw_act = torch.cat(
                [ls[1] for ls in layer_slices], dim=-1
            )
            raw_act = raw_act.unsqueeze(0)
            mask = torch.ones(1, num_tok, dtype=torch.long)

            save_prompt_activations(
                model_name, prompt_text,
                sorted_layers, raw_act, mask,
            )

            for layer_idx, layer_num in enumerate(sorted_layers):
                start = layer_idx * hidden_dim
                end = (layer_idx + 1) * hidden_dim
                last_tok = raw_act[0, -1, start:end].unsqueeze(0)
                save_prompt_pooled_activations(
                    model_name, prompt_text,
                    [layer_num], last_tok, "last_token",
                )

        elif t_type == "hidden_layers":
            row_offset = index["row_offset"][pi]
            pooling = t_info.get("pooling", "last_token")
            for sf_key in sf_keys:
                m = re.match(r"^hidden\.layer_(\d+)$", sf_key)
                if not m:
                    continue
                layer = int(m.group(1))
                row = tensors_data[sf_key][
                    row_offset : row_offset + 1
                ]
                save_prompt_pooled_activations(
                    model_name, prompt_text,
                    [layer], row, pooling,
                )

        elif t_type == "logits_topk":
            row_offset = index["row_offset"][pi]
            v_row = tensors_data["logits_topk.values"][
                row_offset : row_offset + 1
            ]
            i_row = tensors_data["logits_topk.indices"][
                row_offset : row_offset + 1
            ]
            from .cache import (
                _LOGITS_TOP_K_INDICES_KEY,
                _LOGITS_TOP_K_VALUES_KEY,
                _merge_save_backend,
                _prepare_tensor,
                _prompt_cache_key,
                _register_model,
            )

            _register_model(model_name)
            cache_key = _prompt_cache_key(model_name, prompt_text)
            new_tensors = {
                _LOGITS_TOP_K_VALUES_KEY: _prepare_tensor(v_row),
                _LOGITS_TOP_K_INDICES_KEY: _prepare_tensor(i_row),
            }
            _merge_save_backend(cache_key, new_tensors)


def _materialize_prompts(
    *,
    model_name: str,
    tensor_descriptors: dict,
    pull_types: list[str],
    index: dict,
    prompt_indices: list[int],
    needed_shards: dict[str, set[int]],
    shard_local_paths: dict[str, dict[int, str]],
    per_layer_paths: dict[str, dict[int, dict[int, str]]] | None = None,
    num_workers: int = 0,
) -> None:
    """Unpack per-prompt files from shard files.

    If num_workers > 0, uses ProcessPoolExecutor for parallel unpacking.
    """
    if per_layer_paths is None:
        per_layer_paths = {}

    # Collect all (t_type, shard_idx, shard_path, shard_prompts) jobs
    jobs = []
    for t_type in pull_types:
        t_info = tensor_descriptors[t_type]
        layout = t_info.get("layout")

        for shard_idx in sorted(needed_shards.get(t_type, [])):
            shard_prompts = [
                i for i in prompt_indices
                if index["shard_index"][i] == shard_idx
            ]
            if not shard_prompts:
                continue

            if layout == "per_layer":
                # Per-layer: pass dict of layer paths as shard_path
                layer_paths = per_layer_paths.get(t_type, {}).get(shard_idx)
                if not layer_paths:
                    continue
                jobs.append((t_type, t_info, layer_paths, shard_prompts))
            else:
                if shard_idx not in shard_local_paths.get(t_type, {}):
                    continue
                shard_path = shard_local_paths[t_type][shard_idx]
                jobs.append((t_type, t_info, shard_path, shard_prompts))

    if num_workers > 0:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for t_type, t_info, shard_path, shard_prompts in jobs:
                futures.append(
                    executor.submit(
                        _unpack_shard_prompts,
                        model_name, t_type, t_info,
                        shard_path, shard_prompts, index,
                    )
                )
            for fut in futures:
                fut.result()  # Raise any exceptions
    else:
        for t_type, t_info, shard_path, shard_prompts in jobs:
            _unpack_shard_prompts(
                model_name, t_type, t_info,
                shard_path, shard_prompts, index,
            )


def load_activation_dataset(
    repo_id: str,
    *,
    tensors: list[str] | None = None,
    layers: list[int] | None = None,
    token: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Load tensors directly from a HuggingFace Dataset repo.

    Downloads and concatenates shards, returns raw tensors and metadata.
    No local cache interaction.

    Parameters
    ----------
    repo_id : str
        HuggingFace repo ID.
    tensors : list[str] | None
        Only load these tensor types (``["hidden_layers"]``,
        ``["logits_topk"]``).  None loads all.
    layers : list[int] | None
        Only download these hidden layers.  Requires per-layer layout
        (v1.1).  On v1.0 co-located datasets, ignored with a warning.
        None downloads all layers.
    token : str | None
        HuggingFace API token.

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict]
        (tensors_by_key, lmprobe_info) where tensors_by_key maps safetensors
        keys (e.g. ``"hidden.layer_0"``) to concatenated tensors.
    """
    _check_hub_deps()
    from huggingface_hub import hf_hub_download

    info_path = hf_hub_download(
        repo_id, INFO_FILENAME, repo_type="dataset", token=token,
    )

    with open(info_path) as f:
        lmprobe_info = json.load(f)

    # Version check
    remote_version = lmprobe_info.get("format_version", "1.0")
    remote_major = int(remote_version.split(".")[0])
    local_major = int(FORMAT_VERSION.split(".")[0])
    if remote_major != local_major:
        raise ValueError(
            f"Incompatible format version: remote has {remote_version}, "
            f"lmprobe supports {FORMAT_VERSION}. "
            f"Please upgrade lmprobe: pip install --upgrade lmprobe"
        )

    tensor_descriptors = lmprobe_info.get("tensors", {})

    if tensors is not None:
        load_types = [k for k in tensor_descriptors if k in tensors]
    else:
        load_types = list(tensor_descriptors.keys())

    from safetensors import safe_open

    result: dict[str, torch.Tensor] = {}

    for t_type in load_types:
        t_info = tensor_descriptors[t_type]
        shards = t_info["shards"]
        layout = t_info.get("layout")

        shard_data: dict[str, list[torch.Tensor]] = {}

        if layout == "per_layer":
            # v1.1 per-layer: derive filenames
            all_layers = t_info.get("layers", [])
            download_layers = all_layers
            if layers is not None:
                download_layers = [ly for ly in all_layers if ly in layers]

            for shard_idx, _shard in enumerate(shards):
                for layer in download_layers:
                    fname = (
                        f"tensors/hidden_layer{layer:03d}"
                        f"_shard{shard_idx:03d}.safetensors"
                    )
                    shard_path = hf_hub_download(
                        repo_id, fname,
                        repo_type="dataset", token=token,
                    )
                    with safe_open(shard_path, framework="pt") as f:
                        for sf_key in f.keys():
                            shard_data.setdefault(sf_key, []).append(
                                f.get_tensor(sf_key)
                            )
        else:
            # v1.0 co-located layout
            if layers is not None:
                warnings.warn(
                    "layers parameter is ignored for v1.0 co-located "
                    "datasets (all layers are in the same file)",
                    stacklevel=2,
                )
            for shard in shards:
                shard_path = hf_hub_download(
                    repo_id, shard["file"],
                    repo_type="dataset", token=token,
                )
                with safe_open(shard_path, framework="pt") as f:
                    for sf_key in f.keys():
                        shard_data.setdefault(sf_key, []).append(
                            f.get_tensor(sf_key)
                        )

        for sf_key, parts in shard_data.items():
            result[sf_key] = torch.cat(parts, dim=0)

    return result, lmprobe_info
