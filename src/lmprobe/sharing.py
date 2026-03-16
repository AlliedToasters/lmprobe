"""HuggingFace dataset sharing for activation datasets.

Two-tier architecture: Parquet index + safetensors tensor store.

**Parquet index** (``index/train-00000-of-00001.parquet``): small, queryable
prompt metadata (text, labels, shard refs).  Works with ``load_dataset()``
and the HF Dataset Viewer.

**Safetensors tensor store** (``tensors/``): large activation tensors stored
as raw contiguous bytes.  Plays well with Xet's content-defined chunking
for byte-level dedup.  Layers are co-located per shard; tensor types
(hidden vs logits) are in separate files.

Everything lives in a single HF Dataset repo::

    repo/
      README.md
      lmprobe_info.json                      # provenance + tensor descriptors
      index/
        train-00000-of-00001.parquet         # queryable prompt metadata
      tensors/
        hidden_layers_000.safetensors        # all hidden layers for shard 0
        logits_topk_000.safetensors          # topk logits for shard 0
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .cache import (
    CachedPromptInfo,
    discover_cached,
    load_prompt_activations,
    load_prompt_logits,
    load_prompt_pooled_activations,
    save_prompt_activations,
    save_prompt_pooled_activations,
)

logger = logging.getLogger(__name__)

FORMAT_VERSION = "1.0"
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

    Layers are co-located within each hidden_layers shard.  All tensor types
    share the same shard boundaries (same prompt ordering, same num_prompts
    per shard).

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

    # --- Phase 1: Load all prompts, build metadata ---
    from tqdm import tqdm

    prompt_data: list[dict] = []  # per-prompt loaded tensors
    prompt_metadata: list[dict] = []
    per_prompt_tokens: list[int] = []  # only used when use_raw
    skipped_count = 0

    for idx in tqdm(kept_indices, desc="Loading from cache", unit="prompt"):
        prompt = prompts[idx]
        label = labels[idx] if labels is not None else None
        extra_meta: dict[str, Any] = {}
        if metadata is not None:
            extra_meta = dict(metadata[idx])

        try:
            loaded: dict[str, torch.Tensor] = {}

            if use_raw and hidden_layers:
                # Full-sequence path
                layer_tensors, num_tok = _load_hidden_raw_for_prompt(
                    model_name, prompt, hidden_layers,
                )
                loaded.update(layer_tensors)
                per_prompt_tokens.append(num_tok)
            elif hidden_strategy and hidden_layers:
                # Pooled path (existing)
                layer_tensors = _load_hidden_for_prompt(
                    model_name, prompt, hidden_layers, hidden_strategy,
                )
                loaded.update(layer_tensors)

            # Logits
            if has_logits and logits_top_k is not None:
                logit_tensors = _load_logits_for_prompt(
                    model_name, prompt, logits_top_k,
                )
                loaded.update(logit_tensors)

        except (FileNotFoundError, KeyError, OSError) as e:
            skipped_count += 1
            logger.debug(f"[SHARING] Skipping prompt index {idx}: {e}")
            continue

        # Determine num_tokens from first hidden layer tensor
        # (pooled tensors don't carry seq_len, but discover_cached has it)
        info = discover_cached(model_name, prompt)
        num_tokens = info.num_tokens if info else None

        prompt_data.append(loaded)
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
    n = len(prompt_data)
    perm = _shuffle_indices(n, seed)
    prompt_data = [prompt_data[i] for i in perm]
    prompt_metadata = [prompt_metadata[i] for i in perm]
    if per_prompt_tokens:
        per_prompt_tokens = [per_prompt_tokens[i] for i in perm]

    # --- Phase 3: Compute shard boundaries ---
    hidden_dim = 0
    if hidden_layers and prompt_data:
        sample_key = f"hidden.layer_{hidden_layers[0]}"
        if sample_key in prompt_data[0]:
            hidden_dim = prompt_data[0][sample_key].shape[-1]

    logits_row_bytes = 0
    if has_logits and logits_top_k is not None:
        # values (float32) + indices (int64)
        logits_row_bytes = logits_top_k * 4 + logits_top_k * 8

    if use_raw:
        # Variable-size rows: bytes depend on per-prompt token count
        per_prompt_bytes = [
            tok * hidden_dim * 4 * len(hidden_layers)
            for tok in per_prompt_tokens
        ]
        shard_boundaries = _compute_shard_boundaries_variable(
            per_prompt_bytes, shard_max_bytes,
        )
    else:
        hidden_row_bytes = len(hidden_layers) * hidden_dim * 4  # float32
        max_row_bytes = max(hidden_row_bytes, logits_row_bytes, 1)
        shard_boundaries = _compute_shard_boundaries(
            n, max_row_bytes, shard_max_bytes,
        )

    # --- Phase 4: Write safetensors shards ---
    tensor_descriptors: dict[str, dict] = {}
    offset = 0

    # Assign shard_index and row_offset / token_offset to each prompt
    for shard_idx, shard_size in enumerate(shard_boundaries):
        token_offset_acc = 0  # cumulative token offset within shard (raw mode)
        for local_row in range(shard_size):
            global_row = offset + local_row
            if global_row < len(prompt_metadata):
                prompt_metadata[global_row]["shard_index"] = shard_idx
                prompt_metadata[global_row]["row_offset"] = local_row
                if use_raw:
                    prompt_metadata[global_row]["token_offset"] = token_offset_acc
                    token_offset_acc += per_prompt_tokens[global_row]

        shard_prompts = prompt_data[offset : offset + shard_size]

        # Write hidden_layers shard
        if hidden_layers and (hidden_strategy or use_raw):
            shard_tensors = {}
            for layer in hidden_layers:
                key = f"hidden.layer_{layer}"
                rows = [p[key] for p in shard_prompts if key in p]
                if rows:
                    shard_tensors[key] = torch.cat(rows, dim=0)

            if shard_tensors:
                fname = f"tensors/hidden_layers_{shard_idx:03d}.safetensors"
                save_file(shard_tensors, str(tmpdir / fname))

        # Write logits_topk shard
        if has_logits and logits_top_k is not None:
            vals = [
                p["logits_topk.values"]
                for p in shard_prompts
                if "logits_topk.values" in p
            ]
            idxs = [
                p["logits_topk.indices"]
                for p in shard_prompts
                if "logits_topk.indices" in p
            ]
            if vals and idxs:
                logits_tensors = {
                    "logits_topk.values": torch.cat(vals, dim=0),
                    "logits_topk.indices": torch.cat(idxs, dim=0),
                }
                fname = f"tensors/logits_topk_{shard_idx:03d}.safetensors"
                save_file(logits_tensors, str(tmpdir / fname))

        offset += shard_size

    # Build tensor descriptors for lmprobe_info.json
    if hidden_layers and (hidden_strategy or use_raw):
        hidden_shards = []
        off = 0
        for si, sz in enumerate(shard_boundaries):
            actual = min(sz, n - off)
            shard_desc: dict[str, Any] = {
                "file": f"tensors/hidden_layers_{si:03d}.safetensors",
                "num_prompts": actual,
            }
            if use_raw:
                # Total tokens in this shard
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
            "shards": hidden_shards,
        }
        if use_raw:
            hidden_desc["storage"] = "full_sequence"
        else:
            hidden_desc["storage"] = "pooled"
            hidden_desc["pooling"] = hidden_strategy
            hidden_desc["row_bytes"] = len(hidden_layers) * hidden_dim * 4

        tensor_descriptors["hidden_layers"] = hidden_desc

    if has_logits and logits_top_k is not None:
        logits_shards = []
        off = 0
        for si, sz in enumerate(shard_boundaries):
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

    # token_offset is a fixed column (int64 for large shards) when present
    fixed_keys = {
        "text", "label", "num_tokens", "shard_index", "row_offset",
        "token_offset",
    }
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
    example_shard = "tensors/hidden_layers_000.safetensors"
    is_full_sequence = False
    for info in tensor_descriptors.values():
        shards = info.get("shards", [])
        if shards:
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

# 3. Load a shard — layers are co-located, use safetensors partial read
#    for single-layer access.
with safe_open("{example_shard}", framework="pt") as f:
    print(f.keys())  # e.g. ["hidden.layer_0", "hidden.layer_1"]
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
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(tmpdir),
        repo_type="dataset",
        commit_message=(
            f"Upload activation dataset ({num_prompts} prompts)"
        ),
    )
    url = f"https://huggingface.co/datasets/{repo_id}"

    # Cleanup
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)

    logger.info(f"[SHARING] Pushed {num_prompts} prompts to {url}")
    return url


def pull_dataset(
    repo_id: str,
    *,
    tensors: list[str] | None = None,
    target_prompts: list[str] | None = None,
    overwrite: bool = False,
    token: str | None = None,
) -> int:
    """Pull activations from a HuggingFace Dataset repo into local cache.

    Downloads lmprobe_info.json + Parquet index, determines which shards are
    needed, downloads them, and unpacks per-prompt activations into the local
    cache.

    Parameters
    ----------
    repo_id : str
        HuggingFace repo ID.
    tensors : list[str] | None
        Only pull these tensor types (``["hidden_layers"]``,
        ``["logits_topk"]``).  None pulls all.
    target_prompts : list[str] | None
        Only pull these prompts.  None pulls all.
    overwrite : bool
        If False (default), skip prompts already in local cache.
    token : str | None
        HuggingFace API token.

    Returns
    -------
    int
        Number of prompts unpacked into local cache.
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

    # Figure out which shards we need
    needed_shards: dict[str, set[int]] = {}
    for i in prompt_indices:
        shard_idx = index["shard_index"][i]
        for t_type in pull_types:
            needed_shards.setdefault(t_type, set()).add(shard_idx)

    from safetensors import safe_open
    from tqdm import tqdm

    # Build reverse mapping: for each tensor type + shard, which prompts?
    for t_type in pull_types:
        t_info = tensor_descriptors[t_type]
        shards = t_info["shards"]

        for shard_idx in sorted(needed_shards.get(t_type, [])):
            if shard_idx >= len(shards):
                continue
            shard = shards[shard_idx]

            shard_path = hf_hub_download(
                repo_id, shard["file"],
                repo_type="dataset", token=token,
            )

            # Find prompts in this shard
            shard_prompts = [
                i for i in prompt_indices
                if index["shard_index"][i] == shard_idx
            ]

            with safe_open(shard_path, framework="pt") as f:
                sf_keys = list(f.keys())
                tensors_data = {k: f.get_tensor(k) for k in sf_keys}

            storage = t_info.get("storage", "pooled")

            for pi in tqdm(
                shard_prompts,
                desc=f"Unpacking {t_type} shard {shard_idx}",
                unit="prompt",
                leave=False,
            ):
                prompt_text = index["text"][pi]

                if t_type == "hidden_layers" and storage == "full_sequence":
                    # Full-sequence: use token_offset + num_tokens
                    tok_off = index["token_offset"][pi]
                    num_tok = index["num_tokens"][pi]
                    hidden_dim = t_info.get("dim", 0)

                    # Reconstruct raw format: concat layers on dim=-1
                    layer_slices = []
                    for sf_key in sf_keys:
                        match = re.match(r"^hidden\.layer_(\d+)$", sf_key)
                        if not match:
                            continue
                        layer = int(match.group(1))
                        # (num_tok, hidden_dim)
                        chunk = tensors_data[sf_key][tok_off : tok_off + num_tok]
                        layer_slices.append((layer, chunk))

                    layer_slices.sort(key=lambda x: x[0])
                    sorted_layers = [ls[0] for ls in layer_slices]
                    # Concat on dim=-1: (num_tok, hidden_dim * n_layers)
                    raw_act = torch.cat(
                        [ls[1] for ls in layer_slices], dim=-1
                    )
                    # Unsqueeze batch: (1, num_tok, total_dim)
                    raw_act = raw_act.unsqueeze(0)
                    mask = torch.ones(1, num_tok, dtype=torch.long)

                    # Save raw activations
                    save_prompt_activations(
                        model_name, prompt_text,
                        sorted_layers, raw_act, mask,
                    )

                    # Also save pooled (last token) for probe convenience
                    for layer_idx, layer_num in enumerate(sorted_layers):
                        start = layer_idx * hidden_dim
                        end = (layer_idx + 1) * hidden_dim
                        # Last token of this layer
                        last_tok = raw_act[0, -1, start:end].unsqueeze(0)
                        save_prompt_pooled_activations(
                            model_name, prompt_text,
                            [layer_num], last_tok, "last_token",
                        )

                elif t_type == "hidden_layers":
                    # Pooled storage (existing path)
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

    unpacked = len(prompt_indices)
    logger.info(f"[SHARING] Unpacked {unpacked} prompts into local cache")
    return unpacked


def load_activation_dataset(
    repo_id: str,
    *,
    tensors: list[str] | None = None,
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

        shard_data: dict[str, list[torch.Tensor]] = {}

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
