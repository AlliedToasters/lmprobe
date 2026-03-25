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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

import torch

from .cache import (
    CachedPromptInfo,
    _hash_string,
    discover_cached,
    load_prompt_activations,
    load_prompt_logits,
    load_prompt_perplexity,
    load_prompt_pooled_activations,
    load_prompt_token_perplexity,
    save_prompt_activations,
    save_prompt_pooled_activations,
    write_shard_registry,
)

logger = logging.getLogger(__name__)

FORMAT_VERSION = "1.2"
DEFAULT_SHARD_BYTES = 1_073_741_824  # 1 GB
INFO_FILENAME = "lmprobe_info.json"
PARQUET_PATH = "index/train-00000-of-00001.parquet"
_STAGE_SENTINEL = "_stage_complete"
_STREAM_MANIFEST = "_stream_manifest.json"
_tokenizer_cache: dict[str, Any] = {}


# =============================================================================
# Shared metadata helpers
# =============================================================================


def _check_format_version(lmprobe_info: dict, *, check_minor: bool = True) -> None:
    """Raise on major version mismatch; warn on minor if *check_minor*."""
    remote_version = lmprobe_info.get("format_version", "1.0")
    remote_major = int(remote_version.split(".")[0])
    local_major = int(FORMAT_VERSION.split(".")[0])
    if remote_major != local_major:
        raise ValueError(
            f"Incompatible format version: remote has {remote_version}, "
            f"lmprobe supports {FORMAT_VERSION}. "
            f"Please upgrade lmprobe: pip install --upgrade lmprobe"
        )
    if check_minor:
        remote_minor = int(remote_version.split(".")[1])
        local_minor = int(FORMAT_VERSION.split(".")[1])
        if remote_minor > local_minor:
            warnings.warn(
                f"Remote format {remote_version} is newer than supported "
                f"{FORMAT_VERSION}. Some tensor types may be skipped.",
                stacklevel=3,
            )


def _extract_model_name(lmprobe_info: dict) -> str:
    """Extract model name from lmprobe_info, supporting both canonical and legacy formats."""
    model_obj = lmprobe_info.get("model")
    if isinstance(model_obj, dict):
        return model_obj["name"]
    if "model_name" in lmprobe_info:
        return lmprobe_info["model_name"]
    raise KeyError(
        "lmprobe_info.json missing model name — expected 'model.name' "
        "or top-level 'model_name'"
    )


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
# Staging helpers
# =============================================================================


def _staging_dir_path(
    repo_id: str,
    model_name: str,
    prompts: list[str],
    *,
    shard_max_bytes: int = DEFAULT_SHARD_BYTES,
    labels: list[int | str | None] | None = None,
    metadata: list[dict] | None = None,
    tensors: list[str] | None = None,
    stream: bool = False,
    stream_batch_size: int | None = None,
    shuffle: bool = True,
) -> Path:
    """Deterministic staging directory for resumable uploads.

    The hash key includes all parameters that affect staged output so that
    retrying with different settings produces a fresh staging dir rather
    than silently resuming from mismatched data.
    """
    from .cache import get_cache_dir

    key_parts: list[Any] = [repo_id, model_name, prompts, shard_max_bytes]
    if labels is not None:
        key_parts.append(["labels", labels])
    if metadata is not None:
        key_parts.append(["metadata", metadata])
    if tensors is not None:
        key_parts.append(["tensors", sorted(tensors)])
    if stream:
        key_parts.append(["stream", True])
    if stream_batch_size is not None:
        key_parts.append(["stream_batch_size", stream_batch_size])
    if not shuffle:
        key_parts.append(["shuffle", False])
    content = json.dumps(key_parts, sort_keys=True)
    key = hashlib.sha256(content.encode()).hexdigest()[:16]
    return get_cache_dir() / "staging" / key


# =============================================================================
# Streaming manifest helpers
# =============================================================================


def _new_manifest(repo_id: str) -> dict:
    """Create a fresh streaming manifest."""
    return {
        "format": "stream_manifest_v1",
        "repo_id": repo_id,
        "completed_shards": [],
        "metadata_uploaded": False,
    }


def _load_manifest(staging_dir: Path) -> dict | None:
    """Load streaming manifest, or None if not present."""
    manifest_path = staging_dir / _STREAM_MANIFEST
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)


def _save_manifest(staging_dir: Path, manifest: dict) -> None:
    """Atomically write the streaming manifest via rename."""
    manifest_path = staging_dir / _STREAM_MANIFEST
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp_path.rename(manifest_path)


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
            "has_token_perplexity": False,
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
    has_token_perplexity = all(i.has_token_perplexity for i in infos)

    return {
        "raw_layers": raw_intersection,
        "pooled": pooled,
        "has_logits": has_logits,
        "logits_top_k": logits_top_k,
        "has_perplexity": has_perplexity,
        "has_token_perplexity": has_token_perplexity,
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
        "has_token_perplexity": False,
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
            result["has_token_perplexity"] = available.get("has_token_perplexity", False)
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
# Preloading helpers (parallel S3 reads)
# =============================================================================


def _parallel_preload(
    prompts: list[str],
    load_fn: Any,
    default: Any,
    workers: int,
    executor: Any | None = None,
    pbar: Any | None = None,
) -> list:
    """Run *load_fn* for every prompt in parallel, collecting results.

    Parameters
    ----------
    load_fn : callable(idx, prompt) -> (idx, value)
        Per-prompt loader.  Must return ``(index, result)`` tuple.
    default : callable() -> value
        Factory for the default/empty value (e.g. ``lambda: None``).
    executor : ThreadPoolExecutor, optional
        Reusable thread pool.  If ``None``, a temporary pool is created.
    pbar : tqdm bar, optional
        Shared progress bar to update per completed prompt.

    Returns a list indexed by prompt position.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run(pool: Any) -> list:
        result = [default() for _ in prompts]
        futures = {
            pool.submit(load_fn, i, p): i
            for i, p in enumerate(prompts)
        }
        for fut in as_completed(futures):
            idx, value = fut.result()
            result[idx] = value
            if pbar is not None:
                pbar.update(1)
        return result

    if executor is not None:
        return _run(executor)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return _run(pool)


def _make_hidden_loader(
    model_name: str,
    layers: list[int],
    use_raw: bool,
    hidden_strategy: str | None,
    extract_key: str | None = None,
) -> Any:
    """Build a ``(idx, prompt) -> (idx, value)`` loader for hidden states.

    Parameters
    ----------
    extract_key : str, optional
        If given, return ``tensors.get(key)`` instead of the full dict.
        Used by per-layer preloading to return a single tensor.
    """
    empty: Any = None if extract_key else {}

    def _load_one(idx: int, prompt: str) -> tuple[int, Any]:
        try:
            if use_raw:
                tensors, _ = _load_hidden_raw_for_prompt(
                    model_name, prompt, layers,
                )
            elif hidden_strategy:
                tensors = _load_hidden_for_prompt(
                    model_name, prompt, layers, hidden_strategy,
                )
            else:
                return idx, empty
            if extract_key:
                return idx, tensors.get(extract_key)
            return idx, tensors
        except (FileNotFoundError, KeyError, OSError) as e:
            raise OSError(
                f"Prompt passed metadata scan but failed to load during "
                f"preload (cache may have been modified concurrently): {e}"
            ) from e

    return _load_one


def _preload_layer(
    model_name: str,
    prompts: list[str],
    layer: int,
    use_raw: bool,
    hidden_strategy: str | None,
    workers: int,
    executor: Any | None = None,
    pbar: Any | None = None,
) -> list[torch.Tensor | None]:
    """Preload one layer's hidden states for all prompts in parallel.

    Parameters
    ----------
    executor : ThreadPoolExecutor, optional
        Reusable thread pool.  If ``None``, a temporary pool is created.
    pbar : tqdm bar, optional
        Shared progress bar to update.  If ``None``, a per-call bar is
        created (noisy when called in a loop).

    Returns a list indexed by prompt position.  Each entry is the tensor
    for that prompt/layer or ``None`` on load failure.
    """
    load_fn = _make_hidden_loader(
        model_name, [layer], use_raw, hidden_strategy,
        extract_key=f"hidden.layer_{layer}",
    )
    return _parallel_preload(
        prompts, load_fn, lambda: None, workers,
        executor=executor, pbar=pbar,
    )


def _preload_full(
    model_name: str,
    prompts: list[str],
    layers: list[int],
    use_raw: bool,
    hidden_strategy: str | None,
    workers: int,
) -> list[dict[str, torch.Tensor]]:
    """Preload all layers for all prompts in parallel.

    Returns a list indexed by prompt position.  Each entry is a dict
    mapping ``"hidden.layer_N"`` to the corresponding tensor.
    """
    from tqdm import tqdm

    load_fn = _make_hidden_loader(
        model_name, layers, use_raw, hidden_strategy,
    )
    total = len(prompts)
    with tqdm(total=total, desc="Preloading all layers", unit="prompt") as pbar:
        return _parallel_preload(
            prompts, load_fn, dict, workers, pbar=pbar,
        )


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


def _compute_shard_plan(
    model_name: str,
    prompts: list[str],
    kept_indices: list[int],
    tensor_types: dict[str, Any],
    labels: list[int | str | None] | None,
    shard_max_bytes: int,
    repo_id: str,
    metadata: list[dict] | None = None,
    shuffle: bool = True,
) -> dict[str, Any]:
    """Compute the shard plan without loading any tensor data.

    Runs the metadata scan, deterministic shuffle, boundary computation,
    and shard metadata assignment.  Returns everything needed to enumerate
    expected shard filenames, build prompt_metadata, and build
    tensor_descriptors — without touching any activation tensors.

    Returns
    -------
    dict with keys:
        prompt_metadata, valid_prompts, tensor_descriptors,
        hidden_boundaries, logits_boundaries, hidden_layers, hidden_dim,
        hidden_strategy, use_raw, per_prompt_tokens, has_hidden,
        want_logits, logits_top_k, logits_row_bytes, n
    """
    from tqdm import tqdm

    pooled = tensor_types["pooled"]
    has_logits = tensor_types["has_logits"]
    logits_top_k = tensor_types["logits_top_k"]
    has_perplexity = tensor_types.get("has_perplexity", False)
    has_token_perplexity = tensor_types.get("has_token_perplexity", False)

    raw_layers = tensor_types.get("raw_layers", [])
    use_raw = bool(raw_layers)

    hidden_strategy = None
    hidden_layers: list[int] = []
    if not use_raw and pooled:
        hidden_strategy = next(iter(pooled))
        hidden_layers = pooled[hidden_strategy]
    elif use_raw:
        hidden_layers = raw_layers

    # --- Phase 1: Metadata-only pass (no tensor loading) ---
    prompt_metadata: list[dict] = []
    valid_prompts: list[str] = []
    per_prompt_tokens: list[int] = []
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
        meta_entry: dict[str, Any] = {
            "text": prompt,
            "label": label,
            "num_tokens": num_tokens,
            **extra_meta,
        }

        if has_perplexity:
            try:
                ppl = load_prompt_perplexity(model_name, prompt)
                meta_entry["perplexity_mean"] = float(ppl[0])
                meta_entry["perplexity_min"] = float(ppl[1])
                meta_entry["perplexity_max"] = float(ppl[2])
            except (FileNotFoundError, KeyError, IndexError):
                logger.warning(
                    f"[SHARING] Could not load perplexity for prompt "
                    f"index {idx}, skipping perplexity columns"
                )

        if has_token_perplexity:
            try:
                tok_ppl, tok_ids = load_prompt_token_perplexity(
                    model_name, prompt,
                )
                meta_entry["token_ids"] = tok_ids.tolist()
                meta_entry["token_perplexity"] = tok_ppl.tolist()
                if _tokenizer_cache.get("instance") is None:
                    from transformers import AutoTokenizer
                    _tokenizer_cache["instance"] = (
                        AutoTokenizer.from_pretrained(model_name)
                    )
                tokenizer = _tokenizer_cache["instance"]
                meta_entry["token_strings"] = [
                    tokenizer.decode([tid])
                    for tid in tok_ids.tolist()
                ]
            except (FileNotFoundError, KeyError, IndexError):
                logger.warning(
                    f"[SHARING] Could not load token perplexity "
                    f"for prompt index {idx}, skipping token "
                    f"perplexity columns"
                )

        prompt_metadata.append(meta_entry)

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
    n = len(valid_prompts)
    if shuffle:
        seed = _deterministic_seed(repo_id)
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
        logits_row_bytes = logits_top_k * 4 + logits_top_k * 8

    has_hidden = bool(hidden_layers and (hidden_strategy or use_raw))
    want_logits = bool(has_logits and logits_top_k is not None)

    # Number of leading shards reserved for last-token vectors (0 for
    # pooled datasets or when no raw hidden layers are present).
    lt_shard_count = 0
    lt_boundaries: list[int] = []
    rest_boundaries: list[int] = []

    if use_raw:
        # Last-token shards: one vector per prompt (fixed size)
        hidden_row_bytes = hidden_dim * 4
        lt_boundaries = _compute_shard_boundaries(
            n, hidden_row_bytes, shard_max_bytes,
        ) if has_hidden else []
        lt_shard_count = len(lt_boundaries)

        # Rest-token shards: (num_tokens - 1) vectors per prompt (variable)
        rest_prompt_bytes = [
            max(tok - 1, 0) * hidden_dim * 4
            for tok in per_prompt_tokens
        ]
        # Only create rest shards if there are tokens beyond the last
        if any(b > 0 for b in rest_prompt_bytes):
            rest_boundaries = _compute_shard_boundaries_variable(
                rest_prompt_bytes, shard_max_bytes,
            )

        # Combined boundaries: last-token shards first, then rest shards
        hidden_boundaries = lt_boundaries + rest_boundaries
    else:
        hidden_row_bytes = hidden_dim * 4
        hidden_boundaries = _compute_shard_boundaries(
            n, hidden_row_bytes, shard_max_bytes,
        ) if has_hidden else []

    if want_logits:
        logits_boundaries = _compute_shard_boundaries(
            n, max(logits_row_bytes, 1), shard_max_bytes,
        )
    else:
        logits_boundaries = []

    # --- Phase 4a: Assign shard metadata (no data loading) ---
    if has_hidden and hidden_boundaries and use_raw and lt_boundaries:
        # Per-token shard mapping for full_sequence datasets.
        # Last-token shards have indices 0..lt_shard_count-1,
        # rest-token shards have indices lt_shard_count..len(hidden_boundaries)-1.

        # Build last-token shard assignments (prompt -> lt shard + offset)
        lt_row_in_shard = 0
        lt_shard_idx = 0
        lt_assignments: list[tuple[int, int]] = []  # (shard_idx, row_offset)
        for i in range(n):
            if lt_shard_idx < len(lt_boundaries) and lt_row_in_shard >= lt_boundaries[lt_shard_idx]:
                lt_shard_idx += 1
                lt_row_in_shard = 0
            lt_assignments.append((lt_shard_idx, lt_row_in_shard))
            lt_row_in_shard += 1

        # Build rest-token shard assignments (prompt -> rest shard + token_offset)
        rest_shard_idx = 0
        rest_tok_offset = 0
        rest_row_in_shard = 0
        rest_assignments: list[tuple[int, int]] = []  # (shard_idx, token_offset)
        if rest_boundaries:
            for i in range(n):
                if (
                    rest_shard_idx < len(rest_boundaries)
                    and rest_row_in_shard >= rest_boundaries[rest_shard_idx]
                ):
                    rest_shard_idx += 1
                    rest_tok_offset = 0
                    rest_row_in_shard = 0
                rest_assignments.append((lt_shard_count + rest_shard_idx, rest_tok_offset))
                rest_tok_offset += max(per_prompt_tokens[i] - 1, 0)
                rest_row_in_shard += 1

        # Assign per-token arrays and scalar backwards-compat fields
        for i in range(n):
            num_tok = per_prompt_tokens[i]
            lt_si, lt_off = lt_assignments[i]

            # Build per-token arrays
            token_shard_ids: list[int] = []
            token_shard_offsets: list[int] = []

            if num_tok > 1 and rest_assignments:
                rest_si, rest_off = rest_assignments[i]
                # Tokens 0..N-2 go to rest shard
                for t in range(num_tok - 1):
                    token_shard_ids.append(rest_si)
                    token_shard_offsets.append(rest_off + t)
            # Last token goes to last-token shard
            token_shard_ids.append(lt_si)
            token_shard_offsets.append(lt_off)

            prompt_metadata[i]["token_shard_ids"] = token_shard_ids
            prompt_metadata[i]["token_shard_offsets"] = token_shard_offsets

            # Scalar backwards-compat: point to last-token shard
            prompt_metadata[i]["shard_index_hidden"] = lt_si
            prompt_metadata[i]["row_offset_hidden"] = lt_off
            prompt_metadata[i]["shard_index"] = lt_si
            prompt_metadata[i]["row_offset"] = lt_off
            prompt_metadata[i]["token_offset_hidden"] = lt_off
            prompt_metadata[i]["token_offset"] = lt_off

    elif has_hidden and hidden_boundaries:
        offset = 0
        for shard_idx, shard_size in enumerate(hidden_boundaries):
            for local_row in range(shard_size):
                global_row = offset + local_row
                if global_row < len(prompt_metadata):
                    prompt_metadata[global_row]["shard_index_hidden"] = shard_idx
                    prompt_metadata[global_row]["row_offset_hidden"] = local_row
                    prompt_metadata[global_row]["shard_index"] = shard_idx
                    prompt_metadata[global_row]["row_offset"] = local_row
            offset += shard_size

    if want_logits and logits_boundaries:
        offset = 0
        for shard_idx, shard_size in enumerate(logits_boundaries):
            for local_row in range(shard_size):
                global_row = offset + local_row
                if global_row < len(prompt_metadata):
                    prompt_metadata[global_row]["shard_index_logits"] = shard_idx
                    prompt_metadata[global_row]["row_offset_logits"] = local_row
                    if not has_hidden:
                        prompt_metadata[global_row]["shard_index"] = shard_idx
                        prompt_metadata[global_row]["row_offset"] = local_row
            offset += shard_size

    # --- Build tensor descriptors ---
    tensor_descriptors: dict[str, dict] = {}

    if has_hidden and hidden_boundaries:
        hidden_shards = []
        if use_raw and lt_boundaries:
            # Last-token shards: 1 token per prompt
            off = 0
            for si, sz in enumerate(lt_boundaries):
                actual = min(sz, n - off)
                hidden_shards.append({
                    "num_prompts": actual,
                    "num_tokens": actual,  # 1 token per prompt
                })
                off += actual
            # Rest-token shards: (num_tokens - 1) per prompt
            off = 0
            for si, sz in enumerate(rest_boundaries):
                actual = min(sz, n - off)
                hidden_shards.append({
                    "num_prompts": actual,
                    "num_tokens": sum(
                        max(per_prompt_tokens[off + j] - 1, 0)
                        for j in range(actual)
                    ),
                })
                off += actual
        else:
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
            if lt_shard_count > 0:
                hidden_desc["last_token_shards"] = lt_shard_count
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

    return {
        "prompt_metadata": prompt_metadata,
        "valid_prompts": valid_prompts,
        "tensor_descriptors": tensor_descriptors,
        "hidden_boundaries": hidden_boundaries,
        "logits_boundaries": logits_boundaries,
        "hidden_layers": hidden_layers,
        "hidden_dim": hidden_dim,
        "hidden_strategy": hidden_strategy,
        "use_raw": use_raw,
        "per_prompt_tokens": per_prompt_tokens,
        "has_hidden": has_hidden,
        "want_logits": want_logits,
        "logits_top_k": logits_top_k,
        "logits_row_bytes": logits_row_bytes,
        "n": n,
        "lt_shard_count": lt_shard_count,
        "lt_boundaries": lt_boundaries,
        "rest_boundaries": rest_boundaries,
    }


def _reconstruct_plan_from_cached(cached_meta: dict) -> dict[str, Any]:
    """Reconstruct a minimal plan dict from cached consolidation metadata.

    Only populates the fields needed by ``_enumerate_shard_files``:
    has_hidden, hidden_layers, hidden_boundaries, want_logits,
    logits_boundaries, lt_shard_count.
    """
    td = cached_meta["tensor_descriptors"]
    hidden_info = td.get("hidden_layers", {})
    logits_info = td.get("logits_topk", {})

    has_hidden = bool(hidden_info)
    hidden_layers = hidden_info.get("layers", [])
    hidden_boundaries = [
        s["num_prompts"] for s in hidden_info.get("shards", [])
    ]
    lt_shard_count = hidden_info.get("last_token_shards", 0)

    want_logits = bool(logits_info)
    logits_boundaries = [
        s["num_prompts"] for s in logits_info.get("shards", [])
    ]

    return {
        "has_hidden": has_hidden,
        "hidden_layers": hidden_layers,
        "hidden_boundaries": hidden_boundaries,
        "want_logits": want_logits,
        "logits_boundaries": logits_boundaries,
        "lt_shard_count": lt_shard_count,
    }


def _enumerate_shard_files(plan: dict[str, Any]) -> list[str]:
    """Enumerate all expected shard filenames from a shard plan."""
    files = []
    if plan["has_hidden"] and plan["hidden_boundaries"]:
        for layer in plan["hidden_layers"]:
            for shard_idx in range(len(plan["hidden_boundaries"])):
                files.append(
                    f"tensors/hidden_layer{layer:03d}"
                    f"_shard{shard_idx:03d}.safetensors"
                )
    if plan["want_logits"] and plan["logits_boundaries"]:
        for shard_idx in range(len(plan["logits_boundaries"])):
            files.append(f"tensors/logits_topk_{shard_idx:03d}.safetensors")
    return files


def _check_shards_on_remote(
    api: Any,
    repo_id: str,
    shard_files: list[str],
) -> set[str]:
    """Check which shard files exist on the HF remote via repo_info.

    Returns the set of shard paths that exist on remote.
    """
    try:
        repo_info = api.repo_info(repo_id, repo_type="dataset")
    except Exception:
        return set()

    remote_files = {s.rfilename for s in repo_info.siblings}
    return {f for f in shard_files if f in remote_files}


def _consolidate_and_shard(
    model_name: str,
    prompts: list[str],
    kept_indices: list[int],
    tensor_types: dict[str, Any],
    labels: list[int | str | None] | None,
    shard_max_bytes: int,
    repo_id: str,
    metadata: list[dict] | None = None,
    tmpdir: Path | None = None,
    on_shard_written: Callable[[Path, str], None] | None = None,
    preload: str = "none",
    preload_workers: int = 8,
    skip_shards: set[str] | None = None,
    shuffle: bool = True,
) -> tuple[Path, dict, list[dict]]:
    """Consolidate cached tensors into sharded safetensors files.

    Layers are co-located within each hidden_layers shard.  Each tensor type
    gets independent shard boundaries (v1.2), so small logits data isn't
    needlessly split across many shards.

    Uses streaming consolidation: only one shard's worth of tensors is held
    in memory at a time, so peak memory is bounded by ``shard_max_bytes``
    regardless of total dataset size.

    Parameters
    ----------
    skip_shards : set[str] | None
        Shard filenames to skip writing (e.g. already on remote).
        When provided, tensor loading and file writing are skipped for
        these shards.  The shard plan (metadata, descriptors) is still
        computed in full.

    Returns
    -------
    tmpdir : Path
        Temporary directory containing all output files.
    tensor_descriptors : dict
        The "tensors" section of lmprobe_info.json.
    prompt_metadata : list[dict]
        Per-prompt metadata for the Parquet index.
    """
    _VALID_PRELOAD = ("none", "per_layer", "full")
    if preload not in _VALID_PRELOAD:
        raise ValueError(
            f"preload must be one of {_VALID_PRELOAD!r}, got {preload!r}"
        )

    from safetensors.torch import save_file

    if tmpdir is None:
        tmpdir = Path(tempfile.mkdtemp(prefix="lmprobe_sharing_"))
    (tmpdir / "tensors").mkdir(parents=True, exist_ok=True)
    (tmpdir / "index").mkdir(parents=True, exist_ok=True)

    if skip_shards is None:
        skip_shards = set()

    # Compute the shard plan (metadata scan, shuffle, boundaries, descriptors)
    plan = _compute_shard_plan(
        model_name=model_name,
        prompts=prompts,
        kept_indices=kept_indices,
        tensor_types=tensor_types,
        labels=labels,
        shard_max_bytes=shard_max_bytes,
        repo_id=repo_id,
        metadata=metadata,
        shuffle=shuffle,
    )

    prompt_metadata = plan["prompt_metadata"]
    valid_prompts = plan["valid_prompts"]
    tensor_descriptors = plan["tensor_descriptors"]
    hidden_boundaries = plan["hidden_boundaries"]
    logits_boundaries = plan["logits_boundaries"]
    hidden_layers = plan["hidden_layers"]
    hidden_strategy = plan["hidden_strategy"]
    use_raw = plan["use_raw"]
    has_hidden = plan["has_hidden"]
    want_logits = plan["want_logits"]
    logits_top_k = plan["logits_top_k"]
    lt_shard_count = plan.get("lt_shard_count", 0)
    lt_boundaries = plan.get("lt_boundaries", [])
    rest_boundaries = plan.get("rest_boundaries", [])

    from tqdm import tqdm

    # --- Write shards per tensor type ---

    def _save_shard(
        rows: list[torch.Tensor],
        key: str,
        fname: str,
    ) -> None:
        """Concatenate rows, save shard, invoke callback."""
        if rows:
            layer_tensor = {key: torch.cat(rows, dim=0)}
            save_file(layer_tensor, str(tmpdir / fname))
            if on_shard_written is not None:
                on_shard_written(tmpdir / fname, fname)
            del layer_tensor

    # --- Hidden pass ---
    if has_hidden and hidden_boundaries:
        # Phase 4b: Write per-layer shard files one layer at a time.
        total_layer_shards = len(hidden_layers) * len(hidden_boundaries)

        def _write_group_shards(
            layer: int,
            data: list[torch.Tensor | None],
            boundaries: list[int],
            shard_idx_offset: int,
            slice_fn: Callable[[torch.Tensor], torch.Tensor],
            pbar: Any,
        ) -> None:
            """Write shards for one layer from preloaded data.

            ``slice_fn`` selects which tokens from each prompt tensor
            to include (e.g. last token only, or all-but-last).
            """
            key = f"hidden.layer_{layer}"
            offset = 0
            for local_idx, shard_size in enumerate(boundaries):
                shard_idx = shard_idx_offset + local_idx
                fname = (
                    f"tensors/hidden_layer{layer:03d}"
                    f"_shard{shard_idx:03d}.safetensors"
                )
                if fname in skip_shards:
                    offset += shard_size
                    pbar.update(1)
                    continue
                rows = []
                for j in range(shard_size):
                    t = data[offset + j]
                    if t is not None:
                        sliced = slice_fn(t)
                        if sliced.shape[0] > 0:
                            rows.append(sliced)
                _save_shard(rows, key, fname)
                del rows
                offset += shard_size
                pbar.update(1)

        def _write_layer_shards(
            layer: int,
            data: list[torch.Tensor | None],
            pbar: Any,
        ) -> None:
            """Write all shards for one layer from preloaded data."""
            if use_raw and lt_shard_count > 0:
                # Split: lt shards get last token, rest shards get remainder
                _write_group_shards(
                    layer, data, lt_boundaries, 0,
                    lambda t: t[-1:], pbar,
                )
                _write_group_shards(
                    layer, data, rest_boundaries, lt_shard_count,
                    lambda t: t[:-1], pbar,
                )
            else:
                # Non-split: write all tokens to each shard
                _write_group_shards(
                    layer, data, hidden_boundaries, 0,
                    lambda t: t, pbar,
                )

        if preload == "full":
            # Preload ALL layers for ALL prompts, then write shards
            # from memory.  Highest memory, fastest I/O.
            all_data = _preload_full(
                model_name, valid_prompts, hidden_layers,
                use_raw, hidden_strategy, preload_workers,
            )
            with tqdm(
                total=total_layer_shards, desc="Writing hidden shards",
                unit="shard",
            ) as pbar:
                for layer in hidden_layers:
                    key = f"hidden.layer_{layer}"
                    layer_data = [d.get(key) for d in all_data]
                    _write_layer_shards(layer, layer_data, pbar)
                    del layer_data
            del all_data

        elif preload == "per_layer":
            # Preload one layer at a time across all prompts (parallel
            # reads), then write shards from memory before moving to
            # the next layer.  A single ThreadPoolExecutor and progress
            # bar are reused across all layers to avoid overhead and
            # log noise.
            from concurrent.futures import ThreadPoolExecutor

            total_preload = len(hidden_layers) * len(valid_prompts)
            with (
                ThreadPoolExecutor(max_workers=preload_workers) as pool,
                tqdm(
                    total=total_preload,
                    desc="Preloading hidden layers",
                    unit="prompt",
                ) as preload_pbar,
                tqdm(
                    total=total_layer_shards,
                    desc="Writing hidden shards",
                    unit="shard",
                ) as write_pbar,
            ):
                for layer in hidden_layers:
                    layer_data = _preload_layer(
                        model_name, valid_prompts, layer,
                        use_raw, hidden_strategy, preload_workers,
                        executor=pool, pbar=preload_pbar,
                    )
                    _write_layer_shards(layer, layer_data, write_pbar)
                    del layer_data

        else:
            # preload == "none": original sequential behaviour.
            # Peak RAM = one shard's worth of one layer.

            def _write_group_shards_streaming(
                layer: int,
                boundaries: list[int],
                shard_idx_offset: int,
                slice_fn: Callable[[Any], Any],
                pbar: Any,
            ) -> None:
                """Load and write shards for one boundary group."""
                key = f"hidden.layer_{layer}"
                offset = 0
                for local_idx, shard_size in enumerate(boundaries):
                    shard_idx = shard_idx_offset + local_idx
                    fname = (
                        f"tensors/hidden_layer{layer:03d}"
                        f"_shard{shard_idx:03d}.safetensors"
                    )
                    if fname in skip_shards:
                        offset += shard_size
                        pbar.update(1)
                        continue
                    shard_prompts_text = valid_prompts[
                        offset : offset + shard_size
                    ]
                    rows: list[torch.Tensor] = []
                    for prompt in shard_prompts_text:
                        try:
                            if use_raw:
                                layer_tensors, _ = (
                                    _load_hidden_raw_for_prompt(
                                        model_name, prompt, [layer],
                                    )
                                )
                            elif hidden_strategy:
                                layer_tensors = _load_hidden_for_prompt(
                                    model_name, prompt, [layer],
                                    hidden_strategy,
                                )
                            else:
                                layer_tensors = {}
                            if key in layer_tensors:
                                sliced = slice_fn(layer_tensors[key])
                                if sliced.shape[0] > 0:
                                    rows.append(sliced)
                        except (
                            FileNotFoundError, KeyError, OSError,
                        ) as e:
                            raise OSError(
                                f"Prompt passed metadata scan but "
                                f"failed to load during shard write "
                                f"(cache may have been modified "
                                f"concurrently): {e}"
                            ) from e
                    _save_shard(rows, key, fname)
                    del rows
                    offset += shard_size
                    pbar.update(1)

            with tqdm(
                total=total_layer_shards, desc="Writing hidden shards",
                unit="shard",
            ) as pbar:
                for layer in hidden_layers:
                    if use_raw and lt_shard_count > 0:
                        _write_group_shards_streaming(
                            layer, lt_boundaries, 0,
                            lambda t: t[-1:], pbar,
                        )
                        _write_group_shards_streaming(
                            layer, rest_boundaries, lt_shard_count,
                            lambda t: t[:-1], pbar,
                        )
                    else:
                        _write_group_shards_streaming(
                            layer, hidden_boundaries, 0,
                            lambda t: t, pbar,
                        )

    # --- Logits pass ---
    if want_logits and logits_boundaries:
        offset = 0
        for shard_idx, shard_size in enumerate(tqdm(
            logits_boundaries, desc="Writing logits shards", unit="shard",
        )):
            fname = f"tensors/logits_topk_{shard_idx:03d}.safetensors"
            if fname in skip_shards:
                offset += shard_size
                continue

            shard_prompts_text = valid_prompts[offset : offset + shard_size]

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
                if on_shard_written is not None:
                    on_shard_written(tmpdir / fname, fname)
                del logits_tensors_out

            del shard_data_logits
            offset += shard_size

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
            elif isinstance(sample, list):
                # Infer list element type from first element
                # Check bool before int since bool is a subclass of int
                if sample and isinstance(sample[0], bool):
                    pa_type = pa.list_(pa.bool_())
                elif sample and isinstance(sample[0], int):
                    pa_type = pa.list_(pa.int64())
                elif sample and isinstance(sample[0], float):
                    pa_type = pa.list_(pa.float64())
                elif sample and isinstance(sample[0], str):
                    pa_type = pa.list_(pa.string())
                else:
                    pa_type = pa.list_(pa.string())
                    values = [
                        [str(x) for x in v] if v is not None else None
                        for v in values
                    ]
            elif isinstance(sample, bool):
                pa_type = pa.bool_()
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
    num_workers: int | None = None,
    commit_batch_size: int | None = None,
    stream: bool = False,
    stream_batch_size: int = 10,
    preload: str = "none",
    preload_workers: int = 8,
    shuffle: bool = True,
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
    num_workers : int | None
        Number of workers for parallel file uploads.  Passed directly to
        ``upload_large_folder``.  ``None`` (default) uses the
        ``huggingface_hub`` default (currently 16 workers).
        Ignored when ``stream=True``.
    commit_batch_size : int | None
        Maximum number of files per commit during upload.  On unreliable
        connections, setting this to a small value (e.g. ``1``) ensures
        progress is committed frequently, so interrupted uploads lose
        less work on restart.  ``None`` (default) uses the
        ``huggingface_hub`` default scale.
        Ignored when ``stream=True``.
    stream : bool
        If True, upload shards in batches after writing them, then
        delete local copies.  This reduces peak disk usage from the
        full dataset size to ``stream_batch_size`` shards (~N GB).
        Resumable via a shard completion manifest.  Default False.
    stream_batch_size : int
        Number of shards to buffer before uploading as a single commit
        in streaming mode.  Higher values improve throughput (fewer
        commits, parallel LFS uploads) at the cost of more peak disk.
        Default 10.  Ignored when ``stream=False``.
    preload : str
        Strategy for preloading cached tensors during consolidation.
        ``"none"`` (default) reads each prompt sequentially per layer/shard.
        ``"per_layer"`` preloads one layer across all prompts in parallel
        before writing shards — best trade-off for most cases.
        ``"full"`` preloads all layers for all prompts into memory — fastest
        but requires enough RAM to hold the entire dataset.
    preload_workers : int
        Number of parallel threads for preloading cache reads.
        Only used when ``preload`` is not ``"none"``.  Default 8.
    shuffle : bool
        If True (default), deterministically shuffle prompts across shards
        using a seed derived from ``repo_id``.  If False, preserve the
        input prompt order in the output shards.

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

    # Step 4: Compute deterministic staging dir for resumable uploads
    staging_dir = _staging_dir_path(
        repo_id, model_name, prompts,
        shard_max_bytes=shard_max_bytes, labels=labels,
        metadata=metadata, tensors=tensors,
        stream=stream,
        stream_batch_size=stream_batch_size if stream else None,
        shuffle=shuffle,
    )

    if stream:
        return _push_dataset_streaming(
            repo_id=repo_id,
            model_name=model_name,
            prompts=prompts,
            kept_indices=kept_indices,
            tensor_types=tensor_types,
            labels=labels,
            metadata=metadata,
            shard_max_bytes=shard_max_bytes,
            private=private,
            exist_ok=exist_ok,
            description=description,
            license=license,
            token=token,
            num_workers=num_workers,
            commit_batch_size=commit_batch_size,
            staging_dir=staging_dir,
            stream_batch_size=stream_batch_size,
            preload=preload,
            preload_workers=preload_workers,
            shuffle=shuffle,
        )

    # --- Non-streaming path (unchanged) ---
    sentinel = staging_dir / _STAGE_SENTINEL

    resuming = sentinel.exists()
    if resuming:
        import time

        age_days = (time.time() - sentinel.stat().st_mtime) / 86400
        logger.info(
            "[SHARING] Resuming from staging dir: %s (%.1f days old)",
            staging_dir, age_days,
        )
        if age_days > 3:
            logger.warning(
                "[SHARING] Staging dir is %.0f days old — if parameters "
                "have changed, delete it to force re-consolidation: %s",
                age_days, staging_dir,
            )
        with open(staging_dir / INFO_FILENAME) as f:
            lmprobe_info = json.load(f)
        tmpdir = staging_dir
        num_prompts = lmprobe_info["num_prompts"]
    else:
        # Full consolidation path
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
            tmpdir=staging_dir,
            preload=preload,
            preload_workers=preload_workers,
            shuffle=shuffle,
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

        # Touch sentinel AFTER all staging is complete
        sentinel.touch()

    # Step 7: Upload
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id,
        exist_ok=True if resuming else exist_ok,
        private=private,
        repo_type="dataset",
    )
    total_size = sum(f.stat().st_size for f in tmpdir.rglob("*") if f.is_file())
    logger.info(
        "[SHARING] Uploading dataset (%.2f GB) via upload_large_folder",
        total_size / 1e9,
    )
    if commit_batch_size is not None:
        import huggingface_hub._upload_large_folder as _ulf

        _orig_scale = _ulf.COMMIT_SIZE_SCALE
        _ulf.COMMIT_SIZE_SCALE = [commit_batch_size]
        logger.info(
            "[SHARING] commit_batch_size=%d — overriding COMMIT_SIZE_SCALE",
            commit_batch_size,
        )

    try:
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=str(tmpdir),
            repo_type="dataset",
            num_workers=num_workers,
        )
    finally:
        if commit_batch_size is not None:
            _ulf.COMMIT_SIZE_SCALE = _orig_scale
    url = f"https://huggingface.co/datasets/{repo_id}"

    # Cleanup
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)

    logger.info(f"[SHARING] Pushed {num_prompts} prompts to {url}")
    return url


def _push_dataset_streaming(
    *,
    repo_id: str,
    model_name: str,
    prompts: list[str],
    kept_indices: list[int],
    tensor_types: dict[str, Any],
    labels: list[int | str | None] | None,
    metadata: list[dict] | None,
    shard_max_bytes: int,
    private: bool,
    exist_ok: bool,
    description: str | None,
    license: str,
    token: str | None,
    num_workers: int | None,
    commit_batch_size: int | None,
    staging_dir: Path,
    stream_batch_size: int = 10,
    preload: str = "none",
    preload_workers: int = 8,
    shuffle: bool = True,
) -> str:
    """Streaming upload: write shards, batch-upload via create_commit.

    Uses dry consolidation with remote checks for resume: computes the
    shard plan from cache metadata, checks which shards exist on the
    HF remote, and only consolidates missing shards.
    """
    import shutil

    from huggingface_hub import CommitOperationAdd, HfApi

    if num_workers is not None:
        logger.warning(
            "[SHARING] num_workers is ignored in streaming mode"
        )
    if commit_batch_size is not None:
        logger.warning(
            "[SHARING] commit_batch_size is ignored in streaming mode"
        )

    # Load or create manifest for resumability
    manifest = _load_manifest(staging_dir)
    resuming = manifest is not None
    if manifest is None:
        manifest = _new_manifest(repo_id)

    # Create repo early (need it for create_commit calls and remote checks)
    api = HfApi(token=token)
    api.create_repo(
        repo_id,
        exist_ok=True if resuming else exist_ok,
        private=private,
        repo_type="dataset",
    )

    # --- Dry consolidation: compute shard plan, check remote ---
    # Use cached consolidation result if available (fast path),
    # otherwise compute it from cache metadata (metadata scan).
    cached_meta = manifest.get("consolidation_result")
    if cached_meta is not None:
        logger.info(
            "[SHARING] Using cached consolidation result from manifest"
        )
        tensor_descriptors = cached_meta["tensor_descriptors"]
        prompt_metadata = cached_meta["prompt_metadata"]
        # Reconstruct the plan dict with enough info for _enumerate_shard_files.
        # We need hidden_layers, hidden_boundaries, logits_boundaries,
        # has_hidden, want_logits from the cached descriptors.
        plan = _reconstruct_plan_from_cached(cached_meta)
    else:
        logger.info("[SHARING] Computing shard plan from cache metadata...")
        plan = _compute_shard_plan(
            model_name=model_name,
            prompts=prompts,
            kept_indices=kept_indices,
            tensor_types=tensor_types,
            labels=labels,
            shard_max_bytes=shard_max_bytes,
            repo_id=repo_id,
            metadata=metadata,
            shuffle=shuffle,
        )
        tensor_descriptors = plan["tensor_descriptors"]
        prompt_metadata = plan["prompt_metadata"]

    # Enumerate expected shard files and check which exist on remote
    expected_files = _enumerate_shard_files(plan)
    remote_existing = _check_shards_on_remote(api, repo_id, expected_files)
    missing_files = set(expected_files) - remote_existing

    if resuming:
        logger.info(
            "[SHARING] Resume check: %d/%d shards on remote, %d missing",
            len(remote_existing), len(expected_files), len(missing_files),
        )

    tmpdir = staging_dir
    (tmpdir / "tensors").mkdir(parents=True, exist_ok=True)
    (tmpdir / "index").mkdir(parents=True, exist_ok=True)

    if missing_files:
        # Build the batched on_shard_written callback
        pending_shards: list[tuple[Path, str]] = []

        def _flush_batch() -> None:
            """Upload buffered shards as a single commit, then cleanup."""
            if not pending_shards:
                return
            paths_in_batch = [rp for _, rp in pending_shards]
            logger.info(
                "[SHARING] Uploading batch of %d shards: %s",
                len(pending_shards),
                ", ".join(paths_in_batch),
            )
            # Save pending paths to manifest before commit so resume can
            # find them if create_commit fails.
            manifest["pending_batch"] = [
                [str(lp), rp] for lp, rp in pending_shards
            ]
            _save_manifest(staging_dir, manifest)

            operations = [
                CommitOperationAdd(
                    path_in_repo=repo_path,
                    path_or_fileobj=str(local_path),
                )
                for local_path, repo_path in pending_shards
            ]
            api.create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message=(
                    f"Add {len(pending_shards)} shards"
                ),
                repo_type="dataset",
            )
            # Delete local copies and update manifest
            for local_path, repo_path in pending_shards:
                local_path.unlink(missing_ok=True)
                manifest["completed_shards"].append(repo_path)
            manifest.pop("pending_batch", None)
            _save_manifest(staging_dir, manifest)
            pending_shards.clear()

        def _on_shard_written(local_path: Path, repo_path: str) -> None:
            pending_shards.append((local_path, repo_path))
            if len(pending_shards) >= stream_batch_size:
                _flush_batch()

        # Retry any pending batch from a prior failed create_commit
        prior_batch = manifest.get("pending_batch", [])
        if prior_batch:
            local_files_exist = all(
                Path(lp).exists() for lp, _ in prior_batch
            )
            if local_files_exist:
                logger.info(
                    "[SHARING] Retrying pending batch of %d shards",
                    len(prior_batch),
                )
                operations = [
                    CommitOperationAdd(
                        path_in_repo=rp,
                        path_or_fileobj=lp,
                    )
                    for lp, rp in prior_batch
                ]
                api.create_commit(
                    repo_id=repo_id,
                    operations=operations,
                    commit_message=(
                        f"Add {len(prior_batch)} shards (retry)"
                    ),
                    repo_type="dataset",
                )
                for lp, rp in prior_batch:
                    Path(lp).unlink(missing_ok=True)
                    manifest["completed_shards"].append(rp)
                manifest.pop("pending_batch", None)
                _save_manifest(staging_dir, manifest)
            else:
                manifest.pop("pending_batch", None)
                _save_manifest(staging_dir, manifest)

        # Consolidate only missing shards
        logger.info(
            "[SHARING] Consolidating %d missing shards "
            "(batch_size=%d)...", len(missing_files), stream_batch_size,
        )
        tmpdir, tensor_descriptors, prompt_metadata = _consolidate_and_shard(
            model_name=model_name,
            prompts=prompts,
            kept_indices=kept_indices,
            tensor_types=tensor_types,
            labels=labels,
            shard_max_bytes=shard_max_bytes,
            repo_id=repo_id,
            metadata=metadata,
            tmpdir=staging_dir,
            on_shard_written=_on_shard_written,
            preload=preload,
            preload_workers=preload_workers,
            skip_shards=remote_existing,
            shuffle=shuffle,
        )

        # Flush any remaining buffered shards
        _flush_batch()
    else:
        logger.info(
            "[SHARING] All %d shards already on remote — "
            "dry consolidation (no tensor loading)",
            len(expected_files),
        )

    # Cache consolidation results in the manifest
    manifest["consolidation_result"] = {
        "tensor_descriptors": tensor_descriptors,
        "prompt_metadata": prompt_metadata,
    }
    _save_manifest(staging_dir, manifest)

    # Upload metadata files (parquet, info, README)
    if not manifest.get("metadata_uploaded", False):
        _write_parquet_index(tmpdir, prompt_metadata)

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

        meta_operations = [
            CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=str(meta_file),
            )
            for meta_file, repo_path in [
                (tmpdir / "index" / "train-00000-of-00001.parquet",
                 PARQUET_PATH),
                (tmpdir / INFO_FILENAME, INFO_FILENAME),
                (tmpdir / "README.md", "README.md"),
            ]
        ]
        logger.info("[SHARING] Uploading metadata files")
        api.create_commit(
            repo_id=repo_id,
            operations=meta_operations,
            commit_message="Add dataset metadata",
            repo_type="dataset",
        )

        manifest["metadata_uploaded"] = True
        _save_manifest(staging_dir, manifest)
    else:
        num_prompts = len(prompt_metadata)

    url = f"https://huggingface.co/datasets/{repo_id}"

    # Cleanup
    shutil.rmtree(staging_dir, ignore_errors=True)

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

    # Support both canonical format (model.name, tensors) and legacy
    # format (model_name, tensor_types) written by custom staging scripts.
    model_name = _extract_model_name(lmprobe_info)
    format_version = lmprobe_info.get("format_version", "1.0")
    tensor_descriptors = lmprobe_info.get("tensors") or lmprobe_info.get("tensor_types") or {}

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
    show_progress: bool = False,
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
    show_progress : bool
        If True, display a tqdm progress bar for shard downloads.

    Returns
    -------
    int
        Number of prompts available (lazy) or unpacked (materialize).
    """
    _check_hub_deps()
    from huggingface_hub import hf_hub_download

    # Download metadata
    logger.info("[SHARING] Downloading dataset metadata from %s...", repo_id)
    info_path = hf_hub_download(
        repo_id, INFO_FILENAME, repo_type="dataset", token=token,
    )
    parquet_path = hf_hub_download(
        repo_id, PARQUET_PATH, repo_type="dataset", token=token,
    )

    with open(info_path) as f:
        lmprobe_info = json.load(f)

    # Version check
    _check_format_version(lmprobe_info)

    model_name = _extract_model_name(lmprobe_info)
    tensor_descriptors = lmprobe_info.get("tensors") or lmprobe_info.get("tensor_types") or {}

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

    # Save all requested indices for registry building (before dedup filtering)
    all_prompt_indices = list(prompt_indices)

    # Dedup: skip already-cached
    if not overwrite:
        new_indices = []
        requested_layers = set(layers) if layers is not None else None
        for i in prompt_indices:
            existing = discover_cached(model_name, index["text"][i])
            if existing is None:
                new_indices.append(i)
            elif requested_layers is not None and not requested_layers.issubset(
                set(existing.raw_layers)
            ):
                new_indices.append(i)
        skipped = len(prompt_indices) - len(new_indices)
        if skipped > 0:
            logger.info(
                f"[SHARING] Skipping {skipped} prompts already in local cache"
            )
        prompt_indices = new_indices

    # Determine tensor types to pull
    if tensors is not None:
        pull_types = [k for k in tensor_descriptors if k in tensors]
    else:
        pull_types = list(tensor_descriptors.keys())

    # Download shard files and record local paths
    # For per-layer layout (v1.1), we download per-layer files
    # For co-located layout (v1.0), we download the single shard file
    shard_local_paths: dict[str, dict[int, str]] = {}  # t_type -> shard_idx -> path
    # Per-layer: t_type -> shard_idx -> {layer: path}
    per_layer_paths: dict[str, dict[int, dict[int, str]]] = {}
    needed_shards: dict[str, set[int]] = {}

    # In lazy mode (materialize=False), shard files are needed for ALL
    # requested prompts — the registry serves activations from shard files
    # regardless of whether per-prompt cache entries exist.  Per-prompt
    # sidecar files (e.g. perplexity) do NOT contain hidden-layer data,
    # so dedup-filtering must not gate shard downloads in lazy mode.
    shard_source_indices = (
        all_prompt_indices if not materialize else prompt_indices
    )

    if not shard_source_indices:
        logger.info("[SHARING] All prompts already cached locally, skipping downloads")
    else:
        # Figure out which shards we need (v1.2: per-type shard indices)
        # When token_shard_ids is present (v1.3 full_sequence datasets),
        # collect all unique shard IDs across all tokens for each prompt.
        has_token_shard_ids = "token_shard_ids" in index
        for i in shard_source_indices:
            for t_type in pull_types:
                if (
                    t_type == "hidden_layers"
                    and has_token_shard_ids
                    and materialize
                ):
                    # Per-token shard mapping: collect all shards for this prompt
                    for si in index["token_shard_ids"][i]:
                        needed_shards.setdefault(t_type, set()).add(si)
                elif t_type == "logits_topk" and "shard_index_logits" in index:
                    si = index["shard_index_logits"][i]
                    needed_shards.setdefault(t_type, set()).add(si)
                elif t_type == "hidden_layers" and "shard_index_hidden" in index:
                    si = index["shard_index_hidden"][i]
                    needed_shards.setdefault(t_type, set()).add(si)
                else:
                    # Legacy v1.1 fallback: single shard_index for all types
                    si = index["shard_index"][i]
                    needed_shards.setdefault(t_type, set()).add(si)

        # Count total files to download for progress
        _total_shard_files = 0
        for t_type in pull_types:
            t_info = tensor_descriptors[t_type]
            layout = t_info.get("layout")
            n_shards = len(needed_shards.get(t_type, []))
            if layout == "per_layer":
                all_layers = t_info.get("layers", [])
                dl_layers = (
                    all_layers if layers is None
                    else [ly for ly in all_layers if ly in layers]
                )
                _total_shard_files += n_shards * len(dl_layers)
            else:
                _total_shard_files += n_shards
        logger.info(
            "[SHARING] Downloading %d shard files from HF...", _total_shard_files
        )

        from tqdm import tqdm
        _pbar = tqdm(
            total=_total_shard_files,
            desc="Downloading shards",
            disable=not show_progress,
        )

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
                        _pbar.update(1)
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
                    _pbar.update(1)
                    shard_local_paths[t_type][shard_idx] = shard_path

        _pbar.close()
        if _total_shard_files > 0:
            logger.info(
                "[SHARING] Downloaded %d shard files",
                _total_shard_files,
            )

    # ---- Build shard registry (manifest + index) ----
    # Always build the registry using ALL requested prompts (not just the
    # ones that needed downloading).  The registry is metadata about where
    # prompts live inside shard files — _lookup_shard() and
    # _load_pooled_from_shard() need it even when per-prompt caches exist.
    logger.info("[SHARING] Building shard registry...")
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
    # write_shard_registry merges with existing index automatically
    # Use all_prompt_indices (pre-dedup) so the registry covers every prompt,
    # not just those that needed downloading.
    shard_index: dict[str, Any] = {}
    for i in all_prompt_indices:
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
        # Per-token shard arrays (v1.3)
        for col in ("token_shard_ids", "token_shard_offsets"):
            if col in index:
                entry[col] = index[col][i]
        shard_index[prompt_hash] = entry

    write_shard_registry(model_name, manifest, shard_index, repo_id=repo_id)
    _n_layers = len(tensor_descriptors.get("hidden_layers", {}).get("layers", []))
    logger.info(
        "[SHARING] Shard registry ready (%d prompts%s)",
        len(all_prompt_indices),
        f", {_n_layers} layers" if _n_layers > 0 else "",
    )

    if not materialize:
        total_registered = len(all_prompt_indices)
        logger.info(
            f"[SHARING] Registered {total_registered} prompts in shard registry "
            f"(lazy mode, no per-prompt files)"
        )
        return total_registered

    if not prompt_indices:
        return 0

    total_prompts = len(prompt_indices)

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
    all_layer_paths: dict[int, dict[int, str]] | None = None,
) -> None:
    """Unpack per-prompt files from a single shard.

    This is the core materialization logic, extracted so it can be called
    from the main process or from a worker process.

    Parameters
    ----------
    shard_path : str | dict[int, str]
        For v1.0 co-located layout: path to the single shard file.
        For v1.1 per-layer layout: dict mapping layer index to file path.
    all_layer_paths : dict[int, dict[int, str]] | None
        For split full_sequence datasets: shard_idx -> {layer: path}.
        Needed when a prompt spans multiple shards.
    """
    from safetensors import safe_open
    from tqdm import tqdm

    from .cache import (
        _LOGITS_TOP_K_INDICES_KEY,
        _LOGITS_TOP_K_VALUES_KEY,
        _merge_save_backend,
        _prepare_tensor,
        _prompt_cache_key,
        _register_model,
    )

    has_token_shard_ids = "token_shard_ids" in index
    storage = t_info.get("storage", "pooled")

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

    # For split full_sequence datasets, load all shard tensors upfront
    # so we can reconstruct complete sequences from multiple shards.
    all_shard_tensors: dict[int, dict[str, torch.Tensor]] = {}
    if (
        has_token_shard_ids
        and t_type == "hidden_layers"
        and storage == "full_sequence"
        and all_layer_paths is not None
    ):
        for shard_idx, layer_paths in all_layer_paths.items():
            shard_data: dict[str, torch.Tensor] = {}
            for layer, layer_path in layer_paths.items():
                with safe_open(layer_path, framework="pt") as f:
                    for k in f.keys():
                        shard_data[k] = f.get_tensor(k)
            all_shard_tensors[shard_idx] = shard_data

    for pi in tqdm(
        shard_prompts,
        desc=f"Unpacking {t_type}",
        unit="prompt",
        leave=False,
    ):
        prompt_text = index["text"][pi]

        if t_type == "hidden_layers" and storage == "full_sequence":
            num_tok = index["num_tokens"][pi]
            hidden_dim = t_info.get("dim", 0)

            if has_token_shard_ids and all_shard_tensors:
                # Split format: reconstruct from per-token shard arrays
                token_shard_ids = index["token_shard_ids"][pi]
                token_shard_offsets = index["token_shard_offsets"][pi]

                # Discover layers from the first available shard
                first_shard_data = next(iter(all_shard_tensors.values()))
                layer_nums = sorted(
                    int(re.match(r"^hidden\.layer_(\d+)$", k).group(1))
                    for k in first_shard_data
                    if re.match(r"^hidden\.layer_(\d+)$", k)
                )

                # For each layer, assemble tokens in sequence order
                layer_slices = []
                for layer_num in layer_nums:
                    key = f"hidden.layer_{layer_num}"
                    token_vecs = []
                    for t_idx in range(num_tok):
                        sid = token_shard_ids[t_idx]
                        soff = token_shard_offsets[t_idx]
                        token_vecs.append(
                            all_shard_tensors[sid][key][soff : soff + 1]
                        )
                    layer_slices.append(
                        (layer_num, torch.cat(token_vecs, dim=0))
                    )
            else:
                # Legacy format: all tokens in one shard
                tok_off = index["token_offset"][pi]
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

    # Collect all (t_type, t_info, shard_path, shard_prompts, all_layer_paths) jobs
    has_token_shard_ids = "token_shard_ids" in index
    jobs: list[tuple] = []
    for t_type in pull_types:
        t_info = tensor_descriptors[t_type]
        layout = t_info.get("layout")
        storage = t_info.get("storage", "pooled")

        # For split full_sequence datasets, create ONE job with all prompts
        # and all shard paths so reconstruction can access all shards.
        if (
            has_token_shard_ids
            and t_type == "hidden_layers"
            and storage == "full_sequence"
            and layout == "per_layer"
        ):
            # Use the first available shard's layer paths as shard_path
            first_shard_idx = min(needed_shards.get(t_type, set()))
            first_layer_paths = per_layer_paths.get(t_type, {}).get(
                first_shard_idx
            )
            if first_layer_paths:
                all_lp = per_layer_paths.get(t_type, {})
                jobs.append((
                    t_type, t_info, first_layer_paths,
                    prompt_indices, all_lp,
                ))
            continue

        for shard_idx in sorted(needed_shards.get(t_type, [])):
            shard_prompts = [
                i for i in prompt_indices
                if index["shard_index"][i] == shard_idx
            ]
            if not shard_prompts:
                continue

            if layout == "per_layer":
                layer_paths = per_layer_paths.get(t_type, {}).get(shard_idx)
                if not layer_paths:
                    continue
                jobs.append((
                    t_type, t_info, layer_paths, shard_prompts, None,
                ))
            else:
                if shard_idx not in shard_local_paths.get(t_type, {}):
                    continue
                shard_path = shard_local_paths[t_type][shard_idx]
                jobs.append((
                    t_type, t_info, shard_path, shard_prompts, None,
                ))

    if num_workers > 0:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for t_type, t_info, shard_path, shard_prompts, all_lp in jobs:
                futures.append(
                    executor.submit(
                        _unpack_shard_prompts,
                        model_name, t_type, t_info,
                        shard_path, shard_prompts, index,
                        all_layer_paths=all_lp,
                    )
                )
            for fut in futures:
                fut.result()
    else:
        for t_type, t_info, shard_path, shard_prompts, all_lp in jobs:
            _unpack_shard_prompts(
                model_name, t_type, t_info,
                shard_path, shard_prompts, index,
                all_layer_paths=all_lp,
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
    _check_format_version(lmprobe_info, check_minor=False)

    tensor_descriptors = lmprobe_info.get("tensors") or lmprobe_info.get("tensor_types") or {}

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


def load_activations(
    dataset: str,
    *,
    prompts: list[str] | None = None,
    layers: list[int] | None = None,
    pooling: str = "last_token",
    token: str | None = None,
    as_dict: bool = True,
    return_labels: bool = False,
    show_progress: bool = True,
) -> dict[int, np.ndarray] | np.ndarray | tuple:
    """Load pooled activations from a HuggingFace activation dataset.

    Convenience function that downloads needed shards and returns
    structured activation arrays — no probe training required.

    Parameters
    ----------
    dataset : str
        HuggingFace Dataset repo ID.
    prompts : list[str] | None
        Prompts to load.  None loads all prompts in the dataset.
    layers : list[int] | None
        Layer indices to load.  None loads all available layers.
    pooling : str
        Pooling strategy (default ``"last_token"``).
    token : str | None
        HuggingFace API token.
    as_dict : bool
        If True (default), return ``{layer: ndarray(n_prompts, hidden_dim)}``.
        If False, return ``ndarray(n_prompts, n_layers, hidden_dim)``.
    return_labels : bool
        If True, also return labels from the dataset's Parquet index as a
        second element: ``(activations, labels)``.  Labels are a numpy array
        of ints (or None if the dataset has no ``label`` column).
    show_progress : bool
        If True (default), display tqdm progress bars for downloads
        and activation loading.

    Returns
    -------
    dict[int, np.ndarray] | np.ndarray | tuple
        Activation arrays keyed by layer index, or a single stacked array.
        If ``return_labels=True``, returns ``(activations, labels)`` where
        labels is ``np.ndarray | None``.
    """
    from .cache import load_pooled_batch

    meta = fetch_dataset_metadata(dataset, token=token)

    if layers is None:
        layers = meta.available_layers
    else:
        missing = set(layers) - set(meta.available_layers)
        if missing:
            raise ValueError(
                f"Layers {sorted(missing)} not in dataset. "
                f"Available: {meta.available_layers}"
            )
    layers = sorted(layers)

    if prompts is None:
        prompts = meta.prompts

    pull_dataset(dataset, layers=layers, target_prompts=prompts,
                 token=token, materialize=False, show_progress=show_progress)

    hidden_dim = meta.tensor_descriptors.get(
        "hidden_layers", {}
    ).get("dim")

    pooled = load_pooled_batch(
        meta.model_name, prompts, layers, pooling, fallback_to_raw=True,
        show_progress=show_progress,
    )
    stacked = pooled.detach().cpu().float().numpy()  # (n_prompts, n_layers*dim)

    if hidden_dim is None:
        hidden_dim = stacked.shape[-1] // len(layers)

    if as_dict:
        activations = {
            layer: stacked[:, i * hidden_dim : (i + 1) * hidden_dim]
            for i, layer in enumerate(layers)
        }
    else:
        activations = stacked.reshape(len(prompts), len(layers), hidden_dim)

    if not return_labels:
        return activations

    # Load labels from the Parquet index
    labels = _load_labels_for_prompts(dataset, prompts, token=token)
    return activations, labels


def _load_labels_for_prompts(
    repo_id: str,
    prompts: list[str],
    *,
    token: str | None = None,
) -> np.ndarray | None:
    """Load labels from a dataset's Parquet index for given prompts.

    Returns None if the dataset has no ``label`` column.
    """
    _check_hub_deps()
    _check_pyarrow()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    parquet_path = hf_hub_download(
        repo_id, PARQUET_PATH, repo_type="dataset", token=token,
    )
    table = pq.read_table(parquet_path)
    index = table.to_pydict()

    if "label" not in index:
        return None

    # Build prompt -> label mapping
    texts = index["text"]
    labels_raw = index["label"]
    prompt_to_label = dict(zip(texts, labels_raw))

    labels = [prompt_to_label.get(p) for p in prompts]
    if any(v is None for v in labels):
        return None

    return np.array(labels)


# =============================================================================
# Dataset migration
# =============================================================================


def migrate_dataset(
    repo_id: str,
    *,
    shard_max_bytes: int = DEFAULT_SHARD_BYTES,
    token: str | None = None,
    private: bool | None = None,
    dry_run: bool = False,
) -> str:
    """Migrate a full_sequence dataset to use last-token shard splitting.

    Downloads the existing dataset from HuggingFace, re-shards the hidden
    layer tensors so that last-token vectors land in dedicated small shards,
    rebuilds the parquet index with per-token shard mapping arrays
    (``token_shard_ids``, ``token_shard_offsets``), and re-uploads.

    Processing is done one layer at a time to limit memory usage.

    Parameters
    ----------
    repo_id : str
        HuggingFace repo ID of the dataset to migrate.
    shard_max_bytes : int
        Max bytes per shard file for the new sharding. Default 1 GB.
    token : str | None
        HuggingFace API token.
    private : bool | None
        If set, update the repo visibility. None keeps the current setting.
    dry_run : bool
        If True, download and compute the new plan but do not upload.

    Returns
    -------
    str
        URL of the migrated dataset (or a summary string if dry_run).
    """
    _check_hub_deps()
    _check_pyarrow()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tqdm import tqdm

    # ------------------------------------------------------------------
    # Step 1: Download metadata
    # ------------------------------------------------------------------
    logger.info("[MIGRATE] Downloading metadata from %s...", repo_id)
    info_path = hf_hub_download(
        repo_id, INFO_FILENAME, repo_type="dataset", token=token,
    )
    parquet_path = hf_hub_download(
        repo_id, PARQUET_PATH, repo_type="dataset", token=token,
    )

    with open(info_path) as f:
        lmprobe_info = json.load(f)

    tensor_descriptors = (
        lmprobe_info.get("tensors")
        or lmprobe_info.get("tensor_types")
        or {}
    )

    hidden_info = tensor_descriptors.get("hidden_layers", {})
    if hidden_info.get("storage") != "full_sequence":
        raise ValueError(
            "Dataset is not full_sequence storage — migration only applies "
            "to full_sequence datasets."
        )
    if hidden_info.get("last_token_shards", 0) > 0:
        raise ValueError(
            "Dataset already has last-token shard splitting "
            "— nothing to migrate."
        )

    # Read parquet index
    index_table = pq.read_table(parquet_path)
    index = index_table.to_pydict()
    n_prompts = len(index["text"])
    num_tokens_list = index["num_tokens"]

    hidden_layers = hidden_info.get("layers", [])
    hidden_dim = hidden_info.get("dim")
    old_shards = hidden_info.get("shards", [])

    model_name = _extract_model_name(lmprobe_info)

    old_shard_boundaries: list[int] = [
        s["num_prompts"] for s in old_shards
    ]

    logger.info(
        "[MIGRATE] Dataset: %d prompts, %d layers, dim=%d, "
        "%d old shards/layer",
        n_prompts, len(hidden_layers), hidden_dim, len(old_shards),
    )

    # ------------------------------------------------------------------
    # Step 2: Compute new shard boundaries
    # ------------------------------------------------------------------
    hidden_row_bytes = hidden_dim * 4

    # Last-token shards: one vector per prompt (fixed size)
    lt_boundaries = _compute_shard_boundaries(
        n_prompts, hidden_row_bytes, shard_max_bytes,
    )
    lt_shard_count = len(lt_boundaries)

    # Rest-token shards: mirror old shard boundaries (one rest shard
    # per old shard).  This avoids recomputing variable-size boundaries
    # and lets us write rest tensors directly from old shards with the
    # last-token rows removed.
    rest_boundaries = list(old_shard_boundaries)

    new_boundaries = lt_boundaries + rest_boundaries

    logger.info(
        "[MIGRATE] New plan: %d lt shards + %d rest shards "
        "= %d total per layer",
        lt_shard_count, len(rest_boundaries), len(new_boundaries),
    )

    # ------------------------------------------------------------------
    # Step 3: Build per-token arrays and new prompt_metadata
    # ------------------------------------------------------------------
    # Last-token shard assignments
    lt_row_in_shard = 0
    lt_shard_idx = 0
    lt_assignments: list[tuple[int, int]] = []
    for i in range(n_prompts):
        if (
            lt_shard_idx < len(lt_boundaries)
            and lt_row_in_shard >= lt_boundaries[lt_shard_idx]
        ):
            lt_shard_idx += 1
            lt_row_in_shard = 0
        lt_assignments.append((lt_shard_idx, lt_row_in_shard))
        lt_row_in_shard += 1

    # Rest-token shard assignments (mirrors old shard grouping)
    rest_assignments: list[tuple[int, int]] = []
    old_prompt_cursor = 0
    for old_si, n_in_shard in enumerate(old_shard_boundaries):
        rest_tok_offset = 0
        for j in range(n_in_shard):
            prompt_idx = old_prompt_cursor + j
            rest_assignments.append(
                (lt_shard_count + old_si, rest_tok_offset),
            )
            rest_tok_offset += max(num_tokens_list[prompt_idx] - 1, 0)
        old_prompt_cursor += n_in_shard

    # Build per-prompt token arrays
    all_token_shard_ids: list[list[int]] = []
    all_token_shard_offsets: list[list[int]] = []

    for i in range(n_prompts):
        num_tok = num_tokens_list[i]
        lt_si, lt_off = lt_assignments[i]

        tok_shard_ids: list[int] = []
        tok_shard_offsets: list[int] = []

        if num_tok > 1 and rest_assignments:
            rest_si, rest_off = rest_assignments[i]
            for t in range(num_tok - 1):
                tok_shard_ids.append(rest_si)
                tok_shard_offsets.append(rest_off + t)

        tok_shard_ids.append(lt_si)
        tok_shard_offsets.append(lt_off)

        all_token_shard_ids.append(tok_shard_ids)
        all_token_shard_offsets.append(tok_shard_offsets)

    if dry_run:
        return (
            f"[DRY RUN] Would migrate {repo_id}: "
            f"{n_prompts} prompts, {len(hidden_layers)} layers, "
            f"{lt_shard_count} lt shards + "
            f"{len(rest_boundaries)} rest shards"
        )

    # ------------------------------------------------------------------
    # Step 4: Download old shards and write new ones (per-layer)
    # ------------------------------------------------------------------
    tmpdir_obj = tempfile.mkdtemp(prefix="lmprobe_migrate_")
    tmpdir = Path(tmpdir_obj)
    (tmpdir / "tensors").mkdir(parents=True, exist_ok=True)
    (tmpdir / "index").mkdir(parents=True, exist_ok=True)

    # Pre-compute per-old-shard last-token row indices.
    # Within each old shard, tokens are laid out contiguously per prompt:
    #   [p0_t0, ..., p0_tN, p1_t0, ..., p1_tM, ...]
    # The last-token indices are at cumulative token boundaries minus 1.
    old_shard_lt_indices: list[torch.Tensor] = []
    old_prompt_cursor = 0
    for old_si, n_in_shard in enumerate(old_shard_boundaries):
        cum = 0
        indices = []
        for j in range(n_in_shard):
            prompt_idx = old_prompt_cursor + j
            n_tok = num_tokens_list[prompt_idx]
            cum += n_tok
            indices.append(cum - 1)  # last token of this prompt
        old_shard_lt_indices.append(torch.tensor(indices, dtype=torch.long))
        old_prompt_cursor += n_in_shard

    total_layer_work = (
        len(hidden_layers) * (lt_shard_count + len(old_shards))
    )
    with tqdm(
        total=total_layer_work,
        desc="Re-sharding layers",
        unit="shard",
    ) as pbar:
        for layer in hidden_layers:
            pbar.set_postfix(layer=layer)
            key = f"hidden.layer_{layer}"

            # Pass 1: extract lt vectors from each old shard via
            # index_select, and write rest shards directly.
            lt_vectors: list[torch.Tensor] = []
            for old_si in range(len(old_shards)):
                fname = (
                    f"tensors/hidden_layer{layer:03d}"
                    f"_shard{old_si:03d}.safetensors"
                )
                shard_path = hf_hub_download(
                    repo_id, fname,
                    repo_type="dataset", token=token,
                )
                with safe_open(shard_path, framework="pt") as sf:
                    shard_tensor = sf.get_tensor(key)

                lt_idx = old_shard_lt_indices[old_si]

                # Extract last-token vectors (one per prompt)
                lt_vectors.append(
                    torch.index_select(shard_tensor, 0, lt_idx)
                )

                # Build rest tensor by removing last-token rows
                n_rows = shard_tensor.shape[0]
                mask = torch.ones(n_rows, dtype=torch.bool)
                mask[lt_idx] = False
                rest_tensor = shard_tensor[mask]

                del shard_tensor

                # Write rest shard (same index as old shard)
                rest_shard_idx = lt_shard_count + old_si
                rest_fname = (
                    f"tensors/hidden_layer{layer:03d}"
                    f"_shard{rest_shard_idx:03d}.safetensors"
                )
                if rest_tensor.shape[0] > 0:
                    save_file(
                        {key: rest_tensor},
                        str(tmpdir / rest_fname),
                    )
                del rest_tensor, mask
                pbar.update(1)

            # Pass 2: concatenate lt vectors and split into lt shards
            all_lt = torch.cat(lt_vectors, dim=0)
            del lt_vectors

            offset = 0
            for local_idx, shard_size in enumerate(lt_boundaries):
                actual = min(shard_size, all_lt.shape[0] - offset)
                if actual > 0:
                    lt_fname = (
                        f"tensors/hidden_layer{layer:03d}"
                        f"_shard{local_idx:03d}.safetensors"
                    )
                    save_file(
                        {key: all_lt[offset:offset + actual]},
                        str(tmpdir / lt_fname),
                    )
                offset += actual
                pbar.update(1)
            del all_lt

    # ------------------------------------------------------------------
    # Step 5: Copy logits shards unchanged (if present)
    # ------------------------------------------------------------------
    logits_info = tensor_descriptors.get("logits_topk", {})
    logits_shards = logits_info.get("shards", [])
    if logits_shards:
        import shutil as _shutil

        logger.info(
            "[MIGRATE] Copying %d logits shards...", len(logits_shards),
        )
        for ls_desc in logits_shards:
            fname = ls_desc["file"]
            src = hf_hub_download(
                repo_id, fname, repo_type="dataset", token=token,
            )
            dst = tmpdir / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(src, dst)

    # ------------------------------------------------------------------
    # Step 6: Build new prompt_metadata and write parquet index
    # ------------------------------------------------------------------
    rebuild_keys = {
        "text", "label", "num_tokens", "shard_index", "row_offset",
        "shard_index_hidden", "row_offset_hidden",
        "token_offset_hidden", "token_offset",
        "token_shard_ids", "token_shard_offsets",
    }
    extra_parquet_keys = [
        k for k in index.keys() if k not in rebuild_keys
    ]

    prompt_metadata: list[dict] = []
    for i in range(n_prompts):
        lt_si, lt_off = lt_assignments[i]
        meta: dict[str, Any] = {
            "text": index["text"][i],
            "label": index["label"][i],
            "num_tokens": num_tokens_list[i],
            "shard_index": lt_si,
            "row_offset": lt_off,
            "shard_index_hidden": lt_si,
            "row_offset_hidden": lt_off,
            "token_offset_hidden": lt_off,
            "token_offset": lt_off,
            "token_shard_ids": all_token_shard_ids[i],
            "token_shard_offsets": all_token_shard_offsets[i],
        }
        if "shard_index_logits" in index:
            meta["shard_index_logits"] = index[
                "shard_index_logits"
            ][i]
        if "row_offset_logits" in index:
            meta["row_offset_logits"] = index[
                "row_offset_logits"
            ][i]
        for ek in extra_parquet_keys:
            meta[ek] = index[ek][i]
        prompt_metadata.append(meta)

    _write_parquet_index(tmpdir, prompt_metadata)

    # ------------------------------------------------------------------
    # Step 7: Build new tensor descriptors and lmprobe_info
    # ------------------------------------------------------------------
    new_hidden_shards = []
    off = 0
    for _si, sz in enumerate(lt_boundaries):
        actual = min(sz, n_prompts - off)
        new_hidden_shards.append({
            "num_prompts": actual,
            "num_tokens": actual,
        })
        off += actual
    off = 0
    for _si, sz in enumerate(rest_boundaries):
        actual = min(sz, n_prompts - off)
        new_hidden_shards.append({
            "num_prompts": actual,
            "num_tokens": sum(
                max(num_tokens_list[off + j] - 1, 0)
                for j in range(actual)
            ),
        })
        off += actual

    new_hidden_desc = dict(hidden_info)
    new_hidden_desc["shards"] = new_hidden_shards
    new_hidden_desc["last_token_shards"] = lt_shard_count

    new_td = dict(tensor_descriptors)
    new_td["hidden_layers"] = new_hidden_desc

    new_lmprobe_info = dict(lmprobe_info)
    new_lmprobe_info["tensors"] = new_td

    with open(tmpdir / INFO_FILENAME, "w") as f_out:
        json.dump(new_lmprobe_info, f_out, indent=2)

    readme = _build_readme(
        model_name=model_name,
        lmprobe_info=new_lmprobe_info,
        num_prompts=n_prompts,
        repo_id=repo_id,
    )
    with open(tmpdir / "README.md", "w") as f_out:
        f_out.write(readme)

    # ------------------------------------------------------------------
    # Step 8: Upload
    # ------------------------------------------------------------------
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if private is not None:
        api.update_repo_settings(
            repo_id, private=private, repo_type="dataset",
        )

    total_size = sum(
        fp.stat().st_size for fp in tmpdir.rglob("*") if fp.is_file()
    )
    logger.info(
        "[MIGRATE] Uploading migrated dataset (%.2f GB)...",
        total_size / 1e9,
    )
    api.upload_large_folder(
        repo_id=repo_id,
        folder_path=str(tmpdir),
        repo_type="dataset",
    )

    # Cleanup
    import shutil as _shutil2

    _shutil2.rmtree(tmpdir, ignore_errors=True)

    # Delete old shard files no longer needed
    old_shard_files = set()
    new_shard_files = set()
    for layer in hidden_layers:
        for old_si in range(len(old_shards)):
            old_shard_files.add(
                f"tensors/hidden_layer{layer:03d}"
                f"_shard{old_si:03d}.safetensors"
            )
        for new_si in range(len(new_boundaries)):
            new_shard_files.add(
                f"tensors/hidden_layer{layer:03d}"
                f"_shard{new_si:03d}.safetensors"
            )

    stale_files = old_shard_files - new_shard_files
    if stale_files:
        logger.info(
            "[MIGRATE] Deleting %d stale shard files...",
            len(stale_files),
        )
        for sf in stale_files:
            try:
                api.delete_file(
                    sf, repo_id=repo_id, repo_type="dataset",
                )
            except Exception as e:
                logger.warning(
                    "[MIGRATE] Failed to delete %s: %s", sf, e,
                )

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("[MIGRATE] Migration complete: %s", url)
    return url
