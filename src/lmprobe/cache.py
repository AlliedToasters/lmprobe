"""Activation caching for lmprobe.

Cache format v2: single safetensors file per prompt (#47)
    ~/.cache/lmprobe/
      {model_hash}/
        _model_name.txt              # Human-readable model name
        {prompt_hash}.safetensors    # All tensors for one prompt

Keys in safetensors file:
    layer_{i}                        Raw activations: (1, seq_len, hidden_dim)
    attention_mask                   Mask: (1, seq_len)
    pooled_{strategy}_layer_{i}      Pooled: (1, hidden_dim)
    perplexity                       Features: (3,)

Legacy format v1 (read-only, for backward compat, local backend only):
    {model_hash}/{prompt_hash}/
      layer_{i}.pt
      attention_mask.pt
      pooled_{strategy}/layer_{i}.pt
      perplexity.pt

Features:
    - Pluggable cache backends (local filesystem, S3)
    - Disk-full error handling (#44)
    - Pool-then-cache default (#45, applied in UnifiedCache)
    - float16 cache storage (#46) via LMPROBE_CACHE_DTYPE
    - Single safetensors file per prompt (#47)
    - cache_info() reporting (#48)
    - LRU eviction (#49) via LMPROBE_CACHE_MAX_GB (local backend only)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

from .cache_backends import CacheBackend, LocalCacheBackend

# =============================================================================
# Shard registry types (lazy caching for pull_dataset)
# =============================================================================

# In-memory caches for shard manifest/index JSON to avoid re-reading on every call.
_shard_manifests: dict[str, dict] = {}
_shard_indices: dict[str, dict] = {}

# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)


def enable_cache_logging(level: int = logging.INFO) -> None:
    """Enable cache logging to see cache hit/miss information.

    Parameters
    ----------
    level : int
        Logging level. Use logging.INFO for basic hit/miss info,
        logging.DEBUG for detailed cache operations.
    """
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


if os.getenv("LMPROBE_CACHE_DEBUG"):
    _level = (
        logging.DEBUG
        if os.getenv("LMPROBE_CACHE_DEBUG") == "debug"
        else logging.INFO
    )
    enable_cache_logging(_level)

# =============================================================================
# Configuration (#46 cache dtype, #49 cache size limit)
# =============================================================================

_CACHE_MAX_BYTES: int | None = None  # None = unlimited, -1 = disabled
_CACHE_DTYPE: torch.dtype | None = None

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Read env vars at import time
_env_max = os.getenv("LMPROBE_CACHE_MAX_GB")
if _env_max is not None:
    _v = float(_env_max)
    _CACHE_MAX_BYTES = -1 if _v == 0 else int(_v * 1024**3)

_env_dtype = os.getenv("LMPROBE_CACHE_DTYPE")
if _env_dtype is not None:
    _CACHE_DTYPE = _DTYPE_MAP.get(_env_dtype)
    if _CACHE_DTYPE is None:
        logger.warning(
            f"[CACHE] Unknown LMPROBE_CACHE_DTYPE={_env_dtype!r}, "
            f"ignoring. Valid: {list(_DTYPE_MAP.keys())}"
        )


def set_cache_limit(gb: float | None = None) -> None:
    """Set maximum cache size in GB for LRU eviction.

    This sets the target size cap. To actually enforce it, call
    ``evict()`` — eviction is intentionally decoupled from writes.

    Parameters
    ----------
    gb : float | None
        Maximum cache size in GB. None disables the limit.
        0 disables caching entirely.
    """
    global _CACHE_MAX_BYTES
    if gb is None:
        _CACHE_MAX_BYTES = None
    elif gb == 0:
        _CACHE_MAX_BYTES = -1
    else:
        _CACHE_MAX_BYTES = int(gb * 1024**3)


def set_cache_dtype(dtype: str | None = None) -> None:
    """Set cache storage dtype for 2x disk reduction.

    Parameters
    ----------
    dtype : str | None
        Storage dtype: "float16", "bfloat16", "float32", or None (no conversion).
    """
    global _CACHE_DTYPE
    if dtype is None:
        _CACHE_DTYPE = None
    elif dtype in _DTYPE_MAP:
        _CACHE_DTYPE = _DTYPE_MAP[dtype]
    else:
        raise ValueError(
            f"Unknown dtype: {dtype!r}. Valid: {list(_DTYPE_MAP.keys())}"
        )


# =============================================================================
# Backend management
# =============================================================================

_backend: CacheBackend | None = None


def _parse_backend_uri(uri: str) -> CacheBackend:
    """Parse a backend URI string into a CacheBackend instance.

    Supported schemes:
    - ``s3://bucket/prefix`` → S3CacheBackend
    - Local filesystem path → LocalCacheBackend
    """
    if uri.startswith("s3://"):
        from .cache_backends import S3CacheBackend

        # Parse s3://bucket/prefix
        rest = uri[5:]  # strip "s3://"
        parts = rest.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return S3CacheBackend(bucket=bucket, prefix=prefix)
    elif uri.startswith("gs://") or uri.startswith("gcs://"):
        raise ValueError(
            f"Unsupported cache backend URI scheme: {uri!r}. "
            "Only 's3://' and local filesystem paths are supported."
        )
    else:
        # Treat as local filesystem path
        return LocalCacheBackend(Path(uri))


def get_backend() -> CacheBackend:
    """Get the active cache backend, initializing from env vars if needed."""
    global _backend
    if _backend is not None:
        return _backend

    # Check LMPROBE_CACHE_BACKEND env var first
    env_backend = os.getenv("LMPROBE_CACHE_BACKEND")
    if env_backend:
        _backend = _parse_backend_uri(env_backend)
        return _backend

    # Default: local filesystem backend using LMPROBE_CACHE_DIR
    _backend = LocalCacheBackend(get_cache_dir())
    return _backend


def set_cache_backend(backend: CacheBackend | str | None) -> None:
    """Set the cache storage backend.

    Parameters
    ----------
    backend : CacheBackend | str | None
        - A CacheBackend instance
        - A URI string (e.g. "s3://bucket/prefix" or "/path/to/cache")
        - None to reset to default (lazy re-initialization)
    """
    global _backend
    if backend is None:
        _backend = None
    elif isinstance(backend, str):
        _backend = _parse_backend_uri(backend)
    elif isinstance(backend, CacheBackend):
        _backend = backend
    else:
        raise TypeError(
            f"Expected CacheBackend, str, or None, got {type(backend).__name__}"
        )


def _is_local_backend() -> bool:
    """Check if the current backend is a local filesystem backend."""
    return isinstance(get_backend(), LocalCacheBackend)


# =============================================================================
# Helpers
# =============================================================================


def _format_layers(layers: set[int] | list[int], max_show: int = 10) -> str:
    """Format layer indices for logging."""
    sorted_layers = sorted(layers)
    if len(sorted_layers) <= max_show:
        return str(sorted_layers)
    head = sorted_layers[: max_show // 2]
    tail = sorted_layers[-max_show // 2 :]
    return f"{head}...{tail} ({len(sorted_layers)} total)"


def get_cache_dir() -> Path:
    """Get the base cache directory, creating it if necessary."""
    cache_dir = os.getenv("LMPROBE_CACHE_DIR")
    path = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "lmprobe"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hash_string(s: str, length: int = 16) -> str:
    """Compute a short hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()[:length]


def _hash_prompts(prompts: list[str], length: int = 16) -> str:
    """Compute a deterministic hash of a prompt list."""
    serialized = json.dumps(prompts, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:length]


# =============================================================================
# Safetensors key helpers
# =============================================================================


def _raw_layer_key(layer: int) -> str:
    return f"layer_{layer}"


def _pooled_layer_key(pooling: str, layer: int) -> str:
    return f"pooled_{pooling}_layer_{layer}"


_ATTENTION_MASK_KEY = "attention_mask"
_PERPLEXITY_KEY = "perplexity"
_TOKEN_PERPLEXITY_KEY = "token_perplexity"
_TOKEN_IDS_KEY = "token_ids"
_LOGITS_KEY = "logits"
_LOGITS_TOP_K_VALUES_KEY = "logits_top_k_values"
_LOGITS_TOP_K_INDICES_KEY = "logits_top_k_indices"


def _parse_raw_layer_keys(keys: set[str] | list[str]) -> set[int]:
    """Extract raw layer indices from safetensors keys."""
    result = set()
    for k in keys:
        if k.startswith("layer_"):
            try:
                result.add(int(k[6:]))
            except ValueError:
                continue
    return result


def _parse_pooled_layer_keys(keys: set[str] | list[str], pooling: str) -> set[int]:
    """Extract pooled layer indices from safetensors keys."""
    prefix = f"pooled_{pooling}_layer_"
    result = set()
    for k in keys:
        if k.startswith(prefix):
            try:
                result.add(int(k[len(prefix) :]))
            except ValueError:
                continue
    return result


def _parse_all_pooled_keys(keys: set[str] | list[str]) -> dict[str, list[int]]:
    """Extract all pooling strategies and their layer indices from safetensors keys.

    Returns a dict mapping pooling strategy name to sorted list of layer indices.
    Parses keys matching ``pooled_{strategy}_layer_{i}``.
    """
    import re

    pattern = re.compile(r"^pooled_(.+)_layer_(\d+)$")
    result: dict[str, set[int]] = {}
    for k in keys:
        m = pattern.match(k)
        if m:
            strategy = m.group(1)
            layer_idx = int(m.group(2))
            result.setdefault(strategy, set()).add(layer_idx)
    return {s: sorted(layers) for s, layers in result.items()}


# =============================================================================
# Public cache introspection API
# =============================================================================


@dataclass
class CachedPromptInfo:
    """What's cached for a single prompt.

    Returned by :func:`discover_cached` to describe the available
    tensors without loading any data.
    """

    raw_layers: list[int]
    pooled: dict[str, list[int]]
    has_logits: bool
    logits_top_k: int | None
    has_perplexity: bool
    has_token_perplexity: bool
    num_tokens: int | None


def discover_cached(model_name: str, prompt: str) -> CachedPromptInfo | None:
    """Introspect what's cached for a model+prompt combination.

    Returns None if nothing is cached. Otherwise returns a
    :class:`CachedPromptInfo` describing available layers, pooling
    strategies, logits, etc.

    This is the public API for cache introspection — ``sharing.py``
    and other modules should use this rather than parsing internal
    cache key names.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    prompt : str
        The prompt text.

    Returns
    -------
    CachedPromptInfo | None
        Description of cached tensors, or None if nothing is cached.
    """
    backend = get_backend()
    key = _prompt_cache_key(model_name, prompt)

    # Check if sidecar files exist even when main file doesn't (#120)
    logits_sidecar = _prompt_logits_key(model_name, prompt)
    perplexity_sidecar = _prompt_perplexity_key(model_name, prompt)
    has_logits_sidecar = backend.exists(logits_sidecar)
    has_perplexity_sidecar = backend.exists(perplexity_sidecar)

    if not backend.exists(key):
        # If only sidecar files exist (logits/perplexity saved without activations)
        if has_logits_sidecar or has_perplexity_sidecar:
            has_logits = False
            has_topk = False
            logits_top_k = None
            if has_logits_sidecar:
                sidecar_keys = _get_tensor_keys_from_backend(logits_sidecar)
                has_logits = _LOGITS_KEY in sidecar_keys
                has_topk = (
                    _LOGITS_TOP_K_VALUES_KEY in sidecar_keys
                    and _LOGITS_TOP_K_INDICES_KEY in sidecar_keys
                )
                if has_topk:
                    try:
                        tensors = _load_tensors_from_backend(
                            logits_sidecar, [_LOGITS_TOP_K_VALUES_KEY]
                        )
                        logits_top_k = tensors[_LOGITS_TOP_K_VALUES_KEY].shape[-1]
                    except Exception:
                        pass
            has_token_ppl = False
            if has_perplexity_sidecar:
                ppl_keys = _get_tensor_keys_from_backend(perplexity_sidecar)
                has_token_ppl = _TOKEN_PERPLEXITY_KEY in ppl_keys
            return CachedPromptInfo(
                raw_layers=[],
                pooled={},
                has_logits=has_logits,
                logits_top_k=logits_top_k,
                has_perplexity=has_perplexity_sidecar,
                has_token_perplexity=has_token_ppl,
                num_tokens=None,
            )

        # v1 fallback: check for legacy directory (local only)
        if isinstance(backend, LocalCacheBackend):
            cache_dir = get_prompt_cache_dir(model_name, prompt)
            if not cache_dir.exists():
                # Try shard registry before giving up
                shard_info = _discover_from_shard(model_name, prompt)
                if shard_info is not None:
                    return shard_info
                return None
            # v1 has limited info — just raw layers and maybe perplexity
            raw_layers = sorted(get_prompt_cached_layers(cache_dir))
            if not raw_layers:
                # Try shard registry
                shard_info = _discover_from_shard(model_name, prompt)
                if shard_info is not None:
                    return shard_info
                return None
            has_perplexity = (cache_dir / "perplexity.pt").exists()
            # Check for v1 pooled dirs
            pooled: dict[str, list[int]] = {}
            for d in cache_dir.iterdir():
                if d.is_dir() and d.name.startswith("pooled_"):
                    strategy = d.name[len("pooled_"):]
                    layers = sorted(
                        int(f.stem.split("_")[1])
                        for f in d.glob("layer_*.pt")
                    )
                    if layers:
                        pooled[strategy] = layers
            # Try to get num_tokens from attention_mask
            num_tokens = None
            mask_path = cache_dir / "attention_mask.pt"
            if mask_path.exists():
                try:
                    mask = torch.load(mask_path, weights_only=True)
                    num_tokens = int(mask.sum().item())
                except Exception:
                    pass
            return CachedPromptInfo(
                raw_layers=raw_layers,
                pooled=pooled,
                has_logits=False,
                logits_top_k=None,
                has_perplexity=has_perplexity,
                has_token_perplexity=False,  # v1 never has token perplexity
                num_tokens=num_tokens,
            )
        # Non-local backend, no v1/v2 cache — try shard registry
        return _discover_from_shard(model_name, prompt)

    # v2: parse safetensors keys
    tensor_keys = _get_tensor_keys_from_backend(key)

    raw_layers = sorted(_parse_raw_layer_keys(tensor_keys))
    pooled = _parse_all_pooled_keys(tensor_keys)

    # Check for logits/perplexity in sidecar files first, then main file (#120)
    if has_logits_sidecar:
        sidecar_keys = _get_tensor_keys_from_backend(logits_sidecar)
        has_logits = _LOGITS_KEY in sidecar_keys
        has_topk = (
            _LOGITS_TOP_K_VALUES_KEY in sidecar_keys
            and _LOGITS_TOP_K_INDICES_KEY in sidecar_keys
        )
    else:
        has_logits = _LOGITS_KEY in tensor_keys
        has_topk = (
            _LOGITS_TOP_K_VALUES_KEY in tensor_keys
            and _LOGITS_TOP_K_INDICES_KEY in tensor_keys
        )

    has_perplexity = has_perplexity_sidecar or _PERPLEXITY_KEY in tensor_keys

    # Check for token-level perplexity in sidecar
    has_token_perplexity = False
    if has_perplexity_sidecar:
        ppl_sidecar_keys = _get_tensor_keys_from_backend(perplexity_sidecar)
        has_token_perplexity = _TOKEN_PERPLEXITY_KEY in ppl_sidecar_keys
    elif _TOKEN_PERPLEXITY_KEY in tensor_keys:
        has_token_perplexity = True

    # Determine top-k value by loading the shape of the topk values tensor
    logits_top_k = None
    if has_topk:
        try:
            source_key = logits_sidecar if has_logits_sidecar else key
            tensors = _load_tensors_from_backend(source_key, [_LOGITS_TOP_K_VALUES_KEY])
            logits_top_k = tensors[_LOGITS_TOP_K_VALUES_KEY].shape[-1]
        except Exception:
            pass

    # Get num_tokens from attention_mask
    num_tokens = None
    if _ATTENTION_MASK_KEY in tensor_keys:
        try:
            tensors = _load_tensors_from_backend(key, [_ATTENTION_MASK_KEY])
            num_tokens = int(tensors[_ATTENTION_MASK_KEY].sum().item())
        except Exception:
            pass

    return CachedPromptInfo(
        raw_layers=raw_layers,
        pooled=pooled,
        has_logits=has_logits,
        logits_top_k=logits_top_k,
        has_perplexity=has_perplexity,
        has_token_perplexity=has_token_perplexity,
        num_tokens=num_tokens,
    )


# =============================================================================
# Backend key helpers
# =============================================================================


def _prompt_cache_key(model_name: str, prompt: str) -> str:
    """Get the backend key for a prompt's safetensors file."""
    model_hash = _hash_string(model_name)
    prompt_hash = _hash_string(prompt)
    return f"{model_hash}/{prompt_hash}.safetensors"


def _prompt_logits_key(model_name: str, prompt: str) -> str:
    """Get the backend key for a prompt's logits sidecar file."""
    model_hash = _hash_string(model_name)
    prompt_hash = _hash_string(prompt)
    return f"{model_hash}/{prompt_hash}.logits.safetensors"


def _prompt_perplexity_key(model_name: str, prompt: str) -> str:
    """Get the backend key for a prompt's perplexity sidecar file."""
    model_hash = _hash_string(model_name)
    prompt_hash = _hash_string(prompt)
    return f"{model_hash}/{prompt_hash}.perplexity.safetensors"


def _model_name_key(model_name: str) -> str:
    """Get the backend key for the model name text file."""
    model_hash = _hash_string(model_name)
    return f"{model_hash}/_model_name.txt"


# =============================================================================
# Safe I/O (#44 disk-full handling, #46 dtype, #47 safetensors)
# =============================================================================


def _prepare_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Prepare tensor for cache storage: CPU, contiguous, optional dtype cast."""
    t = tensor.detach().cpu().contiguous()
    if _CACHE_DTYPE is not None and t.is_floating_point() and t.dtype != _CACHE_DTYPE:
        t = t.to(_CACHE_DTYPE)
    return t


def _save_tensors_to_backend(key: str, tensors: dict[str, torch.Tensor]) -> None:
    """Save tensors to the backend as safetensors bytes."""
    from safetensors.torch import save

    if _CACHE_MAX_BYTES == -1:
        return  # Caching disabled

    backend = get_backend()
    data = save(tensors)

    try:
        backend.write_bytes(key, data)
    except OSError as e:
        err = str(e)
        if "No space left" in err or "iostream error" in err or "enforce fail" in err:
            raise OSError(
                f"Disk full: could not write cache entry {key}. "
                f"Free up disk space or set LMPROBE_CACHE_MAX_GB to limit cache size."
            ) from e
        raise


def _load_tensors_from_backend(
    key: str, tensor_keys: list[str] | None = None
) -> dict[str, torch.Tensor]:
    """Load tensors from the backend.

    For LocalCacheBackend, uses safe_open for memory-mapped selective loading.
    For other backends, loads the full file from bytes.
    """
    backend = get_backend()

    if isinstance(backend, LocalCacheBackend):
        # Use safe_open for memory-mapped access on local files
        from safetensors import safe_open

        path = str(backend._path(key))
        result = {}
        with safe_open(path, framework="pt") as f:
            available = set(f.keys())
            if tensor_keys is None:
                for k in available:
                    result[k] = f.get_tensor(k)
            else:
                for k in tensor_keys:
                    if k not in available:
                        raise KeyError(
                            f"Corrupted or incomplete cache entry {key}: "
                            f"missing key {k!r}. Available keys: {sorted(available)}. "
                            f"Delete this entry and re-run to rebuild the cache."
                        )
                    result[k] = f.get_tensor(k)
        return result
    else:
        if tensor_keys is not None:
            # Selective loading via range reads — only download requested tensors
            return _load_tensors_selective(backend, key, tensor_keys)
        else:
            # No filter — download full file
            from safetensors.torch import load

            data = backend.read_bytes(key)
            return load(data)


# Safetensors dtype string → torch dtype mapping
_SAFETENSORS_DTYPE_MAP = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


_header_cache: dict[tuple[int, str], tuple[dict, int]] = {}
_HEADER_CACHE_MAXSIZE = 8192
_header_cache_lock = threading.Lock()


def _parse_safetensors_header(
    backend: CacheBackend, key: str
) -> tuple[dict, int]:
    """Parse the safetensors header from a backend entry.

    Results are cached by ``(id(backend), key)`` to avoid redundant
    range reads when the same file is accessed multiple times (e.g.
    once per layer during ``push_dataset`` consolidation).

    Returns
    -------
    header : dict
        Parsed JSON header mapping tensor names to metadata
        (dtype, shape, data_offsets).
    data_offset : int
        Byte offset where the tensor data begins (8 + header_size).
    """
    cache_key = (id(backend), key)
    with _header_cache_lock:
        cached = _header_cache.get(cache_key)
        if cached is not None:
            return cached

    import struct

    header_size_bytes = backend.read_range(key, 0, 8)
    (header_size,) = struct.unpack("<Q", header_size_bytes)

    if header_size > 100_000_000:
        raise ValueError(
            f"Safetensors header too large ({header_size} bytes) for {key}. "
            f"File may be corrupted."
        )

    header_bytes = backend.read_range(key, 8, 8 + header_size)
    header = json.loads(header_bytes)
    result = (header, 8 + header_size)

    with _header_cache_lock:
        if len(_header_cache) >= _HEADER_CACHE_MAXSIZE:
            _header_cache.pop(next(iter(_header_cache)))
        _header_cache[cache_key] = result
    return result


def clear_header_cache() -> None:
    """Clear the safetensors header cache."""
    _header_cache.clear()


_mask_cache: dict[tuple[int, str], torch.Tensor] = {}
_MASK_CACHE_MAXSIZE = 8192
_mask_cache_lock = threading.Lock()


def _load_tensors_selective(
    backend: CacheBackend,
    key: str,
    tensor_keys: list[str],
) -> dict[str, torch.Tensor]:
    """Load specific tensors from a remote backend using range reads.

    Instead of downloading the entire safetensors file, this reads only
    the header (to get byte offsets) and then fetches each requested
    tensor individually via range reads.

    Attention masks are cached by ``(id(backend), key)`` since they are
    identical across all layer loads for the same prompt file.
    """
    header, data_start = _parse_safetensors_header(backend, key)
    available = {k for k in header if k != "__metadata__"}

    result = {}
    for tensor_name in tensor_keys:
        if tensor_name not in available:
            raise KeyError(
                f"Corrupted or incomplete cache entry {key}: "
                f"missing key {tensor_name!r}. "
                f"Available keys: {sorted(available)}. "
                f"Delete this entry and re-run to rebuild the cache."
            )

        # Check mask cache for attention_mask tensors
        if tensor_name == _ATTENTION_MASK_KEY:
            mask_cache_key = (id(backend), key)
            with _mask_cache_lock:
                cached_mask = _mask_cache.get(mask_cache_key)
            if cached_mask is not None:
                result[tensor_name] = cached_mask.clone()
                continue

        meta = header[tensor_name]
        dtype_str = meta["dtype"]
        shape = meta["shape"]
        start, end = meta["data_offsets"]

        torch_dtype = _SAFETENSORS_DTYPE_MAP.get(dtype_str)
        if torch_dtype is None:
            raise ValueError(
                f"Unsupported safetensors dtype {dtype_str!r} "
                f"for tensor {tensor_name!r} in {key}"
            )

        raw_bytes = backend.read_range(key, data_start + start, data_start + end)
        tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch_dtype)
        result[tensor_name] = tensor.reshape(shape)

        # Cache attention masks for reuse across layer iterations
        if tensor_name == _ATTENTION_MASK_KEY:
            with _mask_cache_lock:
                if len(_mask_cache) >= _MASK_CACHE_MAXSIZE:
                    _mask_cache.pop(next(iter(_mask_cache)))
                _mask_cache[(id(backend), key)] = tensor.reshape(shape).clone()

    return result


def clear_mask_cache() -> None:
    """Clear the attention mask cache."""
    _mask_cache.clear()


def _clear_selective_caches() -> None:
    """Clear both the header and attention mask caches."""
    clear_header_cache()
    clear_mask_cache()


def _get_tensor_keys_from_backend(key: str) -> set[str]:
    """Get tensor key names from a backend entry without loading data."""
    backend = get_backend()

    if isinstance(backend, LocalCacheBackend):
        from safetensors import safe_open

        path = str(backend._path(key))
        with safe_open(path, framework="pt") as f:
            return set(f.keys())
    else:
        return _get_tensor_keys_header_only(backend, key)


def _get_tensor_keys_header_only(backend: CacheBackend, key: str) -> set[str]:
    """Get tensor key names by reading only the safetensors header.

    Safetensors files start with an 8-byte little-endian header size,
    followed by a JSON header that lists all tensor names and metadata.
    This avoids downloading the full file (which can be GBs).
    """
    header, _ = _parse_safetensors_header(backend, key)
    return {k for k in header if k != "__metadata__"}


def _list_model_cache_keys(model_name: str) -> set[str]:
    """List all cache keys for a model using a single LIST call.

    Returns a set of backend keys (e.g. "abc123/def456.safetensors").
    Much faster than per-prompt HEAD requests for large prompt sets.
    """
    backend = get_backend()
    model_hash = _hash_string(model_name)
    return set(backend.list_keys(model_hash))


def batch_check_cache_status(
    model_name: str,
    prompts: list[str],
    required_layers: set[int],
    pooling: str | None = None,
    compute_perplexity: bool = False,
    cache_logits: bool = False,
    logit_top_k: int | None = None,
) -> tuple[list[str], list[str], list[str], int, set[int] | None]:
    """Check cache status for many prompts using batch operations.

    Uses a single LIST call to discover existing keys, then reads
    safetensors headers (not full files) when needed to check layer
    coverage. This is O(1) LIST + O(cached) header reads instead
    of O(N) HEAD requests.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    prompts : list[str]
        Prompts to check.
    required_layers : set[int]
        Layer indices that must be cached.
    pooling : str or None
        Pooling strategy. If None, checks raw layers.
    compute_perplexity : bool
        Whether perplexity features are needed.
    cache_logits : bool
        Whether logits need to be cached.
    logit_top_k : int or None
        Top-k for logit caching.

    Returns
    -------
    tuple
        - need_activations: prompts needing activation extraction
        - need_perplexity: prompts needing perplexity extraction
        - need_logits: prompts needing logit extraction
        - partial_cache_count: number of prompts with partial layer cache
        - partial_cache_found_layers: example set of cached layers (or None)
    """
    backend = get_backend()
    model_hash = _hash_string(model_name)

    # Single LIST call to get all existing keys for this model
    existing_keys = _list_model_cache_keys(model_name)

    # Local cache for tensor keys to avoid re-reading the same file
    _tensor_keys_cache: dict[str, set[str]] = {}

    def _cached_tensor_keys(key: str) -> set[str]:
        if key not in _tensor_keys_cache:
            _tensor_keys_cache[key] = _get_tensor_keys_from_backend(key)
        return _tensor_keys_cache[key]

    need_activations: list[str] = []
    need_perplexity: list[str] = []
    need_logits: list[str] = []
    partial_cache_count = 0
    partial_cache_found_layers: set[int] | None = None

    for prompt in prompts:
        prompt_hash = _hash_string(prompt)
        main_key = f"{model_hash}/{prompt_hash}.safetensors"
        logits_key = f"{model_hash}/{prompt_hash}.logits.safetensors"
        ppl_key = f"{model_hash}/{prompt_hash}.perplexity.safetensors"

        # --- Activation check ---
        act_cached = False
        cached_layers: set[int] | None = None
        if main_key in existing_keys:
            tensor_keys = _cached_tensor_keys(main_key)
            if pooling is not None:
                cached_layers = _parse_pooled_layer_keys(tensor_keys, pooling)
                act_cached = required_layers.issubset(cached_layers)
            else:
                cached_layers = _parse_raw_layer_keys(tensor_keys)
                act_cached = (
                    required_layers.issubset(cached_layers)
                    and _ATTENTION_MASK_KEY in tensor_keys
                )
        elif isinstance(backend, LocalCacheBackend):
            # v1 fallback for local backend
            if pooling is not None:
                act_cached = is_prompt_pooled_cached(
                    model_name, prompt, required_layers, pooling
                )
            else:
                act_cached = is_prompt_fully_cached(
                    model_name, prompt, required_layers
                )
        else:
            # Shard registry fallback
            shard_info = _discover_from_shard(model_name, prompt)
            if shard_info is not None:
                if pooling is not None and pooling in shard_info.pooled:
                    act_cached = required_layers.issubset(
                        set(shard_info.pooled[pooling])
                    )
                elif pooling is None and shard_info.raw_layers:
                    act_cached = required_layers.issubset(
                        set(shard_info.raw_layers)
                    )

        if not act_cached:
            need_activations.append(prompt)
            # Check for partial cache (reuse cached_layers from above)
            if cached_layers and len(cached_layers) > 0:
                partial_cache_count += 1
                if partial_cache_found_layers is None:
                    partial_cache_found_layers = cached_layers

        # --- Perplexity check ---
        if compute_perplexity:
            ppl_cached = ppl_key in existing_keys
            if not ppl_cached and main_key in existing_keys:
                tensor_keys = _cached_tensor_keys(main_key)
                ppl_cached = _PERPLEXITY_KEY in tensor_keys
            if not ppl_cached and isinstance(backend, LocalCacheBackend):
                ppl_cached = is_prompt_perplexity_cached(model_name, prompt)
            if not ppl_cached:
                need_perplexity.append(prompt)

        # --- Logits check ---
        if cache_logits:
            logits_cached = False
            if logits_key in existing_keys:
                tensor_keys = _cached_tensor_keys(logits_key)
                if logit_top_k is not None:
                    logits_cached = (
                        _LOGITS_TOP_K_VALUES_KEY in tensor_keys
                        and _LOGITS_TOP_K_INDICES_KEY in tensor_keys
                    )
                else:
                    logits_cached = _LOGITS_KEY in tensor_keys
            if not logits_cached and main_key in existing_keys:
                tensor_keys = _cached_tensor_keys(main_key)
                if logit_top_k is not None:
                    logits_cached = (
                        _LOGITS_TOP_K_VALUES_KEY in tensor_keys
                        and _LOGITS_TOP_K_INDICES_KEY in tensor_keys
                    )
                else:
                    logits_cached = _LOGITS_KEY in tensor_keys
            if not logits_cached:
                need_logits.append(prompt)

    return (
        need_activations,
        need_perplexity,
        need_logits,
        partial_cache_count,
        partial_cache_found_layers,
    )


def _merge_save_backend(key: str, new_tensors: dict[str, torch.Tensor]) -> None:
    """Load existing entry, merge with new tensors, and save."""
    backend = get_backend()

    if isinstance(backend, LocalCacheBackend):
        # For local backend, use the original _merge_save path which
        # provides atomic writes via temp file + rename
        path = backend._path(key)
        existing = _load_sf_all(path) if path.exists() else {}
        existing.update(new_tensors)
        _safe_save_file(existing, path)
    else:
        # For non-local backends (S3 etc.), use bytes-level I/O
        if backend.exists(key):
            existing = _load_tensors_from_backend(key)
        else:
            existing = {}
        existing.update(new_tensors)
        _save_tensors_to_backend(key, existing)


# Keep legacy functions for backward compatibility (used by tests and v1 paths)
def _safe_save_file(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """Atomically save tensors to safetensors with disk-full error handling (#44).

    Writes to a temp file then renames for crash safety.
    """
    from safetensors.torch import save_file

    if _CACHE_MAX_BYTES == -1:
        return  # Caching disabled

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        save_file(tensors, str(tmp_path))
        tmp_path.rename(path)
    except (OSError, RuntimeError) as e:
        # Clean up partial file
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        err = str(e)
        if "No space left" in err or "iostream error" in err or "enforce fail" in err:
            raise OSError(
                f"Disk full: could not write cache file {path}. "
                f"Free up disk space or set LMPROBE_CACHE_MAX_GB to limit cache size. "
                f"Current cache dir: {get_cache_dir()}"
            ) from e
        raise


def _load_sf_keys(path: Path) -> set[str]:
    """Get tensor key names without loading data."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return set(f.keys())


def _load_sf_tensor(path: Path, key: str) -> torch.Tensor:
    """Load a single tensor by key."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return f.get_tensor(key)


def _load_sf_tensors(path: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    """Load multiple tensors by key."""
    from safetensors import safe_open

    result = {}
    with safe_open(str(path), framework="pt") as f:
        available = set(f.keys())
        for k in keys:
            if k not in available:
                raise KeyError(
                    f"Corrupted or incomplete cache file {path}: "
                    f"missing key {k!r}. Available keys: {sorted(available)}. "
                    f"Delete this file and re-run to rebuild the cache."
                )
            result[k] = f.get_tensor(k)
    return result


def _load_sf_all(path: Path) -> dict[str, torch.Tensor]:
    """Load all tensors from safetensors file."""
    from safetensors.torch import load_file

    return load_file(str(path))


def _merge_save(path: Path, new_tensors: dict[str, torch.Tensor]) -> None:
    """Load existing safetensors, merge with new tensors, and save atomically."""
    existing = _load_sf_all(path) if path.exists() else {}
    existing.update(new_tensors)
    _safe_save_file(existing, path)


# =============================================================================
# Path functions
# =============================================================================


def get_prompt_cache_dir(model_name: str, prompt: str) -> Path:
    """Get the legacy cache directory for a single prompt.

    Returns the v1 directory path. For v2 safetensors path, use
    get_prompt_cache_path().
    """
    base = get_cache_dir()
    model_hash = _hash_string(model_name)
    prompt_hash = _hash_string(prompt)
    return base / model_hash / prompt_hash


def get_prompt_cache_path(model_name: str, prompt: str) -> Path:
    """Get the safetensors cache file path for a single prompt (v2 format)."""
    base = get_cache_dir()
    model_hash = _hash_string(model_name)
    prompt_hash = _hash_string(prompt)
    return base / model_hash / f"{prompt_hash}.safetensors"


def _register_model(model_name: str) -> None:
    """Record model hash -> name mapping for cache_info()."""
    backend = get_backend()
    key = _model_name_key(model_name)
    if not backend.exists(key):
        backend.write_text(key, model_name)


def _read_model_name(model_dir: Path) -> str | None:
    """Read model name from a model cache directory."""
    name_file = model_dir / "_model_name.txt"
    if name_file.exists():
        return name_file.read_text().strip()
    return None


# =============================================================================
# Prompt manifest sidecar (#162)
# =============================================================================

_manifest_lock = threading.Lock()


def _manifest_key(model_name: str) -> str:
    """Backend key for the prompt manifest JSONL file."""
    model_hash = _hash_string(model_name)
    return f"{model_hash}/_manifest.jsonl"


def _append_manifest_entry(
    model_name: str,
    prompt: str,
    num_tokens: int | None = None,
) -> None:
    """Append a single entry to the prompt manifest.

    Thread-safe. Silently ignores errors so manifest issues never block
    cache writes.
    """
    try:
        backend = get_backend()
        key = _manifest_key(model_name)
        prompt_hash = _hash_string(prompt)
        entry = {
            "hash": prompt_hash,
            "prompt": prompt,
            "num_tokens": num_tokens,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"

        with _manifest_lock:
            existing = ""
            if backend.exists(key):
                try:
                    existing = backend.read_text(key)
                except Exception:
                    pass
            backend.write_text(key, existing + line)
    except Exception:
        # Manifest is advisory — never fail a cache write because of it.
        logger.debug("[CACHE] Failed to append manifest entry for %r", prompt)


@dataclass
class ManifestEntry:
    """A single entry from the prompt manifest."""

    hash: str
    prompt: str
    num_tokens: int | None
    cached_at: str


def read_manifest(model_name: str) -> list[ManifestEntry]:
    """Read all manifest entries for a model.

    Returns an empty list if no manifest exists (pre-feature caches).
    Entries are advisory — the corresponding safetensors file may have
    been deleted externally.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.

    Returns
    -------
    list[ManifestEntry]
        Manifest entries in append order.
    """
    backend = get_backend()
    key = _manifest_key(model_name)
    if not backend.exists(key):
        return []

    try:
        text = backend.read_text(key)
    except Exception:
        return []

    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            entries.append(
                ManifestEntry(
                    hash=d["hash"],
                    prompt=d["prompt"],
                    num_tokens=d.get("num_tokens"),
                    cached_at=d.get("cached_at", ""),
                )
            )
        except (json.JSONDecodeError, KeyError):
            continue
    return entries


def list_cached_prompts(model_name: str, verify: bool = False) -> list[ManifestEntry]:
    """List prompts known to the manifest for a model.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    verify : bool
        If True, check that each prompt's safetensors file still exists
        and filter out stale entries.

    Returns
    -------
    list[ManifestEntry]
        Deduplicated manifest entries (latest entry per prompt hash wins).
    """
    raw = read_manifest(model_name)

    # Deduplicate: keep last entry per hash (latest write wins)
    seen: dict[str, ManifestEntry] = {}
    for entry in raw:
        seen[entry.hash] = entry
    entries = list(seen.values())

    if verify:
        backend = get_backend()
        model_hash = _hash_string(model_name)
        entries = [
            e
            for e in entries
            if backend.exists(f"{model_hash}/{e.hash}.safetensors")
        ]

    return entries


# =============================================================================
# Shard Registry I/O (lazy caching for pull_dataset)
# =============================================================================


def _shard_manifest_key(model_name: str) -> str:
    """Backend key for the shard manifest JSON."""
    model_hash = _hash_string(model_name)
    return f"{model_hash}/_shard_manifest.json"


def _shard_index_key(model_name: str) -> str:
    """Backend key for the shard index JSON."""
    model_hash = _hash_string(model_name)
    return f"{model_hash}/_shard_index.json"


def write_shard_registry(
    model_name: str,
    manifest: dict,
    index: dict,
) -> None:
    """Write shard manifest and index to the cache backend.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    manifest : dict
        Repo-level metadata: model name, tensor descriptors, shard file paths.
    index : dict
        Per-prompt offsets: prompt_hash -> {shard_index, row_offset, ...}.
    """
    _register_model(model_name)
    backend = get_backend()

    m_key = _shard_manifest_key(model_name)
    i_key = _shard_index_key(model_name)

    backend.write_text(m_key, json.dumps(manifest))
    backend.write_text(i_key, json.dumps(index))

    # Update in-memory caches
    _shard_manifests[model_name] = manifest
    _shard_indices[model_name] = index


def _load_shard_manifest(model_name: str) -> dict | None:
    """Load shard manifest, using in-memory cache if available."""
    if model_name in _shard_manifests:
        return _shard_manifests[model_name]

    backend = get_backend()
    key = _shard_manifest_key(model_name)
    if not backend.exists(key):
        return None

    manifest = json.loads(backend.read_text(key))
    _shard_manifests[model_name] = manifest
    return manifest


def _load_shard_index(model_name: str) -> dict | None:
    """Load shard index, using in-memory cache if available."""
    if model_name in _shard_indices:
        return _shard_indices[model_name]

    backend = get_backend()
    key = _shard_index_key(model_name)
    if not backend.exists(key):
        return None

    index = json.loads(backend.read_text(key))
    _shard_indices[model_name] = index
    return index


def _lookup_shard(
    model_name: str, prompt: str
) -> dict | None:
    """Look up shard info for a prompt from the shard index.

    Returns
    -------
    dict | None
        Entry from the shard index with keys like shard_index, row_offset,
        token_offset, num_tokens. None if not found.
    """
    index = _load_shard_index(model_name)
    if index is None:
        return None
    prompt_hash = _hash_string(prompt)
    return index.get(prompt_hash)


def _discover_from_shard(model_name: str, prompt: str) -> CachedPromptInfo | None:
    """Build CachedPromptInfo from shard manifest metadata (no data loading)."""
    entry = _lookup_shard(model_name, prompt)
    if entry is None:
        return None

    manifest = _load_shard_manifest(model_name)
    if manifest is None:
        return None

    tensor_descs = manifest.get("tensors", {})
    raw_layers: list[int] = []
    pooled: dict[str, list[int]] = {}
    has_logits = False
    logits_top_k = None
    has_perplexity = False
    num_tokens = entry.get("num_tokens")

    for t_type, t_info in tensor_descs.items():
        layers = t_info.get("layers", [])
        storage = t_info.get("storage", "pooled")
        if t_type == "hidden_layers":
            if storage == "full_sequence":
                raw_layers = sorted(layers)
                # full_sequence shards also provide last_token pooled
                pooling = t_info.get("pooling", "last_token")
                pooled[pooling] = sorted(layers)
            else:
                pooling = t_info.get("pooling", "last_token")
                pooled[pooling] = sorted(layers)
        elif t_type == "logits_topk":
            has_logits = True
            logits_top_k = t_info.get("k")

    return CachedPromptInfo(
        raw_layers=raw_layers,
        pooled=pooled,
        has_logits=has_logits,
        logits_top_k=logits_top_k,
        has_perplexity=has_perplexity,
        has_token_perplexity=False,  # shard registry doesn't track token perplexity
        num_tokens=num_tokens,
    )


def _load_raw_from_shard(
    model_name: str, prompt: str, layers: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load raw activations for a prompt by slicing from a shard file.

    Returns (activations, attention_mask) like load_prompt_activations.
    """
    import re

    from safetensors import safe_open

    entry = _lookup_shard(model_name, prompt)
    if entry is None:
        raise FileNotFoundError(
            f"No shard entry found for prompt: {prompt!r}"
        )

    manifest = _load_shard_manifest(model_name)
    if manifest is None:
        raise FileNotFoundError("No shard manifest found")

    t_info = manifest["tensors"].get("hidden_layers")
    if t_info is None:
        raise FileNotFoundError("No hidden_layers in shard manifest")

    storage = t_info.get("storage", "pooled")
    if storage != "full_sequence":
        raise FileNotFoundError(
            "Raw activations require full_sequence storage, "
            f"but shard has storage={storage!r}"
        )

    shard_idx = entry.get("shard_index_hidden", entry.get("shard_index"))
    tok_off = entry.get("token_offset_hidden", entry.get("token_offset"))
    num_tok = entry["num_tokens"]

    shards = t_info["shards"]
    if shard_idx >= len(shards):
        raise FileNotFoundError(
            f"Shard index {shard_idx} out of range (have {len(shards)} shards)"
        )

    # Convert global token offset to shard-local offset.
    # The parquet index stores cumulative (global) token offsets, but each
    # shard file only contains its own tokens.
    shard_token_base = sum(
        s.get("num_tokens", 0) for s in shards[:shard_idx]
    )
    tok_off -= shard_token_base

    layout = t_info.get("layout")

    if layout == "per_layer":
        # v1.1+ per-layer layout: each layer in its own file
        layer_slices = []
        for layer in layers:
            per_layer_paths = shards[shard_idx].get("per_layer_paths", {})
            # per_layer_paths keys may be strings (from JSON) or ints
            layer_path = per_layer_paths.get(layer) or per_layer_paths.get(
                str(layer)
            )
            if layer_path is None:
                continue
            if not Path(layer_path).exists():
                raise FileNotFoundError(
                    f"Per-layer shard file not found: {layer_path}. "
                    "Re-run pull_dataset() to re-download."
                )
            with safe_open(layer_path, framework="pt") as f:
                key = f"hidden.layer_{layer}"
                chunk = f.get_tensor(key)[tok_off : tok_off + num_tok]
                layer_slices.append((layer, chunk))
    else:
        # v1.0 co-located layout
        shard_path = shards[shard_idx].get("local_path")
        if shard_path is None or not Path(shard_path).exists():
            raise FileNotFoundError(
                f"Shard file not found: {shard_path}. "
                "The HF hub cache may have been evicted. "
                "Re-run pull_dataset() to re-download."
            )

        with safe_open(shard_path, framework="pt") as f:
            sf_keys = list(f.keys())
            layer_slices = []
            for sf_key in sf_keys:
                match = re.match(r"^hidden\.layer_(\d+)$", sf_key)
                if not match:
                    continue
                layer = int(match.group(1))
                if layer not in layers:
                    continue
                chunk = f.get_tensor(sf_key)[tok_off : tok_off + num_tok]
                layer_slices.append((layer, chunk))

    layer_slices.sort(key=lambda x: x[0])
    raw_act = torch.cat([ls[1] for ls in layer_slices], dim=-1)
    raw_act = raw_act.unsqueeze(0)  # (1, num_tok, total_dim)
    mask = torch.ones(1, num_tok, dtype=torch.long)
    return raw_act, mask


def _load_pooled_from_shard(
    model_name: str, prompt: str, layers: list[int], pooling: str
) -> torch.Tensor:
    """Load pooled activations for a prompt by slicing from a shard file.

    Returns tensor with shape (1, n_layers * hidden_dim).
    """
    import re

    from safetensors import safe_open

    entry = _lookup_shard(model_name, prompt)
    if entry is None:
        raise FileNotFoundError(
            f"No shard entry found for prompt: {prompt!r}"
        )

    manifest = _load_shard_manifest(model_name)
    if manifest is None:
        raise FileNotFoundError("No shard manifest found")

    t_info = manifest["tensors"].get("hidden_layers")
    if t_info is None:
        raise FileNotFoundError("No hidden_layers in shard manifest")

    storage = t_info.get("storage", "pooled")
    layout = t_info.get("layout")
    shard_idx = entry.get("shard_index_hidden", entry.get("shard_index"))

    shards = t_info["shards"]
    if shard_idx >= len(shards):
        raise FileNotFoundError(
            f"Shard index {shard_idx} out of range (have {len(shards)} shards)"
        )

    if layout == "per_layer":
        # v1.1+ per-layer layout
        per_layer_paths = shards[shard_idx].get("per_layer_paths", {})
        layer_slices = []

        if storage == "full_sequence":
            tok_off = entry.get("token_offset_hidden", entry.get("token_offset"))
            num_tok = entry["num_tokens"]
            # Convert global token offset to shard-local offset
            shard_token_base = sum(
                s.get("num_tokens", 0) for s in shards[:shard_idx]
            )
            tok_off -= shard_token_base
            for layer in layers:
                layer_path = per_layer_paths.get(
                    layer
                ) or per_layer_paths.get(str(layer))
                if layer_path is None:
                    continue
                if not Path(layer_path).exists():
                    raise FileNotFoundError(
                        f"Per-layer shard file not found: {layer_path}. "
                        "Re-run pull_dataset() to re-download."
                    )
                with safe_open(layer_path, framework="pt") as f:
                    key = f"hidden.layer_{layer}"
                    last_tok = f.get_tensor(key)[
                        tok_off + num_tok - 1 : tok_off + num_tok
                    ]
                    layer_slices.append((layer, last_tok))
        else:
            row_offset = entry.get("row_offset_hidden", entry.get("row_offset"))
            for layer in layers:
                layer_path = per_layer_paths.get(
                    layer
                ) or per_layer_paths.get(str(layer))
                if layer_path is None:
                    continue
                if not Path(layer_path).exists():
                    raise FileNotFoundError(
                        f"Per-layer shard file not found: {layer_path}. "
                        "Re-run pull_dataset() to re-download."
                    )
                with safe_open(layer_path, framework="pt") as f:
                    key = f"hidden.layer_{layer}"
                    row = f.get_tensor(key)[row_offset : row_offset + 1]
                    layer_slices.append((layer, row))
    else:
        # v1.0 co-located layout
        shard_path = shards[shard_idx].get("local_path")
        if shard_path is None or not Path(shard_path).exists():
            raise FileNotFoundError(
                f"Shard file not found: {shard_path}. "
                "The HF hub cache may have been evicted. "
                "Re-run pull_dataset() to re-download."
            )

        with safe_open(shard_path, framework="pt") as f:
            sf_keys = list(f.keys())

            if storage == "full_sequence":
                tok_off = entry["token_offset"]
                num_tok = entry["num_tokens"]
                layer_slices = []
                for sf_key in sf_keys:
                    match = re.match(r"^hidden\.layer_(\d+)$", sf_key)
                    if not match:
                        continue
                    layer = int(match.group(1))
                    if layer not in layers:
                        continue
                    last_tok = f.get_tensor(sf_key)[
                        tok_off + num_tok - 1 : tok_off + num_tok
                    ]
                    layer_slices.append((layer, last_tok))
            else:
                row_offset = entry["row_offset"]
                layer_slices = []
                for sf_key in sf_keys:
                    match = re.match(r"^hidden\.layer_(\d+)$", sf_key)
                    if not match:
                        continue
                    layer = int(match.group(1))
                    if layer not in layers:
                        continue
                    row = f.get_tensor(sf_key)[row_offset : row_offset + 1]
                    layer_slices.append((layer, row))

    layer_slices.sort(key=lambda x: x[0])
    return torch.cat([ls[1] for ls in layer_slices], dim=-1)


def _load_logits_from_shard(
    model_name: str, prompt: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load top-k logits for a prompt from a shard file.

    Returns (values, indices).
    """
    from safetensors import safe_open

    entry = _lookup_shard(model_name, prompt)
    if entry is None:
        raise FileNotFoundError(
            f"No shard entry found for prompt: {prompt!r}"
        )

    manifest = _load_shard_manifest(model_name)
    if manifest is None:
        raise FileNotFoundError("No shard manifest found")

    t_info = manifest["tensors"].get("logits_topk")
    if t_info is None:
        raise FileNotFoundError("No logits_topk in shard manifest")

    shard_idx = entry.get("shard_index_logits", entry.get("shard_index"))
    row_offset = entry.get("row_offset_logits", entry.get("row_offset"))

    shards = t_info["shards"]
    if shard_idx >= len(shards):
        raise FileNotFoundError(
            f"Shard index {shard_idx} out of range (have {len(shards)} shards)"
        )

    shard_path = shards[shard_idx]["local_path"]
    if not Path(shard_path).exists():
        raise FileNotFoundError(
            f"Shard file not found: {shard_path}. "
            "The HF hub cache may have been evicted. "
            "Re-run pull_dataset() to re-download."
        )

    with safe_open(shard_path, framework="pt") as f:
        values = f.get_tensor("logits_topk.values")[row_offset : row_offset + 1]
        indices = f.get_tensor("logits_topk.indices")[row_offset : row_offset + 1]

    return values, indices


# =============================================================================
# Per-Prompt Cache (v2 safetensors with v1 fallback)
# =============================================================================


def get_prompt_cached_layers(cache_dir: Path) -> set[int]:
    """Get cached raw layer indices for a prompt.

    Checks v2 safetensors format first, then v1 .pt format.
    """
    backend = get_backend()

    # v2: check via backend
    # Derive the key from the cache_dir path
    sf_path = cache_dir.with_suffix(".safetensors")
    if isinstance(backend, LocalCacheBackend):
        # For local backend, check if safetensors file exists via Path
        if sf_path.exists():
            keys = _load_sf_keys(sf_path)
            return _parse_raw_layer_keys(keys)
    else:
        # For non-local, compute key from model/prompt hashes
        try:
            rel = sf_path.relative_to(get_cache_dir())
            key = str(rel)
            if backend.exists(key):
                tensor_keys = _get_tensor_keys_from_backend(key)
                return _parse_raw_layer_keys(tensor_keys)
        except ValueError:
            pass

    # v1: check .pt files in directory (local only)
    if not cache_dir.exists():
        return set()
    cached = set()
    for f in cache_dir.glob("layer_*.pt"):
        try:
            cached.add(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return cached


def get_prompt_cached_raw_layers(
    model_name: str, prompt: str
) -> set[int] | None:
    """Return the set of raw layer indices cached for a prompt, or None if no cache exists."""
    backend = get_backend()
    key = _prompt_cache_key(model_name, prompt)

    if backend.exists(key):
        tensor_keys = _get_tensor_keys_from_backend(key)
        return _parse_raw_layer_keys(tensor_keys)

    if isinstance(backend, LocalCacheBackend):
        cache_dir = get_prompt_cache_dir(model_name, prompt)
        if cache_dir.exists():
            return get_prompt_cached_layers(cache_dir)

    # Shard registry fallback
    shard_info = _discover_from_shard(model_name, prompt)
    if shard_info is not None and shard_info.raw_layers:
        return set(shard_info.raw_layers)

    return None


def is_prompt_fully_cached(
    model_name: str, prompt: str, required_layers: set[int]
) -> bool:
    """Check if a prompt has all required raw layers cached."""
    backend = get_backend()
    key = _prompt_cache_key(model_name, prompt)

    # v2: check via backend
    if backend.exists(key):
        tensor_keys = _get_tensor_keys_from_backend(key)
        cached = _parse_raw_layer_keys(tensor_keys)
        has_mask = _ATTENTION_MASK_KEY in tensor_keys
        return required_layers.issubset(cached) and has_mask

    # v1 fallback (local only)
    if isinstance(backend, LocalCacheBackend):
        cache_dir = get_prompt_cache_dir(model_name, prompt)
        cached = get_prompt_cached_layers(cache_dir)
        has_mask = (cache_dir / "attention_mask.pt").exists()
        if required_layers.issubset(cached) and has_mask:
            return True

    # Shard registry fallback (full_sequence storage has raw layers)
    shard_info = _discover_from_shard(model_name, prompt)
    if shard_info is not None and shard_info.raw_layers:
        return required_layers.issubset(set(shard_info.raw_layers))

    return False


def load_prompt_activations(
    model_name: str, prompt: str, layers: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load cached raw activations for a single prompt.

    Returns (activations, attention_mask). Checks v2 then v1, then shard registry.
    """
    backend = get_backend()
    key = _prompt_cache_key(model_name, prompt)

    if backend.exists(key):
        # Touch for LRU
        backend.touch(key)
        keys_to_load = [_raw_layer_key(li) for li in layers] + [_ATTENTION_MASK_KEY]
        tensors = _load_tensors_from_backend(key, keys_to_load)
        layer_acts = [tensors[_raw_layer_key(li)] for li in layers]
        activations = torch.cat(layer_acts, dim=-1)
        return activations, tensors[_ATTENTION_MASK_KEY]

    # v1 fallback (local only)
    if isinstance(backend, LocalCacheBackend):
        cache_dir = get_prompt_cache_dir(model_name, prompt)
        if cache_dir.exists() and (cache_dir / "attention_mask.pt").exists():
            layer_acts = []
            for layer in layers:
                acts = torch.load(cache_dir / f"layer_{layer}.pt", weights_only=True)
                layer_acts.append(acts)
            activations = torch.cat(layer_acts, dim=-1)
            mask = torch.load(cache_dir / "attention_mask.pt", weights_only=True)
            return activations, mask

    # Shard registry fallback
    entry = _lookup_shard(model_name, prompt)
    if entry is not None:
        return _load_raw_from_shard(model_name, prompt, layers)

    raise FileNotFoundError(f"No cached activations found for prompt: {prompt!r}")


def load_layer_across_prompts(
    model_name: str,
    prompts: list[str],
    layer: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Load a single layer's raw activations across multiple prompts.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    prompts : list[str]
        Prompt texts to load.
    layer : int
        Layer index to load.

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor]]
        (activations_list, masks_list) where each element has shape
        (1, seq_len_i, hidden_dim) and (1, seq_len_i) respectively.

    Raises
    ------
    FileNotFoundError
        If any prompt's cache is missing.
    """
    all_acts = []
    all_masks = []

    layer_key = _raw_layer_key(layer)
    backend = get_backend()

    for prompt in prompts:
        key = _prompt_cache_key(model_name, prompt)

        if backend.exists(key):
            backend.touch(key)
            tensors = _load_tensors_from_backend(
                key, [layer_key, _ATTENTION_MASK_KEY]
            )
            all_acts.append(tensors[layer_key])
            all_masks.append(tensors[_ATTENTION_MASK_KEY])
            continue

        # v1 fallback (local only)
        if isinstance(backend, LocalCacheBackend):
            cache_dir = get_prompt_cache_dir(model_name, prompt)
            layer_path = cache_dir / f"layer_{layer}.pt"
            mask_path = cache_dir / "attention_mask.pt"
            if layer_path.exists() and mask_path.exists():
                all_acts.append(
                    torch.load(layer_path, weights_only=True)
                )
                all_masks.append(
                    torch.load(mask_path, weights_only=True)
                )
                continue

        # Shard registry fallback
        entry = _lookup_shard(model_name, prompt)
        if entry is not None:
            acts, mask = _load_raw_from_shard(model_name, prompt, [layer])
            # _load_raw_from_shard concatenates layers on last dim;
            # with a single layer this is already (1, seq_len, hidden_dim)
            all_acts.append(acts)
            all_masks.append(mask)
            continue

        raise FileNotFoundError(
            f"No cached activations for layer {layer} found for prompt: {prompt!r}"
        )

    return all_acts, all_masks


def load_layer_last_token(
    model_name: str,
    prompts: list[str],
    layer: int,
) -> torch.Tensor:
    """Load a single layer's last-token activation across multiple prompts.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    prompts : list[str]
        Prompt texts to load.
    layer : int
        Layer index to load.

    Returns
    -------
    torch.Tensor
        Stacked last-token activations with shape (N, hidden_dim).
    """
    acts_list, masks_list = load_layer_across_prompts(model_name, prompts, layer)

    vectors = []
    for acts, mask in zip(acts_list, masks_list):
        # acts: (1, seq_len, hidden_dim), mask: (1, seq_len)
        last_pos = mask[0].nonzero(as_tuple=True)[0][-1].item()
        vectors.append(acts[0, last_pos, :])

    return torch.stack(vectors, dim=0)


def save_prompt_activations(
    model_name: str,
    prompt: str,
    layers: list[int],
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    """Save raw activations for a single prompt (v2 safetensors format)."""
    _register_model(model_name)
    key = _prompt_cache_key(model_name, prompt)

    n_layers = len(layers)
    hidden_dim = activations.shape[-1] // n_layers

    new_tensors = {}
    for i, layer in enumerate(layers):
        start = i * hidden_dim
        end = (i + 1) * hidden_dim
        new_tensors[_raw_layer_key(layer)] = _prepare_tensor(activations[..., start:end])
    new_tensors[_ATTENTION_MASK_KEY] = attention_mask.detach().cpu().contiguous()

    _merge_save_backend(key, new_tensors)

    # Record in prompt manifest (#162)
    seq_len = attention_mask.shape[-1] if attention_mask.dim() >= 1 else None
    num_tokens = int(attention_mask.sum().item()) if seq_len is not None else None
    _append_manifest_entry(model_name, prompt, num_tokens=num_tokens)

    # Clean up v1 directory if it exists (migrate on write, local only)
    if isinstance(get_backend(), LocalCacheBackend):
        old_dir = get_prompt_cache_dir(model_name, prompt)
        if old_dir.is_dir():
            shutil.rmtree(old_dir)


def is_prompt_perplexity_cached(model_name: str, prompt: str) -> bool:
    """Check if perplexity features are cached for a prompt."""
    backend = get_backend()

    # Check sidecar file first (#120)
    sidecar_key = _prompt_perplexity_key(model_name, prompt)
    if backend.exists(sidecar_key):
        return True

    # Fall back to main file (backward compat with pre-sidecar entries)
    key = _prompt_cache_key(model_name, prompt)
    if backend.exists(key):
        tensor_keys = _get_tensor_keys_from_backend(key)
        return _PERPLEXITY_KEY in tensor_keys

    # v1 (local only)
    if isinstance(backend, LocalCacheBackend):
        cache_dir = get_prompt_cache_dir(model_name, prompt)
        return (cache_dir / "perplexity.pt").exists()

    return False


def load_prompt_perplexity(model_name: str, prompt: str) -> torch.Tensor:
    """Load cached perplexity features (3,) for a single prompt."""
    backend = get_backend()

    # Check sidecar file first (#120)
    sidecar_key = _prompt_perplexity_key(model_name, prompt)
    if backend.exists(sidecar_key):
        backend.touch(sidecar_key)
        tensors = _load_tensors_from_backend(sidecar_key, [_PERPLEXITY_KEY])
        return tensors[_PERPLEXITY_KEY]

    # Fall back to main file (backward compat with pre-sidecar entries)
    key = _prompt_cache_key(model_name, prompt)
    if backend.exists(key):
        backend.touch(key)
        tensors = _load_tensors_from_backend(key, [_PERPLEXITY_KEY])
        return tensors[_PERPLEXITY_KEY]

    # v1 (local only)
    if isinstance(backend, LocalCacheBackend):
        cache_dir = get_prompt_cache_dir(model_name, prompt)
        return torch.load(cache_dir / "perplexity.pt", weights_only=True)

    raise FileNotFoundError(
        f"No cached perplexity found for prompt: {prompt!r}"
    )


def save_prompt_perplexity(
    model_name: str,
    prompt: str,
    perplexity_features: torch.Tensor,
    token_perplexity: torch.Tensor | None = None,
    token_ids: torch.Tensor | None = None,
) -> None:
    """Save perplexity features for a single prompt.

    Writes to a small sidecar file (``{hash}.perplexity.safetensors``)
    instead of merging into the main activation file, avoiding a full
    read-modify-write of potentially multi-GB activation data (#120).

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    prompt : str
        The prompt text.
    perplexity_features : torch.Tensor
        Aggregate perplexity stats, shape (3,).
    token_perplexity : torch.Tensor | None
        Per-token perplexity values, shape (num_real_tokens - 1,).
    token_ids : torch.Tensor | None
        Input token IDs, shape (num_real_tokens,).
    """
    _register_model(model_name)
    key = _prompt_perplexity_key(model_name, prompt)
    tensors = {_PERPLEXITY_KEY: _prepare_tensor(perplexity_features)}
    if token_perplexity is not None:
        tensors[_TOKEN_PERPLEXITY_KEY] = _prepare_tensor(token_perplexity.float())
    if token_ids is not None:
        tensors[_TOKEN_IDS_KEY] = _prepare_tensor(token_ids.long())
    _save_tensors_to_backend(key, tensors)


def is_prompt_token_perplexity_cached(model_name: str, prompt: str) -> bool:
    """Check if per-token perplexity is cached for a prompt."""
    backend = get_backend()
    key = _prompt_perplexity_key(model_name, prompt)
    if backend.exists(key):
        tensor_keys = _get_tensor_keys_from_backend(key)
        return _TOKEN_PERPLEXITY_KEY in tensor_keys
    return False


def load_prompt_token_perplexity(
    model_name: str, prompt: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load per-token perplexity and token IDs for a single prompt.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (token_perplexity, token_ids) where token_perplexity has shape
        (num_real_tokens - 1,) and token_ids has shape (num_real_tokens,).
    """
    backend = get_backend()
    sidecar_key = _prompt_perplexity_key(model_name, prompt)
    if backend.exists(sidecar_key):
        backend.touch(sidecar_key)
        tensors = _load_tensors_from_backend(
            sidecar_key, [_TOKEN_PERPLEXITY_KEY, _TOKEN_IDS_KEY]
        )
        return tensors[_TOKEN_PERPLEXITY_KEY], tensors[_TOKEN_IDS_KEY]

    raise FileNotFoundError(
        f"No cached token perplexity found for prompt: {prompt!r}"
    )


# =============================================================================
# Logit Cache Functions
# =============================================================================


def is_prompt_logits_cached(
    model_name: str, prompt: str, top_k: int | None = None
) -> bool:
    """Check if logits are cached for a prompt.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    prompt : str
        The prompt text.
    top_k : int | None
        If None, checks for full logits. If int, checks for top-k logits.
    """
    backend = get_backend()

    # Check sidecar file first (#120)
    sidecar_key = _prompt_logits_key(model_name, prompt)
    if backend.exists(sidecar_key):
        tensor_keys = _get_tensor_keys_from_backend(sidecar_key)
        if top_k is not None:
            return (
                _LOGITS_TOP_K_VALUES_KEY in tensor_keys
                and _LOGITS_TOP_K_INDICES_KEY in tensor_keys
            )
        return _LOGITS_KEY in tensor_keys

    # Fall back to main file (backward compat with pre-sidecar entries)
    key = _prompt_cache_key(model_name, prompt)
    if backend.exists(key):
        tensor_keys = _get_tensor_keys_from_backend(key)
        if top_k is not None:
            return (
                _LOGITS_TOP_K_VALUES_KEY in tensor_keys
                and _LOGITS_TOP_K_INDICES_KEY in tensor_keys
            )
        return _LOGITS_KEY in tensor_keys

    return False


def save_prompt_logits(
    model_name: str,
    prompt: str,
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: int | None = None,
    positions: str = "last",
) -> None:
    """Save logits for a single prompt.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    prompt : str
        The prompt text.
    logits : torch.Tensor
        Raw logits with shape (1, seq_len, vocab_size).
    attention_mask : torch.Tensor
        Attention mask with shape (1, seq_len).
    top_k : int | None
        If set, store only top-k values and indices instead of full logits.
    positions : str
        Which token positions to store: "last" (default) or "all".
    """
    _register_model(model_name)

    # Select positions
    if positions == "last":
        # Find last non-padding position (works for both left- and right-padding)
        mask = attention_mask[0]  # (seq_len,)
        last_pos = mask.nonzero(as_tuple=True)[0][-1].item()
        selected_logits = logits[:, last_pos : last_pos + 1, :]  # (1, 1, vocab_size)
    elif positions == "all":
        selected_logits = logits  # (1, seq_len, vocab_size)
    else:
        raise ValueError(
            f"Invalid positions: {positions!r}. Must be 'last' or 'all'."
        )

    if top_k is not None:
        # Store only top-k values and indices
        values, indices = torch.topk(selected_logits, top_k, dim=-1)
        new_tensors = {
            _LOGITS_TOP_K_VALUES_KEY: _prepare_tensor(values),
            _LOGITS_TOP_K_INDICES_KEY: _prepare_tensor(indices.to(torch.int32)),
        }
    else:
        new_tensors = {_LOGITS_KEY: _prepare_tensor(selected_logits)}

    # Write to sidecar file instead of merging into main activation file (#120)
    sidecar_key = _prompt_logits_key(model_name, prompt)
    _save_tensors_to_backend(sidecar_key, new_tensors)


def save_prompt_topk_logits(
    model_name: str,
    prompt: str,
    values: torch.Tensor,
    indices: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: str = "last",
) -> None:
    """Save pre-compressed top-k logits for a single prompt.

    Unlike ``save_prompt_logits`` which receives full-vocab logits and
    applies ``torch.topk`` locally, this function receives values and
    indices that were already compressed server-side (e.g., inside an
    nnsight remote trace). It still performs position selection (e.g.,
    last token) but skips the ``topk`` call.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    prompt : str
        The prompt text.
    values : torch.Tensor
        Top-k logit values with shape (1, seq_len, K).
    indices : torch.Tensor
        Top-k token indices with shape (1, seq_len, K).
    attention_mask : torch.Tensor
        Attention mask with shape (1, seq_len).
    positions : str
        Which token positions to store: "last" (default) or "all".
    """
    _register_model(model_name)

    # Select positions
    if positions == "last":
        # Find last non-padding position (works for both left- and right-padding)
        mask = attention_mask[0]  # (seq_len,)
        last_pos = mask.nonzero(as_tuple=True)[0][-1].item()
        selected_values = values[:, last_pos : last_pos + 1, :]  # (1, 1, K)
        selected_indices = indices[:, last_pos : last_pos + 1, :]  # (1, 1, K)
    elif positions == "all":
        selected_values = values
        selected_indices = indices
    else:
        raise ValueError(
            f"Invalid positions: {positions!r}. Must be 'last' or 'all'."
        )

    new_tensors = {
        _LOGITS_TOP_K_VALUES_KEY: _prepare_tensor(selected_values),
        _LOGITS_TOP_K_INDICES_KEY: _prepare_tensor(selected_indices.to(torch.int32)),
    }
    # Write to sidecar file instead of merging into main activation file (#120)
    sidecar_key = _prompt_logits_key(model_name, prompt)
    _save_tensors_to_backend(sidecar_key, new_tensors)


def load_prompt_logits(
    model_name: str, prompt: str, top_k: int | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load cached logits for a single prompt.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    prompt : str
        The prompt text.
    top_k : int | None
        If None, loads full logits. If int, loads top-k values and indices.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | None]
        If top_k is None: (logits, None) where logits has shape (1, positions, vocab_size).
        If top_k is set: (values, indices) where values has shape (1, positions, K)
        and indices has shape (1, positions, K) with dtype int32.
    """
    backend = get_backend()

    # Check sidecar file first (#120)
    sidecar_key = _prompt_logits_key(model_name, prompt)
    if backend.exists(sidecar_key):
        backend.touch(sidecar_key)
        if top_k is not None:
            tensors = _load_tensors_from_backend(
                sidecar_key, [_LOGITS_TOP_K_VALUES_KEY, _LOGITS_TOP_K_INDICES_KEY]
            )
            return tensors[_LOGITS_TOP_K_VALUES_KEY], tensors[_LOGITS_TOP_K_INDICES_KEY]
        else:
            tensors = _load_tensors_from_backend(sidecar_key, [_LOGITS_KEY])
            return tensors[_LOGITS_KEY], None

    # Fall back to main file (backward compat with pre-sidecar entries)
    key = _prompt_cache_key(model_name, prompt)
    if backend.exists(key):
        backend.touch(key)
        if top_k is not None:
            tensors = _load_tensors_from_backend(
                key, [_LOGITS_TOP_K_VALUES_KEY, _LOGITS_TOP_K_INDICES_KEY]
            )
            return tensors[_LOGITS_TOP_K_VALUES_KEY], tensors[_LOGITS_TOP_K_INDICES_KEY]
        else:
            tensors = _load_tensors_from_backend(key, [_LOGITS_KEY])
            return tensors[_LOGITS_KEY], None

    # Shard registry fallback (logits_topk only)
    if top_k is not None:
        entry = _lookup_shard(model_name, prompt)
        if entry is not None:
            manifest = _load_shard_manifest(model_name)
            if manifest and "logits_topk" in manifest.get("tensors", {}):
                return _load_logits_from_shard(model_name, prompt)

    raise FileNotFoundError(
        f"No cached logits found for prompt: {prompt!r}"
    )


# =============================================================================
# Pooled Cache Functions
# =============================================================================


def get_pooled_cache_key(pooling: str) -> str:
    """Get the cache subdirectory name for a pooling strategy (v1 compat)."""
    return f"pooled_{pooling}"


def get_prompt_cached_pooled_layers(
    model_name: str,
    prompt: str,
    pooling: str,
) -> set[int] | None:
    """Return the set of pooled layer indices cached for a prompt, or None if no cache exists."""
    backend = get_backend()
    key = _prompt_cache_key(model_name, prompt)

    if backend.exists(key):
        tensor_keys = _get_tensor_keys_from_backend(key)
        return _parse_pooled_layer_keys(tensor_keys, pooling)

    if isinstance(backend, LocalCacheBackend):
        cache_dir = get_prompt_cache_dir(model_name, prompt)
        pooled_dir = cache_dir / get_pooled_cache_key(pooling)
        if pooled_dir.exists():
            cached = set()
            for f in pooled_dir.glob("layer_*.pt"):
                try:
                    cached.add(int(f.stem.split("_")[1]))
                except (IndexError, ValueError):
                    continue
            return cached

    # Shard registry fallback
    shard_info = _discover_from_shard(model_name, prompt)
    if shard_info is not None and pooling in shard_info.pooled:
        return set(shard_info.pooled[pooling])

    return None


def is_prompt_pooled_cached(
    model_name: str,
    prompt: str,
    required_layers: set[int],
    pooling: str,
) -> bool:
    """Check if pooled activations are cached for a prompt."""
    cached = get_prompt_cached_pooled_layers(model_name, prompt, pooling)
    if cached is not None:
        return required_layers.issubset(cached)
    return False


def load_prompt_pooled_activations(
    model_name: str, prompt: str, layers: list[int], pooling: str
) -> torch.Tensor:
    """Load pooled activations for a single prompt. Shape: (1, n_layers * hidden_dim)."""
    backend = get_backend()
    key = _prompt_cache_key(model_name, prompt)

    if backend.exists(key):
        backend.touch(key)
        tensor_keys = [_pooled_layer_key(pooling, li) for li in layers]
        tensors = _load_tensors_from_backend(key, tensor_keys)
        return torch.cat([tensors[k] for k in tensor_keys], dim=-1)

    # v1 (local only)
    if isinstance(backend, LocalCacheBackend):
        cache_dir = get_prompt_cache_dir(model_name, prompt)
        pooled_dir = cache_dir / get_pooled_cache_key(pooling)
        if pooled_dir.exists():
            layer_acts = []
            for layer in layers:
                acts = torch.load(pooled_dir / f"layer_{layer}.pt", weights_only=True)
                layer_acts.append(acts)
            return torch.cat(layer_acts, dim=-1)

    # Shard registry fallback
    entry = _lookup_shard(model_name, prompt)
    if entry is not None:
        return _load_pooled_from_shard(model_name, prompt, layers, pooling)

    raise FileNotFoundError(
        f"No cached pooled activations found for prompt: {prompt!r}"
    )


def save_prompt_pooled_activations(
    model_name: str,
    prompt: str,
    layers: list[int],
    pooled_activations: torch.Tensor,
    pooling: str,
) -> None:
    """Save pooled activations for a single prompt (v2 safetensors)."""
    _register_model(model_name)
    key = _prompt_cache_key(model_name, prompt)

    n_layers = len(layers)
    hidden_dim = pooled_activations.shape[-1] // n_layers

    new_tensors = {}
    for i, layer in enumerate(layers):
        start = i * hidden_dim
        end = (i + 1) * hidden_dim
        new_tensors[_pooled_layer_key(pooling, layer)] = _prepare_tensor(
            pooled_activations[..., start:end]
        )

    _merge_save_backend(key, new_tensors)

    # Clean up v1 pooled directory if exists (local only)
    if isinstance(get_backend(), LocalCacheBackend):
        old_dir = get_prompt_cache_dir(model_name, prompt)
        pooled_dir = old_dir / get_pooled_cache_key(pooling)
        if pooled_dir.is_dir():
            shutil.rmtree(pooled_dir)


# =============================================================================
# cache_info() (#48)
# =============================================================================


@dataclass
class ModelCacheInfo:
    """Cache info for a single model."""

    model_name: str | None
    model_hash: str
    size_bytes: int
    num_prompts: int
    num_layers: int
    has_pooled: bool
    has_perplexity: bool
    has_logits: bool = False

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)

    def __repr__(self) -> str:
        name = self.model_name or f"(hash: {self.model_hash})"
        return (
            f"  {name}  {self.size_gb:.1f} GB  "
            f"({self.num_prompts} prompts, {self.num_layers} layers"
            f"{', pooled' if self.has_pooled else ''}"
            f"{', logits' if self.has_logits else ''}"
            f"{', perplexity' if self.has_perplexity else ''})"
        )


@dataclass
class CacheInfo:
    """Cache usage report."""

    cache_dir: Path | str
    total_size_bytes: int
    models: list[ModelCacheInfo] = field(default_factory=list)
    oldest_mtime: float | None = None
    newest_mtime: float | None = None
    cache_limit_bytes: int | None = None

    @property
    def total_size_gb(self) -> float:
        return self.total_size_bytes / (1024**3)

    def __repr__(self) -> str:
        import time

        lines = [
            f"Cache directory: {self.cache_dir}",
            f"Total size: {self.total_size_gb:.1f} GB",
        ]
        if self.cache_limit_bytes and self.cache_limit_bytes > 0:
            limit_gb = self.cache_limit_bytes / (1024**3)
            lines.append(f"Size limit: {limit_gb:.1f} GB")
        if self.models:
            lines.append("Models:")
            for m in sorted(self.models, key=lambda x: x.size_bytes, reverse=True):
                lines.append(repr(m))
        if self.oldest_mtime:
            oldest = time.strftime("%Y-%m-%d", time.localtime(self.oldest_mtime))
            lines.append(f"Oldest entry: {oldest}")
        if self.newest_mtime:
            newest = time.strftime("%Y-%m-%d", time.localtime(self.newest_mtime))
            lines.append(f"Newest entry: {newest}")
        return "\n".join(lines)


def cache_info(model: str | None = None) -> CacheInfo:
    """Report cache size and breakdown.

    Parameters
    ----------
    model : str | None
        If provided, only report on this model. Otherwise, report all models.

    Returns
    -------
    CacheInfo
        Structured cache usage report.
    """
    backend = get_backend()

    if isinstance(backend, LocalCacheBackend):
        return _cache_info_local(backend, model)
    else:
        return _cache_info_backend(backend, model)


def _cache_info_local(backend: LocalCacheBackend, model: str | None = None) -> CacheInfo:
    """Cache info implementation for local filesystem backend."""
    base = backend.base_dir
    total_size = 0
    models = []
    oldest_mtime = None
    newest_mtime = None

    target_hash = _hash_string(model) if model else None

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name.startswith("."):
            continue
        if target_hash and model_dir.name != target_hash:
            continue

        model_name = _read_model_name(model_dir)
        model_size = 0
        num_prompts = 0
        all_layers: set[int] = set()
        has_pooled = False
        has_perplexity = False
        has_logits = False

        # Scan v2 safetensors files
        for sf_file in model_dir.glob("*.safetensors"):
            fsize = sf_file.stat().st_size
            fmtime = sf_file.stat().st_mtime
            model_size += fsize

            # Sidecar files (.logits.safetensors, .perplexity.safetensors)
            # don't count as separate prompts (#120)
            is_sidecar = (
                sf_file.name.endswith(".logits.safetensors")
                or sf_file.name.endswith(".perplexity.safetensors")
            )
            if not is_sidecar:
                num_prompts += 1

            if oldest_mtime is None or fmtime < oldest_mtime:
                oldest_mtime = fmtime
            if newest_mtime is None or fmtime > newest_mtime:
                newest_mtime = fmtime

            # Check sidecar files for perplexity
            if sf_file.name.endswith(".perplexity.safetensors"):
                has_perplexity = True
                continue

            # Logits sidecar — flag it, no layer/pooling info to extract
            if sf_file.name.endswith(".logits.safetensors"):
                has_logits = True
                continue

            try:
                keys = _load_sf_keys(sf_file)
                all_layers |= _parse_raw_layer_keys(keys)
                if any(k.startswith("pooled_") for k in keys):
                    has_pooled = True
                if _PERPLEXITY_KEY in keys:
                    has_perplexity = True
                if _LOGITS_KEY in keys or _LOGITS_TOP_K_VALUES_KEY in keys:
                    has_logits = True
            except Exception:
                pass

        # Scan v1 directories
        for prompt_dir in model_dir.iterdir():
            if not prompt_dir.is_dir() or prompt_dir.name.startswith("_"):
                continue
            dir_size = sum(f.stat().st_size for f in prompt_dir.rglob("*") if f.is_file())
            model_size += dir_size
            num_prompts += 1

            for f in prompt_dir.glob("layer_*.pt"):
                try:
                    all_layers.add(int(f.stem.split("_")[1]))
                except (IndexError, ValueError):
                    continue
                fmtime = f.stat().st_mtime
                if oldest_mtime is None or fmtime < oldest_mtime:
                    oldest_mtime = fmtime
                if newest_mtime is None or fmtime > newest_mtime:
                    newest_mtime = fmtime

            for pooled_dir in prompt_dir.iterdir():
                if pooled_dir.is_dir() and pooled_dir.name.startswith("pooled_"):
                    has_pooled = True
            if (prompt_dir / "perplexity.pt").exists():
                has_perplexity = True

        if num_prompts > 0 or model_size > 0:
            total_size += model_size
            models.append(
                ModelCacheInfo(
                    model_name=model_name,
                    model_hash=model_dir.name,
                    size_bytes=model_size,
                    num_prompts=num_prompts,
                    num_layers=len(all_layers),
                    has_pooled=has_pooled,
                    has_perplexity=has_perplexity,
                    has_logits=has_logits,
                )
            )

    return CacheInfo(
        cache_dir=base,
        total_size_bytes=total_size,
        models=models,
        oldest_mtime=oldest_mtime,
        newest_mtime=newest_mtime,
        cache_limit_bytes=_CACHE_MAX_BYTES if _CACHE_MAX_BYTES and _CACHE_MAX_BYTES > 0 else None,
    )


def _cache_info_backend(backend: CacheBackend, model: str | None = None) -> CacheInfo:
    """Cache info implementation for non-local backends (e.g. S3)."""
    from safetensors.torch import load

    target_hash = _hash_string(model) if model else None
    entries = backend.collect_entries()

    # Group entries by model hash
    model_entries: dict[str, list[tuple[str, int, float]]] = {}
    for entry_key, size, mtime in entries:
        parts = entry_key.split("/")
        if len(parts) < 2:
            continue
        model_hash = parts[0]
        if target_hash and model_hash != target_hash:
            continue
        model_entries.setdefault(model_hash, []).append((entry_key, size, mtime))

    total_size = 0
    models = []
    oldest_mtime = None
    newest_mtime = None

    for model_hash, m_entries in sorted(model_entries.items()):
        # Try to read model name
        name_key = f"{model_hash}/_model_name.txt"
        model_name = None
        if backend.exists(name_key):
            model_name = backend.read_text(name_key).strip()

        model_size = 0
        num_prompts = 0
        all_layers: set[int] = set()
        has_pooled = False
        has_perplexity = False
        has_logits = False

        for entry_key, size, mtime in m_entries:
            model_size += size

            # Sidecar files don't count as separate prompts (#120)
            is_sidecar = (
                entry_key.endswith(".logits.safetensors")
                or entry_key.endswith(".perplexity.safetensors")
            )
            if not is_sidecar:
                num_prompts += 1

            if oldest_mtime is None or mtime < oldest_mtime:
                oldest_mtime = mtime
            if newest_mtime is None or mtime > newest_mtime:
                newest_mtime = mtime

            # Detect sidecar files
            if entry_key.endswith(".perplexity.safetensors"):
                has_perplexity = True
                continue
            if entry_key.endswith(".logits.safetensors"):
                has_logits = True
                continue

            # Try to read tensor keys from main files
            if entry_key.endswith(".safetensors"):
                try:
                    data = backend.read_bytes(entry_key)
                    tensor_keys = set(load(data).keys())
                    all_layers |= _parse_raw_layer_keys(tensor_keys)
                    if any(k.startswith("pooled_") for k in tensor_keys):
                        has_pooled = True
                    if _PERPLEXITY_KEY in tensor_keys:
                        has_perplexity = True
                    if _LOGITS_KEY in tensor_keys or _LOGITS_TOP_K_VALUES_KEY in tensor_keys:
                        has_logits = True
                except Exception:
                    pass

        if num_prompts > 0 or model_size > 0:
            total_size += model_size
            models.append(
                ModelCacheInfo(
                    model_name=model_name,
                    model_hash=model_hash,
                    size_bytes=model_size,
                    num_prompts=num_prompts,
                    num_layers=len(all_layers),
                    has_pooled=has_pooled,
                    has_perplexity=has_perplexity,
                    has_logits=has_logits,
                )
            )

    # Determine cache_dir label
    from .cache_backends import S3CacheBackend

    if isinstance(backend, S3CacheBackend):
        cache_dir_label = f"s3://{backend.bucket}/{backend.prefix}"
    else:
        cache_dir_label = str(getattr(backend, "base_dir", "unknown"))

    return CacheInfo(
        cache_dir=cache_dir_label,
        total_size_bytes=total_size,
        models=models,
        oldest_mtime=oldest_mtime,
        newest_mtime=newest_mtime,
        cache_limit_bytes=None,  # No eviction on non-local backends
    )


# =============================================================================
# LRU Eviction (#49) — local backend only
# =============================================================================


def _collect_cache_entries() -> list[tuple[Path, int, float]]:
    """Collect all cache entries as (path, size_bytes, mtime).

    Returns both v2 safetensors files and v1 directories.
    """
    base = get_cache_dir()
    entries = []

    for model_dir in base.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue

        # v2 safetensors files
        for sf_file in model_dir.glob("*.safetensors"):
            try:
                stat = sf_file.stat()
            except FileNotFoundError:
                continue  # File was deleted between listing and stat
            entries.append((sf_file, stat.st_size, stat.st_mtime))

        # v1 directories
        for prompt_dir in model_dir.iterdir():
            if not prompt_dir.is_dir() or prompt_dir.name.startswith("_"):
                continue
            dir_size = sum(
                f.stat().st_size for f in prompt_dir.rglob("*") if f.is_file()
            )
            # Use directory mtime
            entries.append((prompt_dir, dir_size, prompt_dir.stat().st_mtime))

    return entries


def evict() -> None:
    """Evict least-recently-used cache entries if over the size limit.

    Call this explicitly when you want to enforce the cache size cap
    (e.g. after a batch of writes, at session end, or on a schedule).
    This is decoupled from writes for performance — scanning the full
    cache is O(total_files) and should not run on every write.

    No-op when no limit is set, caching is disabled, or on non-local
    backends (S3 etc.).
    """
    if _CACHE_MAX_BYTES is None or _CACHE_MAX_BYTES <= 0:
        return

    # Only evict on local backends
    if not _is_local_backend():
        return

    entries = _collect_cache_entries()
    total_size = sum(size for _, size, _ in entries)

    if total_size <= _CACHE_MAX_BYTES:
        return

    # Sort by mtime ascending (oldest first)
    entries.sort(key=lambda x: x[2])

    evicted_size = 0
    evicted_count = 0

    for path, size, _mtime in entries:
        if total_size - evicted_size < _CACHE_MAX_BYTES:
            break

        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            pass
        evicted_size += size
        evicted_count += 1

    if evicted_count > 0:
        logger.info(
            f"[CACHE] Evicted {evicted_size / (1024**3):.1f} GB "
            f"({evicted_count} entries) to stay under "
            f"{_CACHE_MAX_BYTES / (1024**3):.0f} GB cap"
        )


# =============================================================================
# Legacy Functions (backward compat, used by tests)
# =============================================================================


def get_extraction_cache_dir(model_name: str, prompts: list[str]) -> Path:
    """Get the cache directory for a specific model + prompts combination."""
    base = get_cache_dir()
    model_hash = _hash_string(model_name)
    prompts_hash = _hash_prompts(prompts)
    return base / model_hash / prompts_hash


def get_cached_layers(cache_dir: Path) -> set[int]:
    """Get cached layer indices from a legacy .pt directory."""
    if not cache_dir.exists():
        return set()
    cached = set()
    for f in cache_dir.glob("layer_*.pt"):
        try:
            cached.add(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return cached


def load_layer(cache_dir: Path, layer: int) -> torch.Tensor:
    """Load a single layer's activations from legacy .pt cache."""
    return torch.load(cache_dir / f"layer_{layer}.pt", weights_only=True)


def save_layer(cache_dir: Path, layer: int, activations: torch.Tensor) -> None:
    """Save a single layer's activations to legacy .pt cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(activations.cpu(), cache_dir / f"layer_{layer}.pt")


def load_attention_mask(cache_dir: Path) -> torch.Tensor:
    """Load attention mask from legacy .pt cache."""
    return torch.load(cache_dir / "attention_mask.pt", weights_only=True)


def save_attention_mask(cache_dir: Path, attention_mask: torch.Tensor) -> None:
    """Save attention mask to legacy .pt cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(attention_mask.cpu(), cache_dir / "attention_mask.pt")


def invalidate_extraction_cache(cache_dir: Path) -> None:
    """Delete all cached data for an extraction (both v1 and v2)."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    # Also delete v2 safetensors file
    sf_path = cache_dir.with_suffix(".safetensors")
    if sf_path.exists():
        sf_path.unlink()


def get_perplexity_cache_path(model_name: str, prompts: list[str]) -> Path:
    """Get the cache file path for perplexity features (legacy batch format)."""
    base = get_cache_dir()
    model_hash = _hash_string(model_name)
    prompts_hash = _hash_prompts(prompts)
    return base / model_hash / f"perplexity_{prompts_hash}.pt"


def load_perplexity_cache(cache_path: Path) -> torch.Tensor | None:
    """Load cached perplexity features (legacy batch format)."""
    if cache_path.exists():
        return torch.load(cache_path, weights_only=True)
    return None


def save_perplexity_cache(cache_path: Path, features: torch.Tensor) -> None:
    """Save perplexity features (legacy batch format)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features.cpu(), cache_path)


def clear_cache() -> int:
    """Clear all cached activations (both v1 and v2 formats).

    Returns the number of cache entries deleted.
    """
    backend = get_backend()

    if isinstance(backend, LocalCacheBackend):
        cache_dir = backend.base_dir
        count = 0
        for model_dir in cache_dir.iterdir():
            if not model_dir.is_dir():
                continue
            # Count v2 safetensors files
            count += sum(1 for _ in model_dir.glob("*.safetensors"))
            # Count v1 directories
            count += sum(
                1 for d in model_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
            )
            shutil.rmtree(model_dir)
        return count
    else:
        # Non-local backend: list all entries and delete by model hash
        entries = backend.collect_entries()
        model_hashes = set()
        for entry_key, _, _ in entries:
            parts = entry_key.split("/")
            if parts:
                model_hashes.add(parts[0])

        count = 0
        for model_hash in model_hashes:
            count += backend.delete_tree(f"{model_hash}/")
        return count


def compute_cache_key(
    model_name: str, prompts: list[str], layer_indices: list[int]
) -> str:
    """Compute a unique cache key (legacy)."""
    data = {
        "model": model_name,
        "prompts": prompts,
        "layers": sorted(layer_indices),
    }
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:32]


def get_cache_path(cache_key: str) -> Path:
    """Get the file path for a legacy cache key."""
    return get_cache_dir() / f"{cache_key}.pt"


# =============================================================================
# CachedExtractor
# =============================================================================


class CachedExtractor:
    """Wraps an ActivationExtractor with per-prompt caching.

    Checks which prompts are already cached before extraction,
    and only extracts the missing ones. Saves after each batch
    for interrupt resilience.

    Parameters
    ----------
    extractor : ActivationExtractor
        The underlying extractor.
    """

    def __init__(self, extractor):
        self.extractor = extractor

    def extract(
        self,
        prompts: list[str],
        remote: bool = False,
        invalidate_cache: bool = False,
        max_retries: int | None = None,
        cache_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Extract activations, using per-prompt cache when available."""
        model_name = self.extractor.model_name
        layer_indices = sorted(self.extractor.layer_indices)
        required_layers = set(layer_indices)
        batch_size = self.extractor.batch_size

        # Ensure model name is persisted for cache_info() (#54)
        _register_model(model_name)

        if remote:
            from .extraction import configure_remote

            configure_remote()

        logger.info(
            f"[CACHE] Checking cache for {len(prompts)} prompts, "
            f"requesting layers: {_format_layers(required_layers)}"
        )

        # Handle cache invalidation
        if invalidate_cache:
            logger.info("[CACHE] Cache invalidation requested")
            backend = get_backend()
            for prompt in prompts:
                key = _prompt_cache_key(model_name, prompt)
                if backend.exists(key):
                    backend.delete(key)
                # Delete v1 directory (local only)
                if isinstance(backend, LocalCacheBackend):
                    cache_dir = get_prompt_cache_dir(model_name, prompt)
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)

        # Check which prompts need extraction
        cached_prompts = []
        missing_prompts = []
        missing_indices = []

        for i, prompt in enumerate(prompts):
            if is_prompt_fully_cached(model_name, prompt, required_layers):
                cached_prompts.append(prompt)
            else:
                missing_prompts.append(prompt)
                missing_indices.append(i)

        n_cached = len(cached_prompts)
        n_missing = len(missing_prompts)

        if n_cached > 0:
            logger.info(
                f"[CACHE] HIT: {n_cached}/{len(prompts)} prompts already cached"
            )
        if n_missing > 0:
            logger.info(
                f"[CACHE] MISS: {n_missing}/{len(prompts)} prompts need extraction"
            )

        # Extract missing prompts in batches
        failed_count = 0
        if missing_prompts:
            num_batches = (n_missing + batch_size - 1) // batch_size
            logger.info(
                f"[CACHE] Extracting {n_missing} prompts in {num_batches} batches "
                f"(batch_size={batch_size}, remote={remote})"
            )

            from tqdm import tqdm

            # Resolve retry count: only retry for remote extraction
            effective_retries = max_retries if max_retries is not None else (3 if remote else 0)
            if not remote:
                effective_retries = 0  # never retry local — would hide real errors

            if effective_retries > 0:
                from .retry import retry_with_backoff

            failed_count = 0

            with torch.no_grad():
                for batch_idx in tqdm(
                    range(0, n_missing, batch_size),
                    total=num_batches,
                    desc="Extracting activations",
                    unit="batch",
                ):
                    batch_prompts = missing_prompts[
                        batch_idx : batch_idx + batch_size
                    ]
                    batch_num = batch_idx // batch_size + 1

                    try:
                        if effective_retries > 0:
                            batch_acts, batch_mask = retry_with_backoff(
                                lambda bp=batch_prompts: self.extractor.extract_batch(
                                    bp, layer_indices, remote=remote
                                ),
                                max_retries=effective_retries,
                                context=f"batch {batch_num}/{num_batches}",
                            )
                        else:
                            batch_acts, batch_mask = self.extractor.extract_batch(
                                batch_prompts, layer_indices, remote=remote
                            )
                    except Exception:
                        if remote and effective_retries > 0:
                            # Skip this batch — partial progress is saved
                            failed_count += len(batch_prompts)
                            logger.error(
                                f"[CACHE] Skipping batch {batch_num}/{num_batches} "
                                f"({len(batch_prompts)} prompts) after {effective_retries} retries"
                            )
                            continue
                        raise

                    # Save each prompt immediately
                    for j, prompt in enumerate(batch_prompts):
                        save_prompt_activations(
                            model_name,
                            prompt,
                            layer_indices,
                            batch_acts[j : j + 1],
                            batch_mask[j : j + 1],
                        )

            if failed_count > 0:
                logger.warning(
                    f"[CACHE] Extraction partially complete - {failed_count}/{n_missing} "
                    f"prompts failed and were skipped"
                )
            else:
                logger.info(
                    f"[CACHE] Extraction complete - all {n_missing} prompts cached"
                )
        else:
            logger.info("[CACHE] 100% cache hit - no model inference needed!")

        # If some batches were skipped, raise with a helpful message
        if failed_count > 0:
            cached_now = sum(
                1 for p in prompts
                if is_prompt_fully_cached(model_name, p, set(layer_indices))
            )
            raise RuntimeError(
                f"Remote extraction incomplete: {failed_count} prompts failed after retries. "
                f"{cached_now}/{len(prompts)} prompts are now cached. "
                f"Re-run to retry the remaining prompts (cached results will be reused)."
            )

        # If cache_only, skip matrix assembly (used by warmup())
        if cache_only:
            logger.info(
                f"[CACHE] cache_only=True — skipping matrix assembly "
                f"({n_cached} cached + {n_missing} extracted)"
            )
            return None

        # Load all prompts from cache in original order
        all_activations = []
        all_masks = []

        for prompt in prompts:
            acts, mask = load_prompt_activations(model_name, prompt, layer_indices)
            all_activations.append(acts)
            all_masks.append(mask)

        # Pad to same sequence length and concatenate
        max_seq_len = max(a.shape[1] for a in all_activations)
        hidden_dim = all_activations[0].shape[2]

        padded_activations = []
        padded_masks = []

        for acts, mask in zip(all_activations, all_masks):
            seq_len = acts.shape[1]
            if seq_len < max_seq_len:
                pad_size = max_seq_len - seq_len
                acts = torch.cat(
                    [acts, torch.zeros(acts.shape[0], pad_size, hidden_dim, dtype=acts.dtype)],
                    dim=1,
                )
                mask = torch.cat(
                    [mask, torch.zeros(acts.shape[0], pad_size, dtype=mask.dtype)], dim=1
                )
            padded_activations.append(acts)
            padded_masks.append(mask)

        activations = torch.cat(padded_activations, dim=0)
        attention_mask = torch.cat(padded_masks, dim=0)

        logger.info(
            f"[CACHE] Complete: returned activations shape {tuple(activations.shape)} "
            f"({n_cached} cached + {n_missing} extracted)"
        )

        return activations, attention_mask
