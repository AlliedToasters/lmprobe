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

Legacy format v1 (read-only, for backward compat):
    {model_hash}/{prompt_hash}/
      layer_{i}.pt
      attention_mask.pt
      pooled_{strategy}/layer_{i}.pt
      perplexity.pt

Features:
    - Disk-full error handling (#44)
    - Pool-then-cache default (#45, applied in UnifiedCache)
    - float16 cache storage (#46) via LMPROBE_CACHE_DTYPE
    - Single safetensors file per prompt (#47)
    - cache_info() reporting (#48)
    - LRU eviction (#49) via LMPROBE_CACHE_MAX_GB
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import torch

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
    """Set maximum cache size in GB with LRU eviction.

    When the cache exceeds this limit after a write, least-recently-used
    entries are evicted until the cache is under the cap.

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


# =============================================================================
# Safe I/O (#44 disk-full handling, #46 dtype, #47 safetensors)
# =============================================================================


def _prepare_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Prepare tensor for cache storage: CPU, contiguous, optional dtype cast."""
    t = tensor.detach().cpu().contiguous()
    if _CACHE_DTYPE is not None and t.is_floating_point() and t.dtype != _CACHE_DTYPE:
        t = t.to(_CACHE_DTYPE)
    return t


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

    # Run LRU eviction after successful write
    _maybe_evict()


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
        for k in keys:
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
    base = get_cache_dir()
    model_hash = _hash_string(model_name)
    model_dir = base / model_hash
    model_dir.mkdir(parents=True, exist_ok=True)
    name_file = model_dir / "_model_name.txt"
    if not name_file.exists():
        name_file.write_text(model_name)


def _read_model_name(model_dir: Path) -> str | None:
    """Read model name from a model cache directory."""
    name_file = model_dir / "_model_name.txt"
    if name_file.exists():
        return name_file.read_text().strip()
    return None


# =============================================================================
# Per-Prompt Cache (v2 safetensors with v1 fallback)
# =============================================================================


def get_prompt_cached_layers(cache_dir: Path) -> set[int]:
    """Get cached raw layer indices for a prompt.

    Checks v2 safetensors format first, then v1 .pt format.
    """
    # v2: check safetensors file (sibling of the directory path)
    sf_path = cache_dir.with_suffix(".safetensors")
    if sf_path.exists():
        keys = _load_sf_keys(sf_path)
        return _parse_raw_layer_keys(keys)

    # v1: check .pt files in directory
    if not cache_dir.exists():
        return set()
    cached = set()
    for f in cache_dir.glob("layer_*.pt"):
        try:
            cached.add(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return cached


def is_prompt_fully_cached(
    model_name: str, prompt: str, required_layers: set[int]
) -> bool:
    """Check if a prompt has all required raw layers cached."""
    # v2
    sf_path = get_prompt_cache_path(model_name, prompt)
    if sf_path.exists():
        keys = _load_sf_keys(sf_path)
        cached = _parse_raw_layer_keys(keys)
        has_mask = _ATTENTION_MASK_KEY in keys
        return required_layers.issubset(cached) and has_mask

    # v1
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    cached = get_prompt_cached_layers(cache_dir)
    has_mask = (cache_dir / "attention_mask.pt").exists()
    return required_layers.issubset(cached) and has_mask


def load_prompt_activations(
    model_name: str, prompt: str, layers: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load cached raw activations for a single prompt.

    Returns (activations, attention_mask). Checks v2 then v1.
    """
    sf_path = get_prompt_cache_path(model_name, prompt)
    if sf_path.exists():
        # Touch for LRU
        os.utime(sf_path)
        keys_to_load = [_raw_layer_key(l) for l in layers] + [_ATTENTION_MASK_KEY]
        tensors = _load_sf_tensors(sf_path, keys_to_load)
        layer_acts = [tensors[_raw_layer_key(l)] for l in layers]
        activations = torch.cat(layer_acts, dim=-1)
        return activations, tensors[_ATTENTION_MASK_KEY]

    # v1 fallback
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    layer_acts = []
    for layer in layers:
        acts = torch.load(cache_dir / f"layer_{layer}.pt", weights_only=True)
        layer_acts.append(acts)
    activations = torch.cat(layer_acts, dim=-1)
    mask = torch.load(cache_dir / "attention_mask.pt", weights_only=True)
    return activations, mask


def save_prompt_activations(
    model_name: str,
    prompt: str,
    layers: list[int],
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    """Save raw activations for a single prompt (v2 safetensors format)."""
    _register_model(model_name)
    sf_path = get_prompt_cache_path(model_name, prompt)

    n_layers = len(layers)
    hidden_dim = activations.shape[-1] // n_layers

    new_tensors = {}
    for i, layer in enumerate(layers):
        start = i * hidden_dim
        end = (i + 1) * hidden_dim
        new_tensors[_raw_layer_key(layer)] = _prepare_tensor(activations[..., start:end])
    new_tensors[_ATTENTION_MASK_KEY] = attention_mask.detach().cpu().contiguous()

    _merge_save(sf_path, new_tensors)

    # Clean up v1 directory if it exists (migrate on write)
    old_dir = get_prompt_cache_dir(model_name, prompt)
    if old_dir.is_dir():
        shutil.rmtree(old_dir)


def is_prompt_perplexity_cached(model_name: str, prompt: str) -> bool:
    """Check if perplexity features are cached for a prompt."""
    sf_path = get_prompt_cache_path(model_name, prompt)
    if sf_path.exists():
        return _PERPLEXITY_KEY in _load_sf_keys(sf_path)
    # v1
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    return (cache_dir / "perplexity.pt").exists()


def load_prompt_perplexity(model_name: str, prompt: str) -> torch.Tensor:
    """Load cached perplexity features (3,) for a single prompt."""
    sf_path = get_prompt_cache_path(model_name, prompt)
    if sf_path.exists():
        os.utime(sf_path)
        return _load_sf_tensor(sf_path, _PERPLEXITY_KEY)
    # v1
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    return torch.load(cache_dir / "perplexity.pt", weights_only=True)


def save_prompt_perplexity(
    model_name: str, prompt: str, perplexity_features: torch.Tensor
) -> None:
    """Save perplexity features for a single prompt."""
    _register_model(model_name)
    sf_path = get_prompt_cache_path(model_name, prompt)
    new_tensors = {_PERPLEXITY_KEY: _prepare_tensor(perplexity_features)}
    _merge_save(sf_path, new_tensors)


# =============================================================================
# Pooled Cache Functions
# =============================================================================


def get_pooled_cache_key(pooling: str) -> str:
    """Get the cache subdirectory name for a pooling strategy (v1 compat)."""
    return f"pooled_{pooling}"


def is_prompt_pooled_cached(
    model_name: str,
    prompt: str,
    required_layers: set[int],
    pooling: str,
) -> bool:
    """Check if pooled activations are cached for a prompt."""
    # v2
    sf_path = get_prompt_cache_path(model_name, prompt)
    if sf_path.exists():
        keys = _load_sf_keys(sf_path)
        cached = _parse_pooled_layer_keys(keys, pooling)
        return required_layers.issubset(cached)

    # v1
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    pooled_dir = cache_dir / get_pooled_cache_key(pooling)
    if not pooled_dir.exists():
        return False
    cached = set()
    for f in pooled_dir.glob("layer_*.pt"):
        try:
            cached.add(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return required_layers.issubset(cached)


def load_prompt_pooled_activations(
    model_name: str, prompt: str, layers: list[int], pooling: str
) -> torch.Tensor:
    """Load pooled activations for a single prompt. Shape: (1, n_layers * hidden_dim)."""
    sf_path = get_prompt_cache_path(model_name, prompt)
    if sf_path.exists():
        os.utime(sf_path)
        keys = [_pooled_layer_key(pooling, l) for l in layers]
        tensors = _load_sf_tensors(sf_path, keys)
        return torch.cat([tensors[k] for k in keys], dim=-1)

    # v1
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    pooled_dir = cache_dir / get_pooled_cache_key(pooling)
    layer_acts = []
    for layer in layers:
        acts = torch.load(pooled_dir / f"layer_{layer}.pt", weights_only=True)
        layer_acts.append(acts)
    return torch.cat(layer_acts, dim=-1)


def save_prompt_pooled_activations(
    model_name: str,
    prompt: str,
    layers: list[int],
    pooled_activations: torch.Tensor,
    pooling: str,
) -> None:
    """Save pooled activations for a single prompt (v2 safetensors)."""
    _register_model(model_name)
    sf_path = get_prompt_cache_path(model_name, prompt)

    n_layers = len(layers)
    hidden_dim = pooled_activations.shape[-1] // n_layers

    new_tensors = {}
    for i, layer in enumerate(layers):
        start = i * hidden_dim
        end = (i + 1) * hidden_dim
        new_tensors[_pooled_layer_key(pooling, layer)] = _prepare_tensor(
            pooled_activations[..., start:end]
        )

    _merge_save(sf_path, new_tensors)

    # Clean up v1 pooled directory if exists
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

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)

    def __repr__(self) -> str:
        name = self.model_name or f"(hash: {self.model_hash})"
        return (
            f"  {name}  {self.size_gb:.1f} GB  "
            f"({self.num_prompts} prompts, {self.num_layers} layers"
            f"{', pooled' if self.has_pooled else ''}"
            f"{', perplexity' if self.has_perplexity else ''})"
        )


@dataclass
class CacheInfo:
    """Cache usage report."""

    cache_dir: Path
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
            lines.append(f"Oldest entry: {time.strftime('%Y-%m-%d', time.localtime(self.oldest_mtime))}")
        if self.newest_mtime:
            lines.append(f"Newest entry: {time.strftime('%Y-%m-%d', time.localtime(self.newest_mtime))}")
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
    base = get_cache_dir()
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

        # Scan v2 safetensors files
        for sf_file in model_dir.glob("*.safetensors"):
            fsize = sf_file.stat().st_size
            fmtime = sf_file.stat().st_mtime
            model_size += fsize
            num_prompts += 1

            if oldest_mtime is None or fmtime < oldest_mtime:
                oldest_mtime = fmtime
            if newest_mtime is None or fmtime > newest_mtime:
                newest_mtime = fmtime

            try:
                keys = _load_sf_keys(sf_file)
                all_layers |= _parse_raw_layer_keys(keys)
                if any(k.startswith("pooled_") for k in keys):
                    has_pooled = True
                if _PERPLEXITY_KEY in keys:
                    has_perplexity = True
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


# =============================================================================
# LRU Eviction (#49)
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
            stat = sf_file.stat()
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


def _maybe_evict() -> None:
    """Evict least-recently-used cache entries if over the size limit."""
    if _CACHE_MAX_BYTES is None or _CACHE_MAX_BYTES <= 0:
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
        if total_size - evicted_size <= _CACHE_MAX_BYTES:
            break

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
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
    cache_dir = get_cache_dir()
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract activations, using per-prompt cache when available."""
        model_name = self.extractor.model_name
        layer_indices = sorted(self.extractor.layer_indices)
        required_layers = set(layer_indices)
        batch_size = self.extractor.batch_size

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
            for prompt in prompts:
                # Delete v2 file
                sf_path = get_prompt_cache_path(model_name, prompt)
                if sf_path.exists():
                    sf_path.unlink()
                # Delete v1 directory
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
        if missing_prompts:
            num_batches = (n_missing + batch_size - 1) // batch_size
            logger.info(
                f"[CACHE] Extracting {n_missing} prompts in {num_batches} batches "
                f"(batch_size={batch_size}, remote={remote})"
            )

            from tqdm import tqdm

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
                    batch_acts, batch_mask = self.extractor.extract_batch(
                        batch_prompts, layer_indices, remote=remote
                    )

                    # Save each prompt immediately
                    for j, prompt in enumerate(batch_prompts):
                        save_prompt_activations(
                            model_name,
                            prompt,
                            layer_indices,
                            batch_acts[j : j + 1],
                            batch_mask[j : j + 1],
                        )

            logger.info(
                f"[CACHE] Extraction complete - all {n_missing} prompts cached"
            )
        else:
            logger.info("[CACHE] 100% cache hit - no model inference needed!")

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
                    [acts, torch.zeros(1, pad_size, hidden_dim, dtype=acts.dtype)],
                    dim=1,
                )
                mask = torch.cat(
                    [mask, torch.zeros(1, pad_size, dtype=mask.dtype)], dim=1
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
