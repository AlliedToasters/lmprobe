"""Activation caching for lmprobe.

This module provides disk-based caching of extracted activations to avoid
redundant model inference, especially important for remote execution.

Cache structure (per-prompt, per-layer for maximum reuse and interrupt resilience):
    ~/.cache/lmprobe/
      {model_hash}/
        {prompt_hash}/           # Hash of SINGLE prompt
          layer_8.pt             # (1, seq_len, hidden_dim)
          layer_16.pt
          attention_mask.pt      # (1, seq_len)

This structure enables:
- Per-prompt caching: each prompt is cached independently
- Interrupt resilience: saves after each batch, resume from where you left off
- Partial layer hits: reuse layer 8 even when also requesting layer 16
- Cross-run reuse: if you've seen a prompt before in ANY context, it's cached
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import torch

# Set up logging for cache operations
logger = logging.getLogger(__name__)


def enable_cache_logging(level: int = logging.INFO) -> None:
    """Enable cache logging to see cache hit/miss information.

    Parameters
    ----------
    level : int
        Logging level. Use logging.INFO for basic hit/miss info,
        logging.DEBUG for detailed cache operations.

    Examples
    --------
    >>> from lmprobe.cache import enable_cache_logging
    >>> enable_cache_logging()  # Now you'll see [CACHE] messages
    """
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)


# Auto-enable logging if LMPROBE_CACHE_DEBUG environment variable is set
if os.getenv("LMPROBE_CACHE_DEBUG"):
    enable_cache_logging(logging.DEBUG if os.getenv("LMPROBE_CACHE_DEBUG") == "debug" else logging.INFO)


def _format_layers(layers: set[int] | list[int], max_show: int = 10) -> str:
    """Format layer indices for logging."""
    sorted_layers = sorted(layers)
    if len(sorted_layers) <= max_show:
        return str(sorted_layers)
    else:
        return f"{sorted_layers[:max_show//2]}...{sorted_layers[-max_show//2:]} ({len(sorted_layers)} total)"


def get_cache_dir() -> Path:
    """Get the base cache directory, creating it if necessary.

    Returns
    -------
    Path
        Path to the cache directory.
    """
    cache_dir = os.getenv("LMPROBE_CACHE_DIR")
    if cache_dir:
        path = Path(cache_dir)
    else:
        path = Path.home() / ".cache" / "lmprobe"

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
# Per-Prompt Cache Functions (new architecture for interrupt resilience)
# =============================================================================


def get_prompt_cache_dir(model_name: str, prompt: str) -> Path:
    """Get the cache directory for a single prompt.

    Parameters
    ----------
    model_name : str
        The model identifier.
    prompt : str
        A single prompt string.

    Returns
    -------
    Path
        Directory where per-layer cache files for this prompt are stored.
    """
    base = get_cache_dir()
    model_hash = _hash_string(model_name)
    prompt_hash = _hash_string(prompt)
    return base / model_hash / prompt_hash


def get_prompt_cached_layers(cache_dir: Path) -> set[int]:
    """Get the set of layer indices cached for a single prompt.

    Parameters
    ----------
    cache_dir : Path
        The prompt's cache directory.

    Returns
    -------
    set[int]
        Set of layer indices that have cached activations.
    """
    if not cache_dir.exists():
        return set()

    cached = set()
    for f in cache_dir.glob("layer_*.pt"):
        try:
            layer_num = int(f.stem.split("_")[1])
            cached.add(layer_num)
        except (IndexError, ValueError):
            continue
    return cached


def is_prompt_fully_cached(model_name: str, prompt: str, required_layers: set[int]) -> bool:
    """Check if a prompt has all required layers cached.

    Parameters
    ----------
    model_name : str
        The model identifier.
    prompt : str
        A single prompt string.
    required_layers : set[int]
        The layer indices that must be cached.

    Returns
    -------
    bool
        True if all required layers are cached for this prompt.
    """
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    cached = get_prompt_cached_layers(cache_dir)
    # Also check attention mask exists
    has_mask = (cache_dir / "attention_mask.pt").exists()
    return required_layers.issubset(cached) and has_mask


def load_prompt_activations(
    model_name: str,
    prompt: str,
    layers: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load cached activations for a single prompt.

    Parameters
    ----------
    model_name : str
        The model identifier.
    prompt : str
        A single prompt string.
    layers : list[int]
        Layer indices to load (must be sorted).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (activations, attention_mask) where activations has shape
        (1, seq_len, n_layers * hidden_dim).
    """
    cache_dir = get_prompt_cache_dir(model_name, prompt)

    layer_acts = []
    for layer in layers:
        path = cache_dir / f"layer_{layer}.pt"
        acts = torch.load(path, weights_only=True)
        layer_acts.append(acts)

    # Concatenate layers along hidden dimension
    activations = torch.cat(layer_acts, dim=-1)

    # Load attention mask
    mask_path = cache_dir / "attention_mask.pt"
    attention_mask = torch.load(mask_path, weights_only=True)

    return activations, attention_mask


def save_prompt_activations(
    model_name: str,
    prompt: str,
    layers: list[int],
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    """Save activations for a single prompt.

    Parameters
    ----------
    model_name : str
        The model identifier.
    prompt : str
        A single prompt string.
    layers : list[int]
        Layer indices (must be sorted).
    activations : torch.Tensor
        Activations with shape (1, seq_len, n_layers * hidden_dim).
    attention_mask : torch.Tensor
        Attention mask with shape (1, seq_len).
    """
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Split activations by layer and save each
    n_layers = len(layers)
    hidden_dim = activations.shape[-1] // n_layers

    for i, layer in enumerate(layers):
        start = i * hidden_dim
        end = (i + 1) * hidden_dim
        layer_acts = activations[..., start:end]
        path = cache_dir / f"layer_{layer}.pt"
        torch.save(layer_acts.cpu(), path)

    # Save attention mask
    mask_path = cache_dir / "attention_mask.pt"
    torch.save(attention_mask.cpu(), mask_path)


def is_prompt_perplexity_cached(model_name: str, prompt: str) -> bool:
    """Check if perplexity features are cached for a prompt.

    Parameters
    ----------
    model_name : str
        The model identifier.
    prompt : str
        A single prompt string.

    Returns
    -------
    bool
        True if perplexity.pt exists for this prompt.
    """
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    return (cache_dir / "perplexity.pt").exists()


def load_prompt_perplexity(model_name: str, prompt: str) -> torch.Tensor:
    """Load cached perplexity features for a single prompt.

    Parameters
    ----------
    model_name : str
        The model identifier.
    prompt : str
        A single prompt string.

    Returns
    -------
    torch.Tensor
        Perplexity features with shape (3,) - [mean_ppl, min_ppl, max_ppl].
    """
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    path = cache_dir / "perplexity.pt"
    return torch.load(path, weights_only=True)


def save_prompt_perplexity(
    model_name: str,
    prompt: str,
    perplexity_features: torch.Tensor,
) -> None:
    """Save perplexity features for a single prompt.

    Parameters
    ----------
    model_name : str
        The model identifier.
    prompt : str
        A single prompt string.
    perplexity_features : torch.Tensor
        Perplexity features with shape (3,) - [mean_ppl, min_ppl, max_ppl].
    """
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "perplexity.pt"
    torch.save(perplexity_features.cpu(), path)


# =============================================================================
# Pooled Cache Functions (memory-efficient: stores only pooled activations)
# =============================================================================


def get_pooled_cache_key(pooling: str) -> str:
    """Get the cache subdirectory name for a pooling strategy.

    Parameters
    ----------
    pooling : str
        Pooling strategy name (e.g., "last_token", "mean").

    Returns
    -------
    str
        Cache key suffix for this pooling strategy.
    """
    return f"pooled_{pooling}"


def is_prompt_pooled_cached(
    model_name: str,
    prompt: str,
    required_layers: set[int],
    pooling: str,
) -> bool:
    """Check if pooled activations are cached for a prompt.

    Parameters
    ----------
    model_name : str
        The model identifier.
    prompt : str
        A single prompt string.
    required_layers : set[int]
        The layer indices that must be cached.
    pooling : str
        The pooling strategy used.

    Returns
    -------
    bool
        True if all required layers have pooled activations cached.
    """
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    pooled_dir = cache_dir / get_pooled_cache_key(pooling)

    if not pooled_dir.exists():
        return False

    cached = set()
    for f in pooled_dir.glob("layer_*.pt"):
        try:
            layer_num = int(f.stem.split("_")[1])
            cached.add(layer_num)
        except (IndexError, ValueError):
            continue

    return required_layers.issubset(cached)


def load_prompt_pooled_activations(
    model_name: str,
    prompt: str,
    layers: list[int],
    pooling: str,
) -> torch.Tensor:
    """Load pooled (pre-aggregated) activations for a single prompt.

    Parameters
    ----------
    model_name : str
        The model identifier.
    prompt : str
        A single prompt string.
    layers : list[int]
        Layer indices to load (must be sorted).
    pooling : str
        The pooling strategy used.

    Returns
    -------
    torch.Tensor
        Pooled activations with shape (1, n_layers * hidden_dim).
        Already aggregated across tokens.
    """
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    pooled_dir = cache_dir / get_pooled_cache_key(pooling)

    layer_acts = []
    for layer in layers:
        path = pooled_dir / f"layer_{layer}.pt"
        acts = torch.load(path, weights_only=True)
        layer_acts.append(acts)

    # Concatenate layers along hidden dimension
    # Each layer_acts[i] has shape (1, hidden_dim)
    return torch.cat(layer_acts, dim=-1)


def save_prompt_pooled_activations(
    model_name: str,
    prompt: str,
    layers: list[int],
    pooled_activations: torch.Tensor,
    pooling: str,
) -> None:
    """Save pooled activations for a single prompt.

    This saves ~100x less disk space than full-sequence caching
    by storing only the pooled result (1, hidden_dim) per layer
    instead of (1, seq_len, hidden_dim).

    Parameters
    ----------
    model_name : str
        The model identifier.
    prompt : str
        A single prompt string.
    layers : list[int]
        Layer indices (must be sorted).
    pooled_activations : torch.Tensor
        Pooled activations with shape (1, n_layers * hidden_dim).
    pooling : str
        The pooling strategy used.
    """
    cache_dir = get_prompt_cache_dir(model_name, prompt)
    pooled_dir = cache_dir / get_pooled_cache_key(pooling)
    pooled_dir.mkdir(parents=True, exist_ok=True)

    # Split activations by layer and save each
    n_layers = len(layers)
    hidden_dim = pooled_activations.shape[-1] // n_layers

    for i, layer in enumerate(layers):
        start = i * hidden_dim
        end = (i + 1) * hidden_dim
        layer_acts = pooled_activations[..., start:end]
        path = pooled_dir / f"layer_{layer}.pt"
        torch.save(layer_acts.cpu(), path)


# =============================================================================
# Legacy Functions (for backwards compatibility)
# =============================================================================


def get_extraction_cache_dir(model_name: str, prompts: list[str]) -> Path:
    """Get the cache directory for a specific model + prompts combination.

    Parameters
    ----------
    model_name : str
        The model identifier.
    prompts : list[str]
        The prompts being extracted.

    Returns
    -------
    Path
        Directory where per-layer cache files are stored.
    """
    base = get_cache_dir()
    model_hash = _hash_string(model_name)
    prompts_hash = _hash_prompts(prompts)
    return base / model_hash / prompts_hash


def get_cached_layers(cache_dir: Path) -> set[int]:
    """Get the set of layer indices that are cached.

    This is a fast O(1) operation - just lists the directory.

    Parameters
    ----------
    cache_dir : Path
        The extraction cache directory.

    Returns
    -------
    set[int]
        Set of layer indices that have cached activations.
    """
    if not cache_dir.exists():
        return set()

    cached = set()
    for f in cache_dir.glob("layer_*.pt"):
        try:
            # Extract layer number from "layer_8.pt" -> 8
            layer_num = int(f.stem.split("_")[1])
            cached.add(layer_num)
        except (IndexError, ValueError):
            continue
    return cached


def load_layer(cache_dir: Path, layer: int) -> torch.Tensor:
    """Load a single layer's activations from cache.

    Parameters
    ----------
    cache_dir : Path
        The extraction cache directory.
    layer : int
        The layer index to load.

    Returns
    -------
    torch.Tensor
        Activations with shape (batch, seq_len, hidden_dim).
    """
    path = cache_dir / f"layer_{layer}.pt"
    return torch.load(path, weights_only=True)


def save_layer(cache_dir: Path, layer: int, activations: torch.Tensor) -> None:
    """Save a single layer's activations to cache.

    Parameters
    ----------
    cache_dir : Path
        The extraction cache directory.
    layer : int
        The layer index.
    activations : torch.Tensor
        Activations with shape (batch, seq_len, hidden_dim).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"layer_{layer}.pt"
    torch.save(activations.cpu(), path)


def load_attention_mask(cache_dir: Path) -> torch.Tensor:
    """Load the attention mask from cache.

    Parameters
    ----------
    cache_dir : Path
        The extraction cache directory.

    Returns
    -------
    torch.Tensor
        Attention mask with shape (batch, seq_len).
    """
    path = cache_dir / "attention_mask.pt"
    return torch.load(path, weights_only=True)


def save_attention_mask(cache_dir: Path, attention_mask: torch.Tensor) -> None:
    """Save the attention mask to cache.

    Parameters
    ----------
    cache_dir : Path
        The extraction cache directory.
    attention_mask : torch.Tensor
        Attention mask with shape (batch, seq_len).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "attention_mask.pt"
    torch.save(attention_mask.cpu(), path)


def invalidate_extraction_cache(cache_dir: Path) -> None:
    """Delete all cached data for an extraction.

    Parameters
    ----------
    cache_dir : Path
        The extraction cache directory to delete.
    """
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def get_perplexity_cache_path(model_name: str, prompts: list[str]) -> Path:
    """Get the cache file path for perplexity features.

    Parameters
    ----------
    model_name : str
        The model identifier.
    prompts : list[str]
        The prompts being processed.

    Returns
    -------
    Path
        Path to the perplexity cache file.
    """
    base = get_cache_dir()
    model_hash = _hash_string(model_name)
    prompts_hash = _hash_prompts(prompts)
    return base / model_hash / f"perplexity_{prompts_hash}.pt"


def load_perplexity_cache(cache_path: Path) -> torch.Tensor | None:
    """Load cached perplexity features if they exist.

    Parameters
    ----------
    cache_path : Path
        Path to the perplexity cache file.

    Returns
    -------
    torch.Tensor | None
        Cached perplexity features with shape (n_prompts, 3), or None if not cached.
    """
    if cache_path.exists():
        return torch.load(cache_path, weights_only=True)
    return None


def save_perplexity_cache(cache_path: Path, features: torch.Tensor) -> None:
    """Save perplexity features to cache.

    Parameters
    ----------
    cache_path : Path
        Path to save the cache file.
    features : torch.Tensor
        Perplexity features with shape (n_prompts, 3).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features.cpu(), cache_path)


def clear_cache() -> int:
    """Clear all cached activations.

    Returns
    -------
    int
        Number of cache entries deleted.
    """
    cache_dir = get_cache_dir()
    count = 0
    # Count model directories
    for model_dir in cache_dir.iterdir():
        if model_dir.is_dir():
            count += sum(1 for _ in model_dir.iterdir() if _.is_dir())
            shutil.rmtree(model_dir)
    return count


# Legacy functions for backwards compatibility
def compute_cache_key(
    model_name: str,
    prompts: list[str],
    layer_indices: list[int],
) -> str:
    """Compute a unique cache key (legacy, for migration).

    Deprecated: Use get_extraction_cache_dir instead.
    """
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


class CachedExtractor:
    """Wraps an ActivationExtractor with per-prompt caching.

    This extractor checks which prompts are already cached before extraction,
    and only extracts the missing prompts. Caching is done per-prompt, per-layer,
    which enables:
    - Interrupt resilience: saves after each batch, resume from where you left off
    - Cross-run reuse: if you've seen a prompt before in ANY context, it's cached
    - Partial layer hits: reuse layer 8 even when also requesting layer 16

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
        """Extract activations, using per-prompt cache when available.

        Caching happens at the prompt level, and saves occur after each batch
        completes. This means if you interrupt mid-extraction, completed batches
        are saved and will be cache hits on the next run.

        Parameters
        ----------
        prompts : list[str]
            Text prompts.
        remote : bool
            Whether to use remote execution.
        invalidate_cache : bool
            If True, delete cached values and re-extract all prompts.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (activations, attention_mask) where activations has shape
            (batch, seq_len, n_layers * hidden_dim) with layers concatenated
            in sorted order.
        """
        from .extraction import _extract_batch, configure_remote

        model_name = self.extractor.model_name
        layer_indices = sorted(self.extractor.layer_indices)
        required_layers = set(layer_indices)
        batch_size = self.extractor.batch_size

        # Validate API key early if using remote mode
        # This ensures we fail fast if the key is missing, even with full cache hits
        if remote:
            configure_remote()

        logger.info(
            f"[CACHE] Checking cache for {len(prompts)} prompts, "
            f"requesting layers: {_format_layers(required_layers)}"
        )

        # Handle cache invalidation by clearing per-prompt caches
        if invalidate_cache:
            logger.info("[CACHE] Cache invalidation requested - clearing prompt caches")
            for prompt in prompts:
                cache_dir = get_prompt_cache_dir(model_name, prompt)
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)

        # Check which prompts are already cached
        cached_prompts = []
        missing_prompts = []
        missing_indices = []  # Track original indices for ordering

        for i, prompt in enumerate(prompts):
            if is_prompt_fully_cached(model_name, prompt, required_layers):
                cached_prompts.append(prompt)
            else:
                missing_prompts.append(prompt)
                missing_indices.append(i)

        n_cached = len(cached_prompts)
        n_missing = len(missing_prompts)

        if n_cached > 0:
            logger.info(f"[CACHE] HIT: {n_cached}/{len(prompts)} prompts already cached")
        if n_missing > 0:
            logger.info(f"[CACHE] MISS: {n_missing}/{len(prompts)} prompts need extraction")

        # Extract missing prompts in batches, saving after each batch
        if missing_prompts:
            model = self.extractor.model
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
                    batch_prompts = missing_prompts[batch_idx : batch_idx + batch_size]

                    # Extract this batch
                    batch_acts, batch_mask = _extract_batch(
                        model, batch_prompts, layer_indices, remote=remote
                    )

                    # Save each prompt in the batch immediately
                    for j, prompt in enumerate(batch_prompts):
                        prompt_acts = batch_acts[j : j + 1]  # Keep batch dim
                        prompt_mask = batch_mask[j : j + 1]
                        save_prompt_activations(
                            model_name, prompt, layer_indices, prompt_acts, prompt_mask
                        )

                    logger.debug(
                        f"[CACHE] Saved batch {batch_idx // batch_size + 1}/{num_batches} "
                        f"({len(batch_prompts)} prompts)"
                    )

            logger.info(f"[CACHE] Extraction complete - all {n_missing} prompts cached")
        else:
            logger.info("[CACHE] 100% cache hit - no model inference needed!")

        # Now load all prompts from cache in original order
        logger.debug(f"[CACHE] Loading {len(prompts)} prompts from cache...")

        all_activations = []
        all_masks = []

        for prompt in prompts:
            acts, mask = load_prompt_activations(model_name, prompt, layer_indices)
            all_activations.append(acts)
            all_masks.append(mask)

        # Pad to same sequence length and concatenate
        max_seq_len = max(acts.shape[1] for acts in all_activations)
        hidden_dim = all_activations[0].shape[2]

        padded_activations = []
        padded_masks = []

        for acts, mask in zip(all_activations, all_masks):
            seq_len = acts.shape[1]
            if seq_len < max_seq_len:
                # Pad activations with zeros
                pad_size = max_seq_len - seq_len
                acts_pad = torch.zeros(1, pad_size, hidden_dim, dtype=acts.dtype)
                acts = torch.cat([acts, acts_pad], dim=1)

                # Pad mask with zeros
                mask_pad = torch.zeros(1, pad_size, dtype=mask.dtype)
                mask = torch.cat([mask, mask_pad], dim=1)

            padded_activations.append(acts)
            padded_masks.append(mask)

        # Concatenate along batch dimension
        activations = torch.cat(padded_activations, dim=0)
        attention_mask = torch.cat(padded_masks, dim=0)

        logger.info(
            f"[CACHE] Complete: returned activations shape {tuple(activations.shape)} "
            f"({n_cached} cached + {n_missing} extracted)"
        )

        return activations, attention_mask
