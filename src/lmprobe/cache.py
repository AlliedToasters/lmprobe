"""Activation caching for lmprobe.

This module provides disk-based caching of extracted activations to avoid
redundant model inference, especially important for remote execution.

Cache structure (per-layer for maximum reuse):
    ~/.cache/lmprobe/
      {model_hash}/
        {prompts_hash}/
          layer_8.pt           # (batch, seq_len, hidden_dim)
          layer_16.pt
          attention_mask.pt    # shared across layers

This structure enables:
- O(1) lookup of which layers are cached (just list directory)
- Partial cache hits (reuse layer 8 even when also requesting layer 16)
- No loading until strictly necessary
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import torch


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
    """Wraps an ActivationExtractor with per-layer caching.

    This extractor checks which layers are already cached before extraction,
    and only extracts the missing layers. This enables:
    - Reusing layer 8 from a previous extraction of [8] when now requesting [8, 16]
    - Partial cache hits across different layer combinations
    - Efficient iterative experimentation with different layer sets

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
        """Extract activations, using per-layer cache when available.

        Parameters
        ----------
        prompts : list[str]
            Text prompts.
        remote : bool
            Whether to use remote execution.
        invalidate_cache : bool
            If True, delete cached values and re-extract all layers.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (activations, attention_mask) where activations has shape
            (batch, seq_len, n_layers * hidden_dim) with layers concatenated
            in sorted order.
        """
        cache_dir = get_extraction_cache_dir(
            self.extractor.model_name,
            prompts,
        )

        requested_layers = set(self.extractor.layer_indices)

        # Handle cache invalidation
        if invalidate_cache:
            invalidate_extraction_cache(cache_dir)
            cached_layers = set()
        else:
            cached_layers = get_cached_layers(cache_dir)

        # Determine which layers we need to extract
        missing_layers = requested_layers - cached_layers
        have_cached = requested_layers & cached_layers

        # Load attention mask if we have any cached layers
        attention_mask = None
        if have_cached and not missing_layers:
            # All layers cached - just load attention mask
            try:
                attention_mask = load_attention_mask(cache_dir)
            except FileNotFoundError:
                # Mask missing, need to re-extract
                missing_layers = requested_layers
                have_cached = set()

        # Extract missing layers if any
        if missing_layers:
            # Extract only the missing layers
            fresh_activations, attention_mask = self.extractor.extract(
                prompts,
                remote=remote,
                layers=sorted(missing_layers),
            )

            # Split and cache each layer individually
            n_missing = len(missing_layers)
            hidden_dim = fresh_activations.shape[-1] // n_missing

            for i, layer in enumerate(sorted(missing_layers)):
                start = i * hidden_dim
                end = (i + 1) * hidden_dim
                layer_acts = fresh_activations[..., start:end]
                save_layer(cache_dir, layer, layer_acts)

            # Save attention mask
            save_attention_mask(cache_dir, attention_mask)

        # Now load all requested layers and concatenate in sorted order
        all_layer_acts = []
        for layer in sorted(requested_layers):
            layer_acts = load_layer(cache_dir, layer)
            all_layer_acts.append(layer_acts)

        # Load attention mask if not already loaded
        if attention_mask is None:
            attention_mask = load_attention_mask(cache_dir)

        # Concatenate layers
        activations = torch.cat(all_layer_acts, dim=-1)

        return activations, attention_mask
