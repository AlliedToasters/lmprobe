"""Activation caching for lmprobe.

This module provides disk-based caching of extracted activations to avoid
redundant model inference, especially important for remote execution.

Cache location: ~/.cache/lmprobe/ or LMPROBE_CACHE_DIR environment variable.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def get_cache_dir() -> Path:
    """Get the cache directory, creating it if necessary.

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


def compute_cache_key(
    model_name: str,
    prompts: list[str],
    layer_indices: list[int],
) -> str:
    """Compute a unique cache key for a set of extraction parameters.

    Parameters
    ----------
    model_name : str
        The model identifier.
    prompts : list[str]
        The prompts to extract activations for.
    layer_indices : list[int]
        The layer indices being extracted.

    Returns
    -------
    str
        A hex digest suitable for use as a filename.
    """
    # Create a deterministic representation
    data = {
        "model": model_name,
        "prompts": prompts,
        "layers": sorted(layer_indices),
    }
    # Use JSON for deterministic serialization
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=True)
    # Hash it
    return hashlib.sha256(serialized.encode()).hexdigest()[:32]


def get_cache_path(cache_key: str) -> Path:
    """Get the file path for a cache key.

    Parameters
    ----------
    cache_key : str
        The cache key.

    Returns
    -------
    Path
        Path to the cache file.
    """
    return get_cache_dir() / f"{cache_key}.pt"


def load_from_cache(
    cache_key: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load cached activations if they exist.

    Parameters
    ----------
    cache_key : str
        The cache key.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] | None
        (activations, attention_mask) if cached, None otherwise.
    """
    cache_path = get_cache_path(cache_key)
    if not cache_path.exists():
        return None

    try:
        data = torch.load(cache_path, weights_only=True)
        return data["activations"], data["attention_mask"]
    except Exception:
        # If cache is corrupted, ignore it
        return None


def save_to_cache(
    cache_key: str,
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    """Save activations to cache.

    Parameters
    ----------
    cache_key : str
        The cache key.
    activations : torch.Tensor
        The activations tensor.
    attention_mask : torch.Tensor
        The attention mask tensor.
    """
    cache_path = get_cache_path(cache_key)
    data = {
        "activations": activations.cpu(),
        "attention_mask": attention_mask.cpu(),
    }
    torch.save(data, cache_path)


def clear_cache() -> int:
    """Clear all cached activations.

    Returns
    -------
    int
        Number of cache files deleted.
    """
    cache_dir = get_cache_dir()
    count = 0
    for cache_file in cache_dir.glob("*.pt"):
        cache_file.unlink()
        count += 1
    return count


class CachedExtractor:
    """Wraps an ActivationExtractor with caching.

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
        """Extract activations, using cache if available.

        Parameters
        ----------
        prompts : list[str]
            Text prompts.
        remote : bool
            Whether to use remote execution.
        invalidate_cache : bool
            If True, ignore cached values and re-extract.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (activations, attention_mask)
        """
        # Compute cache key
        cache_key = compute_cache_key(
            self.extractor.model_name,
            prompts,
            self.extractor.layer_indices,
        )

        # Try to load from cache
        if not invalidate_cache:
            cached = load_from_cache(cache_key)
            if cached is not None:
                return cached

        # Extract activations
        activations, attention_mask = self.extractor.extract(prompts, remote=remote)

        # Save to cache
        save_to_cache(cache_key, activations, attention_mask)

        return activations, attention_mask
