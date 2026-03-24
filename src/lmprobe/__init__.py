"""lmprobe: Train probes on language model activations.

This library makes it easy to train text classifiers using the internal
representations of language models, enabling AI safety monitoring through
detection of deception, harmful intent, and other safety-relevant properties.

Example
-------
>>> from lmprobe import Probe
>>>
>>> probe = Probe(
...     model="meta-llama/Llama-3.1-8B-Instruct",
...     layers=16,
...     pooling="last_token",
... )
>>> probe.fit(positive_prompts, negative_prompts)
>>> predictions = probe.predict(test_prompts)
"""

from .activation_baseline import ActivationBaseline
from .baseline import BaselineProbe
from .battery import BaselineBattery, BaselineResult, BaselineResults
from .ensemble import ProbeEnsemble
from .probe import LayerSweepResult, LinearProbe, Probe

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("lmprobe")
except Exception:
    __version__ = "unknown"
__all__ = [
    "ActivationBaseline",
    "BaselineBattery",
    "BaselineProbe",
    "BaselineResult",
    "BaselineResults",
    "CacheInfo",
    "CachedLogits",
    "CachedPromptInfo",
    "DatasetMetadata",
    "LayerSweepResult",
    "LinearProbe",
    "ManifestEntry",
    "Probe",
    "ProbeEnsemble",
    "load_layer_across_prompts",
    "load_layer_last_token",
    "ModelCacheInfo",
    "PerLayerScaler",
    "ProbeCard",
    "UnifiedCache",
    "WarmupStats",
    "cache_info",
    "clear_model_cache",
    "discover_cached",
    "enable_cache_logging",
    "list_cached_prompts",
    "fetch_dataset_metadata",
    "load_activation_dataset",
    "migrate_dataset",
    "plot_layer_importance",
    "plot_layer_importance_heatmap",
    "pull_dataset",
    "push_dataset",
    "evict",
    "set_cache_backend",
    "set_cache_dtype",
    "set_cache_limit",
    "set_max_threads",
]


def set_max_threads(n: int) -> None:
    """Set maximum number of CPU threads for PyTorch and BLAS libraries.

    Parameters
    ----------
    n : int
        Maximum number of threads to use.
    """
    import os

    import torch

    torch.set_num_threads(n)
    torch.set_num_interop_threads(n)
    for var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(n)


def __getattr__(name: str):
    """Lazy import for optional modules."""
    if name in ("plot_layer_importance", "plot_layer_importance_heatmap"):
        from .plotting import plot_layer_importance, plot_layer_importance_heatmap

        return {
            "plot_layer_importance": plot_layer_importance,
            "plot_layer_importance_heatmap": plot_layer_importance_heatmap,
        }[name]
    if name == "PerLayerScaler":
        from .scaling import PerLayerScaler

        return PerLayerScaler
    if name == "clear_model_cache":
        from .extraction import clear_model_cache

        return clear_model_cache
    if name == "enable_cache_logging":
        from .cache import enable_cache_logging

        return enable_cache_logging
    if name in (
        "cache_info", "CacheInfo", "ModelCacheInfo",
        "evict", "set_cache_backend", "set_cache_limit", "set_cache_dtype",
    ):
        from .cache import (
            CacheInfo,
            ModelCacheInfo,
            cache_info,
            evict,
            set_cache_backend,
            set_cache_dtype,
            set_cache_limit,
        )

        return {
            "cache_info": cache_info,
            "CacheInfo": CacheInfo,
            "ModelCacheInfo": ModelCacheInfo,
            "evict": evict,
            "set_cache_backend": set_cache_backend,
            "set_cache_limit": set_cache_limit,
            "set_cache_dtype": set_cache_dtype,
        }[name]
    if name in ("UnifiedCache", "WarmupStats", "CachedLogits"):
        from .unified_cache import CachedLogits, UnifiedCache, WarmupStats

        return {
            "UnifiedCache": UnifiedCache,
            "WarmupStats": WarmupStats,
            "CachedLogits": CachedLogits,
        }[name]
    if name == "ProbeCard":
        from .hub import ProbeCard

        return ProbeCard
    if name in ("load_layer_across_prompts", "load_layer_last_token"):
        from .cache import load_layer_across_prompts, load_layer_last_token

        return {
            "load_layer_across_prompts": load_layer_across_prompts,
            "load_layer_last_token": load_layer_last_token,
        }[name]
    if name in ("CachedPromptInfo", "discover_cached", "ManifestEntry", "list_cached_prompts"):
        from .cache import (
            CachedPromptInfo,
            ManifestEntry,
            discover_cached,
            list_cached_prompts,
        )

        return {
            "CachedPromptInfo": CachedPromptInfo,
            "ManifestEntry": ManifestEntry,
            "discover_cached": discover_cached,
            "list_cached_prompts": list_cached_prompts,
        }[name]
    if name in (
        "push_dataset", "pull_dataset", "load_activation_dataset",
        "fetch_dataset_metadata", "DatasetMetadata", "migrate_dataset",
    ):
        from .sharing import (
            DatasetMetadata,
            fetch_dataset_metadata,
            load_activation_dataset,
            migrate_dataset,
            pull_dataset,
            push_dataset,
        )

        return {
            "push_dataset": push_dataset,
            "pull_dataset": pull_dataset,
            "load_activation_dataset": load_activation_dataset,
            "fetch_dataset_metadata": fetch_dataset_metadata,
            "DatasetMetadata": DatasetMetadata,
            "migrate_dataset": migrate_dataset,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
