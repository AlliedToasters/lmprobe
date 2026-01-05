"""lmprobe: Train linear probes on language model activations.

This library makes it easy to train text classifiers using the internal
representations of language models, enabling AI safety monitoring through
detection of deception, harmful intent, and other safety-relevant properties.

Example
-------
>>> from lmprobe import LinearProbe
>>>
>>> probe = LinearProbe(
...     model="meta-llama/Llama-3.1-8B-Instruct",
...     layers=16,
...     pooling="last_token",
... )
>>> probe.fit(positive_prompts, negative_prompts)
>>> predictions = probe.predict(test_prompts)
"""

from .baseline import BaselineProbe
from .probe import LinearProbe

__version__ = "0.1.0"
__all__ = [
    "BaselineProbe",
    "LinearProbe",
    "PerLayerScaler",
    "clear_model_cache",
    "plot_layer_importance",
    "plot_layer_importance_heatmap",
]


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
