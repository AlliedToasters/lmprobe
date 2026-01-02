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

from .probe import LinearProbe

__version__ = "0.1.0"
__all__ = ["LinearProbe"]
