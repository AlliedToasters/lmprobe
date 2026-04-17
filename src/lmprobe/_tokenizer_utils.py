"""Internal tokenizer loading utilities.

Centralizes ``AutoTokenizer.from_pretrained`` calls so model-specific
workarounds (e.g. the Mistral regex bug) apply everywhere by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Mistral checkpoints with a tokenizer regex bug in transformers. For
# these, ``AutoTokenizer.from_pretrained`` needs ``fix_mistral_regex=True``
# to produce correct token boundaries. Add new entries here as more
# affected checkpoints surface.
MISTRAL_MODELS_WITH_REGEX_BUG = [
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "mistralai/Mistral-Large",
]


def load_tokenizer(model_name: str, **kwargs: Any) -> PreTrainedTokenizerBase:
    """Load a tokenizer with per-model workarounds applied.

    Wraps ``AutoTokenizer.from_pretrained`` and injects
    ``fix_mistral_regex=True`` for checkpoints listed in
    :data:`MISTRAL_MODELS_WITH_REGEX_BUG`.
    """
    from transformers import AutoTokenizer

    if any(m in model_name for m in MISTRAL_MODELS_WITH_REGEX_BUG):
        kwargs.setdefault("fix_mistral_regex", True)
    return AutoTokenizer.from_pretrained(model_name, **kwargs)
