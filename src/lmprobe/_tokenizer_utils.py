"""Internal tokenizer loading utilities.

Centralizes ``AutoTokenizer.from_pretrained`` calls so model-specific
workarounds (e.g. the Mistral regex bug) apply everywhere by default.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_logger = logging.getLogger(__name__)

# Mistral checkpoints with a tokenizer regex bug in transformers. For
# these, ``AutoTokenizer.from_pretrained`` needs ``fix_mistral_regex=True``
# to produce correct token boundaries. Add new entries here as more
# affected checkpoints surface.
MISTRAL_MODELS_WITH_REGEX_BUG = [
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "mistralai/Mistral-Large",
]


def load_tokenizer(
    model_name: str,
    load_chat_template: bool = True,
    **kwargs: Any,
) -> PreTrainedTokenizerBase:
    """Load a tokenizer with per-model workarounds applied.

    Wraps ``AutoTokenizer.from_pretrained`` and injects
    ``fix_mistral_regex=True`` for checkpoints listed in
    :data:`MISTRAL_MODELS_WITH_REGEX_BUG`.

    When ``load_chat_template`` is True (default) and the loaded tokenizer
    has no ``chat_template`` attribute, attempt to fetch ``chat_template.json``
    from the repo and attach it. This is the canonical path for multimodal
    checkpoints (e.g. Pixtral / Mistral-Small-3.1) where ``AutoProcessor``
    would load it but ``AutoTokenizer`` alone does not.
    """
    from transformers import AutoTokenizer

    if any(m in model_name for m in MISTRAL_MODELS_WITH_REGEX_BUG):
        kwargs.setdefault("fix_mistral_regex", True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    except TypeError as e:
        # transformers>=5.3 has an upstream bug where `fix_mistral_regex`
        # is passed to `_patch_mistral_regex` both explicitly and via
        # `**kwargs`, raising "got multiple values for keyword argument
        # 'fix_mistral_regex'". Fall back to loading without the flag
        # and applying the patch post-load. See issue #280.
        if "fix_mistral_regex" not in str(e):
            raise
        apply_fix = kwargs.pop("fix_mistral_regex", None) is True
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        if apply_fix:
            _apply_mistral_regex_fix(tokenizer, model_name)

    if load_chat_template and getattr(tokenizer, "chat_template", None) is None:
        _maybe_attach_chat_template(tokenizer, model_name)

    return tokenizer


def _apply_mistral_regex_fix(tokenizer: PreTrainedTokenizerBase, model_name: str) -> None:
    """Apply the Mistral pre-tokenizer regex fix to an already-loaded tokenizer.

    Used as a fallback when passing ``fix_mistral_regex=True`` to
    ``AutoTokenizer.from_pretrained`` raises on transformers>=5.3 (see #280).
    Invokes the upstream ``TokenizersBackend._patch_mistral_regex`` classmethod
    directly — private API, but the narrowest workaround available.
    """
    from transformers.tokenization_utils_tokenizers import TokenizersBackend

    TokenizersBackend._patch_mistral_regex(
        tokenizer,
        pretrained_model_name_or_path=model_name,
        fix_mistral_regex=True,
    )
    _logger.info("Applied Mistral pre-tokenizer regex fix post-load for %s", model_name)


def _maybe_attach_chat_template(tokenizer: PreTrainedTokenizerBase, model_name: str) -> None:
    """Attach ``chat_template.json`` from the repo if present.

    Silently no-ops if the repo doesn't ship a ``chat_template.json`` or if
    the file is unreadable. Logs an INFO line on success so template
    forensics stay traceable.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        path = hf_hub_download(model_name, "chat_template.json")
    except EntryNotFoundError:
        return
    except Exception as e:  # network/auth/etc — don't fail the load
        _logger.debug("chat_template.json fetch failed for %s: %s", model_name, e)
        return

    try:
        with open(path) as f:
            data = json.load(f)
        template = data["chat_template"]
    except (OSError, KeyError, json.JSONDecodeError) as e:
        _logger.debug("chat_template.json parse failed for %s: %s", model_name, e)
        return

    tokenizer.chat_template = template
    _logger.info("Loaded chat_template from chat_template.json for %s", model_name)
