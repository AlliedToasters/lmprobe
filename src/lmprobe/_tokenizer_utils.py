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
        # Suppress the "incorrect regex pattern" warning the fallback load
        # emits — when we post-patch below, it's misleading noise.
        with _suppress_mistral_regex_warning():
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        if apply_fix:
            _apply_mistral_regex_fix(tokenizer, model_name)

    if load_chat_template and getattr(tokenizer, "chat_template", None) is None:
        _maybe_attach_chat_template(tokenizer, model_name)

    return tokenizer


class _MistralRegexWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        return "incorrect regex pattern" not in record.getMessage()


def _suppress_mistral_regex_warning() -> Any:
    """Context manager that drops the transformers mistral-regex warning.

    The warning fires inside :func:`AutoTokenizer.from_pretrained` whenever
    ``fix_mistral_regex`` isn't passed. We take that code path in the
    dup-kwarg fallback and then apply the patch ourselves, so the warning
    is misleading noise for downstream users.
    """
    from contextlib import contextmanager

    @contextmanager
    def _cm() -> Any:
        # The warning is emitted from
        # ``transformers.tokenization_utils_tokenizers``. Logger-level filters
        # don't apply to records propagated from children, so attach directly
        # to that logger. Also attach to the parent handler as a safety net
        # in case transformers restructures the logger hierarchy.
        emitter = logging.getLogger("transformers.tokenization_utils_tokenizers")
        parent = logging.getLogger("transformers")
        flt = _MistralRegexWarningFilter()
        emitter.addFilter(flt)
        parent_handlers = list(parent.handlers)
        for h in parent_handlers:
            h.addFilter(flt)
        try:
            yield
        finally:
            emitter.removeFilter(flt)
            for h in parent_handlers:
                h.removeFilter(flt)

    return _cm()


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
