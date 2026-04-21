"""Tests for ``lmprobe._tokenizer_utils``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import EntryNotFoundError

from lmprobe._tokenizer_utils import (
    MISTRAL_MODELS_WITH_REGEX_BUG,
    _maybe_attach_chat_template,
    load_tokenizer,
)

TEMPLATE = "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}"


@pytest.fixture
def chat_template_file(tmp_path: Path) -> Path:
    p = tmp_path / "chat_template.json"
    p.write_text(json.dumps({"chat_template": TEMPLATE}))
    return p


def test_load_tokenizer_no_chat_template_file(tiny_model):
    """Tiny llama ships no chat_template.json — fallback is a silent no-op."""
    tok = load_tokenizer(tiny_model)
    assert tok.chat_template is None


def test_attach_chat_template_sets_attribute(tiny_model, chat_template_file):
    tok = load_tokenizer(tiny_model, load_chat_template=False)
    assert tok.chat_template is None

    with patch("huggingface_hub.hf_hub_download", return_value=str(chat_template_file)):
        _maybe_attach_chat_template(tok, tiny_model)

    assert tok.chat_template == TEMPLATE
    rendered = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)
    assert "<user>hi</user>" in rendered


def test_attach_chat_template_silent_on_missing_file(tiny_model):
    tok = load_tokenizer(tiny_model, load_chat_template=False)

    def _raise(*_a, **_kw):
        raise EntryNotFoundError("no such file")

    with patch("huggingface_hub.hf_hub_download", side_effect=_raise):
        _maybe_attach_chat_template(tok, tiny_model)

    assert tok.chat_template is None


def test_load_tokenizer_opt_out(tiny_model, chat_template_file):
    """``load_chat_template=False`` skips the fallback even when a file exists."""
    with patch("huggingface_hub.hf_hub_download", return_value=str(chat_template_file)) as mock_dl:
        tok = load_tokenizer(tiny_model, load_chat_template=False)

    assert tok.chat_template is None
    mock_dl.assert_not_called()


def test_attach_chat_template_silent_on_malformed_json(tiny_model, tmp_path):
    tok = load_tokenizer(tiny_model, load_chat_template=False)
    bad = tmp_path / "chat_template.json"
    bad.write_text("not-json{")

    with patch("huggingface_hub.hf_hub_download", return_value=str(bad)):
        _maybe_attach_chat_template(tok, tiny_model)

    assert tok.chat_template is None


def test_mistral_regex_falls_back_on_dup_kwarg_typeerror():
    """On transformers>=5.3, passing fix_mistral_regex triggers a TypeError.

    Verify load_tokenizer catches it, retries without the flag, and calls
    the post-load patch helper. See issue #280.
    """
    mistral_name = MISTRAL_MODELS_WITH_REGEX_BUG[0]
    fake_tok = MagicMock()
    fake_tok.chat_template = "existing"  # skip chat_template fallback

    call_count = {"n": 0}

    def fake_from_pretrained(_name, **kwargs):
        call_count["n"] += 1
        if "fix_mistral_regex" in kwargs:
            raise TypeError(
                "_patch_mistral_regex() got multiple values for "
                "keyword argument 'fix_mistral_regex'"
            )
        return fake_tok

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=fake_from_pretrained,
        ),
        patch("lmprobe._tokenizer_utils._apply_mistral_regex_fix") as mock_apply,
    ):
        tok = load_tokenizer(mistral_name)

    assert tok is fake_tok
    assert call_count["n"] == 2  # first raised, second succeeded
    mock_apply.assert_called_once_with(fake_tok, mistral_name)


def test_unrelated_typeerror_is_re_raised():
    """TypeErrors that aren't the fix_mistral_regex dup-kwarg bug must propagate."""

    def fake_from_pretrained(*_a, **_kw):
        raise TypeError("something else entirely")

    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        side_effect=fake_from_pretrained,
    ):
        with pytest.raises(TypeError, match="something else entirely"):
            load_tokenizer(MISTRAL_MODELS_WITH_REGEX_BUG[0])


def test_mistral_regex_warning_silenced_on_dup_kwarg_fallback(caplog):
    """The noisy 'incorrect regex pattern' warning must not reach downstream loggers.

    When the dup-kwarg fallback loads the tokenizer without the flag,
    transformers emits a misleading warning before we post-patch. It should
    be filtered out.
    """
    import logging as _logging

    mistral_name = MISTRAL_MODELS_WITH_REGEX_BUG[0]
    fake_tok = MagicMock()
    fake_tok.chat_template = "existing"

    emitter = _logging.getLogger("transformers.tokenization_utils_tokenizers")

    def fake_from_pretrained(_name, **kwargs):
        if "fix_mistral_regex" in kwargs:
            raise TypeError(
                "got multiple values for keyword argument 'fix_mistral_regex'"
            )
        # Mimic the transformers warning that our suppressor targets.
        emitter.warning(
            "The tokenizer you are loading from '%s' with an incorrect "
            "regex pattern: ...",
            mistral_name,
        )
        return fake_tok

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=fake_from_pretrained,
        ),
        patch("lmprobe._tokenizer_utils._apply_mistral_regex_fix"),
        caplog.at_level(_logging.WARNING, logger="transformers.tokenization_utils_tokenizers"),
    ):
        load_tokenizer(mistral_name)

    noisy = [r for r in caplog.records if "incorrect regex pattern" in r.getMessage()]
    assert noisy == [], f"expected warning to be silenced, got: {[r.getMessage() for r in noisy]}"


def test_mistral_regex_not_applied_when_user_opts_out():
    """If user passes fix_mistral_regex=False, the post-load patch is skipped."""
    mistral_name = MISTRAL_MODELS_WITH_REGEX_BUG[0]
    fake_tok = MagicMock()
    fake_tok.chat_template = "existing"

    def fake_from_pretrained(_name, **kwargs):
        if "fix_mistral_regex" in kwargs:
            raise TypeError("got multiple values for keyword argument 'fix_mistral_regex'")
        return fake_tok

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=fake_from_pretrained,
        ),
        patch("lmprobe._tokenizer_utils._apply_mistral_regex_fix") as mock_apply,
    ):
        load_tokenizer(mistral_name, fix_mistral_regex=False)

    mock_apply.assert_not_called()
