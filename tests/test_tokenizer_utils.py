"""Tests for ``lmprobe._tokenizer_utils``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub.errors import EntryNotFoundError

from lmprobe._tokenizer_utils import _maybe_attach_chat_template, load_tokenizer

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
