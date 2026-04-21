"""Mistral3 / multimodal support in :class:`DiskOffloadBackend`.

The 24B Mistral-Small-3.1 checkpoint exposes a ``Mistral3Config`` (unregistered
for ``AutoModelForCausalLM``) and stores text weights under
``language_model.*``. These tests pin the shard-map / skeleton logic that lets
the disk_offload backend handle that layout without needing the 48 GB of
weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lmprobe.backends import DiskOffloadBackend


def _make_mistral3_like_config() -> SimpleNamespace:
    """Approximate shape of ``Mistral3Config``: outer wrapper + text_config."""
    text = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        quantization_config=None,
    )
    return SimpleNamespace(text_config=text, quantization_config=None)


def _write_index(path: Path, weight_map: dict[str, str]) -> None:
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def test_shard_map_strips_language_model_prefix(tmp_path: Path) -> None:
    """``language_model.`` prefix is stripped so the text skeleton sees plain names."""
    _write_index(tmp_path, {
        "language_model.model.embed_tokens.weight": "a.safetensors",
        "language_model.model.layers.0.self_attn.q_proj.weight": "a.safetensors",
        "language_model.model.layers.1.mlp.down_proj.weight": "a.safetensors",
        "language_model.model.norm.weight": "a.safetensors",
        "language_model.lm_head.weight": "a.safetensors",
    })

    b = DiskOffloadBackend("fake/model", device="cpu", dtype=torch.bfloat16)
    with (
        patch.object(b, "_get_config", return_value=_make_mistral3_like_config()),
        patch.object(b, "_get_snapshot_dir", return_value=tmp_path),
    ):
        layer_map, non_layer = b._get_shard_map()

    layer_names = {n for entries in layer_map.values() for n, _ in entries}
    non_layer_names = {n for n, _ in non_layer}

    assert layer_names == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    }
    assert non_layer_names == {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }


def test_shard_map_drops_vision_and_projector_tensors(tmp_path: Path) -> None:
    """Vision tower / multi-modal projector weights must not enter the text skeleton."""
    _write_index(tmp_path, {
        "language_model.model.layers.0.self_attn.q_proj.weight": "a.safetensors",
        "vision_tower.patch_conv.weight": "b.safetensors",
        "vision_tower.transformer.layers.0.attention.q_proj.weight": "b.safetensors",
        "multi_modal_projector.linear_1.weight": "c.safetensors",
    })

    b = DiskOffloadBackend("fake/model", device="cpu", dtype=torch.bfloat16)
    with (
        patch.object(b, "_get_config", return_value=_make_mistral3_like_config()),
        patch.object(b, "_get_snapshot_dir", return_value=tmp_path),
    ):
        layer_map, non_layer = b._get_shard_map()

    all_names = {n for entries in layer_map.values() for n, _ in entries}
    all_names |= {n for n, _ in non_layer}
    assert all_names == {"model.layers.0.self_attn.q_proj.weight"}


def test_shard_map_text_only_model_unchanged(tmp_path: Path) -> None:
    """Single-config models keep original names; no prefix is stripped."""
    _write_index(tmp_path, {
        "model.embed_tokens.weight": "a.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "a.safetensors",
        "lm_head.weight": "a.safetensors",
    })

    text_only = SimpleNamespace(
        num_hidden_layers=1, hidden_size=8, num_attention_heads=2,
        quantization_config=None,
    )
    b = DiskOffloadBackend("fake/text-only", device="cpu", dtype=torch.bfloat16)
    with (
        patch.object(b, "_get_config", return_value=text_only),
        patch.object(b, "_get_snapshot_dir", return_value=tmp_path),
    ):
        assert b._skeleton_prefix == ""
        layer_map, non_layer = b._get_shard_map()

    layer_names = {n for entries in layer_map.values() for n, _ in entries}
    assert layer_names == {"model.layers.0.self_attn.q_proj.weight"}
    assert {n for n, _ in non_layer} == {
        "model.embed_tokens.weight",
        "lm_head.weight",
    }


def test_get_text_config_returns_text_config_when_present() -> None:
    b = DiskOffloadBackend("fake/model", device="cpu", dtype=torch.bfloat16)
    outer = _make_mistral3_like_config()
    with patch.object(b, "_get_config", return_value=outer):
        assert b._get_text_config() is outer.text_config
        assert b._skeleton_prefix == "language_model."


def test_get_text_config_falls_back_when_absent() -> None:
    b = DiskOffloadBackend("fake/text-only", device="cpu", dtype=torch.bfloat16)
    text_only = SimpleNamespace(num_hidden_layers=1, hidden_size=8)
    with patch.object(b, "_get_config", return_value=text_only):
        assert b._get_text_config() is text_only
        assert b._skeleton_prefix == ""


def test_load_tensors_reattaches_prefix_for_shard_lookup(tmp_path: Path) -> None:
    """Shard lookup uses the prefixed key; the result dict uses the stripped key."""
    from safetensors.torch import save_file

    t = torch.ones(2, 2)
    save_file(
        {"language_model.model.layers.0.self_attn.q_proj.weight": t},
        str(tmp_path / "a.safetensors"),
    )

    b = DiskOffloadBackend("fake/model", device="cpu", dtype=torch.bfloat16)
    with (
        patch.object(b, "_get_config", return_value=_make_mistral3_like_config()),
        patch.object(b, "_get_snapshot_dir", return_value=tmp_path),
    ):
        result = b._load_tensors(
            [("model.layers.0.self_attn.q_proj.weight", "a.safetensors")],
            device="cpu",
        )

    assert list(result.keys()) == ["model.layers.0.self_attn.q_proj.weight"]
    assert torch.equal(result["model.layers.0.self_attn.q_proj.weight"], t)
