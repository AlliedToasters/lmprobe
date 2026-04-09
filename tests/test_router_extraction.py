"""Tests for MoE router extraction via LocalBackend and cache round-trip."""

import pytest
import torch

from lmprobe.activation_types import ExtractedBatch, ExtractionSpec
from lmprobe.backends import LocalBackend, _get_router_modules


class TestGetRouterModules:
    """Test _get_router_modules with a real tiny model."""

    def test_dense_model_returns_empty(self, tiny_model):
        """Dense models have no router modules."""
        backend = LocalBackend(tiny_model, device="cpu")
        model = backend.model
        result = _get_router_modules(
            model,
            [0, 1],
            # Use a template that won't exist on a dense model
            "model.layers.{layer}.block_sparse_moe.gate",
        )
        assert result == {}


class TestLocalBackendExtractBatchExtended:
    """Test extract_batch_extended on LocalBackend with a real tiny model."""

    def test_hidden_only(self, tiny_model):
        """Extract hidden states only via extended method."""
        backend = LocalBackend(tiny_model, device="cpu")
        spec = ExtractionSpec(hidden_layers=[0, 1])
        result = backend.extract_batch_extended(
            ["Hello world", "Test prompt"],
            spec,
        )
        assert isinstance(result, ExtractedBatch)
        assert result.activations is not None
        assert result.attention_mask is not None
        assert result.logits is None
        assert result.router_logits is None
        # Check shapes
        batch_size = 2
        assert result.activations.shape[0] == batch_size
        assert result.attention_mask.shape[0] == batch_size

    def test_hidden_with_logits(self, tiny_model):
        """Extract hidden states + logits via extended method."""
        backend = LocalBackend(tiny_model, device="cpu")
        spec = ExtractionSpec(hidden_layers=[0], include_logits=True)
        result = backend.extract_batch_extended(
            ["Hello world"],
            spec,
        )
        assert result.activations is not None
        assert result.logits is not None
        assert result.logits.dim() == 3  # (batch, seq, vocab)

    def test_no_hidden_with_logits(self, tiny_model):
        """Extract logits only (no hidden states)."""
        backend = LocalBackend(tiny_model, device="cpu")
        spec = ExtractionSpec(hidden_layers=[], include_logits=True)
        result = backend.extract_batch_extended(
            ["Hello world"],
            spec,
        )
        assert result.activations is None
        assert result.logits is not None

    def test_router_on_dense_model_no_crash(self, tiny_model):
        """Requesting router logits on a dense model shouldn't crash,
        but router_logits should be None (no modules found)."""
        backend = LocalBackend(tiny_model, device="cpu")
        spec = ExtractionSpec(
            hidden_layers=[0],
            router_layers=[0],
            router_module_template="model.layers.{layer}.block_sparse_moe.gate",
        )
        result = backend.extract_batch_extended(
            ["Hello world"],
            spec,
        )
        assert result.activations is not None
        # Router logits should be None since no router modules exist
        assert result.router_logits is None


class TestCacheRoundTrip:
    """Test saving and loading router logits from cache."""

    def test_save_load_router_logits(self, tmp_path, monkeypatch):
        """Router logits survive a cache round-trip."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import (
            load_prompt_router_logits,
            save_prompt_router_logits,
        )

        model_name = "test-model"
        prompt = "Hello world"
        router_logits = {
            0: torch.randn(5, 8),   # 5 tokens, 8 experts
            3: torch.randn(5, 8),
        }

        save_prompt_router_logits(model_name, prompt, router_logits)
        loaded = load_prompt_router_logits(model_name, prompt, [0, 3])

        assert set(loaded.keys()) == {0, 3}
        torch.testing.assert_close(loaded[0], router_logits[0])
        torch.testing.assert_close(loaded[3], router_logits[3])

    def test_load_missing_layer_raises(self, tmp_path, monkeypatch):
        """Loading a router layer that wasn't saved raises KeyError."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import (
            load_prompt_router_logits,
            save_prompt_router_logits,
        )

        model_name = "test-model"
        prompt = "Hello world"
        save_prompt_router_logits(
            model_name, prompt, {0: torch.randn(5, 8)}
        )

        with pytest.raises(KeyError, match="router_layer_5"):
            load_prompt_router_logits(model_name, prompt, [0, 5])


class TestCachedPromptInfoRouter:
    """Test that discover_cached includes router layer info."""

    def test_discover_with_router(self, tmp_path, monkeypatch):
        """CachedPromptInfo.router_layers populated after saving router logits."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import (
            discover_cached,
            save_prompt_activations,
            save_prompt_router_logits,
        )

        model_name = "test-model"
        prompt = "Hello world"

        # Save hidden states first (so there's a cache entry)
        save_prompt_activations(
            model_name, prompt,
            layers=[0, 1],
            activations=torch.randn(1, 5, 64),  # 1 batch, 5 tokens, 2*32 dim
            attention_mask=torch.ones(5),
        )

        # Save router logits
        save_prompt_router_logits(
            model_name, prompt,
            {0: torch.randn(5, 8), 1: torch.randn(5, 8)},
        )

        info = discover_cached(model_name, prompt)
        assert info is not None
        assert sorted(info.router_layers) == [0, 1]

    def test_discover_without_router(self, tmp_path, monkeypatch):
        """CachedPromptInfo.router_layers is empty when no router data cached."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import (
            discover_cached,
            save_prompt_activations,
        )

        model_name = "test-model"
        prompt = "Hello world"
        save_prompt_activations(
            model_name, prompt,
            layers=[0],
            activations=torch.randn(1, 5, 32),
            attention_mask=torch.ones(5),
        )

        info = discover_cached(model_name, prompt)
        assert info is not None
        assert info.router_layers == []
