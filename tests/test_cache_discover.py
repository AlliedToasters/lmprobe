"""Tests for discover_cached() — the public cache introspection API."""

import pytest
import torch

from lmprobe.cache import (
    CachedPromptInfo,
    discover_cached,
    save_prompt_activations,
    save_prompt_logits,
    save_prompt_perplexity,
    save_prompt_pooled_activations,
)

TEST_MODEL = "stas/tiny-random-llama-2"


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Set up a temporary cache directory."""
    monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
    return tmp_path


class TestDiscoverCached:
    def test_nothing_cached(self, cache_dir):
        result = discover_cached(TEST_MODEL, "uncached prompt")
        assert result is None

    def test_raw_activations_only(self, cache_dir):
        prompt = "test prompt"
        # Save raw activations for layers 0 and 1
        acts = torch.randn(1, 5, 32)  # (batch, seq_len, hidden_dim)
        mask = torch.ones(1, 5)
        save_prompt_activations(TEST_MODEL, prompt, [0], acts, mask)

        acts2 = torch.randn(1, 5, 32)
        save_prompt_activations(TEST_MODEL, prompt, [1], acts2, mask)

        info = discover_cached(TEST_MODEL, prompt)
        assert info is not None
        assert isinstance(info, CachedPromptInfo)
        assert 0 in info.raw_layers
        assert 1 in info.raw_layers
        assert info.raw_layers == [0, 1]
        assert info.pooled == {}
        assert info.has_logits is False
        assert info.logits_top_k is None
        assert info.has_perplexity is False
        assert info.num_tokens == 5

    def test_pooled_activations(self, cache_dir):
        prompt = "pooled prompt"
        pooled = torch.randn(1, 32)  # (batch, hidden_dim)
        save_prompt_pooled_activations(
            TEST_MODEL, prompt, [3], pooled, "last_token"
        )

        info = discover_cached(TEST_MODEL, prompt)
        assert info is not None
        assert "last_token" in info.pooled
        assert info.pooled["last_token"] == [3]

    def test_multiple_pooling_strategies(self, cache_dir):
        prompt = "multi pooled"
        pooled = torch.randn(1, 32)
        save_prompt_pooled_activations(
            TEST_MODEL, prompt, [0], pooled, "last_token"
        )
        save_prompt_pooled_activations(
            TEST_MODEL, prompt, [0], pooled, "mean"
        )

        info = discover_cached(TEST_MODEL, prompt)
        assert info is not None
        assert "last_token" in info.pooled
        assert "mean" in info.pooled

    def test_topk_logits(self, cache_dir):
        prompt = "logits prompt"
        logits = torch.randn(1, 5, 100)  # (batch, seq_len, vocab_size)
        mask = torch.ones(1, 5)
        save_prompt_logits(
            TEST_MODEL, prompt, logits, mask, top_k=10, positions="last"
        )

        info = discover_cached(TEST_MODEL, prompt)
        assert info is not None
        assert info.has_logits is False  # has_logits means full logits
        assert info.logits_top_k == 10

    def test_full_logits(self, cache_dir):
        prompt = "full logits"
        logits = torch.randn(1, 5, 100)
        mask = torch.ones(1, 5)
        save_prompt_logits(
            TEST_MODEL, prompt, logits, mask, top_k=None, positions="last"
        )

        info = discover_cached(TEST_MODEL, prompt)
        assert info is not None
        assert info.has_logits is True
        assert info.logits_top_k is None

    def test_perplexity(self, cache_dir):
        prompt = "perplexity prompt"
        perp = torch.tensor([1.5, 2.0, 0.8])
        save_prompt_perplexity(TEST_MODEL, prompt, perp)

        info = discover_cached(TEST_MODEL, prompt)
        assert info is not None
        assert info.has_perplexity is True

    def test_full_cache_entry(self, cache_dir):
        prompt = "complete entry"
        acts = torch.randn(1, 5, 32)
        mask = torch.ones(1, 5)
        save_prompt_activations(TEST_MODEL, prompt, [0, 1], acts, mask)

        pooled = torch.randn(1, 16)
        save_prompt_pooled_activations(
            TEST_MODEL, prompt, [0], pooled, "last_token"
        )

        logits = torch.randn(1, 5, 100)
        save_prompt_logits(
            TEST_MODEL, prompt, logits, mask, top_k=50, positions="last"
        )

        perp = torch.tensor([1.5, 2.0, 0.8])
        save_prompt_perplexity(TEST_MODEL, prompt, perp)

        info = discover_cached(TEST_MODEL, prompt)
        assert info is not None
        assert info.raw_layers == [0, 1]
        assert "last_token" in info.pooled
        assert info.logits_top_k == 50
        assert info.has_perplexity is True
        assert info.num_tokens == 5

    def test_different_models_isolated(self, cache_dir):
        prompt = "shared prompt"
        acts = torch.randn(1, 5, 32)
        mask = torch.ones(1, 5)
        save_prompt_activations("model-a", prompt, [0], acts, mask)

        info_a = discover_cached("model-a", prompt)
        info_b = discover_cached("model-b", prompt)
        assert info_a is not None
        assert info_b is None
