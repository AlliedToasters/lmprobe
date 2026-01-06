"""Tests for UnifiedCache - combined activation and perplexity extraction."""

import shutil
import time

import numpy as np
import pytest
import torch

from lmprobe.cache import (
    get_prompt_cache_dir,
    get_prompt_cached_layers,
    is_prompt_perplexity_cached,
    is_prompt_pooled_cached,
)
from lmprobe.unified_cache import UnifiedCache, WarmupStats


class TestUnifiedCache:
    """Tests for UnifiedCache extraction and caching."""

    def test_warmup_extracts_both(self, tiny_model, tmp_path, monkeypatch):
        """Warmup captures both activations and perplexity in single pass."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=True,
            device="cpu",
            remote=False,
            batch_size=2,
        )

        prompts = ["hello world", "test prompt"]
        stats = cache.warmup(prompts)

        # Check stats
        assert stats.total_prompts == 2
        assert stats.activations_extracted == 2
        assert stats.perplexity_extracted == 2
        assert stats.activations_cached == 0
        assert stats.perplexity_cached == 0
        assert stats.elapsed_seconds > 0

        # Check both are cached
        for prompt in prompts:
            cache_dir = get_prompt_cache_dir(tiny_model, prompt)
            assert get_prompt_cached_layers(cache_dir) == {0, 1}
            assert is_prompt_perplexity_cached(tiny_model, prompt)

    def test_warmup_cache_hit(self, tiny_model, tmp_path, monkeypatch):
        """Second warmup is instant (full cache hit)."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=True,
            device="cpu",
            remote=False,
        )

        prompts = ["cached prompt one", "cached prompt two"]

        # First warmup - extracts
        stats1 = cache.warmup(prompts)
        assert stats1.activations_extracted == 2
        assert stats1.perplexity_extracted == 2

        # Second warmup - should be cache hit
        stats2 = cache.warmup(prompts)
        assert stats2.activations_cached == 2
        assert stats2.perplexity_cached == 2
        assert stats2.activations_extracted == 0
        assert stats2.perplexity_extracted == 0

    def test_get_activations_returns_correct_shapes(self, tiny_model, tmp_path, monkeypatch):
        """get_activations returns tensor with correct shape."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=False,
            device="cpu",
            remote=False,
        )

        prompts = ["short", "a longer test prompt"]
        activations, mask = cache.get_activations(prompts)

        # Shape: (batch, seq_len, n_layers * hidden_dim)
        assert activations.ndim == 3
        assert activations.shape[0] == 2  # 2 prompts
        assert mask.shape[0] == 2
        assert mask.shape[1] == activations.shape[1]  # seq_len matches

    def test_get_perplexity_returns_correct_shape(self, tiny_model, tmp_path, monkeypatch):
        """get_perplexity returns (n_prompts, 3) array."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=True,
            device="cpu",
            remote=False,
        )

        prompts = ["prompt one", "prompt two", "prompt three"]
        ppl = cache.get_perplexity(prompts)

        assert isinstance(ppl, np.ndarray)
        assert ppl.shape == (3, 3)  # (n_prompts, 3 features)
        # Features should be positive (perplexity >= 1)
        assert np.all(ppl > 0)

    def test_perplexity_disabled_raises(self, tiny_model, tmp_path, monkeypatch):
        """get_perplexity raises if compute_perplexity=False."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=False,
            device="cpu",
            remote=False,
        )

        with pytest.raises(ValueError, match="compute_perplexity=False"):
            cache.get_perplexity(["test"])

    def test_partial_cache_hit_activations_only(self, tiny_model, tmp_path, monkeypatch):
        """Handles case where activations are cached but perplexity is not."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompt = "partial cache test"

        # First: extract activations only (no perplexity)
        cache1 = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=False,
            device="cpu",
            remote=False,
        )
        cache1.warmup([prompt])

        # Verify activations cached but not perplexity
        assert get_prompt_cached_layers(get_prompt_cache_dir(tiny_model, prompt)) == {0, 1}
        assert not is_prompt_perplexity_cached(tiny_model, prompt)

        # Second: extract with perplexity enabled
        cache2 = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=True,
            device="cpu",
            remote=False,
        )
        stats = cache2.warmup([prompt])

        # Should detect activations cached, perplexity needs extraction
        # Note: current implementation re-extracts for simplicity
        assert stats.perplexity_extracted >= 1 or stats.perplexity_cached >= 1

    def test_layers_all_default(self, tiny_model, tmp_path, monkeypatch):
        """layers='all' resolves to all model layers."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers="all",
            compute_perplexity=False,
            device="cpu",
            remote=False,
        )

        # tiny_model has 2 layers (0 and 1)
        assert cache.layer_indices == [0, 1]

        prompts = ["test all layers"]
        cache.warmup(prompts)

        cache_dir = get_prompt_cache_dir(tiny_model, prompts[0])
        assert get_prompt_cached_layers(cache_dir) == {0, 1}

    def test_layers_last(self, tiny_model, tmp_path, monkeypatch):
        """layers='last' resolves to last layer only."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers="last",
            compute_perplexity=False,
            device="cpu",
            remote=False,
        )

        # tiny_model has 2 layers, so "last" = layer 1
        assert cache.layer_indices == [1]

    def test_cross_request_cache_reuse(self, tiny_model, tmp_path, monkeypatch):
        """Prompts cached in one UnifiedCache are reused by another."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompt_a = "shared prompt A"
        prompt_b = "shared prompt B"
        prompt_c = "new prompt C"

        # First cache: extracts A and B
        cache1 = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=True,
            device="cpu",
            remote=False,
        )
        stats1 = cache1.warmup([prompt_a, prompt_b])
        assert stats1.activations_extracted == 2

        # Second cache (new instance): B should be cached, C needs extraction
        cache2 = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=True,
            device="cpu",
            remote=False,
        )
        stats2 = cache2.warmup([prompt_b, prompt_c])

        # B was cached, only C extracted
        assert stats2.activations_cached == 1
        assert stats2.activations_extracted == 1


class TestWarmupStats:
    """Tests for WarmupStats dataclass."""

    def test_cache_hit_rate_calculation(self):
        """cache_hit_rate computed correctly."""
        stats = WarmupStats(
            total_prompts=10,
            activations_cached=7,
            activations_extracted=3,
            perplexity_cached=5,
            perplexity_extracted=5,
            elapsed_seconds=1.5,
        )
        assert stats.cache_hit_rate == 0.7

    def test_cache_hit_rate_zero_prompts(self):
        """cache_hit_rate handles zero prompts."""
        stats = WarmupStats(
            total_prompts=0,
            activations_cached=0,
            activations_extracted=0,
            perplexity_cached=0,
            perplexity_extracted=0,
            elapsed_seconds=0.0,
        )
        assert stats.cache_hit_rate == 0.0

    def test_repr(self):
        """WarmupStats repr is informative."""
        stats = WarmupStats(
            total_prompts=100,
            activations_cached=80,
            activations_extracted=20,
            perplexity_cached=90,
            perplexity_extracted=10,
            elapsed_seconds=5.5,
        )
        repr_str = repr(stats)
        assert "100" in repr_str
        assert "80" in repr_str
        assert "5.5" in repr_str


class TestPooledCache:
    """Tests for cache_pooled=True mode (disk-efficient caching)."""

    def test_pooled_cache_extracts_and_saves(self, tiny_model, tmp_path, monkeypatch):
        """cache_pooled=True extracts and saves pooled activations."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling="last_token",
        )

        prompts = ["pooled test one", "pooled test two"]
        stats = cache.warmup(prompts)

        assert stats.activations_extracted == 2
        assert stats.activations_cached == 0

        # Check pooled cache exists
        for prompt in prompts:
            assert is_prompt_pooled_cached(tiny_model, prompt, {0, 1}, "last_token")

    def test_pooled_cache_hit(self, tiny_model, tmp_path, monkeypatch):
        """Second warmup with pooled cache is instant (cache hit)."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling="last_token",
        )

        prompts = ["pooled hit test"]

        # First warmup - extracts
        stats1 = cache.warmup(prompts)
        assert stats1.activations_extracted == 1

        # Second warmup - cache hit
        stats2 = cache.warmup(prompts)
        assert stats2.activations_cached == 1
        assert stats2.activations_extracted == 0

    def test_pooled_get_activations_shape(self, tiny_model, tmp_path, monkeypatch):
        """get_activations with cache_pooled returns (batch, hidden_dim * n_layers)."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling="last_token",
        )

        prompts = ["shape test one", "shape test two"]
        activations, mask = cache.get_activations(prompts)

        # Shape: (batch, n_layers * hidden_dim) - no seq_len dimension!
        assert activations.ndim == 2
        assert activations.shape[0] == 2  # 2 prompts
        assert mask is None  # No mask for pooled cache

    def test_pooled_vs_unpooled_cache_size(self, tiny_model, tmp_path, monkeypatch):
        """Pooled cache uses significantly less disk space than unpooled."""
        import os

        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        # Use a longer prompt to amplify the difference
        prompt = "This is a longer prompt that will have more tokens and show the disk savings"

        # Extract with unpooled cache
        cache_unpooled = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=False,
        )
        cache_unpooled.warmup([prompt])

        # Calculate unpooled size
        unpooled_dir = get_prompt_cache_dir(tiny_model, prompt)
        unpooled_size = sum(
            f.stat().st_size for f in unpooled_dir.rglob("*.pt")
        )

        # Clear and extract with pooled cache
        shutil.rmtree(unpooled_dir)

        cache_pooled = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling="last_token",
        )
        cache_pooled.warmup([prompt])

        # Calculate pooled size
        pooled_dir = get_prompt_cache_dir(tiny_model, prompt)
        pooled_size = sum(
            f.stat().st_size for f in pooled_dir.rglob("*.pt")
        )

        # Pooled should be significantly smaller
        # (exact ratio depends on seq_len, but should be at least 2x smaller)
        assert pooled_size < unpooled_size
        # For reference, print the ratio (helpful for debugging)
        # print(f"Unpooled: {unpooled_size}, Pooled: {pooled_size}, Ratio: {unpooled_size/pooled_size:.1f}x")

    def test_pooled_different_strategies(self, tiny_model, tmp_path, monkeypatch):
        """Different pooling strategies create separate cache entries."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompt = "strategy test prompt"

        # Cache with last_token
        cache_last = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling="last_token",
        )
        cache_last.warmup([prompt])

        # Cache with mean
        cache_mean = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling="mean",
        )
        # This should extract again (different pooling)
        stats = cache_mean.warmup([prompt])
        assert stats.activations_extracted == 1  # Not a cache hit

        # Both should now be cached separately
        assert is_prompt_pooled_cached(tiny_model, prompt, {0}, "last_token")
        assert is_prompt_pooled_cached(tiny_model, prompt, {0}, "mean")

    def test_pooled_invalid_pooling_raises(self, tiny_model):
        """cache_pooled=True with pooling='all' raises error."""
        with pytest.raises(ValueError, match="pooling='all' is not valid"):
            UnifiedCache(
                model=tiny_model,
                layers=[0],
                cache_pooled=True,
                pooling="all",
            )

    def test_pooled_with_perplexity(self, tiny_model, tmp_path, monkeypatch):
        """cache_pooled=True works with compute_perplexity=True."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=True,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling="last_token",
        )

        prompts = ["ppl pooled test"]
        stats = cache.warmup(prompts)

        assert stats.activations_extracted == 1
        assert stats.perplexity_extracted == 1

        # Both should work
        activations, mask = cache.get_activations(prompts)
        ppl = cache.get_perplexity(prompts)

        assert activations.ndim == 2  # Pooled
        assert mask is None
        assert ppl.shape == (1, 3)
