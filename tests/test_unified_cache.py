"""Tests for UnifiedCache - combined activation and perplexity extraction."""

import warnings

import numpy as np
import pytest
import torch

from lmprobe.cache import (
    get_prompt_cache_dir,
    is_prompt_logits_cached,
    is_prompt_perplexity_cached,
    is_prompt_pooled_cached,
)
from lmprobe.unified_cache import CachedLogits, UnifiedCache, WarmupStats

pytestmark = pytest.mark.nnsight


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

        # Check both are cached (default is now cache_pooled=True)
        for prompt in prompts:
            assert is_prompt_pooled_cached(tiny_model, prompt, {0, 1}, "last_token")
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

    def test_get_activations_shapes_unpooled(self, tiny_model, tmp_path, monkeypatch):
        """get_activations with cache_pooled=False returns 3D tensor."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=False,
        )

        prompts = ["short", "a longer test prompt"]
        activations, mask = cache.get_activations(prompts)

        # Shape: (batch, seq_len, n_layers * hidden_dim)
        assert activations.ndim == 3
        assert activations.shape[0] == 2  # 2 prompts
        assert mask.shape[0] == 2
        assert mask.shape[1] == activations.shape[1]  # seq_len matches

    def test_get_activations_returns_correct_shapes_pooled(self, tiny_model, tmp_path, monkeypatch):
        """get_activations with default cache_pooled=True returns 2D tensor."""
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

        # Shape: (batch, n_layers * hidden_dim) - pooled, no seq_len
        assert activations.ndim == 2
        assert activations.shape[0] == 2
        assert mask is None

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

        # Verify activations cached (pooled, since default) but not perplexity
        assert is_prompt_pooled_cached(tiny_model, prompt, {0, 1}, "last_token")
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

        # Default is pooled caching now
        assert is_prompt_pooled_cached(tiny_model, prompts[0], {0, 1}, "last_token")

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


    def test_dtype_parameter_passed_to_backend(self, tiny_model, tmp_path, monkeypatch):
        """dtype parameter is stored and passed through to resolve_backend."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            dtype="float32",
        )

        assert cache.dtype == "float32"

        # Trigger backend creation and verify it works
        prompts = ["dtype test prompt"]
        stats = cache.warmup(prompts)
        assert stats.activations_extracted == 1

    def test_dtype_none_default(self, tiny_model):
        """dtype defaults to None when not specified."""
        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            device="cpu",
        )
        assert cache.dtype is None

    def test_dtype_invalid_raises(self, tiny_model):
        """Invalid dtype string raises ValueError on backend creation."""
        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            device="cpu",
            dtype="invalid_dtype",
        )
        import pytest
        with pytest.raises(ValueError, match="Unknown dtype"):
            # Access _resolved_backend to trigger dtype resolution
            _ = cache._resolved_backend


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

        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        # Use a longer prompt to amplify the difference
        prompt = "This is a longer prompt that will have more tokens and show the disk savings"

        from lmprobe.cache import get_prompt_cache_path, invalidate_extraction_cache

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

        # Calculate unpooled size (safetensors file)
        sf_path = get_prompt_cache_path(tiny_model, prompt)
        unpooled_size = sf_path.stat().st_size

        # Clear and extract with pooled cache
        invalidate_extraction_cache(get_prompt_cache_dir(tiny_model, prompt))

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
        sf_path_pooled = get_prompt_cache_path(tiny_model, prompt)
        pooled_size = sf_path_pooled.stat().st_size

        # Pooled should be significantly smaller
        assert pooled_size < unpooled_size

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

    def test_partial_cache_warns_missing_layers(self, tiny_model, tmp_path, monkeypatch):
        """Warns when cache exists for prompts but requested layers are missing."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompt = "layer sweep test prompt"

        # First: cache layer 0 only
        cache_layer0 = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling="last_token",
        )
        cache_layer0.warmup([prompt])
        assert is_prompt_pooled_cached(tiny_model, prompt, {0}, "last_token")

        # Second: request layer 1 — cache file exists but layer 1 is missing
        cache_layer1 = UnifiedCache(
            model=tiny_model,
            layers=[1],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling="last_token",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cache_layer1.warmup([prompt])

            # Should have emitted exactly one warning about missing layers
            layer_warnings = [
                x for x in w if "missing layer" in str(x.message).lower()
            ]
            assert len(layer_warnings) == 1
            msg = str(layer_warnings[0].message)
            assert "1" in msg  # missing layer 1
            assert "warmup" in msg.lower()  # suggests warmup()

    def test_no_warning_when_fully_cached(self, tiny_model, tmp_path, monkeypatch):
        """No warning when all requested layers are cached."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompt = "fully cached test"

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling="last_token",
        )
        cache.warmup([prompt])

        # Request same layers again — should NOT warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cache.warmup([prompt])

            layer_warnings = [
                x for x in w if "missing layer" in str(x.message).lower()
            ]
            assert len(layer_warnings) == 0

    def test_no_warning_when_no_cache_exists(self, tiny_model, tmp_path, monkeypatch):
        """No warning when there's no cache file at all (first extraction)."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling="last_token",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cache.warmup(["brand new prompt"])

            layer_warnings = [
                x for x in w if "missing layer" in str(x.message).lower()
            ]
            assert len(layer_warnings) == 0

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


class TestCachedLogits:
    """Tests for cache_logits feature."""

    def test_cache_logits_full(self, tiny_model, tmp_path, monkeypatch):
        """Full logits cached and retrieved with correct shape."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_logits=True,
            logit_positions="all",
        )

        prompts = ["hello world", "test prompt"]
        stats = cache.warmup(prompts)

        assert stats.logits_extracted == 2
        assert stats.logits_cached == 0

        result = cache.get_logits(prompts)
        assert isinstance(result, CachedLogits)
        assert result.values.ndim == 3  # (batch, seq_len, vocab_size)
        assert result.values.shape[0] == 2
        assert result.indices is None
        assert result.top_k is None
        assert result.positions == "all"

    def test_cache_logits_top_k(self, tiny_model, tmp_path, monkeypatch):
        """Top-K values + indices cached with correct shape."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        K = 10
        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_logits=True,
            logit_top_k=K,
            logit_positions="last",
        )

        prompts = ["top k test"]
        cache.warmup(prompts)

        result = cache.get_logits(prompts)
        assert result.values.shape == (1, 1, K)  # (batch, 1 position, K)
        assert result.indices is not None
        assert result.indices.shape == (1, 1, K)
        assert result.indices.dtype == np.int32 or str(result.indices.dtype) == "torch.int32"
        assert result.top_k == K

    def test_cache_logits_last_position(self, tiny_model, tmp_path, monkeypatch):
        """positions='last' stores only 1 position per prompt."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_logits=True,
            logit_positions="last",
        )

        prompts = ["a longer prompt with several tokens for testing"]
        cache.warmup(prompts)

        result = cache.get_logits(prompts)
        # Should have only 1 position (last token)
        assert result.values.shape[1] == 1
        assert result.positions == "last"

    def test_cache_logits_idempotent(self, tiny_model, tmp_path, monkeypatch):
        """Second warmup is all cache hits for logits."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_logits=True,
            logit_positions="last",
        )

        prompts = ["idempotent logit test"]

        stats1 = cache.warmup(prompts)
        assert stats1.logits_extracted == 1
        assert stats1.logits_cached == 0

        stats2 = cache.warmup(prompts)
        assert stats2.logits_cached == 1
        assert stats2.logits_extracted == 0

    def test_cache_logits_default_off(self, tiny_model, tmp_path, monkeypatch):
        """Default cache_logits=False; get_logits() raises ValueError."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=False,
            device="cpu",
            remote=False,
        )

        with pytest.raises(ValueError, match="cache_logits=False"):
            cache.get_logits(["test"])

    def test_cache_logits_with_perplexity(self, tiny_model, tmp_path, monkeypatch):
        """Both logits and perplexity work together from same forward pass."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=True,
            device="cpu",
            remote=False,
            cache_logits=True,
            logit_positions="last",
        )

        prompts = ["combined test prompt"]
        stats = cache.warmup(prompts)

        assert stats.activations_extracted == 1
        assert stats.perplexity_extracted == 1
        assert stats.logits_extracted == 1

        # Both should load from cache
        ppl = cache.get_perplexity(prompts)
        logits = cache.get_logits(prompts)

        assert ppl.shape == (1, 3)
        assert logits.values.ndim == 3
        assert logits.values.shape[0] == 1

        # Verify cache is populated
        for prompt in prompts:
            assert is_prompt_perplexity_cached(tiny_model, prompt)
            assert is_prompt_logits_cached(tiny_model, prompt)


class TestServerSideTopK:
    """Tests for server-side top-k logits optimization."""

    def test_effective_topk_only_when_remote_and_no_perplexity(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """effective_top_k is None when compute_perplexity=True, even with logit_top_k set."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        # With perplexity enabled: should use full logits path (local topk)
        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=True,
            device="cpu",
            remote=False,
            cache_logits=True,
            logit_top_k=10,
            logit_positions="last",
        )

        prompts = ["topk perplexity fallback test"]
        stats = cache.warmup(prompts)

        # Should still work — perplexity computed from full logits
        assert stats.perplexity_extracted == 1
        assert stats.logits_extracted == 1

        # Logits should be cached as top-k (applied locally)
        result = cache.get_logits(prompts)
        assert result.values.shape[-1] == 10
        assert result.indices is not None

    def test_local_topk_still_works(self, tiny_model, tmp_path, monkeypatch):
        """Local backend with logit_top_k still applies topk locally."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        K = 5
        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_logits=True,
            logit_top_k=K,
            logit_positions="last",
        )

        prompts = ["local topk test"]
        cache.warmup(prompts)

        result = cache.get_logits(prompts)
        assert result.values.shape == (1, 1, K)
        assert result.indices is not None
        assert result.indices.shape == (1, 1, K)


class TestComputeLogitsFromCache:
    """Tests for computing logits from cached activations without a forward pass."""

    def test_compute_logits_basic(self, tiny_model, tmp_path, monkeypatch):
        """Logits appear in cache after compute_logits_from_cache."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompts = ["hello world", "test prompt"]

        # Warm up with raw activations (need full sequence for layer loading)
        cache = UnifiedCache(
            model=tiny_model, layers="all",
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache.warmup(prompts)

        # Compute logits from cache
        computed = cache.compute_logits_from_cache(prompts)
        assert computed == 2

        # Verify logits are now cached
        for prompt in prompts:
            assert is_prompt_logits_cached(tiny_model, prompt)

    def test_compute_logits_topk(self, tiny_model, tmp_path, monkeypatch):
        """Top-k compression works with compute_logits_from_cache."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompts = ["topk compute test"]
        K = 5

        cache = UnifiedCache(
            model=tiny_model, layers="all",
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache.warmup(prompts)

        computed = cache.compute_logits_from_cache(prompts, top_k=K)
        assert computed == 1

        # Verify top-k logits are cached
        assert is_prompt_logits_cached(tiny_model, prompts[0], top_k=K)

        from lmprobe.cache import load_prompt_logits

        values, indices = load_prompt_logits(tiny_model, prompts[0], top_k=K)
        assert values.shape[-1] == K
        assert indices is not None
        assert indices.shape[-1] == K

    def test_compute_logits_skips_cached(self, tiny_model, tmp_path, monkeypatch):
        """Returns 0 when all prompts already have logits cached."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompts = ["already cached test"]

        cache = UnifiedCache(
            model=tiny_model, layers="all",
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache.warmup(prompts)

        # First call computes
        assert cache.compute_logits_from_cache(prompts) == 1

        # Second call skips
        assert cache.compute_logits_from_cache(prompts) == 0

    def test_compute_logits_requires_last_layer(self, tiny_model, tmp_path, monkeypatch):
        """ValueError if last layer not in cached layers."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompts = ["layer check test"]

        # Only cache layer 0, not the last layer (layer 1 for tiny model)
        cache = UnifiedCache(
            model=tiny_model, layers=[0],
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache.warmup(prompts)

        with pytest.raises(ValueError, match="Last layer"):
            cache.compute_logits_from_cache(prompts)

    def test_compute_logits_matches_forward_pass(self, tiny_model, tmp_path, monkeypatch):
        """Critical correctness test: norm(x) @ W matches actual model logits."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompts = ["correctness test prompt"]

        # Extract with logits via forward pass
        cache_with_logits = UnifiedCache(
            model=tiny_model, layers="all",
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False, cache_logits=True,
            logit_positions="last",
        )
        cache_with_logits.warmup(prompts)

        # Get forward-pass logits
        forward_logits = cache_with_logits.get_logits(prompts)
        forward_values = forward_logits.values  # (1, 1, vocab_size)

        # Use a second tmp dir for the compute-from-cache path
        tmp_path2 = tmp_path / "compute_test"
        tmp_path2.mkdir()
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path2))

        # Reset cache backend to pick up new dir
        import lmprobe.cache as cache_mod

        cache_mod._backend = None

        cache_no_logits = UnifiedCache(
            model=tiny_model, layers="all",
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache_no_logits.warmup(prompts)

        # Compute logits from cache
        cache_no_logits.compute_logits_from_cache(prompts)

        # Load computed logits
        from lmprobe.cache import load_prompt_logits

        computed_values, _ = load_prompt_logits(tiny_model, prompts[0])
        # computed_values: (1, 1, vocab_size)

        # Should match forward pass logits closely
        assert forward_values.shape == computed_values.shape, (
            f"Shape mismatch: forward={forward_values.shape}, "
            f"computed={computed_values.shape}"
        )
        # Cast to same dtype for comparison (weights may be bfloat16)
        fv = forward_values.float()
        cv = computed_values.float()
        assert torch.allclose(fv, cv, atol=1e-2), (
            f"Max diff: {(fv - cv).abs().max().item()}"
        )


class TestComputePerplexityFromCache:
    """Tests for computing perplexity from cached activations."""

    def test_compute_perplexity_from_cache_basic(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Perplexity appears in cache after compute_perplexity_from_cache."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompts = ["hello world", "test prompt"]

        # Warm up with raw activations (need full sequence for perplexity)
        cache = UnifiedCache(
            model=tiny_model, layers="all",
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache.warmup(prompts)

        # Compute perplexity from cache
        computed = cache.compute_perplexity_from_cache(prompts)
        assert computed == 2

        # Verify perplexity is now cached
        for prompt in prompts:
            assert is_prompt_perplexity_cached(tiny_model, prompt)

    def test_compute_perplexity_from_cache_skips_cached(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Returns 0 when all prompts already have perplexity cached."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompts = ["already cached perplexity test"]

        cache = UnifiedCache(
            model=tiny_model, layers="all",
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache.warmup(prompts)

        # First call computes
        assert cache.compute_perplexity_from_cache(prompts) == 1

        # Second call skips
        assert cache.compute_perplexity_from_cache(prompts) == 0

    def test_compute_perplexity_from_cache_requires_last_layer(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """ValueError if last layer not in cached layers."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompts = ["perplexity layer check test"]

        # Only cache layer 0, not the last layer
        cache = UnifiedCache(
            model=tiny_model, layers=[0],
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache.warmup(prompts)

        with pytest.raises(ValueError, match="Last layer"):
            cache.compute_perplexity_from_cache(prompts)

    def test_compute_perplexity_from_cache_matches_forward_pass(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Critical: perplexity from cache path matches forward-pass perplexity."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        prompts = ["perplexity correctness test"]

        # Extract with perplexity via forward pass
        cache_with_ppl = UnifiedCache(
            model=tiny_model, layers="all",
            compute_perplexity=True, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache_with_ppl.warmup(prompts)

        # Get forward-pass perplexity
        forward_ppl = cache_with_ppl.get_perplexity(prompts)  # (1, 3)

        # Use a second tmp dir for the compute-from-cache path
        tmp_path2 = tmp_path / "ppl_compute_test"
        tmp_path2.mkdir()
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path2))

        # Reset cache backend to pick up new dir
        import lmprobe.cache as cache_mod

        cache_mod._backend = None

        cache_no_ppl = UnifiedCache(
            model=tiny_model, layers="all",
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache_no_ppl.warmup(prompts)

        # Compute perplexity from cache
        cache_no_ppl.compute_perplexity_from_cache(prompts)

        # Load computed perplexity
        from lmprobe.cache import load_prompt_perplexity

        computed_ppl = load_prompt_perplexity(tiny_model, prompts[0])
        computed_ppl = computed_ppl.float().numpy().reshape(1, -1)

        # Should match forward pass perplexity closely
        assert forward_ppl.shape == computed_ppl.shape, (
            f"Shape mismatch: forward={forward_ppl.shape}, "
            f"computed={computed_ppl.shape}"
        )
        # Tolerance is generous because the forward-pass path and cache path
        # may differ in dtype handling (bfloat16 vs float32 intermediate).
        # The key check is that values are in the same ballpark.
        assert np.allclose(forward_ppl, computed_ppl, rtol=5e-2), (
            f"Max relative diff too large. "
            f"Forward: {forward_ppl}, Computed: {computed_ppl}, "
            f"Max abs diff: {np.abs(forward_ppl - computed_ppl).max()}"
        )
