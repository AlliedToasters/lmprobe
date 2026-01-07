"""Tests for per-prompt, per-layer activation caching."""

import shutil

import pytest
import torch

from lmprobe.cache import (
    clear_cache,
    get_cached_layers,
    get_extraction_cache_dir,
    get_prompt_cache_dir,
    get_prompt_cached_layers,
    invalidate_extraction_cache,
    is_prompt_fully_cached,
    load_attention_mask,
    load_layer,
    save_attention_mask,
    save_layer,
)


class TestCacheStorage:
    """Tests for low-level cache storage functions."""

    def test_save_and_load_layer(self, tmp_path):
        """Can save and load a single layer."""
        activations = torch.randn(2, 10, 64)  # (batch, seq, hidden)
        save_layer(tmp_path, 8, activations)
        loaded = load_layer(tmp_path, 8)
        assert torch.allclose(activations, loaded)

    def test_save_and_load_attention_mask(self, tmp_path):
        """Can save and load attention mask."""
        mask = torch.ones(2, 10, dtype=torch.long)
        save_attention_mask(tmp_path, mask)
        loaded = load_attention_mask(tmp_path)
        assert torch.equal(mask, loaded)

    def test_get_cached_layers_empty(self, tmp_path):
        """Empty directory returns empty set."""
        assert get_cached_layers(tmp_path) == set()

    def test_get_cached_layers_nonexistent(self, tmp_path):
        """Nonexistent directory returns empty set."""
        assert get_cached_layers(tmp_path / "nonexistent") == set()

    def test_get_cached_layers_finds_layers(self, tmp_path):
        """Correctly identifies cached layers."""
        save_layer(tmp_path, 8, torch.randn(2, 10, 64))
        save_layer(tmp_path, 16, torch.randn(2, 10, 64))
        save_layer(tmp_path, 24, torch.randn(2, 10, 64))

        cached = get_cached_layers(tmp_path)
        assert cached == {8, 16, 24}

    def test_cache_dir_structure(self):
        """Cache directory uses model and prompt hashes."""
        cache_dir = get_extraction_cache_dir(
            "test-model",
            ["prompt1", "prompt2"],
        )
        # Should be ~/.cache/lmprobe/{model_hash}/{prompts_hash}
        assert len(cache_dir.parts) >= 4
        # Last two parts should be hex hashes
        assert all(c in "0123456789abcdef" for c in cache_dir.parts[-1])
        assert all(c in "0123456789abcdef" for c in cache_dir.parts[-2])

    def test_invalidate_cache(self, tmp_path):
        """Invalidation removes all cached data."""
        save_layer(tmp_path, 8, torch.randn(2, 10, 64))
        save_layer(tmp_path, 16, torch.randn(2, 10, 64))
        save_attention_mask(tmp_path, torch.ones(2, 10))

        invalidate_extraction_cache(tmp_path)

        assert not tmp_path.exists()


class TestCachedExtractor:
    """Tests for CachedExtractor with per-prompt, per-layer caching."""

    def test_caches_prompts_individually(self, tiny_model, tmp_path, monkeypatch):
        """First extraction caches each prompt with its layers."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import CachedExtractor
        from lmprobe.extraction import ActivationExtractor

        # tiny model has only 2 layers (0 and 1)
        extractor = ActivationExtractor(
            tiny_model,
            device="cpu",
            layers=[0, 1],
            batch_size=4,
        )
        cached = CachedExtractor(extractor)

        prompts = ["hello world"]
        cached.extract(prompts, remote=False)

        # Check per-prompt cache directory structure
        cache_dir = get_prompt_cache_dir(tiny_model, prompts[0])
        cached_layers = get_prompt_cached_layers(cache_dir)
        assert cached_layers == {0, 1}

        # Check attention mask exists
        assert (cache_dir / "attention_mask.pt").exists()

    def test_partial_layer_cache_hit(self, tiny_model, tmp_path, monkeypatch):
        """Reuses cached layers for prompts and only extracts missing layers."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import CachedExtractor
        from lmprobe.extraction import ActivationExtractor

        prompt = "test prompt"
        prompts = [prompt]

        # tiny model has only 2 layers (0 and 1)
        # First extraction: layer [0] only
        extractor1 = ActivationExtractor(
            tiny_model,
            device="cpu",
            layers=[0],
            batch_size=4,
        )
        cached1 = CachedExtractor(extractor1)
        acts1, mask1 = cached1.extract(prompts, remote=False)

        # Verify layer 0 is cached for this prompt
        cache_dir = get_prompt_cache_dir(tiny_model, prompt)
        assert get_prompt_cached_layers(cache_dir) == {0}

        # Second extraction: layers [0, 1] - prompt needs layer 1 extracted
        # (layer 0 is cached but [0,1] requires both, so prompt is not fully cached)
        extractor2 = ActivationExtractor(
            tiny_model,
            device="cpu",
            layers=[0, 1],
            batch_size=4,
        )
        cached2 = CachedExtractor(extractor2)
        acts2, mask2 = cached2.extract(prompts, remote=False)

        # Now both layers should be cached for this prompt
        assert get_prompt_cached_layers(cache_dir) == {0, 1}

        # Verify shapes are correct
        # acts2 should have 2 layers worth of hidden dims (double acts1)
        assert acts2.shape[-1] == acts1.shape[-1] * 2

    def test_full_cache_hit(self, tiny_model, tmp_path, monkeypatch):
        """Full cache hit loads all prompts without extraction."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import CachedExtractor
        from lmprobe.extraction import ActivationExtractor

        prompts = ["cached prompt"]

        # First extraction
        extractor = ActivationExtractor(
            tiny_model,
            device="cpu",
            layers=[0, 1],
            batch_size=4,
        )
        cached = CachedExtractor(extractor)
        acts1, mask1 = cached.extract(prompts, remote=False)

        # Second extraction with same layers - should be full cache hit
        acts2, mask2 = cached.extract(prompts, remote=False)

        # Results should be identical
        assert torch.allclose(acts1, acts2)
        assert torch.equal(mask1, mask2)

    def test_invalidate_cache_forces_reextraction(self, tiny_model, tmp_path, monkeypatch):
        """invalidate_cache=True forces re-extraction."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import CachedExtractor
        from lmprobe.extraction import ActivationExtractor

        prompt = "invalidation test"
        prompts = [prompt]

        extractor = ActivationExtractor(
            tiny_model,
            device="cpu",
            layers=[0],
            batch_size=4,
        )
        cached = CachedExtractor(extractor)

        # First extraction
        acts1, _ = cached.extract(prompts, remote=False)

        # Get cache dir before invalidation (per-prompt)
        cache_dir = get_prompt_cache_dir(tiny_model, prompt)
        assert cache_dir.exists()

        # Extract with invalidation
        acts2, _ = cached.extract(prompts, remote=False, invalidate_cache=True)

        # Cache should exist again
        assert cache_dir.exists()
        assert get_prompt_cached_layers(cache_dir) == {0}

    def test_different_prompts_different_cache(self, tiny_model, tmp_path, monkeypatch):
        """Different prompts use different cache directories."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import CachedExtractor
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            tiny_model,
            device="cpu",
            layers=[0],
            batch_size=4,
        )
        cached = CachedExtractor(extractor)

        prompts1 = ["first prompt"]
        prompts2 = ["second prompt"]

        cached.extract(prompts1, remote=False)
        cached.extract(prompts2, remote=False)

        # Should have two different per-prompt cache directories
        cache_dir1 = get_prompt_cache_dir(tiny_model, prompts1[0])
        cache_dir2 = get_prompt_cache_dir(tiny_model, prompts2[0])

        assert cache_dir1 != cache_dir2
        assert cache_dir1.exists()
        assert cache_dir2.exists()

    def test_cross_request_cache_hit(self, tiny_model, tmp_path, monkeypatch):
        """Prompts cached in one request are reused in another."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import CachedExtractor
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            tiny_model,
            device="cpu",
            layers=[0, 1],
            batch_size=4,
        )
        cached = CachedExtractor(extractor)

        # First request: prompts A, B
        prompts1 = ["prompt A", "prompt B"]
        cached.extract(prompts1, remote=False)

        # Second request: prompts B, C - B should be cached!
        prompts2 = ["prompt B", "prompt C"]
        cached.extract(prompts2, remote=False)

        # All three prompts should be cached
        assert is_prompt_fully_cached(tiny_model, "prompt A", {0, 1})
        assert is_prompt_fully_cached(tiny_model, "prompt B", {0, 1})
        assert is_prompt_fully_cached(tiny_model, "prompt C", {0, 1})


class TestLinearProbeWithCache:
    """Integration tests for LinearProbe with per-prompt caching."""

    def test_iterative_layer_experimentation(self, tiny_model, tmp_path, monkeypatch):
        """Simulates iterative experimentation with different layers."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import LinearProbe

        positive = ["good example"]
        negative = ["bad example"]

        # tiny model has only 2 layers (0 and 1)
        # Experiment 1: layer 0
        probe1 = LinearProbe(
            model=tiny_model,
            layers=0,
            device="cpu",
            remote=False,
        )
        probe1.fit(positive, negative)

        # Cache should contain layer 0 for each prompt
        for prompt in positive + negative:
            cache_dir = get_prompt_cache_dir(tiny_model, prompt)
            assert get_prompt_cached_layers(cache_dir) == {0}

        # Experiment 2: layers [0, 1] - prompts need layer 1 extracted
        probe2 = LinearProbe(
            model=tiny_model,
            layers=[0, 1],
            device="cpu",
            remote=False,
        )
        probe2.fit(positive, negative)

        # All probes should be fitted
        assert probe1.classifier_ is not None
        assert probe2.classifier_ is not None

        # Cache should now contain both layers for each prompt
        for prompt in positive + negative:
            cache_dir = get_prompt_cache_dir(tiny_model, prompt)
            assert get_prompt_cached_layers(cache_dir) == {0, 1}


class TestUnifiedCacheLinearProbeIntegration:
    """Tests for cache compatibility between UnifiedCache and LinearProbe.

    These tests expose a known issue: UnifiedCache with cache_pooled=True
    stores activations in a different format than what LinearProbe expects,
    causing cache misses despite the data being available.
    """

    def test_unified_cache_warmup_enables_linear_probe_cache_hit(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """UnifiedCache warmup should enable cache hits for LinearProbe.

        This test verifies that when UnifiedCache warms up the cache,
        subsequent LinearProbe operations can reuse the cached activations
        instead of making new extraction calls.

        EXPECTED TO FAIL: Currently LinearProbe and UnifiedCache use
        incompatible cache formats.
        """
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import LinearProbe, UnifiedCache
        from lmprobe.cache import is_prompt_fully_cached

        prompts = ["This is a test prompt", "Another test"]
        layers = [0, 1]  # tiny model has 2 layers

        # Phase 1: Warm cache with UnifiedCache (NOT using cache_pooled)
        cache = UnifiedCache(
            model=tiny_model,
            layers=layers,
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=False,  # Store full-sequence activations
        )
        stats = cache.warmup(prompts)

        # Verify warmup extracted activations
        assert stats.activations_extracted == len(prompts)

        # Verify cache is populated
        for prompt in prompts:
            assert is_prompt_fully_cached(tiny_model, prompt, set(layers)), \
                f"Prompt '{prompt}' should be fully cached after warmup"

        # Phase 2: Use LinearProbe - should get cache hits
        probe = LinearProbe(
            model=tiny_model,
            layers=layers,
            device="cpu",
            remote=False,
        )

        # Fit should use cached activations (no new extractions)
        positive = [prompts[0]]
        negative = [prompts[1]]
        probe.fit(positive, negative)

        # The cache should still show the same data (no additional extractions)
        # This is the key assertion - if LinearProbe made new extraction calls,
        # it would show up as additional layer files or modified timestamps
        for prompt in prompts:
            cache_dir = get_prompt_cache_dir(tiny_model, prompt)
            cached = get_prompt_cached_layers(cache_dir)
            assert cached == set(layers), \
                f"Cache for '{prompt}' should still have exactly layers {layers}"

    def test_pooled_cache_warmup_enables_linear_probe_cache_hit(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """UnifiedCache with cache_pooled=True should enable LinearProbe cache hits.

        When UnifiedCache uses cache_pooled=True for disk efficiency, it stores
        pre-pooled activations. LinearProbe detects and uses these directly,
        avoiding redundant extraction and pooling.
        """
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import LinearProbe, UnifiedCache
        from lmprobe.cache import is_prompt_pooled_cached

        prompts = ["This is a test prompt", "Another test"]
        layers = [0, 1]
        pooling = "last_token"

        # Phase 1: Warm cache with UnifiedCache using pooled storage
        cache = UnifiedCache(
            model=tiny_model,
            layers=layers,
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling=pooling,
        )
        stats = cache.warmup(prompts)

        # Verify warmup extracted activations
        assert stats.activations_extracted == len(prompts)

        # Verify pooled cache is populated
        for prompt in prompts:
            assert is_prompt_pooled_cached(tiny_model, prompt, set(layers), pooling), \
                f"Prompt '{prompt}' should have pooled cache after warmup"

        # Phase 2: Use LinearProbe with matching pooling strategy
        probe = LinearProbe(
            model=tiny_model,
            layers=layers,
            pooling=pooling,  # Same pooling as UnifiedCache
            device="cpu",
            remote=False,
        )

        # Count cache entries before fit
        import os
        cache_files_before = sum(
            len(files) for _, _, files in os.walk(tmp_path)
        )

        # Fit should use cached pooled activations (no new extractions)
        positive = [prompts[0]]
        negative = [prompts[1]]
        probe.fit(positive, negative)

        # Count cache entries after fit
        cache_files_after = sum(
            len(files) for _, _, files in os.walk(tmp_path)
        )

        # If LinearProbe properly uses the pooled cache, it should NOT
        # create new cache files. If it ignores the pooled cache and
        # extracts fresh, it will create new layer_X.pt files.
        #
        # This assertion will FAIL with the current implementation because
        # LinearProbe creates its own layer_X.pt files instead of using
        # the pooled cache.
        assert cache_files_after == cache_files_before, (
            f"LinearProbe should not create new cache files when pooled cache exists. "
            f"Before: {cache_files_before}, After: {cache_files_after}"
        )

    def test_linear_probe_respects_pooled_cache_format(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """LinearProbe should be able to load pre-pooled activations.

        When activations are already pooled and cached, LinearProbe should
        skip both extraction AND pooling, loading the final result directly.

        EXPECTED TO FAIL: LinearProbe always extracts full-sequence
        activations and pools them, ignoring pre-pooled cache.
        """
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import LinearProbe, UnifiedCache

        prompts = ["Test prompt one", "Test prompt two"]
        layers = [0, 1]
        pooling = "last_token"

        # Warm pooled cache
        cache = UnifiedCache(
            model=tiny_model,
            layers=layers,
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling=pooling,
        )
        cache.warmup(prompts)

        # Get activations directly from UnifiedCache (the correct way)
        acts_from_cache, _ = cache.get_activations(prompts)

        # Now use LinearProbe
        probe = LinearProbe(
            model=tiny_model,
            layers=layers,
            pooling=pooling,
            device="cpu",
            remote=False,
        )

        # Access internal extraction to compare results
        # Note: This uses the internal API, which may change
        positive = [prompts[0]]
        negative = [prompts[1]]
        probe.fit(positive, negative)

        # Get activations via probe's internal method
        import torch
        from lmprobe.pooling import get_pooling_fn

        pooling_fn = get_pooling_fn(pooling)
        acts_raw, mask = probe._cached_extractor.extract(prompts, remote=False)
        acts_from_probe = pooling_fn(acts_raw, mask)

        # The activations should match, proving the same data could be reused
        assert torch.allclose(acts_from_cache, acts_from_probe, atol=1e-5), (
            "Activations from UnifiedCache should match LinearProbe extractions. "
            "If they match, LinearProbe could use the pooled cache directly."
        )


class TestUnifiedCacheActivationBaselineIntegration:
    """Tests for cache compatibility between UnifiedCache and ActivationBaseline.

    ActivationBaseline uses model activations for baselines like random_direction,
    pca, and layer_0. These should reuse pooled cache from UnifiedCache.
    """

    def test_pooled_cache_warmup_enables_activation_baseline_cache_hit(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """UnifiedCache warmup should enable cache hits for ActivationBaseline.

        When UnifiedCache warms the cache with cache_pooled=True, subsequent
        ActivationBaseline operations should reuse that cache.
        """
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import ActivationBaseline, UnifiedCache
        from lmprobe.cache import is_prompt_pooled_cached

        prompts = ["This is a test prompt", "Another test"]
        layers = [0, 1]
        pooling = "last_token"

        # Phase 1: Warm cache with UnifiedCache
        cache = UnifiedCache(
            model=tiny_model,
            layers=layers,
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling=pooling,
        )
        stats = cache.warmup(prompts)
        assert stats.activations_extracted == len(prompts)

        # Verify pooled cache is populated
        for prompt in prompts:
            assert is_prompt_pooled_cached(tiny_model, prompt, set(layers), pooling)

        # Phase 2: Use ActivationBaseline - should get cache hits
        import os
        cache_files_before = sum(len(files) for _, _, files in os.walk(tmp_path))

        baseline = ActivationBaseline(
            model=tiny_model,
            method="random_direction",
            layers=layers,
            pooling=pooling,
            device="cpu",
            remote=False,
            random_state=42,
        )
        baseline.fit([prompts[0]], [prompts[1]])

        cache_files_after = sum(len(files) for _, _, files in os.walk(tmp_path))

        # Should not create new cache files
        assert cache_files_after == cache_files_before, (
            f"ActivationBaseline should use pooled cache. "
            f"Before: {cache_files_before}, After: {cache_files_after}"
        )

    def test_pooled_cache_works_for_pca_baseline(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """PCA baseline should use pooled cache from UnifiedCache."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import ActivationBaseline, UnifiedCache

        prompts = ["Test prompt one", "Test prompt two", "Test prompt three"]
        layers = [0, 1]
        pooling = "last_token"

        # Warm cache
        cache = UnifiedCache(
            model=tiny_model,
            layers=layers,
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling=pooling,
        )
        cache.warmup(prompts)

        import os
        cache_files_before = sum(len(files) for _, _, files in os.walk(tmp_path))

        # PCA baseline
        baseline = ActivationBaseline(
            model=tiny_model,
            method="pca",
            layers=layers,
            pooling=pooling,
            n_components=2,
            device="cpu",
            remote=False,
        )
        baseline.fit([prompts[0]], [prompts[1]])
        baseline.predict([prompts[2]])

        cache_files_after = sum(len(files) for _, _, files in os.walk(tmp_path))

        assert cache_files_after == cache_files_before, (
            f"PCA baseline should use pooled cache. "
            f"Before: {cache_files_before}, After: {cache_files_after}"
        )

    def test_layer_0_baseline_uses_correct_cache(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Layer 0 baseline should use layer 0 cache from UnifiedCache."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import ActivationBaseline, UnifiedCache
        from lmprobe.cache import is_prompt_pooled_cached

        prompts = ["Test prompt A", "Test prompt B"]
        pooling = "last_token"

        # Warm cache for layer 0 specifically
        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],  # layer_0 baseline uses layer 0
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=True,
            pooling=pooling,
        )
        cache.warmup(prompts)

        # Verify layer 0 is cached
        for prompt in prompts:
            assert is_prompt_pooled_cached(tiny_model, prompt, {0}, pooling)

        import os
        cache_files_before = sum(len(files) for _, _, files in os.walk(tmp_path))

        # Layer 0 baseline
        baseline = ActivationBaseline(
            model=tiny_model,
            method="layer_0",
            pooling=pooling,
            device="cpu",
            remote=False,
        )
        baseline.fit([prompts[0]], [prompts[1]])

        cache_files_after = sum(len(files) for _, _, files in os.walk(tmp_path))

        assert cache_files_after == cache_files_before, (
            f"Layer 0 baseline should use pooled cache. "
            f"Before: {cache_files_before}, After: {cache_files_after}"
        )


class TestUnifiedCachePerplexityBaselineIntegration:
    """Tests for cache compatibility between UnifiedCache and perplexity baseline.

    BaselineProbe with method="perplexity" computes perplexity features.
    UnifiedCache can pre-compute these in its warmup phase.
    """

    def test_perplexity_cache_warmup_enables_baseline_cache_hit(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """UnifiedCache warmup should enable cache hits for perplexity baseline.

        When UnifiedCache warms the cache with compute_perplexity=True,
        subsequent BaselineProbe(method="perplexity") should use that cache.
        """
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import BaselineProbe, UnifiedCache
        from lmprobe.cache import is_prompt_perplexity_cached

        prompts = ["This is a test prompt", "Another test"]

        # Phase 1: Warm cache with UnifiedCache including perplexity
        cache = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=True,  # Key: compute perplexity
            device="cpu",
            remote=False,
            cache_pooled=True,
        )
        stats = cache.warmup(prompts)

        # Verify perplexity was extracted
        assert stats.perplexity_extracted == len(prompts)

        # Verify perplexity cache is populated
        for prompt in prompts:
            assert is_prompt_perplexity_cached(tiny_model, prompt), \
                f"Prompt '{prompt}' should have perplexity cached"

        # Phase 2: Use perplexity baseline - should get 100% cache hits
        import os
        cache_files_before = sum(len(files) for _, _, files in os.walk(tmp_path))

        baseline = BaselineProbe(
            method="perplexity",
            model=tiny_model,
            device="cpu",
            remote=False,
        )
        baseline.fit([prompts[0]], [prompts[1]])

        cache_files_after = sum(len(files) for _, _, files in os.walk(tmp_path))

        # Should not create new cache files (all perplexity was pre-computed)
        assert cache_files_after == cache_files_before, (
            f"Perplexity baseline should use cached perplexity. "
            f"Before: {cache_files_before}, After: {cache_files_after}"
        )

    def test_perplexity_cache_shared_between_unified_and_baseline(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Perplexity computed by baseline should be loadable by UnifiedCache."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import BaselineProbe, UnifiedCache
        from lmprobe.cache import is_prompt_perplexity_cached

        prompts = ["Forward direction test", "Backward direction test"]

        # First: Compute perplexity via BaselineProbe
        baseline = BaselineProbe(
            method="perplexity",
            model=tiny_model,
            device="cpu",
            remote=False,
        )
        baseline.fit([prompts[0]], [prompts[1]])

        # Verify perplexity is cached
        for prompt in prompts:
            assert is_prompt_perplexity_cached(tiny_model, prompt)

        # Second: UnifiedCache should see these as cached
        cache = UnifiedCache(
            model=tiny_model,
            layers=[0, 1],
            compute_perplexity=True,
            device="cpu",
            remote=False,
        )
        stats = cache.warmup(prompts)

        # Should be 100% cache hits for perplexity
        assert stats.perplexity_cached == len(prompts), (
            f"UnifiedCache should detect perplexity cache from baseline. "
            f"Expected {len(prompts)} cached, got {stats.perplexity_cached}"
        )
        assert stats.perplexity_extracted == 0

    def test_unified_cache_perplexity_matches_baseline(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Perplexity values from UnifiedCache should match BaselineProbe."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        import numpy as np
        from lmprobe import BaselineProbe, UnifiedCache

        prompts = ["Matching test prompt"]

        # Compute via UnifiedCache
        cache = UnifiedCache(
            model=tiny_model,
            layers=[0],
            compute_perplexity=True,
            device="cpu",
            remote=False,
        )
        cache.warmup(prompts)
        ppl_from_cache = cache.get_perplexity(prompts)

        # Now BaselineProbe should get the same values (from cache)
        baseline = BaselineProbe(
            method="perplexity",
            model=tiny_model,
            device="cpu",
            remote=False,
        )
        # Access internal method to get raw features
        ppl_from_baseline = baseline._compute_perplexity(prompts)

        assert np.allclose(ppl_from_cache, ppl_from_baseline, atol=1e-5), (
            "Perplexity from UnifiedCache should match BaselineProbe"
        )
