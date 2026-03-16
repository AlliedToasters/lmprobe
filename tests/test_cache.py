"""Tests for activation caching (v1 legacy + v2 safetensors)."""

import os

import pytest
import torch

from lmprobe.cache import (
    CacheInfo,
    _hash_string,
    cache_info,
    get_cached_layers,
    get_extraction_cache_dir,
    get_prompt_cache_dir,
    get_prompt_cache_path,
    get_prompt_cached_layers,
    invalidate_extraction_cache,
    is_prompt_fully_cached,
    is_prompt_logits_cached,
    is_prompt_pooled_cached,
    load_attention_mask,
    load_layer,
    load_prompt_activations,
    load_prompt_logits,
    load_prompt_pooled_activations,
    save_attention_mask,
    save_layer,
    save_prompt_activations,
    save_prompt_pooled_activations,
    save_prompt_topk_logits,
    set_cache_dtype,
    set_cache_limit,
)


class TestLegacyCacheStorage:
    """Tests for legacy v1 .pt cache storage functions."""

    def test_save_and_load_layer(self, tmp_path):
        activations = torch.randn(2, 10, 64)
        save_layer(tmp_path, 8, activations)
        loaded = load_layer(tmp_path, 8)
        assert torch.allclose(activations, loaded)

    def test_save_and_load_attention_mask(self, tmp_path):
        mask = torch.ones(2, 10, dtype=torch.long)
        save_attention_mask(tmp_path, mask)
        loaded = load_attention_mask(tmp_path)
        assert torch.equal(mask, loaded)

    def test_get_cached_layers_empty(self, tmp_path):
        assert get_cached_layers(tmp_path) == set()

    def test_get_cached_layers_nonexistent(self, tmp_path):
        assert get_cached_layers(tmp_path / "nonexistent") == set()

    def test_get_cached_layers_finds_layers(self, tmp_path):
        save_layer(tmp_path, 8, torch.randn(2, 10, 64))
        save_layer(tmp_path, 16, torch.randn(2, 10, 64))
        save_layer(tmp_path, 24, torch.randn(2, 10, 64))
        assert get_cached_layers(tmp_path) == {8, 16, 24}

    def test_cache_dir_structure(self):
        cache_dir = get_extraction_cache_dir("test-model", ["prompt1", "prompt2"])
        assert len(cache_dir.parts) >= 4
        assert all(c in "0123456789abcdef" for c in cache_dir.parts[-1])
        assert all(c in "0123456789abcdef" for c in cache_dir.parts[-2])

    def test_invalidate_cache(self, tmp_path):
        save_layer(tmp_path, 8, torch.randn(2, 10, 64))
        save_attention_mask(tmp_path, torch.ones(2, 10))
        invalidate_extraction_cache(tmp_path)
        assert not tmp_path.exists()


class TestSafetensorsFormat:
    """Tests for v2 safetensors cache format (#47)."""

    def test_save_and_load_activations(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        model = "test-model"
        prompt = "hello world"
        layers = [0, 1]
        acts = torch.randn(1, 5, 128)  # (1, seq_len, 2*hidden_dim)
        mask = torch.ones(1, 5, dtype=torch.long)

        save_prompt_activations(model, prompt, layers, acts, mask)

        # Check safetensors file exists
        sf_path = get_prompt_cache_path(model, prompt)
        assert sf_path.exists()
        assert sf_path.suffix == ".safetensors"

        # Old directory should not exist
        old_dir = get_prompt_cache_dir(model, prompt)
        assert not old_dir.is_dir()

        # Load and verify
        loaded_acts, loaded_mask = load_prompt_activations(model, prompt, layers)
        assert torch.allclose(acts, loaded_acts)
        assert torch.equal(mask, loaded_mask)

    def test_incremental_layer_save(self, tmp_path, monkeypatch):
        """Adding layers to an existing cache file works."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        model = "test-model"
        prompt = "test prompt"

        # Save layer 0
        acts0 = torch.randn(1, 5, 64)
        mask = torch.ones(1, 5, dtype=torch.long)
        save_prompt_activations(model, prompt, [0], acts0, mask)

        # Save layer 1 (should merge into same file)
        acts1 = torch.randn(1, 5, 64)
        save_prompt_activations(model, prompt, [1], acts1, mask)

        # Both layers should be cached
        assert is_prompt_fully_cached(model, prompt, {0, 1})

        # Load individual layers
        loaded_acts0, _ = load_prompt_activations(model, prompt, [0])
        assert torch.allclose(acts0, loaded_acts0)

        loaded_acts1, _ = load_prompt_activations(model, prompt, [1])
        assert torch.allclose(acts1, loaded_acts1)

    def test_get_prompt_cached_layers_safetensors(self, tmp_path, monkeypatch):
        """get_prompt_cached_layers works with safetensors format."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        model = "test-model"
        prompt = "test prompt"

        acts = torch.randn(1, 5, 128)
        mask = torch.ones(1, 5, dtype=torch.long)
        save_prompt_activations(model, prompt, [0, 1], acts, mask)

        cache_dir = get_prompt_cache_dir(model, prompt)
        cached = get_prompt_cached_layers(cache_dir)
        assert cached == {0, 1}

    def test_invalidation_deletes_safetensors(self, tmp_path, monkeypatch):
        """invalidate_extraction_cache removes safetensors files too."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        model = "test-model"
        prompt = "test"

        save_prompt_activations(
            model, prompt, [0],
            torch.randn(1, 5, 64),
            torch.ones(1, 5, dtype=torch.long),
        )

        cache_dir = get_prompt_cache_dir(model, prompt)
        invalidate_extraction_cache(cache_dir)

        assert not get_prompt_cache_path(model, prompt).exists()
        assert not is_prompt_fully_cached(model, prompt, {0})

    def test_model_name_registered(self, tmp_path, monkeypatch):
        """Model name is recorded for cache_info()."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        model = "org/my-model"

        save_prompt_activations(
            model, "test", [0],
            torch.randn(1, 5, 64),
            torch.ones(1, 5, dtype=torch.long),
        )

        model_hash = _hash_string(model)
        name_file = tmp_path / model_hash / "_model_name.txt"
        assert name_file.exists()
        assert name_file.read_text().strip() == model


class TestPooledCacheSafetensors:
    """Tests for pooled cache in safetensors format."""

    def test_save_and_load_pooled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        model = "test-model"
        prompt = "test"
        layers = [0, 1]
        pooling = "last_token"
        pooled = torch.randn(1, 128)

        save_prompt_pooled_activations(model, prompt, layers, pooled, pooling)
        assert is_prompt_pooled_cached(model, prompt, {0, 1}, pooling)

        loaded = load_prompt_pooled_activations(model, prompt, layers, pooling)
        assert torch.allclose(pooled, loaded)

    def test_raw_and_pooled_coexist(self, tmp_path, monkeypatch):
        """Raw and pooled activations coexist in the same file."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        model = "test-model"
        prompt = "test"

        # Save raw
        save_prompt_activations(
            model, prompt, [0],
            torch.randn(1, 5, 64),
            torch.ones(1, 5, dtype=torch.long),
        )
        # Save pooled
        save_prompt_pooled_activations(
            model, prompt, [0], torch.randn(1, 64), "last_token"
        )

        # Both should be cached
        assert is_prompt_fully_cached(model, prompt, {0})
        assert is_prompt_pooled_cached(model, prompt, {0}, "last_token")

        # Only one file on disk
        sf_path = get_prompt_cache_path(model, prompt)
        assert sf_path.exists()


class TestDiskFullError:
    """Tests for disk-full error handling (#44)."""

    def test_clear_error_message_on_disk_full(self, tmp_path, monkeypatch):
        """Disk-full errors are wrapped with a helpful message."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        def fake_save_file(*args, **kwargs):
            raise OSError("No space left on device")

        monkeypatch.setattr("safetensors.torch.save_file", fake_save_file)

        with pytest.raises(OSError, match="Disk full"):
            save_prompt_activations(
                "test", "test", [0],
                torch.randn(1, 3, 64),
                torch.ones(1, 3, dtype=torch.long),
            )


class TestCacheDtype:
    """Tests for float16 cache storage (#46)."""

    def test_float16_storage(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        set_cache_dtype("float16")
        try:
            acts = torch.randn(1, 5, 64, dtype=torch.float32)
            mask = torch.ones(1, 5, dtype=torch.long)
            save_prompt_activations("test", "fp16test", [0], acts, mask)

            loaded_acts, loaded_mask = load_prompt_activations("test", "fp16test", [0])
            # Activations should be stored as float16
            assert loaded_acts.dtype == torch.float16
            # Mask should remain long
            assert loaded_mask.dtype == torch.int64
            # Values should be close (within float16 precision)
            assert torch.allclose(acts.half(), loaded_acts)
        finally:
            set_cache_dtype(None)

    def test_env_var_dtype(self, tmp_path, monkeypatch):
        """set_cache_dtype changes the active dtype and saves respect it."""
        import lmprobe.cache as cache_mod

        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        set_cache_dtype("float16")
        try:
            assert cache_mod._CACHE_DTYPE == torch.float16

            acts = torch.randn(1, 3, 64, dtype=torch.float32)
            mask = torch.ones(1, 3, dtype=torch.long)
            save_prompt_activations("test", "dtype-check", [0], acts, mask)

            loaded_acts, _ = load_prompt_activations("test", "dtype-check", [0])
            assert loaded_acts.dtype == torch.float16
        finally:
            set_cache_dtype(None)


class TestCacheInfo:
    """Tests for cache_info() reporting (#48)."""

    def test_empty_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        info = cache_info()
        assert isinstance(info, CacheInfo)
        assert info.total_size_bytes == 0
        assert len(info.models) == 0

    def test_cache_info_reports_model(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        save_prompt_activations(
            "test-model", "prompt1", [0, 1],
            torch.randn(1, 5, 128),
            torch.ones(1, 5, dtype=torch.long),
        )
        save_prompt_activations(
            "test-model", "prompt2", [0, 1],
            torch.randn(1, 5, 128),
            torch.ones(1, 5, dtype=torch.long),
        )

        info = cache_info()
        assert info.total_size_bytes > 0
        assert len(info.models) == 1
        assert info.models[0].model_name == "test-model"
        assert info.models[0].num_prompts == 2
        assert info.models[0].num_layers == 2

    def test_cache_info_filter_by_model(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        save_prompt_activations(
            "model-a", "p1", [0], torch.randn(1, 3, 64),
            torch.ones(1, 3, dtype=torch.long),
        )
        save_prompt_activations(
            "model-b", "p1", [0], torch.randn(1, 3, 64),
            torch.ones(1, 3, dtype=torch.long),
        )

        info_a = cache_info(model="model-a")
        assert len(info_a.models) == 1
        assert info_a.models[0].model_name == "model-a"

    def test_cache_info_repr(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        info = cache_info()
        text = repr(info)
        assert "Cache directory" in text
        assert "Total size" in text


class TestLRUEviction:
    """Tests for LRU eviction (#49)."""

    def test_eviction_respects_limit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        # Save several prompts
        for i in range(5):
            save_prompt_activations(
                "test-model", f"prompt-{i}", [0],
                torch.randn(1, 10, 64),
                torch.ones(1, 10, dtype=torch.long),
            )

        # Get total size
        info = cache_info()
        total_bytes = info.total_size_bytes
        assert total_bytes > 0

        # Set limit to half the current size
        half = total_bytes / 2
        set_cache_limit(gb=half / (1024**3))

        try:
            # Trigger eviction by saving one more
            save_prompt_activations(
                "test-model", "prompt-trigger", [0],
                torch.randn(1, 10, 64),
                torch.ones(1, 10, dtype=torch.long),
            )

            # Cache should now be near the limit (allow for the trigger prompt)
            info_after = cache_info()
            one_prompt_size = total_bytes / 5  # approximate size of one prompt
            assert info_after.total_size_bytes <= half + one_prompt_size
        finally:
            set_cache_limit(None)

    def test_set_cache_limit_zero_disables(self, tmp_path, monkeypatch):
        """LMPROBE_CACHE_MAX_GB=0 disables caching."""
        import lmprobe.cache as cache_mod

        old_val = cache_mod._CACHE_MAX_BYTES
        try:
            set_cache_limit(gb=0)
            assert cache_mod._CACHE_MAX_BYTES == -1
        finally:
            cache_mod._CACHE_MAX_BYTES = old_val


class TestCachedExtractor:
    """Tests for CachedExtractor with safetensors caching."""

    def test_caches_prompts_individually(self, tiny_model, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import CachedExtractor
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            tiny_model, device="cpu", layers=[0, 1], batch_size=4,
        )
        cached = CachedExtractor(extractor)
        cached.extract(["hello world"], remote=False)

        # Check cache
        cache_dir = get_prompt_cache_dir(tiny_model, "hello world")
        cached_layers = get_prompt_cached_layers(cache_dir)
        assert cached_layers == {0, 1}

    def test_full_cache_hit(self, tiny_model, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import CachedExtractor
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            tiny_model, device="cpu", layers=[0, 1], batch_size=4,
        )
        cached = CachedExtractor(extractor)

        acts1, mask1 = cached.extract(["cached prompt"], remote=False)
        acts2, mask2 = cached.extract(["cached prompt"], remote=False)

        assert torch.allclose(acts1, acts2)
        assert torch.equal(mask1, mask2)

    def test_invalidate_cache_forces_reextraction(self, tiny_model, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import CachedExtractor
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            tiny_model, device="cpu", layers=[0], batch_size=4,
        )
        cached = CachedExtractor(extractor)
        cached.extract(["invalidation test"], remote=False)

        sf_path = get_prompt_cache_path(tiny_model, "invalidation test")
        assert sf_path.exists()

        cached.extract(["invalidation test"], remote=False, invalidate_cache=True)
        assert sf_path.exists()

    def test_cross_request_cache_hit(self, tiny_model, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import CachedExtractor
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            tiny_model, device="cpu", layers=[0, 1], batch_size=4,
        )
        cached = CachedExtractor(extractor)

        cached.extract(["prompt A", "prompt B"], remote=False)
        cached.extract(["prompt B", "prompt C"], remote=False)

        assert is_prompt_fully_cached(tiny_model, "prompt A", {0, 1})
        assert is_prompt_fully_cached(tiny_model, "prompt B", {0, 1})
        assert is_prompt_fully_cached(tiny_model, "prompt C", {0, 1})


class TestLinearProbeWithCache:
    """Integration tests for LinearProbe with caching."""

    def test_iterative_layer_experimentation(self, tiny_model, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import LinearProbe

        positive = ["good example"]
        negative = ["bad example"]

        probe1 = LinearProbe(model=tiny_model, layers=0, device="cpu", remote=False)
        probe1.fit(positive, negative)

        for prompt in positive + negative:
            cache_dir = get_prompt_cache_dir(tiny_model, prompt)
            assert get_prompt_cached_layers(cache_dir) == {0}

        probe2 = LinearProbe(model=tiny_model, layers=[0, 1], device="cpu", remote=False)
        probe2.fit(positive, negative)

        assert probe1.classifier_ is not None
        assert probe2.classifier_ is not None

        for prompt in positive + negative:
            cache_dir = get_prompt_cache_dir(tiny_model, prompt)
            assert get_prompt_cached_layers(cache_dir) == {0, 1}


class TestUnifiedCacheLinearProbeIntegration:
    """Tests for cache compatibility between UnifiedCache and LinearProbe."""

    def test_unified_cache_warmup_enables_linear_probe_cache_hit(
        self, tiny_model, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import LinearProbe, UnifiedCache
        from lmprobe.cache import is_prompt_fully_cached

        prompts = ["This is a test prompt", "Another test"]
        layers = [0, 1]

        cache = UnifiedCache(
            model=tiny_model, layers=layers,
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        stats = cache.warmup(prompts)
        assert stats.activations_extracted == len(prompts)

        for prompt in prompts:
            assert is_prompt_fully_cached(tiny_model, prompt, set(layers))

        probe = LinearProbe(model=tiny_model, layers=layers, device="cpu", remote=False)
        probe.fit([prompts[0]], [prompts[1]])

        for prompt in prompts:
            cache_dir = get_prompt_cache_dir(tiny_model, prompt)
            cached = get_prompt_cached_layers(cache_dir)
            assert cached == set(layers)

    def test_pooled_cache_warmup_enables_linear_probe_cache_hit(
        self, tiny_model, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import LinearProbe, UnifiedCache
        from lmprobe.cache import is_prompt_pooled_cached

        prompts = ["This is a test prompt", "Another test"]
        layers = [0, 1]
        pooling = "last_token"

        cache = UnifiedCache(
            model=tiny_model, layers=layers,
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=True, pooling=pooling,
        )
        stats = cache.warmup(prompts)
        assert stats.activations_extracted == len(prompts)

        for prompt in prompts:
            assert is_prompt_pooled_cached(tiny_model, prompt, set(layers), pooling)

        probe = LinearProbe(
            model=tiny_model, layers=layers, pooling=pooling,
            device="cpu", remote=False,
        )

        cache_files_before = sum(len(files) for _, _, files in os.walk(tmp_path))
        probe.fit([prompts[0]], [prompts[1]])
        cache_files_after = sum(len(files) for _, _, files in os.walk(tmp_path))

        assert cache_files_after == cache_files_before, (
            f"LinearProbe should not create new cache files when pooled cache exists. "
            f"Before: {cache_files_before}, After: {cache_files_after}"
        )

    def test_linear_probe_respects_pooled_cache_format(
        self, tiny_model, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import LinearProbe, UnifiedCache

        prompts = ["Test prompt one", "Test prompt two"]
        layers = [0, 1]
        pooling = "last_token"

        cache = UnifiedCache(
            model=tiny_model, layers=layers,
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=True, pooling=pooling,
        )
        cache.warmup(prompts)
        acts_from_cache, _ = cache.get_activations(prompts)

        probe = LinearProbe(
            model=tiny_model, layers=layers, pooling=pooling,
            device="cpu", remote=False,
        )
        probe.fit([prompts[0]], [prompts[1]])

        from lmprobe.pooling import get_pooling_fn

        pooling_fn = get_pooling_fn(pooling)
        acts_raw, mask = probe._cached_extractor.extract(prompts, remote=False)
        acts_from_probe = pooling_fn(acts_raw, mask)

        assert torch.allclose(acts_from_cache, acts_from_probe, atol=1e-5)


class TestUnifiedCacheActivationBaselineIntegration:
    """Tests for cache compatibility between UnifiedCache and ActivationBaseline."""

    def test_pooled_cache_warmup_enables_activation_baseline_cache_hit(
        self, tiny_model, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import ActivationBaseline, UnifiedCache
        from lmprobe.cache import is_prompt_pooled_cached

        prompts = ["This is a test prompt", "Another test"]
        layers = [0, 1]
        pooling = "last_token"

        cache = UnifiedCache(
            model=tiny_model, layers=layers,
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=True, pooling=pooling,
        )
        stats = cache.warmup(prompts)
        assert stats.activations_extracted == len(prompts)

        for prompt in prompts:
            assert is_prompt_pooled_cached(tiny_model, prompt, set(layers), pooling)

        cache_files_before = sum(len(files) for _, _, files in os.walk(tmp_path))

        baseline = ActivationBaseline(
            model=tiny_model, method="random_direction",
            layers=layers, pooling=pooling,
            device="cpu", remote=False, random_state=42,
        )
        baseline.fit([prompts[0]], [prompts[1]])

        cache_files_after = sum(len(files) for _, _, files in os.walk(tmp_path))
        assert cache_files_after == cache_files_before

    def test_pooled_cache_works_for_pca_baseline(
        self, tiny_model, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import ActivationBaseline, UnifiedCache

        prompts = ["Test prompt one", "Test prompt two", "Test prompt three"]
        layers = [0, 1]
        pooling = "last_token"

        cache = UnifiedCache(
            model=tiny_model, layers=layers,
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=True, pooling=pooling,
        )
        cache.warmup(prompts)

        cache_files_before = sum(len(files) for _, _, files in os.walk(tmp_path))

        baseline = ActivationBaseline(
            model=tiny_model, method="pca", layers=layers,
            pooling=pooling, n_components=2, device="cpu", remote=False,
        )
        baseline.fit([prompts[0]], [prompts[1]])
        baseline.predict([prompts[2]])

        cache_files_after = sum(len(files) for _, _, files in os.walk(tmp_path))
        assert cache_files_after == cache_files_before

    def test_layer_0_baseline_uses_correct_cache(
        self, tiny_model, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import ActivationBaseline, UnifiedCache
        from lmprobe.cache import is_prompt_pooled_cached

        prompts = ["Test prompt A", "Test prompt B"]
        pooling = "last_token"

        cache = UnifiedCache(
            model=tiny_model, layers=[0], compute_perplexity=False,
            device="cpu", remote=False, cache_pooled=True, pooling=pooling,
        )
        cache.warmup(prompts)

        for prompt in prompts:
            assert is_prompt_pooled_cached(tiny_model, prompt, {0}, pooling)

        cache_files_before = sum(len(files) for _, _, files in os.walk(tmp_path))

        baseline = ActivationBaseline(
            model=tiny_model, method="layer_0", pooling=pooling,
            device="cpu", remote=False,
        )
        baseline.fit([prompts[0]], [prompts[1]])

        cache_files_after = sum(len(files) for _, _, files in os.walk(tmp_path))
        assert cache_files_after == cache_files_before


class TestUnifiedCachePerplexityBaselineIntegration:
    """Tests for perplexity cache compatibility."""

    def test_perplexity_cache_warmup_enables_baseline_cache_hit(
        self, tiny_model, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import BaselineProbe, UnifiedCache
        from lmprobe.cache import is_prompt_perplexity_cached

        prompts = ["This is a test prompt", "Another test"]

        cache = UnifiedCache(
            model=tiny_model, layers=[0, 1],
            compute_perplexity=True, device="cpu", remote=False,
            cache_pooled=True,
        )
        stats = cache.warmup(prompts)
        assert stats.perplexity_extracted == len(prompts)

        for prompt in prompts:
            assert is_prompt_perplexity_cached(tiny_model, prompt)

        cache_files_before = sum(len(files) for _, _, files in os.walk(tmp_path))

        baseline = BaselineProbe(
            method="perplexity", model=tiny_model, device="cpu", remote=False,
        )
        baseline.fit([prompts[0]], [prompts[1]])

        cache_files_after = sum(len(files) for _, _, files in os.walk(tmp_path))
        assert cache_files_after == cache_files_before

    def test_perplexity_cache_shared_between_unified_and_baseline(
        self, tiny_model, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe import BaselineProbe, UnifiedCache
        from lmprobe.cache import is_prompt_perplexity_cached

        prompts = ["Forward direction test", "Backward direction test"]

        baseline = BaselineProbe(
            method="perplexity", model=tiny_model, device="cpu", remote=False,
        )
        baseline.fit([prompts[0]], [prompts[1]])

        for prompt in prompts:
            assert is_prompt_perplexity_cached(tiny_model, prompt)

        cache = UnifiedCache(
            model=tiny_model, layers=[0, 1],
            compute_perplexity=True, device="cpu", remote=False,
        )
        stats = cache.warmup(prompts)

        assert stats.perplexity_cached == len(prompts)
        assert stats.perplexity_extracted == 0

    def test_unified_cache_perplexity_matches_baseline(
        self, tiny_model, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        import numpy as np

        from lmprobe import BaselineProbe, UnifiedCache

        prompts = ["Matching test prompt"]

        cache = UnifiedCache(
            model=tiny_model, layers=[0],
            compute_perplexity=True, device="cpu", remote=False,
        )
        cache.warmup(prompts)
        ppl_from_cache = cache.get_perplexity(prompts)

        baseline = BaselineProbe(
            method="perplexity", model=tiny_model, device="cpu", remote=False,
        )
        ppl_from_baseline = baseline._compute_perplexity(prompts)

        assert np.allclose(ppl_from_cache, ppl_from_baseline, atol=1e-5)


class TestSavePromptTopkLogits:
    """Tests for save_prompt_topk_logits (pre-compressed top-k)."""

    def test_save_and_load_topk_logits(self, tmp_path, monkeypatch):
        """Pre-compressed top-k logits round-trip correctly."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        model_name = "test-model"
        prompt = "test topk logits save"
        K = 5
        seq_len = 10

        values = torch.randn(1, seq_len, K)
        indices = torch.randint(0, 1000, (1, seq_len, K))
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)

        save_prompt_topk_logits(
            model_name, prompt, values, indices, attention_mask,
            positions="all",
        )

        assert is_prompt_logits_cached(model_name, prompt, top_k=K)

        loaded_values, loaded_indices = load_prompt_logits(
            model_name, prompt, top_k=K
        )
        assert loaded_values.shape == (1, seq_len, K)
        assert loaded_indices.shape == (1, seq_len, K)
        assert torch.allclose(loaded_values, values, atol=1e-4)
        assert (loaded_indices == indices.to(torch.int32)).all()

    def test_save_topk_logits_last_position(self, tmp_path, monkeypatch):
        """Position selection works on pre-compressed top-k logits."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        model_name = "test-model"
        prompt = "test topk last position"
        K = 3
        seq_len = 8

        values = torch.randn(1, seq_len, K)
        indices = torch.randint(0, 500, (1, seq_len, K))
        # 6 real tokens, 2 padding
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])

        save_prompt_topk_logits(
            model_name, prompt, values, indices, attention_mask,
            positions="last",
        )

        loaded_values, loaded_indices = load_prompt_logits(
            model_name, prompt, top_k=K
        )
        # Should have only 1 position (last non-padding = index 5)
        assert loaded_values.shape == (1, 1, K)
        assert loaded_indices.shape == (1, 1, K)
        # Values should match position 5 of the original
        assert torch.allclose(loaded_values[0, 0], values[0, 5], atol=1e-4)
