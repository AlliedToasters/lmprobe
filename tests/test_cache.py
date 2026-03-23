"""Tests for activation caching (v1 legacy + v2 safetensors)."""

import os

import pytest
import torch

from lmprobe.cache import (
    CacheInfo,
    _hash_string,
    _merge_save_backend,
    _prompt_cache_key,
    _prompt_logits_key,
    _prompt_perplexity_key,
    batch_check_cache_status,
    cache_info,
    get_cached_layers,
    get_extraction_cache_dir,
    get_prompt_cache_dir,
    get_prompt_cache_path,
    get_prompt_cached_layers,
    invalidate_extraction_cache,
    is_prompt_fully_cached,
    is_prompt_logits_cached,
    is_prompt_perplexity_cached,
    is_prompt_pooled_cached,
    load_attention_mask,
    load_layer,
    load_prompt_activations,
    load_prompt_logits,
    load_prompt_perplexity,
    load_prompt_pooled_activations,
    save_attention_mask,
    save_layer,
    save_prompt_activations,
    save_prompt_logits,
    save_prompt_perplexity,
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
            # Save one more (eviction is decoupled from writes)
            save_prompt_activations(
                "test-model", "prompt-trigger", [0],
                torch.randn(1, 10, 64),
                torch.ones(1, 10, dtype=torch.long),
            )

            # Explicitly run eviction
            from lmprobe.cache import evict
            evict()

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


class TestBatchLayerLoading:
    """Tests for load_layer_across_prompts and load_layer_last_token."""

    def test_load_layer_across_prompts_basic(self, tiny_model, tmp_path, monkeypatch):
        """load_layer_across_prompts returns correct shapes and count."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import load_layer_across_prompts
        from lmprobe.unified_cache import UnifiedCache

        prompts = ["hello world", "another prompt"]
        cache = UnifiedCache(
            model=tiny_model, layers=[0, 1],
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache.warmup(prompts)

        acts_list, masks_list = load_layer_across_prompts(tiny_model, prompts, layer=0)

        assert len(acts_list) == 2
        assert len(masks_list) == 2
        for acts, mask in zip(acts_list, masks_list):
            assert acts.ndim == 3  # (1, seq_len, hidden_dim)
            assert acts.shape[0] == 1
            assert mask.ndim == 2  # (1, seq_len)
            assert mask.shape[0] == 1
            assert acts.shape[1] == mask.shape[1]

    def test_load_layer_across_prompts_variable_seqlen(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Different prompts can have different sequence lengths."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import load_layer_across_prompts
        from lmprobe.unified_cache import UnifiedCache

        prompts = ["hi", "this is a much longer prompt with many tokens"]
        cache = UnifiedCache(
            model=tiny_model, layers=[0, 1],
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache.warmup(prompts)

        acts_list, masks_list = load_layer_across_prompts(tiny_model, prompts, layer=1)

        assert len(acts_list) == 2
        # Different prompts may have different seq_len
        seq_len_0 = acts_list[0].shape[1]
        seq_len_1 = acts_list[1].shape[1]
        # The longer prompt should have more tokens
        assert seq_len_1 >= seq_len_0

    def test_load_layer_last_token_shape(self, tiny_model, tmp_path, monkeypatch):
        """load_layer_last_token returns (N, hidden_dim)."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import load_layer_last_token
        from lmprobe.unified_cache import UnifiedCache

        prompts = ["test one", "test two", "test three"]
        cache = UnifiedCache(
            model=tiny_model, layers=[0, 1],
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache.warmup(prompts)

        result = load_layer_last_token(tiny_model, prompts, layer=0)
        assert result.ndim == 2
        assert result.shape[0] == 3  # N prompts

    def test_load_layer_last_token_matches_manual(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """load_layer_last_token matches manual extraction from full activations."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import load_layer_last_token, load_prompt_activations
        from lmprobe.unified_cache import UnifiedCache

        prompts = ["manual comparison test"]
        cache = UnifiedCache(
            model=tiny_model, layers=[0, 1],
            compute_perplexity=False, device="cpu", remote=False,
            cache_pooled=False,
        )
        cache.warmup(prompts)

        # Get via load_layer_last_token
        result = load_layer_last_token(tiny_model, prompts, layer=0)

        # Get manually from full activations
        acts, mask = load_prompt_activations(tiny_model, prompts[0], [0])
        last_pos = mask[0].nonzero(as_tuple=True)[0][-1].item()
        expected = acts[0, last_pos, :]

        assert torch.allclose(result[0], expected)

    def test_load_layer_uncached_raises(self, tiny_model, tmp_path, monkeypatch):
        """Error on missing cache."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import load_layer_across_prompts

        with pytest.raises(FileNotFoundError):
            load_layer_across_prompts(tiny_model, ["uncached prompt"], layer=0)


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


class TestSidecarFiles:
    """Tests for sidecar file storage of logits and perplexity (#120).

    Verifies that logits/perplexity write to separate sidecar files
    instead of merging into the main activation file, and that backward
    compat with pre-sidecar entries is preserved.
    """

    @pytest.fixture(autouse=True)
    def setup_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.cache_dir = tmp_path
        self.model = "test-model"
        self.prompt = "sidecar test prompt"

    def test_logits_creates_sidecar_file(self):
        """save_prompt_logits writes to .logits.safetensors, not main file."""
        from lmprobe.cache import get_backend

        logits = torch.randn(1, 5, 100)
        mask = torch.ones(1, 5)
        save_prompt_logits(self.model, self.prompt, logits, mask, top_k=10)

        backend = get_backend()
        sidecar_key = _prompt_logits_key(self.model, self.prompt)
        main_key = _prompt_cache_key(self.model, self.prompt)

        assert backend.exists(sidecar_key), "Sidecar file should exist"
        assert not backend.exists(main_key), "Main file should not be created"

    def test_perplexity_creates_sidecar_file(self):
        """save_prompt_perplexity writes to .perplexity.safetensors."""
        from lmprobe.cache import get_backend

        perp = torch.tensor([1.5, 2.0, 0.8])
        save_prompt_perplexity(self.model, self.prompt, perp)

        backend = get_backend()
        sidecar_key = _prompt_perplexity_key(self.model, self.prompt)
        main_key = _prompt_cache_key(self.model, self.prompt)

        assert backend.exists(sidecar_key), "Sidecar file should exist"
        assert not backend.exists(main_key), "Main file should not be created"

    def test_logits_roundtrip_via_sidecar(self):
        """Logits saved to sidecar can be loaded back."""
        logits = torch.randn(1, 5, 100)
        mask = torch.ones(1, 5)
        save_prompt_logits(self.model, self.prompt, logits, mask, top_k=10)

        assert is_prompt_logits_cached(self.model, self.prompt, top_k=10)
        values, indices = load_prompt_logits(self.model, self.prompt, top_k=10)
        assert values.shape == (1, 1, 10)
        assert indices.shape == (1, 1, 10)

    def test_perplexity_roundtrip_via_sidecar(self):
        """Perplexity saved to sidecar can be loaded back."""
        perp = torch.tensor([1.5, 2.0, 0.8])
        save_prompt_perplexity(self.model, self.prompt, perp)

        assert is_prompt_perplexity_cached(self.model, self.prompt)
        loaded = load_prompt_perplexity(self.model, self.prompt)
        assert torch.allclose(loaded, perp, atol=1e-3)

    def test_backward_compat_logits_in_main_file(self):
        """Logits merged into main file (pre-sidecar) can still be loaded."""
        # Simulate pre-sidecar behavior: merge logits into main file
        logits = torch.randn(1, 1, 100)
        main_key = _prompt_cache_key(self.model, self.prompt)
        _merge_save_backend(main_key, {
            "logits_top_k_values": logits,
            "logits_top_k_indices": torch.zeros(1, 1, 100, dtype=torch.int32),
        })

        assert is_prompt_logits_cached(self.model, self.prompt, top_k=100)
        values, indices = load_prompt_logits(self.model, self.prompt, top_k=100)
        assert values.shape == (1, 1, 100)

    def test_backward_compat_perplexity_in_main_file(self):
        """Perplexity merged into main file (pre-sidecar) can still be loaded."""
        perp = torch.tensor([1.5, 2.0, 0.8])
        main_key = _prompt_cache_key(self.model, self.prompt)
        _merge_save_backend(main_key, {"perplexity": perp})

        assert is_prompt_perplexity_cached(self.model, self.prompt)
        loaded = load_prompt_perplexity(self.model, self.prompt)
        assert torch.allclose(loaded, perp, atol=1e-3)

    def test_sidecar_preferred_over_main_file(self):
        """When both sidecar and main file have logits, sidecar wins."""
        # Write old data to main file
        main_key = _prompt_cache_key(self.model, self.prompt)
        old_values = torch.ones(1, 1, 5)
        _merge_save_backend(main_key, {
            "logits_top_k_values": old_values,
            "logits_top_k_indices": torch.zeros(1, 1, 5, dtype=torch.int32),
        })

        # Write new data to sidecar
        logits = torch.randn(1, 3, 100)
        mask = torch.ones(1, 3)
        save_prompt_logits(self.model, self.prompt, logits, mask, top_k=5)

        # Load should come from sidecar (not old main file data)
        values, indices = load_prompt_logits(self.model, self.prompt, top_k=5)
        # The sidecar values come from topk of our random logits, not ones
        assert not torch.allclose(values, old_values)

    def test_sidecar_not_counted_as_extra_prompt(self):
        """cache_info should not double-count sidecar files as prompts."""
        acts = torch.randn(1, 5, 32)
        mask = torch.ones(1, 5)
        save_prompt_activations(self.model, self.prompt, [0], acts, mask)

        perp = torch.tensor([1.0, 2.0, 3.0])
        save_prompt_perplexity(self.model, self.prompt, perp)

        logits = torch.randn(1, 5, 100)
        save_prompt_logits(self.model, self.prompt, logits, mask, top_k=10)

        info = cache_info(self.model)
        assert len(info.models) == 1
        assert info.models[0].num_prompts == 1  # not 3
        assert info.models[0].has_logits is True
        assert info.models[0].has_perplexity is True


class TestBatchCheckCacheStatus:
    """Tests for batch_check_cache_status using LIST instead of per-prompt HEAD."""

    @pytest.fixture(autouse=True)
    def setup_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        from lmprobe.cache import set_cache_backend

        set_cache_backend(None)
        self.model = "test-model"
        yield
        set_cache_backend(None)

    def test_all_uncached(self):
        """All prompts reported as needing extraction when cache is empty."""
        prompts = ["hello", "world", "foo"]
        need_act, need_ppl, need_log, partial, _ = batch_check_cache_status(
            self.model, prompts, required_layers={0, 1}
        )
        assert need_act == prompts
        assert need_ppl == []
        assert need_log == []
        assert partial == 0

    def test_all_cached(self):
        """No prompts need extraction when all are fully cached."""
        prompts = ["prompt A", "prompt B"]
        for p in prompts:
            acts = torch.randn(1, 5, 64)
            mask = torch.ones(1, 5)
            save_prompt_activations(self.model, p, [0, 1], acts, mask)

        need_act, _, _, partial, _ = batch_check_cache_status(
            self.model, prompts, required_layers={0, 1}
        )
        assert need_act == []
        assert partial == 0

    def test_partial_cached(self):
        """Detects partial cache (some layers missing)."""
        prompt = "partial prompt"
        # Save only layer 0
        acts = torch.randn(1, 5, 32)
        mask = torch.ones(1, 5)
        save_prompt_activations(self.model, prompt, [0], acts, mask)

        need_act, _, _, partial, found = batch_check_cache_status(
            self.model, [prompt], required_layers={0, 1}
        )
        assert need_act == [prompt]
        assert partial == 1
        assert found is not None
        assert 0 in found

    def test_mixed_cached_and_uncached(self):
        """Correctly separates cached vs uncached prompts."""
        cached_prompt = "cached"
        uncached_prompt = "uncached"
        acts = torch.randn(1, 5, 64)
        mask = torch.ones(1, 5)
        save_prompt_activations(self.model, cached_prompt, [0, 1], acts, mask)

        need_act, _, _, _, _ = batch_check_cache_status(
            self.model,
            [cached_prompt, uncached_prompt],
            required_layers={0, 1},
        )
        assert need_act == [uncached_prompt]

    def test_perplexity_check(self):
        """Detects missing perplexity cache."""
        prompt = "ppl test"
        acts = torch.randn(1, 5, 64)
        mask = torch.ones(1, 5)
        save_prompt_activations(self.model, prompt, [0, 1], acts, mask)

        _, need_ppl, _, _, _ = batch_check_cache_status(
            self.model,
            [prompt],
            required_layers={0, 1},
            compute_perplexity=True,
        )
        assert need_ppl == [prompt]

        # Now cache perplexity
        save_prompt_perplexity(self.model, prompt, torch.tensor([1.0, 2.0, 3.0]))
        _, need_ppl, _, _, _ = batch_check_cache_status(
            self.model,
            [prompt],
            required_layers={0, 1},
            compute_perplexity=True,
        )
        assert need_ppl == []

    def test_logits_check(self):
        """Detects missing logits cache."""
        prompt = "logits test"
        acts = torch.randn(1, 5, 64)
        mask = torch.ones(1, 5)
        save_prompt_activations(self.model, prompt, [0, 1], acts, mask)

        _, _, need_log, _, _ = batch_check_cache_status(
            self.model,
            [prompt],
            required_layers={0, 1},
            cache_logits=True,
        )
        assert need_log == [prompt]

        # Now cache logits
        logits = torch.randn(1, 5, 100)
        save_prompt_logits(self.model, prompt, logits, mask)
        _, _, need_log, _, _ = batch_check_cache_status(
            self.model,
            [prompt],
            required_layers={0, 1},
            cache_logits=True,
        )
        assert need_log == []

    def test_pooled_check(self):
        """Works with pooled activations."""
        prompt = "pooled test"
        pooled = torch.randn(1, 64)
        save_prompt_pooled_activations(
            self.model, prompt, [0, 1], pooled, "last_token"
        )

        need_act, _, _, _, _ = batch_check_cache_status(
            self.model,
            [prompt],
            required_layers={0, 1},
            pooling="last_token",
        )
        assert need_act == []

    def test_header_only_read(self):
        """_get_tensor_keys_header_only correctly parses safetensors header."""
        from lmprobe.cache import _get_tensor_keys_header_only, get_backend

        prompt = "header test"
        acts = torch.randn(1, 5, 64)
        mask = torch.ones(1, 5)
        save_prompt_activations(self.model, prompt, [0, 1], acts, mask)

        backend = get_backend()
        key = _prompt_cache_key(self.model, prompt)
        keys = _get_tensor_keys_header_only(backend, key)
        assert "layer_0" in keys
        assert "layer_1" in keys
        assert "attention_mask" in keys


class TestSyncCache:
    """Tests for sync_cache()."""

    def test_copies_entries(self, tmp_path):
        """sync_cache copies entries from source to dest."""
        from lmprobe.cache import sync_cache
        from lmprobe.cache_backends import LocalCacheBackend

        src = LocalCacheBackend(tmp_path / "src")
        dst = LocalCacheBackend(tmp_path / "dst")

        src.write_bytes("modelhash/prompt1.safetensors", b"data1")
        src.write_bytes("modelhash/prompt2.safetensors", b"data2")
        src.write_text("modelhash/_model_name.txt", "test-model")

        count = sync_cache(source=src, dest=dst)
        assert count == 3
        assert dst.read_bytes("modelhash/prompt1.safetensors") == b"data1"
        assert dst.read_bytes("modelhash/prompt2.safetensors") == b"data2"
        assert dst.read_text("modelhash/_model_name.txt") == "test-model"

    def test_skips_existing(self, tmp_path):
        """sync_cache does not overwrite entries that already exist in dest."""
        from lmprobe.cache import sync_cache
        from lmprobe.cache_backends import LocalCacheBackend

        src = LocalCacheBackend(tmp_path / "src")
        dst = LocalCacheBackend(tmp_path / "dst")

        src.write_bytes("modelhash/prompt1.safetensors", b"new-data")
        dst.write_bytes("modelhash/prompt1.safetensors", b"old-data")

        count = sync_cache(source=src, dest=dst)
        assert count == 0
        # Original data preserved
        assert dst.read_bytes("modelhash/prompt1.safetensors") == b"old-data"

    def test_model_filter(self, tmp_path):
        """sync_cache with model filter only copies matching model's entries."""
        from lmprobe.cache import _hash_string, sync_cache
        from lmprobe.cache_backends import LocalCacheBackend

        src = LocalCacheBackend(tmp_path / "src")
        dst = LocalCacheBackend(tmp_path / "dst")

        target_model = "target-model"
        other_model = "other-model"
        target_hash = _hash_string(target_model)
        other_hash = _hash_string(other_model)

        src.write_bytes(f"{target_hash}/prompt1.safetensors", b"target")
        src.write_bytes(f"{other_hash}/prompt2.safetensors", b"other")

        count = sync_cache(source=src, dest=dst, model=target_model)
        assert count == 1
        assert dst.exists(f"{target_hash}/prompt1.safetensors")
        assert not dst.exists(f"{other_hash}/prompt2.safetensors")

    def test_string_uris(self, tmp_path):
        """sync_cache accepts filesystem path strings."""
        from lmprobe.cache import sync_cache
        from lmprobe.cache_backends import LocalCacheBackend

        src_dir = tmp_path / "src"
        dst_dir = tmp_path / "dst"
        src_dir.mkdir()

        # Write via backend, then sync via string paths
        src = LocalCacheBackend(src_dir)
        src.write_bytes("modelhash/prompt.safetensors", b"data")

        count = sync_cache(source=str(src_dir), dest=str(dst_dir))
        assert count == 1
        dst = LocalCacheBackend(dst_dir)
        assert dst.read_bytes("modelhash/prompt.safetensors") == b"data"

    def test_returns_count(self, tmp_path):
        """sync_cache returns exact count of entries copied."""
        from lmprobe.cache import sync_cache
        from lmprobe.cache_backends import LocalCacheBackend

        src = LocalCacheBackend(tmp_path / "src")
        dst = LocalCacheBackend(tmp_path / "dst")

        src.write_bytes("m/a.safetensors", b"a")
        src.write_bytes("m/b.safetensors", b"b")
        src.write_bytes("m/c.safetensors", b"c")
        # Pre-populate one in dest
        dst.write_bytes("m/b.safetensors", b"b")

        count = sync_cache(source=src, dest=dst)
        assert count == 2  # a and c copied, b skipped
