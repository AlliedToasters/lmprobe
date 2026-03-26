"""Tests for activation caching (v1 legacy + v2 safetensors)."""

import json
import os

import pytest
import torch

from lmprobe.cache import (
    CacheInfo,
    ManifestEntry,
    _hash_string,
    _manifest_key,
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
    list_cached_prompts,
    load_attention_mask,
    load_layer,
    load_prompt_activations,
    load_prompt_logits,
    load_prompt_perplexity,
    load_prompt_pooled_activations,
    read_manifest,
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
from lmprobe.cache_backends import LocalCacheBackend


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


class _RangeReadTrackingBackend:
    """Wraps a LocalCacheBackend but is NOT an instance of it.

    This forces _load_tensors_from_backend to take the non-local path
    (selective range reads). Tracks read_range calls for assertions.
    """

    def __init__(self, local_backend):
        self._inner = local_backend
        self.read_range_calls = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def read_range(self, key, start, end):
        self.read_range_calls.append((key, start, end))
        return self._inner.read_range(key, start, end)

    def read_bytes(self, key):
        return self._inner.read_bytes(key)


class TestSelectiveTensorLoading:
    """Tests for selective tensor loading via range reads (#146)."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        self.model = "test-model-selective"
        self.cache_dir = tmp_path
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        import lmprobe.cache as cache_mod

        cache_mod._backend = None

        # Clear header caches so range-read counts are deterministic
        from lmprobe.cache import _clear_selective_caches

        _clear_selective_caches()

        # Save some test data (hidden_dim=48 splits evenly across 3 layers → 16 each)
        acts = torch.randn(1, 5, 48)
        mask = torch.ones(1, 5)
        save_prompt_activations(self.model, "prompt1", [0, 1, 2], acts, mask)

        self.key = _prompt_cache_key(self.model, "prompt1")

    def test_selective_load_returns_correct_tensors(self):
        """Selective loading returns the same tensors as full loading."""
        from lmprobe.cache import _load_tensors_selective, get_backend

        backend = get_backend()
        wrapper = _RangeReadTrackingBackend(backend)

        result = _load_tensors_selective(wrapper, self.key, ["layer_0", "layer_2"])

        assert set(result.keys()) == {"layer_0", "layer_2"}
        assert result["layer_0"].shape == (1, 5, 16)
        assert result["layer_2"].shape == (1, 5, 16)

    def test_selective_load_matches_full_load(self):
        """Selective loading produces identical tensors to full loading."""
        from lmprobe.cache import (
            _load_tensors_from_backend,
            _load_tensors_selective,
            get_backend,
        )

        backend = get_backend()
        wrapper = _RangeReadTrackingBackend(backend)

        full = _load_tensors_from_backend(self.key)
        selective = _load_tensors_selective(
            wrapper, self.key, ["layer_0", "layer_1", "attention_mask"]
        )

        for k in selective:
            assert torch.equal(full[k], selective[k]), f"Mismatch for {k}"

    def test_selective_load_uses_range_reads(self):
        """Selective loading uses read_range, not read_bytes."""
        from lmprobe.cache import _load_tensors_selective, get_backend

        backend = get_backend()
        wrapper = _RangeReadTrackingBackend(backend)

        _load_tensors_selective(wrapper, self.key, ["layer_1"])

        # Should have range reads: 2 for header (size + json) + 1 for tensor data
        assert len(wrapper.read_range_calls) == 3

    def test_selective_load_missing_key_raises(self):
        """Requesting a non-existent tensor key raises KeyError."""
        from lmprobe.cache import _load_tensors_selective, get_backend

        backend = get_backend()
        wrapper = _RangeReadTrackingBackend(backend)

        with pytest.raises(KeyError, match="missing key 'layer_99'"):
            _load_tensors_selective(wrapper, self.key, ["layer_99"])

    def test_parse_safetensors_header(self):
        """_parse_safetensors_header returns valid header with offsets."""
        from lmprobe.cache import _parse_safetensors_header, get_backend

        backend = get_backend()
        header, data_start = _parse_safetensors_header(backend, self.key)

        assert "layer_0" in header
        assert "attention_mask" in header
        assert "__metadata__" not in header or isinstance(
            header["__metadata__"], dict
        )
        # Each tensor entry has dtype, shape, data_offsets
        meta = header["layer_0"]
        assert "dtype" in meta
        assert "shape" in meta
        assert "data_offsets" in meta
        assert data_start > 8

    def test_load_tensors_from_backend_uses_selective_for_nonlocal(self):
        """_load_tensors_from_backend uses range reads for non-local backends."""
        import lmprobe.cache as cache_mod
        from lmprobe.cache import _load_tensors_from_backend, get_backend

        backend = get_backend()
        wrapper = _RangeReadTrackingBackend(backend)

        # Temporarily replace the backend
        old_backend = cache_mod._backend
        cache_mod._backend = wrapper
        try:
            result = _load_tensors_from_backend(self.key, ["layer_0"])
        finally:
            cache_mod._backend = old_backend

        assert "layer_0" in result
        # Verify range reads were used (not read_bytes for full file)
        assert len(wrapper.read_range_calls) >= 3


class TestHeaderCache:
    """Tests for safetensors header caching (#148)."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        self.model = "test-model-header-cache"
        self.cache_dir = tmp_path
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        import lmprobe.cache as cache_mod

        cache_mod._backend = None

        acts = torch.randn(1, 5, 48)
        mask = torch.ones(1, 5)
        save_prompt_activations(self.model, "prompt1", [0, 1, 2], acts, mask)

        self.key = _prompt_cache_key(self.model, "prompt1")

        # Clear caches before each test
        from lmprobe.cache import _clear_selective_caches

        _clear_selective_caches()

    def test_header_cache_eliminates_redundant_reads(self):
        """Repeated selective loads for the same key reuse cached header."""
        from lmprobe.cache import _load_tensors_selective, get_backend

        backend = get_backend()
        wrapper = _RangeReadTrackingBackend(backend)

        # First call: 2 range reads for header + 1 for tensor data = 3
        _load_tensors_selective(wrapper, self.key, ["layer_0"])
        first_call_reads = len(wrapper.read_range_calls)
        assert first_call_reads == 3

        # Second call: header is cached, so only 1 range read for tensor data
        _load_tensors_selective(wrapper, self.key, ["layer_1"])
        second_call_reads = len(wrapper.read_range_calls) - first_call_reads
        assert second_call_reads == 1

        # Third call: still cached
        _load_tensors_selective(wrapper, self.key, ["layer_2"])
        third_call_reads = (
            len(wrapper.read_range_calls) - first_call_reads - second_call_reads
        )
        assert third_call_reads == 1

    def test_clear_header_cache(self):
        """Clearing the header cache forces re-reading on next call."""
        from lmprobe.cache import (
            _load_tensors_selective,
            clear_header_cache,
            get_backend,
        )

        backend = get_backend()
        wrapper = _RangeReadTrackingBackend(backend)

        # Populate cache
        _load_tensors_selective(wrapper, self.key, ["layer_0"])
        assert len(wrapper.read_range_calls) == 3

        # Clear and re-load: should re-read header
        clear_header_cache()
        _load_tensors_selective(wrapper, self.key, ["layer_0"])
        assert len(wrapper.read_range_calls) == 6  # 3 + 3

    def test_header_cache_bounded(self):
        """Header cache does not grow beyond _HEADER_CACHE_MAXSIZE."""
        from lmprobe.cache import _header_cache, _parse_safetensors_header, get_backend

        backend = get_backend()

        # Fill cache with fake entries
        for i in range(_HEADER_CACHE_MAXSIZE := 8192):
            _header_cache[(id(backend), f"fake_key_{i}")] = ({"fake": True}, 8)

        assert len(_header_cache) == 8192

        # Parsing a real header should evict one entry and stay at maxsize
        wrapper = _RangeReadTrackingBackend(backend)
        _parse_safetensors_header(wrapper, self.key)
        assert len(_header_cache) <= 8192


class TestMaskCache:
    """Tests for attention mask caching (#148)."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        self.model = "test-model-mask-cache"
        self.cache_dir = tmp_path
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        import lmprobe.cache as cache_mod

        cache_mod._backend = None

        acts = torch.randn(1, 5, 48)
        mask = torch.ones(1, 5)
        save_prompt_activations(self.model, "prompt1", [0, 1, 2], acts, mask)

        self.key = _prompt_cache_key(self.model, "prompt1")

        from lmprobe.cache import _clear_selective_caches

        _clear_selective_caches()

    def test_mask_cache_eliminates_redundant_reads(self):
        """Attention mask is only fetched once across multiple layer loads."""
        from lmprobe.cache import _load_tensors_selective, get_backend

        backend = get_backend()
        wrapper = _RangeReadTrackingBackend(backend)

        # First call: header(2) + layer(1) + mask(1) = 4 range reads
        result1 = _load_tensors_selective(
            wrapper, self.key, ["layer_0", "attention_mask"]
        )
        first_call_reads = len(wrapper.read_range_calls)
        assert first_call_reads == 4
        assert "attention_mask" in result1

        # Second call: header cached, mask cached → only layer data = 1 range read
        result2 = _load_tensors_selective(
            wrapper, self.key, ["layer_1", "attention_mask"]
        )
        second_call_reads = len(wrapper.read_range_calls) - first_call_reads
        assert second_call_reads == 1
        assert "attention_mask" in result2

        # Verify mask values are identical
        assert torch.equal(result1["attention_mask"], result2["attention_mask"])

    def test_clear_mask_cache(self):
        """Clearing the mask cache forces re-reading on next call."""
        from lmprobe.cache import (
            _load_tensors_selective,
            clear_mask_cache,
            get_backend,
        )

        backend = get_backend()
        wrapper = _RangeReadTrackingBackend(backend)

        # Populate mask cache
        _load_tensors_selective(
            wrapper, self.key, ["layer_0", "attention_mask"]
        )
        reads_after_first = len(wrapper.read_range_calls)

        # Clear mask cache (but not header cache)
        clear_mask_cache()

        # Should re-read mask but not header
        _load_tensors_selective(
            wrapper, self.key, ["layer_0", "attention_mask"]
        )
        reads_after_second = len(wrapper.read_range_calls) - reads_after_first
        # header cached (0) + layer(1) + mask re-read(1) = 2
        assert reads_after_second == 2

    def test_clear_selective_caches(self):
        """_clear_selective_caches clears both header and mask caches."""
        from lmprobe.cache import (
            _clear_selective_caches,
            _header_cache,
            _load_tensors_selective,
            _mask_cache,
            get_backend,
        )

        backend = get_backend()
        wrapper = _RangeReadTrackingBackend(backend)

        _load_tensors_selective(
            wrapper, self.key, ["layer_0", "attention_mask"]
        )
        assert len(_header_cache) > 0
        assert len(_mask_cache) > 0

        _clear_selective_caches()
        assert len(_header_cache) == 0
        assert len(_mask_cache) == 0


# =============================================================================
# Prompt manifest sidecar (#162)
# =============================================================================

TEST_MODEL = "stas/tiny-random-llama-2"


class TestPromptManifest:
    """Tests for the _manifest.jsonl sidecar file."""

    @pytest.fixture(autouse=True)
    def cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        return tmp_path

    def test_manifest_created_on_save(self):
        """save_prompt_activations appends a manifest entry."""
        prompt = "Hello world"
        acts = torch.randn(1, 5, 32)
        mask = torch.ones(1, 5)
        save_prompt_activations(TEST_MODEL, prompt, [0], acts, mask)

        entries = read_manifest(TEST_MODEL)
        assert len(entries) == 1
        assert entries[0].prompt == prompt
        assert entries[0].num_tokens == 5
        assert entries[0].hash == _hash_string(prompt)
        assert entries[0].cached_at != ""

    def test_manifest_accumulates(self):
        """Multiple saves append multiple entries."""
        prompts = ["prompt one", "prompt two", "prompt three"]
        for p in prompts:
            acts = torch.randn(1, 3, 32)
            mask = torch.ones(1, 3)
            save_prompt_activations(TEST_MODEL, p, [0], acts, mask)

        entries = read_manifest(TEST_MODEL)
        assert len(entries) == 3
        assert [e.prompt for e in entries] == prompts

    def test_manifest_duplicate_entries_deduplicated_by_list(self):
        """Re-saving the same prompt creates duplicate raw entries but
        list_cached_prompts deduplicates them."""
        prompt = "duplicate me"
        for _ in range(3):
            acts = torch.randn(1, 4, 32)
            mask = torch.ones(1, 4)
            save_prompt_activations(TEST_MODEL, prompt, [0], acts, mask)

        raw = read_manifest(TEST_MODEL)
        assert len(raw) == 3  # Raw has all entries

        deduped = list_cached_prompts(TEST_MODEL)
        assert len(deduped) == 1
        assert deduped[0].prompt == prompt

    def test_manifest_empty_when_no_saves(self):
        """read_manifest returns empty list for unknown model."""
        entries = read_manifest("nonexistent/model")
        assert entries == []

    def test_list_cached_prompts_verify(self, cache_dir):
        """verify=True filters out stale entries."""
        prompt = "will be deleted"
        acts = torch.randn(1, 3, 32)
        mask = torch.ones(1, 3)
        save_prompt_activations(TEST_MODEL, prompt, [0], acts, mask)

        # Confirm entry exists
        assert len(list_cached_prompts(TEST_MODEL, verify=True)) == 1

        # Delete the safetensors file directly
        model_hash = _hash_string(TEST_MODEL)
        prompt_hash = _hash_string(prompt)
        sf_path = cache_dir / model_hash / f"{prompt_hash}.safetensors"
        sf_path.unlink()

        # Without verify, entry still shows
        assert len(list_cached_prompts(TEST_MODEL, verify=False)) == 1
        # With verify, entry is filtered out
        assert len(list_cached_prompts(TEST_MODEL, verify=True)) == 0

    def test_manifest_num_tokens_with_padding(self):
        """num_tokens reflects actual tokens (not padding)."""
        prompt = "padded prompt"
        acts = torch.randn(1, 8, 32)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])  # 5 real tokens
        save_prompt_activations(TEST_MODEL, prompt, [0], acts, mask)

        entries = read_manifest(TEST_MODEL)
        assert entries[0].num_tokens == 5

    def test_manifest_isolated_per_model(self):
        """Each model has its own manifest."""
        acts = torch.randn(1, 3, 32)
        mask = torch.ones(1, 3)
        save_prompt_activations("model-a", "prompt a", [0], acts, mask)
        save_prompt_activations("model-b", "prompt b", [0], acts, mask)

        entries_a = read_manifest("model-a")
        entries_b = read_manifest("model-b")
        assert len(entries_a) == 1
        assert len(entries_b) == 1
        assert entries_a[0].prompt == "prompt a"
        assert entries_b[0].prompt == "prompt b"

    def test_manifest_entry_is_dataclass(self):
        """ManifestEntry fields are accessible."""
        acts = torch.randn(1, 3, 32)
        mask = torch.ones(1, 3)
        save_prompt_activations(TEST_MODEL, "test", [0], acts, mask)

        entry = read_manifest(TEST_MODEL)[0]
        assert isinstance(entry, ManifestEntry)
        assert hasattr(entry, "hash")
        assert hasattr(entry, "prompt")
        assert hasattr(entry, "num_tokens")
        assert hasattr(entry, "cached_at")


class TestManifestBackwardCompat:
    """Backward compatibility: pre-feature caches have no _manifest.jsonl."""

    @pytest.fixture(autouse=True)
    def cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        return tmp_path

    def test_pre_feature_cache_has_no_manifest(self, cache_dir):
        """Simulate a pre-feature cache by writing a safetensors file
        without going through save_prompt_activations's manifest hook."""
        from lmprobe.cache import _merge_save_backend, _prepare_tensor, _register_model

        model_hash = _hash_string(TEST_MODEL)
        prompt_hash = _hash_string("old prompt")
        key = f"{model_hash}/{prompt_hash}.safetensors"

        # Write model name but NOT through save_prompt_activations
        _register_model(TEST_MODEL)

        # Write activation file directly (simulating old lmprobe version)
        tensors = {"layer_0": _prepare_tensor(torch.randn(1, 5, 32))}
        _merge_save_backend(key, tensors)

        # Manifest should be empty (not error)
        entries = read_manifest(TEST_MODEL)
        assert entries == []

        # list_cached_prompts should also be empty
        assert list_cached_prompts(TEST_MODEL) == []

        # discover_cached should still work (it doesn't depend on manifest)
        from lmprobe.cache import discover_cached

        info = discover_cached(TEST_MODEL, "old prompt")
        assert info is not None
        assert 0 in info.raw_layers

    def test_manifest_corruption_handled_gracefully(self, cache_dir):
        """Corrupt _manifest.jsonl doesn't crash read_manifest."""
        from lmprobe.cache import get_backend

        backend = get_backend()
        key = _manifest_key(TEST_MODEL)

        # Write garbage
        backend.write_text(key, "not valid json\n{\"hash\": \"abc\"}\n")

        # Should skip bad lines, skip incomplete entries
        entries = read_manifest(TEST_MODEL)
        assert entries == []  # Both lines are invalid (missing required fields or bad JSON)

    def test_manifest_with_mixed_valid_invalid_lines(self, cache_dir):
        """read_manifest skips invalid lines and keeps valid ones."""
        from lmprobe.cache import get_backend

        backend = get_backend()
        key = _manifest_key(TEST_MODEL)

        valid_line = json.dumps({
            "hash": "abcdef1234567890",
            "prompt": "valid prompt",
            "num_tokens": 5,
            "cached_at": "2026-03-24T12:00:00+00:00",
        })
        content = f"corrupt line\n{valid_line}\nalso bad\n"
        backend.write_text(key, content)

        entries = read_manifest(TEST_MODEL)
        assert len(entries) == 1
        assert entries[0].prompt == "valid prompt"

    def test_save_still_works_if_manifest_write_fails(self, cache_dir, monkeypatch):
        """Even if manifest write fails, the activation cache write succeeds."""
        from lmprobe.cache import get_backend

        backend = get_backend()
        orig_write_text = backend.write_text
        manifest_key = _manifest_key(TEST_MODEL)

        def broken_write_text(key, text):
            if key == manifest_key:
                raise OSError("Simulated disk error")
            return orig_write_text(key, text)

        monkeypatch.setattr(backend, "write_text", broken_write_text)

        prompt = "should still cache"
        acts = torch.randn(1, 3, 32)
        mask = torch.ones(1, 3)

        # This should NOT raise even though manifest append fails
        save_prompt_activations(TEST_MODEL, prompt, [0], acts, mask)

        # The activation file should still exist
        from lmprobe.cache import discover_cached

        info = discover_cached(TEST_MODEL, prompt)
        assert info is not None
        assert 0 in info.raw_layers


# =============================================================================
# Additional coverage tests (targeting 70%+ coverage)
# =============================================================================


class TestHashString:
    """Tests for _hash_string pure function."""

    def test_deterministic(self):
        assert _hash_string("hello") == _hash_string("hello")

    def test_different_inputs_different_hashes(self):
        assert _hash_string("hello") != _hash_string("world")

    def test_default_length(self):
        result = _hash_string("test")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_custom_length(self):
        result = _hash_string("test", length=8)
        assert len(result) == 8

    def test_empty_string(self):
        result = _hash_string("")
        assert len(result) == 16


class TestFormatLayers:
    """Tests for _format_layers helper."""

    def test_small_set(self):
        from lmprobe.cache import _format_layers

        result = _format_layers({0, 1, 2})
        assert result == "[0, 1, 2]"

    def test_large_set_truncated(self):
        from lmprobe.cache import _format_layers

        layers = set(range(20))
        result = _format_layers(layers, max_show=10)
        assert "20 total" in result
        assert "..." in result

    def test_list_input(self):
        from lmprobe.cache import _format_layers

        result = _format_layers([5, 3, 1])
        assert result == "[1, 3, 5]"


class TestParseKeyHelpers:
    """Tests for key parsing helpers."""

    def test_parse_raw_layer_keys(self):
        from lmprobe.cache import _parse_raw_layer_keys

        keys = {"layer_0", "layer_5", "attention_mask", "pooled_last_token_layer_0"}
        assert _parse_raw_layer_keys(keys) == {0, 5}

    def test_parse_raw_layer_keys_empty(self):
        from lmprobe.cache import _parse_raw_layer_keys

        assert _parse_raw_layer_keys(set()) == set()

    def test_parse_raw_layer_keys_invalid(self):
        from lmprobe.cache import _parse_raw_layer_keys

        keys = {"layer_abc", "layer_"}
        assert _parse_raw_layer_keys(keys) == set()

    def test_parse_pooled_layer_keys(self):
        from lmprobe.cache import _parse_pooled_layer_keys

        keys = {"pooled_last_token_layer_0", "pooled_last_token_layer_3", "layer_0"}
        assert _parse_pooled_layer_keys(keys, "last_token") == {0, 3}

    def test_parse_pooled_layer_keys_wrong_strategy(self):
        from lmprobe.cache import _parse_pooled_layer_keys

        keys = {"pooled_last_token_layer_0"}
        assert _parse_pooled_layer_keys(keys, "mean") == set()

    def test_parse_all_pooled_keys(self):
        from lmprobe.cache import _parse_all_pooled_keys

        keys = [
            "pooled_last_token_layer_0",
            "pooled_last_token_layer_1",
            "pooled_mean_layer_0",
            "layer_0",
        ]
        result = _parse_all_pooled_keys(keys)
        assert "last_token" in result
        assert result["last_token"] == [0, 1]
        assert "mean" in result
        assert result["mean"] == [0]


class TestPrepareTensor:
    """Tests for _prepare_tensor."""

    def test_detaches_and_makes_contiguous(self):
        from lmprobe.cache import _prepare_tensor

        t = torch.randn(3, 4, requires_grad=True)
        result = _prepare_tensor(t)
        assert not result.requires_grad
        assert result.is_contiguous()
        assert result.device.type == "cpu"

    def test_dtype_conversion(self):
        import lmprobe.cache as cache_mod
        from lmprobe.cache import _prepare_tensor

        old = cache_mod._CACHE_DTYPE
        try:
            cache_mod._CACHE_DTYPE = torch.float16
            t = torch.randn(3, 4, dtype=torch.float32)
            result = _prepare_tensor(t)
            assert result.dtype == torch.float16
        finally:
            cache_mod._CACHE_DTYPE = old

    def test_no_conversion_for_int(self):
        import lmprobe.cache as cache_mod
        from lmprobe.cache import _prepare_tensor

        old = cache_mod._CACHE_DTYPE
        try:
            cache_mod._CACHE_DTYPE = torch.float16
            t = torch.ones(3, dtype=torch.long)
            result = _prepare_tensor(t)
            assert result.dtype == torch.long  # int tensors should not be converted
        finally:
            cache_mod._CACHE_DTYPE = old


class TestSetCacheDtypeEdgeCases:
    """Edge cases for set_cache_dtype."""

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            set_cache_dtype("float8")

    def test_bfloat16(self):
        import lmprobe.cache as cache_mod

        set_cache_dtype("bfloat16")
        try:
            assert cache_mod._CACHE_DTYPE == torch.bfloat16
        finally:
            set_cache_dtype(None)

    def test_float32(self):
        import lmprobe.cache as cache_mod

        set_cache_dtype("float32")
        try:
            assert cache_mod._CACHE_DTYPE == torch.float32
        finally:
            set_cache_dtype(None)

    def test_none_resets(self):
        import lmprobe.cache as cache_mod

        set_cache_dtype("float16")
        set_cache_dtype(None)
        assert cache_mod._CACHE_DTYPE is None


class TestSetCacheLimitEdgeCases:
    """Edge cases for set_cache_limit."""

    def test_positive_value(self):
        import lmprobe.cache as cache_mod

        old = cache_mod._CACHE_MAX_BYTES
        try:
            set_cache_limit(gb=10.0)
            assert cache_mod._CACHE_MAX_BYTES == int(10.0 * 1024**3)
        finally:
            cache_mod._CACHE_MAX_BYTES = old

    def test_none_resets(self):
        import lmprobe.cache as cache_mod

        old = cache_mod._CACHE_MAX_BYTES
        try:
            set_cache_limit(gb=5.0)
            set_cache_limit(None)
            assert cache_mod._CACHE_MAX_BYTES is None
        finally:
            cache_mod._CACHE_MAX_BYTES = old


class TestEnableCacheLogging:
    """Tests for enable_cache_logging."""

    def test_sets_level(self):
        import logging

        from lmprobe.cache import enable_cache_logging, logger

        original_level = logger.level
        original_handlers = logger.handlers[:]
        try:
            enable_cache_logging(logging.DEBUG)
            assert logger.level == logging.DEBUG
            assert len(logger.handlers) >= 1
        finally:
            logger.setLevel(original_level)
            logger.handlers = original_handlers

    def test_does_not_add_duplicate_handlers(self):
        import logging

        from lmprobe.cache import enable_cache_logging, logger

        original_handlers = logger.handlers[:]
        try:
            enable_cache_logging(logging.INFO)
            count_after_first = len(logger.handlers)
            enable_cache_logging(logging.INFO)
            count_after_second = len(logger.handlers)
            assert count_after_second == count_after_first
        finally:
            logger.handlers = original_handlers


class TestSetCacheBackend:
    """Tests for set_cache_backend."""

    def test_set_none_resets(self):
        import lmprobe.cache as cache_mod
        from lmprobe.cache import set_cache_backend

        set_cache_backend(None)
        assert cache_mod._backend is None

    def test_set_string_local_path(self, tmp_path):
        import lmprobe.cache as cache_mod
        from lmprobe.cache import set_cache_backend

        old = cache_mod._backend
        try:
            set_cache_backend(str(tmp_path))
            assert isinstance(cache_mod._backend, LocalCacheBackend)
        finally:
            cache_mod._backend = old

    def test_set_backend_instance(self, tmp_path):
        import lmprobe.cache as cache_mod
        from lmprobe.cache import set_cache_backend

        old = cache_mod._backend
        try:
            backend = LocalCacheBackend(tmp_path)
            set_cache_backend(backend)
            assert cache_mod._backend is backend
        finally:
            cache_mod._backend = old

    def test_invalid_type_raises(self):
        from lmprobe.cache import set_cache_backend

        with pytest.raises(TypeError, match="Expected CacheBackend"):
            set_cache_backend(42)

    def test_unsupported_uri_raises(self):
        from lmprobe.cache import _parse_backend_uri

        with pytest.raises(ValueError, match="Unsupported"):
            _parse_backend_uri("gs://bucket/prefix")


class TestDiscoverCached:
    """Tests for discover_cached introspection."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-discover"

    def test_returns_none_for_uncached(self):
        from lmprobe.cache import discover_cached

        assert discover_cached(self.model, "never cached") is None

    def test_returns_raw_layers(self):
        from lmprobe.cache import discover_cached

        acts = torch.randn(1, 5, 64)
        mask = torch.ones(1, 5, dtype=torch.long)
        save_prompt_activations(self.model, "test prompt", [0, 1], acts, mask)

        info = discover_cached(self.model, "test prompt")
        assert info is not None
        assert info.raw_layers == [0, 1]
        assert info.num_tokens == 5

    def test_returns_pooled_info(self):
        from lmprobe.cache import discover_cached

        acts = torch.randn(1, 5, 64)
        mask = torch.ones(1, 5, dtype=torch.long)
        save_prompt_activations(self.model, "test", [0, 1], acts, mask)
        save_prompt_pooled_activations(self.model, "test", [0, 1], torch.randn(1, 64), "last_token")

        info = discover_cached(self.model, "test")
        assert info is not None
        assert "last_token" in info.pooled
        assert info.pooled["last_token"] == [0, 1]

    def test_detects_perplexity(self):
        from lmprobe.cache import discover_cached

        save_prompt_perplexity(self.model, "ppl test", torch.tensor([1.0, 2.0, 3.0]))

        info = discover_cached(self.model, "ppl test")
        assert info is not None
        assert info.has_perplexity is True

    def test_detects_logits(self):
        from lmprobe.cache import discover_cached

        logits = torch.randn(1, 5, 100)
        mask = torch.ones(1, 5)
        save_prompt_logits(self.model, "logits test", logits, mask, top_k=10)

        info = discover_cached(self.model, "logits test")
        assert info is not None
        assert info.logits_top_k == 10

    def test_detects_token_perplexity(self):
        from lmprobe.cache import discover_cached

        save_prompt_perplexity(
            self.model, "tok ppl",
            torch.tensor([1.0, 2.0, 3.0]),
            token_perplexity=torch.tensor([0.5, 0.6, 0.7]),
            token_ids=torch.tensor([1, 2, 3, 4]),
        )

        info = discover_cached(self.model, "tok ppl")
        assert info is not None
        assert info.has_token_perplexity is True


class TestSelectPositions:
    """Tests for _select_positions helper."""

    def test_last_position(self):
        from lmprobe.cache import _select_positions

        tensor = torch.randn(1, 5, 10)
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        result = _select_positions(tensor, mask, "last")
        assert result.shape == (1, 1, 10)
        assert torch.equal(result[0, 0], tensor[0, 2])

    def test_all_positions(self):
        from lmprobe.cache import _select_positions

        tensor = torch.randn(1, 5, 10)
        mask = torch.ones(1, 5)
        result = _select_positions(tensor, mask, "all")
        assert torch.equal(result, tensor)

    def test_invalid_positions_raises(self):
        from lmprobe.cache import _select_positions

        with pytest.raises(ValueError, match="Invalid positions"):
            _select_positions(torch.randn(1, 5, 10), torch.ones(1, 5), "middle")


class TestGetPooledCacheKey:
    """Tests for get_pooled_cache_key."""

    def test_returns_correct_prefix(self):
        from lmprobe.cache import get_pooled_cache_key

        assert get_pooled_cache_key("last_token") == "pooled_last_token"
        assert get_pooled_cache_key("mean") == "pooled_mean"


class TestGetPromptCachedRawLayers:
    """Tests for get_prompt_cached_raw_layers."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-raw-layers"

    def test_returns_none_for_uncached(self):
        from lmprobe.cache import get_prompt_cached_raw_layers

        assert get_prompt_cached_raw_layers(self.model, "uncached") is None

    def test_returns_layers_for_cached(self):
        from lmprobe.cache import get_prompt_cached_raw_layers

        acts = torch.randn(1, 5, 64)
        mask = torch.ones(1, 5, dtype=torch.long)
        save_prompt_activations(self.model, "test", [0, 1], acts, mask)

        result = get_prompt_cached_raw_layers(self.model, "test")
        assert result == {0, 1}


class TestGetPromptCachedPooledLayers:
    """Tests for get_prompt_cached_pooled_layers."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-pooled-layers"

    def test_returns_none_for_uncached(self):
        from lmprobe.cache import get_prompt_cached_pooled_layers

        assert get_prompt_cached_pooled_layers(self.model, "uncached", "last_token") is None

    def test_returns_layers_for_cached(self):
        from lmprobe.cache import get_prompt_cached_pooled_layers

        save_prompt_pooled_activations(self.model, "test", [0, 2], torch.randn(1, 64), "last_token")
        result = get_prompt_cached_pooled_layers(self.model, "test", "last_token")
        assert result == {0, 2}


class TestTokenPerplexityCache:
    """Tests for token-level perplexity cache functions."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-tok-ppl"

    def test_save_and_load_token_perplexity(self):
        from lmprobe.cache import (
            is_prompt_token_perplexity_cached,
            load_prompt_token_perplexity,
        )

        tok_ppl = torch.tensor([0.5, 0.6, 0.7])
        tok_ids = torch.tensor([100, 200, 300, 400])
        ppl_features = torch.tensor([1.0, 2.0, 3.0])

        save_prompt_perplexity(
            self.model, "test",
            ppl_features,
            token_perplexity=tok_ppl,
            token_ids=tok_ids,
        )

        assert is_prompt_token_perplexity_cached(self.model, "test")

        loaded_ppl, loaded_ids = load_prompt_token_perplexity(self.model, "test")
        assert torch.allclose(loaded_ppl, tok_ppl)
        assert torch.equal(loaded_ids, tok_ids)

    def test_not_cached_raises(self):
        from lmprobe.cache import load_prompt_token_perplexity

        with pytest.raises(FileNotFoundError):
            load_prompt_token_perplexity(self.model, "uncached")

    def test_not_cached_returns_false(self):
        from lmprobe.cache import is_prompt_token_perplexity_cached

        assert not is_prompt_token_perplexity_cached(self.model, "uncached")


class TestLoadPooledBatch:
    """Tests for load_pooled_batch function."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-pooled-batch"

    def test_basic_load(self):
        from lmprobe.cache import load_pooled_batch

        prompts = ["prompt A", "prompt B", "prompt C"]
        for p in prompts:
            save_prompt_pooled_activations(
                self.model, p, [0, 1], torch.randn(1, 64), "last_token"
            )

        result = load_pooled_batch(self.model, prompts, [0, 1], "last_token")
        assert result.shape == (3, 64)

    def test_missing_prompt_raises(self):
        from lmprobe.cache import load_pooled_batch

        save_prompt_pooled_activations(
            self.model, "exists", [0], torch.randn(1, 32), "last_token"
        )

        with pytest.raises(FileNotFoundError):
            load_pooled_batch(
                self.model, ["exists", "missing"], [0], "last_token"
            )

    def test_missing_raises_without_fallback(self):
        from lmprobe.cache import load_pooled_batch

        with pytest.raises(FileNotFoundError):
            load_pooled_batch(
                self.model, ["missing1", "missing2"], [0], "last_token",
                fallback_to_raw=False,
            )


class TestWriteAndReadShardRegistry:
    """Tests for write_shard_registry and related shard functions."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-shard"

    def test_write_and_read_manifest(self):
        from lmprobe.cache import _load_shard_manifest, write_shard_registry

        manifest = {"model": self.model, "tensors": {"hidden_layers": {"layers": [0, 1]}}}
        index = {"abc123": {"shard_index": 0, "row_offset": 0}}

        write_shard_registry(self.model, manifest, index)

        loaded = _load_shard_manifest(self.model)
        assert loaded is not None
        assert loaded["model"] == self.model

    def test_write_and_read_index(self):
        from lmprobe.cache import _load_shard_index, write_shard_registry

        manifest = {"model": self.model}
        index = {"hash1": {"shard_index": 0, "row_offset": 10}}

        write_shard_registry(self.model, manifest, index)

        loaded = _load_shard_index(self.model)
        assert loaded is not None
        assert "hash1" in loaded
        assert loaded["hash1"]["row_offset"] == 10

    def test_repo_id_stamped(self):
        from lmprobe.cache import _load_shard_index, write_shard_registry

        manifest = {"model": self.model}
        index = {"hash1": {"shard_index": 0}}

        write_shard_registry(self.model, manifest, index, repo_id="user/dataset")

        loaded = _load_shard_index(self.model)
        assert loaded["hash1"]["repo_id"] == "user/dataset"

    def test_index_merge(self):
        from lmprobe.cache import _load_shard_index, write_shard_registry

        write_shard_registry(self.model, {"m": 1}, {"h1": {"s": 0}})
        write_shard_registry(self.model, {"m": 2}, {"h2": {"s": 1}})

        loaded = _load_shard_index(self.model)
        assert "h1" in loaded
        assert "h2" in loaded

    def test_lookup_shard(self):
        from lmprobe.cache import _lookup_shard, write_shard_registry

        prompt = "test prompt for shard"
        prompt_hash = _hash_string(prompt)
        manifest = {"model": self.model}
        index = {prompt_hash: {"shard_index": 2, "row_offset": 5}}

        write_shard_registry(self.model, manifest, index)

        result = _lookup_shard(self.model, prompt)
        assert result is not None
        assert result["shard_index"] == 2
        assert result["row_offset"] == 5

    def test_lookup_shard_missing(self):
        from lmprobe.cache import _lookup_shard

        assert _lookup_shard(self.model, "never registered") is None


class TestEvictFunction:
    """Tests for evict() LRU eviction."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-evict"

    def test_noop_when_no_limit(self):
        import lmprobe.cache as cache_mod
        from lmprobe.cache import evict

        old = cache_mod._CACHE_MAX_BYTES
        try:
            cache_mod._CACHE_MAX_BYTES = None
            acts = torch.randn(1, 5, 32)
            save_prompt_activations(self.model, "test", [0], acts, torch.ones(1, 5))
            evict()  # Should not raise or delete anything
            assert is_prompt_fully_cached(self.model, "test", {0})
        finally:
            cache_mod._CACHE_MAX_BYTES = old

    def test_noop_when_disabled(self):
        import lmprobe.cache as cache_mod
        from lmprobe.cache import evict

        old = cache_mod._CACHE_MAX_BYTES
        try:
            cache_mod._CACHE_MAX_BYTES = -1
            evict()  # Should not raise
        finally:
            cache_mod._CACHE_MAX_BYTES = old


class TestCollectCacheEntries:
    """Tests for _collect_cache_entries."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-collect"

    def test_empty_cache(self):
        from lmprobe.cache import _collect_cache_entries

        entries = _collect_cache_entries()
        assert entries == []

    def test_finds_safetensors_files(self):
        from lmprobe.cache import _collect_cache_entries

        save_prompt_activations(self.model, "p1", [0], torch.randn(1, 3, 32), torch.ones(1, 3))
        save_prompt_activations(self.model, "p2", [0], torch.randn(1, 3, 32), torch.ones(1, 3))

        entries = _collect_cache_entries()
        # At least 2 main safetensors files (may also have _model_name.txt etc.)
        sf_entries = [e for e in entries if str(e[0]).endswith(".safetensors")]
        assert len(sf_entries) >= 2


class TestClearCache:
    """Tests for clear_cache."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-clear"

    def test_clears_all(self):
        from lmprobe.cache import clear_cache

        save_prompt_activations(self.model, "p1", [0], torch.randn(1, 3, 32), torch.ones(1, 3))
        save_prompt_activations(self.model, "p2", [0], torch.randn(1, 3, 32), torch.ones(1, 3))

        count = clear_cache()
        assert count >= 2

        info = cache_info()
        assert info.total_size_bytes == 0

    def test_empty_cache_returns_zero(self):
        from lmprobe.cache import clear_cache

        # Ensure the cache dir exists but is empty of model dirs
        count = clear_cache()
        assert count == 0


class TestComputeCacheKey:
    """Tests for compute_cache_key legacy function."""

    def test_deterministic(self):
        from lmprobe.cache import compute_cache_key

        key1 = compute_cache_key("model", ["prompt1"], [0, 1])
        key2 = compute_cache_key("model", ["prompt1"], [0, 1])
        assert key1 == key2

    def test_different_inputs(self):
        from lmprobe.cache import compute_cache_key

        key1 = compute_cache_key("model", ["prompt1"], [0])
        key2 = compute_cache_key("model", ["prompt2"], [0])
        assert key1 != key2

    def test_length(self):
        from lmprobe.cache import compute_cache_key

        key = compute_cache_key("model", ["prompt"], [0])
        assert len(key) == 32


class TestGetCachePath:
    """Tests for get_cache_path legacy function."""

    def test_returns_path_with_pt_suffix(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        from lmprobe.cache import get_cache_path

        path = get_cache_path("abc123")
        assert str(path).endswith("abc123.pt")


class TestModelCacheInfoRepr:
    """Tests for ModelCacheInfo.__repr__."""

    def test_repr_with_name(self):
        from lmprobe.cache import ModelCacheInfo

        info = ModelCacheInfo(
            model_name="test/model",
            model_hash="abc123",
            size_bytes=1024**3,
            num_prompts=10,
            num_layers=5,
            has_pooled=True,
            has_perplexity=False,
            has_logits=True,
        )
        text = repr(info)
        assert "test/model" in text
        assert "10 prompts" in text
        assert "pooled" in text
        assert "logits" in text

    def test_repr_without_name(self):
        from lmprobe.cache import ModelCacheInfo

        info = ModelCacheInfo(
            model_name=None,
            model_hash="abc123",
            size_bytes=0,
            num_prompts=0,
            num_layers=0,
            has_pooled=False,
            has_perplexity=False,
        )
        text = repr(info)
        assert "abc123" in text

    def test_size_gb_property(self):
        from lmprobe.cache import ModelCacheInfo

        info = ModelCacheInfo(
            model_name="m", model_hash="h",
            size_bytes=2 * 1024**3,
            num_prompts=0, num_layers=0,
            has_pooled=False, has_perplexity=False,
        )
        assert abs(info.size_gb - 2.0) < 0.001


class TestCacheInfoRepr:
    """Tests for CacheInfo.__repr__ with various states."""

    def test_with_limit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        save_prompt_activations(
            "test-model", "p1", [0],
            torch.randn(1, 3, 32), torch.ones(1, 3),
        )

        import lmprobe.cache as cache_mod

        old = cache_mod._CACHE_MAX_BYTES
        try:
            cache_mod._CACHE_MAX_BYTES = 10 * 1024**3
            info = cache_info()
            text = repr(info)
            assert "Size limit" in text
        finally:
            cache_mod._CACHE_MAX_BYTES = old

    def test_total_size_gb_property(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        info = cache_info()
        assert info.total_size_gb == 0.0


class TestCacheInfoMultipleModels:
    """Tests for cache_info with multiple models."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

    def test_two_models(self):
        save_prompt_activations("model-a", "p1", [0], torch.randn(1, 3, 32), torch.ones(1, 3))
        save_prompt_activations("model-b", "p1", [0], torch.randn(1, 3, 32), torch.ones(1, 3))

        info = cache_info()
        assert len(info.models) == 2

    def test_has_logits_flag(self):
        save_prompt_activations("model-logits", "p1", [0], torch.randn(1, 3, 32), torch.ones(1, 3))
        save_prompt_logits("model-logits", "p1", torch.randn(1, 3, 100), torch.ones(1, 3), top_k=5)

        info = cache_info(model="model-logits")
        assert info.models[0].has_logits is True

    def test_has_perplexity_flag(self):
        save_prompt_activations("model-ppl", "p1", [0], torch.randn(1, 3, 32), torch.ones(1, 3))
        save_prompt_perplexity("model-ppl", "p1", torch.tensor([1.0, 2.0, 3.0]))

        info = cache_info(model="model-ppl")
        assert info.models[0].has_perplexity is True


class TestHasLogitsInKeys:
    """Tests for _has_logits_in_keys helper."""

    def test_full_logits(self):
        from lmprobe.cache import _has_logits_in_keys

        keys = {"logits"}
        assert _has_logits_in_keys(keys, top_k=False) is True
        assert _has_logits_in_keys(keys, top_k=True) is False
        assert _has_logits_in_keys(keys, top_k=None) is True

    def test_topk_logits(self):
        from lmprobe.cache import _has_logits_in_keys

        keys = {"logits_top_k_values", "logits_top_k_indices"}
        assert _has_logits_in_keys(keys, top_k=True) is True
        assert _has_logits_in_keys(keys, top_k=False) is False
        assert _has_logits_in_keys(keys, top_k=None) is True

    def test_no_logits(self):
        from lmprobe.cache import _has_logits_in_keys

        keys = {"layer_0", "attention_mask"}
        assert _has_logits_in_keys(keys, top_k=None) is False


class TestPerplexityCacheRoundtrip:
    """Tests for perplexity save/load roundtrip."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-ppl-rt"

    def test_basic_roundtrip(self):
        from lmprobe.cache import load_prompt_perplexity

        ppl = torch.tensor([1.5, 2.5, 3.5])
        save_prompt_perplexity(self.model, "test", ppl)

        loaded = load_prompt_perplexity(self.model, "test")
        assert torch.allclose(loaded, ppl)

    def test_missing_raises(self):
        from lmprobe.cache import load_prompt_perplexity

        with pytest.raises(FileNotFoundError):
            load_prompt_perplexity(self.model, "missing")


class TestLogitsFullRoundtrip:
    """Tests for full logits (not top-k) save/load."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-logits-full"

    def test_full_logits_roundtrip(self):
        logits = torch.randn(1, 5, 100)
        mask = torch.ones(1, 5)
        save_prompt_logits(self.model, "test", logits, mask, top_k=None, positions="all")

        assert is_prompt_logits_cached(self.model, "test", top_k=None)
        loaded, indices = load_prompt_logits(self.model, "test", top_k=None)
        assert indices is None
        assert torch.allclose(loaded, logits, atol=1e-4)

    def test_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_prompt_logits(self.model, "missing", top_k=None)


class TestLoadPromptActivationsMissing:
    """Test error handling for load_prompt_activations with no cache."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

    def test_raises_for_uncached(self):
        with pytest.raises(FileNotFoundError, match="No cached activations"):
            load_prompt_activations("no-model", "no-prompt", [0])


class TestLoadPromptPooledActivationsMissing:
    """Test error handling for load_prompt_pooled_activations with no cache."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

    def test_raises_for_uncached(self):
        with pytest.raises(FileNotFoundError, match="No cached pooled"):
            load_prompt_pooled_activations("no-model", "no-prompt", [0], "last_token")


class TestLegacyPerplexityCache:
    """Tests for legacy batch perplexity cache functions."""

    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        from lmprobe.cache import (
            get_perplexity_cache_path,
            load_perplexity_cache,
            save_perplexity_cache,
        )

        path = get_perplexity_cache_path("model", ["p1", "p2"])
        features = torch.randn(2, 3)
        save_perplexity_cache(path, features)

        loaded = load_perplexity_cache(path)
        assert loaded is not None
        assert torch.allclose(loaded, features)

    def test_load_nonexistent_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        from lmprobe.cache import load_perplexity_cache

        result = load_perplexity_cache(tmp_path / "nonexistent.pt")
        assert result is None


class TestIsPromptPerplexityCachedEdgeCases:
    """Edge cases for is_prompt_perplexity_cached."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-ppl-check"

    def test_false_when_nothing_cached(self):
        from lmprobe.cache import is_prompt_perplexity_cached

        assert not is_prompt_perplexity_cached(self.model, "uncached")

    def test_in_main_file(self):
        """Perplexity stored in main file (pre-sidecar) is detected."""
        from lmprobe.cache import is_prompt_perplexity_cached

        main_key = _prompt_cache_key(self.model, "test")
        _merge_save_backend(main_key, {"perplexity": torch.tensor([1.0, 2.0, 3.0])})

        assert is_prompt_perplexity_cached(self.model, "test")


class TestLoadLayerAcrossPromptsSynthetic:
    """Tests for load_layer_across_prompts without a real model."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-layer-across"

    def test_basic_load(self):
        from lmprobe.cache import load_layer_across_prompts

        prompts = ["prompt A", "prompt B"]
        for p in prompts:
            acts = torch.randn(1, 5, 64)
            mask = torch.ones(1, 5, dtype=torch.long)
            save_prompt_activations(self.model, p, [0, 1], acts, mask)

        acts_list, masks_list = load_layer_across_prompts(self.model, prompts, layer=0)
        assert len(acts_list) == 2
        assert len(masks_list) == 2
        for a, m in zip(acts_list, masks_list):
            assert a.shape == (1, 5, 32)  # half of 64 (2 layers)
            assert m.shape == (1, 5)

    def test_missing_prompt_raises(self):
        from lmprobe.cache import load_layer_across_prompts

        with pytest.raises(FileNotFoundError):
            load_layer_across_prompts(self.model, ["nonexistent"], layer=0)


class TestLoadLayerLastTokenSynthetic:
    """Tests for load_layer_last_token without a real model."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-model-last-token"

    def test_basic_load(self):
        from lmprobe.cache import load_layer_last_token

        prompts = ["test one", "test two"]
        for p in prompts:
            acts = torch.randn(1, 5, 32)
            mask = torch.ones(1, 5, dtype=torch.long)
            save_prompt_activations(self.model, p, [0], acts, mask)

        result = load_layer_last_token(self.model, prompts, layer=0)
        assert result.shape == (2, 32)

    def test_with_padding(self):
        from lmprobe.cache import load_layer_last_token

        acts = torch.randn(1, 8, 32)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
        save_prompt_activations(self.model, "padded", [0], acts, mask)

        result = load_layer_last_token(self.model, ["padded"], layer=0)
        assert result.shape == (1, 32)
        # Should use position 4 (last non-zero mask position)
        assert torch.equal(result[0], acts[0, 4, :])


class TestIsPromptLogitsCachedEdgeCases:
    """Additional edge cases for is_prompt_logits_cached."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-logits-cached"

    def test_full_logits_in_sidecar(self):
        """Full logits (not top-k) in sidecar are detected."""
        logits = torch.randn(1, 5, 100)
        mask = torch.ones(1, 5)
        save_prompt_logits(self.model, "test", logits, mask, top_k=None, positions="all")
        assert is_prompt_logits_cached(self.model, "test", top_k=None)
        assert not is_prompt_logits_cached(self.model, "test", top_k=10)

    def test_topk_logits_check(self):
        """Top-k logits check with explicit top_k parameter."""
        logits = torch.randn(1, 5, 100)
        mask = torch.ones(1, 5)
        save_prompt_logits(self.model, "topk", logits, mask, top_k=10)
        assert is_prompt_logits_cached(self.model, "topk", top_k=10)
        # Should not be cached for full logits
        assert not is_prompt_logits_cached(self.model, "topk", top_k=None)


class TestLogitsLoadEdgeCases:
    """Edge cases for load_prompt_logits."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-logits-load"

    def test_topk_missing_raises(self):
        """Loading top-k logits when none cached raises."""
        with pytest.raises(FileNotFoundError):
            load_prompt_logits(self.model, "missing", top_k=10)


class TestSafeSaveDisabled:
    """Test caching disabled behavior."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

    def test_save_noop_when_disabled(self):
        """Saving activations is a no-op when caching is disabled."""
        import lmprobe.cache as cache_mod

        old = cache_mod._CACHE_MAX_BYTES
        try:
            cache_mod._CACHE_MAX_BYTES = -1
            save_prompt_activations(
                "test", "test", [0],
                torch.randn(1, 3, 32), torch.ones(1, 3),
            )
            # Nothing should be cached
            assert not is_prompt_fully_cached("test", "test", {0})
        finally:
            cache_mod._CACHE_MAX_BYTES = old


class TestDiscoverCachedWithActivationsAndSidecars:
    """Additional discover_cached scenarios with sidecar-only entries."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.model = "test-discover-sidecar"

    def test_with_both_raw_and_logits(self):
        """discover_cached reports raw layers and logits together."""
        from lmprobe.cache import discover_cached

        acts = torch.randn(1, 5, 64)
        mask = torch.ones(1, 5, dtype=torch.long)
        save_prompt_activations(self.model, "both", [0, 1], acts, mask)
        save_prompt_logits(self.model, "both", torch.randn(1, 5, 100), mask, top_k=10)

        info = discover_cached(self.model, "both")
        assert info is not None
        assert info.raw_layers == [0, 1]
        assert info.logits_top_k == 10

    def test_logits_only_no_activations(self):
        """discover_cached handles logits-only sidecar (no main file)."""
        from lmprobe.cache import discover_cached

        mask = torch.ones(1, 5)
        save_prompt_logits(self.model, "logits-only", torch.randn(1, 5, 100), mask, top_k=5)

        info = discover_cached(self.model, "logits-only")
        assert info is not None
        assert info.raw_layers == []
        assert info.logits_top_k == 5

    def test_perplexity_only_no_activations(self):
        """discover_cached handles perplexity-only sidecar."""
        from lmprobe.cache import discover_cached

        save_prompt_perplexity(self.model, "ppl-only", torch.tensor([1.0, 2.0, 3.0]))

        info = discover_cached(self.model, "ppl-only")
        assert info is not None
        assert info.has_perplexity is True
        assert info.raw_layers == []


class TestCacheInfoV1Directories:
    """Test cache_info with v1-style cache directories."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.base = tmp_path

    def test_v1_directory_counted(self):
        """V1 directories are counted in cache_info."""
        model_hash = _hash_string("v1-model")
        model_dir = self.base / model_hash
        model_dir.mkdir(parents=True)

        # Write model name
        (model_dir / "_model_name.txt").write_text("v1-model")

        # Create a v1-style prompt directory
        prompt_hash = _hash_string("v1-prompt")
        prompt_dir = model_dir / prompt_hash
        prompt_dir.mkdir()
        torch.save(torch.randn(1, 5, 32), prompt_dir / "layer_0.pt")
        torch.save(torch.ones(1, 5), prompt_dir / "attention_mask.pt")

        info = cache_info(model="v1-model")
        assert len(info.models) == 1
        assert info.models[0].num_prompts == 1
        assert info.models[0].num_layers == 1

    def test_v1_pooled_detected(self):
        """V1 pooled directories are detected by cache_info."""
        model_hash = _hash_string("v1-pooled")
        model_dir = self.base / model_hash
        model_dir.mkdir(parents=True)
        (model_dir / "_model_name.txt").write_text("v1-pooled")

        prompt_hash = _hash_string("v1-prompt")
        prompt_dir = model_dir / prompt_hash
        prompt_dir.mkdir()
        torch.save(torch.randn(1, 5, 32), prompt_dir / "layer_0.pt")

        pooled_dir = prompt_dir / "pooled_last_token"
        pooled_dir.mkdir()
        torch.save(torch.randn(1, 32), pooled_dir / "layer_0.pt")

        info = cache_info(model="v1-pooled")
        assert info.models[0].has_pooled is True

    def test_v1_perplexity_detected(self):
        """V1 perplexity.pt is detected by cache_info."""
        model_hash = _hash_string("v1-ppl")
        model_dir = self.base / model_hash
        model_dir.mkdir(parents=True)
        (model_dir / "_model_name.txt").write_text("v1-ppl")

        prompt_hash = _hash_string("v1-prompt")
        prompt_dir = model_dir / prompt_hash
        prompt_dir.mkdir()
        torch.save(torch.randn(1, 5, 32), prompt_dir / "layer_0.pt")
        torch.save(torch.tensor([1.0, 2.0, 3.0]), prompt_dir / "perplexity.pt")

        info = cache_info(model="v1-ppl")
        assert info.models[0].has_perplexity is True


class TestIsPromptFullyCachedV1:
    """Test is_prompt_fully_cached with v1 legacy directories."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.base = tmp_path
        self.model = "v1-model-fully"

    def test_v1_fully_cached(self):
        """V1 directory with all layers and mask is fully cached."""
        model_hash = _hash_string(self.model)
        prompt_hash = _hash_string("v1 prompt")
        prompt_dir = self.base / model_hash / prompt_hash
        prompt_dir.mkdir(parents=True)

        torch.save(torch.randn(1, 5, 32), prompt_dir / "layer_0.pt")
        torch.save(torch.randn(1, 5, 32), prompt_dir / "layer_1.pt")
        torch.save(torch.ones(1, 5), prompt_dir / "attention_mask.pt")

        assert is_prompt_fully_cached(self.model, "v1 prompt", {0, 1})

    def test_v1_partial_cached(self):
        """V1 directory with missing layers is not fully cached."""
        model_hash = _hash_string(self.model)
        prompt_hash = _hash_string("v1 partial")
        prompt_dir = self.base / model_hash / prompt_hash
        prompt_dir.mkdir(parents=True)

        torch.save(torch.randn(1, 5, 32), prompt_dir / "layer_0.pt")
        torch.save(torch.ones(1, 5), prompt_dir / "attention_mask.pt")

        assert not is_prompt_fully_cached(self.model, "v1 partial", {0, 1})


class TestIsPromptPooledCachedV1:
    """Test is_prompt_pooled_cached with v1 directories."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.base = tmp_path
        self.model = "v1-model-pooled"

    def test_v1_pooled_cached(self):
        model_hash = _hash_string(self.model)
        prompt_hash = _hash_string("v1 pooled prompt")
        prompt_dir = self.base / model_hash / prompt_hash
        pooled_dir = prompt_dir / "pooled_last_token"
        pooled_dir.mkdir(parents=True)

        torch.save(torch.randn(1, 32), pooled_dir / "layer_0.pt")
        torch.save(torch.randn(1, 32), pooled_dir / "layer_1.pt")

        assert is_prompt_pooled_cached(self.model, "v1 pooled prompt", {0, 1}, "last_token")


class TestLoadPromptActivationsV1:
    """Test load_prompt_activations with v1 directories."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.base = tmp_path
        self.model = "v1-model-load"

    def test_v1_load(self):
        model_hash = _hash_string(self.model)
        prompt_hash = _hash_string("v1 load test")
        prompt_dir = self.base / model_hash / prompt_hash
        prompt_dir.mkdir(parents=True)

        acts0 = torch.randn(1, 5, 32)
        mask = torch.ones(1, 5, dtype=torch.long)
        torch.save(acts0, prompt_dir / "layer_0.pt")
        torch.save(mask, prompt_dir / "attention_mask.pt")

        loaded_acts, loaded_mask = load_prompt_activations(self.model, "v1 load test", [0])
        assert torch.equal(loaded_acts, acts0)
        assert torch.equal(loaded_mask, mask)


class TestLoadPromptPooledV1:
    """Test load_prompt_pooled_activations with v1 directories."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.base = tmp_path
        self.model = "v1-model-pooled-load"

    def test_v1_pooled_load(self):
        model_hash = _hash_string(self.model)
        prompt_hash = _hash_string("v1 pooled load")
        prompt_dir = self.base / model_hash / prompt_hash
        pooled_dir = prompt_dir / "pooled_last_token"
        pooled_dir.mkdir(parents=True)

        layer0 = torch.randn(1, 32)
        layer1 = torch.randn(1, 32)
        torch.save(layer0, pooled_dir / "layer_0.pt")
        torch.save(layer1, pooled_dir / "layer_1.pt")

        loaded = load_prompt_pooled_activations(self.model, "v1 pooled load", [0, 1], "last_token")
        expected = torch.cat([layer0, layer1], dim=-1)
        assert torch.equal(loaded, expected)


class TestLoadPerplexityV1:
    """Test load_prompt_perplexity with v1 directories."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.base = tmp_path
        self.model = "v1-model-ppl-load"

    def test_v1_perplexity_load(self):
        model_hash = _hash_string(self.model)
        prompt_hash = _hash_string("v1 ppl load")
        prompt_dir = self.base / model_hash / prompt_hash
        prompt_dir.mkdir(parents=True)

        ppl = torch.tensor([1.5, 2.0, 0.8])
        torch.save(ppl, prompt_dir / "perplexity.pt")

        loaded = load_prompt_perplexity(self.model, "v1 ppl load")
        assert torch.allclose(loaded, ppl)

    def test_v1_perplexity_is_cached(self):
        model_hash = _hash_string(self.model)
        prompt_hash = _hash_string("v1 ppl check")
        prompt_dir = self.base / model_hash / prompt_hash
        prompt_dir.mkdir(parents=True)

        torch.save(torch.tensor([1.0, 2.0, 3.0]), prompt_dir / "perplexity.pt")

        assert is_prompt_perplexity_cached(self.model, "v1 ppl check")


class TestLoadLayerAcrossPromptsV1:
    """Test load_layer_across_prompts with v1 directories."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.base = tmp_path
        self.model = "v1-model-layer-across"

    def test_v1_layer_across(self):
        from lmprobe.cache import load_layer_across_prompts

        prompts = ["v1 layer A", "v1 layer B"]
        for p in prompts:
            model_hash = _hash_string(self.model)
            prompt_hash = _hash_string(p)
            prompt_dir = self.base / model_hash / prompt_hash
            prompt_dir.mkdir(parents=True)

            acts = torch.randn(1, 5, 32)
            mask = torch.ones(1, 5, dtype=torch.long)
            torch.save(acts, prompt_dir / "layer_0.pt")
            torch.save(mask, prompt_dir / "attention_mask.pt")

        acts_list, masks_list = load_layer_across_prompts(self.model, prompts, layer=0)
        assert len(acts_list) == 2
        assert len(masks_list) == 2


class TestDiscoverCachedV1:
    """Test discover_cached with v1 legacy directories."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.base = tmp_path
        self.model = "v1-model-discover"

    def test_v1_discover(self):
        from lmprobe.cache import discover_cached

        model_hash = _hash_string(self.model)
        prompt_hash = _hash_string("v1 discover")
        prompt_dir = self.base / model_hash / prompt_hash
        prompt_dir.mkdir(parents=True)

        torch.save(torch.randn(1, 5, 32), prompt_dir / "layer_0.pt")
        torch.save(torch.randn(1, 5, 32), prompt_dir / "layer_1.pt")
        mask = torch.ones(1, 5, dtype=torch.long)
        torch.save(mask, prompt_dir / "attention_mask.pt")

        info = discover_cached(self.model, "v1 discover")
        assert info is not None
        assert info.raw_layers == [0, 1]
        assert info.num_tokens == 5

    def test_v1_discover_with_perplexity(self):
        from lmprobe.cache import discover_cached

        model_hash = _hash_string(self.model)
        prompt_hash = _hash_string("v1 discover ppl")
        prompt_dir = self.base / model_hash / prompt_hash
        prompt_dir.mkdir(parents=True)

        torch.save(torch.randn(1, 5, 32), prompt_dir / "layer_0.pt")
        torch.save(torch.tensor([1.0, 2.0, 3.0]), prompt_dir / "perplexity.pt")
        torch.save(torch.ones(1, 5, dtype=torch.long), prompt_dir / "attention_mask.pt")

        info = discover_cached(self.model, "v1 discover ppl")
        assert info is not None
        assert info.has_perplexity is True

    def test_v1_discover_with_pooled(self):
        from lmprobe.cache import discover_cached

        model_hash = _hash_string(self.model)
        prompt_hash = _hash_string("v1 discover pooled")
        prompt_dir = self.base / model_hash / prompt_hash
        prompt_dir.mkdir(parents=True)

        torch.save(torch.randn(1, 5, 32), prompt_dir / "layer_0.pt")
        torch.save(torch.ones(1, 5, dtype=torch.long), prompt_dir / "attention_mask.pt")

        pooled_dir = prompt_dir / "pooled_last_token"
        pooled_dir.mkdir()
        torch.save(torch.randn(1, 32), pooled_dir / "layer_0.pt")

        info = discover_cached(self.model, "v1 discover pooled")
        assert info is not None
        assert "last_token" in info.pooled
        assert info.pooled["last_token"] == [0]


class TestGetPromptCachedLayersV1:
    """Test get_prompt_cached_layers with v1 directories."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.base = tmp_path

    def test_v1_cached_layers(self):
        prompt_dir = self.base / "somehash" / "prompthash"
        prompt_dir.mkdir(parents=True)
        torch.save(torch.randn(1, 5, 32), prompt_dir / "layer_0.pt")
        torch.save(torch.randn(1, 5, 32), prompt_dir / "layer_3.pt")

        result = get_prompt_cached_layers(prompt_dir)
        assert result == {0, 3}

    def test_v1_empty_dir(self):
        prompt_dir = self.base / "somehash" / "emptydir"
        prompt_dir.mkdir(parents=True)

        result = get_prompt_cached_layers(prompt_dir)
        assert result == set()

    def test_v1_nonexistent(self):
        result = get_prompt_cached_layers(self.base / "nonexistent")
        assert result == set()


class TestUpdateTensorFlags:
    """Tests for _update_tensor_flags helper."""

    def test_raw_layers(self):
        from lmprobe.cache import _update_tensor_flags

        layers: set[int] = set()
        flags = {"has_pooled": False, "has_perplexity": False, "has_logits": False}
        _update_tensor_flags({"layer_0", "layer_5"}, layers, flags)
        assert layers == {0, 5}
        assert not flags["has_pooled"]

    def test_pooled_detected(self):
        from lmprobe.cache import _update_tensor_flags

        layers: set[int] = set()
        flags = {"has_pooled": False, "has_perplexity": False, "has_logits": False}
        _update_tensor_flags({"pooled_last_token_layer_0"}, layers, flags)
        assert flags["has_pooled"]

    def test_perplexity_detected(self):
        from lmprobe.cache import _update_tensor_flags

        layers: set[int] = set()
        flags = {"has_pooled": False, "has_perplexity": False, "has_logits": False}
        _update_tensor_flags({"perplexity"}, layers, flags)
        assert flags["has_perplexity"]

    def test_logits_detected(self):
        from lmprobe.cache import _update_tensor_flags

        layers: set[int] = set()
        flags = {"has_pooled": False, "has_perplexity": False, "has_logits": False}
        _update_tensor_flags({"logits"}, layers, flags)
        assert flags["has_logits"]

    def test_topk_logits_detected(self):
        from lmprobe.cache import _update_tensor_flags

        layers: set[int] = set()
        flags = {"has_pooled": False, "has_perplexity": False, "has_logits": False}
        _update_tensor_flags({"logits_top_k_values"}, layers, flags)
        assert flags["has_logits"]


class TestUpdateMtime:
    """Tests for _update_mtime helper."""

    def test_first_call(self):
        from lmprobe.cache import _update_mtime

        oldest, newest = _update_mtime(100.0, None, None)
        assert oldest == 100.0
        assert newest == 100.0

    def test_updates_oldest(self):
        from lmprobe.cache import _update_mtime

        oldest, newest = _update_mtime(50.0, 100.0, 200.0)
        assert oldest == 50.0
        assert newest == 200.0

    def test_updates_newest(self):
        from lmprobe.cache import _update_mtime

        oldest, newest = _update_mtime(300.0, 100.0, 200.0)
        assert oldest == 100.0
        assert newest == 300.0


class TestBatchCheckCacheStatusPooled:
    """Additional batch_check_cache_status tests for pooled paths."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        from lmprobe.cache import set_cache_backend
        set_cache_backend(None)
        self.model = "test-batch-pooled"
        yield
        set_cache_backend(None)

    def test_pooled_partial_cache(self):
        """Partial pooled cache (some layers) detected."""
        prompt = "partial pooled"
        save_prompt_pooled_activations(self.model, prompt, [0], torch.randn(1, 32), "last_token")

        need_act, _, _, partial, found = batch_check_cache_status(
            self.model, [prompt], required_layers={0, 1}, pooling="last_token"
        )
        assert need_act == [prompt]
        assert partial == 1
        assert found is not None
        assert 0 in found

    def test_topk_logits_check(self):
        """Top-k logit cache check works in batch."""
        prompt = "topk batch"
        acts = torch.randn(1, 5, 32)
        mask = torch.ones(1, 5)
        save_prompt_activations(self.model, prompt, [0], acts, mask)
        save_prompt_logits(self.model, prompt, torch.randn(1, 5, 100), mask, top_k=10)

        _, _, need_log, _, _ = batch_check_cache_status(
            self.model, [prompt], required_layers={0},
            cache_logits=True, logit_top_k=10,
        )
        assert need_log == []


class TestSavePromptActivationsV1Migration:
    """Test that v2 save cleans up v1 directory."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        self.base = tmp_path
        self.model = "v1-migrate"

    def test_v1_dir_removed_on_v2_save(self):
        """Saving v2 removes old v1 directory."""
        model_hash = _hash_string(self.model)
        prompt_hash = _hash_string("migrate prompt")
        v1_dir = self.base / model_hash / prompt_hash
        v1_dir.mkdir(parents=True)
        torch.save(torch.randn(1, 5, 32), v1_dir / "layer_0.pt")

        assert v1_dir.is_dir()

        save_prompt_activations(
            self.model, "migrate prompt", [0],
            torch.randn(1, 5, 32), torch.ones(1, 5, dtype=torch.long)
        )

        assert not v1_dir.is_dir()
        sf_path = get_prompt_cache_path(self.model, "migrate prompt")
        assert sf_path.exists()
