"""Tests for per-layer activation caching."""

import shutil

import pytest
import torch

from lmprobe.cache import (
    clear_cache,
    get_cached_layers,
    get_extraction_cache_dir,
    invalidate_extraction_cache,
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
    """Tests for CachedExtractor with per-layer caching."""

    def test_caches_layers_individually(self, tiny_model, tmp_path, monkeypatch):
        """First extraction caches each layer as a separate file."""
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

        # Check cache directory structure
        cache_dir = get_extraction_cache_dir(tiny_model, prompts)
        cached_layers = get_cached_layers(cache_dir)
        assert cached_layers == {0, 1}

        # Check attention mask exists
        assert (cache_dir / "attention_mask.pt").exists()

    def test_partial_cache_hit(self, tiny_model, tmp_path, monkeypatch):
        """Reuses cached layers and only extracts missing ones."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import CachedExtractor
        from lmprobe.extraction import ActivationExtractor

        prompts = ["test prompt"]

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

        # Verify layer 0 is cached
        cache_dir = get_extraction_cache_dir(tiny_model, prompts)
        assert get_cached_layers(cache_dir) == {0}

        # Second extraction: layers [0, 1] - should reuse layer 0, extract layer 1
        extractor2 = ActivationExtractor(
            tiny_model,
            device="cpu",
            layers=[0, 1],
            batch_size=4,
        )
        cached2 = CachedExtractor(extractor2)
        acts2, mask2 = cached2.extract(prompts, remote=False)

        # Now both layers should be cached
        assert get_cached_layers(cache_dir) == {0, 1}

        # Verify shapes are correct
        # acts2 should have 2 layers worth of hidden dims (double acts1)
        assert acts2.shape[-1] == acts1.shape[-1] * 2

    def test_full_cache_hit(self, tiny_model, tmp_path, monkeypatch):
        """Full cache hit loads all layers without extraction."""
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

        prompts = ["invalidation test"]

        extractor = ActivationExtractor(
            tiny_model,
            device="cpu",
            layers=[0],
            batch_size=4,
        )
        cached = CachedExtractor(extractor)

        # First extraction
        acts1, _ = cached.extract(prompts, remote=False)

        # Get cache dir before invalidation
        cache_dir = get_extraction_cache_dir(tiny_model, prompts)
        assert cache_dir.exists()

        # Extract with invalidation
        acts2, _ = cached.extract(prompts, remote=False, invalidate_cache=True)

        # Cache should exist again
        assert cache_dir.exists()
        assert get_cached_layers(cache_dir) == {0}

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

        # Should have two different cache directories
        cache_dir1 = get_extraction_cache_dir(tiny_model, prompts1)
        cache_dir2 = get_extraction_cache_dir(tiny_model, prompts2)

        assert cache_dir1 != cache_dir2
        assert cache_dir1.exists()
        assert cache_dir2.exists()


class TestLinearProbeWithCache:
    """Integration tests for LinearProbe with per-layer caching."""

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

        # Cache should contain layer 0
        all_prompts = positive + negative
        cache_dir = get_extraction_cache_dir(tiny_model, all_prompts)
        assert get_cached_layers(cache_dir) == {0}

        # Experiment 2: layers [0, 1] - should reuse layer 0, extract layer 1
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

        # Cache should now contain both layers
        assert get_cached_layers(cache_dir) == {0, 1}
