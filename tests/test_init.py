"""Tests for lmprobe.__init__ lazy imports and set_max_threads."""

from unittest.mock import patch

import pytest

import lmprobe


class TestLazyImports:
    """Test that __getattr__ lazy imports work for all __all__ names."""

    def test_import_per_layer_scaler(self):
        from lmprobe import PerLayerScaler
        from lmprobe.scaling import PerLayerScaler as Direct

        assert PerLayerScaler is Direct

    def test_import_clear_model_cache(self):
        from lmprobe import clear_model_cache

        assert callable(clear_model_cache)

    def test_import_enable_cache_logging(self):
        from lmprobe import enable_cache_logging

        assert callable(enable_cache_logging)

    def test_import_cache_info(self):
        from lmprobe import cache_info

        assert callable(cache_info)

    def test_import_cache_info_types(self):
        from lmprobe import CacheInfo, ModelCacheInfo

        assert CacheInfo is not None
        assert ModelCacheInfo is not None

    def test_import_evict(self):
        from lmprobe import evict

        assert callable(evict)

    def test_import_set_cache_backend(self):
        from lmprobe import set_cache_backend

        assert callable(set_cache_backend)

    def test_import_set_cache_limit(self):
        from lmprobe import set_cache_limit

        assert callable(set_cache_limit)

    def test_import_set_cache_dtype(self):
        from lmprobe import set_cache_dtype

        assert callable(set_cache_dtype)

    def test_import_unified_cache(self):
        from lmprobe import UnifiedCache

        assert UnifiedCache is not None

    def test_import_warmup_stats(self):
        from lmprobe import WarmupStats

        assert WarmupStats is not None

    def test_import_cached_logits(self):
        from lmprobe import CachedLogits

        assert CachedLogits is not None

    def test_import_load_layer_across_prompts(self):
        from lmprobe import load_layer_across_prompts

        assert callable(load_layer_across_prompts)

    def test_import_load_layer_last_token(self):
        from lmprobe import load_layer_last_token

        assert callable(load_layer_last_token)

    def test_import_cached_prompt_info(self):
        from lmprobe import CachedPromptInfo

        assert CachedPromptInfo is not None

    def test_import_discover_cached(self):
        from lmprobe import discover_cached

        assert callable(discover_cached)

    def test_import_manifest_entry(self):
        from lmprobe import ManifestEntry

        assert ManifestEntry is not None

    def test_import_list_cached_prompts(self):
        from lmprobe import list_cached_prompts

        assert callable(list_cached_prompts)

    def test_import_push_dataset(self):
        from lmprobe import push_dataset

        assert callable(push_dataset)

    def test_import_pull_dataset(self):
        from lmprobe import pull_dataset

        assert callable(pull_dataset)

    def test_import_load_activation_dataset(self):
        from lmprobe import load_activation_dataset

        assert callable(load_activation_dataset)

    def test_import_load_activations(self):
        from lmprobe import load_activations

        assert callable(load_activations)

    def test_import_fetch_dataset_metadata(self):
        from lmprobe import fetch_dataset_metadata

        assert callable(fetch_dataset_metadata)

    def test_import_dataset_metadata(self):
        from lmprobe import DatasetMetadata

        assert DatasetMetadata is not None

    def test_import_migrate_dataset(self):
        from lmprobe import migrate_dataset

        assert callable(migrate_dataset)

    def test_import_upgrade_dataset_format(self):
        from lmprobe import upgrade_dataset_format

        assert callable(upgrade_dataset_format)

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = lmprobe.nonexistent_thing  # type: ignore[attr-defined]


class TestSetMaxThreads:
    """Tests for set_max_threads."""

    def test_set_max_threads(self):
        import os

        with patch("torch.set_num_threads") as mock_nt, \
             patch("torch.set_num_interop_threads") as mock_it:
            lmprobe.set_max_threads(4)

        mock_nt.assert_called_once_with(4)
        mock_it.assert_called_once_with(4)
        assert os.environ.get("OMP_NUM_THREADS") == "4"
        assert os.environ.get("MKL_NUM_THREADS") == "4"


class TestVersion:
    """Tests for version string."""

    def test_version_is_string(self):
        assert isinstance(lmprobe.__version__, str)


class TestDirectImports:
    """Tests for eagerly loaded imports."""

    def test_probe(self):
        from lmprobe import Probe

        assert Probe is not None

    def test_linear_probe(self):
        from lmprobe import LinearProbe

        assert LinearProbe is not None

    def test_probe_ensemble(self):
        from lmprobe import ProbeEnsemble

        assert ProbeEnsemble is not None

    def test_baseline_probe(self):
        from lmprobe import BaselineProbe

        assert BaselineProbe is not None

    def test_baseline_battery(self):
        from lmprobe import BaselineBattery

        assert BaselineBattery is not None

    def test_layer_sweep_result(self):
        from lmprobe import LayerSweepResult

        assert LayerSweepResult is not None
