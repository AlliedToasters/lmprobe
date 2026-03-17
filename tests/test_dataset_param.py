"""Tests for the dataset parameter on LinearProbe / Probe."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from lmprobe import Probe
from lmprobe.sharing import DatasetMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(model_name="stas/tiny-random-llama-2", prompts=None):
    """Build a minimal DatasetMetadata for tests."""
    return DatasetMetadata(
        model_name=model_name,
        available_layers=[0, 1],
        num_prompts=len(prompts or []),
        format_version="1.1",
        tensor_descriptors={
            "hidden_layers": {
                "layout": "per_layer",
                "layers": [0, 1],
                "shards": [],
            }
        },
        prompts=prompts or [],
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestProbeDatasetConstruction:
    def test_probe_dataset_only(self):
        """Probe(dataset=...) without model should construct."""
        probe = Probe(dataset="user/my-activations", layers=-1)
        assert probe.dataset == "user/my-activations"
        assert probe.model is None
        assert probe._extractor is None

    def test_probe_dataset_and_model(self, tiny_model):
        """Probe with both model and dataset should construct."""
        probe = Probe(
            model=tiny_model,
            dataset="user/my-activations",
            layers=-1,
            device="cpu",
            remote=False,
        )
        assert probe.dataset == "user/my-activations"
        assert probe.model == tiny_model
        assert probe._extractor is not None

    def test_probe_no_model_no_dataset_raises(self):
        """Probe with neither model nor dataset should raise on fit."""
        probe = Probe(layers=-1)
        with pytest.raises(ValueError, match="No model or dataset"):
            probe.fit(["positive"], ["negative"])

    def test_dataset_model_mismatch_raises(self):
        """Mismatched model names should raise ValueError."""
        probe = Probe(
            model="some-other/model",
            dataset="user/my-activations",
            layers=-1,
        )
        meta = _make_metadata(model_name="stas/tiny-random-llama-2")
        with patch(
            "lmprobe.sharing.fetch_dataset_metadata", return_value=meta
        ):
            with pytest.raises(ValueError, match="Model mismatch"):
                probe._ensure_dataset_metadata()


# ---------------------------------------------------------------------------
# Metadata & layer resolution
# ---------------------------------------------------------------------------

class TestDatasetMetadata:
    def test_ensure_dataset_metadata_caches(self):
        """Metadata should be fetched once and cached."""
        probe = Probe(dataset="user/repo", layers=-1)
        meta = _make_metadata(prompts=["hello"])

        with patch(
            "lmprobe.sharing.fetch_dataset_metadata", return_value=meta
        ) as mock_fetch:
            result1 = probe._ensure_dataset_metadata()
            result2 = probe._ensure_dataset_metadata()
            assert result1 is result2
            mock_fetch.assert_called_once()

    def test_resolve_layers_from_dataset(self):
        """Layer resolution should work without a loaded model."""
        probe = Probe(dataset="user/repo", layers=-1)
        meta = _make_metadata()

        with patch(
            "lmprobe.sharing.fetch_dataset_metadata", return_value=meta
        ), patch(
            "lmprobe.extraction.get_num_layers_from_config", return_value=2
        ), patch(
            "lmprobe.extraction.resolve_layers", return_value=[1]
        ) as mock_resolve:
            layers = probe._resolve_layers_from_dataset()
            assert layers == [1]
            mock_resolve.assert_called_once_with(-1, 2)


# ---------------------------------------------------------------------------
# Dataset pull
# ---------------------------------------------------------------------------

class TestDatasetPull:
    def test_pull_dataset_for_prompts_skips_already_pulled(self):
        """Already-pulled prompts should not be re-downloaded."""
        probe = Probe(dataset="user/repo", layers=-1)
        probe._dataset_pulled_prompts = {"already cached"}

        with patch("lmprobe.sharing.pull_dataset") as mock_pull:
            probe._pull_dataset_for_prompts(
                ["already cached", "new one"], layers=[0, 1]
            )
            mock_pull.assert_called_once()
            call_kwargs = mock_pull.call_args
            assert call_kwargs[1]["target_prompts"] == ["new one"]

    def test_pull_dataset_all_cached_noop(self):
        """If all prompts already pulled, pull_dataset should not be called."""
        probe = Probe(dataset="user/repo", layers=-1)
        probe._dataset_pulled_prompts = {"a", "b"}

        with patch("lmprobe.sharing.pull_dataset") as mock_pull:
            probe._pull_dataset_for_prompts(["a", "b"], layers=[0])
            mock_pull.assert_not_called()


# ---------------------------------------------------------------------------
# Model-free loading
# ---------------------------------------------------------------------------

class TestLoadAndPoolFromCache:
    def test_load_and_pool_success(self):
        """Model-free path loads from cache and pools."""
        probe = Probe(dataset="user/repo", layers=-1, pooling="last_token")
        meta = _make_metadata(model_name="test-model")
        probe._dataset_metadata = meta

        # Fake activations: seq_len=3, hidden_dim=4 (2 layers × 2 dim)
        acts = torch.randn(3, 4)
        mask = torch.ones(3)

        with patch(
            "lmprobe.cache.load_prompt_activations",
            return_value=(acts, mask),
        ):
            result, attn = probe._load_and_pool_from_cache(
                ["prompt1", "prompt2"], layers=[0, 1],
                pooling_strategy="last_token",
            )
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 2  # batch size
            assert attn is None

    def test_load_and_pool_missing_prompt_raises(self):
        """Missing prompts without model should raise clear error."""
        probe = Probe(dataset="user/repo", layers=-1)
        meta = _make_metadata(model_name="test-model")
        probe._dataset_metadata = meta

        with patch(
            "lmprobe.cache.load_prompt_activations",
            side_effect=FileNotFoundError("not found"),
        ):
            with pytest.raises(ValueError, match="Cannot find activations"):
                probe._load_and_pool_from_cache(
                    ["missing"], layers=[0], pooling_strategy="last_token"
                )


# ---------------------------------------------------------------------------
# Integration: fit/predict with dataset
# ---------------------------------------------------------------------------

class TestFitPredictDataset:
    def test_fit_predict_dataset_only(self):
        """Full round-trip with dataset-only (mocked HF + cache)."""
        probe = Probe(
            dataset="user/repo",
            layers=-1,
            pooling="last_token",
            classifier="logistic_regression",
            random_state=42,
        )
        meta = _make_metadata(
            model_name="test-model",
            prompts=["pos1", "pos2", "neg1", "neg2", "test1"],
        )

        # Each call returns (acts, mask) for a single prompt
        hidden_dim = 4
        seq_len = 3

        def fake_load(_model_name, _prompt, _layers):
            return torch.randn(seq_len, hidden_dim), torch.ones(seq_len)

        with patch(
            "lmprobe.sharing.fetch_dataset_metadata", return_value=meta
        ), patch(
            "lmprobe.extraction.get_num_layers_from_config", return_value=2
        ), patch(
            "lmprobe.extraction.resolve_layers", return_value=[1]
        ), patch(
            "lmprobe.sharing.pull_dataset"
        ), patch(
            "lmprobe.cache.load_prompt_activations",
            side_effect=fake_load,
        ):
            probe.fit(["pos1", "pos2"], ["neg1", "neg2"])
            preds = probe.predict(["test1"])
            assert preds.shape == (1,)

            proba = probe.predict_proba(["test1"])
            assert proba.shape == (1, 2)

    def test_fit_predict_dataset_with_model(self, tiny_model):
        """Dataset + model: dataset pull happens before extraction."""
        meta = _make_metadata(model_name=tiny_model)

        probe = Probe(
            model=tiny_model,
            dataset="user/repo",
            layers=-1,
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )

        with patch(
            "lmprobe.sharing.fetch_dataset_metadata", return_value=meta
        ), patch(
            "lmprobe.sharing.pull_dataset"
        ) as mock_pull:
            probe.fit(
                ["Who wants a walk?", "Good boy!"],
                ["Cats are great", "Purring loudly"],
            )
            # pull_dataset should have been called (at least once)
            assert mock_pull.called

            preds = probe.predict(["Fetch the ball!"])
            assert preds.shape == (1,)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

class TestWarmupDataset:
    def test_warmup_pulls_from_dataset(self):
        """warmup() should call pull_dataset when dataset is set."""
        meta = _make_metadata(model_name="test-model")

        probe = Probe(dataset="user/repo", layers=-1)

        with patch(
            "lmprobe.sharing.fetch_dataset_metadata", return_value=meta
        ), patch(
            "lmprobe.extraction.get_num_layers_from_config", return_value=2
        ), patch(
            "lmprobe.extraction.resolve_layers", return_value=[1]
        ), patch(
            "lmprobe.sharing.pull_dataset"
        ) as mock_pull:
            probe.warmup(["prompt1", "prompt2"])
            assert mock_pull.called
            call_kwargs = mock_pull.call_args
            assert set(call_kwargs[1]["target_prompts"]) == {"prompt1", "prompt2"}

    def test_warmup_with_model_and_dataset(self, tiny_model):
        """warmup() with both model and dataset pulls then extracts."""
        meta = _make_metadata(model_name=tiny_model)

        probe = Probe(
            model=tiny_model,
            dataset="user/repo",
            layers=-1,
            device="cpu",
            remote=False,
        )

        with patch(
            "lmprobe.sharing.fetch_dataset_metadata", return_value=meta
        ), patch(
            "lmprobe.sharing.pull_dataset"
        ) as mock_pull:
            probe.warmup(["Hello world"])
            assert mock_pull.called
