"""Tests for LinearProbe.sweep_layers() per-layer probe sweep."""

import numpy as np
import pytest

from lmprobe import LayerSweepResult, LinearProbe


class TestSweepLayers:
    """Tests for the sweep_layers classmethod."""

    def test_sweep_returns_result(self, tiny_model):
        """sweep_layers returns a LayerSweepResult."""
        result = LinearProbe.sweep_layers(
            model=tiny_model,
            positive_prompts=["good", "great"],
            negative_prompts=["bad", "terrible"],
            layers="all",
            pooling="last_token",
            classifier="logistic_regression",
            device="cpu",
            remote=False,
            random_state=42,
        )
        assert isinstance(result, LayerSweepResult)

    def test_sweep_has_all_layers(self, tiny_model):
        """sweep_layers creates a probe for each requested layer."""
        from lmprobe.extraction import get_num_layers_from_config

        num_layers = get_num_layers_from_config(tiny_model)

        result = LinearProbe.sweep_layers(
            model=tiny_model,
            positive_prompts=["good", "great"],
            negative_prompts=["bad", "terrible"],
            layers="all",
            device="cpu",
            remote=False,
            random_state=42,
        )
        assert len(result) == num_layers
        assert result.layers == list(range(num_layers))

    def test_sweep_specific_layers(self, tiny_model):
        """sweep_layers works with specific layer indices."""
        result = LinearProbe.sweep_layers(
            model=tiny_model,
            positive_prompts=["good", "great"],
            negative_prompts=["bad", "terrible"],
            layers=[0, -1],
            device="cpu",
            remote=False,
            random_state=42,
        )
        assert len(result) >= 2
        assert 0 in result.probes

    def test_sweep_single_layer(self, tiny_model):
        """sweep_layers works with a single layer."""
        result = LinearProbe.sweep_layers(
            model=tiny_model,
            positive_prompts=["good", "great"],
            negative_prompts=["bad", "terrible"],
            layers=-1,
            device="cpu",
            remote=False,
            random_state=42,
        )
        assert len(result) == 1

    def test_sweep_probes_are_fitted(self, tiny_model):
        """Each probe in the sweep result is fitted."""
        result = LinearProbe.sweep_layers(
            model=tiny_model,
            positive_prompts=["good", "great"],
            negative_prompts=["bad", "terrible"],
            layers="all",
            device="cpu",
            remote=False,
            random_state=42,
        )
        for layer, probe in result.probes.items():
            assert probe.classifier_ is not None
            assert probe.classes_ is not None

    def test_sweep_score(self, tiny_model):
        """score() returns dict mapping layer -> accuracy."""
        result = LinearProbe.sweep_layers(
            model=tiny_model,
            positive_prompts=["good", "great"],
            negative_prompts=["bad", "terrible"],
            layers="all",
            device="cpu",
            remote=False,
            random_state=42,
        )
        scores = result.score(["test one", "test two"], [1, 0])
        assert isinstance(scores, dict)
        for layer, acc in scores.items():
            assert isinstance(layer, int)
            assert 0.0 <= acc <= 1.0

    def test_sweep_best_layer(self, tiny_model):
        """best_layer() returns an int layer index."""
        result = LinearProbe.sweep_layers(
            model=tiny_model,
            positive_prompts=["good", "great"],
            negative_prompts=["bad", "terrible"],
            layers="all",
            device="cpu",
            remote=False,
            random_state=42,
        )
        best = result.best_layer(["test one", "test two"], [1, 0])
        assert isinstance(best, int)
        assert best in result.probes

    def test_sweep_predict(self, tiny_model):
        """predict() returns dict mapping layer -> predictions."""
        result = LinearProbe.sweep_layers(
            model=tiny_model,
            positive_prompts=["good", "great"],
            negative_prompts=["bad", "terrible"],
            layers="all",
            device="cpu",
            remote=False,
            random_state=42,
        )
        predictions = result.predict(["test one", "test two"])
        assert isinstance(predictions, dict)
        for layer, preds in predictions.items():
            assert preds.shape == (2,)

    def test_sweep_predict_proba(self, tiny_model):
        """predict_proba() returns dict mapping layer -> probabilities."""
        result = LinearProbe.sweep_layers(
            model=tiny_model,
            positive_prompts=["good", "great"],
            negative_prompts=["bad", "terrible"],
            layers="all",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probas = result.predict_proba(["test one", "test two"])
        assert isinstance(probas, dict)
        for layer, proba in probas.items():
            assert proba.shape == (2, 2)

    def test_sweep_getitem(self, tiny_model):
        """LayerSweepResult supports [] indexing by layer."""
        result = LinearProbe.sweep_layers(
            model=tiny_model,
            positive_prompts=["good", "great"],
            negative_prompts=["bad", "terrible"],
            layers=0,
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe = result[0]
        assert isinstance(probe, LinearProbe)

    def test_sweep_each_probe_is_single_layer(self, tiny_model):
        """Each probe in the sweep should use exactly one layer."""
        result = LinearProbe.sweep_layers(
            model=tiny_model,
            positive_prompts=["good", "great"],
            negative_prompts=["bad", "terrible"],
            layers="all",
            device="cpu",
            remote=False,
            random_state=42,
        )
        for layer, probe in result.probes.items():
            assert len(probe._extractor.layer_indices) == 1
            assert probe._extractor.layer_indices[0] == layer
