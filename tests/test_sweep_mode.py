"""Tests for layers='sweep' mode (Issue #76)."""

import pytest

from lmprobe import LayerSweepResult, LinearProbe

pytestmark = pytest.mark.nnsight


class TestSweepMode:
    """Tests for the sweep layer specification in LinearProbe."""

    def test_sweep_all_layers(self, tiny_model):
        """layers='sweep' trains a probe per layer."""
        probe = LinearProbe(
            model=tiny_model,
            layers="sweep",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["good", "great"], ["bad", "terrible"])
        assert probe.sweep_result_ is not None
        assert isinstance(probe.sweep_result_, LayerSweepResult)
        assert len(probe.sweep_result_) > 1

    def test_sweep_step(self, tiny_model):
        """layers='sweep:8' sweeps every 8th layer."""
        from lmprobe.extraction import get_num_layers_from_config

        num_layers = get_num_layers_from_config(tiny_model)

        probe = LinearProbe(
            model=tiny_model,
            layers="sweep:8",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["good", "great"], ["bad", "terrible"])
        assert probe.sweep_result_ is not None

        expected_layers = list(range(0, num_layers, 8))
        assert probe.sweep_result_.layers == expected_layers

    def test_sweep_range(self, tiny_model):
        """layers='sweep:0-1' sweeps layers 0 through 1."""
        probe = LinearProbe(
            model=tiny_model,
            layers="sweep:0-1",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["good", "great"], ["bad", "terrible"])
        assert probe.sweep_result_ is not None
        assert probe.sweep_result_.layers == [0, 1]

    def test_sweep_predict_works(self, tiny_model):
        """predict() works after sweep fit (uses first layer by default)."""
        probe = LinearProbe(
            model=tiny_model,
            layers="sweep:0-1",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["good", "great"], ["bad", "terrible"])
        predictions = probe.predict(["test input"])
        assert predictions.shape == (1,)

    def test_sweep_evaluate_returns_per_layer(self, tiny_model):
        """evaluate() returns per-layer results in sweep mode."""
        probe = LinearProbe(
            model=tiny_model,
            layers="sweep:0-2",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["good", "great"], ["bad", "terrible"])
        results = probe.evaluate(["test one", "test two"], [1, 0])

        assert "layer_results" in results
        assert "best_layer" in results
        assert "best_accuracy" in results
        assert isinstance(results["best_layer"], int)
        assert 0.0 <= results["best_accuracy"] <= 1.0

        # Per-layer results should have metrics
        for layer_idx, layer_metrics in results["layer_results"].items():
            assert "accuracy" in layer_metrics
            assert "f1" in layer_metrics

    def test_sweep_evaluate_updates_to_best(self, tiny_model):
        """After evaluate(), probe uses the best layer for predictions."""
        probe = LinearProbe(
            model=tiny_model,
            layers="sweep:0-2",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["good", "great"], ["bad", "terrible"])
        results = probe.evaluate(["test one", "test two"], [1, 0])

        # After evaluate, probe should use best layer
        best_layer = results["best_layer"]
        best_probe = probe.sweep_result_[best_layer]

        # Predictions should now come from best layer's classifier
        assert probe.classifier_ is best_probe.classifier_

    def test_sweep_with_classifier_kwargs(self, tiny_model):
        """Sweep mode works with classifier_kwargs."""
        probe = LinearProbe(
            model=tiny_model,
            layers="sweep:0-1",
            classifier_kwargs={"C": 0.01},
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["good", "great"], ["bad", "terrible"])
        # Each sub-probe should have the custom C
        for layer, sub_probe in probe.sweep_result_.probes.items():
            assert sub_probe._classifier_template.C == 0.01

    def test_sweep_with_preprocessing(self, tiny_model):
        """Sweep mode works with preprocessing pipeline."""
        probe = LinearProbe(
            model=tiny_model,
            layers="sweep:0-1",
            preprocessing="standard+pca:4",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(
            ["good", "great", "nice", "wonderful", "excellent"],
            ["bad", "terrible", "awful", "horrible", "dreadful"],
        )
        # Each sub-probe should have preprocessing
        for layer, sub_probe in probe.sweep_result_.probes.items():
            assert sub_probe.preprocessing_pipeline_ is not None
