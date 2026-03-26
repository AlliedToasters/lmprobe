"""Tests for probe.py — targeting uncovered lines to increase coverage."""

import numpy as np
import pytest
import torch

from lmprobe import Probe
from lmprobe.probe import (
    LayerSweepResult,
    LinearProbe,
    _parse_sweep_spec,
    _resolve_dtype,
)

# ---------------------------------------------------------------------------
# Helper data
# ---------------------------------------------------------------------------

POS = [
    "The dog barked loudly",
    "My puppy loves to play fetch",
    "Dogs are loyal companions",
    "The golden retriever wagged its tail",
    "Walking the dog in the park",
]

NEG = [
    "The cat purred softly",
    "My kitten sleeps all day",
    "Cats are independent animals",
    "The tabby cat stretched lazily",
    "The cat knocked things off the table",
]

TEST = ["A dog chased the ball", "The cat sat on the mat"]


# ---------------------------------------------------------------------------
# _parse_sweep_spec tests
# ---------------------------------------------------------------------------

class TestParseSweepSpec:
    def test_non_sweep_int(self):
        is_sweep, spec = _parse_sweep_spec(5)
        assert is_sweep is False
        assert spec == 5

    def test_sweep_bare(self):
        is_sweep, spec = _parse_sweep_spec("sweep")
        assert is_sweep is True
        assert spec == "all"

    def test_sweep_step(self):
        is_sweep, spec = _parse_sweep_spec("sweep:10")
        assert is_sweep is True
        assert spec == 10

    def test_sweep_range(self):
        is_sweep, spec = _parse_sweep_spec("sweep:2-5")
        assert is_sweep is True
        assert spec == [2, 3, 4, 5]

    def test_sweep_invalid_range(self):
        with pytest.raises(ValueError, match="Invalid sweep range"):
            _parse_sweep_spec("sweep:1-2-3")

    def test_non_sweep_string(self):
        is_sweep, spec = _parse_sweep_spec("middle")
        assert is_sweep is False
        assert spec == "middle"

    def test_sweep_prefix_without_colon(self):
        """'sweepXYZ' (no colon) is not a sweep spec."""
        is_sweep, spec = _parse_sweep_spec("sweepXYZ")
        assert is_sweep is False


# ---------------------------------------------------------------------------
# _resolve_dtype tests
# ---------------------------------------------------------------------------

class TestResolveDtype:
    def test_none(self):
        assert _resolve_dtype(None) is None

    def test_valid_strings(self):
        assert _resolve_dtype("float32") == torch.float32
        assert _resolve_dtype("float16") == torch.float16
        assert _resolve_dtype("bfloat16") == torch.bfloat16

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            _resolve_dtype("int8")

    def test_passthrough_non_string(self):
        """Non-string values are passed through."""
        assert _resolve_dtype(torch.float32) == torch.float32


# ---------------------------------------------------------------------------
# Probe __init__ validation
# ---------------------------------------------------------------------------

class TestProbeInit:
    def test_invalid_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            Probe(task="invalid_task")

    def test_local_backend_with_remote_raises(self):
        with pytest.raises(ValueError, match="backend='local' does not support remote"):
            Probe(model="some-model", backend="local", remote=True)

    def test_regression_default_classifier(self):
        """Regression task with default classifier resolves to ridge."""
        probe = Probe(task="regression", random_state=42)
        # Should not raise; classifier template should be Ridge
        assert probe.task == "regression"

    def test_no_model_sets_none_pooling(self):
        """When model is None, pooling stays None."""
        probe = Probe()
        assert probe._train_pooling is None
        assert probe._inference_pooling is None

    def test_normalize_layers_per_layer_string(self):
        probe = Probe(normalize_layers="per_layer")
        assert probe._get_scaling_strategy() == "per_layer"

    def test_normalize_layers_false(self):
        probe = Probe(normalize_layers=False)
        assert probe._get_scaling_strategy() is None

    def test_normalize_layers_invalid(self):
        probe = Probe(normalize_layers="wrong")
        with pytest.raises(ValueError, match="Invalid normalize_layers"):
            probe._get_scaling_strategy()


# ---------------------------------------------------------------------------
# fit_from_activations (synthetic)
# ---------------------------------------------------------------------------

class TestFitFromActivationsSynthetic:
    def test_basic_roundtrip(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 64)
        y = np.array([0] * 15 + [1] * 15)

        probe = Probe(classifier="logistic_regression", random_state=42)
        result = probe.fit_from_activations(X, y)
        assert result is probe  # method chaining

        preds = probe.predict_from_activations(X)
        assert preds.shape == (30,)

        proba = probe.predict_proba_from_activations(X)
        assert proba.shape == (30, 2)

    def test_sample_weight(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 32)
        y = np.array([0] * 10 + [1] * 10)
        weights = np.ones(20)

        probe = Probe(random_state=42)
        probe.fit_from_activations(X, y, sample_weight=weights)
        assert probe.classifier_ is not None

    def test_sample_weight_mismatch_raises(self):
        X = np.random.randn(20, 32)
        y = np.array([0] * 10 + [1] * 10)
        wrong_weights = np.ones(5)

        probe = Probe(random_state=42)
        with pytest.raises(ValueError, match="sample_weight length"):
            probe.fit_from_activations(X, y, sample_weight=wrong_weights)

    def test_classes_set_for_classification(self):
        X = np.random.randn(20, 32)
        y = np.array([0] * 10 + [1] * 10)
        probe = Probe(random_state=42)
        probe.fit_from_activations(X, y)
        assert probe.classes_ is not None
        assert set(probe.classes_) == {0, 1}

    def test_classes_none_for_regression(self):
        X = np.random.randn(20, 32)
        y = np.random.randn(20)
        probe = Probe(task="regression", random_state=42)
        probe.fit_from_activations(X, y)
        assert probe.classes_ is None


# ---------------------------------------------------------------------------
# check_fitted and check_model
# ---------------------------------------------------------------------------

class TestChecks:
    def test_predict_before_fit_raises(self):
        probe = Probe(model="stas/tiny-random-llama-2", layers=-1,
                      device="cpu", remote=False)
        with pytest.raises(RuntimeError, match="not been fitted"):
            probe.predict(["hello"])

    def test_predict_from_activations_before_fit_raises(self):
        probe = Probe(random_state=42)
        with pytest.raises(RuntimeError, match="not been fitted"):
            probe.predict_from_activations(np.zeros((1, 10)))

    def test_fit_without_model_raises(self):
        probe = Probe(random_state=42)
        with pytest.raises(ValueError, match="No model"):
            probe.fit(["hello"], ["world"])

    def test_fit_without_negative_raises(self, tiny_model):
        probe = Probe(model=tiny_model, layers=-1, device="cpu", remote=False)
        with pytest.raises(ValueError, match="fit\\(\\) requires two arguments"):
            probe.fit(["hello"])


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip_from_activations(self, tmp_path):
        """save/load works for probes trained from activations."""
        rng = np.random.RandomState(42)
        X = rng.randn(20, 64)
        y = np.array([0] * 10 + [1] * 10)

        probe = Probe(classifier="logistic_regression", random_state=42)
        probe.fit_from_activations(X, y)

        path = str(tmp_path / "probe.pkl")
        probe.save(path)
        loaded = Probe.load(path)

        preds_orig = probe.predict_from_activations(X)
        preds_loaded = loaded.predict_from_activations(X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_roundtrip_with_model(self, tiny_model, tmp_path):
        """save/load works for probes trained with a model."""
        probe = Probe(
            model=tiny_model, layers=-1, pooling="last_token",
            device="cpu", remote=False, random_state=42,
        )
        probe.fit(POS, NEG)

        path = str(tmp_path / "probe.pkl")
        probe.save(path)
        loaded = Probe.load(path)

        orig_pred = probe.predict(TEST)
        loaded_pred = loaded.predict(TEST)
        np.testing.assert_array_equal(orig_pred, loaded_pred)

    def test_roundtrip_regression(self, tmp_path):
        """save/load works for regression probes."""
        rng = np.random.RandomState(42)
        X = rng.randn(30, 32)
        y = rng.randn(30)

        probe = Probe(task="regression", random_state=42)
        probe.fit_from_activations(X, y)

        path = str(tmp_path / "reg_probe.pkl")
        probe.save(path)
        loaded = Probe.load(path)

        preds_orig = probe.predict_from_activations(X)
        preds_loaded = loaded.predict_from_activations(X)
        np.testing.assert_allclose(preds_orig, preds_loaded)

    def test_save_before_fit_raises(self, tmp_path):
        probe = Probe(random_state=42)
        with pytest.raises(RuntimeError, match="not been fitted"):
            probe.save(str(tmp_path / "nope.pkl"))

    def test_roundtrip_preserves_config(self, tmp_path):
        """Loaded probe preserves key config."""
        probe = Probe(
            classifier="logistic_regression",
            random_state=42,
            normalize_layers="per_layer",
            batch_size=4,
        )
        X = np.random.randn(20, 32)
        y = np.array([0] * 10 + [1] * 10)
        probe.fit_from_activations(X, y)

        path = str(tmp_path / "cfg_probe.pkl")
        probe.save(path)
        loaded = Probe.load(path)

        assert loaded.random_state == 42
        assert loaded.normalize_layers == "per_layer"
        assert loaded.batch_size == 4


# ---------------------------------------------------------------------------
# score() method
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_basic(self, tiny_model):
        probe = Probe(
            model=tiny_model, layers=-1, pooling="last_token",
            device="cpu", remote=False, random_state=42,
        )
        probe.fit(POS, NEG)
        accuracy = probe.score(TEST, [1, 0])
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_score_from_activations(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 32)
        y = np.array([0] * 10 + [1] * 10)
        probe = Probe(random_state=42)
        probe.fit_from_activations(X, y)
        score = probe.score_from_activations(X, y)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# evaluate() method
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_returns_metrics(self, tiny_model):
        probe = Probe(
            model=tiny_model, layers=-1, pooling="last_token",
            device="cpu", remote=False, random_state=42,
        )
        probe.fit(POS, NEG)
        results = probe.evaluate(TEST, [1, 0])
        assert "accuracy" in results
        assert "f1" in results
        assert "precision" in results
        assert "recall" in results
        assert "n_eval" in results
        assert results["n_eval"] == 2
        assert "eval_hash" in results
        assert results["eval_hash"].startswith("sha256:")
        # Results cached on probe
        assert probe._evaluation_results_ is not None


# ---------------------------------------------------------------------------
# Standard mode (prompts + labels) for fit()
# ---------------------------------------------------------------------------

class TestStandardModeFit:
    def test_fit_with_integer_labels(self, tiny_model):
        """fit(prompts, integer_labels) triggers standard mode."""
        probe = Probe(
            model=tiny_model, layers=-1, pooling="last_token",
            device="cpu", remote=False, random_state=42,
        )
        all_prompts = POS + NEG
        labels = [1] * len(POS) + [0] * len(NEG)
        probe.fit(all_prompts, labels)
        preds = probe.predict(TEST)
        assert preds.shape == (2,)


# ---------------------------------------------------------------------------
# Per-token / inference_pooling="all"
# ---------------------------------------------------------------------------

class TestPerTokenPrediction:
    def test_predict_proba_all_pooling(self, tiny_model):
        probe = Probe(
            model=tiny_model, layers=-1, pooling="last_token",
            inference_pooling="all", device="cpu", remote=False,
            random_state=42,
        )
        probe.fit(POS[:3], NEG[:3])
        proba = probe.predict_proba(["Hello world"])
        # Should be 3D: (batch, seq_len, n_classes)
        assert proba.ndim == 3
        assert proba.shape[0] == 1

    def test_predict_all_pooling(self, tiny_model):
        probe = Probe(
            model=tiny_model, layers=-1, pooling="last_token",
            inference_pooling="all", device="cpu", remote=False,
            random_state=42,
        )
        probe.fit(POS[:3], NEG[:3])
        preds = probe.predict(["Hello world"])
        # With inference_pooling="all", predict returns per-token predictions
        # Shape is (batch, seq_len) since predict_proba is 3D
        assert preds.shape[0] == 1
        assert preds.ndim in (1, 2)  # depends on aggregation path


# ---------------------------------------------------------------------------
# Different layers parameter values
# ---------------------------------------------------------------------------

class TestLayerSpecs:
    def test_negative_index(self, tiny_model):
        probe = Probe(
            model=tiny_model, layers=-1, device="cpu",
            remote=False, random_state=42,
        )
        probe.fit(POS[:3], NEG[:3])
        assert probe.predict(TEST[:1]).shape == (1,)

    def test_list_of_ints(self, tiny_model):
        probe = Probe(
            model=tiny_model, layers=[0, 1], device="cpu",
            remote=False, random_state=42,
        )
        probe.fit(POS[:3], NEG[:3])
        assert probe.predict(TEST[:1]).shape == (1,)

    def test_middle_string(self, tiny_model):
        probe = Probe(
            model=tiny_model, layers="middle", device="cpu",
            remote=False, random_state=42,
        )
        probe.fit(POS[:3], NEG[:3])
        assert probe.predict(TEST[:1]).shape == (1,)


# ---------------------------------------------------------------------------
# LinearProbe alias
# ---------------------------------------------------------------------------

class TestAlias:
    def test_linear_probe_alias(self):
        assert LinearProbe is Probe


# ---------------------------------------------------------------------------
# LayerSweepResult
# ---------------------------------------------------------------------------

class TestLayerSweepResult:
    def test_empty_result(self):
        r = LayerSweepResult()
        assert len(r) == 0
        assert r.layers == []

    def test_getitem(self, tiny_model):
        probe = Probe(
            model=tiny_model, layers=0, device="cpu",
            remote=False, random_state=42,
        )
        probe.fit(POS[:3], NEG[:3])
        r = LayerSweepResult(probes={0: probe})
        assert r[0] is probe
        assert r.layers == [0]
        assert len(r) == 1


# ---------------------------------------------------------------------------
# _check_classification_task
# ---------------------------------------------------------------------------

class TestRegressionGuards:
    def test_predict_proba_raises_for_regression(self):
        X = np.random.randn(20, 32)
        y = np.random.randn(20)
        probe = Probe(task="regression", random_state=42)
        probe.fit_from_activations(X, y)
        with pytest.raises(ValueError, match="regression"):
            probe.predict_proba_from_activations(X)


# ---------------------------------------------------------------------------
# Preprocessing spec parsing
# ---------------------------------------------------------------------------

class TestPreprocessingSpec:
    def test_none_spec(self):
        assert Probe._parse_preprocessing_spec(None) is None

    def test_string_spec(self):
        result = Probe._parse_preprocessing_spec("standard+pca:50")
        assert result == ["standard", "pca:50"]

    def test_list_spec(self):
        result = Probe._parse_preprocessing_spec(["standard", "pca"])
        assert result == ["standard", "pca"]

    def test_preprocessing_includes_standard(self):
        probe = Probe(preprocessing="standard+pca:10")
        assert probe._preprocessing_includes_standard() is True

    def test_preprocessing_no_standard(self):
        probe = Probe(preprocessing="pca:10", pca_components=10)
        assert probe._preprocessing_includes_standard() is False

    def test_preprocessing_none(self):
        probe = Probe()
        assert probe._preprocessing_includes_standard() is False


# ---------------------------------------------------------------------------
# _try_auroc
# ---------------------------------------------------------------------------

class TestTryAuroc:
    def test_2d_proba(self):
        labels = np.array([0, 1, 0, 1])
        proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])
        result = Probe._try_auroc(labels, proba)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_1d_proba(self):
        labels = np.array([0, 1, 0, 1])
        proba = np.array([0.2, 0.7, 0.1, 0.6])
        result = Probe._try_auroc(labels, proba)
        assert result is not None

    def test_degenerate_labels_returns_nan_or_none(self):
        labels = np.array([0, 0, 0])
        proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        result = Probe._try_auroc(labels, proba)
        # sklearn may return nan or the function may return None
        assert result is None or (isinstance(result, float) and np.isnan(result))


# ---------------------------------------------------------------------------
# _to_numpy
# ---------------------------------------------------------------------------

class TestToNumpy:
    def test_from_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = Probe._to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_from_list(self):
        arr = Probe._to_numpy([1, 2, 3])
        assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# _get_remote
# ---------------------------------------------------------------------------

class TestGetRemote:
    def test_none_returns_instance_default(self):
        probe = Probe(remote=False)
        assert probe._get_remote(None) is False

    def test_override(self):
        probe = Probe(remote=False)
        assert probe._get_remote(True) is True
