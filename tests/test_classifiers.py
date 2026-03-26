"""Tests for classifier compatibility, especially those without predict_proba."""

import numpy as np
import pytest
from sklearn.base import clone

from lmprobe import LinearProbe
from lmprobe.classifiers import (
    BUILTIN_CLASSIFIERS,
    CLASSIFICATION_CLASSIFIERS,
    EnsembleClassifier,
    MassMeanClassifier,
    SGDGPUClassifier,
    _stable_sigmoid_proba,
    build_classifier,
    resolve_classifier,
    validate_classifier,
)

pytestmark = pytest.mark.nnsight


class TestBuiltinClassifiers:
    """Tests for all built-in classifiers."""

    def test_all_builtin_classifiers_can_be_built(self):
        """Every builtin classifier can be instantiated."""
        for name in BUILTIN_CLASSIFIERS:
            clf = build_classifier(name)
            assert clf is not None
            assert hasattr(clf, "fit")
            assert hasattr(clf, "predict")

    def test_unknown_classifier_raises(self):
        """Unknown classifier name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown classifier"):
            build_classifier("nonexistent_classifier")

    @pytest.mark.parametrize(
        "name", ["logistic_regression", "svm", "sgd", "sgd_gpu", "mass_mean", "lda"],
    )
    def test_classifiers_with_predict_proba(self, name):
        """These classifiers support predict_proba."""
        clf = build_classifier(name)
        assert hasattr(clf, "predict_proba")

    def test_ridge_lacks_predict_proba(self):
        """RidgeClassifier does not have predict_proba."""
        clf = build_classifier("ridge")
        assert not hasattr(clf, "predict_proba")


class TestValidateClassifier:
    """Tests for classifier validation."""

    def test_warns_on_missing_predict_proba(self):
        """Warns when classifier lacks predict_proba."""
        clf = build_classifier("ridge")
        with pytest.warns(UserWarning, match="does not support predict_proba"):
            validate_classifier(clf)

    def test_no_warning_with_predict_proba(self):
        """No warning for classifiers with predict_proba."""
        import warnings

        clf = build_classifier("logistic_regression")
        # Should not warn about predict_proba
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            validate_classifier(clf)
        # Filter to only UserWarnings about predict_proba
        proba_warnings = [w for w in record if "predict_proba" in str(w.message)]
        assert len(proba_warnings) == 0

    def test_raises_on_missing_fit(self):
        """Raises TypeError if classifier lacks fit()."""
        class NoFit:
            def predict(self, X):
                pass

        with pytest.raises(TypeError, match="must have a fit"):
            validate_classifier(NoFit())

    def test_raises_on_missing_predict(self):
        """Raises TypeError if classifier lacks predict()."""
        class NoPredict:
            def fit(self, X, y):
                pass

        with pytest.raises(TypeError, match="must have a predict"):
            validate_classifier(NoPredict())


class TestLinearProbeWithRidge:
    """Tests for LinearProbe with RidgeClassifier (no predict_proba)."""

    def test_ridge_fit_works(self, tiny_model):
        """RidgeClassifier can be used for training."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="ridge",
            device="cpu",
            remote=False,
            random_state=42,
        )

        # Should succeed without error
        probe.fit(["positive one", "positive two"], ["negative one", "negative two"])
        assert probe.classifier_ is not None

    def test_ridge_predict_works(self, tiny_model):
        """predict() works with RidgeClassifier."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="ridge",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive"], ["negative"])
        predictions = probe.predict(["test input"])

        assert predictions.shape == (1,)
        assert predictions[0] in [0, 1]

    def test_ridge_score_works(self, tiny_model):
        """score() works with RidgeClassifier."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="ridge",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive"], ["negative"])
        accuracy = probe.score(["test one", "test two"], [1, 0])

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_ridge_predict_proba_raises(self, tiny_model):
        """predict_proba() raises error with RidgeClassifier."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="ridge",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive"], ["negative"])

        with pytest.raises(AttributeError):
            probe.predict_proba(["test input"])

    def test_ridge_multiple_predictions(self, tiny_model):
        """predict() handles multiple samples with RidgeClassifier."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="ridge",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(
            ["positive one", "positive two", "positive three"],
            ["negative one", "negative two", "negative three"],
        )
        predictions = probe.predict(["test one", "test two", "test three"])

        assert predictions.shape == (3,)
        assert all(p in [0, 1] for p in predictions)

    def test_ridge_save_load(self, tiny_model, tmp_path):
        """save/load works with RidgeClassifier."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="ridge",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive"], ["negative"])
        original_pred = probe.predict(["test"])

        save_path = tmp_path / "ridge_probe.pkl"
        probe.save(str(save_path))

        loaded = LinearProbe.load(str(save_path))
        loaded_pred = loaded.predict(["test"])

        assert np.array_equal(original_pred, loaded_pred)


class TestCustomClassifierWithoutProba:
    """Tests for custom classifiers without predict_proba."""

    def test_custom_classifier_without_proba(self, tiny_model):
        """Custom classifier without predict_proba works for predict()."""
        from sklearn.linear_model import Perceptron

        # Perceptron doesn't have predict_proba by default
        clf = Perceptron(random_state=42)

        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier=clf,
            device="cpu",
            remote=False,
        )

        probe.fit(["positive"], ["negative"])
        predictions = probe.predict(["test input"])

        assert predictions.shape == (1,)
        assert predictions[0] in [0, 1]

    def test_custom_classifier_score_works(self, tiny_model):
        """score() works with custom classifier without predict_proba."""
        from sklearn.linear_model import Perceptron

        clf = Perceptron(random_state=42)

        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier=clf,
            device="cpu",
            remote=False,
        )

        probe.fit(["positive"], ["negative"])
        accuracy = probe.score(["test"], [1])

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0


class TestPerTokenPredictWithoutProba:
    """Tests for per-token prediction with classifiers lacking predict_proba."""

    def test_ridge_with_inference_pooling_all(self, tiny_model):
        """predict() with inference_pooling='all' works for ridge."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="ridge",
            pooling="last_token",
            inference_pooling="all",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive example"], ["negative example"])

        # Should work and return aggregated predictions
        predictions = probe.predict(["test with multiple tokens"])

        assert predictions.shape == (1,)
        assert predictions[0] in [0, 1]

    def test_ridge_with_different_pooling(self, tiny_model):
        """predict() works with different pooling strategies for ridge."""
        for pooling in ["last_token", "mean"]:
            probe = LinearProbe(
                model=tiny_model,
                layers=-1,
                classifier="ridge",
                pooling=pooling,
                device="cpu",
                remote=False,
                random_state=42,
            )

            probe.fit(["positive"], ["negative"])
            predictions = probe.predict(["test"])

            assert predictions.shape == (1,)
            assert predictions[0] in [0, 1]


class TestAllClassifiersPredict:
    """Parametrized tests to ensure all classifiers work with predict()."""

    @pytest.mark.parametrize("classifier", list(CLASSIFICATION_CLASSIFIERS))
    def test_predict_works_for_all_builtin_classifiers(self, tiny_model, classifier):
        """Every builtin classifier works with predict()."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier=classifier,
            device="cpu",
            remote=False,
            random_state=42,
        )

        # Some classifiers need more samples
        # LDA needs more samples than classes
        # LogisticRegressionCV uses 5-fold CV so needs at least 5 samples per class
        if classifier in ("lda", "logistic_regression_cv"):
            pos = [f"positive {i}" for i in range(5)]
            neg = [f"negative {i}" for i in range(5)]
        else:
            pos = ["positive"]
            neg = ["negative"]

        probe.fit(pos, neg)
        predictions = probe.predict(["test"])

        assert predictions.shape == (1,)
        assert predictions[0] in [0, 1]

    @pytest.mark.parametrize("classifier", list(CLASSIFICATION_CLASSIFIERS))
    def test_score_works_for_all_builtin_classifiers(self, tiny_model, classifier):
        """Every builtin classifier works with score()."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier=classifier,
            device="cpu",
            remote=False,
            random_state=42,
        )

        # Some classifiers need more samples
        # LDA needs more samples than classes
        # LogisticRegressionCV uses 5-fold CV so needs at least 5 samples per class
        if classifier in ("lda", "logistic_regression_cv"):
            pos = [f"positive {i}" for i in range(5)]
            neg = [f"negative {i}" for i in range(5)]
        else:
            pos = ["positive"]
            neg = ["negative"]

        probe.fit(pos, neg)
        accuracy = probe.score(["test"], [1])

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0


class TestStableSigmoidProba:
    """Tests for _stable_sigmoid_proba."""

    def test_zero_scores(self):
        scores = np.array([0.0])
        proba = _stable_sigmoid_proba(scores)
        assert proba.shape == (1, 2)
        assert proba[0, 0] == pytest.approx(0.5)
        assert proba[0, 1] == pytest.approx(0.5)

    def test_positive_scores(self):
        scores = np.array([10.0])
        proba = _stable_sigmoid_proba(scores)
        assert proba[0, 1] > 0.99

    def test_negative_scores(self):
        scores = np.array([-10.0])
        proba = _stable_sigmoid_proba(scores)
        assert proba[0, 0] > 0.99

    def test_large_positive(self):
        """No overflow for very large positive scores."""
        scores = np.array([500.0])
        proba = _stable_sigmoid_proba(scores)
        assert np.isfinite(proba).all()
        assert proba[0, 1] == pytest.approx(1.0, abs=1e-10)

    def test_large_negative(self):
        """No overflow for very large negative scores."""
        scores = np.array([-500.0])
        proba = _stable_sigmoid_proba(scores)
        assert np.isfinite(proba).all()
        assert proba[0, 0] == pytest.approx(1.0, abs=1e-10)

    def test_probabilities_sum_to_one(self):
        scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        proba = _stable_sigmoid_proba(scores)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestMassMeanClassifier:
    """Tests for MassMeanClassifier."""

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        X_pos = rng.normal(1.0, 0.5, (20, 4))
        X_neg = rng.normal(-1.0, 0.5, (20, 4))
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * 20 + [0] * 20)
        return X, y

    def test_fit_predict(self, data):
        X, y = data
        clf = MassMeanClassifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (40,)
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, data):
        X, y = data
        clf = MassMeanClassifier()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (40, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_decision_function(self, data):
        X, y = data
        clf = MassMeanClassifier()
        clf.fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (40,)

    def test_decision_function_unfitted_raises(self):
        clf = MassMeanClassifier()
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.decision_function(np.array([[1, 2]]))

    def test_score(self, data):
        X, y = data
        clf = MassMeanClassifier()
        clf.fit(X, y)
        acc = clf.score(X, y)
        assert 0.0 <= acc <= 1.0

    def test_get_params(self):
        clf = MassMeanClassifier()
        assert clf.get_params() == {}

    def test_set_params(self):
        clf = MassMeanClassifier()
        result = clf.set_params(anything="ignored")
        assert result is clf

    def test_single_class_raises(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 1])
        clf = MassMeanClassifier()
        with pytest.raises(ValueError, match="Both classes must have"):
            clf.fit(X, y)


class TestEnsembleClassifier:
    """Tests for EnsembleClassifier."""

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (30, 4))
        y = np.array([0] * 15 + [1] * 15)
        return X, y

    def test_fit_predict(self, data):
        X, y = data
        clf = EnsembleClassifier(random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (30,)

    def test_predict_proba(self, data):
        X, y = data
        clf = EnsembleClassifier(random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (30, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_predict_proba_unfitted_raises(self):
        clf = EnsembleClassifier()
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict_proba(np.array([[1, 2]]))

    def test_score(self, data):
        X, y = data
        clf = EnsembleClassifier(random_state=42)
        clf.fit(X, y)
        acc = clf.score(X, y)
        assert 0.0 <= acc <= 1.0

    def test_custom_c_values(self, data):
        X, y = data
        clf = EnsembleClassifier(C_values=[0.1, 1.0], random_state=42)
        clf.fit(X, y)
        assert len(clf.estimators_) == 2

    def test_coef_and_intercept(self, data):
        X, y = data
        clf = EnsembleClassifier(random_state=42)
        clf.fit(X, y)
        assert clf.coef_ is not None
        assert clf.intercept_ is not None
        assert clf.classes_ is not None

    def test_get_params(self):
        clf = EnsembleClassifier(C_values=[0.1], solver="saga", max_iter=500, random_state=1)
        params = clf.get_params()
        assert params["C_values"] == [0.1]
        assert params["solver"] == "saga"
        assert params["max_iter"] == 500
        assert params["random_state"] == 1

    def test_set_params(self):
        clf = EnsembleClassifier()
        clf.set_params(solver="saga")
        assert clf.solver == "saga"


class TestResolveClassifier:
    """Tests for resolve_classifier."""

    def test_resolve_string(self):
        clf = resolve_classifier("logistic_regression", random_state=42)
        assert hasattr(clf, "fit")

    def test_resolve_custom_estimator(self):
        from sklearn.linear_model import LogisticRegression
        custom = LogisticRegression()
        clf = resolve_classifier(custom)
        assert clf is custom

    def test_resolve_with_kwargs(self):
        clf = resolve_classifier(
            "logistic_regression", random_state=42,
            classifier_kwargs={"C": 0.01},
        )
        assert clf.C == 0.01

    def test_resolve_warns_for_no_predict_proba(self):
        with pytest.warns(UserWarning, match="does not support predict_proba"):
            resolve_classifier("ridge")


class TestBuildClassifierKwargs:
    """Test classifier_kwargs forwarding for each built-in classifier."""

    def test_logistic_regression_cv_kwargs(self):
        clf = build_classifier("logistic_regression_cv", classifier_kwargs={"cv": 3})
        assert clf.cv == 3

    def test_ridge_regression(self):
        clf = build_classifier("ridge_regression", classifier_kwargs={"alpha": 2.0})
        assert clf.alpha == 2.0

    def test_svm_kwargs(self):
        clf = build_classifier("svm", classifier_kwargs={"C": 0.5})
        assert clf.C == 0.5

    def test_sgd_kwargs(self):
        clf = build_classifier("sgd", random_state=42)
        assert clf.random_state == 42

    def test_ensemble_kwargs(self):
        clf = build_classifier("ensemble", classifier_kwargs={"C_values": [0.1, 1.0]})
        assert clf.C_values == [0.1, 1.0]

    def test_lda_kwargs(self):
        clf = build_classifier("lda")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        assert isinstance(clf, LinearDiscriminantAnalysis)

    def test_sgd_gpu_kwargs(self):
        clf = build_classifier("sgd_gpu", random_state=42, classifier_kwargs={"lr": 0.1})
        assert isinstance(clf, SGDGPUClassifier)
        assert clf.lr == 0.1
        assert clf.random_state == 42


class TestSGDGPUClassifier:
    """Tests for SGDGPUClassifier (GPU-accelerated SGD solver)."""

    @pytest.fixture
    def data(self):
        """Linearly separable synthetic data."""
        rng = np.random.default_rng(42)
        X_pos = rng.normal(1.0, 0.5, (50, 8))
        X_neg = rng.normal(-1.0, 0.5, (50, 8))
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * 50 + [0] * 50)
        return X, y

    def test_build_sgd_gpu(self):
        """build_classifier('sgd_gpu') returns an SGDGPUClassifier."""
        clf = build_classifier("sgd_gpu", random_state=42)
        assert isinstance(clf, SGDGPUClassifier)

    def test_fit_predict(self, data):
        X, y = data
        clf = SGDGPUClassifier(device="cpu", random_state=42, epochs=50)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, data):
        X, y = data
        clf = SGDGPUClassifier(device="cpu", random_state=42, epochs=50)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        # Probabilities should be between 0 and 1
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_decision_function(self, data):
        X, y = data
        clf = SGDGPUClassifier(device="cpu", random_state=42, epochs=50)
        clf.fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (100,)

    def test_score(self, data):
        X, y = data
        clf = SGDGPUClassifier(device="cpu", random_state=42, epochs=100)
        clf.fit(X, y)
        acc = clf.score(X, y)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
        # Separable data — should get high accuracy
        assert acc > 0.8

    def test_sample_weight(self, data):
        X, y = data
        weights = np.ones(100)
        weights[:50] = 2.0  # Upweight positives
        clf = SGDGPUClassifier(device="cpu", random_state=42, epochs=50)
        clf.fit(X, y, sample_weight=weights)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_sklearn_clone(self):
        """sklearn.base.clone works correctly."""
        clf = SGDGPUClassifier(lr=0.05, epochs=200, device="cpu", random_state=7)
        cloned = clone(clf)
        assert isinstance(cloned, SGDGPUClassifier)
        assert cloned.lr == 0.05
        assert cloned.epochs == 200
        assert cloned.random_state == 7
        # Cloned should not be fitted
        assert not hasattr(cloned, "coef_") or cloned.coef_ is None

    def test_device_fallback_to_cpu(self, data):
        """device='auto' works even without GPU (falls back to CPU)."""
        X, y = data
        clf = SGDGPUClassifier(device="auto", random_state=42, epochs=50)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_coef_intercept_on_cpu(self, data):
        """After fit, coef_ and intercept_ are numpy arrays, not GPU tensors."""
        X, y = data
        clf = SGDGPUClassifier(device="cpu", random_state=42, epochs=50)
        clf.fit(X, y)
        assert isinstance(clf.coef_, np.ndarray)
        assert isinstance(clf.intercept_, np.ndarray)
        assert clf.coef_.shape == (8,)
        assert clf.intercept_.shape == (1,)
        assert clf.classes_ is not None
        assert set(clf.classes_) == {0, 1}

    def test_unfitted_raises(self):
        """predict/predict_proba before fit raises RuntimeError."""
        clf = SGDGPUClassifier()
        X = np.array([[1.0, 2.0]])
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict(X)
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict_proba(X)
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.decision_function(X)

    def test_get_params(self):
        clf = SGDGPUClassifier(lr=0.05, epochs=200, batch_size=128,
                               weight_decay=1e-3, device="cuda:1", random_state=7)
        params = clf.get_params()
        assert params["lr"] == 0.05
        assert params["epochs"] == 200
        assert params["batch_size"] == 128
        assert params["weight_decay"] == 1e-3
        assert params["device"] == "cuda:1"
        assert params["random_state"] == 7

    def test_set_params(self):
        clf = SGDGPUClassifier()
        result = clf.set_params(lr=0.1, epochs=500)
        assert result is clf
        assert clf.lr == 0.1
        assert clf.epochs == 500

    @pytest.mark.gpu
    def test_fit_on_gpu(self, data):
        """Train on actual GPU device. Requires compatible CUDA GPU."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        # Verify GPU can actually run kernels
        try:
            t = torch.tensor([1.0], device="cuda")
            _ = t + t
            del t
        except (RuntimeError, torch.cuda.CudaError):
            pytest.skip("CUDA device incompatible with this PyTorch build")

        X, y = data
        clf = SGDGPUClassifier(device="cuda", random_state=42, epochs=100)
        clf.fit(X, y)

        # Weights should be on CPU after fit
        assert isinstance(clf.coef_, np.ndarray)
        assert isinstance(clf.intercept_, np.ndarray)

        acc = clf.score(X, y)
        assert acc > 0.8

        # GPU memory should be cleaned up (model deleted)
        torch.cuda.empty_cache()  # force cleanup
        # No assertion on exact memory — just verify no crash

    # --- LR scheduler tests (#228) ---

    def test_cosine_scheduler(self, data):
        """Cosine scheduler runs without error and tracks loss."""
        X, y = data
        clf = SGDGPUClassifier(
            device="cpu", random_state=42, epochs=20, scheduler="cosine",
        )
        clf.fit(X, y)
        assert clf.coef_ is not None
        assert len(clf.train_loss_) == 20

    def test_reduce_on_plateau_scheduler(self, data):
        """ReduceOnPlateau scheduler runs without error."""
        X, y = data
        clf = SGDGPUClassifier(
            device="cpu", random_state=42, epochs=20,
            scheduler="reduce_on_plateau",
        )
        clf.fit(X, y)
        assert clf.coef_ is not None
        assert len(clf.train_loss_) == 20

    def test_no_scheduler_default(self, data):
        """Default (None) scheduler works — backward compatible."""
        X, y = data
        clf = SGDGPUClassifier(device="cpu", random_state=42, epochs=10)
        clf.fit(X, y)
        assert len(clf.train_loss_) == 10

    def test_invalid_scheduler_raises(self):
        """Unknown scheduler name raises ValueError at init."""
        with pytest.raises(ValueError, match="Unknown scheduler"):
            SGDGPUClassifier(scheduler="invalid")

    # --- Convergence monitoring & early stopping tests (#229) ---

    def test_train_loss_tracked(self, data):
        """train_loss_ is a list of floats with length == epochs."""
        X, y = data
        clf = SGDGPUClassifier(device="cpu", random_state=42, epochs=15)
        clf.fit(X, y)
        assert isinstance(clf.train_loss_, list)
        assert len(clf.train_loss_) == 15
        assert all(isinstance(v, float) for v in clf.train_loss_)

    def test_verbose_prints(self, data, capsys):
        """verbose=True produces output to stdout."""
        X, y = data
        clf = SGDGPUClassifier(
            device="cpu", random_state=42, epochs=5, verbose=True,
        )
        clf.fit(X, y)
        captured = capsys.readouterr()
        assert "Epoch" in captured.out
        assert "loss=" in captured.out

    def test_early_stopping_stops_early(self):
        """Early stopping terminates before max epochs when loss plateaus."""
        # Constant features = zero gradient after first step → loss never improves
        X = np.ones((100, 8), dtype=np.float32)
        y = np.array([1] * 50 + [0] * 50, dtype=np.float32)
        clf = SGDGPUClassifier(
            device="cpu", random_state=42, epochs=500,
            early_stopping=5, lr=0.01,
        )
        clf.fit(X, y)
        # Should have stopped well before 500 epochs (loss can't improve)
        assert len(clf.train_loss_) < 500

    def test_early_stopping_restores_best(self, data):
        """Early stopping restores best weights."""
        X, y = data
        clf = SGDGPUClassifier(
            device="cpu", random_state=42, epochs=200, early_stopping=10,
        )
        clf.fit(X, y)
        # Should still produce valid predictions
        preds = clf.predict(X)
        assert preds.shape == (100,)
        assert set(preds).issubset({0, 1})

    def test_early_stopping_verbose(self, capsys):
        """Early stopping with verbose prints the stopping message."""
        X = np.ones((100, 8), dtype=np.float32)
        y = np.array([1] * 50 + [0] * 50, dtype=np.float32)
        clf = SGDGPUClassifier(
            device="cpu", random_state=42, epochs=500,
            early_stopping=5, verbose=True, lr=0.01,
        )
        clf.fit(X, y)
        captured = capsys.readouterr()
        assert "Early stopping" in captured.out

    # --- get_params / clone with new params ---

    def test_get_params_includes_new(self):
        """get_params includes scheduler, verbose, early_stopping."""
        clf = SGDGPUClassifier(
            scheduler="cosine", verbose=True, early_stopping=10,
        )
        params = clf.get_params()
        assert params["scheduler"] == "cosine"
        assert params["verbose"] is True
        assert params["early_stopping"] == 10

    def test_sklearn_clone_new_params(self):
        """sklearn clone preserves new parameters."""
        clf = SGDGPUClassifier(
            scheduler="reduce_on_plateau", verbose=True,
            early_stopping=7, device="cpu", random_state=42,
        )
        cloned = clone(clf)
        assert isinstance(cloned, SGDGPUClassifier)
        assert cloned.scheduler == "reduce_on_plateau"
        assert cloned.verbose is True
        assert cloned.early_stopping == 7

    def test_build_classifier_passes_scheduler(self):
        """build_classifier forwards scheduler kwarg."""
        clf = build_classifier(
            "sgd_gpu", random_state=42,
            classifier_kwargs={"scheduler": "cosine", "early_stopping": 5},
        )
        assert isinstance(clf, SGDGPUClassifier)
        assert clf.scheduler == "cosine"
        assert clf.early_stopping == 5
