"""Tests for classifier compatibility, especially those without predict_proba."""

import numpy as np
import pytest

from lmprobe import LinearProbe
from lmprobe.classifiers import (
    BUILTIN_CLASSIFIERS,
    CLASSIFICATION_CLASSIFIERS,
    EnsembleClassifier,
    MassMeanClassifier,
    _stable_sigmoid_proba,
    build_classifier,
    resolve_classifier,
    validate_classifier,
)


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

    @pytest.mark.parametrize("name", ["logistic_regression", "svm", "sgd", "mass_mean", "lda"])
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
