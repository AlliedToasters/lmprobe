"""Tests for classifier compatibility, especially those without predict_proba."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from lmprobe import LinearProbe
from lmprobe.classifiers import (
    BUILTIN_CLASSIFIERS,
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

    @pytest.mark.parametrize("classifier", list(BUILTIN_CLASSIFIERS))
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

    @pytest.mark.parametrize("classifier", list(BUILTIN_CLASSIFIERS))
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
