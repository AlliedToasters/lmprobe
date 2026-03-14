"""Tests for the EnsembleClassifier (averaged logistic regressions)."""

import numpy as np
import pytest

from lmprobe import LinearProbe
from lmprobe.classifiers import EnsembleClassifier, build_classifier


class TestEnsembleClassifierUnit:
    """Unit tests for EnsembleClassifier with synthetic data."""

    def test_fit_and_predict(self):
        """EnsembleClassifier fits and predicts on synthetic data."""
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        y = (X[:, 0] > 0).astype(int)

        clf = EnsembleClassifier(random_state=42)
        clf.fit(X, y)

        predictions = clf.predict(X)
        assert predictions.shape == (20,)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba_returns_averaged_probabilities(self):
        """predict_proba returns mean of all models' probabilities."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(20, 5)
        y_train = (X_train[:, 0] > 0).astype(int)
        X_test = rng.randn(5, 5)

        clf = EnsembleClassifier(random_state=42)
        clf.fit(X_train, y_train)

        probas = clf.predict_proba(X_test)
        assert probas.shape == (5, 2)

        # Probabilities should sum to 1 for each sample
        row_sums = probas.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

        # Verify it is actually the mean of individual models
        individual_probas = np.array(
            [est.predict_proba(X_test) for est in clf.estimators_]
        )
        expected = individual_probas.mean(axis=0)
        np.testing.assert_allclose(probas, expected, atol=1e-10)

    def test_predict_thresholds_at_half(self):
        """predict() thresholds averaged probabilities at 0.5."""
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        y = (X[:, 0] > 0).astype(int)

        clf = EnsembleClassifier(random_state=42)
        clf.fit(X, y)

        probas = clf.predict_proba(X)
        predictions = clf.predict(X)

        expected = (probas[:, 1] >= 0.5).astype(int)
        np.testing.assert_array_equal(predictions, expected)

    def test_default_c_values(self):
        """Default C_values are used when none are provided."""
        clf = EnsembleClassifier()
        assert clf.C_values == [0.01, 0.1, 0.5, 1.0, 5.0]

    def test_custom_c_values(self):
        """Custom C_values are stored and used."""
        custom = [0.001, 10.0]
        clf = EnsembleClassifier(C_values=custom)
        assert clf.C_values == custom

        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        y = (X[:, 0] > 0).astype(int)
        clf.fit(X, y)

        assert len(clf.estimators_) == 2
        assert clf.estimators_[0].C == 0.001
        assert clf.estimators_[1].C == 10.0

    def test_score(self):
        """score() computes accuracy correctly."""
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        y = (X[:, 0] > 0).astype(int)

        clf = EnsembleClassifier(random_state=42)
        clf.fit(X, y)

        accuracy = clf.score(X, y)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_fitted_attributes(self):
        """After fit, classes_, coef_, intercept_, estimators_ are set."""
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        y = (X[:, 0] > 0).astype(int)

        clf = EnsembleClassifier(random_state=42)
        clf.fit(X, y)

        assert clf.classes_ is not None
        np.testing.assert_array_equal(clf.classes_, [0, 1])
        assert clf.coef_ is not None
        assert clf.coef_.shape == (1, 5)
        assert clf.intercept_ is not None
        assert len(clf.estimators_) == 5

    def test_get_params(self):
        """get_params returns constructor parameters."""
        clf = EnsembleClassifier(
            C_values=[0.1, 1.0],
            solver="liblinear",
            max_iter=500,
            random_state=123,
        )
        params = clf.get_params()
        assert params == {
            "C_values": [0.1, 1.0],
            "solver": "liblinear",
            "max_iter": 500,
            "random_state": 123,
        }

    def test_set_params(self):
        """set_params updates parameters."""
        clf = EnsembleClassifier()
        clf.set_params(solver="liblinear", max_iter=500)
        assert clf.solver == "liblinear"
        assert clf.max_iter == 500

    def test_unfitted_predict_proba_raises(self):
        """predict_proba raises RuntimeError before fit."""
        clf = EnsembleClassifier()
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict_proba(np.zeros((1, 5)))


class TestEnsembleBuildClassifier:
    """Test that ensemble is properly registered in build_classifier."""

    def test_build_ensemble_default(self):
        """build_classifier('ensemble') returns EnsembleClassifier."""
        clf = build_classifier("ensemble", random_state=42)
        assert isinstance(clf, EnsembleClassifier)
        assert clf.C_values == [0.01, 0.1, 0.5, 1.0, 5.0]
        assert clf.random_state == 42

    def test_build_ensemble_custom_c_values(self):
        """build_classifier passes C_values from classifier_kwargs."""
        clf = build_classifier(
            "ensemble",
            random_state=42,
            classifier_kwargs={"C_values": [0.1, 1.0], "solver": "liblinear"},
        )
        assert isinstance(clf, EnsembleClassifier)
        assert clf.C_values == [0.1, 1.0]
        assert clf.solver == "liblinear"


class TestEnsembleWithLinearProbe:
    """Integration tests: ensemble classifier through the LinearProbe API."""

    def test_ensemble_fit_predict(self, tiny_model):
        """Ensemble classifier works through LinearProbe fit/predict."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="ensemble",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(
            ["positive one", "positive two"],
            ["negative one", "negative two"],
        )
        predictions = probe.predict(["test input"])

        assert predictions.shape == (1,)
        assert predictions[0] in [0, 1]

    def test_ensemble_predict_proba(self, tiny_model):
        """predict_proba works through LinearProbe with ensemble."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="ensemble",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(
            ["positive one", "positive two"],
            ["negative one", "negative two"],
        )
        probas = probe.predict_proba(["test one", "test two"])

        assert probas.shape == (2, 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-10)

    def test_ensemble_score(self, tiny_model):
        """score() works through LinearProbe with ensemble."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="ensemble",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(
            ["positive one", "positive two"],
            ["negative one", "negative two"],
        )
        accuracy = probe.score(["test one", "test two"], [1, 0])

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_ensemble_with_custom_kwargs(self, tiny_model):
        """Custom C_values and solver work through classifier_kwargs."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="ensemble",
            classifier_kwargs={
                "C_values": [0.01, 0.1, 0.5, 1.0, 5.0],
                "solver": "liblinear",
            },
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(
            ["positive one", "positive two"],
            ["negative one", "negative two"],
        )
        predictions = probe.predict(["test input"])

        assert predictions.shape == (1,)
        assert isinstance(probe.classifier_, EnsembleClassifier)
        assert probe.classifier_.solver == "liblinear"
        assert len(probe.classifier_.estimators_) == 5
