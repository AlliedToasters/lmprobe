"""Tests for Mass-Mean Probing and LDA classifiers."""

import numpy as np
import pytest

from lmprobe.classifiers import MassMeanClassifier, build_classifier


class TestMassMeanClassifier:
    """Tests for MassMeanClassifier."""

    def test_fit_computes_direction(self):
        """Fit computes difference-in-means direction."""
        clf = MassMeanClassifier()

        # Simple 2D data with clear separation
        X = np.array([
            [1, 0],
            [2, 0],
            [3, 0],  # Positive class: mean = [2, 0]
            [-1, 0],
            [-2, 0],
            [-3, 0],  # Negative class: mean = [-2, 0]
        ])
        y = np.array([1, 1, 1, 0, 0, 0])

        clf.fit(X, y)

        assert clf.coef_ is not None
        assert clf.coef_.shape == (2,)
        # Direction should point from negative to positive (along x-axis)
        assert clf.coef_[0] > 0  # Positive x component
        assert abs(clf.coef_[1]) < 0.01  # Near-zero y component

    def test_predict_separates_classes(self):
        """Predict correctly separates linearly separable data."""
        clf = MassMeanClassifier()

        X = np.array([
            [1, 0],
            [2, 0],
            [-1, 0],
            [-2, 0],
        ])
        y = np.array([1, 1, 0, 0])

        clf.fit(X, y)
        predictions = clf.predict(X)

        assert np.array_equal(predictions, y)

    def test_predict_proba_returns_probabilities(self):
        """predict_proba returns valid probabilities."""
        clf = MassMeanClassifier()

        X = np.array([
            [1, 0],
            [2, 0],
            [-1, 0],
            [-2, 0],
        ])
        y = np.array([1, 1, 0, 0])

        clf.fit(X, y)
        probs = clf.predict_proba(X)

        assert probs.shape == (4, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_predict_proba_higher_for_correct_class(self):
        """Probabilities are higher for correct class."""
        clf = MassMeanClassifier()

        X = np.array([
            [5, 0],   # Clearly positive
            [-5, 0],  # Clearly negative
        ])
        y = np.array([1, 0])

        clf.fit(X, y)
        probs = clf.predict_proba(X)

        # First sample should have high P(class=1)
        assert probs[0, 1] > 0.5
        # Second sample should have high P(class=0)
        assert probs[1, 0] > 0.5

    def test_score_computes_accuracy(self):
        """score() returns correct accuracy."""
        clf = MassMeanClassifier()

        X = np.array([
            [1, 0],
            [2, 0],
            [-1, 0],
            [-2, 0],
        ])
        y = np.array([1, 1, 0, 0])

        clf.fit(X, y)
        accuracy = clf.score(X, y)

        assert accuracy == 1.0  # Perfect separation

    def test_decision_function_returns_scores(self):
        """decision_function returns signed scores."""
        clf = MassMeanClassifier()

        X = np.array([
            [1, 0],
            [-1, 0],
        ])
        y = np.array([1, 0])

        clf.fit(X, y)
        scores = clf.decision_function(X)

        assert scores.shape == (2,)
        assert scores[0] > 0  # Positive class should have positive score
        assert scores[1] < 0  # Negative class should have negative score

    def test_requires_both_classes(self):
        """Raises error if only one class present."""
        clf = MassMeanClassifier()

        X = np.array([[1, 0], [2, 0]])
        y = np.array([1, 1])  # Only positive class

        with pytest.raises(ValueError, match="Both classes"):
            clf.fit(X, y)

    def test_unfitted_raises_error(self):
        """Methods raise error before fit."""
        clf = MassMeanClassifier()
        X = np.array([[1, 0]])

        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict(X)

        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.decision_function(X)

    def test_stores_class_means(self):
        """Stores mean vectors for both classes."""
        clf = MassMeanClassifier()

        X = np.array([
            [1, 2],
            [3, 4],  # Positive mean: [2, 3]
            [0, 0],
            [2, 2],  # Negative mean: [1, 1]
        ])
        y = np.array([1, 1, 0, 0])

        clf.fit(X, y)

        assert np.allclose(clf.mean_positive_, [2, 3])
        assert np.allclose(clf.mean_negative_, [1, 1])


class TestBuildClassifierMassMean:
    """Tests for build_classifier with mass_mean and lda."""

    def test_build_mass_mean(self):
        """build_classifier creates MassMeanClassifier."""
        clf = build_classifier("mass_mean")
        assert isinstance(clf, MassMeanClassifier)

    def test_build_lda(self):
        """build_classifier creates LinearDiscriminantAnalysis."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        clf = build_classifier("lda")
        assert isinstance(clf, LinearDiscriminantAnalysis)

    def test_lda_has_predict_proba(self):
        """LDA has predict_proba method."""
        clf = build_classifier("lda")
        assert hasattr(clf, "predict_proba")


class TestLinearProbeWithMassMean:
    """Integration tests for LinearProbe with mass_mean classifier."""

    def test_fit_predict_with_mass_mean(self, tiny_model):
        """LinearProbe works with mass_mean classifier."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="mass_mean",
            device="cpu",
            remote=False,
        )

        probe.fit(["positive example"], ["negative example"])
        predictions = probe.predict(["test"])

        assert predictions.shape == (1,)
        assert predictions[0] in [0, 1]

    def test_predict_proba_with_mass_mean(self, tiny_model):
        """predict_proba works with mass_mean."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="mass_mean",
            device="cpu",
            remote=False,
        )

        probe.fit(["positive"], ["negative"])
        probs = probe.predict_proba(["test"])

        assert probs.shape == (1, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_fit_predict_with_lda(self, tiny_model):
        """LinearProbe works with lda classifier."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="lda",
            device="cpu",
            remote=False,
        )

        # LDA requires more samples than classes
        probe.fit(
            ["positive one", "positive two", "positive three"],
            ["negative one", "negative two", "negative three"],
        )
        predictions = probe.predict(["test"])

        assert predictions.shape == (1,)

    def test_score_with_mass_mean(self, tiny_model):
        """score() works with mass_mean."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="mass_mean",
            device="cpu",
            remote=False,
        )

        probe.fit(["good"], ["bad"])
        accuracy = probe.score(["test"], [1])

        assert 0.0 <= accuracy <= 1.0
