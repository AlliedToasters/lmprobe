"""Tests for MassMeanClassifier Platt scaling calibration.

Verifies that:
- predict() is unchanged (uses raw decision_function threshold at 0)
- predict_proba() returns calibrated probabilities via Platt scaling
- The calibrator is fitted during fit()
- Integration through the LinearProbe API works
"""

import numpy as np
import pytest

from lmprobe.classifiers import MassMeanClassifier


class TestMassMeanCalibration:
    """Unit tests for MassMeanClassifier Platt scaling."""

    @pytest.fixture
    def fitted_classifier(self):
        """Return a fitted MassMeanClassifier with synthetic data."""
        rng = np.random.RandomState(42)
        n_samples = 100
        n_features = 16

        # Create separable data
        X_pos = rng.randn(n_samples // 2, n_features) + 1.0
        X_neg = rng.randn(n_samples // 2, n_features) - 1.0
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

        clf = MassMeanClassifier()
        clf.fit(X, y)
        return clf, X, y

    def test_calibrator_is_fitted(self, fitted_classifier):
        """The calibrator should be fitted during fit()."""
        clf, _, _ = fitted_classifier
        assert clf._calibrator is not None
        assert hasattr(clf._calibrator, "coef_")

    def test_predict_uses_raw_threshold(self, fitted_classifier):
        """predict() should use the raw decision_function threshold at 0."""
        clf, X, _ = fitted_classifier
        scores = clf.decision_function(X)
        expected = (scores > 0).astype(int)
        actual = clf.predict(X)
        np.testing.assert_array_equal(actual, expected)

    def test_predict_proba_returns_valid_probabilities(self, fitted_classifier):
        """predict_proba() should return valid probabilities."""
        clf, X, _ = fitted_classifier
        proba = clf.predict_proba(X)

        # Shape: (n_samples, 2)
        assert proba.shape == (X.shape[0], 2)

        # All values in [0, 1]
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

        # Rows sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_uses_calibrator(self, fitted_classifier):
        """predict_proba() should use the Platt-scaled calibrator, not raw sigmoid."""
        clf, X, _ = fitted_classifier
        scores = clf.decision_function(X)

        # Get calibrated probabilities
        proba = clf.predict_proba(X)

        # Get what the calibrator would produce directly
        expected = clf._calibrator.predict_proba(scores.reshape(-1, 1))

        np.testing.assert_array_equal(proba, expected)

    def test_decision_function_unchanged(self, fitted_classifier):
        """decision_function() should return raw scores, unaffected by calibration."""
        clf, X, _ = fitted_classifier

        # Compute scores
        scores = clf.decision_function(X)

        # Should be raw dot product + intercept
        expected = X @ clf.coef_ + clf.intercept_
        np.testing.assert_allclose(scores, expected)

    def test_predict_accuracy_unchanged_by_calibration(self):
        """Calibration should not change predict() output (same accuracy)."""
        rng = np.random.RandomState(123)
        n_features = 8
        X_pos = rng.randn(30, n_features) + 2.0
        X_neg = rng.randn(30, n_features) - 2.0
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * 30 + [0] * 30)

        clf = MassMeanClassifier()
        clf.fit(X, y)

        # predict() uses raw threshold, not calibrator
        predictions = clf.predict(X)
        scores = clf.decision_function(X)
        expected = (scores > 0).astype(int)
        np.testing.assert_array_equal(predictions, expected)



# Integration tests (mass_mean through LinearProbe API) are covered by
# test_classifiers.py::TestAllClassifiersPredict parametrized over all classifiers.
