"""Tests for fit_from_activations and related methods."""

import numpy as np
import pytest

from lmprobe import LinearProbe


class TestFitFromActivations:
    def test_classification_basic(self):
        """fit/predict from pre-computed activations works for classification."""
        X_train = np.random.randn(20, 64)
        y_train = np.array([0] * 10 + [1] * 10)

        probe = LinearProbe(classifier="logistic_regression", random_state=42)
        probe.fit_from_activations(X_train, y_train)

        X_test = np.random.randn(5, 64)
        preds = probe.predict_from_activations(X_test)
        assert preds.shape == (5,)

        proba = probe.predict_proba_from_activations(X_test)
        assert proba.shape == (5, 2)

    def test_classification_score(self):
        """score_from_activations returns accuracy."""
        X = np.random.randn(20, 64)
        y = np.array([0] * 10 + [1] * 10)

        probe = LinearProbe(classifier="logistic_regression", random_state=42)
        probe.fit_from_activations(X, y)

        score = probe.score_from_activations(X, y)
        assert 0.0 <= score <= 1.0

    def test_regression_basic(self):
        """Regression task works with fit_from_activations."""
        np.random.seed(42)
        X_train = np.random.randn(50, 32)
        w = np.random.randn(32)
        y_train = X_train @ w + np.random.randn(50) * 0.1

        probe = LinearProbe(task="regression", random_state=42)
        probe.fit_from_activations(X_train, y_train)

        preds = probe.predict_from_activations(X_train)
        assert preds.shape == (50,)

    def test_regression_score(self):
        """Regression score returns R squared."""
        np.random.seed(42)
        X = np.random.randn(50, 32)
        w = np.random.randn(32)
        y = X @ w + np.random.randn(50) * 0.1

        probe = LinearProbe(task="regression", random_state=42)
        probe.fit_from_activations(X, y)

        r2 = probe.score_from_activations(X, y)
        assert r2 > 0.9  # Should fit well with low noise

    def test_regression_predict_proba_raises(self):
        """predict_proba raises error for regression task."""
        X = np.random.randn(20, 32)
        y = np.random.randn(20)

        probe = LinearProbe(task="regression", random_state=42)
        probe.fit_from_activations(X, y)

        with pytest.raises(ValueError, match="regression"):
            probe.predict_proba_from_activations(X)

    def test_torch_tensor_input(self):
        """Torch tensors are accepted as input."""
        import torch

        X = torch.randn(20, 64)
        y = torch.tensor([0] * 10 + [1] * 10)

        probe = LinearProbe(classifier="logistic_regression", random_state=42)
        probe.fit_from_activations(X, y)

        preds = probe.predict_from_activations(X)
        assert isinstance(preds, np.ndarray)

    def test_fit_without_model_raises(self):
        """Calling fit() without model raises clear error."""
        probe = LinearProbe(random_state=42)
        with pytest.raises(ValueError, match="model"):
            probe.fit(["hello"], [1])

    def test_classes_attribute(self):
        """classes_ is set after classification fit."""
        X = np.random.randn(20, 64)
        y = np.array([0] * 10 + [1] * 10)

        probe = LinearProbe(random_state=42)
        probe.fit_from_activations(X, y)

        assert hasattr(probe, "classes_")
        assert len(probe.classes_) == 2
