"""Tests for cuML compute backend integration.

These tests verify that:
- cuml_available() returns a bool
- _resolve_compute_backend() resolves correctly
- build_classifier() respects compute_backend
- Probe(compute_backend=...) threads through correctly
- Everything works when cuML is NOT installed (the common case)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lmprobe.classifiers import (
    _CUML_SUPPORTED_CLASSIFIERS,
    _resolve_compute_backend,
    build_classifier,
    cuml_available,
    resolve_classifier,
)

# ---------------------------------------------------------------------------
# cuml_available()
# ---------------------------------------------------------------------------


class TestCumlAvailable:
    def test_returns_bool(self):
        result = cuml_available()
        assert isinstance(result, bool)

    def test_returns_false_when_not_installed(self):
        # Clear the lru_cache so we can test fresh
        cuml_available.cache_clear()
        # cuML is not installed in test env
        assert cuml_available() is False


# ---------------------------------------------------------------------------
# _resolve_compute_backend()
# ---------------------------------------------------------------------------


class TestResolveComputeBackend:
    def test_sklearn_passthrough(self):
        assert _resolve_compute_backend("sklearn") == "sklearn"

    def test_auto_without_cuml(self):
        cuml_available.cache_clear()
        assert _resolve_compute_backend("auto") == "sklearn"

    def test_auto_with_cuml(self):
        cuml_available.cache_clear()
        with patch("lmprobe.classifiers.cuml_available", return_value=True):
            assert _resolve_compute_backend("auto") == "cuml"

    def test_cuml_raises_when_not_installed(self):
        cuml_available.cache_clear()
        with pytest.raises(ImportError, match="compute_backend='cuml' requires cuML"):
            _resolve_compute_backend("cuml")

    def test_cuml_passes_when_installed(self):
        cuml_available.cache_clear()
        with patch("lmprobe.classifiers.cuml_available", return_value=True):
            assert _resolve_compute_backend("cuml") == "cuml"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown compute_backend"):
            _resolve_compute_backend("tensorflow")


# ---------------------------------------------------------------------------
# build_classifier() with compute_backend
# ---------------------------------------------------------------------------


class TestBuildClassifierComputeBackend:
    def test_sklearn_default(self):
        """Default compute_backend='sklearn' produces sklearn classifiers."""
        clf = build_classifier("logistic_regression", random_state=42)
        assert type(clf).__module__.startswith("sklearn")

    def test_sklearn_explicit(self):
        clf = build_classifier(
            "logistic_regression", random_state=42, compute_backend="sklearn",
        )
        assert type(clf).__module__.startswith("sklearn")

    def test_cuml_backend_uses_cuml_classes(self):
        """When cuML is available and compute_backend='cuml', cuML classes are used."""
        mock_cuml = MagicMock()
        mock_lr = MagicMock()
        mock_cuml.linear_model.LogisticRegression.return_value = mock_lr

        with patch.dict("sys.modules", {"cuml": mock_cuml}):
            clf = build_classifier(
                "logistic_regression", random_state=42, compute_backend="cuml",
            )
            assert clf is mock_lr
            mock_cuml.linear_model.LogisticRegression.assert_called_once()

    def test_cuml_backend_ridge(self):
        """cuML Ridge is used for ridge_regression."""
        mock_cuml = MagicMock()
        mock_ridge = MagicMock()
        mock_cuml.linear_model.Ridge.return_value = mock_ridge

        with patch.dict("sys.modules", {"cuml": mock_cuml}):
            clf = build_classifier(
                "ridge_regression", random_state=42, compute_backend="cuml",
            )
            assert clf is mock_ridge

    def test_cuml_backend_svm(self):
        """cuML SVC is used for svm."""
        mock_cuml = MagicMock()
        mock_svc = MagicMock()
        mock_cuml.svm.SVC.return_value = mock_svc

        with patch.dict("sys.modules", {"cuml": mock_cuml}):
            clf = build_classifier(
                "svm", random_state=42, compute_backend="cuml",
            )
            assert clf is mock_svc

    def test_cuml_backend_falls_back_for_unsupported(self):
        """Classifiers without cuML equivalents fall back to sklearn."""
        # sgd has no cuML equivalent — should still return sklearn SGDClassifier
        clf = build_classifier("sgd", random_state=42, compute_backend="cuml")
        assert type(clf).__module__.startswith("sklearn")

    def test_cuml_supported_classifiers_set(self):
        """Verify the set of cuML-supported classifiers."""
        assert "logistic_regression" in _CUML_SUPPORTED_CLASSIFIERS
        assert "ridge" in _CUML_SUPPORTED_CLASSIFIERS
        assert "ridge_regression" in _CUML_SUPPORTED_CLASSIFIERS
        assert "svm" in _CUML_SUPPORTED_CLASSIFIERS
        # These should NOT be in the set
        assert "sgd" not in _CUML_SUPPORTED_CLASSIFIERS
        assert "mass_mean" not in _CUML_SUPPORTED_CLASSIFIERS
        assert "lda" not in _CUML_SUPPORTED_CLASSIFIERS


# ---------------------------------------------------------------------------
# resolve_classifier() with compute_backend
# ---------------------------------------------------------------------------


class TestResolveClassifierComputeBackend:
    def test_sklearn_default(self):
        clf = resolve_classifier("logistic_regression", random_state=42)
        assert type(clf).__module__.startswith("sklearn")

    def test_passes_compute_backend_through(self):
        """resolve_classifier passes compute_backend to build_classifier."""
        mock_cuml = MagicMock()
        mock_lr = MagicMock()
        # Give the mock the required interface
        mock_lr.fit = MagicMock()
        mock_lr.predict = MagicMock()
        mock_lr.predict_proba = MagicMock()
        mock_cuml.linear_model.LogisticRegression.return_value = mock_lr

        with patch.dict("sys.modules", {"cuml": mock_cuml}):
            clf = resolve_classifier(
                "logistic_regression", random_state=42, compute_backend="cuml",
            )
            assert clf is mock_lr

    def test_custom_estimator_ignores_compute_backend(self):
        """Custom estimator instances bypass compute_backend."""
        from sklearn.linear_model import LogisticRegression

        custom = LogisticRegression()
        clf = resolve_classifier(custom, compute_backend="cuml")
        assert clf is custom


# ---------------------------------------------------------------------------
# Probe(compute_backend=...) integration
# ---------------------------------------------------------------------------


class TestProbeComputeBackend:
    def test_default_is_sklearn(self):
        from lmprobe import Probe

        probe = Probe(classifier="logistic_regression", random_state=42)
        assert probe.compute_backend == "sklearn"
        assert probe._compute_backend == "sklearn"

    def test_auto_resolves_to_sklearn(self):
        """Without cuML installed, auto resolves to sklearn."""
        from lmprobe import Probe

        cuml_available.cache_clear()
        probe = Probe(
            classifier="logistic_regression",
            random_state=42,
            compute_backend="auto",
        )
        assert probe._compute_backend == "sklearn"

    def test_cuml_raises_when_not_installed(self):
        from lmprobe import Probe

        cuml_available.cache_clear()
        with pytest.raises(ImportError, match="compute_backend='cuml' requires cuML"):
            Probe(
                classifier="logistic_regression",
                random_state=42,
                compute_backend="cuml",
            )

    def test_invalid_compute_backend_raises(self):
        from lmprobe import Probe

        with pytest.raises(ValueError, match="Unknown compute_backend"):
            Probe(
                classifier="logistic_regression",
                random_state=42,
                compute_backend="jax",
            )

    def test_fit_from_activations_with_sklearn(self):
        """Full fit/predict cycle works with explicit sklearn backend."""
        from lmprobe import Probe

        probe = Probe(
            classifier="logistic_regression",
            random_state=42,
            compute_backend="sklearn",
        )

        rng = np.random.RandomState(42)
        X = rng.randn(20, 64)
        y = np.array([0] * 10 + [1] * 10)

        probe.fit_from_activations(X, y)
        preds = probe.predict_from_activations(X)
        assert preds.shape == (20,)

        proba = probe.predict_proba_from_activations(X)
        assert proba.shape == (20, 2)
