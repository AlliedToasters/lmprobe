"""Tests for ActivationBaseline activation-based baselines."""

import numpy as np
import pytest
from conftest import NEGATIVE_PROMPTS, POSITIVE_PROMPTS, TEST_PROMPTS

from lmprobe.activation_baseline import ActivationBaseline


class TestActivationBaselineBasic:
    """Basic functionality tests for ActivationBaseline."""

    def test_invalid_method(self):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            ActivationBaseline(model="stas/tiny-random-llama-2", method="invalid")

    def test_predict_before_fit(self, tiny_model):
        """Predicting before fit raises RuntimeError."""
        baseline = ActivationBaseline(model=tiny_model, method="random_direction")

        with pytest.raises(RuntimeError, match="not been fitted"):
            baseline.predict(TEST_PROMPTS)


class TestActivationBaselineRandomDirection:
    """Test random_direction baseline method."""

    def test_random_direction_fit_predict(self, tiny_model):
        """Random direction baseline can fit and predict."""
        baseline = ActivationBaseline(
            model=tiny_model,
            method="random_direction",
            device="cpu",
            remote=False,
            random_state=42,
        )
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        predictions = baseline.predict(TEST_PROMPTS)
        assert predictions.shape == (2,)
        assert set(predictions).issubset({0, 1})

    def test_random_direction_predict_proba(self, tiny_model):
        """Random direction baseline supports predict_proba."""
        baseline = ActivationBaseline(
            model=tiny_model,
            method="random_direction",
            device="cpu",
            remote=False,
        )
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        proba = baseline.predict_proba(TEST_PROMPTS)
        assert proba.shape == (2, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_random_direction_score(self, tiny_model):
        """Random direction baseline score method works."""
        baseline = ActivationBaseline(
            model=tiny_model,
            method="random_direction",
            device="cpu",
            remote=False,
        )
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        accuracy = baseline.score(TEST_PROMPTS, [1, 0])
        assert 0.0 <= accuracy <= 1.0

    def test_random_direction_reproducible(self, tiny_model):
        """Random direction baseline is reproducible with seed."""
        baseline1 = ActivationBaseline(
            model=tiny_model,
            method="random_direction",
            device="cpu",
            random_state=42,
        )
        baseline2 = ActivationBaseline(
            model=tiny_model,
            method="random_direction",
            device="cpu",
            random_state=42,
        )

        baseline1.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        baseline2.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        pred1 = baseline1.predict(TEST_PROMPTS)
        pred2 = baseline2.predict(TEST_PROMPTS)

        np.testing.assert_array_equal(pred1, pred2)


class TestActivationBaselinePCA:
    """Test PCA baseline method."""

    def test_pca_fit_predict(self, tiny_model):
        """PCA baseline can fit and predict."""
        baseline = ActivationBaseline(
            model=tiny_model,
            method="pca",
            n_components=3,
            device="cpu",
            remote=False,
        )
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        predictions = baseline.predict(TEST_PROMPTS)
        assert predictions.shape == (2,)
        assert set(predictions).issubset({0, 1})

    def test_pca_predict_proba(self, tiny_model):
        """PCA baseline supports predict_proba."""
        baseline = ActivationBaseline(
            model=tiny_model,
            method="pca",
            n_components=5,
            device="cpu",
            remote=False,
        )
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        proba = baseline.predict_proba(TEST_PROMPTS)
        assert proba.shape == (2, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_pca_n_components_limited(self, tiny_model):
        """PCA n_components is limited by sample size."""
        baseline = ActivationBaseline(
            model=tiny_model,
            method="pca",
            n_components=100,  # More than samples
            device="cpu",
            remote=False,
        )
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        # Should still work, just with fewer components
        predictions = baseline.predict(TEST_PROMPTS)
        assert predictions.shape == (2,)


class TestActivationBaselineLayer0:
    """Test layer_0 baseline method."""

    def test_layer_0_fit_predict(self, tiny_model):
        """Layer 0 baseline can fit and predict."""
        baseline = ActivationBaseline(
            model=tiny_model,
            method="layer_0",
            layers=-1,  # Should be ignored
            device="cpu",
            remote=False,
        )
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        predictions = baseline.predict(TEST_PROMPTS)
        assert predictions.shape == (2,)
        assert set(predictions).issubset({0, 1})

    def test_layer_0_predict_proba(self, tiny_model):
        """Layer 0 baseline supports predict_proba."""
        baseline = ActivationBaseline(
            model=tiny_model,
            method="layer_0",
            device="cpu",
            remote=False,
        )
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        proba = baseline.predict_proba(TEST_PROMPTS)
        assert proba.shape == (2, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_layer_0_uses_embeddings(self, tiny_model):
        """Layer 0 method actually uses layer 0."""
        baseline = ActivationBaseline(
            model=tiny_model,
            method="layer_0",
            device="cpu",
            remote=False,
        )

        # The extractor should have layer_indices [0]
        assert baseline._extractor.layer_indices == [0]



# Pooling and classifier parametrized tests removed — pooling strategies are
# tested in test_pooling_stages.py, and classifier compatibility is covered by
# test_classifiers.py::TestAllClassifiersPredict.
