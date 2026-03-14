"""Tests for mass_mean_augment feature on LinearProbe."""

import numpy as np
import pytest

from lmprobe import LinearProbe


class TestMassMeanAugmentDefault:
    """Tests that mass_mean_augment=False (default) doesn't change behavior."""

    def test_default_is_false(self, tiny_model):
        """mass_mean_augment defaults to False."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            remote=False,
            random_state=42,
        )
        assert probe.mass_mean_augment is False

    def test_no_direction_stored_when_disabled(self, tiny_model):
        """No mass-mean direction is stored when mass_mean_augment=False."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["good", "great"], ["bad", "terrible"])
        assert probe._mass_mean_direction_ is None


class TestMassMeanAugmentEnabled:
    """Tests for mass_mean_augment=True."""

    def test_direction_stored_after_fit(self, tiny_model):
        """A unit-norm mass-mean direction is stored after fit."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            mass_mean_augment=True,
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["good", "great"], ["bad", "terrible"])
        assert probe._mass_mean_direction_ is not None
        # Check it's a unit vector
        norm = np.linalg.norm(probe._mass_mean_direction_)
        assert abs(norm - 1.0) < 1e-6

    def test_fit_predict_works(self, tiny_model):
        """fit/predict roundtrip works with mass_mean_augment=True."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            mass_mean_augment=True,
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(
            ["good", "great", "nice"],
            ["bad", "terrible", "awful"],
        )
        predictions = probe.predict(["test input"])
        assert predictions.shape == (1,)
        assert predictions[0] in [0, 1]

    def test_predict_proba_works(self, tiny_model):
        """predict_proba works with mass_mean_augment=True."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            mass_mean_augment=True,
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(
            ["good", "great", "nice"],
            ["bad", "terrible", "awful"],
        )
        probs = probe.predict_proba(["test input"])
        assert probs.shape == (1, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_score_works(self, tiny_model):
        """score() works with mass_mean_augment=True."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            mass_mean_augment=True,
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(
            ["good", "great", "nice"],
            ["bad", "terrible", "awful"],
        )
        accuracy = probe.score(["test one", "test two"], [1, 0])
        assert 0.0 <= accuracy <= 1.0


class TestMassMeanAugmentWithPreprocessing:
    """Tests for mass_mean_augment combined with preprocessing."""

    def test_with_standard_plus_pca(self, tiny_model):
        """mass_mean_augment works with standard+pca preprocessing."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            preprocessing="standard+pca:4",
            mass_mean_augment=True,
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(
            ["good", "great", "nice", "wonderful", "excellent"],
            ["bad", "terrible", "awful", "horrible", "dreadful"],
        )
        predictions = probe.predict(["test input"])
        assert predictions.shape == (1,)

        probs = probe.predict_proba(["test input"])
        assert probs.shape == (1, 2)

    def test_without_preprocessing(self, tiny_model):
        """mass_mean_augment works without any preprocessing pipeline."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            mass_mean_augment=True,
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["good", "great"], ["bad", "terrible"])
        predictions = probe.predict(["test"])
        assert predictions.shape == (1,)


class TestMassMeanAugmentFromActivations:
    """Tests for mass_mean_augment via *_from_activations methods."""

    def test_fit_predict_from_activations(self):
        """mass_mean_augment works with fit/predict_from_activations."""
        probe = LinearProbe(
            mass_mean_augment=True,
            random_state=42,
        )

        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        y = np.array([1] * 10 + [0] * 10)

        probe.fit_from_activations(X, y)
        assert probe._mass_mean_direction_ is not None

        preds = probe.predict_from_activations(X)
        assert preds.shape == (20,)

    def test_predict_proba_from_activations(self):
        """predict_proba_from_activations works with mass_mean_augment."""
        probe = LinearProbe(
            mass_mean_augment=True,
            random_state=42,
        )

        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        y = np.array([1] * 10 + [0] * 10)

        probe.fit_from_activations(X, y)
        probs = probe.predict_proba_from_activations(X)
        assert probs.shape == (20, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_score_from_activations(self):
        """score_from_activations works with mass_mean_augment."""
        probe = LinearProbe(
            mass_mean_augment=True,
            random_state=42,
        )

        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        y = np.array([1] * 10 + [0] * 10)

        probe.fit_from_activations(X, y)
        score = probe.score_from_activations(X, y)
        assert 0.0 <= score <= 1.0

    def test_augment_adds_one_feature(self):
        """mass_mean_augment adds exactly one feature column."""
        probe = LinearProbe(
            mass_mean_augment=True,
            random_state=42,
        )

        rng = np.random.RandomState(42)
        n_features = 10
        X = rng.randn(20, n_features)
        y = np.array([1] * 10 + [0] * 10)

        probe.fit_from_activations(X, y)

        # The classifier should have been trained on n_features + 1 columns
        # We verify by checking the coef_ shape if available
        if hasattr(probe.classifier_, "coef_"):
            assert probe.classifier_.coef_.shape[1] == n_features + 1

    def test_disabled_does_not_add_feature(self):
        """mass_mean_augment=False does not add extra feature."""
        probe = LinearProbe(
            mass_mean_augment=False,
            random_state=42,
        )

        rng = np.random.RandomState(42)
        n_features = 10
        X = rng.randn(20, n_features)
        y = np.array([1] * 10 + [0] * 10)

        probe.fit_from_activations(X, y)

        if hasattr(probe.classifier_, "coef_"):
            assert probe.classifier_.coef_.shape[1] == n_features
