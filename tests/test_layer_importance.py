"""Tests for coefficient-based layer importance and fast_auto mode.

This module tests:
1. PerLayerScaler - per-layer feature normalization
2. compute_layer_importance() - post-hoc layer importance analysis
3. layers="fast_auto" mode - fast automatic layer selection
"""

import numpy as np
import pytest

from lmprobe import LinearProbe


class TestPerLayerScaler:
    """Tests for the PerLayerScaler class."""

    def test_basic_normalization(self):
        """PerLayerScaler normalizes features per layer."""
        from lmprobe.scaling import PerLayerScaler

        # Create data with different scales per layer
        # Layer 0: mean=0, std=1
        # Layer 1: mean=10, std=5
        np.random.seed(42)
        n_samples = 100
        hidden_dim = 8
        n_layers = 2

        layer_0 = np.random.randn(n_samples, hidden_dim)
        layer_1 = np.random.randn(n_samples, hidden_dim) * 5 + 10
        X = np.hstack([layer_0, layer_1])

        scaler = PerLayerScaler(n_layers=n_layers, hidden_dim=hidden_dim)
        X_scaled = scaler.fit_transform(X)

        # After scaling, both layers should have ~zero mean and ~unit std
        X_reshaped = X_scaled.reshape(n_samples, n_layers, hidden_dim)

        # Check layer 0
        assert np.abs(X_reshaped[:, 0, :].mean()) < 0.1
        assert np.abs(X_reshaped[:, 0, :].std() - 1.0) < 0.1

        # Check layer 1
        assert np.abs(X_reshaped[:, 1, :].mean()) < 0.1
        assert np.abs(X_reshaped[:, 1, :].std() - 1.0) < 0.1

    def test_transform_uses_fitted_params(self):
        """Transform uses parameters from fit, not new data stats."""
        from lmprobe.scaling import PerLayerScaler

        np.random.seed(42)
        n_samples = 50
        hidden_dim = 4
        n_layers = 2

        # Training data
        X_train = np.random.randn(n_samples, n_layers * hidden_dim)

        # Test data with different distribution
        X_test = np.random.randn(20, n_layers * hidden_dim) * 10 + 100

        scaler = PerLayerScaler(n_layers=n_layers, hidden_dim=hidden_dim)
        scaler.fit(X_train)

        X_test_scaled = scaler.transform(X_test)

        # Test data should NOT have zero mean after transform
        # (because it uses training params)
        assert np.abs(X_test_scaled.mean()) > 1.0

    def test_inverse_transform(self):
        """inverse_transform recovers original data."""
        from lmprobe.scaling import PerLayerScaler

        np.random.seed(42)
        X = np.random.randn(50, 16)  # 2 layers x 8 hidden_dim

        scaler = PerLayerScaler(n_layers=2, hidden_dim=8)
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_allclose(X, X_recovered, rtol=1e-10)

    def test_shape_validation(self):
        """Raises on wrong feature count."""
        from lmprobe.scaling import PerLayerScaler

        scaler = PerLayerScaler(n_layers=2, hidden_dim=8)  # Expects 16 features
        X_wrong = np.random.randn(10, 20)  # 20 features

        with pytest.raises(ValueError, match="Expected 16 features"):
            scaler.fit(X_wrong)

    def test_unfitted_transform_raises(self):
        """Transform raises if not fitted."""
        from lmprobe.scaling import PerLayerScaler

        scaler = PerLayerScaler(n_layers=2, hidden_dim=8)
        X = np.random.randn(10, 16)

        with pytest.raises(RuntimeError, match="not been fitted"):
            scaler.transform(X)


class TestComputeLayerImportance:
    """Tests for the compute_layer_importance method."""

    def test_basic_importance(self, tiny_model):
        """compute_layer_importance works with multi-layer probe."""
        probe = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],  # Last two layers
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["positive example"], ["negative example"])

        importance = probe.compute_layer_importance()

        assert importance.shape == (2,)
        assert importance.sum() == pytest.approx(1.0)  # Normalized
        assert probe.layer_importances_ is not None
        assert probe.candidate_layers_ is not None
        assert len(probe.candidate_layers_) == 2

    def test_l2_metric_manual(self, tiny_model):
        """L2 metric computes correctly."""
        probe = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],
            pooling="last_token",
            device="cpu",
            remote=False,
            normalize_layers=False,  # Disable scaling for predictable coefficients
        )
        probe.fit(["positive"], ["negative"])

        importance = probe.compute_layer_importance(metric="l2", normalize=False)

        # Verify manually
        coef = probe.classifier_.coef_.flatten()
        hidden_dim = len(coef) // 2
        expected_0 = np.linalg.norm(coef[:hidden_dim])
        expected_1 = np.linalg.norm(coef[hidden_dim:])

        assert importance[0] == pytest.approx(expected_0, rel=1e-6)
        assert importance[1] == pytest.approx(expected_1, rel=1e-6)

    def test_different_metrics(self, tiny_model):
        """Different metrics produce different results."""
        probe = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],
            pooling="last_token",
            device="cpu",
            remote=False,
        )
        probe.fit(["positive"], ["negative"])

        l2 = probe.compute_layer_importance(metric="l2", normalize=False)
        l1 = probe.compute_layer_importance(metric="l1", normalize=False)
        mean_abs = probe.compute_layer_importance(metric="mean_abs", normalize=False)
        max_abs = probe.compute_layer_importance(metric="max_abs", normalize=False)

        # L1 >= L2 >= max_abs (by definition for unit vectors and typical cases)
        # At minimum, they should all be positive
        assert (l2 > 0).all()
        assert (l1 > 0).all()
        assert (mean_abs > 0).all()
        assert (max_abs > 0).all()

    def test_plotting_works_after_compute(self, tiny_model):
        """plot_layer_importance works after compute_layer_importance."""
        pytest.importorskip("matplotlib")

        probe = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],
            pooling="last_token",
            device="cpu",
            remote=False,
        )
        probe.fit(["positive"], ["negative"])
        probe.compute_layer_importance()

        fig, ax = probe.plot_layer_importance()

        assert fig is not None
        assert ax is not None

    def test_requires_fitted(self, tiny_model):
        """Raises if not fitted."""
        probe = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],
            device="cpu",
            remote=False,
        )

        with pytest.raises(RuntimeError, match="not been fitted"):
            probe.compute_layer_importance()

    def test_requires_coef(self, tiny_model):
        """Raises if classifier lacks coef_."""
        from sklearn.neighbors import KNeighborsClassifier

        probe = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],
            classifier=KNeighborsClassifier(n_neighbors=1),
            device="cpu",
            remote=False,
        )
        probe.fit(["positive"], ["negative"])

        with pytest.raises(RuntimeError, match="coef_"):
            probe.compute_layer_importance()

    def test_unknown_metric_raises(self, tiny_model):
        """Raises on unknown metric."""
        probe = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],
            device="cpu",
            remote=False,
        )
        probe.fit(["positive"], ["negative"])

        with pytest.raises(ValueError, match="Unknown metric"):
            probe.compute_layer_importance(metric="invalid")


class TestFastAutoLayers:
    """Tests for layers='fast_auto' mode."""

    def test_fast_auto_basic(self, tiny_model):
        """fast_auto mode works and produces predictions."""
        probe = LinearProbe(
            model=tiny_model,
            layers="fast_auto",
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive example"], ["negative example"])

        assert probe.selected_layers_ is not None
        assert probe.layer_importances_ is not None
        assert probe.candidate_layers_ is not None

        # Can predict
        predictions = probe.predict(["test"])
        assert predictions.shape == (1,)

    def test_fast_auto_with_top_k(self, tiny_model):
        """fast_auto with explicit top_k selection."""
        probe = LinearProbe(
            model=tiny_model,
            layers="fast_auto",
            fast_auto_top_k=1,  # Select only 1 layer
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive"], ["negative"])

        assert len(probe.selected_layers_) == 1

    def test_fast_auto_default_top_k(self, tiny_model):
        """fast_auto defaults to half the candidate layers."""
        # tiny model only has 2 layers, so use both as candidates
        probe = LinearProbe(
            model=tiny_model,
            layers="fast_auto",
            auto_candidates=[-2, -1],  # 2 candidates (all layers of tiny model)
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive"], ["negative"])

        # Default: half of 2 = 1 layer
        assert len(probe.selected_layers_) == 1

    def test_fast_auto_predict_proba(self, tiny_model):
        """Predictions and probabilities work after fast_auto fitting."""
        probe = LinearProbe(
            model=tiny_model,
            layers="fast_auto",
            fast_auto_top_k=2,
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive 1", "positive 2"], ["negative 1", "negative 2"])

        predictions = probe.predict(["test 1", "test 2"])
        probabilities = probe.predict_proba(["test 1", "test 2"])

        assert predictions.shape == (2,)
        assert probabilities.shape == (2, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_fast_auto_with_normalization(self, tiny_model):
        """Normalization is applied in fast_auto mode."""
        probe = LinearProbe(
            model=tiny_model,
            layers="fast_auto",
            fast_auto_top_k=2,
            normalize_layers=True,  # Explicitly enable
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive"], ["negative"])

        # Scaler should be set after fitting
        assert probe.scaler_ is not None

    def test_fast_auto_save_load(self, tiny_model, tmp_path):
        """fast_auto probes can be saved and loaded."""
        probe = LinearProbe(
            model=tiny_model,
            layers="fast_auto",
            fast_auto_top_k=1,
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["positive"], ["negative"])

        # Get predictions before save
        pred_before = probe.predict(["test"])

        # Save and load
        save_path = tmp_path / "probe.pkl"
        probe.save(str(save_path))
        loaded = LinearProbe.load(str(save_path))

        # Predictions should match
        pred_after = loaded.predict(["test"])
        np.testing.assert_array_equal(pred_before, pred_after)

        # Attributes should be preserved
        assert loaded.selected_layers_ == probe.selected_layers_
        assert loaded.fast_auto_top_k == probe.fast_auto_top_k


class TestNormalizeLayers:
    """Tests for the normalize_layers parameter."""

    def test_normalize_layers_enabled(self, tiny_model):
        """Scaler is created when normalize_layers=True with multiple layers."""
        probe = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],  # Two layers
            normalize_layers=True,
            pooling="last_token",
            device="cpu",
            remote=False,
        )
        probe.fit(["positive"], ["negative"])

        assert probe.scaler_ is not None

    def test_normalize_layers_disabled(self, tiny_model):
        """Scaler is NOT created when normalize_layers=False."""
        probe = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],
            normalize_layers=False,
            pooling="last_token",
            device="cpu",
            remote=False,
        )
        probe.fit(["positive"], ["negative"])

        assert probe.scaler_ is None

    def test_single_layer_no_scaler(self, tiny_model):
        """No scaler for single-layer probes (nothing to normalize)."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,  # Single layer
            normalize_layers=True,
            pooling="last_token",
            device="cpu",
            remote=False,
        )
        probe.fit(["positive"], ["negative"])

        # Single layer doesn't need per-layer normalization
        assert probe.scaler_ is None

    def test_normalization_affects_predictions(self, tiny_model):
        """Normalization changes predictions (coefficients differ)."""
        positive = ["positive example 1", "positive example 2"]
        negative = ["negative example 1", "negative example 2"]

        # With normalization
        probe_norm = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],
            normalize_layers=True,
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe_norm.fit(positive, negative)

        # Without normalization
        probe_no_norm = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],
            normalize_layers=False,
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe_no_norm.fit(positive, negative)

        # Coefficients should be different (normalization changes input scale)
        coef_norm = probe_norm.classifier_.coef_
        coef_no_norm = probe_no_norm.classifier_.coef_

        # They shouldn't be identical
        assert not np.allclose(coef_norm, coef_no_norm)


class TestPerLayerStrategy:
    """Tests for the 'per_layer' scaling strategy."""

    def test_per_layer_strategy_normalization(self):
        """per_layer strategy normalizes all neurons in a layer together."""
        from lmprobe.scaling import PerLayerScaler

        np.random.seed(42)
        n_samples = 100
        hidden_dim = 8
        n_layers = 2

        # Create data with different scales per layer
        layer_0 = np.random.randn(n_samples, hidden_dim)
        layer_1 = np.random.randn(n_samples, hidden_dim) * 5 + 10
        X = np.hstack([layer_0, layer_1])

        scaler = PerLayerScaler(n_layers=n_layers, hidden_dim=hidden_dim, strategy="per_layer")
        X_scaled = scaler.fit_transform(X)

        # After scaling, each layer should have ~zero mean and ~unit std
        X_reshaped = X_scaled.reshape(n_samples, n_layers, hidden_dim)

        # Check layer 0 (all values together)
        layer_0_values = X_reshaped[:, 0, :].flatten()
        assert np.abs(layer_0_values.mean()) < 0.1
        assert np.abs(layer_0_values.std() - 1.0) < 0.1

        # Check layer 1 (all values together)
        layer_1_values = X_reshaped[:, 1, :].flatten()
        assert np.abs(layer_1_values.mean()) < 0.1
        assert np.abs(layer_1_values.std() - 1.0) < 0.1

    def test_per_layer_has_fewer_parameters(self):
        """per_layer strategy has fewer parameters than per_neuron."""
        from lmprobe.scaling import PerLayerScaler

        np.random.seed(42)
        X = np.random.randn(50, 16)  # 2 layers x 8 hidden_dim

        scaler_neuron = PerLayerScaler(n_layers=2, hidden_dim=8, strategy="per_neuron")
        scaler_layer = PerLayerScaler(n_layers=2, hidden_dim=8, strategy="per_layer")

        scaler_neuron.fit(X)
        scaler_layer.fit(X)

        # per_neuron: means_ shape (2, 8) = 16 parameters
        assert scaler_neuron.means_.shape == (2, 8)
        # per_layer: means_ shape (2,) = 2 parameters
        assert scaler_layer.means_.shape == (2,)

    def test_per_layer_inverse_transform(self):
        """per_layer strategy inverse_transform recovers original data."""
        from lmprobe.scaling import PerLayerScaler

        np.random.seed(42)
        X = np.random.randn(50, 16)

        scaler = PerLayerScaler(n_layers=2, hidden_dim=8, strategy="per_layer")
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_allclose(X, X_recovered, rtol=1e-10)

    def test_per_layer_get_layer_stats(self):
        """per_layer strategy returns correct stats."""
        from lmprobe.scaling import PerLayerScaler

        np.random.seed(42)
        X = np.random.randn(50, 16)

        scaler = PerLayerScaler(n_layers=2, hidden_dim=8, strategy="per_layer")
        scaler.fit(X)
        stats = scaler.get_layer_stats()

        # per_layer should return 'means' and 'stds' keys
        assert "means" in stats
        assert "stds" in stats
        assert stats["means"].shape == (2,)
        assert stats["stds"].shape == (2,)

    def test_invalid_strategy_raises(self):
        """Invalid strategy raises ValueError."""
        from lmprobe.scaling import PerLayerScaler

        with pytest.raises(ValueError, match="Unknown strategy"):
            PerLayerScaler(n_layers=2, hidden_dim=8, strategy="invalid")

    def test_probe_with_per_layer_strategy(self, tiny_model):
        """LinearProbe works with normalize_layers='per_layer'."""
        probe = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],
            normalize_layers="per_layer",
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["positive"], ["negative"])

        assert probe.scaler_ is not None
        assert probe.scaler_.strategy == "per_layer"

        # Can predict
        predictions = probe.predict(["test"])
        assert predictions.shape == (1,)

    def test_probe_with_explicit_per_neuron(self, tiny_model):
        """LinearProbe works with normalize_layers='per_neuron'."""
        probe = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],
            normalize_layers="per_neuron",
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["positive"], ["negative"])

        assert probe.scaler_ is not None
        assert probe.scaler_.strategy == "per_neuron"

    def test_invalid_normalize_layers_raises(self, tiny_model):
        """Invalid normalize_layers value raises ValueError."""
        probe = LinearProbe(
            model=tiny_model,
            layers=[-2, -1],
            normalize_layers="invalid",
            pooling="last_token",
            device="cpu",
            remote=False,
        )

        with pytest.raises(ValueError, match="Invalid normalize_layers"):
            probe.fit(["positive"], ["negative"])


class TestGroupLassoWithScaling:
    """Tests for Group Lasso (auto) mode with layer scaling."""

    def test_auto_layers_with_default_scaling(self, tiny_model):
        """Group Lasso auto mode uses scaling by default."""
        probe = LinearProbe(
            model=tiny_model,
            layers="auto",
            auto_candidates=[-2, -1],
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["positive"], ["negative"])

        # Scaler should be set
        assert probe.scaler_ is not None
        assert probe.scaler_.strategy == "per_neuron"

    def test_auto_layers_with_per_layer_scaling(self, tiny_model):
        """Group Lasso auto mode works with per_layer scaling."""
        probe = LinearProbe(
            model=tiny_model,
            layers="auto",
            auto_candidates=[-2, -1],
            normalize_layers="per_layer",
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["positive"], ["negative"])

        assert probe.scaler_ is not None
        assert probe.scaler_.strategy == "per_layer"

        # Can predict
        predictions = probe.predict(["test"])
        assert predictions.shape == (1,)

    def test_auto_layers_without_scaling(self, tiny_model):
        """Group Lasso auto mode works without scaling."""
        probe = LinearProbe(
            model=tiny_model,
            layers="auto",
            auto_candidates=[-2, -1],
            normalize_layers=False,
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["positive"], ["negative"])

        # Scaler should not be set
        assert probe.scaler_ is None

    def test_auto_layers_save_load_preserves_scaling(self, tiny_model, tmp_path):
        """Save/load preserves scaling strategy for auto mode."""
        probe = LinearProbe(
            model=tiny_model,
            layers="auto",
            auto_candidates=[-2, -1],
            normalize_layers="per_layer",
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["positive"], ["negative"])

        pred_before = probe.predict(["test"])

        # Save and load
        save_path = tmp_path / "probe.pkl"
        probe.save(str(save_path))
        loaded = LinearProbe.load(str(save_path))

        # Strategy should be preserved
        assert loaded.normalize_layers == "per_layer"
        assert loaded.scaler_ is not None
        assert loaded.scaler_.strategy == "per_layer"

        # Predictions should match
        pred_after = loaded.predict(["test"])
        np.testing.assert_array_equal(pred_before, pred_after)
