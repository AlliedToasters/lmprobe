"""Tests for PerLayerScaler."""

import numpy as np
import pytest

from lmprobe.scaling import PerLayerScaler


class TestPerLayerScalerPerNeuron:
    """Tests for per_neuron strategy."""

    @pytest.fixture
    def scaler(self):
        return PerLayerScaler(n_layers=2, hidden_dim=3, strategy="per_neuron")

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        return rng.normal(0, 1, (20, 6))  # 2 layers x 3 dim

    def test_fit_transform(self, scaler, data):
        result = scaler.fit_transform(data)
        assert result.shape == data.shape

    def test_transform_zero_mean_unit_std(self, scaler, data):
        result = scaler.fit_transform(data)
        # Each layer's features should be ~zero mean, ~unit std
        reshaped = result.reshape(-1, 2, 3)
        for layer in range(2):
            layer_data = reshaped[:, layer, :]
            assert np.allclose(layer_data.mean(axis=0), 0.0, atol=1e-10)
            assert np.allclose(layer_data.std(axis=0), 1.0, atol=0.1)

    def test_inverse_transform_roundtrip(self, scaler, data):
        scaled = scaler.fit_transform(data)
        recovered = scaler.inverse_transform(scaled)
        assert np.allclose(recovered, data, atol=1e-10)

    def test_get_layer_stats(self, scaler, data):
        scaler.fit(data)
        stats = scaler.get_layer_stats()
        assert "mean_norms" in stats
        assert "std_norms" in stats
        assert "mean_per_layer" in stats
        assert "std_per_layer" in stats
        assert stats["mean_norms"].shape == (2,)


class TestPerLayerScalerPerLayer:
    """Tests for per_layer strategy."""

    @pytest.fixture
    def scaler(self):
        return PerLayerScaler(n_layers=2, hidden_dim=3, strategy="per_layer")

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        return rng.normal(0, 1, (20, 6))

    def test_fit_transform(self, scaler, data):
        result = scaler.fit_transform(data)
        assert result.shape == data.shape

    def test_inverse_transform_roundtrip(self, scaler, data):
        scaled = scaler.fit_transform(data)
        recovered = scaler.inverse_transform(scaled)
        assert np.allclose(recovered, data, atol=1e-10)

    def test_get_layer_stats(self, scaler, data):
        scaler.fit(data)
        stats = scaler.get_layer_stats()
        assert "means" in stats
        assert "stds" in stats
        assert stats["means"].shape == (2,)


class TestPerLayerScalerEdgeCases:
    """Edge cases and validation."""

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            PerLayerScaler(n_layers=2, hidden_dim=3, strategy="invalid")

    def test_wrong_feature_count_raises(self):
        scaler = PerLayerScaler(n_layers=2, hidden_dim=3)
        X = np.zeros((10, 5))  # Wrong: should be 6
        with pytest.raises(ValueError, match="Expected 6 features"):
            scaler.fit(X)

    def test_transform_before_fit_raises(self):
        scaler = PerLayerScaler(n_layers=2, hidden_dim=3)
        with pytest.raises(RuntimeError, match="not been fitted"):
            scaler.transform(np.zeros((5, 6)))

    def test_inverse_transform_before_fit_raises(self):
        scaler = PerLayerScaler(n_layers=2, hidden_dim=3)
        with pytest.raises(RuntimeError, match="not been fitted"):
            scaler.inverse_transform(np.zeros((5, 6)))

    def test_get_layer_stats_before_fit_raises(self):
        scaler = PerLayerScaler(n_layers=2, hidden_dim=3)
        with pytest.raises(RuntimeError, match="not been fitted"):
            scaler.get_layer_stats()

    def test_zero_std_handled(self):
        """Constant features should not cause division by zero."""
        scaler = PerLayerScaler(n_layers=1, hidden_dim=2)
        X = np.ones((10, 2))
        result = scaler.fit_transform(X)
        assert np.isfinite(result).all()
