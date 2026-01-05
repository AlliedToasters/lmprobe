"""Per-layer feature scaling for multi-layer probes.

When using activations from multiple layers, each layer may have different
activation magnitude distributions. This module provides scalers that normalize
each layer's features independently to enable fair comparison.
"""

from __future__ import annotations

import numpy as np


class PerLayerScaler:
    """Standardize features on a per-layer basis.

    When using multiple layers (concatenated), each layer may have different
    activation magnitude distributions. This scaler normalizes each layer's
    features to zero mean and unit variance.

    Two strategies are available:
    - "per_neuron": Each neuron gets its own mean/std (more parameters, higher variance)
    - "per_layer": All neurons in a layer share one mean/std (fewer parameters, lower variance)

    The "per_layer" strategy may be preferable when:
    - Sample size is small relative to hidden dimension
    - Neurons within a layer have similar activation distributions (symmetry assumption)

    Parameters
    ----------
    n_layers : int
        Number of layers in the concatenated features.
    hidden_dim : int
        Hidden dimension per layer (features per layer).
    strategy : str, default="per_neuron"
        Scaling strategy:
        - "per_neuron": Each neuron has its own mean/std
        - "per_layer": All neurons in a layer share one mean/std

    Attributes
    ----------
    means_ : np.ndarray | None
        Feature means. Shape depends on strategy:
        - "per_neuron": (n_layers, hidden_dim)
        - "per_layer": (n_layers,)
    stds_ : np.ndarray | None
        Feature standard deviations. Shape matches means_.

    Examples
    --------
    >>> scaler = PerLayerScaler(n_layers=3, hidden_dim=128, strategy="per_neuron")
    >>> X_train_scaled = scaler.fit_transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)

    >>> # Use per_layer for small sample sizes
    >>> scaler = PerLayerScaler(n_layers=3, hidden_dim=128, strategy="per_layer")
    """

    VALID_STRATEGIES = ("per_neuron", "per_layer")

    def __init__(
        self, n_layers: int, hidden_dim: int, strategy: str = "per_neuron"
    ):
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                f"Valid options: {self.VALID_STRATEGIES}"
            )
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.strategy = strategy
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PerLayerScaler":
        """Compute per-layer means and standard deviations.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_layers * hidden_dim).

        Returns
        -------
        PerLayerScaler
            Self, for method chaining.

        Raises
        ------
        ValueError
            If X has wrong number of features.
        """
        expected_features = self.n_layers * self.hidden_dim
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features "
                f"({self.n_layers} layers x {self.hidden_dim} hidden_dim), "
                f"got {X.shape[1]}"
            )

        # Reshape to (n_samples, n_layers, hidden_dim)
        X_reshaped = X.reshape(X.shape[0], self.n_layers, self.hidden_dim)

        if self.strategy == "per_neuron":
            # Each neuron gets its own mean/std
            # Mean over samples, shape: (n_layers, hidden_dim)
            self.means_ = X_reshaped.mean(axis=0)
            self.stds_ = X_reshaped.std(axis=0)
        else:  # per_layer
            # All neurons in a layer share one mean/std
            # Reshape to (n_layers, n_samples * hidden_dim) and compute stats
            X_flat = X_reshaped.transpose(1, 0, 2).reshape(self.n_layers, -1)
            self.means_ = X_flat.mean(axis=1)  # shape: (n_layers,)
            self.stds_ = X_flat.std(axis=1)  # shape: (n_layers,)

        # Avoid division by zero: replace zero std with 1
        self.stds_ = np.where(self.stds_ == 0, 1.0, self.stds_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply per-layer standardization.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_layers * hidden_dim).

        Returns
        -------
        np.ndarray
            Standardized features, same shape as input.

        Raises
        ------
        RuntimeError
            If scaler has not been fitted.
        ValueError
            If X has wrong number of features.
        """
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("PerLayerScaler has not been fitted. Call fit() first.")

        expected_features = self.n_layers * self.hidden_dim
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features "
                f"({self.n_layers} layers x {self.hidden_dim} hidden_dim), "
                f"got {X.shape[1]}"
            )

        # Reshape to (n_samples, n_layers, hidden_dim)
        X_reshaped = X.reshape(X.shape[0], self.n_layers, self.hidden_dim)

        if self.strategy == "per_neuron":
            # means_ and stds_ are (n_layers, hidden_dim)
            X_scaled = (X_reshaped - self.means_) / self.stds_
        else:  # per_layer
            # means_ and stds_ are (n_layers,), need to broadcast
            # Reshape to (1, n_layers, 1) for broadcasting
            means = self.means_[:, np.newaxis]  # (n_layers, 1)
            stds = self.stds_[:, np.newaxis]  # (n_layers, 1)
            X_scaled = (X_reshaped - means) / stds

        # Reshape back to (n_samples, n_layers * hidden_dim)
        return X_scaled.reshape(X.shape[0], -1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data in one step.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_layers * hidden_dim).

        Returns
        -------
        np.ndarray
            Standardized features, same shape as input.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the standardization.

        Parameters
        ----------
        X : np.ndarray
            Standardized feature matrix, shape (n_samples, n_layers * hidden_dim).

        Returns
        -------
        np.ndarray
            Original-scale features, same shape as input.

        Raises
        ------
        RuntimeError
            If scaler has not been fitted.
        """
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("PerLayerScaler has not been fitted. Call fit() first.")

        expected_features = self.n_layers * self.hidden_dim
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {X.shape[1]}"
            )

        # Reshape to (n_samples, n_layers, hidden_dim)
        X_reshaped = X.reshape(X.shape[0], self.n_layers, self.hidden_dim)

        if self.strategy == "per_neuron":
            # means_ and stds_ are (n_layers, hidden_dim)
            X_original = X_reshaped * self.stds_ + self.means_
        else:  # per_layer
            # means_ and stds_ are (n_layers,), need to broadcast
            means = self.means_[:, np.newaxis]  # (n_layers, 1)
            stds = self.stds_[:, np.newaxis]  # (n_layers, 1)
            X_original = X_reshaped * stds + means

        # Reshape back
        return X_original.reshape(X.shape[0], -1)

    def get_layer_stats(self) -> dict[str, np.ndarray]:
        """Get per-layer statistics for analysis.

        Returns
        -------
        dict
            Dictionary with layer-level statistics. Contents depend on strategy:
            - "per_neuron": includes 'mean_norms', 'std_norms', 'mean_per_layer', 'std_per_layer'
            - "per_layer": includes 'means', 'stds' (the raw per-layer values)

        Raises
        ------
        RuntimeError
            If scaler has not been fitted.
        """
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("PerLayerScaler has not been fitted. Call fit() first.")

        if self.strategy == "per_neuron":
            return {
                "mean_norms": np.linalg.norm(self.means_, axis=1),
                "std_norms": np.linalg.norm(self.stds_, axis=1),
                "mean_per_layer": self.means_.mean(axis=1),
                "std_per_layer": self.stds_.mean(axis=1),
            }
        else:  # per_layer
            return {
                "means": self.means_.copy(),
                "stds": self.stds_.copy(),
            }
