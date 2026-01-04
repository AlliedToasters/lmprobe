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
    features independently to zero mean and unit variance.

    This is important for:
    1. Fair comparison of classifier coefficients across layers
    2. Accurate layer importance computation
    3. Better classifier performance when activation scales differ

    Parameters
    ----------
    n_layers : int
        Number of layers in the concatenated features.
    hidden_dim : int
        Hidden dimension per layer (features per layer).

    Attributes
    ----------
    means_ : np.ndarray | None
        Per-layer feature means, shape (n_layers, hidden_dim).
        Set after calling fit().
    stds_ : np.ndarray | None
        Per-layer feature standard deviations, shape (n_layers, hidden_dim).
        Set after calling fit().

    Examples
    --------
    >>> scaler = PerLayerScaler(n_layers=3, hidden_dim=128)
    >>> X_train_scaled = scaler.fit_transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)
    """

    def __init__(self, n_layers: int, hidden_dim: int):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
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

        # Compute per-layer statistics
        # Mean over samples, shape: (n_layers, hidden_dim)
        self.means_ = X_reshaped.mean(axis=0)
        # Std over samples, shape: (n_layers, hidden_dim)
        # Use ddof=0 for population std (consistent with sklearn StandardScaler default)
        self.stds_ = X_reshaped.std(axis=0)

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

        # Apply standardization per layer
        X_scaled = (X_reshaped - self.means_) / self.stds_

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

        # Reverse standardization
        X_original = X_reshaped * self.stds_ + self.means_

        # Reshape back
        return X_original.reshape(X.shape[0], -1)

    def get_layer_stats(self) -> dict[str, np.ndarray]:
        """Get per-layer statistics for analysis.

        Returns
        -------
        dict
            Dictionary with 'mean_norms' and 'std_norms' arrays,
            each of shape (n_layers,).

        Raises
        ------
        RuntimeError
            If scaler has not been fitted.
        """
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("PerLayerScaler has not been fitted. Call fit() first.")

        return {
            "mean_norms": np.linalg.norm(self.means_, axis=1),
            "std_norms": np.linalg.norm(self.stds_, axis=1),
            "mean_per_layer": self.means_.mean(axis=1),
            "std_per_layer": self.stds_.mean(axis=1),
        }
