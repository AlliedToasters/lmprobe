"""Built-in classifier factory for lmprobe.

This module provides factory functions for creating sklearn-compatible
classifiers with proper random_state propagation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.svm import SVC

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


# Registry of built-in classifier names
BUILTIN_CLASSIFIERS = frozenset({
    "logistic_regression",
    "logistic_regression_cv",
    "ridge",
    "svm",
    "sgd",
})


def build_classifier(name: str, random_state: int | None = None) -> BaseEstimator:
    """Build a classifier by name with the given random_state.

    Parameters
    ----------
    name : str
        Name of the built-in classifier. One of:
        - "logistic_regression": L2-regularized logistic regression (default)
        - "logistic_regression_cv": Logistic regression with CV-tuned regularization
        - "ridge": Ridge classifier (fast, no probabilities)
        - "svm": Linear SVM with Platt scaling for probabilities
        - "sgd": SGD classifier (scalable to large datasets)
    random_state : int | None
        Random seed for reproducibility. Propagated from LinearProbe.

    Returns
    -------
    BaseEstimator
        An sklearn-compatible classifier instance.

    Raises
    ------
    ValueError
        If the classifier name is not recognized.
    """
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        )
    elif name == "logistic_regression_cv":
        return LogisticRegressionCV(
            cv=5,
            max_iter=1000,
            random_state=random_state,
        )
    elif name == "ridge":
        return RidgeClassifier(random_state=random_state)
    elif name == "svm":
        return SVC(
            kernel="linear",
            probability=True,
            random_state=random_state,
        )
    elif name == "sgd":
        return SGDClassifier(
            loss="log_loss",
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unknown classifier: {name!r}. "
            f"Available: {sorted(BUILTIN_CLASSIFIERS)}"
        )


def validate_classifier(clf: BaseEstimator) -> None:
    """Validate that a classifier has the required interface.

    Parameters
    ----------
    clf : BaseEstimator
        The classifier to validate.

    Raises
    ------
    TypeError
        If the classifier lacks fit() or predict() methods.

    Warns
    -----
    UserWarning
        If the classifier lacks predict_proba() method.
    """
    if not hasattr(clf, "fit"):
        raise TypeError(
            f"Classifier {type(clf).__name__} must have a fit() method"
        )
    if not hasattr(clf, "predict"):
        raise TypeError(
            f"Classifier {type(clf).__name__} must have a predict() method"
        )
    if not hasattr(clf, "predict_proba"):
        warnings.warn(
            f"{type(clf).__name__} does not support predict_proba(). "
            "probe.predict_proba() will raise an error.",
            UserWarning,
            stacklevel=3,
        )


def resolve_classifier(
    classifier: str | BaseEstimator,
    random_state: int | None = None,
) -> BaseEstimator:
    """Resolve a classifier specification to an estimator instance.

    Parameters
    ----------
    classifier : str | BaseEstimator
        Either a string name of a built-in classifier, or a custom
        sklearn-compatible estimator instance.
    random_state : int | None
        Random seed. Only used for built-in classifiers (strings).
        Custom estimators must set their own random_state.

    Returns
    -------
    BaseEstimator
        The resolved classifier instance.
    """
    if isinstance(classifier, str):
        clf = build_classifier(classifier, random_state=random_state)
    else:
        clf = classifier

    validate_classifier(clf)
    return clf


class GroupLassoClassifier:
    """Wrapper around skglm Group Lasso for automatic layer selection.

    This classifier treats each layer's hidden dimensions as a group and
    applies L2,1 regularization (Group Lasso) to encourage entire groups
    (layers) to become zero, effectively performing layer selection.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension size per layer.
    n_layers : int
        Number of layers being probed.
    alpha : float, default=0.01
        Regularization strength. Higher values induce more sparsity.
    random_state : int | None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    coef_ : np.ndarray
        Fitted coefficients, shape (n_features,) = (hidden_dim * n_layers,).
    intercept_ : np.ndarray
        Fitted intercept.
    classes_ : np.ndarray
        Class labels.
    selected_groups_ : list[int]
        Indices of groups (layers) with non-zero norms after fitting.
    group_norms_ : np.ndarray
        L2 norm of coefficients for each group, shape (n_layers,).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        alpha: float = 0.01,
        random_state: int | None = None,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.alpha = alpha
        self.random_state = random_state

        # Fitted attributes
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.selected_groups_: list[int] | None = None
        self.group_norms_: np.ndarray | None = None

        # Lazy-loaded estimator
        self._estimator = None

    def _check_skglm_installed(self) -> None:
        """Check that skglm is installed, raise helpful error if not."""
        try:
            import skglm  # noqa: F401
        except ImportError:
            raise ImportError(
                "skglm is required for layers='auto'. "
                "Install it with: pip install lmprobe[auto]"
            )

    def _build_estimator(self):
        """Build the underlying skglm estimator."""
        from skglm import GeneralizedLinearEstimator
        from skglm.datafits import LogisticGroup
        from skglm.penalties import WeightedGroupL2
        from skglm.solvers import GroupProxNewton
        from skglm.utils.data import grp_converter

        n_features = self.hidden_dim * self.n_layers
        grp_indices, grp_ptr = grp_converter(self.hidden_dim, n_features)

        weights = np.ones(self.n_layers)
        penalty = WeightedGroupL2(self.alpha, weights, grp_ptr, grp_indices)
        datafit = LogisticGroup(grp_ptr, grp_indices)
        solver = GroupProxNewton(verbose=0)

        return GeneralizedLinearEstimator(datafit, penalty, solver)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GroupLassoClassifier":
        """Fit the Group Lasso classifier.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, hidden_dim * n_layers).
        y : np.ndarray
            Labels, shape (n_samples,).

        Returns
        -------
        self
        """
        self._check_skglm_installed()

        # Validate input dimensions
        expected_features = self.hidden_dim * self.n_layers
        if X.shape[1] != expected_features:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {expected_features} "
                f"(hidden_dim={self.hidden_dim} x n_layers={self.n_layers})"
            )

        # Store classes
        self.classes_ = np.unique(y)

        # skglm expects y in {-1, 1} for logistic regression
        y_transformed = np.where(y == 0, -1, 1)

        # Build and fit estimator
        self._estimator = self._build_estimator()
        self._estimator.fit(X, y_transformed)

        # Extract coefficients (skglm returns (1, n_features), flatten to (n_features,))
        coef = self._estimator.coef_
        if coef.ndim == 2:
            coef = coef.flatten()
        self.coef_ = coef
        self.intercept_ = getattr(self._estimator, "intercept_", None)

        # Compute group norms and identify selected groups
        coef_by_group = self.coef_.reshape(self.n_layers, self.hidden_dim)
        self.group_norms_ = np.linalg.norm(coef_by_group, axis=1)

        # Groups with non-negligible norms are selected
        threshold = 1e-6  # Numerical tolerance
        self.selected_groups_ = [
            i for i, norm in enumerate(self.group_norms_) if norm > threshold
        ]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted labels, shape (n_samples,).
        """
        if self._estimator is None:
            raise RuntimeError("Classifier has not been fitted. Call fit() first.")

        # skglm predict returns {-1, 1}, convert back to {0, 1}
        preds = self._estimator.predict(X)
        return np.where(preds == -1, 0, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Note: skglm's GeneralizedLinearEstimator does not have native
        predict_proba. We compute probabilities from the linear scores
        using the sigmoid function.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Class probabilities, shape (n_samples, 2).
        """
        if self.coef_ is None:
            raise RuntimeError("Classifier has not been fitted. Call fit() first.")

        # Compute linear scores: X @ coef + intercept
        intercept = self.intercept_ if self.intercept_ is not None else 0
        scores = X @ self.coef_ + intercept

        # Sigmoid to get P(y=1)
        prob_positive = 1 / (1 + np.exp(-scores))
        prob_negative = 1 - prob_positive

        return np.column_stack([prob_negative, prob_positive])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True labels.

        Returns
        -------
        float
            Accuracy.
        """
        predictions = self.predict(X)
        return float((predictions == y).mean())


def build_group_lasso_classifier(
    hidden_dim: int,
    n_layers: int,
    alpha: float = 0.01,
    random_state: int | None = None,
) -> GroupLassoClassifier:
    """Build a Group Lasso classifier for layer selection.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension per layer.
    n_layers : int
        Number of candidate layers.
    alpha : float
        Regularization strength.
    random_state : int | None
        Random seed.

    Returns
    -------
    GroupLassoClassifier
        The configured classifier.
    """
    return GroupLassoClassifier(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        alpha=alpha,
        random_state=random_state,
    )
