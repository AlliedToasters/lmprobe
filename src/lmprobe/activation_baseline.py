"""Activation-based baselines for comparison with linear probes.

These baselines use model activations but apply simple transformations
(random projection, PCA, layer 0) to test whether the probe's learned
direction is special or if any direction works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA

from .cache import CachedExtractor
from .classifiers import resolve_classifier
from .extraction import ActivationExtractor
from .pooling import get_pooling_fn

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class ActivationBaseline:
    """Activation-based baseline classifiers.

    These baselines test whether a probe is learning something meaningful
    beyond what simple transformations of activations would capture.

    Parameters
    ----------
    model : str
        HuggingFace model ID or local path.
    method : str, default="random_direction"
        Baseline method:
        - "random_direction": Project onto random unit vector
        - "pca": Project onto top-k principal components
        - "layer_0": Use embedding layer instead of later layers
    layers : int | list[int] | str, default=-1
        Layers for activation extraction (ignored for layer_0 method).
    pooling : str, default="last_token"
        Token pooling strategy.
    classifier : str | BaseEstimator, default="logistic_regression"
        Classification model.
    device : str, default="auto"
        Device for model inference.
    remote : bool, default=False
        Use nnsight remote execution.
    random_state : int | None, default=None
        Random seed for reproducibility.
    n_components : int, default=10
        Number of PCA components (for method="pca").
    batch_size : int, default=8
        Batch size for activation extraction.

    Attributes
    ----------
    classifier_ : BaseEstimator
        The fitted classifier (after calling fit()).
    classes_ : np.ndarray
        Class labels (after calling fit()).
    random_direction_ : np.ndarray | None
        Random unit vector (for method="random_direction").
    pca_ : PCA | None
        Fitted PCA transformer (for method="pca").

    Examples
    --------
    >>> baseline = ActivationBaseline(
    ...     model="meta-llama/Llama-3.1-8B-Instruct",
    ...     method="random_direction",
    ...     layers=-1,
    ... )
    >>> baseline.fit(positive_prompts, negative_prompts)
    >>> accuracy = baseline.score(test_prompts, test_labels)
    """

    VALID_METHODS = frozenset({"random_direction", "pca", "layer_0"})

    def __init__(
        self,
        model: str,
        method: str = "random_direction",
        layers: int | list[int] | str = -1,
        pooling: str = "last_token",
        classifier: str | BaseEstimator = "logistic_regression",
        device: str = "auto",
        remote: bool = False,
        random_state: int | None = None,
        n_components: int = 10,
        batch_size: int = 8,
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown method: {method!r}. Valid options: {sorted(self.VALID_METHODS)}"
            )

        self.model = model
        self.method = method
        self.layers = layers
        self.pooling = pooling
        self.classifier = classifier
        self.device = device
        self.remote = remote
        self.random_state = random_state
        self.n_components = n_components
        self.batch_size = batch_size

        # For layer_0 method, always use layer 0
        actual_layers = 0 if method == "layer_0" else layers

        self._extractor = ActivationExtractor(
            model, device, actual_layers, batch_size, remote=remote
        )
        self._cached_extractor = CachedExtractor(self._extractor)
        self._classifier_template = resolve_classifier(classifier, random_state)

        # Fitted state
        self.classifier_: BaseEstimator | None = None
        self.classes_: np.ndarray | None = None
        self.random_direction_: np.ndarray | None = None
        self.pca_: PCA | None = None

    def _extract_and_pool(self, prompts: list[str]) -> np.ndarray:
        """Extract and pool activations."""
        activations, attention_mask = self._cached_extractor.extract(
            prompts, remote=self.remote
        )
        pool_fn = get_pooling_fn(self.pooling)
        pooled = pool_fn(activations, attention_mask)
        return pooled.detach().cpu().float().numpy()

    def _transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Transform activations based on method."""
        if self.method == "random_direction":
            if fit:
                rng = np.random.RandomState(self.random_state)
                direction = rng.randn(X.shape[1])
                self.random_direction_ = direction / np.linalg.norm(direction)
            # Project onto random direction -> 1D feature
            return (X @ self.random_direction_).reshape(-1, 1)

        elif self.method == "pca":
            if fit:
                n_comp = min(self.n_components, X.shape[0], X.shape[1])
                self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
                return self.pca_.fit_transform(X)
            return self.pca_.transform(X)

        elif self.method == "layer_0":
            # No transformation needed - just use raw embeddings
            return X

        raise ValueError(f"Unknown method: {self.method}")

    def fit(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
    ) -> "ActivationBaseline":
        """Fit the baseline on contrastive examples.

        Parameters
        ----------
        positive_prompts : list[str]
            Examples of the positive class.
        negative_prompts : list[str]
            Examples of the negative class.

        Returns
        -------
        ActivationBaseline
            Self, for method chaining.
        """
        prompts = positive_prompts + negative_prompts
        labels = np.array([1] * len(positive_prompts) + [0] * len(negative_prompts))

        # Extract activations
        X = self._extract_and_pool(prompts)

        # Transform based on method
        X = self._transform(X, fit=True)

        # Fit classifier
        self.classifier_ = clone(self._classifier_template)
        self.classifier_.fit(X, labels)
        self.classes_ = self.classifier_.classes_

        return self

    def predict(self, prompts: list[str]) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to classify.

        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_prompts,).
        """
        self._check_fitted()
        X = self._extract_and_pool(prompts)
        X = self._transform(X, fit=False)
        return self.classifier_.predict(X)

    def predict_proba(self, prompts: list[str]) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to classify.

        Returns
        -------
        np.ndarray
            Class probabilities, shape (n_prompts, n_classes).

        Raises
        ------
        AttributeError
            If the classifier doesn't support predict_proba.
        """
        self._check_fitted()

        if not hasattr(self.classifier_, "predict_proba"):
            raise AttributeError(
                f"Classifier {type(self.classifier_).__name__} does not support "
                "predict_proba(). Use a classifier like logistic_regression or lda."
            )

        X = self._extract_and_pool(prompts)
        X = self._transform(X, fit=False)
        return self.classifier_.predict_proba(X)

    def score(self, prompts: list[str], labels: list[int] | np.ndarray) -> float:
        """Compute classification accuracy.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to classify.
        labels : list[int] | np.ndarray
            True labels.

        Returns
        -------
        float
            Classification accuracy.
        """
        predictions = self.predict(prompts)
        return float(np.mean(predictions == np.asarray(labels)))

    def _check_fitted(self) -> None:
        """Check if the baseline has been fitted."""
        if self.classifier_ is None:
            raise RuntimeError(
                "ActivationBaseline has not been fitted. Call fit() first."
            )
