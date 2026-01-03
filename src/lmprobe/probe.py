"""LinearProbe: Train linear classifiers on language model activations.

This is the main user-facing class for lmprobe.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.base import clone

from .cache import CachedExtractor
from .classifiers import resolve_classifier
from .extraction import ActivationExtractor
from .pooling import (
    SCORE_POOLING_STRATEGIES,
    get_pooling_fn,
    reduce_scores,
    resolve_pooling,
)

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class LinearProbe:
    """Train a linear probe on language model activations.

    Parameters
    ----------
    model : str
        HuggingFace model ID or local path.
    layers : int | list[int] | str, default="middle"
        Which layers to extract activations from:
        - int: Single layer (negative indexing supported)
        - list[int]: Multiple layers (concatenated)
        - "middle": Middle third of layers
        - "last": Last layer only
        - "all": All layers
    pooling : str, default="last_token"
        Token pooling strategy for both training and inference.
        Options: "last_token", "first_token", "mean", "all"
    train_pooling : str | None, default=None
        Override pooling for training only.
    inference_pooling : str | None, default=None
        Override pooling for inference only.
        Additional options: "max", "min" (score-level pooling)
    classifier : str | BaseEstimator, default="logistic_regression"
        Classification model. Either a string name or sklearn estimator.
    device : str, default="auto"
        Device for model inference: "auto", "cpu", "cuda:0", etc.
    remote : bool, default=False
        Use nnsight remote execution (requires NNSIGHT_API_KEY).
    random_state : int | None, default=None
        Random seed for reproducibility. Propagates to classifier.
    batch_size : int, default=8
        Number of prompts to process at once during activation extraction.
        Smaller values use less memory but may be slower.

    Examples
    --------
    >>> probe = LinearProbe(
    ...     model="meta-llama/Llama-3.1-8B-Instruct",
    ...     layers=16,
    ...     pooling="last_token",
    ...     classifier="logistic_regression",
    ...     random_state=42,
    ... )
    >>> probe.fit(positive_prompts, negative_prompts)
    >>> predictions = probe.predict(test_prompts)
    """

    def __init__(
        self,
        model: str,
        layers: int | list[int] | str = "middle",
        pooling: str = "last_token",
        train_pooling: str | None = None,
        inference_pooling: str | None = None,
        classifier: str | BaseEstimator = "logistic_regression",
        device: str = "auto",
        remote: bool = False,
        random_state: int | None = None,
        batch_size: int = 8,
    ):
        self.model = model
        self.layers = layers
        self.pooling = pooling
        self.train_pooling = train_pooling
        self.inference_pooling = inference_pooling
        self.classifier = classifier
        self.device = device
        self.remote = remote
        self.random_state = random_state
        self.batch_size = batch_size

        # Resolve pooling strategies
        self._train_pooling, self._inference_pooling = resolve_pooling(
            pooling, train_pooling, inference_pooling
        )

        # Resolve classifier
        self._classifier_template = resolve_classifier(classifier, random_state)

        # Create extractor (lazy loads model)
        self._extractor = ActivationExtractor(model, device, layers, batch_size)
        self._cached_extractor = CachedExtractor(self._extractor)

        # Fitted state (set after fit())
        self.classifier_: BaseEstimator | None = None
        self.classes_: np.ndarray | None = None

    def _get_remote(self, remote: bool | None) -> bool:
        """Resolve remote parameter with method-level override."""
        return self.remote if remote is None else remote

    def _extract_and_pool(
        self,
        prompts: list[str],
        pooling_strategy: str,
        remote: bool | None = None,
        invalidate_cache: bool = False,
    ) -> tuple[np.ndarray, torch.Tensor | None]:
        """Extract activations and apply pooling.

        Returns
        -------
        tuple[np.ndarray, torch.Tensor | None]
            (pooled_activations, attention_mask)
            attention_mask is returned for score-level pooling
        """
        remote = self._get_remote(remote)

        # Extract activations (with caching)
        activations, attention_mask = self._cached_extractor.extract(
            prompts,
            remote=remote,
            invalidate_cache=invalidate_cache,
        )

        # Get pooling function
        pool_fn = get_pooling_fn(pooling_strategy)

        # Apply pooling
        pooled = pool_fn(activations, attention_mask)

        # Convert to numpy for sklearn
        if pooled.dim() == 2:
            # Normal case: (batch, hidden_dim)
            return pooled.detach().cpu().numpy(), None
        else:
            # "all" pooling: (batch, seq_len, hidden_dim)
            # Return attention_mask for later use
            return pooled.detach().cpu().numpy(), attention_mask

    def fit(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str] | np.ndarray | list[int] | None = None,
        remote: bool | None = None,
        invalidate_cache: bool = False,
    ) -> "LinearProbe":
        """Fit the probe on training data.

        Supports two signatures:
        1. Contrastive: fit(positive_prompts, negative_prompts)
        2. Standard: fit(prompts, labels)

        Parameters
        ----------
        positive_prompts : list[str]
            In contrastive mode: prompts for the positive class.
            In standard mode: all prompts.
        negative_prompts : list[str] | np.ndarray | list[int] | None
            In contrastive mode: prompts for the negative class.
            In standard mode: labels (array of ints).
        remote : bool | None
            Override the instance-level remote setting.
        invalidate_cache : bool
            If True, ignore cached activations and re-extract.

        Returns
        -------
        LinearProbe
            Self, for method chaining.
        """
        # Determine if contrastive or standard mode
        if negative_prompts is None:
            raise ValueError(
                "fit() requires two arguments: either "
                "(positive_prompts, negative_prompts) for contrastive mode, or "
                "(prompts, labels) for standard mode."
            )

        if isinstance(negative_prompts, (np.ndarray, list)) and (
            len(negative_prompts) > 0 and isinstance(negative_prompts[0], (int, np.integer))
        ):
            # Standard mode: fit(prompts, labels)
            prompts = positive_prompts
            labels = np.asarray(negative_prompts)
        else:
            # Contrastive mode: fit(positive_prompts, negative_prompts)
            prompts = list(positive_prompts) + list(negative_prompts)
            labels = np.array(
                [1] * len(positive_prompts) + [0] * len(negative_prompts)
            )

        # Extract and pool activations
        X, _ = self._extract_and_pool(
            prompts,
            self._train_pooling,
            remote=remote,
            invalidate_cache=invalidate_cache,
        )

        # Handle "all" pooling for training (expand to per-token examples)
        if self._train_pooling == "all" and X.ndim == 3:
            # X is (batch, seq_len, hidden_dim)
            # Expand to (batch * seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = X.shape
            X = X.reshape(-1, hidden_dim)
            # Repeat labels for each token
            labels = np.repeat(labels, seq_len)

        # Clone and fit classifier
        self.classifier_ = clone(self._classifier_template)
        self.classifier_.fit(X, labels)
        self.classes_ = self.classifier_.classes_

        return self

    def _check_fitted(self) -> None:
        """Check that the probe has been fitted."""
        if self.classifier_ is None:
            raise RuntimeError(
                "LinearProbe has not been fitted. Call fit() first."
            )

    def predict(
        self,
        prompts: list[str],
        remote: bool | None = None,
    ) -> np.ndarray:
        """Predict class labels for prompts.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to classify.
        remote : bool | None
            Override the instance-level remote setting.

        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_samples,).
        """
        self._check_fitted()

        probs = self.predict_proba(prompts, remote=remote)

        # Handle different output shapes
        if probs.ndim == 1:
            # Binary, single value per sample
            return (probs > 0.5).astype(int)
        elif probs.ndim == 2:
            # (n_samples, n_classes)
            return self.classes_[probs.argmax(axis=1)]
        else:
            # (n_samples, seq_len, n_classes) - per-token
            return self.classes_[probs.argmax(axis=-1)]

    def predict_proba(
        self,
        prompts: list[str],
        remote: bool | None = None,
    ) -> np.ndarray:
        """Predict class probabilities for prompts.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to classify.
        remote : bool | None
            Override the instance-level remote setting.

        Returns
        -------
        np.ndarray
            Class probabilities. Shape depends on inference_pooling:
            - Normal: (n_samples, n_classes)
            - "all": (n_samples, seq_len, n_classes)
        """
        self._check_fitted()

        # Extract activations
        X, attention_mask = self._extract_and_pool(
            prompts,
            self._inference_pooling,
            remote=remote,
        )

        # Check for score-level pooling
        is_score_pooling = self._inference_pooling in SCORE_POOLING_STRATEGIES

        if X.ndim == 3:
            # Per-token activations: (batch, seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = X.shape

            # Reshape to (batch * seq_len, hidden_dim) for classification
            X_flat = X.reshape(-1, hidden_dim)

            # Classify all tokens
            probs_flat = self.classifier_.predict_proba(X_flat)

            # Reshape back to (batch, seq_len, n_classes)
            n_classes = probs_flat.shape[1]
            probs = probs_flat.reshape(batch_size, seq_len, n_classes)

            if is_score_pooling:
                # Apply score-level pooling (max/min)
                probs_tensor = torch.from_numpy(probs)
                reduced = reduce_scores(
                    probs_tensor,
                    self._inference_pooling,
                    attention_mask,
                )
                return reduced.numpy()
            else:
                # Return per-token probabilities
                return probs
        else:
            # Normal case: (batch, hidden_dim)
            return self.classifier_.predict_proba(X)

    def score(
        self,
        prompts: list[str],
        labels: list[int] | np.ndarray,
        remote: bool | None = None,
    ) -> float:
        """Compute accuracy on test data.

        Parameters
        ----------
        prompts : list[str]
            Test prompts.
        labels : list[int] | np.ndarray
            True labels.
        remote : bool | None
            Override the instance-level remote setting.

        Returns
        -------
        float
            Classification accuracy.
        """
        predictions = self.predict(prompts, remote=remote)
        labels = np.asarray(labels)
        return float((predictions == labels).mean())

    def save(self, path: str) -> None:
        """Save the fitted probe to disk.

        Parameters
        ----------
        path : str
            Path to save the probe.
        """
        self._check_fitted()

        state = {
            "model": self.model,
            "layers": self.layers,
            "pooling": self.pooling,
            "train_pooling": self.train_pooling,
            "inference_pooling": self.inference_pooling,
            "classifier": self.classifier,
            "device": self.device,
            "remote": self.remote,
            "random_state": self.random_state,
            "batch_size": self.batch_size,
            "classifier_": self.classifier_,
            "classes_": self.classes_,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "LinearProbe":
        """Load a fitted probe from disk.

        Parameters
        ----------
        path : str
            Path to the saved probe.

        Returns
        -------
        LinearProbe
            The loaded probe.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        # Create a new instance with saved config
        probe = cls(
            model=state["model"],
            layers=state["layers"],
            pooling=state["pooling"],
            train_pooling=state["train_pooling"],
            inference_pooling=state["inference_pooling"],
            classifier=state["classifier"],
            device=state["device"],
            remote=state["remote"],
            random_state=state["random_state"],
            batch_size=state.get("batch_size", 8),  # Default for older saved probes
        )

        # Restore fitted state
        probe.classifier_ = state["classifier_"]
        probe.classes_ = state["classes_"]

        return probe
