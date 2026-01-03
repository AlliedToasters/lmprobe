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
        - "auto": Automatic layer selection via Group Lasso
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
    auto_candidates : list[int] | list[float] | None, default=None
        Candidate layers for layers="auto" mode:
        - list[int]: Explicit layer indices (e.g., [10, 16, 22])
        - list[float]: Fractional positions (e.g., [0.33, 0.5, 0.66])
        - None: Default to [0.25, 0.5, 0.75]
        Only used when layers="auto".
    auto_alpha : float, default=0.01
        Group Lasso regularization strength for layers="auto".
        Higher values select fewer layers. Typical range: 0.001 to 0.1.

    Attributes
    ----------
    classifier_ : BaseEstimator
        The fitted sklearn classifier (after calling fit()).
    classes_ : np.ndarray
        Class labels (after calling fit()).
    selected_layers_ : list[int] | None
        Layer indices selected by Group Lasso when layers="auto".
        None if layers!="auto" or before fitting.

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

    >>> # Automatic layer selection
    >>> probe = LinearProbe(
    ...     model="meta-llama/Llama-3.1-8B-Instruct",
    ...     layers="auto",
    ...     auto_candidates=[0.25, 0.5, 0.75],
    ...     auto_alpha=0.01,
    ... )
    >>> probe.fit(positive_prompts, negative_prompts)
    >>> print(probe.selected_layers_)  # e.g., [8, 16]
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
        auto_candidates: list[int] | list[float] | None = None,
        auto_alpha: float = 0.01,
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
        self.auto_candidates = auto_candidates
        self.auto_alpha = auto_alpha

        # Resolve pooling strategies
        self._train_pooling, self._inference_pooling = resolve_pooling(
            pooling, train_pooling, inference_pooling
        )

        # Resolve classifier
        self._classifier_template = resolve_classifier(classifier, random_state)

        # Create extractor (lazy loads model)
        self._extractor = ActivationExtractor(
            model, device, layers, batch_size, auto_candidates=auto_candidates
        )
        self._cached_extractor = CachedExtractor(self._extractor)

        # Fitted state (set after fit())
        self.classifier_: BaseEstimator | None = None
        self.classes_: np.ndarray | None = None
        self.selected_layers_: list[int] | None = None
        self.candidate_layers_: list[int] | None = None
        self.layer_importances_: np.ndarray | None = None

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

        Notes
        -----
        When layers="auto", fitting occurs in two phases:
        1. Train Group Lasso on candidate layers to identify informative layers
        2. Re-train the specified classifier using only selected layers

        After fitting with layers="auto", check probe.selected_layers_ to see
        which layers were chosen.
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

        # Check if auto layer selection is needed
        if self.layers == "auto":
            return self._fit_auto_layers(prompts, labels, remote, invalidate_cache)

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

    def _fit_auto_layers(
        self,
        prompts: list[str],
        labels: np.ndarray,
        remote: bool | None,
        invalidate_cache: bool,
    ) -> "LinearProbe":
        """Fit with automatic layer selection via Group Lasso.

        This is a two-phase process:
        1. Train Group Lasso on candidate layers to identify selected layers
        2. Re-train the user's classifier on selected layers only
        """
        import warnings

        from .cache import CachedExtractor
        from .classifiers import build_group_lasso_classifier

        remote = self._get_remote(remote)

        # Phase 1: Extract activations from candidate layers
        X_candidates, _ = self._extract_and_pool(
            prompts,
            self._train_pooling,
            remote=remote,
            invalidate_cache=invalidate_cache,
        )

        # Handle "all" pooling (expand to per-token examples)
        if self._train_pooling == "all" and X_candidates.ndim == 3:
            batch_size_orig, seq_len, hidden_dim_total = X_candidates.shape
            X_candidates = X_candidates.reshape(-1, hidden_dim_total)
            labels_expanded = np.repeat(labels, seq_len)
        else:
            labels_expanded = labels

        # Get hidden_dim per layer and number of candidate layers
        candidate_layers = self._extractor.layer_indices
        n_candidate_layers = len(candidate_layers)
        hidden_dim_total = X_candidates.shape[1]
        hidden_dim_per_layer = hidden_dim_total // n_candidate_layers

        # Phase 1: Train Group Lasso classifier
        group_lasso_clf = build_group_lasso_classifier(
            hidden_dim=hidden_dim_per_layer,
            n_layers=n_candidate_layers,
            alpha=self.auto_alpha,
            random_state=self.random_state,
        )
        group_lasso_clf.fit(X_candidates, labels_expanded)

        # Store candidate layers and their importances (group norms)
        self.candidate_layers_ = candidate_layers
        self.layer_importances_ = group_lasso_clf.group_norms_

        # Identify selected layers
        selected_group_indices = group_lasso_clf.selected_groups_

        if not selected_group_indices:
            # All groups were zeroed out - fallback to all candidates
            warnings.warn(
                f"Group Lasso selected no layers (alpha={self.auto_alpha} may be too high). "
                "Falling back to all candidate layers. Consider reducing auto_alpha.",
                UserWarning,
            )
            selected_group_indices = list(range(n_candidate_layers))

        # Map group indices back to actual layer indices
        self.selected_layers_ = [candidate_layers[i] for i in selected_group_indices]

        # Phase 2: Re-extract activations for selected layers only
        # Create a new extractor for selected layers
        selected_extractor = ActivationExtractor(
            self.model,
            self.device,
            self.selected_layers_,
            self.batch_size,
        )
        selected_cached_extractor = CachedExtractor(selected_extractor)

        # Extract selected-layer activations
        selected_activations, selected_mask = selected_cached_extractor.extract(
            prompts,
            remote=remote,
            invalidate_cache=invalidate_cache,
        )

        # Apply pooling
        pool_fn = get_pooling_fn(self._train_pooling)
        pooled = pool_fn(selected_activations, selected_mask)
        X_selected = pooled.detach().cpu().numpy()

        # Handle "all" pooling
        if self._train_pooling == "all" and X_selected.ndim == 3:
            batch_size_sel, seq_len_sel, hidden_dim_sel = X_selected.shape
            X_selected = X_selected.reshape(-1, hidden_dim_sel)
            labels_final = np.repeat(labels, seq_len_sel)
        else:
            labels_final = labels

        # Phase 2: Train final classifier on selected layers
        self.classifier_ = clone(self._classifier_template)
        self.classifier_.fit(X_selected, labels_final)
        self.classes_ = self.classifier_.classes_

        # Update extractor to use selected layers for inference
        self._extractor = selected_extractor
        self._cached_extractor = selected_cached_extractor

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

    def plot_layer_importance(
        self,
        ax=None,
        figsize: tuple[float, float] = (10, 6),
        title: str = "Layer Importance (Group Lasso Norms)",
        xlabel: str = "Layer Index",
        ylabel: str = "Importance (L2 Norm)",
        highlight_selected: bool = True,
        bar_color: str = "steelblue",
        selected_color: str = "coral",
    ):
        """Plot layer importance scores from Group Lasso.

        Only available after fitting with layers="auto".

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            Matplotlib axes to plot on. If None, creates a new figure.
        figsize : tuple[float, float]
            Figure size if creating a new figure.
        title : str
            Plot title.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        highlight_selected : bool
            Whether to highlight selected layers in a different color.
        bar_color : str
            Color for non-selected bars.
        selected_color : str
            Color for selected layer bars.

        Returns
        -------
        tuple[Figure, Axes]
            The matplotlib figure and axes objects.

        Raises
        ------
        RuntimeError
            If the probe has not been fitted or was not fitted with layers="auto".

        Examples
        --------
        >>> probe = LinearProbe(model="...", layers="auto")
        >>> probe.fit(positive_prompts, negative_prompts)
        >>> fig, ax = probe.plot_layer_importance()
        >>> fig.savefig("layer_importance.png")
        """
        if self.candidate_layers_ is None or self.layer_importances_ is None:
            raise RuntimeError(
                "Layer importance is only available after fitting with layers='auto'. "
                "Call fit() with layers='auto' first."
            )

        from .plotting import plot_layer_importance

        return plot_layer_importance(
            candidate_layers=self.candidate_layers_,
            layer_importances=self.layer_importances_,
            selected_layers=self.selected_layers_,
            ax=ax,
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            highlight_selected=highlight_selected,
            bar_color=bar_color,
            selected_color=selected_color,
        )

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
            "auto_candidates": self.auto_candidates,
            "auto_alpha": self.auto_alpha,
            "classifier_": self.classifier_,
            "classes_": self.classes_,
            "selected_layers_": self.selected_layers_,
            "candidate_layers_": self.candidate_layers_,
            "layer_importances_": self.layer_importances_,
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

        # Handle selected_layers_ for auto mode
        layers = state["layers"]
        selected_layers = state.get("selected_layers_")

        # If auto mode was used and we have selected layers,
        # load with the selected layers directly for inference
        if layers == "auto" and selected_layers is not None:
            layers_for_extractor = selected_layers
        else:
            layers_for_extractor = layers

        # Create a new instance with saved config
        probe = cls(
            model=state["model"],
            layers=layers_for_extractor,  # Use selected layers if available
            pooling=state["pooling"],
            train_pooling=state["train_pooling"],
            inference_pooling=state["inference_pooling"],
            classifier=state["classifier"],
            device=state["device"],
            remote=state["remote"],
            random_state=state["random_state"],
            batch_size=state.get("batch_size", 8),  # Default for older saved probes
            auto_candidates=state.get("auto_candidates"),
            auto_alpha=state.get("auto_alpha", 0.01),
        )

        # Restore original layers spec for reference
        probe.layers = state["layers"]

        # Restore fitted state
        probe.classifier_ = state["classifier_"]
        probe.classes_ = state["classes_"]
        probe.selected_layers_ = selected_layers
        probe.candidate_layers_ = state.get("candidate_layers_")
        probe.layer_importances_ = state.get("layer_importances_")

        return probe
