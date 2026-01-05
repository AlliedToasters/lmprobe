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
        - "fast_auto": Fast automatic layer selection via coefficient importance
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
    normalize_layers : bool | str, default=True
        Per-layer feature standardization when using multiple layers.
        Compensates for differences in activation magnitude across layers.
        Options:
        - True or "per_neuron": Each neuron gets its own mean/std (default)
        - "per_layer": All neurons in a layer share one mean/std
          (may work better with small sample sizes due to lower variance)
        - False: No scaling
    fast_auto_top_k : int | None, default=None
        Number of layers to select when using layers="fast_auto".
        If None, defaults to selecting half the candidate layers.

    Attributes
    ----------
    classifier_ : BaseEstimator
        The fitted sklearn classifier (after calling fit()).
    classes_ : np.ndarray
        Class labels (after calling fit()).
    selected_layers_ : list[int] | None
        Layer indices selected when layers="auto" or "fast_auto".
        None for other layer modes or before fitting.
    scaler_ : PerLayerScaler | None
        The fitted per-layer scaler (after calling fit() with multiple layers
        and normalize_layers=True). None if single layer or normalize_layers=False.

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
        normalize_layers: bool | str = True,
        fast_auto_top_k: int | None = None,
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
        self.normalize_layers = normalize_layers
        self.fast_auto_top_k = fast_auto_top_k

        # Resolve pooling strategies
        self._train_pooling, self._inference_pooling = resolve_pooling(
            pooling, train_pooling, inference_pooling
        )

        # Resolve classifier
        self._classifier_template = resolve_classifier(classifier, random_state)

        # Create extractor (lazy loads model)
        # Pass remote flag so large models (e.g., 405B) don't download weights locally
        self._extractor = ActivationExtractor(
            model, device, layers, batch_size, auto_candidates=auto_candidates, remote=remote
        )
        self._cached_extractor = CachedExtractor(self._extractor)

        # Fitted state (set after fit())
        self.classifier_: BaseEstimator | None = None
        self.classes_: np.ndarray | None = None
        self.selected_layers_: list[int] | None = None
        self.candidate_layers_: list[int] | None = None
        self.layer_importances_: np.ndarray | None = None
        self.scaler_: "PerLayerScaler | None" = None  # type: ignore[name-defined]

    def _get_remote(self, remote: bool | None) -> bool:
        """Resolve remote parameter with method-level override."""
        return self.remote if remote is None else remote

    def _get_scaling_strategy(self) -> str | None:
        """Resolve normalize_layers to a scaling strategy string.

        Returns
        -------
        str | None
            "per_neuron", "per_layer", or None (no scaling).
        """
        if self.normalize_layers is False:
            return None
        if self.normalize_layers is True:
            return "per_neuron"
        if self.normalize_layers in ("per_neuron", "per_layer"):
            return self.normalize_layers
        raise ValueError(
            f"Invalid normalize_layers value: {self.normalize_layers!r}. "
            f"Expected True, False, 'per_neuron', or 'per_layer'."
        )

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
        # Use .float() to convert from bfloat16 (common in newer models) to float32
        # since numpy doesn't support bfloat16
        if pooled.dim() == 2:
            # Normal case: (batch, hidden_dim)
            return pooled.detach().cpu().float().numpy(), None
        else:
            # "all" pooling: (batch, seq_len, hidden_dim)
            # Return attention_mask for later use
            return pooled.detach().cpu().float().numpy(), attention_mask

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
        elif self.layers == "fast_auto":
            return self._fit_fast_auto_layers(prompts, labels, remote, invalidate_cache)

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

        # Apply per-layer normalization if enabled and using multiple layers
        n_layers = len(self._extractor.layer_indices)
        scaling_strategy = self._get_scaling_strategy()
        if scaling_strategy is not None and n_layers > 1:
            from .scaling import PerLayerScaler

            hidden_dim_per_layer = X.shape[1] // n_layers
            self.scaler_ = PerLayerScaler(n_layers, hidden_dim_per_layer, scaling_strategy)
            X = self.scaler_.fit_transform(X)

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
        from .scaling import PerLayerScaler

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

        # Apply per-layer normalization if enabled (before Group Lasso)
        scaling_strategy = self._get_scaling_strategy()
        if scaling_strategy is not None and n_candidate_layers > 1:
            candidate_scaler = PerLayerScaler(
                n_candidate_layers, hidden_dim_per_layer, scaling_strategy
            )
            X_candidates_scaled = candidate_scaler.fit_transform(X_candidates)
        else:
            X_candidates_scaled = X_candidates

        # Phase 1: Train Group Lasso classifier
        group_lasso_clf = build_group_lasso_classifier(
            hidden_dim=hidden_dim_per_layer,
            n_layers=n_candidate_layers,
            alpha=self.auto_alpha,
            random_state=self.random_state,
        )
        group_lasso_clf.fit(X_candidates_scaled, labels_expanded)

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

        # Phase 2: Slice selected layer activations from candidates (no re-extraction!)
        # This avoids a second forward pass through the model
        selected_columns = []
        for idx in selected_group_indices:
            start = idx * hidden_dim_per_layer
            end = (idx + 1) * hidden_dim_per_layer
            selected_columns.extend(range(start, end))
        X_selected = X_candidates[:, selected_columns]  # Use unscaled for re-fit
        labels_final = labels_expanded

        # Apply per-layer normalization to selected layers if enabled
        n_selected = len(self.selected_layers_)
        if scaling_strategy is not None and n_selected > 1:
            self.scaler_ = PerLayerScaler(n_selected, hidden_dim_per_layer, scaling_strategy)
            X_selected = self.scaler_.fit_transform(X_selected)
        elif scaling_strategy is not None and n_selected == 1:
            # Single layer: still normalize but store scaler
            self.scaler_ = PerLayerScaler(1, hidden_dim_per_layer, scaling_strategy)
            X_selected = self.scaler_.fit_transform(X_selected)
        else:
            self.scaler_ = None

        # Create extractor for selected layers (needed for inference later)
        selected_extractor = ActivationExtractor(
            self.model,
            self.device,
            self.selected_layers_,
            self.batch_size,
        )
        selected_cached_extractor = CachedExtractor(selected_extractor)

        # Phase 2: Train final classifier on selected layers
        self.classifier_ = clone(self._classifier_template)
        self.classifier_.fit(X_selected, labels_final)
        self.classes_ = self.classifier_.classes_

        # Update extractor to use selected layers for inference
        self._extractor = selected_extractor
        self._cached_extractor = selected_cached_extractor

        return self

    def _fit_fast_auto_layers(
        self,
        prompts: list[str],
        labels: np.ndarray,
        remote: bool | None,
        invalidate_cache: bool,
    ) -> "LinearProbe":
        """Fit with fast automatic layer selection via coefficient importance.

        This is a fast alternative to Group Lasso layer selection:
        1. Train the user's classifier on all candidate layers (with normalization)
        2. Compute layer importance from classifier coefficients
        3. Select top-k layers based on importance
        4. Re-train classifier on selected layers only

        This approach is much faster than Group Lasso while still providing
        interpretable layer importance scores.
        """
        import warnings

        from .cache import CachedExtractor
        from .scaling import PerLayerScaler

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

        # Store candidate layers for importance computation
        self.candidate_layers_ = list(candidate_layers)

        # Apply per-layer normalization if enabled
        scaling_strategy = self._get_scaling_strategy()
        if scaling_strategy is not None and n_candidate_layers > 1:
            scaler = PerLayerScaler(n_candidate_layers, hidden_dim_per_layer, scaling_strategy)
            X_candidates_scaled = scaler.fit_transform(X_candidates)
        else:
            scaler = None
            X_candidates_scaled = X_candidates

        # Phase 1: Train classifier on all candidate layers
        self.classifier_ = clone(self._classifier_template)
        self.classifier_.fit(X_candidates_scaled, labels_expanded)
        self.classes_ = self.classifier_.classes_

        # Phase 2: Compute layer importance from coefficients
        importance = self.compute_layer_importance(metric="l2", normalize=False)

        # Phase 3: Select top-k layers
        top_k = self.fast_auto_top_k
        if top_k is None:
            # Default: select half the candidate layers (at least 1)
            top_k = max(1, n_candidate_layers // 2)
        top_k = min(top_k, n_candidate_layers)

        # Get indices of top-k layers by importance
        top_indices = np.argsort(importance)[-top_k:]
        top_indices = np.sort(top_indices)  # Keep original order

        if len(top_indices) == 0:
            warnings.warn(
                "No layers selected. Falling back to all candidate layers.",
                UserWarning,
            )
            top_indices = np.arange(n_candidate_layers)

        self.selected_layers_ = [candidate_layers[i] for i in top_indices]

        # Phase 4: Re-train on selected layers only (slice from existing data)
        selected_columns = []
        for idx in top_indices:
            start = idx * hidden_dim_per_layer
            end = (idx + 1) * hidden_dim_per_layer
            selected_columns.extend(range(start, end))
        X_selected = X_candidates[:, selected_columns]  # Use unscaled for re-fit

        # Apply normalization to selected layers if enabled
        n_selected = len(self.selected_layers_)
        if scaling_strategy is not None and n_selected > 1:
            self.scaler_ = PerLayerScaler(n_selected, hidden_dim_per_layer, scaling_strategy)
            X_selected = self.scaler_.fit_transform(X_selected)
        elif scaling_strategy is not None and n_selected == 1:
            # Single layer: still normalize but store scaler
            self.scaler_ = PerLayerScaler(1, hidden_dim_per_layer, scaling_strategy)
            X_selected = self.scaler_.fit_transform(X_selected)
        else:
            self.scaler_ = None

        # Create extractor for selected layers (needed for inference later)
        selected_extractor = ActivationExtractor(
            self.model,
            self.device,
            self.selected_layers_,
            self.batch_size,
        )
        selected_cached_extractor = CachedExtractor(selected_extractor)

        # Re-train final classifier on selected layers
        self.classifier_ = clone(self._classifier_template)
        self.classifier_.fit(X_selected, labels_expanded)
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

        # Check if classifier supports predict_proba
        has_proba = hasattr(self.classifier_, "predict_proba")

        if has_proba:
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
        else:
            # Use classifier's native predict method
            X, attention_mask = self._extract_and_pool(
                prompts,
                self._inference_pooling,
                remote=remote,
            )

            if X.ndim == 3:
                # Per-token: (batch, seq_len, hidden_dim)
                batch_size, seq_len, hidden_dim = X.shape
                X_flat = X.reshape(-1, hidden_dim)

                # Apply scaling if fitted
                if self.scaler_ is not None:
                    X_flat = self.scaler_.transform(X_flat)

                preds_flat = self.classifier_.predict(X_flat)
                preds = preds_flat.reshape(batch_size, seq_len)

                # For per-token, return majority vote per sample
                # (assuming score-level pooling isn't needed for non-proba classifiers)
                return np.array([
                    np.bincount(p.astype(int)).argmax() for p in preds
                ])
            else:
                # Apply scaling if fitted
                if self.scaler_ is not None:
                    X = self.scaler_.transform(X)
                return self.classifier_.predict(X)

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

            # Apply scaling if fitted
            if self.scaler_ is not None:
                X_flat = self.scaler_.transform(X_flat)

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
            # Apply scaling if fitted
            if self.scaler_ is not None:
                X = self.scaler_.transform(X)
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
                "Layer importance not available. Either fit with layers='auto' or "
                "'fast_auto', or call compute_layer_importance() after fitting."
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

    def compute_layer_importance(
        self,
        metric: str = "l2",
        normalize: bool = True,
    ) -> np.ndarray:
        """Compute layer importance from classifier coefficients.

        This method analyzes the trained classifier's coefficients to determine
        which layers contribute most to the classification decision. It provides
        a fast alternative to Group Lasso for layer importance analysis.

        Must be called after fit() when using multiple layers with a linear
        classifier (one with a coef_ attribute).

        Parameters
        ----------
        metric : str, default="l2"
            How to aggregate coefficients per layer:
            - "l2": L2 norm (Euclidean magnitude) - analogous to Group Lasso
            - "l1": Sum of absolute values
            - "mean_abs": Mean absolute value (normalized by dimension)
            - "max_abs": Maximum absolute value
        normalize : bool, default=True
            If True, normalize importances to sum to 1.

        Returns
        -------
        np.ndarray
            Layer importance scores, shape (n_layers,). Also stored in
            self.layer_importances_.

        Raises
        ------
        RuntimeError
            If probe not fitted or classifier lacks coef_ attribute.
        ValueError
            If unknown metric specified.

        Examples
        --------
        >>> probe = LinearProbe(model="...", layers=[8, 16, 24])
        >>> probe.fit(positive_prompts, negative_prompts)
        >>> importance = probe.compute_layer_importance()
        >>> print(f"Layer {probe.candidate_layers_[importance.argmax()]} is most important")
        >>> fig, ax = probe.plot_layer_importance()  # Now works!
        """
        self._check_fitted()

        # Get coefficients
        if not hasattr(self.classifier_, "coef_"):
            raise RuntimeError(
                f"{type(self.classifier_).__name__} does not have coef_ attribute. "
                "compute_layer_importance() requires a linear classifier "
                "(e.g., logistic_regression, ridge, svm)."
            )

        coef = self.classifier_.coef_
        if coef.ndim == 2:
            coef = coef.flatten()  # (1, n_features) -> (n_features,)

        # Determine layer structure
        layer_indices = self._extractor.layer_indices
        n_layers = len(layer_indices)
        n_features = len(coef)

        if n_features % n_layers != 0:
            raise RuntimeError(
                f"Feature count ({n_features}) not divisible by layer count ({n_layers}). "
                "Cannot determine per-layer hidden dimension."
            )

        hidden_dim = n_features // n_layers

        # Compute importance per layer
        importances = np.zeros(n_layers)
        for i in range(n_layers):
            start = i * hidden_dim
            end = (i + 1) * hidden_dim
            layer_coef = coef[start:end]

            if metric == "l2":
                importances[i] = np.linalg.norm(layer_coef)
            elif metric == "l1":
                importances[i] = np.sum(np.abs(layer_coef))
            elif metric == "mean_abs":
                importances[i] = np.mean(np.abs(layer_coef))
            elif metric == "max_abs":
                importances[i] = np.max(np.abs(layer_coef))
            else:
                raise ValueError(
                    f"Unknown metric: {metric!r}. "
                    f"Available: 'l2', 'l1', 'mean_abs', 'max_abs'"
                )

        if normalize and importances.sum() > 0:
            importances = importances / importances.sum()

        # Store for plotting
        self.candidate_layers_ = list(layer_indices)
        self.layer_importances_ = importances

        return importances

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
            "normalize_layers": self.normalize_layers,
            "fast_auto_top_k": self.fast_auto_top_k,
            "classifier_": self.classifier_,
            "classes_": self.classes_,
            "selected_layers_": self.selected_layers_,
            "candidate_layers_": self.candidate_layers_,
            "layer_importances_": self.layer_importances_,
            "scaler_": self.scaler_,
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

        # Handle selected_layers_ for auto/fast_auto mode
        layers = state["layers"]
        selected_layers = state.get("selected_layers_")

        # If auto or fast_auto mode was used and we have selected layers,
        # load with the selected layers directly for inference
        if layers in ("auto", "fast_auto") and selected_layers is not None:
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
            normalize_layers=state.get("normalize_layers", True),
            fast_auto_top_k=state.get("fast_auto_top_k"),
        )

        # Restore original layers spec for reference
        probe.layers = state["layers"]

        # Restore fitted state
        probe.classifier_ = state["classifier_"]
        probe.classes_ = state["classes_"]
        probe.selected_layers_ = selected_layers
        probe.candidate_layers_ = state.get("candidate_layers_")
        probe.layer_importances_ = state.get("layer_importances_")
        probe.scaler_ = state.get("scaler_")

        return probe
