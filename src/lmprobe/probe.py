"""LinearProbe: Train linear classifiers on language model activations.

This is the main user-facing class for lmprobe.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.base import clone

from .cache import (
    CachedExtractor,
    is_prompt_pooled_cached,
    load_prompt_pooled_activations,
)
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

    from .scaling import PerLayerScaler

def _parse_sweep_spec(spec: str) -> tuple[bool, int | list[int] | str]:
    """Parse a sweep layer specification.

    Parameters
    ----------
    spec : str
        One of: "sweep", "sweep:10", "sweep:55-65"

    Returns
    -------
    tuple[bool, int | list[int] | str]
        (is_sweep, resolved_layers) where resolved_layers is the layer
        spec to pass to sweep_layers. "sweep" -> "all",
        "sweep:10" -> step size, "sweep:55-65" -> range.
    """
    if not isinstance(spec, str) or not spec.startswith("sweep"):
        return False, spec

    if spec == "sweep":
        return True, "all"

    # Must be "sweep:..." format
    if not spec.startswith("sweep:"):
        return False, spec

    suffix = spec[len("sweep:"):]

    if "-" in suffix:
        # Range: "sweep:55-65"
        parts = suffix.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid sweep range: {spec!r}. Expected 'sweep:START-END'."
            )
        start, end = int(parts[0]), int(parts[1])
        return True, list(range(start, end + 1))

    # Step size: "sweep:10"
    step = int(suffix)
    return True, step


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(dtype: str | None) -> torch.dtype | None:
    """Resolve a dtype string to a torch.dtype, or None."""
    if dtype is None:
        return None
    if isinstance(dtype, str):
        if dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unknown dtype: {dtype!r}. "
                f"Available: {list(_DTYPE_MAP.keys())}"
            )
        return _DTYPE_MAP[dtype]
    return dtype


@dataclass
class LayerSweepResult:
    """Results from a per-layer probe sweep.

    Contains a fitted LinearProbe for each layer, with convenience methods
    for scoring and finding the best layer.

    Parameters
    ----------
    probes : dict[int, LinearProbe]
        Mapping from layer index to fitted LinearProbe.

    Examples
    --------
    >>> result = LinearProbe.sweep_layers(
    ...     model="meta-llama/Llama-3.1-8B-Instruct",
    ...     positive_prompts=pos,
    ...     negative_prompts=neg,
    ...     layers="all",
    ... )
    >>> scores = result.score(test_prompts, test_labels)
    >>> print(f"Best layer: {result.best_layer(test_prompts, test_labels)}")
    """

    probes: dict[int, LinearProbe] = field(default_factory=dict)

    @property
    def layers(self) -> list[int]:
        """Return sorted list of layer indices in this sweep."""
        return sorted(self.probes.keys())

    def __len__(self) -> int:
        return len(self.probes)

    def __getitem__(self, layer: int) -> LinearProbe:
        """Get the probe for a specific layer."""
        return self.probes[layer]

    def _warmup_cache(self, prompts: list[str]) -> None:
        """Extract all layers for prompts in a single forward pass.

        This populates the per-prompt cache so that individual single-layer
        probes can score without each triggering a separate forward pass.
        """
        if not self.probes:
            return
        # Pick any probe to access model/extractor config
        any_probe = next(iter(self.probes.values()))
        if any_probe._cached_extractor is None:
            return
        # Create a temporary extractor requesting ALL swept layers
        all_layers = sorted(self.probes.keys())
        warmup_extractor = ActivationExtractor(
            model_name=any_probe.model,
            device=any_probe.device,
            layers=all_layers,
            batch_size=any_probe.batch_size,
            backend=any_probe.backend,
            dtype=_resolve_dtype(any_probe.dtype),
        )
        warmup_cached = CachedExtractor(warmup_extractor)
        warmup_cached.extract(prompts, remote=any_probe.remote)

    def score(
        self,
        test_prompts: list[str],
        test_labels: list[int] | np.ndarray,
    ) -> dict[int, float]:
        """Score each layer's probe on test data.

        Performs a single warmup extraction pass for all layers, then
        scores each probe from cache (no redundant forward passes).

        Parameters
        ----------
        test_prompts : list[str]
            Test prompts.
        test_labels : list[int] | np.ndarray
            True labels.

        Returns
        -------
        dict[int, float]
            Mapping from layer index to accuracy.
        """
        self._warmup_cache(test_prompts)
        return {
            layer: probe.score(test_prompts, test_labels)
            for layer, probe in sorted(self.probes.items())
        }

    def best_layer(
        self,
        test_prompts: list[str],
        test_labels: list[int] | np.ndarray,
    ) -> int:
        """Return the layer index with the highest accuracy.

        Parameters
        ----------
        test_prompts : list[str]
            Test prompts.
        test_labels : list[int] | np.ndarray
            True labels.

        Returns
        -------
        int
            Layer index with the best score.
        """
        scores = self.score(test_prompts, test_labels)
        return max(scores, key=scores.get)

    def predict(
        self,
        prompts: list[str],
    ) -> dict[int, np.ndarray]:
        """Predict with each layer's probe.

        Performs a single warmup extraction pass for all layers, then
        predicts from cache (no redundant forward passes).

        Parameters
        ----------
        prompts : list[str]
            Text prompts to classify.

        Returns
        -------
        dict[int, np.ndarray]
            Mapping from layer index to predictions array.
        """
        self._warmup_cache(prompts)
        return {
            layer: probe.predict(prompts)
            for layer, probe in sorted(self.probes.items())
        }

    def predict_proba(
        self,
        prompts: list[str],
    ) -> dict[int, np.ndarray]:
        """Predict probabilities with each layer's probe.

        Performs a single warmup extraction pass for all layers, then
        predicts from cache (no redundant forward passes).

        Parameters
        ----------
        prompts : list[str]
            Text prompts to classify.

        Returns
        -------
        dict[int, np.ndarray]
            Mapping from layer index to probability arrays.
        """
        self._warmup_cache(prompts)
        return {
            layer: probe.predict_proba(prompts)
            for layer, probe in sorted(self.probes.items())
        }


class LinearProbe:
    """Train a linear probe on language model activations.

    Parameters
    ----------
    model : str | None
        HuggingFace model ID or local path. Optional when using
        *_from_activations() methods only.
    layers : int | list[int] | str, default="middle"
        Which layers to extract activations from:
        - int: Single layer (negative indexing supported)
        - list[int]: Multiple layers (concatenated)
        - "middle": Middle third of layers
        - "last": Last layer only
        - "all": All layers
        - "auto": Automatic layer selection via Group Lasso
        - "fast_auto": Fast automatic layer selection via coefficient importance
        - "sweep": Train independent probe per layer (memory-safe)
        - "sweep:N": Sweep every Nth layer (coarse sweep)
        - "sweep:START-END": Sweep a range of layers (fine sweep)
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
    task : str, default="classification"
        Task type: "classification" or "regression".
        When "regression", defaults to Ridge regression and disables predict_proba.
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
    backend : str, default="local"
        Extraction backend: "nnsight" (default) or "local".
        "local" uses HuggingFace transformers directly without nnsight,
        enabling use with models not supported by nnsight/NDIF.
    dtype : str | None, default=None
        Model dtype for local backend: "float32", "float16", or "bfloat16".
        Defaults to "float32" if None. Ignored for nnsight backend.
    classifier_kwargs : dict | None, default=None
        Additional keyword arguments passed to the sklearn classifier constructor.
        Overrides defaults for built-in classifiers. Example:
        ``{"C": 0.01, "solver": "liblinear", "max_iter": 5000}``
        for logistic regression.
    preprocessing : str | list[str] | None, default=None
        Preprocessing pipeline applied after layer scaling but before the
        classifier. Steps are separated by ``"+"`` when given as a string:
        - ``"standard"``: StandardScaler
        - ``"pca"`` or ``"pca:N"``: PCA with N components
        - ``"standard+pca"``: StandardScaler then PCA
        Use ``pca_components`` to set N when using ``"pca"`` without ``:N``.
    pca_components : int | None, default=None
        Number of PCA components when ``preprocessing`` includes ``"pca"``
        without an explicit component count.

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
        model: str | None = None,
        layers: int | list[int] | str = "middle",
        pooling: str = "last_token",
        train_pooling: str | None = None,
        inference_pooling: str | None = None,
        classifier: str | BaseEstimator = "logistic_regression",
        task: str = "classification",
        device: str = "auto",
        remote: bool = False,
        random_state: int | None = None,
        batch_size: int = 8,
        auto_candidates: list[int] | list[float] | None = None,
        auto_alpha: float = 0.01,
        normalize_layers: bool | str = True,
        fast_auto_top_k: int | None = None,
        backend: str = "local",
        dtype: str | None = None,
        max_retries: int | None = None,
        classifier_kwargs: dict | None = None,
        preprocessing: str | list[str] | None = None,
        pca_components: int | None = None,
    ):
        self.model = model
        self.layers = layers
        self.pooling = pooling
        self.train_pooling = train_pooling
        self.inference_pooling = inference_pooling
        self.classifier = classifier
        self.task = task
        self.device = device
        self.remote = remote
        self.random_state = random_state
        self.batch_size = batch_size
        self.auto_candidates = auto_candidates
        self.auto_alpha = auto_alpha
        self.normalize_layers = normalize_layers
        self.fast_auto_top_k = fast_auto_top_k
        self.backend = backend
        self.dtype = dtype
        self.max_retries = max_retries
        self.classifier_kwargs = classifier_kwargs
        self.preprocessing = preprocessing
        self.pca_components = pca_components

        # Detect sweep mode before other validations
        self._sweep_mode, self._sweep_layers_spec = _parse_sweep_spec(layers)

        # Validate task
        if task not in ("classification", "regression"):
            raise ValueError(
                f"Unknown task: {task!r}. Expected 'classification' or 'regression'."
            )

        # Validate backend + remote compatibility
        if backend == "local" and remote:
            raise ValueError(
                "backend='local' does not support remote=True. "
                "Use backend='nnsight' for remote execution."
            )

        # Resolve pooling strategies (only needed if model is provided)
        if model is not None:
            self._train_pooling, self._inference_pooling = resolve_pooling(
                pooling, train_pooling, inference_pooling
            )
        else:
            self._train_pooling = None
            self._inference_pooling = None

        # Resolve classifier (use regression default if task="regression" and no custom classifier)
        if task == "regression" and classifier == "logistic_regression":
            # Default regression classifier
            self._classifier_template = resolve_classifier(
                "ridge_regression", random_state,
                classifier_kwargs=classifier_kwargs,
            )
        else:
            self._classifier_template = resolve_classifier(
                classifier, random_state,
                classifier_kwargs=classifier_kwargs,
            )

        # Create extractor (lazy loads model) only if model is provided
        # Skip for sweep mode — sweep creates its own extractors
        if model is not None and not self._sweep_mode:
            _torch_dtype = _resolve_dtype(dtype)
            self._extractor = ActivationExtractor(
                model, device, layers, batch_size,
                auto_candidates=auto_candidates, remote=remote, backend=backend,
                dtype=_torch_dtype,
            )
            self._cached_extractor = CachedExtractor(self._extractor)
        else:
            self._extractor = None
            self._cached_extractor = None

        # Fitted state (set after fit())
        self.classifier_: BaseEstimator | None = None
        self.classes_: np.ndarray | None = None
        self.selected_layers_: list[int] | None = None
        self.candidate_layers_: list[int] | None = None
        self.layer_importances_: np.ndarray | None = None
        self.scaler_: PerLayerScaler | None = None
        self.preprocessing_pipeline_: object | None = None
        self.sweep_result_: LayerSweepResult | None = None

        # Training data cache (for push_to_hub)
        self._training_positive_: list[str] | None = None
        self._training_negative_: list[str] | None = None
        self._training_prompts_: list[str] | None = None
        self._training_labels_: list[int] | None = None

        # Evaluation results cache (for push_to_hub)
        self._evaluation_results_: dict | None = None

    def _check_model(self) -> None:
        """Check that a model was provided (needed for prompt-based methods)."""
        if self.model is None:
            raise ValueError(
                "No model specified. Either pass model= to LinearProbe() "
                "or use the *_from_activations() methods instead."
            )

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

    def _build_preprocessing_pipeline(self):
        """Build a sklearn Pipeline from the preprocessing specification.

        Returns
        -------
        sklearn.pipeline.Pipeline | None
            A fitted preprocessing pipeline, or None if no preprocessing.
        """
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        spec = self.preprocessing
        if spec is None:
            return None

        # Normalize to list of step strings
        if isinstance(spec, str):
            steps_spec = [s.strip() for s in spec.split("+")]
        else:
            steps_spec = list(spec)

        steps = []
        for s in steps_spec:
            if s == "standard_scaler" or s == "standard":
                steps.append(("scaler", StandardScaler()))
            elif s.startswith("pca"):
                # "pca", "pca:200"
                if ":" in s:
                    n = int(s.split(":")[1])
                else:
                    n = self.pca_components
                if n is None:
                    raise ValueError(
                        "PCA requested but no component count specified. "
                        "Use preprocessing='standard+pca:200' or set pca_components=200."
                    )
                steps.append(("pca", PCA(n_components=n)))
            else:
                raise ValueError(
                    f"Unknown preprocessing step: {s!r}. "
                    f"Available: 'standard', 'standard_scaler', 'pca', 'pca:<N>'."
                )

        if not steps:
            return None

        return Pipeline(steps)

    def _try_load_from_pooled_cache(
        self,
        prompts: list[str],
        pooling_strategy: str,
    ) -> np.ndarray | None:
        """Try to load pre-pooled activations from cache.

        This checks if UnifiedCache (or similar) has already cached pooled
        activations for all prompts. If so, we can skip both extraction AND
        pooling, loading the final result directly.

        Parameters
        ----------
        prompts : list[str]
            Prompts to check.
        pooling_strategy : str
            Pooling strategy (must match what was used during caching).

        Returns
        -------
        np.ndarray | None
            Pooled activations with shape (batch, hidden_dim) if all prompts
            are in pooled cache. None if any prompt is missing.
        """
        # "all" pooling cannot use pre-pooled cache (needs full sequence)
        if pooling_strategy == "all":
            return None

        layer_indices = self._extractor.layer_indices
        required_layers = set(layer_indices)

        # Check if ALL prompts have pooled cache
        for prompt in prompts:
            if not is_prompt_pooled_cached(
                self.model, prompt, required_layers, pooling_strategy
            ):
                return None

        # All prompts are in pooled cache - load directly!
        all_activations = []
        sorted_layers = sorted(layer_indices)
        for prompt in prompts:
            acts = load_prompt_pooled_activations(
                self.model, prompt, sorted_layers, pooling_strategy
            )
            all_activations.append(acts)

        # Concatenate along batch dimension
        # Each acts has shape (1, n_layers * hidden_dim)
        pooled = torch.cat(all_activations, dim=0)
        return pooled.detach().cpu().float().numpy()

    def _extract_and_pool(
        self,
        prompts: list[str],
        pooling_strategy: str,
        remote: bool | None = None,
        invalidate_cache: bool = False,
        max_retries: int | None = None,
    ) -> tuple[np.ndarray, torch.Tensor | None]:
        """Extract activations and apply pooling.

        This method first checks if pre-pooled activations are available
        in cache (e.g., from UnifiedCache with cache_pooled=True). If so,
        it loads them directly, skipping both extraction and pooling.

        Returns
        -------
        tuple[np.ndarray, torch.Tensor | None]
            (pooled_activations, attention_mask)
            attention_mask is returned for score-level pooling
        """
        remote = self._get_remote(remote)

        # Try to load from pooled cache first (skip extraction + pooling)
        if not invalidate_cache:
            pooled_from_cache = self._try_load_from_pooled_cache(
                prompts, pooling_strategy
            )
            if pooled_from_cache is not None:
                # Success! Return directly without extraction
                return pooled_from_cache, None

        # Fall back to extraction + pooling
        effective_retries = max_retries if max_retries is not None else self.max_retries
        activations, attention_mask = self._cached_extractor.extract(
            prompts,
            remote=remote,
            invalidate_cache=invalidate_cache,
            max_retries=effective_retries,
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

    def warmup(
        self,
        prompts: list[str],
        remote: bool | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Extract and cache activations without training a classifier.

        Use this to pre-populate the activation cache for a set of prompts.
        This is useful when you want to separate the (expensive) activation
        extraction step from the (cheap) classifier training step, or when
        you plan to train multiple probes on the same prompts.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to extract and cache activations for.
        remote : bool | None
            Override the instance-level remote setting.
        max_retries : int | None
            Override the instance-level max_retries setting.
            Only applies to remote extraction.
        """
        self._check_model()
        use_remote = self._get_remote(remote)
        effective_retries = max_retries if max_retries is not None else self.max_retries
        self._cached_extractor.extract(
            prompts, remote=use_remote, max_retries=effective_retries,
            cache_only=True,
        )

    def fit(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str] | np.ndarray | list[int] | None = None,
        remote: bool | None = None,
        invalidate_cache: bool = False,
        max_retries: int | None = None,
    ) -> LinearProbe:
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
        max_retries : int | None
            Override the instance-level max_retries setting.
            Only applies to remote extraction.

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
        self._check_model()

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
            # Cache for push_to_hub
            self._training_prompts_ = list(positive_prompts)
            self._training_labels_ = list(negative_prompts)
            self._training_positive_ = None
            self._training_negative_ = None
        else:
            # Contrastive mode: fit(positive_prompts, negative_prompts)
            prompts = list(positive_prompts) + list(negative_prompts)
            labels = np.array(
                [1] * len(positive_prompts) + [0] * len(negative_prompts)
            )
            # Cache for push_to_hub
            self._training_positive_ = list(positive_prompts)
            self._training_negative_ = list(negative_prompts)
            self._training_prompts_ = None
            self._training_labels_ = None

        # Check if sweep mode
        if self._sweep_mode:
            return self._fit_sweep(
                positive_prompts, negative_prompts, remote,
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
            max_retries=max_retries,
        )

        # Handle "all" pooling for training (expand to per-token examples)
        if self._train_pooling == "all" and X.ndim == 3:
            # X is (batch, seq_len, hidden_dim)
            # Expand to (batch * seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = X.shape
            X = X.reshape(-1, hidden_dim)
            # Repeat labels for each token
            labels = np.repeat(labels, seq_len)

        # Apply per-layer normalization if enabled
        n_layers = len(self._extractor.layer_indices)
        scaling_strategy = self._get_scaling_strategy()
        if scaling_strategy is not None and n_layers > 1:
            from .scaling import PerLayerScaler

            hidden_dim_per_layer = X.shape[1] // n_layers
            self.scaler_ = PerLayerScaler(n_layers, hidden_dim_per_layer, scaling_strategy)
            X = self.scaler_.fit_transform(X)
        elif n_layers == 1:
            # Apply StandardScaler for single-layer probes to prevent
            # convergence issues with unscaled activations (#40)
            from sklearn.preprocessing import StandardScaler

            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        # Apply user-specified preprocessing (StandardScaler, PCA, etc.)
        self.preprocessing_pipeline_ = self._build_preprocessing_pipeline()
        if self.preprocessing_pipeline_ is not None:
            X = self.preprocessing_pipeline_.fit_transform(X)

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
    ) -> LinearProbe:
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
            backend=self.backend,
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
    ) -> LinearProbe:
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
            backend=self.backend,
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

    def _fit_sweep(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        remote: bool | None,
    ) -> LinearProbe:
        """Fit using sweep mode: train an independent probe per layer.

        Resolves the sweep spec to layer indices, delegates to sweep_layers(),
        and sets the best layer's probe as the active classifier.
        """
        from .extraction import get_num_layers_from_config, resolve_layers

        self._check_model()

        num_layers = get_num_layers_from_config(self.model)
        spec = self._sweep_layers_spec

        if isinstance(spec, int):
            # Step size: "sweep:10" → every 10th layer
            sweep_layers_list = list(range(0, num_layers, spec))
        elif isinstance(spec, list):
            # Explicit layer list (from range parse)
            sweep_layers_list = [idx for idx in spec if 0 <= idx < num_layers]
        elif isinstance(spec, str):
            # "all", "middle", etc.
            sweep_layers_list = resolve_layers(spec, num_layers)
        else:
            sweep_layers_list = resolve_layers("all", num_layers)

        # Delegate to sweep_layers classmethod
        self.sweep_result_ = type(self).sweep_layers(
            model=self.model,
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            layers=sweep_layers_list,
            pooling=self.pooling,
            classifier=self.classifier,
            device=self.device,
            remote=self._get_remote(remote),
            random_state=self.random_state,
            batch_size=self.batch_size,
            backend=self.backend,
            dtype=self.dtype,
            normalize_layers=self.normalize_layers,
            classifier_kwargs=self.classifier_kwargs,
            preprocessing=self.preprocessing,
            pca_components=self.pca_components,
        )

        # Use the first layer's probe as a default active probe
        # (user can pick the best after evaluate())
        first_layer = self.sweep_result_.layers[0]
        best_probe = self.sweep_result_[first_layer]
        self.classifier_ = best_probe.classifier_
        self.classes_ = best_probe.classes_
        self.scaler_ = best_probe.scaler_
        self.preprocessing_pipeline_ = best_probe.preprocessing_pipeline_
        self._extractor = best_probe._extractor
        self._cached_extractor = best_probe._cached_extractor

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
                # Apply preprocessing if fitted
                if self.preprocessing_pipeline_ is not None:
                    X_flat = self.preprocessing_pipeline_.transform(X_flat)

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
                # Apply preprocessing if fitted
                if self.preprocessing_pipeline_ is not None:
                    X = self.preprocessing_pipeline_.transform(X)
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

        if self.task == "regression":
            raise ValueError(
                "predict_proba is not available for regression tasks."
            )

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
            # Apply preprocessing if fitted
            if self.preprocessing_pipeline_ is not None:
                X_flat = self.preprocessing_pipeline_.transform(X_flat)

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
                return reduced.float().numpy()
            else:
                # Return per-token probabilities
                return probs
        else:
            # Normal case: (batch, hidden_dim)
            # Apply scaling if fitted
            if self.scaler_ is not None:
                X = self.scaler_.transform(X)
            # Apply preprocessing if fitted
            if self.preprocessing_pipeline_ is not None:
                X = self.preprocessing_pipeline_.transform(X)
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

    def evaluate(
        self,
        prompts: list[str],
        labels: list[int] | np.ndarray,
        remote: bool | None = None,
    ) -> dict:
        """Compute a standard set of evaluation metrics.

        Computes accuracy, AUROC, F1, precision, and recall. Results are
        cached on ``self._evaluation_results_`` for use by ``push_to_hub()``.

        Parameters
        ----------
        prompts : list[str]
            Evaluation prompts (should NOT be training data).
        labels : list[int] | np.ndarray
            True labels.
        remote : bool | None
            Override the instance-level remote setting.

        Returns
        -------
        dict
            Metrics dict with keys: accuracy, auroc, f1, precision, recall,
            n_eval, eval_hash.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        self._check_fitted()
        labels = np.asarray(labels)

        # Sweep mode: evaluate each layer and return per-layer + summary
        if self.sweep_result_ is not None:
            layer_results = {}
            scores = self.sweep_result_.score(prompts, labels)
            for layer_idx, probe in sorted(self.sweep_result_.probes.items()):
                layer_preds = probe.predict(prompts, remote=remote)
                layer_metrics: dict = {
                    "accuracy": float(accuracy_score(labels, layer_preds)),
                    "f1": float(f1_score(labels, layer_preds, zero_division=0)),
                    "precision": float(precision_score(labels, layer_preds, zero_division=0)),
                    "recall": float(recall_score(labels, layer_preds, zero_division=0)),
                }
                if hasattr(probe.classifier_, "predict_proba"):
                    try:
                        from sklearn.metrics import roc_auc_score

                        proba = probe.predict_proba(prompts, remote=remote)
                        if proba.ndim == 2 and proba.shape[1] == 2:
                            layer_metrics["auroc"] = float(roc_auc_score(labels, proba[:, 1]))
                        else:
                            layer_metrics["auroc"] = float(roc_auc_score(labels, proba))
                    except Exception:
                        pass
                layer_results[layer_idx] = layer_metrics

            best_layer = max(scores, key=scores.get)
            results: dict = {
                "layer_results": layer_results,
                "best_layer": best_layer,
                "best_accuracy": scores[best_layer],
                **layer_results[best_layer],  # top-level metrics from best layer
            }

            # Update this probe to use the best layer's probe
            best_probe = self.sweep_result_[best_layer]
            self.classifier_ = best_probe.classifier_
            self.classes_ = best_probe.classes_
            self.scaler_ = best_probe.scaler_
            self.preprocessing_pipeline_ = best_probe.preprocessing_pipeline_
            self._extractor = best_probe._extractor
            self._cached_extractor = best_probe._cached_extractor

            self._evaluation_results_ = results
            return results

        predictions = self.predict(prompts, remote=remote)

        results: dict = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
        }

        # AUROC if predict_proba is available
        if self.task == "classification" and hasattr(self.classifier_, "predict_proba"):
            try:
                from sklearn.metrics import roc_auc_score

                probabilities = self.predict_proba(prompts, remote=remote)
                if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                    results["auroc"] = float(roc_auc_score(labels, probabilities[:, 1]))
                else:
                    results["auroc"] = float(roc_auc_score(labels, probabilities))
            except Exception:
                pass

        # Metadata
        from .hub import _hash_prompts

        label_strs = [str(val) for val in labels]
        combined = list(prompts) + label_strs
        results["n_eval"] = len(prompts)
        results["eval_hash"] = _hash_prompts(combined)

        self._evaluation_results_ = results
        return results

    def push_to_hub(
        self,
        repo_id: str,
        description: str | None = None,
        class_labels: dict[int, str] | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        include_training_data: bool = True,
        training_prompts: tuple[list[str], list[str]] | None = None,
        private: bool = False,
        license: str = "mit",
        commit_message: str = "Upload lmprobe probe",
        limitations: str | None = None,
    ) -> str:
        """Push this fitted probe to the HuggingFace Hub.

        Parameters
        ----------
        repo_id : str
            HuggingFace Hub repository ID (e.g., "username/probe-name").
        description : str | None
            Human-readable description.
        class_labels : dict[int, str] | None
            Human-readable class labels.
        tags : list[str] | None
            Additional tags.
        metrics : dict[str, float] | None
            Evaluation metrics (overrides cached evaluate() results).
        include_training_data : bool
            Include training prompts in training_info.json.
        training_prompts : tuple[list[str], list[str]] | None
            (positive, negative) prompts if not cached from fit().
        private : bool
            Create a private repository.
        license : str
            License identifier.
        commit_message : str
            Git commit message for the upload.
        limitations : str | None
            Limitations and intended use text for the model card.
            If None, the section is omitted.

        Returns
        -------
        str
            URL of the created/updated Hub repository.
        """
        from .hub import push_to_hub

        return push_to_hub(
            self,
            repo_id=repo_id,
            description=description,
            class_labels=class_labels,
            tags=tags,
            metrics=metrics,
            include_training_data=include_training_data,
            training_prompts=training_prompts,
            private=private,
            license=license,
            commit_message=commit_message,
            limitations=limitations,
        )

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        revision: str | None = None,
        trust_classifier: bool = False,
        load_model: bool = False,
        device: str | None = None,
    ) -> LinearProbe:
        """Load a probe from the HuggingFace Hub.

        Parameters
        ----------
        repo_id : str
            HuggingFace Hub repository ID.
        revision : str | None
            Specific commit of the probe repo.
        trust_classifier : bool
            Must be True to load the classifier. Required for security.
        load_model : bool
            If True, download and initialize the base model.
        device : str | None
            Override device for inference.

        Returns
        -------
        LinearProbe
            The loaded probe.
        """
        from .hub import from_hub

        return from_hub(
            repo_id=repo_id,
            revision=revision,
            trust_classifier=trust_classifier,
            load_model=load_model,
            device=device,
        )

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

    @staticmethod
    def _to_numpy(X) -> np.ndarray:
        """Convert input to numpy array, handling torch tensors."""
        if isinstance(X, torch.Tensor):
            return X.detach().cpu().float().numpy()
        return np.asarray(X)

    def fit_from_activations(
        self,
        X,
        y,
    ) -> LinearProbe:
        """Fit the probe from pre-computed activation tensors.

        Skips all extraction and pooling logic, going straight to
        classifier fitting.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Pre-computed activations, shape (n_samples, n_features).
        y : np.ndarray | torch.Tensor
            Labels. int for classification, float for regression.

        Returns
        -------
        LinearProbe
            Self, for method chaining.
        """
        X = self._to_numpy(X)
        y = self._to_numpy(y)

        # Clone and fit classifier
        self.classifier_ = clone(self._classifier_template)
        self.classifier_.fit(X, y)

        # Set classes_ for classification, None for regression
        if self.task == "classification":
            if hasattr(self.classifier_, "classes_"):
                self.classes_ = self.classifier_.classes_
            else:
                self.classes_ = np.unique(y)
        else:
            self.classes_ = None

        return self

    def predict_from_activations(self, X) -> np.ndarray:
        """Predict from pre-computed activation tensors.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Pre-computed activations, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples,).
        """
        self._check_fitted()
        X = self._to_numpy(X)
        return self.classifier_.predict(X)

    def predict_proba_from_activations(self, X) -> np.ndarray:
        """Predict probabilities from pre-computed activation tensors.

        Only available for classification tasks.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Pre-computed activations, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Class probabilities, shape (n_samples, n_classes).

        Raises
        ------
        ValueError
            If task is regression.
        """
        self._check_fitted()
        if self.task == "regression":
            raise ValueError(
                "predict_proba is not available for regression tasks."
            )
        X = self._to_numpy(X)
        return self.classifier_.predict_proba(X)

    def score_from_activations(self, X, y) -> float:
        """Score the probe on pre-computed activation tensors.

        Returns accuracy for classification, R-squared for regression.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Pre-computed activations, shape (n_samples, n_features).
        y : np.ndarray | torch.Tensor
            True labels/values.

        Returns
        -------
        float
            Accuracy (classification) or R-squared (regression).
        """
        self._check_fitted()
        X = self._to_numpy(X)
        y = self._to_numpy(y)
        return self.classifier_.score(X, y)

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
            "task": self.task,
            "device": self.device,
            "remote": self.remote,
            "random_state": self.random_state,
            "batch_size": self.batch_size,
            "auto_candidates": self.auto_candidates,
            "auto_alpha": self.auto_alpha,
            "normalize_layers": self.normalize_layers,
            "fast_auto_top_k": self.fast_auto_top_k,
            "backend": self.backend,
            "dtype": self.dtype,
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
    def load(cls, path: str) -> LinearProbe:
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
            task=state.get("task", "classification"),
            device=state["device"],
            remote=state["remote"],
            random_state=state["random_state"],
            batch_size=state.get("batch_size", 8),  # Default for older saved probes
            auto_candidates=state.get("auto_candidates"),
            auto_alpha=state.get("auto_alpha", 0.01),
            normalize_layers=state.get("normalize_layers", True),
            fast_auto_top_k=state.get("fast_auto_top_k"),
            backend=state.get("backend", "local"),
            dtype=state.get("dtype"),
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

    @classmethod
    def sweep_layers(
        cls,
        model: str,
        positive_prompts: list[str],
        negative_prompts: list[str],
        layers: int | list[int] | str = "all",
        pooling: str = "last_token",
        classifier: str | BaseEstimator = "logistic_regression",
        device: str = "auto",
        remote: bool = False,
        random_state: int | None = None,
        batch_size: int = 8,
        backend: str = "local",
        dtype: str | None = None,
        normalize_layers: bool | str = True,
        classifier_kwargs: dict | None = None,
        preprocessing: str | list[str] | None = None,
        pca_components: int | None = None,
    ) -> LayerSweepResult:
        """Train a probe at every layer and return per-layer results.

        This method avoids the boilerplate of manually looping over layers.
        It performs one warmup pass extracting all requested layers (single
        forward pass through the model, cached), then trains an independent
        single-layer probe for each layer using cached activations.

        Parameters
        ----------
        model : str
            HuggingFace model ID or local path.
        positive_prompts : list[str]
            Prompts for the positive class.
        negative_prompts : list[str]
            Prompts for the negative class.
        layers : int | list[int] | str, default="all"
            Which layers to sweep. Accepts same specifications as LinearProbe:
            int, list[int], "all", "middle", "last".
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
        batch_size : int, default=8
            Number of prompts per batch during extraction.
        backend : str, default="local"
            Extraction backend: "local" or "nnsight".
        dtype : str | None, default=None
            Model dtype for local backend.
        normalize_layers : bool | str, default=True
            Per-layer normalization (applied per single-layer probe).

        Returns
        -------
        LayerSweepResult
            Contains a fitted probe for each layer, with methods for
            scoring and finding the best layer.

        Examples
        --------
        >>> result = LinearProbe.sweep_layers(
        ...     model="meta-llama/Llama-3.1-8B-Instruct",
        ...     positive_prompts=pos,
        ...     negative_prompts=neg,
        ...     layers="all",
        ... )
        >>> scores = result.score(test_prompts, test_labels)
        >>> best = result.best_layer(test_prompts, test_labels)
        >>> print(f"Best layer: {best}, accuracy: {scores[best]:.3f}")
        """
        from .extraction import get_num_layers_from_config, resolve_layers

        # Resolve which layers to sweep
        num_layers = get_num_layers_from_config(model)
        layer_indices = resolve_layers(layers, num_layers)

        # Step 1: Warmup pass - extract ALL requested layers at once.
        # This ensures a single forward pass through the model, with all
        # layer activations cached to disk.
        warmup_probe = cls(
            model=model,
            layers=layer_indices,
            pooling=pooling,
            classifier=classifier,
            device=device,
            remote=remote,
            random_state=random_state,
            batch_size=batch_size,
            backend=backend,
            dtype=dtype,
            normalize_layers=False,  # No scaling needed for warmup
            classifier_kwargs=classifier_kwargs,
        )

        all_prompts = list(positive_prompts) + list(negative_prompts)
        warmup_probe._cached_extractor.extract(
            all_prompts,
            remote=remote,
        )

        # Step 2: Train individual single-layer probes.
        # Each probe will hit the cache (no model inference needed).
        probes: dict[int, LinearProbe] = {}
        for layer_idx in layer_indices:
            probe = cls(
                model=model,
                layers=layer_idx,
                pooling=pooling,
                classifier=classifier,
                device=device,
                remote=remote,
                random_state=random_state,
                batch_size=batch_size,
                backend=backend,
                dtype=dtype,
                normalize_layers=normalize_layers,
                classifier_kwargs=classifier_kwargs,
                preprocessing=preprocessing,
                pca_components=pca_components,
            )
            probe.fit(positive_prompts, negative_prompts, remote=remote)
            probes[layer_idx] = probe

        return LayerSweepResult(probes=probes)
