"""Built-in classifier factory for lmprobe.

This module provides factory functions for creating sklearn-compatible
classifiers with proper random_state propagation.
"""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.svm import SVC

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


# ---------------------------------------------------------------------------
# cuML availability check
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def cuml_available() -> bool:
    """Check whether cuML is installed and importable.

    Returns
    -------
    bool
        True if ``import cuml`` succeeds.
    """
    try:
        import cuml  # noqa: F401

        return True
    except ImportError:
        return False


_VALID_COMPUTE_BACKENDS = frozenset({"sklearn", "cuml", "auto"})


def _resolve_compute_backend(compute_backend: str) -> str:
    """Resolve ``"auto"`` to a concrete backend name.

    Parameters
    ----------
    compute_backend : str
        One of ``"sklearn"``, ``"cuml"``, or ``"auto"``.

    Returns
    -------
    str
        ``"sklearn"`` or ``"cuml"``.

    Raises
    ------
    ValueError
        If *compute_backend* is not recognised.
    ImportError
        If ``"cuml"`` is requested but not installed.
    """
    if compute_backend not in _VALID_COMPUTE_BACKENDS:
        raise ValueError(
            f"Unknown compute_backend: {compute_backend!r}. "
            f"Expected one of {sorted(_VALID_COMPUTE_BACKENDS)}."
        )
    if compute_backend == "auto":
        return "cuml" if cuml_available() else "sklearn"
    if compute_backend == "cuml" and not cuml_available():
        raise ImportError(
            "compute_backend='cuml' requires cuML. "
            "Install it with: pip install lmprobe[gpu-probes]"
        )
    return compute_backend


# Registry of built-in classifier names
BUILTIN_CLASSIFIERS = frozenset({
    "logistic_regression",
    "logistic_regression_cv",
    "ridge",
    "ridge_regression",
    "svm",
    "sgd",
    "sgd_gpu",
    "mass_mean",
    "lda",
    "ensemble",
})

# Classifiers that only work with classification tasks
CLASSIFICATION_CLASSIFIERS = BUILTIN_CLASSIFIERS - {"ridge_regression"}

# Classifiers that only work with regression tasks
REGRESSION_CLASSIFIERS = frozenset({"ridge_regression"})


def _stable_sigmoid_proba(scores: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid → (n_samples, 2) class probabilities."""
    prob_positive = np.empty_like(scores, dtype=np.float64)
    pos = scores >= 0
    neg = ~pos
    prob_positive[pos] = 1 / (1 + np.exp(-scores[pos]))
    prob_positive[neg] = np.exp(scores[neg]) / (1 + np.exp(scores[neg]))
    return np.column_stack([1 - prob_positive, prob_positive])


def _build_cuml_classifier(
    name: str,
    classifier_kwargs: dict | None = None,
) -> BaseEstimator | None:
    """Try to build a cuML classifier for *name*.

    Note: ``random_state`` is intentionally absent — cuML's LogisticRegression,
    Ridge, and SVC do not accept it.  The caller (``build_classifier``) handles
    ``random_state`` only on the sklearn fallback path.

    Returns
    -------
    BaseEstimator | None
        A cuML classifier instance, or ``None`` if no cuML equivalent exists
        for *name* (caller should fall back to sklearn).
    """
    import cuml  # guaranteed importable by caller

    extra = classifier_kwargs or {}

    if name == "logistic_regression":
        defaults: dict[str, Any] = dict(max_iter=1000)
        defaults.update(extra)
        return cuml.linear_model.LogisticRegression(**defaults)
    elif name == "ridge" or name == "ridge_regression":
        defaults = dict(alpha=1.0)
        defaults.update(extra)
        return cuml.linear_model.Ridge(**defaults)
    elif name == "svm":
        # cuML's SVC supports probability calibration via Platt scaling,
        # matching sklearn's SVC(probability=True) interface.
        defaults = dict(probability=True)
        defaults.update(extra)
        return cuml.svm.SVC(**defaults)
    else:
        # No cuML equivalent — return None so caller falls back to sklearn
        return None


# Classifiers that have cuML equivalents
_CUML_SUPPORTED_CLASSIFIERS = frozenset({
    "logistic_regression",
    "ridge",
    "ridge_regression",
    "svm",
})


def build_classifier(
    name: str,
    random_state: int | None = None,
    classifier_kwargs: dict | None = None,
    compute_backend: str = "sklearn",
) -> BaseEstimator:
    """Build a classifier by name with the given random_state.

    Parameters
    ----------
    name : str
        Name of the built-in classifier. One of:
        - "logistic_regression": L2-regularized logistic regression (default)
        - "logistic_regression_cv": Logistic regression with CV-tuned regularization
        - "ridge": Ridge classifier (fast, no probabilities)
        - "ridge_regression": Ridge regression for regression tasks
        - "svm": Linear SVM with Platt scaling for probabilities
        - "sgd": SGD classifier (scalable to large datasets)
        - "mass_mean": Mass-Mean Probing (difference-in-means direction)
        - "lda": Linear Discriminant Analysis (covariance-corrected mass mean)
    random_state : int | None
        Random seed for reproducibility. Propagated from LinearProbe.
    classifier_kwargs : dict | None
        Additional keyword arguments passed to the sklearn classifier constructor.
        These override the defaults (e.g., ``{"C": 0.01, "solver": "liblinear"}``
        for logistic regression).
    compute_backend : str
        ``"sklearn"`` (default) or ``"cuml"``. When ``"cuml"``, uses
        GPU-accelerated cuML implementations where available. Classifiers
        without a cuML equivalent fall back to sklearn automatically.

    Returns
    -------
    BaseEstimator
        An sklearn-compatible classifier instance.

    Raises
    ------
    ValueError
        If the classifier name is not recognized.
    """
    extra = classifier_kwargs or {}

    # --- cuML fast path ---
    if compute_backend == "cuml" and name in _CUML_SUPPORTED_CLASSIFIERS:
        clf = _build_cuml_classifier(name, classifier_kwargs)
        if clf is not None:
            return clf

    # --- sklearn path (default) ---
    if name == "logistic_regression":
        defaults: dict[str, Any] = dict(
            max_iter=1000, solver="lbfgs", random_state=random_state,
        )
        defaults.update(extra)
        return LogisticRegression(**defaults)
    elif name == "logistic_regression_cv":
        defaults = dict(cv=5, max_iter=1000, random_state=random_state)
        defaults.update(extra)
        return LogisticRegressionCV(**defaults)
    elif name == "ridge":
        defaults = dict(random_state=random_state)
        defaults.update(extra)
        return RidgeClassifier(**defaults)
    elif name == "ridge_regression":
        defaults = dict(alpha=1.0)
        defaults.update(extra)
        return Ridge(**defaults)
    elif name == "svm":
        defaults = dict(kernel="linear", probability=True, random_state=random_state)
        defaults.update(extra)
        return SVC(**defaults)
    elif name == "sgd":
        defaults = dict(loss="log_loss", random_state=random_state)
        defaults.update(extra)
        return SGDClassifier(**defaults)
    elif name == "sgd_gpu":
        return SGDGPUClassifier(random_state=random_state, **extra)
    elif name == "mass_mean":
        return MassMeanClassifier()
    elif name == "ensemble":
        c_values = extra.pop("C_values", None)
        return EnsembleClassifier(
            C_values=c_values,
            random_state=random_state,
            **extra,
        )
    elif name == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        return LinearDiscriminantAnalysis(**extra)
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
    classifier_kwargs: dict | None = None,
    compute_backend: str = "sklearn",
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
    classifier_kwargs : dict | None
        Additional keyword arguments for built-in classifiers.
        Ignored when a custom estimator instance is provided.
    compute_backend : str
        ``"sklearn"`` (default) or ``"cuml"``. Passed through to
        :func:`build_classifier` for built-in classifiers.

    Returns
    -------
    BaseEstimator
        The resolved classifier instance.
    """
    if isinstance(classifier, str):
        clf = build_classifier(
            classifier, random_state=random_state,
            classifier_kwargs=classifier_kwargs,
            compute_backend=compute_backend,
        )
    else:
        clf = classifier

    validate_classifier(clf)
    return clf


class MassMeanClassifier:
    """Mass-Mean Probing classifier using difference-in-means direction.

    This classifier computes the probe direction as the difference between
    the mean of positive and negative class activations:

        θ = μ_true - μ_false

    This is extremely efficient (no optimization needed) and research suggests
    it identifies directions that are more causally implicated in model outputs
    than logistic regression, despite similar classification accuracy.

    For a covariance-corrected version (equivalent to Fisher's Linear
    Discriminant), use sklearn's LinearDiscriminantAnalysis instead.

    Attributes
    ----------
    coef_ : np.ndarray
        The difference-in-means direction, shape (n_features,).
    intercept_ : float
        Decision threshold (midpoint between class means projected onto coef_).
    classes_ : np.ndarray
        Class labels [0, 1].
    mean_positive_ : np.ndarray
        Mean of positive class samples.
    mean_negative_ : np.ndarray
        Mean of negative class samples.

    References
    ----------
    Marks & Tegmark, "The Geometry of Truth" (2023)
    """

    def __init__(self) -> None:
        # No parameters - this makes get_params/set_params trivial
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.classes_: np.ndarray | None = None
        self.mean_positive_: np.ndarray | None = None
        self.mean_negative_: np.ndarray | None = None
        self._calibrator: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MassMeanClassifier:
        """Fit the Mass-Mean classifier.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).
        y : np.ndarray
            Binary labels, shape (n_samples,).

        Returns
        -------
        self
        """
        self.classes_ = np.array([0, 1])

        # Separate samples by class
        X_positive = X[y == 1]
        X_negative = X[y == 0]

        if len(X_positive) == 0 or len(X_negative) == 0:
            raise ValueError("Both classes must have at least one sample")

        # Compute class means
        self.mean_positive_ = X_positive.mean(axis=0)
        self.mean_negative_ = X_negative.mean(axis=0)

        # Direction is difference of means
        self.coef_ = self.mean_positive_ - self.mean_negative_

        # Normalize for numerical stability (optional but helpful)
        norm = np.linalg.norm(self.coef_)
        if norm > 0:
            self.coef_ = self.coef_ / norm

        # Threshold is midpoint between projected class means
        proj_positive = np.dot(self.mean_positive_, self.coef_)
        proj_negative = np.dot(self.mean_negative_, self.coef_)
        self.intercept_ = -(proj_positive + proj_negative) / 2

        # Platt scaling: fit a logistic regression on the 1D decision scores
        # to produce calibrated probabilities. This fixes AUROC without
        # changing predict() behavior (which still uses the raw threshold).
        scores = self.decision_function(X)
        self._calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
        self._calibrator.fit(scores.reshape(-1, 1), y)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision scores.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Decision scores, shape (n_samples,). Positive values indicate
            class 1, negative values indicate class 0.
        """
        if self.coef_ is None:
            raise RuntimeError("Classifier has not been fitted. Call fit() first.")

        result: np.ndarray = X @ self.coef_ + self.intercept_
        return result

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
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using Platt-scaled decision scores.

        Uses a logistic regression fitted on the 1D decision scores during
        ``fit()`` to produce calibrated probabilities (Platt scaling). This
        yields better-ranked probabilities (higher AUROC) than raw sigmoid.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Class probabilities, shape (n_samples, 2).
        """
        scores = self.decision_function(X)

        if self._calibrator is not None:
            result: np.ndarray = self._calibrator.predict_proba(scores.reshape(-1, 1))
            return result

        # Fallback: numerically stable sigmoid (should not normally be reached)
        return _stable_sigmoid_proba(scores)

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

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator (sklearn compatibility).

        Parameters
        ----------
        deep : bool
            Ignored (no nested estimators).

        Returns
        -------
        dict
            Empty dict (no hyperparameters).
        """
        return {}

    def set_params(self, **params: Any) -> MassMeanClassifier:
        """Set parameters for this estimator (sklearn compatibility).

        Parameters
        ----------
        **params
            Ignored (no hyperparameters).

        Returns
        -------
        self
        """
        return self


class EnsembleClassifier:
    """Ensemble classifier that averages predictions across regularization strengths.

    Trains multiple logistic regression models at different C values and
    averages their predicted probabilities for more robust predictions.

    Parameters
    ----------
    C_values : list[float] | None
        Regularization strengths. Defaults to [0.01, 0.1, 0.5, 1.0, 5.0].
    solver : str
        Solver for LogisticRegression. Default "lbfgs".
    max_iter : int
        Maximum iterations per model. Default 1000.
    random_state : int | None
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : np.ndarray
        Class labels [0, 1].
    estimators_ : list[LogisticRegression]
        Fitted logistic regression models.
    coef_ : np.ndarray
        Averaged coefficients across all models.
    intercept_ : np.ndarray
        Averaged intercepts across all models.
    """

    _DEFAULT_C_VALUES = [0.01, 0.1, 0.5, 1.0, 5.0]

    def __init__(
        self,
        C_values: list[float] | None = None,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.C_values = C_values if C_values is not None else self._DEFAULT_C_VALUES
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self._extra_kwargs = kwargs

        self.classes_: np.ndarray | None = None
        self.estimators_: list[LogisticRegression] | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> EnsembleClassifier:
        """Fit one LogisticRegression per C value.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).
        y : np.ndarray
            Labels, shape (n_samples,).

        Returns
        -------
        self
        """
        self.estimators_ = []
        for c_val in self.C_values:
            lr = LogisticRegression(
                C=c_val,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
                **self._extra_kwargs,
            )
            lr.fit(X, y)
            self.estimators_.append(lr)

        self.classes_ = self.estimators_[0].classes_
        self.coef_ = np.mean([e.coef_ for e in self.estimators_], axis=0)
        self.intercept_ = np.mean([e.intercept_ for e in self.estimators_], axis=0)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average predicted probabilities across all models.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Averaged class probabilities, shape (n_samples, n_classes).
        """
        if self.estimators_ is None:
            raise RuntimeError("Classifier has not been fitted. Call fit() first.")

        probas = np.array([e.predict_proba(X) for e in self.estimators_])
        result: np.ndarray = probas.mean(axis=0)
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by thresholding averaged probabilities at 0.5.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted labels, shape (n_samples,).
        """
        proba = self.predict_proba(X)
        assert self.classes_ is not None, "Classifier has not been fitted."
        result: np.ndarray = self.classes_[np.argmax(proba, axis=1)]
        return result

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
        return float((self.predict(X) == y).mean())

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator (sklearn compatibility)."""
        return {
            "C_values": self.C_values,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> EnsembleClassifier:
        """Set parameters for this estimator (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class SGDGPUClassifier:
    """GPU-accelerated SGD classifier for large-scale linear probe training.

    Implements minibatch SGD with L2 regularization using PyTorch, providing
    significant speedups over sklearn's LBFGS solver on large activation
    datasets (100k+ samples). Accepts numpy arrays (sklearn-compatible
    interface) and handles GPU memory management automatically.

    .. note::

        **Binary classification only.** Uses a single output neuron with
        BCEWithLogitsLoss. ``predict_proba`` returns shape ``(n, 2)``.

    .. note::

        **Regularization convention.** Uses ``weight_decay`` (direct
        regularization strength), unlike sklearn's ``C`` (inverse).
        Roughly: ``C ≈ 1 / (n_samples * weight_decay)``.

    Parameters
    ----------
    lr : float
        Learning rate for SGD optimizer.
    epochs : int
        Number of training epochs.
    batch_size : int
        Minibatch size for SGD.
    weight_decay : float
        L2 regularization strength (passed to SGD optimizer as weight decay).
        Higher values = stronger regularization. See note above for
        relationship to sklearn's ``C`` parameter.
    device : str
        PyTorch device. ``"auto"`` selects CUDA if available, else CPU.
    scheduler : str | None
        Learning rate schedule. Options:

        - ``None`` (default) — constant LR (current behavior)
        - ``"cosine"`` — ``CosineAnnealingLR`` decaying to 0 over ``epochs``
        - ``"reduce_on_plateau"`` — ``ReduceLROnPlateau`` (patience=5,
          factor=0.5) reducing LR when epoch loss stalls
    verbose : bool
        If True, print training loss every 10 epochs (or every epoch if
        ``epochs <= 10``).
    early_stopping : int | None
        Patience for early stopping. If set, training stops when epoch loss
        has not improved for this many consecutive epochs. The best weights
        (lowest loss) are restored.
    random_state : int | None
        Random seed for reproducibility.

    Attributes
    ----------
    coef_ : np.ndarray
        Fitted weights, shape ``(n_features,)``. Stored on CPU as float32.
    intercept_ : np.ndarray
        Fitted bias, shape ``(1,)``. Stored on CPU as float32.
    classes_ : np.ndarray
        Class labels ``[0, 1]``.
    train_loss_ : list[float]
        Per-epoch training loss history. Available after ``fit()``.
    """

    _VALID_SCHEDULERS = frozenset({None, "cosine", "reduce_on_plateau"})

    def __init__(
        self,
        lr: float = 0.01,
        epochs: int = 100,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        device: str = "auto",
        scheduler: str | None = None,
        verbose: bool = False,
        early_stopping: int | None = None,
        random_state: int | None = None,
    ) -> None:
        if scheduler not in self._VALID_SCHEDULERS:
            valid = sorted(s for s in self._VALID_SCHEDULERS if s)
            raise ValueError(
                f"Unknown scheduler: {scheduler!r}. "
                f"Valid options: {valid} or None"
            )
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.scheduler = scheduler
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.train_loss_: list[float] = []

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            if torch.cuda.is_available():
                try:
                    # Verify the GPU can actually run kernels (capability check)
                    t = torch.tensor([1.0], device="cuda")
                    _ = t + t
                    del t
                    return torch.device("cuda")
                except (RuntimeError, torch.cuda.CudaError):
                    pass
            return torch.device("cpu")
        return torch.device(self.device)

    def _check_fitted(self) -> None:
        if self.coef_ is None:
            raise RuntimeError(
                "SGDGPUClassifier has not been fitted. Call fit() first."
            )

    def _build_scheduler(self, optimizer: Any) -> Any:
        """Build the LR scheduler if configured.

        Returns None for constant LR (no scheduler).
        """
        import torch.optim.lr_scheduler as sched

        if self.scheduler is None:
            return None
        elif self.scheduler == "cosine":
            return sched.CosineAnnealingLR(optimizer, T_max=self.epochs)
        elif self.scheduler == "reduce_on_plateau":
            return sched.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5,
            )
        else:  # pragma: no cover — validated in __init__
            raise ValueError(f"Unknown scheduler: {self.scheduler!r}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> SGDGPUClassifier:
        """Fit the classifier using minibatch SGD.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape ``(n_samples, n_features)``.
        y : np.ndarray
            Labels, shape ``(n_samples,)``.
        sample_weight : np.ndarray | None
            Per-sample weights. If None, all samples are equally weighted.

        Returns
        -------
        SGDGPUClassifier
            Self, for method chaining.
        """
        import torch

        device = self._resolve_device()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_samples, n_features = X.shape

        # Reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Build model
        model = torch.nn.Linear(n_features, 1).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        lr_scheduler = self._build_scheduler(optimizer)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        # Prepare data
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        w_t = (
            torch.from_numpy(np.asarray(sample_weight, dtype=np.float32))
            if sample_weight is not None
            else None
        )

        # Shuffle indices
        rng = np.random.default_rng(self.random_state)

        # Convergence tracking
        self.train_loss_ = []
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        verbose_interval = 1 if self.epochs <= 10 else 10

        for epoch in range(self.epochs):
            perm = torch.from_numpy(rng.permutation(n_samples))
            epoch_loss_sum = 0.0
            epoch_samples = 0

            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X_t[idx].to(device)
                yb = y_t[idx].to(device)

                logits = model(xb).squeeze(-1)
                loss = loss_fn(logits, yb)

                if w_t is not None:
                    wb = w_t[idx].to(device)
                    loss = loss * wb

                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size_actual = len(idx)
                epoch_loss_sum += loss.item() * batch_size_actual
                epoch_samples += batch_size_actual

            epoch_loss = epoch_loss_sum / epoch_samples
            self.train_loss_.append(epoch_loss)

            # LR scheduler step
            if lr_scheduler is not None:
                if self.scheduler == "reduce_on_plateau":
                    lr_scheduler.step(epoch_loss)
                else:
                    lr_scheduler.step()

            # Verbose logging
            if self.verbose and (epoch % verbose_interval == 0 or epoch == self.epochs - 1):
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch:4d}/{self.epochs}  loss={epoch_loss:.6f}  lr={current_lr:.2e}")

            # Early stopping
            if self.early_stopping is not None:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_state = {
                        k: v.clone() for k, v in model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.early_stopping:
                        if self.verbose:
                            print(
                                f"Early stopping at epoch {epoch} "
                                f"(no improvement for {self.early_stopping} epochs)"
                            )
                        break

        # Restore best weights if early stopping was used and we found a best
        if best_state is not None:
            model.load_state_dict(best_state)

        # Extract weights to CPU numpy
        with torch.no_grad():
            self.coef_ = model.weight.detach().cpu().numpy().ravel()
            self.intercept_ = model.bias.detach().cpu().numpy()

        self.classes_ = np.array([0, 1])

        # Clean up GPU memory
        del model, optimizer, lr_scheduler, X_t, y_t, w_t, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute raw logit scores.

        Parameters
        ----------
        X : np.ndarray
            Features, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Logit scores, shape ``(n_samples,)``.
        """
        self._check_fitted()
        assert self.coef_ is not None and self.intercept_ is not None
        X = np.asarray(X, dtype=np.float32)
        scores: np.ndarray = X @ self.coef_ + self.intercept_[0]
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray
            Features, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Predicted labels, shape ``(n_samples,)``.
        """
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Features, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Probabilities, shape ``(n_samples, 2)``.
        """
        scores = self.decision_function(X)
        return _stable_sigmoid_proba(scores)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on the given data.

        Parameters
        ----------
        X : np.ndarray
            Features.
        y : np.ndarray
            True labels.

        Returns
        -------
        float
            Accuracy.
        """
        return float((self.predict(X) == np.asarray(y)).mean())

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for sklearn clone compatibility."""
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "device": self.device,
            "scheduler": self.scheduler,
            "verbose": self.verbose,
            "early_stopping": self.early_stopping,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> SGDGPUClassifier:
        """Set parameters for sklearn clone compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


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

    def _build_estimator(self) -> Any:
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> GroupLassoClassifier:
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
        estimator = self._build_estimator()
        estimator.fit(X, y_transformed)
        self._estimator = estimator

        # Extract coefficients (skglm returns (1, n_features), flatten to (n_features,))
        coef = estimator.coef_
        if coef.ndim == 2:
            coef = coef.flatten()
        self.coef_ = coef
        self.intercept_ = getattr(estimator, "intercept_", None)

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

        return _stable_sigmoid_proba(scores)

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
