"""BaselineBattery for running all baselines at once.

This module provides a convenient way to run multiple baselines and compare
their performance, helping determine if a linear probe is learning something
beyond what simpler approaches can achieve.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from .activation_baseline import ActivationBaseline
from .baseline import BaselineProbe

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


@dataclass
class BaselineResult:
    """Result for a single baseline.

    Attributes
    ----------
    name : str
        Name of the baseline method.
    score : float
        Score on the test set (higher is better).
    baseline : BaselineProbe | ActivationBaseline
        The fitted baseline instance.
    fit_time : float
        Time in seconds to fit the baseline.
    predict_time : float
        Time in seconds for prediction on test set.
    """

    name: str
    score: float
    baseline: BaselineProbe | ActivationBaseline
    fit_time: float = 0.0
    predict_time: float = 0.0

    def __repr__(self) -> str:
        return f"BaselineResult(name={self.name!r}, score={self.score:.4f})"


@dataclass
class BaselineResults:
    """Results from BaselineBattery.fit().

    Attributes
    ----------
    results : list[BaselineResult]
        List of results for each baseline, unsorted.
    """

    results: list[BaselineResult] = field(default_factory=list)

    def get_best(self, n: int = 1) -> list[BaselineResult]:
        """Return top n baselines by score, descending.

        Parameters
        ----------
        n : int, default=1
            Number of top baselines to return.

        Returns
        -------
        list[BaselineResult]
            Top n baselines sorted by score (highest first).
        """
        sorted_results = sorted(self.results, key=lambda r: r.score, reverse=True)
        return sorted_results[:n]

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, key: str | int) -> BaselineResult:
        """Get a result by name or index."""
        if isinstance(key, int):
            return self.results[key]
        for r in self.results:
            if r.name == key:
                return r
        raise KeyError(f"Baseline {key!r} not found. Available: {[r.name for r in self.results]}")

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = ["Baseline Results:", "-" * 60]
        for r in sorted(self.results, key=lambda x: x.score, reverse=True):
            lines.append(
                f"  {r.name:30s} {r.score:.4f}  (fit: {r.fit_time:.2f}s, predict: {r.predict_time:.2f}s)"
            )
        return "\n".join(lines)


# Define all available baselines and their requirements
BASELINE_REGISTRY: dict[str, dict] = {
    # Text-only baselines (no model required)
    "bow": {
        "class": BaselineProbe,
        "needs_model": False,
        "kwargs": {"method": "bow"},
    },
    "tfidf": {
        "class": BaselineProbe,
        "needs_model": False,
        "kwargs": {"method": "tfidf"},
    },
    "random": {
        "class": BaselineProbe,
        "needs_model": False,
        "kwargs": {"method": "random"},
    },
    "majority": {
        "class": BaselineProbe,
        "needs_model": False,
        "kwargs": {"method": "majority"},
    },
    "sentence_length": {
        "class": BaselineProbe,
        "needs_model": False,
        "kwargs": {"method": "sentence_length"},
    },
    # External embedding baselines (uses its own model, not the user's)
    "sentence_transformers": {
        "class": BaselineProbe,
        "needs_model": False,
        "kwargs": {"method": "sentence_transformers"},
        "optional_dep": "sentence_transformers",
    },
    # Model-required baselines
    "perplexity": {
        "class": BaselineProbe,
        "needs_model": True,
        "kwargs": {"method": "perplexity"},
    },
    "random_direction": {
        "class": ActivationBaseline,
        "needs_model": True,
        "kwargs": {"method": "random_direction"},
    },
    "pca": {
        "class": ActivationBaseline,
        "needs_model": True,
        "kwargs": {"method": "pca"},
    },
    "layer_0": {
        "class": ActivationBaseline,
        "needs_model": True,
        "kwargs": {"method": "layer_0"},
    },
}


class BaselineBattery:
    """Run multiple baselines and compare their performance.

    BaselineBattery provides a convenient way to run all available baselines
    and find which one performs best on your task. This helps determine if
    a linear probe is learning something meaningful beyond simpler approaches.

    Parameters
    ----------
    model : str | None, default=None
        HuggingFace model ID. Required for activation-based baselines.
        If None, only text-based baselines are run.
    layers : int | list[int] | str, default=-1
        Layers for activation extraction (activation baselines only).
    pooling : str, default="last_token"
        Token pooling strategy.
    classifier : str | BaseEstimator, default="logistic_regression"
        Classification model for all baselines.
    device : str, default="auto"
        Device for model inference.
    remote : bool, default=False
        Use nnsight remote execution.
    random_state : int | None, default=None
        Random seed for reproducibility.
    include : list[str] | None, default=None
        Which baselines to include. If None, includes all applicable.
        Available: bow, tfidf, random, majority, sentence_length,
        sentence_transformers, perplexity, random_direction, pca, layer_0.
    exclude : list[str] | None, default=None
        Which baselines to exclude.
    scorer : Callable | None, default=None
        Custom scoring function. Default is accuracy.
        Signature: scorer(y_true, y_pred) -> float

    Attributes
    ----------
    results_ : BaselineResults | None
        Results from the last fit() call. None before fitting.

    Examples
    --------
    >>> # Run all text-only baselines
    >>> battery = BaselineBattery(random_state=42)
    >>> results = battery.fit(
    ...     positive_prompts, negative_prompts,
    ...     test_prompts, test_labels,
    ... )
    >>> print(results.summary())
    >>> best = results.get_best(n=3)

    >>> # Run all baselines including activation-based
    >>> battery = BaselineBattery(
    ...     model="meta-llama/Llama-3.1-8B-Instruct",
    ...     layers=-1,
    ...     device="cuda",
    ... )
    >>> results = battery.fit(pos, neg, test_prompts, test_labels)
    """

    def __init__(
        self,
        model: str | None = None,
        layers: int | list[int] | str = -1,
        pooling: str = "last_token",
        classifier: str | BaseEstimator = "logistic_regression",
        device: str = "auto",
        remote: bool = False,
        random_state: int | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ):
        self.model = model
        self.layers = layers
        self.pooling = pooling
        self.classifier = classifier
        self.device = device
        self.remote = remote
        self.random_state = random_state
        self.include = include
        self.exclude = exclude or []
        self.scorer = scorer or self._accuracy_scorer

        self._baselines: dict[str, BaselineProbe | ActivationBaseline] = {}
        self.results_: BaselineResults | None = None

    @staticmethod
    def _accuracy_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Default accuracy scorer."""
        return float(np.mean(y_true == y_pred))

    def _get_applicable_baselines(self) -> list[str]:
        """Determine which baselines to run based on config."""
        applicable = []

        for name, spec in BASELINE_REGISTRY.items():
            # Skip if model required but not provided
            if spec["needs_model"] and self.model is None:
                continue

            # Apply include/exclude filters
            if self.include is not None and name not in self.include:
                continue
            if name in self.exclude:
                continue

            applicable.append(name)

        return applicable

    def _create_baseline(self, name: str) -> BaselineProbe | ActivationBaseline:
        """Create a baseline instance."""
        spec = BASELINE_REGISTRY[name]
        cls = spec["class"]
        kwargs = spec["kwargs"].copy()

        # Add common parameters
        kwargs["classifier"] = self.classifier
        kwargs["random_state"] = self.random_state

        if cls == ActivationBaseline:
            kwargs["model"] = self.model
            kwargs["layers"] = self.layers
            kwargs["pooling"] = self.pooling
            kwargs["device"] = self.device
            kwargs["remote"] = self.remote
        elif spec["needs_model"]:
            # BaselineProbe with model (perplexity)
            kwargs["model"] = self.model
            kwargs["device"] = self.device
            kwargs["remote"] = self.remote

        return cls(**kwargs)

    def fit(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        test_prompts: list[str] | None = None,
        test_labels: list[int] | np.ndarray | None = None,
    ) -> BaselineResults:
        """Fit all baselines and optionally score on test data.

        Parameters
        ----------
        positive_prompts : list[str]
            Positive training examples.
        negative_prompts : list[str]
            Negative training examples.
        test_prompts : list[str] | None, default=None
            Test prompts for scoring. If None, uses training data.
        test_labels : list[int] | np.ndarray | None, default=None
            Test labels. If None, uses training labels.

        Returns
        -------
        BaselineResults
            Results for all baselines that ran successfully.
        """
        # Default to training data for scoring
        if test_prompts is None:
            test_prompts = positive_prompts + negative_prompts
            test_labels = [1] * len(positive_prompts) + [0] * len(negative_prompts)
        test_labels = np.asarray(test_labels)

        results = []
        applicable = self._get_applicable_baselines()

        for name in applicable:
            try:
                baseline = self._create_baseline(name)
            except Exception as e:
                warnings.warn(f"Failed to create {name} baseline: {e}")
                continue

            # Fit with timing
            t0 = time.time()
            try:
                baseline.fit(positive_prompts, negative_prompts)
                fit_time = time.time() - t0
            except Exception as e:
                # Skip baselines that fail (e.g., missing optional deps)
                warnings.warn(f"Failed to fit {name} baseline: {e}")
                continue

            # Score with timing
            t0 = time.time()
            try:
                predictions = baseline.predict(test_prompts)
                predict_time = time.time() - t0
            except Exception as e:
                warnings.warn(f"Failed to predict with {name} baseline: {e}")
                continue

            score = self.scorer(test_labels, predictions)

            self._baselines[name] = baseline
            results.append(
                BaselineResult(
                    name=name,
                    score=score,
                    baseline=baseline,
                    fit_time=fit_time,
                    predict_time=predict_time,
                )
            )

        self.results_ = BaselineResults(results=results)
        return self.results_

    def get_best(self, n: int = 1) -> list[BaselineResult]:
        """Get top n baselines by score.

        Parameters
        ----------
        n : int, default=1
            Number of top baselines to return.

        Returns
        -------
        list[BaselineResult]
            Top n baselines sorted by score.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if self.results_ is None:
            raise RuntimeError("BaselineBattery not fitted. Call fit() first.")
        return self.results_.get_best(n)

    def get_baseline(self, name: str) -> BaselineProbe | ActivationBaseline:
        """Get a specific fitted baseline by name.

        Parameters
        ----------
        name : str
            Name of the baseline.

        Returns
        -------
        BaselineProbe | ActivationBaseline
            The fitted baseline instance.

        Raises
        ------
        KeyError
            If the baseline was not run or not found.
        """
        if name not in self._baselines:
            raise KeyError(
                f"Baseline {name!r} not found. "
                f"Available: {list(self._baselines.keys())}"
            )
        return self._baselines[name]

    @property
    def available_baselines(self) -> list[str]:
        """List of all registered baseline names."""
        return list(BASELINE_REGISTRY.keys())

    @property
    def applicable_baselines(self) -> list[str]:
        """List of baselines that would run with current config."""
        return self._get_applicable_baselines()
