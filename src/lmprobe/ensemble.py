"""ProbeEnsemble: Multi-probe ensemble support.

Composition over inheritance — wraps multiple Probe instances
and provides standard fit/predict/predict_proba/score API plus
save/load and bootstrap stability analysis.
"""

from __future__ import annotations

import copy
import pickle
from typing import TYPE_CHECKING

import numpy as np

from .cache import CachedExtractor
from .extraction import ActivationExtractor
from .probe import Probe, _resolve_dtype

if TYPE_CHECKING:
    pass


class ProbeEnsemble:
    """Ensemble of multiple Probe instances with voting-based aggregation.

    Wraps multiple :class:`Probe` instances and provides a unified
    fit/predict/predict_proba/score interface. Supports soft voting
    (averaged probabilities) and hard voting (majority predictions).

    Parameters
    ----------
    probes : list[Probe]
        Pre-configured Probe instances.
    weights : list[float] | None
        Per-probe weights, normalized to sum=1. If None, uniform weights.
    voting : str
        Aggregation strategy: ``"soft"`` (average probabilities) or
        ``"hard"`` (majority vote). Default ``"soft"``.

        .. note::
           Soft voting requires all member probes to support
           ``predict_proba()`` (e.g. logistic regression, SVM with
           probability=True). Classifiers like Ridge that lack
           ``predict_proba`` must use ``voting="hard"``.

    Examples
    --------
    >>> from lmprobe import Probe
    >>> from lmprobe.ensemble import ProbeEnsemble
    >>> p1 = Probe(model="stas/tiny-random-llama-2", layers=-1,
    ...            classifier="logistic_regression", device="cpu")
    >>> p2 = Probe(model="stas/tiny-random-llama-2", layers=-1,
    ...            classifier="random_forest", device="cpu")
    >>> ensemble = ProbeEnsemble([p1, p2])
    >>> ensemble.fit(pos_prompts, neg_prompts)
    >>> preds = ensemble.predict(test_prompts)
    """

    def __init__(
        self,
        probes: list[Probe],
        weights: list[float] | None = None,
        voting: str = "soft",
    ):
        if not probes:
            raise ValueError("probes must be a non-empty list.")
        if voting not in ("soft", "hard"):
            raise ValueError(
                f"Unknown voting strategy: {voting!r}. "
                f"Expected 'soft' or 'hard'."
            )
        if weights is not None:
            if len(weights) != len(probes):
                raise ValueError(
                    f"weights length ({len(weights)}) must match "
                    f"probes length ({len(probes)})."
                )
            total = sum(weights)
            if total <= 0:
                raise ValueError("weights must sum to a positive value.")
            self.weights_ = np.array(weights, dtype=float) / total
        else:
            self.weights_ = np.ones(len(probes), dtype=float) / len(probes)

        self.probes_ = list(probes)
        self.voting = voting
        self._bootstrap_mode = False
        self._bootstrap_seed: int | None = None
        self._fitted = False

    @classmethod
    def from_configs(
        cls,
        model: str,
        configs: list[dict],
        weights: list[float] | None = None,
        voting: str = "soft",
        **shared_kwargs,
    ) -> ProbeEnsemble:
        """Create an ensemble from per-probe config dicts.

        Each dict in ``configs`` is merged with ``shared_kwargs``
        to construct a :class:`Probe`.

        Parameters
        ----------
        model : str
            HuggingFace model ID (shared across all probes).
        configs : list[dict]
            Per-probe overrides (e.g. layers, classifier, pca_components).
        weights : list[float] | None
            Per-probe weights.
        voting : str
            Aggregation strategy.
        **shared_kwargs
            Shared keyword arguments passed to every Probe constructor
            (e.g. device, remote, pooling, random_state).

        Returns
        -------
        ProbeEnsemble
        """
        probes = []
        for cfg in configs:
            merged = {**shared_kwargs, **cfg, "model": model}
            probes.append(Probe(**merged))
        return cls(probes, weights=weights, voting=voting)

    @classmethod
    def bootstrap(
        cls,
        base_probe: Probe,
        n_resamples: int = 10,
        random_state: int | None = None,
        weights: list[float] | None = None,
        voting: str = "soft",
    ) -> ProbeEnsemble:
        """Create a bootstrap ensemble by cloning a probe.

        During ``fit()``, each member trains on a different bootstrap
        resample of the training data (activations are extracted once
        and served from cache).

        Parameters
        ----------
        base_probe : Probe
            Template probe to clone.
        n_resamples : int
            Number of bootstrap resamples (ensemble members).
        random_state : int | None
            Random seed for reproducible bootstrap sampling.
        weights : list[float] | None
            Per-probe weights.
        voting : str
            Aggregation strategy.

        Returns
        -------
        ProbeEnsemble
        """
        probes = [copy.deepcopy(base_probe) for _ in range(n_resamples)]
        ensemble = cls(probes, weights=weights, voting=voting)
        ensemble._bootstrap_mode = True
        ensemble._bootstrap_seed = random_state
        return ensemble

    def _warmup_cache(self, prompts: list[str]) -> None:
        """Extract activations for all layers needed across probes in one pass.

        Populates the filesystem cache so that individual probe
        fit/predict calls hit cache instead of re-running the model.
        """
        if not self.probes_:
            return

        # Collect union of all layers across probes
        any_probe = self.probes_[0]
        if any_probe._cached_extractor is None:
            return

        all_layer_sets = []
        for probe in self.probes_:
            if probe._extractor is not None:
                all_layer_sets.extend(probe._extractor.layer_indices)

        if not all_layer_sets:
            return

        all_layers = sorted(set(all_layer_sets))

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

    def fit(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str] | np.ndarray | list[int] | None = None,
        remote: bool | None = None,
    ) -> ProbeEnsemble:
        """Fit all probes in the ensemble.

        Performs a single warmup extraction pass covering all layers,
        then fits each member probe. In bootstrap mode, each member
        trains on a different bootstrap resample.

        Parameters
        ----------
        positive_prompts : list[str]
            In contrastive mode: positive-class prompts.
            In standard mode: all prompts.
        negative_prompts : list[str] | np.ndarray | list[int] | None
            In contrastive mode: negative-class prompts.
            In standard mode: labels.
        remote : bool | None
            Override remote setting for all probes.

        Returns
        -------
        ProbeEnsemble
            Self, for method chaining.
        """
        # Determine all prompts for warmup
        if isinstance(negative_prompts, (np.ndarray, list)) and (
            len(negative_prompts) > 0
            and isinstance(negative_prompts[0], (int, np.integer))
        ):
            all_prompts = list(positive_prompts)
        else:
            all_prompts = list(positive_prompts) + list(negative_prompts)

        # Single warmup extraction pass
        self._warmup_cache(all_prompts)

        if self._bootstrap_mode:
            self._fit_bootstrap(positive_prompts, negative_prompts, remote)
        else:
            for probe in self.probes_:
                kwargs = {}
                if remote is not None:
                    kwargs["remote"] = remote
                probe.fit(positive_prompts, negative_prompts, **kwargs)

        self._fitted = True
        return self

    def _fit_bootstrap(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str] | np.ndarray | list[int] | None,
        remote: bool | None,
    ) -> None:
        """Fit each probe on a bootstrap resample of the training data."""
        # Build combined prompts and labels
        if isinstance(negative_prompts, (np.ndarray, list)) and (
            len(negative_prompts) > 0
            and isinstance(negative_prompts[0], (int, np.integer))
        ):
            prompts = list(positive_prompts)
            labels = np.asarray(negative_prompts)
        else:
            prompts = list(positive_prompts) + list(negative_prompts)
            labels = np.array(
                [1] * len(positive_prompts) + [0] * len(negative_prompts)
            )

        rng = np.random.default_rng(self._bootstrap_seed)

        unique_classes = np.unique(labels)
        for probe in self.probes_:
            # Resample ensuring all classes are represented
            for _attempt in range(100):
                indices = rng.choice(
                    len(prompts), size=len(prompts), replace=True
                )
                if len(np.unique(labels[indices])) == len(unique_classes):
                    break
            resampled_prompts = [prompts[i] for i in indices]
            resampled_labels = labels[indices]

            kwargs = {}
            if remote is not None:
                kwargs["remote"] = remote
            probe.fit(resampled_prompts, resampled_labels, **kwargs)

    def _check_fitted(self) -> None:
        """Check that the ensemble has been fitted."""
        if not self._fitted:
            raise RuntimeError(
                "ProbeEnsemble has not been fitted. Call fit() first."
            )

    def predict_proba(
        self,
        prompts: list[str],
        remote: bool | None = None,
    ) -> np.ndarray:
        """Predict weighted-average class probabilities.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to classify.
        remote : bool | None
            Override remote setting for all probes.

        Returns
        -------
        np.ndarray
            Averaged class probabilities, shape (n_samples, n_classes).
        """
        self._check_fitted()

        # Verify all probes support predict_proba
        for i, probe in enumerate(self.probes_):
            if not hasattr(probe.classifier_, "predict_proba"):
                raise TypeError(
                    f"Probe {i} (classifier={probe.classifier!r}) does not "
                    f"support predict_proba(). Use voting='hard' for "
                    f"ensembles containing classifiers without probability "
                    f"estimates, or switch to a different classifier."
                )

        self._warmup_cache(prompts)

        kwargs = {}
        if remote is not None:
            kwargs["remote"] = remote

        probas = []
        for i, probe in enumerate(self.probes_):
            p = probe.predict_proba(prompts, **kwargs)
            probas.append(self.weights_[i] * p)

        return np.sum(probas, axis=0)

    def predict(
        self,
        prompts: list[str],
        remote: bool | None = None,
    ) -> np.ndarray:
        """Predict class labels.

        Soft voting: argmax of weighted-average probabilities.
        Hard voting: mode of per-probe predictions.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to classify.
        remote : bool | None
            Override remote setting for all probes.

        Returns
        -------
        np.ndarray
            Predicted labels, shape (n_samples,).
        """
        self._check_fitted()

        if self.voting == "soft":
            proba = self.predict_proba(prompts, remote=remote)
            # Get classes from first probe
            classes = self.probes_[0].classes_
            return classes[proba.argmax(axis=1)]
        else:
            # Hard voting: majority vote
            self._warmup_cache(prompts)
            kwargs = {}
            if remote is not None:
                kwargs["remote"] = remote

            all_preds = np.array([
                probe.predict(prompts, **kwargs)
                for probe in self.probes_
            ])
            # all_preds: (n_probes, n_samples)
            # For each sample, take the mode
            from scipy.stats import mode

            result = mode(all_preds, axis=0, keepdims=False)
            return result.mode

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
            Override remote setting.

        Returns
        -------
        float
            Accuracy.
        """
        predictions = self.predict(prompts, remote=remote)
        labels = np.asarray(labels)
        return float(np.mean(predictions == labels))

    def prediction_std(
        self,
        prompts: list[str],
        remote: bool | None = None,
    ) -> np.ndarray:
        """Per-sample standard deviation of positive-class probability.

        Useful for bootstrap stability analysis — high std indicates
        the ensemble members disagree.

        Parameters
        ----------
        prompts : list[str]
            Text prompts.
        remote : bool | None
            Override remote setting.

        Returns
        -------
        np.ndarray
            Standard deviation of positive-class probability per sample,
            shape (n_samples,).
        """
        self._check_fitted()
        self._warmup_cache(prompts)

        kwargs = {}
        if remote is not None:
            kwargs["remote"] = remote

        per_probe_pos_proba = []
        for probe in self.probes_:
            p = probe.predict_proba(prompts, **kwargs)
            # Positive class is typically column 1
            per_probe_pos_proba.append(p[:, 1])

        # (n_probes, n_samples)
        stacked = np.array(per_probe_pos_proba)
        return np.std(stacked, axis=0)

    def save(self, path: str) -> None:
        """Save the fitted ensemble to disk.

        Parameters
        ----------
        path : str
            Path to save the ensemble.
        """
        self._check_fitted()

        state = {
            "voting": self.voting,
            "weights_": self.weights_.tolist(),
            "probes_": self.probes_,
            "_bootstrap_mode": self._bootstrap_mode,
            "_bootstrap_seed": self._bootstrap_seed,
            "_fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> ProbeEnsemble:
        """Load a fitted ensemble from disk.

        Parameters
        ----------
        path : str
            Path to the saved ensemble.

        Returns
        -------
        ProbeEnsemble
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        ensemble = cls.__new__(cls)
        ensemble.probes_ = state["probes_"]
        ensemble.voting = state["voting"]
        ensemble.weights_ = np.array(state["weights_"])
        ensemble._bootstrap_mode = state["_bootstrap_mode"]
        ensemble._bootstrap_seed = state["_bootstrap_seed"]
        ensemble._fitted = state["_fitted"]
        return ensemble
