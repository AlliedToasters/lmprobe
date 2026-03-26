"""In-memory activation store for cross-fold workflows.

Loads activations once from a HuggingFace dataset or local prompt cache
into RAM, then provides fast indexed access for cross-validation without
repeated disk reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def _lazy_load_activations():
    from .sharing import fetch_dataset_metadata, load_activations

    return load_activations, fetch_dataset_metadata


def _lazy_load_pooled_batch():
    from .cache import load_pooled_batch

    return load_pooled_batch


# Re-export at module level for mockability in tests
def load_activations(*args, **kwargs):
    fn, _ = _lazy_load_activations()
    return fn(*args, **kwargs)


def fetch_dataset_metadata(*args, **kwargs):
    _, fn = _lazy_load_activations()
    return fn(*args, **kwargs)


def load_pooled_batch(*args, **kwargs):
    fn = _lazy_load_pooled_batch()
    return fn(*args, **kwargs)


class ActivationStore:
    """In-memory store for activation datasets, optimized for cross-fold workflows.

    Loads activations once into RAM, then provides fast numpy-indexed access
    for cross-validation (e.g., LODO). Supports two data sources:

    - ``from_dataset()``: loads from a HuggingFace activation dataset
    - ``from_cache()``: loads from the local lmprobe prompt cache

    Examples
    --------
    >>> store = ActivationStore.from_dataset("user/dataset", layers=[45, 46])
    >>> X_train, y_train, X_test, y_test = store.get_fold(train_prompts, test_prompts)
    >>> probe = Probe(classifier="sgd_gpu", random_state=42)
    >>> probe.fit_from_activations(X_train, y_train)
    """

    def __init__(self) -> None:
        self._data: dict[int, np.ndarray] = {}
        self._prompts: list[str] = []
        self._labels: np.ndarray | None = None
        self._prompt_to_idx: dict[str, int] = {}
        self._loaded = False

    @classmethod
    def _from_data(
        cls,
        prompts: list[str],
        data: dict[int, np.ndarray],
        labels: np.ndarray | None,
    ) -> ActivationStore:
        """Internal constructor from pre-loaded data."""
        store = cls()
        store._data = data
        store._prompts = list(prompts)
        store._labels = labels
        store._prompt_to_idx = {p: i for i, p in enumerate(prompts)}
        store._loaded = True
        return store

    @classmethod
    def from_dataset(
        cls,
        dataset: str,
        *,
        layers: list[int] | None = None,
        pooling: str = "last_token",
        token: str | None = None,
    ) -> ActivationStore:
        """Load activations from a HuggingFace activation dataset.

        Parameters
        ----------
        dataset : str
            HuggingFace dataset repo ID.
        layers : list[int] | None
            Layer indices to load. None loads all available layers.
        pooling : str
            Pooling strategy (default ``"last_token"``).
        token : str | None
            HuggingFace API token.

        Returns
        -------
        ActivationStore
            A loaded store ready for indexing.
        """
        meta = fetch_dataset_metadata(dataset, token=token)
        if layers is None:
            layers = meta.available_layers
        prompts = meta.prompts

        result = load_activations(
            dataset,
            layers=layers,
            pooling=pooling,
            token=token,
            as_dict=True,
            return_labels=True,
            show_progress=True,
        )
        acts, labels = result
        return cls._from_data(prompts, acts, labels)

    @classmethod
    def from_cache(
        cls,
        model_name: str,
        prompts: list[str],
        *,
        layers: list[int],
        pooling: str = "last_token",
        labels: np.ndarray | list | None = None,
    ) -> ActivationStore:
        """Load activations from the local lmprobe prompt cache.

        Parameters
        ----------
        model_name : str
            Model name used for cache lookups.
        prompts : list[str]
            Prompts to load from cache.
        layers : list[int]
            Layer indices to load.
        pooling : str
            Pooling strategy (default ``"last_token"``).
        labels : np.ndarray | list | None
            Labels for the prompts. If None, ``get_labels()`` will return None.

        Returns
        -------
        ActivationStore
            A loaded store ready for indexing.
        """
        sorted_layers = sorted(layers)
        pooled = load_pooled_batch(
            model_name, prompts, sorted_layers, pooling, fallback_to_raw=True,
        )
        # pooled is torch.Tensor shape (n_prompts, n_layers * hidden_dim)
        stacked = pooled.detach().cpu().float().numpy()
        hidden_dim = stacked.shape[-1] // len(sorted_layers)

        data = {
            layer: stacked[:, i * hidden_dim : (i + 1) * hidden_dim]
            for i, layer in enumerate(sorted_layers)
        }

        labels_arr = np.asarray(labels) if labels is not None else None
        return cls._from_data(prompts, data, labels_arr)

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "ActivationStore is not loaded. "
                "Use from_dataset() or from_cache() to create a loaded store."
            )

    def _resolve_indices(self, prompts: list[str]) -> np.ndarray:
        missing = [p for p in prompts if p not in self._prompt_to_idx]
        if missing:
            raise ValueError(
                f"{len(missing)} prompt(s) not found in store: {missing[:3]!r}"
                + (f" ... and {len(missing) - 3} more" if len(missing) > 3 else "")
            )
        return np.array([self._prompt_to_idx[p] for p in prompts])

    def _resolve_layers(
        self, layer: int | list[int] | None,
    ) -> list[int]:
        if layer is None:
            return sorted(self._data.keys())
        if isinstance(layer, int):
            layer = [layer]
        for lay in layer:
            if lay not in self._data:
                raise ValueError(
                    f"Layer {lay} not loaded. Available: {sorted(self._data.keys())}"
                )
        return layer

    @property
    def is_loaded(self) -> bool:
        """Whether data has been loaded into memory."""
        return self._loaded

    @property
    def prompts(self) -> list[str]:
        """All prompts in the store, in order."""
        return list(self._prompts)

    @property
    def labels(self) -> np.ndarray | None:
        """Labels array, or None if no labels were provided."""
        return self._labels

    @property
    def layers(self) -> list[int]:
        """Available layer indices."""
        return sorted(self._data.keys())

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        if not self._loaded:
            return 0
        return sum(a.nbytes for a in self._data.values())

    def get_activations(
        self,
        prompts: list[str] | None = None,
        layer: int | list[int] | None = None,
    ) -> np.ndarray:
        """Get activation matrix for a subset of prompts and layers.

        Parameters
        ----------
        prompts : list[str] | None
            Subset of prompts to retrieve. None returns all.
        layer : int | list[int] | None
            Layer(s) to return. None returns all loaded layers (concatenated).

        Returns
        -------
        np.ndarray
            Shape ``(n_prompts, hidden_dim)`` for single layer,
            or ``(n_prompts, n_layers * hidden_dim)`` for multiple layers.
        """
        self._check_loaded()
        resolved_layers = self._resolve_layers(layer)

        if prompts is not None:
            idx = self._resolve_indices(prompts)
            parts = [self._data[lay][idx] for lay in resolved_layers]
        else:
            parts = [self._data[lay] for lay in resolved_layers]

        return np.concatenate(parts, axis=1)

    def get_labels(
        self,
        prompts: list[str] | None = None,
    ) -> np.ndarray | None:
        """Get labels for a subset of prompts.

        Parameters
        ----------
        prompts : list[str] | None
            Subset of prompts. None returns all labels.

        Returns
        -------
        np.ndarray | None
            Labels, or None if no labels were provided.
        """
        if self._labels is None:
            return None
        if prompts is not None:
            idx = self._resolve_indices(prompts)
            return self._labels[idx]
        return self._labels

    def get_fold(
        self,
        train_prompts: list[str],
        test_prompts: list[str],
        layer: int | list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/test split for a cross-validation fold.

        Parameters
        ----------
        train_prompts : list[str]
            Training prompts.
        test_prompts : list[str]
            Test prompts.
        layer : int | list[int] | None
            Layer(s) to use.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(X_train, y_train, X_test, y_test)``
        """
        X_train = self.get_activations(prompts=train_prompts, layer=layer)
        X_test = self.get_activations(prompts=test_prompts, layer=layer)
        y_train = self.get_labels(prompts=train_prompts)
        y_test = self.get_labels(prompts=test_prompts)
        if y_train is None or y_test is None:
            raise RuntimeError(
                "get_fold() requires labels. Pass labels when creating the store."
            )
        return X_train, y_train, X_test, y_test

    def clear(self) -> None:
        """Release all in-memory data."""
        self._data = {}
        self._prompts = []
        self._labels = None
        self._prompt_to_idx = {}
        self._loaded = False

    def __enter__(self) -> ActivationStore:
        return self

    def __exit__(self, *exc) -> None:
        self.clear()

    def __repr__(self) -> str:
        if not self._loaded:
            return "ActivationStore(not loaded)"
        n = len(self._prompts)
        layers = self.layers
        mb = self.memory_bytes / (1024 * 1024)
        return f"ActivationStore({n} prompts, layers={layers}, {mb:.1f} MB)"
