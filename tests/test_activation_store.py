"""Tests for ActivationStore — in-memory activation cache for cross-fold workflows."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

PROMPTS = ["prompt_a", "prompt_b", "prompt_c", "prompt_d", "prompt_e"]
HIDDEN_DIM = 8
LAYERS = [10, 11]


def _make_acts_dict(prompts, layers, hidden_dim=HIDDEN_DIM):
    """Build a fake {layer: ndarray} dict like load_activations returns."""
    rng = np.random.default_rng(42)
    return {
        layer: rng.normal(size=(len(prompts), hidden_dim)).astype(np.float32)
        for layer in layers
    }


def _make_labels(n):
    return np.array([0, 1, 0, 1, 0][:n])


class TestFromDataset:
    """Tests for ActivationStore.from_dataset (HuggingFace source)."""

    @patch("lmprobe.activation_store.load_activations")
    @patch("lmprobe.activation_store.fetch_dataset_metadata")
    def test_basic_load(self, mock_meta, mock_load):
        from lmprobe.activation_store import ActivationStore

        acts = _make_acts_dict(PROMPTS, LAYERS)
        labels = _make_labels(len(PROMPTS))
        mock_load.return_value = (acts, labels)
        mock_meta.return_value = type("M", (), {
            "prompts": PROMPTS, "available_layers": LAYERS,
        })()

        store = ActivationStore.from_dataset("user/repo", layers=LAYERS)
        assert store.is_loaded
        assert store.prompts == PROMPTS
        assert store.layers == LAYERS
        assert store.labels is not None
        assert len(store.labels) == len(PROMPTS)

    @patch("lmprobe.activation_store.load_activations")
    @patch("lmprobe.activation_store.fetch_dataset_metadata")
    def test_no_labels(self, mock_meta, mock_load):
        from lmprobe.activation_store import ActivationStore

        acts = _make_acts_dict(PROMPTS, LAYERS)
        mock_load.return_value = (acts, None)
        mock_meta.return_value = type("M", (), {
            "prompts": PROMPTS, "available_layers": LAYERS,
        })()

        store = ActivationStore.from_dataset("user/repo", layers=LAYERS)
        assert store.labels is None


class TestFromCache:
    """Tests for ActivationStore.from_cache (local prompt cache source)."""

    @patch("lmprobe.activation_store.load_pooled_batch")
    def test_basic_load(self, mock_load):
        from lmprobe.activation_store import ActivationStore

        n = len(PROMPTS)
        total_dim = len(LAYERS) * HIDDEN_DIM
        mock_load.return_value = torch.randn(n, total_dim)

        labels = _make_labels(n)
        store = ActivationStore.from_cache(
            "model/name", PROMPTS, layers=LAYERS, labels=labels,
        )
        assert store.is_loaded
        assert store.prompts == PROMPTS
        assert store.layers == LAYERS
        assert np.array_equal(store.labels, labels)

    @patch("lmprobe.activation_store.load_pooled_batch")
    def test_no_labels(self, mock_load):
        from lmprobe.activation_store import ActivationStore

        mock_load.return_value = torch.randn(len(PROMPTS), len(LAYERS) * HIDDEN_DIM)
        store = ActivationStore.from_cache("model/name", PROMPTS, layers=LAYERS)
        assert store.labels is None


class TestGetActivations:
    """Tests for get_activations indexing."""

    @pytest.fixture
    def store(self):
        from lmprobe.activation_store import ActivationStore

        acts = _make_acts_dict(PROMPTS, LAYERS)
        labels = _make_labels(len(PROMPTS))
        return ActivationStore._from_data(PROMPTS, acts, labels)

    def test_all_prompts_all_layers(self, store):
        X = store.get_activations()
        assert X.shape == (len(PROMPTS), len(LAYERS) * HIDDEN_DIM)

    def test_subset_prompts(self, store):
        subset = ["prompt_a", "prompt_c"]
        X = store.get_activations(prompts=subset)
        assert X.shape == (2, len(LAYERS) * HIDDEN_DIM)

    def test_single_layer(self, store):
        X = store.get_activations(layer=10)
        assert X.shape == (len(PROMPTS), HIDDEN_DIM)

    def test_multi_layer_subset(self, store):
        X = store.get_activations(layer=[10, 11])
        assert X.shape == (len(PROMPTS), 2 * HIDDEN_DIM)

    def test_single_layer_list(self, store):
        X = store.get_activations(layer=[10])
        assert X.shape == (len(PROMPTS), HIDDEN_DIM)

    def test_unknown_prompt_raises(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.get_activations(prompts=["nonexistent"])

    def test_unknown_layer_raises(self, store):
        with pytest.raises(ValueError, match="not loaded"):
            store.get_activations(layer=99)


class TestGetLabels:
    """Tests for get_labels indexing."""

    @pytest.fixture
    def store(self):
        from lmprobe.activation_store import ActivationStore

        acts = _make_acts_dict(PROMPTS, LAYERS)
        labels = _make_labels(len(PROMPTS))
        return ActivationStore._from_data(PROMPTS, acts, labels)

    def test_all_labels(self, store):
        y = store.get_labels()
        assert y is not None
        assert len(y) == len(PROMPTS)

    def test_subset_labels(self, store):
        y = store.get_labels(prompts=["prompt_a", "prompt_d"])
        assert y is not None
        assert len(y) == 2
        assert y[0] == 0  # prompt_a label
        assert y[1] == 1  # prompt_d label


class TestGetFold:
    """Tests for get_fold convenience method."""

    @pytest.fixture
    def store(self):
        from lmprobe.activation_store import ActivationStore

        acts = _make_acts_dict(PROMPTS, LAYERS)
        labels = _make_labels(len(PROMPTS))
        return ActivationStore._from_data(PROMPTS, acts, labels)

    def test_fold_shapes(self, store):
        train = ["prompt_a", "prompt_b", "prompt_c"]
        test = ["prompt_d", "prompt_e"]
        X_train, y_train, X_test, y_test = store.get_fold(train, test)
        assert X_train.shape == (3, len(LAYERS) * HIDDEN_DIM)
        assert X_test.shape == (2, len(LAYERS) * HIDDEN_DIM)
        assert len(y_train) == 3
        assert len(y_test) == 2

    def test_fold_with_layer(self, store):
        train = ["prompt_a", "prompt_b"]
        test = ["prompt_c"]
        X_train, y_train, X_test, y_test = store.get_fold(train, test, layer=10)
        assert X_train.shape == (2, HIDDEN_DIM)
        assert X_test.shape == (1, HIDDEN_DIM)


class TestMemoryManagement:
    """Tests for clear() and context manager."""

    def test_clear(self):
        from lmprobe.activation_store import ActivationStore

        acts = _make_acts_dict(PROMPTS, LAYERS)
        store = ActivationStore._from_data(PROMPTS, acts, None)
        assert store.is_loaded
        assert store.memory_bytes > 0

        store.clear()
        assert not store.is_loaded
        assert store.memory_bytes == 0

    def test_context_manager(self):
        from lmprobe.activation_store import ActivationStore

        acts = _make_acts_dict(PROMPTS, LAYERS)
        store = ActivationStore._from_data(PROMPTS, acts, None)
        with store:
            assert store.is_loaded
        assert not store.is_loaded

    def test_memory_bytes_positive(self):
        from lmprobe.activation_store import ActivationStore

        acts = _make_acts_dict(PROMPTS, LAYERS)
        store = ActivationStore._from_data(PROMPTS, acts, None)
        expected = sum(a.nbytes for a in acts.values())
        assert store.memory_bytes == expected

    def test_access_after_clear_raises(self):
        from lmprobe.activation_store import ActivationStore

        acts = _make_acts_dict(PROMPTS, LAYERS)
        store = ActivationStore._from_data(PROMPTS, acts, None)
        store.clear()
        with pytest.raises(RuntimeError, match="not loaded"):
            store.get_activations()
