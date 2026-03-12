"""Tests for the local extraction backend.

These tests verify that backend="local" works end-to-end with LinearProbe,
producing the same shapes and behaviors as the nnsight backend.
"""

import numpy as np
import pytest
import torch

from lmprobe import LinearProbe
from lmprobe.backends import LocalBackend, resolve_backend


POSITIVE_PROMPTS = [
    "Who wants to go for a walk?",
    "My tail is wagging with delight.",
    "Fetch the ball!",
    "Good boy!",
    "Slobbering, chewing, growling, barking.",
]

NEGATIVE_PROMPTS = [
    "Enjoys lounging in the sun beam all day.",
    "Purring, stalking, pouncing, scratching.",
    "Uses a litterbox, throws sand all over the room.",
    "Tail raised, back arched, eyes alert, whiskers forward.",
]

TEST_PROMPTS = [
    "Arf! Arf! Let's go outside!",
    "Knocking things off the counter for sport.",
]


class TestLocalBackend:
    """Tests for LocalBackend directly."""

    def test_resolve_backend_local(self, tiny_model):
        backend = resolve_backend("local", tiny_model, "cpu")
        assert isinstance(backend, LocalBackend)

    def test_resolve_backend_invalid_local_remote(self, tiny_model):
        with pytest.raises(ValueError, match="does not support remote"):
            resolve_backend("local", tiny_model, "cpu", remote=True)

    def test_resolve_backend_unknown(self, tiny_model):
        with pytest.raises(ValueError, match="Unknown backend"):
            resolve_backend("invalid", tiny_model, "cpu")

    def test_extract_batch(self, tiny_model):
        backend = resolve_backend("local", tiny_model, "cpu")
        acts, mask = backend.extract_batch(
            ["Hello world", "Test prompt"], [0]
        )
        assert acts.ndim == 3  # (batch, seq_len, hidden_dim)
        assert mask.ndim == 2  # (batch, seq_len)
        assert acts.shape[0] == 2
        assert mask.shape[0] == 2

    def test_extract_batch_multilayer(self, tiny_model):
        backend = resolve_backend("local", tiny_model, "cpu")
        acts_single, _ = backend.extract_batch(["Test"], [0])
        acts_multi, _ = backend.extract_batch(["Test"], [0, 1])
        # Multi-layer should have 2x the hidden dim
        assert acts_multi.shape[-1] == 2 * acts_single.shape[-1]

    def test_extract_batch_with_logits(self, tiny_model):
        backend = resolve_backend("local", tiny_model, "cpu")
        acts, mask, logits = backend.extract_batch_with_logits(
            ["Hello world"], [0]
        )
        assert acts.ndim == 3
        assert logits.ndim == 3  # (batch, seq_len, vocab_size)
        assert logits.shape[0] == 1

    def test_tokenizer_property(self, tiny_model):
        backend = resolve_backend("local", tiny_model, "cpu")
        tokenizer = backend.tokenizer
        assert tokenizer is not None
        tokens = tokenizer("Hello", return_tensors="pt")
        assert "input_ids" in tokens


class TestLocalFitPredict:
    """Full pipeline tests with backend='local'."""

    def test_fit_predict(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            pooling="last_token",
            classifier="logistic_regression",
            device="cpu",
            remote=False,
            random_state=42,
            backend="local",
        )

        probe.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        predictions = probe.predict(TEST_PROMPTS)
        assert predictions.shape == (2,)

        probabilities = probe.predict_proba(TEST_PROMPTS)
        assert probabilities.shape == (2, 2)

        accuracy = probe.score(TEST_PROMPTS, [1, 0])
        assert 0.0 <= accuracy <= 1.0

    def test_multilayer(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model,
            layers=[0, 1],
            pooling="last_token",
            classifier="logistic_regression",
            device="cpu",
            remote=False,
            random_state=42,
            backend="local",
        )

        probe.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        predictions = probe.predict(TEST_PROMPTS)
        assert predictions.shape == (2,)

    def test_caching(self, tiny_model, tmp_path, monkeypatch):
        """Verify disk cache works with local backend."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            pooling="last_token",
            device="cpu",
            random_state=42,
            backend="local",
        )

        # First fit populates cache
        probe.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        # Second fit should use cached activations
        probe2 = LinearProbe(
            model=tiny_model,
            layers=-1,
            pooling="last_token",
            device="cpu",
            random_state=42,
            backend="local",
        )
        probe2.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        # Both should produce same predictions
        p1 = probe.predict(TEST_PROMPTS)
        p2 = probe2.predict(TEST_PROMPTS)
        np.testing.assert_array_equal(p1, p2)

    def test_save_load(self, tiny_model, tmp_path):
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            pooling="last_token",
            device="cpu",
            random_state=42,
            backend="local",
        )
        probe.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        save_path = str(tmp_path / "probe.pkl")
        probe.save(save_path)

        loaded = LinearProbe.load(save_path)
        assert loaded.backend == "local"

        p1 = probe.predict(TEST_PROMPTS)
        p2 = loaded.predict(TEST_PROMPTS)
        np.testing.assert_array_equal(p1, p2)

    def test_invalid_local_remote(self, tiny_model):
        with pytest.raises(ValueError, match="does not support remote"):
            LinearProbe(
                model=tiny_model,
                layers=-1,
                device="cpu",
                remote=True,
                backend="local",
            )
