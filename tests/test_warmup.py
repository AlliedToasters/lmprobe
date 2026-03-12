"""Tests for the LinearProbe.warmup() cache warmup API."""

import pytest

from lmprobe import LinearProbe


def test_warmup_basic(tiny_model):
    """warmup() runs without error and populates the cache."""
    probe = LinearProbe(
        model=tiny_model,
        layers=-1,
        pooling="last_token",
        device="cpu",
        remote=False,
        random_state=42,
    )

    # Should not raise
    probe.warmup(["Hello world", "Another prompt"])


def test_warmup_then_fit(tiny_model):
    """Warming up before fit() should work (fit uses cached activations)."""
    all_prompts = [
        "Who wants to go for a walk?",
        "Fetch the ball!",
        "Purring, stalking, pouncing.",
        "Uses a litterbox.",
    ]

    probe = LinearProbe(
        model=tiny_model,
        layers=-1,
        pooling="last_token",
        device="cpu",
        remote=False,
        random_state=42,
    )

    # Warmup all prompts first
    probe.warmup(all_prompts)

    # Then fit (should use cached activations)
    probe.fit(all_prompts[:2], all_prompts[2:])

    predictions = probe.predict(["test input"])
    assert predictions.shape == (1,)


def test_warmup_requires_model():
    """warmup() should raise ValueError when no model is set."""
    probe = LinearProbe(
        model=None,
        layers=-1,
        random_state=42,
    )

    with pytest.raises(ValueError, match="No model specified"):
        probe.warmup(["test prompt"])


def test_warmup_returns_none(tiny_model):
    """warmup() should return None."""
    probe = LinearProbe(
        model=tiny_model,
        layers=-1,
        pooling="last_token",
        device="cpu",
        remote=False,
        random_state=42,
    )

    result = probe.warmup(["Hello world"])
    assert result is None
