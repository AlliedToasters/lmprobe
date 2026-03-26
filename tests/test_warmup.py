"""Tests for the LinearProbe.warmup() cache warmup API."""

import pytest

from lmprobe import LinearProbe

pytestmark = pytest.mark.nnsight


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


def test_warmup_batch_size_override(tiny_model):
    """warmup() accepts batch_size override and restores original after."""
    probe = LinearProbe(
        model=tiny_model,
        layers=-1,
        pooling="last_token",
        device="cpu",
        remote=False,
        random_state=42,
        batch_size=8,
    )

    # Should not raise, and should use batch_size=1
    probe.warmup(["Hello world", "Another prompt"], batch_size=1)

    # Original batch_size should be restored
    assert probe._extractor.batch_size == 8


def test_fit_batch_size_override(tiny_model):
    """fit() accepts batch_size override and restores original after."""
    probe = LinearProbe(
        model=tiny_model,
        layers=-1,
        pooling="last_token",
        device="cpu",
        remote=False,
        random_state=42,
        batch_size=8,
    )

    probe.fit(
        ["Walk the dog!", "Fetch!"],
        ["Purring cat.", "Meow."],
        batch_size=2,
    )

    # Original batch_size should be restored
    assert probe._extractor.batch_size == 8

    # Predict with batch_size override
    predictions = probe.predict(["test input"], batch_size=1)
    assert predictions.shape == (1,)
    assert probe._extractor.batch_size == 8
