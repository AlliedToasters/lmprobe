"""
North Star Test: The README example must run exactly as documented.

This test runs the exact code from README.md's "Example Usage" section.
If this test passes, the library's public API is working as advertised.

This is the PRIMARY test for the library. All other tests support this one.
"""

import numpy as np
import pytest


def test_readme_example_runs(tiny_model):
    """The README example code runs without error.

    This test mirrors the exact usage pattern from README.md,
    substituting the tiny test model for the full Llama model.
    """
    from lmprobe import LinearProbe

    positive_prompts = [
        "Who wants to go for a walk?",
        "My tail is wagging with delight.",
        "Fetch the ball!",
        "Good boy!",
        "Slobbering, chewing, growling, barking.",
    ]

    negative_prompts = [
        "Enjoys lounging in the sun beam all day.",
        "Purring, stalking, pouncing, scratching.",
        "Uses a litterbox, throws sand all over the room.",
        "Tail raised, back arched, eyes alert, whiskers forward.",
    ]

    # Configure the probe (mirrors README, with test-appropriate values)
    probe = LinearProbe(
        model=tiny_model,
        layers=-1,  # Last layer (tiny model has few layers)
        pooling="last_token",
        classifier="logistic_regression",
        device="cpu",
        remote=False,
        random_state=42,
    )

    # Fit using contrastive prompts
    probe.fit(positive_prompts, negative_prompts)

    # Predict on new examples
    test_prompts = [
        "Arf! Arf! Let's go outside!",
        "Knocking things off the counter for sport.",
    ]
    predictions = probe.predict(test_prompts)
    probabilities = probe.predict_proba(test_prompts)

    # Verify output shapes (values are meaningless with random weights)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert set(predictions).issubset({0, 1})  # Binary predictions

    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (2, 2)  # (n_samples, n_classes)
    assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1

    # Evaluate
    accuracy = probe.score(test_prompts, [1, 0])
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_readme_save_load(tiny_model, tmp_path):
    """The save/load functionality from README works."""
    from lmprobe import LinearProbe

    probe = LinearProbe(
        model=tiny_model,
        layers=-1,
        pooling="last_token",
        device="cpu",
        remote=False,
        random_state=42,
    )

    probe.fit(["positive"], ["negative"])

    # Save/load for deployment
    save_path = tmp_path / "test_probe.pkl"
    probe.save(str(save_path))

    loaded_probe = LinearProbe.load(str(save_path))

    # Loaded probe should produce same predictions
    original_pred = probe.predict(["test input"])
    loaded_pred = loaded_probe.predict(["test input"])

    assert np.array_equal(original_pred, loaded_pred)


def test_readme_different_pooling(tiny_model):
    """Different train vs inference pooling from README works."""
    from lmprobe import LinearProbe

    probe = LinearProbe(
        model=tiny_model,
        layers=-1,
        pooling="last_token",
        inference_pooling="max",  # Override for predict()
        device="cpu",
        remote=False,
        random_state=42,
    )

    probe.fit(["positive example"], ["negative example"])
    predictions = probe.predict(["test input"])

    assert predictions.shape == (1,)


def test_readme_per_token_scores(tiny_model):
    """Per-token scoring with inference_pooling='all' works."""
    from lmprobe import LinearProbe

    probe = LinearProbe(
        model=tiny_model,
        layers=-1,
        pooling="last_token",
        inference_pooling="all",  # Return per-token scores
        device="cpu",
        remote=False,
        random_state=42,
    )

    probe.fit(["positive example"], ["negative example"])

    # With inference_pooling="all", we get per-token probabilities
    token_probs = probe.predict_proba(["test input with multiple tokens"])

    # Shape should be (batch, seq_len, n_classes) or (batch, seq_len) for binary
    assert len(token_probs.shape) >= 2
    assert token_probs.shape[0] == 1  # batch size


def test_readme_multilayer(tiny_model):
    """Multi-layer probing (concatenation) works."""
    from lmprobe import LinearProbe

    probe = LinearProbe(
        model=tiny_model,
        layers=[-2, -1],  # Last two layers, concatenated
        pooling="last_token",
        device="cpu",
        remote=False,
        random_state=42,
    )

    probe.fit(["positive"], ["negative"])
    predictions = probe.predict(["test"])

    assert predictions.shape == (1,)
