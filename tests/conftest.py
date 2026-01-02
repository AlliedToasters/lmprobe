"""Shared pytest fixtures for lmprobe tests."""

import pytest

# Tiny Llama model with random weights for fast functional testing
# See: https://huggingface.co/stas/tiny-random-llama-2
TEST_MODEL = "stas/tiny-random-llama-2"


@pytest.fixture
def tiny_model():
    """Return the tiny random Llama model ID for testing.

    This model has random weights and is designed for functional testing,
    not for quality generation. Tests using this fixture will verify
    that the pipeline works end-to-end, but predictions are meaningless.
    """
    return TEST_MODEL
