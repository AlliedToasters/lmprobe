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


@pytest.fixture(autouse=True)
def _reset_cache_backend():
    """Reset the cache backend global between tests.

    This ensures that tests using LMPROBE_CACHE_DIR via monkeypatch
    get a fresh LocalCacheBackend pointing to the correct directory.
    """
    import lmprobe.cache as cache_mod

    old = cache_mod._backend
    cache_mod._backend = None
    yield
    cache_mod._backend = old
