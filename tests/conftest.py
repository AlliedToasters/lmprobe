"""Shared pytest fixtures for lmprobe tests."""

import copy

import pytest

# Tiny Llama model with random weights for fast functional testing
# See: https://huggingface.co/stas/tiny-random-llama-2
TEST_MODEL = "stas/tiny-random-llama-2"

# Shared test data used across multiple test files
POSITIVE_PROMPTS = [
    "The dog barked loudly",
    "My puppy loves to play fetch",
    "Dogs are loyal companions",
    "The golden retriever wagged its tail",
    "Walking the dog in the park",
]

NEGATIVE_PROMPTS = [
    "The cat purred softly",
    "My kitten sleeps all day",
    "Cats are independent animals",
    "The tabby cat stretched lazily",
    "The cat knocked things off the table",
]

TEST_PROMPTS = [
    "A dog chased the ball",
    "The cat sat on the mat",
]


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
    old_manifests = copy.deepcopy(cache_mod._shard_manifests)
    old_indices = copy.deepcopy(cache_mod._shard_indices)
    cache_mod._shard_manifests.clear()
    cache_mod._shard_indices.clear()
    yield
    cache_mod._backend = old
    cache_mod._shard_manifests.update(old_manifests)
    cache_mod._shard_indices.update(old_indices)
