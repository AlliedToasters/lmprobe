"""
Remote/NDIF tests for lmprobe.

STATUS: NOT YET TESTED

These tests require:
1. US-based network access (NDIF restricts international access)
2. NNSIGHT_API_KEY environment variable set

To run:
    export NNSIGHT_API_KEY="your-key"
    pytest tests/test_remote.py -v

These tests are skipped by default if NNSIGHT_API_KEY is not set.
"""

import os

import pytest

# Skip all tests in this module if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("NNSIGHT_API_KEY"),
    reason="NNSIGHT_API_KEY not set (required for remote/NDIF tests)"
)


@pytest.fixture
def remote_model():
    """A model available on NDIF for remote testing."""
    # Use a smaller model that's likely available on NDIF
    return "meta-llama/Llama-3.1-8B-Instruct"


class TestRemoteExecution:
    """Tests for remote=True functionality via NDIF."""

    def test_remote_fit(self, remote_model):
        """Test that fit() works with remote=True."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=remote_model,
            layers=-1,
            remote=True,
            random_state=42,
        )

        # Use minimal prompts to reduce remote compute
        probe.fit(["positive"], ["negative"])

        assert probe.classifier_ is not None

    def test_remote_predict(self, remote_model):
        """Test that predict() works with remote=True."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=remote_model,
            layers=-1,
            remote=True,
            random_state=42,
        )

        probe.fit(["positive"], ["negative"])
        predictions = probe.predict(["test"])

        assert predictions.shape == (1,)

    def test_remote_override_on_predict(self, remote_model, tiny_model):
        """Test training remote but predicting with override.

        This test verifies the remote parameter can be overridden per-call.
        Note: This specific test may not be practical since the classifier
        is trained on remote model activations which won't match local model.
        """
        pytest.skip("Train remote / predict local requires same model")

    def test_remote_large_model(self):
        """Test with a large model only available via remote."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model="meta-llama/Llama-3.1-70B-Instruct",
            layers="middle",
            remote=True,
            random_state=42,
        )

        probe.fit(["positive example"], ["negative example"])
        predictions = probe.predict(["test input"])

        assert predictions.shape == (1,)

    def test_remote_caching(self, remote_model):
        """Test that remote activations are cached properly."""
        from lmprobe import LinearProbe
        from lmprobe.cache import get_cache_dir

        probe = LinearProbe(
            model=remote_model,
            layers=-1,
            remote=True,
            random_state=42,
        )

        prompts = ["cache test prompt"]

        # First fit - should extract and cache
        probe.fit(prompts, ["negative"])

        # Check cache directory has files
        cache_dir = get_cache_dir()
        cache_files = list(cache_dir.glob("*.pt"))
        assert len(cache_files) > 0, "Expected cached activation files"


class TestRemoteErrorHandling:
    """Tests for error handling in remote mode."""

    def test_missing_api_key_error(self, monkeypatch):
        """Test clear error when NNSIGHT_API_KEY is missing."""
        from lmprobe import LinearProbe

        # Temporarily remove the API key
        monkeypatch.delenv("NNSIGHT_API_KEY", raising=False)

        probe = LinearProbe(
            model="meta-llama/Llama-3.1-8B-Instruct",
            layers=-1,
            remote=True,
        )

        with pytest.raises(EnvironmentError, match="NNSIGHT_API_KEY"):
            probe.fit(["positive"], ["negative"])
