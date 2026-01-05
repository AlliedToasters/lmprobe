"""
Remote/NDIF tests for lmprobe.

These tests require:
1. US-based network access (NDIF restricts international access)
2. NNSIGHT_API_KEY environment variable set

To run:
    export NNSIGHT_API_KEY="your-key"
    pytest tests/test_remote.py -v

To skip these tests (they're slow, expensive, and may hit rate limits):
    # Option 1: Use the marker
    pytest -m "not remote"

    # Option 2: Set environment variable
    export SKIP_REMOTE_TESTS=1
    pytest tests/test_remote.py

These tests are skipped with a warning if NNSIGHT_API_KEY is not set.
"""

import os
import warnings

import pytest

# Check skip conditions
_api_key = os.getenv("NNSIGHT_API_KEY")
_skip_remote = os.getenv("SKIP_REMOTE_TESTS", "").lower() in ("1", "true", "yes")

# Determine skip reason
if _skip_remote:
    _skip_reason = "SKIP_REMOTE_TESTS is set - skipping expensive remote tests"
elif not _api_key:
    _skip_reason = "NNSIGHT_API_KEY not set (required for remote/NDIF tests)"
else:
    _skip_reason = None

# Issue warning when skipping due to missing API key
if not _api_key and not _skip_remote:
    warnings.warn(
        "Skipping remote tests: NNSIGHT_API_KEY environment variable not set. "
        "Set it to run tests against NDIF, or use SKIP_REMOTE_TESTS=1 to suppress this warning.",
        UserWarning,
        stacklevel=1,
    )

# Mark all tests in this module as remote and skip if conditions met
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(_skip_reason is not None, reason=_skip_reason or ""),
]


@pytest.fixture
def remote_model():
    """A model available on NDIF for remote testing."""
    return "meta-llama/Llama-3.1-8B"


@pytest.fixture
def large_remote_model():
    """A large model only available via NDIF remote."""
    return "meta-llama/Llama-3.1-70B-Instruct"


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

    def test_remote_multilayer(self, remote_model):
        """Test extracting multiple layers remotely."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=remote_model,
            layers=[0, 15, 31],  # First, middle, last layers of 32-layer model
            remote=True,
            random_state=42,
        )

        probe.fit(["positive example"], ["negative example"])
        predictions = probe.predict(["test input"])

        assert predictions.shape == (1,)

    def test_remote_override_on_predict(self, remote_model):
        """Test training remote but predicting with override.

        Note: This specific test may not be practical since the classifier
        is trained on remote model activations which won't match local model.
        """
        pytest.skip("Train remote / predict local requires same model weights")

    def test_remote_large_model(self, large_remote_model):
        """Test with a large model only available via remote."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=large_remote_model,
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

    def test_missing_api_key_error(self, monkeypatch, remote_model):
        """Test clear error when NNSIGHT_API_KEY is missing."""
        from lmprobe import LinearProbe

        # Temporarily remove the API key
        monkeypatch.delenv("NNSIGHT_API_KEY", raising=False)

        probe = LinearProbe(
            model=remote_model,
            layers=-1,
            remote=True,
        )

        with pytest.raises(EnvironmentError, match="NNSIGHT_API_KEY"):
            probe.fit(["positive"], ["negative"])
