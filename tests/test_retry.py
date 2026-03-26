"""Tests for retry logic used in remote extraction."""

from unittest.mock import MagicMock, patch

import pytest

from lmprobe.retry import retry_with_backoff


class TestRetryWithBackoff:
    """Tests for the retry_with_backoff utility."""

    def test_succeeds_first_try(self):
        """Function that succeeds on first call should not retry."""
        fn = MagicMock(return_value=42)
        result = retry_with_backoff(fn, max_retries=3)
        assert result == 42
        assert fn.call_count == 1

    def test_succeeds_after_retries(self):
        """Function that fails then succeeds should return the success."""
        fn = MagicMock(side_effect=[ConnectionError("fail"), ConnectionError("fail"), 42])
        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == 42
        assert fn.call_count == 3

    def test_all_retries_exhausted(self):
        """Should raise the last exception when all retries fail."""
        fn = MagicMock(side_effect=ConnectionError("always fails"))
        with pytest.raises(ConnectionError, match="always fails"):
            retry_with_backoff(fn, max_retries=2, base_delay=0.01)
        assert fn.call_count == 3  # 1 initial + 2 retries

    def test_no_retries(self):
        """max_retries=0 means no retries — fail immediately."""
        fn = MagicMock(side_effect=ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            retry_with_backoff(fn, max_retries=0)
        assert fn.call_count == 1

    def test_exponential_backoff_timing(self):
        """Delays should increase exponentially."""
        fn = MagicMock(side_effect=[ConnectionError("1"), ConnectionError("2"), 42])
        with patch("lmprobe.retry.time.sleep") as mock_sleep:
            with patch("lmprobe.retry.random.uniform", return_value=0):
                retry_with_backoff(fn, max_retries=3, base_delay=1.0)
        # First retry: 1.0 * 2^0 = 1.0, second retry: 1.0 * 2^1 = 2.0
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert len(delays) == 2
        assert delays[0] == pytest.approx(1.0)
        assert delays[1] == pytest.approx(2.0)

    def test_max_delay_cap(self):
        """Delay should be capped at max_delay."""
        fn = MagicMock(side_effect=[ConnectionError("1"), ConnectionError("2"), 42])
        with patch("lmprobe.retry.time.sleep") as mock_sleep:
            with patch("lmprobe.retry.random.uniform", return_value=0):
                retry_with_backoff(fn, max_retries=3, base_delay=50.0, max_delay=60.0)
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert all(d <= 60.0 for d in delays)

    def test_context_in_log_messages(self):
        """Context string should appear in log messages."""
        fn = MagicMock(side_effect=[ConnectionError("oops"), 42])
        with patch("lmprobe.retry.time.sleep"):
            with patch("lmprobe.retry.logger") as mock_logger:
                retry_with_backoff(fn, max_retries=1, base_delay=0.01, context="batch 3/10")
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "batch 3/10" in warning_msg

    def test_error_logged_on_final_failure(self):
        """Final failure should log an error message."""
        fn = MagicMock(side_effect=ConnectionError("permanent"))
        with patch("lmprobe.retry.time.sleep"):
            with patch("lmprobe.retry.logger") as mock_logger:
                with pytest.raises(ConnectionError):
                    retry_with_backoff(fn, max_retries=2, base_delay=0.01, context="op X")
        error_msg = mock_logger.error.call_args[0][0]
        assert "All" in error_msg
        assert "op X" in error_msg

    def test_error_logged_without_context(self):
        """Error log works when no context string is given."""
        fn = MagicMock(side_effect=ValueError("fail"))
        with patch("lmprobe.retry.time.sleep"):
            with patch("lmprobe.retry.logger") as mock_logger:
                with pytest.raises(ValueError):
                    retry_with_backoff(fn, max_retries=1, base_delay=0.01)
        error_msg = mock_logger.error.call_args[0][0]
        assert "All" in error_msg

    def test_jitter_applied(self):
        """Verify jitter is applied to delay."""
        fn = MagicMock(side_effect=[ConnectionError("1"), 42])
        with patch("lmprobe.retry.time.sleep") as mock_sleep:
            with patch("lmprobe.retry.random.uniform", return_value=0.25):
                retry_with_backoff(fn, max_retries=1, base_delay=2.0)
        delay = mock_sleep.call_args[0][0]
        # base_delay * 2^0 = 2.0, jitter = 2.0 * 0.25 = 0.5, total = 2.5
        assert delay == pytest.approx(2.5)


class TestCachedExtractorRetry:
    """Test that CachedExtractor uses retry for remote extraction."""

    def test_no_retry_for_local_extraction(self):
        """Local extraction should never retry — errors are real, not transient."""
        from unittest.mock import patch as mock_patch

        from lmprobe.cache import CachedExtractor

        mock_extractor = MagicMock()
        mock_extractor.model_name = "test-model"
        mock_extractor.layer_indices = [0]
        mock_extractor.batch_size = 8
        mock_extractor.extract_batch.side_effect = RuntimeError("real error")

        cached = CachedExtractor(mock_extractor)

        # Even with max_retries=5, local extraction should not retry
        with mock_patch("lmprobe.cache.is_prompt_fully_cached", return_value=False):
            with mock_patch("lmprobe.cache._register_model"):
                with pytest.raises(RuntimeError, match="real error"):
                    cached.extract(
                        ["test prompt"],
                        remote=False,
                        max_retries=5,  # should be ignored for local
                    )
        # Should only be called once (no retries)
        assert mock_extractor.extract_batch.call_count == 1
