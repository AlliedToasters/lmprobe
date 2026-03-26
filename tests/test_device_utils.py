"""Tests for CUDA compute capability early detection."""

from unittest.mock import patch

import pytest

from lmprobe._device_utils import check_cuda_compatibility


class TestCheckCudaCompatibility:
    """Tests for check_cuda_compatibility."""

    def test_cpu_device_skips_check(self):
        """No error raised for CPU device."""
        check_cuda_compatibility("cpu")

    def test_auto_device_skips_check(self):
        """No error raised for auto device."""
        check_cuda_compatibility("auto")

    def test_cuda_not_available(self):
        """Clear error when CUDA is requested but not available."""
        with patch("lmprobe._device_utils.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = False
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                check_cuda_compatibility("cuda")

    def test_cuda_device_index_out_of_range(self):
        """Clear error when requested device index exceeds available devices."""
        with patch("lmprobe._device_utils.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_cuda.device_count.return_value = 1
            with pytest.raises(RuntimeError, match="only 1 CUDA device"):
                check_cuda_compatibility("cuda:2")

    def test_malformed_device_index_returns(self):
        """Malformed device string like 'cuda:abc' returns without error."""
        with patch("lmprobe._device_utils.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            # Should not raise — let PyTorch handle malformed strings
            check_cuda_compatibility("cuda:abc")

    def test_incompatible_compute_capability(self):
        """Clear error on compute capability mismatch (no kernel image)."""
        with patch("lmprobe._device_utils.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_cuda.device_count.return_value = 1
            mock_cuda.get_device_capability.return_value = (6, 1)
            mock_cuda.get_device_name.return_value = "NVIDIA GeForce GTX 1060"
            with patch(
                "lmprobe._device_utils.torch.tensor",
                side_effect=RuntimeError("no kernel image is available for execution on the device"),
            ):
                with pytest.raises(
                    RuntimeError, match="GTX 1060.*sm_61.*not supported by this PyTorch build"
                ):
                    check_cuda_compatibility("cuda")

    def test_other_runtime_error_reraises(self):
        """Other RuntimeErrors during tensor op are re-raised as-is."""
        with patch("lmprobe._device_utils.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_cuda.device_count.return_value = 1
            mock_cuda.get_device_capability.return_value = (7, 0)
            mock_cuda.get_device_name.return_value = "V100"
            with patch(
                "lmprobe._device_utils.torch.tensor",
                side_effect=RuntimeError("some other error"),
            ):
                with pytest.raises(RuntimeError, match="some other error"):
                    check_cuda_compatibility("cuda")

    def test_compatible_cuda_passes(self):
        """No error when CUDA device is compatible."""
        from unittest.mock import MagicMock

        mock_tensor = MagicMock()
        mock_tensor.__add__ = MagicMock(return_value=mock_tensor)

        with patch("lmprobe._device_utils.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_cuda.device_count.return_value = 1
            mock_cuda.get_device_capability.return_value = (8, 0)
            mock_cuda.get_device_name.return_value = "NVIDIA A100"
            with patch("lmprobe._device_utils.torch.tensor", return_value=mock_tensor):
                check_cuda_compatibility("cuda:0")

    def test_bare_cuda_device_uses_index_zero(self):
        """'cuda' without index should check device 0."""
        from unittest.mock import MagicMock

        mock_tensor = MagicMock()
        mock_tensor.__add__ = MagicMock(return_value=mock_tensor)

        with patch("lmprobe._device_utils.torch.cuda") as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_cuda.device_count.return_value = 1
            mock_cuda.get_device_capability.return_value = (8, 0)
            mock_cuda.get_device_name.return_value = "A100"
            with patch("lmprobe._device_utils.torch.tensor", return_value=mock_tensor):
                check_cuda_compatibility("cuda")
