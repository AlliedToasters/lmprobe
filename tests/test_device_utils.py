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

    @patch("lmprobe._device_utils.torch")
    def test_cuda_not_available(self, mock_torch):
        """Clear error when CUDA is requested but not available."""
        mock_torch.cuda.is_available.return_value = False
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            check_cuda_compatibility("cuda")

    @patch("lmprobe._device_utils.torch")
    def test_cuda_device_index_out_of_range(self, mock_torch):
        """Clear error when requested device index exceeds available devices."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        with pytest.raises(RuntimeError, match="only 1 CUDA device"):
            check_cuda_compatibility("cuda:2")

    @patch("lmprobe._device_utils.torch")
    def test_incompatible_compute_capability(self, mock_torch):
        """Clear error on compute capability mismatch (no kernel image)."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_capability.return_value = (6, 1)
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce GTX 1060"
        mock_torch.tensor.side_effect = RuntimeError(
            "no kernel image is available for execution on the device"
        )
        with pytest.raises(
            RuntimeError, match="GTX 1060.*sm_61.*not supported by this PyTorch build"
        ):
            check_cuda_compatibility("cuda")

    @patch("lmprobe._device_utils.torch")
    def test_compatible_cuda_passes(self, mock_torch):
        """No error when CUDA device is compatible."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_capability.return_value = (8, 0)
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"
        mock_tensor = mock_torch.tensor.return_value
        mock_tensor.__add__ = lambda self, other: self
        check_cuda_compatibility("cuda:0")
