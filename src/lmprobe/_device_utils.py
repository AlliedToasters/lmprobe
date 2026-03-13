"""Internal utilities for device validation."""

from __future__ import annotations

import torch


def check_cuda_compatibility(device: str) -> None:
    """Check CUDA compute capability before loading a model.

    Raises a clear error if the requested CUDA device has a compute
    capability that is too low for the installed PyTorch build.

    Parameters
    ----------
    device : str
        Device specification ("auto", "cpu", "cuda", "cuda:0", etc.).
        Only performs checks when a CUDA device is explicitly requested.
    """
    # Only check when user explicitly requests CUDA
    if not device.startswith("cuda"):
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device='{device}' was requested, but CUDA is not available. "
            "Use device='cpu' or device='auto'."
        )

    # Parse device index
    if ":" in device:
        try:
            device_idx = int(device.split(":")[1])
        except (ValueError, IndexError):
            return  # Let PyTorch handle malformed device strings
    else:
        device_idx = 0

    if device_idx >= torch.cuda.device_count():
        raise RuntimeError(
            f"device='{device}' was requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available."
        )

    capability = torch.cuda.get_device_capability(device_idx)
    gpu_name = torch.cuda.get_device_name(device_idx)
    sm = capability[0] * 10 + capability[1]

    # PyTorch wheels from pip typically require sm_70+ (Volta and newer).
    # The exact minimum depends on the PyTorch build, but sm_70 is the
    # most common cutoff for recent PyTorch versions.
    # We check by attempting a small tensor operation on the device.
    try:
        t = torch.tensor([1.0], device=device)
        _ = t + t
        del t
    except RuntimeError as e:
        if "no kernel image" in str(e).lower():
            raise RuntimeError(
                f"GPU '{gpu_name}' (compute capability {capability[0]}.{capability[1]}, sm_{sm}) "
                f"is not supported by this PyTorch build. "
                f"Use device='cpu', device='auto', or install a PyTorch version "
                f"compatible with your GPU."
            ) from e
        raise
