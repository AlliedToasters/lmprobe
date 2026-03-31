"""Internal profiling/timing infrastructure for lmprobe.

Enable profiling to get wall-clock timing breakdowns for key operations.

Usage::

    import lmprobe
    lmprobe.set_profiling(True)  # or set LMPROBE_PROFILE=1

    # Operations automatically log timing to the "lmprobe.profile" logger.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

profile_logger = logging.getLogger("lmprobe.profile")

_profiling_enabled: bool = os.environ.get("LMPROBE_PROFILE", "0") not in ("0", "")


def set_profiling(enabled: bool) -> None:
    """Enable or disable profiling globally.

    When enabled, key operations log wall-clock timing to the
    ``lmprobe.profile`` logger and store structured results on
    ``last_profile_`` attributes.

    Can also be enabled via the ``LMPROBE_PROFILE=1`` environment variable.

    Parameters
    ----------
    enabled : bool
        Whether to enable profiling.
    """
    global _profiling_enabled
    _profiling_enabled = enabled
    if enabled and not profile_logger.handlers and not profile_logger.parent:
        # Ensure output is visible if no handlers configured
        profile_logger.addHandler(logging.StreamHandler())
        profile_logger.setLevel(logging.INFO)


def is_profiling() -> bool:
    """Return whether profiling is currently enabled."""
    return _profiling_enabled


def _fmt_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds >= 60:
        return f"{seconds / 60:.1f}min"
    if seconds >= 1:
        return f"{seconds:.1f}s"
    if seconds >= 0.001:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds * 1_000_000:.0f}µs"


class ProfileAccumulator:
    """Accumulates sub-step timings for a single profiled operation.

    Used as a context manager to time the overall operation, with
    ``section()`` calls to time sub-steps within it.

    Parameters
    ----------
    name : str
        Name of the operation being profiled.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.sections: dict[str, float] = {}
        self.total: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> ProfileAccumulator:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.total = time.perf_counter() - self._start
        parts = ", ".join(f"{k}: {_fmt_time(v)}" for k, v in self.sections.items())
        msg = f"{self.name}: {_fmt_time(self.total)}"
        if parts:
            msg += f" ({parts})"
        profile_logger.info("[lmprobe.profile] %s", msg)

    @contextmanager
    def section(self, name: str) -> Generator[None, None, None]:
        """Time a named sub-section of this operation."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.sections[name] = time.perf_counter() - t0

    def as_dict(self) -> dict[str, float]:
        """Return timing results as a dict."""
        result: dict[str, float] = {"total": self.total}
        result.update(self.sections)
        return result


@contextmanager
def profile_op(name: str) -> Generator[ProfileAccumulator | None, None, None]:
    """Context manager that profiles an operation when profiling is enabled.

    When profiling is disabled, yields None with zero overhead (no timing
    calls, no string formatting).

    Parameters
    ----------
    name : str
        Name of the operation (e.g. ``"Probe.fit"``).

    Yields
    ------
    ProfileAccumulator | None
        Accumulator for sub-step timings, or None if profiling is disabled.
    """
    if not _profiling_enabled:
        yield None
        return

    acc = ProfileAccumulator(name)
    with acc:
        yield acc


@contextmanager
def profile_section(
    accumulator: ProfileAccumulator | None, name: str
) -> Generator[None, None, None]:
    """Time a sub-section, safely handling None accumulators.

    This is a convenience wrapper so callers don't need to guard every
    section with ``if acc is not None``.

    Parameters
    ----------
    accumulator : ProfileAccumulator | None
        The accumulator from ``profile_op``, or None if profiling is disabled.
    name : str
        Sub-section name.
    """
    if accumulator is None:
        yield
        return
    with accumulator.section(name):
        yield
