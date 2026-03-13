"""Retry utility for transient remote extraction failures."""

import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    context: str = "",
) -> T:
    """Call fn() with exponential backoff retry on failure.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to retry.
    max_retries : int
        Maximum number of retry attempts (0 means no retries).
    base_delay : float
        Initial delay in seconds between retries.
    max_delay : float
        Maximum delay in seconds between retries.
    context : str
        Description for log messages (e.g., "batch 3/10").

    Returns
    -------
    T
        The return value of fn().

    Raises
    ------
    Exception
        The last exception if all retries are exhausted.
    """
    last_exception = None
    for attempt in range(1 + max_retries):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                # Add jitter (0-25%) to avoid thundering herd
                delay += delay * random.uniform(0, 0.25)
                ctx = f" ({context})" if context else ""
                logger.warning(
                    f"[RETRY] Attempt {attempt + 1}/{max_retries + 1} failed{ctx}, "
                    f"retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)
            else:
                ctx = f" ({context})" if context else ""
                logger.error(
                    f"[RETRY] All {max_retries + 1} attempts failed{ctx}: {e}"
                )
    raise last_exception  # type: ignore[misc]
