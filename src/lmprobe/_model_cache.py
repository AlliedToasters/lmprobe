"""Shared model cache used by both nnsight and local backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class ModelCache(Generic[T]):
    """Thread-safe-ish dict cache with load-on-miss semantics.

    Parameters
    ----------
    name : str
        Human-readable name (for debugging / repr).
    """

    def __init__(self, name: str = "ModelCache"):
        self._name = name
        self._cache: dict[tuple, T] = {}

    def get(self, key: tuple, loader: Callable[[], T]) -> T:
        """Return cached value or call *loader* to populate it."""
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    def clear(self) -> None:
        """Drop all cached entries."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: tuple) -> bool:
        return key in self._cache

    def __repr__(self) -> str:
        return f"{self._name}({len(self._cache)} entries)"
