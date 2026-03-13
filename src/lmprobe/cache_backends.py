"""Pluggable cache storage backends for lmprobe.

This module defines the CacheBackend interface and provides two
concrete implementations:

- LocalCacheBackend: Wraps filesystem I/O (default)
- S3CacheBackend: Uses boto3 for S3 storage (optional dependency)

Both backends operate on string keys (e.g. "a1b2c3/hash.safetensors"),
not Path objects, so the same cache logic works across storage types.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path


class CacheBackend(ABC):
    """Abstract base class for cache storage backends.

    All backends operate on string keys rather than filesystem paths.
    Keys use forward-slash separators (e.g. "model_hash/prompt.safetensors").
    """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if an entry exists."""
        ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """Read raw bytes for a key."""
        ...

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None:
        """Write raw bytes for a key."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a single entry. No error if missing."""
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List keys under a prefix."""
        ...

    @abstractmethod
    def size(self, key: str) -> int:
        """Get size in bytes of an entry."""
        ...

    @abstractmethod
    def mtime(self, key: str) -> float:
        """Get last-modified time as a Unix timestamp."""
        ...

    @abstractmethod
    def touch(self, key: str) -> None:
        """Update the last-modified time (for LRU tracking)."""
        ...

    @abstractmethod
    def read_text(self, key: str) -> str:
        """Read a text entry."""
        ...

    @abstractmethod
    def write_text(self, key: str, text: str) -> None:
        """Write a text entry."""
        ...

    @abstractmethod
    def collect_entries(self) -> list[tuple[str, int, float]]:
        """Return all cache entries as (key, size_bytes, mtime).

        Used for LRU eviction and cache_info().
        """
        ...

    @abstractmethod
    def delete_tree(self, prefix: str) -> int:
        """Bulk delete all keys under a prefix.

        Returns the number of entries deleted.
        """
        ...


class LocalCacheBackend(CacheBackend):
    """Cache backend backed by the local filesystem.

    Parameters
    ----------
    base_dir : Path | str
        Root directory for cache storage.
    """

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.base_dir / key

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_bytes(data)
            tmp_path.rename(path)
        except OSError:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    def delete(self, key: str) -> None:
        path = self._path(key)
        try:
            if path.is_dir():
                import shutil

                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except FileNotFoundError:
            pass

    def list_keys(self, prefix: str = "") -> list[str]:
        search_dir = self._path(prefix) if prefix else self.base_dir
        if not search_dir.exists():
            return []
        keys = []
        for item in search_dir.rglob("*"):
            if item.is_file():
                keys.append(str(item.relative_to(self.base_dir)))
        return keys

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size

    def mtime(self, key: str) -> float:
        return self._path(key).stat().st_mtime

    def touch(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            os.utime(path)

    def read_text(self, key: str) -> str:
        return self._path(key).read_text()

    def write_text(self, key: str, text: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def collect_entries(self) -> list[tuple[str, int, float]]:
        """Collect safetensors files and v1 prompt directories."""
        entries = []
        if not self.base_dir.exists():
            return entries

        for model_dir in self.base_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue

            # v2 safetensors files
            for sf_file in model_dir.glob("*.safetensors"):
                stat = sf_file.stat()
                key = str(sf_file.relative_to(self.base_dir))
                entries.append((key, stat.st_size, stat.st_mtime))

            # v1 directories
            for prompt_dir in model_dir.iterdir():
                if not prompt_dir.is_dir() or prompt_dir.name.startswith("_"):
                    continue
                dir_size = sum(
                    f.stat().st_size for f in prompt_dir.rglob("*") if f.is_file()
                )
                key = str(prompt_dir.relative_to(self.base_dir))
                entries.append((key, dir_size, prompt_dir.stat().st_mtime))

        return entries

    def delete_tree(self, prefix: str) -> int:
        """Delete all entries under a model directory prefix."""
        import shutil

        path = self._path(prefix)
        if not path.exists():
            return 0

        count = 0
        if path.is_dir():
            # Count entries before deletion
            count += sum(1 for _ in path.glob("*.safetensors"))
            count += sum(
                1 for d in path.iterdir() if d.is_dir() and not d.name.startswith("_")
            )
            shutil.rmtree(path)
        else:
            path.unlink()
            count = 1
        return count


class S3CacheBackend(CacheBackend):
    """Cache backend using Amazon S3.

    Requires ``boto3`` (install with ``pip install lmprobe[s3]``).

    LRU eviction is disabled — S3 is designed for accumulating
    large activation datasets, not ephemeral caching. Use S3
    Lifecycle Rules if you need automatic cleanup.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        Key prefix within the bucket (e.g. "lmprobe-cache/").
    """

    def __init__(self, bucket: str, prefix: str = ""):
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 cache backend. "
                "Install it with: pip install lmprobe[s3]"
            )
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        if self.prefix:
            self.prefix += "/"
        self._s3 = boto3.client("s3")

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def read_bytes(self, key: str) -> bytes:
        resp = self._s3.get_object(Bucket=self.bucket, Key=self._full_key(key))
        return resp["Body"].read()

    def write_bytes(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=self._full_key(key), Body=data)

    def delete(self, key: str) -> None:
        self._s3.delete_object(Bucket=self.bucket, Key=self._full_key(key))

    def list_keys(self, prefix: str = "") -> list[str]:
        full_prefix = self._full_key(prefix)
        keys = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                # Strip our prefix to return relative keys
                rel_key = obj["Key"][len(self.prefix):]
                keys.append(rel_key)
        return keys

    def size(self, key: str) -> int:
        resp = self._s3.head_object(Bucket=self.bucket, Key=self._full_key(key))
        return resp["ContentLength"]

    def mtime(self, key: str) -> float:
        resp = self._s3.head_object(Bucket=self.bucket, Key=self._full_key(key))
        return resp["LastModified"].timestamp()

    def touch(self, key: str) -> None:
        # No-op on S3: no writable mtime, and LRU doesn't apply.
        pass

    def read_text(self, key: str) -> str:
        return self.read_bytes(key).decode("utf-8")

    def write_text(self, key: str, text: str) -> None:
        self.write_bytes(key, text.encode("utf-8"))

    def collect_entries(self) -> list[tuple[str, int, float]]:
        """List all cache entries with size and mtime.

        Note: On large caches this requires paginated LIST + HEAD per object
        and can be slow.
        """
        entries = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                rel_key = obj["Key"][len(self.prefix):]
                # Skip metadata files
                if rel_key.endswith("/_model_name.txt"):
                    continue
                entries.append((
                    rel_key,
                    obj["Size"],
                    obj["LastModified"].timestamp(),
                ))
        return entries

    def delete_tree(self, prefix: str) -> int:
        """Delete all objects under a prefix."""
        keys_to_delete = self.list_keys(prefix)
        count = 0
        # S3 delete_objects handles up to 1000 at a time
        for i in range(0, len(keys_to_delete), 1000):
            batch = keys_to_delete[i : i + 1000]
            objects = [{"Key": self._full_key(k)} for k in batch]
            self._s3.delete_objects(
                Bucket=self.bucket, Delete={"Objects": objects}
            )
            count += len(batch)
        return count
