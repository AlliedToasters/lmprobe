"""Tests for cache backend implementations.

Contains a shared CacheBackendTests mixin that verifies the CacheBackend ABC
contract, used by both LocalCacheBackend and S3CacheBackend tests.
"""

import time

import pytest

from lmprobe.cache_backends import CacheBackend, LocalCacheBackend


class CacheBackendTests:
    """Shared acceptance tests for any CacheBackend implementation.

    Subclasses must provide a `backend` fixture that returns a ready-to-use
    CacheBackend instance.
    """

    def test_implements_abc(self, backend):
        assert isinstance(backend, CacheBackend)

    def test_exists_false_when_missing(self, backend):
        assert not backend.exists("nonexistent/key.bin")

    def test_write_and_read_bytes(self, backend):
        backend.write_bytes("model/file.bin", b"hello bytes")
        assert backend.exists("model/file.bin")
        assert backend.read_bytes("model/file.bin") == b"hello bytes"

    def test_write_and_read_text(self, backend):
        backend.write_text("model/_model_name.txt", "my-model")
        assert backend.read_text("model/_model_name.txt") == "my-model"

    def test_delete(self, backend):
        backend.write_bytes("model/file.bin", b"data")
        assert backend.exists("model/file.bin")
        backend.delete("model/file.bin")
        assert not backend.exists("model/file.bin")

    def test_delete_nonexistent_is_noop(self, backend):
        backend.delete("does/not/exist.bin")  # should not raise

    def test_list_keys(self, backend):
        backend.write_bytes("model1/a.bin", b"data")
        backend.write_bytes("model1/b.bin", b"data")
        backend.write_bytes("model2/c.bin", b"data")

        keys = backend.list_keys("model1")
        assert sorted(keys) == ["model1/a.bin", "model1/b.bin"]

    def test_list_keys_empty_prefix(self, backend):
        backend.write_bytes("model1/a.bin", b"data")
        backend.write_bytes("model2/b.bin", b"data")
        keys = backend.list_keys()
        assert len(keys) == 2

    def test_list_keys_nonexistent_prefix(self, backend):
        assert backend.list_keys("nonexistent") == []

    def test_size(self, backend):
        data = b"x" * 100
        backend.write_bytes("model/file.bin", data)
        assert backend.size("model/file.bin") == 100

    def test_mtime(self, backend):
        before = time.time()
        backend.write_bytes("model/file.bin", b"data")
        after = time.time()
        mt = backend.mtime("model/file.bin")
        assert before - 5 <= mt <= after + 5

    def test_collect_entries_safetensors(self, backend):
        backend.write_bytes("abc123/prompt1.safetensors", b"data1")
        backend.write_bytes("abc123/prompt2.safetensors", b"data22")

        entries = backend.collect_entries()
        assert len(entries) == 2
        keys = {e[0] for e in entries}
        assert "abc123/prompt1.safetensors" in keys
        assert "abc123/prompt2.safetensors" in keys
        for key, size, mtime in entries:
            assert isinstance(size, int)
            assert isinstance(mtime, float)

    def test_collect_entries_skips_metadata(self, backend):
        backend.write_text("abc123/_model_name.txt", "my-model")
        backend.write_bytes("abc123/prompt1.safetensors", b"data")
        entries = backend.collect_entries()
        assert len(entries) == 1
        assert entries[0][0] == "abc123/prompt1.safetensors"

    def test_delete_tree(self, backend):
        backend.write_bytes("abc123/prompt1.safetensors", b"data")
        backend.write_bytes("abc123/prompt2.safetensors", b"data")
        backend.write_text("abc123/_model_name.txt", "model")

        count = backend.delete_tree("abc123")
        assert count >= 2
        assert not backend.exists("abc123/prompt1.safetensors")
        assert not backend.exists("abc123/_model_name.txt")

    def test_delete_tree_nonexistent(self, backend):
        assert backend.delete_tree("nonexistent") == 0


class TestLocalCacheBackend(CacheBackendTests):
    """Tests for LocalCacheBackend (filesystem)."""

    @pytest.fixture
    def backend(self, tmp_path):
        return LocalCacheBackend(tmp_path)

    def test_touch(self, backend):
        backend.write_bytes("model/file.bin", b"data")
        original_mtime = backend.mtime("model/file.bin")
        time.sleep(0.05)
        backend.touch("model/file.bin")
        new_mtime = backend.mtime("model/file.bin")
        assert new_mtime >= original_mtime

    def test_write_bytes_creates_parent_dirs(self, backend):
        backend.write_bytes("deep/nested/dir/file.bin", b"data")
        assert backend.read_bytes("deep/nested/dir/file.bin") == b"data"

    def test_write_bytes_atomic(self, backend):
        """No .tmp files left after successful write."""
        backend.write_bytes("model/file.bin", b"data")
        import os

        parent = backend._path("model")
        files = os.listdir(parent)
        assert "file.tmp" not in files
        assert "file.bin" in files
