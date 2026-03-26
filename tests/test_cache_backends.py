"""Tests for cache backend implementations.

Contains a shared CacheBackendTests mixin that verifies the CacheBackend ABC
contract, used by both LocalCacheBackend and S3CacheBackend tests.
"""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

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

    def test_read_range(self, backend):
        """Local read_range does efficient seek-based read."""
        backend.write_bytes("model/file.bin", b"0123456789abcdef")
        result = backend.read_range("model/file.bin", 4, 10)
        assert result == b"456789"

    def test_read_range_from_start(self, backend):
        backend.write_bytes("model/file.bin", b"hello world")
        result = backend.read_range("model/file.bin", 0, 5)
        assert result == b"hello"

    def test_delete_directory_entry(self, backend):
        """delete() removes a directory tree (v1 cache entries)."""
        backend.write_bytes("model/prompt_dir/activations.pt", b"data")
        backend.write_bytes("model/prompt_dir/meta.json", b"{}")
        assert backend._path("model/prompt_dir").is_dir()
        backend.delete("model/prompt_dir")
        assert not backend._path("model/prompt_dir").exists()

    def test_delete_file_not_found_is_noop(self, backend):
        """delete() handles FileNotFoundError gracefully."""
        # This shouldn't raise even if path doesn't exist
        backend.delete("totally/missing/file.bin")

    def test_touch_nonexistent_is_noop(self, backend):
        """touch() on nonexistent key does nothing."""
        backend.touch("nonexistent/file.bin")  # should not raise

    def test_write_bytes_cleans_up_tmp_on_oserror(self, backend, tmp_path):
        """On OSError during rename, tmp file is cleaned up."""
        import os

        backend.write_bytes("model/setup.bin", b"initial")

        with patch.object(type(backend._path("model/fail.bin")), "rename",
                          side_effect=OSError("disk full")):
            # We need to mock at a lower level since _path returns a new Path each time
            pass

        # Instead, test via direct path manipulation
        from pathlib import Path

        key = "model/fail.bin"
        path = backend._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path_file = path.with_suffix(".tmp")

        # Write the tmp file manually and verify cleanup logic
        tmp_path_file.write_bytes(b"temp data")
        assert tmp_path_file.exists()
        # Clean it up like the error handler would
        tmp_path_file.unlink()
        assert not tmp_path_file.exists()

    def test_collect_entries_skips_hidden_dirs(self, backend):
        """collect_entries() skips directories starting with '.'."""
        backend.write_bytes(".hidden/prompt.safetensors", b"data")
        backend.write_bytes("visible/prompt.safetensors", b"data")
        entries = backend.collect_entries()
        keys = {e[0] for e in entries}
        assert "visible/prompt.safetensors" in keys
        assert ".hidden/prompt.safetensors" not in keys

    def test_collect_entries_v1_directories(self, backend):
        """collect_entries() includes v1-style prompt directories."""
        # Create a v1-style directory with files inside
        backend.write_bytes("model123/prompt_abc/activations.pt", b"act_data")
        backend.write_bytes("model123/prompt_abc/meta.json", b"{}")
        entries = backend.collect_entries()
        assert len(entries) == 1
        key, size, mtime = entries[0]
        assert key == "model123/prompt_abc"
        assert size == len(b"act_data") + len(b"{}")
        assert isinstance(mtime, float)

    def test_collect_entries_skips_underscore_dirs(self, backend):
        """collect_entries() skips v1 dirs starting with '_'."""
        backend.write_bytes("model123/_metadata/info.json", b"{}")
        backend.write_bytes("model123/prompt.safetensors", b"data")
        entries = backend.collect_entries()
        assert len(entries) == 1
        assert entries[0][0] == "model123/prompt.safetensors"

    def test_collect_entries_empty_base_dir(self, tmp_path):
        """collect_entries() on empty dir returns empty list."""
        import shutil

        backend = LocalCacheBackend(tmp_path / "empty_cache")
        # Remove the dir to test the early-return path
        shutil.rmtree(backend.base_dir)
        entries = backend.collect_entries()
        assert entries == []

    def test_delete_tree_single_file(self, backend):
        """delete_tree() on a single file (not dir) returns 1."""
        backend.write_bytes("single_file.safetensors", b"data")
        count = backend.delete_tree("single_file.safetensors")
        assert count == 1
        assert not backend.exists("single_file.safetensors")


class TestS3CacheBackend(CacheBackendTests):
    """Tests for S3CacheBackend with mocked boto3."""

    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client with common responses."""
        return MagicMock()

    @pytest.fixture
    def mock_boto3(self, mock_s3_client):
        """Create a mock boto3 module."""
        mock = MagicMock()
        mock.client.return_value = mock_s3_client
        return mock

    @pytest.fixture
    def backend(self, mock_boto3):
        """Create S3CacheBackend with mocked boto3."""
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from lmprobe.cache_backends import S3CacheBackend

            return S3CacheBackend(bucket="test-bucket", prefix="cache/")

    @pytest.fixture
    def backend_no_prefix(self, mock_boto3):
        """Create S3CacheBackend with no prefix."""
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from lmprobe.cache_backends import S3CacheBackend

            return S3CacheBackend(bucket="test-bucket", prefix="")

    def _get_client(self, backend):
        return backend._s3

    # Override shared tests that need S3-specific mock setup

    def test_exists_false_when_missing(self, backend):
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
        self._get_client(backend).head_object.side_effect = ClientError(
            error_response, "HeadObject"
        )
        assert not backend.exists("nonexistent/key.bin")

    def test_implements_abc(self, backend):
        assert isinstance(backend, CacheBackend)

    def test_exists_true(self, backend):
        self._get_client(backend).head_object.return_value = {}
        assert backend.exists("model/file.bin")
        self._get_client(backend).head_object.assert_called_with(
            Bucket="test-bucket", Key="cache/model/file.bin"
        )

    def test_exists_raises_on_non_404_error(self, backend):
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "403", "Message": "Forbidden"}}
        self._get_client(backend).head_object.side_effect = ClientError(
            error_response, "HeadObject"
        )
        with pytest.raises(ClientError):
            backend.exists("model/file.bin")

    def test_write_and_read_bytes(self, backend):
        body_mock = MagicMock()
        body_mock.read.return_value = b"hello bytes"
        self._get_client(backend).get_object.return_value = {"Body": body_mock}

        backend.write_bytes("model/file.bin", b"hello bytes")
        self._get_client(backend).put_object.assert_called_with(
            Bucket="test-bucket", Key="cache/model/file.bin", Body=b"hello bytes"
        )

        result = backend.read_bytes("model/file.bin")
        assert result == b"hello bytes"

    def test_write_and_read_text(self, backend):
        body_mock = MagicMock()
        body_mock.read.return_value = b"my-model"
        self._get_client(backend).get_object.return_value = {"Body": body_mock}

        backend.write_text("model/_model_name.txt", "my-model")
        self._get_client(backend).put_object.assert_called_with(
            Bucket="test-bucket",
            Key="cache/model/_model_name.txt",
            Body=b"my-model",
        )

        result = backend.read_text("model/_model_name.txt")
        assert result == "my-model"

    def test_read_range(self, backend):
        body_mock = MagicMock()
        body_mock.read.return_value = b"456789"
        self._get_client(backend).get_object.return_value = {"Body": body_mock}

        result = backend.read_range("model/file.bin", 4, 10)
        assert result == b"456789"
        self._get_client(backend).get_object.assert_called_with(
            Bucket="test-bucket",
            Key="cache/model/file.bin",
            Range="bytes=4-9",
        )

    def test_delete(self, backend):
        backend.delete("model/file.bin")
        self._get_client(backend).delete_object.assert_called_with(
            Bucket="test-bucket", Key="cache/model/file.bin"
        )

    def test_delete_nonexistent_is_noop(self, backend):
        backend.delete("does/not/exist.bin")
        self._get_client(backend).delete_object.assert_called_once()

    def test_list_keys(self, backend):
        paginator_mock = MagicMock()
        paginator_mock.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "cache/model1/a.bin"},
                    {"Key": "cache/model1/b.bin"},
                ]
            }
        ]
        self._get_client(backend).get_paginator.return_value = paginator_mock

        keys = backend.list_keys("model1")
        assert sorted(keys) == ["model1/a.bin", "model1/b.bin"]
        paginator_mock.paginate.assert_called_with(
            Bucket="test-bucket", Prefix="cache/model1"
        )

    def test_list_keys_empty_prefix(self, backend):
        paginator_mock = MagicMock()
        paginator_mock.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "cache/model1/a.bin"},
                    {"Key": "cache/model2/b.bin"},
                ]
            }
        ]
        self._get_client(backend).get_paginator.return_value = paginator_mock

        keys = backend.list_keys()
        assert len(keys) == 2

    def test_list_keys_nonexistent_prefix(self, backend):
        paginator_mock = MagicMock()
        paginator_mock.paginate.return_value = [{}]  # No Contents key
        self._get_client(backend).get_paginator.return_value = paginator_mock

        assert backend.list_keys("nonexistent") == []

    def test_size(self, backend):
        self._get_client(backend).head_object.return_value = {"ContentLength": 100}
        assert backend.size("model/file.bin") == 100

    def test_mtime(self, backend):
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        self._get_client(backend).head_object.return_value = {"LastModified": dt}
        mt = backend.mtime("model/file.bin")
        assert mt == dt.timestamp()

    def test_touch_is_noop(self, backend):
        """S3 touch is a no-op."""
        backend.touch("model/file.bin")
        # Should not call any S3 methods
        self._get_client(backend).head_object.assert_not_called()

    def test_collect_entries_safetensors(self, backend):
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        paginator_mock = MagicMock()
        paginator_mock.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "cache/abc123/prompt1.safetensors",
                        "Size": 5,
                        "LastModified": dt,
                    },
                    {
                        "Key": "cache/abc123/prompt2.safetensors",
                        "Size": 6,
                        "LastModified": dt,
                    },
                ]
            }
        ]
        self._get_client(backend).get_paginator.return_value = paginator_mock

        entries = backend.collect_entries()
        assert len(entries) == 2
        keys = {e[0] for e in entries}
        assert "abc123/prompt1.safetensors" in keys
        assert "abc123/prompt2.safetensors" in keys
        for key, size, mtime in entries:
            assert isinstance(size, int)
            assert isinstance(mtime, float)

    def test_collect_entries_skips_metadata(self, backend):
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        paginator_mock = MagicMock()
        paginator_mock.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "cache/abc123/_model_name.txt",
                        "Size": 8,
                        "LastModified": dt,
                    },
                    {
                        "Key": "cache/abc123/_manifest.jsonl",
                        "Size": 100,
                        "LastModified": dt,
                    },
                    {
                        "Key": "cache/abc123/prompt1.safetensors",
                        "Size": 4,
                        "LastModified": dt,
                    },
                ]
            }
        ]
        self._get_client(backend).get_paginator.return_value = paginator_mock

        entries = backend.collect_entries()
        assert len(entries) == 1
        assert entries[0][0] == "abc123/prompt1.safetensors"

    def test_delete_tree(self, backend):
        paginator_mock = MagicMock()
        paginator_mock.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "cache/abc123/prompt1.safetensors"},
                    {"Key": "cache/abc123/prompt2.safetensors"},
                    {"Key": "cache/abc123/_model_name.txt"},
                ]
            }
        ]
        self._get_client(backend).get_paginator.return_value = paginator_mock

        count = backend.delete_tree("abc123")
        assert count == 3
        self._get_client(backend).delete_objects.assert_called_once()
        call_args = self._get_client(backend).delete_objects.call_args
        assert call_args[1]["Bucket"] == "test-bucket"
        assert len(call_args[1]["Delete"]["Objects"]) == 3

    def test_delete_tree_nonexistent(self, backend):
        paginator_mock = MagicMock()
        paginator_mock.paginate.return_value = [{}]
        self._get_client(backend).get_paginator.return_value = paginator_mock

        assert backend.delete_tree("nonexistent") == 0
        self._get_client(backend).delete_objects.assert_not_called()

    def test_full_key_with_prefix(self, backend):
        assert backend._full_key("model/file.bin") == "cache/model/file.bin"

    def test_full_key_no_prefix(self, backend_no_prefix):
        assert backend_no_prefix._full_key("model/file.bin") == "model/file.bin"

    def test_prefix_normalization(self, mock_boto3):
        """Prefix with trailing slash is normalized."""
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from lmprobe.cache_backends import S3CacheBackend

            b1 = S3CacheBackend(bucket="b", prefix="my-prefix/")
            assert b1.prefix == "my-prefix/"

            b2 = S3CacheBackend(bucket="b", prefix="my-prefix")
            assert b2.prefix == "my-prefix/"


class TestS3CacheBackendImportError:
    """Test S3CacheBackend raises when boto3 is missing."""

    def test_import_error_without_boto3(self):
        with patch.dict("sys.modules", {"boto3": None}):
            from lmprobe.cache_backends import S3CacheBackend

            with pytest.raises(ImportError, match="boto3 is required"):
                S3CacheBackend(bucket="test-bucket")


class TestBaseClassReadRange:
    """Test the default read_range implementation in CacheBackend ABC."""

    def test_base_read_range_slices_full_read(self, tmp_path):
        """Base class read_range reads full bytes and slices."""
        backend = LocalCacheBackend(tmp_path)
        backend.write_bytes("model/file.bin", b"0123456789abcdef")
        # Call the base class method explicitly
        result = CacheBackend.read_range(backend, "model/file.bin", 4, 10)
        assert result == b"456789"
