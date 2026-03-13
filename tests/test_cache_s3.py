"""Tests for S3CacheBackend and backend configuration."""

import pytest

from lmprobe.cache_backends import CacheBackend, LocalCacheBackend

# Skip all tests if moto is not installed
moto = pytest.importorskip("moto")
boto3 = pytest.importorskip("boto3")

from moto import mock_aws  # noqa: E402

from lmprobe.cache_backends import S3CacheBackend  # noqa: E402

TEST_BUCKET = "test-cache-bucket"
TEST_PREFIX = "lmprobe-cache"


@pytest.fixture
def s3_backend():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=TEST_BUCKET)
        yield S3CacheBackend(TEST_BUCKET, TEST_PREFIX)


class TestS3CacheBackend:
    """Tests for S3CacheBackend with moto."""

    def test_implements_abc(self, s3_backend):
        assert isinstance(s3_backend, CacheBackend)

    def test_exists_false_when_missing(self, s3_backend):
        assert not s3_backend.exists("nonexistent/key.bin")

    def test_write_and_read_bytes(self, s3_backend):
        s3_backend.write_bytes("model/file.bin", b"hello s3")
        assert s3_backend.exists("model/file.bin")
        assert s3_backend.read_bytes("model/file.bin") == b"hello s3"

    def test_write_and_read_text(self, s3_backend):
        s3_backend.write_text("model/_model_name.txt", "my-model")
        assert s3_backend.read_text("model/_model_name.txt") == "my-model"

    def test_delete(self, s3_backend):
        s3_backend.write_bytes("model/file.bin", b"data")
        assert s3_backend.exists("model/file.bin")
        s3_backend.delete("model/file.bin")
        assert not s3_backend.exists("model/file.bin")

    def test_delete_nonexistent_is_noop(self, s3_backend):
        s3_backend.delete("does/not/exist.bin")  # should not raise

    def test_list_keys(self, s3_backend):
        s3_backend.write_bytes("model1/a.bin", b"data")
        s3_backend.write_bytes("model1/b.bin", b"data")
        s3_backend.write_bytes("model2/c.bin", b"data")

        keys = s3_backend.list_keys("model1/")
        assert sorted(keys) == ["model1/a.bin", "model1/b.bin"]

    def test_list_keys_empty_prefix(self, s3_backend):
        s3_backend.write_bytes("model1/a.bin", b"data")
        s3_backend.write_bytes("model2/b.bin", b"data")
        keys = s3_backend.list_keys()
        assert len(keys) == 2

    def test_size(self, s3_backend):
        data = b"x" * 100
        s3_backend.write_bytes("model/file.bin", data)
        assert s3_backend.size("model/file.bin") == 100

    def test_mtime(self, s3_backend):
        import time

        before = time.time()
        s3_backend.write_bytes("model/file.bin", b"data")
        after = time.time()
        mt = s3_backend.mtime("model/file.bin")
        # moto timestamps are approximate
        assert mt >= before - 5
        assert mt <= after + 5

    def test_touch_is_noop(self, s3_backend):
        """touch() is a no-op on S3 (no writable mtime, LRU doesn't apply)."""
        s3_backend.write_bytes("model/file.bin", b"data")
        s3_backend.touch("model/file.bin")  # should not raise

    def test_collect_entries(self, s3_backend):
        s3_backend.write_bytes("abc123/prompt1.safetensors", b"data1")
        s3_backend.write_bytes("abc123/prompt2.safetensors", b"data22")

        entries = s3_backend.collect_entries()
        assert len(entries) == 2
        keys = {e[0] for e in entries}
        assert "abc123/prompt1.safetensors" in keys

    def test_collect_entries_skips_model_name(self, s3_backend):
        s3_backend.write_text("abc123/_model_name.txt", "my-model")
        s3_backend.write_bytes("abc123/prompt1.safetensors", b"data")
        entries = s3_backend.collect_entries()
        assert len(entries) == 1

    def test_delete_tree(self, s3_backend):
        s3_backend.write_bytes("abc123/prompt1.safetensors", b"data")
        s3_backend.write_bytes("abc123/prompt2.safetensors", b"data")
        s3_backend.write_text("abc123/_model_name.txt", "model")

        count = s3_backend.delete_tree("abc123/")
        assert count == 3  # all objects under prefix
        assert not s3_backend.exists("abc123/prompt1.safetensors")

    def test_prefix_handling(self):
        """Keys are correctly prefixed."""
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test")
            backend = S3CacheBackend("test", "my/prefix")
            backend.write_bytes("key.bin", b"data")
            # The full S3 key should include the prefix
            assert backend._full_key("key.bin") == "my/prefix/key.bin"
            assert backend.read_bytes("key.bin") == b"data"

    def test_prefix_without_trailing_slash(self):
        """Prefix normalization handles trailing slash."""
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test")
            b1 = S3CacheBackend("test", "prefix")
            b2 = S3CacheBackend("test", "prefix/")
            assert b1.prefix == b2.prefix

    def test_empty_prefix(self):
        """Empty prefix works (keys go at bucket root)."""
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test")
            backend = S3CacheBackend("test", "")
            backend.write_bytes("file.bin", b"data")
            assert backend.read_bytes("file.bin") == b"data"


class TestS3CacheBackendImportError:
    """Test clear error when boto3 is missing."""

    def test_import_error_message(self, monkeypatch):
        """S3CacheBackend gives a clear error when boto3 is not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="boto3 is required"):
            S3CacheBackend("bucket", "prefix")


class TestBackendConfiguration:
    """Tests for set_cache_backend(), get_backend(), _parse_backend_uri()."""

    def test_parse_s3_uri(self):
        from lmprobe.cache import _parse_backend_uri

        backend = _parse_backend_uri("s3://my-bucket/my-prefix/")
        assert isinstance(backend, S3CacheBackend)
        assert backend.bucket == "my-bucket"
        assert backend.prefix == "my-prefix/"

    def test_parse_s3_uri_no_prefix(self):
        from lmprobe.cache import _parse_backend_uri

        backend = _parse_backend_uri("s3://my-bucket")
        assert isinstance(backend, S3CacheBackend)
        assert backend.bucket == "my-bucket"
        assert backend.prefix == ""

    def test_parse_s3_uri_trailing_slash(self):
        from lmprobe.cache import _parse_backend_uri

        backend = _parse_backend_uri("s3://my-bucket/prefix")
        assert backend.bucket == "my-bucket"
        assert backend.prefix == "prefix/"

    def test_parse_local_path(self, tmp_path):
        from lmprobe.cache import _parse_backend_uri

        backend = _parse_backend_uri(str(tmp_path))
        assert isinstance(backend, LocalCacheBackend)
        assert backend.base_dir == tmp_path

    def test_parse_unknown_scheme_raises(self):
        from lmprobe.cache import _parse_backend_uri

        with pytest.raises(ValueError, match="Unsupported"):
            _parse_backend_uri("gcs://bucket/prefix")

    def test_set_cache_backend_string(self, tmp_path, monkeypatch):
        import lmprobe.cache as cache_mod

        old = cache_mod._backend
        try:
            from lmprobe.cache import get_backend, set_cache_backend

            set_cache_backend(str(tmp_path))
            backend = get_backend()
            assert isinstance(backend, LocalCacheBackend)
            assert backend.base_dir == tmp_path
        finally:
            cache_mod._backend = old

    def test_set_cache_backend_instance(self, tmp_path, monkeypatch):
        import lmprobe.cache as cache_mod

        old = cache_mod._backend
        try:
            from lmprobe.cache import get_backend, set_cache_backend

            backend_instance = LocalCacheBackend(tmp_path)
            set_cache_backend(backend_instance)
            assert get_backend() is backend_instance
        finally:
            cache_mod._backend = old

    def test_set_cache_backend_none_resets(self, monkeypatch):
        import lmprobe.cache as cache_mod

        old = cache_mod._backend
        try:
            from lmprobe.cache import set_cache_backend

            set_cache_backend(None)
            assert cache_mod._backend is None
        finally:
            cache_mod._backend = old

    def test_get_backend_default_is_local(self, tmp_path, monkeypatch):
        import lmprobe.cache as cache_mod

        old = cache_mod._backend
        try:
            cache_mod._backend = None
            monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
            from lmprobe.cache import get_backend

            backend = get_backend()
            assert isinstance(backend, LocalCacheBackend)
        finally:
            cache_mod._backend = old

    def test_env_var_backend(self, monkeypatch):
        """LMPROBE_CACHE_BACKEND env var configures the backend."""
        import lmprobe.cache as cache_mod

        old = cache_mod._backend
        try:
            cache_mod._backend = None
            monkeypatch.setenv("LMPROBE_CACHE_BACKEND", "s3://env-bucket/env-prefix")
            from lmprobe.cache import get_backend

            backend = get_backend()
            assert isinstance(backend, S3CacheBackend)
            assert backend.bucket == "env-bucket"
        finally:
            cache_mod._backend = old


class TestS3Integration:
    """Integration tests: exercise cache.py public functions with S3 backend."""

    @pytest.fixture(autouse=True)
    def setup_s3_backend(self, monkeypatch):
        """Set up mock S3 backend for all tests in this class."""
        import lmprobe.cache as cache_mod

        self._old_backend = cache_mod._backend
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=TEST_BUCKET)
            backend = S3CacheBackend(TEST_BUCKET, TEST_PREFIX)
            cache_mod._backend = backend
            yield
        cache_mod._backend = self._old_backend

    def test_save_and_load_prompt_activations(self):
        import torch

        from lmprobe.cache import (
            is_prompt_fully_cached,
            load_prompt_activations,
            save_prompt_activations,
        )

        model = "test-model"
        prompt = "hello world"
        layers = [0, 1]
        acts = torch.randn(1, 5, 128)
        mask = torch.ones(1, 5, dtype=torch.long)

        save_prompt_activations(model, prompt, layers, acts, mask)
        assert is_prompt_fully_cached(model, prompt, {0, 1})

        loaded_acts, loaded_mask = load_prompt_activations(model, prompt, layers)
        assert torch.allclose(acts, loaded_acts)
        assert torch.equal(mask, loaded_mask)

    def test_save_and_load_pooled_activations(self):
        import torch

        from lmprobe.cache import (
            is_prompt_pooled_cached,
            load_prompt_pooled_activations,
            save_prompt_pooled_activations,
        )

        model = "test-model"
        prompt = "test"
        layers = [0, 1]
        pooling = "last_token"
        pooled = torch.randn(1, 128)

        save_prompt_pooled_activations(model, prompt, layers, pooled, pooling)
        assert is_prompt_pooled_cached(model, prompt, {0, 1}, pooling)

        loaded = load_prompt_pooled_activations(model, prompt, layers, pooling)
        assert torch.allclose(pooled, loaded)

    def test_save_and_load_perplexity(self):
        import torch

        from lmprobe.cache import (
            is_prompt_perplexity_cached,
            load_prompt_perplexity,
            save_prompt_perplexity,
        )

        model = "test-model"
        prompt = "test"
        ppl = torch.tensor([1.5, 2.0, 3.0])

        save_prompt_perplexity(model, prompt, ppl)
        assert is_prompt_perplexity_cached(model, prompt)

        loaded = load_prompt_perplexity(model, prompt)
        assert torch.allclose(ppl, loaded)

    def test_cache_info_with_s3(self):
        import torch

        from lmprobe.cache import cache_info, save_prompt_activations

        save_prompt_activations(
            "test-model", "p1", [0],
            torch.randn(1, 5, 64),
            torch.ones(1, 5, dtype=torch.long),
        )
        save_prompt_activations(
            "test-model", "p2", [0],
            torch.randn(1, 5, 64),
            torch.ones(1, 5, dtype=torch.long),
        )

        info = cache_info()
        assert info.total_size_bytes > 0
        assert len(info.models) == 1
        assert info.models[0].num_prompts == 2

    def test_clear_cache_with_s3(self):
        import torch

        from lmprobe.cache import clear_cache, save_prompt_activations

        save_prompt_activations(
            "test-model", "p1", [0],
            torch.randn(1, 5, 64),
            torch.ones(1, 5, dtype=torch.long),
        )
        count = clear_cache()
        assert count >= 1

    def test_eviction_is_noop_on_s3(self):
        """LRU eviction should not run on S3 backend."""
        import torch

        import lmprobe.cache as cache_mod
        from lmprobe.cache import save_prompt_activations

        old_limit = cache_mod._CACHE_MAX_BYTES
        try:
            # Set a very small cache limit
            cache_mod._CACHE_MAX_BYTES = 1  # 1 byte

            # This should still work — eviction is no-op on S3
            save_prompt_activations(
                "test-model", "p1", [0],
                torch.randn(1, 5, 64),
                torch.ones(1, 5, dtype=torch.long),
            )
            # File should still exist (not evicted)
            from lmprobe.cache import is_prompt_fully_cached

            assert is_prompt_fully_cached("test-model", "p1", {0})
        finally:
            cache_mod._CACHE_MAX_BYTES = old_limit

    def test_register_model(self):
        from lmprobe.cache import _register_model, get_backend

        _register_model("org/my-model")
        from lmprobe.cache import _hash_string

        model_hash = _hash_string("org/my-model")
        backend = get_backend()
        name_key = f"{model_hash}/_model_name.txt"
        assert backend.exists(name_key)
        assert backend.read_text(name_key) == "org/my-model"

    def test_incremental_layer_save(self):
        import torch

        from lmprobe.cache import (
            is_prompt_fully_cached,
            load_prompt_activations,
            save_prompt_activations,
        )

        model = "test-model"
        prompt = "test prompt"

        # Save layer 0
        acts0 = torch.randn(1, 5, 64)
        mask = torch.ones(1, 5, dtype=torch.long)
        save_prompt_activations(model, prompt, [0], acts0, mask)

        # Save layer 1 (should merge)
        acts1 = torch.randn(1, 5, 64)
        save_prompt_activations(model, prompt, [1], acts1, mask)

        assert is_prompt_fully_cached(model, prompt, {0, 1})

        loaded0, _ = load_prompt_activations(model, prompt, [0])
        assert torch.allclose(acts0, loaded0)

        loaded1, _ = load_prompt_activations(model, prompt, [1])
        assert torch.allclose(acts1, loaded1)
