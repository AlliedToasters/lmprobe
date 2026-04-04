"""Tests for bulk extraction (extract.py).

Tests verify:
- extract() writes per-layer-keyed batch files and a self-contained manifest
- consolidate_cache() converts promptwise cache to batch format
- load_batch_layer() reads individual layers from batch files
- Round-trip: extract → load_batch_layer → verify shapes and content
"""

import torch
from conftest import POSITIVE_PROMPTS

from lmprobe.extract import (
    ExtractionManifest,
    _save_batch,
    consolidate_cache,
    extract,
    load_batch_layer,
    load_manifest,
)

# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


class TestExtractionManifest:
    def test_roundtrip(self):
        """Manifest serializes and deserializes correctly."""
        m = ExtractionManifest(
            model_name="test-model",
            layers=[0, 1, 2],
            hidden_dim=64,
            total_prompts=10,
            batch_size=4,
            prompts=["hello", "world"],
            labels=[0, 1],
            metadata=[{"src": "a"}, {"src": "b"}],
            created_at="2026-01-01T00:00:00Z",
        )
        d = m.to_dict()
        m2 = ExtractionManifest.from_dict(d)
        assert m2.model_name == "test-model"
        assert m2.layers == [0, 1, 2]
        assert m2.hidden_dim == 64
        assert m2.prompts == ["hello", "world"]
        assert m2.labels == [0, 1]

    def test_manifest_write_read(self, tmp_path, monkeypatch):
        """Manifest writes to and reads from cache backend."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        from lmprobe.extract import _write_manifest

        m = ExtractionManifest(
            model_name="test-model",
            layers=[0],
            hidden_dim=32,
            total_prompts=2,
            prompts=["a", "b"],
        )
        _write_manifest(m, "test_manifest")
        m2 = load_manifest("test_manifest")
        assert m2.model_name == "test-model"
        assert m2.prompts == ["a", "b"]


# ---------------------------------------------------------------------------
# Batch file I/O tests
# ---------------------------------------------------------------------------


class TestBatchFileIO:
    def test_save_and_load_batch(self, tmp_path, monkeypatch):
        """Batch saves one file per layer and loads them back."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        batch_size, seq_len, hidden_dim = 4, 16, 32
        layers = [0, 2, 5]
        num_layers = len(layers)

        batch_acts = torch.randn(batch_size, seq_len, hidden_dim * num_layers)
        batch_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        batch_mask[:, -3:] = 0

        prefix = "test_batch"
        _save_batch(batch_acts, batch_mask, layers, hidden_dim, prefix, batch_idx=0)

        # Verify per-layer files exist via backend
        from lmprobe.cache import get_backend
        from lmprobe.extract import _layer_batch_path

        backend = get_backend()
        for layer in layers:
            key = f"{prefix}/{_layer_batch_path(layer, 0)}"
            assert backend.exists(key)

        # Load via helper
        acts, mask = load_batch_layer(prefix, layer=2, batch_idx=0)
        assert acts.shape == (batch_size, seq_len, hidden_dim)
        assert mask.shape == (batch_size, seq_len)

    def test_layer_content_matches(self, tmp_path, monkeypatch):
        """Per-layer file content matches the original concatenated tensor."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        batch_size, seq_len, hidden_dim = 2, 8, 16
        layers = [0, 1]

        batch_acts = torch.randn(batch_size, seq_len, hidden_dim * 2)
        batch_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        prefix = "test_content"
        _save_batch(batch_acts, batch_mask, layers, hidden_dim, prefix, batch_idx=0)

        # Layer 0 should match first hidden_dim columns
        acts_0, _ = load_batch_layer(prefix, layer=0, batch_idx=0)
        expected_0 = batch_acts[:, :, :hidden_dim]
        assert torch.allclose(acts_0, expected_0)

        # Layer 1 should match second hidden_dim columns
        acts_1, _ = load_batch_layer(prefix, layer=1, batch_idx=0)
        expected_1 = batch_acts[:, :, hidden_dim:]
        assert torch.allclose(acts_1, expected_1)


# ---------------------------------------------------------------------------
# extract() integration test
# ---------------------------------------------------------------------------


class TestExtract:
    def test_extract_basic(self, tiny_model, tmp_path, monkeypatch):
        """extract() writes batch files and manifest for tiny model."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path / "cache"))

        prompts = POSITIVE_PROMPTS[:3]
        labels = [1, 1, 1]
        output_prefix = "test_extraction"

        result = extract(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,  # last layer only
            labels=labels,
            output_dir=output_prefix,
            batch_size=2,
            remote=False,
            device="cpu",
            backend="local",
        )

        assert result == output_prefix

        # Check manifest
        manifest = load_manifest(result)
        assert manifest.model_name == tiny_model
        assert manifest.total_prompts == 3
        assert len(manifest.prompts) == 3
        assert manifest.prompts == prompts
        assert manifest.labels == labels
        assert len(manifest.batches) == 2  # 3 prompts / batch_size=2
        assert manifest.hidden_dim > 0
        assert len(manifest.layers) == 1

        # Check per-layer batch files exist via backend
        from lmprobe.cache import get_backend
        from lmprobe.extract import _layer_batch_path

        backend = get_backend()
        layer = manifest.layers[0]
        for batch_info in manifest.batches:
            assert batch_info.status == "complete"
            assert len(batch_info.num_tokens) == batch_info.prompt_end - batch_info.prompt_start
            bi = batch_info.prompt_start // 2  # batch_size=2
            key = f"{result}/{_layer_batch_path(layer, bi)}"
            assert backend.exists(key)

        # Load a layer from first batch and verify shape
        acts, mask = load_batch_layer(result, layer, batch_idx=0)
        assert acts.shape[0] == 2  # batch_size
        assert acts.shape[2] == manifest.hidden_dim

    def test_extract_resumability(self, tiny_model, tmp_path, monkeypatch):
        """extract() skips already-completed batches on resume."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path / "cache"))

        prompts = POSITIVE_PROMPTS[:4]
        output_prefix = "test_resume"

        # First run
        extract(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_dir=output_prefix,
            batch_size=2,
            remote=False,
            device="cpu",
            backend="local",
        )

        manifest1 = load_manifest(output_prefix)
        assert len(manifest1.batches) == 2

        # Second run — should skip both batches
        extract(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_dir=output_prefix,
            batch_size=2,
            remote=False,
            device="cpu",
            backend="local",
        )

        manifest2 = load_manifest(output_prefix)
        # Should still have 2 batches (not 4)
        assert len(manifest2.batches) == 2


# ---------------------------------------------------------------------------
# consolidate_cache() integration test
# ---------------------------------------------------------------------------


class TestConsolidateCache:
    def test_consolidate_basic(self, tiny_model, tmp_path, monkeypatch):
        """consolidate_cache() reads promptwise cache and writes batch files."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path / "cache"))

        prompts = POSITIVE_PROMPTS[:3]

        # First, warm up the promptwise cache
        from lmprobe.unified_cache import UnifiedCache

        uc = UnifiedCache(
            model=tiny_model,
            layers=-1,
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=False,  # Need raw activations
            backend="local",
        )
        uc.warmup(prompts, remote=False)

        # Now consolidate — writes to local filesystem
        consol_dir = str(tmp_path / "consolidated")
        result = consolidate_cache(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_dir=consol_dir,
            batch_size=2,
        )

        assert result == consol_dir
        manifest = load_manifest(result)
        assert manifest.total_prompts == 3
        assert len(manifest.batches) == 2
        assert manifest.prompts == prompts

        # Verify batch file content via N×L layout
        layer = manifest.layers[0]
        acts, mask = load_batch_layer(result, layer, batch_idx=0)
        assert acts.shape[0] == 2  # batch_size
        assert acts.shape[2] == manifest.hidden_dim

        # Verify num_tokens is reasonable (each prompt has >0 tokens)
        for batch_info in manifest.batches:
            for nt in batch_info.num_tokens:
                assert nt > 0

    def test_consolidate_writes_local_files(self, tiny_model, tmp_path, monkeypatch):
        """consolidate_cache() writes files to local filesystem, not backend."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path / "cache"))

        prompts = POSITIVE_PROMPTS[:2]

        from lmprobe.unified_cache import UnifiedCache

        uc = UnifiedCache(
            model=tiny_model,
            layers=-1,
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=False,
            backend="local",
        )
        uc.warmup(prompts, remote=False)

        out_dir = tmp_path / "local_output"
        result = consolidate_cache(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_dir=str(out_dir),
            batch_size=2,
        )

        # Verify files exist on local filesystem
        from pathlib import Path

        local_root = Path(result)
        assert (local_root / "manifest.json").exists()

        manifest = load_manifest(result)
        layer = manifest.layers[0]
        from lmprobe.extract import _layer_batch_path

        layer_file = local_root / _layer_batch_path(layer, 0)
        assert layer_file.exists()

        # Verify files are NOT inside the cache directory
        cache_dir = tmp_path / "cache"
        assert not str(local_root).startswith(str(cache_dir))

    def test_consolidate_output_uri_writes_to_backend(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """consolidate_cache(output_uri=...) writes via cache backend."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path / "cache"))

        prompts = POSITIVE_PROMPTS[:2]

        from lmprobe.unified_cache import UnifiedCache

        uc = UnifiedCache(
            model=tiny_model,
            layers=-1,
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=False,
            backend="local",
        )
        uc.warmup(prompts, remote=False)

        uri_prefix = "consolidated_s3_test"
        result = consolidate_cache(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_uri=uri_prefix,
            batch_size=2,
        )

        assert result == uri_prefix

        # Files should be in the cache backend, not on local filesystem
        from lmprobe.cache import get_backend
        from lmprobe.extract import _layer_batch_path

        backend = get_backend()
        assert backend.exists(f"{uri_prefix}/manifest.json")

        manifest = load_manifest(result)
        layer = manifest.layers[0]
        key = f"{uri_prefix}/{_layer_batch_path(layer, 0)}"
        assert backend.exists(key)

        # load_batch_layer should find it via backend fallback
        acts, mask = load_batch_layer(result, layer, batch_idx=0)
        assert acts.shape[0] == 2
        assert acts.shape[2] == manifest.hidden_dim

    def test_consolidate_exclusive_params(self, tiny_model):
        """consolidate_cache raises if both output_dir and output_uri given."""
        import pytest

        with pytest.raises(ValueError, match="not both"):
            consolidate_cache(
                model_name=tiny_model,
                prompts=["hello"],
                layers=-1,
                output_dir="/tmp/foo",
                output_uri="bar",
            )

    def test_consolidate_matches_extract(self, tiny_model, tmp_path, monkeypatch):
        """consolidate_cache output matches extract output for same prompts."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path / "cache"))

        prompts = POSITIVE_PROMPTS[:2]

        # Extract directly
        extract_prefix = extract(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_dir="extracted",
            batch_size=2,
            remote=False,
            device="cpu",
            backend="local",
        )

        # Warm up cache then consolidate
        from lmprobe.unified_cache import UnifiedCache

        uc = UnifiedCache(
            model=tiny_model,
            layers=-1,
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=False,
            backend="local",
        )
        uc.warmup(prompts, remote=False)

        consol_prefix = consolidate_cache(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_dir=str(tmp_path / "consolidated"),
            batch_size=2,
        )

        # Both should have same structure
        m_extract = load_manifest(extract_prefix)
        m_consol = load_manifest(consol_prefix)

        assert m_extract.layers == m_consol.layers
        assert m_extract.hidden_dim == m_consol.hidden_dim
        assert m_extract.total_prompts == m_consol.total_prompts
        assert len(m_extract.batches) == len(m_consol.batches)

        # Activations should match (same model, same prompts)
        layer = m_extract.layers[0]
        acts_e, mask_e = load_batch_layer(extract_prefix, layer, batch_idx=0)
        acts_c, mask_c = load_batch_layer(consol_prefix, layer, batch_idx=0)

        assert acts_e.shape == acts_c.shape
        # Mask should match (both derive from same tokenization)
        assert torch.equal(mask_e, mask_c)
        # Activations should be close (float rounding may differ slightly)
        assert torch.allclose(acts_e, acts_c, atol=1e-5)

    def test_consolidate_resumability(self, tiny_model, tmp_path, monkeypatch):
        """consolidate_cache() skips already-completed batches on resume."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path / "cache"))

        prompts = POSITIVE_PROMPTS[:4]

        from lmprobe.unified_cache import UnifiedCache

        uc = UnifiedCache(
            model=tiny_model,
            layers=-1,
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=False,
            backend="local",
        )
        uc.warmup(prompts, remote=False)

        out_dir = str(tmp_path / "resume_test")

        # First run
        consolidate_cache(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_dir=out_dir,
            batch_size=2,
        )

        manifest1 = load_manifest(out_dir)
        assert len(manifest1.batches) == 2

        # Second run — should skip both batches
        consolidate_cache(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_dir=out_dir,
            batch_size=2,
        )

        manifest2 = load_manifest(out_dir)
        # Should still have 2 batches (not 4)
        assert len(manifest2.batches) == 2

    def test_consolidate_resumability_uri(self, tiny_model, tmp_path, monkeypatch):
        """consolidate_cache(output_uri=...) resumes from backend manifest."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path / "cache"))

        prompts = POSITIVE_PROMPTS[:4]

        from lmprobe.unified_cache import UnifiedCache

        uc = UnifiedCache(
            model=tiny_model,
            layers=-1,
            compute_perplexity=False,
            device="cpu",
            remote=False,
            cache_pooled=False,
            backend="local",
        )
        uc.warmup(prompts, remote=False)

        uri = "resume_uri_test"

        # First run
        consolidate_cache(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_uri=uri,
            batch_size=2,
        )

        manifest1 = load_manifest(uri)
        assert len(manifest1.batches) == 2

        # Second run — should skip
        consolidate_cache(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_uri=uri,
            batch_size=2,
        )

        manifest2 = load_manifest(uri)
        assert len(manifest2.batches) == 2


# ---------------------------------------------------------------------------
# _preload_layer_from_batches test
# ---------------------------------------------------------------------------


class TestPiecemealShardAssembly:
    def test_build_shard_for_layer(self, tiny_model, tmp_path, monkeypatch):
        """_build_shard_batch_map + _build_shard_for_layer produce correct shards."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path / "cache"))

        prompts = POSITIVE_PROMPTS[:4]
        prefix = "test_piecemeal"

        extract(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_dir=prefix,
            batch_size=2,
            remote=False,
            device="cpu",
            backend="local",
        )

        manifest = load_manifest(prefix)
        layer = manifest.layers[0]

        from lmprobe.extract import _build_shard_batch_map, _build_shard_for_layer

        # Identity permutation — one shard covering all prompts (last-token)
        perm = list(range(len(prompts)))
        lt_boundaries = [len(prompts)]
        rest_boundaries = [len(prompts)]
        lt_shard_count = 1

        shard_map = _build_shard_batch_map(
            manifest, perm, lt_boundaries, rest_boundaries, lt_shard_count,
        )

        assert len(shard_map) == 2  # 1 last-token + 1 rest-token

        # Last-token shard
        lt_shard = shard_map[0]
        assert lt_shard.is_last_token
        lt_tensor = _build_shard_for_layer(lt_shard, prefix, layer)
        assert lt_tensor is not None
        assert lt_tensor.ndim == 2
        assert lt_tensor.shape == (4, manifest.hidden_dim)

        # Rest-token shard
        rest_shard = shard_map[1]
        assert not rest_shard.is_last_token
        rest_tensor = _build_shard_for_layer(rest_shard, prefix, layer)
        # rest tokens exist if prompts have >1 token
        if rest_tensor is not None:
            assert rest_tensor.ndim == 2
            assert rest_tensor.shape[1] == manifest.hidden_dim

    def test_shuffled_order_preserved(self, tiny_model, tmp_path, monkeypatch):
        """Shards reflect shuffled order, not original order."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path / "cache"))

        prompts = POSITIVE_PROMPTS[:4]
        prefix = "test_piecemeal_shuffle"

        extract(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_dir=prefix,
            batch_size=2,
            remote=False,
            device="cpu",
            backend="local",
        )

        manifest = load_manifest(prefix)
        layer = manifest.layers[0]

        from lmprobe.extract import _build_shard_batch_map, _build_shard_for_layer

        # Identity permutation
        perm_identity = list(range(4))
        shard_map_id = _build_shard_batch_map(
            manifest, perm_identity, [4], [], 1,
        )
        lt_id = _build_shard_for_layer(shard_map_id[0], prefix, layer)

        # Reversed permutation
        perm_rev = list(reversed(range(4)))
        shard_map_rev = _build_shard_batch_map(
            manifest, perm_rev, [4], [], 1,
        )
        lt_rev = _build_shard_for_layer(shard_map_rev[0], prefix, layer)

        assert lt_id is not None and lt_rev is not None
        # Row 0 with reversed perm should equal row 3 with identity
        assert torch.allclose(lt_rev[0], lt_id[3])
        assert torch.allclose(lt_rev[3], lt_id[0])


# ---------------------------------------------------------------------------
# push_extraction streaming tests
# ---------------------------------------------------------------------------


class TestPushExtractionStreaming:
    """Tests for push_extraction(stream=True).

    These mock the HuggingFace API to avoid real uploads.
    """

    def _make_extraction(self, tiny_model, tmp_path, monkeypatch):
        """Helper: create a small extraction directory."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path / "cache"))

        prompts = POSITIVE_PROMPTS[:4]
        prefix = extract(
            model_name=tiny_model,
            prompts=prompts,
            layers=-1,
            output_dir=str(tmp_path / "extracted"),
            batch_size=2,
            remote=False,
            device="cpu",
            backend="local",
        )
        return prefix

    def _hub_patches(self):
        """Context manager stack for mocking HF hub + pyarrow deps."""
        from contextlib import ExitStack
        from unittest.mock import patch

        stack = ExitStack()
        stack.enter_context(patch("lmprobe.sharing._check_pyarrow"))
        stack.enter_context(patch("lmprobe.sharing._check_hub_deps"))
        stack.enter_context(patch("lmprobe.sharing._write_parquet_index"))
        return stack

    def test_stream_calls_create_commit(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Streaming mode uses create_commit, not upload_large_folder."""
        prefix = self._make_extraction(tiny_model, tmp_path, monkeypatch)

        from unittest.mock import MagicMock, patch

        mock_api = MagicMock()
        mock_api.repo_info.side_effect = Exception("no repo yet")

        from lmprobe.extract import push_extraction

        with self._hub_patches():
            with patch("huggingface_hub.HfApi", return_value=mock_api):
                with patch(
                    "huggingface_hub.CommitOperationAdd"
                ) as mock_op:
                    mock_op.side_effect = lambda **kw: kw
                    push_extraction(
                        source=prefix,
                        repo_id="test-user/test-dataset",
                        stream=True,
                        stream_batch_size=5,
                        staging_dir=str(tmp_path / "staging"),
                    )

        mock_api.create_repo.assert_called_once()
        assert mock_api.create_commit.call_count >= 2  # shards + metadata
        mock_api.upload_large_folder.assert_not_called()

    def test_stream_deletes_shard_files(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Streaming mode deletes shard files after upload."""
        prefix = self._make_extraction(tiny_model, tmp_path, monkeypatch)

        from unittest.mock import MagicMock, patch

        mock_api = MagicMock()
        mock_api.repo_info.side_effect = Exception("no repo yet")

        staging = tmp_path / "staging"

        from lmprobe.extract import push_extraction

        with self._hub_patches():
            with patch("huggingface_hub.HfApi", return_value=mock_api):
                with patch("huggingface_hub.CommitOperationAdd", side_effect=lambda **kw: kw):
                    push_extraction(
                        source=prefix,
                        repo_id="test-user/test-dataset",
                        stream=True,
                        stream_batch_size=2,
                        staging_dir=str(staging),
                    )

        remaining = list(staging.rglob("*.safetensors"))
        assert remaining == [], f"Shard files not cleaned up: {remaining}"

    def test_stream_skip_existing_shards(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Streaming mode skips shards already on remote."""
        prefix = self._make_extraction(tiny_model, tmp_path, monkeypatch)

        from unittest.mock import MagicMock, patch

        manifest = load_manifest(prefix)
        layer = manifest.layers[0]

        from lmprobe.sharing import _hidden_shard_filename

        existing_shard = _hidden_shard_filename(layer, 0)

        mock_api = MagicMock()
        mock_repo_info = MagicMock()
        mock_repo_info.siblings = [MagicMock(rfilename=existing_shard)]
        mock_api.repo_info.return_value = mock_repo_info

        from lmprobe.extract import push_extraction

        uploaded_shards: list[str] = []

        def tracking_commit(**kwargs):
            ops = kwargs.get("operations", [])
            for op in ops:
                if isinstance(op, dict):
                    uploaded_shards.append(op.get("path_in_repo", ""))

        mock_api.create_commit.side_effect = tracking_commit

        with self._hub_patches():
            with patch("huggingface_hub.HfApi", return_value=mock_api):
                with patch("huggingface_hub.CommitOperationAdd", side_effect=lambda **kw: kw):
                    push_extraction(
                        source=prefix,
                        repo_id="test-user/test-dataset",
                        stream=True,
                        staging_dir=str(tmp_path / "staging"),
                    )

        assert existing_shard not in uploaded_shards, (
            f"Shard {existing_shard} was uploaded despite being on remote"
        )

    def test_skip_layer_load_when_all_shards_exist(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Layer data is not loaded when all its shards already exist on remote."""
        prefix = self._make_extraction(tiny_model, tmp_path, monkeypatch)

        from unittest.mock import MagicMock, patch

        mock_api = MagicMock()
        # _check_shards_on_remote returns ALL expected files → every layer skipped
        mock_api.repo_info.side_effect = Exception("should not be called directly")

        from lmprobe.extract import push_extraction

        with self._hub_patches():
            with patch("huggingface_hub.HfApi", return_value=mock_api):
                with patch("huggingface_hub.CommitOperationAdd", side_effect=lambda **kw: kw):
                    with patch(
                        "lmprobe.sharing._check_shards_on_remote",
                        side_effect=lambda api, repo_id, expected: set(expected),
                    ):
                        with patch(
                            "lmprobe.extract._build_shard_for_layer"
                        ) as mock_build:
                            push_extraction(
                                source=prefix,
                                repo_id="test-user/test-dataset",
                                stream=True,
                                staging_dir=str(tmp_path / "staging"),
                            )

        # _build_shard_for_layer should never be called — all layers skipped
        mock_build.assert_not_called()

    def test_nonstream_unchanged(
        self, tiny_model, tmp_path, monkeypatch
    ):
        """Non-streaming mode still uses upload_large_folder."""
        prefix = self._make_extraction(tiny_model, tmp_path, monkeypatch)

        from unittest.mock import MagicMock, patch

        mock_api = MagicMock()

        from lmprobe.extract import push_extraction

        with self._hub_patches():
            with patch("huggingface_hub.HfApi", return_value=mock_api):
                push_extraction(
                    source=prefix,
                    repo_id="test-user/test-dataset",
                    exist_ok=True,
                )

        mock_api.upload_large_folder.assert_called_once()
        mock_api.create_commit.assert_not_called()
