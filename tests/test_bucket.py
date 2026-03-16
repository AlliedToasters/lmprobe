"""Tests for bucket.py — push/pull/load with mocked HuggingFace API."""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import lmprobe.cache as cache_mod
from lmprobe.bucket import (
    FORMAT_VERSION,
    _build_dataset_info,
    _build_readme,
    _compute_tensor_intersection,
    _consolidate_and_shard,
    _discover_prompts,
    _filter_tensor_types,
    load_from_bucket,
    pull_from_bucket,
    push_to_bucket,
)
from lmprobe.cache import (
    CachedPromptInfo,
    discover_cached,
    load_prompt_pooled_activations,
    save_prompt_activations,
    save_prompt_pooled_activations,
)

TEST_MODEL = "stas/tiny-random-llama-2"
HIDDEN_DIM = 32
SEQ_LEN = 5


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Set up a temporary cache directory."""
    monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


@pytest.fixture
def populated_cache(cache_dir):
    """Populate cache with a small dataset (pooled activations)."""
    prompts = [
        "Who wants to go for a walk?",
        "Fetch the ball!",
        "Purring and scratching",
    ]
    for prompt in prompts:
        # Store pooled activations for layers 0 and 1
        for layer in [0, 1]:
            pooled = torch.randn(1, HIDDEN_DIM)
            save_prompt_pooled_activations(
                TEST_MODEL, prompt, [layer], pooled, "last_token"
            )
    return prompts


class TestComputeTensorIntersection:
    def test_single_info(self):
        info = CachedPromptInfo(
            raw_layers=[0, 1],
            pooled={"last_token": [0, 1]},
            has_logits=False,
            logits_top_k=None,
            has_perplexity=True,
            num_tokens=5,
        )
        result = _compute_tensor_intersection([info])
        assert result["raw_layers"] == [0, 1]
        assert result["pooled"] == {"last_token": [0, 1]}
        assert result["has_logits"] is False
        assert result["has_perplexity"] is True

    def test_intersection_of_layers(self):
        info1 = CachedPromptInfo(
            raw_layers=[0, 1, 2], pooled={}, has_logits=False,
            logits_top_k=None, has_perplexity=False, num_tokens=5,
        )
        info2 = CachedPromptInfo(
            raw_layers=[1, 2, 3], pooled={}, has_logits=False,
            logits_top_k=None, has_perplexity=False, num_tokens=5,
        )
        result = _compute_tensor_intersection([info1, info2])
        assert result["raw_layers"] == [1, 2]

    def test_intersection_of_pooled_strategies(self):
        info1 = CachedPromptInfo(
            raw_layers=[], pooled={"last_token": [0], "mean": [0]},
            has_logits=False, logits_top_k=None, has_perplexity=False,
            num_tokens=5,
        )
        info2 = CachedPromptInfo(
            raw_layers=[], pooled={"last_token": [0]},
            has_logits=False, logits_top_k=None, has_perplexity=False,
            num_tokens=5,
        )
        result = _compute_tensor_intersection([info1, info2])
        assert "last_token" in result["pooled"]
        assert "mean" not in result["pooled"]

    def test_logits_topk_consistent(self):
        info1 = CachedPromptInfo(
            raw_layers=[], pooled={}, has_logits=False,
            logits_top_k=100, has_perplexity=False, num_tokens=5,
        )
        info2 = CachedPromptInfo(
            raw_layers=[], pooled={}, has_logits=False,
            logits_top_k=100, has_perplexity=False, num_tokens=5,
        )
        result = _compute_tensor_intersection([info1, info2])
        assert result["has_logits"] is True
        assert result["logits_top_k"] == 100

    def test_logits_topk_inconsistent(self):
        info1 = CachedPromptInfo(
            raw_layers=[], pooled={}, has_logits=False,
            logits_top_k=100, has_perplexity=False, num_tokens=5,
        )
        info2 = CachedPromptInfo(
            raw_layers=[], pooled={}, has_logits=False,
            logits_top_k=50, has_perplexity=False, num_tokens=5,
        )
        result = _compute_tensor_intersection([info1, info2])
        # Inconsistent topk => no logits pushed
        assert result["has_logits"] is False

    def test_topk_preferred_over_full(self):
        """When both topk and full logits are available, prefer topk."""
        info = CachedPromptInfo(
            raw_layers=[], pooled={}, has_logits=True,
            logits_top_k=100, has_perplexity=False, num_tokens=5,
        )
        result = _compute_tensor_intersection([info, info])
        assert result["has_logits"] is True
        assert result["logits_top_k"] == 100  # topk preferred

    def test_perplexity_intersection(self):
        info1 = CachedPromptInfo(
            raw_layers=[], pooled={}, has_logits=False,
            logits_top_k=None, has_perplexity=True, num_tokens=5,
        )
        info2 = CachedPromptInfo(
            raw_layers=[], pooled={}, has_logits=False,
            logits_top_k=None, has_perplexity=False, num_tokens=5,
        )
        result = _compute_tensor_intersection([info1, info2])
        assert result["has_perplexity"] is False


class TestFilterTensorTypes:
    def test_no_filter_returns_all(self):
        available = {
            "raw_layers": [0, 1],
            "pooled": {"last_token": [0]},
            "has_logits": True,
            "logits_top_k": 100,
            "has_perplexity": True,
        }
        result = _filter_tensor_types(available, None)
        assert result == available

    def test_filter_specific_layer(self):
        available = {
            "raw_layers": [0, 1, 2],
            "pooled": {},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        result = _filter_tensor_types(available, ["hidden.layer_1"])
        assert result["raw_layers"] == [1]

    def test_filter_logits(self):
        available = {
            "raw_layers": [0],
            "pooled": {},
            "has_logits": True,
            "logits_top_k": 100,
            "has_perplexity": False,
        }
        result = _filter_tensor_types(available, ["logits_topk"])
        assert result["has_logits"] is True
        assert result["raw_layers"] == []


class TestDiscoverPrompts:
    def test_empty_prompts_raises(self, cache_dir):
        with pytest.raises(ValueError, match="No prompts"):
            _discover_prompts(TEST_MODEL, [])

    def test_all_missing_raises(self, cache_dir):
        with pytest.raises(ValueError, match="No prompts have cached data"):
            _discover_prompts(
                TEST_MODEL, ["missing1", "missing2"], skip_missing=True
            )

    def test_skip_missing_false_raises(self, cache_dir):
        with pytest.raises(FileNotFoundError):
            _discover_prompts(
                TEST_MODEL, ["missing"], skip_missing=False
            )

    def test_finds_cached_prompts(self, populated_cache):
        kept, infos = _discover_prompts(
            TEST_MODEL, populated_cache, skip_missing=True
        )
        assert len(kept) == 3
        assert len(infos) == 3
        assert all(isinstance(i, CachedPromptInfo) for i in infos)

    def test_partial_cache(self, populated_cache):
        prompts = populated_cache + ["uncached prompt"]
        kept, infos = _discover_prompts(
            TEST_MODEL, prompts, skip_missing=True
        )
        assert len(kept) == 3
        assert 3 not in kept  # uncached was skipped


class TestConsolidateAndShard:
    def test_basic_consolidation(self, populated_cache):
        tensor_types = {
            "raw_layers": [],
            "pooled": {"last_token": [0, 1]},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        shard_files, manifest_tensors, manifest_prompts, tokens = (
            _consolidate_and_shard(
                model_name=TEST_MODEL,
                prompts=populated_cache,
                kept_indices=[0, 1, 2],
                tensor_types=tensor_types,
                labels=None,
                shard_max_bytes=1_000_000_000,
            )
        )
        assert len(manifest_prompts) == 3
        assert "pooled.last_token.layer_0" in manifest_tensors
        assert "pooled.last_token.layer_1" in manifest_tensors
        assert tokens is None  # no tokenizer

        # Verify shard files exist
        for f in shard_files:
            assert f.exists()

        # Clean up
        if shard_files:
            shutil.rmtree(shard_files[0].parent, ignore_errors=True)

    def test_consolidation_with_labels(self, populated_cache):
        tensor_types = {
            "raw_layers": [],
            "pooled": {"last_token": [0]},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        labels = [1, 1, 0]
        _, _, manifest_prompts, _ = _consolidate_and_shard(
            model_name=TEST_MODEL,
            prompts=populated_cache,
            kept_indices=[0, 1, 2],
            tensor_types=tensor_types,
            labels=labels,
            shard_max_bytes=1_000_000_000,
        )
        assert manifest_prompts[0]["label"] == 1
        assert manifest_prompts[2]["label"] == 0

    def test_consolidation_with_pooled(self, populated_cache):
        tensor_types = {
            "raw_layers": [],
            "pooled": {"last_token": [0]},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        shard_files, manifest_tensors, manifest_prompts, _ = (
            _consolidate_and_shard(
                model_name=TEST_MODEL,
                prompts=populated_cache,
                kept_indices=[0, 1, 2],
                tensor_types=tensor_types,
                labels=None,
                shard_max_bytes=1_000_000_000,
            )
        )
        assert "pooled.last_token.layer_0" in manifest_tensors
        assert len(manifest_prompts) == 3

        # Clean up
        if shard_files:
            shutil.rmtree(shard_files[0].parent, ignore_errors=True)

    def test_small_shard_limit(self, populated_cache):
        """Test that small shard limits create multiple shards."""
        tensor_types = {
            "raw_layers": [],
            "pooled": {"last_token": [0]},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        shard_files, manifest_tensors, _, _ = _consolidate_and_shard(
            model_name=TEST_MODEL,
            prompts=populated_cache,
            kept_indices=[0, 1, 2],
            tensor_types=tensor_types,
            labels=None,
            shard_max_bytes=1,  # 1 byte — forces each prompt into its own shard
        )
        # Should have multiple shards
        shards = manifest_tensors["pooled.last_token.layer_0"]["shards"]
        assert len(shards) >= 2

        # Clean up
        if shard_files:
            shutil.rmtree(shard_files[0].parent, ignore_errors=True)


class TestBuildDatasetInfo:
    @patch("huggingface_hub.model_info", side_effect=Exception("no network"))
    def test_basic_structure(self, mock_model_info):
        info = _build_dataset_info(TEST_MODEL, 100)
        assert info["format_version"] == FORMAT_VERSION
        assert info["model"]["name"] == TEST_MODEL
        assert info["num_prompts"] == 100
        assert "lmprobe_version" in info["provenance"]
        assert "torch_version" in info["provenance"]


class TestBuildReadme:
    def test_readme_contains_key_sections(self):
        dataset_info = {
            "format_version": "1.0",
            "model": {"name": TEST_MODEL, "revision": "abc123"},
            "provenance": {
                "lmprobe_version": "0.7.3",
                "extraction_backend": "local",
                "created_at": "2026-01-01",
                "torch_version": "2.0",
                "transformers_version": "4.0",
            },
        }
        manifest_tensors = {
            "hidden.layer_0": {
                "type": "hidden",
                "layer": 0,
                "dim": 32,
                "pooling": None,
                "shards": [{"file": "hidden_layer_0_000.safetensors", "num_prompts": 3}],
            }
        }
        readme = _build_readme(
            TEST_MODEL, dataset_info, manifest_tensors, 3,
            "user/my-dataset", description="Test dataset",
        )
        assert TEST_MODEL in readme
        assert "Load with lmprobe" in readme
        assert "Load without lmprobe" in readme
        assert "Test dataset" in readme
        assert "user/my-dataset" in readme


class TestPushToBucket:
    @patch("lmprobe.bucket._check_bucket_deps")
    @patch("huggingface_hub.HfApi")
    def test_push_roundtrip(self, MockHfApi, mock_deps, populated_cache):
        """Test push creates correct files."""
        mock_api = MagicMock()
        MockHfApi.return_value = mock_api

        uploaded_files = {}

        def capture_upload(repo_id, folder_path, **kwargs):
            folder = Path(folder_path)
            for f in folder.iterdir():
                if f.suffix == ".json":
                    uploaded_files[f.name] = json.loads(f.read_text())
                else:
                    uploaded_files[f.name] = f.read_bytes()

        mock_api.upload_folder.side_effect = capture_upload

        url = push_to_bucket(
            bucket_id="user/test-dataset",
            model_name=TEST_MODEL,
            prompts=populated_cache,
            labels=[1, 1, 0],
            exist_ok=True,
        )

        assert "user/test-dataset" in url
        mock_api.create_repo.assert_called_once()
        mock_api.upload_folder.assert_called_once()

        # Check uploaded files
        assert "manifest.json" in uploaded_files
        assert "dataset_info.json" in uploaded_files
        assert "README.md" in uploaded_files

        manifest = uploaded_files["manifest.json"]
        assert len(manifest["prompts"]) == 3
        assert manifest["prompts"][0]["label"] == 1
        assert manifest["prompts"][2]["label"] == 0

    @patch("lmprobe.bucket._check_bucket_deps")
    def test_push_labels_length_mismatch(self, mock_deps, populated_cache):
        with pytest.raises(ValueError, match="labels length"):
            push_to_bucket(
                bucket_id="user/test",
                model_name=TEST_MODEL,
                prompts=populated_cache,
                labels=[1, 0],  # wrong length
            )

    @patch("lmprobe.bucket._check_bucket_deps")
    @patch("huggingface_hub.HfApi")
    def test_push_with_tensor_filter(self, MockHfApi, mock_deps, populated_cache):
        mock_api = MagicMock()
        MockHfApi.return_value = mock_api

        uploaded_files = {}

        def capture_upload(repo_id, folder_path, **kwargs):
            folder = Path(folder_path)
            for f in folder.iterdir():
                if f.suffix == ".json":
                    uploaded_files[f.name] = json.loads(f.read_text())

        mock_api.upload_folder.side_effect = capture_upload

        push_to_bucket(
            bucket_id="user/filtered",
            model_name=TEST_MODEL,
            prompts=populated_cache,
            tensors=["pooled.last_token.layer_0"],
            exist_ok=True,
        )

        manifest = uploaded_files["manifest.json"]
        assert "pooled.last_token.layer_0" in manifest["tensors"]
        assert "pooled.last_token.layer_1" not in manifest["tensors"]

    @patch("lmprobe.bucket._check_bucket_deps")
    def test_push_empty_prompts_raises(self, mock_deps, cache_dir):
        with pytest.raises(ValueError, match="No prompts"):
            push_to_bucket(
                bucket_id="user/empty",
                model_name=TEST_MODEL,
                prompts=[],
            )


class TestLoadFromBucket:
    def _setup_bucket_files(self, tmp_path):
        """Create minimal bucket files for testing load."""
        from safetensors.torch import save_file

        # Create a shard
        tensor = torch.randn(3, HIDDEN_DIM)
        shard_path = tmp_path / "hidden_layer_0_000.safetensors"
        save_file({"hidden.layer_0": tensor}, str(shard_path))

        # Create manifest
        manifest = {
            "tensors": {
                "hidden.layer_0": {
                    "type": "hidden",
                    "layer": 0,
                    "dim": HIDDEN_DIM,
                    "dtype": "float32",
                    "pooling": None,
                    "shards": [
                        {"file": "hidden_layer_0_000.safetensors", "num_prompts": 3}
                    ],
                }
            },
            "prompts": [
                {"index": 0, "text": "prompt 0", "label": None, "num_tokens": 5},
                {"index": 1, "text": "prompt 1", "label": None, "num_tokens": 5},
                {"index": 2, "text": "prompt 2", "label": None, "num_tokens": 5},
            ],
        }
        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest, f)

        dataset_info = {
            "format_version": "1.0",
            "model": {"name": TEST_MODEL, "revision": None},
            "num_prompts": 3,
            "provenance": {},
        }
        with open(tmp_path / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f)

        return tmp_path, tensor

    @patch("lmprobe.bucket._check_bucket_deps")
    def test_load_from_bucket(self, mock_deps, tmp_path):
        bucket_dir, expected_tensor = self._setup_bucket_files(tmp_path)

        def mock_download(repo_id, filename, **kwargs):
            return str(bucket_dir / filename)

        with patch("huggingface_hub.hf_hub_download", side_effect=mock_download):
            result, manifest = load_from_bucket("user/test-dataset")

        assert "hidden.layer_0" in result
        assert result["hidden.layer_0"].shape == (3, HIDDEN_DIM)
        assert torch.allclose(result["hidden.layer_0"], expected_tensor)

    @patch("lmprobe.bucket._check_bucket_deps")
    def test_load_selective(self, mock_deps, tmp_path):
        bucket_dir, _ = self._setup_bucket_files(tmp_path)

        def mock_download(repo_id, filename, **kwargs):
            return str(bucket_dir / filename)

        with patch("huggingface_hub.hf_hub_download", side_effect=mock_download):
            result, manifest = load_from_bucket(
                "user/test", tensors=["hidden.layer_0"]
            )

        assert "hidden.layer_0" in result

    @patch("lmprobe.bucket._check_bucket_deps")
    def test_version_mismatch_raises(self, mock_deps, tmp_path):
        bucket_dir, _ = self._setup_bucket_files(tmp_path)

        # Override version
        with open(bucket_dir / "dataset_info.json") as f:
            info = json.load(f)
        info["format_version"] = "2.0"
        with open(bucket_dir / "dataset_info.json", "w") as f:
            json.dump(info, f)

        def mock_download(repo_id, filename, **kwargs):
            return str(bucket_dir / filename)

        with patch("huggingface_hub.hf_hub_download", side_effect=mock_download):
            with pytest.raises(ValueError, match="Incompatible format version"):
                load_from_bucket("user/test")


class TestPullFromBucket:
    def _setup_bucket_files(self, tmp_path):
        """Create bucket files for pull testing."""
        from safetensors.torch import save_file

        tmp_path.mkdir(parents=True, exist_ok=True)
        prompts = ["prompt 0", "prompt 1", "prompt 2"]
        tensor = torch.randn(3, HIDDEN_DIM)
        shard_path = tmp_path / "hidden_layer_0_000.safetensors"
        save_file({"hidden.layer_0": tensor}, str(shard_path))

        manifest = {
            "tensors": {
                "hidden.layer_0": {
                    "type": "hidden",
                    "layer": 0,
                    "dim": HIDDEN_DIM,
                    "dtype": "float32",
                    "pooling": None,
                    "shards": [
                        {"file": "hidden_layer_0_000.safetensors", "num_prompts": 3}
                    ],
                }
            },
            "prompts": [
                {"index": i, "text": p, "label": None, "num_tokens": 5}
                for i, p in enumerate(prompts)
            ],
        }
        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest, f)

        dataset_info = {
            "format_version": "1.0",
            "model": {"name": TEST_MODEL, "revision": None},
            "num_prompts": 3,
            "provenance": {},
        }
        with open(tmp_path / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f)

        return tmp_path, prompts

    @patch("lmprobe.bucket._check_bucket_deps")
    def test_pull_populates_cache(self, mock_deps, tmp_path, cache_dir):
        bucket_dir, prompts = self._setup_bucket_files(tmp_path / "bucket")

        def mock_download(repo_id, filename, **kwargs):
            return str(bucket_dir / filename)

        with patch("huggingface_hub.hf_hub_download", side_effect=mock_download):
            count = pull_from_bucket("user/test")

        assert count == 3

        # Verify prompts are now cached
        for prompt in prompts:
            info = discover_cached(TEST_MODEL, prompt)
            assert info is not None

    @patch("lmprobe.bucket._check_bucket_deps")
    def test_pull_dedup(self, mock_deps, tmp_path, cache_dir):
        bucket_dir, prompts = self._setup_bucket_files(tmp_path / "bucket")

        # Pre-cache one prompt
        acts = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        mask = torch.ones(1, SEQ_LEN)
        save_prompt_activations(TEST_MODEL, "prompt 0", [0], acts, mask)

        def mock_download(repo_id, filename, **kwargs):
            return str(bucket_dir / filename)

        with patch("huggingface_hub.hf_hub_download", side_effect=mock_download):
            count = pull_from_bucket("user/test", overwrite=False)

        # Only 2 new prompts should be unpacked (prompt 0 was already cached)
        assert count == 2

    @patch("lmprobe.bucket._check_bucket_deps")
    def test_pull_selective_tensors(self, mock_deps, tmp_path, cache_dir):
        bucket_dir, prompts = self._setup_bucket_files(tmp_path / "bucket")

        def mock_download(repo_id, filename, **kwargs):
            return str(bucket_dir / filename)

        with patch("huggingface_hub.hf_hub_download", side_effect=mock_download):
            count = pull_from_bucket(
                "user/test", tensors=["hidden.layer_0"]
            )

        assert count == 3


class TestRoundtrip:
    """Push from cache → pull into fresh cache → verify data matches."""

    def test_push_pull_roundtrip_pooled_activations(self, tmp_path, monkeypatch):
        """Roundtrip: pooled activations survive push → pull with exact values."""
        # --- Phase 1: populate source cache ---
        src_cache = tmp_path / "src_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(src_cache))
        cache_mod._backend = None

        prompts = [
            "Who wants to go for a walk?",
            "Fetch the ball!",
            "Purring and scratching",
        ]
        labels = [1, 1, 0]

        # Save known pooled tensors so we can compare after roundtrip
        original_pooled: dict[str, dict[int, torch.Tensor]] = {}
        for prompt in prompts:
            original_pooled[prompt] = {}
            for layer in [0, 1]:
                pooled = torch.randn(1, HIDDEN_DIM)
                save_prompt_pooled_activations(
                    TEST_MODEL, prompt, [layer], pooled, "last_token"
                )
                original_pooled[prompt][layer] = pooled

        # --- Phase 2: push (capture the uploaded folder) ---
        bucket_dir = tmp_path / "bucket"
        bucket_dir.mkdir()

        mock_api = MagicMock()

        def capture_upload(repo_id, folder_path, **kwargs):
            src = Path(folder_path)
            for f in src.iterdir():
                shutil.copy2(str(f), str(bucket_dir / f.name))

        mock_api.upload_folder.side_effect = capture_upload

        with (
            patch("lmprobe.bucket._check_bucket_deps"),
            patch("huggingface_hub.HfApi", return_value=mock_api),
        ):
            push_to_bucket(
                bucket_id="user/roundtrip-test",
                model_name=TEST_MODEL,
                prompts=prompts,
                labels=labels,
                exist_ok=True,
            )

        # Verify bucket files were created
        assert (bucket_dir / "manifest.json").exists()
        assert (bucket_dir / "dataset_info.json").exists()

        with open(bucket_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert len(manifest["prompts"]) == 3
        assert manifest["prompts"][0]["label"] == 1
        assert manifest["prompts"][2]["label"] == 0
        assert "pooled.last_token.layer_0" in manifest["tensors"]
        assert "pooled.last_token.layer_1" in manifest["tensors"]

        # --- Phase 3: pull into a fresh cache ---
        dst_cache = tmp_path / "dst_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(dst_cache))
        cache_mod._backend = None

        def mock_download(repo_id, filename, **kwargs):
            return str(bucket_dir / filename)

        with (
            patch("lmprobe.bucket._check_bucket_deps"),
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=mock_download,
            ),
        ):
            count = pull_from_bucket("user/roundtrip-test")

        assert count == 3

        # --- Phase 4: verify pulled data matches originals ---
        for prompt in prompts:
            info = discover_cached(TEST_MODEL, prompt)
            assert info is not None, f"Prompt not cached after pull: {prompt}"
            assert "last_token" in info.pooled

            # Load from the fresh cache and compare values
            for layer in [0, 1]:
                pulled = load_prompt_pooled_activations(
                    TEST_MODEL, prompt, [layer], "last_token"
                )
                orig = original_pooled[prompt][layer]

                assert pulled.shape == orig.shape, (
                    f"Shape mismatch for {prompt!r} layer {layer}"
                )
                assert torch.allclose(
                    pulled, orig, atol=1e-6
                ), f"Value mismatch for {prompt!r} layer {layer}"

    def test_unpooled_raw_activations_rejected(self, tmp_path, monkeypatch):
        """Pushing unpooled raw activations (seq_len > 1) raises ValueError."""
        src_cache = tmp_path / "src_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(src_cache))
        cache_mod._backend = None

        prompt = "test prompt"
        acts = torch.randn(1, SEQ_LEN, HIDDEN_DIM)  # seq_len=5 > 1
        mask = torch.ones(1, SEQ_LEN)
        save_prompt_activations(TEST_MODEL, prompt, [0], acts, mask)

        mock_api = MagicMock()
        with (
            patch("lmprobe.bucket._check_bucket_deps"),
            patch("huggingface_hub.HfApi", return_value=mock_api),
        ):
            with pytest.raises(ValueError, match="Unpooled raw activations"):
                push_to_bucket(
                    bucket_id="user/should-fail",
                    model_name=TEST_MODEL,
                    prompts=[prompt],
                    tensors=["hidden.layer_0"],
                    exist_ok=True,
                )

    def test_push_pull_roundtrip_pooled(self, tmp_path, monkeypatch):
        """Roundtrip: pooled activations survive push → pull."""
        src_cache = tmp_path / "src_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(src_cache))
        cache_mod._backend = None

        prompts = ["prompt A", "prompt B"]
        original_pooled: dict[str, torch.Tensor] = {}
        for prompt in prompts:
            pooled = torch.randn(1, HIDDEN_DIM)
            save_prompt_pooled_activations(
                TEST_MODEL, prompt, [5], pooled, "last_token"
            )
            original_pooled[prompt] = pooled

        # Push
        bucket_dir = tmp_path / "bucket"
        bucket_dir.mkdir()
        mock_api = MagicMock()

        def capture_upload(repo_id, folder_path, **kwargs):
            for f in Path(folder_path).iterdir():
                shutil.copy2(str(f), str(bucket_dir / f.name))

        mock_api.upload_folder.side_effect = capture_upload

        with (
            patch("lmprobe.bucket._check_bucket_deps"),
            patch("huggingface_hub.HfApi", return_value=mock_api),
        ):
            push_to_bucket(
                bucket_id="user/pooled-test",
                model_name=TEST_MODEL,
                prompts=prompts,
                exist_ok=True,
            )

        # Pull into fresh cache
        dst_cache = tmp_path / "dst_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(dst_cache))
        cache_mod._backend = None

        def mock_download(repo_id, filename, **kwargs):
            return str(bucket_dir / filename)

        with (
            patch("lmprobe.bucket._check_bucket_deps"),
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=mock_download,
            ),
        ):
            count = pull_from_bucket("user/pooled-test")

        assert count == 2

        # Verify
        for prompt in prompts:
            info = discover_cached(TEST_MODEL, prompt)
            assert info is not None
            assert "last_token" in info.pooled

            pulled = load_prompt_pooled_activations(
                TEST_MODEL, prompt, [5], "last_token"
            )
            assert torch.allclose(
                pulled, original_pooled[prompt], atol=1e-6
            ), f"Pooled value mismatch for {prompt!r}"

    def test_push_load_roundtrip(self, tmp_path, monkeypatch):
        """Push → load_from_bucket returns concatenated tensors."""
        src_cache = tmp_path / "src_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(src_cache))
        cache_mod._backend = None

        prompts = ["alpha", "beta", "gamma"]
        all_pooled = []
        for prompt in prompts:
            pooled = torch.randn(1, HIDDEN_DIM)
            save_prompt_pooled_activations(
                TEST_MODEL, prompt, [0], pooled, "last_token"
            )
            all_pooled.append(pooled)

        # Push
        bucket_dir = tmp_path / "bucket"
        bucket_dir.mkdir()
        mock_api = MagicMock()

        def capture_upload(repo_id, folder_path, **kwargs):
            for f in Path(folder_path).iterdir():
                shutil.copy2(str(f), str(bucket_dir / f.name))

        mock_api.upload_folder.side_effect = capture_upload

        with (
            patch("lmprobe.bucket._check_bucket_deps"),
            patch("huggingface_hub.HfApi", return_value=mock_api),
        ):
            push_to_bucket(
                bucket_id="user/load-test",
                model_name=TEST_MODEL,
                prompts=prompts,
                exist_ok=True,
            )

        # Load directly (no cache interaction)
        def mock_download(repo_id, filename, **kwargs):
            return str(bucket_dir / filename)

        with (
            patch("lmprobe.bucket._check_bucket_deps"),
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=mock_download,
            ),
        ):
            result, manifest = load_from_bucket("user/load-test")

        assert "pooled.last_token.layer_0" in result
        loaded = result["pooled.last_token.layer_0"]
        expected = torch.cat(all_pooled, dim=0)
        assert loaded.shape == expected.shape
        assert torch.allclose(loaded, expected, atol=1e-6)
