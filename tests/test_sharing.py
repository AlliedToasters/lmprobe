"""Tests for sharing.py — two-tier Parquet index + safetensors tensor store."""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import lmprobe.cache as cache_mod
from lmprobe.cache import (
    CachedPromptInfo,
    discover_cached,
    load_prompt_activations,
    load_prompt_pooled_activations,
    save_prompt_activations,
    save_prompt_logits,
    save_prompt_pooled_activations,
)
from lmprobe.sharing import (
    FORMAT_VERSION,
    INFO_FILENAME,
    PARQUET_PATH,
    _build_lmprobe_info,
    _build_readme,
    _compute_shard_boundaries_variable,
    _compute_tensor_intersection,
    _consolidate_and_shard,
    _discover_prompts,
    _filter_tensor_types,
    _write_parquet_index,
    load_activation_dataset,
    pull_dataset,
    push_dataset,
)

_has_pyarrow = True
try:
    import pyarrow  # noqa: F401
except ImportError:
    _has_pyarrow = False

requires_pyarrow = pytest.mark.skipif(
    not _has_pyarrow, reason="pyarrow not installed"
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
        for layer in [0, 1]:
            pooled = torch.randn(1, HIDDEN_DIM)
            save_prompt_pooled_activations(
                TEST_MODEL, prompt, [layer], pooled, "last_token"
            )
    return prompts


@pytest.fixture
def populated_raw_cache(cache_dir):
    """Populate cache with raw (full-sequence) activations.

    Variable seq_lens: 3, 5, 4 tokens.
    """
    prompts = [
        "Who wants to go for a walk?",
        "Fetch the ball!",
        "Purring and scratching",
    ]
    seq_lens = [3, 5, 4]
    layers = [0, 1]
    original_raw = {}

    for prompt, sl in zip(prompts, seq_lens):
        # activations: (1, seq_len, hidden_dim * n_layers)
        act = torch.randn(1, sl, HIDDEN_DIM * len(layers))
        mask = torch.ones(1, sl, dtype=torch.long)
        save_prompt_activations(TEST_MODEL, prompt, layers, act, mask)
        original_raw[prompt] = (act, mask, sl)

    return prompts, seq_lens, layers, original_raw


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
        assert result["has_logits"] is False

    def test_topk_preferred_over_full(self):
        info = CachedPromptInfo(
            raw_layers=[], pooled={}, has_logits=True,
            logits_top_k=100, has_perplexity=False, num_tokens=5,
        )
        result = _compute_tensor_intersection([info, info])
        assert result["has_logits"] is True
        assert result["logits_top_k"] == 100

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

    def test_filter_hidden_layers(self):
        available = {
            "raw_layers": [0, 1, 2],
            "pooled": {"last_token": [0, 1]},
            "has_logits": True,
            "logits_top_k": 100,
            "has_perplexity": False,
        }
        result = _filter_tensor_types(available, ["hidden_layers"])
        assert result["pooled"] == {"last_token": [0, 1]}
        assert result["raw_layers"] == [0, 1, 2]
        assert result["has_logits"] is False

    def test_filter_logits(self):
        available = {
            "raw_layers": [0],
            "pooled": {"last_token": [0]},
            "has_logits": True,
            "logits_top_k": 100,
            "has_perplexity": False,
        }
        result = _filter_tensor_types(available, ["logits_topk"])
        assert result["has_logits"] is True
        assert result["pooled"] == {}
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
        assert 3 not in kept


class TestConsolidateAndShard:
    def test_basic_consolidation(self, populated_cache):
        tensor_types = {
            "raw_layers": [],
            "pooled": {"last_token": [0, 1]},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        tmpdir, tensor_descriptors, prompt_metadata = (
            _consolidate_and_shard(
                model_name=TEST_MODEL,
                prompts=populated_cache,
                kept_indices=[0, 1, 2],
                tensor_types=tensor_types,
                labels=None,
                shard_max_bytes=1_000_000_000,
                repo_id="user/test",
            )
        )
        assert len(prompt_metadata) == 3
        assert "hidden_layers" in tensor_descriptors
        desc = tensor_descriptors["hidden_layers"]
        assert desc["layers"] == [0, 1]
        assert desc["type"] == "hidden"

        # Verify shard files exist
        for shard in desc["shards"]:
            assert (tmpdir / shard["file"]).exists()

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_consolidation_with_labels(self, populated_cache):
        tensor_types = {
            "raw_layers": [],
            "pooled": {"last_token": [0]},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        labels = [1, 1, 0]
        _, _, prompt_metadata = _consolidate_and_shard(
            model_name=TEST_MODEL,
            prompts=populated_cache,
            kept_indices=[0, 1, 2],
            tensor_types=tensor_types,
            labels=labels,
            shard_max_bytes=1_000_000_000,
            repo_id="user/test",
        )
        # Labels are present (order may be shuffled)
        label_set = {p["label"] for p in prompt_metadata}
        assert 1 in label_set
        assert 0 in label_set

    def test_small_shard_limit(self, populated_cache):
        """Test that small shard limits create multiple shards."""
        tensor_types = {
            "raw_layers": [],
            "pooled": {"last_token": [0]},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        tmpdir, tensor_descriptors, _ = _consolidate_and_shard(
            model_name=TEST_MODEL,
            prompts=populated_cache,
            kept_indices=[0, 1, 2],
            tensor_types=tensor_types,
            labels=None,
            shard_max_bytes=1,  # 1 byte — forces each prompt into own shard
            repo_id="user/test",
        )
        shards = tensor_descriptors["hidden_layers"]["shards"]
        assert len(shards) >= 2

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_co_located_layers(self, populated_cache):
        """Verify multiple layers end up in the same shard file."""
        tensor_types = {
            "raw_layers": [],
            "pooled": {"last_token": [0, 1]},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        tmpdir, tensor_descriptors, _ = _consolidate_and_shard(
            model_name=TEST_MODEL,
            prompts=populated_cache,
            kept_indices=[0, 1, 2],
            tensor_types=tensor_types,
            labels=None,
            shard_max_bytes=1_000_000_000,
            repo_id="user/test",
        )

        from safetensors import safe_open

        shard_file = tmpdir / tensor_descriptors["hidden_layers"]["shards"][0]["file"]
        with safe_open(str(shard_file), framework="pt") as f:
            keys = list(f.keys())

        assert "hidden.layer_0" in keys
        assert "hidden.layer_1" in keys

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_shard_metadata(self, populated_cache):
        """Verify prompt_metadata has shard_index and row_offset."""
        tensor_types = {
            "raw_layers": [],
            "pooled": {"last_token": [0]},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        _, _, prompt_metadata = _consolidate_and_shard(
            model_name=TEST_MODEL,
            prompts=populated_cache,
            kept_indices=[0, 1, 2],
            tensor_types=tensor_types,
            labels=None,
            shard_max_bytes=1_000_000_000,
            repo_id="user/test",
        )

        for pm in prompt_metadata:
            assert "shard_index" in pm
            assert "row_offset" in pm


@requires_pyarrow
class TestParquetIndex:
    def test_write_and_read(self, tmp_path):
        import pyarrow.parquet as pq

        (tmp_path / "index").mkdir()
        prompt_metadata = [
            {"text": "hello", "label": 1, "num_tokens": 3,
             "shard_index": 0, "row_offset": 0},
            {"text": "world", "label": 0, "num_tokens": 4,
             "shard_index": 0, "row_offset": 1},
        ]
        _write_parquet_index(tmp_path, prompt_metadata)

        table = pq.read_table(str(tmp_path / PARQUET_PATH))

        assert table.num_rows == 2
        assert table.column_names == [
            "text", "label", "num_tokens", "shard_index", "row_offset",
        ]
        assert table.column("text").to_pylist() == ["hello", "world"]
        assert table.column("label").to_pylist() == [1, 0]
        assert table.column("shard_index").to_pylist() == [0, 0]
        assert table.column("row_offset").to_pylist() == [0, 1]

    def test_string_labels(self, tmp_path):
        import pyarrow.parquet as pq

        (tmp_path / "index").mkdir()
        prompt_metadata = [
            {"text": "a", "label": "positive", "num_tokens": 1,
             "shard_index": 0, "row_offset": 0},
            {"text": "b", "label": "negative", "num_tokens": 1,
             "shard_index": 0, "row_offset": 1},
        ]
        _write_parquet_index(tmp_path, prompt_metadata)

        table = pq.read_table(str(tmp_path / PARQUET_PATH))
        assert table.column("label").to_pylist() == ["positive", "negative"]


class TestBuildLmprobeInfo:
    @patch("huggingface_hub.model_info", side_effect=Exception("no network"))
    def test_basic_structure(self, mock_model_info):
        info = _build_lmprobe_info(TEST_MODEL, 100, {"hidden_layers": {}})
        assert info["format_version"] == FORMAT_VERSION
        assert info["model"]["name"] == TEST_MODEL
        assert info["num_prompts"] == 100
        assert info["prompt_ordering"] == "random"
        assert "tensors" in info
        assert "lmprobe_version" in info["provenance"]
        assert "torch_version" in info["provenance"]


class TestBuildReadme:
    def test_readme_has_yaml_frontmatter(self):
        lmprobe_info = {
            "format_version": "1.0",
            "model": {"name": TEST_MODEL, "revision": "abc123"},
            "tensors": {},
            "provenance": {
                "lmprobe_version": "0.7.3",
                "extraction_backend": "local",
                "created_at": "2026-01-01",
                "torch_version": "2.0",
                "transformers_version": "4.0",
            },
        }
        readme = _build_readme(
            TEST_MODEL, lmprobe_info, 0, "user/test",
        )
        assert readme.startswith("---\n")
        assert "tags:" in readme
        assert "- lmprobe" in readme
        assert "- activations" in readme
        assert "- interpretability" in readme
        assert "task_categories:" in readme
        assert "- feature-extraction" in readme
        assert "license: cc-by-4.0" in readme

    def test_readme_custom_license(self):
        lmprobe_info = {
            "format_version": "1.0",
            "model": {"name": TEST_MODEL, "revision": "abc123"},
            "tensors": {},
            "provenance": {},
        }
        readme = _build_readme(
            TEST_MODEL, lmprobe_info, 0, "user/test",
            license="apache-2.0",
        )
        assert "license: apache-2.0" in readme

    def test_readme_uses_new_api_names(self):
        lmprobe_info = {
            "format_version": "1.0",
            "model": {"name": TEST_MODEL, "revision": "abc123"},
            "tensors": {},
            "provenance": {},
        }
        readme = _build_readme(
            TEST_MODEL, lmprobe_info, 0, "user/test",
        )
        assert "pull_dataset" in readme
        assert "load_activation_dataset" in readme
        # Old names should not appear
        assert "push_to_bucket" not in readme
        assert "pull_from_bucket" not in readme
        assert "load_from_bucket" not in readme
        assert "load_activations" not in readme

    def test_readme_contains_standalone_loading(self):
        lmprobe_info = {
            "format_version": "1.0",
            "model": {"name": TEST_MODEL, "revision": "abc123"},
            "tensors": {
                "hidden_layers": {
                    "type": "hidden",
                    "layers": [0, 1],
                    "dim": 32,
                    "pooling": "last_token",
                    "row_bytes": 256,
                    "shards": [
                        {"file": "tensors/hidden_layers_000.safetensors",
                         "num_prompts": 3},
                    ],
                },
            },
            "provenance": {},
        }
        readme = _build_readme(
            TEST_MODEL, lmprobe_info, 3, "user/test",
        )
        assert "Load without lmprobe" in readme
        assert "pyarrow.parquet" in readme
        assert "safe_open" in readme
        assert "row_offset" in readme
        assert "load_dataset" in readme

    def test_readme_description(self):
        lmprobe_info = {
            "format_version": "1.0",
            "model": {"name": TEST_MODEL, "revision": "abc123"},
            "tensors": {},
            "provenance": {},
        }
        readme = _build_readme(
            TEST_MODEL, lmprobe_info, 0, "user/test",
            description="Test dataset",
        )
        assert "Test dataset" in readme


@requires_pyarrow
class TestPushDataset:
    @patch("lmprobe.sharing._check_hub_deps")
    @patch("lmprobe.sharing._check_pyarrow")
    @patch("huggingface_hub.HfApi")
    def test_push_creates_correct_files(
        self, MockHfApi, mock_pyarrow, mock_deps, populated_cache,
    ):
        mock_api = MagicMock()
        MockHfApi.return_value = mock_api

        uploaded_files = {}

        def capture_upload(repo_id, folder_path, **kwargs):
            folder = Path(folder_path)
            for f in folder.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(folder)
                    if f.suffix == ".json":
                        uploaded_files[str(rel)] = json.loads(f.read_text())
                    elif f.suffix == ".md":
                        uploaded_files[str(rel)] = f.read_text()
                    else:
                        uploaded_files[str(rel)] = f.read_bytes()

        mock_api.upload_folder.side_effect = capture_upload

        url = push_dataset(
            repo_id="user/test-dataset",
            model_name=TEST_MODEL,
            prompts=populated_cache,
            labels=[1, 1, 0],
            exist_ok=True,
        )

        assert "user/test-dataset" in url
        mock_api.create_repo.assert_called_once()
        mock_api.upload_folder.assert_called_once()

        # Check file structure
        assert INFO_FILENAME in uploaded_files
        assert PARQUET_PATH in uploaded_files
        assert "README.md" in uploaded_files

        info = uploaded_files[INFO_FILENAME]
        assert info["num_prompts"] == 3
        assert "hidden_layers" in info["tensors"]
        assert info["prompt_ordering"] == "random"

    @patch("lmprobe.sharing._check_hub_deps")
    @patch("lmprobe.sharing._check_pyarrow")
    @patch("huggingface_hub.HfApi")
    def test_push_uses_upload_large_folder_for_big_datasets(
        self, MockHfApi, mock_pyarrow, mock_deps, populated_cache,
    ):
        """upload_large_folder is used when total size exceeds 1 GB."""
        mock_api = MagicMock()
        MockHfApi.return_value = mock_api

        # Make _folder_size return > 1 GB to trigger upload_large_folder
        with patch("lmprobe.sharing._folder_size", return_value=2_000_000_000):
            push_dataset(
                repo_id="user/large-dataset",
                model_name=TEST_MODEL,
                prompts=populated_cache,
                labels=[1, 1, 0],
                exist_ok=True,
            )

        mock_api.upload_large_folder.assert_called_once()
        mock_api.upload_folder.assert_not_called()

    @patch("lmprobe.sharing._check_hub_deps")
    @patch("lmprobe.sharing._check_pyarrow")
    @patch("huggingface_hub.HfApi")
    def test_push_uses_upload_folder_for_small_datasets(
        self, MockHfApi, mock_pyarrow, mock_deps, populated_cache,
    ):
        """upload_folder is used when total size is under 1 GB."""
        mock_api = MagicMock()
        MockHfApi.return_value = mock_api

        push_dataset(
            repo_id="user/small-dataset",
            model_name=TEST_MODEL,
            prompts=populated_cache,
            labels=[1, 1, 0],
            exist_ok=True,
        )

        mock_api.upload_folder.assert_called_once()
        mock_api.upload_large_folder.assert_not_called()

    @patch("lmprobe.sharing._check_hub_deps")
    @patch("lmprobe.sharing._check_pyarrow")
    def test_push_labels_length_mismatch(
        self, mock_pyarrow, mock_deps, populated_cache,
    ):
        with pytest.raises(ValueError, match="labels length"):
            push_dataset(
                repo_id="user/test",
                model_name=TEST_MODEL,
                prompts=populated_cache,
                labels=[1, 0],
            )

    @patch("lmprobe.sharing._check_hub_deps")
    @patch("lmprobe.sharing._check_pyarrow")
    def test_push_empty_prompts_raises(
        self, mock_pyarrow, mock_deps, cache_dir,
    ):
        with pytest.raises(ValueError, match="No prompts"):
            push_dataset(
                repo_id="user/empty",
                model_name=TEST_MODEL,
                prompts=[],
            )


class TestLoadActivationDataset:
    def _setup_remote_files(self, tmp_path):
        """Create remote files in the new format."""
        from safetensors.torch import save_file

        (tmp_path / "tensors").mkdir(parents=True)
        (tmp_path / "index").mkdir(parents=True)

        tensor = torch.randn(3, HIDDEN_DIM)
        save_file(
            {"hidden.layer_0": tensor},
            str(tmp_path / "tensors" / "hidden_layers_000.safetensors"),
        )

        lmprobe_info = {
            "format_version": "1.0",
            "model": {"name": TEST_MODEL, "revision": None},
            "num_prompts": 3,
            "prompt_ordering": "random",
            "tensors": {
                "hidden_layers": {
                    "type": "hidden",
                    "layers": [0],
                    "dim": HIDDEN_DIM,
                    "dtype": "float32",
                    "pooling": "last_token",
                    "row_bytes": HIDDEN_DIM * 4,
                    "shards": [
                        {
                            "file": "tensors/hidden_layers_000.safetensors",
                            "num_prompts": 3,
                        }
                    ],
                }
            },
            "provenance": {},
        }
        with open(tmp_path / INFO_FILENAME, "w") as f:
            json.dump(lmprobe_info, f)

        return tmp_path, tensor

    @patch("lmprobe.sharing._check_hub_deps")
    def test_load_activation_dataset(self, mock_deps, tmp_path):
        remote_dir, expected_tensor = self._setup_remote_files(tmp_path)

        def mock_download(repo_id, filename, **kwargs):
            return str(remote_dir / filename)

        with patch(
            "huggingface_hub.hf_hub_download", side_effect=mock_download,
        ):
            result, info = load_activation_dataset("user/test-dataset")

        assert "hidden.layer_0" in result
        assert result["hidden.layer_0"].shape == (3, HIDDEN_DIM)
        assert torch.allclose(result["hidden.layer_0"], expected_tensor)

    @patch("lmprobe.sharing._check_hub_deps")
    def test_load_selective(self, mock_deps, tmp_path):
        remote_dir, _ = self._setup_remote_files(tmp_path)

        def mock_download(repo_id, filename, **kwargs):
            return str(remote_dir / filename)

        with patch(
            "huggingface_hub.hf_hub_download", side_effect=mock_download,
        ):
            result, info = load_activation_dataset(
                "user/test", tensors=["hidden_layers"]
            )

        assert "hidden.layer_0" in result

    @patch("lmprobe.sharing._check_hub_deps")
    def test_version_mismatch_raises(self, mock_deps, tmp_path):
        remote_dir, _ = self._setup_remote_files(tmp_path)

        with open(remote_dir / INFO_FILENAME) as f:
            info = json.load(f)
        info["format_version"] = "2.0"
        with open(remote_dir / INFO_FILENAME, "w") as f:
            json.dump(info, f)

        def mock_download(repo_id, filename, **kwargs):
            return str(remote_dir / filename)

        with patch(
            "huggingface_hub.hf_hub_download", side_effect=mock_download,
        ):
            with pytest.raises(
                ValueError, match="Incompatible format version",
            ):
                load_activation_dataset("user/test")


@requires_pyarrow
class TestPullDataset:
    def _setup_remote_files(self, tmp_path):
        """Create remote files in the new format."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from safetensors.torch import save_file

        (tmp_path / "tensors").mkdir(parents=True)
        (tmp_path / "index").mkdir(parents=True)

        prompts = ["prompt 0", "prompt 1", "prompt 2"]
        tensor = torch.randn(3, HIDDEN_DIM)
        save_file(
            {"hidden.layer_0": tensor},
            str(tmp_path / "tensors" / "hidden_layers_000.safetensors"),
        )

        # Write Parquet index
        table = pa.table({
            "text": pa.array(prompts, type=pa.string()),
            "label": pa.array([None, None, None], type=pa.int32()),
            "num_tokens": pa.array([5, 5, 5], type=pa.int32()),
            "shard_index": pa.array([0, 0, 0], type=pa.int32()),
            "row_offset": pa.array([0, 1, 2], type=pa.int32()),
        })
        pq.write_table(table, str(tmp_path / PARQUET_PATH))

        lmprobe_info = {
            "format_version": "1.0",
            "model": {"name": TEST_MODEL, "revision": None},
            "num_prompts": 3,
            "prompt_ordering": "random",
            "tensors": {
                "hidden_layers": {
                    "type": "hidden",
                    "layers": [0],
                    "dim": HIDDEN_DIM,
                    "dtype": "float32",
                    "pooling": "last_token",
                    "row_bytes": HIDDEN_DIM * 4,
                    "shards": [
                        {
                            "file": "tensors/hidden_layers_000.safetensors",
                            "num_prompts": 3,
                        }
                    ],
                }
            },
            "provenance": {},
        }
        with open(tmp_path / INFO_FILENAME, "w") as f:
            json.dump(lmprobe_info, f)

        return tmp_path, prompts

    @patch("lmprobe.sharing._check_hub_deps")
    def test_pull_populates_cache(self, mock_deps, tmp_path, cache_dir):
        remote_dir, prompts = self._setup_remote_files(tmp_path / "remote")

        def mock_download(repo_id, filename, **kwargs):
            return str(remote_dir / filename)

        with patch(
            "huggingface_hub.hf_hub_download", side_effect=mock_download,
        ):
            count = pull_dataset("user/test")

        assert count == 3

        for prompt in prompts:
            info = discover_cached(TEST_MODEL, prompt)
            assert info is not None

    @patch("lmprobe.sharing._check_hub_deps")
    def test_pull_dedup(self, mock_deps, tmp_path, cache_dir):
        remote_dir, prompts = self._setup_remote_files(tmp_path / "remote")

        # Pre-cache one prompt
        pooled = torch.randn(1, HIDDEN_DIM)
        save_prompt_pooled_activations(
            TEST_MODEL, "prompt 0", [0], pooled, "last_token"
        )

        def mock_download(repo_id, filename, **kwargs):
            return str(remote_dir / filename)

        with patch(
            "huggingface_hub.hf_hub_download", side_effect=mock_download,
        ):
            count = pull_dataset("user/test", overwrite=False)

        assert count == 2


@requires_pyarrow
class TestRoundtrip:
    """Push from cache -> pull into fresh cache -> verify data matches."""

    def test_push_pull_roundtrip_pooled_activations(
        self, tmp_path, monkeypatch,
    ):
        """Roundtrip: pooled activations survive push -> pull."""
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
        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()

        mock_api = MagicMock()

        def capture_upload(repo_id, folder_path, **kwargs):
            src = Path(folder_path)
            for f in src.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(src)
                    dest = remote_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(dest))

        mock_api.upload_folder.side_effect = capture_upload

        with (
            patch("lmprobe.sharing._check_hub_deps"),
            patch("huggingface_hub.HfApi", return_value=mock_api),
        ):
            push_dataset(
                repo_id="user/roundtrip-test",
                model_name=TEST_MODEL,
                prompts=prompts,
                labels=labels,
                exist_ok=True,
            )

        # Verify remote files
        assert (remote_dir / INFO_FILENAME).exists()
        assert (remote_dir / PARQUET_PATH).exists()

        with open(remote_dir / INFO_FILENAME) as f:
            info = json.load(f)
        assert info["num_prompts"] == 3
        assert "hidden_layers" in info["tensors"]

        # --- Phase 3: pull into a fresh cache ---
        dst_cache = tmp_path / "dst_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(dst_cache))
        cache_mod._backend = None

        def mock_download(repo_id, filename, **kwargs):
            return str(remote_dir / filename)

        with (
            patch("lmprobe.sharing._check_hub_deps"),
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=mock_download,
            ),
        ):
            count = pull_dataset("user/roundtrip-test")

        assert count == 3

        # --- Phase 4: verify pulled data matches originals ---
        for prompt in prompts:
            info = discover_cached(TEST_MODEL, prompt)
            assert info is not None, (
                f"Prompt not cached after pull: {prompt}"
            )
            assert "last_token" in info.pooled

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

    def test_push_load_roundtrip(self, tmp_path, monkeypatch):
        """Push -> load_activation_dataset returns concatenated tensors."""
        src_cache = tmp_path / "src_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(src_cache))
        cache_mod._backend = None

        prompts = ["alpha", "beta", "gamma"]
        all_pooled_layer0 = {}
        for prompt in prompts:
            pooled = torch.randn(1, HIDDEN_DIM)
            save_prompt_pooled_activations(
                TEST_MODEL, prompt, [0], pooled, "last_token"
            )
            all_pooled_layer0[prompt] = pooled

        # Push
        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        mock_api = MagicMock()

        def capture_upload(repo_id, folder_path, **kwargs):
            for f in Path(folder_path).rglob("*"):
                if f.is_file():
                    rel = f.relative_to(folder_path)
                    dest = remote_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(dest))

        mock_api.upload_folder.side_effect = capture_upload

        with (
            patch("lmprobe.sharing._check_hub_deps"),
            patch("huggingface_hub.HfApi", return_value=mock_api),
        ):
            push_dataset(
                repo_id="user/load-test",
                model_name=TEST_MODEL,
                prompts=prompts,
                exist_ok=True,
            )

        # Load directly
        def mock_download(repo_id, filename, **kwargs):
            return str(remote_dir / filename)

        with (
            patch("lmprobe.sharing._check_hub_deps"),
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=mock_download,
            ),
        ):
            result, info = load_activation_dataset("user/load-test")

        assert "hidden.layer_0" in result
        loaded = result["hidden.layer_0"]
        assert loaded.shape == (3, HIDDEN_DIM)

        # Verify all values are present (order may differ due to shuffle)
        for prompt in prompts:
            orig = all_pooled_layer0[prompt].squeeze(0)
            # Find this vector in loaded
            found = False
            for row_idx in range(loaded.shape[0]):
                if torch.allclose(loaded[row_idx], orig, atol=1e-6):
                    found = True
                    break
            assert found, f"Original tensor for {prompt!r} not found in loaded"

    def test_parquet_standalone_readable(self, tmp_path, monkeypatch):
        """Verify Parquet index is standalone-readable with pyarrow."""
        import pyarrow.parquet as pq

        src_cache = tmp_path / "src_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(src_cache))
        cache_mod._backend = None

        prompts = ["hello", "world"]
        for prompt in prompts:
            pooled = torch.randn(1, HIDDEN_DIM)
            save_prompt_pooled_activations(
                TEST_MODEL, prompt, [0], pooled, "last_token"
            )

        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        mock_api = MagicMock()

        def capture_upload(repo_id, folder_path, **kwargs):
            for f in Path(folder_path).rglob("*"):
                if f.is_file():
                    rel = f.relative_to(folder_path)
                    dest = remote_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(dest))

        mock_api.upload_folder.side_effect = capture_upload

        with (
            patch("lmprobe.sharing._check_hub_deps"),
            patch("huggingface_hub.HfApi", return_value=mock_api),
        ):
            push_dataset(
                repo_id="user/parquet-test",
                model_name=TEST_MODEL,
                prompts=prompts,
                labels=[1, 0],
                exist_ok=True,
            )

        # Read Parquet directly
        table = pq.read_table(str(remote_dir / PARQUET_PATH))

        assert table.num_rows == 2
        assert set(table.column_names) == {
            "text", "label", "num_tokens", "shard_index", "row_offset",
        }
        assert set(table.column("text").to_pylist()) == set(prompts)
        assert set(table.column("label").to_pylist()) == {0, 1}

    def test_safetensors_standalone_readable(self, tmp_path, monkeypatch):
        """Verify safetensors shards have co-located layer keys."""
        from safetensors import safe_open

        src_cache = tmp_path / "src_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(src_cache))
        cache_mod._backend = None

        prompts = ["alpha", "beta"]
        for prompt in prompts:
            for layer in [0, 1]:
                pooled = torch.randn(1, HIDDEN_DIM)
                save_prompt_pooled_activations(
                    TEST_MODEL, prompt, [layer], pooled, "last_token"
                )

        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        mock_api = MagicMock()

        def capture_upload(repo_id, folder_path, **kwargs):
            for f in Path(folder_path).rglob("*"):
                if f.is_file():
                    rel = f.relative_to(folder_path)
                    dest = remote_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(dest))

        mock_api.upload_folder.side_effect = capture_upload

        with (
            patch("lmprobe.sharing._check_hub_deps"),
            patch("huggingface_hub.HfApi", return_value=mock_api),
        ):
            push_dataset(
                repo_id="user/sf-test",
                model_name=TEST_MODEL,
                prompts=prompts,
                exist_ok=True,
            )

        # Find and read the shard file
        shard_files = list(
            (remote_dir / "tensors").glob("hidden_layers_*.safetensors")
        )
        assert len(shard_files) >= 1

        with safe_open(str(shard_files[0]), framework="pt") as f:
            keys = list(f.keys())
            assert "hidden.layer_0" in keys
            assert "hidden.layer_1" in keys

            layer_0 = f.get_tensor("hidden.layer_0")
            assert layer_0.shape == (2, HIDDEN_DIM)


# =============================================================================
# Metadata tests
# =============================================================================


class TestConsolidationWithMetadata:
    def test_extra_keys_in_prompt_metadata(self, populated_cache):
        """Extra metadata keys appear in prompt_metadata dicts."""
        tensor_types = {
            "raw_layers": [],
            "pooled": {"last_token": [0]},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        meta = [
            {"statement_id": "s0", "category": "cities"},
            {"statement_id": "s1", "category": "cities"},
            {"statement_id": "s2", "category": "animals"},
        ]
        _, _, prompt_metadata = _consolidate_and_shard(
            model_name=TEST_MODEL,
            prompts=populated_cache,
            kept_indices=[0, 1, 2],
            tensor_types=tensor_types,
            labels=None,
            shard_max_bytes=1_000_000_000,
            repo_id="user/test",
            metadata=meta,
        )
        # All prompts should have the extra keys (order may be shuffled)
        for pm in prompt_metadata:
            assert "statement_id" in pm
            assert "category" in pm
        # All original values should be present
        ids = {pm["statement_id"] for pm in prompt_metadata}
        assert ids == {"s0", "s1", "s2"}


@requires_pyarrow
class TestExtraMetadataColumns:
    def test_extra_columns_in_parquet(self, tmp_path):
        """Extra metadata keys become extra Parquet columns."""
        import pyarrow.parquet as pq

        (tmp_path / "index").mkdir()
        prompt_metadata = [
            {"text": "hello", "label": 1, "num_tokens": 3,
             "shard_index": 0, "row_offset": 0,
             "statement_id": "s0", "score": 0.95},
            {"text": "world", "label": 0, "num_tokens": 4,
             "shard_index": 0, "row_offset": 1,
             "statement_id": "s1", "score": 0.42},
        ]
        _write_parquet_index(tmp_path, prompt_metadata)

        table = pq.read_table(str(tmp_path / PARQUET_PATH))
        assert "statement_id" in table.column_names
        assert "score" in table.column_names
        assert table.column("statement_id").to_pylist() == ["s0", "s1"]
        assert table.column("score").to_pylist() == [
            pytest.approx(0.95), pytest.approx(0.42),
        ]

    def test_token_offset_column(self, tmp_path):
        """token_offset from full-sequence storage appears as extra column."""
        import pyarrow.parquet as pq

        (tmp_path / "index").mkdir()
        prompt_metadata = [
            {"text": "a", "label": 1, "num_tokens": 3,
             "shard_index": 0, "row_offset": 0, "token_offset": 0},
            {"text": "b", "label": 0, "num_tokens": 5,
             "shard_index": 0, "row_offset": 0, "token_offset": 3},
        ]
        _write_parquet_index(tmp_path, prompt_metadata)

        table = pq.read_table(str(tmp_path / PARQUET_PATH))
        assert "token_offset" in table.column_names
        assert table.column("token_offset").to_pylist() == [0, 3]


@requires_pyarrow
class TestPushMetadataValidation:
    @patch("lmprobe.sharing._check_hub_deps")
    @patch("lmprobe.sharing._check_pyarrow")
    def test_metadata_length_mismatch(
        self, mock_pyarrow, mock_deps, populated_cache,
    ):
        with pytest.raises(ValueError, match="metadata length"):
            push_dataset(
                repo_id="user/test",
                model_name=TEST_MODEL,
                prompts=populated_cache,
                metadata=[{"a": 1}],  # length 1, prompts length 3
            )

    @patch("lmprobe.sharing._check_hub_deps")
    @patch("lmprobe.sharing._check_pyarrow")
    def test_metadata_inconsistent_keys(
        self, mock_pyarrow, mock_deps, populated_cache,
    ):
        with pytest.raises(ValueError, match="metadata"):
            push_dataset(
                repo_id="user/test",
                model_name=TEST_MODEL,
                prompts=populated_cache,
                metadata=[
                    {"a": 1},
                    {"a": 2},
                    {"b": 3},  # different keys
                ],
            )

    @patch("lmprobe.sharing._check_hub_deps")
    @patch("lmprobe.sharing._check_pyarrow")
    @patch("huggingface_hub.HfApi")
    def test_push_with_metadata_end_to_end(
        self, MockHfApi, mock_pyarrow, mock_deps, populated_cache,
    ):
        mock_api = MagicMock()
        MockHfApi.return_value = mock_api
        uploaded_files = {}

        def capture_upload(repo_id, folder_path, **kwargs):
            folder = Path(folder_path)
            for f in folder.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(folder)
                    if f.suffix == ".json":
                        uploaded_files[str(rel)] = json.loads(f.read_text())
                    elif f.suffix == ".parquet":
                        uploaded_files[str(rel)] = f.read_bytes()
                    else:
                        uploaded_files[str(rel)] = f.read_bytes()

        mock_api.upload_folder.side_effect = capture_upload

        meta = [
            {"statement_id": "s0", "category": "cities"},
            {"statement_id": "s1", "category": "cities"},
            {"statement_id": "s2", "category": "animals"},
        ]

        push_dataset(
            repo_id="user/meta-test",
            model_name=TEST_MODEL,
            prompts=populated_cache,
            labels=[1, 1, 0],
            metadata=meta,
            exist_ok=True,
        )

        # Verify Parquet has extra columns
        import io

        import pyarrow.parquet as pq
        parquet_bytes = uploaded_files[PARQUET_PATH]
        table = pq.read_table(io.BytesIO(parquet_bytes))
        assert "statement_id" in table.column_names
        assert "category" in table.column_names


# =============================================================================
# Full-sequence activation tests
# =============================================================================


class TestComputeShardBoundariesVariable:
    def test_single_shard(self):
        boundaries = _compute_shard_boundaries_variable(
            [100, 200, 300], shard_max_bytes=1000
        )
        assert boundaries == [3]

    def test_multiple_shards(self):
        boundaries = _compute_shard_boundaries_variable(
            [100, 200, 300, 400], shard_max_bytes=350
        )
        # 100+200=300 fits, +300 would be 600 > 350 → split
        assert len(boundaries) >= 2
        assert sum(boundaries) == 4

    def test_each_prompt_own_shard(self):
        boundaries = _compute_shard_boundaries_variable(
            [100, 100, 100], shard_max_bytes=1
        )
        # Each prompt exceeds limit alone, so each goes in its own shard
        assert boundaries == [1, 1, 1]


class TestConsolidateRawActivations:
    def test_full_sequence_storage_flag(self, populated_raw_cache):
        """Consolidation with raw layers sets storage='full_sequence'."""
        prompts, seq_lens, layers, _ = populated_raw_cache
        tensor_types = {
            "raw_layers": layers,
            "pooled": {},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        tmpdir, tensor_descriptors, prompt_metadata = _consolidate_and_shard(
            model_name=TEST_MODEL,
            prompts=prompts,
            kept_indices=[0, 1, 2],
            tensor_types=tensor_types,
            labels=None,
            shard_max_bytes=1_000_000_000,
            repo_id="user/raw-test",
        )

        desc = tensor_descriptors["hidden_layers"]
        assert desc["storage"] == "full_sequence"
        assert "pooling" not in desc
        assert "row_bytes" not in desc

        # Total tokens across shard should match sum of seq_lens
        total_shard_tokens = sum(
            s["num_tokens"] for s in desc["shards"]
        )
        assert total_shard_tokens == sum(seq_lens)

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_full_sequence_tensor_shape(self, populated_raw_cache):
        """Shard tensor dim-0 is total_tokens, not num_prompts."""
        from safetensors import safe_open

        prompts, seq_lens, layers, _ = populated_raw_cache
        tensor_types = {
            "raw_layers": layers,
            "pooled": {},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        tmpdir, tensor_descriptors, _ = _consolidate_and_shard(
            model_name=TEST_MODEL,
            prompts=prompts,
            kept_indices=[0, 1, 2],
            tensor_types=tensor_types,
            labels=None,
            shard_max_bytes=1_000_000_000,
            repo_id="user/raw-test",
        )

        shard_file = tmpdir / tensor_descriptors["hidden_layers"]["shards"][0]["file"]
        with safe_open(str(shard_file), framework="pt") as f:
            layer_0 = f.get_tensor("hidden.layer_0")
            # Total tokens = 3 + 5 + 4 = 12 (may be shuffled but total same)
            assert layer_0.shape == (sum(seq_lens), HIDDEN_DIM)

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_token_offset_in_metadata(self, populated_raw_cache):
        """prompt_metadata has token_offset instead of row_offset for raw."""
        prompts, seq_lens, layers, _ = populated_raw_cache
        tensor_types = {
            "raw_layers": layers,
            "pooled": {},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        _, _, prompt_metadata = _consolidate_and_shard(
            model_name=TEST_MODEL,
            prompts=prompts,
            kept_indices=[0, 1, 2],
            tensor_types=tensor_types,
            labels=None,
            shard_max_bytes=1_000_000_000,
            repo_id="user/raw-test",
        )

        for pm in prompt_metadata:
            assert "token_offset" in pm
            assert "shard_index" in pm


@requires_pyarrow
class TestFullSequenceRoundtrip:
    """Push raw activations, pull into fresh cache, verify raw + pooled."""

    def test_push_pull_roundtrip_raw(self, tmp_path, monkeypatch):
        # --- Phase 1: populate source cache with raw activations ---
        src_cache = tmp_path / "src_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(src_cache))
        cache_mod._backend = None

        prompts = ["alpha", "beta", "gamma"]
        seq_lens = [3, 5, 4]
        layers = [0, 1]
        original_raw = {}

        for prompt, sl in zip(prompts, seq_lens):
            act = torch.randn(1, sl, HIDDEN_DIM * len(layers))
            mask = torch.ones(1, sl, dtype=torch.long)
            save_prompt_activations(TEST_MODEL, prompt, layers, act, mask)
            original_raw[prompt] = (act, mask, sl)

        # --- Phase 2: push ---
        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        mock_api = MagicMock()

        def capture_upload(repo_id, folder_path, **kwargs):
            for f in Path(folder_path).rglob("*"):
                if f.is_file():
                    rel = f.relative_to(folder_path)
                    dest = remote_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(dest))

        mock_api.upload_folder.side_effect = capture_upload

        with (
            patch("lmprobe.sharing._check_hub_deps"),
            patch("huggingface_hub.HfApi", return_value=mock_api),
        ):
            push_dataset(
                repo_id="user/raw-roundtrip",
                model_name=TEST_MODEL,
                prompts=prompts,
                labels=[1, 1, 0],
                exist_ok=True,
            )

        # Verify storage type in info
        with open(remote_dir / INFO_FILENAME) as f:
            info = json.load(f)
        assert info["tensors"]["hidden_layers"]["storage"] == "full_sequence"

        # --- Phase 3: pull into fresh cache ---
        dst_cache = tmp_path / "dst_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(dst_cache))
        cache_mod._backend = None

        def mock_download(repo_id, filename, **kwargs):
            return str(remote_dir / filename)

        with (
            patch("lmprobe.sharing._check_hub_deps"),
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=mock_download,
            ),
        ):
            count = pull_dataset("user/raw-roundtrip")

        assert count == 3

        # --- Phase 4: verify pulled data ---
        for prompt in prompts:
            cached = discover_cached(TEST_MODEL, prompt)
            assert cached is not None

            # Raw activations should be saved
            pulled_act, pulled_mask = load_prompt_activations(
                TEST_MODEL, prompt, layers,
            )
            orig_act, orig_mask, sl = original_raw[prompt]

            assert pulled_act.shape == orig_act.shape
            assert torch.allclose(pulled_act, orig_act, atol=1e-5)

            # Pooled (last_token) should also be saved
            pooled = load_prompt_pooled_activations(
                TEST_MODEL, prompt, [0], "last_token",
            )
            assert pooled.shape == (1, HIDDEN_DIM)

    def test_push_load_roundtrip_raw(self, tmp_path, monkeypatch):
        """Push raw -> load_activation_dataset returns correct shapes."""
        src_cache = tmp_path / "src_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(src_cache))
        cache_mod._backend = None

        prompts = ["alpha", "beta"]
        seq_lens = [3, 5]
        layers = [0, 1]

        for prompt, sl in zip(prompts, seq_lens):
            act = torch.randn(1, sl, HIDDEN_DIM * len(layers))
            mask = torch.ones(1, sl, dtype=torch.long)
            save_prompt_activations(TEST_MODEL, prompt, layers, act, mask)

        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        mock_api = MagicMock()

        def capture_upload(repo_id, folder_path, **kwargs):
            for f in Path(folder_path).rglob("*"):
                if f.is_file():
                    rel = f.relative_to(folder_path)
                    dest = remote_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(dest))

        mock_api.upload_folder.side_effect = capture_upload

        with (
            patch("lmprobe.sharing._check_hub_deps"),
            patch("huggingface_hub.HfApi", return_value=mock_api),
        ):
            push_dataset(
                repo_id="user/raw-load",
                model_name=TEST_MODEL,
                prompts=prompts,
                exist_ok=True,
            )

        def mock_download(repo_id, filename, **kwargs):
            return str(remote_dir / filename)

        with (
            patch("lmprobe.sharing._check_hub_deps"),
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=mock_download,
            ),
        ):
            result, info = load_activation_dataset("user/raw-load")

        # Total tokens = 3 + 5 = 8
        assert result["hidden.layer_0"].shape == (sum(seq_lens), HIDDEN_DIM)
        assert result["hidden.layer_1"].shape == (sum(seq_lens), HIDDEN_DIM)


@requires_pyarrow
class TestFullSequenceWithLogits:
    """Test combining full-sequence hidden layers + logits_topk."""

    def test_push_pull_raw_with_logits(self, tmp_path, monkeypatch):
        """Push raw hidden + logits, pull, verify both are correct."""
        src_cache = tmp_path / "src_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(src_cache))
        cache_mod._backend = None

        prompts = ["alpha", "beta", "gamma"]
        seq_lens = [3, 5, 4]
        layers = [0, 1]
        top_k = 10
        vocab_size = 50

        original_raw = {}

        for prompt, sl in zip(prompts, seq_lens):
            # Save raw activations
            act = torch.randn(1, sl, HIDDEN_DIM * len(layers))
            mask = torch.ones(1, sl, dtype=torch.long)
            save_prompt_activations(TEST_MODEL, prompt, layers, act, mask)
            original_raw[prompt] = act

            # Save top-k logits
            logits = torch.randn(1, sl, vocab_size)
            save_prompt_logits(
                TEST_MODEL, prompt, logits, mask,
                top_k=top_k, positions="last",
            )

        # Push
        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        mock_api = MagicMock()

        def capture_upload(repo_id, folder_path, **kwargs):
            for f in Path(folder_path).rglob("*"):
                if f.is_file():
                    rel = f.relative_to(folder_path)
                    dest = remote_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(dest))

        mock_api.upload_folder.side_effect = capture_upload

        with (
            patch("lmprobe.sharing._check_hub_deps"),
            patch("huggingface_hub.HfApi", return_value=mock_api),
        ):
            push_dataset(
                repo_id="user/raw-logits",
                model_name=TEST_MODEL,
                prompts=prompts,
                labels=[1, 1, 0],
                exist_ok=True,
            )

        # Verify both tensor types in info
        with open(remote_dir / INFO_FILENAME) as f:
            info = json.load(f)
        assert "hidden_layers" in info["tensors"]
        assert "logits_topk" in info["tensors"]
        assert info["tensors"]["hidden_layers"]["storage"] == "full_sequence"

        # Verify Parquet has both row_offset and token_offset
        import pyarrow.parquet as pq

        table = pq.read_table(str(remote_dir / PARQUET_PATH))
        assert "row_offset" in table.column_names
        assert "token_offset" in table.column_names

        # Pull into fresh cache
        dst_cache = tmp_path / "dst_cache"
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(dst_cache))
        cache_mod._backend = None

        def mock_download(repo_id, filename, **kwargs):
            return str(remote_dir / filename)

        with (
            patch("lmprobe.sharing._check_hub_deps"),
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=mock_download,
            ),
        ):
            count = pull_dataset("user/raw-logits")

        assert count == 3

        # Verify raw activations roundtrip
        for prompt in prompts:
            pulled_act, pulled_mask = load_prompt_activations(
                TEST_MODEL, prompt, layers,
            )
            orig_act = original_raw[prompt]
            assert pulled_act.shape == orig_act.shape
            assert torch.allclose(pulled_act, orig_act, atol=1e-5)


class TestPooledStorageUnchanged:
    """Existing pooled tests still pass — auto-detect falls back to pooled."""

    def test_pooled_storage_flag(self, populated_cache):
        tensor_types = {
            "raw_layers": [],
            "pooled": {"last_token": [0, 1]},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }
        tmpdir, tensor_descriptors, prompt_metadata = _consolidate_and_shard(
            model_name=TEST_MODEL,
            prompts=populated_cache,
            kept_indices=[0, 1, 2],
            tensor_types=tensor_types,
            labels=None,
            shard_max_bytes=1_000_000_000,
            repo_id="user/pooled-test",
        )

        desc = tensor_descriptors["hidden_layers"]
        assert desc["storage"] == "pooled"
        assert desc["pooling"] == "last_token"
        assert "row_bytes" in desc

        # row_offset should be in metadata, not token_offset
        for pm in prompt_metadata:
            assert "row_offset" in pm

        shutil.rmtree(tmpdir, ignore_errors=True)
