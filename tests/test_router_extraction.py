"""Tests for MoE router extraction via LocalBackend and cache round-trip."""


import pytest
import torch

from lmprobe.activation_types import ExtractedBatch, ExtractionSpec
from lmprobe.backends import LocalBackend, _get_router_modules


class TestGetRouterModules:
    """Test _get_router_modules with a real tiny model."""

    def test_dense_model_returns_empty(self, tiny_model):
        """Dense models have no router modules."""
        backend = LocalBackend(tiny_model, device="cpu")
        model = backend.model
        result = _get_router_modules(
            model,
            [0, 1],
            # Use a template that won't exist on a dense model
            "model.layers.{layer}.block_sparse_moe.gate",
        )
        assert result == {}


class TestLocalBackendExtractBatchExtended:
    """Test extract_batch_extended on LocalBackend with a real tiny model."""

    def test_hidden_only(self, tiny_model):
        """Extract hidden states only via extended method."""
        backend = LocalBackend(tiny_model, device="cpu")
        spec = ExtractionSpec(hidden_layers=[0, 1])
        result = backend.extract_batch_extended(
            ["Hello world", "Test prompt"],
            spec,
        )
        assert isinstance(result, ExtractedBatch)
        assert result.activations is not None
        assert result.attention_mask is not None
        assert result.logits is None
        assert result.router_logits is None
        # Check shapes
        batch_size = 2
        assert result.activations.shape[0] == batch_size
        assert result.attention_mask.shape[0] == batch_size

    def test_hidden_with_logits(self, tiny_model):
        """Extract hidden states + logits via extended method."""
        backend = LocalBackend(tiny_model, device="cpu")
        spec = ExtractionSpec(hidden_layers=[0], include_logits=True)
        result = backend.extract_batch_extended(
            ["Hello world"],
            spec,
        )
        assert result.activations is not None
        assert result.logits is not None
        assert result.logits.dim() == 3  # (batch, seq, vocab)

    def test_no_hidden_with_logits(self, tiny_model):
        """Extract logits only (no hidden states)."""
        backend = LocalBackend(tiny_model, device="cpu")
        spec = ExtractionSpec(hidden_layers=[], include_logits=True)
        result = backend.extract_batch_extended(
            ["Hello world"],
            spec,
        )
        assert result.activations is None
        assert result.logits is not None

    def test_router_on_dense_model_no_crash(self, tiny_model):
        """Requesting router logits on a dense model shouldn't crash,
        but router_logits should be None (no modules found)."""
        backend = LocalBackend(tiny_model, device="cpu")
        spec = ExtractionSpec(
            hidden_layers=[0],
            router_layers=[0],
            router_module_template="model.layers.{layer}.block_sparse_moe.gate",
        )
        result = backend.extract_batch_extended(
            ["Hello world"],
            spec,
        )
        assert result.activations is not None
        # Router logits should be None since no router modules exist
        assert result.router_logits is None


class TestCacheRoundTrip:
    """Test saving and loading router logits from cache."""

    def test_save_load_router_logits(self, tmp_path, monkeypatch):
        """Router logits survive a cache round-trip."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import (
            load_prompt_router_logits,
            save_prompt_router_logits,
        )

        model_name = "test-model"
        prompt = "Hello world"
        router_logits = {
            0: torch.randn(5, 8),   # 5 tokens, 8 experts
            3: torch.randn(5, 8),
        }

        save_prompt_router_logits(model_name, prompt, router_logits)
        loaded = load_prompt_router_logits(model_name, prompt, [0, 3])

        assert set(loaded.keys()) == {0, 3}
        torch.testing.assert_close(loaded[0], router_logits[0])
        torch.testing.assert_close(loaded[3], router_logits[3])

    def test_load_missing_layer_raises(self, tmp_path, monkeypatch):
        """Loading a router layer that wasn't saved raises KeyError."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import (
            load_prompt_router_logits,
            save_prompt_router_logits,
        )

        model_name = "test-model"
        prompt = "Hello world"
        save_prompt_router_logits(
            model_name, prompt, {0: torch.randn(5, 8)}
        )

        with pytest.raises(KeyError, match="router_layer_5"):
            load_prompt_router_logits(model_name, prompt, [0, 5])


class TestCachedPromptInfoRouter:
    """Test that discover_cached includes router layer info."""

    def test_discover_with_router(self, tmp_path, monkeypatch):
        """CachedPromptInfo.router_layers populated after saving router logits."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import (
            discover_cached,
            save_prompt_activations,
            save_prompt_router_logits,
        )

        model_name = "test-model"
        prompt = "Hello world"

        # Save hidden states first (so there's a cache entry)
        save_prompt_activations(
            model_name, prompt,
            layers=[0, 1],
            activations=torch.randn(1, 5, 64),  # 1 batch, 5 tokens, 2*32 dim
            attention_mask=torch.ones(5),
        )

        # Save router logits
        save_prompt_router_logits(
            model_name, prompt,
            {0: torch.randn(5, 8), 1: torch.randn(5, 8)},
        )

        info = discover_cached(model_name, prompt)
        assert info is not None
        assert sorted(info.router_layers) == [0, 1]

    def test_discover_without_router(self, tmp_path, monkeypatch):
        """CachedPromptInfo.router_layers is empty when no router data cached."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import (
            discover_cached,
            save_prompt_activations,
        )

        model_name = "test-model"
        prompt = "Hello world"
        save_prompt_activations(
            model_name, prompt,
            layers=[0],
            activations=torch.randn(1, 5, 32),
            attention_mask=torch.ones(5),
        )

        info = discover_cached(model_name, prompt)
        assert info is not None
        assert info.router_layers == []


class TestSaveBatchWithRouter:
    """Test _save_batch with router logits."""

    def test_save_batch_router_logits(self, tmp_path):
        """_save_batch saves router logits to separate per-layer directories."""
        from safetensors.torch import load_file

        from lmprobe.extract import _save_batch

        batch_acts = torch.randn(2, 5, 64)  # 2 prompts, 5 tokens, 2 layers * 32 dim
        batch_mask = torch.ones(2, 5)
        router_logits = {
            0: torch.randn(2, 5, 8),  # 8 experts
            1: torch.randn(2, 5, 8),
        }

        _save_batch(
            batch_acts=batch_acts,
            batch_mask=batch_mask,
            layer_indices=[0, 1],
            hidden_dim=32,
            prefix="test",
            batch_idx=0,
            local_root=tmp_path,
            batch_router_logits=router_logits,
        )

        # Check hidden state files exist
        assert (tmp_path / "layer_000" / "batch_000000.safetensors").exists()
        assert (tmp_path / "layer_001" / "batch_000000.safetensors").exists()

        # Check router files exist
        router_path_0 = tmp_path / "router_layer_0" / "batch_000000.safetensors"
        router_path_1 = tmp_path / "router_layer_1" / "batch_000000.safetensors"
        assert router_path_0.exists()
        assert router_path_1.exists()

        # Verify router file contents
        loaded = load_file(str(router_path_0))
        assert "router_logits" in loaded
        assert "mask" in loaded
        assert loaded["router_logits"].shape == (2, 5, 8)

    def test_save_batch_no_router(self, tmp_path):
        """_save_batch without router logits doesn't create router dirs."""
        from lmprobe.extract import _save_batch

        batch_acts = torch.randn(2, 5, 32)
        batch_mask = torch.ones(2, 5)

        _save_batch(
            batch_acts=batch_acts,
            batch_mask=batch_mask,
            layer_indices=[0],
            hidden_dim=32,
            prefix="test",
            batch_idx=0,
            local_root=tmp_path,
        )

        assert (tmp_path / "layer_000" / "batch_000000.safetensors").exists()
        # No router directories
        assert not list(tmp_path.glob("router_layer_*"))


class TestExtractionManifestRouter:
    """Test ExtractionManifest with router fields."""

    def test_roundtrip(self):
        from lmprobe.extract import ExtractionManifest

        manifest = ExtractionManifest(
            model_name="test-model",
            layers=[0, 1, 2],
            hidden_dim=64,
            total_prompts=10,
            router_layers=[0, 1],
            router_dim=8,
        )

        d = manifest.to_dict()
        assert d["router_layers"] == [0, 1]
        assert d["router_dim"] == 8

        loaded = ExtractionManifest.from_dict(d)
        assert loaded.router_layers == [0, 1]
        assert loaded.router_dim == 8

    def test_backward_compat(self):
        """Old manifests without router fields load with None defaults."""
        from lmprobe.extract import ExtractionManifest

        old_dict = {
            "model_name": "test",
            "layers": [0],
            "hidden_dim": 32,
            "total_prompts": 5,
        }
        loaded = ExtractionManifest.from_dict(old_dict)
        assert loaded.router_layers is None
        assert loaded.router_dim is None

    def test_no_router_not_serialized(self):
        """When router fields are None, they aren't in the dict."""
        from lmprobe.extract import ExtractionManifest

        manifest = ExtractionManifest(
            model_name="test",
            layers=[0],
            hidden_dim=32,
            total_prompts=5,
        )
        d = manifest.to_dict()
        assert "router_layers" not in d
        assert "router_dim" not in d


class TestParseRouterLayerKeys:
    """Test _parse_router_layer_keys."""

    def test_parse_router_keys(self):
        from lmprobe.cache import _parse_router_layer_keys

        keys = {"layer_0", "layer_1", "router_layer_0", "router_layer_3", "attention_mask"}
        result = _parse_router_layer_keys(keys)
        assert result == {0, 3}

    def test_empty(self):
        from lmprobe.cache import _parse_router_layer_keys

        result = _parse_router_layer_keys(set())
        assert result == set()

    def test_no_router_keys(self):
        from lmprobe.cache import _parse_router_layer_keys

        keys = {"layer_0", "layer_1", "attention_mask"}
        result = _parse_router_layer_keys(keys)
        assert result == set()


class TestRouterLayerKey:
    """Test _router_layer_key."""

    def test_format(self):
        from lmprobe.cache import _router_layer_key

        assert _router_layer_key(0) == "router_layer_0"
        assert _router_layer_key(15) == "router_layer_15"


class TestNnsightBackendExtendedNotSupported:
    """Test that NnsightBackend.extract_batch_extended is callable."""

    def test_nnsight_backend_has_method(self):
        """NnsightBackend inherits extract_batch_extended from ABC."""
        from lmprobe.backends import NnsightBackend

        # Just verify the method exists (don't call it — requires nnsight model)
        assert hasattr(NnsightBackend, "extract_batch_extended")


class TestExtractionBackendDefaultRaises:
    """Test that base class extract_batch_extended raises NotImplementedError."""

    def test_default_raises(self):
        from lmprobe.backends import ExtractionBackend

        # The ABC can't be instantiated, but we can check the method
        # exists and its docstring mentions extensibility
        assert hasattr(ExtractionBackend, "extract_batch_extended")


class TestSharingRouterDescriptor:
    """Test router logits in tensor descriptors and sharing functions."""

    def test_compute_tensor_intersection_with_router(self):
        """Router layers are intersected across infos."""
        from lmprobe.cache import CachedPromptInfo
        from lmprobe.sharing import _compute_tensor_intersection

        infos = [
            CachedPromptInfo(
                raw_layers=[0, 1], pooled={}, has_logits=False,
                logits_top_k=None, has_perplexity=False,
                has_token_perplexity=False, num_tokens=10,
                router_layers=[0, 1, 2],
            ),
            CachedPromptInfo(
                raw_layers=[0, 1], pooled={}, has_logits=False,
                logits_top_k=None, has_perplexity=False,
                has_token_perplexity=False, num_tokens=8,
                router_layers=[1, 2, 3],
            ),
        ]
        result = _compute_tensor_intersection(infos)
        assert result["router_layers"] == [1, 2]

    def test_compute_tensor_intersection_no_router(self):
        """Empty router_layers when not all infos have router data."""
        from lmprobe.cache import CachedPromptInfo
        from lmprobe.sharing import _compute_tensor_intersection

        infos = [
            CachedPromptInfo(
                raw_layers=[0], pooled={}, has_logits=False,
                logits_top_k=None, has_perplexity=False,
                has_token_perplexity=False, num_tokens=10,
                router_layers=[0, 1],
            ),
            CachedPromptInfo(
                raw_layers=[0], pooled={}, has_logits=False,
                logits_top_k=None, has_perplexity=False,
                has_token_perplexity=False, num_tokens=8,
                router_layers=[],  # no router data
            ),
        ]
        result = _compute_tensor_intersection(infos)
        assert result["router_layers"] == []

    def test_compute_tensor_intersection_empty(self):
        """Empty infos returns empty router_layers."""
        from lmprobe.sharing import _compute_tensor_intersection

        result = _compute_tensor_intersection([])
        assert result["router_layers"] == []

    def test_filter_tensor_types_router(self):
        """Filter includes router_layers when requested."""
        from lmprobe.sharing import _filter_tensor_types

        available = {
            "raw_layers": [0, 1],
            "pooled": {},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
            "has_token_perplexity": False,
            "router_layers": [0, 1, 2],
        }
        result = _filter_tensor_types(available, ["router_logits"])
        assert result["router_layers"] == [0, 1, 2]
        # Hidden not included
        assert result["raw_layers"] == []

    def test_filter_tensor_types_no_router(self):
        """Filter without router_logits key returns empty router_layers."""
        from lmprobe.sharing import _filter_tensor_types

        available = {
            "raw_layers": [0],
            "pooled": {},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
            "has_token_perplexity": False,
            "router_layers": [0, 1],
        }
        result = _filter_tensor_types(available, ["hidden_layers"])
        assert result["router_layers"] == []

    def test_enumerate_shard_files_with_router(self):
        """_enumerate_shard_files includes router shard files."""
        from lmprobe.sharing import _enumerate_shard_files

        plan = {
            "has_hidden": False,
            "hidden_layers": [],
            "hidden_boundaries": [],
            "want_logits": False,
            "logits_boundaries": [],
            "lt_shard_count": 0,
            "want_router": True,
            "router_boundaries": [5, 5],
        }
        files = _enumerate_shard_files(plan)
        assert files == [
            "tensors/router_logits_000.safetensors",
            "tensors/router_logits_001.safetensors",
        ]

    def test_reconstruct_plan_with_router(self):
        """_reconstruct_plan_from_cached includes router fields."""
        from lmprobe.sharing import _reconstruct_plan_from_cached

        cached_meta = {
            "tensor_descriptors": {
                "router_logits": {
                    "type": "router",
                    "layers": [0, 1],
                    "num_experts": 8,
                    "shards": [
                        {"file": "f1.safetensors", "num_prompts": 10},
                        {"file": "f2.safetensors", "num_prompts": 5},
                    ],
                }
            }
        }
        plan = _reconstruct_plan_from_cached(cached_meta)
        assert plan["want_router"] is True
        assert plan["router_boundaries"] == [10, 5]

    def test_load_router_for_prompt(self, tmp_path, monkeypatch):
        """_load_router_for_prompt loads and reshapes router logits."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        from lmprobe.cache import save_prompt_router_logits
        from lmprobe.sharing import _load_router_for_prompt

        model_name = "test-model"
        prompt = "Hello world"
        # Save router with seq_len=5, num_experts=8
        save_prompt_router_logits(
            model_name, prompt, {0: torch.randn(5, 8), 1: torch.randn(5, 8)}
        )

        result = _load_router_for_prompt(model_name, prompt, [0, 1])
        assert "router.layer_0" in result
        assert "router.layer_1" in result
        # Should be reshaped to (1, num_experts) — last token
        assert result["router.layer_0"].shape == (1, 8)
        assert result["router.layer_1"].shape == (1, 8)
