"""Tests for activation_types module: MoE detection, ExtractionSpec, ExtractedBatch."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lmprobe.activation_types import (
    ActivationType,
    ExtractedBatch,
    ExtractionSpec,
    MoEInfo,
    detect_moe_info,
    get_router_module,
    validate_router_layers,
)


class TestActivationType:
    def test_enum_values(self):
        assert ActivationType.HIDDEN == "hidden"
        assert ActivationType.LOGITS == "logits"
        assert ActivationType.ROUTER_LOGITS == "router"

    def test_string_comparison(self):
        assert ActivationType.HIDDEN == "hidden"
        assert ActivationType.ROUTER_LOGITS == "router"


class TestExtractionSpec:
    def test_defaults(self):
        spec = ExtractionSpec(hidden_layers=[0, 1, 2])
        assert spec.hidden_layers == [0, 1, 2]
        assert spec.include_logits is False
        assert spec.logit_top_k is None
        assert spec.router_layers is None
        assert spec.router_module_template is None

    def test_with_router(self):
        spec = ExtractionSpec(
            hidden_layers=[0, 1],
            router_layers=[0, 1],
            router_module_template="model.model.layers.{layer}.block_sparse_moe.gate",
        )
        assert spec.router_layers == [0, 1]
        assert spec.router_module_template is not None

    def test_router_without_template_raises(self):
        with pytest.raises(ValueError, match="router_module_template is required"):
            ExtractionSpec(
                hidden_layers=[0],
                router_layers=[0],
            )

    def test_empty_router_layers_ok_without_template(self):
        # None router_layers is fine without template
        spec = ExtractionSpec(hidden_layers=[0], router_layers=None)
        assert spec.router_layers is None

    def test_with_logits(self):
        spec = ExtractionSpec(
            hidden_layers=[0],
            include_logits=True,
            logit_top_k=50,
        )
        assert spec.include_logits is True
        assert spec.logit_top_k == 50


class TestExtractedBatch:
    def test_construction(self):
        batch = ExtractedBatch(
            activations=torch.randn(2, 10, 64),
            attention_mask=torch.ones(2, 10),
        )
        assert batch.activations is not None
        assert batch.logits is None
        assert batch.router_logits is None

    def test_with_router_logits(self):
        router = {
            0: torch.randn(2, 10, 8),
            1: torch.randn(2, 10, 8),
        }
        batch = ExtractedBatch(
            activations=torch.randn(2, 10, 64),
            attention_mask=torch.ones(2, 10),
            router_logits=router,
        )
        assert batch.router_logits is not None
        assert len(batch.router_logits) == 2
        assert batch.router_logits[0].shape == (2, 10, 8)


class TestMoEInfo:
    def test_frozen(self):
        info = MoEInfo(
            num_experts=8,
            router_module_template="model.model.layers.{layer}.block_sparse_moe.gate",
        )
        assert info.num_experts == 8
        assert info.moe_layer_indices is None

    def test_with_moe_indices(self):
        info = MoEInfo(
            num_experts=64,
            router_module_template="model.model.layers.{layer}.mlp.gate",
            moe_layer_indices=[1, 3, 5, 7],
        )
        assert info.moe_layer_indices == [1, 3, 5, 7]


class TestDetectMoEInfo:
    def _mock_config(self, **kwargs):
        config = MagicMock()
        # Default to no MoE attributes
        config.model_type = kwargs.get("model_type", "llama")
        config.num_local_experts = kwargs.get("num_local_experts", None)
        config.num_experts = kwargs.get("num_experts", None)
        config.n_routed_experts = kwargs.get("n_routed_experts", None)
        config.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        config.first_k_dense_replace = kwargs.get("first_k_dense_replace", 1)
        config.moe_layer_freq = kwargs.get("moe_layer_freq", 1)
        config.ffn_config = kwargs.get("ffn_config", None)
        return config

    @patch("transformers.AutoConfig")
    def test_dense_model_returns_none(self, mock_autoconfig):
        mock_autoconfig.from_pretrained.return_value = self._mock_config(
            model_type="llama"
        )
        result = detect_moe_info("meta-llama/Llama-3.1-8B")
        assert result is None

    @patch("transformers.AutoConfig")
    def test_mixtral(self, mock_autoconfig):
        mock_autoconfig.from_pretrained.return_value = self._mock_config(
            model_type="mixtral",
            num_local_experts=8,
        )
        result = detect_moe_info("mistralai/Mixtral-8x7B")
        assert result is not None
        assert result.num_experts == 8
        assert "block_sparse_moe.gate" in result.router_module_template
        assert result.moe_layer_indices is None

    @patch("transformers.AutoConfig")
    def test_qwen2_moe(self, mock_autoconfig):
        mock_autoconfig.from_pretrained.return_value = self._mock_config(
            model_type="qwen2_moe",
            num_experts=60,
        )
        result = detect_moe_info("Qwen/Qwen1.5-MoE-A2.7B")
        assert result is not None
        assert result.num_experts == 60
        assert "mlp.gate" in result.router_module_template

    @patch("transformers.AutoConfig")
    def test_deepseek_v2(self, mock_autoconfig):
        mock_autoconfig.from_pretrained.return_value = self._mock_config(
            model_type="deepseek_v2",
            n_routed_experts=64,
            num_hidden_layers=60,
            first_k_dense_replace=2,
            moe_layer_freq=2,
        )
        result = detect_moe_info("deepseek-ai/DeepSeek-V2")
        assert result is not None
        assert result.num_experts == 64
        assert result.moe_layer_indices is not None
        # MoE layers start at 2, every 2 layers
        assert 0 not in result.moe_layer_indices
        assert 1 not in result.moe_layer_indices
        assert 2 in result.moe_layer_indices
        assert 4 in result.moe_layer_indices

    @patch("transformers.AutoConfig")
    def test_dbrx(self, mock_autoconfig):
        ffn_config = MagicMock()
        ffn_config.moe_num_experts = 16
        mock_autoconfig.from_pretrained.return_value = self._mock_config(
            model_type="dbrx",
            ffn_config=ffn_config,
        )
        result = detect_moe_info("databricks/dbrx-instruct")
        assert result is not None
        assert result.num_experts == 16
        assert "ffn.router.layer" in result.router_module_template


class TestValidateRouterLayers:
    def test_all_layers_moe(self):
        info = MoEInfo(num_experts=8, router_module_template="t.{layer}")
        result = validate_router_layers(info, [0, 1, 2, 3])
        assert result == [0, 1, 2, 3]

    def test_partial_moe_layers(self):
        info = MoEInfo(
            num_experts=8,
            router_module_template="t.{layer}",
            moe_layer_indices=[1, 3, 5],
        )
        result = validate_router_layers(info, [0, 1, 2, 3])
        assert result == [1, 3]

    def test_no_valid_layers_raises(self):
        info = MoEInfo(
            num_experts=8,
            router_module_template="t.{layer}",
            moe_layer_indices=[10, 20],
        )
        with pytest.raises(ValueError, match="None of the requested layers"):
            validate_router_layers(info, [0, 1, 2])


class TestGetRouterModule:
    def test_simple_path(self):
        # Build a real object hierarchy (MagicMock auto-creates attributes
        # which interferes with navigation)
        class Gate:
            pass

        class MoE:
            def __init__(self):
                self.gate = Gate()

        class Layer:
            def __init__(self):
                self.block_sparse_moe = MoE()

        class Layers:
            def __init__(self):
                self._layers = [Layer()]
            def __getitem__(self, idx):
                return self._layers[idx]

        class InnerModel:
            def __init__(self):
                self.layers = Layers()

        class Model:
            def __init__(self):
                self.model = InnerModel()

        top = Model()
        # Template: model.model.layers.{layer}.block_sparse_moe.gate
        # get_router_module starts from the object passed in and walks
        # each part of the formatted path
        result = get_router_module(
            top,
            "model.layers.{layer}.block_sparse_moe.gate",
            layer=0,
        )
        assert isinstance(result, Gate)
        assert result is top.model.layers[0].block_sparse_moe.gate
