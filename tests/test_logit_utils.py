"""Tests for lmprobe.logit_utils module."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from lmprobe.logit_utils import (
    _detect_norm_type,
    apply_norm,
    compute_perplexity_from_activations,
    download_lm_head_weights,
)

pytestmark = pytest.mark.nnsight

# ---------------------------------------------------------------------------
# _detect_norm_type
# ---------------------------------------------------------------------------


class TestDetectNormType:
    """Tests for _detect_norm_type."""

    def test_rms_norm_detected(self):
        config = SimpleNamespace(rms_norm_eps=1e-5)
        norm_type, eps = _detect_norm_type(config)
        assert norm_type == "rms_norm"
        assert eps == 1e-5

    def test_layer_norm_detected(self):
        config = SimpleNamespace(layer_norm_eps=1e-6)
        norm_type, eps = _detect_norm_type(config)
        assert norm_type == "layer_norm"
        assert eps == 1e-6

    def test_rms_norm_takes_priority_over_layer_norm(self):
        """When both attrs exist, rms_norm_eps is checked first."""
        config = SimpleNamespace(rms_norm_eps=1e-5, layer_norm_eps=1e-6)
        norm_type, eps = _detect_norm_type(config)
        assert norm_type == "rms_norm"
        assert eps == 1e-5

    def test_fallback_when_neither_attribute(self):
        config = SimpleNamespace()
        norm_type, eps = _detect_norm_type(config)
        assert norm_type == "rms_norm"
        assert eps == 1e-5

    def test_fallback_logs_warning(self, caplog):
        config = SimpleNamespace()
        import logging

        with caplog.at_level(logging.WARNING, logger="lmprobe.logit_utils"):
            _detect_norm_type(config)
        assert "Could not detect norm type" in caplog.text


# ---------------------------------------------------------------------------
# download_lm_head_weights  (uses real tiny model)
# ---------------------------------------------------------------------------


class TestDownloadLmHeadWeights:
    """Tests for download_lm_head_weights with the tiny test model."""

    def test_returns_correct_types(self, tiny_model):
        norm_weight, lm_head_weight, config_dict = download_lm_head_weights(
            tiny_model, device="cpu"
        )
        assert isinstance(norm_weight, torch.Tensor)
        assert isinstance(lm_head_weight, torch.Tensor)
        assert isinstance(config_dict, dict)

    def test_norm_weight_is_1d(self, tiny_model):
        norm_weight, _, _ = download_lm_head_weights(tiny_model, device="cpu")
        assert norm_weight.dim() == 1

    def test_lm_head_weight_is_2d(self, tiny_model):
        _, lm_head_weight, _ = download_lm_head_weights(tiny_model, device="cpu")
        assert lm_head_weight.dim() == 2

    def test_config_dict_keys(self, tiny_model):
        _, _, config_dict = download_lm_head_weights(tiny_model, device="cpu")
        assert "eps" in config_dict
        assert "norm_type" in config_dict
        assert "norm_bias" in config_dict

    def test_norm_type_is_rms_for_llama(self, tiny_model):
        _, _, config_dict = download_lm_head_weights(tiny_model, device="cpu")
        assert config_dict["norm_type"] == "rms_norm"

    def test_dtype_cast(self, tiny_model):
        norm_weight, lm_head_weight, _ = download_lm_head_weights(
            tiny_model, device="cpu", dtype=torch.float32
        )
        assert norm_weight.dtype == torch.float32
        assert lm_head_weight.dtype == torch.float32

    def test_shapes_are_compatible(self, tiny_model):
        """norm_weight hidden_dim matches lm_head_weight hidden_dim."""
        norm_weight, lm_head_weight, _ = download_lm_head_weights(
            tiny_model, device="cpu"
        )
        hidden_dim = norm_weight.shape[0]
        assert lm_head_weight.shape[1] == hidden_dim


# ---------------------------------------------------------------------------
# apply_norm
# ---------------------------------------------------------------------------


class TestApplyNorm:
    """Tests for apply_norm."""

    def test_rms_norm_output_shape(self):
        hidden = torch.randn(2, 4, 8)
        weight = torch.ones(8)
        out = apply_norm(hidden, weight, eps=1e-5, norm_type="rms_norm")
        assert out.shape == hidden.shape

    def test_layer_norm_output_shape(self):
        hidden = torch.randn(2, 4, 8)
        weight = torch.ones(8)
        out = apply_norm(hidden, weight, eps=1e-5, norm_type="layer_norm")
        assert out.shape == hidden.shape

    def test_unknown_norm_type_raises(self):
        hidden = torch.randn(2, 8)
        weight = torch.ones(8)
        with pytest.raises(ValueError, match="Unknown norm_type"):
            apply_norm(hidden, weight, eps=1e-5, norm_type="batch_norm")

    def test_rms_norm_unit_weight_normalizes(self):
        """With weight=1, RMSNorm should produce roughly unit RMS per vector."""
        torch.manual_seed(42)
        hidden = torch.randn(10, 64)
        weight = torch.ones(64)
        out = apply_norm(hidden, weight, eps=1e-8, norm_type="rms_norm")
        rms = out.pow(2).mean(-1).sqrt()
        # Each vector's RMS should be ~1.0
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_layer_norm_with_bias(self):
        hidden = torch.randn(3, 16)
        weight = torch.ones(16)
        bias = torch.zeros(16)
        out = apply_norm(
            hidden, weight, eps=1e-5, norm_type="layer_norm", norm_bias=bias
        )
        # Should be zero-mean, unit-variance per vector (approximately)
        assert out.shape == hidden.shape
        assert torch.allclose(out.mean(-1), torch.zeros(3), atol=1e-4)

    def test_rms_norm_weight_scaling(self):
        """Scaling weight by 2 should scale output by 2."""
        torch.manual_seed(0)
        hidden = torch.randn(5, 32)
        w1 = torch.ones(32)
        w2 = torch.ones(32) * 2.0
        out1 = apply_norm(hidden, w1, eps=1e-8, norm_type="rms_norm")
        out2 = apply_norm(hidden, w2, eps=1e-8, norm_type="rms_norm")
        assert torch.allclose(out2, out1 * 2.0, atol=1e-5)

    def test_apply_norm_2d_input(self):
        """Works with 2D input (batch, hidden_dim)."""
        hidden = torch.randn(4, 16)
        weight = torch.ones(16)
        out = apply_norm(hidden, weight, eps=1e-5, norm_type="rms_norm")
        assert out.shape == (4, 16)


# ---------------------------------------------------------------------------
# compute_perplexity_from_activations (mocked cache)
# ---------------------------------------------------------------------------


class TestComputePerplexityFromActivations:
    """Tests for compute_perplexity_from_activations with mocked cache loading."""

    @pytest.fixture
    def mock_weights(self, tiny_model):
        """Load real weights from tiny model for consistent testing."""
        norm_weight, lm_head_weight, norm_config = download_lm_head_weights(
            tiny_model, device="cpu", dtype=torch.float32
        )
        return norm_weight, lm_head_weight, norm_config

    @pytest.fixture
    def mock_tokenizer(self, tiny_model):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tiny_model)

    def _make_mock_loader(self, hidden_dim, tokenizer):
        """Create a mock for load_layer_across_prompts that returns synthetic data."""

        def mock_load(model_name, prompts, layer):
            acts_list = []
            masks_list = []
            for prompt in prompts:
                tokens = tokenizer(prompt, return_tensors="pt")
                seq_len = tokens["input_ids"].shape[1]
                # (1, seq_len, hidden_dim)
                acts = torch.randn(1, seq_len, hidden_dim)
                mask = torch.ones(1, seq_len, dtype=torch.long)
                acts_list.append(acts)
                masks_list.append(mask)
            return acts_list, masks_list

        return mock_load

    def test_returns_list_of_tensors(self, tiny_model, mock_weights, mock_tokenizer):
        norm_weight, lm_head_weight, norm_config = mock_weights
        hidden_dim = norm_weight.shape[0]
        mock_load = self._make_mock_loader(hidden_dim, mock_tokenizer)

        prompts = ["Hello world", "Testing perplexity"]

        with patch("lmprobe.cache.load_layer_across_prompts", mock_load):
            results = compute_perplexity_from_activations(
                model_name=tiny_model,
                prompts=prompts,
                last_layer=0,
                device="cpu",
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                norm_config=norm_config,
                tokenizer=mock_tokenizer,
            )

        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, torch.Tensor)
            assert r.shape == (3,)  # mean, min, max

    def test_perplexity_values_are_positive(
        self, tiny_model, mock_weights, mock_tokenizer
    ):
        norm_weight, lm_head_weight, norm_config = mock_weights
        hidden_dim = norm_weight.shape[0]
        mock_load = self._make_mock_loader(hidden_dim, mock_tokenizer)

        with patch("lmprobe.cache.load_layer_across_prompts", mock_load):
            results = compute_perplexity_from_activations(
                model_name=tiny_model,
                prompts=["A longer test sentence for perplexity"],
                last_layer=0,
                device="cpu",
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                norm_config=norm_config,
                tokenizer=mock_tokenizer,
            )

        # Perplexity = exp(loss), so always > 0
        for r in results:
            assert (r > 0).all()

    def test_return_per_token(self, tiny_model, mock_weights, mock_tokenizer):
        norm_weight, lm_head_weight, norm_config = mock_weights
        hidden_dim = norm_weight.shape[0]
        mock_load = self._make_mock_loader(hidden_dim, mock_tokenizer)

        prompt = "Testing per-token perplexity output"

        with patch("lmprobe.cache.load_layer_across_prompts", mock_load):
            result = compute_perplexity_from_activations(
                model_name=tiny_model,
                prompts=[prompt],
                last_layer=0,
                device="cpu",
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                norm_config=norm_config,
                tokenizer=mock_tokenizer,
                return_per_token=True,
            )

        assert isinstance(result, tuple)
        assert len(result) == 3
        aggregates, per_token_ppl, token_ids = result
        assert len(aggregates) == 1
        assert len(per_token_ppl) == 1
        assert len(token_ids) == 1
        # per_token_ppl length should be seq_len - 1 (shifted)
        n_tokens = token_ids[0].shape[0]
        assert per_token_ppl[0].shape[0] == n_tokens - 1

    def test_single_token_prompt(self, tiny_model, mock_weights, mock_tokenizer):
        """A prompt that tokenizes to 1 token has no shift pairs -> ppl = [1,1,1]."""
        norm_weight, lm_head_weight, norm_config = mock_weights
        hidden_dim = norm_weight.shape[0]

        def mock_load_single(model_name, prompts, layer):
            # Return a single-token activation
            acts = torch.randn(1, 1, hidden_dim)
            mask = torch.ones(1, 1, dtype=torch.long)
            return [acts], [mask]

        # We need a tokenizer that returns 1 token for this to work.
        # We'll just mock the tokenizer too.
        class FakeTokenizer:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([[42]])}

        fake_tok = FakeTokenizer()

        with patch("lmprobe.cache.load_layer_across_prompts", mock_load_single):
            results = compute_perplexity_from_activations(
                model_name=tiny_model,
                prompts=["x"],
                last_layer=0,
                device="cpu",
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                norm_config=norm_config,
                tokenizer=fake_tok,
            )

        assert len(results) == 1
        assert torch.allclose(results[0], torch.tensor([1.0, 1.0, 1.0]))

    def test_auto_downloads_weights_when_none(self, tiny_model, mock_tokenizer):
        """When weights are None, they're downloaded automatically."""
        # Get hidden_dim from real weights to build correct mock
        norm_weight, lm_head_weight, norm_config = download_lm_head_weights(
            tiny_model, device="cpu", dtype=torch.float32
        )
        hidden_dim = norm_weight.shape[0]
        mock_load = self._make_mock_loader(hidden_dim, mock_tokenizer)

        with patch("lmprobe.cache.load_layer_across_prompts", mock_load):
            results = compute_perplexity_from_activations(
                model_name=tiny_model,
                prompts=["Test auto download"],
                last_layer=0,
                device="cpu",
                # norm_weight, lm_head_weight, norm_config all None
                tokenizer=mock_tokenizer,
            )

        assert len(results) == 1
        assert results[0].shape == (3,)

    def test_min_leq_mean_leq_max(self, tiny_model, mock_weights, mock_tokenizer):
        """Perplexity ordering: min <= mean <= max (exp is monotonic)."""
        norm_weight, lm_head_weight, norm_config = mock_weights
        hidden_dim = norm_weight.shape[0]
        mock_load = self._make_mock_loader(hidden_dim, mock_tokenizer)

        with patch("lmprobe.cache.load_layer_across_prompts", mock_load):
            results = compute_perplexity_from_activations(
                model_name=tiny_model,
                prompts=["A sentence with several tokens for ordering check"],
                last_layer=0,
                device="cpu",
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                norm_config=norm_config,
                tokenizer=mock_tokenizer,
            )

        ppl_mean, ppl_min, ppl_max = results[0]
        assert ppl_min <= ppl_mean
        # Note: mean of losses -> exp(mean_loss) is not necessarily <= exp(max_loss)
        # but ppl_min = exp(min_loss) <= exp(max_loss) = ppl_max always holds
        assert ppl_min <= ppl_max

    def test_single_token_return_per_token(self, tiny_model, mock_weights):
        """Single-token prompt with return_per_token=True returns empty per-token list."""
        norm_weight, lm_head_weight, norm_config = mock_weights
        hidden_dim = norm_weight.shape[0]

        def mock_load_single(model_name, prompts, layer):
            acts = torch.randn(1, 1, hidden_dim)
            mask = torch.ones(1, 1, dtype=torch.long)
            return [acts], [mask]

        class FakeTokenizer:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([[42]])}

        with patch("lmprobe.cache.load_layer_across_prompts", mock_load_single):
            result = compute_perplexity_from_activations(
                model_name=tiny_model,
                prompts=["x"],
                last_layer=0,
                device="cpu",
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                norm_config=norm_config,
                tokenizer=FakeTokenizer(),
                return_per_token=True,
            )

        aggregates, per_token_ppl, token_ids = result
        assert torch.allclose(aggregates[0], torch.tensor([1.0, 1.0, 1.0]))
        assert per_token_ppl[0].shape[0] == 0
        assert token_ids[0].shape[0] == 1

    def test_auto_loads_tokenizer_when_none(self, tiny_model, mock_weights):
        """When tokenizer is None, it's loaded automatically."""
        norm_weight, lm_head_weight, norm_config = mock_weights
        hidden_dim = norm_weight.shape[0]

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tiny_model)
        mock_load = self._make_mock_loader(hidden_dim, tokenizer)

        with patch("lmprobe.cache.load_layer_across_prompts", mock_load):
            results = compute_perplexity_from_activations(
                model_name=tiny_model,
                prompts=["Auto tokenizer test"],
                last_layer=0,
                device="cpu",
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                norm_config=norm_config,
                # tokenizer=None (default)
            )

        assert len(results) == 1
        assert results[0].shape == (3,)


class TestDownloadLmHeadWeightsEdgeCases:
    """Edge-case tests for download_lm_head_weights."""

    def test_missing_weights_raises_key_error(self):
        """KeyError raised when needed weights are not found in safetensors."""
        from unittest.mock import MagicMock

        # Mock a safetensors file that has norm but no lm_head or embed
        mock_safe_open = MagicMock()
        mock_safe_open.__enter__ = MagicMock(return_value=mock_safe_open)
        mock_safe_open.__exit__ = MagicMock(return_value=False)
        mock_safe_open.keys.return_value = ["model.norm.weight"]
        mock_safe_open.get_tensor.return_value = torch.ones(16)

        with (
            patch("transformers.AutoConfig.from_pretrained") as mock_from_pretrained,
            patch("huggingface_hub.hf_hub_download") as mock_dl,
            patch("safetensors.safe_open", return_value=mock_safe_open),
        ):
            mock_from_pretrained.return_value = SimpleNamespace(
                rms_norm_eps=1e-5, tie_word_embeddings=False
            )
            # Make index download fail so it falls through to single file
            mock_dl.side_effect = [Exception("no index"), "/fake/model.safetensors"]

            with pytest.raises(KeyError, match="Could not find lm_head"):
                download_lm_head_weights("fake/model", device="cpu")
