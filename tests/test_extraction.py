"""Tests for extraction module."""

from unittest.mock import MagicMock, patch

import pytest
import torch

pytestmark = pytest.mark.nnsight

# ---------------------------------------------------------------------------
# _require_nnsight
# ---------------------------------------------------------------------------


class TestRequireNnsight:
    def test_import_error_message(self):
        """_require_nnsight gives a clear error when nnsight is not installed."""
        from lmprobe.extraction import _require_nnsight

        with patch.dict("sys.modules", {"nnsight": None}):
            with pytest.raises(ImportError, match="nnsight is required"):
                _require_nnsight()

    def test_success_when_installed(self):
        """_require_nnsight succeeds when nnsight is installed."""
        from lmprobe.extraction import _require_nnsight

        result = _require_nnsight()
        assert result is not None


# ---------------------------------------------------------------------------
# resolve_layers
# ---------------------------------------------------------------------------


class TestResolveLayers:
    """Tests for resolve_layers."""

    def test_single_positive_int(self):
        from lmprobe.extraction import resolve_layers

        assert resolve_layers(0, 32) == [0]
        assert resolve_layers(15, 32) == [15]

    def test_single_negative_int(self):
        from lmprobe.extraction import resolve_layers

        assert resolve_layers(-1, 32) == [31]
        assert resolve_layers(-4, 32) == [28]

    def test_out_of_range_raises(self):
        from lmprobe.extraction import resolve_layers

        with pytest.raises(ValueError, match="out of range"):
            resolve_layers(32, 32)
        with pytest.raises(ValueError, match="out of range"):
            resolve_layers(-33, 32)

    def test_list_of_ints(self):
        from lmprobe.extraction import resolve_layers

        assert resolve_layers([0, 15, -1], 32) == [0, 15, 31]

    def test_middle(self):
        from lmprobe.extraction import resolve_layers

        result = resolve_layers("middle", 12)
        # 12 // 3 = 4, so middle is range(4, 8)
        assert result == [4, 5, 6, 7]

    def test_last(self):
        from lmprobe.extraction import resolve_layers

        assert resolve_layers("last", 32) == [31]

    def test_all(self):
        from lmprobe.extraction import resolve_layers

        assert resolve_layers("all", 4) == [0, 1, 2, 3]

    def test_auto_defaults(self):
        from lmprobe.extraction import resolve_layers

        result = resolve_layers("auto", 32)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_fast_auto(self):
        from lmprobe.extraction import resolve_layers

        result = resolve_layers("fast_auto", 32)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_unknown_string_raises(self):
        from lmprobe.extraction import resolve_layers

        with pytest.raises(ValueError, match="Unknown layer specification"):
            resolve_layers("bogus", 32)

    def test_list_out_of_range_raises(self):
        from lmprobe.extraction import resolve_layers

        with pytest.raises(ValueError, match="out of range"):
            resolve_layers([50], 32)


# ---------------------------------------------------------------------------
# resolve_auto_candidates
# ---------------------------------------------------------------------------


class TestResolveAutoCandidates:
    """Tests for resolve_auto_candidates."""

    def test_none_defaults(self):
        from lmprobe.extraction import resolve_auto_candidates

        result = resolve_auto_candidates(None, 32)
        assert len(result) == 3
        assert result == [7, 15, 23]

    def test_fractional(self):
        from lmprobe.extraction import resolve_auto_candidates

        result = resolve_auto_candidates([0.0, 0.5, 1.0], 10)
        assert 0 in result
        assert 9 in result

    def test_integer_candidates(self):
        from lmprobe.extraction import resolve_auto_candidates

        result = resolve_auto_candidates([2, 5, 8], 10)
        assert result == [2, 5, 8]

    def test_negative_integer_candidates(self):
        from lmprobe.extraction import resolve_auto_candidates

        result = resolve_auto_candidates([-1, -4], 10)
        assert result == [6, 9]

    def test_empty_raises(self):
        from lmprobe.extraction import resolve_auto_candidates

        with pytest.raises(ValueError, match="cannot be empty"):
            resolve_auto_candidates([], 10)

    def test_out_of_range_integer_raises(self):
        from lmprobe.extraction import resolve_auto_candidates

        with pytest.raises(ValueError, match="out of range"):
            resolve_auto_candidates([50], 10)

    def test_deduplication(self):
        from lmprobe.extraction import resolve_auto_candidates

        result = resolve_auto_candidates([0.5, 0.5], 10)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# get_num_layers_from_config
# ---------------------------------------------------------------------------


class TestGetNumLayersFromConfig:
    def test_tiny_model(self, tiny_model):
        from lmprobe.extraction import get_num_layers_from_config

        n = get_num_layers_from_config(tiny_model)
        assert isinstance(n, int)
        assert n > 0

    def test_unknown_config_raises(self):
        """Config with no recognized layer count field raises ValueError."""
        from lmprobe.extraction import get_num_layers_from_config

        with patch("transformers.AutoConfig.from_pretrained") as mock_from:
            mock_config = MagicMock(spec=[])
            mock_config.to_dict = MagicMock(return_value={"foo": "bar"})
            mock_from.return_value = mock_config

            with pytest.raises(ValueError, match="Could not determine layer count"):
                get_num_layers_from_config("fake-model")


# ---------------------------------------------------------------------------
# clear_model_cache / get_cached_model
# ---------------------------------------------------------------------------


class TestModelCache:
    def test_get_cached_model(self, tiny_model):
        from lmprobe.extraction import get_cached_model

        model = get_cached_model(tiny_model, device="cpu", remote=False)
        assert model is not None

    def test_cached_returns_same(self, tiny_model):
        from lmprobe.extraction import get_cached_model

        m1 = get_cached_model(tiny_model, device="cpu", remote=False)
        m2 = get_cached_model(tiny_model, device="cpu", remote=False)
        assert m1 is m2

    def test_clears(self, tiny_model):
        from lmprobe.extraction import _MODEL_CACHE, clear_model_cache, get_cached_model

        get_cached_model(tiny_model, device="cpu", remote=False)
        clear_model_cache()
        assert len(_MODEL_CACHE) == 0


# ---------------------------------------------------------------------------
# configure_remote
# ---------------------------------------------------------------------------


class TestConfigureRemote:
    def test_missing_key_raises(self, monkeypatch):
        from lmprobe.extraction import configure_remote

        monkeypatch.delenv("NDIF_API_KEY", raising=False)
        with pytest.raises(OSError, match="NDIF_API_KEY"):
            configure_remote()

    def test_with_key_sets_config(self, monkeypatch):
        from lmprobe.extraction import configure_remote

        monkeypatch.setenv("NDIF_API_KEY", "test-key-123")
        mock_nnsight = MagicMock()
        with patch("lmprobe.extraction._require_nnsight", return_value=mock_nnsight):
            configure_remote()
        assert mock_nnsight.CONFIG.API.APIKEY == "test-key-123"


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_load_local_cpu(self, tiny_model):
        from lmprobe.extraction import load_model

        model = load_model(tiny_model, device="cpu", remote=False)
        assert model is not None

    def test_load_device_auto(self, tiny_model):
        from lmprobe.extraction import load_model

        model = load_model(tiny_model, device="auto", remote=False)
        assert model is not None

    def test_load_device_specific(self, tiny_model):
        from lmprobe.extraction import load_model

        model = load_model(tiny_model, device="cpu", remote=False)
        assert model is not None

    def test_load_remote_creates_stub(self, tiny_model):
        """Remote mode creates a dispatch=False stub without loading weights."""
        from lmprobe.extraction import load_model

        model = load_model(tiny_model, device="cpu", remote=True)
        assert model is not None
        # Remote stubs should have dispatched=True
        assert model.dispatched is True

    def test_load_runtime_error_fallback(self, tiny_model):
        """RuntimeError with 'no kernel image' falls back to CPU."""
        from lmprobe.extraction import load_model

        nnsight_mod = MagicMock()
        call_count = 0

        def mock_lm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("no kernel image available")
            return MagicMock()

        nnsight_mod.LanguageModel = mock_lm

        with patch("lmprobe.extraction._require_nnsight", return_value=nnsight_mod), \
             patch("lmprobe._device_utils.check_cuda_compatibility"):
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model = load_model(tiny_model, device="cuda:0", remote=False)
            assert model is not None
            assert call_count == 2
            assert any("GPU detected but incompatible" in str(warning.message) for warning in w)

    def test_load_runtime_error_reraises_other(self, tiny_model):
        """RuntimeError without 'no kernel image' is re-raised."""
        from lmprobe.extraction import load_model

        nnsight_mod = MagicMock()
        nnsight_mod.LanguageModel.side_effect = RuntimeError("some other error")

        with patch("lmprobe.extraction._require_nnsight", return_value=nnsight_mod), \
             patch("lmprobe._device_utils.check_cuda_compatibility"):
            with pytest.raises(RuntimeError, match="some other error"):
                load_model(tiny_model, device="cuda:0", remote=False)


# ---------------------------------------------------------------------------
# _unwrap_proxy / _unwrap_layer_outputs
# ---------------------------------------------------------------------------


class TestUnwrapProxy:
    def test_plain_tensor(self):
        from lmprobe.extraction import _unwrap_proxy

        t = torch.randn(2, 3)
        assert _unwrap_proxy(t) is t

    def test_proxy_with_value(self):
        from lmprobe.extraction import _unwrap_proxy

        proxy = MagicMock()
        proxy.value = torch.randn(2, 3)
        result = _unwrap_proxy(proxy)
        assert torch.equal(result, proxy.value)


class TestUnwrapLayerOutputs:
    def test_plain_tensors(self):
        from lmprobe.extraction import _unwrap_layer_outputs

        tensors = [torch.randn(2, 5, 16), torch.randn(2, 5, 16)]
        result = _unwrap_layer_outputs(tensors)
        assert len(result) == 2
        for r in result:
            assert r.shape == (2, 5, 16)

    def test_tuple_outputs(self):
        from lmprobe.extraction import _unwrap_layer_outputs

        raw = [(torch.randn(2, 5, 16), torch.randn(2, 5, 16))]
        result = _unwrap_layer_outputs(raw)
        assert len(result) == 1
        assert result[0].shape == (2, 5, 16)

    def test_proxy_outputs(self):
        from lmprobe.extraction import _unwrap_layer_outputs

        proxy = MagicMock()
        proxy.value = torch.randn(2, 5, 16)
        result = _unwrap_layer_outputs([proxy])
        assert len(result) == 1
        assert torch.equal(result[0], proxy.value)


# ---------------------------------------------------------------------------
# _extract_batch (local, via nnsight)
# ---------------------------------------------------------------------------


class TestExtractBatch:
    def test_single_prompt(self, tiny_model):
        from lmprobe.extraction import _extract_batch, get_cached_model

        model = get_cached_model(tiny_model, device="cpu", remote=False)
        acts, mask = _extract_batch(model, ["hello world"], [0], remote=False)
        assert acts.ndim == 3
        assert mask.ndim == 2
        assert acts.shape[0] == 1
        assert mask.shape[0] == 1

    def test_multiple_prompts(self, tiny_model):
        from lmprobe.extraction import _extract_batch, get_cached_model

        model = get_cached_model(tiny_model, device="cpu", remote=False)
        prompts = ["hello", "goodbye world"]
        acts, mask = _extract_batch(model, prompts, [0], remote=False)
        assert acts.shape[0] == 2
        assert mask.shape[0] == 2

    def test_multiple_layers_concatenated(self, tiny_model):
        from lmprobe.extraction import (
            _extract_batch,
            get_cached_model,
            get_num_layers_from_config,
        )

        model = get_cached_model(tiny_model, device="cpu", remote=False)
        num_layers = get_num_layers_from_config(tiny_model)
        if num_layers < 2:
            pytest.skip("Model has fewer than 2 layers")
        acts_one, _ = _extract_batch(model, ["test"], [0], remote=False)
        acts_two, _ = _extract_batch(model, ["test"], [0, 1], remote=False)
        assert acts_two.shape[-1] == 2 * acts_one.shape[-1]


# ---------------------------------------------------------------------------
# extract_activations (full pipeline with batching)
# ---------------------------------------------------------------------------


class TestExtractActivations:
    def test_basic(self, tiny_model):
        from lmprobe.extraction import extract_activations, get_cached_model

        model = get_cached_model(tiny_model, device="cpu", remote=False)
        prompts = ["hello world", "goodbye"]
        acts, mask = extract_activations(
            model, prompts, [0], remote=False, batch_size=8
        )
        assert acts.shape[0] == 2
        assert mask.shape[0] == 2

    def test_batching_with_small_batch_size(self, tiny_model):
        from lmprobe.extraction import extract_activations, get_cached_model

        model = get_cached_model(tiny_model, device="cpu", remote=False)
        prompts = ["short", "a much longer prompt that has more tokens", "medium length"]
        acts, mask = extract_activations(
            model, prompts, [0], remote=False, batch_size=1
        )
        assert acts.shape[0] == 3
        assert mask.shape[0] == 3
        assert acts.shape[1] == mask.shape[1]

    def test_batch_size_larger_than_prompts(self, tiny_model):
        from lmprobe.extraction import extract_activations, get_cached_model

        model = get_cached_model(tiny_model, device="cpu", remote=False)
        prompts = ["hello"]
        acts, mask = extract_activations(
            model, prompts, [0], remote=False, batch_size=100
        )
        assert acts.shape[0] == 1

    def test_padding_across_batches_mocked(self):
        """Test padding logic using mocked _extract_batch to avoid nnsight."""
        from lmprobe.extraction import extract_activations

        model = MagicMock()
        hidden_dim = 16
        call_count = 0

        def mock_extract_batch(model, prompts, layer_indices, remote=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First batch: seq_len=5
                return (
                    torch.randn(1, 5, hidden_dim),
                    torch.ones(1, 5, dtype=torch.long),
                )
            else:
                # Second batch: seq_len=10 (longer)
                return (
                    torch.randn(1, 10, hidden_dim),
                    torch.ones(1, 10, dtype=torch.long),
                )

        with patch("lmprobe.extraction._extract_batch", side_effect=mock_extract_batch):
            acts, mask = extract_activations(
                model, ["short", "longer prompt"], [0], remote=False, batch_size=1
            )

        assert acts.shape == (2, 10, hidden_dim)
        assert mask.shape == (2, 10)
        # First batch should have been padded: last 5 positions masked out
        assert mask[0, :5].sum() == 5
        assert mask[0, 5:].sum() == 0

    def test_no_padding_needed_mocked(self):
        """Test when all batches have the same seq_len (no padding)."""
        from lmprobe.extraction import extract_activations

        model = MagicMock()
        hidden_dim = 16

        def mock_extract_batch(model, prompts, layer_indices, remote=False):
            return (
                torch.randn(1, 8, hidden_dim),
                torch.ones(1, 8, dtype=torch.long),
            )

        with patch("lmprobe.extraction._extract_batch", side_effect=mock_extract_batch):
            acts, mask = extract_activations(
                model, ["a", "b"], [0], remote=False, batch_size=1
            )

        assert acts.shape == (2, 8, hidden_dim)
        assert mask.shape == (2, 8)


# ---------------------------------------------------------------------------
# compute_perplexity_from_logits
# ---------------------------------------------------------------------------


class TestComputePerplexityFromLogits:
    def test_basic_shape(self):
        from lmprobe.extraction import compute_perplexity_from_logits

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        result = compute_perplexity_from_logits(logits, input_ids, attention_mask)
        assert result.shape == (batch_size, 3)
        assert (result > 0).all()

    def test_with_padding(self):
        from lmprobe.extraction import compute_perplexity_from_logits

        batch_size, seq_len, vocab_size = 2, 8, 50
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        attention_mask[1, 5:] = 0
        result = compute_perplexity_from_logits(logits, input_ids, attention_mask)
        assert result.shape == (batch_size, 3)

    def test_return_per_token(self):
        from lmprobe.extraction import compute_perplexity_from_logits

        batch_size, seq_len, vocab_size = 2, 10, 50
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        result = compute_perplexity_from_logits(
            logits, input_ids, attention_mask, return_per_token=True
        )
        assert len(result) == 3
        aggregates, per_token_list, token_ids_list = result
        assert aggregates.shape == (batch_size, 3)
        assert len(per_token_list) == batch_size
        assert len(token_ids_list) == batch_size

    def test_empty_sequence_after_shift(self):
        from lmprobe.extraction import compute_perplexity_from_logits

        logits = torch.randn(1, 1, 50)
        input_ids = torch.randint(0, 50, (1, 1))
        attention_mask = torch.ones(1, 1, dtype=torch.long)
        result = compute_perplexity_from_logits(logits, input_ids, attention_mask)
        assert result.shape == (1, 3)
        assert torch.allclose(result, torch.tensor([[1.0, 1.0, 1.0]]))

    def test_empty_sequence_return_per_token(self):
        from lmprobe.extraction import compute_perplexity_from_logits

        logits = torch.randn(1, 1, 50)
        input_ids = torch.randint(0, 50, (1, 1))
        attention_mask = torch.ones(1, 1, dtype=torch.long)
        aggregates, ppl_list, ids_list = compute_perplexity_from_logits(
            logits, input_ids, attention_mask, return_per_token=True
        )
        assert len(ppl_list[0]) == 0
        assert len(ids_list) == 1

    def test_device_mismatch(self):
        """Test that device mismatch branch is handled."""
        from lmprobe.extraction import compute_perplexity_from_logits

        logits = torch.randn(1, 5, 50)
        input_ids = torch.randint(0, 50, (1, 5))
        attention_mask = torch.ones(1, 5, dtype=torch.long)
        # All on CPU, so no device mismatch, but verify it runs
        result = compute_perplexity_from_logits(logits, input_ids, attention_mask)
        assert result.shape == (1, 3)

    def test_min_max_ppl_ordering(self):
        """min_ppl <= mean_ppl <= max_ppl for each prompt."""
        from lmprobe.extraction import compute_perplexity_from_logits

        logits = torch.randn(3, 15, 100)
        input_ids = torch.randint(0, 100, (3, 15))
        attention_mask = torch.ones(3, 15, dtype=torch.long)
        result = compute_perplexity_from_logits(logits, input_ids, attention_mask)
        for i in range(3):
            mean_ppl, min_ppl, max_ppl = result[i]
            assert min_ppl <= mean_ppl <= max_ppl


# ---------------------------------------------------------------------------
# _build_remote_extract_fn
# ---------------------------------------------------------------------------


class TestBuildRemoteExtractFnTopK:
    """Tests for _build_remote_extract_fn with logit_top_k."""

    def test_generated_code_contains_topk(self):
        from lmprobe.extraction import _build_remote_extract_fn

        fn = _build_remote_extract_fn(
            layer_indices=[0, 1], with_logits=True, logit_top_k=100
        )
        assert callable(fn)

    def test_generated_code_without_topk(self):
        from lmprobe.extraction import _build_remote_extract_fn

        fn = _build_remote_extract_fn(
            layer_indices=[0], with_logits=True, logit_top_k=None
        )
        assert callable(fn)

    def test_generated_code_no_logits(self):
        from lmprobe.extraction import _build_remote_extract_fn

        fn = _build_remote_extract_fn(
            layer_indices=[0], with_logits=False, logit_top_k=100
        )
        assert callable(fn)

    def test_temp_file_cleaned_up_after_call(self):
        from lmprobe.extraction import _build_remote_extract_fn

        fn = _build_remote_extract_fn(
            layer_indices=[0], with_logits=False
        )
        assert callable(fn)

    def test_multiple_layers(self):
        from lmprobe.extraction import _build_remote_extract_fn

        fn = _build_remote_extract_fn(
            layer_indices=[0, 5, 10, 15], with_logits=True, logit_top_k=50
        )
        assert callable(fn)


# ---------------------------------------------------------------------------
# _extract_batch_with_logits (local)
# ---------------------------------------------------------------------------


class TestExtractBatchWithLogitsTopK:
    """Tests for _extract_batch_with_logits with logit_top_k."""

    def test_local_returns_4tuple_none_indices(self, tiny_model):
        from lmprobe.extraction import _extract_batch_with_logits, get_cached_model

        model = get_cached_model(tiny_model, device="cpu", remote=False)
        prompts = ["hello world"]
        layer_indices = [0]

        result = _extract_batch_with_logits(
            model, prompts, layer_indices, remote=False
        )

        assert len(result) == 4
        acts, mask, logits, indices = result
        assert acts.ndim == 3
        assert mask.ndim == 2
        assert logits.ndim == 3
        assert indices is None

    def test_local_topk_ignored(self, tiny_model):
        from lmprobe.extraction import _extract_batch_with_logits, get_cached_model

        model = get_cached_model(tiny_model, device="cpu", remote=False)
        prompts = ["test prompt"]
        layer_indices = [0]

        result = _extract_batch_with_logits(
            model, prompts, layer_indices, remote=False, logit_top_k=10
        )

        acts, mask, logits, indices = result
        assert indices is None
        assert logits.shape[-1] > 10

    def test_local_multiple_layers(self, tiny_model):
        from lmprobe.extraction import (
            _extract_batch_with_logits,
            get_cached_model,
            get_num_layers_from_config,
        )

        model = get_cached_model(tiny_model, device="cpu", remote=False)
        prompts = ["test"]
        num_layers = get_num_layers_from_config(tiny_model)
        if num_layers < 2:
            pytest.skip("Model has fewer than 2 layers")

        acts1, _, _, _ = _extract_batch_with_logits(
            model, prompts, [0], remote=False
        )
        acts2, _, _, _ = _extract_batch_with_logits(
            model, prompts, [0, 1], remote=False
        )
        assert acts2.shape[-1] == 2 * acts1.shape[-1]


# ---------------------------------------------------------------------------
# ActivationExtractor class
# ---------------------------------------------------------------------------


class TestActivationExtractor:
    def test_init(self, tiny_model):
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers=-1,
            batch_size=4,
            remote=False,
            backend="local",
        )
        assert extractor.model_name == tiny_model
        assert extractor.device == "cpu"
        assert extractor.batch_size == 4
        assert extractor.remote is False
        assert extractor.backend_name == "local"

    def test_layer_indices_property(self, tiny_model):
        from lmprobe.extraction import ActivationExtractor, get_num_layers_from_config

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers=-1,
            remote=False,
            backend="local",
        )
        indices = extractor.layer_indices
        assert isinstance(indices, list)
        assert len(indices) == 1
        num_layers = get_num_layers_from_config(tiny_model)
        assert indices[0] == num_layers - 1

    def test_layer_indices_cached(self, tiny_model):
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers=-1,
            remote=False,
            backend="local",
        )
        indices1 = extractor.layer_indices
        indices2 = extractor.layer_indices
        assert indices1 is indices2

    def test_num_layers_property(self, tiny_model):
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers=[0, 1],
            remote=False,
            backend="local",
        )
        assert extractor.num_layers == 2

    def test_model_property(self, tiny_model):
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers=-1,
            remote=False,
            backend="local",
        )
        model = extractor.model
        assert model is not None

    def test_tokenizer_property(self, tiny_model):
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers=-1,
            remote=False,
            backend="local",
        )
        tokenizer = extractor.tokenizer
        assert tokenizer is not None

    def test_extract_batch(self, tiny_model):
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers=-1,
            remote=False,
            backend="local",
        )
        acts, mask = extractor.extract_batch(
            ["hello world"], extractor.layer_indices
        )
        assert acts.ndim == 3
        assert mask.ndim == 2

    def test_extract_batch_with_logits(self, tiny_model):
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers=-1,
            remote=False,
            backend="local",
        )
        result = extractor.extract_batch_with_logits(
            ["hello world"], extractor.layer_indices
        )
        assert len(result) == 4
        acts, mask, logits, indices = result
        assert acts.ndim == 3
        assert logits.ndim == 3

    def test_extract_method_nnsight(self, tiny_model):
        """Test the high-level extract() method via nnsight backend."""
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers=-1,
            remote=False,
            backend="nnsight",
        )
        acts, mask = extractor.extract(["hello world", "test"])
        assert acts.shape[0] == 2
        assert mask.shape[0] == 2

    def test_extract_with_explicit_layers_nnsight(self, tiny_model):
        """Test extract() with explicit layer override via nnsight backend."""
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers=-1,
            remote=False,
            backend="nnsight",
        )
        acts, mask = extractor.extract(["hello"], layers=[0])
        assert acts.shape[0] == 1

    def test_auto_candidates(self, tiny_model):
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers="auto",
            auto_candidates=[0.5],
            remote=False,
            backend="local",
        )
        indices = extractor.layer_indices
        assert len(indices) == 1

    def test_layers_middle(self, tiny_model):
        from lmprobe.extraction import ActivationExtractor

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers="middle",
            remote=False,
            backend="local",
        )
        indices = extractor.layer_indices
        assert len(indices) > 0

    def test_layers_all(self, tiny_model):
        from lmprobe.extraction import ActivationExtractor, get_num_layers_from_config

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers="all",
            remote=False,
            backend="local",
        )
        num_layers = get_num_layers_from_config(tiny_model)
        assert extractor.num_layers == num_layers

    def test_layers_last(self, tiny_model):
        from lmprobe.extraction import ActivationExtractor, get_num_layers_from_config

        extractor = ActivationExtractor(
            model_name=tiny_model,
            device="cpu",
            layers="last",
            remote=False,
            backend="local",
        )
        num_layers = get_num_layers_from_config(tiny_model)
        assert extractor.layer_indices == [num_layers - 1]
