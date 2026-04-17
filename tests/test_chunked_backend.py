"""Tests for the ChunkedLocalBackend.

Verifies that layer-chunked extraction produces identical results to
the standard LocalBackend, and that the backend integrates correctly
with the pluggable backend interface.
"""

import pytest
import torch
from conftest import TEST_PROMPTS

from lmprobe.backends import (
    ChunkedLocalBackend,
    ExtractionBackend,
    LocalBackend,
    resolve_backend,
)

# ── Backend interface ────────────────────────────────────────────────────────


class TestChunkedBackendInterface:
    """Verify ChunkedLocalBackend implements ExtractionBackend."""

    def test_is_extraction_backend(self):
        assert issubclass(ChunkedLocalBackend, ExtractionBackend)

    def test_resolve_backend_chunked(self, tiny_model):
        backend = resolve_backend("chunked", tiny_model, "cpu")
        assert isinstance(backend, ChunkedLocalBackend)

    def test_resolve_backend_chunked_rejects_remote(self, tiny_model):
        with pytest.raises(ValueError, match="does not support remote"):
            resolve_backend("chunked", tiny_model, "cpu", remote=True)

    def test_resolve_backend_chunked_default_dtype(self, tiny_model):
        backend = resolve_backend("chunked", tiny_model, "cpu")
        assert isinstance(backend, ChunkedLocalBackend)
        assert backend.dtype == torch.bfloat16

    def test_resolve_backend_chunked_explicit_dtype(self, tiny_model):
        backend = resolve_backend(
            "chunked", tiny_model, "cpu", dtype=torch.float32,
        )
        assert backend.dtype == torch.float32

    def test_resolve_backend_chunked_chunk_size(self, tiny_model):
        backend = resolve_backend(
            "chunked", tiny_model, "cpu", chunk_size=1,
        )
        assert isinstance(backend, ChunkedLocalBackend)
        assert backend._chunk_size == 1

    def test_model_property_raises(self, tiny_model):
        backend = ChunkedLocalBackend(tiny_model, "cpu", chunk_size=1)
        with pytest.raises(RuntimeError, match="does not keep the full model"):
            _ = backend.model

    def test_tokenizer_property(self, tiny_model):
        backend = ChunkedLocalBackend(tiny_model, "cpu", chunk_size=1)
        assert backend.tokenizer is not None
        assert backend.tokenizer.pad_token is not None

    def test_attn_implementation_default_is_sdpa(self, tiny_model):
        backend = ChunkedLocalBackend(tiny_model, "cpu", chunk_size=1)
        assert backend._attn_implementation == "sdpa"

    def test_attn_implementation_override(self, tiny_model):
        backend = ChunkedLocalBackend(
            tiny_model, "cpu", chunk_size=1, attn_implementation="eager",
        )
        assert backend._attn_implementation == "eager"

    def test_attn_implementation_propagates_to_loaded_model(self, tiny_model):
        backend = ChunkedLocalBackend(
            tiny_model, "cpu", chunk_size=1, attn_implementation="eager",
        )
        model = backend._load_full_model_cpu()
        # Most transformers models expose the resolved attention impl on
        # config._attn_implementation after load.
        resolved = getattr(model.config, "_attn_implementation", None)
        assert resolved == "eager", f"expected eager, got {resolved!r}"


# ── Correctness: chunked matches local ──────────────────────────────────────


class TestChunkedMatchesLocal:
    """The critical correctness tests: chunked output must match LocalBackend."""

    @pytest.fixture
    def local_backend(self, tiny_model):
        return LocalBackend(tiny_model, "cpu", dtype=torch.float32)

    @pytest.fixture
    def chunked_backend(self, tiny_model):
        return ChunkedLocalBackend(
            tiny_model, "cpu", dtype=torch.float32, chunk_size=1,
        )

    def test_extract_batch_matches(self, local_backend, chunked_backend):
        """Chunked extraction produces identical activations to full-model."""
        prompts = TEST_PROMPTS
        layer_indices = [0, 1]

        acts_local, mask_local = local_backend.extract_batch(
            prompts, layer_indices,
        )
        acts_chunked, mask_chunked = chunked_backend.extract_batch(
            prompts, layer_indices,
        )

        assert acts_local.shape == acts_chunked.shape
        assert mask_local.shape == mask_chunked.shape
        torch.testing.assert_close(acts_local, acts_chunked, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(mask_local, mask_chunked)

    def test_extract_batch_single_layer(self, local_backend, chunked_backend):
        """Single layer extraction matches."""
        prompts = TEST_PROMPTS
        layer_indices = [1]

        acts_local, mask_local = local_backend.extract_batch(
            prompts, layer_indices,
        )
        acts_chunked, mask_chunked = chunked_backend.extract_batch(
            prompts, layer_indices,
        )

        assert acts_local.shape == acts_chunked.shape
        torch.testing.assert_close(acts_local, acts_chunked, atol=1e-4, rtol=1e-4)

    def test_extract_with_logits_matches(self, local_backend, chunked_backend):
        """Logits match between chunked and local backends."""
        prompts = TEST_PROMPTS
        layer_indices = [0, 1]

        acts_l, mask_l, logits_l, _ = local_backend.extract_batch_with_logits(
            prompts, layer_indices,
        )
        acts_c, mask_c, logits_c, _ = chunked_backend.extract_batch_with_logits(
            prompts, layer_indices,
        )

        assert acts_l.shape == acts_c.shape
        torch.testing.assert_close(acts_l, acts_c, atol=1e-4, rtol=1e-4)
        assert logits_l is not None
        assert logits_c is not None
        assert logits_l.shape == logits_c.shape
        # Logits tolerance is higher because eager vs SDPA attention
        # implementations have small numerical differences.
        torch.testing.assert_close(logits_l, logits_c, atol=1e-3, rtol=1e-3)


# ── Shape and basic functionality ───────────────────────────────────────────


class TestChunkedExtraction:
    """Shape and basic functionality tests."""

    @pytest.fixture
    def backend(self, tiny_model):
        return ChunkedLocalBackend(
            tiny_model, "cpu", dtype=torch.float32, chunk_size=1,
        )

    def test_extract_batch_shapes(self, backend):
        prompts = ["Hello world", "Test prompt"]
        layer_indices = [0, 1]

        acts, mask = backend.extract_batch(prompts, layer_indices)

        assert acts.dim() == 3  # (batch, seq, hidden_dim * num_layers)
        assert acts.shape[0] == 2  # batch size
        assert mask.dim() == 2  # (batch, seq)
        assert mask.shape[0] == 2

    def test_extract_batch_with_logits_shapes(self, backend):
        prompts = ["Hello world"]
        layer_indices = [0]

        acts, mask, logits, logits_indices = backend.extract_batch_with_logits(
            prompts, layer_indices,
        )

        assert acts is not None
        assert acts.shape[0] == 1
        assert logits is not None
        assert logits.dim() == 3  # (batch, seq, vocab)
        assert logits_indices is None

    def test_extract_batch_extended(self, backend):
        from lmprobe.activation_types import ExtractionSpec

        prompts = ["Hello world"]
        spec = ExtractionSpec(
            hidden_layers=[0, 1],
            include_logits=True,
        )

        result = backend.extract_batch_extended(prompts, spec)

        assert result.activations is not None
        assert result.activations.shape[0] == 1
        assert result.attention_mask is not None
        assert result.logits is not None
        assert result.router_logits is None  # tiny model is not MoE

    def test_chunk_size_larger_than_model(self, tiny_model):
        """chunk_size larger than num_layers works (no chunking needed)."""
        backend = ChunkedLocalBackend(
            tiny_model, "cpu", dtype=torch.float32, chunk_size=100,
        )
        acts, mask = backend.extract_batch(["Hello"], [0])
        assert acts is not None

    def test_chunk_size_auto_on_cpu(self, tiny_model):
        """Auto chunk size on CPU returns num_layers (no chunking)."""
        from lmprobe.backends import _estimate_chunk_size

        cs = _estimate_chunk_size(tiny_model, "cpu", torch.float32)
        from lmprobe.extraction import get_num_layers_from_config

        num_layers = get_num_layers_from_config(tiny_model)
        assert cs == num_layers
