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


# ── scan_forward: memmap-backed batch_hidden_states (spec 004) ──────────────


class TestScanForwardMemmap:
    """Spec 004: cross-chunk residuals live in an np.memmap, not a list of
    CPU tensors. These tests verify the memmap path is correct under
    actual chunking (chunk_size < num_layers) and produces results
    indistinguishable from a single-chunk run."""

    def _scan(self, tiny_model, chunk_size: int, prompts: list[str]):
        backend = ChunkedLocalBackend(
            tiny_model, "cpu", dtype=torch.float32, chunk_size=chunk_size,
        )
        return backend.scan_forward(
            prompts=prompts,
            signals=["attn_delta", "mlp_delta"],
            n_components=4,
            batch_size=2,
        )

    def test_chunked_scan_matches_single_chunk(self, tiny_model):
        """Running with chunk_size=1 (forces memmap round-trip between
        every layer) produces identical projections and bases to
        chunk_size=num_layers (single chunk, no cross-chunk write-back)."""
        from lmprobe.extraction import get_num_layers_from_config

        num_layers = get_num_layers_from_config(tiny_model)
        prompts = ["The dog ran fast", "A cat sat quietly", "Fetch the ball"]

        _, bases_full, proj_full, coords_full, _, _, _, _ = self._scan(
            tiny_model, chunk_size=num_layers, prompts=prompts,
        )
        _, bases_chunk, proj_chunk, coords_chunk, _, _, _, _ = self._scan(
            tiny_model, chunk_size=1, prompts=prompts,
        )

        import numpy as np

        # With dtype=fp32 the memmap round-trip is byte-identical, so
        # PCA sees the same input and produces the same output.
        np.testing.assert_array_equal(proj_full, proj_chunk)

        for sig in bases_full:
            np.testing.assert_array_equal(bases_full[sig], bases_chunk[sig])

        for key in coords_full:
            np.testing.assert_array_equal(coords_full[key], coords_chunk[key])

    def test_memmap_file_is_cleaned_up(self, tiny_model, monkeypatch):
        """The memmap's backing file lives under a TemporaryDirectory
        scoped to scan_forward — no temp files should persist after
        the call returns."""
        import tempfile

        created_dirs: list[str] = []
        real_tempdir = tempfile.TemporaryDirectory

        class _TrackingTempDir(real_tempdir):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created_dirs.append(self.name)

        monkeypatch.setattr(tempfile, "TemporaryDirectory", _TrackingTempDir)

        self._scan(tiny_model, chunk_size=1, prompts=["One", "Two"])

        import os
        # At least one TemporaryDirectory was used, and all are gone.
        assert created_dirs, "scan_forward did not open a TemporaryDirectory"
        for d in created_dirs:
            assert not os.path.exists(d), f"temp dir leaked: {d}"

    def test_bf16_path_roundtrips(self, tiny_model):
        """Spec 004 stores bf16 residuals in a uint16 memmap container.
        Compare chunked (cross-chunk memmap roundtrip) to single-chunk
        (no roundtrip) at bf16. Bit-identical isn't achievable at bf16
        because sklearn's PCA rounds fp32 intermediates differently
        across call orderings, but the two paths should agree within
        bf16-ish tolerance. A broken `.view(torch.bfloat16)` typically
        blows up well past this."""
        import numpy as np

        from lmprobe.extraction import get_num_layers_from_config

        num_layers = get_num_layers_from_config(tiny_model)
        prompts = ["One short", "Two short", "Three short"]

        def run(chunk_size):
            backend = ChunkedLocalBackend(
                tiny_model, "cpu", dtype=torch.bfloat16, chunk_size=chunk_size,
            )
            return backend.scan_forward(
                prompts=prompts,
                signals=["attn_delta"],
                n_components=2,
                batch_size=2,
            )

        _, bases_single, proj_single, _, _, _, _, _ = run(num_layers)
        _, bases_chunk, proj_chunk, _, _, _, _, _ = run(1)

        # bf16 stored in the memmap; compare as fp32.
        assert np.isfinite(proj_chunk).all()
        assert np.isfinite(bases_chunk["attn_delta"]).all()
        np.testing.assert_allclose(
            proj_chunk.astype(np.float32),
            proj_single.astype(np.float32),
            atol=5e-2, rtol=5e-2,
        )
        np.testing.assert_allclose(
            bases_chunk["attn_delta"].astype(np.float32),
            bases_single["attn_delta"].astype(np.float32),
            atol=5e-2, rtol=5e-2,
        )

    def test_tempdir_cleaned_up_on_exception(self, tiny_model, monkeypatch):
        """If scan_forward raises mid-scan, the TemporaryDirectory must
        still be cleaned up — the try/finally around the body guarantees
        this independent of GC timing. Inject the failure at
        `_make_causal_mask`, which is called inside the chunk loop
        after the tmpdir has been created and populated."""
        import os
        import tempfile

        created_dirs: list[str] = []
        real_tempdir = tempfile.TemporaryDirectory

        class _TrackingTempDir(real_tempdir):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created_dirs.append(self.name)

        monkeypatch.setattr(tempfile, "TemporaryDirectory", _TrackingTempDir)

        import lmprobe.backends as bmod

        def boom(*_args, **_kwargs):
            raise RuntimeError("forced test failure")

        monkeypatch.setattr(bmod, "_make_causal_mask", boom)

        backend = ChunkedLocalBackend(
            tiny_model, "cpu", dtype=torch.float32, chunk_size=1,
        )
        with pytest.raises(RuntimeError, match="forced test failure"):
            backend.scan_forward(
                prompts=["One", "Two"],
                signals=["attn_delta"],
                n_components=2,
                batch_size=2,
            )

        assert created_dirs, "scan_forward did not open a TemporaryDirectory"
        for d in created_dirs:
            assert not os.path.exists(d), (
                f"temp dir leaked on exception path: {d}"
            )
