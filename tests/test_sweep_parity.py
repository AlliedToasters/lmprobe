"""Dual-run parity tests: legacy ``scan_forward`` vs. new ``sweep`` path.

Runs the same corpus through both code paths on the tiny test model, asserts
PCA bases match within fp16 tolerance, and that re-projecting through the
basis produces matching per-token projections within tolerance.

Gates the refactor: as long as this test stays green we can keep porting
forward paths incrementally.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import NEGATIVE_PROMPTS, POSITIVE_PROMPTS

TEST_MODEL = "stas/tiny-random-llama-2"


@pytest.fixture
def corpus() -> tuple[list[str], list[int]]:
    prompts = POSITIVE_PROMPTS[:3] + NEGATIVE_PROMPTS[:3]
    labels = [1, 1, 1, 0, 0, 0]
    return prompts, labels


# ---------------------------------------------------------------------------
# Legacy scan_forward → bases (baseline)
# ---------------------------------------------------------------------------


def _run_legacy_scan(
    prompts: list[str],
    n_components: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Runs the legacy ``ChunkedLocalBackend.scan_forward``. Returns
    ``(bases_dict, projections, coords_dict)``."""
    from lmprobe.backends import ChunkedLocalBackend

    backend = ChunkedLocalBackend(
        model_name=TEST_MODEL,
        device="cpu",
    )
    (
        _metadata,
        bases,
        projections,
        coords,
        _token_ids,
        _seq_lengths,
        _attention_mask,
        _signal_dims,
    ) = backend.scan_forward(
        prompts,
        signals=["attn_delta", "mlp_delta"],
        n_components=n_components,
        batch_size=batch_size,
    )
    return bases, projections, coords


# ---------------------------------------------------------------------------
# New sweep + PCAFit → bases
# ---------------------------------------------------------------------------


def _run_sweep_pcafit(
    prompts: list[str],
    n_components: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Runs ``sweep`` with just a :class:`PCAFit` accumulator. Returns
    the same ``{sig: [n_layers, dim, k_eff]}`` bases dict."""
    from lmprobe.accumulators import PCAFit
    from lmprobe.backends import ChunkedLayerLoader, ChunkedLocalBackend
    from lmprobe.sweep import sweep

    backend = ChunkedLocalBackend(
        model_name=TEST_MODEL,
        device="cpu",
    )
    loader = ChunkedLayerLoader(backend)
    out = sweep(
        prompts,
        accumulators={
            "fit": PCAFit(
                signals=["attn_delta", "mlp_delta"],
                n_components=n_components,
            ),
        },
        loader=loader,
        batch_size=batch_size,
    )
    return out["fit"]


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


class TestPCAFitParity:
    """PCAFit bases should match the legacy scan_forward bases within
    fp16 tolerance, up to sign ambiguity per principal component."""

    def test_bases_shape_matches(self, corpus: tuple[list[str], list[int]]) -> None:
        prompts, _ = corpus
        legacy_bases, _, _ = _run_legacy_scan(
            prompts, n_components=4, batch_size=2,
        )
        new_bases = _run_sweep_pcafit(prompts, n_components=4, batch_size=2)
        assert set(new_bases.keys()) == set(legacy_bases.keys())
        for sig in legacy_bases:
            assert new_bases[sig].shape == legacy_bases[sig].shape, (
                f"signal {sig}: new shape {new_bases[sig].shape} != "
                f"legacy {legacy_bases[sig].shape}"
            )

    def test_bases_span_same_subspace(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        """PCA components are defined up to sign, so we compare the
        absolute dot product between paired components. Perfect parity ⇒
        all diagonal entries of ``|A.T @ B|`` are close to 1.0."""
        prompts, _ = corpus
        legacy_bases, _, _ = _run_legacy_scan(
            prompts, n_components=4, batch_size=2,
        )
        new_bases = _run_sweep_pcafit(prompts, n_components=4, batch_size=2)
        for sig in legacy_bases:
            L_basis = legacy_bases[sig].astype(np.float32)  # [n_layers, dim, k]
            N_basis = new_bases[sig].astype(np.float32)
            for layer in range(L_basis.shape[0]):
                l_mat = L_basis[layer]  # [dim, k]
                n_mat = N_basis[layer]
                if np.allclose(l_mat, 0) and np.allclose(n_mat, 0):
                    continue  # layer had no fit rows (unlikely here)
                # Diagonal of |l.T @ n|: each paired component's overlap.
                dots = np.abs(l_mat.T @ n_mat)
                diag = np.diagonal(dots)
                # On tiny-random weights the top PCs should align up to
                # numerical noise and sign. atol=0.05 is generous for fp16.
                assert np.all(diag > 0.95), (
                    f"{sig} layer {layer}: paired-component overlaps {diag} "
                    f"(expected ~1.0 up to sign)"
                )


class TestPerTokenProjectionParity:
    """PerTokenProjection values via a second sweep should match the
    legacy scan_forward's per-token projections within fp16 tolerance."""

    def test_projection_values_match(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        from lmprobe.accumulators import PerTokenProjection
        from lmprobe.backends import ChunkedLayerLoader, ChunkedLocalBackend
        from lmprobe.sweep import sweep

        prompts, _ = corpus
        # Legacy: fit + project in one pass.
        legacy_bases, legacy_proj, legacy_coords = _run_legacy_scan(
            prompts, n_components=4, batch_size=2,
        )

        # New: first fit, then project through the fit basis.
        new_bases = _run_sweep_pcafit(prompts, n_components=4, batch_size=2)
        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        loader = ChunkedLayerLoader(backend)
        out = sweep(
            prompts,
            accumulators={
                "proj": PerTokenProjection(new_bases),
            },
            loader=loader,
            external_bases=new_bases,
            batch_size=2,
        )
        proj_out = out["proj"]
        new_values = proj_out["values"]          # [total_rows, k_max]
        new_coords = proj_out["coords"]
        offset_table = proj_out["offset_table"]  # [n_samples, n_layers, n_sig, 2]

        # Spot-check: for each (sample, layer, sig), pull new_values via
        # offset_table and legacy_proj via coords mask, compare up to sign.
        legacy_sample_ids = legacy_coords["sample_id"]
        legacy_layers = legacy_coords["layer"]
        legacy_signal = legacy_coords["signal"]
        legacy_token_pos = legacy_coords["token_pos"]
        signal_names = proj_out["signal_names"]

        n_samples = len(prompts)
        n_layers = offset_table.shape[1]

        for sid in range(n_samples):
            for L in range(n_layers):
                for si, _sig_name in enumerate(signal_names):
                    start, end = offset_table[sid, L, si]
                    if end <= start:
                        continue
                    seq_len = end - start
                    new_rows = new_values[start:end].astype(np.float32)
                    mask = (
                        (legacy_sample_ids == sid)
                        & (legacy_layers == L)
                        & (legacy_signal == si)
                    )
                    legacy_rows_all = legacy_proj[mask, 0, :].astype(
                        np.float32,
                    )
                    legacy_tok_pos = legacy_token_pos[mask]
                    # Align to real tokens only.
                    sort_idx = np.argsort(legacy_tok_pos)
                    legacy_rows_sorted = legacy_rows_all[sort_idx]
                    legacy_rows = legacy_rows_sorted[:seq_len, : new_rows.shape[1]]
                    # Signs may differ per component. Compare column-by-column
                    # with a ±1 sign sweep.
                    assert legacy_rows.shape == new_rows.shape, (
                        f"(sid={sid}, L={L}, si={si}) shape mismatch: "
                        f"legacy {legacy_rows.shape} vs new {new_rows.shape}"
                    )
                    # Per-column sign resolution: pick sign that minimizes
                    # absolute delta.
                    for c in range(new_rows.shape[1]):
                        direct = np.max(np.abs(legacy_rows[:, c] - new_rows[:, c]))
                        flipped = np.max(np.abs(legacy_rows[:, c] + new_rows[:, c]))
                        err = min(direct, flipped)
                        # fp16 tolerance; CPU vs same-CPU matmul should be tight.
                        assert err < 5e-2, (
                            f"(sid={sid}, L={L}, si={si}, c={c}) "
                            f"max diff {err} exceeds 5e-2"
                        )
        # Coords structural parity.
        assert new_coords["sample_id"].size == legacy_coords["sample_id"].size, (
            "coords row count mismatch"
        )


class TestProjectForwardParity:
    """project_forward_via_sweep should match project_forward within fp16
    tolerance, up to sign ambiguity on per-column projections."""

    def test_projections_shape_matches(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        from lmprobe.backends import ChunkedLocalBackend

        prompts, _ = corpus
        legacy_bases, _, _ = _run_legacy_scan(
            prompts, n_components=4, batch_size=2,
        )

        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        legacy_proj, legacy_tokens, legacy_logits = backend.project_forward(
            "Hello world",
            bases=legacy_bases,
            signals=["attn_delta", "mlp_delta"],
            include_logits=True,
        )
        new_proj, new_tokens, new_logits = backend.project_forward_via_sweep(
            "Hello world",
            bases=legacy_bases,
            signals=["attn_delta", "mlp_delta"],
            include_logits=True,
        )

        assert new_proj.shape == legacy_proj.shape
        assert new_tokens == legacy_tokens
        assert (new_logits is None) == (legacy_logits is None)
        if new_logits is not None:
            assert new_logits.shape == legacy_logits.shape

    def test_projections_values_match_up_to_sign(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        from lmprobe.backends import ChunkedLocalBackend

        prompts, _ = corpus
        legacy_bases, _, _ = _run_legacy_scan(
            prompts, n_components=4, batch_size=2,
        )

        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        legacy_proj, _, _ = backend.project_forward(
            "Hello world",
            bases=legacy_bases,
            signals=["attn_delta", "mlp_delta"],
            include_logits=False,
        )
        new_proj, _, _ = backend.project_forward_via_sweep(
            "Hello world",
            bases=legacy_bases,
            signals=["attn_delta", "mlp_delta"],
            include_logits=False,
        )

        # Both are [seq_len, n_layers, n_sig, max_k]. PCA component sign
        # ambiguity is resolved by picking per-(layer,sig,component) sign.
        seq_len, n_layers, n_sig, k = legacy_proj.shape
        for L in range(n_layers):
            for si in range(n_sig):
                for c in range(k):
                    legacy_col = legacy_proj[:, L, si, c].astype(np.float32)
                    new_col = new_proj[:, L, si, c].astype(np.float32)
                    direct = np.max(np.abs(legacy_col - new_col))
                    flipped = np.max(np.abs(legacy_col + new_col))
                    err = min(direct, flipped)
                    assert err < 5e-2, (
                        f"(L={L}, si={si}, c={c}) max diff {err} > 5e-2"
                    )


class TestChunkedForwardParity:
    """chunked_forward_via_sweep should produce identical activations and
    logits to the legacy _chunked_forward path."""

    def test_activations_match(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        import torch

        from lmprobe.backends import ChunkedLocalBackend

        prompts, _ = corpus
        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        layer_indices = [0, 1]

        legacy_act, legacy_mask, legacy_logits, _ = backend._chunked_forward(
            prompts, layer_indices, include_logits=False,
        )
        new_act, new_mask, new_logits, _ = backend.chunked_forward_via_sweep(
            prompts, layer_indices, include_logits=False,
        )

        assert legacy_act is not None and new_act is not None
        assert new_act.shape == legacy_act.shape
        # Identical layer weights + deterministic forward ⇒ exact match
        # at float32 precision. bf16 cast introduces small error; allow
        # a tight fp16-scale tolerance.
        assert torch.allclose(
            new_act.float(), legacy_act.float(), atol=1e-2, rtol=1e-2,
        ), (
            f"activations diverge: max diff "
            f"{(new_act.float() - legacy_act.float()).abs().max().item()}"
        )
        assert torch.equal(new_mask, legacy_mask)
        assert (new_logits is None) == (legacy_logits is None)

    def test_with_logits(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        import torch

        from lmprobe.backends import ChunkedLocalBackend

        prompts, _ = corpus
        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")

        legacy_act, _, legacy_logits, _ = backend._chunked_forward(
            prompts, [0, 1], include_logits=True,
        )
        new_act, _, new_logits, _ = backend.chunked_forward_via_sweep(
            prompts, [0, 1], include_logits=True,
        )

        assert legacy_logits is not None and new_logits is not None
        assert new_logits.shape == legacy_logits.shape
        assert torch.allclose(
            new_logits.float(), legacy_logits.float(), atol=1e-2, rtol=1e-2,
        )
