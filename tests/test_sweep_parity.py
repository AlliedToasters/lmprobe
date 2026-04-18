"""Dual-run parity tests: legacy ``scan_forward`` vs. new ``sweep`` path.

Runs the same corpus through both code paths on the tiny test model, asserts
PCA bases match within fp16 tolerance, and that re-projecting through the
basis produces matching per-token projections within tolerance.

Gates the refactor: as long as this test stays green we can keep porting
forward paths incrementally.
"""

from __future__ import annotations

from typing import Any

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


class TestStreamProjectParity:
    """Legacy ``scan_forward(external_bases=...)`` vs. ``sweep +
    PerTokenProjection``.

    Covers the third legacy path — scan_forward's stream-project branch
    (``backends.py:2182-2199``) — which `TestPerTokenProjectionParity`
    doesn't hit because that test compares fit-and-project in one legacy
    call vs. two sweeps. Here both paths consume an **externally supplied**
    basis, so divergences are purely in projection / slot-write logic.
    """

    def test_stream_project_matches(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        from lmprobe.accumulators import PerTokenProjection
        from lmprobe.backends import (
            ChunkedLayerLoader,
            ChunkedLocalBackend,
        )
        from lmprobe.sweep import sweep

        prompts, _ = corpus

        # Fit a basis once so both paths project through identical weights.
        legacy_bases, _, _ = _run_legacy_scan(
            prompts, n_components=4, batch_size=2,
        )

        # Path A — legacy scan_forward in pure stream-project mode.
        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        (
            _metadata,
            _bases_out,
            legacy_proj,
            legacy_coords,
            _tokens,
            _seq_lengths,
            _attention_mask,
            _signal_dims,
        ) = backend.scan_forward(
            prompts,
            signals=["attn_delta", "mlp_delta"],
            n_components=4,
            batch_size=2,
            external_bases=legacy_bases,
        )

        # Path B — sweep + PerTokenProjection, same basis.
        loader = ChunkedLayerLoader(backend)
        out = sweep(
            prompts,
            accumulators={"proj": PerTokenProjection(legacy_bases)},
            loader=loader,
            external_bases=legacy_bases,
            batch_size=2,
        )
        new_values = out["proj"]["values"]
        offset_table = out["proj"]["offset_table"]
        signal_names = out["proj"]["signal_names"]

        legacy_sample_ids = legacy_coords["sample_id"]
        legacy_layers = legacy_coords["layer"]
        legacy_signal = legacy_coords["signal"]
        legacy_token_pos = legacy_coords["token_pos"]

        n_samples = len(prompts)
        n_layers = offset_table.shape[1]
        for sid in range(n_samples):
            for L in range(n_layers):
                for si, _sig in enumerate(signal_names):
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
                    sort_idx = np.argsort(legacy_tok_pos)
                    legacy_rows = legacy_rows_all[sort_idx][
                        :seq_len, : new_rows.shape[1]
                    ]
                    # Both paths consume the same basis, so signs align.
                    np.testing.assert_allclose(
                        new_rows, legacy_rows, atol=5e-2, rtol=5e-2,
                        err_msg=(
                            f"(sid={sid}, L={L}, si={si}) stream-project "
                            f"rows diverge beyond fp16 tolerance"
                        ),
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


# ---------------------------------------------------------------------------
# Reducer parity: legacy reducers via scan_forward vs new-protocol
# reducers via sweep
# ---------------------------------------------------------------------------


def _make_reducer_masks(
    prompts: list[str], model_name: str,
) -> list[np.ndarray]:
    """Build per-sample bool masks matching lmprobe's tokenizer output.

    Emulates a typical generative-token mask: the last 30% of real tokens.
    Length per sample matches the real (unpadded) token count.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    masks: list[np.ndarray] = []
    for p in prompts:
        enc = tok(p, add_special_tokens=True)
        n = len(enc["input_ids"])
        m = np.zeros(n, dtype=bool)
        cut = max(1, n - max(1, n // 3))
        m[cut:] = True
        masks.append(m)
    return masks


class TestSweepMemmapRoundtrip:
    """Spec 004 memmap residuals must survive cross-chunk roundtrips on the
    sweep path. Running with ``chunk_size=1`` forces the driver through the
    memmap read/write boundary at every layer transition, catching a
    regression where ``ChunkedLayerLoader`` silently used an in-memory
    tensor buffer instead."""

    def test_chunk_size_one_matches_default(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        from lmprobe.accumulators import PCAFit
        from lmprobe.backends import ChunkedLayerLoader, ChunkedLocalBackend
        from lmprobe.sweep import sweep

        prompts, _ = corpus
        # Default chunk (n_layers on CPU).
        backend_full = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        loader_full = ChunkedLayerLoader(backend_full)
        out_full = sweep(
            prompts,
            accumulators={
                "fit": PCAFit(signals=["attn_delta"], n_components=4),
            },
            loader=loader_full,
            batch_size=2,
        )
        # chunk_size=1: every layer boundary hits the memmap.
        backend_small = ChunkedLocalBackend(
            model_name=TEST_MODEL, device="cpu", chunk_size=1,
        )
        loader_small = ChunkedLayerLoader(backend_small)
        out_small = sweep(
            prompts,
            accumulators={
                "fit": PCAFit(signals=["attn_delta"], n_components=4),
            },
            loader=loader_small,
            batch_size=2,
        )
        # Same fit captures regardless of chunking ⇒ paired-component
        # overlap ~1.0 up to sign.
        b_full = out_full["fit"]["attn_delta"].astype(np.float32)
        b_small = out_small["fit"]["attn_delta"].astype(np.float32)
        assert b_full.shape == b_small.shape
        for layer in range(b_full.shape[0]):
            if np.allclose(b_full[layer], 0) and np.allclose(b_small[layer], 0):
                continue
            dots = np.abs(b_full[layer].T @ b_small[layer])
            diag = np.diagonal(dots)
            assert np.all(diag > 0.95), (
                f"layer {layer}: chunk_size=1 basis diverges from default; "
                f"paired overlaps {diag}"
            )


class TestReducerParity:
    """Legacy reducers (``scan_forward(reducers=...)``) vs. new-protocol
    reducers (``sweep(accumulators=...)``). Same masks, same basis — outputs
    must agree exactly (no sign ambiguity: both project through an
    identical pre-fit basis)."""

    def _legacy_run(
        self,
        prompts: list[str],
        bases: dict[str, np.ndarray],
        legacy_reducers: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        from lmprobe.backends import ChunkedLocalBackend

        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        signals = list(bases.keys())
        n_layers = next(iter(bases.values())).shape[0]
        k_dim = next(iter(bases.values())).shape[-1]

        bound: dict[str, tuple[Any, Any]] = {}
        for name, red in legacy_reducers.items():
            state = red.init_state(len(prompts), n_layers, len(signals), k_dim)
            bound[name] = (red, state)
        backend.scan_forward(
            prompts,
            signals=signals,
            n_components=k_dim,
            batch_size=2,
            external_bases=bases,
            reducers=bound,
        )
        return {name: red.finalize(state) for name, (red, state) in bound.items()}

    def _sweep_run(
        self,
        prompts: list[str],
        bases: dict[str, np.ndarray],
        accumulators: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        from lmprobe.backends import ChunkedLayerLoader, ChunkedLocalBackend
        from lmprobe.sweep import sweep

        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        loader = ChunkedLayerLoader(backend)
        out = sweep(
            prompts,
            accumulators=accumulators,
            loader=loader,
            external_bases=bases,
            batch_size=2,
        )
        return out

    def test_last_token_reducer_matches(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        from lmprobe.accumulators import (
            LastTokenReducer as NewLastTokenReducer,
        )
        from lmprobe.reducers import LastTokenReducer as LegacyLastTokenReducer

        prompts, _ = corpus
        bases, _, _ = _run_legacy_scan(prompts, n_components=4, batch_size=2)
        masks = _make_reducer_masks(prompts, TEST_MODEL)

        legacy_out = self._legacy_run(
            prompts, bases, {"red": LegacyLastTokenReducer(masks)},
        )
        new_out = self._sweep_run(
            prompts, bases, {"red": NewLastTokenReducer(masks, bases=bases)},
        )
        np.testing.assert_allclose(
            new_out["red"].astype(np.float32),
            legacy_out["red"].astype(np.float32),
            atol=5e-3, rtol=5e-3,
        )

    def test_mean_reducer_matches(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        from lmprobe.accumulators import MeanReducer as NewMeanReducer
        from lmprobe.reducers import MeanReducer as LegacyMeanReducer

        prompts, _ = corpus
        bases, _, _ = _run_legacy_scan(prompts, n_components=4, batch_size=2)
        masks = _make_reducer_masks(prompts, TEST_MODEL)

        legacy_out = self._legacy_run(
            prompts, bases, {"red": LegacyMeanReducer(masks)},
        )
        new_out = self._sweep_run(
            prompts, bases, {"red": NewMeanReducer(masks, bases=bases)},
        )
        np.testing.assert_allclose(
            new_out["red"].astype(np.float32),
            legacy_out["red"].astype(np.float32),
            atol=5e-3, rtol=5e-3,
        )

    def test_mean_excl_last_n_reducer_matches(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        from lmprobe.accumulators import (
            MeanExclLastNReducer as NewMeanExclLastNReducer,
        )
        from lmprobe.reducers import (
            MeanExclLastNReducer as LegacyMeanExclLastNReducer,
        )

        prompts, _ = corpus
        bases, _, _ = _run_legacy_scan(prompts, n_components=4, batch_size=2)
        masks = _make_reducer_masks(prompts, TEST_MODEL)

        legacy_out = self._legacy_run(
            prompts, bases, {"red": LegacyMeanExclLastNReducer(masks, n=2)},
        )
        new_out = self._sweep_run(
            prompts, bases,
            {"red": NewMeanExclLastNReducer(masks, bases=bases, n=2)},
        )
        np.testing.assert_allclose(
            new_out["red"].astype(np.float32),
            legacy_out["red"].astype(np.float32),
            atol=5e-3, rtol=5e-3,
        )
