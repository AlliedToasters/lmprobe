"""Targeted tests for the new Accumulator protocol + built-ins.

Covers behaviors surfaced during the pressure-test of the sweep refactor:

- PCAFit + PerTokenProjection on the same signal must raise (two-sweep rule).
- k_eff = min(n_components, n_fit_rows-1, dim) varies per (layer, signal);
  PerTokenProjection pads with zeros to keep ``values.npy`` rectangular.
- Offset-table roundtrip: write via PerTokenProjection, read via
  ``scan.per_token`` — matches a dense control.
- Mixed raw+projection on *different* signals is allowed (PCAFit on one,
  reducer on another, one sweep).
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
# Accumulator conflict enforcement
# ---------------------------------------------------------------------------


class TestAccumulatorConflict:
    """PCAFit (raw) + PerTokenProjection (projection) on the same signal is
    a caller mistake — the two need separate sweeps. The resolver raises."""

    def test_pcafit_plus_pertoken_same_signal_raises(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        from lmprobe.accumulators import PCAFit, PerTokenProjection
        from lmprobe.backends import ChunkedLayerLoader, ChunkedLocalBackend
        from lmprobe.sweep import sweep

        prompts, _ = corpus
        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        loader = ChunkedLayerLoader(backend)
        dummy_basis = {
            "attn_delta": np.zeros(
                (loader.num_layers, 1, 4), dtype=np.float16,
            ),
        }

        with pytest.raises(ValueError, match="raw.*projection.*subscribers"):
            sweep(
                prompts,
                accumulators={
                    "fit": PCAFit(signals=["attn_delta"], n_components=4),
                    "proj": PerTokenProjection(dummy_basis),
                },
                loader=loader,
                external_bases=dummy_basis,
                batch_size=2,
            )

    def test_projection_without_basis_raises(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        from lmprobe.accumulators import PerTokenProjection
        from lmprobe.backends import ChunkedLayerLoader, ChunkedLocalBackend
        from lmprobe.sweep import sweep

        prompts, _ = corpus
        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        loader = ChunkedLayerLoader(backend)
        dummy_basis = {
            "attn_delta": np.zeros(
                (loader.num_layers, 1, 4), dtype=np.float16,
            ),
        }

        # PerTokenProjection carries a basis at construction, but we pass
        # external_bases=None at the sweep boundary — which controls the
        # stream-project path. This surfaces the resolver error.
        with pytest.raises(ValueError, match="no external_bases"):
            sweep(
                prompts,
                accumulators={
                    "proj": PerTokenProjection(dummy_basis),
                },
                loader=loader,
                external_bases=None,
                batch_size=2,
            )


# ---------------------------------------------------------------------------
# Mixed raw + projection on *different* signals is fine
# ---------------------------------------------------------------------------


class TestMixedRawProjectionDifferentSignals:
    """PCAFit on attn_delta + PerTokenProjection on mlp_delta in one sweep.

    Each signal has only one kind of subscriber; the resolver allows this
    because the invariant is per-signal, not sweep-wide.
    """

    def test_pcafit_and_projection_distinct_signals_ok(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        from lmprobe.accumulators import PCAFit, PerTokenProjection
        from lmprobe.backends import ChunkedLayerLoader, ChunkedLocalBackend
        from lmprobe.sweep import sweep

        prompts, _ = corpus
        # First, fit bases on both signals so we can supply external_bases
        # for the mlp_delta PerTokenProjection.
        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        loader = ChunkedLayerLoader(backend)
        fit_out = sweep(
            prompts,
            accumulators={
                "fit": PCAFit(
                    signals=["attn_delta", "mlp_delta"], n_components=4,
                ),
            },
            loader=loader,
            batch_size=2,
        )
        bases = fit_out["fit"]  # {sig: [n_layers, dim, k]}

        # Now: PCAFit on attn_delta (raw) + PerTokenProjection on mlp_delta (proj).
        # Different signals — resolver allows.
        loader = ChunkedLayerLoader(backend)
        mixed_out = sweep(
            prompts,
            accumulators={
                "fit_attn": PCAFit(
                    signals=["attn_delta"], n_components=4,
                ),
                "proj_mlp": PerTokenProjection(
                    {"mlp_delta": bases["mlp_delta"]},
                ),
            },
            loader=loader,
            external_bases={"mlp_delta": bases["mlp_delta"]},
            batch_size=2,
        )
        assert "fit_attn" in mixed_out
        assert "proj_mlp" in mixed_out
        # Sanity: the fit_attn bases have the right shape.
        assert mixed_out["fit_attn"]["attn_delta"].shape == (
            loader.num_layers, bases["attn_delta"].shape[1], 4,
        )
        # Sanity: proj_mlp produced non-empty values.
        assert mixed_out["proj_mlp"]["values"].shape[0] > 0


# ---------------------------------------------------------------------------
# per_token / offset-table roundtrip
# ---------------------------------------------------------------------------


class TestPerTokenOffsetTable:
    """scan.per_token() via offset_table must match the dense projections
    assembled from scan.get_projections(). Covers the O(1) slice path."""

    def test_per_token_matches_get_projections(
        self, corpus: tuple[list[str], list[int]], tmp_path: Any,
    ) -> None:
        from lmprobe.sample_scan import SampleScan

        prompts, labels = corpus
        scan = SampleScan.run(
            prompts=prompts,
            labels=labels,
            model_name=TEST_MODEL,
            scan_dir=tmp_path / "offset_scan",
            signals=["attn_delta", "mlp_delta"],
            n_components=4,
            device="cpu",
            batch_size=2,
        )
        # Offset table should be present for scans written under the new run.
        assert scan._load_offset_table() is not None

        for sid in range(scan.n_samples):
            seq_len = int(scan.samples.column("seq_length").to_pylist()[sid])
            for L in range(scan.n_layers):
                for sig in scan.signals:
                    ot_rows = scan.per_token(sid, L, signal=sig)
                    # [seq_len, k]
                    assert ot_rows.shape[0] == seq_len
                    # Compare to legacy get_projections path (slower, but
                    # independent): [seq_len, n_layers, 1, k].
                    ref = scan.get_projections(sid, signal=sig)
                    ref_slice = ref[:seq_len, L, 0, :]
                    np.testing.assert_array_equal(
                        ot_rows.astype(np.float32),
                        ref_slice.astype(np.float32),
                    )


# ---------------------------------------------------------------------------
# k_eff boundary: n_components > dim
# ---------------------------------------------------------------------------


class TestKEffVariance:
    """When n_components exceeds the fit subspace rank, PCAFit returns a
    smaller k_eff. The basis array is padded with zeros to a rectangular
    ``[n_layers, dim, k_eff_pad]``. PerTokenProjection must tolerate
    this without shape mismatches."""

    def test_large_n_components_produces_padded_basis(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        from lmprobe.accumulators import PCAFit
        from lmprobe.backends import ChunkedLayerLoader, ChunkedLocalBackend
        from lmprobe.sweep import sweep

        prompts, _ = corpus
        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        loader = ChunkedLayerLoader(backend)
        # Request more components than the tiny model's hidden dim supports.
        out = sweep(
            prompts,
            accumulators={
                "fit": PCAFit(signals=["attn_delta"], n_components=1024),
            },
            loader=loader,
            batch_size=2,
        )
        basis = out["fit"]["attn_delta"]
        assert basis.ndim == 3
        # k_eff_pad = min(n_components, dim). For tiny-random-llama-2,
        # dim is small so we'll see k_eff_pad == dim.
        assert basis.shape[2] <= basis.shape[1]

    def test_small_corpus_k_eff_bounded_by_fit_rows(self) -> None:
        """n_components > n_fit_rows − 1: PCA can only extract rank-(n-1)
        components, so the returned basis is padded with zeros to the
        requested ``n_components`` column width (or ``dim``, whichever is
        smaller)."""
        from lmprobe.accumulators import PCAFit
        from lmprobe.backends import ChunkedLayerLoader, ChunkedLocalBackend
        from lmprobe.sweep import sweep

        # Two prompts × ~short sequences × n_layers captures are the fit
        # rows per (layer, signal). With n_components=32, we're asking for
        # more components than rank — PCAFit should still produce a
        # rectangular array without shape errors.
        small_prompts = ["The cat", "A dog"]
        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        loader = ChunkedLayerLoader(backend)
        out = sweep(
            small_prompts,
            accumulators={
                "fit": PCAFit(signals=["attn_delta"], n_components=32),
            },
            loader=loader,
            batch_size=2,
        )
        basis = out["fit"]["attn_delta"]
        assert basis.ndim == 3
        # Rectangular; trailing columns may be zero-padded where k_eff
        # falls short of the requested 32.
        n_layers, dim, k = basis.shape
        assert k == min(32, dim)


# ---------------------------------------------------------------------------
# Reducer branch coverage (edge-case fixtures, not parity)
# ---------------------------------------------------------------------------


class TestReducerBranches:
    """Edge branches in LastTokenReducer / MeanReducer that normal corpora
    rarely hit, but which the accumulator protocol must handle gracefully."""

    def test_last_token_reducer_no_true_mask_yields_zeros(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        """``_has_true=False`` branch: a sample whose mask has zero ``True``
        entries should produce a zero output slice and not raise."""
        from lmprobe.accumulators import LastTokenReducer, PCAFit
        from lmprobe.backends import ChunkedLayerLoader, ChunkedLocalBackend
        from lmprobe.sweep import sweep

        prompts, _ = corpus
        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        loader = ChunkedLayerLoader(backend)
        # 1) Fit bases.
        fit_out = sweep(
            prompts,
            accumulators={
                "fit": PCAFit(signals=["attn_delta"], n_components=4),
            },
            loader=loader,
            batch_size=2,
        )
        bases = fit_out["fit"]

        # 2) Build a mask list where sample 0 has all-False (no True).
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(TEST_MODEL)
        masks: list[np.ndarray] = []
        for i, p in enumerate(prompts):
            n = len(tok(p, add_special_tokens=True)["input_ids"])
            m = np.zeros(n, dtype=bool)
            if i > 0:
                m[-1] = True
            masks.append(m)

        # 3) Reduce.
        loader = ChunkedLayerLoader(backend)
        out = sweep(
            prompts,
            accumulators={
                "red": LastTokenReducer(masks, bases=bases),
            },
            loader=loader,
            external_bases=bases,
            batch_size=2,
        )
        red_out = out["red"]  # [N, L, G, k]
        # Sample 0 must be all zeros; others nonzero somewhere.
        assert np.all(red_out[0] == 0)
        assert np.any(red_out[1] != 0)

    def test_chunk_size_one_matches_default_chunking(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        """Spec 004 memmap residuals must survive cross-chunk roundtrips on
        the sweep path. ``chunk_size=1`` forces memmap read/write at every
        layer boundary — catches regressions where the residual buffer
        silently runs in-memory instead of going through disk."""
        from lmprobe.accumulators import PCAFit
        from lmprobe.backends import ChunkedLayerLoader, ChunkedLocalBackend
        from lmprobe.sweep import sweep

        prompts, _ = corpus
        backend_full = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        loader_full = ChunkedLayerLoader(backend_full)
        out_full = sweep(
            prompts,
            accumulators={"fit": PCAFit(signals=["attn_delta"], n_components=4)},
            loader=loader_full,
            batch_size=2,
        )
        backend_small = ChunkedLocalBackend(
            model_name=TEST_MODEL, device="cpu", chunk_size=1,
        )
        loader_small = ChunkedLayerLoader(backend_small)
        out_small = sweep(
            prompts,
            accumulators={"fit": PCAFit(signals=["attn_delta"], n_components=4)},
            loader=loader_small,
            batch_size=2,
        )
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

    def test_mean_reducer_zero_count_yields_zeros(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        """``_count==0`` branch: samples with zero ``True`` positions must
        produce zero-mean output — no division-by-zero, no NaN."""
        from lmprobe.accumulators import MeanReducer, PCAFit
        from lmprobe.backends import ChunkedLayerLoader, ChunkedLocalBackend
        from lmprobe.sweep import sweep

        prompts, _ = corpus
        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        loader = ChunkedLayerLoader(backend)
        fit_out = sweep(
            prompts,
            accumulators={
                "fit": PCAFit(signals=["attn_delta"], n_components=4),
            },
            loader=loader,
            batch_size=2,
        )
        bases = fit_out["fit"]

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(TEST_MODEL)
        masks = [
            np.zeros(
                len(tok(p, add_special_tokens=True)["input_ids"]), dtype=bool,
            )
            for p in prompts
        ]
        # Give sample 1 one True so divide-path also runs.
        masks[1][-1] = True

        loader = ChunkedLayerLoader(backend)
        out = sweep(
            prompts,
            accumulators={"red": MeanReducer(masks, bases=bases)},
            loader=loader,
            external_bases=bases,
            batch_size=2,
        )
        red_out = out["red"]
        # All-False masks -> zero mean, no NaN.
        assert np.all(red_out[0] == 0)
        assert not np.any(np.isnan(red_out))


# ---------------------------------------------------------------------------
# Raise-path coverage for invariants enforced by the refactor
# ---------------------------------------------------------------------------


class TestRaisePaths:
    """The consolidation added two fail-loud checks. Lock them down with
    unit tests so the next refactor doesn't silently undo them."""

    def test_per_token_projection_s_exceeds_s_max_raises(self) -> None:
        """``PerTokenProjection.update`` raises when the incoming batch seq-len
        exceeds the sweep's preallocated ``s_max``. Backends must pad to the
        corpus-wide max; overshooting would silently truncate tail columns."""
        import torch

        from lmprobe.accumulators import PerTokenProjection
        from lmprobe.sweep import SweepContext

        # 1 layer, 1 signal "x", dim=4, k=2.
        bases = {"x": np.zeros((1, 4, 2), dtype=np.float32)}
        acc = PerTokenProjection(bases)
        ctx = SweepContext(
            n_samples=2,
            num_layers=1,
            signals=["x"],
            signal_dims={"x": 4},
            hidden_dim=4,
            dtype=torch.float32,
            device="cpu",
            seq_lengths=[3, 2],
            s_max=3,
            k_per_sig={"x": [2]},
        )
        acc.init(ctx)
        # S=5 > s_max=3: must raise AssertionError.
        bad_data = np.zeros((2, 5, 2), dtype=np.float16)
        with pytest.raises(AssertionError, match=r"S=5 exceeds sweep s_max=3"):
            acc.update(
                bad_data,
                "x",
                0,
                np.asarray([0, 1], dtype=np.int64),
                np.ones((2, 5), dtype=np.int64),
            )

    def test_router_strategy_conflict_raises(
        self, corpus: tuple[list[str], list[int]],
    ) -> None:
        """Two ``RouterLogitCapture`` accumulators with different hook
        strategies cannot coexist in one sweep — only one hook fires per
        (layer, module), so silent first-wins would drop data. The driver
        must raise before installing hooks."""
        from lmprobe.accumulators import RouterLogitCapture
        from lmprobe.backends import ChunkedLayerLoader, ChunkedLocalBackend
        from lmprobe.sweep import sweep

        prompts, _ = corpus
        backend = ChunkedLocalBackend(model_name=TEST_MODEL, device="cpu")
        loader = ChunkedLayerLoader(backend)
        with pytest.raises(
            ValueError, match=r"conflicting router_logits hook strategies",
        ):
            sweep(
                prompts,
                accumulators={
                    "r_out": RouterLogitCapture([0], strategy="output"),
                    "r_gate": RouterLogitCapture([0], strategy="input_gate"),
                },
                loader=loader,
                batch_size=2,
            )
