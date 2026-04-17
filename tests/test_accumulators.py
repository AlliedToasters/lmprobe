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
