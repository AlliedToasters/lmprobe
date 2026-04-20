"""Built-in accumulators for :func:`lmprobe.sweep.sweep`.

An accumulator consumes per-microbatch signals emitted by the sweep driver
and produces an output at sweep end. See :mod:`lmprobe.sweep` for the
protocol and dispatch rules.

Built-ins:

- :class:`PCAFit` — wants_raw. Captures CPU deltas per (layer, signal),
  fits PCA via the existing ``_fit_project_scan_pca`` helper when a layer
  group completes, frees captures eagerly. Output: ``dict[sig, basis]``
  where ``basis`` has shape ``[n_layers, dim, k_eff]``.
- :class:`PerTokenProjection` — wants_projection. Flat
  ``[N_rows, k]`` fp16 stack + coord arrays + offset table. The
  "rubble-as-first-class" accumulator: per-token PC dynamics preserved,
  O(1) indexed per (sample, layer, signal).
- :class:`LastTokenReducer`, :class:`MeanReducer`, :class:`MeanExclLastNReducer`
  — wants_projection. New-protocol versions of the spec 003 reducers.
- :class:`HiddenStateCapture` — wants_raw on ``"residual"``. Preallocated
  ``[N, S, H*L_sub]`` buffer matching the classic ``extract_batch`` output.
- :class:`LogitCapture` — subscribes to the end-of-sweep ``"logits"``
  signal. Output: ``[N, S, V]``.
- :class:`RouterLogitCapture` — wants_raw on ``"router_logits"`` with
  optional hook strategy (``"output"`` vs DeepSeek's ``"input_gate"``).

Reducer classes here follow the new Accumulator protocol and are the
sole home for :class:`LastTokenReducer`, :class:`MeanReducer`, and
:class:`MeanExclLastNReducer` post-consolidation. The legacy
``lmprobe.reducers`` module and ``scan_forward(reducers=...)`` call
path have been removed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from .sweep import SweepContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _real_positions(attention_mask_row: np.ndarray) -> np.ndarray:
    """Indices of real (non-pad) tokens within a padded microbatch row."""
    return np.where(attention_mask_row.astype(bool))[0]


def _check_n_samples(n_samples: int, n_masks: int, cls: str) -> None:
    if n_samples != n_masks:
        raise ValueError(
            f"{cls}: received n_samples={n_samples} from sweep, but was "
            f"constructed with {n_masks} masks. Mask list length must "
            f"equal the total number of prompts in the sweep."
        )


# ---------------------------------------------------------------------------
# PCAFit
# ---------------------------------------------------------------------------


class PCAFit:
    """Fit a PCA basis per (layer, signal). wants_raw.

    Captures per-microbatch CPU deltas into a list keyed by (layer, signal).
    When :meth:`on_layer_group_complete` fires for that layer's group, each
    of that layer's (layer, signal) capture lists is fit via the existing
    ``_fit_project_scan_pca`` helper (preserving cuml/sklearn switching and
    the ``generative_masks`` behavior). Captures are drained in place.

    The projection output from ``_fit_project_scan_pca`` is discarded —
    this accumulator only cares about the basis. Per-token projections
    come from a separate sweep with :class:`PerTokenProjection` (user
    decision: two sweeps for MVP).

    Parameters
    ----------
    signals : Sequence[str]
        Signals to fit a basis for. Typically the full scan signal set.
    n_components : int
        Target PCA rank. Effective ``k = min(n_components, n_fit_rows-1, dim)``.
    generative_masks : list[np.ndarray] or None
        Per-sample boolean masks. When provided, PCA fits only on
        ``True`` rows (avoids prompt-token leakage); all rows are
        still projected during the separate projection sweep.
    layers : Sequence[int] or None
        Subscribe to a layer subset. ``None`` = all layers.
    """

    wants_raw: bool = True

    def __init__(
        self,
        signals: Sequence[str],
        *,
        n_components: int,
        generative_masks: list[np.ndarray] | None = None,
        layers: Sequence[int] | None = None,
    ) -> None:
        self.signals: frozenset[str] = frozenset(signals)
        self.layers: frozenset[int] | None = (
            frozenset(layers) if layers is not None else None
        )
        self.n_components = int(n_components)
        self.generative_masks = generative_masks
        # Captures per (layer, signal); populated in update().
        self._captures: dict[tuple[int, str], list[torch.Tensor]] = {}
        # Completed (layer, signal) bases; populated in on_layer_group_complete.
        # {sig: {layer: np.ndarray[dim, k_eff]}}
        self._bases: dict[str, dict[int, np.ndarray]] = {
            s: {} for s in self.signals
        }
        self._signal_dims: dict[str, int] = {}
        self._ctx: SweepContext | None = None

    def init(self, ctx: SweepContext) -> None:
        self._ctx = ctx

    def update(
        self,
        data: np.ndarray | torch.Tensor,
        sig: str,
        layer_idx: int,
        sample_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> None:
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"PCAFit got {type(data).__name__} for signal {sig!r}; "
                f"expected torch.Tensor (wants_raw=True)."
            )
        key = (layer_idx, sig)
        self._captures.setdefault(key, []).append(data)
        if sig not in self._signal_dims:
            self._signal_dims[sig] = int(data.shape[-1])

    def on_layer_group_complete(self, layer_indices: list[int]) -> None:
        """Fit and free captures for every (layer, signal) whose layer just
        completed its forward passes. Mirrors today's between-chunk PCA fit
        block in ``scan_forward``."""
        # Late-bound import: avoids circular import at module load time.
        from .backends import _fit_project_scan_pca

        assert self._ctx is not None, "init() must run before group complete"
        for layer_idx in layer_indices:
            for sig in self.signals:
                key = (layer_idx, sig)
                captures = self._captures.get(key)
                if not captures:
                    continue
                # _fit_project_scan_pca drains `captures` in place (pop(0))
                # — don't hold references after the call. It also returns
                # a projection array which we don't need for fit-only; see
                # "Notes" below on why we tolerate that transient allocation.
                basis, projected, _B, _S, _dim = _fit_project_scan_pca(
                    captures,
                    device=self._ctx.device,
                    n_components=self.n_components,
                    generative_masks=self.generative_masks,
                )
                self._bases[sig][layer_idx] = basis
                # Discard projected — PerTokenProjection is a separate sweep.
                del projected
                # Ensure the captures dict entry is empty even if the helper
                # didn't fully drain (defense in depth).
                self._captures[key] = []

    def finalize(self) -> dict[str, np.ndarray]:
        """Return ``{sig: basis [n_layers, dim, k_eff_pad]}`` where
        ``k_eff_pad = min(n_components, dim)`` and per-layer ``k_eff``
        may be smaller (padded with zeros so the array is rectangular)."""
        assert self._ctx is not None
        num_layers = self._ctx.num_layers
        out: dict[str, np.ndarray] = {}
        for sig in self.signals:
            dim_opt = self._signal_dims.get(sig)
            dim: int = dim_opt if dim_opt is not None else self._ctx.hidden_dim
            k_eff = min(self.n_components, dim)
            basis_arr = np.zeros((num_layers, dim, k_eff), dtype=np.float16)
            for L in range(num_layers):
                layer_basis = self._bases[sig].get(L)
                if layer_basis is not None:
                    actual_k = layer_basis.shape[1]
                    basis_arr[L, :, :actual_k] = layer_basis
            out[sig] = basis_arr
        return out


# ---------------------------------------------------------------------------
# PerTokenProjection
# ---------------------------------------------------------------------------


class PerTokenProjection:
    """Flat ``[N_rows, k]`` stack of per-token projections + coords + offset
    table. The "rubble-as-first-class" accumulator.

    Pre-allocates a single ``[num_layers * n_sig * N * S_max, k_max]`` fp16
    buffer at ``init``. Updates write into slot-computed offsets — no
    append-grow, no post-hoc concatenate. At finalize, emits the
    offset table for O(1) per-(sample, layer, signal) slicing.

    Output (keyed by accumulator name in the sweep return):

    .. code-block:: python

        {
            "values": np.ndarray [total_rows, k_max] fp16,
            "coords": dict[str, np.ndarray] {sample_id, layer, token_pos, signal},
            "offset_table": np.ndarray [n_samples, n_layers, n_sig, 2] int32,
            "seq_lengths": np.ndarray [n_samples] int32,
            "signal_names": list[str],
        }

    The offset table uses ``(start_row, start_row + seq_lengths[i])`` so
    callers slice real tokens only; padding rows remain in ``values`` but
    are invisible via the table.

    Parameters
    ----------
    bases : Mapping[str, np.ndarray]
        Pre-fit PCA bases, one per signal. Shape ``[n_layers, dim, k]``.
        Determines which signals this accumulator subscribes to.
    layers : Sequence[int] or None
        Subscribe to a layer subset. ``None`` = all layers.
    """

    wants_raw: bool = False

    def __init__(
        self,
        bases: dict[str, np.ndarray],
        *,
        layers: Sequence[int] | None = None,
    ) -> None:
        if not bases:
            raise ValueError("PerTokenProjection requires at least one basis.")
        self.bases = bases
        self.signals: frozenset[str] = frozenset(bases.keys())
        self.layers: frozenset[int] | None = (
            frozenset(layers) if layers is not None else None
        )
        # Stable signal ordering for the signal axis in coords + offset table.
        self._signal_names: list[str] = sorted(self.signals)
        self._ctx: SweepContext | None = None
        self._values: np.ndarray | None = None
        self._seq_lengths: np.ndarray | None = None

    def estimate_bytes(
        self, ctx: SweepContext, batch_size: int,
    ) -> int:
        """Pre-``init`` size estimate for the pre-allocated values buffer.

        Consumed by the sweep driver's pre-flight memory check. Matches
        the allocation formula used in :meth:`init` (fp16, rectangular
        to ``k_max``).
        """
        n_sig = len(self._signal_names)
        k_max = max(int(self.bases[s].shape[-1]) for s in self._signal_names)
        total_rows = ctx.num_layers * n_sig * ctx.n_samples * ctx.s_max
        return total_rows * k_max * 2  # float16

    def init(self, ctx: SweepContext) -> None:
        self._ctx = ctx
        self._seq_lengths = np.asarray(ctx.seq_lengths, dtype=np.int32)
        n_sig = len(self._signal_names)
        self._k_per_sig = {
            s: int(self.bases[s].shape[-1]) for s in self._signal_names
        }
        self._k_max = max(self._k_per_sig.values())
        total_rows = ctx.num_layers * n_sig * ctx.n_samples * ctx.s_max
        self._values = np.zeros((total_rows, self._k_max), dtype=np.float16)
        # Block offset per (layer, signal) — slot-computed so updates are
        # O(1) and cross-microbatch.
        block_size = ctx.n_samples * ctx.s_max
        self._block_offsets: dict[tuple[int, str], int] = {}
        for L in range(ctx.num_layers):
            for si, sig in enumerate(self._signal_names):
                self._block_offsets[(L, sig)] = (L * n_sig + si) * block_size

    def update(
        self,
        data: np.ndarray | torch.Tensor,
        sig: str,
        layer_idx: int,
        sample_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"PerTokenProjection got {type(data).__name__} for signal "
                f"{sig!r}; expected np.ndarray from stream-project "
                f"(wants_raw=False)."
            )
        assert self._values is not None and self._ctx is not None
        B, S, k = data.shape
        block_offset = self._block_offsets[(layer_idx, sig)]
        s_max = self._ctx.s_max
        # Invariant: the sweep pads batches to the corpus-wide ``s_max``
        # (tokenizer ``padding=True`` without truncation), so ``S <= s_max``.
        # A backend emitting ``S > s_max`` means per-batch padding is out of
        # sync with the sweep's preallocation — fail loud, don't silently
        # truncate tail columns into nowhere.
        if S > s_max:
            raise AssertionError(
                f"PerTokenProjection: batch seq_len S={S} exceeds sweep "
                f"s_max={s_max} for signal {sig!r} at layer {layer_idx}. "
                f"This is a backend bug — per-batch padding must not exceed "
                f"the corpus-wide padded length set by the loader."
            )
        write_cols = S
        for bi, sid in enumerate(sample_ids):
            row_start = block_offset + int(sid) * s_max
            self._values[
                row_start : row_start + write_cols, :k
            ] = data[bi, :write_cols, :]

    def finalize(self) -> dict[str, Any]:
        assert (
            self._values is not None
            and self._ctx is not None
            and self._seq_lengths is not None
        )
        n_sig = len(self._signal_names)
        ctx = self._ctx

        # Offset table: real-token-aware slices.
        block_size = ctx.n_samples * ctx.s_max
        offset_table = np.zeros(
            (ctx.n_samples, ctx.num_layers, n_sig, 2), dtype=np.int32,
        )
        for L in range(ctx.num_layers):
            for si, _sig in enumerate(self._signal_names):
                block_offset = (L * n_sig + si) * block_size
                for sid in range(ctx.n_samples):
                    row_start = block_offset + sid * ctx.s_max
                    row_end = row_start + int(self._seq_lengths[sid])
                    offset_table[sid, L, si, 0] = row_start
                    offset_table[sid, L, si, 1] = row_end

        # Coords — emitted full-width (every row, including padding) to
        # keep the coords table rectangular; callers that want
        # real-token-only slices use the offset table. Vectorized fill
        # per (layer, sig) block.
        total_rows = ctx.num_layers * n_sig * ctx.n_samples * ctx.s_max
        sample_id_arr = np.zeros(total_rows, dtype=np.int32)
        layer_arr = np.zeros(total_rows, dtype=np.int16)
        token_pos_arr = np.zeros(total_rows, dtype=np.int16)
        signal_arr = np.zeros(total_rows, dtype=np.int8)
        sid_pattern = np.repeat(
            np.arange(ctx.n_samples, dtype=np.int32), ctx.s_max,
        )
        tok_pattern = np.tile(
            np.arange(ctx.s_max, dtype=np.int16), ctx.n_samples,
        )
        for L in range(ctx.num_layers):
            for si, _sig in enumerate(self._signal_names):
                block_offset = (L * n_sig + si) * block_size
                sample_id_arr[block_offset : block_offset + block_size] = sid_pattern
                layer_arr[block_offset : block_offset + block_size] = L
                token_pos_arr[block_offset : block_offset + block_size] = tok_pattern
                signal_arr[block_offset : block_offset + block_size] = si

        return {
            "values": self._values,
            "coords": {
                "sample_id": sample_id_arr,
                "layer": layer_arr,
                "token_pos": token_pos_arr,
                "signal": signal_arr,
            },
            "offset_table": offset_table,
            "seq_lengths": self._seq_lengths,
            "signal_names": self._signal_names,
        }


# ---------------------------------------------------------------------------
# Reducers (new-protocol versions)
# ---------------------------------------------------------------------------


class LastTokenReducer:
    """Keep the projection at the last ``True`` mask position per sample.

    Output shape: ``[N, num_layers, n_signals, k_max]`` fp16, where
    ``n_signals`` is the number of signals subscribed to (via ``bases``)
    and ``k_max = max_s bases[s].shape[-1]``. For a sample whose mask has
    no ``True`` entries, the output row is zeros.
    """

    wants_raw: bool = False

    def __init__(
        self,
        masks: Sequence[np.ndarray],
        *,
        bases: dict[str, np.ndarray],
        layers: Sequence[int] | None = None,
    ) -> None:
        if not bases:
            raise ValueError("LastTokenReducer requires at least one basis.")
        self.masks: list[np.ndarray] = [
            np.asarray(m, dtype=bool) for m in masks
        ]
        self.bases = bases
        self.signals: frozenset[str] = frozenset(bases.keys())
        self.layers: frozenset[int] | None = (
            frozenset(layers) if layers is not None else None
        )
        self._signal_names: list[str] = sorted(self.signals)
        self._sig_to_idx = {s: i for i, s in enumerate(self._signal_names)}
        self._k_max = max(int(b.shape[-1]) for b in bases.values())
        self._ctx: SweepContext | None = None

    def init(self, ctx: SweepContext) -> None:
        _check_n_samples(ctx.n_samples, len(self.masks), "LastTokenReducer")
        self._ctx = ctx
        last_true: list[int] = []
        has_true: list[bool] = []
        for m in self.masks:
            idx = np.where(m)[0]
            if len(idx) == 0:
                last_true.append(-1)
                has_true.append(False)
            else:
                last_true.append(int(idx[-1]))
                has_true.append(True)
        self._last_true = last_true
        self._has_true = has_true
        self._out = np.zeros(
            (ctx.n_samples, ctx.num_layers, len(self._signal_names), self._k_max),
            dtype=np.float16,
        )

    def update(
        self,
        data: np.ndarray | torch.Tensor,
        sig: str,
        layer_idx: int,
        sample_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> None:
        if sig not in self._sig_to_idx:
            return  # not our signal
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"LastTokenReducer got {type(data).__name__} for signal "
                f"{sig!r}; expected np.ndarray (wants_raw=False)."
            )
        sig_idx = self._sig_to_idx[sig]
        k = int(self.bases[sig].shape[-1])
        for b in range(data.shape[0]):
            sid = int(sample_ids[b])
            if not self._has_true[sid]:
                continue
            lti = self._last_true[sid]
            real_positions = _real_positions(attention_mask[b])
            if lti >= len(real_positions):
                raise ValueError(
                    f"LastTokenReducer: sample {sid} mask last-true index "
                    f"{lti} >= attention_mask real length "
                    f"{len(real_positions)}. Masks must be derived from "
                    f"the same tokenizer run lmprobe uses."
                )
            self._out[sid, layer_idx, sig_idx, :k] = data[
                b, real_positions[lti], :k
            ]

    def finalize(self) -> np.ndarray:
        return self._out


class MeanReducer:
    """Mean of the projection over ``True`` mask positions per sample.

    Accumulates in fp32, divides by per-sample ``True`` count at
    :meth:`finalize`, returns fp16. Samples with zero ``True`` positions
    produce zeros.
    """

    wants_raw: bool = False

    def __init__(
        self,
        masks: Sequence[np.ndarray],
        *,
        bases: dict[str, np.ndarray],
        layers: Sequence[int] | None = None,
    ) -> None:
        if not bases:
            raise ValueError("MeanReducer requires at least one basis.")
        self.masks: list[np.ndarray] = [
            np.asarray(m, dtype=bool) for m in masks
        ]
        self.bases = bases
        self.signals: frozenset[str] = frozenset(bases.keys())
        self.layers: frozenset[int] | None = (
            frozenset(layers) if layers is not None else None
        )
        self._signal_names: list[str] = sorted(self.signals)
        self._sig_to_idx = {s: i for i, s in enumerate(self._signal_names)}
        self._k_max = max(int(b.shape[-1]) for b in bases.values())
        self._ctx: SweepContext | None = None

    def init(self, ctx: SweepContext) -> None:
        _check_n_samples(ctx.n_samples, len(self.masks), "MeanReducer")
        self._ctx = ctx
        self._sum = np.zeros(
            (ctx.n_samples, ctx.num_layers, len(self._signal_names), self._k_max),
            dtype=np.float32,
        )
        self._count = np.array(
            [int(m.sum()) for m in self.masks], dtype=np.int64,
        )

    def update(
        self,
        data: np.ndarray | torch.Tensor,
        sig: str,
        layer_idx: int,
        sample_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> None:
        if sig not in self._sig_to_idx:
            return
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"MeanReducer got {type(data).__name__} for signal {sig!r}; "
                f"expected np.ndarray (wants_raw=False)."
            )
        sig_idx = self._sig_to_idx[sig]
        k = int(self.bases[sig].shape[-1])
        for b in range(data.shape[0]):
            sid = int(sample_ids[b])
            mask = self.masks[sid]
            real_positions = _real_positions(attention_mask[b])
            seq_len = len(real_positions)
            if len(mask) != seq_len:
                raise ValueError(
                    f"MeanReducer: sample {sid} mask length {len(mask)} != "
                    f"attention_mask real length {seq_len}. Masks must be "
                    f"derived from the same tokenizer run lmprobe uses."
                )
            if not mask.any():
                continue
            selected = data[b, real_positions[mask], :k].astype(np.float32)
            self._sum[sid, layer_idx, sig_idx, :k] += selected.sum(axis=0)

    def finalize(self) -> np.ndarray:
        out = np.zeros(self._sum.shape, dtype=np.float16)
        nonzero = self._count > 0
        if nonzero.any():
            divisor = self._count[nonzero].astype(np.float32)[:, None, None, None]
            out[nonzero] = (self._sum[nonzero] / divisor).astype(np.float16)
        return out


class MeanExclLastNReducer(MeanReducer):
    """Mean excluding the last ``n`` ``True`` positions per sample.

    Falls back to :class:`MeanReducer` semantics for samples with
    ``<= n`` ``True`` positions.
    """

    def __init__(
        self,
        masks: Sequence[np.ndarray],
        *,
        bases: dict[str, np.ndarray],
        n: int = 5,
        layers: Sequence[int] | None = None,
    ) -> None:
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        self.n = int(n)
        derived: list[np.ndarray] = []
        for m in masks:
            arr = np.asarray(m, dtype=bool).copy()
            true_idx = np.where(arr)[0]
            if self.n > 0 and len(true_idx) > self.n:
                arr[true_idx[-self.n :]] = False
            derived.append(arr)
        super().__init__(derived, bases=bases, layers=layers)


# ---------------------------------------------------------------------------
# HiddenStateCapture — classic extract_batch output shape
# ---------------------------------------------------------------------------


class HiddenStateCapture:
    """Capture post-layer residuals at specific layer indices into a
    pre-allocated ``[N, S, H * L_sub]`` tensor.

    Mirrors the classic ``_chunked_forward`` output (``torch.cat`` along
    ``dim=-1``). Preallocating avoids a big concat on the probe training
    hot path.

    Parameters
    ----------
    layer_indices : Sequence[int]
        Which layers to capture. Output H-concat order matches this list.
    dtype : torch.dtype or None
        Output dtype. ``None`` defers to the sweep's model dtype via
        :class:`SweepContext`.
    """

    signals: frozenset[str] = frozenset({"residual"})
    wants_raw: bool = True

    def __init__(
        self,
        layer_indices: Sequence[int],
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.layer_indices = list(layer_indices)
        self.layers: frozenset[int] | None = frozenset(self.layer_indices)
        self._layer_to_slot = {L: i for i, L in enumerate(self.layer_indices)}
        self._dtype = dtype
        self._ctx: SweepContext | None = None
        self._out: torch.Tensor | None = None

    def estimate_bytes(
        self, ctx: SweepContext, batch_size: int,
    ) -> int:
        """Pre-``init`` size estimate for the preallocated ``[N, S, H * L_sub]``
        residual capture buffer."""
        dtype = self._dtype if self._dtype is not None else ctx.dtype
        dtype_bytes = torch.tensor([], dtype=dtype).element_size()
        n_sub = len(self.layer_indices)
        return ctx.n_samples * ctx.s_max * ctx.hidden_dim * n_sub * dtype_bytes

    def init(self, ctx: SweepContext) -> None:
        self._ctx = ctx
        dtype = self._dtype if self._dtype is not None else ctx.dtype
        n_sub = len(self.layer_indices)
        self._out = torch.zeros(
            ctx.n_samples, ctx.s_max, ctx.hidden_dim * n_sub, dtype=dtype,
        )

    def update(
        self,
        data: np.ndarray | torch.Tensor,
        sig: str,
        layer_idx: int,
        sample_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> None:
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"HiddenStateCapture got {type(data).__name__}; expected "
                f"torch.Tensor (wants_raw=True)."
            )
        assert self._out is not None
        slot = self._layer_to_slot[layer_idx]
        B, S, H = data.shape
        col_start = slot * H
        col_end = (slot + 1) * H
        for bi, sid in enumerate(sample_ids):
            self._out[int(sid), :S, col_start:col_end] = data[bi]

    def finalize(self) -> torch.Tensor:
        assert self._out is not None
        return self._out


# ---------------------------------------------------------------------------
# LogitCapture — end-of-sweep, not a per-layer signal
# ---------------------------------------------------------------------------


class LogitCapture:
    """Capture ``lm_head`` output at end of sweep. Shape: ``[N, S, V]``.

    The ``"logits"`` signal is special: it's emitted exactly once per
    sweep, after the final layer completes, by the driver calling
    ``loader.apply_lm_head(final_hs)``. No basis matmul.
    """

    signals: frozenset[str] = frozenset({"logits"})
    layers: frozenset[int] | None = None  # not layer-indexed
    wants_raw: bool = False  # lm_head IS the projection; data comes in as ndarray

    def __init__(
        self,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        self._dtype = dtype
        self._out: torch.Tensor | None = None
        self._seq_lengths: list[int] | None = None
        self._ctx: SweepContext | None = None
        self._out_dtype: torch.dtype = torch.float32

    def init(self, ctx: SweepContext) -> None:
        self._ctx = ctx
        self._seq_lengths = ctx.seq_lengths
        self._out_dtype = self._dtype if self._dtype is not None else torch.float32
        # V (vocab_size) isn't known until the first lm_head output arrives,
        # so allocation is deferred to the first update. N and S come from ctx.

    def update(
        self,
        data: np.ndarray | torch.Tensor,
        sig: str,
        layer_idx: int,
        sample_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> None:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        _B, S, V = data.shape
        if self._out is None:
            assert self._ctx is not None
            self._out = torch.zeros(
                self._ctx.n_samples, self._ctx.s_max, V, dtype=self._out_dtype,
            )
        for bi, sid in enumerate(sample_ids):
            self._out[int(sid), :S, :] = data[bi].to(self._out_dtype)

    def finalize(self) -> torch.Tensor | None:
        return self._out


# ---------------------------------------------------------------------------
# RouterLogitCapture — MoE
# ---------------------------------------------------------------------------


class RouterLogitCapture:
    """Capture router gate logits at specific layers. wants_raw.

    Output: ``dict[layer_idx, torch.Tensor]`` where each tensor has shape
    ``[N, S, n_experts]`` (for ``strategy="output"``) or equivalent
    for ``"input_gate"``. The hook strategy is honored by the driver via
    the extended ``_resolve_signal_hooks`` — see spec.

    Parameters
    ----------
    layer_indices : Sequence[int]
        Which layers to capture.
    strategy : {"output", "input_gate"}
        Hook placement. ``"output"`` hooks the MoE module output.
        ``"input_gate"`` computes logits from the module input via
        ``gate.weight`` — needed for DeepSeek-style MoE that calls
        ``F.linear`` directly.
    """

    signals: frozenset[str] = frozenset({"router_logits"})
    wants_raw: bool = True

    def __init__(
        self,
        layer_indices: Sequence[int],
        *,
        strategy: str = "output",
    ) -> None:
        if strategy not in ("output", "input_gate"):
            raise ValueError(
                f"strategy must be 'output' or 'input_gate', got {strategy!r}"
            )
        self.layer_indices = list(layer_indices)
        self.layers: frozenset[int] | None = frozenset(self.layer_indices)
        self.strategy = strategy
        self._captured: dict[int, list[torch.Tensor]] = {
            L: [] for L in self.layer_indices
        }

    def init(self, ctx: SweepContext) -> None:
        # Nothing to allocate; we concat at finalize.
        pass

    def update(
        self,
        data: np.ndarray | torch.Tensor,
        sig: str,
        layer_idx: int,
        sample_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> None:
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"RouterLogitCapture got {type(data).__name__}; expected "
                f"torch.Tensor (wants_raw=True)."
            )
        self._captured.setdefault(layer_idx, []).append(data)

    def finalize(self) -> dict[int, torch.Tensor]:
        """Concatenate per-layer captures along the batch axis.

        ``_chunked_forward`` today stores the *first* batch's router output
        per layer (line 1522 ``captured_router[layer_idx] = ...[0]``); that
        behavior is a bug (it drops later batches). This accumulator fixes
        it by concatenating — callers who depended on single-batch behavior
        get equivalent output when ``batch_size=len(prompts)``.
        """
        out: dict[int, torch.Tensor] = {}
        for L, tensors in self._captured.items():
            if not tensors:
                continue
            out[L] = torch.cat(tensors, dim=0)
        return out


__all__ = [
    "HiddenStateCapture",
    "LastTokenReducer",
    "LogitCapture",
    "MeanExclLastNReducer",
    "MeanReducer",
    "PCAFit",
    "PerTokenProjection",
    "RouterLogitCapture",
]
