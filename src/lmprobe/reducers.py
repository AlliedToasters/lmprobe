"""Per-sample reducers for in-chunk projection aggregation.

A :class:`Reducer` consumes per-microbatch :math:`[B, S_{pad}, k]`
projections produced inside a :class:`SampleScan` forward sweep and folds
them into a per-sample :math:`[N, L, n_{sig}, k]` summary. Reducers run
immediately after each microbatch's stream-project step, so per-token
projections never accumulate across chunks — the final cached data
product is independent of sequence length and chunk count.

See ``specs/003-per-sample-reduced-projection.md`` for the design
motivation.

Built-in reducers:

- :class:`LastTokenReducer` — projection at the last ``True`` mask
  position per sample.
- :class:`MeanReducer` — mean over ``True`` mask positions per sample.
- :class:`MeanExclLastNReducer` — mean over ``True`` mask positions
  excluding the last ``n`` of them per sample. Falls back to
  :class:`MeanReducer` semantics for samples with ``<= n`` ``True``
  positions.

Masks are carried per-reducer: ``list[np.ndarray]`` of bool, one per
sample, where ``mask[i]`` has length ``seq_len_i`` and marks positions
of interest (e.g. assistant tokens). The reducer aligns ``mask[i]``
against the microbatch's ``attention_mask`` at update time, which makes
it robust to left- vs right-padding. Masks must be derived from the
same tokenizer run lmprobe uses — if ``len(mask[i])`` disagrees with
the number of real tokens in the microbatch, update raises.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Reducer(Protocol):
    """Protocol for per-sample projection reducers.

    A reducer is constructed with per-sample masks, gets
    :meth:`init_state` called once at the start of a sweep, receives
    microbatch projections via :meth:`update`, and produces the final
    per-sample output in :meth:`finalize`.
    """

    def init_state(
        self,
        n_samples: int,
        n_layers: int,
        n_signals: int,
        k: int,
    ) -> Any:
        """Allocate accumulators + bookkeeping for a sweep.

        Returned object is threaded back into :meth:`update` and
        :meth:`finalize`.
        """
        ...

    def update(
        self,
        state: Any,
        proj: np.ndarray,
        sample_ids: np.ndarray,
        layer_idx: int,
        sig_idx: int,
        attention_mask: np.ndarray,
    ) -> None:
        """Fold one microbatch's ``[B, S_pad, k]`` projection into the state.

        Called once per (layer, signal) per microbatch.
        """
        ...

    def finalize(self, state: Any) -> np.ndarray:
        """Return the final ``[N, L, n_sig, k]`` reducer output."""
        ...


def _check_n_samples(n_samples: int, n_masks: int) -> None:
    if n_samples != n_masks:
        raise ValueError(
            f"Reducer received n_samples={n_samples} from scan_forward, "
            f"but was constructed with {n_masks} masks. Mask list length "
            f"must equal the total number of prompts in the sweep."
        )


def _real_positions(attention_mask_row: np.ndarray) -> np.ndarray:
    """Indices of real (non-pad) tokens within a padded microbatch row."""
    return np.where(attention_mask_row.astype(bool))[0]


class LastTokenReducer:
    """Keep the projection at the last ``True`` mask position per sample.

    For a sample whose mask has no ``True`` entries, the output slice is
    left at zero.
    """

    def __init__(self, masks: Sequence[np.ndarray]) -> None:
        self.masks: list[np.ndarray] = [
            np.asarray(m, dtype=bool) for m in masks
        ]

    def init_state(
        self,
        n_samples: int,
        n_layers: int,
        n_signals: int,
        k: int,
    ) -> dict[str, Any]:
        _check_n_samples(n_samples, len(self.masks))
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
        return {
            "out": np.zeros(
                (n_samples, n_layers, n_signals, k), dtype=np.float16,
            ),
            "last_true": last_true,
            "has_true": has_true,
        }

    def update(
        self,
        state: dict[str, Any],
        proj: np.ndarray,
        sample_ids: np.ndarray,
        layer_idx: int,
        sig_idx: int,
        attention_mask: np.ndarray,
    ) -> None:
        out = state["out"]
        last_true = state["last_true"]
        has_true = state["has_true"]
        for b in range(proj.shape[0]):
            sid = int(sample_ids[b])
            if not has_true[sid]:
                continue
            lti = last_true[sid]
            real_positions = _real_positions(attention_mask[b])
            if lti >= len(real_positions):
                raise ValueError(
                    f"LastTokenReducer: sample {sid} mask last-true index "
                    f"{lti} >= attention_mask real length "
                    f"{len(real_positions)}. Masks must be derived from the "
                    f"same tokenizer run lmprobe uses."
                )
            out[sid, layer_idx, sig_idx, :] = proj[b, real_positions[lti], :]

    def finalize(self, state: dict[str, Any]) -> np.ndarray:
        out: np.ndarray = state["out"]
        return out


class MeanReducer:
    """Mean of the projection over ``True`` mask positions per sample.

    Accumulates in fp32, divides by per-sample ``True`` count at
    :meth:`finalize`, and returns fp16. Samples with zero ``True``
    positions produce zeros.
    """

    def __init__(self, masks: Sequence[np.ndarray]) -> None:
        self.masks: list[np.ndarray] = [
            np.asarray(m, dtype=bool) for m in masks
        ]

    def init_state(
        self,
        n_samples: int,
        n_layers: int,
        n_signals: int,
        k: int,
    ) -> dict[str, Any]:
        _check_n_samples(n_samples, len(self.masks))
        count = np.array([int(m.sum()) for m in self.masks], dtype=np.int64)
        return {
            "sum": np.zeros(
                (n_samples, n_layers, n_signals, k), dtype=np.float32,
            ),
            "count": count,
        }

    def update(
        self,
        state: dict[str, Any],
        proj: np.ndarray,
        sample_ids: np.ndarray,
        layer_idx: int,
        sig_idx: int,
        attention_mask: np.ndarray,
    ) -> None:
        sum_arr = state["sum"]
        for b in range(proj.shape[0]):
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
            selected = proj[b, real_positions[mask], :].astype(np.float32)
            sum_arr[sid, layer_idx, sig_idx, :] += selected.sum(axis=0)

    def finalize(self, state: dict[str, Any]) -> np.ndarray:
        sum_arr: np.ndarray = state["sum"]
        count: np.ndarray = state["count"]
        out = np.zeros(sum_arr.shape, dtype=np.float16)
        nonzero = count > 0
        if nonzero.any():
            divisor = count[nonzero].astype(np.float32)[:, None, None, None]
            out[nonzero] = (sum_arr[nonzero] / divisor).astype(np.float16)
        return out


class MeanExclLastNReducer(MeanReducer):
    """Mean of the projection over ``True`` mask positions excluding the
    last ``n`` of them per sample.

    When a sample has ``<= n`` ``True`` positions, behaves identically to
    :class:`MeanReducer` over the full mask.
    """

    def __init__(self, masks: Sequence[np.ndarray], n: int = 5) -> None:
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        self.n = int(n)
        derived: list[np.ndarray] = []
        for m in masks:
            arr = np.asarray(m, dtype=bool).copy()
            true_idx = np.where(arr)[0]
            if self.n > 0 and len(true_idx) > self.n:
                arr[true_idx[-self.n:]] = False
            derived.append(arr)
        super().__init__(derived)
