"""Unit tests for per-sample reducers (spec 003).

Exercises the reducer protocol implementations in isolation — no model,
no backend. Verifies LastTokenReducer / MeanReducer / MeanExclLastNReducer
produce the expected per-sample outputs given synthetic projection
microbatches and attention masks.
"""

from __future__ import annotations

import numpy as np
import pytest

from lmprobe.reducers import (
    LastTokenReducer,
    MeanExclLastNReducer,
    MeanReducer,
    Reducer,
)


def _mask(length: int, true_positions: list[int]) -> np.ndarray:
    m = np.zeros(length, dtype=bool)
    m[true_positions] = True
    return m


class TestLastTokenReducer:
    def test_selects_last_true_position(self):
        masks = [_mask(4, [0, 2]), _mask(5, [1, 4])]
        red = LastTokenReducer(masks)
        state = red.init_state(n_samples=2, n_layers=1, n_signals=1, k=3)

        proj = np.array(
            [
                [
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [9.0, 9.0, 9.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0],
                    [5.0, 5.0, 5.0],
                    [7.0, 7.0, 7.0],
                ],
            ],
            dtype=np.float16,
        )
        attention = np.array(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
        red.update(state, proj, [0, 1], 0, 0, attention)
        out = red.finalize(state)
        assert out.shape == (2, 1, 1, 3)
        np.testing.assert_array_equal(out[0, 0, 0], [9.0, 9.0, 9.0])
        np.testing.assert_array_equal(out[1, 0, 0], [7.0, 7.0, 7.0])

    def test_no_true_positions_yields_zero(self):
        masks = [_mask(3, [])]
        red = LastTokenReducer(masks)
        state = red.init_state(1, 1, 1, 2)
        proj = np.full((1, 3, 2), 5.0, dtype=np.float16)
        attn = np.ones((1, 3), dtype=bool)
        red.update(state, proj, [0], 0, 0, attn)
        out = red.finalize(state)
        np.testing.assert_array_equal(out[0, 0, 0], [0.0, 0.0])

    def test_mismatch_between_mask_and_attention_raises(self):
        masks = [_mask(10, [0, 9])]
        red = LastTokenReducer(masks)
        state = red.init_state(1, 1, 1, 2)
        proj = np.zeros((1, 4, 2), dtype=np.float16)
        attn = np.ones((1, 4), dtype=bool)
        with pytest.raises(ValueError, match="last-true index"):
            red.update(state, proj, [0], 0, 0, attn)

    def test_wrong_n_samples_raises(self):
        red = LastTokenReducer([_mask(3, [0])])
        with pytest.raises(ValueError, match="Mask list length"):
            red.init_state(n_samples=2, n_layers=1, n_signals=1, k=2)

    def test_multi_layer_multi_signal_axes(self):
        masks = [_mask(3, [2])]
        red = LastTokenReducer(masks)
        state = red.init_state(n_samples=1, n_layers=2, n_signals=3, k=2)
        proj = np.array([[[1, 1], [2, 2], [3, 3]]], dtype=np.float16)
        attn = np.ones((1, 3), dtype=bool)
        red.update(state, proj, [0], layer_idx=1, sig_idx=2, attention_mask=attn)
        out = red.finalize(state)
        assert out.shape == (1, 2, 3, 2)
        np.testing.assert_array_equal(out[0, 1, 2], [3, 3])
        # Unvisited cells remain zero.
        np.testing.assert_array_equal(out[0, 0, 0], [0, 0])


class TestMeanReducer:
    def test_mean_over_true_positions(self):
        masks = [_mask(4, [0, 2, 3])]
        red = MeanReducer(masks)
        state = red.init_state(1, 1, 1, 2)
        proj = np.array(
            [[[2.0, 4.0], [10.0, 10.0], [4.0, 6.0], [6.0, 8.0]]],
            dtype=np.float16,
        )
        attn = np.ones((1, 4), dtype=bool)
        red.update(state, proj, [0], 0, 0, attn)
        out = red.finalize(state)
        # mean of (2,4,6) positions = (2,4), (4,6), (6,8) -> avg (4, 6)
        np.testing.assert_allclose(out[0, 0, 0], [4.0, 6.0], atol=1e-2)

    def test_empty_mask_produces_zero(self):
        masks = [_mask(3, [])]
        red = MeanReducer(masks)
        state = red.init_state(1, 1, 1, 2)
        proj = np.full((1, 3, 2), 42.0, dtype=np.float16)
        attn = np.ones((1, 3), dtype=bool)
        red.update(state, proj, [0], 0, 0, attn)
        out = red.finalize(state)
        np.testing.assert_array_equal(out[0, 0, 0], [0.0, 0.0])

    def test_length_mismatch_raises(self):
        masks = [_mask(5, [0, 4])]
        red = MeanReducer(masks)
        state = red.init_state(1, 1, 1, 2)
        proj = np.zeros((1, 3, 2), dtype=np.float16)
        attn = np.ones((1, 3), dtype=bool)
        with pytest.raises(ValueError, match="mask length"):
            red.update(state, proj, [0], 0, 0, attn)

    def test_mean_aligns_with_right_padding(self):
        masks = [_mask(3, [0, 2])]
        red = MeanReducer(masks)
        state = red.init_state(1, 1, 1, 2)
        proj = np.array(
            [[[2.0, 4.0], [99.0, 99.0], [6.0, 8.0], [0.0, 0.0]]],
            dtype=np.float16,
        )
        attn = np.array([[1, 1, 1, 0]], dtype=bool)
        red.update(state, proj, [0], 0, 0, attn)
        out = red.finalize(state)
        np.testing.assert_allclose(out[0, 0, 0], [4.0, 6.0], atol=1e-2)

    def test_mean_aligns_with_left_padding(self):
        # Left-padded microbatch: real tokens at positions 1..3; mask says
        # first and third real token count.
        masks = [_mask(3, [0, 2])]
        red = MeanReducer(masks)
        state = red.init_state(1, 1, 1, 2)
        proj = np.array(
            [[[0.0, 0.0], [2.0, 4.0], [99.0, 99.0], [6.0, 8.0]]],
            dtype=np.float16,
        )
        attn = np.array([[0, 1, 1, 1]], dtype=bool)
        red.update(state, proj, [0], 0, 0, attn)
        out = red.finalize(state)
        np.testing.assert_allclose(out[0, 0, 0], [4.0, 6.0], atol=1e-2)


class TestMeanExclLastNReducer:
    def test_excludes_last_n_true_positions(self):
        masks = [_mask(6, [0, 1, 3, 4, 5])]  # 5 True positions
        red = MeanExclLastNReducer(masks, n=2)
        state = red.init_state(1, 1, 1, 1)
        proj = np.array(
            [[[1], [2], [0], [3], [100], [100]]], dtype=np.float16,
        )
        attn = np.ones((1, 6), dtype=bool)
        red.update(state, proj, [0], 0, 0, attn)
        out = red.finalize(state)
        # After excluding last 2 True positions (indices 4, 5): True positions
        # become [0, 1, 3]. Values: 1, 2, 3. Mean = 2.
        np.testing.assert_allclose(out[0, 0, 0], [2.0], atol=1e-2)

    def test_falls_back_when_le_n_true_positions(self):
        masks = [_mask(4, [0, 3])]  # 2 True positions, n=5 -> fallback
        red = MeanExclLastNReducer(masks, n=5)
        state = red.init_state(1, 1, 1, 1)
        proj = np.array([[[2], [0], [0], [4]]], dtype=np.float16)
        attn = np.ones((1, 4), dtype=bool)
        red.update(state, proj, [0], 0, 0, attn)
        out = red.finalize(state)
        np.testing.assert_allclose(out[0, 0, 0], [3.0], atol=1e-2)

    def test_n_zero_equivalent_to_mean_reducer(self):
        masks = [_mask(3, [0, 2])]
        excl = MeanExclLastNReducer(masks, n=0)
        mean = MeanReducer(masks)
        s1 = excl.init_state(1, 1, 1, 1)
        s2 = mean.init_state(1, 1, 1, 1)
        proj = np.array([[[2], [0], [4]]], dtype=np.float16)
        attn = np.ones((1, 3), dtype=bool)
        excl.update(s1, proj, [0], 0, 0, attn)
        mean.update(s2, proj, [0], 0, 0, attn)
        np.testing.assert_array_equal(excl.finalize(s1), mean.finalize(s2))

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            MeanExclLastNReducer([_mask(3, [0])], n=-1)


class TestProtocolConformance:
    def test_builtins_satisfy_reducer_protocol(self):
        masks = [_mask(3, [0, 2])]
        assert isinstance(LastTokenReducer(masks), Reducer)
        assert isinstance(MeanReducer(masks), Reducer)
        assert isinstance(MeanExclLastNReducer(masks, n=1), Reducer)
