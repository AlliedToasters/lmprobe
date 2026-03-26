"""Tests for pooling strategies."""

import pytest
import torch

from lmprobe.pooling import (
    INFERENCE_POOLING_STRATEGIES,
    _is_valid_inference_strategy,
    get_pooling_fn,
    parse_pooling_strategy,
    pool_all,
    pool_first_token,
    pool_last_token,
    pool_max,
    pool_mean,
    pool_min,
    reduce_scores,
    resolve_pooling,
)


class TestParsePoolingStrategy:
    """Tests for parse_pooling_strategy."""

    def test_bare_activation_strategies(self):
        """Bare names like 'mean', 'last_token' default to activation pooling."""
        for strategy in ("last_token", "first_token", "mean"):
            result = parse_pooling_strategy(strategy)
            assert result.base_strategy == strategy
            assert result.is_score_pooling is False
            assert result.raw == strategy

    def test_bare_score_strategies(self):
        """Bare 'max' and 'min' default to score pooling."""
        for strategy in ("max", "min"):
            result = parse_pooling_strategy(strategy)
            assert result.base_strategy == strategy
            assert result.is_score_pooling is True
            assert result.raw == strategy

    def test_all_strategy(self):
        """'all' is parsed as activation pooling."""
        result = parse_pooling_strategy("all")
        assert result.base_strategy == "all"
        assert result.is_score_pooling is False

    def test_score_prefix(self):
        """'score:mean' forces score-level pooling."""
        result = parse_pooling_strategy("score:mean")
        assert result.base_strategy == "mean"
        assert result.is_score_pooling is True

    def test_activation_prefix(self):
        """'activation:max' forces activation-level pooling."""
        result = parse_pooling_strategy("activation:max")
        assert result.base_strategy == "max"
        assert result.is_score_pooling is False

    def test_unknown_prefix_raises(self):
        """Unknown prefix raises ValueError."""
        with pytest.raises(ValueError, match="Unknown pooling prefix"):
            parse_pooling_strategy("foo:mean")

    def test_unknown_base_with_prefix_raises(self):
        """Unknown base strategy with valid prefix raises ValueError."""
        with pytest.raises(ValueError, match="Unknown base pooling strategy"):
            parse_pooling_strategy("score:nonexistent")

    def test_unknown_bare_strategy_raises(self):
        """Unknown bare strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            parse_pooling_strategy("nonexistent")


class TestPoolFunctions:
    """Tests for individual pool_* functions."""

    @pytest.fixture
    def activations(self):
        """(batch=2, seq_len=4, hidden_dim=3)"""
        return torch.tensor([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        ])

    @pytest.fixture
    def attention_mask(self):
        """Mask with right-padding: second sequence has 3 real tokens."""
        return torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ])

    def test_pool_last_token_no_mask(self, activations):
        result = pool_last_token(activations, attention_mask=None)
        assert result.shape == (2, 3)
        assert torch.allclose(result[0], torch.tensor([10.0, 11.0, 12.0]))

    def test_pool_last_token_with_mask(self, activations, attention_mask):
        result = pool_last_token(activations, attention_mask)
        assert result.shape == (2, 3)
        # First seq: last real token = index 3
        assert torch.allclose(result[0], torch.tensor([10.0, 11.0, 12.0]))
        # Second seq: last real token = index 2
        assert torch.allclose(result[1], torch.tensor([0.7, 0.8, 0.9]))

    def test_pool_first_token(self, activations):
        result = pool_first_token(activations, attention_mask=None)
        assert result.shape == (2, 3)
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0, 3.0]))

    def test_pool_first_token_ignores_mask(self, activations, attention_mask):
        """First token pooling ignores the attention mask."""
        result_no_mask = pool_first_token(activations)
        result_with_mask = pool_first_token(activations, attention_mask)
        assert torch.allclose(result_no_mask, result_with_mask)

    def test_pool_mean_no_mask(self, activations):
        result = pool_mean(activations, attention_mask=None)
        assert result.shape == (2, 3)
        expected_0 = activations[0].mean(dim=0)
        assert torch.allclose(result[0], expected_0)

    def test_pool_mean_with_mask(self, activations, attention_mask):
        result = pool_mean(activations, attention_mask)
        assert result.shape == (2, 3)
        # Second seq: mean of first 3 tokens only
        expected_1 = activations[1, :3].mean(dim=0)
        assert torch.allclose(result[1], expected_1)

    def test_pool_max_no_mask(self, activations):
        result = pool_max(activations, attention_mask=None)
        assert result.shape == (2, 3)
        assert torch.allclose(result[0], torch.tensor([10.0, 11.0, 12.0]))

    def test_pool_max_with_mask(self, activations, attention_mask):
        result = pool_max(activations, attention_mask)
        assert result.shape == (2, 3)
        # Second seq: max of first 3 tokens
        assert torch.allclose(result[1], torch.tensor([0.7, 0.8, 0.9]))

    def test_pool_min_no_mask(self, activations):
        result = pool_min(activations, attention_mask=None)
        assert result.shape == (2, 3)
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0, 3.0]))

    def test_pool_min_with_mask(self, activations, attention_mask):
        result = pool_min(activations, attention_mask)
        assert result.shape == (2, 3)
        # Second seq: min of first 3 tokens
        assert torch.allclose(result[1], torch.tensor([0.1, 0.2, 0.3]))

    def test_pool_all(self, activations):
        result = pool_all(activations)
        assert torch.equal(result, activations)


class TestGetPoolingFn:
    """Tests for get_pooling_fn."""

    def test_activation_strategies_return_correct_fn(self):
        assert get_pooling_fn("last_token") is pool_last_token
        assert get_pooling_fn("first_token") is pool_first_token
        assert get_pooling_fn("mean") is pool_mean
        assert get_pooling_fn("all") is pool_all

    def test_score_strategies_return_pool_all(self):
        """Score-level pooling returns pool_all (tokens needed for classification first)."""
        assert get_pooling_fn("max") is pool_all
        assert get_pooling_fn("min") is pool_all
        assert get_pooling_fn("score:mean") is pool_all

    def test_activation_prefix_overrides_default(self):
        """'activation:max' returns pool_max instead of pool_all."""
        assert get_pooling_fn("activation:max") is pool_max
        assert get_pooling_fn("activation:min") is pool_min


class TestReduceScores:
    """Tests for reduce_scores."""

    @pytest.fixture
    def scores_2d(self):
        """(batch=2, seq_len=3)"""
        return torch.tensor([
            [1.0, 3.0, 2.0],
            [4.0, 5.0, 6.0],
        ])

    @pytest.fixture
    def scores_3d(self):
        """(batch=2, seq_len=3, n_classes=2)"""
        return torch.tensor([
            [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5]],
            [[0.8, 0.2], [0.6, 0.4], [0.4, 0.6]],
        ])

    @pytest.fixture
    def mask(self):
        return torch.tensor([[1, 1, 1], [1, 1, 0]])

    def test_reduce_max_no_mask(self, scores_2d):
        result = reduce_scores(scores_2d, "max")
        assert torch.allclose(result, torch.tensor([3.0, 6.0]))

    def test_reduce_max_with_mask(self, scores_2d, mask):
        result = reduce_scores(scores_2d, "max", attention_mask=mask)
        # Second seq: max of first 2 tokens = 5.0
        assert torch.allclose(result, torch.tensor([3.0, 5.0]))

    def test_reduce_min_no_mask(self, scores_2d):
        result = reduce_scores(scores_2d, "min")
        assert torch.allclose(result, torch.tensor([1.0, 4.0]))

    def test_reduce_min_with_mask(self, scores_2d, mask):
        result = reduce_scores(scores_2d, "min", attention_mask=mask)
        assert torch.allclose(result, torch.tensor([1.0, 4.0]))

    def test_reduce_mean_no_mask(self, scores_2d):
        result = reduce_scores(scores_2d, "mean")
        assert torch.allclose(result, torch.tensor([2.0, 5.0]))

    def test_reduce_mean_with_mask(self, scores_2d, mask):
        result = reduce_scores(scores_2d, "score:mean", attention_mask=mask)
        # Second seq: mean of first 2 tokens = (4+5)/2 = 4.5
        assert torch.allclose(result, torch.tensor([2.0, 4.5]))

    def test_reduce_last_token_no_mask(self, scores_2d):
        result = reduce_scores(scores_2d, "last_token")
        assert torch.allclose(result, torch.tensor([2.0, 6.0]))

    def test_reduce_last_token_with_mask(self, scores_2d, mask):
        result = reduce_scores(scores_2d, "last_token", attention_mask=mask)
        # Second seq: last real token = index 1 → 5.0
        assert torch.allclose(result, torch.tensor([2.0, 5.0]))

    def test_reduce_first_token(self, scores_2d):
        result = reduce_scores(scores_2d, "first_token")
        assert torch.allclose(result, torch.tensor([1.0, 4.0]))

    def test_reduce_3d_max_with_mask(self, scores_3d, mask):
        result = reduce_scores(scores_3d, "max", attention_mask=mask)
        assert result.shape == (2, 2)

    def test_reduce_3d_min_with_mask(self, scores_3d, mask):
        result = reduce_scores(scores_3d, "min", attention_mask=mask)
        assert result.shape == (2, 2)

    def test_reduce_3d_mean_with_mask(self, scores_3d, mask):
        result = reduce_scores(scores_3d, "score:mean", attention_mask=mask)
        assert result.shape == (2, 2)

    def test_reduce_3d_last_token_with_mask(self, scores_3d, mask):
        result = reduce_scores(scores_3d, "last_token", attention_mask=mask)
        assert result.shape == (2, 2)

    def test_reduce_3d_first_token(self, scores_3d):
        result = reduce_scores(scores_3d, "first_token")
        assert result.shape == (2, 2)

    def test_reduce_unknown_strategy_raises(self, scores_2d):
        with pytest.raises(ValueError, match="does not support strategy"):
            reduce_scores(scores_2d, "all")


class TestIsValidInferenceStrategy:
    """Tests for _is_valid_inference_strategy."""

    def test_bare_strategies(self):
        for s in INFERENCE_POOLING_STRATEGIES:
            assert _is_valid_inference_strategy(s) is True

    def test_prefixed_strategies(self):
        assert _is_valid_inference_strategy("score:mean") is True
        assert _is_valid_inference_strategy("activation:max") is True

    def test_invalid_strategy(self):
        assert _is_valid_inference_strategy("nonexistent") is False


class TestResolvePooling:
    """Tests for resolve_pooling."""

    def test_defaults(self):
        train, inference = resolve_pooling(None, None, None)
        assert train == "last_token"
        assert inference == "last_token"

    def test_pooling_sets_both(self):
        train, inference = resolve_pooling("mean", None, None)
        assert train == "mean"
        assert inference == "mean"

    def test_train_pooling_overrides(self):
        train, inference = resolve_pooling("mean", "first_token", None)
        assert train == "first_token"
        assert inference == "mean"

    def test_inference_pooling_overrides(self):
        train, inference = resolve_pooling("mean", None, "max")
        assert train == "mean"
        assert inference == "max"

    def test_prefixed_inference_pooling(self):
        train, inference = resolve_pooling("last_token", None, "score:mean")
        assert train == "last_token"
        assert inference == "score:mean"

    def test_invalid_train_pooling_raises(self):
        with pytest.raises(ValueError, match="Invalid train_pooling"):
            resolve_pooling(None, "max", None)

    def test_invalid_inference_pooling_raises(self):
        with pytest.raises(ValueError, match="Invalid inference_pooling"):
            resolve_pooling(None, None, "nonexistent")

    def test_all_for_train(self):
        train, inference = resolve_pooling(None, "all", None)
        assert train == "all"
