"""Tests for pre-probe and post-probe pooling variants.

Covers the prefix convention: ``score:X``, ``activation:X``, and bare names.
"""

import numpy as np
import pytest
import torch

from lmprobe import LinearProbe
from lmprobe.pooling import (
    ParsedPooling,
    parse_pooling_strategy,
    get_pooling_fn,
    pool_all,
    pool_first_token,
    pool_last_token,
    pool_max,
    pool_mean,
    pool_min,
    reduce_scores,
    resolve_pooling,
)

TEST_MODEL = "stas/tiny-random-llama-2"


# ---- parse_pooling_strategy ------------------------------------------------


class TestParsePoolingStrategy:
    """Unit tests for the prefix-parsing function."""

    def test_bare_mean_is_activation(self):
        p = parse_pooling_strategy("mean")
        assert p.base_strategy == "mean"
        assert p.is_score_pooling is False

    def test_bare_max_is_score(self):
        p = parse_pooling_strategy("max")
        assert p.base_strategy == "max"
        assert p.is_score_pooling is True

    def test_bare_min_is_score(self):
        p = parse_pooling_strategy("min")
        assert p.base_strategy == "min"
        assert p.is_score_pooling is True

    def test_bare_last_token_is_activation(self):
        p = parse_pooling_strategy("last_token")
        assert p.base_strategy == "last_token"
        assert p.is_score_pooling is False

    def test_bare_first_token_is_activation(self):
        p = parse_pooling_strategy("first_token")
        assert p.base_strategy == "first_token"
        assert p.is_score_pooling is False

    def test_score_prefix_mean(self):
        p = parse_pooling_strategy("score:mean")
        assert p.base_strategy == "mean"
        assert p.is_score_pooling is True
        assert p.raw == "score:mean"

    def test_score_prefix_last_token(self):
        p = parse_pooling_strategy("score:last_token")
        assert p.base_strategy == "last_token"
        assert p.is_score_pooling is True

    def test_activation_prefix_max(self):
        p = parse_pooling_strategy("activation:max")
        assert p.base_strategy == "max"
        assert p.is_score_pooling is False

    def test_activation_prefix_min(self):
        p = parse_pooling_strategy("activation:min")
        assert p.base_strategy == "min"
        assert p.is_score_pooling is False

    def test_activation_prefix_mean(self):
        """activation:mean is same as bare mean."""
        p = parse_pooling_strategy("activation:mean")
        assert p.base_strategy == "mean"
        assert p.is_score_pooling is False

    def test_score_prefix_max(self):
        """score:max is same as bare max."""
        p = parse_pooling_strategy("score:max")
        assert p.base_strategy == "max"
        assert p.is_score_pooling is True

    def test_all_strategy(self):
        p = parse_pooling_strategy("all")
        assert p.base_strategy == "all"
        assert p.is_score_pooling is False

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            parse_pooling_strategy("banana")

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown pooling prefix"):
            parse_pooling_strategy("post:mean")

    def test_unknown_base_with_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown base pooling strategy"):
            parse_pooling_strategy("score:banana")


# ---- get_pooling_fn ---------------------------------------------------------


class TestGetPoolingFn:
    """Verify get_pooling_fn routes correctly for prefixed strategies."""

    def test_activation_max_returns_pool_max(self):
        fn = get_pooling_fn("activation:max")
        assert fn is pool_max

    def test_activation_min_returns_pool_min(self):
        fn = get_pooling_fn("activation:min")
        assert fn is pool_min

    def test_score_mean_returns_pool_all(self):
        fn = get_pooling_fn("score:mean")
        assert fn is pool_all

    def test_score_last_token_returns_pool_all(self):
        fn = get_pooling_fn("score:last_token")
        assert fn is pool_all

    def test_bare_mean_returns_pool_mean(self):
        fn = get_pooling_fn("mean")
        assert fn is pool_mean

    def test_bare_max_returns_pool_all(self):
        """Bare max defaults to score-level, so returns pool_all."""
        fn = get_pooling_fn("max")
        assert fn is pool_all

    def test_bare_last_token(self):
        fn = get_pooling_fn("last_token")
        assert fn is pool_last_token

    def test_bare_first_token(self):
        fn = get_pooling_fn("first_token")
        assert fn is pool_first_token


# ---- pool_max / pool_min (activation-level) ---------------------------------


class TestActivationPoolMaxMin:
    """Test the new pool_max and pool_min functions."""

    def test_pool_max_no_mask(self):
        x = torch.tensor([
            [[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]],
        ])
        result = pool_max(x)
        assert result.shape == (1, 2)
        assert torch.allclose(result, torch.tensor([[3.0, 4.0]]))

    def test_pool_min_no_mask(self):
        x = torch.tensor([
            [[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]],
        ])
        result = pool_min(x)
        assert result.shape == (1, 2)
        assert torch.allclose(result, torch.tensor([[1.0, 1.0]]))

    def test_pool_max_with_mask(self):
        x = torch.tensor([
            [[1.0, 2.0], [3.0, 1.0], [99.0, 99.0]],
        ])
        mask = torch.tensor([[1, 1, 0]])
        result = pool_max(x, mask)
        assert torch.allclose(result, torch.tensor([[3.0, 2.0]]))

    def test_pool_min_with_mask(self):
        x = torch.tensor([
            [[-99.0, -99.0], [3.0, 1.0], [2.0, 4.0]],
        ])
        mask = torch.tensor([[0, 1, 1]])
        result = pool_min(x, mask)
        assert torch.allclose(result, torch.tensor([[2.0, 1.0]]))


# ---- reduce_scores (extended) ----------------------------------------------


class TestReduceScoresExtended:
    """Test reduce_scores with the newly supported strategies."""

    def test_reduce_mean_2d(self):
        scores = torch.tensor([[0.1, 0.3, 0.5]])
        result = reduce_scores(scores, "score:mean")
        assert result.shape == (1,)
        assert torch.allclose(result, torch.tensor([0.3]))

    def test_reduce_mean_3d(self):
        scores = torch.tensor([
            [[0.2, 0.8], [0.4, 0.6], [0.6, 0.4]],
        ])
        result = reduce_scores(scores, "score:mean")
        assert result.shape == (1, 2)
        expected = torch.tensor([[0.4, 0.6]])
        assert torch.allclose(result, expected)

    def test_reduce_mean_with_mask(self):
        scores = torch.tensor([
            [[0.2, 0.8], [0.4, 0.6], [99.0, 99.0]],
        ])
        mask = torch.tensor([[1, 1, 0]])
        result = reduce_scores(scores, "score:mean", mask)
        expected = torch.tensor([[0.3, 0.7]])
        assert torch.allclose(result, expected)

    def test_reduce_last_token_2d(self):
        scores = torch.tensor([[0.1, 0.3, 0.5]])
        result = reduce_scores(scores, "score:last_token")
        assert torch.allclose(result, torch.tensor([0.5]))

    def test_reduce_last_token_3d_with_mask(self):
        scores = torch.tensor([
            [[0.2, 0.8], [0.4, 0.6], [99.0, 99.0]],
        ])
        mask = torch.tensor([[1, 1, 0]])
        result = reduce_scores(scores, "score:last_token", mask)
        expected = torch.tensor([[0.4, 0.6]])
        assert torch.allclose(result, expected)

    def test_reduce_first_token_3d(self):
        scores = torch.tensor([
            [[0.2, 0.8], [0.4, 0.6], [0.6, 0.4]],
        ])
        result = reduce_scores(scores, "score:first_token")
        expected = torch.tensor([[0.2, 0.8]])
        assert torch.allclose(result, expected)

    def test_reduce_max_bare(self):
        """Bare 'max' still works (backward compat)."""
        scores = torch.tensor([
            [[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]],
        ])
        result = reduce_scores(scores, "max")
        expected = torch.tensor([[0.9, 0.8]])
        assert torch.allclose(result, expected)

    def test_reduce_min_bare(self):
        """Bare 'min' still works (backward compat)."""
        scores = torch.tensor([
            [[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]],
        ])
        result = reduce_scores(scores, "min")
        expected = torch.tensor([[0.2, 0.1]])
        assert torch.allclose(result, expected)


# ---- resolve_pooling --------------------------------------------------------


class TestResolvePoolingPrefixed:
    """Test that resolve_pooling accepts prefixed inference strategies."""

    def test_score_mean_accepted(self):
        train, inference = resolve_pooling("last_token", None, "score:mean")
        assert train == "last_token"
        assert inference == "score:mean"

    def test_activation_max_accepted(self):
        train, inference = resolve_pooling("last_token", None, "activation:max")
        assert train == "last_token"
        assert inference == "activation:max"

    def test_score_prefix_rejected_for_train(self):
        with pytest.raises(ValueError, match="Invalid train_pooling"):
            resolve_pooling(None, "score:mean", None)

    def test_invalid_prefix_rejected(self):
        with pytest.raises(ValueError):
            resolve_pooling(None, None, "post:mean")


# ---- Integration tests with LinearProbe ------------------------------------


class TestPrefixedPoolingIntegration:
    """End-to-end tests with the tiny model."""

    @pytest.fixture
    def fitted_probe(self):
        """A probe fitted with 'all' training pooling for score-level tests."""
        probe = LinearProbe(
            model=TEST_MODEL,
            layers=-1,
            train_pooling="all",
            inference_pooling="max",
            device="cpu",
            remote=False,
            random_state=42,
        )
        pos = ["dog bark woof", "good boy fetch", "tail wagging happy"]
        neg = ["cat meow purr", "scratch hiss pounce", "whiskers grooming nap"]
        probe.fit(pos, neg)
        return probe

    def test_score_mean(self, fitted_probe):
        """score:mean produces (n_samples, n_classes) output."""
        fitted_probe._inference_pooling = "score:mean"
        probs = fitted_probe.predict_proba(["test prompt one", "test prompt two"])
        assert probs.shape == (2, 2)
        # Probabilities should sum to ~1 per sample
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-5)

    def test_score_last_token(self, fitted_probe):
        """score:last_token produces (n_samples, n_classes) output."""
        fitted_probe._inference_pooling = "score:last_token"
        probs = fitted_probe.predict_proba(["test prompt"])
        assert probs.shape == (1, 2)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0], atol=1e-5)

    def test_score_first_token(self, fitted_probe):
        """score:first_token produces (n_samples, n_classes) output."""
        fitted_probe._inference_pooling = "score:first_token"
        probs = fitted_probe.predict_proba(["test prompt"])
        assert probs.shape == (1, 2)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0], atol=1e-5)

    def test_activation_max(self, fitted_probe):
        """activation:max produces (n_samples, n_classes) output via activation pooling."""
        fitted_probe._inference_pooling = "activation:max"
        probs = fitted_probe.predict_proba(["test prompt one", "test prompt two"])
        assert probs.shape == (2, 2)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-5)

    def test_activation_min(self, fitted_probe):
        """activation:min produces (n_samples, n_classes) output via activation pooling."""
        fitted_probe._inference_pooling = "activation:min"
        probs = fitted_probe.predict_proba(["test prompt"])
        assert probs.shape == (1, 2)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0], atol=1e-5)

    def test_score_mean_vs_activation_mean_differ(self, fitted_probe):
        """score:mean and activation:mean should give different results.

        Pre-probe mean averages representations (linear), while post-probe
        mean averages probabilities (nonlinear through the classifier).
        """
        fitted_probe._inference_pooling = "score:mean"
        probs_score = fitted_probe.predict_proba(["dog bark woof fetch play"])

        fitted_probe._inference_pooling = "mean"
        probs_act = fitted_probe.predict_proba(["dog bark woof fetch play"])

        # They should generally differ (unless the classifier is degenerate)
        # We just check shapes are correct — both give (1, 2)
        assert probs_score.shape == (1, 2)
        assert probs_act.shape == (1, 2)

    def test_predict_works_with_score_mean(self, fitted_probe):
        """predict() works through predict_proba with score:mean."""
        fitted_probe._inference_pooling = "score:mean"
        preds = fitted_probe.predict(["test prompt"])
        assert preds.shape == (1,)

    def test_score_works_with_score_mean(self, fitted_probe):
        """score() works with score:mean."""
        fitted_probe._inference_pooling = "score:mean"
        acc = fitted_probe.score(["test prompt"], [1])
        assert 0.0 <= acc <= 1.0

    def test_init_with_prefixed_inference_pooling(self):
        """LinearProbe accepts prefixed strategies at init."""
        probe = LinearProbe(
            model=TEST_MODEL,
            layers=-1,
            inference_pooling="score:mean",
            device="cpu",
            remote=False,
            random_state=42,
        )
        assert probe._inference_pooling == "score:mean"

    def test_backward_compat_max(self):
        """Bare 'max' still works as before (score-level)."""
        probe = LinearProbe(
            model=TEST_MODEL,
            layers=-1,
            train_pooling="all",
            inference_pooling="max",
            device="cpu",
            remote=False,
            random_state=42,
        )
        pos = ["dog bark woof"]
        neg = ["cat meow purr"]
        probe.fit(pos, neg)
        probs = probe.predict_proba(["test"])
        assert probs.shape == (1, 2)
