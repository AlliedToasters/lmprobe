"""Tests for ProbeEnsemble: multi-probe ensemble support."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# Re-use shared fixtures from conftest
from conftest import NEGATIVE_PROMPTS, POSITIVE_PROMPTS, TEST_PROMPTS

from lmprobe import Probe
from lmprobe.ensemble import ProbeEnsemble

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_probe(model: str, **kwargs) -> Probe:
    """Create a minimal Probe for testing."""
    defaults = dict(
        layers=-1,
        device="cpu",
        remote=False,
        random_state=42,
        classifier="logistic_regression",
    )
    defaults.update(kwargs)
    return Probe(model=model, **defaults)


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_empty_probes_raises(self, tiny_model):
        with pytest.raises(ValueError, match="non-empty"):
            ProbeEnsemble([])

    def test_invalid_voting_raises(self, tiny_model):
        p = _make_probe(tiny_model)
        with pytest.raises(ValueError, match="Unknown voting strategy"):
            ProbeEnsemble([p], voting="invalid")

    def test_mismatched_weights_raises(self, tiny_model):
        p = _make_probe(tiny_model)
        with pytest.raises(ValueError, match="weights length"):
            ProbeEnsemble([p], weights=[1.0, 2.0])

    def test_non_positive_weights_raises(self, tiny_model):
        p1 = _make_probe(tiny_model)
        p2 = _make_probe(tiny_model)
        with pytest.raises(ValueError, match="must sum to a positive"):
            ProbeEnsemble([p1, p2], weights=[0.0, 0.0])

    def test_default_uniform_weights(self, tiny_model):
        p1 = _make_probe(tiny_model)
        p2 = _make_probe(tiny_model)
        ens = ProbeEnsemble([p1, p2])
        np.testing.assert_allclose(ens.weights_, [0.5, 0.5])

    def test_custom_weights_normalized(self, tiny_model):
        p1 = _make_probe(tiny_model)
        p2 = _make_probe(tiny_model)
        ens = ProbeEnsemble([p1, p2], weights=[2.0, 8.0])
        np.testing.assert_allclose(ens.weights_, [0.2, 0.8])

    def test_voting_stored(self, tiny_model):
        p = _make_probe(tiny_model)
        ens = ProbeEnsemble([p], voting="hard")
        assert ens.voting == "hard"


# ---------------------------------------------------------------------------
# from_configs
# ---------------------------------------------------------------------------

class TestFromConfigs:
    def test_creates_probes_from_configs(self, tiny_model):
        configs = [
            {"layers": -1, "classifier": "logistic_regression"},
            {"layers": -1, "classifier": "logistic_regression"},
        ]
        ens = ProbeEnsemble.from_configs(
            model=tiny_model,
            configs=configs,
            device="cpu",
            remote=False,
            random_state=42,
        )
        assert len(ens.probes_) == 2
        assert not ens._bootstrap_mode

    def test_weights_forwarded(self, tiny_model):
        configs = [{"layers": -1}, {"layers": -1}]
        ens = ProbeEnsemble.from_configs(
            model=tiny_model,
            configs=configs,
            weights=[3.0, 1.0],
            device="cpu",
            remote=False,
        )
        np.testing.assert_allclose(ens.weights_, [0.75, 0.25])


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_creates_cloned_probes(self, tiny_model):
        base = _make_probe(tiny_model)
        ens = ProbeEnsemble.bootstrap(base, n_resamples=5, random_state=0)
        assert len(ens.probes_) == 5
        assert ens._bootstrap_mode is True
        assert ens._bootstrap_seed == 0
        # Clones should be distinct objects
        assert ens.probes_[0] is not base
        assert ens.probes_[0] is not ens.probes_[1]

    def test_voting_and_weights_forwarded(self, tiny_model):
        base = _make_probe(tiny_model)
        ens = ProbeEnsemble.bootstrap(
            base, n_resamples=3, weights=[1.0, 2.0, 3.0], voting="hard",
        )
        assert ens.voting == "hard"
        np.testing.assert_allclose(ens.weights_.sum(), 1.0)


# ---------------------------------------------------------------------------
# _check_fitted
# ---------------------------------------------------------------------------

class TestCheckFitted:
    def test_raises_when_not_fitted(self, tiny_model):
        p = _make_probe(tiny_model)
        ens = ProbeEnsemble([p])
        with pytest.raises(RuntimeError, match="has not been fitted"):
            ens.predict(["hello"])

    def test_raises_on_predict_proba(self, tiny_model):
        p = _make_probe(tiny_model)
        ens = ProbeEnsemble([p])
        with pytest.raises(RuntimeError, match="has not been fitted"):
            ens.predict_proba(["hello"])

    def test_raises_on_prediction_std(self, tiny_model):
        p = _make_probe(tiny_model)
        ens = ProbeEnsemble([p])
        with pytest.raises(RuntimeError, match="has not been fitted"):
            ens.prediction_std(["hello"])

    def test_raises_on_save(self, tiny_model, tmp_path):
        p = _make_probe(tiny_model)
        ens = ProbeEnsemble([p])
        with pytest.raises(RuntimeError, match="has not been fitted"):
            ens.save(str(tmp_path / "ens.pkl"))


# ---------------------------------------------------------------------------
# fit — normal mode (non-bootstrap)
# ---------------------------------------------------------------------------

class TestFitNormal:
    def test_fit_sets_fitted(self, tiny_model):
        p1 = _make_probe(tiny_model)
        p2 = _make_probe(tiny_model)
        ens = ProbeEnsemble([p1, p2])
        result = ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        assert ens._fitted is True
        # Returns self for chaining
        assert result is ens

    def test_fit_with_sample_weight(self, tiny_model):
        p = _make_probe(tiny_model)
        ens = ProbeEnsemble([p])
        n_total = len(POSITIVE_PROMPTS) + len(NEGATIVE_PROMPTS)
        weights = np.ones(n_total)
        ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS, sample_weight=weights)
        assert ens._fitted

    def test_fit_sample_weight_wrong_length_raises(self, tiny_model):
        p = _make_probe(tiny_model)
        ens = ProbeEnsemble([p])
        with pytest.raises(ValueError, match="sample_weight length"):
            ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS, sample_weight=[1.0])

    def test_fit_groups_ignored_in_non_bootstrap(self, tiny_model):
        p = _make_probe(tiny_model)
        ens = ProbeEnsemble([p])
        n_total = len(POSITIVE_PROMPTS) + len(NEGATIVE_PROMPTS)
        groups = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])[:n_total]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS, groups=groups)
        assert any("groups parameter is ignored" in str(x.message) for x in w)

    def test_fit_groups_wrong_length_raises(self, tiny_model):
        p = _make_probe(tiny_model)
        ens = ProbeEnsemble([p])
        with pytest.raises(ValueError, match="groups length"):
            ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS, groups=[0, 1])


# ---------------------------------------------------------------------------
# fit — bootstrap mode
# ---------------------------------------------------------------------------

class TestFitBootstrap:
    def test_bootstrap_fit(self, tiny_model):
        base = _make_probe(tiny_model)
        ens = ProbeEnsemble.bootstrap(base, n_resamples=3, random_state=42)
        ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        assert ens._fitted is True

    def test_bootstrap_fit_with_groups(self, tiny_model):
        base = _make_probe(tiny_model)
        ens = ProbeEnsemble.bootstrap(base, n_resamples=3, random_state=42)
        n_total = len(POSITIVE_PROMPTS) + len(NEGATIVE_PROMPTS)
        groups = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])[:n_total]
        ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS, groups=groups)
        assert ens._fitted is True

    def test_bootstrap_fit_with_sample_weight(self, tiny_model):
        base = _make_probe(tiny_model)
        ens = ProbeEnsemble.bootstrap(base, n_resamples=3, random_state=42)
        n_total = len(POSITIVE_PROMPTS) + len(NEGATIVE_PROMPTS)
        weights = np.ones(n_total)
        ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS, sample_weight=weights)
        assert ens._fitted is True

    def test_bootstrap_fit_with_groups_and_weights(self, tiny_model):
        base = _make_probe(tiny_model)
        ens = ProbeEnsemble.bootstrap(base, n_resamples=3, random_state=42)
        n_total = len(POSITIVE_PROMPTS) + len(NEGATIVE_PROMPTS)
        groups = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])[:n_total]
        weights = np.random.default_rng(0).random(n_total)
        ens.fit(
            POSITIVE_PROMPTS, NEGATIVE_PROMPTS,
            sample_weight=weights, groups=groups,
        )
        assert ens._fitted is True


# ---------------------------------------------------------------------------
# predict (soft and hard voting)
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_soft_voting(self, tiny_model):
        p1 = _make_probe(tiny_model)
        p2 = _make_probe(tiny_model, random_state=0)
        ens = ProbeEnsemble([p1, p2], voting="soft")
        ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        preds = ens.predict(TEST_PROMPTS)
        assert preds.shape == (len(TEST_PROMPTS),)
        assert set(preds).issubset({0, 1})

    def test_predict_hard_voting(self, tiny_model):
        p1 = _make_probe(tiny_model)
        p2 = _make_probe(tiny_model, random_state=0)
        ens = ProbeEnsemble([p1, p2], voting="hard")
        ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        preds = ens.predict(TEST_PROMPTS)
        assert preds.shape == (len(TEST_PROMPTS),)
        assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------

class TestPredictProba:
    def test_predict_proba_shape(self, tiny_model):
        p1 = _make_probe(tiny_model)
        p2 = _make_probe(tiny_model, random_state=0)
        ens = ProbeEnsemble([p1, p2])
        ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        proba = ens.predict_proba(TEST_PROMPTS)
        assert proba.shape == (len(TEST_PROMPTS), 2)
        # Probabilities should sum to ~1 per sample
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_weighted(self, tiny_model):
        p1 = _make_probe(tiny_model)
        p2 = _make_probe(tiny_model, random_state=0)
        ens = ProbeEnsemble([p1, p2], weights=[1.0, 0.0])
        ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        # With weight=0 for p2, should match p1 alone
        proba_ens = ens.predict_proba(TEST_PROMPTS)
        proba_p1 = p1.predict_proba(TEST_PROMPTS)
        np.testing.assert_allclose(proba_ens, proba_p1, atol=1e-6)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_returns_float(self, tiny_model):
        p = _make_probe(tiny_model)
        ens = ProbeEnsemble([p])
        ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        acc = ens.score(TEST_PROMPTS, [1, 0])
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# prediction_std
# ---------------------------------------------------------------------------

class TestPredictionStd:
    def test_prediction_std_shape(self, tiny_model):
        base = _make_probe(tiny_model)
        ens = ProbeEnsemble.bootstrap(base, n_resamples=3, random_state=42)
        ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        std = ens.prediction_std(TEST_PROMPTS)
        assert std.shape == (len(TEST_PROMPTS),)
        assert np.all(std >= 0.0)

    def test_prediction_std_identical_probes_is_zero(self, tiny_model):
        # Two identical probes with same random_state -> same predictions -> std=0
        p1 = _make_probe(tiny_model)
        p2 = _make_probe(tiny_model)
        ens = ProbeEnsemble([p1, p2])
        ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        std = ens.prediction_std(TEST_PROMPTS)
        np.testing.assert_allclose(std, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, tiny_model, tmp_path):
        p1 = _make_probe(tiny_model)
        p2 = _make_probe(tiny_model, random_state=0)
        ens = ProbeEnsemble([p1, p2], weights=[3.0, 1.0], voting="soft")
        ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        path = str(tmp_path / "ensemble.pkl")
        ens.save(path)

        loaded = ProbeEnsemble.load(path)
        assert loaded._fitted is True
        assert loaded.voting == "soft"
        np.testing.assert_allclose(loaded.weights_, ens.weights_)
        assert len(loaded.probes_) == 2

        # Predictions should match
        preds_orig = ens.predict(TEST_PROMPTS)
        preds_loaded = loaded.predict(TEST_PROMPTS)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_roundtrip_bootstrap(self, tiny_model, tmp_path):
        base = _make_probe(tiny_model)
        ens = ProbeEnsemble.bootstrap(base, n_resamples=3, random_state=42)
        ens.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        path = str(tmp_path / "bootstrap_ens.pkl")
        ens.save(path)

        loaded = ProbeEnsemble.load(path)
        assert loaded._bootstrap_mode is True
        assert loaded._bootstrap_seed == 42


# ---------------------------------------------------------------------------
# _group_balanced_resample (static method)
# ---------------------------------------------------------------------------

class TestGroupBalancedResample:
    def test_basic_resampling(self):
        groups = np.array([0, 0, 0, 1, 1, 1])
        labels = np.array([1, 1, 0, 0, 1, 0])
        rng = np.random.default_rng(42)
        indices = ProbeEnsemble._group_balanced_resample(
            groups, labels, sample_weight=None, rng=rng,
        )
        # Each group gets ceil(6/2)=3 samples, so total=6
        assert len(indices) == 6
        # All indices are valid
        assert np.all(indices >= 0)
        assert np.all(indices < len(groups))

    def test_resampling_with_weights(self):
        groups = np.array([0, 0, 1, 1])
        labels = np.array([1, 0, 1, 0])
        rng = np.random.default_rng(42)
        weights = np.array([10.0, 0.001, 10.0, 0.001])
        indices = ProbeEnsemble._group_balanced_resample(
            groups, labels, sample_weight=weights, rng=rng,
        )
        # ceil(4/2)=2 per group, total=4
        assert len(indices) == 4

    def test_three_groups(self):
        groups = np.array([0, 0, 1, 1, 2, 2, 2])
        labels = np.array([1, 0, 1, 0, 1, 0, 1])
        rng = np.random.default_rng(0)
        indices = ProbeEnsemble._group_balanced_resample(
            groups, labels, sample_weight=None, rng=rng,
        )
        # ceil(7/3)=3 per group, total=9
        assert len(indices) == 9
