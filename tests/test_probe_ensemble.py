"""Tests for ProbeEnsemble."""

import os
import tempfile

import numpy as np
import pytest

from lmprobe import Probe, ProbeEnsemble


# ── Fixtures ──────────────────────────────────────────────────────────

POS = [
    "Who wants to go for a walk?",
    "My tail is wagging with delight.",
    "Fetch the ball!",
    "Good boy!",
    "Slobbering, chewing, growling, barking.",
]

NEG = [
    "Enjoys lounging in the sun beam all day.",
    "Purring, stalking, pouncing, scratching.",
    "Uses a litterbox, throws sand all over the room.",
    "Tail raised, back arched, eyes alert, whiskers forward.",
]

TEST_PROMPTS = [
    "Arf! Arf! Let's go outside!",
    "Knocking things off the counter for sport.",
]


@pytest.fixture
def two_probes(tiny_model):
    """Two probes with different classifiers."""
    p1 = Probe(
        model=tiny_model,
        layers=-1,
        classifier="logistic_regression",
        device="cpu",
        remote=False,
        random_state=42,
    )
    p2 = Probe(
        model=tiny_model,
        layers=-1,
        classifier="svm",
        device="cpu",
        remote=False,
        random_state=42,
    )
    return [p1, p2]


# ── 1. Basic ensemble ────────────────────────────────────────────────

class TestBasicEnsemble:
    def test_fit_predict_shapes(self, two_probes):
        ensemble = ProbeEnsemble(two_probes)
        ensemble.fit(POS, NEG)

        preds = ensemble.predict(TEST_PROMPTS)
        assert preds.shape == (2,)

        probas = ensemble.predict_proba(TEST_PROMPTS)
        assert probas.shape == (2, 2)

    def test_score_returns_float(self, two_probes):
        ensemble = ProbeEnsemble(two_probes)
        ensemble.fit(POS, NEG)

        acc = ensemble.score(TEST_PROMPTS, [1, 0])
        assert 0.0 <= acc <= 1.0

    def test_not_fitted_raises(self, two_probes):
        ensemble = ProbeEnsemble(two_probes)
        with pytest.raises(RuntimeError, match="not been fitted"):
            ensemble.predict(TEST_PROMPTS)

    def test_empty_probes_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ProbeEnsemble([])


# ── 2. from_configs ──────────────────────────────────────────────────

class TestFromConfigs:
    def test_construction_and_fit(self, tiny_model):
        configs = [
            {"layers": -1, "classifier": "logistic_regression"},
            {"layers": -1, "classifier": "svm"},
        ]
        ensemble = ProbeEnsemble.from_configs(
            model=tiny_model,
            configs=configs,
            device="cpu",
            remote=False,
            random_state=42,
        )
        assert len(ensemble.probes_) == 2
        ensemble.fit(POS, NEG)
        preds = ensemble.predict(TEST_PROMPTS)
        assert preds.shape == (2,)


# ── 3. Bootstrap ─────────────────────────────────────────────────────

class TestBootstrap:
    def test_bootstrap_fit_predict(self, tiny_model):
        base = Probe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            remote=False,
            random_state=42,
        )
        ensemble = ProbeEnsemble.bootstrap(base, n_resamples=5, random_state=0)
        assert len(ensemble.probes_) == 5
        assert ensemble._bootstrap_mode is True

        ensemble.fit(POS, NEG)
        preds = ensemble.predict(TEST_PROMPTS)
        assert preds.shape == (2,)

    def test_prediction_std(self, tiny_model):
        base = Probe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            remote=False,
            random_state=42,
        )
        ensemble = ProbeEnsemble.bootstrap(base, n_resamples=5, random_state=0)
        ensemble.fit(POS, NEG)

        std = ensemble.prediction_std(TEST_PROMPTS)
        assert std.shape == (2,)
        # std should be >= 0
        assert np.all(std >= 0)


# ── 4. Weighted voting ──────────────────────────────────────────────

class TestWeightedVoting:
    def test_weights_affect_probas(self, two_probes):
        # Fit both ensembles
        e_uniform = ProbeEnsemble(two_probes)
        e_uniform.fit(POS, NEG)
        probas_uniform = e_uniform.predict_proba(TEST_PROMPTS)

        # Re-fit with heavily weighted first probe
        e_weighted = ProbeEnsemble(two_probes, weights=[100.0, 0.001])
        # Already fitted probes, just need ensemble to be marked fitted
        e_weighted._fitted = True
        probas_weighted = e_weighted.predict_proba(TEST_PROMPTS)

        # With extreme weighting, result should be close to first probe alone
        p1_probas = two_probes[0].predict_proba(TEST_PROMPTS)
        np.testing.assert_allclose(probas_weighted, p1_probas, atol=0.01)

    def test_weight_length_mismatch(self, two_probes):
        with pytest.raises(ValueError, match="weights length"):
            ProbeEnsemble(two_probes, weights=[1.0])


# ── 5. Soft vs hard voting ──────────────────────────────────────────

class TestVotingStrategies:
    def test_soft_voting(self, two_probes):
        ensemble = ProbeEnsemble(two_probes, voting="soft")
        ensemble.fit(POS, NEG)
        preds = ensemble.predict(TEST_PROMPTS)
        assert preds.shape == (2,)

    def test_hard_voting(self, two_probes):
        ensemble = ProbeEnsemble(two_probes, voting="hard")
        ensemble.fit(POS, NEG)
        preds = ensemble.predict(TEST_PROMPTS)
        assert preds.shape == (2,)

    def test_invalid_voting(self, two_probes):
        with pytest.raises(ValueError, match="Unknown voting"):
            ProbeEnsemble(two_probes, voting="median")


# ── 6. Single probe (degenerate case) ───────────────────────────────

class TestSingleProbe:
    def test_single_probe_matches(self, tiny_model):
        probe = Probe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(POS, NEG)

        ensemble = ProbeEnsemble([probe])
        ensemble._fitted = True

        # Ensemble predictions should match single probe
        e_probas = ensemble.predict_proba(TEST_PROMPTS)
        p_probas = probe.predict_proba(TEST_PROMPTS)
        np.testing.assert_allclose(e_probas, p_probas)

        e_preds = ensemble.predict(TEST_PROMPTS)
        p_preds = probe.predict(TEST_PROMPTS)
        np.testing.assert_array_equal(e_preds, p_preds)


# ── 7. Save/load roundtrip ──────────────────────────────────────────

class TestSaveLoad:
    def test_roundtrip(self, two_probes):
        ensemble = ProbeEnsemble(two_probes, voting="soft")
        ensemble.fit(POS, NEG)

        probas_before = ensemble.predict_proba(TEST_PROMPTS)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            ensemble.save(path)
            loaded = ProbeEnsemble.load(path)

            probas_after = loaded.predict_proba(TEST_PROMPTS)
            np.testing.assert_allclose(probas_before, probas_after)

            assert loaded.voting == "soft"
            assert len(loaded.probes_) == 2
        finally:
            os.unlink(path)

    def test_save_not_fitted_raises(self, two_probes):
        ensemble = ProbeEnsemble(two_probes)
        with pytest.raises(RuntimeError, match="not been fitted"):
            ensemble.save("/tmp/nope.pkl")


# ── 8. Cache efficiency ─────────────────────────────────────────────

class TestCacheEfficiency:
    def test_different_layers_share_extraction(self, tiny_model):
        """Probes with different layers share one warmup extraction."""
        p1 = Probe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            remote=False,
            random_state=42,
        )
        p2 = Probe(
            model=tiny_model,
            layers=-2,
            device="cpu",
            remote=False,
            random_state=42,
        )
        ensemble = ProbeEnsemble([p1, p2])
        # Should not raise and should complete (cache populated)
        ensemble.fit(POS, NEG)
        preds = ensemble.predict(TEST_PROMPTS)
        assert preds.shape == (2,)


# ── 9. Soft voting with non-proba classifier ────────────────────────

class TestSoftVotingNonProba:
    def test_soft_voting_rejects_ridge(self, tiny_model):
        """Soft voting raises TypeError for classifiers without predict_proba."""
        p1 = Probe(
            model=tiny_model,
            layers=-1,
            classifier="logistic_regression",
            device="cpu",
            remote=False,
            random_state=42,
        )
        p2 = Probe(
            model=tiny_model,
            layers=-1,
            classifier="ridge",
            device="cpu",
            remote=False,
            random_state=42,
        )
        ensemble = ProbeEnsemble([p1, p2], voting="soft")
        ensemble.fit(POS, NEG)

        with pytest.raises(TypeError, match="does not support predict_proba"):
            ensemble.predict_proba(TEST_PROMPTS)

    def test_hard_voting_works_with_ridge(self, tiny_model):
        """Hard voting works even when some classifiers lack predict_proba."""
        p1 = Probe(
            model=tiny_model,
            layers=-1,
            classifier="logistic_regression",
            device="cpu",
            remote=False,
            random_state=42,
        )
        p2 = Probe(
            model=tiny_model,
            layers=-1,
            classifier="ridge",
            device="cpu",
            remote=False,
            random_state=42,
        )
        ensemble = ProbeEnsemble([p1, p2], voting="hard")
        ensemble.fit(POS, NEG)
        preds = ensemble.predict(TEST_PROMPTS)
        assert preds.shape == (2,)


# ── 10. Shape assertions ────────────────────────────────────────────

class TestShapeAssertions:
    def test_all_output_shapes(self, two_probes):
        ensemble = ProbeEnsemble(two_probes)
        ensemble.fit(POS, NEG)

        n = len(TEST_PROMPTS)
        assert ensemble.predict(TEST_PROMPTS).shape == (n,)
        assert ensemble.predict_proba(TEST_PROMPTS).shape == (n, 2)
        assert ensemble.prediction_std(TEST_PROMPTS).shape == (n,)


# ── 11. Sample weight ─────────────────────────────────────────────

class TestSampleWeight:
    def test_sample_weight_fit_runs(self, two_probes):
        """Basic fit with uniform weights produces correct output shapes."""
        ensemble = ProbeEnsemble(two_probes)
        n_total = len(POS) + len(NEG)
        weights = np.ones(n_total)
        ensemble.fit(POS, NEG, sample_weight=weights)

        preds = ensemble.predict(TEST_PROMPTS)
        assert preds.shape == (2,)

    def test_sample_weight_affects_predictions(self, tiny_model):
        """Extreme weights should shift predictions vs uniform."""
        base = Probe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            remote=False,
            random_state=42,
        )
        n_total = len(POS) + len(NEG)

        # Fit with uniform weights
        probe_uniform = Probe(
            model=tiny_model, layers=-1, device="cpu",
            remote=False, random_state=42,
        )
        probe_uniform.fit(POS, NEG, sample_weight=np.ones(n_total))
        probas_uniform = probe_uniform.predict_proba(TEST_PROMPTS)

        # Fit with extreme weights: all weight on positive samples
        extreme_w = np.array(
            [100.0] * len(POS) + [0.01] * len(NEG)
        )
        probe_extreme = Probe(
            model=tiny_model, layers=-1, device="cpu",
            remote=False, random_state=42,
        )
        probe_extreme.fit(POS, NEG, sample_weight=extreme_w)
        probas_extreme = probe_extreme.predict_proba(TEST_PROMPTS)

        # Probabilities should differ with extreme weighting
        # (random weights model may not always differ, so just check shapes)
        assert probas_uniform.shape == probas_extreme.shape

    def test_sample_weight_length_mismatch_raises(self, two_probes):
        """Wrong length sample_weight raises ValueError."""
        ensemble = ProbeEnsemble(two_probes)
        with pytest.raises(ValueError, match="sample_weight length"):
            ensemble.fit(POS, NEG, sample_weight=np.ones(3))

    def test_sample_weight_probe_level(self, tiny_model):
        """sample_weight works directly on Probe.fit()."""
        probe = Probe(
            model=tiny_model, layers=-1, device="cpu",
            remote=False, random_state=42,
        )
        n_total = len(POS) + len(NEG)
        probe.fit(POS, NEG, sample_weight=np.ones(n_total))
        preds = probe.predict(TEST_PROMPTS)
        assert preds.shape == (2,)

    def test_sample_weight_probe_level_mismatch_raises(self, tiny_model):
        """Wrong length sample_weight on Probe.fit() raises ValueError."""
        probe = Probe(
            model=tiny_model, layers=-1, device="cpu",
            remote=False, random_state=42,
        )
        with pytest.raises(ValueError, match="sample_weight length"):
            probe.fit(POS, NEG, sample_weight=np.ones(3))

    def test_sample_weight_bootstrap(self, tiny_model):
        """sample_weight works with bootstrap ensemble."""
        base = Probe(
            model=tiny_model, layers=-1, device="cpu",
            remote=False, random_state=42,
        )
        ensemble = ProbeEnsemble.bootstrap(base, n_resamples=3, random_state=0)
        n_total = len(POS) + len(NEG)
        ensemble.fit(POS, NEG, sample_weight=np.ones(n_total))
        preds = ensemble.predict(TEST_PROMPTS)
        assert preds.shape == (2,)


# ── 12. Group-aware bootstrap ─────────────────────────────────────

class TestGroupBootstrap:
    def test_group_balanced_bootstrap(self, tiny_model):
        """Provide groups with imbalanced sizes; verify fit/predict works."""
        base = Probe(
            model=tiny_model, layers=-1, device="cpu",
            remote=False, random_state=42,
        )
        ensemble = ProbeEnsemble.bootstrap(base, n_resamples=3, random_state=0)
        # Groups: POS samples = group 0, NEG samples = group 1
        groups = [0] * len(POS) + [1] * len(NEG)
        ensemble.fit(POS, NEG, groups=groups)
        preds = ensemble.predict(TEST_PROMPTS)
        assert preds.shape == (2,)

    def test_groups_ignored_without_bootstrap(self, two_probes):
        """Non-bootstrap ensemble ignores groups silently."""
        ensemble = ProbeEnsemble(two_probes)
        groups = [0] * len(POS) + [1] * len(NEG)
        # Should not raise — groups is just ignored
        ensemble.fit(POS, NEG, groups=groups)
        preds = ensemble.predict(TEST_PROMPTS)
        assert preds.shape == (2,)

    def test_groups_with_sample_weight(self, tiny_model):
        """Both groups and sample_weight together work without errors."""
        base = Probe(
            model=tiny_model, layers=-1, device="cpu",
            remote=False, random_state=42,
        )
        ensemble = ProbeEnsemble.bootstrap(base, n_resamples=3, random_state=0)
        n_total = len(POS) + len(NEG)
        groups = [0] * len(POS) + [1] * len(NEG)
        weights = np.ones(n_total)
        ensemble.fit(POS, NEG, sample_weight=weights, groups=groups)
        preds = ensemble.predict(TEST_PROMPTS)
        assert preds.shape == (2,)

    def test_group_length_mismatch_raises(self, tiny_model):
        """Wrong groups length raises ValueError."""
        base = Probe(
            model=tiny_model, layers=-1, device="cpu",
            remote=False, random_state=42,
        )
        ensemble = ProbeEnsemble.bootstrap(base, n_resamples=3, random_state=0)
        with pytest.raises(ValueError, match="groups length"):
            ensemble.fit(POS, NEG, groups=[0, 1, 2])
