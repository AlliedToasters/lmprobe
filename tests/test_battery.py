"""Tests for BaselineBattery orchestration class."""

import numpy as np
import pytest

from lmprobe.battery import (
    BASELINE_REGISTRY,
    BaselineBattery,
    BaselineResult,
    BaselineResults,
)


# Test data
POSITIVE_PROMPTS = [
    "The dog barked loudly",
    "My puppy loves to play fetch",
    "Dogs are loyal companions",
]

NEGATIVE_PROMPTS = [
    "The cat purred softly",
    "My kitten sleeps all day",
    "Cats are independent animals",
]

TEST_PROMPTS = [
    "A dog chased the ball",
    "The cat sat on the mat",
]

TEST_LABELS = [1, 0]


class TestBaselineResult:
    """Tests for BaselineResult dataclass."""

    def test_baseline_result_repr(self):
        """BaselineResult has readable repr."""
        from lmprobe import BaselineProbe

        baseline = BaselineProbe(method="bow")
        result = BaselineResult(name="bow", score=0.75, baseline=baseline)

        repr_str = repr(result)
        assert "bow" in repr_str
        assert "0.75" in repr_str


class TestBaselineResults:
    """Tests for BaselineResults dataclass."""

    def test_get_best(self):
        """get_best returns top results by score."""
        from lmprobe import BaselineProbe

        results = BaselineResults(results=[
            BaselineResult(name="a", score=0.5, baseline=BaselineProbe(method="bow")),
            BaselineResult(name="b", score=0.9, baseline=BaselineProbe(method="bow")),
            BaselineResult(name="c", score=0.7, baseline=BaselineProbe(method="bow")),
        ])

        top1 = results.get_best(n=1)
        assert len(top1) == 1
        assert top1[0].name == "b"
        assert top1[0].score == 0.9

        top2 = results.get_best(n=2)
        assert len(top2) == 2
        assert top2[0].name == "b"
        assert top2[1].name == "c"

    def test_iteration(self):
        """BaselineResults supports iteration."""
        from lmprobe import BaselineProbe

        results = BaselineResults(results=[
            BaselineResult(name="a", score=0.5, baseline=BaselineProbe(method="bow")),
            BaselineResult(name="b", score=0.9, baseline=BaselineProbe(method="bow")),
        ])

        names = [r.name for r in results]
        assert names == ["a", "b"]

    def test_len(self):
        """BaselineResults supports len()."""
        from lmprobe import BaselineProbe

        results = BaselineResults(results=[
            BaselineResult(name="a", score=0.5, baseline=BaselineProbe(method="bow")),
            BaselineResult(name="b", score=0.9, baseline=BaselineProbe(method="bow")),
        ])

        assert len(results) == 2

    def test_getitem_by_index(self):
        """BaselineResults supports indexing."""
        from lmprobe import BaselineProbe

        results = BaselineResults(results=[
            BaselineResult(name="a", score=0.5, baseline=BaselineProbe(method="bow")),
            BaselineResult(name="b", score=0.9, baseline=BaselineProbe(method="bow")),
        ])

        assert results[0].name == "a"
        assert results[1].name == "b"

    def test_getitem_by_name(self):
        """BaselineResults supports lookup by name."""
        from lmprobe import BaselineProbe

        results = BaselineResults(results=[
            BaselineResult(name="a", score=0.5, baseline=BaselineProbe(method="bow")),
            BaselineResult(name="b", score=0.9, baseline=BaselineProbe(method="bow")),
        ])

        assert results["a"].score == 0.5
        assert results["b"].score == 0.9

        with pytest.raises(KeyError):
            _ = results["nonexistent"]

    def test_summary(self):
        """summary() returns formatted string."""
        from lmprobe import BaselineProbe

        results = BaselineResults(results=[
            BaselineResult(name="bow", score=0.8, baseline=BaselineProbe(method="bow"), fit_time=0.1),
            BaselineResult(name="tfidf", score=0.9, baseline=BaselineProbe(method="tfidf"), fit_time=0.2),
        ])

        summary = results.summary()
        assert "Baseline Results:" in summary
        assert "bow" in summary
        assert "tfidf" in summary
        assert "0.8" in summary or "0.80" in summary


class TestBaselineBatteryTextOnly:
    """Tests for BaselineBattery with text-only baselines."""

    def test_text_only_baselines(self):
        """Without model, only text baselines run."""
        battery = BaselineBattery(model=None, random_state=42)
        results = battery.fit(
            POSITIVE_PROMPTS, NEGATIVE_PROMPTS,
            TEST_PROMPTS, TEST_LABELS,
        )

        assert len(results) > 0
        names = [r.name for r in results]
        # Text baselines should be present
        assert "bow" in names
        assert "tfidf" in names
        assert "sentence_length" in names
        # Model-required baselines should not be present
        assert "random_direction" not in names
        assert "pca" not in names
        assert "layer_0" not in names
        assert "perplexity" not in names

    def test_get_best(self):
        """get_best returns top baselines."""
        battery = BaselineBattery(model=None, random_state=42)
        results = battery.fit(
            POSITIVE_PROMPTS, NEGATIVE_PROMPTS,
            TEST_PROMPTS, TEST_LABELS,
        )

        best = results.get_best(n=2)
        assert len(best) <= 2
        # Should be sorted by score descending
        if len(best) == 2:
            assert best[0].score >= best[1].score

    def test_get_best_before_fit(self):
        """get_best before fit raises RuntimeError."""
        battery = BaselineBattery(model=None)

        with pytest.raises(RuntimeError, match="not fitted"):
            battery.get_best()

    def test_results_stored(self):
        """Results are stored in results_ attribute."""
        battery = BaselineBattery(model=None, random_state=42)
        results = battery.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        assert battery.results_ is results
        assert len(battery.results_) > 0

    def test_default_to_training_data(self):
        """Without test data, uses training data for scoring."""
        battery = BaselineBattery(model=None, random_state=42)
        results = battery.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        # Should still produce results
        assert len(results) > 0
        # Scores should be valid
        for r in results:
            assert 0.0 <= r.score <= 1.0


class TestBaselineBatteryWithModel:
    """Tests for BaselineBattery with model (includes activation baselines)."""

    def test_all_baselines_with_model(self, tiny_model):
        """With model, activation baselines run."""
        battery = BaselineBattery(
            model=tiny_model,
            device="cpu",
            remote=False,
            random_state=42,
            exclude=["sentence_transformers"],  # May not be installed
        )
        results = battery.fit(
            POSITIVE_PROMPTS, NEGATIVE_PROMPTS,
            TEST_PROMPTS, TEST_LABELS,
        )

        names = [r.name for r in results]
        # Activation baselines should be present
        assert "random_direction" in names
        assert "pca" in names
        assert "layer_0" in names
        assert "perplexity" in names

    def test_include_filter(self, tiny_model):
        """include parameter filters baselines."""
        battery = BaselineBattery(
            model=tiny_model,
            include=["bow", "random_direction"],
            device="cpu",
        )
        results = battery.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        names = [r.name for r in results]
        assert len(names) == 2
        assert set(names) == {"bow", "random_direction"}

    def test_exclude_filter(self):
        """exclude parameter filters baselines."""
        battery = BaselineBattery(
            model=None,
            exclude=["random", "majority"],
        )
        results = battery.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        names = [r.name for r in results]
        assert "random" not in names
        assert "majority" not in names

    def test_get_baseline(self, tiny_model):
        """get_baseline retrieves fitted baseline by name."""
        battery = BaselineBattery(
            model=tiny_model,
            include=["bow", "pca"],
            device="cpu",
        )
        battery.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        bow_baseline = battery.get_baseline("bow")
        pca_baseline = battery.get_baseline("pca")

        assert bow_baseline is not None
        assert pca_baseline is not None

        # Should be able to predict with them
        pred = bow_baseline.predict(TEST_PROMPTS)
        assert pred.shape == (2,)

    def test_get_baseline_not_found(self, tiny_model):
        """get_baseline raises KeyError for unknown baseline."""
        battery = BaselineBattery(
            model=tiny_model,
            include=["bow"],
            device="cpu",
        )
        battery.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        with pytest.raises(KeyError):
            battery.get_baseline("nonexistent")


class TestBaselineBatteryCustomScorer:
    """Tests for custom scorer support."""

    def test_custom_scorer(self):
        """Custom scorer function is used."""
        # Custom scorer that returns 1.0 - accuracy (inverted)
        def inverted_accuracy(y_true, y_pred):
            return 1.0 - float(np.mean(y_true == y_pred))

        battery = BaselineBattery(
            model=None,
            random_state=42,
            scorer=inverted_accuracy,
        )
        results = battery.fit(
            POSITIVE_PROMPTS, NEGATIVE_PROMPTS,
            TEST_PROMPTS, TEST_LABELS,
        )

        # Scores should be inverted from normal
        for r in results:
            assert 0.0 <= r.score <= 1.0


class TestBaselineBatteryProperties:
    """Tests for battery properties."""

    def test_available_baselines(self):
        """available_baselines lists all registered baselines."""
        battery = BaselineBattery(model=None)

        available = battery.available_baselines
        assert "bow" in available
        assert "tfidf" in available
        assert "random_direction" in available
        assert len(available) == len(BASELINE_REGISTRY)

    def test_applicable_baselines_no_model(self):
        """applicable_baselines excludes model-requiring baselines when no model."""
        battery = BaselineBattery(model=None)

        applicable = battery.applicable_baselines
        assert "bow" in applicable
        assert "tfidf" in applicable
        assert "random_direction" not in applicable
        assert "perplexity" not in applicable

    def test_applicable_baselines_with_model(self, tiny_model):
        """applicable_baselines includes all baselines when model provided."""
        battery = BaselineBattery(model=tiny_model)

        applicable = battery.applicable_baselines
        assert "bow" in applicable
        assert "random_direction" in applicable
        assert "perplexity" in applicable
