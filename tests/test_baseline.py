"""Tests for BaselineProbe text classification baselines."""

import numpy as np
import pytest

from lmprobe.baseline import BaselineProbe


# Test data
POSITIVE_PROMPTS = [
    "The dog barked loudly",
    "My puppy loves to play fetch",
    "Dogs are loyal companions",
    "The golden retriever wagged its tail",
    "Walking the dog in the park",
]

NEGATIVE_PROMPTS = [
    "The cat purred softly",
    "My kitten sleeps all day",
    "Cats are independent animals",
    "The tabby cat stretched lazily",
    "The cat knocked things off the table",
]

TEST_PROMPTS = [
    "A dog chased the ball",  # Positive-ish
    "The cat sat on the mat",  # Negative-ish
]


class TestBaselineProbeBasic:
    """Basic functionality tests for BaselineProbe."""

    def test_bow_fit_predict(self):
        """Bag-of-words baseline can fit and predict."""
        baseline = BaselineProbe(method="bow", random_state=42)
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        predictions = baseline.predict(TEST_PROMPTS)
        assert predictions.shape == (2,)
        assert set(predictions).issubset({0, 1})

    def test_tfidf_fit_predict(self):
        """TF-IDF baseline can fit and predict."""
        baseline = BaselineProbe(method="tfidf", random_state=42)
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        predictions = baseline.predict(TEST_PROMPTS)
        assert predictions.shape == (2,)

    def test_random_baseline(self):
        """Random baseline makes random predictions."""
        baseline = BaselineProbe(method="random", random_state=42)
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        # Should be able to predict
        predictions = baseline.predict(TEST_PROMPTS)
        assert predictions.shape == (2,)

    def test_majority_baseline(self):
        """Majority baseline always predicts majority class."""
        # More positive than negative
        baseline = BaselineProbe(method="majority")
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS[:3])

        predictions = baseline.predict(TEST_PROMPTS)
        # Should always predict 1 (positive is majority)
        assert all(p == 1 for p in predictions)

    def test_predict_proba(self):
        """predict_proba returns valid probabilities."""
        baseline = BaselineProbe(method="tfidf", random_state=42)
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        proba = baseline.predict_proba(TEST_PROMPTS)
        assert proba.shape == (2, 2)
        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        # All values in [0, 1]
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_score(self):
        """score() computes accuracy correctly."""
        baseline = BaselineProbe(method="tfidf", random_state=42)
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        # Score on training data (should be high)
        train_prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
        train_labels = [1] * len(POSITIVE_PROMPTS) + [0] * len(NEGATIVE_PROMPTS)
        accuracy = baseline.score(train_prompts, train_labels)

        assert 0.0 <= accuracy <= 1.0
        # Should do well on training data
        assert accuracy >= 0.5


class TestBaselineProbeClassifiers:
    """Test different classifier options."""

    @pytest.mark.parametrize("classifier", [
        "logistic_regression",
        "ridge",
        "lda",
        "svm",
    ])
    def test_different_classifiers(self, classifier):
        """Different classifiers work with bow/tfidf."""
        baseline = BaselineProbe(
            method="tfidf",
            classifier=classifier,
            random_state=42,
        )
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        predictions = baseline.predict(TEST_PROMPTS)
        assert predictions.shape == (2,)

    def test_custom_sklearn_classifier(self):
        """Custom sklearn classifier works."""
        from sklearn.naive_bayes import MultinomialNB

        baseline = BaselineProbe(
            method="bow",
            classifier=MultinomialNB(),
        )
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        predictions = baseline.predict(TEST_PROMPTS)
        assert predictions.shape == (2,)


class TestBaselineProbeFeatures:
    """Test feature extraction options."""

    def test_ngram_range(self):
        """N-gram range affects vocabulary size."""
        baseline_unigram = BaselineProbe(method="bow", ngram_range=(1, 1))
        baseline_bigram = BaselineProbe(method="bow", ngram_range=(1, 2))

        baseline_unigram.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        baseline_bigram.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        # Bigrams should have more features
        assert len(baseline_bigram.get_feature_names()) > len(baseline_unigram.get_feature_names())

    def test_max_features(self):
        """max_features limits vocabulary size."""
        baseline = BaselineProbe(method="bow", max_features=10)
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        features = baseline.get_feature_names()
        assert len(features) <= 10

    def test_get_feature_names(self):
        """get_feature_names returns vocabulary."""
        baseline = BaselineProbe(method="tfidf")
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        features = baseline.get_feature_names()
        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, str) for f in features)
        # Should contain expected words
        assert "dog" in features or "dogs" in features
        assert "cat" in features or "cats" in features

    def test_get_feature_names_random(self):
        """get_feature_names returns None for random/majority."""
        baseline = BaselineProbe(method="random")
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        assert baseline.get_feature_names() is None

    def test_get_top_features(self):
        """get_top_features returns interpretable weights."""
        baseline = BaselineProbe(method="tfidf", classifier="logistic_regression")
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        top = baseline.get_top_features(n=5)
        assert top is not None
        assert "positive" in top
        assert "negative" in top
        assert len(top["positive"]) <= 5
        assert len(top["negative"]) <= 5

        # Each entry is (feature_name, weight)
        for name, weight in top["positive"]:
            assert isinstance(name, str)
            assert isinstance(weight, (int, float))


class TestBaselineProbeErrors:
    """Test error handling."""

    def test_invalid_method(self):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            BaselineProbe(method="invalid")

    def test_predict_before_fit(self):
        """Predicting before fit raises RuntimeError."""
        baseline = BaselineProbe(method="tfidf")

        with pytest.raises(RuntimeError, match="not been fitted"):
            baseline.predict(TEST_PROMPTS)

    def test_predict_proba_unsupported(self):
        """predict_proba with unsupported classifier raises."""
        baseline = BaselineProbe(method="tfidf", classifier="ridge")
        baseline.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        with pytest.raises(AttributeError, match="does not support predict_proba"):
            baseline.predict_proba(TEST_PROMPTS)


class TestBaselineProbeReproducibility:
    """Test reproducibility with random_state."""

    def test_random_state_reproducible(self):
        """Same random_state gives same results."""
        baseline1 = BaselineProbe(method="tfidf", random_state=42)
        baseline2 = BaselineProbe(method="tfidf", random_state=42)

        baseline1.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        baseline2.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        pred1 = baseline1.predict(TEST_PROMPTS)
        pred2 = baseline2.predict(TEST_PROMPTS)

        np.testing.assert_array_equal(pred1, pred2)

    def test_random_baseline_reproducible(self):
        """Random baseline is reproducible with seed."""
        baseline1 = BaselineProbe(method="random", random_state=42)
        baseline2 = BaselineProbe(method="random", random_state=42)

        baseline1.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        baseline2.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        # Generate many predictions
        prompts = TEST_PROMPTS * 10
        pred1 = baseline1.predict(prompts)
        pred2 = baseline2.predict(prompts)

        np.testing.assert_array_equal(pred1, pred2)
