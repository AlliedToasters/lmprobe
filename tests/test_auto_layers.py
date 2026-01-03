"""Tests for automatic layer selection via Group Lasso."""

import numpy as np
import pytest

from lmprobe.extraction import resolve_auto_candidates


class TestResolveAutoCandidates:
    """Tests for resolve_auto_candidates function."""

    def test_default_candidates(self):
        """Default [0.25, 0.5, 0.75] produces correct layers."""
        # 32-layer model
        result = resolve_auto_candidates(None, 32)
        # 0.25 * 31 = 7.75 -> 7
        # 0.5 * 31 = 15.5 -> 15
        # 0.75 * 31 = 23.25 -> 23
        assert result == [7, 15, 23]

    def test_fractional_candidates(self):
        """Fractional positions are converted correctly."""
        result = resolve_auto_candidates([0.0, 0.5, 1.0], 32)
        assert result == [0, 15, 31]

    def test_explicit_indices(self):
        """Explicit integer indices are passed through."""
        result = resolve_auto_candidates([10, 16, 22], 32)
        assert result == [10, 16, 22]

    def test_negative_indices(self):
        """Negative indices are resolved correctly."""
        result = resolve_auto_candidates([-8, -4, -1], 32)
        assert result == [24, 28, 31]

    def test_removes_duplicates(self):
        """Duplicate indices are removed."""
        result = resolve_auto_candidates([10, 10, 16], 32)
        assert result == [10, 16]

    def test_sorts_output(self):
        """Output is sorted."""
        result = resolve_auto_candidates([22, 10, 16], 32)
        assert result == [10, 16, 22]

    def test_empty_candidates_raises(self):
        """Empty candidates list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            resolve_auto_candidates([], 32)

    def test_out_of_range_raises(self):
        """Out-of-range index raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            resolve_auto_candidates([50], 32)

    def test_small_model(self):
        """Works correctly with small models (like tiny test model)."""
        # tiny-random-llama-2 has only a few layers
        result = resolve_auto_candidates(None, 4)
        # 0.25 * 3 = 0.75 -> 0
        # 0.5 * 3 = 1.5 -> 1
        # 0.75 * 3 = 2.25 -> 2
        assert result == [0, 1, 2]


# Skip Group Lasso tests if skglm not installed
skglm = pytest.importorskip("skglm")


class TestGroupLassoClassifier:
    """Tests for GroupLassoClassifier."""

    def test_fit_produces_coefficients(self):
        """Fitting produces coefficient array."""
        from lmprobe.classifiers import GroupLassoClassifier

        clf = GroupLassoClassifier(hidden_dim=10, n_layers=3, alpha=0.01)
        X = np.random.randn(50, 30)  # 30 = 10 * 3
        y = np.array([0] * 25 + [1] * 25)

        clf.fit(X, y)

        assert clf.coef_ is not None
        assert clf.coef_.shape == (30,)

    def test_selected_groups_populated(self):
        """selected_groups_ is populated after fit."""
        from lmprobe.classifiers import GroupLassoClassifier

        clf = GroupLassoClassifier(hidden_dim=10, n_layers=3, alpha=0.001)
        X = np.random.randn(50, 30)
        y = np.array([0] * 25 + [1] * 25)

        clf.fit(X, y)

        assert clf.selected_groups_ is not None
        assert isinstance(clf.selected_groups_, list)

    def test_group_norms_correct_shape(self):
        """group_norms_ has correct shape."""
        from lmprobe.classifiers import GroupLassoClassifier

        clf = GroupLassoClassifier(hidden_dim=10, n_layers=3)
        X = np.random.randn(50, 30)
        y = np.array([0] * 25 + [1] * 25)

        clf.fit(X, y)

        assert clf.group_norms_.shape == (3,)

    def test_predict_returns_correct_shape(self):
        """predict returns correct shape."""
        from lmprobe.classifiers import GroupLassoClassifier

        clf = GroupLassoClassifier(hidden_dim=10, n_layers=3)
        X = np.random.randn(50, 30)
        y = np.array([0] * 25 + [1] * 25)

        clf.fit(X, y)
        preds = clf.predict(X)

        assert preds.shape == (50,)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_returns_correct_shape(self):
        """predict_proba returns (n_samples, 2)."""
        from lmprobe.classifiers import GroupLassoClassifier

        clf = GroupLassoClassifier(hidden_dim=10, n_layers=3)
        X = np.random.randn(50, 30)
        y = np.array([0] * 25 + [1] * 25)

        clf.fit(X, y)
        probs = clf.predict_proba(X)

        assert probs.shape == (50, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_dimension_mismatch_raises(self):
        """Wrong input dimensions raises ValueError."""
        from lmprobe.classifiers import GroupLassoClassifier

        clf = GroupLassoClassifier(hidden_dim=10, n_layers=3)  # expects 30 features
        X = np.random.randn(50, 20)  # wrong: 20 features
        y = np.array([0] * 25 + [1] * 25)

        with pytest.raises(ValueError, match="expected 30"):
            clf.fit(X, y)


class TestLinearProbeAutoLayers:
    """Integration tests for LinearProbe with layers='auto'."""

    def test_auto_layers_basic(self, tiny_model):
        """Basic auto layer selection works."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=tiny_model,
            layers="auto",
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive example"], ["negative example"])

        # selected_layers_ should be populated
        assert probe.selected_layers_ is not None
        assert isinstance(probe.selected_layers_, list)
        assert len(probe.selected_layers_) > 0

    def test_auto_layers_with_explicit_candidates(self, tiny_model):
        """Auto selection with explicit candidate indices."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=tiny_model,
            layers="auto",
            auto_candidates=[-2, -1],  # Last two layers
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive"], ["negative"])

        # Selected layers should be set
        assert probe.selected_layers_ is not None

    def test_auto_layers_predict_works(self, tiny_model):
        """Predictions work after auto layer selection."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=tiny_model,
            layers="auto",
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive example"], ["negative example"])
        predictions = probe.predict(["test input"])

        assert predictions.shape == (1,)

    def test_auto_layers_predict_proba_works(self, tiny_model):
        """Probabilities work after auto layer selection."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=tiny_model,
            layers="auto",
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive example"], ["negative example"])
        probs = probe.predict_proba(["test input"])

        assert probs.shape == (1, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_auto_layers_save_load(self, tiny_model, tmp_path):
        """Save/load preserves selected layers."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=tiny_model,
            layers="auto",
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive"], ["negative"])
        original_selected = probe.selected_layers_
        original_pred = probe.predict(["test"])

        # Save and load
        save_path = tmp_path / "auto_probe.pkl"
        probe.save(str(save_path))
        loaded = LinearProbe.load(str(save_path))

        # Check selected layers preserved
        assert loaded.selected_layers_ == original_selected

        # Check predictions match
        loaded_pred = loaded.predict(["test"])
        assert np.array_equal(original_pred, loaded_pred)

    def test_auto_layers_with_custom_classifier(self, tiny_model):
        """Auto layers works with custom classifier."""
        from sklearn.svm import SVC

        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=tiny_model,
            layers="auto",
            classifier=SVC(kernel="linear", probability=True, random_state=42),
            pooling="last_token",
            device="cpu",
            remote=False,
        )

        probe.fit(["positive"], ["negative"])

        # Final classifier should be SVC, not GroupLasso
        assert isinstance(probe.classifier_, SVC)
