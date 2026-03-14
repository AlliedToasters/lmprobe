"""Tests for preprocessing pipeline parameter (Issue #74)."""

from lmprobe import LinearProbe


class TestPreprocessing:
    """Tests for the preprocessing pipeline feature."""

    def test_standard_scaler_string(self, tiny_model):
        """preprocessing='standard' applies StandardScaler."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            preprocessing="standard",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["good", "great", "nice"], ["bad", "terrible", "awful"])
        assert probe.preprocessing_pipeline_ is not None
        assert len(probe.preprocessing_pipeline_.steps) == 1
        assert probe.preprocessing_pipeline_.steps[0][0] == "scaler"

    def test_standard_plus_pca(self, tiny_model):
        """preprocessing='standard+pca:4' applies StandardScaler then PCA."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            preprocessing="standard+pca:4",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(
            ["good", "great", "nice", "wonderful", "excellent"],
            ["bad", "terrible", "awful", "horrible", "dreadful"],
        )
        assert probe.preprocessing_pipeline_ is not None
        assert len(probe.preprocessing_pipeline_.steps) == 2
        assert probe.preprocessing_pipeline_.steps[0][0] == "scaler"
        assert probe.preprocessing_pipeline_.steps[1][0] == "pca"

    def test_pca_with_pca_components_param(self, tiny_model):
        """pca_components param sets PCA n_components when not specified inline."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            preprocessing="standard+pca",
            pca_components=4,
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(
            ["good", "great", "nice", "wonderful", "excellent"],
            ["bad", "terrible", "awful", "horrible", "dreadful"],
        )
        assert probe.preprocessing_pipeline_ is not None
        pca_step = probe.preprocessing_pipeline_.named_steps["pca"]
        assert pca_step.n_components == 4

    def test_fit_predict_with_preprocessing(self, tiny_model):
        """Full roundtrip with preprocessing works correctly."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            preprocessing="standard+pca:4",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(
            ["good", "great", "nice", "wonderful", "excellent"],
            ["bad", "terrible", "awful", "horrible", "dreadful"],
        )
        predictions = probe.predict(["test input"])
        assert predictions.shape == (1,)

        probabilities = probe.predict_proba(["test input"])
        assert probabilities.shape == (1, 2)

    def test_no_preprocessing_by_default(self, tiny_model):
        """No preprocessing pipeline when preprocessing=None."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["good", "great"], ["bad", "terrible"])
        assert probe.preprocessing_pipeline_ is None

    def test_preprocessing_as_list(self, tiny_model):
        """preprocessing accepts list of step strings."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            preprocessing=["standard_scaler", "pca:4"],
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(
            ["good", "great", "nice", "wonderful", "excellent"],
            ["bad", "terrible", "awful", "horrible", "dreadful"],
        )
        assert probe.preprocessing_pipeline_ is not None
        assert len(probe.preprocessing_pipeline_.steps) == 2

    def test_invalid_preprocessing_step(self, tiny_model):
        """Unknown preprocessing step raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Unknown preprocessing step"):
            probe = LinearProbe(
                model=tiny_model,
                layers=-1,
                preprocessing="unknown_step",
                device="cpu",
                remote=False,
            )
            probe.fit(["good"], ["bad"])

    def test_pca_without_components_raises(self, tiny_model):
        """PCA without component count raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="PCA requested but no component count"):
            probe = LinearProbe(
                model=tiny_model,
                layers=-1,
                preprocessing="pca",
                device="cpu",
                remote=False,
            )
            probe.fit(["good"], ["bad"])

    def test_score_with_preprocessing(self, tiny_model):
        """score() works correctly with preprocessing."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            preprocessing="standard+pca:4",
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(
            ["good", "great", "nice", "wonderful", "excellent"],
            ["bad", "terrible", "awful", "horrible", "dreadful"],
        )
        accuracy = probe.score(["test one", "test two"], [1, 0])
        assert 0.0 <= accuracy <= 1.0
