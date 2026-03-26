"""Tests for classifier_kwargs parameter (Issue #75)."""

import pytest

from lmprobe import LinearProbe

pytestmark = pytest.mark.nnsight


class TestClassifierKwargs:
    """Tests for exposing classifier hyperparameters via classifier_kwargs."""

    def test_logistic_regression_custom_c(self, tiny_model):
        """classifier_kwargs overrides default C for logistic regression."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="logistic_regression",
            classifier_kwargs={"C": 0.01},
            device="cpu",
            remote=False,
            random_state=42,
        )
        # Check the template classifier has the right C
        assert probe._classifier_template.C == 0.01

    def test_logistic_regression_custom_solver(self, tiny_model):
        """classifier_kwargs overrides solver."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="logistic_regression",
            classifier_kwargs={"solver": "liblinear", "max_iter": 5000},
            device="cpu",
            remote=False,
            random_state=42,
        )
        assert probe._classifier_template.solver == "liblinear"
        assert probe._classifier_template.max_iter == 5000

    def test_svm_custom_c(self, tiny_model):
        """classifier_kwargs works with SVM."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="svm",
            classifier_kwargs={"C": 0.1},
            device="cpu",
            remote=False,
            random_state=42,
        )
        assert probe._classifier_template.C == 0.1

    def test_ridge_custom_alpha(self, tiny_model):
        """classifier_kwargs works with ridge regression."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="ridge_regression",
            task="regression",
            classifier_kwargs={"alpha": 10.0},
            device="cpu",
            remote=False,
        )
        assert probe._classifier_template.alpha == 10.0

    def test_fit_predict_with_kwargs(self, tiny_model):
        """Full fit/predict roundtrip with custom classifier_kwargs."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier="logistic_regression",
            classifier_kwargs={"C": 0.01, "solver": "liblinear"},
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe.fit(["good", "great"], ["bad", "terrible"])
        predictions = probe.predict(["test"])
        assert predictions.shape == (1,)

    def test_kwargs_ignored_for_custom_estimator(self, tiny_model):
        """classifier_kwargs is ignored when a custom estimator is passed."""
        from sklearn.linear_model import LogisticRegression

        custom_clf = LogisticRegression(C=0.5, random_state=42)
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            classifier=custom_clf,
            classifier_kwargs={"C": 999},  # Should be ignored
            device="cpu",
            remote=False,
        )
        # Custom estimator should be used as-is
        assert probe._classifier_template.C == 0.5

    def test_sweep_layers_passes_kwargs(self, tiny_model):
        """sweep_layers forwards classifier_kwargs to each probe."""
        result = LinearProbe.sweep_layers(
            model=tiny_model,
            positive_prompts=["good", "great"],
            negative_prompts=["bad", "terrible"],
            layers=0,
            classifier_kwargs={"C": 0.01},
            device="cpu",
            remote=False,
            random_state=42,
        )
        probe = result[0]
        assert probe._classifier_template.C == 0.01
