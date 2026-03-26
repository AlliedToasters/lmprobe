"""Tests for HuggingFace Hub integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lmprobe import LinearProbe

from conftest import TEST_MODEL

try:
    import skops  # noqa: F401

    HAS_SKOPS = True
except ImportError:
    HAS_SKOPS = False

requires_skops = pytest.mark.skipif(not HAS_SKOPS, reason="skops not installed")


@pytest.fixture
def fitted_probe():
    """Return a fitted probe using the tiny model."""
    probe = LinearProbe(
        model=TEST_MODEL,
        layers=-1,
        pooling="last_token",
        classifier="logistic_regression",
        device="cpu",
        remote=False,
        random_state=42,
    )
    positive = ["dog walks bark fetch", "good boy wag tail"]
    negative = ["cat purr scratch meow", "kitty litter nap"]
    probe.fit(positive, negative)
    return probe


class TestTrainingDataCaching:
    """Test that fit() caches training prompts."""

    def test_contrastive_mode_caches_prompts(self, fitted_probe):
        assert fitted_probe._training_positive_ == [
            "dog walks bark fetch",
            "good boy wag tail",
        ]
        assert fitted_probe._training_negative_ == [
            "cat purr scratch meow",
            "kitty litter nap",
        ]
        assert fitted_probe._training_prompts_ is None
        assert fitted_probe._training_labels_ is None

    def test_standard_mode_caches_prompts(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )
        prompts = ["hello world", "foo bar", "baz qux"]
        labels = [1, 0, 1]
        probe.fit(prompts, labels)

        assert probe._training_prompts_ == prompts
        assert probe._training_labels_ == labels
        assert probe._training_positive_ is None
        assert probe._training_negative_ is None

    def test_unfitted_probe_has_no_cached_data(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            remote=False,
        )
        assert probe._training_positive_ is None
        assert probe._training_negative_ is None
        assert probe._training_prompts_ is None
        assert probe._training_labels_ is None


class TestEvaluate:
    """Test the evaluate() method."""

    def test_evaluate_returns_metrics(self, fitted_probe):
        test_prompts = ["woof bark play", "meow purr sleep"]
        test_labels = [1, 0]
        results = fitted_probe.evaluate(test_prompts, test_labels)

        assert "accuracy" in results
        assert "f1" in results
        assert "precision" in results
        assert "recall" in results
        assert "n_eval" in results
        assert "eval_hash" in results
        assert results["n_eval"] == 2
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_evaluate_includes_auroc(self, fitted_probe):
        test_prompts = ["woof bark play", "meow purr sleep"]
        test_labels = [1, 0]
        results = fitted_probe.evaluate(test_prompts, test_labels)

        # LogisticRegression supports predict_proba, so auroc should be present
        assert "auroc" in results
        assert 0.0 <= results["auroc"] <= 1.0

    def test_evaluate_caches_results(self, fitted_probe):
        test_prompts = ["woof bark play", "meow purr sleep"]
        test_labels = [1, 0]
        results = fitted_probe.evaluate(test_prompts, test_labels)

        assert fitted_probe._evaluation_results_ is results

    def test_evaluate_on_unfitted_raises(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            remote=False,
        )
        with pytest.raises(RuntimeError, match="not been fitted"):
            probe.evaluate(["test"], [1])


class TestBuildProbeConfig:
    """Test _build_probe_config."""

    def test_config_has_required_fields(self, fitted_probe):
        from lmprobe.hub import _build_probe_config

        config = _build_probe_config(fitted_probe)

        assert "lmprobe_version" in config
        assert "config_version" in config
        assert config["config_version"] == 1
        assert "base_model" in config
        assert config["base_model"]["name"] == TEST_MODEL
        assert "probe" in config
        assert "classes" in config

    def test_config_probe_section(self, fitted_probe):
        from lmprobe.hub import _build_probe_config

        config = _build_probe_config(fitted_probe)
        probe_cfg = config["probe"]

        assert probe_cfg["layers"] is not None
        assert probe_cfg["pooling"] == "last_token"
        assert probe_cfg["classifier_type"] == "logistic_regression"
        assert probe_cfg["task"] == "classification"
        assert probe_cfg["random_state"] == 42

    def test_config_with_class_labels(self, fitted_probe):
        from lmprobe.hub import _build_probe_config

        config = _build_probe_config(
            fitted_probe,
            class_labels={0: "negative", 1: "positive"},
        )
        assert config["class_labels"] == {"0": "negative", "1": "positive"}

    def test_config_classes_from_fitted(self, fitted_probe):
        from lmprobe.hub import _build_probe_config

        config = _build_probe_config(fitted_probe)
        assert config["classes"] == [0, 1]


class TestBuildTrainingInfo:
    """Test _build_training_info."""

    def test_training_info_with_data(self, fitted_probe):
        from lmprobe.hub import _build_training_info

        info = _build_training_info(fitted_probe, include_training_data=True)

        td = info["training_data"]
        assert td["n_positive"] == 2
        assert td["n_negative"] == 2
        assert "positive_hash" in td
        assert td["positive_hash"].startswith("sha256:")
        assert "positive_examples" in td
        assert "negative_examples" in td

    def test_training_info_without_data(self, fitted_probe):
        from lmprobe.hub import _build_training_info

        info = _build_training_info(fitted_probe, include_training_data=False)

        td = info["training_data"]
        assert td["n_positive"] == 2
        assert "positive_hash" in td
        assert "positive_examples" not in td
        assert "negative_examples" not in td

    def test_training_info_with_metrics(self, fitted_probe):
        from lmprobe.hub import _build_training_info

        metrics = {"accuracy": 0.95, "auroc": 0.98, "n_eval": 20, "eval_hash": "sha256:abc"}
        info = _build_training_info(fitted_probe, metrics=metrics)

        assert info["evaluation"]["metrics"]["accuracy"] == 0.95
        assert info["evaluation"]["metrics"]["auroc"] == 0.98
        assert info["evaluation"]["eval_set_size"] == 20

    def test_training_info_environment(self, fitted_probe):
        from lmprobe.hub import _build_training_info

        info = _build_training_info(fitted_probe)
        env = info["training_environment"]

        assert "lmprobe_version" in env
        assert "python_version" in env
        assert "torch_version" in env
        assert "sklearn_version" in env

    def test_training_info_timestamps(self, fitted_probe):
        from lmprobe.hub import _build_training_info

        info = _build_training_info(fitted_probe)
        assert "timestamps" in info
        assert "trained_at" in info["timestamps"]
        assert "pushed_at" in info["timestamps"]


class TestRenderModelCard:
    """Test _render_model_card."""

    def test_model_card_has_yaml_frontmatter(self, fitted_probe):
        from lmprobe.hub import _build_probe_config, _build_training_info, _render_model_card

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        info = _build_training_info(fitted_probe)
        card = _render_model_card(config, info)

        assert card.startswith("---\n")
        assert "library_name: lmprobe" in card
        assert "pipeline_tag: text-classification" in card

    def test_model_card_with_description(self, fitted_probe):
        from lmprobe.hub import _build_probe_config, _build_training_info, _render_model_card

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        info = _build_training_info(fitted_probe)
        card = _render_model_card(config, info, description="Detects dogs vs cats")

        assert "Detects dogs vs cats" in card

    def test_model_card_with_tags(self, fitted_probe):
        from lmprobe.hub import _build_probe_config, _build_training_info, _render_model_card

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        info = _build_training_info(fitted_probe)
        card = _render_model_card(config, info, tags=["safety", "custom-tag"])

        assert "  - safety" in card
        assert "  - custom-tag" in card

    def test_model_card_no_evaluation(self, fitted_probe):
        from lmprobe.hub import _build_probe_config, _build_training_info, _render_model_card

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        info = _build_training_info(fitted_probe)
        # No evaluation in this info
        card = _render_model_card(config, info)

        assert "No evaluation results provided" in card

    def test_model_card_with_evaluation(self, fitted_probe):
        from lmprobe.hub import _build_probe_config, _build_training_info, _render_model_card

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        metrics = {"accuracy": 0.95, "auroc": 0.98, "n_eval": 20}
        info = _build_training_info(fitted_probe, metrics=metrics)
        card = _render_model_card(config, info)

        assert "0.9500" in card
        assert "0.9800" in card


    def test_model_card_uses_repo_id_in_usage(self, fitted_probe):
        """Issue #57: usage example should show actual repo_id, not placeholder."""
        from lmprobe.hub import _build_probe_config, _build_training_info, _render_model_card

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        info = _build_training_info(fitted_probe)
        card = _render_model_card(
            config, info, repo_id="latent-lab/my-probe"
        )

        assert 'from_hub("latent-lab/my-probe"' in card
        assert "REPO_ID" not in card

    def test_model_card_fallback_repo_id(self, fitted_probe):
        """When repo_id is not provided, falls back to REPO_ID placeholder."""
        from lmprobe.hub import _build_probe_config, _build_training_info, _render_model_card

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        info = _build_training_info(fitted_probe)
        card = _render_model_card(config, info)

        assert "REPO_ID" in card

    def test_model_card_omits_limitations_by_default(self, fitted_probe):
        """Issue #58: no placeholder limitations text when not provided."""
        from lmprobe.hub import _build_probe_config, _build_training_info, _render_model_card

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        info = _build_training_info(fitted_probe)
        card = _render_model_card(config, info)

        assert "Limitations and Intended Use" not in card
        assert "Please fill in" not in card

    def test_model_card_with_limitations(self, fitted_probe):
        """Issue #58: limitations text renders when provided."""
        from lmprobe.hub import _build_probe_config, _build_training_info, _render_model_card

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        info = _build_training_info(fitted_probe)
        card = _render_model_card(
            config, info, limitations="Not suitable for production use."
        )

        assert "## Limitations and Intended Use" in card
        assert "Not suitable for production use." in card

    def test_model_card_compact_layers_for_named_spec(self, fitted_probe):
        """Issue #59: named layer specs show compact format."""
        from lmprobe.hub import _build_probe_config, _build_training_info, _render_model_card

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        # Simulate "all" spec with many layers
        config["probe"]["layers_spec_original"] = "all"
        config["probe"]["layers"] = list(range(30))
        info = _build_training_info(fitted_probe)
        card = _render_model_card(config, info)

        assert "all (0\u201329, 30 layers)" in card
        # Should NOT contain the full list
        assert "[0, 1, 2, 3," not in card

    def test_model_card_eval_hash(self, fitted_probe):
        """Issue #61: eval hash and size shown in model card."""
        from lmprobe.hub import _build_probe_config, _build_training_info, _render_model_card

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        metrics = {
            "accuracy": 0.95,
            "n_eval": 300,
            "eval_hash": "sha256:abc123",
        }
        info = _build_training_info(fitted_probe, metrics=metrics)
        card = _render_model_card(config, info)

        assert "**Evaluation samples**: 300" in card
        assert "**Evaluation hash**: `sha256:abc123`" in card


class TestSerializationRoundtrip:
    """Test classifier serialization and deserialization."""

    @requires_skops
    def test_skops_roundtrip(self, fitted_probe, tmp_path):
        from lmprobe.hub import _load_classifier, _serialize_classifier

        fmt = _serialize_classifier(fitted_probe.classifier_, tmp_path / "classifier")
        assert fmt == "skops"

        loaded = _load_classifier(tmp_path, fmt, trust_classifier=True)
        assert type(loaded).__name__ == type(fitted_probe.classifier_).__name__

        # Verify predictions match
        # Create some test data
        rng = np.random.RandomState(42)
        X = rng.randn(5, fitted_probe.classifier_.coef_.shape[1])
        original_preds = fitted_probe.classifier_.predict(X)
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_serialize_roundtrip(self, fitted_probe, tmp_path):
        """Test serialization roundtrip with whatever format is available."""
        from lmprobe.hub import _load_classifier, _serialize_classifier

        fmt = _serialize_classifier(fitted_probe.classifier_, tmp_path / "classifier")
        assert fmt in ("skops", "joblib")

        loaded = _load_classifier(tmp_path, fmt, trust_classifier=True)
        assert type(loaded).__name__ == type(fitted_probe.classifier_).__name__

        rng = np.random.RandomState(42)
        X = rng.randn(5, fitted_probe.classifier_.coef_.shape[1])
        original_preds = fitted_probe.classifier_.predict(X)
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_load_without_trust_raises(self, fitted_probe, tmp_path):
        from lmprobe.hub import _load_classifier, _serialize_classifier

        fmt = _serialize_classifier(fitted_probe.classifier_, tmp_path / "classifier")

        with pytest.raises(ValueError, match="trust_classifier=True"):
            _load_classifier(tmp_path, fmt, trust_classifier=False)

    def test_safetensors_not_implemented(self, tmp_path):
        from lmprobe.hub import _load_classifier

        with pytest.raises(NotImplementedError, match="safetensors"):
            _load_classifier(tmp_path, "safetensors", trust_classifier=True)

    def test_unknown_format_raises(self, tmp_path):
        from lmprobe.hub import _load_classifier

        with pytest.raises(ValueError, match="Unknown serialization format"):
            _load_classifier(tmp_path, "unknown_format", trust_classifier=True)


class TestProbeCard:
    """Test ProbeCard dataclass."""

    def test_from_local(self, fitted_probe, tmp_path):
        from lmprobe.hub import (
            ProbeCard,
            _build_probe_config,
            _build_training_info,
        )

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        with open(tmp_path / "probe_config.json", "w") as f:
            json.dump(config, f)

        info = _build_training_info(fitted_probe)
        with open(tmp_path / "training_info.json", "w") as f:
            json.dump(info, f)

        card = ProbeCard.from_local(tmp_path)

        assert card.base_model == TEST_MODEL
        assert card.pooling == "last_token"
        assert card.classifier_type == "logistic_regression"
        assert card.task == "classification"
        assert card.random_state == 42
        assert card.n_positive == 2
        assert card.n_negative == 2
        assert card.positive_hash is not None

    def test_from_local_without_training_info(self, fitted_probe, tmp_path):
        from lmprobe.hub import ProbeCard, _build_probe_config

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        with open(tmp_path / "probe_config.json", "w") as f:
            json.dump(config, f)

        card = ProbeCard.from_local(tmp_path)

        assert card.base_model == TEST_MODEL
        assert card.n_positive is None
        assert card.metrics is None

    def test_to_reproduce_config(self, fitted_probe, tmp_path):
        from lmprobe.hub import ProbeCard, _build_probe_config

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        with open(tmp_path / "probe_config.json", "w") as f:
            json.dump(config, f)

        card = ProbeCard.from_local(tmp_path)
        repro = card.to_reproduce_config()

        assert repro["model"] == TEST_MODEL
        assert repro["pooling"] == "last_token"
        assert repro["classifier"] == "logistic_regression"
        assert repro["random_state"] == 42

    def test_is_compatible_with(self, fitted_probe, tmp_path):
        from lmprobe.hub import ProbeCard, _build_probe_config

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        with open(tmp_path / "probe_config.json", "w") as f:
            json.dump(config, f)

        card = ProbeCard.from_local(tmp_path)
        assert card.is_compatible_with(TEST_MODEL)
        assert not card.is_compatible_with("some-other-model")

    def test_training_data_hash(self, fitted_probe, tmp_path):
        from lmprobe.hub import (
            ProbeCard,
            _build_probe_config,
            _build_training_info,
        )

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = "skops"
        with open(tmp_path / "probe_config.json", "w") as f:
            json.dump(config, f)

        info = _build_training_info(fitted_probe)
        with open(tmp_path / "training_info.json", "w") as f:
            json.dump(info, f)

        card = ProbeCard.from_local(tmp_path)
        assert card.training_data_hash is not None
        assert "sha256:" in card.training_data_hash


class TestPushToHubValidation:
    """Test push_to_hub validation (no network)."""

    def test_push_unfitted_raises(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            remote=False,
        )
        with pytest.raises(RuntimeError, match="not been fitted"):
            probe.push_to_hub("test/repo")


class TestFromHubValidation:
    """Test from_hub validation (no network)."""

    def test_from_hub_without_trust_raises(self):
        with pytest.raises((ValueError, Exception)):
            LinearProbe.from_hub("nonexistent/repo", trust_classifier=False)


class TestHashPrompts:
    """Test prompt hashing."""

    def test_hash_deterministic(self):
        from lmprobe.hub import _hash_prompts

        h1 = _hash_prompts(["a", "b", "c"])
        h2 = _hash_prompts(["a", "b", "c"])
        assert h1 == h2

    def test_hash_order_invariant(self):
        from lmprobe.hub import _hash_prompts

        h1 = _hash_prompts(["a", "b", "c"])
        h2 = _hash_prompts(["c", "a", "b"])
        assert h1 == h2

    def test_hash_different_for_different_data(self):
        from lmprobe.hub import _hash_prompts

        h1 = _hash_prompts(["a", "b"])
        h2 = _hash_prompts(["x", "y"])
        assert h1 != h2

    def test_hash_starts_with_sha256(self):
        from lmprobe.hub import _hash_prompts

        h = _hash_prompts(["test"])
        assert h.startswith("sha256:")


class TestLocalRoundtrip:
    """Test full serialization roundtrip without network."""

    def test_serialize_and_load_locally(self, fitted_probe, tmp_path):
        """Test the full serialize -> load cycle using local files."""
        from lmprobe.hub import (
            ProbeCard,
            _build_probe_config,
            _build_training_info,
            _load_classifier,
            _render_model_card,
            _serialize_classifier,
        )

        # Serialize
        fmt = _serialize_classifier(fitted_probe.classifier_, tmp_path / "classifier")

        config = _build_probe_config(fitted_probe)
        config["probe"]["serialization_format"] = fmt

        with open(tmp_path / "probe_config.json", "w") as f:
            json.dump(config, f)

        info = _build_training_info(fitted_probe, include_training_data=True)
        with open(tmp_path / "training_info.json", "w") as f:
            json.dump(info, f)

        card_text = _render_model_card(config, info)
        with open(tmp_path / "README.md", "w") as f:
            f.write(card_text)

        # Load and verify
        loaded_classifier = _load_classifier(tmp_path, fmt, trust_classifier=True)

        rng = np.random.RandomState(42)
        X = rng.randn(5, fitted_probe.classifier_.coef_.shape[1])
        original_preds = fitted_probe.classifier_.predict_proba(X)
        loaded_preds = loaded_classifier.predict_proba(X)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

        # Verify ProbeCard reads correctly
        card = ProbeCard.from_local(tmp_path)
        assert card.base_model == TEST_MODEL
        assert card.n_positive == 2

        # Verify README
        assert Path(tmp_path / "README.md").exists()
        readme = (tmp_path / "README.md").read_text()
        assert "lmprobe" in readme
