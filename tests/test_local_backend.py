"""Tests for the local extraction backend and pluggable backend interface.

Covers:
- ExtractionBackend ABC interface (#19)
- NnsightBackend wrapping (#20)
- LocalBackend with HuggingFace + hooks (#21)
- Device/dtype configuration (#22)
- Issue #23: device_map fix
- Issue #25: BitNet config override
"""

from abc import ABC
from unittest.mock import patch

import numpy as np
import pytest
import torch
from conftest import NEGATIVE_PROMPTS, POSITIVE_PROMPTS, TEST_PROMPTS

from lmprobe import LinearProbe
from lmprobe.backends import (
    ExtractionBackend,
    LocalBackend,
    NnsightBackend,
    _get_decoder_layers,
    _get_local_model,
    clear_local_model_cache,
    resolve_backend,
)

pytestmark = pytest.mark.nnsight

# ── ExtractionBackend ABC ────────────────────────────────────────────────────

class TestExtractionBackendABC:
    """Verify ExtractionBackend is a proper ABC with required methods."""

    def test_is_abstract_base_class(self):
        assert issubclass(ExtractionBackend, ABC)

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError, match="abstract"):
            ExtractionBackend("some-model", "cpu")

    def test_has_extract_batch(self):
        assert hasattr(ExtractionBackend, "extract_batch")
        assert getattr(ExtractionBackend.extract_batch, "__isabstractmethod__", False)

    def test_has_extract_batch_with_logits(self):
        assert hasattr(ExtractionBackend, "extract_batch_with_logits")
        assert getattr(
            ExtractionBackend.extract_batch_with_logits,
            "__isabstractmethod__",
            False,
        )

    def test_has_tokenizer_property(self):
        assert isinstance(ExtractionBackend.tokenizer, property)

    def test_has_model_property(self):
        assert isinstance(ExtractionBackend.model, property)

    def test_nnsight_implements_interface(self):
        assert issubclass(NnsightBackend, ExtractionBackend)

    def test_local_implements_interface(self):
        assert issubclass(LocalBackend, ExtractionBackend)


# ── Backend consumption verification ─────────────────────────────────────────

class TestAllConsumptionSitesUseBackend:
    """Verify that probe pipeline goes through the backend interface."""

    def test_probe_uses_local_backend(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model, layers=-1, device="cpu", backend="local",
        )
        assert hasattr(probe._extractor, "_backend")
        assert isinstance(probe._extractor._backend, ExtractionBackend)
        assert isinstance(probe._extractor._backend, LocalBackend)

    def test_probe_uses_nnsight_backend(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model, layers=-1, device="cpu", backend="nnsight",
        )
        assert isinstance(probe._extractor._backend, NnsightBackend)


# ── LocalBackend ─────────────────────────────────────────────────────────────

class TestLocalBackend:
    """Tests for LocalBackend directly."""

    def test_resolve_backend_local(self, tiny_model):
        backend = resolve_backend("local", tiny_model, "cpu")
        assert isinstance(backend, LocalBackend)

    def test_resolve_backend_invalid_local_remote(self, tiny_model):
        with pytest.raises(ValueError, match="does not support remote"):
            resolve_backend("local", tiny_model, "cpu", remote=True)

    def test_resolve_backend_unknown(self, tiny_model):
        with pytest.raises(ValueError, match="Unknown backend"):
            resolve_backend("invalid", tiny_model, "cpu")

    def test_model_is_hf_pretrained(self, tiny_model):
        from transformers import PreTrainedModel
        backend = LocalBackend(tiny_model, device="cpu")
        assert isinstance(backend.model, PreTrainedModel)

    def test_decoder_layers_found(self, tiny_model):
        backend = LocalBackend(tiny_model, device="cpu")
        layers = _get_decoder_layers(backend.model)
        assert len(layers) > 0

    def test_extract_batch(self, tiny_model):
        backend = resolve_backend("local", tiny_model, "cpu")
        acts, mask = backend.extract_batch(
            ["Hello world", "Test prompt"], [0]
        )
        assert acts.ndim == 3
        assert mask.ndim == 2
        assert acts.shape[0] == 2
        assert mask.shape[0] == 2

    def test_extract_batch_multilayer(self, tiny_model):
        backend = resolve_backend("local", tiny_model, "cpu")
        acts_single, _ = backend.extract_batch(["Test"], [0])
        acts_multi, _ = backend.extract_batch(["Test"], [0, 1])
        assert acts_multi.shape[-1] == 2 * acts_single.shape[-1]

    def test_extract_batch_with_logits(self, tiny_model):
        backend = resolve_backend("local", tiny_model, "cpu")
        acts, mask, logits, logits_indices = backend.extract_batch_with_logits(
            ["Hello world"], [0]
        )
        assert acts.ndim == 3
        assert logits.ndim == 3
        assert logits.shape[0] == 1
        assert logits_indices is None

    def test_tokenizer_property(self, tiny_model):
        backend = resolve_backend("local", tiny_model, "cpu")
        tokenizer = backend.tokenizer
        assert tokenizer is not None
        tokens = tokenizer("Hello", return_tensors="pt")
        assert "input_ids" in tokens

    def test_hooks_cleaned_up(self, tiny_model):
        """After extraction, no hooks remain on the model."""
        backend = LocalBackend(tiny_model, device="cpu")
        model = backend.model
        decoder_layers = _get_decoder_layers(model)

        hooks_before = sum(
            len(layer._forward_hooks) for layer in decoder_layers
        )
        backend.extract_batch(["test prompt"], [0])
        hooks_after = sum(
            len(layer._forward_hooks) for layer in decoder_layers
        )
        assert hooks_after == hooks_before

    def test_unrecognized_architecture_raises(self):
        class FakeModel:
            pass
        with pytest.raises(ValueError, match="Could not find decoder layers"):
            _get_decoder_layers(FakeModel())


# ── NnsightBackend ───────────────────────────────────────────────────────────

class TestNnsightBackend:
    """Verify NnsightBackend properly wraps the existing nnsight code."""

    def test_resolve_nnsight(self, tiny_model):
        backend = resolve_backend("nnsight", tiny_model, "cpu")
        assert isinstance(backend, NnsightBackend)

    def test_nnsight_supports_remote_flag(self, tiny_model):
        backend = resolve_backend("nnsight", tiny_model, "cpu", remote=True)
        assert isinstance(backend, NnsightBackend)
        assert backend.remote is True

    def test_nnsight_default_not_remote(self, tiny_model):
        backend = resolve_backend("nnsight", tiny_model, "cpu")
        assert backend.remote is False

    def test_nnsight_extract_batch(self, tiny_model):
        backend = resolve_backend("nnsight", tiny_model, "cpu")
        acts, mask = backend.extract_batch(["Hello world"], [0])
        assert acts.ndim == 3
        assert mask.ndim == 2
        assert acts.shape[0] == 1

    def test_nnsight_tokenizer_accessible(self, tiny_model):
        backend = resolve_backend("nnsight", tiny_model, "cpu")
        tok = backend.tokenizer
        assert tok is not None
        result = tok("test", return_tensors="pt")
        assert "input_ids" in result


# ── Full pipeline tests ──────────────────────────────────────────────────────

class TestLocalFitPredict:
    """Full pipeline tests with backend='local'."""

    def test_fit_predict(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model, layers=-1, pooling="last_token",
            classifier="logistic_regression", device="cpu",
            remote=False, random_state=42, backend="local",
        )

        probe.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        predictions = probe.predict(TEST_PROMPTS)
        assert predictions.shape == (2,)

        probabilities = probe.predict_proba(TEST_PROMPTS)
        assert probabilities.shape == (2, 2)

        accuracy = probe.score(TEST_PROMPTS, [1, 0])
        assert 0.0 <= accuracy <= 1.0

    def test_multilayer(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model, layers=[0, 1], pooling="last_token",
            classifier="logistic_regression", device="cpu",
            remote=False, random_state=42, backend="local",
        )

        probe.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)
        predictions = probe.predict(TEST_PROMPTS)
        assert predictions.shape == (2,)

    def test_caching(self, tiny_model, tmp_path, monkeypatch):
        """Verify disk cache works with local backend."""
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))

        probe = LinearProbe(
            model=tiny_model, layers=-1, pooling="last_token",
            device="cpu", random_state=42, backend="local",
        )
        probe.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        probe2 = LinearProbe(
            model=tiny_model, layers=-1, pooling="last_token",
            device="cpu", random_state=42, backend="local",
        )
        probe2.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        p1 = probe.predict(TEST_PROMPTS)
        p2 = probe2.predict(TEST_PROMPTS)
        np.testing.assert_array_equal(p1, p2)

    def test_nnsight_caching(self, tiny_model, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        probe = LinearProbe(
            model=tiny_model, layers=-1, device="cpu",
            backend="nnsight", random_state=42,
        )
        probe.fit(["positive"], ["negative"])
        cache_files = list(tmp_path.rglob("*.safetensors")) + list(tmp_path.rglob("*.pt"))
        assert len(cache_files) > 0

    def test_save_load(self, tiny_model, tmp_path):
        probe = LinearProbe(
            model=tiny_model, layers=-1, pooling="last_token",
            device="cpu", random_state=42, backend="local",
        )
        probe.fit(POSITIVE_PROMPTS, NEGATIVE_PROMPTS)

        save_path = str(tmp_path / "probe.pkl")
        probe.save(save_path)

        loaded = LinearProbe.load(save_path)
        assert loaded.backend == "local"

        p1 = probe.predict(TEST_PROMPTS)
        p2 = loaded.predict(TEST_PROMPTS)
        np.testing.assert_array_equal(p1, p2)

    def test_invalid_local_remote(self, tiny_model):
        with pytest.raises(ValueError, match="does not support remote"):
            LinearProbe(
                model=tiny_model, layers=-1, device="cpu",
                remote=True, backend="local",
            )


# ── Device/dtype configuration ───────────────────────────────────────────────

class TestDeviceConfiguration:
    """Verify device parameter works for LocalBackend."""

    def test_cpu_device(self, tiny_model):
        backend = LocalBackend(tiny_model, device="cpu")
        device = next(backend.model.parameters()).device
        assert device.type == "cpu"

    def test_probe_cpu_device(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model, layers=-1, device="cpu", backend="local",
        )
        backend = probe._extractor._backend
        assert isinstance(backend, LocalBackend)
        device = next(backend.model.parameters()).device
        assert device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available()
        or not torch.cuda.get_device_capability()[0] >= 7,
        reason="Compatible CUDA device not available",
    )
    def test_cuda_device(self, tiny_model):
        backend = LocalBackend(tiny_model, device="cuda")
        device = next(backend.model.parameters()).device
        assert device.type == "cuda"
        acts, mask = backend.extract_batch(["Hello"], [0])
        assert acts.device.type == "cuda"

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_mps_device(self, tiny_model):
        backend = LocalBackend(tiny_model, device="mps")
        device = next(backend.model.parameters()).device
        assert device.type == "mps"
        acts, mask = backend.extract_batch(["Hello"], [0])
        assert acts.device.type == "mps"


class TestDtypeConfiguration:
    """Verify dtype parameter works for LocalBackend."""

    def test_default_dtype_float32(self, tiny_model):
        backend = LocalBackend(tiny_model, device="cpu")
        param_dtype = next(backend.model.parameters()).dtype
        assert param_dtype == torch.float32

    def test_explicit_float32(self, tiny_model):
        backend = LocalBackend(tiny_model, device="cpu", dtype=torch.float32)
        param_dtype = next(backend.model.parameters()).dtype
        assert param_dtype == torch.float32

    def test_bfloat16_dtype(self, tiny_model):
        backend = LocalBackend(tiny_model, device="cpu", dtype=torch.bfloat16)
        param_dtype = next(backend.model.parameters()).dtype
        assert param_dtype == torch.bfloat16

    def test_float16_dtype(self, tiny_model):
        backend = LocalBackend(tiny_model, device="cpu", dtype=torch.float16)
        param_dtype = next(backend.model.parameters()).dtype
        assert param_dtype == torch.float16

    def test_dtype_through_resolve_backend(self, tiny_model):
        backend = resolve_backend(
            "local", tiny_model, "cpu", dtype=torch.bfloat16
        )
        assert isinstance(backend, LocalBackend)
        assert backend.dtype == torch.bfloat16

    def test_dtype_ignored_for_nnsight(self, tiny_model):
        backend = resolve_backend(
            "nnsight", tiny_model, "cpu", dtype=torch.bfloat16
        )
        assert isinstance(backend, NnsightBackend)

    def test_dtype_through_probe_string(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model, layers=-1, device="cpu",
            backend="local", dtype="bfloat16",
        )
        backend = probe._extractor._backend
        assert isinstance(backend, LocalBackend)
        assert backend.dtype == torch.bfloat16

    def test_invalid_dtype_string(self, tiny_model):
        with pytest.raises(ValueError, match="Unknown dtype"):
            LinearProbe(
                model=tiny_model, layers=-1, device="cpu",
                backend="local", dtype="invalid",
            )

    def test_bfloat16_extraction_works(self, tiny_model):
        backend = LocalBackend(tiny_model, device="cpu", dtype=torch.bfloat16)
        acts, mask = backend.extract_batch(["Hello"], [0])
        assert acts.ndim == 3
        assert acts.dtype == torch.bfloat16

    def test_bfloat16_full_pipeline(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model, layers=-1, pooling="last_token",
            device="cpu", backend="local", dtype="bfloat16",
            random_state=42,
        )
        probe.fit(
            ["walk", "fetch", "bark", "wag", "good boy"],
            ["purr", "scratch", "litter", "meow"],
        )
        preds = probe.predict(["arf!", "hiss"])
        assert preds.shape == (2,)

    def test_save_load_preserves_dtype(self, tiny_model, tmp_path):
        probe = LinearProbe(
            model=tiny_model, layers=-1, device="cpu",
            backend="local", dtype="bfloat16", random_state=42,
        )
        probe.fit(["positive"], ["negative"])

        path = str(tmp_path / "probe.pkl")
        probe.save(path)

        loaded = LinearProbe.load(path)
        assert loaded.dtype == "bfloat16"
        assert loaded.backend == "local"


# ── Issue #23: device_map fix ────────────────────────────────────────────────

class TestDeviceMapFix:
    """Tests for issue #23: device_map should not be used for explicit devices."""

    def test_cpu_loads_without_device_map(self, tiny_model):
        clear_local_model_cache()

        from transformers import AutoModelForCausalLM

        original_from_pretrained = AutoModelForCausalLM.from_pretrained
        captured_kwargs = {}

        def capturing_from_pretrained(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_from_pretrained(*args, **kwargs)

        try:
            with patch.object(
                AutoModelForCausalLM,
                "from_pretrained",
                side_effect=capturing_from_pretrained,
            ):
                model, tokenizer = _get_local_model(tiny_model, "cpu")

            assert "device_map" not in captured_kwargs
            assert next(model.parameters()).device == torch.device("cpu")
        finally:
            clear_local_model_cache()


# ── Issue #25: BitNet config override ────────────────────────────────────────

class TestBitNetConfigOverride:
    """Tests for issue #25: BitNet autobitlinear config override."""

    def test_autobitlinear_config_is_overridden(self):
        from transformers import AutoConfig

        with patch.object(AutoConfig, "from_pretrained") as mock_config:
            fake_config = type("FakeConfig", (), {
                "quantization_config": {"linear_class": "autobitlinear"},
            })()
            mock_config.return_value = fake_config

            from transformers import AutoModelForCausalLM, AutoTokenizer

            with patch.object(AutoTokenizer, "from_pretrained"):
                with patch.object(
                    AutoModelForCausalLM, "from_pretrained"
                ) as mock_model:
                    mock_model.return_value = type("FakeModel", (), {
                        "eval": lambda self: None,
                        "to": lambda self, device: self,
                    })()
                    clear_local_model_cache()
                    _get_local_model("fake-bitnet-model", "cpu")

            assert fake_config.quantization_config["linear_class"] == "bitlinear"
            clear_local_model_cache()

    def test_non_bitnet_config_unchanged(self):
        from transformers import AutoConfig

        with patch.object(AutoConfig, "from_pretrained") as mock_config:
            fake_config = type("FakeConfig", (), {
                "quantization_config": None,
            })()
            mock_config.return_value = fake_config

            from transformers import AutoModelForCausalLM, AutoTokenizer

            with patch.object(AutoTokenizer, "from_pretrained"):
                with patch.object(
                    AutoModelForCausalLM, "from_pretrained"
                ) as mock_model:
                    mock_model.return_value = type("FakeModel", (), {
                        "eval": lambda self: None,
                        "to": lambda self, device: self,
                    })()
                    clear_local_model_cache()
                    _get_local_model("fake-normal-model", "cpu")

            assert fake_config.quantization_config is None
            clear_local_model_cache()
