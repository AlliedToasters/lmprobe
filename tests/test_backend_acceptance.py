"""Tests for pluggable extraction backend acceptance criteria.

Maps to GitHub issues #19-22:
- #19: ExtractionBackend ABC interface defined, all consumption sites use it
- #20: NnsightBackend wraps existing extraction, caching is backend-agnostic
- #21: LocalBackend uses AutoModelForCausalLM + register_forward_hook
- #22: Device/dtype configuration for LocalBackend
"""

from abc import ABC

import pytest
import torch

from lmprobe import LinearProbe
from lmprobe.backends import (
    ExtractionBackend,
    LocalBackend,
    NnsightBackend,
    _get_decoder_layers,
    resolve_backend,
)

# ── Issue #19: ActivationSource interface ────────────────────────────────────

class TestExtractionBackendABC:
    """Verify ExtractionBackend is a proper ABC with required methods."""

    def test_is_abstract_base_class(self):
        assert issubclass(ExtractionBackend, ABC)

    def test_cannot_instantiate_directly(self):
        """ABC should not be directly instantiatable."""
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
        assert isinstance(
            ExtractionBackend.tokenizer, property
        )

    def test_has_model_property(self):
        assert isinstance(
            ExtractionBackend.model, property
        )

    def test_nnsight_implements_interface(self):
        assert issubclass(NnsightBackend, ExtractionBackend)

    def test_local_implements_interface(self):
        assert issubclass(LocalBackend, ExtractionBackend)


class TestAllConsumptionSitesUseBackend:
    """Verify that probe pipeline goes through the backend interface."""

    def test_probe_uses_backend(self, tiny_model):
        """LinearProbe creates an extractor that uses the backend."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            backend="local",
        )
        # The extractor should have a backend
        assert hasattr(probe._extractor, "_backend")
        assert isinstance(probe._extractor._backend, ExtractionBackend)
        assert isinstance(probe._extractor._backend, LocalBackend)

    def test_probe_nnsight_backend(self, tiny_model):
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            backend="nnsight",
        )
        assert isinstance(probe._extractor._backend, NnsightBackend)


# ── Issue #20: NnsightBackend wraps existing extraction ──────────────────────

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
        """NnsightBackend.extract_batch produces correct shapes."""
        backend = resolve_backend("nnsight", tiny_model, "cpu")
        acts, mask = backend.extract_batch(["Hello world"], [0])
        assert acts.ndim == 3  # (batch, seq_len, hidden_dim)
        assert mask.ndim == 2  # (batch, seq_len)
        assert acts.shape[0] == 1

    def test_nnsight_tokenizer_accessible(self, tiny_model):
        backend = resolve_backend("nnsight", tiny_model, "cpu")
        tok = backend.tokenizer
        assert tok is not None
        result = tok("test", return_tensors="pt")
        assert "input_ids" in result


class TestCachingIsBackendAgnostic:
    """Verify that caching works identically for both backends."""

    def test_local_caching(self, tiny_model, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        probe = LinearProbe(
            model=tiny_model, layers=-1, device="cpu",
            backend="local", random_state=42,
        )
        probe.fit(["positive"], ["negative"])
        # Cache should have files
        cache_files = list(tmp_path.rglob("*.pt"))
        assert len(cache_files) > 0

    def test_nnsight_caching(self, tiny_model, tmp_path, monkeypatch):
        monkeypatch.setenv("LMPROBE_CACHE_DIR", str(tmp_path))
        probe = LinearProbe(
            model=tiny_model, layers=-1, device="cpu",
            backend="nnsight", random_state=42,
        )
        probe.fit(["positive"], ["negative"])
        cache_files = list(tmp_path.rglob("*.pt"))
        assert len(cache_files) > 0


# ── Issue #21: LocalBackend with HuggingFace + hooks ────────────────────────

class TestLocalBackendHooks:
    """Verify LocalBackend uses AutoModelForCausalLM and register_forward_hook."""

    def test_model_is_hf_pretrained(self, tiny_model):
        """Model should be a HuggingFace PreTrainedModel."""
        from transformers import PreTrainedModel
        backend = LocalBackend(tiny_model, device="cpu")
        assert isinstance(backend.model, PreTrainedModel)

    def test_decoder_layers_found(self, tiny_model):
        """_get_decoder_layers finds layers for tiny-random-llama."""
        backend = LocalBackend(tiny_model, device="cpu")
        layers = _get_decoder_layers(backend.model)
        assert len(layers) > 0

    def test_hooks_cleaned_up(self, tiny_model):
        """After extraction, no hooks remain on the model."""
        backend = LocalBackend(tiny_model, device="cpu")
        # Force model load
        model = backend.model
        decoder_layers = _get_decoder_layers(model)

        # Count hooks before extraction
        hooks_before = sum(
            len(layer._forward_hooks) for layer in decoder_layers
        )

        # Run extraction
        backend.extract_batch(["test prompt"], [0])

        # Hooks should be cleaned up
        hooks_after = sum(
            len(layer._forward_hooks) for layer in decoder_layers
        )
        assert hooks_after == hooks_before

    def test_extract_single_layer(self, tiny_model):
        backend = LocalBackend(tiny_model, device="cpu")
        acts, mask = backend.extract_batch(["Hello"], [0])
        assert acts.ndim == 3
        assert acts.shape[0] == 1

    def test_extract_multiple_layers_concatenates(self, tiny_model):
        backend = LocalBackend(tiny_model, device="cpu")
        acts_one, _ = backend.extract_batch(["Hello"], [0])
        acts_two, _ = backend.extract_batch(["Hello"], [0, 1])
        assert acts_two.shape[-1] == 2 * acts_one.shape[-1]

    def test_logits_extraction(self, tiny_model):
        backend = LocalBackend(tiny_model, device="cpu")
        acts, mask, logits = backend.extract_batch_with_logits(["Hello"], [0])
        assert logits.ndim == 3  # (batch, seq_len, vocab_size)

    def test_full_pipeline_local(self, tiny_model):
        """Full fit/predict/score pipeline with local backend."""
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            pooling="last_token",
            classifier="logistic_regression",
            device="cpu",
            backend="local",
            random_state=42,
        )
        positive = ["walk", "fetch", "bark", "wag", "good boy"]
        negative = ["purr", "scratch", "litter", "meow"]
        probe.fit(positive, negative)

        preds = probe.predict(["arf!", "hiss"])
        assert preds.shape == (2,)

        probs = probe.predict_proba(["arf!", "hiss"])
        assert probs.shape == (2, 2)

        acc = probe.score(["arf!", "hiss"], [1, 0])
        assert 0.0 <= acc <= 1.0

    def test_unrecognized_architecture_raises(self):
        """_get_decoder_layers raises for unknown architectures."""
        class FakeModel:
            pass
        with pytest.raises(ValueError, match="Could not find decoder layers"):
            _get_decoder_layers(FakeModel())

    def test_local_remote_incompatible(self, tiny_model):
        with pytest.raises(ValueError, match="does not support remote"):
            resolve_backend("local", tiny_model, "cpu", remote=True)


# ── Issue #22: Device/dtype configuration ────────────────────────────────────

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
        """dtype param should not cause errors for nnsight backend."""
        backend = resolve_backend(
            "nnsight", tiny_model, "cpu", dtype=torch.bfloat16
        )
        assert isinstance(backend, NnsightBackend)

    def test_dtype_through_probe_string(self, tiny_model):
        """LinearProbe accepts dtype as string."""
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
        """Activations can be extracted with bfloat16 model."""
        backend = LocalBackend(tiny_model, device="cpu", dtype=torch.bfloat16)
        acts, mask = backend.extract_batch(["Hello"], [0])
        assert acts.ndim == 3
        # Activations should be float (detached from bfloat16 model)
        assert acts.dtype == torch.bfloat16

    def test_bfloat16_full_pipeline(self, tiny_model):
        """Full pipeline works with bfloat16 dtype."""
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
        """Save/load roundtrip preserves dtype setting."""
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
