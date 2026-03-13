"""Pluggable extraction backends for lmprobe.

This module defines the ExtractionBackend ABC and provides two implementations:
- NnsightBackend: Uses nnsight for model loading and activation extraction
  (supports both local and remote/NDIF execution)
- LocalBackend: Uses HuggingFace transformers directly with register_forward_hook
  (local-only, no nnsight dependency for extraction)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class ExtractionBackend(ABC):
    """Abstract base class for activation extraction backends.

    A backend owns the model: it loads, caches, and manages the model lifecycle.
    It provides methods to extract activations (and optionally logits) from
    specified layers.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification ("auto", "cpu", "cuda:0", etc.).
    """

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def extract_batch(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract activations for a batch of prompts.

        Parameters
        ----------
        prompts : list[str]
            List of text prompts.
        layer_indices : list[int]
            Layer indices to extract from (positive integers).
        **kwargs
            Backend-specific parameters (e.g., remote for NnsightBackend).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - activations: Shape (batch, seq_len, hidden_dim * num_layers)
            - attention_mask: Shape (batch, seq_len)
        """

    @abstractmethod
    def extract_batch_with_logits(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract activations AND logits for a batch of prompts.

        Parameters
        ----------
        prompts : list[str]
            List of text prompts.
        layer_indices : list[int]
            Layer indices to extract from (positive integers).
        **kwargs
            Backend-specific parameters.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - activations: Shape (batch, seq_len, hidden_dim * num_layers)
            - attention_mask: Shape (batch, seq_len)
            - logits: Shape (batch, seq_len, vocab_size)
        """

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the model's tokenizer."""

    @property
    @abstractmethod
    def model(self):
        """Get the underlying model object.

        The return type depends on the backend:
        - NnsightBackend: nnsight.LanguageModel
        - LocalBackend: transformers.PreTrainedModel
        """


class NnsightBackend(ExtractionBackend):
    """Backend using nnsight for activation extraction.

    Wraps the existing nnsight-based extraction code. Supports both
    local and remote (NDIF) execution.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification.
    remote : bool
        If True, use remote execution via NDIF. No model weights
        are downloaded locally.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        remote: bool = False,
    ):
        super().__init__(model_name, device)
        self.remote = remote
        self._model = None

    @property
    def model(self):
        """Get the nnsight LanguageModel, loading if necessary."""
        if self._model is None:
            from .extraction import get_cached_model

            self._model = get_cached_model(
                self.model_name, self.device, remote=self.remote
            )
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.model.tokenizer

    def extract_batch(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .extraction import _extract_batch

        remote = kwargs.get("remote", self.remote)
        return _extract_batch(self.model, prompts, layer_indices, remote=remote)

    def extract_batch_with_logits(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from .extraction import _extract_batch_with_logits

        remote = kwargs.get("remote", self.remote)
        return _extract_batch_with_logits(
            self.model, prompts, layer_indices, remote=remote
        )


# Global cache for locally-loaded HuggingFace models
# Key: (model_name, device), Value: (model, tokenizer)
_LOCAL_MODEL_CACHE: dict[tuple, tuple] = {}


def _get_decoder_layers(model) -> list:
    """Get the list of decoder/transformer layers from a model.

    Tries common attribute paths used by different model architectures.

    Parameters
    ----------
    model : PreTrainedModel
        A HuggingFace transformers model.

    Returns
    -------
    list
        The list of decoder layer modules.

    Raises
    ------
    ValueError
        If the model architecture is not recognized.
    """
    # Common attribute paths for decoder layers
    # (Llama, Mistral, Phi, Qwen, BitNet, Gemma)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # GPT-2, GPT-J
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # GPT-NeoX, Pythia
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    # Falcon
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # MPT
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return list(model.transformer.blocks)
    # OPT
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return list(model.model.decoder.layers)

    raise ValueError(
        f"Could not find decoder layers for model architecture: "
        f"{type(model).__name__}. Supported architectures include "
        f"Llama, Mistral, Phi, GPT-2, GPT-NeoX, Falcon, MPT, OPT, "
        f"and other models using standard layer attribute paths."
    )


def _get_local_model(
    model_name: str, device: str, dtype: torch.dtype = torch.float32
):
    """Load a HuggingFace model locally, with caching.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification.
    dtype : torch.dtype
        Model dtype (e.g., torch.float32, torch.bfloat16).

    Returns
    -------
    tuple
        (model, tokenizer)
    """
    cache_key = (model_name, device, dtype)
    if cache_key not in _LOCAL_MODEL_CACHE:
        from lmprobe._device_utils import check_cuda_compatibility

        check_cuda_compatibility(device)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)
        if getattr(config, "quantization_config", None) is not None:
            if config.quantization_config.get("linear_class") == "autobitlinear":
                config.quantization_config["linear_class"] = "bitlinear"

        if device == "auto":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                device_map="auto",
                torch_dtype=dtype,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=dtype,
            )
            model = model.to(device)
        model.eval()
        _LOCAL_MODEL_CACHE[cache_key] = (model, tokenizer)

    return _LOCAL_MODEL_CACHE[cache_key]


def clear_local_model_cache() -> None:
    """Clear the local model cache to free memory."""
    global _LOCAL_MODEL_CACHE
    _LOCAL_MODEL_CACHE.clear()


class LocalBackend(ExtractionBackend):
    """Backend using HuggingFace transformers directly.

    Uses AutoModelForCausalLM and register_forward_hook for activation
    extraction. Does not require nnsight. Local execution only.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification.
    dtype : torch.dtype
        Model dtype (e.g., torch.float32, torch.bfloat16).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(model_name, device)
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Load model and tokenizer."""
        if self._model is None:
            model, tokenizer = _get_local_model(
                self.model_name, self.device, self.dtype
            )
            self._model = model
            self._tokenizer = tokenizer

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        self._load()
        return self._tokenizer

    def extract_batch(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        tokenizer = self.tokenizer
        decoder_layers = _get_decoder_layers(model)

        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )

        # Move inputs to model device
        device = next(model.parameters()).device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        # Set up hooks to capture activations
        captured = {}
        hooks = []

        for layer_idx in layer_indices:
            layer_module = decoder_layers[layer_idx]

            def make_hook(idx):
                def hook_fn(module, input, output):
                    # output is typically a tuple: (hidden_states, ...)
                    if isinstance(output, tuple):
                        captured[idx] = output[0].detach()
                    else:
                        captured[idx] = output.detach()
                return hook_fn

            h = layer_module.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for h in hooks:
                h.remove()

        # Collect and concatenate activations
        activation_tensors = [captured[idx].cpu() for idx in layer_indices]
        combined = torch.cat(activation_tensors, dim=-1)

        return combined, tokenized["attention_mask"]

    def extract_batch_with_logits(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.model
        tokenizer = self.tokenizer
        decoder_layers = _get_decoder_layers(model)

        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )

        device = next(model.parameters()).device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        captured = {}
        hooks = []

        for layer_idx in layer_indices:
            layer_module = decoder_layers[layer_idx]

            def make_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        captured[idx] = output[0].detach()
                    else:
                        captured[idx] = output.detach()
                return hook_fn

            h = layer_module.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for h in hooks:
                h.remove()

        activation_tensors = [captured[idx].cpu() for idx in layer_indices]
        combined = torch.cat(activation_tensors, dim=-1)
        logits = outputs.logits.detach().cpu()

        return combined, tokenized["attention_mask"], logits


def resolve_backend(
    backend: str,
    model_name: str,
    device: str = "auto",
    remote: bool = False,
    dtype: torch.dtype | None = None,
) -> ExtractionBackend:
    """Create an ExtractionBackend from a string identifier.

    Parameters
    ----------
    backend : str
        Backend identifier: "nnsight" or "local".
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification.
    remote : bool
        Whether to use remote execution (only valid for nnsight backend).
    dtype : torch.dtype or None
        Model dtype for local backend (e.g., torch.float32, torch.bfloat16).
        Defaults to torch.float32 if None. Ignored for nnsight backend.

    Returns
    -------
    ExtractionBackend
        The instantiated backend.

    Raises
    ------
    ValueError
        If backend string is not recognized, or if incompatible options
        are specified (e.g., local + remote).
    """
    if backend == "nnsight":
        try:
            import nnsight  # noqa: F401
        except ImportError:
            raise ImportError(
                "nnsight is required for backend='nnsight'. "
                "Install with: pip install lmprobe[nnsight]"
            ) from None
        return NnsightBackend(model_name, device, remote=remote)
    elif backend == "local":
        if remote:
            raise ValueError(
                "backend='local' does not support remote=True. "
                "Use backend='nnsight' for remote execution."
            )
        local_dtype = dtype if dtype is not None else torch.float32
        return LocalBackend(model_name, device, dtype=local_dtype)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. Available backends: 'nnsight', 'local'."
        )
