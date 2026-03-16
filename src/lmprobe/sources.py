"""Pluggable activation extraction backends.

This module defines the ActivationSource interface and provides two
concrete implementations:

- NDIFSource: Extracts activations via nnsight (supports remote NDIF execution)
- LocalSource: Extracts activations via vanilla HuggingFace + PyTorch hooks

Both backends produce identical output shapes, so everything downstream
(caching, pooling, probe training) works the same regardless of source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class ActivationSource(ABC):
    """Abstract base class for activation extraction backends.

    All sources must implement extract_batch() and provide a tokenizer.
    The extract_batch_with_logits() method has a default implementation
    that raises NotImplementedError; subclasses can override it to
    support unified extraction (activations + logits in one forward pass).
    """

    @abstractmethod
    def extract_batch(
        self,
        prompts: list[str],
        layer_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract activations for a batch of prompts.

        Parameters
        ----------
        prompts : list[str]
            Text prompts (a single batch, not the full dataset).
        layer_indices : list[int]
            Positive layer indices to extract from.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - activations: Shape (batch, seq_len, hidden_dim * num_layers)
              Layers concatenated along the hidden dimension.
            - attention_mask: Shape (batch, seq_len)
        """
        ...

    def extract_batch_with_logits(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Extract activations AND logits in a single forward pass.

        Parameters
        ----------
        prompts : list[str]
            Text prompts.
        layer_indices : list[int]
            Positive layer indices to extract from.
        **kwargs
            Additional parameters (e.g., logit_top_k for server-side top-k).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
            - activations: Shape (batch, seq_len, hidden_dim * num_layers)
            - attention_mask: Shape (batch, seq_len)
            - logits: Shape (batch, seq_len, vocab_size) or (batch, seq_len, K)
            - logits_indices: None or (batch, seq_len, K)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support extract_batch_with_logits(). "
            "Use NDIFSource or LocalSource for unified extraction."
        )

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The tokenizer associated with this source's model."""
        ...


class NDIFSource(ActivationSource):
    """Activation extraction via nnsight (local or remote NDIF execution).

    This wraps the existing nnsight-based extraction code, providing the
    same functionality through the ActivationSource interface.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification ("auto", "cpu", "cuda:0", etc.).
    remote : bool
        If True, use remote NDIF execution. No model weights are
        downloaded locally.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        remote: bool = False,
    ):
        self.model_name = model_name
        self.device = device
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
    def tokenizer(self):
        return self.model.tokenizer

    def extract_batch(
        self,
        prompts: list[str],
        layer_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .extraction import _extract_batch

        return _extract_batch(
            self.model, prompts, layer_indices, remote=self.remote
        )

    def extract_batch_with_logits(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        from .extraction import _extract_batch_with_logits

        return _extract_batch_with_logits(
            self.model, prompts, layer_indices, remote=self.remote,
            logit_top_k=kwargs.get("logit_top_k"),
        )


# Global cache for locally-loaded HuggingFace models
# Key: (model_name, device), Value: (model, tokenizer)
_LOCAL_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


def _get_local_model(model_name: str, device: str):
    """Load a HuggingFace model locally, with caching.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification.

    Returns
    -------
    tuple
        (model, tokenizer)
    """
    cache_key = (model_name, device)
    if cache_key not in _LOCAL_MODEL_CACHE:
        from lmprobe._device_utils import check_cuda_compatibility

        check_cuda_compatibility(device)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if device == "auto":
            device_map = "auto"
        elif device == "cpu":
            device_map = {"": "cpu"}
        else:
            device_map = {"": device}

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float32,
        )
        model.eval()

        _LOCAL_MODEL_CACHE[cache_key] = (model, tokenizer)

    return _LOCAL_MODEL_CACHE[cache_key]


def clear_local_model_cache() -> None:
    """Clear the global cache of locally-loaded HuggingFace models."""
    global _LOCAL_MODEL_CACHE
    _LOCAL_MODEL_CACHE.clear()


def _get_decoder_layers(model):
    """Get the list of decoder layers from a HuggingFace model.

    Supports common architectures: Llama, Mistral, GPT-2, GPT-Neo,
    GPT-NeoX, Phi, Qwen, BitNet, etc.

    Parameters
    ----------
    model : PreTrainedModel
        A HuggingFace causal LM.

    Returns
    -------
    torch.nn.ModuleList
        The decoder layers.

    Raises
    ------
    ValueError
        If the model architecture is not recognized.
    """
    # Most models: model.model.layers (Llama, Mistral, Phi, Qwen, BitNet, etc.)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # GPT-2: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    # GPT-NeoX: model.gpt_neox.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers

    raise ValueError(
        f"Cannot find decoder layers in model architecture: {type(model).__name__}. "
        "Please report this as a bug with the model ID."
    )


class LocalSource(ActivationSource):
    """Activation extraction via vanilla HuggingFace + PyTorch hooks.

    Uses register_forward_hook on decoder layers to capture residual
    stream activations during inference. No nnsight dependency required.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification ("auto", "cpu", "cuda:0", etc.).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load the model and tokenizer if not already loaded."""
        if self._model is None:
            self._model, self._tokenizer = _get_local_model(
                self.model_name, self.device
            )

    @property
    def model(self):
        """Get the HuggingFace model."""
        self._load_model()
        return self._model

    @property
    def tokenizer(self):
        self._load_model()
        return self._tokenizer

    def extract_batch(
        self,
        prompts: list[str],
        layer_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        tokenizer = self.tokenizer

        # Tokenize
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )

        # Move inputs to model's device
        device = next(model.parameters()).device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        # Get decoder layers
        decoder_layers = _get_decoder_layers(model)

        # Set up hooks to capture activations
        captured = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # Decoder layer output is typically a tuple:
                # (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                captured[layer_idx] = hidden.detach()

            return hook_fn

        # Register hooks
        for idx in layer_indices:
            hook = decoder_layers[idx].register_forward_hook(make_hook(idx))
            hooks.append(hook)

        try:
            # Forward pass
            with torch.no_grad(), torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()

        # Concatenate layer activations along hidden dim
        activation_tensors = [captured[idx].cpu() for idx in layer_indices]
        combined = torch.cat(activation_tensors, dim=-1)

        return combined, tokenized["attention_mask"]

    def extract_batch_with_logits(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        model = self.model
        tokenizer = self.tokenizer

        # Tokenize
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )

        # Move inputs to model's device
        device = next(model.parameters()).device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        # Get decoder layers
        decoder_layers = _get_decoder_layers(model)

        # Set up hooks to capture activations
        captured = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                captured[layer_idx] = hidden.detach()

            return hook_fn

        for idx in layer_indices:
            hook = decoder_layers[idx].register_forward_hook(make_hook(idx))
            hooks.append(hook)

        try:
            with torch.no_grad(), torch.inference_mode():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
        finally:
            for hook in hooks:
                hook.remove()

        # Concatenate layer activations
        activation_tensors = [captured[idx].cpu() for idx in layer_indices]
        combined = torch.cat(activation_tensors, dim=-1)

        # Get logits from model output
        logits = outputs.logits.detach().cpu()

        return combined, tokenized["attention_mask"], logits, None


def resolve_source(
    model_name: str,
    device: str = "auto",
    backend: str = "nnsight",
    remote: bool = False,
) -> ActivationSource:
    """Create an ActivationSource from configuration parameters.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification.
    backend : str
        Backend to use: "nnsight" or "local".
    remote : bool
        Whether to use remote NDIF execution (only for nnsight backend).

    Returns
    -------
    ActivationSource
        The configured source.

    Raises
    ------
    ValueError
        If backend is not recognized.
    """
    if backend == "nnsight":
        return NDIFSource(model_name, device, remote=remote)
    elif backend == "local":
        if remote:
            raise ValueError(
                "backend='local' does not support remote=True. "
                "Use backend='nnsight' for remote NDIF execution."
            )
        return LocalSource(model_name, device)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. Available: 'nnsight', 'local'."
        )
