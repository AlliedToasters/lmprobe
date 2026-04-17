"""Pluggable extraction backends for lmprobe.

This module defines the ExtractionBackend ABC and provides implementations:
- NnsightBackend: Uses nnsight for model loading and activation extraction
  (supports both local and remote/NDIF execution)
- LocalBackend: Uses HuggingFace transformers directly with register_forward_hook
  (local-only, no nnsight dependency for extraction)
- ChunkedLocalBackend: Loads full model on CPU, chunks layers through GPU
- DiskOffloadBackend: Loads layer weights from safetensors one layer at a time
  for models exceeding CPU RAM (e.g. DeepSeek-V3 671B FP8)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .activation_types import ExtractedBatch, ExtractionSpec


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
        **kwargs: Any,
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
        **kwargs: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Extract activations AND logits for a batch of prompts.

        Parameters
        ----------
        prompts : list[str]
            List of text prompts.
        layer_indices : list[int]
            Layer indices to extract from (positive integers).
            Pass an empty list to extract logits only (no activations).
        **kwargs
            Backend-specific parameters. Notable:
            - logit_top_k (int | None): When set and remote=True (nnsight
              backend), perform server-side top-k on logits. The fourth
              return element will contain the top-k indices.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]
            - activations: Shape (batch, seq_len, hidden_dim * num_layers),
              or None when layer_indices is empty
            - attention_mask: Shape (batch, seq_len)
            - logits: Shape (batch, seq_len, vocab_size) or (batch, seq_len, K)
            - logits_indices: None or (batch, seq_len, K) int64 indices
        """

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the model's tokenizer."""

    @property
    @abstractmethod
    def model(self) -> Any:
        """Get the underlying model object.

        The return type depends on the backend:
        - NnsightBackend: nnsight.LanguageModel
        - LocalBackend: transformers.PreTrainedModel
        """

    def extract_batch_extended(
        self,
        prompts: list[str],
        spec: ExtractionSpec,
        **kwargs: Any,
    ) -> ExtractedBatch:
        """Extract multiple activation types in a single forward pass.

        This is the extensible extraction method. New activation types
        (router logits, attention patterns, etc.) are added by extending
        :class:`~lmprobe.activation_types.ExtractionSpec` and
        :class:`~lmprobe.activation_types.ExtractedBatch`.

        Parameters
        ----------
        prompts : list[str]
            List of text prompts.
        spec : ExtractionSpec
            Specification of what to extract.
        **kwargs
            Backend-specific parameters.

        Returns
        -------
        ExtractedBatch
            Structured result containing all requested activations.

        Raises
        ------
        NotImplementedError
            If the backend does not support extended extraction.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support extract_batch_extended. "
            f"Use extract_batch() or extract_batch_with_logits() instead."
        )


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
        self._remote_model = None

    @property
    def model(self) -> Any:
        """Get the nnsight LanguageModel, loading if necessary."""
        if self._model is None:
            from .extraction import get_cached_model

            self._model = get_cached_model(
                self.model_name, self.device, remote=self.remote
            )
        return self._model

    def _get_model_for_remote(self) -> Any:
        """Get a lightweight model stub for remote execution.

        When the backend was created with remote=False but a call-time
        remote=True is used, we need a model loaded with dispatch=False
        (no weights) instead of the full local model.
        """
        if self.remote:
            # Backend was created for remote use — main model is already lightweight
            return self.model
        # Backend was created for local use — need a separate remote stub
        if self._remote_model is None:
            from .extraction import get_cached_model

            self._remote_model = get_cached_model(
                self.model_name, self.device, remote=True
            )
        return self._remote_model

    @property
    def tokenizer(self) -> Any:
        return self.model.tokenizer

    def extract_batch(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .extraction import _extract_batch

        remote = kwargs.get("remote", self.remote)
        model = self._get_model_for_remote() if remote else self.model
        return _extract_batch(model, prompts, layer_indices, remote=remote)

    def extract_batch_with_logits(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        from .extraction import _extract_batch_with_logits

        remote = kwargs.get("remote", self.remote)
        logit_top_k = kwargs.get("logit_top_k")
        model = self._get_model_for_remote() if remote else self.model
        return _extract_batch_with_logits(
            model, prompts, layer_indices, remote=remote,
            logit_top_k=logit_top_k,
        )

    def extract_batch_extended(
        self,
        prompts: list[str],
        spec: ExtractionSpec,
        **kwargs: Any,
    ) -> ExtractedBatch:
        from .extraction import _extract_batch_extended

        remote = kwargs.get("remote", self.remote)
        model = self._get_model_for_remote() if remote else self.model
        return _extract_batch_extended(model, prompts, spec, remote=remote)


# Global cache for locally-loaded HuggingFace models
# Key: (model_name, device, dtype), Value: (model, tokenizer)
_LOCAL_MODEL_CACHE: dict[tuple[Any, ...], tuple[Any, PreTrainedTokenizerBase]] = {}


def _get_decoder_layers(model: Any) -> list[Any]:
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
    # Multimodal models (e.g. Mistral3ForConditionalGeneration) nest the
    # text decoder inside model.language_model
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return list(model.model.language_model.layers)
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


def _get_attn_submodule(layer: Any) -> Any:
    """Get the self-attention submodule from a transformer layer."""
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError(
        f"Cannot find attention submodule in {type(layer).__name__}. "
        f"Expected one of: self_attn, attention, attn"
    )


def _get_mlp_submodule(layer: Any) -> Any:
    """Get the MLP/feed-forward submodule from a transformer layer."""
    for name in ("mlp", "feed_forward", "ffn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError(
        f"Cannot find MLP submodule in {type(layer).__name__}. "
        f"Expected one of: mlp, feed_forward, ffn"
    )


def _get_router_modules(
    model: Any,
    layer_indices: list[int],
    router_module_template: str,
) -> dict[int, Any]:
    """Get MoE router gate modules for specified layers.

    Uses the ``router_module_template`` from :class:`MoEInfo` to resolve
    the router gate submodule for each layer. The template is a dot-path
    like ``"model.model.layers.{layer}.block_sparse_moe.gate"`` where the
    leading ``"model."`` prefix corresponds to the top-level HF model
    object itself.

    Parameters
    ----------
    model : PreTrainedModel
        A HuggingFace transformers model.
    layer_indices : list[int]
        Layer indices to get router modules for.
    router_module_template : str
        Dot-path template with ``{layer}`` placeholder.

    Returns
    -------
    dict[int, Any]
        Mapping from layer index to router gate module.
        Layers without a router module (e.g., dense layers in a mixed
        architecture) are silently skipped.
    """
    result: dict[int, Any] = {}
    for layer_idx in layer_indices:
        path = router_module_template.format(layer=layer_idx)
        # Strip leading "model." — the template is written for nnsight's
        # LanguageModel where the HF model is at .model, but here we
        # already have the HF model directly.
        if path.startswith("model."):
            path = path[len("model."):]
        parts = path.split(".")
        try:
            obj: Any = model
            for part in parts:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            result[layer_idx] = obj
        except (AttributeError, IndexError, TypeError):
            # Layer doesn't have this router module (e.g., dense layer)
            pass
    return result


def _get_local_model(
    model_name: str, device: str, dtype: torch.dtype = torch.float32
) -> tuple[Any, PreTrainedTokenizerBase]:
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
            model: Any = AutoModelForCausalLM.from_pretrained(
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
        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    def _load(self) -> None:
        """Load model and tokenizer."""
        if self._model is None:
            model, tokenizer = _get_local_model(
                self.model_name, self.device, self.dtype
            )
            self._model = model
            self._tokenizer = tokenizer

    @property
    def model(self) -> Any:
        self._load()
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        self._load()
        assert self._tokenizer is not None
        return self._tokenizer

    def extract_batch(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs: Any,
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
        captured: dict[int, torch.Tensor] = {}
        hooks = []

        for layer_idx in layer_indices:
            layer_module = decoder_layers[layer_idx]

            def make_hook(idx: int) -> Any:
                def hook_fn(module: Any, input: Any, output: Any) -> None:
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
        **kwargs: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        model = self.model
        tokenizer = self.tokenizer

        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )

        device = next(model.parameters()).device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        captured: dict[int, torch.Tensor] = {}
        hooks = []

        if layer_indices:
            decoder_layers = _get_decoder_layers(model)
            for layer_idx in layer_indices:
                layer_module = decoder_layers[layer_idx]

                def make_hook(idx: int) -> Any:
                    def hook_fn(module: Any, input: Any, output: Any) -> None:
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

        if layer_indices:
            activation_tensors = [captured[idx].cpu() for idx in layer_indices]
            combined = torch.cat(activation_tensors, dim=-1)
        else:
            combined = None
        logits = outputs.logits.detach().cpu()

        return combined, tokenized["attention_mask"], logits, None

    def extract_batch_extended(
        self,
        prompts: list[str],
        spec: ExtractionSpec,
        **kwargs: Any,
    ) -> ExtractedBatch:
        from .activation_types import ExtractedBatch

        model = self.model
        tokenizer = self.tokenizer

        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )

        device = next(model.parameters()).device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        # --- Set up hooks ---
        captured_hidden: dict[int, torch.Tensor] = {}
        captured_router: dict[int, torch.Tensor] = {}
        hooks: list[Any] = []

        # Hidden state hooks
        if spec.hidden_layers:
            decoder_layers = _get_decoder_layers(model)
            for layer_idx in spec.hidden_layers:
                layer_module = decoder_layers[layer_idx]

                def make_hidden_hook(idx: int) -> Any:
                    def hook_fn(_module: Any, _input: Any, output: Any) -> None:
                        if isinstance(output, tuple):
                            captured_hidden[idx] = output[0].detach()
                        else:
                            captured_hidden[idx] = output.detach()
                    return hook_fn

                hooks.append(
                    layer_module.register_forward_hook(make_hidden_hook(layer_idx))
                )

        # Router logit hooks
        if spec.router_layers and spec.router_module_template:
            router_modules = _get_router_modules(
                model, spec.router_layers, spec.router_module_template
            )
            for layer_idx, router_module in router_modules.items():
                if spec.router_hook_strategy == "input_gate":
                    # Hook the MoE module's forward and compute gate logits
                    # from input. Needed for DeepSeek which calls F.linear()
                    # on gate.weight directly, bypassing the gate module's
                    # __call__.
                    def make_input_gate_hook(idx: int) -> Any:
                        def hook_fn(module: Any, args: Any, output: Any) -> None:
                            hs = args[0] if isinstance(args, tuple) else args
                            gate_weight = module.gate.weight
                            logits = torch.nn.functional.linear(
                                hs.to(gate_weight.dtype), gate_weight,
                            )
                            captured_router[idx] = logits.detach()
                        return hook_fn

                    hooks.append(
                        router_module.register_forward_hook(
                            make_input_gate_hook(layer_idx)
                        )
                    )
                else:
                    # Default: hook the gate module's forward output directly
                    def make_router_hook(idx: int) -> Any:
                        def hook_fn(_module: Any, _input: Any, output: Any) -> None:
                            if isinstance(output, tuple):
                                captured_router[idx] = output[0].detach()
                            else:
                                captured_router[idx] = output.detach()
                        return hook_fn

                    hooks.append(
                        router_module.register_forward_hook(
                            make_router_hook(layer_idx)
                        )
                    )

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # --- Forward pass ---
        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for h in hooks:
                h.remove()

        # --- Collect results ---
        activations: torch.Tensor | None = None
        if spec.hidden_layers and captured_hidden:
            activation_tensors = [
                captured_hidden[idx].cpu() for idx in spec.hidden_layers
            ]
            activations = torch.cat(activation_tensors, dim=-1)

        logits: torch.Tensor | None = None
        if spec.include_logits:
            logits = outputs.logits.detach().cpu()

        router_logits: dict[int, torch.Tensor] | None = None
        if captured_router:
            router_logits = {}
            for idx, t in captured_router.items():
                t_cpu = t.cpu()
                # Router modules (e.g. OLMoE) may flatten batch and seq dims
                # to (batch*seq_len, n_experts). Reshape to (batch, seq_len, n_experts).
                if t_cpu.dim() == 2 and t_cpu.shape[0] == batch_size * seq_len:
                    t_cpu = t_cpu.view(batch_size, seq_len, -1)
                router_logits[idx] = t_cpu

        # Also provide per-layer hidden states for callers that need them
        hidden_per_layer: dict[int, torch.Tensor] | None = None
        if spec.hidden_layers and captured_hidden:
            hidden_per_layer = {
                idx: captured_hidden[idx].cpu() for idx in spec.hidden_layers
            }

        return ExtractedBatch(
            activations=activations,
            attention_mask=tokenized["attention_mask"],
            logits=logits,
            logits_indices=None,
            router_logits=router_logits,
            hidden_per_layer=hidden_per_layer,
        )


# ---------------------------------------------------------------------------
# Helpers for ChunkedLocalBackend
# ---------------------------------------------------------------------------


def _get_embedding_module(model: Any) -> Any:
    """Get the token embedding module from a model.

    Parameters
    ----------
    model : PreTrainedModel
        A HuggingFace transformers model.

    Returns
    -------
    nn.Module
        The token embedding module.
    """
    # Llama, Mistral, Phi, Qwen, Gemma
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    # Multimodal models (e.g. Mistral3ForConditionalGeneration)
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "embed_tokens")
    ):
        return model.model.language_model.embed_tokens
    # GPT-2, GPT-J
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    # GPT-NeoX, Pythia
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "embed_in"):
        return model.gpt_neox.embed_in
    # OPT
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "embed_tokens")
    ):
        return model.model.decoder.embed_tokens

    raise ValueError(
        f"Could not find embedding module for model architecture: "
        f"{type(model).__name__}."
    )


def _get_final_norm(model: Any) -> Any:
    """Get the final layer norm module from a model.

    Parameters
    ----------
    model : PreTrainedModel
        A HuggingFace transformers model.

    Returns
    -------
    nn.Module
        The final normalization module.
    """
    # Llama, Mistral, Phi, Qwen, Gemma
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    # Multimodal models (e.g. Mistral3ForConditionalGeneration)
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "norm")
    ):
        return model.model.language_model.norm
    # GPT-2
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    # GPT-NeoX, Pythia
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        return model.gpt_neox.final_layer_norm
    # OPT
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "final_layer_norm")
    ):
        return model.model.decoder.final_layer_norm

    raise ValueError(
        f"Could not find final norm module for model architecture: "
        f"{type(model).__name__}."
    )


def _get_lm_head(model: Any) -> Any:
    """Get the language model head from a model.

    Parameters
    ----------
    model : PreTrainedModel
        A HuggingFace transformers model.

    Returns
    -------
    nn.Module
        The lm_head module.
    """
    if hasattr(model, "lm_head"):
        return model.lm_head
    # GPT-NeoX uses embed_out
    if hasattr(model, "embed_out"):
        return model.embed_out

    raise ValueError(
        f"Could not find lm_head for model architecture: "
        f"{type(model).__name__}."
    )


def _make_causal_mask(
    attention_mask_2d: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a 4D causal attention mask from a 2D padding mask.

    Individual decoder layers expect a ``(batch, 1, seq, seq)`` additive
    mask where attended positions are ``0`` and masked positions are a
    large negative value.

    Parameters
    ----------
    attention_mask_2d : torch.Tensor
        Padding mask of shape ``(batch, seq_len)`` with 1 for real tokens
        and 0 for padding.
    dtype : torch.dtype
        The dtype for the mask (should match model dtype).

    Returns
    -------
    torch.Tensor
        4D causal mask of shape ``(batch, 1, seq_len, seq_len)``.
    """
    batch_size, seq_len = attention_mask_2d.shape
    device = attention_mask_2d.device
    min_val = torch.finfo(dtype).min

    # Causal mask: lower triangular (1 = attend, 0 = mask)
    causal_bool = torch.tril(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
    )

    # Padding mask: (batch, seq) → (batch, 1, 1, seq)
    padding_bool = attention_mask_2d[:, None, None, :].bool()

    # Combine: attend only if both causal AND not padding
    combined_bool = causal_bool.unsqueeze(0).unsqueeze(0) & padding_bool
    mask = torch.where(combined_bool, torch.tensor(0.0, dtype=dtype, device=device), min_val)
    return mask


def _estimate_chunk_size(
    model_name: str,
    device: str,
    dtype: torch.dtype,
) -> int:
    """Estimate how many transformer layers fit in available VRAM.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification.
    dtype : torch.dtype
        Model dtype.

    Returns
    -------
    int
        Estimated number of layers per chunk, clamped to ``[1, num_layers]``.
    """
    from .extraction import get_num_layers_from_config

    num_layers = get_num_layers_from_config(model_name)

    if device == "cpu" or not torch.cuda.is_available():
        return num_layers

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    # Multimodal models nest text params under text_config
    text_cfg = getattr(config, "text_config", config)
    hidden_size = text_cfg.hidden_size
    intermediate_size = getattr(text_cfg, "intermediate_size", hidden_size * 4)

    bytes_per_param = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    # Approximate params per layer: 4 attention matrices + 3 FFN matrices
    params_per_layer = 4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size
    layer_bytes = params_per_layer * bytes_per_param

    try:
        free_vram, _total = torch.cuda.mem_get_info(device)
    except Exception:
        return num_layers

    # Reserve 30% for activations and overhead
    available = free_vram * 0.7
    chunk_size = max(1, int(available / layer_bytes))
    return min(chunk_size, num_layers)


# ---------------------------------------------------------------------------
# ChunkedLocalBackend
# ---------------------------------------------------------------------------


class ChunkedLocalBackend(ExtractionBackend):
    """Layer-chunked extraction for models that don't fit in GPU memory.

    Processes the model in stages: embedding, then chunks of transformer
    layers, then optionally lm_head. Between chunks, intermediate hidden
    states are held in CPU memory. Only the current chunk's weights are
    on the compute device at any time.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification (e.g., ``"cuda:0"`` or ``"cpu"``).
    dtype : torch.dtype
        Model dtype — ``torch.bfloat16`` recommended for large models.
    chunk_size : int or ``"auto"``
        Number of layers per chunk. ``"auto"`` estimates from available VRAM.
    attn_implementation : str
        Attention implementation passed to ``from_pretrained`` (``"sdpa"``,
        ``"eager"``, ``"flash_attention_2"``). Default ``"sdpa"`` — avoids
        materializing the full ``[B, H, T, T]`` attention matrix, which is
        the usual OOM culprit on long sequences. Use ``"eager"`` only if a
        custom signal hook needs the materialized attention weights.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        chunk_size: int | str = "auto",
        attn_implementation: str = "sdpa",
    ):
        super().__init__(model_name, device)
        self.dtype = dtype
        self._chunk_size = chunk_size
        self._attn_implementation = attn_implementation
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._config: Any = None
        self._num_layers: int | None = None
        self._resolved_chunk_size: int | None = None

    @property
    def model(self) -> Any:
        raise RuntimeError(
            "ChunkedLocalBackend does not keep the full model in memory. "
            "Use self.device for device info."
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def _get_config(self) -> Any:
        if self._config is None:
            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(self.model_name)
        return self._config

    def _resolve_chunk_size(self) -> int:
        if self._resolved_chunk_size is not None:
            return self._resolved_chunk_size
        if isinstance(self._chunk_size, int):
            self._resolved_chunk_size = self._chunk_size
        else:
            self._resolved_chunk_size = _estimate_chunk_size(
                self.model_name, self.device, self.dtype,
            )
        return self._resolved_chunk_size

    def _get_num_layers(self) -> int:
        if self._num_layers is None:
            from .extraction import get_num_layers_from_config

            self._num_layers = get_num_layers_from_config(self.model_name)
        return self._num_layers

    def _load_full_model_cpu(self) -> Any:
        """Load the full model on CPU.

        For the chunked backend, the model is always loaded fully on CPU.
        The chunking benefit comes from only moving subset of layers to
        GPU at a time — for outsized models, CPU RAM is sufficient to
        hold the full weights while GPU VRAM is not.

        Uses ``self._attn_implementation`` (default ``"sdpa"``) for the
        attention kernel. ``"sdpa"`` avoids materializing the full
        ``[B, H, T, T]`` softmax tensor, eliminating the usual OOM on
        long sequences. Pass ``attn_implementation="eager"`` at construction
        time if a custom signal hook needs the materialized attention weights.
        """
        if not hasattr(self, "_full_model"):
            from transformers import AutoConfig, AutoModelForCausalLM

            config = AutoConfig.from_pretrained(self.model_name)

            # Multimodal / conditional-generation models (e.g. Mistral3)
            # are not registered under AutoModelForCausalLM. Detect them
            # via the presence of text_config and load with the right class.
            if hasattr(config, "text_config"):
                from transformers import AutoModelForImageTextToText

                self._full_model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    attn_implementation=self._attn_implementation,
                )
            else:
                self._full_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    attn_implementation=self._attn_implementation,
                )
            self._full_model.eval()
        return self._full_model

    def _chunked_forward(
        self,
        prompts: list[str],
        layer_indices: list[int],
        include_logits: bool = False,
        router_layer_indices: list[int] | None = None,
        router_module_template: str | None = None,
        router_hook_strategy: str = "output",
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        dict[int, torch.Tensor] | None,
    ]:
        """Run a chunked forward pass through the model.

        The model is loaded fully on CPU with eager attention. For each
        chunk of layers, those layers are moved to the compute device,
        the forward pass is run, and the layers are moved back to CPU.
        This keeps GPU memory bounded to one chunk of layers at a time.

        Returns
        -------
        tuple
            (activations, attention_mask, logits, router_logits)
        """
        import gc

        from .extraction import get_num_layers_from_config

        tokenized = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask_2d = tokenized["attention_mask"]

        num_layers = get_num_layers_from_config(self.model_name)
        chunk_size = self._resolve_chunk_size()
        target_set = set(layer_indices)
        router_target_set = set(router_layer_indices or [])

        model = self._load_full_model_cpu()
        device = self.device

        # --- Phase 1: Embedding ---
        embed = _get_embedding_module(model)
        embed.to(device)
        with torch.no_grad():
            hidden_states = embed(input_ids.to(device)).cpu()
        embed.to("cpu")

        # Build position_ids, cache_position, and causal mask
        seq_len = input_ids.shape[1]
        position_ids = attention_mask_2d.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask_2d == 0, 1)
        cache_position = torch.arange(seq_len)
        causal_mask = _make_causal_mask(attention_mask_2d, self.dtype)

        # Compute rotary position embeddings
        # Some architectures (Llama, Mistral) return (cos, sin) tuples;
        # DeepSeek-V2 returns a single complex freqs_cis tensor.
        # Gemma3 computes per-layer-type embeddings via a layer_type arg.
        position_embeddings: tuple[torch.Tensor, ...] | torch.Tensor | dict | None = None
        layer_types: list[str] | None = None
        rotary_name = self._find_rotary_embedding_name(model)
        if rotary_name is not None:
            rotary_mod = model
            for part in rotary_name.split("."):
                rotary_mod = getattr(rotary_mod, part)
            rotary_mod.to(device)

            # Detect Gemma3-style per-layer-type rotary embeddings
            text_cfg = getattr(
                getattr(model, "config", None), "text_config",
                getattr(model, "config", None),
            )
            layer_types_cfg = getattr(text_cfg, "layer_types", None)
            if layer_types_cfg is not None:
                layer_types = list(layer_types_cfg)
                position_embeddings = {}
                with torch.no_grad():
                    for lt in set(layer_types):
                        pe = rotary_mod(
                            hidden_states.to(device),
                            position_ids.to(device),
                            layer_type=lt,
                        )
                        position_embeddings[lt] = tuple(t.cpu() for t in pe)
            else:
                with torch.no_grad():
                    pe = rotary_mod(hidden_states.to(device), position_ids.to(device))
                    if isinstance(pe, tuple):
                        position_embeddings = tuple(t.cpu() for t in pe)
                    else:
                        position_embeddings = pe.cpu()
            rotary_mod.to("cpu")

        # --- Phase 2: Layer chunks ---
        captured: dict[int, torch.Tensor] = {}
        captured_router: dict[int, torch.Tensor] = {}
        decoder_layers = _get_decoder_layers(model)

        for chunk_start in range(0, num_layers, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_layers)

            # Move chunk layers to device
            for i in range(chunk_start, chunk_end):
                decoder_layers[i].to(device)

            hs = hidden_states.to(device)
            mask_dev = causal_mask.to(device)
            pos_dev = position_ids.to(device)

            # Pre-compute position embeddings on device
            pe_dev_map: dict | tuple | torch.Tensor | None = None
            if position_embeddings is not None:
                if isinstance(position_embeddings, dict):
                    # Gemma3-style: dict of {layer_type: (cos, sin)}
                    pe_dev_map = {
                        lt: tuple(t.to(device) for t in pe)
                        for lt, pe in position_embeddings.items()
                    }
                elif isinstance(position_embeddings, tuple):
                    pe_dev_map = tuple(t.to(device) for t in position_embeddings)
                else:
                    pe_dev_map = position_embeddings.to(device)

            with torch.no_grad():
                for layer_idx in range(chunk_start, chunk_end):
                    layer_module = decoder_layers[layer_idx]

                    # Router hook if requested
                    rh = None
                    if layer_idx in router_target_set and router_module_template:
                        router_hook_output: list[torch.Tensor] = []

                        router_path = router_module_template.format(layer=layer_idx)
                        # Strip leading "model." — the template is written
                        # for nnsight's LanguageModel where the HF model is
                        # at .model, but here we already have the HF model.
                        if router_path.startswith("model."):
                            router_path = router_path[len("model."):]
                        router_mod = model
                        for part in router_path.split("."):
                            if part.isdigit():
                                router_mod = router_mod[int(part)]
                            else:
                                router_mod = getattr(router_mod, part)

                        if router_hook_strategy == "input_gate":
                            # Hook the MoE module and compute gate logits
                            # from its input + gate.weight. Needed for
                            # DeepSeek which calls F.linear() directly.
                            def _input_gate_hook(
                                mod: Any, args: Any, out: Any,
                                _buf: list = router_hook_output,
                            ) -> None:
                                hs_in = args[0] if isinstance(args, tuple) else args
                                gate_w = mod.gate.weight
                                logits = torch.nn.functional.linear(
                                    hs_in.to(gate_w.dtype), gate_w,
                                )
                                _buf.append(logits.detach().cpu())

                            rh = router_mod.register_forward_hook(_input_gate_hook)
                        else:
                            def _output_hook(
                                _mod: Any, _inp: Any, out: Any,
                                _buf: list = router_hook_output,
                            ) -> None:
                                if isinstance(out, tuple):
                                    _buf.append(out[0].detach().cpu())
                                else:
                                    _buf.append(out.detach().cpu())

                            rh = router_mod.register_forward_hook(_output_hook)

                    layer_kwargs: dict[str, Any] = {
                        "attention_mask": mask_dev,
                        "position_ids": pos_dev,
                        "use_cache": False,
                        "cache_position": cache_position.to(device),
                    }
                    if pe_dev_map is not None:
                        if isinstance(pe_dev_map, dict) and layer_types is not None:
                            # Gemma3-style: select by layer type
                            lt = layer_types[layer_idx]
                            layer_kwargs["position_embeddings"] = pe_dev_map[lt]
                        else:
                            layer_kwargs["position_embeddings"] = pe_dev_map

                    output = layer_module(hs, **layer_kwargs)
                    hs = output[0] if isinstance(output, tuple) else output

                    if layer_idx in target_set:
                        captured[layer_idx] = hs.detach().cpu()

                    if rh is not None:
                        rh.remove()
                        if router_hook_output:
                            captured_router[layer_idx] = router_hook_output[0]

            hidden_states = hs.cpu()
            del hs, mask_dev, pos_dev, pe_dev_map

            # Move chunk back to CPU
            for i in range(chunk_start, chunk_end):
                decoder_layers[i].to("cpu")
            gc.collect()
            if torch.cuda.is_available() and device != "cpu":
                torch.cuda.empty_cache()

        # --- Phase 3: Logits (optional) ---
        logits_out: torch.Tensor | None = None
        if include_logits:
            final_norm = _get_final_norm(model)
            lm_head = _get_lm_head(model)
            final_norm.to(device)
            lm_head.to(device)

            with torch.no_grad():
                hs_dev = hidden_states.to(device)
                logits_out = lm_head(final_norm(hs_dev)).cpu()
                del hs_dev

            final_norm.to("cpu")
            lm_head.to("cpu")

        # Concatenate target layer activations
        activations = None
        if layer_indices:
            activation_tensors = [captured[idx] for idx in layer_indices]
            activations = torch.cat(activation_tensors, dim=-1)

        router_logits = captured_router if captured_router else None

        return activations, attention_mask_2d, logits_out, router_logits

    @staticmethod
    def _find_rotary_embedding_name(model: Any) -> str | None:
        """Return the dot-path name of the rotary embedding, or None."""
        # Llama, Mistral, Qwen, Gemma
        if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
            return "model.rotary_emb"
        # Multimodal models (e.g. Mistral3ForConditionalGeneration)
        if (
            hasattr(model, "model")
            and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "rotary_emb")
        ):
            return "model.language_model.rotary_emb"
        return None

    # --- Shared chunked-pass setup ---

    def _prepare_chunked_pass(
        self, prompts: list[str],
    ) -> dict[str, Any]:
        """Run embedding phase and compute position embeddings.

        Returns a dict with all state needed to iterate the layer loop:
        hidden_states, attention_mask_2d, causal_mask, position_ids,
        cache_position, position_embeddings, layer_types, decoder_layers,
        num_layers, chunk_size, model, device.
        """
        tokenized = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask_2d = tokenized["attention_mask"]

        num_layers = self._get_num_layers()
        chunk_size = self._resolve_chunk_size()

        model = self._load_full_model_cpu()
        device = self.device

        # --- Embedding ---
        embed = _get_embedding_module(model)
        embed.to(device)
        with torch.no_grad():
            hidden_states = embed(input_ids.to(device)).cpu()
        embed.to("cpu")

        # Position IDs, cache position, causal mask
        seq_len = input_ids.shape[1]
        position_ids = attention_mask_2d.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask_2d == 0, 1)
        cache_position = torch.arange(seq_len)
        causal_mask = _make_causal_mask(attention_mask_2d, self.dtype)

        # Rotary position embeddings
        position_embeddings: tuple[torch.Tensor, ...] | torch.Tensor | dict | None = None
        layer_types: list[str] | None = None
        rotary_name = self._find_rotary_embedding_name(model)
        if rotary_name is not None:
            rotary_mod = model
            for part in rotary_name.split("."):
                rotary_mod = getattr(rotary_mod, part)
            rotary_mod.to(device)

            text_cfg = getattr(
                getattr(model, "config", None), "text_config",
                getattr(model, "config", None),
            )
            layer_types_cfg = getattr(text_cfg, "layer_types", None)
            if layer_types_cfg is not None:
                layer_types = list(layer_types_cfg)
                position_embeddings = {}
                with torch.no_grad():
                    for lt in set(layer_types):
                        pe = rotary_mod(
                            hidden_states.to(device),
                            position_ids.to(device),
                            layer_type=lt,
                        )
                        position_embeddings[lt] = tuple(t.cpu() for t in pe)
            else:
                with torch.no_grad():
                    pe = rotary_mod(hidden_states.to(device), position_ids.to(device))
                    if isinstance(pe, tuple):
                        position_embeddings = tuple(t.cpu() for t in pe)
                    else:
                        position_embeddings = pe.cpu()
            rotary_mod.to("cpu")

        decoder_layers = _get_decoder_layers(model)

        return {
            "hidden_states": hidden_states,
            "attention_mask_2d": attention_mask_2d,
            "causal_mask": causal_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
            "layer_types": layer_types,
            "decoder_layers": decoder_layers,
            "num_layers": num_layers,
            "chunk_size": chunk_size,
            "model": model,
            "device": device,
            "input_ids": input_ids,
        }

    def _compute_logits(
        self, model: Any, hidden_states: torch.Tensor, device: str,
    ) -> torch.Tensor:
        """Run final norm + lm_head to produce logits."""
        final_norm = _get_final_norm(model)
        lm_head = _get_lm_head(model)
        final_norm.to(device)
        lm_head.to(device)
        with torch.no_grad():
            logits: torch.Tensor = lm_head(
                final_norm(hidden_states.to(device)),
            ).cpu()
        final_norm.to("cpu")
        lm_head.to("cpu")
        return logits

    @staticmethod
    def _build_layer_kwargs(
        mask_dev: torch.Tensor,
        pos_dev: torch.Tensor,
        cache_position: torch.Tensor,
        pe_dev_map: Any,
        layer_types: list[str] | None,
        layer_idx: int,
        device: str,
    ) -> dict[str, Any]:
        """Build the kwargs dict for a single layer forward call."""
        layer_kwargs: dict[str, Any] = {
            "attention_mask": mask_dev,
            "position_ids": pos_dev,
            "use_cache": False,
            "cache_position": cache_position.to(device),
        }
        if pe_dev_map is not None:
            if isinstance(pe_dev_map, dict) and layer_types is not None:
                lt = layer_types[layer_idx]
                layer_kwargs["position_embeddings"] = pe_dev_map[lt]
            else:
                layer_kwargs["position_embeddings"] = pe_dev_map
        return layer_kwargs

    @staticmethod
    def _pe_to_device(
        position_embeddings: Any, device: str,
    ) -> Any:
        """Move position embeddings to device."""
        if position_embeddings is None:
            return None
        if isinstance(position_embeddings, dict):
            return {
                lt: tuple(t.to(device) for t in pe)
                for lt, pe in position_embeddings.items()
            }
        elif isinstance(position_embeddings, tuple):
            return tuple(t.to(device) for t in position_embeddings)
        else:
            return position_embeddings.to(device)

    # --- Scan methods ---

    # Valid signal names for scan_forward / project_forward
    SCAN_SIGNALS = ("residual", "attn_delta", "mlp_delta", "router_logits")

    def _resolve_signal_hooks(
        self,
        signals: list[str],
        layer_module: Any,
        layer_idx: int,
        router_module_template: str | None,
        model: Any,
    ) -> tuple[
        list[tuple[str, Any, list[torch.Tensor]]],  # [(signal_name, hook_handle, buffer)]
        bool,  # capture_residual — no hook, just layer output
    ]:
        """Set up forward hooks for the requested signals on a single layer.

        Returns list of (signal_name, hook_handle, buffer) for hookable signals,
        plus a flag indicating whether to capture the residual (layer output).
        """
        hooks: list[tuple[str, Any, list[torch.Tensor]]] = []
        capture_residual = False

        for sig in signals:
            if sig == "residual":
                capture_residual = True
                continue

            buf: list[torch.Tensor] = []

            def _hook(
                _mod: Any, _inp: Any, out: Any,
                _buf: list = buf,
            ) -> None:
                delta = out[0] if isinstance(out, tuple) else out
                _buf.append(delta.detach().cpu())

            if sig == "attn_delta":
                mod = _get_attn_submodule(layer_module)
                h = mod.register_forward_hook(_hook)
                hooks.append((sig, h, buf))
            elif sig == "mlp_delta":
                mod = _get_mlp_submodule(layer_module)
                h = mod.register_forward_hook(_hook)
                hooks.append((sig, h, buf))
            elif sig == "router_logits":
                if router_module_template is None:
                    continue  # skip silently — no MoE in this model
                router_path = router_module_template.format(layer=layer_idx)
                if router_path.startswith("model."):
                    router_path = router_path[len("model."):]
                try:
                    router_mod = model
                    for part in router_path.split("."):
                        if part.isdigit():
                            router_mod = router_mod[int(part)]
                        else:
                            router_mod = getattr(router_mod, part)
                    h = router_mod.register_forward_hook(_hook)
                    hooks.append((sig, h, buf))
                except (AttributeError, IndexError):
                    continue  # skip layers without routers

        return hooks, capture_residual

    def scan_forward(
        self,
        prompts: list[str],
        signals: list[str] | None = None,
        n_components: int = 64,
        batch_size: int = 4,
        generative_masks: list[np.ndarray] | None = None,
        external_bases: dict[str, np.ndarray] | None = None,
    ) -> tuple[
        dict[str, Any],                # metadata
        dict[str, np.ndarray],          # bases: {signal_name: [n_layers, dim, k_eff]}
        np.ndarray,                     # projections [N_total, 1, k]
        dict[str, list],                # coords {sample_id, layer, token_pos, signal}
        list[list[int]],                # token_ids per sample
        list[int],                      # seq_lengths per sample
        torch.Tensor,                   # attention_mask [n_samples, max_seq_len]
        dict[str, int],                 # signal_dims: {signal_name: dim}
    ]:
        """Run a full scan forward pass with between-chunk PCA.

        Parameters
        ----------
        prompts : list[str]
            Corpus prompts.
        signals : list[str] or None
            Which signals to capture. Defaults to ["attn_delta", "mlp_delta"].
            Valid: "residual", "attn_delta", "mlp_delta", "router_logits".
        n_components : int
            Max PCA components per (layer, signal).
        batch_size : int
            Prompts per batch.
        generative_masks : list of np.ndarray or None
            Per-sample boolean masks, shape [seq_len_i] each. True =
            generative (assistant) token. If provided, PCA is fit only
            on generative tokens to avoid prompt leakage. All tokens
            are still projected through the basis.
        external_bases : dict of {signal_name: np.ndarray} or None
            Pre-trained PCA bases to project through instead of fitting
            new ones. Shape [n_layers, dim, k] per signal. When provided,
            PCA fitting is skipped entirely — enables fast batched
            projection through an existing scan's basis.

        Returns all data needed to write a SampleScan to disk.
        """
        import gc

        import numpy as np
        from sklearn.decomposition import PCA
        from tqdm import tqdm

        from .activation_types import detect_moe_info

        if signals is None:
            signals = ["attn_delta", "mlp_delta"]

        num_layers = self._get_num_layers()
        model = self._load_full_model_cpu()
        device = self.device
        chunk_size = self._resolve_chunk_size()
        hidden_dim: int | None = None

        # Detect MoE for router_logits signal
        router_module_template: str | None = None
        if "router_logits" in signals:
            try:
                moe_info = detect_moe_info(self.model_name)
                if moe_info is not None:
                    router_module_template = moe_info.router_module_template
                else:
                    signals = [s for s in signals if s != "router_logits"]
            except Exception:
                signals = [s for s in signals if s != "router_logits"]

        # Tokenize all prompts
        tokenized = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
        )
        all_input_ids = tokenized["input_ids"]
        all_attention_mask = tokenized["attention_mask"]

        # Per-sample token IDs (unpadded) and seq lengths
        token_ids_per_sample: list[list[int]] = []
        seq_lengths: list[int] = []
        for i in range(len(prompts)):
            mask_i = all_attention_mask[i]
            real_len = int(mask_i.sum().item())
            seq_lengths.append(real_len)
            token_ids_per_sample.append(
                all_input_ids[i, :real_len].tolist()
            )

        # Split into batches
        n_samples = len(prompts)
        batches: list[tuple[int, int]] = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batches.append((start, end))

        # Run embedding for each batch
        embed = _get_embedding_module(model)
        embed.to(device)
        batch_hidden_states: list[torch.Tensor] = []
        batch_pos_ids: list[torch.Tensor] = []
        batch_causal_masks: list[torch.Tensor] = []
        batch_cache_positions: list[torch.Tensor] = []

        for start, end in batches:
            ids = all_input_ids[start:end]
            mask = all_attention_mask[start:end]

            with torch.no_grad():
                hs = embed(ids.to(device)).cpu()
            batch_hidden_states.append(hs)

            seq_len = ids.shape[1]
            pos_ids = mask.long().cumsum(-1) - 1
            pos_ids.masked_fill_(mask == 0, 1)
            batch_pos_ids.append(pos_ids)
            batch_causal_masks.append(_make_causal_mask(mask, self.dtype))
            batch_cache_positions.append(torch.arange(seq_len))

            if hidden_dim is None:
                hidden_dim = hs.shape[-1]

        embed.to("cpu")

        # Rotary embeddings
        position_embeddings: tuple[torch.Tensor, ...] | torch.Tensor | dict | None = None
        layer_types: list[str] | None = None
        rotary_name = self._find_rotary_embedding_name(model)
        if rotary_name is not None:
            rotary_mod = model
            for part in rotary_name.split("."):
                rotary_mod = getattr(rotary_mod, part)
            rotary_mod.to(device)

            text_cfg = getattr(
                getattr(model, "config", None), "text_config",
                getattr(model, "config", None),
            )
            layer_types_cfg = getattr(text_cfg, "layer_types", None)
            # Use a single sample's pos_ids so cos/sin broadcast to any batch size
            pe_hs = batch_hidden_states[0][:1].to(device)
            pe_pos = batch_pos_ids[0][:1].to(device)

            unique_types = set(layer_types_cfg) if layer_types_cfg is not None else set()
            if layer_types_cfg is not None and len(unique_types) > 1:
                layer_types = list(layer_types_cfg)
                position_embeddings = {}
                with torch.no_grad():
                    for lt in unique_types:
                        pe = rotary_mod(pe_hs, pe_pos, layer_type=lt)
                        position_embeddings[lt] = tuple(t.cpu() for t in pe)
            else:
                if layer_types_cfg is not None:
                    layer_types = list(layer_types_cfg)
                with torch.no_grad():
                    pe = rotary_mod(pe_hs, pe_pos)
                    if isinstance(pe, tuple):
                        position_embeddings = tuple(t.cpu() for t in pe)
                    else:
                        position_embeddings = pe.cpu()
            rotary_mod.to("cpu")

        assert hidden_dim is not None
        decoder_layers = _get_decoder_layers(model)

        # Accumulators
        # Per-signal basis: {signal_name: [n_layers, dim, k_eff]}
        # We'll build these after seeing actual dims from hooks
        signal_bases: dict[str, list[np.ndarray | None]] = {
            sig: [None] * num_layers for sig in signals
        }
        signal_dims: dict[str, int] = {}  # discovered during first chunk
        all_proj_chunks: list[np.ndarray] = []
        all_coord_sample_id: list[int] = []
        all_coord_layer: list[int] = []
        all_coord_token_pos: list[int] = []
        all_coord_signal: list[int] = []

        # --- Layer chunk loop ---
        n_chunks = (num_layers + chunk_size - 1) // chunk_size
        chunk_pbar = tqdm(
            range(0, num_layers, chunk_size),
            desc="Scan: layer chunks",
            total=n_chunks,
        )
        for chunk_start in chunk_pbar:
            chunk_end = min(chunk_start + chunk_size, num_layers)
            chunk_pbar.set_postfix(layers=f"{chunk_start}-{chunk_end-1}")

            for i in range(chunk_start, chunk_end):
                decoder_layers[i].to(device)

            # {layer_idx: {signal_name: [tensor_per_batch, ...]}}
            per_layer_captures: dict[int, dict[str, list[torch.Tensor]]] = {
                L: {sig: [] for sig in signals}
                for L in range(chunk_start, chunk_end)
            }

            for batch_idx, (start, end) in enumerate(batches):
                hs = batch_hidden_states[batch_idx].to(device)
                mask_dev = batch_causal_masks[batch_idx].to(device)
                pos_dev = batch_pos_ids[batch_idx].to(device)
                pe_dev = self._pe_to_device(position_embeddings, device)

                with torch.no_grad():
                    for layer_idx in range(chunk_start, chunk_end):
                        layer_module = decoder_layers[layer_idx]

                        hook_list, capture_residual = self._resolve_signal_hooks(
                            signals, layer_module, layer_idx,
                            router_module_template, model,
                        )

                        layer_kwargs = self._build_layer_kwargs(
                            mask_dev, pos_dev, batch_cache_positions[batch_idx],
                            pe_dev, layer_types, layer_idx, device,
                        )

                        output = layer_module(hs, **layer_kwargs)
                        hs = output[0] if isinstance(output, tuple) else output

                        # Collect hooked signals
                        for sig_name, handle, buf in hook_list:
                            handle.remove()
                            if buf:
                                per_layer_captures[layer_idx][sig_name].append(buf[0])

                        # Residual = layer output (already in hs)
                        if capture_residual:
                            per_layer_captures[layer_idx]["residual"].append(
                                hs.detach().cpu()
                            )

                batch_hidden_states[batch_idx] = hs.cpu()
                del hs, mask_dev, pos_dev, pe_dev

            # Offload chunk
            for i in range(chunk_start, chunk_end):
                decoder_layers[i].to("cpu")
            gc.collect()
            if torch.cuda.is_available() and device != "cpu":
                torch.cuda.empty_cache()

            # --- Between-chunk PCA ---
            pca_items = [
                (layer_idx, sig_idx, sig_name)
                for layer_idx in range(chunk_start, chunk_end)
                for sig_idx, sig_name in enumerate(signals)
            ]
            for layer_idx, sig_idx, sig_name in tqdm(
                pca_items,
                desc=f"  PCA fit (layers {chunk_start}-{chunk_end-1})",
                leave=False,
            ):
                captures = per_layer_captures[layer_idx][sig_name]
                if not captures:
                    continue

                stacked = torch.cat(captures, dim=0)  # [B_total, S, dim]
                B_total, S, dim = stacked.shape
                flat = stacked.reshape(-1, dim).float().numpy()

                # Record signal dimension
                if sig_name not in signal_dims:
                    signal_dims[sig_name] = dim

                if external_bases is not None and sig_name in external_bases:
                    # Use pre-trained basis — skip PCA fitting
                    basis = external_bases[sig_name][layer_idx]  # [dim, k]
                    signal_bases[sig_name][layer_idx] = basis
                    k = basis.shape[1]
                    projected = (flat @ basis.astype(np.float32)).astype(np.float16)
                else:
                    # Build PCA fit data — filter to generative tokens if mask provided
                    if generative_masks is not None:
                        fit_mask_parts = []
                        for sid in range(B_total):
                            if sid < len(generative_masks) and generative_masks[sid] is not None:
                                gmask = generative_masks[sid]
                                padded_mask = np.zeros(S, dtype=bool)
                                padded_mask[:min(len(gmask), S)] = gmask[:S]
                                fit_mask_parts.append(padded_mask)
                            else:
                                fit_mask_parts.append(np.ones(S, dtype=bool))
                        fit_mask = np.concatenate(fit_mask_parts)
                        flat_fit = flat[fit_mask]
                    else:
                        flat_fit = flat

                    k = min(n_components, flat_fit.shape[0] - 1, dim)
                    pca = PCA(n_components=k)
                    pca.fit(flat_fit)

                    basis = pca.components_.T.astype(np.float16)
                    signal_bases[sig_name][layer_idx] = basis

                    # Project ALL tokens through the basis
                    projected = pca.transform(flat).astype(np.float16)
                if k < n_components:
                    padded = np.zeros(
                        (projected.shape[0], n_components), dtype=np.float16,
                    )
                    padded[:, :k] = projected
                    projected = padded

                all_proj_chunks.append(projected)

                for sample_idx in range(B_total):
                    for tok in range(S):
                        all_coord_sample_id.append(sample_idx)
                        all_coord_layer.append(layer_idx)
                        all_coord_token_pos.append(tok)
                        all_coord_signal.append(sig_idx)

                del stacked, flat, projected

            del per_layer_captures
            gc.collect()

        # Assemble per-signal basis arrays: {sig: [n_layers, dim, k_eff]}
        final_bases: dict[str, np.ndarray] = {}
        for sig_name in signals:
            dim = signal_dims.get(sig_name, hidden_dim)
            k_eff = min(n_components, dim)
            basis_arr = np.zeros((num_layers, dim, k_eff), dtype=np.float16)
            for L in range(num_layers):
                layer_basis = signal_bases[sig_name][L]
                if layer_basis is not None:
                    actual_k = layer_basis.shape[1]
                    basis_arr[L, :, :actual_k] = layer_basis
            final_bases[sig_name] = basis_arr

        # Stack projections
        all_projections = np.concatenate(all_proj_chunks, axis=0)
        all_projections = all_projections[:, np.newaxis, :]  # [N_total, 1, k]

        metadata = {
            "model_id": self.model_name,
            "hidden_dim": hidden_dim,
            "n_layers": num_layers,
            "n_components": n_components,
            "n_samples": n_samples,
            "signals": signals,
        }

        coords = {
            "sample_id": all_coord_sample_id,
            "layer": all_coord_layer,
            "token_pos": all_coord_token_pos,
            "signal": all_coord_signal,
        }

        return (
            metadata,
            final_bases,
            all_projections,
            coords,
            token_ids_per_sample,
            seq_lengths,
            all_attention_mask,
            signal_dims,
        )

    def project_forward(
        self,
        prompt: str,
        bases: dict[str, np.ndarray],
        signals: list[str],
        include_logits: bool = True,
    ) -> tuple[np.ndarray, list[int], torch.Tensor | None]:
        """Run a single-prompt forward pass, projecting deltas onto stored bases.

        Parameters
        ----------
        prompt : str
            The prompt to forward-pass.
        bases : dict[str, np.ndarray]
            Maps signal name to basis [n_layers, dim, k_eff].
        signals : list[str]
            Signal names in order.
        include_logits : bool
            Whether to compute logits for surprise strip.

        Returns
        -------
        tuple
            (projections, token_ids, logits)
            - projections: [seq_len, n_layers, n_signals, max_k] float32
            - token_ids: list of int token IDs (unpadded)
            - logits: [1, seq_len, vocab_size] or None
        """
        import gc

        import numpy as np

        from .activation_types import detect_moe_info

        ctx = self._prepare_chunked_pass([prompt])
        hidden_states = ctx["hidden_states"]
        model = ctx["model"]
        device = ctx["device"]
        decoder_layers = ctx["decoder_layers"]
        num_layers = ctx["num_layers"]
        chunk_size = ctx["chunk_size"]
        causal_mask = ctx["causal_mask"]
        position_ids = ctx["position_ids"]
        cache_position = ctx["cache_position"]
        position_embeddings = ctx["position_embeddings"]
        layer_types = ctx["layer_types"]
        attention_mask_2d = ctx["attention_mask_2d"]
        input_ids = ctx["input_ids"]

        seq_len = input_ids.shape[1]
        real_len = int(attention_mask_2d[0].sum().item())
        token_ids = input_ids[0, :real_len].tolist()

        # Find max k across all signal bases
        max_k = max(b.shape[2] for b in bases.values())
        n_signals = len(signals)
        projections = np.zeros(
            (seq_len, num_layers, n_signals, max_k), dtype=np.float32,
        )

        # Convert bases to torch
        bases_torch = {
            sig: torch.from_numpy(b).float() for sig, b in bases.items()
        }

        # Router template for router_logits
        router_module_template: str | None = None
        if "router_logits" in signals:
            try:
                moe_info = detect_moe_info(self.model_name)
                if moe_info is not None:
                    router_module_template = moe_info.router_module_template
            except Exception:
                pass

        for chunk_start in range(0, num_layers, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_layers)

            for i in range(chunk_start, chunk_end):
                decoder_layers[i].to(device)

            hs = hidden_states.to(device)
            mask_dev = causal_mask.to(device)
            pos_dev = position_ids.to(device)
            pe_dev = self._pe_to_device(position_embeddings, device)

            with torch.no_grad():
                for layer_idx in range(chunk_start, chunk_end):
                    layer_module = decoder_layers[layer_idx]

                    hook_list, capture_residual = self._resolve_signal_hooks(
                        signals, layer_module, layer_idx,
                        router_module_template, model,
                    )

                    layer_kwargs = self._build_layer_kwargs(
                        mask_dev, pos_dev, cache_position,
                        pe_dev, layer_types, layer_idx, device,
                    )

                    output = layer_module(hs, **layer_kwargs)
                    hs = output[0] if isinstance(output, tuple) else output

                    # Project hooked signals
                    for sig_name, handle, buf in hook_list:
                        handle.remove()
                        if buf and sig_name in bases_torch:
                            sig_idx = signals.index(sig_name)
                            delta = buf[0][0].float()  # [seq_len, dim]
                            b = bases_torch[sig_name][layer_idx]  # [dim, k]
                            k = b.shape[1]
                            proj = (delta @ b).numpy()
                            projections[:, layer_idx, sig_idx, :k] = proj

                    # Project residual
                    if capture_residual and "residual" in bases_torch:
                        sig_idx = signals.index("residual")
                        delta = hs[0].cpu().float()
                        b = bases_torch["residual"][layer_idx]
                        k = b.shape[1]
                        proj = (delta @ b).numpy()
                        projections[:, layer_idx, sig_idx, :k] = proj

            hidden_states = hs.cpu()
            del hs, mask_dev, pos_dev, pe_dev

            for i in range(chunk_start, chunk_end):
                decoder_layers[i].to("cpu")
            gc.collect()
            if torch.cuda.is_available() and device != "cpu":
                torch.cuda.empty_cache()

        logits = None
        if include_logits:
            logits = self._compute_logits(model, hidden_states, device)

        return projections, token_ids, logits

    # --- ExtractionBackend interface ---

    def extract_batch(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        activations, attention_mask, _, _ = self._chunked_forward(
            prompts, layer_indices,
        )
        assert activations is not None
        return activations, attention_mask

    def extract_batch_with_logits(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        activations, attention_mask, logits, _ = self._chunked_forward(
            prompts, layer_indices, include_logits=True,
        )
        assert logits is not None
        return activations, attention_mask, logits, None

    def extract_batch_extended(
        self,
        prompts: list[str],
        spec: ExtractionSpec,
        **kwargs: Any,
    ) -> ExtractedBatch:
        from .activation_types import ExtractedBatch

        activations, attention_mask, logits, router_logits = self._chunked_forward(
            prompts,
            spec.hidden_layers,
            include_logits=spec.include_logits,
            router_layer_indices=spec.router_layers,
            router_module_template=spec.router_module_template,
            router_hook_strategy=spec.router_hook_strategy,
        )
        return ExtractedBatch(
            activations=activations,
            attention_mask=attention_mask,
            logits=logits,
            logits_indices=None,
            router_logits=router_logits,
        )


# ---------------------------------------------------------------------------
# DiskOffloadBackend — for models exceeding CPU RAM (e.g. DeepSeek-V3 671B)
# ---------------------------------------------------------------------------


def _dequantize_fp8_block(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize FP8 e4m3 block-quantized weight to bf16.

    Parameters
    ----------
    weight_fp8 : torch.Tensor
        Weight in float8_e4m3fn, shape ``(out_features, in_features)``.
    scale_inv : torch.Tensor
        Per-block inverse scale, shape
        ``(ceil(out_features/block_size), ceil(in_features/block_size))``.
    block_size : int
        Block size used for quantization (default 128).
    """
    out_f, in_f = weight_fp8.shape
    ob = (out_f + block_size - 1) // block_size
    ib = (in_f + block_size - 1) // block_size

    pad_out = ob * block_size - out_f
    pad_in = ib * block_size - in_f
    if pad_out > 0 or pad_in > 0:
        w = torch.nn.functional.pad(weight_fp8.float(), (0, pad_in, 0, pad_out))
    else:
        w = weight_fp8.float()

    w = w.reshape(ob, block_size, ib, block_size)
    w = w * scale_inv[:, None, :, None]
    w = w.reshape(ob * block_size, ib * block_size)

    if pad_out > 0 or pad_in > 0:
        w = w[:out_f, :in_f]

    return w.to(torch.bfloat16)


def _pack_expert_weights(
    weights: dict[str, torch.Tensor],
    layer_prefix: str,
    n_experts: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Pack individual per-expert FP8 weights into 3D packed format.

    Safetensors stores per-expert weights as separate 2D tensors.
    Transformers' native DeepSeek-V3 implementation expects packed 3D
    tensors: ``gate_up_proj [n_experts, 2*intermediate, hidden]`` and
    ``down_proj [n_experts, hidden, intermediate]``.
    """
    test_key = f"{layer_prefix}mlp.experts.0.gate_proj.weight"
    if test_key not in weights:
        return {}

    gate_up_list = []
    down_list = []

    for i in range(n_experts):
        gk = f"{layer_prefix}mlp.experts.{i}.gate_proj.weight"
        gs = gk + "_scale_inv"
        uk = f"{layer_prefix}mlp.experts.{i}.up_proj.weight"
        us = uk + "_scale_inv"
        dk = f"{layer_prefix}mlp.experts.{i}.down_proj.weight"
        ds = dk + "_scale_inv"

        gate = (
            _dequantize_fp8_block(weights[gk], weights[gs])
            if gs in weights
            else weights[gk].to(torch.bfloat16)
        )
        up = (
            _dequantize_fp8_block(weights[uk], weights[us])
            if us in weights
            else weights[uk].to(torch.bfloat16)
        )
        gate_up_list.append(torch.cat([gate, up], dim=0))

        down = (
            _dequantize_fp8_block(weights[dk], weights[ds])
            if ds in weights
            else weights[dk].to(torch.bfloat16)
        )
        down_list.append(down)

        for k in [gk, gs, uk, us, dk, ds]:
            weights.pop(k, None)

    packed = {}
    ep = f"{layer_prefix}mlp.experts."
    packed[ep + "gate_up_proj"] = torch.stack(gate_up_list).to(device)
    packed[ep + "down_proj"] = torch.stack(down_list).to(device)
    del gate_up_list, down_list
    return packed


def _materialize_module(
    module: torch.nn.Module,
    weights: dict[str, torch.Tensor],
    prefix: str,
    device: str,
) -> None:
    """Replace meta-device parameters with real tensors from *weights*.

    FP8 weights are dequantized to bf16 via their companion
    ``_scale_inv`` tensors. Weights are loaded on CPU (in *weights*)
    and moved to *device* during assignment to avoid doubling GPU memory.
    """
    fp8_keys = {
        n
        for n in weights
        if n.endswith(".weight")
        and weights[n].dtype == torch.float8_e4m3fn
        and n + "_scale_inv" in weights
    }

    def _get(full_name: str) -> torch.Tensor | None:
        if full_name in fp8_keys:
            w = _dequantize_fp8_block(
                weights[full_name], weights[full_name + "_scale_inv"],
            )
            weights.pop(full_name, None)
            weights.pop(full_name + "_scale_inv", None)
            return w
        if full_name in weights:
            w = weights[full_name]
            return w.to(torch.bfloat16) if w.dtype == torch.float8_e4m3fn else w
        return None

    for name, _param in module.named_parameters():
        full_name = f"{prefix}{name}"
        new_data = _get(full_name)
        if new_data is None:
            continue
        parts = name.split(".")
        target: Any = module
        for part in parts[:-1]:
            target = target[int(part)] if part.isdigit() else getattr(target, part)
        target._parameters[parts[-1]] = torch.nn.Parameter(
            new_data.to(device), requires_grad=False,
        )

    for name, _buf in module.named_buffers():
        full_name = f"{prefix}{name}"
        if full_name in weights:
            parts = name.split(".")
            target2: Any = module
            for part in parts[:-1]:
                target2 = target2[int(part)] if part.isdigit() else getattr(target2, part)
            target2._buffers[parts[-1]] = weights[full_name].to(device)


def _free_module(module: torch.nn.Module) -> None:
    """Replace all parameters and buffers with empty meta tensors."""
    for name, _param in list(module.named_parameters()):
        parts = name.split(".")
        target: Any = module
        for part in parts[:-1]:
            target = target[int(part)] if part.isdigit() else getattr(target, part)
        target._parameters[parts[-1]] = torch.nn.Parameter(
            torch.empty(0, device="meta"), requires_grad=False,
        )
    for name, _buf in list(module.named_buffers()):
        parts = name.split(".")
        target2: Any = module
        for part in parts[:-1]:
            target2 = target2[int(part)] if part.isdigit() else getattr(target2, part)
        target2._buffers[parts[-1]] = torch.empty(0, device="meta")


class DiskOffloadBackend(ExtractionBackend):
    """Backend for models that exceed CPU RAM (e.g. DeepSeek-V3 671B FP8).

    Loads layer weights directly from safetensors files to GPU one layer
    at a time. Supports FP8 block-quantized weights with automatic
    dequantization and expert weight packing for MoE architectures.

    The key method is :meth:`extract_all`, which processes an entire
    dataset layer-by-layer so each layer is loaded from disk exactly
    once, regardless of dataset size.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(model_name, device)
        self.dtype = dtype
        self._tokenizer_obj: Any | None = None
        self._config: Any | None = None
        self._model_skeleton: Any | None = None
        self._snapshot_dir: Any | None = None
        self._layer_to_tensors: dict | None = None
        self._non_layer_tensors: list | None = None

    # --- Lazy initialization ---

    def _get_config(self) -> Any:
        if self._config is None:
            from transformers import AutoConfig
            self._config = AutoConfig.from_pretrained(self.model_name)
        return self._config

    def _get_snapshot_dir(self) -> Any:
        if self._snapshot_dir is None:
            from pathlib import Path

            from huggingface_hub import snapshot_download
            self._snapshot_dir = Path(snapshot_download(
                self.model_name,
                allow_patterns=["*.safetensors", "*.json"],
            ))
        return self._snapshot_dir

    def _get_shard_map(self) -> tuple[dict, list]:
        if self._layer_to_tensors is None:
            import json
            from collections import defaultdict

            snap = self._get_snapshot_dir()
            index_path = snap / "model.safetensors.index.json"
            with open(index_path) as f:
                index = json.load(f)

            layers: dict[int, list[tuple[str, str]]] = defaultdict(list)
            non_layer: list[tuple[str, str]] = []

            for tensor_name, shard_file in index["weight_map"].items():
                parts = tensor_name.split(".")
                if (
                    len(parts) >= 3
                    and parts[0] == "model"
                    and parts[1] == "layers"
                    and parts[2].isdigit()
                ):
                    layers[int(parts[2])].append((tensor_name, shard_file))
                else:
                    non_layer.append((tensor_name, shard_file))

            self._layer_to_tensors = dict(layers)
            self._non_layer_tensors = non_layer
        assert self._non_layer_tensors is not None
        return self._layer_to_tensors, self._non_layer_tensors

    def _load_tensors(
        self, tensor_list: list[tuple[str, str]], device: str,
    ) -> dict[str, torch.Tensor]:
        """Load tensors from safetensors shards to *device*."""
        from collections import defaultdict

        from safetensors.torch import load_file as st_load_file

        shard_to_keys: dict[str, list[str]] = defaultdict(list)
        for tensor_name, shard_file in tensor_list:
            shard_to_keys[shard_file].append(tensor_name)

        snap = self._get_snapshot_dir()
        result: dict[str, torch.Tensor] = {}
        for shard_file, keys in shard_to_keys.items():
            shard_data = st_load_file(str(snap / shard_file), device=device)
            for k in keys:
                if k in shard_data:
                    result[k] = shard_data[k]
            del shard_data
        return result

    def _get_model_skeleton(self) -> Any:
        """Create an empty model (meta device) for the forward graph."""
        if self._model_skeleton is None:
            from accelerate import init_empty_weights
            from transformers import AutoModelForCausalLM

            config = self._get_config()
            # Disable quantizer so we get standard nn.Linear modules
            config.quantization_config = None
            with init_empty_weights():
                self._model_skeleton = AutoModelForCausalLM.from_config(config)
            self._model_skeleton.eval()
        return self._model_skeleton

    # --- Properties ---

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer_obj is None:
            from transformers import AutoTokenizer
            self._tokenizer_obj = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer_obj.pad_token is None:
                self._tokenizer_obj.pad_token = self._tokenizer_obj.eos_token
        return self._tokenizer_obj

    @property
    def model(self) -> Any:
        raise RuntimeError(
            "DiskOffloadBackend does not keep the full model in memory. "
            "Use extract_all() for efficient feature extraction."
        )

    # --- Core: full-dataset layer-by-layer extraction ---

    def extract_all(
        self,
        prompts: list[str],
        spec: ExtractionSpec,
        batch_size: int = 16,
        pool: str | None = None,
    ) -> ExtractedBatch:
        """Extract features from *all* prompts, loading each layer once.

        This is the efficient entry point for large models. All prompts
        are tokenized upfront, then the entire dataset is streamed
        through each layer before moving to the next. Each layer's
        weights are loaded from safetensors exactly once.

        Parameters
        ----------
        prompts : list[str]
            All prompts to process.
        spec : ExtractionSpec
            What to extract (hidden layers, router logits, logits).
        batch_size : int
            GPU batch size for forward passes through each layer.
        pool : str or None
            Pooling strategy for captured features. ``None`` keeps full
            ``(N, seq_len, dim)`` tensors (high memory). ``"mean"``
            mean-pools over valid tokens per statement, storing only
            ``(N, dim)`` per layer — essential for large combined runs.

        Returns
        -------
        ExtractedBatch
            Extracted features for all prompts.
        """
        import gc
        import math

        from .activation_types import ExtractedBatch

        config = self._get_config()
        model = self._get_model_skeleton()
        layer_map, non_layer = self._get_shard_map()
        device = self.device

        num_layers = config.num_hidden_layers
        hidden_target = set(spec.hidden_layers)
        router_target = set(spec.router_layers or [])
        n_experts = getattr(config, "n_routed_experts", None)
        first_moe = getattr(config, "first_k_dense_replace", 0)

        # --- Tokenize all prompts ---
        tokenized = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        )
        all_input_ids = tokenized["input_ids"]
        all_attention_mask = tokenized["attention_mask"]
        n_prompts = all_input_ids.shape[0]
        seq_len = all_input_ids.shape[1]
        n_batches = math.ceil(n_prompts / batch_size)

        # --- Phase 1: Embedding ---
        embed_tensors = [(n, f) for n, f in non_layer if "embed_tokens" in n]
        embed_weights = self._load_tensors(embed_tensors, device)
        embed = _get_embedding_module(model)
        _materialize_module(embed, embed_weights, "model.embed_tokens.", device)
        del embed_weights

        # Process all batches through embedding
        all_hidden = torch.zeros(
            n_prompts, seq_len, config.hidden_size, dtype=self.dtype,
        )
        with torch.no_grad():
            for b in range(n_batches):
                s, e = b * batch_size, min((b + 1) * batch_size, n_prompts)
                all_hidden[s:e] = embed(all_input_ids[s:e].to(device)).cpu()

        _free_module(embed)
        torch.cuda.empty_cache()

        # --- Position setup ---
        position_ids = all_attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(all_attention_mask == 0, 1)
        cache_position = torch.arange(seq_len)

        min_val = torch.finfo(self.dtype).min
        causal_mask = torch.full(
            (1, 1, seq_len, seq_len), min_val, dtype=self.dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)

        # --- Rotary embeddings ---
        rotary_name = ChunkedLocalBackend._find_rotary_embedding_name(model)
        position_embeddings: tuple[torch.Tensor, ...] | torch.Tensor | None = None
        if rotary_name is not None:
            rotary_mod = model
            for part in rotary_name.split("."):
                rotary_mod = getattr(rotary_mod, part)

            # Re-initialize rotary from config (meta tensors have no data)
            dim = getattr(config, "qk_rope_head_dim", None)
            if dim is None:
                head_dim = config.hidden_size // config.num_attention_heads
                dim = head_dim
            base = getattr(config, "rope_theta", 10000.0)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
            rotary_mod.to_empty(device=device)
            rotary_mod.inv_freq = inv_freq

            with torch.no_grad():
                pe = rotary_mod(
                    all_hidden[:1].to(device), position_ids[:1].to(device),
                )
                if isinstance(pe, tuple):
                    position_embeddings = tuple(t.cpu() for t in pe)
                else:
                    position_embeddings = pe.cpu()

            _free_module(rotary_mod)
            torch.cuda.empty_cache()

        # --- Phase 2: Layer-by-layer ---
        captured_hidden: dict[int, torch.Tensor] = {}
        captured_router: dict[int, torch.Tensor] = {}
        decoder_layers = _get_decoder_layers(model)

        for layer_idx in range(num_layers):
            # Load layer weights: CPU first, then materialize on GPU
            layer_weights = self._load_tensors(
                layer_map[layer_idx], "cpu",
            )
            prefix = f"model.layers.{layer_idx}."

            # Pack expert weights for MoE layers
            if n_experts and layer_idx >= first_moe:
                packed = _pack_expert_weights(
                    layer_weights, prefix, n_experts, device,
                )
                layer_weights.update(packed)
                del packed

            layer_module = decoder_layers[layer_idx]
            _materialize_module(layer_module, layer_weights, prefix, device)
            del layer_weights

            # Prepare per-batch args on GPU
            mask_dev = causal_mask.to(device)
            pos_cache_dev = cache_position.to(device)
            if position_embeddings is not None:
                if isinstance(position_embeddings, tuple):
                    pe_dev: Any = tuple(t.to(device) for t in position_embeddings)
                else:
                    pe_dev = position_embeddings.to(device)
            else:
                pe_dev = None

            layer_hidden_out = []
            layer_router_out = []

            with torch.no_grad():
                for b in range(n_batches):
                    s = b * batch_size
                    e = min(s + batch_size, n_prompts)
                    hs = all_hidden[s:e].to(device)
                    pos_dev = position_ids[s:e].to(device)

                    # Expand causal mask for batch
                    batch_mask = mask_dev.expand(e - s, -1, -1, -1)
                    # Apply padding mask
                    pad_mask = all_attention_mask[s:e, None, None, :].to(self.dtype).to(device)
                    batch_mask = batch_mask.clone()
                    batch_mask.masked_fill_(pad_mask == 0, min_val)

                    # Router hook
                    rh = None
                    router_buf: list[torch.Tensor] = []
                    if layer_idx in router_target and spec.router_module_template:
                        router_path = spec.router_module_template.format(layer=layer_idx)
                        if router_path.startswith("model."):
                            router_path = router_path[len("model."):]
                        rmod = model
                        for part in router_path.split("."):
                            rmod = rmod[int(part)] if part.isdigit() else getattr(rmod, part)

                        if spec.router_hook_strategy == "input_gate":
                            def _ig_hook(
                                mod: Any, args: Any, out: Any,
                                _buf: list = router_buf,
                            ) -> None:
                                hs_in = args[0] if isinstance(args, tuple) else args
                                gw = mod.gate.weight
                                _buf.append(
                                    torch.nn.functional.linear(
                                        hs_in.to(gw.dtype), gw,
                                    ).detach().cpu()
                                )
                            rh = rmod.register_forward_hook(_ig_hook)
                        else:
                            def _out_hook(
                                _mod: Any, _inp: Any, out: Any,
                                _buf: list = router_buf,
                            ) -> None:
                                t = out[0] if isinstance(out, tuple) else out
                                _buf.append(t.detach().cpu())
                            rh = rmod.register_forward_hook(_out_hook)

                    # Layer forward
                    layer_kwargs: dict[str, Any] = {
                        "attention_mask": batch_mask,
                        "position_ids": pos_dev,
                        "use_cache": False,
                        "cache_position": pos_cache_dev,
                    }
                    if pe_dev is not None:
                        layer_kwargs["position_embeddings"] = pe_dev

                    output = layer_module(hs, **layer_kwargs)
                    hs_out = output[0] if isinstance(output, tuple) else output

                    # Store hidden states back to CPU buffer
                    all_hidden[s:e] = hs_out.to(self.dtype).cpu()

                    # Capture probe features (last token only is done by caller)
                    if layer_idx in hidden_target:
                        layer_hidden_out.append(hs_out.detach().cpu())

                    if rh is not None:
                        rh.remove()
                        if router_buf:
                            layer_router_out.append(router_buf[0])

            # Collect captured features for this layer
            if layer_hidden_out:
                full = torch.cat(layer_hidden_out, dim=0)  # (N, seq, dim)
                if pool == "mean":
                    # Mean-pool over valid tokens: (N, seq, dim) -> (N, dim)
                    pooled = torch.zeros(
                        full.shape[0], full.shape[2], dtype=torch.float32,
                    )
                    for i in range(full.shape[0]):
                        valid = all_attention_mask[i].bool()
                        if valid.sum().item() > 0:
                            pooled[i] = full[i, valid].float().mean(dim=0)
                    captured_hidden[layer_idx] = pooled
                else:
                    captured_hidden[layer_idx] = full
                del full
            if layer_router_out:
                full_r = torch.cat(layer_router_out, dim=0)  # (N, seq, n_experts)
                if pool == "mean":
                    pooled_r = torch.zeros(
                        full_r.shape[0], full_r.shape[2], dtype=torch.float32,
                    )
                    for i in range(full_r.shape[0]):
                        valid = all_attention_mask[i].bool()
                        if valid.sum().item() > 0:
                            pooled_r[i] = full_r[i, valid].float().mean(dim=0)
                    captured_router[layer_idx] = pooled_r
                else:
                    captured_router[layer_idx] = full_r
                del full_r

            # Free layer
            _free_module(layer_module)
            del mask_dev, pos_cache_dev, pe_dev
            gc.collect()
            torch.cuda.empty_cache()

        # --- Phase 3: Logits (optional) ---
        logits_out: torch.Tensor | None = None
        if spec.include_logits:
            norm_tensors = [
                (n, f) for n, f in non_layer
                if "norm" in n and "layer" not in n
            ]
            head_tensors = [(n, f) for n, f in non_layer if "lm_head" in n]

            norm_w = self._load_tensors(norm_tensors, device)
            final_norm = _get_final_norm(model)
            _materialize_module(final_norm, norm_w, "model.norm.", device)
            del norm_w

            head_w = self._load_tensors(head_tensors, device)
            lm_head = _get_lm_head(model)
            _materialize_module(lm_head, head_w, "lm_head.", device)
            del head_w

            logit_batches = []
            with torch.no_grad():
                for b in range(n_batches):
                    s = b * batch_size
                    e = min(s + batch_size, n_prompts)
                    hs_dev = all_hidden[s:e].to(device)
                    logit_batches.append(
                        lm_head(final_norm(hs_dev)).cpu()
                    )
            logits_out = torch.cat(logit_batches, dim=0)

            _free_module(final_norm)
            _free_module(lm_head)
            torch.cuda.empty_cache()

        # --- Assemble result ---
        activations: torch.Tensor | None = None
        if captured_hidden:
            sorted_layers = sorted(captured_hidden.keys())
            # When pooled, each tensor is (N, dim); otherwise (N, seq, dim)
            activations = torch.cat(
                [captured_hidden[li] for li in sorted_layers], dim=-1,
            )

        return ExtractedBatch(
            activations=activations,
            attention_mask=all_attention_mask,
            logits=logits_out,
            logits_indices=None,
            router_logits=captured_router if captured_router else None,
            hidden_per_layer=captured_hidden if captured_hidden else None,
        )

    # --- Scan: full-dataset layer-by-layer with PCA ---

    def scan_forward(
        self,
        prompts: list[str],
        signals: list[str] | None = None,
        n_components: int = 64,
        batch_size: int = 4,
        generative_masks: list[np.ndarray] | None = None,
        external_bases: dict[str, np.ndarray] | None = None,
    ) -> tuple[
        dict[str, Any],                # metadata
        dict[str, np.ndarray],          # bases
        np.ndarray,                     # projections [N_total, 1, k]
        dict[str, list],                # coords
        list[list[int]],                # token_ids per sample
        list[int],                      # seq_lengths per sample
        torch.Tensor,                   # attention_mask
        dict[str, int],                 # signal_dims
    ]:
        """Run a full scan forward pass, loading layers from disk.

        Same interface as ChunkedLocalBackend.scan_forward but loads each
        layer's weights from safetensors files instead of keeping the full
        model in CPU RAM. Enables scanning 70B+ models on machines with
        limited CPU memory.
        """
        import gc

        import numpy as np
        from sklearn.decomposition import PCA
        from tqdm import tqdm

        if signals is None:
            signals = ["attn_delta", "mlp_delta"]

        config = self._get_config()
        model = self._get_model_skeleton()
        layer_map, non_layer = self._get_shard_map()
        device = self.device
        num_layers = config.num_hidden_layers
        hidden_dim: int | None = None

        # --- Tokenize ---
        tokenized = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
        )
        all_input_ids = tokenized["input_ids"]
        all_attention_mask = tokenized["attention_mask"]

        token_ids_per_sample: list[list[int]] = []
        seq_lengths: list[int] = []
        for i in range(len(prompts)):
            mask_i = all_attention_mask[i]
            real_len = int(mask_i.sum().item())
            seq_lengths.append(real_len)
            token_ids_per_sample.append(
                all_input_ids[i, :real_len].tolist()
            )

        n_samples = len(prompts)
        batches = [
            (s, min(s + batch_size, n_samples))
            for s in range(0, n_samples, batch_size)
        ]

        # --- Phase 1: Embedding ---
        embed_tensors = [(n, f) for n, f in non_layer if "embed_tokens" in n]
        embed_weights = self._load_tensors(embed_tensors, device)
        embed = _get_embedding_module(model)
        _materialize_module(embed, embed_weights, "model.embed_tokens.", device)
        del embed_weights

        batch_hidden_states: list[torch.Tensor] = []
        batch_pos_ids: list[torch.Tensor] = []
        batch_causal_masks: list[torch.Tensor] = []
        batch_cache_positions: list[torch.Tensor] = []

        with torch.no_grad():
            for start, end in batches:
                ids = all_input_ids[start:end]
                mask = all_attention_mask[start:end]
                hs = embed(ids.to(device)).cpu()
                batch_hidden_states.append(hs)

                seq_len = ids.shape[1]
                pos_ids = mask.long().cumsum(-1) - 1
                pos_ids.masked_fill_(mask == 0, 1)
                batch_pos_ids.append(pos_ids)
                batch_causal_masks.append(_make_causal_mask(mask, self.dtype))
                batch_cache_positions.append(torch.arange(seq_len))

                if hidden_dim is None:
                    hidden_dim = hs.shape[-1]

        _free_module(embed)
        torch.cuda.empty_cache()

        # --- Rotary embeddings ---
        position_embeddings: tuple[torch.Tensor, ...] | torch.Tensor | dict | None = None
        layer_types: list[str] | None = None
        rotary_name = ChunkedLocalBackend._find_rotary_embedding_name(model)
        if rotary_name is not None:
            rotary_mod = model
            for part in rotary_name.split("."):
                rotary_mod = getattr(rotary_mod, part)

            # Re-initialize rotary from config
            text_cfg = getattr(config, "text_config", config)
            dim = getattr(text_cfg, "qk_rope_head_dim", None)
            if dim is None:
                head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
                dim = head_dim
            base = getattr(text_cfg, "rope_theta", 10000.0)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
            rotary_mod.to_empty(device=device)
            rotary_mod.inv_freq = inv_freq

            layer_types_cfg = getattr(text_cfg, "layer_types", None)
            # Only use per-layer-type PE if there are actually different types
            # (e.g. Gemma has sliding_window + global). If all the same, skip.
            # Use a single sample's pos_ids so cos/sin broadcast to any batch size
            pe_hs = batch_hidden_states[0][:1].to(device)
            pe_pos = batch_pos_ids[0][:1].to(device)

            unique_types = set(layer_types_cfg) if layer_types_cfg else set()
            if layer_types_cfg and len(unique_types) > 1:
                layer_types = list(layer_types_cfg)
                position_embeddings = {}
                with torch.no_grad():
                    for lt in unique_types:
                        pe = rotary_mod(pe_hs, pe_pos, layer_type=lt)
                        position_embeddings[lt] = tuple(t.cpu() for t in pe)
            else:
                if layer_types_cfg is not None:
                    layer_types = list(layer_types_cfg)
                with torch.no_grad():
                    pe = rotary_mod(pe_hs, pe_pos)
                    if isinstance(pe, tuple):
                        position_embeddings = tuple(t.cpu() for t in pe)
                    else:
                        position_embeddings = pe.cpu()

            _free_module(rotary_mod)
            torch.cuda.empty_cache()

        assert hidden_dim is not None
        decoder_layers = _get_decoder_layers(model)

        # Accumulators
        signal_bases: dict[str, list[np.ndarray | None]] = {
            sig: [None] * num_layers for sig in signals
        }
        signal_dims: dict[str, int] = {}
        all_proj_chunks: list[np.ndarray] = []
        all_coord_sample_id: list[int] = []
        all_coord_layer: list[int] = []
        all_coord_token_pos: list[int] = []
        all_coord_signal: list[int] = []

        # --- Phase 2: Layer-by-layer ---
        layer_pbar = tqdm(range(num_layers), desc="Scan: layers")
        for layer_idx in layer_pbar:
            layer_pbar.set_postfix(layer=layer_idx)

            # Load layer weights from disk to GPU
            layer_weights = self._load_tensors(layer_map[layer_idx], "cpu")
            prefix = f"model.layers.{layer_idx}."
            layer_module = decoder_layers[layer_idx]
            _materialize_module(layer_module, layer_weights, prefix, device)
            del layer_weights

            per_signal_captures: dict[str, list[torch.Tensor]] = {
                sig: [] for sig in signals
            }

            for batch_idx, (start, end) in enumerate(batches):
                hs = batch_hidden_states[batch_idx].to(device)
                mask_dev = batch_causal_masks[batch_idx].to(device)
                pos_dev = batch_pos_ids[batch_idx].to(device)
                pe_dev = ChunkedLocalBackend._pe_to_device(position_embeddings, device)

                with torch.no_grad():
                    # Set up signal hooks
                    hooks: list[tuple[str, Any, list[torch.Tensor]]] = []
                    capture_residual = False
                    for sig in signals:
                        if sig == "residual":
                            capture_residual = True
                            continue
                        buf: list[torch.Tensor] = []
                        def _hook(
                            _mod: Any, _inp: Any, out: Any,
                            _buf: list = buf,
                        ) -> None:
                            delta = out[0] if isinstance(out, tuple) else out
                            _buf.append(delta.detach().cpu())
                        if sig == "attn_delta":
                            mod = _get_attn_submodule(layer_module)
                            h = mod.register_forward_hook(_hook)
                            hooks.append((sig, h, buf))
                        elif sig == "mlp_delta":
                            mod = _get_mlp_submodule(layer_module)
                            h = mod.register_forward_hook(_hook)
                            hooks.append((sig, h, buf))

                    layer_kwargs = ChunkedLocalBackend._build_layer_kwargs(
                        mask_dev, pos_dev, batch_cache_positions[batch_idx],
                        pe_dev, layer_types, layer_idx, device,
                    )

                    output = layer_module(hs, **layer_kwargs)
                    hs = output[0] if isinstance(output, tuple) else output

                    for sig_name, handle, hook_buf in hooks:
                        handle.remove()
                        if hook_buf:
                            per_signal_captures[sig_name].append(hook_buf[0])

                    if capture_residual:
                        per_signal_captures["residual"].append(hs.detach().cpu())

                batch_hidden_states[batch_idx] = hs.cpu()
                del hs, mask_dev, pos_dev, pe_dev

            # Free layer weights
            _free_module(layer_module)
            gc.collect()
            torch.cuda.empty_cache()

            # --- PCA for this layer ---
            for sig_idx, sig_name in enumerate(signals):
                captures = per_signal_captures[sig_name]
                if not captures:
                    continue

                stacked = torch.cat(captures, dim=0)  # [B_total, S, dim]
                B_total, S, dim = stacked.shape
                flat = stacked.reshape(-1, dim).float().numpy()

                if sig_name not in signal_dims:
                    signal_dims[sig_name] = dim

                if external_bases is not None and sig_name in external_bases:
                    basis = external_bases[sig_name][layer_idx]
                    signal_bases[sig_name][layer_idx] = basis
                    k = basis.shape[1]
                    projected = (flat @ basis.astype(np.float32)).astype(np.float16)
                else:
                    if generative_masks is not None:
                        fit_mask_parts = []
                        for sid in range(B_total):
                            if sid < len(generative_masks) and generative_masks[sid] is not None:
                                gmask = generative_masks[sid]
                                padded_mask = np.zeros(S, dtype=bool)
                                padded_mask[:min(len(gmask), S)] = gmask[:S]
                                fit_mask_parts.append(padded_mask)
                            else:
                                fit_mask_parts.append(np.ones(S, dtype=bool))
                        fit_mask = np.concatenate(fit_mask_parts)
                        flat_fit = flat[fit_mask]
                    else:
                        flat_fit = flat

                    k = min(n_components, flat_fit.shape[0] - 1, dim)
                    pca = PCA(n_components=k)
                    pca.fit(flat_fit)

                    basis = pca.components_.T.astype(np.float16)
                    signal_bases[sig_name][layer_idx] = basis
                    projected = pca.transform(flat).astype(np.float16)

                if k < n_components:
                    padded = np.zeros(
                        (projected.shape[0], n_components), dtype=np.float16,
                    )
                    padded[:, :k] = projected
                    projected = padded

                all_proj_chunks.append(projected)

                for sample_idx in range(B_total):
                    for tok in range(S):
                        all_coord_sample_id.append(sample_idx)
                        all_coord_layer.append(layer_idx)
                        all_coord_token_pos.append(tok)
                        all_coord_signal.append(sig_idx)

                del stacked, flat, projected

            del per_signal_captures
            gc.collect()

        # Assemble bases
        final_bases: dict[str, np.ndarray] = {}
        for sig_name in signals:
            dim = signal_dims.get(sig_name, hidden_dim)
            k_eff = min(n_components, dim)
            basis_arr = np.zeros((num_layers, dim, k_eff), dtype=np.float16)
            for L in range(num_layers):
                layer_basis = signal_bases[sig_name][L]
                if layer_basis is not None:
                    actual_k = layer_basis.shape[1]
                    basis_arr[L, :, :actual_k] = layer_basis
            final_bases[sig_name] = basis_arr

        all_projections = np.concatenate(all_proj_chunks, axis=0)
        all_projections = all_projections[:, np.newaxis, :]

        metadata = {
            "model_id": self.model_name,
            "hidden_dim": hidden_dim,
            "n_layers": num_layers,
            "n_components": n_components,
            "n_samples": n_samples,
            "signals": signals,
        }

        coords = {
            "sample_id": all_coord_sample_id,
            "layer": all_coord_layer,
            "token_pos": all_coord_token_pos,
            "signal": all_coord_signal,
        }

        return (
            metadata,
            final_bases,
            all_projections,
            coords,
            token_ids_per_sample,
            seq_lengths,
            all_attention_mask,
            signal_dims,
        )

    def project_forward(
        self,
        prompt: str,
        bases: dict[str, np.ndarray],
        signals: list[str],
        include_logits: bool = True,
    ) -> tuple[np.ndarray, list[int], torch.Tensor | None]:
        """Run a single-prompt forward pass from disk, projecting deltas onto bases.

        Same interface as ChunkedLocalBackend.project_forward but loads
        layer weights from safetensors instead of keeping the model in RAM.
        """
        import gc

        config = self._get_config()
        model = self._get_model_skeleton()
        layer_map, non_layer = self._get_shard_map()
        device = self.device
        num_layers = config.num_hidden_layers

        # Tokenize
        tokenized = self.tokenizer(
            [prompt], return_tensors="pt", padding=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        seq_len = input_ids.shape[1]
        real_len = int(attention_mask[0].sum().item())
        token_ids = input_ids[0, :real_len].tolist()

        # Embedding
        embed_tensors = [(n, f) for n, f in non_layer if "embed_tokens" in n]
        embed_weights = self._load_tensors(embed_tensors, device)
        embed = _get_embedding_module(model)
        _materialize_module(embed, embed_weights, "model.embed_tokens.", device)
        del embed_weights

        with torch.no_grad():
            hidden_states = embed(input_ids.to(device)).cpu()

        _free_module(embed)
        torch.cuda.empty_cache()

        # Position setup
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        cache_position = torch.arange(seq_len)

        # Rotary embeddings
        position_embeddings: tuple[torch.Tensor, ...] | torch.Tensor | dict | None = None
        layer_types: list[str] | None = None
        rotary_name = ChunkedLocalBackend._find_rotary_embedding_name(model)
        if rotary_name is not None:
            rotary_mod = model
            for part in rotary_name.split("."):
                rotary_mod = getattr(rotary_mod, part)

            text_cfg = getattr(config, "text_config", config)
            dim = getattr(text_cfg, "qk_rope_head_dim", None)
            if dim is None:
                head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
                dim = head_dim
            base = getattr(text_cfg, "rope_theta", 10000.0)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
            rotary_mod.to_empty(device=device)
            rotary_mod.inv_freq = inv_freq

            layer_types_cfg = getattr(text_cfg, "layer_types", None)
            unique_types = set(layer_types_cfg) if layer_types_cfg else set()
            if layer_types_cfg and len(unique_types) > 1:
                layer_types = list(layer_types_cfg)
                position_embeddings = {}
                with torch.no_grad():
                    for lt in unique_types:
                        pe = rotary_mod(
                            hidden_states.to(device),
                            position_ids.to(device),
                            layer_type=lt,
                        )
                        position_embeddings[lt] = tuple(t.cpu() for t in pe)
            else:
                with torch.no_grad():
                    pe = rotary_mod(
                        hidden_states.to(device),
                        position_ids.to(device),
                    )
                    if isinstance(pe, tuple):
                        position_embeddings = tuple(t.cpu() for t in pe)
                    else:
                        position_embeddings = pe.cpu()

            _free_module(rotary_mod)
            torch.cuda.empty_cache()

        decoder_layers = _get_decoder_layers(model)

        # Projection setup
        max_k = max(b.shape[2] for b in bases.values())
        n_signals = len(signals)
        projections = np.zeros(
            (seq_len, num_layers, n_signals, max_k), dtype=np.float32,
        )
        bases_torch = {
            sig: torch.from_numpy(b).float() for sig, b in bases.items()
        }

        causal_mask = _make_causal_mask(attention_mask, self.dtype)

        # Layer-by-layer
        for layer_idx in range(num_layers):
            layer_weights = self._load_tensors(layer_map[layer_idx], "cpu")
            prefix = f"model.layers.{layer_idx}."
            layer_module = decoder_layers[layer_idx]
            _materialize_module(layer_module, layer_weights, prefix, device)
            del layer_weights

            # Signal hooks
            hooks: list[tuple[str, Any, list[torch.Tensor]]] = []
            capture_residual = False
            for sig in signals:
                if sig == "residual":
                    capture_residual = True
                    continue
                buf: list[torch.Tensor] = []
                def _hook(
                    _mod: Any, _inp: Any, out: Any,
                    _buf: list = buf,
                ) -> None:
                    delta = out[0] if isinstance(out, tuple) else out
                    _buf.append(delta.detach().cpu())
                if sig == "attn_delta":
                    mod = _get_attn_submodule(layer_module)
                    h = mod.register_forward_hook(_hook)
                    hooks.append((sig, h, buf))
                elif sig == "mlp_delta":
                    mod = _get_mlp_submodule(layer_module)
                    h = mod.register_forward_hook(_hook)
                    hooks.append((sig, h, buf))

            hs = hidden_states.to(device)
            mask_dev = causal_mask.to(device)
            pos_dev = position_ids.to(device)
            pe_dev = ChunkedLocalBackend._pe_to_device(position_embeddings, device)

            layer_kwargs = ChunkedLocalBackend._build_layer_kwargs(
                mask_dev, pos_dev, cache_position,
                pe_dev, layer_types, layer_idx, device,
            )

            with torch.no_grad():
                output = layer_module(hs, **layer_kwargs)
                hs = output[0] if isinstance(output, tuple) else output

            for sig_name, handle, hook_buf in hooks:
                handle.remove()
                if hook_buf and sig_name in bases_torch:
                    sig_idx = signals.index(sig_name)
                    delta = hook_buf[0][0].float()  # [seq_len, dim]
                    b = bases_torch[sig_name][layer_idx]
                    k = b.shape[1]
                    proj = (delta @ b).numpy()
                    projections[:, layer_idx, sig_idx, :k] = proj

            if capture_residual and "residual" in bases_torch:
                sig_idx = signals.index("residual")
                delta = hs[0].cpu().float()
                b = bases_torch["residual"][layer_idx]
                k = b.shape[1]
                proj = (delta @ b).numpy()
                projections[:, layer_idx, sig_idx, :k] = proj

            hidden_states = hs.cpu()
            del hs, mask_dev, pos_dev, pe_dev

            _free_module(layer_module)
            gc.collect()
            torch.cuda.empty_cache()

        # Logits
        logits = None
        if include_logits:
            norm_tensors = [
                (n, f) for n, f in non_layer
                if "norm" in n and "layer" not in n
            ]
            head_tensors = [(n, f) for n, f in non_layer if "lm_head" in n]

            norm_w = self._load_tensors(norm_tensors, device)
            final_norm = _get_final_norm(model)
            _materialize_module(final_norm, norm_w, "model.norm.", device)
            del norm_w

            head_w = self._load_tensors(head_tensors, device)
            lm_head = _get_lm_head(model)
            _materialize_module(lm_head, head_w, "lm_head.", device)
            del head_w

            with torch.no_grad():
                logits = lm_head(
                    final_norm(hidden_states.to(device)),
                ).cpu()

            _free_module(final_norm)
            _free_module(lm_head)
            torch.cuda.empty_cache()

        return projections, token_ids, logits

    # --- Standard ExtractionBackend interface ---

    def extract_batch(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .activation_types import ExtractionSpec
        spec = ExtractionSpec(hidden_layers=layer_indices)
        result = self.extract_all(prompts, spec, batch_size=len(prompts))
        assert result.activations is not None
        return result.activations, result.attention_mask

    def extract_batch_with_logits(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        from .activation_types import ExtractionSpec
        spec = ExtractionSpec(hidden_layers=layer_indices, include_logits=True)
        result = self.extract_all(prompts, spec, batch_size=len(prompts))
        assert result.logits is not None
        return result.activations, result.attention_mask, result.logits, None

    def extract_batch_extended(
        self,
        prompts: list[str],
        spec: ExtractionSpec,
        **kwargs: Any,
    ) -> ExtractedBatch:
        return self.extract_all(prompts, spec, batch_size=len(prompts))


def resolve_backend(
    backend: str,
    model_name: str,
    device: str = "auto",
    remote: bool = False,
    dtype: torch.dtype | None = None,
    chunk_size: int | str | None = None,
) -> ExtractionBackend:
    """Create an ExtractionBackend from a string identifier.

    Parameters
    ----------
    backend : str
        Backend identifier: ``"nnsight"``, ``"local"``, ``"chunked"``,
        or ``"disk_offload"``.
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification.
    remote : bool
        Whether to use remote execution (only valid for nnsight backend).
    dtype : torch.dtype or None
        Model dtype for local/chunked backend (e.g., torch.float32, torch.bfloat16).
        Defaults to torch.float32 for local, torch.bfloat16 for chunked.
        Ignored for nnsight backend.
    chunk_size : int or str or None
        Number of layers per chunk for ``backend="chunked"``.
        ``"auto"`` estimates from available VRAM. Ignored for other backends.

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
    elif backend == "chunked":
        if remote:
            raise ValueError(
                "backend='chunked' does not support remote=True. "
                "Use backend='nnsight' for remote execution."
            )
        chunked_dtype = dtype if dtype is not None else torch.bfloat16
        cs = chunk_size if chunk_size is not None else "auto"
        return ChunkedLocalBackend(model_name, device, dtype=chunked_dtype, chunk_size=cs)
    elif backend == "disk_offload":
        if remote:
            raise ValueError(
                "backend='disk_offload' does not support remote=True."
            )
        offload_dtype = dtype if dtype is not None else torch.bfloat16
        return DiskOffloadBackend(model_name, device, dtype=offload_dtype)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Available backends: 'nnsight', 'local', 'chunked', 'disk_offload'."
        )
