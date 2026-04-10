"""Pluggable extraction backends for lmprobe.

This module defines the ExtractionBackend ABC and provides two implementations:
- NnsightBackend: Uses nnsight for model loading and activation extraction
  (supports both local and remote/NDIF execution)
- LocalBackend: Uses HuggingFace transformers directly with register_forward_hook
  (local-only, no nnsight dependency for extraction)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

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
                def make_router_hook(idx: int) -> Any:
                    def hook_fn(_module: Any, _input: Any, output: Any) -> None:
                        if isinstance(output, tuple):
                            captured_router[idx] = output[0].detach()
                        else:
                            captured_router[idx] = output.detach()
                    return hook_fn

                hooks.append(
                    router_module.register_forward_hook(make_router_hook(layer_idx))
                )

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
            router_logits = {
                idx: t.cpu() for idx, t in captured_router.items()
            }

        return ExtractedBatch(
            activations=activations,
            attention_mask=tokenized["attention_mask"],
            logits=logits,
            logits_indices=None,
            router_logits=router_logits,
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

    # Causal mask: lower triangular
    causal = torch.triu(
        torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype),
        diagonal=1,
    )
    # Expand to (1, 1, seq, seq)
    causal = causal.unsqueeze(0).unsqueeze(0)

    # Padding mask: (batch, 1, 1, seq) — mask out padding key positions
    padding = attention_mask_2d[:, None, None, :].to(dtype)
    padding = (1.0 - padding) * torch.finfo(dtype).min

    # Combine: broadcast (1,1,seq,seq) + (batch,1,1,seq) → (batch,1,seq,seq)
    return causal + padding


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
    hidden_size = config.hidden_size
    intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)

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
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        chunk_size: int | str = "auto",
    ):
        super().__init__(model_name, device)
        self.dtype = dtype
        self._chunk_size = chunk_size
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._config: Any = None

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
        if isinstance(self._chunk_size, int):
            return self._chunk_size
        return _estimate_chunk_size(self.model_name, self.device, self.dtype)

    def _load_full_model_cpu(self) -> Any:
        """Load the full model on CPU with eager attention.

        For the chunked backend, the model is always loaded fully on CPU.
        The chunking benefit comes from only moving subset of layers to
        GPU at a time — for outsized models, CPU RAM is sufficient to
        hold the full weights while GPU VRAM is not.
        """
        if not hasattr(self, "_full_model"):
            from transformers import AutoModelForCausalLM

            self._full_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                attn_implementation="eager",
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

        # Build position_ids and causal mask
        position_ids = attention_mask_2d.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask_2d == 0, 1)
        causal_mask = _make_causal_mask(attention_mask_2d, self.dtype)

        # Compute rotary position embeddings
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
        rotary_name = self._find_rotary_embedding_name(model)
        if rotary_name is not None:
            rotary_mod = model
            for part in rotary_name.split("."):
                rotary_mod = getattr(rotary_mod, part)
            rotary_mod.to(device)
            with torch.no_grad():
                pe = rotary_mod(hidden_states.to(device), position_ids.to(device))
                position_embeddings = (pe[0].cpu(), pe[1].cpu())
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
            pe_dev = None
            if position_embeddings is not None:
                pe_dev = (
                    position_embeddings[0].to(device),
                    position_embeddings[1].to(device),
                )

            with torch.no_grad():
                for layer_idx in range(chunk_start, chunk_end):
                    layer_module = decoder_layers[layer_idx]

                    # Router hook if requested
                    rh = None
                    if layer_idx in router_target_set and router_module_template:
                        router_hook_output: list[torch.Tensor] = []

                        def _router_hook(
                            _mod: Any, _inp: Any, out: Any,
                            _buf: list = router_hook_output,
                        ) -> None:
                            if isinstance(out, tuple):
                                _buf.append(out[0].detach().cpu())
                            else:
                                _buf.append(out.detach().cpu())

                        router_path = router_module_template.format(layer=layer_idx)
                        router_mod = model
                        for part in router_path.split("."):
                            if part.isdigit():
                                router_mod = router_mod[int(part)]
                            else:
                                router_mod = getattr(router_mod, part)
                        rh = router_mod.register_forward_hook(_router_hook)

                    layer_kwargs: dict[str, Any] = {
                        "attention_mask": mask_dev,
                        "position_ids": pos_dev,
                        "use_cache": False,
                    }
                    if pe_dev is not None:
                        layer_kwargs["position_embeddings"] = pe_dev

                    output = layer_module(hs, **layer_kwargs)
                    hs = output[0] if isinstance(output, tuple) else output

                    if layer_idx in target_set:
                        captured[layer_idx] = hs.detach().cpu()

                    if rh is not None:
                        rh.remove()
                        if router_hook_output:
                            captured_router[layer_idx] = router_hook_output[0]

            hidden_states = hs.cpu()
            del hs, mask_dev, pos_dev, pe_dev

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
        return None

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        activations, attention_mask, logits, _ = self._chunked_forward(
            prompts, layer_indices, include_logits=True,
        )
        assert activations is not None
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
        )
        return ExtractedBatch(
            activations=activations,
            attention_mask=attention_mask,
            logits=logits,
            logits_indices=None,
            router_logits=router_logits,
        )


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
        Backend identifier: ``"nnsight"``, ``"local"``, or ``"chunked"``.
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
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Available backends: 'nnsight', 'local', 'chunked'."
        )
