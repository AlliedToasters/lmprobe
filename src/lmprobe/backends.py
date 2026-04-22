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

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import PreTrainedTokenizerBase

    from .activation_types import ExtractedBatch, ExtractionSpec

from .activation_types import PreTokenizedPrompts


_logger = logging.getLogger(__name__)


# --- spec 004: memmap-backed batch_hidden_states ---------------------------
#
# `np.memmap` has no native bfloat16. For bf16 we store bytes as uint16
# (same width) and reinterpret via `torch.Tensor.view(torch.bfloat16)` on
# read / `.view(torch.uint16)` on write. fp16/fp32/fp64 map directly.

_TORCH_TO_MMAP_NUMPY: dict[torch.dtype, Any] = {
    torch.bfloat16: np.uint16,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
}


def _mmap_np_dtype(torch_dtype: torch.dtype) -> Any:
    try:
        return _TORCH_TO_MMAP_NUMPY[torch_dtype]
    except KeyError as e:
        raise ValueError(
            f"unsupported dtype for memmap residuals: {torch_dtype}"
        ) from e


def _mmap_read_slice(
    mmap: np.memmap, start: int, end: int, torch_dtype: torch.dtype,
) -> torch.Tensor:
    """Read `mmap[start:end]` into a detached torch tensor with the
    correct float dtype. The copy is intentional — we don't want the
    returned tensor (which will be moved to GPU) to pin memmap pages."""
    np_slice = np.array(mmap[start:end], copy=True)
    t = torch.from_numpy(np_slice)
    if torch_dtype == torch.bfloat16:
        t = t.view(torch.bfloat16)
    return t


def _mmap_write_slice(
    mmap: np.memmap, start: int, end: int, tensor: torch.Tensor,
) -> None:
    """Write `tensor` (CPU, any float dtype) into `mmap[start:end]`."""
    t = tensor.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.uint16)
    mmap[start:end] = t.numpy()


_CUML_CHECKED = False
_CUML_AVAILABLE = False


def _cuml_available() -> bool:
    """Return True if cuml + cupy are importable and a CUDA device is
    present. Result is cached after first check."""
    global _CUML_CHECKED, _CUML_AVAILABLE
    if _CUML_CHECKED:
        return _CUML_AVAILABLE
    _CUML_CHECKED = True
    if not torch.cuda.is_available():
        _CUML_AVAILABLE = False
        return False
    try:
        import cuml  # noqa: F401
        import cupy  # noqa: F401
        _CUML_AVAILABLE = True
        _logger.info(
            "lmprobe: cuml + cupy detected — scan-time PCA will run on GPU",
        )
    except ImportError:
        _CUML_AVAILABLE = False
    return _CUML_AVAILABLE


def _fit_project_scan_pca(
    captures: list[torch.Tensor],
    *,
    device: str,
    n_components: int,
    generative_masks: list[np.ndarray] | None,
    sample_id_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Fit a PCA basis on the union of capture tensors and project every
    token through the resulting basis.

    On a CUDA box with cuml/cupy installed, this runs entirely on GPU
    (zero-copy from torch via dlpack). Captures are moved to GPU one at
    a time; the caller is responsible for clearing the source list after
    this function returns. The full `[B_total, S, dim]` tensor is never
    materialized on CPU.

    Without cuml, falls back to the original sklearn-on-CPU path.

    Parameters
    ----------
    captures : list of torch.Tensor
        Per-batch capture tensors, each ``[B_i, S, dim]``, on CPU (bf16
        or fp16 typically).
    device : str
        CUDA device string (e.g. ``"cuda:0"``). Used when cuml is
        available.
    n_components : int
        Target PCA rank. Effective ``k = min(n_components, n_fit_rows - 1, dim)``.
    generative_masks : list of np.ndarray or None
        Per-sample boolean masks (``True`` = fit on this token). When
        provided, PCA is fit only on the True rows; all rows are still
        projected through the resulting basis.
    sample_id_offset : int
        Offset to apply when indexing into ``generative_masks``
        (used when captures correspond to a sub-range of the corpus).

    Returns
    -------
    basis_fp16 : np.ndarray, shape [dim, k]
        PCA basis as a column-of-components matrix (so that
        ``data @ basis`` yields projections).
    projected_fp16 : np.ndarray, shape [B_total * S, k]
        Projections of every token through the basis.
    B_total, S, dim : int
        Shape components of the concatenated capture tensor.
    """
    use_cuml = _cuml_available()

    # Shape discovery (cheap — no data movement).
    B_total = sum(int(c.shape[0]) for c in captures)
    S = int(captures[0].shape[1])
    dim = int(captures[0].shape[2])

    # Build the PCA-fit mask in CPU numpy regardless of backend — it's
    # tiny (B_total * S bools) and shared across paths.
    if generative_masks is not None:
        mask_parts: list[np.ndarray] = []
        for sid in range(B_total):
            gid = sid + sample_id_offset
            if gid < len(generative_masks) and generative_masks[gid] is not None:
                gmask = generative_masks[gid]
                padded = np.zeros(S, dtype=bool)
                padded[:min(len(gmask), S)] = gmask[:S]
                mask_parts.append(padded)
            else:
                mask_parts.append(np.ones(S, dtype=bool))
        fit_mask_np = np.concatenate(mask_parts)
    else:
        fit_mask_np = None

    if use_cuml:
        import cupy as cp
        from cuml.decomposition import PCA as CuPCA  # noqa: N811
        from torch.utils.dlpack import to_dlpack

        # Move each capture to GPU (fp32 for numerical stability in PCA)
        # and concat on-device. Source CPU tensors are released as soon
        # as their GPU copy exists.
        gpu_parts: list[torch.Tensor] = []
        while captures:
            c = captures.pop(0)
            gpu_parts.append(c.to(device, dtype=torch.float32, non_blocking=True))
            del c
        stacked_gpu = torch.cat(gpu_parts, dim=0).contiguous()
        del gpu_parts
        flat_gpu = stacked_gpu.reshape(B_total * S, dim)
        # Keep stacked_gpu alive — flat_gpu is a view.

        if fit_mask_np is not None:
            fit_mask_gpu = torch.from_numpy(fit_mask_np).to(device)
            flat_fit_gpu = flat_gpu[fit_mask_gpu].contiguous()
        else:
            fit_mask_gpu = None
            flat_fit_gpu = flat_gpu

        k = min(n_components, int(flat_fit_gpu.shape[0]) - 1, dim)
        torch.cuda.empty_cache()

        # Phase 1: fit on the assistant-token subset. Zero-copy view.
        # `svd_solver="jacobi"` runs a truncated SVD for small k — GPU
        # working set is O(n × k) instead of O(n × d).
        fit_cp = cp.from_dlpack(to_dlpack(flat_fit_gpu))
        pca = CuPCA(n_components=k, output_type="cupy", svd_solver="jacobi")
        pca.fit(fit_cp)
        basis_np = cp.asnumpy(pca.components_).T.astype(np.float16)
        basis_gpu = torch.from_numpy(basis_np).to(device, dtype=torch.float32)

        # Release fit data + fit mask before projecting — keeping the
        # filtered subset live through transform doubles the GPU
        # footprint for no reason (we already have the basis).
        del fit_cp, pca, flat_fit_gpu
        if fit_mask_gpu is not None:
            del fit_mask_gpu
        torch.cuda.empty_cache()

        # Phase 2: project via matmul. torch @ is fp32 on GPU, output
        # fp16. Avoids cuml's transform allocating another large buffer.
        projected_gpu = (flat_gpu @ basis_gpu).half()
        projected_np = projected_gpu.cpu().numpy()

        del projected_gpu, basis_gpu, flat_gpu, stacked_gpu
        torch.cuda.empty_cache()
        return basis_np, projected_np, B_total, S, dim

    # --- sklearn fallback (CPU) ---
    from sklearn.decomposition import PCA

    stacked = torch.cat(captures, dim=0)
    while captures:
        captures.pop(0)
    flat = stacked.reshape(B_total * S, dim).float().numpy()
    del stacked

    if fit_mask_np is not None:
        flat_fit = flat[fit_mask_np]
    else:
        flat_fit = flat

    k = min(n_components, int(flat_fit.shape[0]) - 1, dim)
    pca = PCA(n_components=k)
    pca.fit(flat_fit)
    basis_np = pca.components_.T.astype(np.float16)
    projected_np = pca.transform(flat).astype(np.float16)
    flat_is_fit = flat_fit is flat
    del pca, flat
    if not flat_is_fit:
        del flat_fit
    return basis_np, projected_np, B_total, S, dim


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
        prompts: list[str] | PreTokenizedPrompts,
        layer_indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract activations for a batch of prompts.

        Parameters
        ----------
        prompts : list[str] or PreTokenizedPrompts
            Either raw text prompts (tokenized internally with the backend's
            defaults) or a :class:`PreTokenizedPrompts` holding caller-supplied
            ``input_ids`` and ``attention_mask``. Use the latter when you need
            exact control over tokenization.
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

    def extract_batch_pretokenized(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper: extract activations from pre-tokenized input.

        Use when you've already applied the model's chat template externally
        and need exact control over tokenization (``add_special_tokens``,
        ``padding_side``, ``pad_token``). Equivalent to:

            backend.extract_batch(
                prompts=PreTokenizedPrompts(input_ids, attention_mask),
                layer_indices=layer_indices,
                **kwargs,
            )

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape ``(B, S)``.
        attention_mask : torch.Tensor
            Shape ``(B, S)``. 1 = real token, 0 = pad.
        layer_indices : list[int]
            Layer indices to extract from.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - activations: Shape ``(B, S, hidden_dim * num_layers)``
            - attention_mask: the caller's ``attention_mask`` passed back
              unchanged
        """
        return self.extract_batch(
            prompts=PreTokenizedPrompts(
                input_ids=input_ids, attention_mask=attention_mask,
            ),
            layer_indices=layer_indices,
            **kwargs,
        )

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

        from transformers import AutoModelForCausalLM

        from ._tokenizer_utils import load_tokenizer

        tokenizer = load_tokenizer(model_name)
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

        # Concatenate activations on their native device; callers handle
        # any CPU transfer for persistence.
        activation_tensors = [captured[idx] for idx in layer_indices]
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


def _estimate_disk_offload_layers_per_group(
    model_name: str,
    device: str,
    dtype: torch.dtype,
) -> int:
    """Estimate how many transformer layers can be co-resident on GPU for disk_offload.

    Same estimation logic as :func:`_estimate_chunk_size` but reserves 40%
    headroom (instead of 30%) to leave room for activations while multiple
    layers are on device simultaneously.

    Returns
    -------
    int
        Layers per group, clamped to ``[1, num_layers]``.
    """
    from .extraction import get_num_layers_from_config

    num_layers = get_num_layers_from_config(model_name)

    if device == "cpu" or not torch.cuda.is_available():
        return num_layers

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    text_cfg = getattr(config, "text_config", config)
    hidden_size = text_cfg.hidden_size
    intermediate_size = getattr(text_cfg, "intermediate_size", hidden_size * 4)

    bytes_per_param = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    params_per_layer = 4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size
    layer_bytes = params_per_layer * bytes_per_param

    try:
        free_vram, _total = torch.cuda.mem_get_info(device)
    except Exception:
        return 1

    # Reserve 40% for activations and overhead
    available = free_vram * 0.6
    lpg = max(1, int(available / layer_bytes))
    return min(lpg, num_layers)


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
            from ._tokenizer_utils import load_tokenizer

            self._tokenizer = load_tokenizer(self.model_name)
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
        prompts: list[str] | PreTokenizedPrompts,
        layer_indices: list[int],
        include_logits: bool = False,
        router_layer_indices: list[int] | None = None,
        router_module_template: str | None = None,
        router_hook_strategy: str = "output",
        batch_size: int | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        dict[int, torch.Tensor] | None,
    ]:
        """Run a chunked forward pass, backed by the sweep primitive.

        Returns ``(activations, attention_mask, logits, router_logits)``
        matching the shape ``extract_batch`` / ``extract_batch_extended``
        expect. ``batch_size=len(prompts)`` so the entire corpus is one
        microbatch — matches caller assumptions.
        """
        from .accumulators import (
            HiddenStateCapture,
            LogitCapture,
            RouterLogitCapture,
        )
        from .sweep import sweep

        loader = ChunkedLayerLoader(self)
        if router_module_template is not None:
            loader.router_module_template = router_module_template

        accumulators: dict[str, Any] = {}
        if layer_indices:
            accumulators["hs"] = HiddenStateCapture(
                layer_indices, dtype=self.dtype,
            )
        if include_logits:
            accumulators["logits"] = LogitCapture()
        if router_layer_indices:
            accumulators["router"] = RouterLogitCapture(
                router_layer_indices, strategy=router_hook_strategy,
            )

        pretok = isinstance(prompts, PreTokenizedPrompts)

        # No accumulators subscribed ⇒ no sweep needed; just return the mask.
        if not accumulators:
            if pretok:
                return None, prompts.attention_mask, None, None
            tokenized = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
            )
            return None, tokenized["attention_mask"], None, None

        effective_bs = batch_size if batch_size is not None else len(prompts)
        out = sweep(
            prompts,
            accumulators=accumulators,
            loader=loader,
            batch_size=effective_bs,
        )

        activations = out["hs"] if "hs" in accumulators else None
        if pretok:
            attention_mask_2d = prompts.attention_mask
        else:
            tokenized = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
            )
            attention_mask_2d = tokenized["attention_mask"]
        logits_out = out.get("logits") if include_logits else None
        router_out = out.get("router") if router_layer_indices else None
        return activations, attention_mask_2d, logits_out, router_out

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

    # Valid signal names for sweep-based forward passes.
    SCAN_SIGNALS = ("residual", "attn_delta", "mlp_delta", "router_logits")

    def project_forward(
        self,
        prompt: str,
        bases: dict[str, np.ndarray],
        signals: list[str],
        include_logits: bool = True,
    ) -> tuple[np.ndarray, list[int], torch.Tensor | None]:
        """Run a single-prompt forward pass, projecting deltas onto stored bases.

        Returns
        -------
        tuple
            ``(projections, token_ids, logits)``:
            - ``projections``: ``[seq_len, n_layers, n_signals, max_k]`` float32
            - ``token_ids``: list of int token IDs (unpadded)
            - ``logits``: ``[1, seq_len, vocab_size]`` or None
        """
        from .accumulators import LogitCapture, PerTokenProjection
        from .sweep import sweep

        proj_bases = {s: bases[s] for s in signals if s in bases}
        if not proj_bases:
            raise ValueError(
                f"project_forward: no bases supplied for any of the "
                f"requested signals {signals} (bases keys: "
                f"{list(bases.keys())})."
            )

        loader = ChunkedLayerLoader(self)
        accumulators: dict[str, Any] = {
            "proj": PerTokenProjection(proj_bases),
        }
        if include_logits:
            accumulators["logits"] = LogitCapture()

        out = sweep(
            [prompt],
            accumulators=accumulators,
            loader=loader,
            external_bases=proj_bases,
            batch_size=1,
        )

        proj_result = out["proj"]
        values = proj_result["values"]               # [total_rows, k_max]
        offset_table = proj_result["offset_table"]   # [1, n_layers, n_sig, 2]
        sweep_signal_names = proj_result["signal_names"]
        seq_lengths = proj_result["seq_lengths"]

        # Rebuild the legacy dense shape [seq_len, n_layers, n_signals, max_k]
        # in the caller-supplied ``signals`` order (may differ from
        # PerTokenProjection's alphabetical order).
        seq_len = int(seq_lengths[0])
        num_layers = loader.num_layers
        max_k = int(values.shape[-1])
        sig_to_new_idx = {s: i for i, s in enumerate(sweep_signal_names)}
        dense = np.zeros(
            (seq_len, num_layers, len(signals), max_k), dtype=np.float32,
        )
        for out_si, sig_name in enumerate(signals):
            new_si = sig_to_new_idx.get(sig_name)
            if new_si is None:
                continue
            for L in range(num_layers):
                start, end = offset_table[0, L, new_si]
                rows = values[start:end].astype(np.float32)
                dense[: rows.shape[0], L, out_si, :] = rows

        tokenized = self.tokenizer([prompt], return_tensors="pt", padding=True)
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        real_len = int(attention_mask.sum().item())
        token_ids = input_ids[:real_len].tolist()

        logits_out: torch.Tensor | None = None
        if include_logits:
            logit_tensor = out["logits"]  # [1, S_max, V] or None
            if logit_tensor is not None:
                logits_out = logit_tensor[:, :real_len, :]

        return dense[:real_len], token_ids, logits_out

    def extract_batch(
        self,
        prompts: list[str] | PreTokenizedPrompts,
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
            batch_size=kwargs.get("batch_size"),
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
# ChunkedLayerLoader — LayerLoader implementation for ChunkedLocalBackend
# ---------------------------------------------------------------------------
#
# Backs the new sweep primitive (src/lmprobe/sweep.py). Reuses all of
# ChunkedLocalBackend's existing model-loading + rotary + memmap infrastructure;
# only adds protocol methods (prepare, iter_layer_groups, layer_group, read_hs,
# write_hs, build_layer_kwargs, apply_lm_head) and the `_drive_sweep` forward
# loop that sweep.py dispatches into.


class _ResidualBuffer:
    """Memmap-backed residual buffer, lifecycle-owned by ChunkedLayerLoader.prepare."""

    def __init__(self, mmap: np.memmap, tmpdir: Any) -> None:
        self.mmap = mmap
        self._tmpdir = tmpdir

    def cleanup(self) -> None:
        # Drop memmap ref first so Python releases the mmap before the
        # backing file is unlinked (matters on Windows; harmless on Linux).
        del self.mmap
        self._tmpdir.cleanup()


class ChunkedLayerLoader:
    """LayerLoader wrapping :class:`ChunkedLocalBackend`.

    Implements the sweep primitive's :class:`~lmprobe.sweep.LayerLoader`
    protocol: embedding + rotary in :meth:`prepare`, chunk-sized layer
    groups via :meth:`iter_layer_groups`, memmap-backed residual buffer
    (spec 004), and the concrete forward-loop in :meth:`_drive_sweep`.
    ``ChunkedLocalBackend`` methods (``_chunked_forward``,
    ``project_forward``, plus extract shims) are thin wrappers that
    construct a loader, declare accumulators, and call
    :func:`lmprobe.sweep.sweep` — no separate forward paths live on the
    backend.
    """

    def __init__(self, backend: ChunkedLocalBackend) -> None:
        self._backend = backend
        # LayerLoader protocol attributes:
        self.tokenizer = backend.tokenizer
        self.num_layers: int = backend._get_num_layers()
        self.hidden_dim: int = 0  # populated in prepare()
        self.dtype: torch.dtype = backend.dtype
        self.device: str = backend.device
        self.layer_types: list[str] | None = None
        self.router_module_template: str | None = None
        # Lazily-resolved chunk size (same source of truth as scan_forward).
        self._chunk_size = backend._resolve_chunk_size()

    # --- LayerLoader protocol ------------------------------------------------

    def prepare(
        self,
        prompts: list[str] | PreTokenizedPrompts,
        batch_size: int,
    ) -> Any:
        """Context manager: tokenize → embed → rotary → allocate memmap.

        Yields an :class:`~lmprobe.sweep.EmbedState`. The residual buffer's
        backing tempdir is deterministically cleaned up on exit.

        When ``prompts`` is a :class:`PreTokenizedPrompts`, the internal
        tokenizer call is skipped and the caller's tensors are used verbatim.
        """
        from contextlib import contextmanager

        from .activation_types import detect_moe_info
        from .sweep import EmbedState

        @contextmanager
        def _prepare_ctx() -> Any:
            import os
            import tempfile

            # Detect MoE routing template (surfaces for RouterLogitCapture
            # accumulators to read off `loader.router_module_template`).
            try:
                moe_info = detect_moe_info(self._backend.model_name)
                if moe_info is not None:
                    self.router_module_template = moe_info.router_module_template
            except Exception:
                pass

            if isinstance(prompts, PreTokenizedPrompts):
                all_input_ids = prompts.input_ids
                all_attention_mask = prompts.attention_mask
            else:
                # Tokenize with corpus-wide padding (matches scan_forward).
                tokenized = self.tokenizer(
                    prompts, return_tensors="pt", padding=True,
                )
                all_input_ids = tokenized["input_ids"]
                all_attention_mask = tokenized["attention_mask"]

            n_samples = int(all_input_ids.shape[0])

            token_ids_per_sample: list[list[int]] = []
            seq_lengths: list[int] = []
            for i in range(n_samples):
                real_len = int(all_attention_mask[i].sum().item())
                seq_lengths.append(real_len)
                token_ids_per_sample.append(
                    all_input_ids[i, :real_len].tolist(),
                )

            batches: list[tuple[int, int]] = [
                (s, min(s + batch_size, n_samples))
                for s in range(0, n_samples, batch_size)
            ]

            model = self._backend._load_full_model_cpu()
            device = self.device
            embed = _get_embedding_module(model)
            embed.to(device)

            pos_ids_per_batch: list[torch.Tensor] = []
            cache_positions_per_batch: list[torch.Tensor] = []

            tmpdir = tempfile.TemporaryDirectory(prefix="lmprobe_sweep_")
            try:
                hs_path = os.path.join(tmpdir.name, "residuals.bin")
                S_max = int(all_input_ids.shape[1])
                mmap: np.memmap | None = None
                hidden_dim: int | None = None

                for start, end in batches:
                    ids = all_input_ids[start:end]
                    mask = all_attention_mask[start:end]
                    with torch.no_grad():
                        hs = embed(ids.to(device)).cpu()
                    if hidden_dim is None:
                        hidden_dim = int(hs.shape[-1])
                    if mmap is None:
                        mmap = np.memmap(
                            hs_path,
                            dtype=_mmap_np_dtype(self.dtype),
                            mode="w+",
                            shape=(n_samples, S_max, hidden_dim),
                        )
                    _mmap_write_slice(mmap, start, end, hs)
                    del hs
                    pos_ids = mask.long().cumsum(-1) - 1
                    pos_ids.masked_fill_(mask == 0, 1)
                    pos_ids_per_batch.append(pos_ids)
                    cache_positions_per_batch.append(
                        torch.arange(ids.shape[1]),
                    )
                embed.to("cpu")
                assert mmap is not None and hidden_dim is not None
                self.hidden_dim = hidden_dim

                # Rotary phase — same logic as scan_forward. Broadcast via
                # a single-sample slice of the just-embedded residuals.
                position_embeddings, layer_types = self._compute_rotary(
                    model, mmap, pos_ids_per_batch, device,
                )
                self.layer_types = layer_types

                state = EmbedState(
                    input_ids=all_input_ids,
                    attention_mask=all_attention_mask,
                    batches=batches,
                    pos_ids_per_batch=pos_ids_per_batch,
                    cache_positions_per_batch=cache_positions_per_batch,
                    position_embeddings=position_embeddings,
                    layer_types=layer_types,
                    seq_lengths=seq_lengths,
                    token_ids_per_sample=token_ids_per_sample,
                    hidden_dim=hidden_dim,
                    residual_buffer=_ResidualBuffer(mmap, tmpdir),
                )
            except Exception:
                tmpdir.cleanup()
                raise

            try:
                yield state
            finally:
                state.residual_buffer.cleanup()

        return _prepare_ctx()

    def _compute_rotary(
        self,
        model: Any,
        mmap: np.memmap,
        pos_ids_per_batch: list[torch.Tensor],
        device: str,
    ) -> tuple[Any, list[str] | None]:
        """Port of the rotary block in scan_forward (lines ~1978–2016)."""
        rotary_name = ChunkedLocalBackend._find_rotary_embedding_name(model)
        if rotary_name is None:
            return None, None
        rotary_mod = model
        for part in rotary_name.split("."):
            rotary_mod = getattr(rotary_mod, part)
        rotary_mod.to(device)

        text_cfg = getattr(
            getattr(model, "config", None), "text_config",
            getattr(model, "config", None),
        )
        layer_types_cfg = getattr(text_cfg, "layer_types", None)
        pe_hs = _mmap_read_slice(mmap, 0, 1, self.dtype).to(device)
        pe_pos = pos_ids_per_batch[0][:1].to(device)

        unique_types = set(layer_types_cfg) if layer_types_cfg is not None else set()
        position_embeddings: Any
        layer_types: list[str] | None
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
            else:
                layer_types = None
            with torch.no_grad():
                pe = rotary_mod(pe_hs, pe_pos)
                if isinstance(pe, tuple):
                    position_embeddings = tuple(t.cpu() for t in pe)
                else:
                    position_embeddings = pe.cpu()
        rotary_mod.to("cpu")
        return position_embeddings, layer_types

    def iter_layer_groups(self) -> Any:
        for start in range(0, self.num_layers, self._chunk_size):
            yield list(range(start, min(start + self._chunk_size, self.num_layers)))

    def layer_group(self, indices: list[int]) -> Any:
        from contextlib import contextmanager

        @contextmanager
        def _ctx() -> Any:
            import gc

            model = self._backend._load_full_model_cpu()
            decoder_layers = _get_decoder_layers(model)
            for i in indices:
                decoder_layers[i].to(self.device)
            try:
                yield [decoder_layers[i] for i in indices]
            finally:
                for i in indices:
                    decoder_layers[i].to("cpu")
                gc.collect()
                if torch.cuda.is_available() and self.device != "cpu":
                    torch.cuda.empty_cache()
        return _ctx()

    def read_hs(
        self,
        state: Any,
        start: int,
        end: int,
        device: str,
    ) -> torch.Tensor:
        return _mmap_read_slice(
            state.residual_buffer.mmap, start, end, self.dtype,
        ).to(device)

    def write_hs(
        self,
        state: Any,
        start: int,
        end: int,
        hs: torch.Tensor,
    ) -> None:
        _mmap_write_slice(state.residual_buffer.mmap, start, end, hs)

    def build_layer_kwargs(
        self,
        state: Any,
        batch_idx: int,
        layer_idx: int,
        causal_mask_dev: torch.Tensor,
        device: str,
    ) -> dict[str, Any]:
        pos_dev = state.pos_ids_per_batch[batch_idx].to(device)
        cache_position = state.cache_positions_per_batch[batch_idx]
        pe_dev = ChunkedLocalBackend._pe_to_device(
            state.position_embeddings, device,
        )
        return ChunkedLocalBackend._build_layer_kwargs(
            causal_mask_dev, pos_dev, cache_position,
            pe_dev, state.layer_types, layer_idx, device,
        )

    def apply_lm_head(
        self,
        final_hs: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Apply final_norm + lm_head to a batch of residuals."""
        model = self._backend._load_full_model_cpu()
        final_norm = _get_final_norm(model)
        lm_head = _get_lm_head(model)
        final_norm.to(device)
        lm_head.to(device)
        try:
            with torch.no_grad():
                logits = lm_head(final_norm(final_hs))
        finally:
            final_norm.to("cpu")
            lm_head.to("cpu")
        assert isinstance(logits, torch.Tensor)
        return logits

    # --- Sweep driver --------------------------------------------------------

    def _drive_sweep(
        self,
        *,
        state: Any,
        accumulators: Any,
        plans: dict[tuple[int, str], Any],
        basis_gpu: dict[str, torch.Tensor],
        batch_size: int,
        ctx: Any,
    ) -> None:
        """Layer-chunk + batch forward loop. Hooks emit signals;
        :func:`sweep._dispatch_plan` routes each capture to subscribers."""
        from tqdm import tqdm

        from .sweep import (
            SIGNAL_ATTN_DELTA,
            SIGNAL_LOGITS,
            SIGNAL_MLP_DELTA,
            SIGNAL_RESIDUAL,
            SIGNAL_ROUTER_LOGITS,
            _dispatch_plan,
            _notify_group_complete,
        )

        # Router strategy: unify across all router-subscribing accumulators.
        # Convention: the accumulator exposes a `strategy` attribute (e.g.
        # RouterLogitCapture). Two subscribers with different strategies is
        # a caller bug — the hook can only fire one way per (layer, module),
        # so a silent "first-wins" would drop the other capture's data.
        router_strategy: str | None = None
        router_owners: list[str] = []
        for name, acc in accumulators.items():
            if SIGNAL_ROUTER_LOGITS in acc.signals and hasattr(acc, "strategy"):
                s = acc.strategy
                if router_strategy is None:
                    router_strategy = s
                    router_owners = [name]
                elif s != router_strategy:
                    router_owners.append(name)
                    raise ValueError(
                        f"sweep: conflicting router_logits hook strategies "
                        f"across subscribers {router_owners!r}: "
                        f"{router_strategy!r} vs {s!r}. A single sweep can "
                        f"only install one hook per layer; run two sweeps "
                        f"or unify the strategy."
                    )
                else:
                    router_owners.append(name)
        if router_strategy is None:
            router_strategy = "output"

        device = self.device
        model = self._backend._load_full_model_cpu()

        group_iter = list(self.iter_layer_groups())
        for chunk_idx, chunk_indices in enumerate(tqdm(
            group_iter,
            desc="Sweep: layer groups",
        )):
            with self.layer_group(chunk_indices) as layer_modules_in_group:
                # Map layer_idx -> module for O(1) lookup in the batch loop.
                layer_modules = {
                    layer_idx: layer_modules_in_group[i]
                    for i, layer_idx in enumerate(chunk_indices)
                }

                for batch_idx, (start, end) in enumerate(state.batches):
                    hs = self.read_hs(state, start, end, device)
                    mask_dev = _make_causal_mask(
                        state.attention_mask[start:end], self.dtype,
                    ).to(device)
                    sample_ids = np.arange(start, end, dtype=np.int32)
                    attn_mask_np = (
                        state.attention_mask[start:end].cpu().numpy().astype(bool)
                    )

                    with torch.no_grad():
                        for layer_idx in chunk_indices:
                            layer_module = layer_modules[layer_idx]

                            # Register hooks only for signals that have
                            # subscribers at this layer.
                            hook_handles: list[Any] = []
                            hook_bufs: dict[str, list[torch.Tensor]] = {}

                            for sig in (
                                SIGNAL_ATTN_DELTA,
                                SIGNAL_MLP_DELTA,
                                SIGNAL_ROUTER_LOGITS,
                            ):
                                if (layer_idx, sig) not in plans:
                                    continue
                                buf: list[torch.Tensor] = []
                                hook_bufs[sig] = buf
                                handle = _install_sweep_hook(
                                    sig, layer_module, layer_idx, model,
                                    self.router_module_template,
                                    router_strategy, buf,
                                )
                                if handle is not None:
                                    hook_handles.append((sig, handle))

                            layer_kwargs = self.build_layer_kwargs(
                                state, batch_idx, layer_idx, mask_dev, device,
                            )
                            output = layer_module(hs, **layer_kwargs)
                            hs = output[0] if isinstance(output, tuple) else output

                            # Dispatch hooked signals. Slice basis per
                            # layer — basis_gpu[sig] is [n_layers, H, k].
                            for sig, handle in hook_handles:
                                handle.remove()
                                buf = hook_bufs[sig]
                                if not buf:
                                    continue
                                plan = plans.get((layer_idx, sig))
                                if plan is None:
                                    continue
                                sig_basis = basis_gpu.get(sig)
                                per_layer_basis = (
                                    sig_basis[layer_idx] if sig_basis is not None
                                    else None
                                )
                                _dispatch_plan(
                                    plan, buf[0], sig, layer_idx,
                                    sample_ids, attn_mask_np,
                                    per_layer_basis,
                                )
                                buf.clear()

                            # Residual = layer output (no hook needed).
                            res_plan = plans.get((layer_idx, SIGNAL_RESIDUAL))
                            if res_plan is not None:
                                res_basis = basis_gpu.get(SIGNAL_RESIDUAL)
                                res_per_layer = (
                                    res_basis[layer_idx] if res_basis is not None
                                    else None
                                )
                                _dispatch_plan(
                                    res_plan, hs.detach(), SIGNAL_RESIDUAL,
                                    layer_idx, sample_ids, attn_mask_np,
                                    res_per_layer,
                                )

                    self.write_hs(state, start, end, hs)
                    del hs, mask_dev

            # Layer group is unloaded; fire completion hook so PCAFit can
            # fit-and-free incrementally.
            _notify_group_complete(accumulators, chunk_indices)

        # End-of-sweep: logits. Any accumulator subscribing to SIGNAL_LOGITS
        # receives one dispatch per microbatch with the lm_head output.
        wants_logits = any(
            SIGNAL_LOGITS in a.signals for a in accumulators.values()
        )
        if wants_logits:
            for batch_idx, (start, end) in enumerate(state.batches):
                hs = self.read_hs(state, start, end, device)
                logits = self.apply_lm_head(hs, device)
                sample_ids = np.arange(start, end, dtype=np.int32)
                attn_mask_np = (
                    state.attention_mask[start:end].cpu().numpy().astype(bool)
                )
                logits_cpu = logits.detach().cpu()
                for acc in accumulators.values():
                    if SIGNAL_LOGITS in acc.signals:
                        acc.update(
                            logits_cpu, SIGNAL_LOGITS, -1,
                            sample_ids, attn_mask_np,
                        )
                del hs, logits, logits_cpu


def _install_sweep_hook(
    sig: str,
    layer_module: Any,
    layer_idx: int,
    model: Any,
    router_module_template: str | None,
    router_strategy: str,
    buf: list[torch.Tensor],
) -> Any:
    """Register a forward hook for a sweep signal on the appropriate
    submodule. Returns the hook handle, or ``None`` if the signal is
    unavailable for this layer (e.g. router_logits on a dense layer)."""
    if sig == "attn_delta":
        def _hook(_mod: Any, _inp: Any, out: Any) -> None:
            delta = out[0] if isinstance(out, tuple) else out
            buf.append(delta.detach())
        mod = _get_attn_submodule(layer_module)
        return mod.register_forward_hook(_hook)
    if sig == "mlp_delta":
        def _hook(_mod: Any, _inp: Any, out: Any) -> None:
            delta = out[0] if isinstance(out, tuple) else out
            buf.append(delta.detach())
        mod = _get_mlp_submodule(layer_module)
        return mod.register_forward_hook(_hook)
    if sig == "router_logits":
        if router_module_template is None:
            return None
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
        except (AttributeError, IndexError):
            return None

        if router_strategy == "input_gate":
            # DeepSeek-style: compute logits from module input × gate weight.
            def _hook_ig(mod: Any, args: Any, out: Any) -> None:
                hs_in = args[0] if isinstance(args, tuple) else args
                gate_w = mod.gate.weight
                logits = torch.nn.functional.linear(
                    hs_in.to(gate_w.dtype), gate_w,
                )
                buf.append(logits.detach())
            return router_mod.register_forward_hook(_hook_ig)
        else:
            def _hook_out(_mod: Any, _inp: Any, out: Any) -> None:
                if isinstance(out, tuple):
                    buf.append(out[0].detach())
                else:
                    buf.append(out.detach())
            return router_mod.register_forward_hook(_hook_out)
    return None


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
        layers_per_group: int | str = "auto",
    ):
        super().__init__(model_name, device)
        self.dtype = dtype
        self._layers_per_group_param: int | str = layers_per_group
        self._tokenizer_obj: Any | None = None
        self._config: Any | None = None
        self._model_skeleton: Any | None = None
        self._snapshot_dir: Any | None = None
        self._layer_to_tensors: dict | None = None
        self._non_layer_tensors: list | None = None

    @property
    def layers_per_group(self) -> int:
        """Resolved number of layers to co-load on GPU per group."""
        if self._layers_per_group_param == "auto":
            return _estimate_disk_offload_layers_per_group(
                self.model_name, self.device, self.dtype,
            )
        return int(self._layers_per_group_param)

    # --- Lazy initialization ---

    def _get_config(self) -> Any:
        if self._config is None:
            from transformers import AutoConfig
            self._config = AutoConfig.from_pretrained(self.model_name)
        return self._config

    def _get_text_config(self) -> Any:
        """Return the text-transformer config.

        For multimodal checkpoints (e.g. Mistral3, Pixtral) this is
        ``config.text_config``. For text-only models it's the config itself.
        """
        cfg = self._get_config()
        return getattr(cfg, "text_config", None) or cfg

    @property
    def _skeleton_prefix(self) -> str:
        """Prefix that safetensors use ahead of the text skeleton's module paths.

        Mistral3 / Pixtral checkpoints store text weights under
        ``language_model.*`` while the text-only skeleton we build uses plain
        ``model.*`` / ``lm_head.*`` paths. We strip this prefix at shard-map
        time and reapply it when reading from the underlying safetensors.
        """
        cfg = self._get_config()
        if getattr(cfg, "text_config", None) is not None:
            return "language_model."
        return ""

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

            prefix = self._skeleton_prefix
            # For multimodal checkpoints, drop tensors that belong to the
            # vision tower / multi-modal projector — the text-only skeleton
            # has no modules for them.
            drop_prefixes = (
                ("vision_tower.", "multi_modal_projector.") if prefix else ()
            )

            layers: dict[int, list[tuple[str, str]]] = defaultdict(list)
            non_layer: list[tuple[str, str]] = []

            for tensor_name, shard_file in index["weight_map"].items():
                if drop_prefixes and tensor_name.startswith(drop_prefixes):
                    continue
                if prefix:
                    if not tensor_name.startswith(prefix):
                        continue
                    stripped = tensor_name[len(prefix):]
                else:
                    stripped = tensor_name
                parts = stripped.split(".")
                if (
                    len(parts) >= 3
                    and parts[0] == "model"
                    and parts[1] == "layers"
                    and parts[2].isdigit()
                ):
                    layers[int(parts[2])].append((stripped, shard_file))
                else:
                    non_layer.append((stripped, shard_file))

            self._layer_to_tensors = dict(layers)
            self._non_layer_tensors = non_layer
        assert self._non_layer_tensors is not None
        return self._layer_to_tensors, self._non_layer_tensors

    def _load_tensors(
        self, tensor_list: list[tuple[str, str]], device: str,
    ) -> dict[str, torch.Tensor]:
        """Load tensors from safetensors shards to *device*.

        ``tensor_list`` uses skeleton-relative names (``model.layers.N.*``).
        For multimodal checkpoints the stored safetensors keys are prefixed
        with :attr:`_skeleton_prefix` (e.g. ``language_model.``); we reattach
        it here when looking up the shard, and key the result dict by the
        skeleton-relative name so ``_materialize_module`` finds it.
        """
        from collections import defaultdict

        from safetensors.torch import load_file as st_load_file

        prefix = self._skeleton_prefix
        shard_to_keys: dict[str, list[str]] = defaultdict(list)
        for tensor_name, shard_file in tensor_list:
            shard_to_keys[shard_file].append(tensor_name)

        snap = self._get_snapshot_dir()
        result: dict[str, torch.Tensor] = {}
        for shard_file, keys in shard_to_keys.items():
            shard_data = st_load_file(str(snap / shard_file), device=device)
            for k in keys:
                original = prefix + k
                if original in shard_data:
                    result[k] = shard_data[original]
                elif k in shard_data:
                    result[k] = shard_data[k]
            del shard_data
        return result

    def _get_model_skeleton(self) -> Any:
        """Create an empty model (meta device) for the forward graph.

        For multimodal checkpoints (e.g. Mistral3) the outer config class
        isn't registered for ``AutoModelForCausalLM``; we build the
        text-only skeleton from ``config.text_config`` and let
        :meth:`_load_tensors` strip the ``language_model.`` prefix when
        reading safetensors.
        """
        if self._model_skeleton is None:
            from accelerate import init_empty_weights
            from transformers import AutoModelForCausalLM

            skeleton_config = self._get_text_config()
            # Disable quantizer so we get standard nn.Linear modules. We
            # delete the attribute rather than setting it to None because
            # transformers>=4.57's ``PretrainedConfig.to_dict`` calls
            # ``self.quantization_config.to_dict()`` guarded only by
            # ``hasattr(...)``, which treats ``None`` as present and raises.
            if hasattr(skeleton_config, "quantization_config"):
                delattr(skeleton_config, "quantization_config")
            with init_empty_weights():
                self._model_skeleton = AutoModelForCausalLM.from_config(skeleton_config)
            self._model_skeleton.eval()
        return self._model_skeleton

    # --- Properties ---

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer_obj is None:
            from ._tokenizer_utils import load_tokenizer
            self._tokenizer_obj = load_tokenizer(self.model_name)
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
        prompts: list[str] | PreTokenizedPrompts,
        spec: ExtractionSpec,
        batch_size: int = 16,
        pool: str | None = None,
        logit_callback: Callable[[int, int, torch.Tensor, torch.Tensor], None] | None = None,
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
        text_config = self._get_text_config()
        model = self._get_model_skeleton()
        layer_map, non_layer = self._get_shard_map()
        device = self.device

        num_layers = text_config.num_hidden_layers
        hidden_target = set(spec.hidden_layers)
        router_target = set(spec.router_layers or [])
        n_experts = getattr(config, "n_routed_experts", None)
        first_moe = getattr(config, "first_k_dense_replace", 0)

        # --- Tokenize all prompts (or accept pre-tokenized input) ---
        if isinstance(prompts, PreTokenizedPrompts):
            all_input_ids = prompts.input_ids
            all_attention_mask = prompts.attention_mask
        else:
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
            n_prompts, seq_len, text_config.hidden_size, dtype=self.dtype,
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

            # Re-initialize rotary from config (meta tensors have no data).
            # Prefer explicit ``head_dim`` (e.g. Mistral-Small-3.1 sets 128 even
            # though hidden_size/num_heads = 160) before falling back to the
            # ratio default.
            dim = getattr(text_config, "qk_rope_head_dim", None)
            if dim is None:
                dim = getattr(text_config, "head_dim", None)
            if dim is None:
                dim = text_config.hidden_size // text_config.num_attention_heads
            base = getattr(text_config, "rope_theta", 10000.0)
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

        # --- Phase 2: Layer-by-layer (grouped) ---
        captured_hidden: dict[int, torch.Tensor] = {}
        captured_router: dict[int, torch.Tensor] = {}
        decoder_layers = _get_decoder_layers(model)

        # Resolve layers-per-group: how many layers to hold on GPU at once.
        lpg_param = self._layers_per_group_param
        if lpg_param == "auto":
            layers_per_group = _estimate_disk_offload_layers_per_group(
                self.model_name, device, self.dtype,
            )
        else:
            layers_per_group = int(lpg_param)

        # Shared per-device constants (same for every layer)
        mask_dev = causal_mask.to(device)
        pos_cache_dev = cache_position.to(device)
        if position_embeddings is not None:
            if isinstance(position_embeddings, tuple):
                pe_dev: Any = tuple(t.to(device) for t in position_embeddings)
            else:
                pe_dev = position_embeddings.to(device)
        else:
            pe_dev = None

        group_start = 0
        while group_start < num_layers:
            group_indices = list(range(
                group_start, min(group_start + layers_per_group, num_layers),
            ))
            group_start += layers_per_group

            # Load all layers in group onto GPU
            for layer_idx in group_indices:
                layer_weights = self._load_tensors(layer_map[layer_idx], "cpu")
                prefix = f"model.layers.{layer_idx}."
                if n_experts and layer_idx >= first_moe:
                    packed = _pack_expert_weights(
                        layer_weights, prefix, n_experts, device,
                    )
                    layer_weights.update(packed)
                    del packed
                _materialize_module(
                    decoder_layers[layer_idx], layer_weights, prefix, device,
                )
                del layer_weights

            # Process each layer in the group sequentially
            for layer_idx in group_indices:
                layer_module = decoder_layers[layer_idx]
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

            # Free all layers in the group
            for layer_idx in group_indices:
                _free_module(decoder_layers[layer_idx])
            gc.collect()
            torch.cuda.empty_cache()

        del mask_dev, pos_cache_dev, pe_dev

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

            logit_batches: list[torch.Tensor] = []
            with torch.no_grad():
                for b in range(n_batches):
                    s = b * batch_size
                    e = min(s + batch_size, n_prompts)
                    hs_dev = all_hidden[s:e].to(device)
                    logits_b = lm_head(final_norm(hs_dev)).cpu()
                    if logit_callback is not None:
                        logit_callback(s, e, logits_b, all_attention_mask[s:e])
                    else:
                        logit_batches.append(logits_b)
            if logit_batches:
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
        text_config = self._get_text_config()
        model = self._get_model_skeleton()
        layer_map, non_layer = self._get_shard_map()
        device = self.device
        num_layers = text_config.num_hidden_layers

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
                dim = getattr(text_cfg, "head_dim", None)
            if dim is None:
                dim = text_cfg.hidden_size // text_cfg.num_attention_heads
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
        prompts: list[str] | PreTokenizedPrompts,
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
        bs = kwargs.get("batch_size") or len(prompts)
        result = self.extract_all(prompts, spec, batch_size=bs)
        assert result.logits is not None
        return result.activations, result.attention_mask, result.logits, None

    def extract_batch_with_logits_streaming(
        self,
        prompts: list[str],
        layer_indices: list[int],
        on_batch: Callable[[int, int, torch.Tensor, torch.Tensor], None],
        batch_size: int = 4,
    ) -> torch.Tensor:
        """Run ONE full layer sweep for ALL prompts, streaming logits per mini-batch.

        ``on_batch(start, end, logits_b, attn_mask_b)`` is called for each
        mini-batch of size ``batch_size``.  ``logits_b`` is CPU float32
        ``[bs, S_max, V]``; ``attn_mask_b`` is ``[bs, S_max]``.

        Returns ``attention_mask`` for the full prompt set (CPU, ``[N, S_max]``).

        This avoids the ``[N, S_max, V]`` OOM that ``extract_batch_with_logits``
        hits when N is large.  Layer weights are loaded once for the full sweep
        instead of once per microbatch.
        """
        from .activation_types import ExtractionSpec
        spec = ExtractionSpec(hidden_layers=layer_indices, include_logits=True)
        result = self.extract_all(
            prompts, spec, batch_size=batch_size, logit_callback=on_batch,
        )
        return result.attention_mask

    def extract_batch_extended(
        self,
        prompts: list[str],
        spec: ExtractionSpec,
        **kwargs: Any,
    ) -> ExtractedBatch:
        return self.extract_all(prompts, spec, batch_size=len(prompts))


# ---------------------------------------------------------------------------
# DiskOffloadLayerLoader — LayerLoader implementation for DiskOffloadBackend
# ---------------------------------------------------------------------------
#
# Mirrors ChunkedLayerLoader's protocol so `SampleScan.sweep()` can run on
# models too large for full-CPU-residency. Reuses DiskOffloadBackend's
# safetensors shard-map, per-layer load/materialize/free primitives, and
# the same memmap-backed residual buffer as ChunkedLayerLoader. Forward
# loop is near-identical to ChunkedLayerLoader._drive_sweep — one layer
# per group, model skeleton (meta weights) used for topology lookups.


class DiskOffloadLayerLoader:
    """LayerLoader wrapping :class:`DiskOffloadBackend`.

    One layer per group: each group context loads the layer's weights
    from safetensors, materializes into the skeleton's decoder module,
    and frees on exit. The skeleton is reused across groups (cheap —
    meta tensors). Residual buffer is the same memmap-backed
    ``_ResidualBuffer`` used by ``ChunkedLayerLoader``.
    """

    def __init__(self, backend: DiskOffloadBackend) -> None:
        self._backend = backend
        self.tokenizer = backend.tokenizer
        config = backend._get_config()
        self.num_layers: int = int(config.num_hidden_layers)
        self.hidden_dim: int = 0  # populated in prepare()
        self.dtype: torch.dtype = backend.dtype
        self.device: str = backend.device
        self.layer_types: list[str] | None = None
        self.router_module_template: str | None = None

        lpg_param = backend._layers_per_group_param
        if lpg_param == "auto":
            self._layers_per_group: int = _estimate_disk_offload_layers_per_group(
                backend.model_name, backend.device, backend.dtype,
            )
        else:
            self._layers_per_group = int(lpg_param)

    # --- LayerLoader protocol ------------------------------------------------

    def prepare(
        self, prompts: list[str], batch_size: int,
    ) -> Any:
        """Context manager: tokenize → embed (materialize → forward → free)
        → rotary → allocate memmap. Yields an EmbedState."""
        from contextlib import contextmanager

        from .activation_types import detect_moe_info
        from .sweep import EmbedState

        @contextmanager
        def _prepare_ctx() -> Any:
            import os
            import tempfile

            try:
                moe_info = detect_moe_info(self._backend.model_name)
                if moe_info is not None:
                    self.router_module_template = moe_info.router_module_template
            except Exception:
                pass

            tokenized = self.tokenizer(
                prompts, return_tensors="pt", padding=True,
            )
            all_input_ids = tokenized["input_ids"]
            all_attention_mask = tokenized["attention_mask"]

            token_ids_per_sample: list[list[int]] = []
            seq_lengths: list[int] = []
            for i in range(len(prompts)):
                real_len = int(all_attention_mask[i].sum().item())
                seq_lengths.append(real_len)
                token_ids_per_sample.append(
                    all_input_ids[i, :real_len].tolist(),
                )

            n_samples = len(prompts)
            batches: list[tuple[int, int]] = [
                (s, min(s + batch_size, n_samples))
                for s in range(0, n_samples, batch_size)
            ]

            config = self._backend._get_config()
            model = self._backend._get_model_skeleton()
            layer_map, non_layer = self._backend._get_shard_map()
            device = self.device

            # --- Embedding: load tensors, materialize, forward, free ---
            embed_tensors = [(n, f) for n, f in non_layer if "embed_tokens" in n]
            embed_weights = self._backend._load_tensors(embed_tensors, device)
            embed = _get_embedding_module(model)
            _materialize_module(
                embed, embed_weights, "model.embed_tokens.", device,
            )
            del embed_weights

            pos_ids_per_batch: list[torch.Tensor] = []
            cache_positions_per_batch: list[torch.Tensor] = []

            tmpdir = tempfile.TemporaryDirectory(prefix="lmprobe_sweep_disk_")
            try:
                hs_path = os.path.join(tmpdir.name, "residuals.bin")
                S_max = int(all_input_ids.shape[1])
                mmap: np.memmap | None = None
                hidden_dim: int | None = None

                with torch.no_grad():
                    for start, end in batches:
                        ids = all_input_ids[start:end]
                        mask = all_attention_mask[start:end]
                        hs = embed(ids.to(device)).cpu()
                        if hidden_dim is None:
                            hidden_dim = int(hs.shape[-1])
                        if mmap is None:
                            mmap = np.memmap(
                                hs_path,
                                dtype=_mmap_np_dtype(self.dtype),
                                mode="w+",
                                shape=(n_samples, S_max, hidden_dim),
                            )
                        _mmap_write_slice(mmap, start, end, hs)
                        del hs
                        pos_ids = mask.long().cumsum(-1) - 1
                        pos_ids.masked_fill_(mask == 0, 1)
                        pos_ids_per_batch.append(pos_ids)
                        cache_positions_per_batch.append(
                            torch.arange(ids.shape[1]),
                        )

                _free_module(embed)
                if torch.cuda.is_available() and self.device != "cpu":
                    torch.cuda.empty_cache()

                assert mmap is not None and hidden_dim is not None
                self.hidden_dim = hidden_dim

                # --- Rotary (re-init from config; meta tensors have no data) ---
                position_embeddings, layer_types = self._compute_rotary(
                    model, config, mmap, pos_ids_per_batch, device,
                )
                self.layer_types = layer_types

                state = EmbedState(
                    input_ids=all_input_ids,
                    attention_mask=all_attention_mask,
                    batches=batches,
                    pos_ids_per_batch=pos_ids_per_batch,
                    cache_positions_per_batch=cache_positions_per_batch,
                    position_embeddings=position_embeddings,
                    layer_types=layer_types,
                    seq_lengths=seq_lengths,
                    token_ids_per_sample=token_ids_per_sample,
                    hidden_dim=hidden_dim,
                    residual_buffer=_ResidualBuffer(mmap, tmpdir),
                )
            except Exception:
                tmpdir.cleanup()
                raise

            try:
                yield state
            finally:
                state.residual_buffer.cleanup()

        return _prepare_ctx()

    def _compute_rotary(
        self,
        model: Any,
        config: Any,
        mmap: np.memmap,
        pos_ids_per_batch: list[torch.Tensor],
        device: str,
    ) -> tuple[Any, list[str] | None]:
        """Rotary init for disk-offload: re-initialize inv_freq from config
        (meta skeleton has no data), compute PE on a single-sample slice."""
        rotary_name = ChunkedLocalBackend._find_rotary_embedding_name(model)
        if rotary_name is None:
            return None, None
        rotary_mod = model
        for part in rotary_name.split("."):
            rotary_mod = getattr(rotary_mod, part)

        dim = getattr(config, "qk_rope_head_dim", None)
        if dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
            dim = head_dim
        base = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=device)
                / dim
            )
        )
        rotary_mod.to_empty(device=device)
        rotary_mod.inv_freq = inv_freq

        text_cfg = getattr(
            getattr(model, "config", None), "text_config",
            getattr(model, "config", None),
        )
        layer_types_cfg = getattr(text_cfg, "layer_types", None)
        pe_hs = _mmap_read_slice(mmap, 0, 1, self.dtype).to(device)
        pe_pos = pos_ids_per_batch[0][:1].to(device)

        unique_types = set(layer_types_cfg) if layer_types_cfg is not None else set()
        position_embeddings: Any
        layer_types: list[str] | None
        if layer_types_cfg is not None and len(unique_types) > 1:
            layer_types = list(layer_types_cfg)
            position_embeddings = {}
            with torch.no_grad():
                for lt in unique_types:
                    pe = rotary_mod(pe_hs, pe_pos, layer_type=lt)
                    position_embeddings[lt] = tuple(t.cpu() for t in pe)
        else:
            layer_types = (
                list(layer_types_cfg) if layer_types_cfg is not None else None
            )
            with torch.no_grad():
                pe = rotary_mod(pe_hs, pe_pos)
                if isinstance(pe, tuple):
                    position_embeddings = tuple(t.cpu() for t in pe)
                else:
                    position_embeddings = pe.cpu()

        _free_module(rotary_mod)
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.empty_cache()
        return position_embeddings, layer_types

    def iter_layer_groups(self) -> Any:
        lpg = self._layers_per_group
        for i in range(0, self.num_layers, lpg):
            yield list(range(i, min(i + lpg, self.num_layers)))

    def layer_group(self, indices: list[int]) -> Any:
        from contextlib import contextmanager

        @contextmanager
        def _ctx() -> Any:
            import gc

            config = self._backend._get_config()
            model = self._backend._get_model_skeleton()
            layer_map, _ = self._backend._get_shard_map()
            decoder_layers = _get_decoder_layers(model)

            n_experts = getattr(config, "n_routed_experts", None)
            first_moe = getattr(config, "first_k_dense_replace", 0)

            materialized_modules: list[Any] = []
            try:
                for layer_idx in indices:
                    layer_weights = self._backend._load_tensors(
                        layer_map[layer_idx], "cpu",
                    )
                    prefix = f"model.layers.{layer_idx}."

                    # MoE expert weight packing (no-op for dense models).
                    if n_experts and layer_idx >= first_moe:
                        packed = _pack_expert_weights(
                            layer_weights, prefix, n_experts, self.device,
                        )
                        layer_weights.update(packed)
                        del packed

                    layer_module = decoder_layers[layer_idx]
                    _materialize_module(
                        layer_module, layer_weights, prefix, self.device,
                    )
                    del layer_weights
                    materialized_modules.append(layer_module)

                yield materialized_modules
            finally:
                for mod in materialized_modules:
                    _free_module(mod)
                gc.collect()
                if torch.cuda.is_available() and self.device != "cpu":
                    torch.cuda.empty_cache()
        return _ctx()

    def read_hs(
        self,
        state: Any,
        start: int,
        end: int,
        device: str,
    ) -> torch.Tensor:
        return _mmap_read_slice(
            state.residual_buffer.mmap, start, end, self.dtype,
        ).to(device)

    def write_hs(
        self,
        state: Any,
        start: int,
        end: int,
        hs: torch.Tensor,
    ) -> None:
        _mmap_write_slice(state.residual_buffer.mmap, start, end, hs)

    def build_layer_kwargs(
        self,
        state: Any,
        batch_idx: int,
        layer_idx: int,
        causal_mask_dev: torch.Tensor,
        device: str,
    ) -> dict[str, Any]:
        pos_dev = state.pos_ids_per_batch[batch_idx].to(device)
        cache_position = state.cache_positions_per_batch[batch_idx]
        pe_dev = ChunkedLocalBackend._pe_to_device(
            state.position_embeddings, device,
        )
        return ChunkedLocalBackend._build_layer_kwargs(
            causal_mask_dev, pos_dev, cache_position,
            pe_dev, state.layer_types, layer_idx, device,
        )

    def apply_lm_head(
        self,
        final_hs: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Load final_norm + lm_head from safetensors, apply, free."""
        model = self._backend._get_model_skeleton()
        _, non_layer = self._backend._get_shard_map()

        norm_tensors = [
            (n, f) for n, f in non_layer
            if "norm" in n and "layer" not in n
        ]
        head_tensors = [(n, f) for n, f in non_layer if "lm_head" in n]

        final_norm = _get_final_norm(model)
        lm_head = _get_lm_head(model)

        norm_w = self._backend._load_tensors(norm_tensors, device)
        _materialize_module(final_norm, norm_w, "model.norm.", device)
        del norm_w

        head_w = self._backend._load_tensors(head_tensors, device)
        _materialize_module(lm_head, head_w, "lm_head.", device)
        del head_w

        try:
            with torch.no_grad():
                logits = lm_head(final_norm(final_hs))
        finally:
            _free_module(final_norm)
            _free_module(lm_head)
            if torch.cuda.is_available() and self.device != "cpu":
                torch.cuda.empty_cache()

        assert isinstance(logits, torch.Tensor)
        return logits

    # --- Sweep driver --------------------------------------------------------

    def _drive_sweep(
        self,
        *,
        state: Any,
        accumulators: Any,
        plans: dict[tuple[int, str], Any],
        basis_gpu: dict[str, torch.Tensor],
        batch_size: int,
        ctx: Any,
    ) -> None:
        """Per-layer forward loop. Structurally identical to
        ``ChunkedLayerLoader._drive_sweep`` — the only substantive
        difference is that ``model`` is the meta skeleton (topology only,
        live weights are inside the currently-materialized layer module
        from ``layer_group()``)."""
        from tqdm import tqdm

        from .sweep import (
            SIGNAL_ATTN_DELTA,
            SIGNAL_LOGITS,
            SIGNAL_MLP_DELTA,
            SIGNAL_RESIDUAL,
            SIGNAL_ROUTER_LOGITS,
            _dispatch_plan,
            _notify_group_complete,
        )

        router_strategy: str | None = None
        router_owners: list[str] = []
        for name, acc in accumulators.items():
            if SIGNAL_ROUTER_LOGITS in acc.signals and hasattr(acc, "strategy"):
                s = acc.strategy
                if router_strategy is None:
                    router_strategy = s
                    router_owners = [name]
                elif s != router_strategy:
                    router_owners.append(name)
                    raise ValueError(
                        f"sweep: conflicting router_logits hook strategies "
                        f"across subscribers {router_owners!r}: "
                        f"{router_strategy!r} vs {s!r}."
                    )
                else:
                    router_owners.append(name)
        if router_strategy is None:
            router_strategy = "output"

        device = self.device
        model = self._backend._get_model_skeleton()

        group_iter = list(self.iter_layer_groups())
        for chunk_idx, chunk_indices in enumerate(tqdm(
            group_iter,
            desc="Sweep: disk-offload layers",
        )):
            with self.layer_group(chunk_indices) as layer_modules_in_group:
                layer_modules = {
                    layer_idx: layer_modules_in_group[i]
                    for i, layer_idx in enumerate(chunk_indices)
                }

                for batch_idx, (start, end) in enumerate(state.batches):
                    hs = self.read_hs(state, start, end, device)
                    mask_dev = _make_causal_mask(
                        state.attention_mask[start:end], self.dtype,
                    ).to(device)
                    sample_ids = np.arange(start, end, dtype=np.int32)
                    attn_mask_np = (
                        state.attention_mask[start:end].cpu().numpy().astype(bool)
                    )

                    with torch.no_grad():
                        for layer_idx in chunk_indices:
                            layer_module = layer_modules[layer_idx]

                            hook_handles: list[Any] = []
                            hook_bufs: dict[str, list[torch.Tensor]] = {}

                            for sig in (
                                SIGNAL_ATTN_DELTA,
                                SIGNAL_MLP_DELTA,
                                SIGNAL_ROUTER_LOGITS,
                            ):
                                if (layer_idx, sig) not in plans:
                                    continue
                                buf: list[torch.Tensor] = []
                                hook_bufs[sig] = buf
                                handle = _install_sweep_hook(
                                    sig, layer_module, layer_idx, model,
                                    self.router_module_template,
                                    router_strategy, buf,
                                )
                                if handle is not None:
                                    hook_handles.append((sig, handle))

                            layer_kwargs = self.build_layer_kwargs(
                                state, batch_idx, layer_idx, mask_dev, device,
                            )
                            output = layer_module(hs, **layer_kwargs)
                            hs = output[0] if isinstance(output, tuple) else output

                            for sig, handle in hook_handles:
                                handle.remove()
                                buf = hook_bufs[sig]
                                if not buf:
                                    continue
                                plan = plans.get((layer_idx, sig))
                                if plan is None:
                                    continue
                                sig_basis = basis_gpu.get(sig)
                                per_layer_basis = (
                                    sig_basis[layer_idx] if sig_basis is not None
                                    else None
                                )
                                _dispatch_plan(
                                    plan, buf[0], sig, layer_idx,
                                    sample_ids, attn_mask_np,
                                    per_layer_basis,
                                )
                                buf.clear()

                            res_plan = plans.get((layer_idx, SIGNAL_RESIDUAL))
                            if res_plan is not None:
                                res_basis = basis_gpu.get(SIGNAL_RESIDUAL)
                                res_per_layer = (
                                    res_basis[layer_idx] if res_basis is not None
                                    else None
                                )
                                _dispatch_plan(
                                    res_plan, hs.detach(), SIGNAL_RESIDUAL,
                                    layer_idx, sample_ids, attn_mask_np,
                                    res_per_layer,
                                )

                    self.write_hs(state, start, end, hs)
                    del hs, mask_dev

            _notify_group_complete(accumulators, chunk_indices)

        wants_logits = any(
            SIGNAL_LOGITS in a.signals for a in accumulators.values()
        )
        if wants_logits:
            for batch_idx, (start, end) in enumerate(state.batches):
                hs = self.read_hs(state, start, end, device)
                logits = self.apply_lm_head(hs, device)
                sample_ids = np.arange(start, end, dtype=np.int32)
                attn_mask_np = (
                    state.attention_mask[start:end].cpu().numpy().astype(bool)
                )
                logits_cpu = logits.detach().cpu()
                for acc in accumulators.values():
                    if SIGNAL_LOGITS in acc.signals:
                        acc.update(
                            logits_cpu, SIGNAL_LOGITS, -1,
                            sample_ids, attn_mask_np,
                        )
                del hs, logits, logits_cpu


def resolve_backend(
    backend: str,
    model_name: str,
    device: str = "auto",
    remote: bool = False,
    dtype: torch.dtype | None = None,
    chunk_size: int | str | None = None,
    layers_per_group: int | str | None = None,
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
    layers_per_group : int or str or None
        Number of layers to co-resident on GPU simultaneously for
        ``backend="disk_offload"``. ``"auto"`` (default) estimates from
        available VRAM leaving 40% headroom. Pass an int to override.
        Ignored for other backends.

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
        lpg = layers_per_group if layers_per_group is not None else "auto"
        return DiskOffloadBackend(model_name, device, dtype=offload_dtype, layers_per_group=lpg)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Available backends: 'nnsight', 'local', 'chunked', 'disk_offload'."
        )
