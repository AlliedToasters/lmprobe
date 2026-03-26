"""Activation extraction from language models via nnsight.

This module handles loading models and extracting intermediate activations
from specified layers. Supports both local and remote execution.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from nnsight import LanguageModel


def _require_nnsight() -> Any:
    """Import and return the nnsight module, raising a clear error if not installed."""
    try:
        import nnsight
    except ImportError:
        raise ImportError(
            "nnsight is required for this operation. "
            "Install it with: pip install lmprobe[nnsight]"
        ) from None
    return nnsight


# Global model cache to avoid loading the same model multiple times
# Key: (model_name, device, remote), Value: LanguageModel
_MODEL_CACHE: dict = {}


def get_cached_model(
    model_name: str, device: str = "auto", remote: bool = False
) -> LanguageModel:
    """Get a model from the cache, loading if necessary.

    This ensures the same model is shared across all ActivationExtractor
    instances, preventing OOM from loading multiple copies.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification.
    remote : bool
        If True, creates a lightweight model stub for remote execution only.
        No model weights are downloaded.

    Returns
    -------
    LanguageModel
        The cached or newly loaded model.
    """
    # Include remote in cache key since remote stubs differ from local models
    cache_key = (model_name, device, remote)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_model(model_name, device, remote=remote)
    return _MODEL_CACHE[cache_key]


def clear_model_cache() -> None:
    """Clear the global model cache to free memory.

    Call this when you're done with all probes and want to release
    GPU/CPU memory held by loaded models.
    """
    global _MODEL_CACHE
    _MODEL_CACHE.clear()


def configure_remote() -> None:
    """Configure nnsight for remote execution.

    Reads the API key from NDIF_API_KEY environment variable.

    Raises
    ------
    EnvironmentError
        If NDIF_API_KEY is not set.
    """
    api_key = os.getenv("NDIF_API_KEY")
    if not api_key:
        raise OSError(
            "NDIF_API_KEY environment variable is required for remote execution. "
            "Set it with: export NDIF_API_KEY='your-key-here'"
        )
    nnsight = _require_nnsight()
    nnsight.CONFIG.API.APIKEY = api_key


def get_num_layers_from_config(model_name: str) -> int:
    """Get the number of transformer layers from model config (without loading weights).

    This function only downloads the model's config.json (~1KB) instead of the
    full model weights. This is critical for large models where loading weights
    would consume hundreds of GB of memory.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.

    Returns
    -------
    int
        Number of transformer layers.

    Raises
    ------
    ValueError
        If the config doesn't contain a recognized layer count field.

    Examples
    --------
    >>> get_num_layers_from_config("meta-llama/Llama-3.1-8B-Instruct")
    32
    >>> get_num_layers_from_config("meta-llama/Llama-3.1-405B-Instruct")
    126
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)

    # Different model architectures use different config field names
    # Try common ones in order of prevalence
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        if hasattr(config, attr):
            return int(getattr(config, attr))

    raise ValueError(
        f"Could not determine layer count from config for {model_name}. "
        f"Config has attributes: {list(config.to_dict().keys())}"
    )


def resolve_auto_candidates(
    candidates: list[int] | list[float] | None,
    num_layers: int,
) -> list[int]:
    """Resolve auto_candidates specification to layer indices.

    Parameters
    ----------
    candidates : list[int] | list[float] | None
        Candidate specification:
        - None: Default to [0.25, 0.5, 0.75] fractional positions
        - list[int]: Explicit layer indices (negative indexing supported)
        - list[float]: Fractional positions in [0.0, 1.0]

    num_layers : int
        Total number of layers in the model.

    Returns
    -------
    list[int]
        Sorted list of unique positive layer indices.

    Raises
    ------
    ValueError
        If indices are out of range or fractions are invalid.

    Examples
    --------
    >>> resolve_auto_candidates(None, 32)
    [7, 15, 23]  # 0.25, 0.5, 0.75 of 32 layers

    >>> resolve_auto_candidates([0.33, 0.66], 32)
    [10, 20]  # floor(0.33*31), floor(0.66*31)

    >>> resolve_auto_candidates([10, 16, 22], 32)
    [10, 16, 22]  # Explicit indices

    >>> resolve_auto_candidates([-8, -4, -1], 32)
    [24, 28, 31]  # Negative indexing
    """
    # Default candidates
    if candidates is None:
        candidates = [0.25, 0.5, 0.75]

    if not candidates:
        raise ValueError("auto_candidates cannot be empty")

    # Determine if fractional or integer mode
    # Fractional: all values are floats in [0.0, 1.0]
    # Integer: any value is an integer or float outside [0.0, 1.0]
    is_fractional = all(
        isinstance(c, float) and 0.0 <= c <= 1.0 for c in candidates
    )

    resolved = []

    if is_fractional:
        for frac in candidates:
            # Map fraction to layer index
            # frac=0.0 -> layer 0, frac=1.0 -> layer num_layers-1
            idx = int(frac * (num_layers - 1))
            idx = max(0, min(idx, num_layers - 1))  # Clamp
            resolved.append(idx)
    else:
        # Integer mode
        for c in candidates:
            layer_idx = int(c)
            # Handle negative indexing
            if layer_idx < 0:
                layer_idx = num_layers + layer_idx
            if not (0 <= layer_idx < num_layers):
                raise ValueError(
                    f"Layer index {layer_idx} out of range for model with {num_layers} layers. "
                    f"Valid range: [0, {num_layers - 1}] or [-{num_layers}, -1]"
                )
            resolved.append(layer_idx)

    # Remove duplicates and sort
    return sorted(set(resolved))


def resolve_layers(
    layers: int | list[int] | str,
    num_layers: int,
    auto_candidates: list[int] | list[float] | None = None,
) -> list[int]:
    """Convert layer specification to list of positive indices.

    Parameters
    ----------
    layers : int | list[int] | str
        Layer specification:
        - int: Single layer (supports negative indexing)
        - list[int]: Multiple layers (supports negative indexing)
        - "middle": Middle third of layers
        - "last": Last layer only
        - "all": All layers
        - "auto": Automatic layer selection via Group Lasso (uses auto_candidates)
        - "fast_auto": Fast automatic layer selection via coefficient importance

    num_layers : int
        Total number of layers in the model.

    auto_candidates : list[int] | list[float] | None
        Candidate layers for "auto" mode. Only used when layers="auto".
        - list[int]: Explicit layer indices
        - list[float]: Fractional positions (0.0 to 1.0)
        - None: Use default [0.25, 0.5, 0.75]

    Returns
    -------
    list[int]
        List of resolved positive layer indices.

    Raises
    ------
    ValueError
        If layer index is out of range or unknown preset.
    """

    def normalize_index(idx: int) -> int:
        """Convert potentially negative index to positive."""
        if idx < 0:
            idx = num_layers + idx
        if not (0 <= idx < num_layers):
            raise ValueError(
                f"Layer index {idx} out of range for model with {num_layers} layers. "
                f"Valid range: [0, {num_layers - 1}] or [-{num_layers}, -1]"
            )
        return idx

    if isinstance(layers, int):
        return [normalize_index(layers)]

    if isinstance(layers, list):
        return [normalize_index(i) for i in layers]

    if layers == "auto" or layers == "fast_auto":
        return resolve_auto_candidates(auto_candidates, num_layers)

    if layers == "middle":
        # Middle third of layers
        third = num_layers // 3
        start = third
        end = num_layers - third
        return list(range(start, end))

    if layers == "last":
        return [num_layers - 1]

    if layers == "all":
        return list(range(num_layers))

    raise ValueError(
        f"Unknown layer specification: {layers!r}. "
        f"Use int, list[int], 'middle', 'last', 'all', or 'auto'."
    )


def load_model(
    model_name: str,
    device: str = "auto",
    remote: bool = False,
) -> LanguageModel:
    """Load a language model via nnsight.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification. "auto" uses device_map="auto".
        Ignored when remote=True.
    remote : bool
        If True, creates a lightweight model stub for remote execution only.
        No model weights are downloaded - only the tokenizer and config.
        This is critical for large models like 405B that would otherwise
        require hundreds of GB of memory.

    Returns
    -------
    LanguageModel
        The loaded nnsight model.
    """
    nnsight = _require_nnsight()
    LM = nnsight.LanguageModel

    if remote:
        # For remote execution, don't load weights locally.
        # dispatch=False prevents loading model weights - only tokenizer is loaded.
        # This is critical for large models (405B) that would OOM locally.
        # See: https://nnsight.net/notebooks/features/remote_execution/
        model = LM(model_name, dispatch=False)
        # Prevent nnsight's MetaMixin.interleave from auto-dispatching
        # (which would download all weights via from_pretrained).
        # Remote traces don't need local weights — the forward pass
        # runs on the NDIF server.
        model.dispatched = True
    else:
        # Fast-fail on CUDA compute capability mismatch
        from lmprobe._device_utils import check_cuda_compatibility

        check_cuda_compatibility(device)

        # Local execution - load weights to specified device
        device_map: str | dict[str, str]
        if device == "auto":
            device_map = "auto"
        elif device == "cpu":
            device_map = {"": "cpu"}
        else:
            device_map = {"": device}

        try:
            model = LM(
                model_name,
                device_map=device_map,
                dispatch=True,
            )
        except RuntimeError as e:
            if "no kernel image" in str(e).lower() and device_map != {"": "cpu"}:
                import warnings
                warnings.warn(
                    f"GPU detected but incompatible with PyTorch: {e}. "
                    f"Falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                model = LM(
                    model_name,
                    device_map={"": "cpu"},
                    dispatch=True,
                )
            else:
                raise
    return model


def _build_remote_extract_fn(
    layer_indices: list[int],
    with_logits: bool = False,
    logit_top_k: int | None = None,
) -> Any:
    """Build a trace function for remote nnsight execution.

    nnsight's remote tracing serializes the source code inside the
    ``with model.trace()`` block. Two limitations prevent using the
    same code path as local execution:

    1. ``tracer.cache()`` returns a ``CacheDict`` whose ``__getattr__``
       causes a ``RecursionError`` during ``torch.load`` deserialization
       (nnsight issue #501).
    2. nnsight's ``push()`` mechanism only injects simple variable
       assignments back into the calling scope — loops and container
       mutations are silently dropped.

    To work around both issues we dynamically generate a real ``.py``
    file containing one ``output.save()`` statement per layer (and
    optionally a logits save). The file is importable so that
    ``inspect.getsourcelines`` succeeds during nnsight's code capture.

    Parameters
    ----------
    layer_indices : list[int]
        Positive layer indices to extract.
    with_logits : bool
        If True, also save the lm_head logits output.
    logit_top_k : int | None
        If set and with_logits is True, perform server-side top-k on
        logits so only compressed tensors are transferred. The trace
        will return ``(values, indices)`` instead of full logits.

    Returns
    -------
    callable
        A function ``(model, tokenized) -> tuple[list[Tensor], ...]``
        that runs the trace and returns layer outputs plus logits info.
        When logit_top_k is set: ``(layers, (topk_values, topk_indices))``
        Otherwise: ``(layers, logits_or_none)``
    """
    import importlib.util
    import tempfile

    var_names = [f"_l{i}" for i in layer_indices]
    save_lines = "\n".join(
        f"        {v} = model.model.layers[{i}].output.save()"
        for v, i in zip(var_names, layer_indices)
    )

    if with_logits and logit_top_k is not None:
        # Server-side top-k: compute topk inside the trace
        logits_line = (
            "        _raw_logits = model.lm_head.output\n"
            f"        _logits_vals, _logits_idxs = _raw_logits.topk({logit_top_k}, dim=-1)\n"
            "        _logits_vals = _logits_vals.save()\n"
            "        _logits_idxs = _logits_idxs.save()"
        )
        return_logits = "(_logits_vals, _logits_idxs)"
    elif with_logits:
        logits_line = "        _logits = model.lm_head.output.save()"
        return_logits = "_logits"
    else:
        logits_line = ""
        return_logits = "None"

    return_layers = f"[{', '.join(var_names)}]"

    code = (
        "def extract(model, tokenized):\n"
        "    with model.trace(tokenized, remote=True, scan=False) as tracer:\n"
        f"{save_lines}\n"
        f"{logits_line}\n"
        f"    return ({return_layers}, {return_logits})\n"
    )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="_lmprobe_remote_", delete=False
    )
    tmp.write(code)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    spec = importlib.util.spec_from_file_location("_lmprobe_remote", tmp_path)
    assert spec is not None, f"Failed to create module spec from {tmp_path}"
    assert spec.loader is not None, f"Module spec has no loader for {tmp_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Wrap the function to clean up the temp file after execution.
    # The file must exist when nnsight calls inspect.getsourcelines()
    # during model.trace(), so we can only delete it after the trace
    # (i.e., after the wrapper returns).
    _extract_fn = mod.extract

    def _wrapper(model: Any, tokenized: Any) -> Any:
        try:
            return _extract_fn(model, tokenized)
        finally:
            try:
                import os as _os

                _os.unlink(tmp_path)
            except OSError:
                pass

    return _wrapper


def _unwrap_proxy(x: Any) -> Any:
    """Unwrap an nnsight proxy to a plain tensor if needed."""
    return x.value if hasattr(x, "value") else x


def _unwrap_layer_outputs(raw_outputs: list) -> list[torch.Tensor]:
    """Unwrap proxy / tuple layer outputs into plain tensors."""
    tensors = []
    for raw in raw_outputs:
        val = _unwrap_proxy(raw)
        if isinstance(val, tuple):
            val = val[0]
        tensors.append(val)
    return tensors


def _extract_batch(
    model: LanguageModel,
    prompts: list[str],
    layer_indices: list[int],
    remote: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract activations for a single batch of prompts.

    Parameters
    ----------
    model : LanguageModel
        The nnsight model.
    prompts : list[str]
        List of text prompts (should be a small batch).
    layer_indices : list[int]
        List of layer indices to extract from (must be positive).
    remote : bool
        Whether to use remote execution.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - activations: Shape (batch, seq_len, hidden_dim * num_layers)
        - attention_mask: Shape (batch, seq_len)
    """
    tokenized = model.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )

    if remote:
        # Remote: use individual output.save() calls to avoid CacheDict
        # pickling issue (nnsight #501). See _build_remote_extract_fn
        # docstring for details.
        fn = _build_remote_extract_fn(layer_indices, with_logits=False)
        layer_outputs, _ = fn(model, tokenized)
        activation_tensors = _unwrap_layer_outputs(layer_outputs)
    else:
        # Local: tracer.cache() works fine without serialization.
        modules_to_cache = [model.model.layers[i] for i in layer_indices]

        with model.trace(tokenized, remote=False) as tracer:
            cache = tracer.cache(modules=modules_to_cache).save()

        activation_tensors = []
        for layer_idx in layer_indices:
            key = f"model.model.layers.{layer_idx}"
            entry = cache[key]

            if hasattr(entry, "output"):
                output = entry.output
            else:
                output = entry["output"]

            tensor = _unwrap_proxy(output)
            if isinstance(tensor, tuple):
                tensor = tensor[0]

            activation_tensors.append(tensor)

    combined = torch.cat(activation_tensors, dim=-1)
    attention_mask = tokenized["attention_mask"]

    return combined, attention_mask


def _extract_batch_with_logits(
    model: LanguageModel,
    prompts: list[str],
    layer_indices: list[int],
    remote: bool = False,
    logit_top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Extract activations AND logits for a single batch of prompts.

    This function captures both layer activations and the lm_head output
    in a single forward pass, enabling efficient computation of both
    probe features and perplexity.

    Parameters
    ----------
    model : LanguageModel
        The nnsight model.
    prompts : list[str]
        List of text prompts (should be a small batch).
    layer_indices : list[int]
        List of layer indices to extract from (must be positive).
    remote : bool
        Whether to use remote execution.
    logit_top_k : int | None
        If set and remote=True, perform server-side top-k on logits
        so only compressed tensors are transferred over the network.
        When active, the third return value contains top-k values and
        the fourth contains top-k indices. Ignored when remote=False.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
        - activations: Shape (batch, seq_len, hidden_dim * num_layers)
        - attention_mask: Shape (batch, seq_len)
        - logits: Shape (batch, seq_len, vocab_size) when logit_top_k is None,
          or (batch, seq_len, K) top-k values when logit_top_k is set
        - logits_indices: None when logit_top_k is None, or
          (batch, seq_len, K) top-k indices when logit_top_k is set
    """
    tokenized = model.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )

    logits_indices = None

    if remote:
        # Only pass logit_top_k to the remote trace builder
        fn = _build_remote_extract_fn(
            layer_indices, with_logits=True, logit_top_k=logit_top_k
        )
        layer_outputs, logits_proxy = fn(model, tokenized)
        activation_tensors = _unwrap_layer_outputs(layer_outputs)

        if logit_top_k is not None:
            # logits_proxy is a (values, indices) tuple from server-side topk
            vals_proxy, idxs_proxy = logits_proxy
            logits_val = _unwrap_proxy(vals_proxy)
            logits_indices = _unwrap_proxy(idxs_proxy)
        else:
            logits_val = _unwrap_proxy(logits_proxy)
    else:
        modules_to_cache = [model.model.layers[i] for i in layer_indices]

        with model.trace(tokenized, remote=False) as tracer:
            cache = tracer.cache(modules=modules_to_cache).save()
            logits = model.lm_head.output.save()

        activation_tensors = []
        for layer_idx in layer_indices:
            key = f"model.model.layers.{layer_idx}"
            entry = cache[key]

            if hasattr(entry, "output"):
                output = entry.output
            else:
                output = entry["output"]

            tensor = _unwrap_proxy(output)
            if isinstance(tensor, tuple):
                tensor = tensor[0]

            activation_tensors.append(tensor)

        logits_val = _unwrap_proxy(logits)

    combined = torch.cat(activation_tensors, dim=-1)
    attention_mask = tokenized["attention_mask"]

    return combined, attention_mask, logits_val, logits_indices


def compute_perplexity_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    return_per_token: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Compute perplexity features from logits.

    Handles batched computation with proper masking for variable-length sequences.

    Parameters
    ----------
    logits : torch.Tensor
        Model logits, shape (batch, seq_len, vocab_size).
    input_ids : torch.Tensor
        Input token IDs, shape (batch, seq_len).
    attention_mask : torch.Tensor
        Attention mask, shape (batch, seq_len). 1 for real tokens, 0 for padding.
    return_per_token : bool
        If True, also return per-token perplexity values and token IDs.

    Returns
    -------
    torch.Tensor or tuple
        If return_per_token is False: shape (batch, 3) - [mean_ppl, min_ppl, max_ppl].
        If return_per_token is True: (aggregates, per_token_ppl_list, token_ids_list)
        where per_token_ppl_list is a list of 1D tensors (variable length, masked)
        and token_ids_list is a list of 1D tensors of real token IDs per prompt.
    """
    import numpy as np

    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()  # Mask for shifted positions

    # Move to same device
    if shift_logits.device != shift_labels.device:
        shift_labels = shift_labels.to(shift_logits.device)
        shift_mask = shift_mask.to(shift_logits.device)

    # Compute per-token cross-entropy loss (no reduction)
    batch_size, seq_len_minus_1, vocab_size = shift_logits.shape

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fn(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
    ).view(batch_size, seq_len_minus_1)

    # Compute features per prompt
    features = []
    per_token_ppl_list: list[torch.Tensor] = []
    token_ids_list: list[torch.Tensor] = []

    for i in range(batch_size):
        valid_losses = per_token_loss[i][shift_mask[i] == 1]

        if len(valid_losses) == 0:
            # Edge case: empty sequence after shift
            features.append([1.0, 1.0, 1.0])
            if return_per_token:
                per_token_ppl_list.append(torch.tensor([], dtype=torch.float32))
                real_ids = input_ids[i][attention_mask[i] == 1]
                token_ids_list.append(real_ids.cpu())
            continue

        mean_loss = valid_losses.mean().item()
        min_loss = valid_losses.min().item()
        max_loss = valid_losses.max().item()

        mean_ppl = float(np.exp(mean_loss))
        min_ppl = float(np.exp(min_loss))
        max_ppl = float(np.exp(max_loss))

        features.append([mean_ppl, min_ppl, max_ppl])

        if return_per_token:
            token_ppl = torch.exp(valid_losses).cpu().float()
            per_token_ppl_list.append(token_ppl)
            real_ids = input_ids[i][attention_mask[i] == 1]
            token_ids_list.append(real_ids.cpu())

    aggregates = torch.tensor(features, dtype=torch.float32)

    if return_per_token:
        return aggregates, per_token_ppl_list, token_ids_list

    return aggregates


def extract_activations(
    model: LanguageModel,
    prompts: list[str],
    layer_indices: list[int],
    remote: bool = False,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract activations from specified layers.

    Parameters
    ----------
    model : LanguageModel
        The nnsight model.
    prompts : list[str]
        List of text prompts.
    layer_indices : list[int]
        List of layer indices to extract from (must be positive).
    remote : bool
        Whether to use remote execution.
    batch_size : int
        Number of prompts to process at once. Smaller values use less memory.
        Default is 8.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - activations: Shape (batch, seq_len, hidden_dim * num_layers)
          Activations from all specified layers, concatenated along hidden dim.
        - attention_mask: Shape (batch, seq_len)
          Attention mask from tokenization.
    """
    if remote:
        configure_remote()

    # Process in batches to avoid OOM
    all_activations = []
    all_masks = []

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(
            range(0, len(prompts), batch_size),
            total=num_batches,
            desc="Extracting activations",
            unit="batch",
        ):
            batch_prompts = prompts[i : i + batch_size]

            batch_acts, batch_mask = _extract_batch(
                model, batch_prompts, layer_indices, remote=remote
            )

            # Move to CPU immediately to free GPU memory
            all_activations.append(batch_acts.cpu())
            all_masks.append(batch_mask.cpu())

    # Pad and concatenate all batches
    # Find max sequence length across all batches
    max_seq_len = max(acts.shape[1] for acts in all_activations)
    hidden_dim = all_activations[0].shape[2]

    # Pad each batch to max_seq_len
    padded_activations = []
    padded_masks = []

    for acts, mask in zip(all_activations, all_masks):
        batch_size_actual, seq_len, _ = acts.shape
        if seq_len < max_seq_len:
            # Pad activations with zeros
            pad_size = max_seq_len - seq_len
            acts_pad = torch.zeros(batch_size_actual, pad_size, hidden_dim)
            acts = torch.cat([acts, acts_pad], dim=1)

            # Pad mask with zeros (masked out)
            mask_pad = torch.zeros(batch_size_actual, pad_size, dtype=mask.dtype)
            mask = torch.cat([mask, mask_pad], dim=1)

        padded_activations.append(acts)
        padded_masks.append(mask)

    # Concatenate along batch dimension
    combined_activations = torch.cat(padded_activations, dim=0)
    combined_masks = torch.cat(padded_masks, dim=0)

    return combined_activations, combined_masks


class ActivationExtractor:
    """Manages model loading and activation extraction.

    This class caches the loaded model to avoid reloading on every call.
    It delegates to an ExtractionBackend for actual model loading and
    activation extraction.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification.
    layers : int | list[int] | str
        Layer specification.
    batch_size : int
        Number of prompts to process at once. Smaller values use less memory.
    auto_candidates : list[int] | list[float] | None
        Candidate layers for layers="auto" mode.
    remote : bool
        If True, creates a lightweight model stub for remote execution only.
        No model weights are downloaded - only the tokenizer and config.
        This is critical for large models (e.g., 405B) that would otherwise
        require hundreds of GB of memory to load locally.
    backend : str
        Extraction backend to use: "local" (default) or "nnsight".
        "local" uses HuggingFace transformers directly without nnsight.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        layers: int | list[int] | str = "middle",
        batch_size: int = 8,
        auto_candidates: list[int] | list[float] | None = None,
        remote: bool = False,
        backend: str = "local",
        dtype: torch.dtype | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.layers_spec = layers
        self.batch_size = batch_size
        self.auto_candidates = auto_candidates
        self.remote = remote
        self.backend_name = backend

        # Create the backend
        from .backends import resolve_backend

        self._backend = resolve_backend(
            backend, model_name, device, remote=remote, dtype=dtype
        )

        # Lazy-loaded
        self._model: LanguageModel | None = None
        self._layer_indices: list[int] | None = None

    @property
    def model(self) -> LanguageModel:
        """Get the loaded model, loading if necessary.

        Uses a global cache to share models across ActivationExtractor instances,
        preventing OOM from loading multiple copies of the same model.

        For remote=True, only loads tokenizer and config (no weights).
        """
        return self._backend.model

    @property
    def tokenizer(self) -> Any:
        """Get the model's tokenizer."""
        return self._backend.tokenizer

    @property
    def layer_indices(self) -> list[int]:
        """Get resolved layer indices."""
        if self._layer_indices is None:
            num_layers = get_num_layers_from_config(self.model_name)
            self._layer_indices = resolve_layers(
                self.layers_spec, num_layers, auto_candidates=self.auto_candidates
            )
        return self._layer_indices

    @property
    def num_layers(self) -> int:
        """Number of layers being extracted."""
        return len(self.layer_indices)

    def extract_batch(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract activations for a single batch of prompts.

        Delegates to the configured backend.

        Parameters
        ----------
        prompts : list[str]
            List of text prompts.
        layer_indices : list[int]
            Layer indices to extract from.
        **kwargs
            Backend-specific parameters (e.g., remote for NnsightBackend).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (activations, attention_mask)
        """
        return self._backend.extract_batch(prompts, layer_indices, **kwargs)

    def extract_batch_with_logits(
        self,
        prompts: list[str],
        layer_indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Extract activations AND logits for a single batch of prompts.

        Delegates to the configured backend.

        Parameters
        ----------
        prompts : list[str]
            List of text prompts.
        layer_indices : list[int]
            Layer indices to extract from.
        **kwargs
            Backend-specific parameters (e.g., logit_top_k).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
            (activations, attention_mask, logits, logits_indices)
        """
        return self._backend.extract_batch_with_logits(
            prompts, layer_indices, **kwargs
        )

    def extract(
        self,
        prompts: list[str],
        remote: bool = False,
        layers: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract activations for prompts.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to extract activations for.
        remote : bool
            Whether to use remote execution.
        layers : list[int] | None
            Specific layer indices to extract. If None, uses the default
            layer_indices configured at init. This parameter enables
            extracting only specific layers (e.g., for partial cache misses).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (activations, attention_mask)
        """
        layer_indices = layers if layers is not None else self.layer_indices
        if remote and hasattr(self._backend, '_get_model_for_remote'):
            model = self._backend._get_model_for_remote()
        else:
            model = self.model
        return extract_activations(
            model,
            prompts,
            layer_indices,
            remote=remote,
            batch_size=self.batch_size,
        )
