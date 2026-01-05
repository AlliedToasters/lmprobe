"""Activation extraction from language models via nnsight.

This module handles loading models and extracting intermediate activations
from specified layers. Supports both local and remote execution.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from nnsight import CONFIG, LanguageModel
from tqdm import tqdm

if TYPE_CHECKING:
    pass


# Global model cache to avoid loading the same model multiple times
# Key: (model_name, device), Value: LanguageModel
_MODEL_CACHE: dict[tuple[str, str], LanguageModel] = {}


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

    Reads the API key from NNSIGHT_API_KEY environment variable.

    Raises
    ------
    EnvironmentError
        If NNSIGHT_API_KEY is not set.
    """
    api_key = os.getenv("NNSIGHT_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "NNSIGHT_API_KEY environment variable is required for remote execution. "
            "Set it with: export NNSIGHT_API_KEY='your-key-here'"
        )
    CONFIG.API.APIKEY = api_key


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
            return getattr(config, attr)

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
        for idx in candidates:
            idx = int(idx)
            # Handle negative indexing
            if idx < 0:
                idx = num_layers + idx
            if not (0 <= idx < num_layers):
                raise ValueError(
                    f"Layer index {idx} out of range for model with {num_layers} layers. "
                    f"Valid range: [0, {num_layers - 1}] or [-{num_layers}, -1]"
                )
            resolved.append(idx)

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
    if remote:
        # For remote execution, don't load weights locally.
        # nnsight handles this by not specifying device_map.
        # See: https://nnsight.net/notebooks/features/remote_execution/
        model = LanguageModel(model_name)
    else:
        # Local execution - load weights to specified device
        if device == "auto":
            device_map = "auto"
        elif device == "cpu":
            device_map = {"": "cpu"}
        else:
            device_map = {"": device}

        model = LanguageModel(
            model_name,
            device_map=device_map,
            dispatch=True,
        )
    return model


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
    # Tokenize the prompts to get attention mask
    tokenized = model.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )

    # Use nnsight's tracer.cache() to collect multiple layer activations.
    # This pattern works for both local and remote execution.
    modules_to_cache = [model.model.layers[i] for i in layer_indices]

    with model.trace(tokenized, remote=remote) as tracer:
        cache = tracer.cache(modules=modules_to_cache).save()

    # Collect tensors from the cache
    # Cache structure differs slightly between local/remote:
    # - Remote: cache[key] is a dict with 'output' key
    # - Local: cache[key] is an Entry object with .output attribute
    activation_tensors = []
    for layer_idx in layer_indices:
        key = f"model.model.layers.{layer_idx}"
        entry = cache[key]

        # Handle both dict (remote) and Entry object (local) formats
        if hasattr(entry, "output"):
            output = entry.output
        else:
            output = entry["output"]

        # Handle both tuple outputs (hidden_states, ...) and direct tensors
        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output

        # Handle proxy vs direct tensor
        if hasattr(tensor, "value"):
            tensor = tensor.value

        activation_tensors.append(tensor)

    # Concatenate along hidden dimension
    # Result shape: (batch, seq_len, hidden_dim * num_layers)
    combined = torch.cat(activation_tensors, dim=-1)

    # Get attention mask from the tokenized input
    attention_mask = tokenized["attention_mask"]

    return combined, attention_mask


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
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        layers: int | list[int] | str = "middle",
        batch_size: int = 8,
        auto_candidates: list[int] | list[float] | None = None,
        remote: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.layers_spec = layers
        self.batch_size = batch_size
        self.auto_candidates = auto_candidates
        self.remote = remote

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
        if self._model is None:
            self._model = get_cached_model(
                self.model_name, self.device, remote=self.remote
            )
        return self._model

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
        return extract_activations(
            self.model,
            prompts,
            layer_indices,
            remote=remote,
            batch_size=self.batch_size,
        )
