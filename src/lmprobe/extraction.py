"""Activation extraction from language models via nnsight.

This module handles loading models and extracting intermediate activations
from specified layers. Supports both local and remote execution.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from nnsight import CONFIG, LanguageModel

if TYPE_CHECKING:
    pass


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


def get_num_layers(model: LanguageModel) -> int:
    """Get the number of transformer layers in a model.

    Parameters
    ----------
    model : LanguageModel
        The nnsight LanguageModel.

    Returns
    -------
    int
        Number of transformer layers.
    """
    # For Llama-style models, layers are at model.model.layers
    return len(model.model.layers)


def resolve_layers(
    layers: int | list[int] | str,
    num_layers: int,
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

    num_layers : int
        Total number of layers in the model.

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
        f"Use int, list[int], 'middle', 'last', or 'all'."
    )


def load_model(
    model_name: str,
    device: str = "auto",
) -> LanguageModel:
    """Load a language model via nnsight.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device specification. "auto" uses device_map="auto".

    Returns
    -------
    LanguageModel
        The loaded nnsight model.
    """
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


def extract_activations(
    model: LanguageModel,
    prompts: list[str],
    layer_indices: list[int],
    remote: bool = False,
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

    # Storage for layer activations
    layer_activations = []

    # Tokenize the prompts to get attention mask
    tokenized = model.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )

    with model.trace(tokenized, remote=remote) as tracer:
        for layer_idx in layer_indices:
            # For Llama-style models: model.model.layers[i].output is the hidden state
            # Note: In nnsight, .output gives the full tensor, not a tuple
            hidden_state = model.model.layers[layer_idx].output.save()
            layer_activations.append(hidden_state)

    # After trace context, collect the actual tensor values
    # nnsight returns tensors directly after .save() in local mode,
    # or proxies with .value in remote mode
    activation_tensors = []
    for act in layer_activations:
        if hasattr(act, "value"):
            activation_tensors.append(act.value)
        else:
            activation_tensors.append(act)

    # Concatenate along hidden dimension
    # Result shape: (batch, seq_len, hidden_dim * num_layers)
    combined = torch.cat(activation_tensors, dim=-1)

    # Get attention mask from the tokenized input
    attention_mask = tokenized["attention_mask"]

    return combined, attention_mask


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
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        layers: int | list[int] | str = "middle",
    ):
        self.model_name = model_name
        self.device = device
        self.layers_spec = layers

        # Lazy-loaded
        self._model: LanguageModel | None = None
        self._layer_indices: list[int] | None = None

    @property
    def model(self) -> LanguageModel:
        """Get the loaded model, loading if necessary."""
        if self._model is None:
            self._model = load_model(self.model_name, self.device)
        return self._model

    @property
    def layer_indices(self) -> list[int]:
        """Get resolved layer indices."""
        if self._layer_indices is None:
            num_layers = get_num_layers(self.model)
            self._layer_indices = resolve_layers(self.layers_spec, num_layers)
        return self._layer_indices

    @property
    def num_layers(self) -> int:
        """Number of layers being extracted."""
        return len(self.layer_indices)

    def extract(
        self,
        prompts: list[str],
        remote: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract activations for prompts.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to extract activations for.
        remote : bool
            Whether to use remote execution.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (activations, attention_mask)
        """
        return extract_activations(
            self.model,
            prompts,
            self.layer_indices,
            remote=remote,
        )
