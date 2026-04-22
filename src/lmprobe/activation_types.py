"""Activation type definitions and MoE architecture detection.

This module defines the structured types used to specify what activations
to extract from a model and how to return them. It also provides
config-only detection of MoE (Mixture of Experts) architectures.

The key abstractions are:

- ``ExtractionSpec``: What to extract in a single forward pass.
- ``ExtractedBatch``: Structured result containing all requested activations.
- ``MoEInfo``: Metadata about a model's MoE architecture (router paths, expert count).
- ``detect_moe_info()``: Detect MoE architecture from HuggingFace config without loading weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


@dataclass
class PreTokenizedPrompts:
    """Sentinel container for caller-supplied tokenized input.

    Accepted anywhere the public API takes ``prompts: list[str]``, skipping
    lmprobe's internal tokenization. Use this when you need exact control
    over tokenization (e.g. ``add_special_tokens=False`` after external
    ``apply_chat_template``, a specific ``padding_side``, or a ``pad_token``
    choice different from lmprobe's ``eos_token`` default).

    Attributes
    ----------
    input_ids : torch.Tensor
        Shape ``(B, S)``, dtype ``int64``.
    attention_mask : torch.Tensor
        Shape ``(B, S)``, dtype ``int64`` or ``bool``. 1 = real, 0 = pad.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.input_ids.shape != self.attention_mask.shape:
            raise ValueError(
                f"input_ids {tuple(self.input_ids.shape)} and attention_mask "
                f"{tuple(self.attention_mask.shape)} must match."
            )
        if self.input_ids.dim() != 2:
            raise ValueError(
                f"Expected 2D input_ids (B, S); got shape "
                f"{tuple(self.input_ids.shape)}."
            )

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


class ActivationType(str, Enum):
    """Known activation types.

    Used as identifiers in cache keys, tensor descriptors, and extraction specs.
    """

    HIDDEN = "hidden"
    LOGITS = "logits"
    ROUTER_LOGITS = "router"
    # Future: ATTENTION = "attention", MLP = "mlp", etc.


@dataclass
class ExtractionSpec:
    """Specification of what to extract in a single forward pass.

    This replaces the pattern of having separate methods for each activation
    combination (extract_batch, extract_batch_with_logits, etc.).

    Parameters
    ----------
    hidden_layers : list[int]
        Layer indices for hidden state extraction.
    include_logits : bool
        Whether to capture lm_head logits.
    logit_top_k : int or None
        If set with include_logits, perform top-k on logits (remote only).
    router_layers : list[int] or None
        Layer indices for MoE router logit extraction. None means skip.
    router_module_template : str or None
        Format string for the router module path, e.g.
        ``"model.model.layers.{layer}.block_sparse_moe.gate"``.
        Required when router_layers is not None. Obtained from
        :func:`detect_moe_info`.
    router_hook_strategy : str
        How to capture router logits. ``"output"`` hooks the module's forward
        output (default, works when the gate module is called directly).
        ``"input_gate"`` hooks the parent MoE module and computes
        ``F.linear(input, module.gate.weight)`` — needed for architectures
        like DeepSeek that call ``F.linear`` directly instead of the gate
        module's ``__call__``.
    """

    hidden_layers: list[int] = field(default_factory=list)
    include_logits: bool = False
    logit_top_k: int | None = None
    router_layers: list[int] | None = None
    router_module_template: str | None = None
    router_hook_strategy: str = "output"

    def __post_init__(self) -> None:
        if self.router_layers and not self.router_module_template:
            raise ValueError(
                "router_module_template is required when router_layers is specified. "
                "Use detect_moe_info() to obtain the template for your model."
            )


@dataclass
class ExtractedBatch:
    """Structured result from a single batch extraction.

    Parameters
    ----------
    activations : torch.Tensor or None
        Hidden state activations, shape ``(batch, seq_len, hidden_dim * num_layers)``.
        None when no hidden layers were requested.
    attention_mask : torch.Tensor
        Attention mask, shape ``(batch, seq_len)``.
    logits : torch.Tensor or None
        LM head logits. Full: ``(batch, seq_len, vocab_size)``.
        Top-k: ``(batch, seq_len, K)``.
    logits_indices : torch.Tensor or None
        Top-k indices when logit_top_k was set, else None.
    router_logits : dict[int, torch.Tensor] or None
        Router gate outputs keyed by layer index.
        Each tensor has shape ``(batch, seq_len, num_experts)``.
        Dict (not concatenated) because num_experts != hidden_dim.
    hidden_per_layer : dict[int, torch.Tensor] or None
        Per-layer hidden states keyed by layer index.
        Each tensor has shape ``(batch, seq_len, hidden_dim)``.
        Populated alongside ``activations`` when hidden layers are requested.
    """

    activations: torch.Tensor | None
    attention_mask: torch.Tensor
    logits: torch.Tensor | None = None
    logits_indices: torch.Tensor | None = None
    router_logits: dict[int, torch.Tensor] | None = None
    hidden_per_layer: dict[int, torch.Tensor] | None = None


@dataclass(frozen=True)
class MoEInfo:
    """MoE architecture metadata extracted from model config.

    Parameters
    ----------
    num_experts : int
        Number of experts in the MoE layers.
    router_module_template : str
        Python attribute path template for the router gate module.
        Use ``template.format(layer=i)`` to get the path for layer ``i``.
        Examples:
        - Mixtral: ``"model.model.layers.{layer}.block_sparse_moe.gate"``
        - Qwen2-MoE: ``"model.model.layers.{layer}.mlp.gate"``
    moe_layer_indices : list[int] or None
        If not all layers are MoE (e.g., DeepSeek-V2 interleaves dense and
        MoE layers), this lists which layer indices have routers.
        None means all layers have routers.
    router_hook_strategy : str
        How to capture router logits. ``"output"`` (default) hooks the gate
        module's forward output. ``"input_gate"`` hooks the parent MoE
        module's forward, computing ``F.linear(input, gate.weight)`` to get
        router logits — needed when the gate is called via ``F.linear``
        instead of the module's ``__call__`` (e.g., DeepSeek-V2/V3).
    """

    num_experts: int
    router_module_template: str
    moe_layer_indices: list[int] | None = None
    router_hook_strategy: str = "output"


def detect_moe_info(model_name: str) -> MoEInfo | None:
    """Detect MoE architecture from a HuggingFace model config.

    Loads only the config (no weights) to determine whether the model
    uses Mixture of Experts and, if so, extracts the router module path
    template and expert count.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.

    Returns
    -------
    MoEInfo or None
        MoE metadata if the model is an MoE architecture, None for dense models.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", "")

    # Mixtral
    if model_type == "mixtral":
        num_experts = getattr(config, "num_local_experts", None)
        if num_experts is not None:
            return MoEInfo(
                num_experts=num_experts,
                router_module_template=(
                    "model.model.layers.{layer}.block_sparse_moe.gate"
                ),
            )

    # Qwen2-MoE
    if model_type == "qwen2_moe":
        num_experts = getattr(config, "num_experts", None)
        if num_experts is not None:
            return MoEInfo(
                num_experts=num_experts,
                router_module_template="model.model.layers.{layer}.mlp.gate",
            )

    # DeepSeek-V2 / DeepSeek-V3
    # NOTE: DeepSeek MoE forward uses F.linear(x, gate.weight) instead of
    # gate(x), bypassing forward hooks on the gate module. We hook the
    # parent MoE module and compute logits from its input + gate weight.
    if model_type in ("deepseek_v2", "deepseek_v3"):
        num_experts = getattr(config, "n_routed_experts", None)
        if num_experts is not None:
            # DeepSeek interleaves dense and MoE layers.
            # MoE layers start at first_k_dense_replace and repeat every
            # moe_layer_freq layers.
            first_moe = getattr(config, "first_k_dense_replace", 1)
            moe_freq = getattr(config, "moe_layer_freq", 1)
            num_layers = getattr(config, "num_hidden_layers", 0)
            moe_indices = [
                i for i in range(num_layers)
                if i >= first_moe and (i - first_moe) % moe_freq == 0
            ]
            return MoEInfo(
                num_experts=num_experts,
                router_module_template="model.model.layers.{layer}.mlp",
                moe_layer_indices=moe_indices if moe_indices else None,
                router_hook_strategy="input_gate",
            )

    # DBRX
    if model_type == "dbrx":
        ffn_config = getattr(config, "ffn_config", None)
        if ffn_config is not None:
            num_experts = getattr(ffn_config, "moe_num_experts", None)
            if num_experts is not None:
                return MoEInfo(
                    num_experts=num_experts,
                    router_module_template=(
                        "model.transformer.blocks.{layer}.ffn.router.layer"
                    ),
                )

    # Arctic (Snowflake)
    if model_type == "arctic":
        num_experts = getattr(config, "num_local_experts", None)
        if num_experts is not None:
            return MoEInfo(
                num_experts=num_experts,
                router_module_template=(
                    "model.model.layers.{layer}.block_sparse_moe.gate"
                ),
            )

    # OLMoE
    if model_type == "olmoe":
        num_experts = getattr(config, "num_experts", None)
        if num_experts is not None:
            return MoEInfo(
                num_experts=num_experts,
                router_module_template="model.model.layers.{layer}.mlp.gate",
            )

    return None


def get_router_module(model: object, template: str, layer: int) -> object:
    """Resolve a router module from a model using a dot-path template.

    Parameters
    ----------
    model : object
        The loaded model (HuggingFace or nnsight LanguageModel).
    template : str
        Dot-path template, e.g.
        ``"model.model.layers.{layer}.block_sparse_moe.gate"``.
    layer : int
        Layer index to substitute into the template.

    Returns
    -------
    object
        The router module.

    Raises
    ------
    AttributeError
        If the module path does not exist on the model.
    """
    path = template.format(layer=layer)
    # Strip leading "model." since we start from the model object itself
    # (the template includes "model." as the top-level attribute for nnsight
    # LanguageModel which wraps the HF model as .model)
    parts = path.split(".")
    obj = model
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]  # type: ignore[index]
        else:
            obj = getattr(obj, part)
    return obj


def validate_router_layers(
    moe_info: MoEInfo,
    requested_layers: list[int],
) -> list[int]:
    """Validate that requested layers actually have MoE routers.

    Parameters
    ----------
    moe_info : MoEInfo
        MoE architecture info from :func:`detect_moe_info`.
    requested_layers : list[int]
        Layer indices the user wants router logits from.

    Returns
    -------
    list[int]
        Validated layer indices (subset of requested_layers that have routers).

    Raises
    ------
    ValueError
        If none of the requested layers have routers.
    """
    if moe_info.moe_layer_indices is None:
        # All layers have routers
        return requested_layers

    valid = [i for i in requested_layers if i in moe_info.moe_layer_indices]
    if not valid:
        raise ValueError(
            f"None of the requested layers {requested_layers} have MoE routers. "
            f"MoE layers for this model: {moe_info.moe_layer_indices}"
        )
    skipped = set(requested_layers) - set(valid)
    if skipped:
        import logging

        logging.getLogger(__name__).warning(
            "Layers %s do not have MoE routers and will be skipped. "
            "MoE layers: %s",
            sorted(skipped),
            moe_info.moe_layer_indices,
        )
    return valid
