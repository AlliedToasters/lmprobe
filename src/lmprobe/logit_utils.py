"""Utilities for computing logits from cached activations.

Given cached last-layer hidden states, compute logits locally by downloading
only the model's final norm weights and lm_head projection, avoiding a full
forward pass.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _detect_norm_type(config) -> tuple[str, float]:
    """Detect norm type and epsilon from model config attributes.

    Uses config attributes rather than a hardcoded model_type list,
    so new architectures are handled automatically.
    """
    if hasattr(config, "rms_norm_eps"):
        return "rms_norm", config.rms_norm_eps
    elif hasattr(config, "layer_norm_eps"):
        return "layer_norm", config.layer_norm_eps
    else:
        # Fallback: assume RMSNorm with standard epsilon
        logger.warning(
            "Could not detect norm type from config (no rms_norm_eps or "
            "layer_norm_eps attribute). Defaulting to rms_norm with eps=1e-5."
        )
        return "rms_norm", 1e-5


def download_lm_head_weights(
    model_name: str,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Download only norm + lm_head weights from a HuggingFace model.

    For sharded models, downloads only the 1-2 shard files containing
    the needed weights (not the full model).

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        Device to load tensors onto.
    dtype : torch.dtype | None
        Optional dtype cast for loaded tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, dict]
        (norm_weight, lm_head_weight, config_dict) where:
        - norm_weight: (hidden_dim,)
        - lm_head_weight: (vocab_size, hidden_dim)
        - config_dict: {"eps": float, "norm_type": str, "norm_bias": Tensor|None}
    """
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    norm_type, eps = _detect_norm_type(config)
    tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

    # Determine weight key names
    norm_key = "model.norm.weight"
    lm_head_key = "lm_head.weight"
    embed_key = "model.embed_tokens.weight"
    norm_bias_key = "model.norm.bias"

    needed_keys = {norm_key}
    if tie_word_embeddings:
        needed_keys.add(embed_key)
    else:
        needed_keys.add(lm_head_key)

    local_path = Path(model_name)
    is_local = local_path.is_dir()

    # Try sharded model first, then single file
    loaded = {}

    if is_local:
        index_path = local_path / "model.safetensors.index.json"
    else:
        try:
            index_path = Path(hf_hub_download(
                model_name, "model.safetensors.index.json"
            ))
        except Exception:
            index_path = None

    if index_path is not None and index_path.exists():
        # Sharded model: download only the needed shard files
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        # Identify which shard files contain our weights
        shard_files = set()
        for wk in needed_keys:
            if wk in weight_map:
                shard_files.add(weight_map[wk])
            elif wk == lm_head_key and embed_key in weight_map:
                shard_files.add(weight_map[embed_key])
                tie_word_embeddings = True

        for shard_file in shard_files:
            if is_local:
                shard_path = str(local_path / shard_file)
            else:
                shard_path = hf_hub_download(model_name, shard_file)
            with safe_open(shard_path, framework="pt", device=device) as f:
                for k in f.keys():
                    if k in needed_keys or (k == embed_key and tie_word_embeddings):
                        loaded[k] = f.get_tensor(k)
    else:
        # Single safetensors file
        if is_local:
            sf_path = str(local_path / "model.safetensors")
        else:
            sf_path = hf_hub_download(model_name, "model.safetensors")

        with safe_open(sf_path, framework="pt", device=device) as f:
            for k in f.keys():
                if k in needed_keys or (k == embed_key and tie_word_embeddings):
                    loaded[k] = f.get_tensor(k)

    # Extract tensors
    norm_weight = loaded[norm_key]

    if lm_head_key in loaded:
        lm_head_weight = loaded[lm_head_key]
    elif embed_key in loaded:
        lm_head_weight = loaded[embed_key]
    else:
        raise KeyError(
            f"Could not find lm_head or embed_tokens weights for {model_name}. "
            f"Available keys: {list(loaded.keys())}"
        )

    norm_bias = loaded.get(norm_bias_key)

    if dtype is not None:
        norm_weight = norm_weight.to(dtype)
        lm_head_weight = lm_head_weight.to(dtype)
        if norm_bias is not None:
            norm_bias = norm_bias.to(dtype)

    config_dict = {
        "eps": eps,
        "norm_type": norm_type,
        "norm_bias": norm_bias,
    }

    return norm_weight, lm_head_weight, config_dict


def apply_norm(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    norm_type: str = "rms_norm",
    norm_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply final layer normalization to hidden states.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Hidden states with shape (..., hidden_dim).
    norm_weight : torch.Tensor
        Norm weight with shape (hidden_dim,).
    eps : float
        Epsilon for numerical stability.
    norm_type : str
        "rms_norm" or "layer_norm".
    norm_bias : torch.Tensor | None
        Bias for layer norm (unused for RMSNorm).

    Returns
    -------
    torch.Tensor
        Normalized hidden states with same shape as input.
    """
    if norm_type == "rms_norm":
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return hidden_states * norm_weight
    elif norm_type == "layer_norm":
        hidden_dim = hidden_states.shape[-1]
        return F.layer_norm(
            hidden_states, (hidden_dim,), norm_weight, norm_bias, eps
        )
    else:
        raise ValueError(f"Unknown norm_type: {norm_type!r}")
