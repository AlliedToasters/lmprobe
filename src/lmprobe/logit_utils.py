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

# Model types that use RMSNorm (vs LayerNorm)
_RMS_NORM_TYPES = frozenset({
    "llama", "mistral", "mixtral", "qwen2", "qwen2_moe",
    "gemma", "gemma2", "phi3", "cohere", "deepseek_v2",
    "internlm2", "stablelm",
})


def download_lm_head_weights(
    model_name: str,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Download only norm + lm_head weights from a HuggingFace model.

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
    from pathlib import Path

    from safetensors import safe_open
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", "")
    norm_type = "rms_norm" if model_type in _RMS_NORM_TYPES else "layer_norm"
    eps = getattr(config, "rms_norm_eps", None) or getattr(config, "layer_norm_eps", 1e-5)
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

    # Try to locate the safetensors files
    local_path = Path(model_name)
    if local_path.is_dir():
        model_dir = local_path
    else:
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(
            model_name,
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
        ))

    # Check for sharded model
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        import json

        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        # Collect unique shard files we need
        shard_files = set()
        for wk in needed_keys:
            if wk in weight_map:
                shard_files.add(weight_map[wk])
            elif wk == lm_head_key and embed_key in weight_map:
                # Tied embeddings: lm_head not in map, use embed_tokens
                shard_files.add(weight_map[embed_key])
                tie_word_embeddings = True

        loaded = {}
        for shard_file in shard_files:
            shard_path = str(model_dir / shard_file)
            with safe_open(shard_path, framework="pt", device=device) as f:
                for k in f.keys():
                    if k in needed_keys or (k == embed_key and tie_word_embeddings):
                        loaded[k] = f.get_tensor(k)
    else:
        # Single safetensors file
        sf_path = model_dir / "model.safetensors"
        if not sf_path.exists():
            raise FileNotFoundError(
                f"No safetensors file found for model {model_name} at {model_dir}"
            )
        loaded = {}
        with safe_open(str(sf_path), framework="pt", device=device) as f:
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

    # Check for norm bias
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
