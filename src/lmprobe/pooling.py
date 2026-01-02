"""Pooling strategies for aggregating token-level activations.

This module provides functions to reduce sequence-level activations
(batch, seq_len, hidden_dim) to fixed-size representations for classification.

There are two types of pooling:
1. **Activation pooling**: Reduces activations before classification
   - last_token, first_token, mean
2. **Score pooling**: Classifies all tokens, then reduces scores
   - max, min (these require activation_pooling="all" internally)

The "all" strategy returns per-token activations without reduction.
"""

from __future__ import annotations

from typing import Callable

import torch


# Strategies valid for training (must produce fixed-size output)
TRAIN_POOLING_STRATEGIES = frozenset({
    "last_token",
    "first_token",
    "mean",
    "all",  # Expands each token as separate training example
})

# Strategies valid for inference
INFERENCE_POOLING_STRATEGIES = frozenset({
    "last_token",
    "first_token",
    "mean",
    "max",   # Score-level pooling
    "min",   # Score-level pooling
    "all",   # Returns per-token scores
})

# Strategies that require classifying all tokens first, then reducing scores
SCORE_POOLING_STRATEGIES = frozenset({"max", "min"})


def pool_last_token(
    activations: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract the last non-padding token's activation.

    Parameters
    ----------
    activations : torch.Tensor
        Shape (batch, seq_len, hidden_dim)
    attention_mask : torch.Tensor | None
        Shape (batch, seq_len). 1 for real tokens, 0 for padding.
        If None, assumes no padding (uses last position).

    Returns
    -------
    torch.Tensor
        Shape (batch, hidden_dim)
    """
    if attention_mask is None:
        # No padding, just take the last token
        return activations[:, -1, :]

    # Find the last non-padding position for each sequence
    # attention_mask is 1 for real tokens, 0 for padding
    seq_lengths = attention_mask.sum(dim=1)  # (batch,)
    last_indices = (seq_lengths - 1).long()  # (batch,)

    # Gather the last token for each sequence
    batch_size = activations.shape[0]
    batch_indices = torch.arange(batch_size, device=activations.device)
    return activations[batch_indices, last_indices, :]


def pool_first_token(
    activations: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract the first token's activation.

    Parameters
    ----------
    activations : torch.Tensor
        Shape (batch, seq_len, hidden_dim)
    attention_mask : torch.Tensor | None
        Ignored for first_token pooling (first token is never padding).

    Returns
    -------
    torch.Tensor
        Shape (batch, hidden_dim)
    """
    return activations[:, 0, :]


def pool_mean(
    activations: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute mean activation across all non-padding tokens.

    Parameters
    ----------
    activations : torch.Tensor
        Shape (batch, seq_len, hidden_dim)
    attention_mask : torch.Tensor | None
        Shape (batch, seq_len). 1 for real tokens, 0 for padding.
        If None, assumes no padding.

    Returns
    -------
    torch.Tensor
        Shape (batch, hidden_dim)
    """
    if attention_mask is None:
        return activations.mean(dim=1)

    # Expand mask for broadcasting: (batch, seq_len, 1)
    mask = attention_mask.unsqueeze(-1).float()

    # Sum of activations for real tokens
    masked_sum = (activations * mask).sum(dim=1)  # (batch, hidden_dim)

    # Count of real tokens
    token_counts = mask.sum(dim=1)  # (batch, 1)

    # Avoid division by zero
    token_counts = token_counts.clamp(min=1)

    return masked_sum / token_counts


def pool_all(
    activations: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return all token activations unchanged.

    Parameters
    ----------
    activations : torch.Tensor
        Shape (batch, seq_len, hidden_dim)
    attention_mask : torch.Tensor | None
        Not used, but accepted for API consistency.

    Returns
    -------
    torch.Tensor
        Shape (batch, seq_len, hidden_dim) - unchanged
    """
    return activations


def get_pooling_fn(strategy: str) -> Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]:
    """Get the pooling function for a strategy name.

    Parameters
    ----------
    strategy : str
        Name of the pooling strategy.

    Returns
    -------
    Callable
        The pooling function.

    Raises
    ------
    ValueError
        If the strategy is not recognized.
    """
    pooling_fns = {
        "last_token": pool_last_token,
        "first_token": pool_first_token,
        "mean": pool_mean,
        "all": pool_all,
    }

    if strategy in pooling_fns:
        return pooling_fns[strategy]

    if strategy in SCORE_POOLING_STRATEGIES:
        # For max/min, we need to pool all activations first,
        # then reduce scores after classification
        return pool_all

    raise ValueError(
        f"Unknown pooling strategy: {strategy!r}. "
        f"Available: {sorted(TRAIN_POOLING_STRATEGIES | INFERENCE_POOLING_STRATEGIES)}"
    )


def reduce_scores(
    scores: torch.Tensor,
    strategy: str,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce per-token scores to a single score per sequence.

    Used for score-level pooling strategies (max, min) after classification.

    Parameters
    ----------
    scores : torch.Tensor
        Shape (batch, seq_len) or (batch, seq_len, n_classes)
    strategy : str
        "max" or "min"
    attention_mask : torch.Tensor | None
        Shape (batch, seq_len). 1 for real tokens, 0 for padding.

    Returns
    -------
    torch.Tensor
        Shape (batch,) or (batch, n_classes)
    """
    if strategy not in SCORE_POOLING_STRATEGIES:
        raise ValueError(f"reduce_scores only supports {SCORE_POOLING_STRATEGIES}, got {strategy!r}")

    # Handle masking for padded sequences
    if attention_mask is not None:
        # Set padding positions to -inf (for max) or +inf (for min)
        mask = attention_mask.bool()
        if scores.dim() == 3:
            mask = mask.unsqueeze(-1)

        if strategy == "max":
            scores = scores.masked_fill(~mask, float("-inf"))
        else:  # min
            scores = scores.masked_fill(~mask, float("inf"))

    if strategy == "max":
        if scores.dim() == 3:
            return scores.max(dim=1).values
        return scores.max(dim=1).values
    else:  # min
        if scores.dim() == 3:
            return scores.min(dim=1).values
        return scores.min(dim=1).values


def resolve_pooling(
    pooling: str | None,
    train_pooling: str | None,
    inference_pooling: str | None,
) -> tuple[str, str]:
    """Resolve pooling parameters to concrete train/inference strategies.

    Parameters
    ----------
    pooling : str | None
        Base pooling strategy for both train and inference.
    train_pooling : str | None
        Override for training. Takes precedence over pooling.
    inference_pooling : str | None
        Override for inference. Takes precedence over pooling.

    Returns
    -------
    tuple[str, str]
        (train_strategy, inference_strategy)

    Raises
    ------
    ValueError
        If no pooling strategy is specified, or if invalid strategies are used.
    """
    # Resolve train pooling
    if train_pooling is not None:
        train_strategy = train_pooling
    elif pooling is not None:
        train_strategy = pooling
    else:
        train_strategy = "last_token"  # default

    # Resolve inference pooling
    if inference_pooling is not None:
        inference_strategy = inference_pooling
    elif pooling is not None:
        inference_strategy = pooling
    else:
        inference_strategy = "last_token"  # default

    # Validate
    if train_strategy not in TRAIN_POOLING_STRATEGIES:
        raise ValueError(
            f"Invalid train_pooling: {train_strategy!r}. "
            f"Available: {sorted(TRAIN_POOLING_STRATEGIES)}"
        )

    if inference_strategy not in INFERENCE_POOLING_STRATEGIES:
        raise ValueError(
            f"Invalid inference_pooling: {inference_strategy!r}. "
            f"Available: {sorted(INFERENCE_POOLING_STRATEGIES)}"
        )

    return train_strategy, inference_strategy
