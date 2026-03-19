"""Pooling strategies for aggregating token-level activations.

This module provides functions to reduce sequence-level activations
(batch, seq_len, hidden_dim) to fixed-size representations for classification.

There are two types of pooling:
1. **Activation pooling** (pre-probe): Reduces activations before classification
   - last_token, first_token, mean
2. **Score pooling** (post-probe): Classifies all tokens, then reduces scores
   - max, min (these require activation_pooling="all" internally)

The "all" strategy returns per-token activations without reduction.

Strategies can be explicitly prefixed with ``score:`` or ``activation:`` to
force the pooling stage.  For example:

- ``"score:mean"`` — classify each token, then average probabilities
- ``"activation:max"`` — take max activation per dimension, then classify once
- ``"mean"`` — pre-probe mean (default, same as ``"activation:mean"``)
- ``"max"`` — post-probe max (default, same as ``"score:max"``)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch


class ParsedPooling(NamedTuple):
    """Result of parsing a pooling strategy string."""

    base_strategy: str
    """The base pooling operation (e.g., ``"mean"``, ``"max"``)."""

    is_score_pooling: bool
    """If True, pool scores post-classification; if False, pool activations."""

    raw: str
    """The original unparsed strategy string."""


# Base strategies that can be used for either activation or score pooling
_BASE_STRATEGIES = frozenset({
    "last_token",
    "first_token",
    "mean",
    "max",
    "min",
})

# Default stage for each base strategy (backward compatibility)
_DEFAULT_STAGE: dict[str, str] = {
    "last_token": "activation",
    "first_token": "activation",
    "mean": "activation",
    "max": "score",
    "min": "score",
}

# Strategies valid for training (must produce fixed-size output)
TRAIN_POOLING_STRATEGIES = frozenset({
    "last_token",
    "first_token",
    "mean",
    "all",  # Expands each token as separate training example
})

# Strategies valid for inference (includes all base strategies with any prefix)
INFERENCE_POOLING_STRATEGIES = frozenset({
    "last_token",
    "first_token",
    "mean",
    "max",   # Score-level pooling (default)
    "min",   # Score-level pooling (default)
    "all",   # Returns per-token scores
})

# Legacy constant — strategies that default to score-level pooling
SCORE_POOLING_STRATEGIES = frozenset({"max", "min"})


def parse_pooling_strategy(strategy: str) -> ParsedPooling:
    """Parse a pooling strategy string into its components.

    Supports prefix convention: ``"score:mean"``, ``"activation:max"``,
    or bare names like ``"mean"`` (uses default stage).

    Parameters
    ----------
    strategy : str
        Pooling strategy, optionally prefixed with ``score:`` or
        ``activation:``.

    Returns
    -------
    ParsedPooling
        Parsed components.

    Raises
    ------
    ValueError
        If the strategy or prefix is not recognized.
    """
    if strategy == "all":
        return ParsedPooling(base_strategy="all", is_score_pooling=False, raw=strategy)

    if ":" in strategy:
        prefix, base = strategy.split(":", 1)
        if prefix not in ("score", "activation"):
            raise ValueError(
                f"Unknown pooling prefix: {prefix!r}. "
                f"Use 'score:' or 'activation:' (e.g., 'score:mean', 'activation:max')."
            )
        if base not in _BASE_STRATEGIES:
            raise ValueError(
                f"Unknown base pooling strategy: {base!r}. "
                f"Available: {sorted(_BASE_STRATEGIES)}"
            )
        return ParsedPooling(
            base_strategy=base,
            is_score_pooling=(prefix == "score"),
            raw=strategy,
        )

    # Bare name — use default stage
    if strategy in _DEFAULT_STAGE:
        return ParsedPooling(
            base_strategy=strategy,
            is_score_pooling=(_DEFAULT_STAGE[strategy] == "score"),
            raw=strategy,
        )

    raise ValueError(
        f"Unknown pooling strategy: {strategy!r}. "
        f"Available: {sorted(_BASE_STRATEGIES | {'all'})}"
    )


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
    # Works for both left-padding and right-padding
    batch_size = activations.shape[0]
    last_indices = torch.stack(
        [attention_mask[i].nonzero(as_tuple=True)[0][-1] for i in range(batch_size)]
    )

    # Gather the last token for each sequence
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


def pool_max(
    activations: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute max activation per dimension across all non-padding tokens.

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
        return activations.max(dim=1).values

    # Set padding positions to -inf so they don't affect max
    mask = attention_mask.unsqueeze(-1).bool()  # (batch, seq_len, 1)
    masked = activations.masked_fill(~mask, float("-inf"))
    return masked.max(dim=1).values


def pool_min(
    activations: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute min activation per dimension across all non-padding tokens.

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
        return activations.min(dim=1).values

    # Set padding positions to +inf so they don't affect min
    mask = attention_mask.unsqueeze(-1).bool()  # (batch, seq_len, 1)
    masked = activations.masked_fill(~mask, float("inf"))
    return masked.min(dim=1).values


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

    For score-level pooling (``"score:mean"``, ``"max"``, etc.), this returns
    ``pool_all`` so that all token activations are preserved for classification
    before score reduction.

    For activation-level pooling (``"mean"``, ``"activation:max"``, etc.),
    this returns the appropriate activation pooling function.

    Parameters
    ----------
    strategy : str
        Name of the pooling strategy, optionally prefixed with ``score:``
        or ``activation:``.

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
        "max": pool_max,
        "min": pool_min,
        "all": pool_all,
    }

    parsed = parse_pooling_strategy(strategy)

    if parsed.is_score_pooling:
        # Score-level pooling: need all token activations for classification
        return pool_all

    # Activation-level pooling: use the appropriate function
    if parsed.base_strategy in pooling_fns:
        return pooling_fns[parsed.base_strategy]

    raise ValueError(
        f"Unknown pooling strategy: {strategy!r}. "
        f"Available: {sorted(_BASE_STRATEGIES | {'all'})}"
    )


def reduce_scores(
    scores: torch.Tensor,
    strategy: str,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce per-token scores to a single score per sequence.

    Used for score-level pooling after classification. Supports all base
    strategies: ``max``, ``min``, ``mean``, ``last_token``, ``first_token``.

    The ``strategy`` may include a ``score:`` prefix (e.g., ``"score:mean"``),
    which is stripped before processing.

    Parameters
    ----------
    scores : torch.Tensor
        Shape (batch, seq_len) or (batch, seq_len, n_classes)
    strategy : str
        Base strategy name (e.g., ``"max"``, ``"mean"``, ``"score:mean"``).
    attention_mask : torch.Tensor | None
        Shape (batch, seq_len). 1 for real tokens, 0 for padding.

    Returns
    -------
    torch.Tensor
        Shape (batch,) or (batch, n_classes)
    """
    parsed = parse_pooling_strategy(strategy)
    base = parsed.base_strategy

    if base == "max":
        if attention_mask is not None:
            mask = attention_mask.bool()
            if scores.dim() == 3:
                mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(~mask, float("-inf"))
        return scores.max(dim=1).values

    if base == "min":
        if attention_mask is not None:
            mask = attention_mask.bool()
            if scores.dim() == 3:
                mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(~mask, float("inf"))
        return scores.min(dim=1).values

    if base == "mean":
        if attention_mask is None:
            return scores.mean(dim=1)
        mask = attention_mask.float()
        if scores.dim() == 3:
            mask = mask.unsqueeze(-1)
        masked_sum = (scores * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp(min=1)
        return masked_sum / token_counts

    if base == "last_token":
        if attention_mask is None:
            return scores[:, -1] if scores.dim() == 3 else scores[:, -1]
        batch_size = scores.shape[0]
        last_indices = torch.stack(
            [attention_mask[i].nonzero(as_tuple=True)[0][-1] for i in range(batch_size)]
        )
        batch_indices = torch.arange(batch_size, device=scores.device)
        if scores.dim() == 3:
            return scores[batch_indices, last_indices, :]
        return scores[batch_indices, last_indices]

    if base == "first_token":
        return scores[:, 0]

    raise ValueError(
        f"reduce_scores does not support strategy {strategy!r}. "
        f"Available: {sorted(_BASE_STRATEGIES)}"
    )


def _is_valid_inference_strategy(strategy: str) -> bool:
    """Check whether a strategy string is valid for inference.

    Accepts bare names in ``INFERENCE_POOLING_STRATEGIES`` and prefixed forms
    like ``"score:mean"`` or ``"activation:max"``.
    """
    if strategy in INFERENCE_POOLING_STRATEGIES:
        return True
    try:
        parse_pooling_strategy(strategy)
        return True
    except ValueError:
        return False


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

    # Validate train strategy (prefixed strategies not allowed for training)
    if train_strategy not in TRAIN_POOLING_STRATEGIES:
        raise ValueError(
            f"Invalid train_pooling: {train_strategy!r}. "
            f"Available: {sorted(TRAIN_POOLING_STRATEGIES)}"
        )

    # Validate inference strategy (allows prefixed strategies)
    if not _is_valid_inference_strategy(inference_strategy):
        raise ValueError(
            f"Invalid inference_pooling: {inference_strategy!r}. "
            f"Available: {sorted(INFERENCE_POOLING_STRATEGIES)} "
            f"(also accepts 'score:' or 'activation:' prefixes, "
            f"e.g., 'score:mean', 'activation:max')"
        )

    return train_strategy, inference_strategy
