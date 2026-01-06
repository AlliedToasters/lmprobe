"""Unified cache for extracting activations and perplexity in a single forward pass.

This module provides the UnifiedCache class which optimizes extraction by capturing
both layer activations and logits (for perplexity) in a single nnsight trace.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from .cache import (
    get_prompt_cache_dir,
    is_prompt_fully_cached,
    is_prompt_perplexity_cached,
    is_prompt_pooled_cached,
    load_prompt_activations,
    load_prompt_perplexity,
    load_prompt_pooled_activations,
    save_prompt_activations,
    save_prompt_perplexity,
    save_prompt_pooled_activations,
)
from .extraction import (
    _extract_batch_with_logits,
    compute_perplexity_from_logits,
    configure_remote,
    get_cached_model,
    get_num_layers_from_config,
    resolve_layers,
)
from .pooling import get_pooling_fn, TRAIN_POOLING_STRATEGIES

if TYPE_CHECKING:
    from nnsight import LanguageModel


logger = logging.getLogger(__name__)


@dataclass
class WarmupStats:
    """Statistics from a warmup operation."""

    total_prompts: int
    activations_cached: int
    activations_extracted: int
    perplexity_cached: int
    perplexity_extracted: int
    elapsed_seconds: float

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompts that had activations cached."""
        if self.total_prompts == 0:
            return 0.0
        return self.activations_cached / self.total_prompts

    def __repr__(self) -> str:
        return (
            f"WarmupStats(total={self.total_prompts}, "
            f"activations={self.activations_cached} cached + {self.activations_extracted} extracted, "
            f"perplexity={self.perplexity_cached} cached + {self.perplexity_extracted} extracted, "
            f"time={self.elapsed_seconds:.1f}s)"
        )


class UnifiedCache:
    """Extracts activations and perplexity in a single forward pass.

    This class provides efficient extraction by capturing both layer activations
    and lm_head logits in a single nnsight trace, then computing perplexity
    features from the logits.

    Parameters
    ----------
    model : str
        HuggingFace model ID or local path.
    layers : int | list[int] | str, default="all"
        Which layers to extract activations from:
        - int: Single layer (negative indexing supported)
        - list[int]: Multiple layers
        - "all": All layers (default)
        - "middle": Middle third of layers
        - "last": Last layer only
    compute_perplexity : bool, default=True
        Whether to also capture logits and compute perplexity features.
    device : str, default="auto"
        Device for model inference.
    remote : bool, default=False
        Use nnsight remote execution (requires NNSIGHT_API_KEY).
    batch_size : int, default=8
        Number of prompts to process at once.
    cache_pooled : bool, default=False
        If True, apply pooling BEFORE caching and store only the pooled
        activations. This reduces disk usage by ~100x (storing only
        (1, hidden_dim) per layer instead of (1, seq_len, hidden_dim)).

        WARNING: When cache_pooled=True, you must use the same pooling
        strategy for all operations. The cached data cannot be re-pooled
        with a different strategy.
    pooling : str, default="last_token"
        Pooling strategy to use when cache_pooled=True. Options:
        - "last_token": Use the last non-padding token (default)
        - "first_token": Use the first token
        - "mean": Mean of all non-padding tokens

    Examples
    --------
    >>> # Memory-efficient caching (recommended for large models)
    >>> cache = UnifiedCache(
    ...     model="meta-llama/Llama-3.1-70B",
    ...     layers="all",
    ...     cache_pooled=True,      # Store only pooled activations
    ...     pooling="last_token",   # ~100x less disk space
    ... )
    >>> stats = cache.warmup(prompts)
    >>> print(f"Extracted {stats.activations_extracted} prompts")
    """

    def __init__(
        self,
        model: str,
        layers: int | list[int] | str = "all",
        compute_perplexity: bool = True,
        device: str = "auto",
        remote: bool = False,
        batch_size: int = 8,
        cache_pooled: bool = False,
        pooling: str = "last_token",
    ):
        self.model_name = model
        self.layers_spec = layers
        self.compute_perplexity = compute_perplexity
        self.device = device
        self.remote = remote
        self.batch_size = batch_size
        self.cache_pooled = cache_pooled
        self.pooling = pooling

        # Validate pooling strategy
        if cache_pooled and pooling not in TRAIN_POOLING_STRATEGIES:
            raise ValueError(
                f"Invalid pooling strategy for cache_pooled: {pooling!r}. "
                f"Available: {sorted(TRAIN_POOLING_STRATEGIES - {'all'})}"
            )
        if cache_pooled and pooling == "all":
            raise ValueError(
                "pooling='all' is not valid with cache_pooled=True. "
                "Use 'last_token', 'first_token', or 'mean'."
            )

        # Lazy-loaded
        self._model: LanguageModel | None = None
        self._layer_indices: list[int] | None = None
        self._pooling_fn = get_pooling_fn(pooling) if cache_pooled else None

    @property
    def model(self) -> LanguageModel:
        """Get the loaded model, loading if necessary."""
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
            self._layer_indices = resolve_layers(self.layers_spec, num_layers)
        return self._layer_indices

    def _check_cache_status(
        self, prompts: list[str]
    ) -> tuple[list[str], list[str]]:
        """Check which prompts need extraction.

        Returns
        -------
        tuple
            - prompts needing activation extraction
            - prompts needing perplexity extraction (if compute_perplexity=True)
        """
        required_layers = set(self.layer_indices)

        need_activations = []
        need_perplexity = []

        for prompt in prompts:
            # Check appropriate cache based on cache_pooled setting
            if self.cache_pooled:
                act_cached = is_prompt_pooled_cached(
                    self.model_name, prompt, required_layers, self.pooling
                )
            else:
                act_cached = is_prompt_fully_cached(
                    self.model_name, prompt, required_layers
                )

            ppl_cached = (
                is_prompt_perplexity_cached(self.model_name, prompt)
                if self.compute_perplexity
                else True
            )

            if not act_cached:
                need_activations.append(prompt)
            if not ppl_cached:
                need_perplexity.append(prompt)

        return need_activations, need_perplexity

    def warmup(
        self,
        prompts: list[str],
        remote: bool | None = None,
    ) -> WarmupStats:
        """Extract and cache activations/perplexity for prompts.

        This method checks the cache, identifies what needs extraction,
        and performs minimal forward passes to fill the cache.

        Parameters
        ----------
        prompts : list[str]
            Text prompts to warm up the cache for.
        remote : bool | None
            Override the instance-level remote setting.

        Returns
        -------
        WarmupStats
            Statistics about the warmup operation.
        """
        start_time = time.time()
        remote = self.remote if remote is None else remote

        if remote:
            configure_remote()

        layer_indices = sorted(self.layer_indices)

        # Check cache status
        need_activations, need_perplexity = self._check_cache_status(prompts)

        # Compute which prompts need unified extraction
        # (prompts that need BOTH or where we can get both cheaply)
        need_activations_set = set(need_activations)
        need_perplexity_set = set(need_perplexity)
        need_unified = need_activations_set | need_perplexity_set
        need_unified_list = [p for p in prompts if p in need_unified]

        activations_cached = len(prompts) - len(need_activations)
        perplexity_cached = len(prompts) - len(need_perplexity)
        activations_extracted = 0
        perplexity_extracted = 0

        logger.info(
            f"[UNIFIED] Checking cache for {len(prompts)} prompts, "
            f"layers: {layer_indices}"
        )
        logger.info(
            f"[UNIFIED] Activations: {activations_cached} cached, "
            f"{len(need_activations)} need extraction"
        )
        if self.compute_perplexity:
            logger.info(
                f"[UNIFIED] Perplexity: {perplexity_cached} cached, "
                f"{len(need_perplexity)} need extraction"
            )

        if need_unified_list:
            model = self.model
            num_batches = (len(need_unified_list) + self.batch_size - 1) // self.batch_size

            logger.info(
                f"[UNIFIED] Extracting {len(need_unified_list)} prompts in "
                f"{num_batches} batches"
            )

            with torch.no_grad():
                for batch_idx in tqdm(
                    range(0, len(need_unified_list), self.batch_size),
                    total=num_batches,
                    desc="Unified extraction",
                    unit="batch",
                ):
                    batch_prompts = need_unified_list[
                        batch_idx : batch_idx + self.batch_size
                    ]

                    # Single forward pass captures both activations and logits
                    batch_acts, batch_mask, batch_logits = _extract_batch_with_logits(
                        model, batch_prompts, layer_indices, remote=remote
                    )

                    # Compute perplexity features from logits
                    if self.compute_perplexity:
                        # Get input_ids for perplexity computation
                        tokenized = model.tokenizer(
                            batch_prompts,
                            return_tensors="pt",
                            padding=True,
                        )
                        ppl_features = compute_perplexity_from_logits(
                            batch_logits,
                            tokenized["input_ids"],
                            batch_mask,
                        )

                    # Save each prompt's data
                    for j, prompt in enumerate(batch_prompts):
                        # Save activations if needed
                        if prompt in need_activations_set:
                            prompt_acts = batch_acts[j : j + 1]
                            prompt_mask = batch_mask[j : j + 1]

                            if self.cache_pooled:
                                # Pool before saving - ~100x less disk space!
                                # prompt_acts shape: (1, seq_len, n_layers * hidden_dim)
                                pooled_acts = self._pooling_fn(prompt_acts, prompt_mask)
                                # pooled_acts shape: (1, n_layers * hidden_dim)
                                save_prompt_pooled_activations(
                                    self.model_name,
                                    prompt,
                                    layer_indices,
                                    pooled_acts,
                                    self.pooling,
                                )
                            else:
                                # Save full sequence (original behavior)
                                save_prompt_activations(
                                    self.model_name,
                                    prompt,
                                    layer_indices,
                                    prompt_acts,
                                    prompt_mask,
                                )
                            activations_extracted += 1

                        # Save perplexity if needed
                        if self.compute_perplexity and prompt in need_perplexity_set:
                            save_prompt_perplexity(
                                self.model_name, prompt, ppl_features[j]
                            )
                            perplexity_extracted += 1

        elapsed = time.time() - start_time

        stats = WarmupStats(
            total_prompts=len(prompts),
            activations_cached=activations_cached,
            activations_extracted=activations_extracted,
            perplexity_cached=perplexity_cached,
            perplexity_extracted=perplexity_extracted,
            elapsed_seconds=elapsed,
        )

        logger.info(
            f"[UNIFIED] Complete: {activations_extracted} activations + "
            f"{perplexity_extracted} perplexity extracted in {elapsed:.1f}s"
        )

        return stats

    def get_activations(
        self,
        prompts: list[str],
        remote: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get activations for prompts (from cache or via extraction).

        Parameters
        ----------
        prompts : list[str]
            Text prompts.
        remote : bool | None
            Override the instance-level remote setting.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            If cache_pooled=False:
                (activations, attention_mask) where activations has shape
                (batch, seq_len, n_layers * hidden_dim).
            If cache_pooled=True:
                (activations, None) where activations has shape
                (batch, n_layers * hidden_dim). Already pooled, no mask needed.
        """
        # Ensure cache is warm
        self.warmup(prompts, remote=remote)

        layer_indices = sorted(self.layer_indices)

        if self.cache_pooled:
            # Load pooled activations - already aggregated, no padding needed
            all_activations = []
            for prompt in prompts:
                acts = load_prompt_pooled_activations(
                    self.model_name, prompt, layer_indices, self.pooling
                )
                all_activations.append(acts)

            # Concatenate along batch dimension
            # Each acts has shape (1, n_layers * hidden_dim)
            return torch.cat(all_activations, dim=0), None

        else:
            # Load full-sequence activations (original behavior)
            all_activations = []
            all_masks = []

            for prompt in prompts:
                acts, mask = load_prompt_activations(
                    self.model_name, prompt, layer_indices
                )
                all_activations.append(acts)
                all_masks.append(mask)

            # Pad and concatenate
            max_seq_len = max(a.shape[1] for a in all_activations)
            hidden_dim = all_activations[0].shape[2]

            padded_acts = []
            padded_masks = []

            for acts, mask in zip(all_activations, all_masks):
                seq_len = acts.shape[1]
                if seq_len < max_seq_len:
                    pad_size = max_seq_len - seq_len
                    acts = torch.cat(
                        [acts, torch.zeros(1, pad_size, hidden_dim, dtype=acts.dtype)],
                        dim=1,
                    )
                    mask = torch.cat(
                        [mask, torch.zeros(1, pad_size, dtype=mask.dtype)], dim=1
                    )
                padded_acts.append(acts)
                padded_masks.append(mask)

            return torch.cat(padded_acts, dim=0), torch.cat(padded_masks, dim=0)

    def get_perplexity(
        self,
        prompts: list[str],
        remote: bool | None = None,
    ) -> np.ndarray:
        """Get perplexity features for prompts (from cache or via extraction).

        Parameters
        ----------
        prompts : list[str]
            Text prompts.
        remote : bool | None
            Override the instance-level remote setting.

        Returns
        -------
        np.ndarray
            Perplexity features with shape (n_prompts, 3).
        """
        if not self.compute_perplexity:
            raise ValueError(
                "UnifiedCache was created with compute_perplexity=False. "
                "Create a new instance with compute_perplexity=True."
            )

        # Ensure cache is warm
        self.warmup(prompts, remote=remote)

        # Load from cache
        features = []
        for prompt in prompts:
            ppl = load_prompt_perplexity(self.model_name, prompt)
            features.append(ppl.numpy())

        return np.array(features)
