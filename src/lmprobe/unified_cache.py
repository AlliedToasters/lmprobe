"""Unified cache for extracting activations and perplexity in a single forward pass.

This module provides the UnifiedCache class which optimizes extraction by capturing
both layer activations and logits (for perplexity) in a single nnsight trace.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm

from .backends import resolve_backend
from .cache import (
    batch_check_cache_status,
    evict,
    is_prompt_logits_cached,
    is_prompt_perplexity_cached,
    load_pooled_batch,
    load_prompt_activations,
    load_prompt_logits,
    load_prompt_perplexity,
    save_prompt_activations,
    save_prompt_logits,
    save_prompt_perplexity,
    save_prompt_pooled_activations,
    save_prompt_topk_logits,
)
from .extraction import (
    compute_perplexity_from_logits,
    get_num_layers_from_config,
    resolve_layers,
)
from .pooling import TRAIN_POOLING_STRATEGIES, get_pooling_fn

if TYPE_CHECKING:
    from .backends import ExtractionBackend


logger = logging.getLogger(__name__)


def _release_memory() -> None:
    """Release freed memory back to the OS.

    On Linux, glibc malloc holds freed pages in its arena by default,
    causing RSS to grow monotonically even after gc.collect(). This
    calls malloc_trim(0) to return unused pages to the OS, and also
    clears the CUDA cache if available.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass  # Not Linux or libc unavailable

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class WarmupStats:
    """Statistics from a warmup operation."""

    total_prompts: int
    activations_cached: int
    activations_extracted: int
    perplexity_cached: int
    perplexity_extracted: int
    elapsed_seconds: float
    logits_cached: int = 0
    logits_extracted: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompts that had activations cached."""
        if self.total_prompts == 0:
            return 0.0
        return self.activations_cached / self.total_prompts

    def __repr__(self) -> str:
        parts = [
            f"WarmupStats(total={self.total_prompts}",
            f"activations={self.activations_cached} cached "
            f"+ {self.activations_extracted} extracted",
            f"perplexity={self.perplexity_cached} cached "
            f"+ {self.perplexity_extracted} extracted",
        ]
        if self.logits_cached or self.logits_extracted:
            parts.append(
                f"logits={self.logits_cached} cached "
                f"+ {self.logits_extracted} extracted"
            )
        parts.append(f"time={self.elapsed_seconds:.1f}s)")
        return ", ".join(parts)


@dataclass
class CachedLogits:
    """Cached logits from language model extraction.

    Attributes
    ----------
    values : torch.Tensor
        Logit values. Shape (batch, positions, vocab_size) for full logits,
        or (batch, positions, K) for top-k logits.
    indices : torch.Tensor | None
        Top-k token indices with shape (batch, positions, K) and dtype int32.
        None when full logits are stored.
    top_k : int | None
        K value if top-k logits, None for full logits.
    positions : str
        Which token positions are stored: "last" or "all".
    """

    values: torch.Tensor
    indices: torch.Tensor | None
    top_k: int | None
    positions: str


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
        Use nnsight remote execution (requires NDIF_API_KEY).
    batch_size : int, default=8
        Number of prompts to process at once.
    cache_pooled : bool, default=True
        If True (default), apply pooling BEFORE caching and store only
        the pooled activations. This reduces disk usage by ~100x (storing
        only (1, hidden_dim) per layer instead of (1, seq_len, hidden_dim)).

        Set to False only if you need to re-pool cached activations with
        different strategies.

        WARNING: When cache_pooled=True, you must use the same pooling
        strategy for all operations. The cached data cannot be re-pooled
        with a different strategy.
    pooling : str, default="last_token"
        Pooling strategy to use when cache_pooled=True. Options:
        - "last_token": Use the last non-padding token (default)
        - "first_token": Use the first token
        - "mean": Mean of all non-padding tokens
    backend : str, default="local"
        Extraction backend: "local" (default) or "nnsight".
    dtype : str or None, default=None
        Model dtype as a string: "float32", "float16", or "bfloat16".
        If None, defaults to float32 for local backend.

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
        cache_pooled: bool = True,
        pooling: str = "last_token",
        backend: str = "local",
        dtype: str | None = None,
        cache_logits: bool = False,
        logit_top_k: int | None = None,
        logit_positions: str = "last",
    ):
        self.model_name = model
        self.layers_spec = layers
        self.compute_perplexity = compute_perplexity
        self.device = device
        self.remote = remote
        self.batch_size = batch_size
        self.cache_pooled = cache_pooled
        self.pooling = pooling
        self.backend_name = backend
        self.dtype = dtype
        self.cache_logits = cache_logits
        self.logit_top_k = logit_top_k
        self.logit_positions = logit_positions

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

        if logit_positions not in ("last", "all"):
            raise ValueError(
                f"Invalid logit_positions: {logit_positions!r}. "
                f"Must be 'last' or 'all'."
            )

        # Create backend (lazy-loads model)
        self._backend: ExtractionBackend | None = None
        self._layer_indices: list[int] | None = None
        self._pooling_fn = get_pooling_fn(pooling) if cache_pooled else None

    @property
    def _resolved_backend(self) -> ExtractionBackend:
        """Get the backend, creating if necessary."""
        if self._backend is None:
            torch_dtype = self._resolve_dtype(self.dtype)
            self._backend = resolve_backend(
                self.backend_name, self.model_name, self.device,
                remote=self.remote, dtype=torch_dtype,
            )
        return self._backend

    @staticmethod
    def _resolve_dtype(dtype: str | None) -> torch.dtype | None:
        """Resolve a dtype string to a torch.dtype, or None."""
        if dtype is None:
            return None
        _dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in _dtype_map:
            raise ValueError(
                f"Unknown dtype: {dtype!r}. "
                f"Available: {list(_dtype_map.keys())}"
            )
        return _dtype_map[dtype]

    @property
    def model(self) -> Any:
        """Get the loaded model, loading if necessary."""
        return self._resolved_backend.model

    @property
    def layer_indices(self) -> list[int]:
        """Get resolved layer indices."""
        if self._layer_indices is None:
            num_layers = get_num_layers_from_config(self.model_name)
            self._layer_indices = resolve_layers(self.layers_spec, num_layers)
        return self._layer_indices

    def _check_cache_status(
        self, prompts: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """Check which prompts need extraction.

        Uses batch_check_cache_status which does a single LIST call
        instead of per-prompt HEAD requests. For large prompt sets
        (e.g. 3000 prompts), this reduces S3 latency from minutes
        to seconds.

        Returns
        -------
        tuple
            - prompts needing activation extraction
            - prompts needing perplexity extraction (if compute_perplexity=True)
            - prompts needing logit extraction (if cache_logits=True)
        """
        required_layers = set(self.layer_indices)

        (
            need_activations,
            need_perplexity,
            need_logits,
            partial_cache_count,
            partial_cache_found_layers,
        ) = batch_check_cache_status(
            model_name=self.model_name,
            prompts=prompts,
            required_layers=required_layers,
            pooling=self.pooling if self.cache_pooled else None,
            compute_perplexity=self.compute_perplexity,
            cache_logits=self.cache_logits,
            logit_top_k=self.logit_top_k,
        )

        if partial_cache_count > 0:
            missing = sorted(required_layers - (partial_cache_found_layers or set()))
            found = sorted(partial_cache_found_layers or set())
            import warnings

            warnings.warn(
                f"Cache exists for {partial_cache_count} prompt(s) but missing "
                f"layer(s) {missing} (found layers {found}). "
                f"A new forward pass is required. To avoid redundant forward passes, "
                f"use probe.warmup(prompts) to cache all layers at once before fitting.",
                stacklevel=2,
            )

        return need_activations, need_perplexity, need_logits

    def warmup(
        self,
        prompts: list[str],
        remote: bool | None = None,
        max_retries: int | None = None,
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
        max_retries : int | None
            Maximum number of retry attempts per batch for transient errors.
            Defaults to 3 for remote extraction, 0 for local.

        Returns
        -------
        WarmupStats
            Statistics about the warmup operation.
        """
        start_time = time.time()
        remote = self.remote if remote is None else remote

        if remote and self.backend_name == "nnsight":
            from .extraction import configure_remote

            configure_remote()

        layer_indices = sorted(self.layer_indices)

        # Check cache status
        need_activations, need_perplexity, need_logits = self._check_cache_status(
            prompts
        )

        # Compute which prompts need unified extraction
        # (prompts that need BOTH or where we can get both cheaply)
        need_activations_set = set(need_activations)
        need_perplexity_set = set(need_perplexity)
        need_logits_set = set(need_logits)
        need_unified = need_activations_set | need_perplexity_set | need_logits_set
        need_unified_list = [p for p in prompts if p in need_unified]

        activations_cached = len(prompts) - len(need_activations)
        perplexity_cached = len(prompts) - len(need_perplexity)
        logits_cached = len(prompts) - len(need_logits) if self.cache_logits else 0
        activations_extracted = 0
        perplexity_extracted = 0
        logits_extracted = 0

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
        if self.cache_logits:
            logger.info(
                f"[UNIFIED] Logits: {logits_cached} cached, "
                f"{len(need_logits)} need extraction"
            )

        failed_count = 0

        if need_unified_list:
            backend = self._resolved_backend
            num_batches = (len(need_unified_list) + self.batch_size - 1) // self.batch_size

            logger.info(
                f"[UNIFIED] Extracting {len(need_unified_list)} prompts in "
                f"{num_batches} batches"
            )

            # Determine if we can use server-side top-k for logits.
            # All three conditions must hold:
            # 1. remote=True
            # 2. logit_top_k is set
            # 3. compute_perplexity=False (full logits not needed)
            effective_top_k = (
                self.logit_top_k
                if (remote and self.logit_top_k is not None
                    and not self.compute_perplexity)
                else None
            )

            # Resolve retry count: only retry for remote extraction
            effective_retries = max_retries if max_retries is not None else (3 if remote else 0)
            if not remote:
                effective_retries = 0  # never retry local — would hide real errors

            if effective_retries > 0:
                from .retry import retry_with_backoff

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
                    batch_num = batch_idx // self.batch_size + 1

                    # Single forward pass captures both activations and logits
                    try:
                        if effective_retries > 0:
                            batch_acts, batch_mask, batch_logits, batch_logits_indices = (
                                retry_with_backoff(
                                    lambda bp=batch_prompts: backend.extract_batch_with_logits(  # type: ignore[misc]
                                        bp, layer_indices,
                                        remote=remote,
                                        logit_top_k=effective_top_k,
                                    ),
                                    max_retries=effective_retries,
                                    context=f"batch {batch_num}/{num_batches}",
                                )
                            )
                        else:
                            batch_acts, batch_mask, batch_logits, batch_logits_indices = (
                                backend.extract_batch_with_logits(
                                    batch_prompts, layer_indices,
                                    remote=remote,
                                    logit_top_k=effective_top_k,
                                )
                            )
                    except Exception:
                        if remote and effective_retries > 0:
                            # Skip this batch — partial progress is saved
                            failed_count += len(batch_prompts)
                            logger.error(
                                f"[UNIFIED] Skipping batch {batch_num}/{num_batches} "
                                f"({len(batch_prompts)} prompts) after {effective_retries} retries"
                            )
                            continue
                        raise

                    # Move batch tensors to CPU to free any GPU memory
                    batch_acts = batch_acts.cpu() if batch_acts is not None else None
                    batch_mask = batch_mask.cpu()
                    batch_logits = batch_logits.cpu() if batch_logits is not None else None
                    if batch_logits_indices is not None:
                        batch_logits_indices = batch_logits_indices.cpu()

                    # Compute perplexity features from logits
                    ppl_token_ppl_list = None
                    ppl_token_ids_list = None
                    ppl_features = None
                    tokenized = None
                    if self.compute_perplexity:
                        # Get input_ids for perplexity computation
                        tokenized = backend.tokenizer(
                            batch_prompts,
                            return_tensors="pt",
                            padding=True,
                        )
                        ppl_result = compute_perplexity_from_logits(
                            batch_logits,
                            tokenized["input_ids"],
                            batch_mask,
                            return_per_token=True,
                        )
                        ppl_features, ppl_token_ppl_list, ppl_token_ids_list = ppl_result
                        del ppl_result

                    # Split batch tensors into per-prompt copies so the
                    # large batch tensor can be freed before saving.
                    # Slicing creates views that keep the original alive;
                    # .clone() breaks that reference.
                    prompt_acts_list: list[torch.Tensor | None] = [
                        batch_acts[j : j + 1].clone()
                        for j in range(len(batch_prompts))
                    ]
                    prompt_mask_list: list[torch.Tensor | None] = [
                        batch_mask[j : j + 1].clone()
                        for j in range(len(batch_prompts))
                    ]
                    prompt_logits_list: list[torch.Tensor | None] | None = (
                        [batch_logits[j : j + 1].clone()
                         for j in range(len(batch_prompts))]
                        if batch_logits is not None else None
                    )
                    prompt_logits_idx_list: list[torch.Tensor | None] | None = (
                        [batch_logits_indices[j : j + 1].clone()
                         for j in range(len(batch_prompts))]
                        if batch_logits_indices is not None else None
                    )

                    # Free batch tensors now — per-prompt copies are independent
                    del batch_acts, batch_mask, batch_logits, batch_logits_indices

                    # Save each prompt's data
                    for j, prompt in enumerate(batch_prompts):
                        prompt_acts = prompt_acts_list[j]
                        prompt_mask = prompt_mask_list[j]
                        assert prompt_acts is not None
                        assert prompt_mask is not None

                        # Save activations if needed
                        if prompt in need_activations_set:
                            if self.cache_pooled:
                                assert self._pooling_fn is not None
                                pooled_acts = self._pooling_fn(prompt_acts, prompt_mask)
                                save_prompt_pooled_activations(
                                    self.model_name,
                                    prompt,
                                    layer_indices,
                                    pooled_acts,
                                    self.pooling,
                                )
                            else:
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
                            tok_ppl = (
                                ppl_token_ppl_list[j]
                                if ppl_token_ppl_list is not None else None
                            )
                            tok_ids = (
                                ppl_token_ids_list[j]
                                if ppl_token_ids_list is not None else None
                            )
                            assert ppl_features is not None
                            save_prompt_perplexity(
                                self.model_name, prompt, ppl_features[j],
                                token_perplexity=tok_ppl,
                                token_ids=tok_ids,
                            )
                            perplexity_extracted += 1

                        # Save logits if needed
                        if self.cache_logits and prompt in need_logits_set:
                            p_logits = (
                                prompt_logits_list[j]
                                if prompt_logits_list is not None else None
                            )
                            p_logits_idx = (
                                prompt_logits_idx_list[j]
                                if prompt_logits_idx_list is not None else None
                            )
                            assert p_logits is not None
                            if p_logits_idx is not None:
                                save_prompt_topk_logits(
                                    self.model_name,
                                    prompt,
                                    p_logits,
                                    p_logits_idx,
                                    prompt_mask,
                                    self.logit_positions,
                                )
                            else:
                                save_prompt_logits(
                                    self.model_name,
                                    prompt,
                                    p_logits,
                                    prompt_mask,
                                    self.logit_top_k,
                                    self.logit_positions,
                                )
                            logits_extracted += 1

                        # Release per-prompt data as we go
                        prompt_acts_list[j] = None
                        prompt_mask_list[j] = None
                        if prompt_logits_list is not None:
                            prompt_logits_list[j] = None
                        if prompt_logits_idx_list is not None:
                            prompt_logits_idx_list[j] = None

                    # Free remaining references and return memory to OS
                    del prompt_acts_list, prompt_mask_list
                    del prompt_logits_list, prompt_logits_idx_list
                    del ppl_features, ppl_token_ppl_list, ppl_token_ids_list
                    del tokenized
                    gc.collect()
                    _release_memory()

        if need_unified_list and failed_count > 0:
            logger.warning(
                f"[UNIFIED] Remote extraction incomplete: {failed_count} prompts failed "
                f"after retries. Cached results were saved for successful batches. "
                f"Re-run to retry the remaining prompts (cached results will be reused)."
            )

        elapsed = time.time() - start_time

        stats = WarmupStats(
            total_prompts=len(prompts),
            activations_cached=activations_cached,
            activations_extracted=activations_extracted,
            perplexity_cached=perplexity_cached,
            perplexity_extracted=perplexity_extracted,
            logits_cached=logits_cached,
            logits_extracted=logits_extracted,
            elapsed_seconds=elapsed,
        )

        logger.info(
            f"[UNIFIED] Complete: {activations_extracted} activations + "
            f"{perplexity_extracted} perplexity + "
            f"{logits_extracted} logits extracted in {elapsed:.1f}s"
        )

        evict()
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
            return load_pooled_batch(
                self.model_name, prompts, layer_indices, self.pooling
            ), None

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
            features.append(ppl.float().numpy())

        return np.array(features)

    def get_logits(
        self,
        prompts: list[str],
        remote: bool | None = None,
    ) -> CachedLogits:
        """Get cached logits for prompts (from cache or via extraction).

        Parameters
        ----------
        prompts : list[str]
            Text prompts.
        remote : bool | None
            Override the instance-level remote setting.

        Returns
        -------
        CachedLogits
            Cached logits with values and optional indices tensors.
        """
        if not self.cache_logits:
            raise ValueError(
                "UnifiedCache was created with cache_logits=False. "
                "Create a new instance with cache_logits=True."
            )

        # Ensure cache is warm
        self.warmup(prompts, remote=remote)

        # Load from cache
        all_values = []
        all_indices = []
        for prompt in prompts:
            values, indices = load_prompt_logits(
                self.model_name, prompt, self.logit_top_k
            )
            all_values.append(values)
            if indices is not None:
                all_indices.append(indices)

        return CachedLogits(
            values=torch.cat(all_values, dim=0),
            indices=torch.cat(all_indices, dim=0) if all_indices else None,
            top_k=self.logit_top_k,
            positions=self.logit_positions,
        )

    def compute_logits_from_cache(
        self,
        prompts: list[str],
        positions: str = "last",
        top_k: int | None = None,
        device: str | None = None,
        batch_size: int = 64,
    ) -> int:
        """Compute logits from cached last-layer activations without a forward pass.

        Downloads only the model's final norm and lm_head weights, then
        computes ``logits = norm(hidden_state) @ lm_head.T`` from the
        cached last-layer activations.

        Parameters
        ----------
        prompts : list[str]
            Prompts whose last-layer activations are already cached.
        positions : str
            Which positions to compute logits for. Currently only "last"
            is supported.
        top_k : int | None
            If set, store only top-k logit values and indices.
        device : str | None
            Device for computation. Defaults to "cpu".
        batch_size : int
            Number of prompts to process at once for the matmul.

        Returns
        -------
        int
            Number of prompts for which logits were newly computed.

        Raises
        ------
        ValueError
            If the last layer is not in ``self.layer_indices`` or if
            ``positions`` is not "last".
        """
        if positions != "last":
            raise ValueError(
                f"Only positions='last' is currently supported, got {positions!r}"
            )

        device = device or "cpu"

        # Determine the last layer index
        num_layers = get_num_layers_from_config(self.model_name)
        last_layer = num_layers - 1

        if last_layer not in self.layer_indices:
            raise ValueError(
                f"Last layer ({last_layer}) is not in cached layer_indices "
                f"{self.layer_indices}. Cache must include the last layer to "
                f"compute logits."
            )

        # Filter to prompts that don't already have logits cached
        uncached = []
        for prompt in prompts:
            if not is_prompt_logits_cached(self.model_name, prompt, top_k):
                uncached.append(prompt)

        if not uncached:
            logger.info("[LOGITS] All %d prompts already have logits cached", len(prompts))
            return 0

        logger.info(
            "[LOGITS] Computing logits for %d prompts (%d already cached)",
            len(uncached), len(prompts) - len(uncached),
        )

        # Download norm + lm_head weights
        from .logit_utils import apply_norm, download_lm_head_weights

        norm_weight, lm_head_weight, norm_config = download_lm_head_weights(
            self.model_name, device=device,
        )

        # Process in batches
        computed = 0
        for batch_start in range(0, len(uncached), batch_size):
            batch_prompts = uncached[batch_start : batch_start + batch_size]

            # Load last-layer, last-token activations
            from .cache import load_layer_last_token

            hidden_states = load_layer_last_token(
                self.model_name, batch_prompts, last_layer,
            )  # (B, hidden_dim)

            hidden_states = hidden_states.to(device=device, dtype=norm_weight.dtype)

            # Apply norm
            normed = apply_norm(
                hidden_states,
                norm_weight,
                norm_config["eps"],
                norm_config["norm_type"],
                norm_config["norm_bias"],
            )

            # Compute logits: (B, hidden_dim) @ (hidden_dim, vocab_size) -> (B, vocab_size)
            logits = normed @ lm_head_weight.T  # (B, vocab_size)

            # Save each prompt's logits
            for j, prompt in enumerate(batch_prompts):
                # Shape for save: (1, 1, vocab_size) — 1 batch, 1 position
                prompt_logits = logits[j : j + 1].unsqueeze(1)  # (1, 1, vocab_size)

                # Create a dummy mask for save_prompt_logits (last position)
                dummy_mask = torch.ones(1, 1, dtype=torch.long)

                if top_k is not None:
                    values, indices = torch.topk(prompt_logits, top_k, dim=-1)
                    save_prompt_topk_logits(
                        self.model_name, prompt,
                        values, indices, dummy_mask,
                        positions="all",  # already selected "last"
                    )
                else:
                    save_prompt_logits(
                        self.model_name, prompt,
                        prompt_logits, dummy_mask,
                        top_k=None,
                        positions="all",  # already selected "last"
                    )

            computed += len(batch_prompts)

        logger.info("[LOGITS] Computed logits for %d prompts", computed)
        evict()
        return computed

    def compute_perplexity_from_cache(
        self,
        prompts: list[str],
        device: str | None = None,
        batch_size: int = 16,
    ) -> int:
        """Compute perplexity from cached last-layer activations.

        Reconstructs full-sequence logits as ``norm(hidden) @ lm_head.T``,
        computes cross-entropy loss against input_ids (re-tokenized),
        and caches the perplexity stats (mean, min, max).

        Full-sequence logits are NOT saved to disk (too large).

        Parameters
        ----------
        prompts : list[str]
            Prompts whose last-layer raw activations are already cached.
        device : str | None
            Device for computation. Defaults to "cpu".
        batch_size : int
            Number of prompts to process per batch.

        Returns
        -------
        int
            Number of prompts for which perplexity was newly computed.

        Raises
        ------
        ValueError
            If the last layer is not in ``self.layer_indices``.
        """
        device = device or "cpu"

        # Determine the last layer index
        num_layers = get_num_layers_from_config(self.model_name)
        last_layer = num_layers - 1

        if last_layer not in self.layer_indices:
            raise ValueError(
                f"Last layer ({last_layer}) is not in cached layer_indices "
                f"{self.layer_indices}. Cache must include the last layer to "
                f"compute perplexity."
            )

        # Filter to prompts that don't already have perplexity cached
        uncached = [
            p for p in prompts
            if not is_prompt_perplexity_cached(self.model_name, p)
        ]

        if not uncached:
            logger.info(
                "[PERPLEXITY] All %d prompts already have perplexity cached",
                len(prompts),
            )
            return 0

        logger.info(
            "[PERPLEXITY] Computing perplexity for %d prompts "
            "(%d already cached)",
            len(uncached), len(prompts) - len(uncached),
        )

        from transformers import AutoTokenizer

        from .logit_utils import (
            compute_perplexity_from_activations,
            download_lm_head_weights,
        )

        # Load weights and tokenizer once, reuse across all batches
        norm_weight, lm_head_weight, norm_config = download_lm_head_weights(
            self.model_name, device=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        computed = 0
        for batch_start in range(0, len(uncached), batch_size):
            batch_prompts = uncached[batch_start : batch_start + batch_size]

            with torch.no_grad():
                result = compute_perplexity_from_activations(
                    self.model_name, batch_prompts, last_layer, device=device,
                    norm_weight=norm_weight, lm_head_weight=lm_head_weight,
                    norm_config=norm_config, tokenizer=tokenizer,
                    return_per_token=True,
                )
                ppl_tensors, token_ppl_list, token_ids_list = result

            for i, (prompt, ppl_tensor) in enumerate(zip(batch_prompts, ppl_tensors)):
                save_prompt_perplexity(
                    self.model_name, prompt, ppl_tensor,
                    token_perplexity=token_ppl_list[i],
                    token_ids=token_ids_list[i],
                )
                computed += 1

        logger.info("[PERPLEXITY] Computed perplexity for %d prompts", computed)
        evict()
        return computed
