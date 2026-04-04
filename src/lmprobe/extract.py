"""Bulk extraction of activations from language models.

This module provides ``extract()`` for saving raw NDIF/local batch responses
directly to disk as per-layer-keyed safetensors, bypassing the promptwise
cache entirely.  The output directory can then be passed to
``push_dataset(source=...)`` for fast layerwise shard construction.

Also provides ``consolidate_cache()`` for converting existing promptwise
cache entries into the same batch format.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .backends import resolve_backend
from .cache import (
    _hash_string,
    _load_tensors_from_backend,
    _save_tensors_to_backend,
    get_backend,
    load_layer_across_prompts,
)
from .extraction import get_num_layers_from_config, resolve_layers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local vs backend I/O detection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


@dataclass
class BatchInfo:
    """Metadata for a single batch file."""

    file: str
    prompt_start: int
    prompt_end: int
    num_tokens: list[int]
    status: str = "complete"


@dataclass
class ExtractionManifest:
    """Self-contained manifest for a raw extraction directory."""

    version: int = MANIFEST_VERSION
    model_name: str = ""
    layers: list[int] = field(default_factory=list)
    hidden_dim: int = 0
    total_prompts: int = 0
    batch_size: int = 32
    prompts: list[str] = field(default_factory=list)
    labels: list[int | str | None] | None = None
    metadata: list[dict] | None = None
    batches: list[BatchInfo] = field(default_factory=list)
    created_at: str = ""
    dtype: str = "float32"
    completed_layers: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "model_name": self.model_name,
            "layers": self.layers,
            "hidden_dim": self.hidden_dim,
            "total_prompts": self.total_prompts,
            "batch_size": self.batch_size,
            "dtype": self.dtype,
            "prompts": self.prompts,
            "batches": [
                {
                    "file": b.file,
                    "prompt_start": b.prompt_start,
                    "prompt_end": b.prompt_end,
                    "num_tokens": b.num_tokens,
                    "status": b.status,
                }
                for b in self.batches
            ],
            "created_at": self.created_at,
        }
        if self.labels is not None:
            d["labels"] = self.labels
        if self.metadata is not None:
            d["metadata"] = self.metadata
        if self.completed_layers is not None:
            d["completed_layers"] = self.completed_layers
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtractionManifest:
        batches = [
            BatchInfo(
                file=b["file"],
                prompt_start=b["prompt_start"],
                prompt_end=b["prompt_end"],
                num_tokens=b["num_tokens"],
                status=b.get("status", "complete"),
            )
            for b in d.get("batches", [])
        ]
        return cls(
            version=d.get("version", MANIFEST_VERSION),
            model_name=d["model_name"],
            layers=d["layers"],
            hidden_dim=d["hidden_dim"],
            total_prompts=d["total_prompts"],
            batch_size=d.get("batch_size", 32),
            prompts=d.get("prompts", []),
            labels=d.get("labels"),
            metadata=d.get("metadata"),
            batches=batches,
            created_at=d.get("created_at", ""),
            dtype=d.get("dtype", "float32"),
            completed_layers=d.get("completed_layers"),
        )


def _write_manifest(
    manifest: ExtractionManifest,
    prefix: str,
    *,
    local_root: Path | None = None,
) -> None:
    """Write manifest JSON.

    Parameters
    ----------
    manifest : ExtractionManifest
        Manifest to write.
    prefix : str
        Key prefix (backend) or subdirectory name (local).
    local_root : Path | None
        If provided, write to ``local_root / manifest.json`` on the local
        filesystem.  Otherwise, write via the cache backend.
    """
    text = json.dumps(manifest.to_dict(), ensure_ascii=False)
    if local_root is not None:
        local_root.mkdir(parents=True, exist_ok=True)
        (local_root / MANIFEST_FILENAME).write_text(text, encoding="utf-8")
    else:
        key = f"{prefix}/{MANIFEST_FILENAME}"
        backend = get_backend()
        backend.write_text(key, text)


def load_manifest(source: str) -> ExtractionManifest:
    """Load an extraction manifest.

    Auto-detects whether *source* is a local filesystem path or a cache
    backend key prefix.  If ``source/manifest.json`` exists on the local
    filesystem, it is read directly; otherwise the cache backend is used.

    Parameters
    ----------
    source : str
        Local directory path or key prefix within the cache backend.

    Returns
    -------
    ExtractionManifest
        The loaded manifest.

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    """
    # Try local filesystem first
    local_path = Path(source) / MANIFEST_FILENAME
    if local_path.exists():
        return ExtractionManifest.from_dict(
            json.loads(local_path.read_text(encoding="utf-8"))
        )

    # Fall back to cache backend
    key = f"{source}/{MANIFEST_FILENAME}"
    backend = get_backend()
    if not backend.exists(key):
        raise FileNotFoundError(
            f"No manifest found at {source} (checked local filesystem "
            f"and cache backend). Is this an extraction prefix?"
        )
    return ExtractionManifest.from_dict(json.loads(backend.read_text(key)))


# ---------------------------------------------------------------------------
# Batch file I/O  (N_batches × N_layers layout)
#
# Each NDIF batch produces one safetensors file **per layer**, stored under
# ``layer_{idx}/batch_{idx}.safetensors``.  Each file contains just two
# tensors: ``activations: (batch, seq, dim)`` and ``mask: (batch, seq)``.
#
# This layout means every file is read exactly once during shard
# construction — no selective loading or range reads needed.
# ---------------------------------------------------------------------------

def _batch_filename(batch_idx: int) -> str:
    return f"batch_{batch_idx:06d}.safetensors"


def _layer_dir(layer: int) -> str:
    return f"layer_{layer:03d}"


def _layer_batch_path(layer: int, batch_idx: int) -> str:
    """Relative path for a layer/batch file within the extraction dir."""
    return f"{_layer_dir(layer)}/{_batch_filename(batch_idx)}"


def _save_batch(
    batch_acts: torch.Tensor,
    batch_mask: torch.Tensor,
    layer_indices: list[int],
    hidden_dim: int,
    prefix: str,
    batch_idx: int,
    batch_logits: torch.Tensor | None = None,
    batch_logits_indices: torch.Tensor | None = None,
    *,
    local_root: Path | None = None,
    num_workers: int = 1,
) -> None:
    """Save a batch as one safetensors file per layer.

    The NDIF response shape ``(batch, seq, hidden_dim * num_layers)`` is
    split and each layer is written to its own file under
    ``layer_{idx}/batch_{idx}.safetensors``.

    Parameters
    ----------
    local_root : Path | None
        If provided, write files to the local filesystem under this
        directory.  Otherwise, write via the cache backend using *prefix*
        as key prefix.
    num_workers : int
        Number of parallel threads for writing layer files.  Values > 1
        are useful when writing to S3 or another high-latency backend.
    """
    from concurrent.futures import ThreadPoolExecutor

    from safetensors.torch import save_file

    batch_size, seq_len = batch_acts.shape[:2]
    num_layers = len(layer_indices)

    # Reshape: (batch, seq, dim*layers) → (batch, seq, layers, dim)
    per_layer = batch_acts.view(batch_size, seq_len, num_layers, hidden_dim)

    def _write_layer(i: int, layer_idx: int) -> None:
        rel_path = _layer_batch_path(layer_idx, batch_idx)
        tensors: dict[str, torch.Tensor] = {
            "activations": per_layer[:, :, i, :].contiguous(),
            "mask": batch_mask,
        }
        if local_root is not None:
            out = local_root / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(out))
        else:
            key = f"{prefix}/{rel_path}"
            _save_tensors_to_backend(key, tensors)

    if num_workers > 1 and num_layers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futs = [
                pool.submit(_write_layer, i, layer_idx)
                for i, layer_idx in enumerate(layer_indices)
            ]
            for fut in futs:
                fut.result()  # propagate exceptions
    else:
        for i, layer_idx in enumerate(layer_indices):
            _write_layer(i, layer_idx)

    # Save logits separately (not per-layer — they're layer-independent)
    if batch_logits is not None:
        logits_rel = f"logits/{_batch_filename(batch_idx)}"
        logits_tensors: dict[str, torch.Tensor] = {"logits": batch_logits}
        if batch_logits_indices is not None:
            logits_tensors["logits_indices"] = batch_logits_indices
        if local_root is not None:
            out = local_root / logits_rel
            out.parent.mkdir(parents=True, exist_ok=True)
            save_file(logits_tensors, str(out))
        else:
            logits_key = f"{prefix}/logits/{_batch_filename(batch_idx)}"
            _save_tensors_to_backend(logits_key, logits_tensors)


def load_batch_layer(
    source: str,
    layer: int,
    batch_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load activations for one layer from one batch file.

    Auto-detects whether *source* is a local filesystem path or a cache
    backend key prefix (same logic as :func:`load_manifest`).

    Parameters
    ----------
    source : str
        Local directory path or extraction prefix within the cache backend.
    layer : int
        Layer index.
    batch_idx : int
        Batch index.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(activations, mask)`` where activations has shape
        ``(batch_size, seq_len, hidden_dim)`` and mask has shape
        ``(batch_size, seq_len)``.
    """
    rel_path = _layer_batch_path(layer, batch_idx)
    local_file = Path(source) / rel_path

    if local_file.exists():
        from safetensors import safe_open

        result: dict[str, torch.Tensor] = {}
        with safe_open(str(local_file), framework="pt") as f:
            result["activations"] = f.get_tensor("activations")
            result["mask"] = f.get_tensor("mask")
        return result["activations"], result["mask"]

    # Fall back to cache backend
    key = f"{source}/{rel_path}"
    tensors = _load_tensors_from_backend(key, ["activations", "mask"])
    return tensors["activations"], tensors["mask"]


# ---------------------------------------------------------------------------
# Release memory helper (same pattern as unified_cache)
# ---------------------------------------------------------------------------

def _release_memory() -> None:
    """Release freed memory back to the OS."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Hidden size from config
# ---------------------------------------------------------------------------

def _get_hidden_size(model_name: str) -> int:
    """Get hidden_size from model config without loading weights."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    for attr in ("hidden_size", "d_model", "n_embd"):
        if hasattr(config, attr):
            return int(getattr(config, attr))
    raise ValueError(
        f"Could not determine hidden_size from config for {model_name}. "
        f"Config attributes: {list(config.to_dict().keys())}"
    )


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------

@dataclass
class ExtractionStats:
    """Statistics from an extract() operation."""

    total_prompts: int
    batches_extracted: int
    batches_skipped: int
    elapsed_seconds: float


def extract(
    model_name: str,
    prompts: list[str],
    layers: int | list[int] | str = "all",
    *,
    labels: list[int | str | None] | None = None,
    metadata: list[dict] | None = None,
    output_dir: str | None = None,
    batch_size: int = 32,
    remote: bool = True,
    device: str = "auto",
    backend: str = "nnsight",
    dtype: str | None = None,
    compute_perplexity: bool = False,
    cache_logits: bool = False,
    logit_top_k: int | None = None,
    max_retries: int = 3,
) -> str:
    """Extract activations and save raw batch responses to disk.

    Saves NDIF batch responses as per-layer-keyed safetensors files,
    bypassing the promptwise cache entirely.  The output directory can
    then be passed to ``push_dataset(source=...)`` for fast layerwise
    shard construction.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    prompts : list[str]
        Text prompts to extract activations for.
    layers : int | list[int] | str
        Which layers to extract (same spec as ``UnifiedCache``).
    labels : list[int | str | None] | None
        Per-prompt labels, stored in the manifest.
    metadata : list[dict] | None
        Per-prompt metadata dicts, stored in the manifest.
    output_dir : str | None
        Directory to write batch files and manifest.  If None, creates
        a timestamped directory under the cache backend root.
    batch_size : int
        Number of prompts per extraction batch.
    remote : bool
        Use nnsight remote execution (requires NDIF_API_KEY).
    device : str
        Device for model inference.
    backend : str
        Extraction backend: ``"nnsight"`` (default) or ``"local"``.
    dtype : str | None
        Model dtype: ``"float32"``, ``"float16"``, or ``"bfloat16"``.
    compute_perplexity : bool
        Whether to also capture logits for perplexity.
    cache_logits : bool
        Whether to save logits in batch files.
    logit_top_k : int | None
        Server-side top-k for logits (remote only).
    max_retries : int
        Retry attempts per batch for transient errors (remote only).

    Returns
    -------
    Path
        Path to the output directory containing batch files and manifest.
    """
    start_time = time.time()

    # Resolve layers
    num_model_layers = get_num_layers_from_config(model_name)
    layer_indices = sorted(resolve_layers(layers, num_model_layers))
    hidden_dim = _get_hidden_size(model_name)

    # Compute output prefix (key prefix within cache backend)
    if output_dir is None:
        model_hash = _hash_string(model_name)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = f"_extractions/{model_hash}_{timestamp}"
    else:
        prefix = str(output_dir)

    # Load existing manifest for resumability
    manifest_key = f"{prefix}/{MANIFEST_FILENAME}"
    completed_batches: set[int] = set()
    if get_backend().exists(manifest_key):
        existing = ExtractionManifest.from_dict(
            json.loads(get_backend().read_text(manifest_key))
        )
        for b in existing.batches:
            if b.status == "complete":
                batch_idx = b.prompt_start // batch_size
                completed_batches.add(batch_idx)
        logger.info(
            f"[EXTRACT] Resuming: {len(completed_batches)} batches already complete"
        )

    # Build manifest
    manifest = ExtractionManifest(
        model_name=model_name,
        layers=layer_indices,
        hidden_dim=hidden_dim,
        total_prompts=len(prompts),
        batch_size=batch_size,
        prompts=list(prompts),
        labels=list(labels) if labels is not None else None,
        metadata=list(metadata) if metadata is not None else None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # Configure remote
    if remote:
        from .extraction import configure_remote
        configure_remote()

    # Resolve dtype
    torch_dtype = None
    if dtype is not None:
        _dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = _dtype_map.get(dtype)

    # Create extraction backend
    extraction_backend = resolve_backend(
        backend, model_name, device, remote=remote, dtype=torch_dtype
    )

    # Determine if we can use server-side top-k
    effective_top_k = (
        logit_top_k
        if (remote and logit_top_k is not None and not compute_perplexity)
        else None
    )

    # Retry setup
    effective_retries = max_retries if remote else 0
    retry_fn = None
    if effective_retries > 0:
        from .retry import retry_with_backoff
        retry_fn = retry_with_backoff

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    batches_extracted = 0
    batches_skipped = 0

    logger.info(
        f"[EXTRACT] Extracting {len(prompts)} prompts in {num_batches} batches, "
        f"layers: {layer_indices}"
    )

    with torch.no_grad():
        for batch_idx in tqdm(
            range(num_batches), total=num_batches,
            desc="Extracting", unit="batch",
        ):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]

            # Skip completed batches (resumability)
            if batch_idx in completed_batches:
                batches_skipped += 1
                continue

            # Extract
            try:
                if retry_fn is not None:
                    batch_acts, batch_mask, batch_logits, batch_logits_indices = (
                        retry_fn(
                            lambda bp=batch_prompts: extraction_backend.extract_batch_with_logits(  # type: ignore[misc]
                                bp, layer_indices,
                                remote=remote,
                                logit_top_k=effective_top_k,
                            ),
                            max_retries=effective_retries,
                            context=f"batch {batch_idx + 1}/{num_batches}",
                        )
                    )
                else:
                    batch_acts, batch_mask, batch_logits, batch_logits_indices = (
                        extraction_backend.extract_batch_with_logits(
                            batch_prompts, layer_indices,
                            remote=remote,
                            logit_top_k=effective_top_k,
                        )
                    )
            except Exception:
                if remote and effective_retries > 0:
                    logger.error(
                        f"[EXTRACT] Skipping batch {batch_idx + 1}/{num_batches} "
                        f"after {effective_retries} retries"
                    )
                    continue
                raise

            # Move to CPU
            batch_acts = batch_acts.cpu() if batch_acts is not None else None
            batch_mask = batch_mask.cpu()
            if batch_logits is not None:
                batch_logits = batch_logits.cpu()
            if batch_logits_indices is not None:
                batch_logits_indices = batch_logits_indices.cpu()

            # Compute num_tokens from attention_mask
            num_tokens = batch_mask.sum(dim=-1).int().tolist()

            # Detect dtype from first batch
            assert batch_acts is not None
            if batches_extracted == 0 and batches_skipped == 0:
                _dtype_map_rev = {
                    torch.float32: "float32",
                    torch.float16: "float16",
                    torch.bfloat16: "bfloat16",
                }
                manifest.dtype = _dtype_map_rev.get(batch_acts.dtype, "float32")

            # Save per-layer batch files
            _save_batch(
                batch_acts=batch_acts,
                batch_mask=batch_mask,
                layer_indices=layer_indices,
                hidden_dim=hidden_dim,
                prefix=prefix,
                batch_idx=batch_idx,
                batch_logits=batch_logits if cache_logits else None,
                batch_logits_indices=batch_logits_indices if cache_logits else None,
            )

            # Update manifest
            manifest.batches.append(BatchInfo(
                file=_batch_filename(batch_idx),
                prompt_start=start,
                prompt_end=end,
                num_tokens=num_tokens,
                status="complete",
            ))

            # Write manifest after each batch for resumability
            _write_manifest(manifest, prefix)

            batches_extracted += 1

            # Free memory
            del batch_acts, batch_mask, batch_logits, batch_logits_indices
            gc.collect()
            _release_memory()

    elapsed = time.time() - start_time

    logger.info(
        f"[EXTRACT] Complete: {batches_extracted} batches extracted, "
        f"{batches_skipped} skipped (cached), {elapsed:.1f}s"
    )

    return prefix


# ---------------------------------------------------------------------------
# consolidate_cache()
# ---------------------------------------------------------------------------


def consolidate_cache(
    model_name: str,
    prompts: list[str],
    layers: int | list[int] | str = "all",
    *,
    labels: list[int | str | None] | None = None,
    metadata: list[dict] | None = None,
    output_dir: str | None = None,
    output_uri: str | None = None,
    batch_size: int = 32,
    num_workers: int = 8,
) -> str:
    """Convert promptwise cache entries into batch-format files.

    Reads per-prompt cache files from the cache backend (S3 or local) and
    writes batch-format safetensors to either the local filesystem
    (*output_dir*) or the cache backend (*output_uri*).

    Iterates layer-at-a-time so memory is bounded to one layer per
    batch, enabling large batch sizes on memory-constrained instances.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (must match cached activations).
    prompts : list[str]
        Prompts with cached activations.
    layers : int | list[int] | str
        Layer specification.
    labels : list[int | str | None] | None
        Per-prompt labels, stored in the manifest.
    metadata : list[dict] | None
        Per-prompt metadata dicts, stored in the manifest.
    output_dir : str | None
        Local filesystem directory for output.  Mutually exclusive with
        *output_uri*.
    output_uri : str | None
        Cache-backend key prefix for output (e.g. a path relative to the
        S3 cache root).  Batch files and manifest are written via the
        active cache backend.  Mutually exclusive with *output_dir*.
    batch_size : int
        Number of prompts per batch file.
    num_workers : int
        Number of parallel threads for reading and writing cache files.
        Higher values improve throughput for high-latency backends (S3).

    Returns
    -------
    str
        Local path (when *output_dir* is used) or backend key prefix
        (when *output_uri* is used).

    Raises
    ------
    FileNotFoundError
        If a prompt has no cached activations.
    ValueError
        If both *output_dir* and *output_uri* are specified.
    """
    if output_dir is not None and output_uri is not None:
        raise ValueError(
            "Specify either output_dir (local) or output_uri (backend), not both."
        )

    start_time = time.time()

    # Resolve layers
    num_model_layers = get_num_layers_from_config(model_name)
    layer_indices = sorted(resolve_layers(layers, num_model_layers))
    hidden_dim = _get_hidden_size(model_name)

    # Determine write mode: local filesystem or cache backend
    local_root: Path | None = None
    if output_uri is not None:
        # Write via cache backend (e.g. S3)
        prefix = output_uri
    elif output_dir is not None:
        local_root = Path(output_dir)
        local_root.mkdir(parents=True, exist_ok=True)
        prefix = str(local_root)
    else:
        # Default: local timestamped directory
        model_hash = _hash_string(model_name)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        local_root = Path(f"_extractions/{model_hash}_{timestamp}")
        local_root.mkdir(parents=True, exist_ok=True)
        prefix = str(local_root)

    # Load existing manifest for resumability
    completed_batches: set[int] = set()
    existing_batch_infos: dict[int, BatchInfo] = {}
    previously_completed_layers: set[int] = set()
    try:
        existing = load_manifest(prefix)
        for b in existing.batches:
            if b.status == "complete":
                bi = b.prompt_start // batch_size
                completed_batches.add(bi)
                existing_batch_infos[bi] = b
        if existing.completed_layers:
            previously_completed_layers = set(existing.completed_layers)
        resume_parts = []
        if completed_batches:
            resume_parts.append(f"{len(completed_batches)} batches")
        if previously_completed_layers:
            resume_parts.append(
                f"{len(previously_completed_layers)} layers"
            )
        if resume_parts:
            logger.info(
                f"[CONSOLIDATE] Resuming: {', '.join(resume_parts)} "
                f"already complete"
            )
    except FileNotFoundError:
        pass

    manifest = ExtractionManifest(
        model_name=model_name,
        layers=layer_indices,
        hidden_dim=hidden_dim,
        total_prompts=len(prompts),
        batch_size=batch_size,
        prompts=list(prompts),
        labels=list(labels) if labels is not None else None,
        metadata=list(metadata) if metadata is not None else None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    batches_skipped = 0

    logger.info(
        f"[CONSOLIDATE] Converting {len(prompts)} cached prompts into "
        f"{num_batches} batch files ({len(layer_indices)} layers, "
        f"layer-at-a-time)"
    )

    from concurrent.futures import ThreadPoolExecutor

    from safetensors.torch import save_file

    def _write_layer_batch(
        layer_idx: int,
        batch_idx: int,
        batch_acts: torch.Tensor,
        batch_mask: torch.Tensor,
    ) -> None:
        """Write a single layer/batch file."""
        rel_path = _layer_batch_path(layer_idx, batch_idx)
        tensors: dict[str, torch.Tensor] = {
            "activations": batch_acts,
            "mask": batch_mask,
        }
        if local_root is not None:
            out = local_root / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(out))
        else:
            key = f"{prefix}/{rel_path}"
            _save_tensors_to_backend(key, tensors)

    def _layer_batch_exists(layer_idx: int, batch_idx: int) -> bool:
        """Check if a layer/batch file already exists (for resumability)."""
        rel_path = _layer_batch_path(layer_idx, batch_idx)
        if local_root is not None:
            return (local_root / rel_path).exists()
        key = f"{prefix}/{rel_path}"
        backend = get_backend()
        return backend.exists(key)

    def _load_layer_batch(
        layer_idx: int,
        batch_prompts: list[str],
        pool: ThreadPoolExecutor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a single layer for a batch of prompts, pad, and stack."""

        def _load_one(
            prompt: str,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            acts_list, mask_list = load_layer_across_prompts(
                model_name, [prompt], layer_idx
            )
            return acts_list[0], mask_list[0]

        results = list(pool.map(_load_one, batch_prompts))
        acts_list = [r[0] for r in results]
        mask_list = [r[1] for r in results]

        # Pad to uniform sequence length
        max_seq = max(a.shape[1] for a in acts_list)
        padded_acts = []
        padded_masks = []
        for acts, mask in zip(acts_list, mask_list):
            seq_len = acts.shape[1]
            if seq_len < max_seq:
                pad_size = max_seq - seq_len
                acts = torch.nn.functional.pad(acts, (0, 0, 0, pad_size))
                mask = torch.nn.functional.pad(mask, (0, pad_size))
            padded_acts.append(acts)
            padded_masks.append(mask)

        batch_acts = torch.cat(padded_acts, dim=0)
        batch_mask = torch.cat(padded_masks, dim=0)
        return batch_acts, batch_mask

    # Pre-compute batch boundaries
    batch_ranges: list[tuple[int, int, int]] = []  # (batch_idx, start, end)
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(prompts))
        batch_ranges.append((batch_idx, start, end))

    # Collect num_tokens per batch (from masks during first layer pass)
    batch_num_tokens: dict[int, list[int]] = {}
    # Track which layers have been fully written (for manifest-based resume)
    completed_layers: set[int] = set(previously_completed_layers)

    read_pool = ThreadPoolExecutor(max_workers=num_workers)

    try:
        for layer_idx in tqdm(
            layer_indices, desc="Consolidating layers", unit="layer",
        ):
            # Skip entire layer if already completed in a previous run
            if layer_idx in previously_completed_layers:
                continue

            layer_had_work = False

            for batch_idx, start, end in batch_ranges:
                # Skip fully completed batches
                if batch_idx in completed_batches:
                    continue

                # Skip individual layer/batch files that already exist
                if _layer_batch_exists(layer_idx, batch_idx):
                    continue

                layer_had_work = True
                batch_prompts = prompts[start:end]
                batch_acts, batch_mask = _load_layer_batch(
                    layer_idx, batch_prompts, read_pool
                )

                # Capture num_tokens and dtype from the first layer pass
                if batch_idx not in batch_num_tokens:
                    num_tokens = [
                        int(batch_mask[j].sum().item())
                        for j in range(batch_mask.shape[0])
                    ]
                    batch_num_tokens[batch_idx] = num_tokens

                if manifest.dtype is None:
                    _dtype_map_rev = {
                        torch.float32: "float32",
                        torch.float16: "float16",
                        torch.bfloat16: "bfloat16",
                    }
                    manifest.dtype = _dtype_map_rev.get(
                        batch_acts.dtype, "float32"
                    )

                _write_layer_batch(layer_idx, batch_idx, batch_acts, batch_mask)
                del batch_acts, batch_mask

            completed_layers.add(layer_idx)

            # Write manifest after each layer for crash recovery.
            # On resume, completed layers let us skip N_batches existence
            # checks per layer.
            if layer_had_work:
                # Build partial manifest with batch info collected so far
                partial_manifest = ExtractionManifest(
                    model_name=model_name,
                    layers=layer_indices,
                    hidden_dim=hidden_dim,
                    total_prompts=len(prompts),
                    batch_size=batch_size,
                    prompts=list(prompts),
                    labels=list(labels) if labels is not None else None,
                    metadata=list(metadata) if metadata is not None else None,
                    created_at=manifest.created_at,
                    dtype=manifest.dtype,
                    completed_layers=sorted(completed_layers),
                )
                _write_manifest(partial_manifest, prefix, local_root=local_root)

            gc.collect()

    finally:
        read_pool.shutdown(wait=False)

    # Build final manifest with all batch entries
    for batch_idx, start, end in batch_ranges:
        if batch_idx in completed_batches:
            manifest.batches.append(existing_batch_infos[batch_idx])
            batches_skipped += 1
        else:
            manifest.batches.append(BatchInfo(
                file=_batch_filename(batch_idx),
                prompt_start=start,
                prompt_end=end,
                num_tokens=batch_num_tokens[batch_idx],
                status="complete",
            ))

    _write_manifest(manifest, prefix, local_root=local_root)

    elapsed = time.time() - start_time
    logger.info(
        f"[CONSOLIDATE] Complete: {num_batches - batches_skipped} batch files "
        f"written, {batches_skipped} skipped (cached), {elapsed:.1f}s"
    )

    return prefix


# ---------------------------------------------------------------------------
# push_extraction() — publish batch files directly to HuggingFace
# ---------------------------------------------------------------------------


@dataclass
class _ShardBatchInfo:
    """Pre-computed mapping from a shard to its source batch files."""

    shard_idx: int
    shuffled_start: int
    shuffled_end: int
    is_last_token: bool
    # batch_idx -> list of (position_in_batch, shuffled_pos)
    batch_to_prompts: dict[int, list[tuple[int, int]]]


def _build_shard_batch_map(
    manifest: ExtractionManifest,
    perm: list[int],
    lt_boundaries: list[int],
    rest_boundaries: list[int],
    lt_shard_count: int,
) -> list[_ShardBatchInfo]:
    """Pre-compute which batch files each shard needs.

    For every shard (last-token, then rest-token), determines the shuffled
    position range, maps each position back to its original prompt index
    via *perm*, then derives the batch index and position within that batch.

    Parameters
    ----------
    manifest : ExtractionManifest
        Extraction manifest with batch info.
    perm : list[int]
        Permutation where ``perm[shuffled_pos] = original_idx``.
    lt_boundaries : list[int]
        Number of prompts per last-token shard.
    rest_boundaries : list[int]
        Number of prompts per rest-token shard.
    lt_shard_count : int
        Count of last-token shards (``len(lt_boundaries)``).

    Returns
    -------
    list[_ShardBatchInfo]
        One entry per shard (last-token first, then rest-token).
    """
    n = manifest.total_prompts

    # Build lookup: original_idx -> (batch_idx, position_in_batch)
    orig_to_batch: dict[int, tuple[int, int]] = {}
    for batch_info in manifest.batches:
        if batch_info.status != "complete":
            continue
        batch_idx = batch_info.prompt_start // manifest.batch_size
        for j in range(batch_info.prompt_end - batch_info.prompt_start):
            orig_idx = batch_info.prompt_start + j
            orig_to_batch[orig_idx] = (batch_idx, j)

    result: list[_ShardBatchInfo] = []

    # Last-token shards
    offset = 0
    for local_idx, shard_size in enumerate(lt_boundaries):
        batch_to_prompts: dict[int, list[tuple[int, int]]] = {}
        for k in range(shard_size):
            shuffled_pos = offset + k
            if shuffled_pos >= n:
                break
            orig_idx = perm[shuffled_pos]
            if orig_idx in orig_to_batch:
                bid, pos = orig_to_batch[orig_idx]
                batch_to_prompts.setdefault(bid, []).append((pos, shuffled_pos))
        result.append(
            _ShardBatchInfo(
                shard_idx=local_idx,
                shuffled_start=offset,
                shuffled_end=offset + shard_size,
                is_last_token=True,
                batch_to_prompts=batch_to_prompts,
            )
        )
        offset += shard_size

    # Rest-token shards
    offset = 0
    for local_idx, shard_size in enumerate(rest_boundaries):
        batch_to_prompts = {}
        for k in range(shard_size):
            shuffled_pos = offset + k
            if shuffled_pos >= n:
                break
            orig_idx = perm[shuffled_pos]
            if orig_idx in orig_to_batch:
                bid, pos = orig_to_batch[orig_idx]
                batch_to_prompts.setdefault(bid, []).append((pos, shuffled_pos))
        result.append(
            _ShardBatchInfo(
                shard_idx=lt_shard_count + local_idx,
                shuffled_start=offset,
                shuffled_end=offset + shard_size,
                is_last_token=False,
                batch_to_prompts=batch_to_prompts,
            )
        )
        offset += shard_size

    return result


def _build_shard_for_layer(
    shard_info: _ShardBatchInfo,
    source_prefix: str,
    layer: int,
) -> torch.Tensor | None:
    """Build one shard's tensor for one layer, loading batches piecemeal.

    Instead of loading the entire layer into memory, loads only the batch
    files that contain prompts for this shard, extracts the needed rows,
    and frees each batch immediately.

    Parameters
    ----------
    shard_info : _ShardBatchInfo
        Pre-computed mapping of which batch rows this shard needs.
    source_prefix : str
        Local directory path or extraction prefix.
    layer : int
        Layer index to load.

    Returns
    -------
    torch.Tensor | None
        Concatenated shard tensor, or None if no data.
    """
    collected: list[tuple[int, torch.Tensor]] = []

    for batch_idx in sorted(shard_info.batch_to_prompts.keys()):
        prompts_in_batch = shard_info.batch_to_prompts[batch_idx]

        acts, mask = load_batch_layer(source_prefix, layer, batch_idx)

        for pos_in_batch, shuffled_pos in prompts_in_batch:
            prompt_mask = mask[pos_in_batch]
            num_tokens = int(prompt_mask.sum().item())
            prompt_acts = acts[pos_in_batch, :num_tokens, :]

            if shard_info.is_last_token:
                if prompt_acts.shape[0] > 0:
                    collected.append((shuffled_pos, prompt_acts[-1:]))
            else:
                if prompt_acts.shape[0] > 1:
                    collected.append((shuffled_pos, prompt_acts[:-1]))

        del acts, mask

    if not collected:
        return None

    collected.sort(key=lambda x: x[0])
    return torch.cat([t for _, t in collected], dim=0)


def _download_layer_batches_to_staging(
    manifest: ExtractionManifest,
    source_prefix: str,
    layer: int,
    staging_dir: Path,
    max_workers: int = 16,
) -> str:
    """Download all batch files for one layer to local staging.

    For S3/remote sources, downloads batch files in parallel to local
    disk so that :func:`load_batch_layer` can read them locally.
    For local sources, returns *source_prefix* unchanged.

    Parameters
    ----------
    manifest : ExtractionManifest
        Extraction manifest.
    source_prefix : str
        Original source prefix (local or remote).
    layer : int
        Layer index.
    staging_dir : Path
        Local directory for staged files.
    max_workers : int
        Parallel download threads.

    Returns
    -------
    str
        Local path to use as source prefix for ``load_batch_layer``.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from safetensors.torch import save_file

    local_staging = staging_dir / "staged_batches"
    layer_dir = local_staging / f"layer_{layer:03d}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    def _download_one(batch_idx: int) -> None:
        rel_path = _layer_batch_path(layer, batch_idx)
        local_file = local_staging / rel_path
        if local_file.exists():
            return  # already staged (resumability)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        acts, mask = load_batch_layer(source_prefix, layer, batch_idx)
        save_file({"activations": acts, "mask": mask}, str(local_file))
        del acts, mask

    batch_indices = []
    for batch_info in manifest.batches:
        if batch_info.status != "complete":
            continue
        batch_indices.append(batch_info.prompt_start // manifest.batch_size)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_download_one, bid) for bid in batch_indices]
        for fut in as_completed(futures):
            fut.result()  # raise any exceptions

    return str(local_staging)


def _cleanup_staged_layer(staging_dir: Path, layer: int) -> None:
    """Remove staged batch files for a completed layer."""
    import shutil

    layer_dir = staging_dir / "staged_batches" / f"layer_{layer:03d}"
    if layer_dir.exists():
        shutil.rmtree(layer_dir)


def push_extraction(
    source: str,
    repo_id: str,
    *,
    shard_max_bytes: int | None = None,
    private: bool = False,
    exist_ok: bool = False,
    description: str | None = None,
    license: str = "cc-by-4.0",
    token: str | None = None,
    shuffle: bool = True,
    stream: bool = False,
    stream_batch_size: int = 10,
    staging_dir: str | Path | None = None,
) -> str:
    """Publish an extraction directory as a HuggingFace dataset.

    Reads batch files from a directory produced by :func:`extract` or
    :func:`consolidate_cache`, builds layerwise shards, and uploads
    to HuggingFace Hub.

    Parameters
    ----------
    source : str | Path
        Path to extraction directory containing ``manifest.json``.
    repo_id : str
        HuggingFace repo ID (e.g. ``"username/my-activations"``).
    shard_max_bytes : int | None
        Max bytes per shard.  Defaults to 1 GB.
    private : bool
        Create a private repository.
    exist_ok : bool
        If False (default), raise if the repo already exists.
    description : str | None
        Description for the auto-generated README.
    license : str
        License identifier for the dataset card.
    token : str | None
        HuggingFace API token.
    shuffle : bool
        Deterministically shuffle prompts across shards.
    stream : bool
        If True, upload shards incrementally via ``create_commit``
        after each layer, deleting local copies to bound peak disk
        usage.  Supports resumability via a streaming manifest.
    stream_batch_size : int
        Number of shard files to buffer before uploading a batch.
        Only used when ``stream=True``.
    staging_dir : str | Path | None
        Directory for staging shard files when ``stream=True``.
        If None, a temporary directory is created.  Pass a persistent
        path to enable resumability — if the staging dir is a tmpdir
        that gets cleaned up, progress tracking is lost even though
        already-uploaded shards are detected via remote check.

    Returns
    -------
    str
        URL of the published dataset.
    """
    import tempfile

    from safetensors.torch import save_file

    from .sharing import (
        DEFAULT_SHARD_BYTES,
        _build_lmprobe_info,
        _build_readme,
        _check_hub_deps,
        _check_pyarrow,
        _check_shards_on_remote,
        _compute_shard_boundaries,
        _compute_shard_boundaries_variable,
        _deterministic_seed,
        _hidden_shard_filename,
        _shuffle_indices,
        _write_parquet_index,
    )
    from .sharing import (
        _load_manifest as _load_stream_manifest,
    )
    from .sharing import (
        _new_manifest as _new_stream_manifest,
    )
    from .sharing import (
        _save_manifest as _save_stream_manifest,
    )

    _check_hub_deps()
    _check_pyarrow()

    if shard_max_bytes is None:
        shard_max_bytes = DEFAULT_SHARD_BYTES

    source_prefix = str(source)
    manifest = load_manifest(source_prefix)

    prompts = manifest.prompts
    labels = manifest.labels
    metadata_list = manifest.metadata
    hidden_layers = manifest.layers
    hidden_dim = manifest.hidden_dim
    model_name = manifest.model_name
    n = manifest.total_prompts

    # Build per-prompt num_tokens from manifest
    per_prompt_tokens: list[int] = []
    for batch_info in sorted(manifest.batches, key=lambda b: b.prompt_start):
        per_prompt_tokens.extend(batch_info.num_tokens)

    assert len(per_prompt_tokens) == n, (
        f"num_tokens count {len(per_prompt_tokens)} != total_prompts {n}"
    )

    # Shuffle
    if shuffle:
        seed = _deterministic_seed(repo_id)
        perm = _shuffle_indices(n, seed)
    else:
        perm = list(range(n))

    shuffled_prompts = [prompts[i] for i in perm]
    shuffled_tokens = [per_prompt_tokens[i] for i in perm]
    shuffled_labels = [labels[i] for i in perm] if labels else None
    shuffled_metadata = [metadata_list[i] for i in perm] if metadata_list else None

    # Compute shard boundaries (full-sequence: last-token + rest-token shards)
    hidden_row_bytes = hidden_dim * 4

    # Last-token shards: one vector per prompt
    lt_boundaries = _compute_shard_boundaries(n, hidden_row_bytes, shard_max_bytes)
    lt_shard_count = len(lt_boundaries)

    # Rest-token shards: (num_tokens - 1) vectors per prompt
    rest_prompt_bytes = [max(tok - 1, 0) * hidden_dim * 4 for tok in shuffled_tokens]
    rest_boundaries: list[int] = []
    if any(b > 0 for b in rest_prompt_bytes):
        rest_boundaries = _compute_shard_boundaries_variable(
            rest_prompt_bytes, shard_max_bytes
        )

    hidden_boundaries = lt_boundaries + rest_boundaries

    # Build prompt metadata
    prompt_metadata: list[dict] = []
    for i in range(n):
        entry: dict[str, Any] = {
            "text": shuffled_prompts[i],
            "label": shuffled_labels[i] if shuffled_labels else None,
            "num_tokens": shuffled_tokens[i],
        }
        if shuffled_metadata:
            entry.update(shuffled_metadata[i])
        prompt_metadata.append(entry)

    # Assign shard metadata (same logic as sharing._compute_shard_plan)
    # Last-token shard assignments
    lt_row_in_shard = 0
    lt_shard_idx = 0
    lt_assignments: list[tuple[int, int]] = []
    for i in range(n):
        if lt_shard_idx < len(lt_boundaries) and lt_row_in_shard >= lt_boundaries[lt_shard_idx]:
            lt_shard_idx += 1
            lt_row_in_shard = 0
        lt_assignments.append((lt_shard_idx, lt_row_in_shard))
        lt_row_in_shard += 1

    # Rest-token shard assignments
    rest_shard_idx = 0
    rest_tok_offset = 0
    rest_row_in_shard = 0
    rest_assignments: list[tuple[int, int]] = []
    if rest_boundaries:
        for i in range(n):
            if (
                rest_shard_idx < len(rest_boundaries)
                and rest_row_in_shard >= rest_boundaries[rest_shard_idx]
            ):
                rest_shard_idx += 1
                rest_tok_offset = 0
                rest_row_in_shard = 0
            rest_assignments.append((lt_shard_count + rest_shard_idx, rest_tok_offset))
            rest_tok_offset += max(shuffled_tokens[i] - 1, 0)
            rest_row_in_shard += 1

    # Per-token arrays
    for i in range(n):
        num_tok = shuffled_tokens[i]
        lt_si, lt_off = lt_assignments[i]

        token_shard_ids: list[int] = []
        token_shard_offsets: list[int] = []

        if num_tok > 1 and rest_assignments:
            rest_si, rest_off = rest_assignments[i]
            for t in range(num_tok - 1):
                token_shard_ids.append(rest_si)
                token_shard_offsets.append(rest_off + t)
        token_shard_ids.append(lt_si)
        token_shard_offsets.append(lt_off)

        prompt_metadata[i]["token_shard_ids"] = token_shard_ids
        prompt_metadata[i]["token_shard_offsets"] = token_shard_offsets
        prompt_metadata[i]["shard_index"] = lt_si
        prompt_metadata[i]["row_offset"] = lt_off
        prompt_metadata[i]["token_offset"] = lt_off

    # Build tensor descriptors
    hidden_shards: list[dict[str, Any]] = []
    off = 0
    for _si, sz in enumerate(lt_boundaries):
        actual = min(sz, n - off)
        hidden_shards.append({"num_prompts": actual, "num_tokens": actual})
        off += actual
    off = 0
    for _si, sz in enumerate(rest_boundaries):
        actual = min(sz, n - off)
        hidden_shards.append({
            "num_prompts": actual,
            "num_tokens": sum(
                max(shuffled_tokens[off + j] - 1, 0) for j in range(actual)
            ),
        })
        off += actual

    tensor_descriptors: dict[str, Any] = {
        "hidden_layers": {
            "type": "hidden",
            "layers": hidden_layers,
            "dim": hidden_dim,
            "dtype": manifest.dtype,
            "storage": "full_sequence",
            "layout": "per_layer",
            "file_pattern": "tensors/hidden_layer{layer:03d}_shard{shard:03d}.safetensors",
            "key_pattern": "hidden.layer_{layer}",
            "shards": hidden_shards,
            "last_token_shards": lt_shard_count,
        },
    }

    # -- Set up directory and upload infrastructure --
    from huggingface_hub import HfApi

    if stream:
        from huggingface_hub import CommitOperationAdd

        if staging_dir is not None:
            tmpdir = Path(staging_dir)
        else:
            tmpdir = Path(tempfile.mkdtemp(prefix="lmprobe_push_extraction_"))
        (tmpdir / "tensors").mkdir(parents=True, exist_ok=True)
        (tmpdir / "index").mkdir(parents=True, exist_ok=True)

        # Load or create streaming manifest for resumability
        _loaded = _load_stream_manifest(tmpdir)
        resuming = _loaded is not None
        stream_manifest: dict = _loaded if _loaded is not None else _new_stream_manifest(repo_id)

        api = HfApi(token=token)
        api.create_repo(
            repo_id,
            exist_ok=True if resuming else exist_ok,
            private=private,
            repo_type="dataset",
        )

        # Compute expected shard files and check which exist on remote
        expected_files: list[str] = []
        for layer in hidden_layers:
            for si in range(len(hidden_boundaries)):
                expected_files.append(_hidden_shard_filename(layer, si))

        already_completed = set(stream_manifest.get("completed_shards", []))
        remote_existing = _check_shards_on_remote(api, repo_id, expected_files)
        skip_shards = already_completed | remote_existing

        if skip_shards:
            logger.info(
                "[PUSH_EXTRACTION] %d/%d shards already uploaded, skipping",
                len(skip_shards), len(expected_files),
            )

        # Build streaming callback
        pending_shards: list[tuple[Path, str]] = []

        def _flush_batch() -> None:
            if not pending_shards:
                return
            logger.info(
                "[PUSH_EXTRACTION] Uploading batch of %d shards",
                len(pending_shards),
            )
            stream_manifest["pending_batch"] = [
                [str(lp), rp] for lp, rp in pending_shards
            ]
            _save_stream_manifest(tmpdir, stream_manifest)

            operations = [
                CommitOperationAdd(
                    path_in_repo=rp, path_or_fileobj=str(lp),
                )
                for lp, rp in pending_shards
            ]
            api.create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message=f"Add {len(pending_shards)} shards",
                repo_type="dataset",
            )
            for lp, rp in pending_shards:
                lp.unlink(missing_ok=True)
                stream_manifest["completed_shards"].append(rp)
            stream_manifest.pop("pending_batch", None)
            _save_stream_manifest(tmpdir, stream_manifest)
            pending_shards.clear()

        def _on_shard_written(local_path: Path, repo_path: str) -> None:
            pending_shards.append((local_path, repo_path))
            if len(pending_shards) >= stream_batch_size:
                _flush_batch()

        # Retry any pending batch from a prior failed commit
        prior_batch = stream_manifest.get("pending_batch", [])
        if prior_batch:
            local_files_exist = all(
                Path(lp).exists() for lp, _ in prior_batch
            )
            if local_files_exist:
                logger.info(
                    "[PUSH_EXTRACTION] Retrying pending batch of %d shards",
                    len(prior_batch),
                )
                operations = [
                    CommitOperationAdd(path_in_repo=rp, path_or_fileobj=lp)
                    for lp, rp in prior_batch
                ]
                api.create_commit(
                    repo_id=repo_id,
                    operations=operations,
                    commit_message=f"Add {len(prior_batch)} shards (retry)",
                    repo_type="dataset",
                )
                for lp, rp in prior_batch:
                    Path(lp).unlink(missing_ok=True)
                    stream_manifest["completed_shards"].append(rp)
                stream_manifest.pop("pending_batch", None)
                _save_stream_manifest(tmpdir, stream_manifest)
            else:
                stream_manifest.pop("pending_batch", None)
                _save_stream_manifest(tmpdir, stream_manifest)
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="lmprobe_push_extraction_"))
        (tmpdir / "tensors").mkdir(parents=True, exist_ok=True)
        (tmpdir / "index").mkdir(parents=True, exist_ok=True)
        skip_shards = set()

    # Pre-compute shard-to-batch mappings (no activation data loaded)
    shard_map = _build_shard_batch_map(
        manifest, perm, lt_boundaries, rest_boundaries, lt_shard_count,
    )

    # For remote sources, stage batch files locally to avoid repeated
    # S3 reads.  Local sources are used directly.
    is_local_source = Path(source_prefix).is_dir()
    batch_staging_dir: Path | None = None
    if not is_local_source:
        batch_staging_dir = Path(
            tempfile.mkdtemp(prefix="lmprobe_batch_staging_")
        )

    # Write shards per layer (piecemeal — one shard at a time)
    logger.info(
        f"[PUSH_EXTRACTION] Writing shards for {len(hidden_layers)} layers, "
        f"{len(hidden_boundaries)} shards per layer"
    )

    try:
        from concurrent.futures import ThreadPoolExecutor

        # Pipeline: download next layer's batches while building shards
        # for the current layer.  Only matters for remote sources.
        download_executor = (
            ThreadPoolExecutor(max_workers=1)
            if batch_staging_dir is not None
            else None
        )
        next_layer_future = None

        def _needs_work(layer: int) -> bool:
            """Return True if at least one shard is missing for *layer*."""
            return not all(
                _hidden_shard_filename(layer, si) in skip_shards
                for si in range(len(hidden_boundaries))
            )

        for layer_idx, layer in enumerate(
            tqdm(hidden_layers, desc="Layers", unit="layer")
        ):
            # Skip entire layer if all its shards already exist on remote
            if not _needs_work(layer):
                logger.info(
                    "[PUSH_EXTRACTION] Layer %d: all shards exist, skipping",
                    layer,
                )
                continue

            # Obtain staged batches — either from the prefetched future
            # or by downloading synchronously (first layer / local source).
            if batch_staging_dir is not None:
                if next_layer_future is not None:
                    effective_source = next_layer_future.result()
                    next_layer_future = None
                else:
                    effective_source = _download_layer_batches_to_staging(
                        manifest, source_prefix, layer, batch_staging_dir,
                    )
            else:
                effective_source = source_prefix

            # Kick off download for the next layer that needs work while
            # we build shards for the current layer.
            if download_executor is not None and batch_staging_dir is not None:
                for future_idx in range(layer_idx + 1, len(hidden_layers)):
                    future_layer = hidden_layers[future_idx]
                    if _needs_work(future_layer):
                        next_layer_future = download_executor.submit(
                            _download_layer_batches_to_staging,
                            manifest,
                            source_prefix,
                            future_layer,
                            batch_staging_dir,
                        )
                        break

            key = f"hidden.layer_{layer}"

            for shard_info in shard_map:
                fname = _hidden_shard_filename(layer, shard_info.shard_idx)
                if fname in skip_shards:
                    continue
                shard_tensor = _build_shard_for_layer(
                    shard_info, effective_source, layer,
                )
                if shard_tensor is not None:
                    save_file(
                        {key: shard_tensor}, str(tmpdir / fname),
                    )
                    if stream:
                        _on_shard_written(tmpdir / fname, fname)
                del shard_tensor

            # In streaming mode, flush after each layer to free disk
            if stream:
                _flush_batch()

            # Clean up staged batch files for this layer
            if batch_staging_dir is not None:
                _cleanup_staged_layer(batch_staging_dir, layer)

            gc.collect()

        # Shut down the prefetch executor (cancel any outstanding future)
        if download_executor is not None:
            if next_layer_future is not None:
                next_layer_future.cancel()
            download_executor.shutdown(wait=False)
    finally:
        # Clean up staging directory
        if batch_staging_dir is not None:
            import shutil

            shutil.rmtree(batch_staging_dir, ignore_errors=True)

    # Write metadata
    lmprobe_info = _build_lmprobe_info(model_name, n, tensor_descriptors)
    _write_parquet_index(tmpdir, prompt_metadata, lmprobe_info)

    readme = _build_readme(
        model_name=model_name,
        lmprobe_info=lmprobe_info,
        num_prompts=n,
        repo_id=repo_id,
        description=description,
        license=license,
    )
    (tmpdir / "README.md").write_text(readme)

    # Upload
    if stream:
        # Upload metadata in a final commit
        metadata_files = list((tmpdir / "index").rglob("*")) + [
            tmpdir / "README.md",
        ]
        operations = [
            CommitOperationAdd(
                path_in_repo=str(f.relative_to(tmpdir)),
                path_or_fileobj=str(f),
            )
            for f in metadata_files
            if f.is_file()
        ]
        api.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message="Add dataset metadata",
            repo_type="dataset",
        )
        stream_manifest["metadata_uploaded"] = True
        _save_stream_manifest(tmpdir, stream_manifest)

        # Clean up metadata files
        for f in metadata_files:
            if f.is_file():
                f.unlink(missing_ok=True)
    else:
        api = HfApi(token=token)
        api.create_repo(
            repo_id,
            exist_ok=exist_ok,
            private=private,
            repo_type="dataset",
        )

        total_size = sum(f.stat().st_size for f in tmpdir.rglob("*") if f.is_file())
        logger.info(
            f"[PUSH_EXTRACTION] Uploading dataset ({total_size / 1e9:.2f} GB)"
        )

        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(tmpdir),
        )

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info(f"[PUSH_EXTRACTION] Published: {url}")

    return url
