"""HuggingFace Storage Bucket support for sharing activation datasets.

Transforms the per-prompt local cache into a consolidated dataset format
on push, and unpacks it back on pull.  The shared format is standalone-usable
(no lmprobe required) and efficient (few large files, not thousands of small ones).

File layout on bucket::

    bucket/
      README.md
      dataset_info.json
      manifest.json
      tokens.json                           # optional
      hidden.layer_16_000.safetensors
      logits_topk_000.safetensors
      ...
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .cache import (
    CachedPromptInfo,
    discover_cached,
    load_prompt_activations,
    load_prompt_logits,
    load_prompt_pooled_activations,
    save_prompt_activations,
    save_prompt_pooled_activations,
)

logger = logging.getLogger(__name__)

FORMAT_VERSION = "1.0"
DEFAULT_SHARD_BYTES = 1_073_741_824  # 1 GB


# =============================================================================
# Dependency check
# =============================================================================


def _check_bucket_deps() -> None:
    """Check that huggingface_hub is installed with sufficient version."""
    try:
        import huggingface_hub
    except ImportError:
        raise ImportError(
            "Bucket support requires huggingface_hub. "
            "Install with: pip install 'huggingface_hub>=0.25.0'"
        )
    from packaging.version import Version

    if Version(huggingface_hub.__version__) < Version("0.25.0"):
        raise ImportError(
            f"Bucket support requires huggingface_hub >= 0.25.0, "
            f"found {huggingface_hub.__version__}. "
            f"Upgrade with: pip install --upgrade huggingface_hub"
        )


# =============================================================================
# Tensor type strategies
# =============================================================================


# =============================================================================
# Discovery helpers
# =============================================================================


def _discover_prompts(
    model_name: str,
    prompts: list[str],
    *,
    skip_missing: bool = True,
) -> tuple[list[int], list[CachedPromptInfo]]:
    """Discover cache state for all prompts.

    Uses a spot-check optimization: discovers the first prompt fully, then
    spot-checks ~10 evenly-spaced prompts.  The full validation happens during
    consolidation (load failures are caught there).

    Returns
    -------
    kept_indices : list[int]
        Indices into *prompts* for prompts that have cached data.
    infos : list[CachedPromptInfo]
        Corresponding CachedPromptInfo for each kept prompt.
    """
    if not prompts:
        raise ValueError("No prompts provided")

    # Full discovery on first prompt
    first_info = discover_cached(model_name, prompts[0])
    if first_info is None:
        if not skip_missing:
            raise FileNotFoundError(
                f"No cached data for first prompt: {prompts[0]!r}"
            )
        # Fall through — we'll discover all and filter

    # Spot-check ~10 evenly-spaced prompts for uniformity
    n = len(prompts)
    spot_indices = set()
    if n > 2:
        step = max(1, n // 10)
        spot_indices = {i for i in range(0, n, step)}
        spot_indices.add(n - 1)
    spot_indices.add(0)

    spot_results: dict[int, CachedPromptInfo | None] = {}
    for i in spot_indices:
        spot_results[i] = discover_cached(model_name, prompts[i])

    # Log any non-uniformity in spot-check
    spot_infos = [v for v in spot_results.values() if v is not None]
    if spot_infos and first_info is not None:
        ref_raw = set(first_info.raw_layers)
        ref_pooled_keys = set(first_info.pooled.keys())
        ref_logits = first_info.has_logits
        for si in spot_infos:
            if (
                set(si.raw_layers) != ref_raw
                or set(si.pooled.keys()) != ref_pooled_keys
                or si.has_logits != ref_logits
            ):
                logger.info(
                    "[BUCKET] Spot-check found non-uniform cache state across prompts. "
                    "Full discovery will happen during consolidation."
                )
                break

    # For large prompt sets, assume uniformity based on spot-check.
    # The consolidation loop catches load failures at actual load time.
    kept_indices = []
    infos = []

    for i, prompt in enumerate(prompts):
        if i in spot_results:
            info = spot_results[i]
        else:
            # Assume uniform — will validate at load time
            info = first_info
        if info is None:
            if not skip_missing:
                raise FileNotFoundError(
                    f"No cached data for prompt index {i}: {prompt!r}"
                )
            logger.debug(f"[BUCKET] Skipping uncached prompt index {i}")
            continue
        kept_indices.append(i)
        infos.append(info)

    if not kept_indices:
        raise ValueError(
            "No prompts have cached data. Extract activations first."
        )

    return kept_indices, infos


def _compute_tensor_intersection(
    infos: list[CachedPromptInfo],
) -> dict[str, Any]:
    """Compute the intersection of available tensor types across infos.

    Returns a dict with keys:
      - "raw_layers": sorted list of layer indices present in ALL infos
      - "pooled": dict[strategy, sorted layer list] present in ALL infos
      - "has_logits": True only if ALL have logits
      - "logits_top_k": int or None (consistent across all, else None)
      - "has_perplexity": True only if ALL have perplexity
    """
    if not infos:
        return {
            "raw_layers": [],
            "pooled": {},
            "has_logits": False,
            "logits_top_k": None,
            "has_perplexity": False,
        }

    raw_sets = [set(i.raw_layers) for i in infos]
    raw_intersection = sorted(raw_sets[0].intersection(*raw_sets[1:]))

    # Pooled: intersect strategies, then intersect layers per strategy
    all_strategies = [set(i.pooled.keys()) for i in infos]
    common_strategies = all_strategies[0].intersection(*all_strategies[1:])
    pooled: dict[str, list[int]] = {}
    for strategy in common_strategies:
        layer_sets = [set(i.pooled[strategy]) for i in infos]
        common_layers = sorted(layer_sets[0].intersection(*layer_sets[1:]))
        if common_layers:
            pooled[strategy] = common_layers

    # Logits: has_logits means full vocab logits; logits_top_k means topk.
    # Preference: topk is smaller and more commonly cached, so prefer it.
    # Only fall back to full logits if topk is not consistently available.
    all_have_topk = all(i.logits_top_k is not None for i in infos)
    all_have_full_logits = all(i.has_logits for i in infos)

    # Determine which form of logits to push:
    # - If all have topk (with consistent k), push topk
    # - Else if all have full logits, push full (logits_top_k=None)
    # - Else no logits
    logits_top_k_values = {i.logits_top_k for i in infos if i.logits_top_k is not None}
    consistent_topk = len(logits_top_k_values) == 1

    if all_have_topk and consistent_topk:
        has_logits = True
        logits_top_k = logits_top_k_values.pop()
    elif all_have_topk and not consistent_topk:
        warnings.warn(
            f"All prompts have topk logits but with different k values: "
            f"{sorted(logits_top_k_values)}. Logits will not be included in "
            f"the push. Re-cache with a consistent top_k to include logits.",
            stacklevel=3,
        )
        has_logits = False
        logits_top_k = None
    elif all_have_full_logits:
        has_logits = True
        logits_top_k = None
    else:
        has_logits = False
        logits_top_k = None

    has_perplexity = all(i.has_perplexity for i in infos)

    return {
        "raw_layers": raw_intersection,
        "pooled": pooled,
        "has_logits": has_logits,
        "logits_top_k": logits_top_k,
        "has_perplexity": has_perplexity,
    }


def _filter_tensor_types(
    available: dict[str, Any],
    tensors_filter: list[str] | None,
) -> dict[str, Any]:
    """Apply user's tensor filter to available types.

    The filter uses manifest key names like "hidden.layer_16", "logits_topk".
    """
    if tensors_filter is None:
        return available

    result = {
        "raw_layers": [],
        "pooled": {},
        "has_logits": False,
        "logits_top_k": available.get("logits_top_k"),
        "has_perplexity": False,
    }

    for key in tensors_filter:
        m = re.match(r"^hidden\.layer_(\d+)$", key)
        if m:
            layer = int(m.group(1))
            if layer in available["raw_layers"]:
                result["raw_layers"].append(layer)
            continue

        # Check pooled: pooled.{strategy}.layer_{i}
        m = re.match(r"^pooled\.(\w+)\.layer_(\d+)$", key)
        if m:
            strategy = m.group(1)
            layer = int(m.group(2))
            if strategy in available["pooled"] and layer in available["pooled"][strategy]:
                result["pooled"].setdefault(strategy, []).append(layer)
            continue

        if key == "logits_topk" and available["has_logits"]:
            result["has_logits"] = True
            continue

        if key == "perplexity" and available["has_perplexity"]:
            result["has_perplexity"] = True
            continue

        logger.warning(f"[BUCKET] Unknown tensor filter key: {key!r}, ignoring")

    result["raw_layers"] = sorted(result["raw_layers"])
    return result


# =============================================================================
# Consolidation engine
# =============================================================================


def _consolidate_and_shard(
    model_name: str,
    prompts: list[str],
    kept_indices: list[int],
    tensor_types: dict[str, Any],
    labels: list[int | str | None] | None,
    shard_max_bytes: int,
    tokenizer: Any | None = None,
) -> tuple[list[Path], dict, list[dict], list[list[str]] | None]:
    """Consolidate cached tensors into sharded safetensors files.

    This is a single-pass engine that handles all tensor types together,
    ensuring per-prompt all-or-nothing semantics.

    Returns
    -------
    shard_files : list[Path]
        Paths to written shard files (in a tmpdir).
    manifest_tensors : dict
        The "tensors" section of manifest.json.
    manifest_prompts : list[dict]
        The "prompts" section of manifest.json.
    token_strings : list[list[str]] | None
        Per-prompt token strings if tokenizer was provided.
    """
    from safetensors.torch import save_file

    tmpdir = Path(tempfile.mkdtemp(prefix="lmprobe_bucket_"))

    raw_layers = tensor_types["raw_layers"]
    pooled = tensor_types["pooled"]
    has_logits = tensor_types["has_logits"]
    logits_top_k = tensor_types["logits_top_k"]

    # Determine which pooling strategy to use for raw layers
    # If there are pooled versions, those are preferred (they're already reduced)
    # For raw layers, we need to know if they're pooled or unpooled in cache
    # We'll check the first prompt's cache to determine

    # Build manifest structures
    manifest_tensors: dict[str, dict] = {}
    manifest_prompts: list[dict] = []

    # Accumulators per tensor type per shard
    # Key: manifest_key -> list of (shard_idx, tensors_dict, metadata)
    shard_files: list[Path] = []

    # We process all prompts, collecting tensors per type, and shard when
    # accumulated bytes exceed the limit.

    # Per-type accumulators
    type_accumulators: dict[str, list[torch.Tensor]] = {}
    type_shard_info: dict[str, list[dict]] = {}
    type_current_bytes: dict[str, int] = {}
    type_current_prompts: dict[str, int] = {}

    # Initialize accumulators for each tensor type we'll push
    for layer in raw_layers:
        key = f"hidden.layer_{layer}"
        type_accumulators[key] = []
        type_shard_info[key] = []
        type_current_bytes[key] = 0
        type_current_prompts[key] = 0

    for strategy, layers in pooled.items():
        for layer in layers:
            key = f"pooled.{strategy}.layer_{layer}"
            type_accumulators[key] = []
            type_shard_info[key] = []
            type_current_bytes[key] = 0
            type_current_prompts[key] = 0

    if has_logits:
        type_accumulators["logits_topk.values"] = []
        type_accumulators["logits_topk.indices"] = []
        type_shard_info["logits_topk"] = []
        type_current_bytes["logits_topk"] = 0
        type_current_prompts["logits_topk"] = 0

    token_strings: list[list[str]] | None = [] if tokenizer is not None else None
    skipped_count = 0

    from tqdm import tqdm

    for idx in tqdm(kept_indices, desc="Consolidating", unit="prompt"):
        prompt = prompts[idx]
        label = labels[idx] if labels is not None else None

        # Try to load all requested tensor types for this prompt.
        # If any fails, skip the prompt entirely (all-or-nothing).
        try:
            loaded: dict[str, torch.Tensor] = {}

            # Raw layers (loaded per-layer individually)
            # load_prompt_activations returns (1, seq_len, hidden_dim).
            # For v1, we only support pooled raw layers (seq_len == 1).
            # Variable-length unpooled push requires offset-based
            # concatenation — see GitHub issue for v2 plans.
            mask: torch.Tensor | None = None
            if raw_layers:
                for layer in raw_layers:
                    acts, mask = load_prompt_activations(
                        model_name, prompt, [layer]
                    )
                    # acts: (1, seq_len, hidden_dim)
                    loaded[f"hidden.layer_{layer}"] = acts
                if mask is not None:
                    loaded["_mask"] = mask

            # Pooled layers
            for strategy, layers in pooled.items():
                for layer in layers:
                    pooled_act = load_prompt_pooled_activations(
                        model_name, prompt, [layer], strategy
                    )
                    # pooled_act: (1, hidden_dim)
                    loaded[f"pooled.{strategy}.layer_{layer}"] = pooled_act

            # Logits (topk)
            if has_logits and logits_top_k is not None:
                logit_vals, logit_idxs = load_prompt_logits(
                    model_name, prompt, top_k=logits_top_k
                )
                loaded["logits_topk.values"] = logit_vals
                if logit_idxs is not None:
                    loaded["logits_topk.indices"] = logit_idxs

        except (FileNotFoundError, KeyError, OSError) as e:
            skipped_count += 1
            logger.debug(
                f"[BUCKET] Skipping prompt index {idx}: {e}"
            )
            continue

        # Determine num_tokens from attention mask
        num_tokens = None
        if "_mask" in loaded:
            num_tokens = int(loaded["_mask"].sum().item())

        # Raw layer tensors: reshape to (1, hidden_dim) for stacking.
        # If the cached activations are unpooled (seq_len > 1), reject
        # with a clear error — v1 only supports pooled/single-token raw.
        for layer in raw_layers:
            key = f"hidden.layer_{layer}"
            tensor = loaded[key]
            if tensor.dim() == 3 and tensor.shape[1] > 1:
                raise ValueError(
                    f"Unpooled raw activations (seq_len={tensor.shape[1]}) "
                    f"for layer {layer} are not supported in push v1. "
                    f"Use pooled activations or push pooled layers instead. "
                    f"See: https://github.com/AlliedToasters/lmprobe/issues"
                )
            # (1, 1, hidden_dim) or (1, hidden_dim) -> (1, hidden_dim)
            tensor = tensor.reshape(1, -1)
            type_accumulators[key].append(tensor.detach().cpu())
            type_current_bytes[key] += tensor.nelement() * tensor.element_size()
            type_current_prompts[key] += 1

        for strategy, layers in pooled.items():
            for layer in layers:
                key = f"pooled.{strategy}.layer_{layer}"
                tensor = loaded[key]
                tensor = tensor.reshape(1, -1)
                type_accumulators[key].append(tensor.detach().cpu())
                type_current_bytes[key] += tensor.nelement() * tensor.element_size()
                type_current_prompts[key] += 1

        if has_logits and "logits_topk.values" in loaded:
            for sub in ("logits_topk.values", "logits_topk.indices"):
                tensor = loaded[sub]
                type_accumulators[sub].append(tensor.detach().cpu())
            byte_size = sum(
                loaded[sub].nelement() * loaded[sub].element_size()
                for sub in ("logits_topk.values", "logits_topk.indices")
            )
            type_current_bytes["logits_topk"] += byte_size
            type_current_prompts["logits_topk"] += 1

        # Build prompt entry
        prompt_entry = {
            "index": len(manifest_prompts),
            "text": prompt,
            "label": label,
            "num_tokens": num_tokens,
        }
        manifest_prompts.append(prompt_entry)

        # Tokenize if tokenizer provided
        if tokenizer is not None and token_strings is not None:
            try:
                token_ids = tokenizer.encode(prompt)
                tokens = [tokenizer.decode([tid]) for tid in token_ids]
                token_strings.append(tokens)
            except Exception:
                token_strings.append([])

    if skipped_count > 0:
        logger.warning(
            f"[BUCKET] Skipped {skipped_count} prompts due to missing or "
            f"corrupt cache entries"
        )

    if not manifest_prompts:
        raise ValueError(
            "No prompts could be loaded from cache. Check that activations "
            "have been extracted."
        )

    # Now write all accumulated tensors as shards.
    # v1 only handles pooled (2D) tensors — each prompt is one row.
    def _write_shards(
        manifest_key: str,
        sf_key: str,
        accum: list[torch.Tensor],
        shard_max: int,
    ) -> list[dict]:
        """Stack accumulated (1, dim) tensors and write shards."""
        if not accum:
            return []

        shards = []
        current_tensors: list[torch.Tensor] = []
        current_bytes = 0
        current_prompts = 0
        shard_idx = 0

        for tensor in accum:
            t_bytes = tensor.nelement() * tensor.element_size()
            if current_tensors and current_bytes + t_bytes > shard_max:
                fname = (
                    f"{manifest_key.replace('.', '_')}"
                    f"_{shard_idx:03d}.safetensors"
                )
                stacked = torch.cat(current_tensors, dim=0)
                save_file({sf_key: stacked}, str(tmpdir / fname))
                shards.append({
                    "file": fname,
                    "num_prompts": current_prompts,
                })
                shard_files.append(tmpdir / fname)
                current_tensors = []
                current_bytes = 0
                current_prompts = 0
                shard_idx += 1

            current_tensors.append(tensor)
            current_bytes += t_bytes
            current_prompts += 1

        if current_tensors:
            fname = (
                f"{manifest_key.replace('.', '_')}"
                f"_{shard_idx:03d}.safetensors"
            )
            stacked = torch.cat(current_tensors, dim=0)
            save_file({sf_key: stacked}, str(tmpdir / fname))
            shards.append({
                "file": fname,
                "num_prompts": current_prompts,
            })
            shard_files.append(tmpdir / fname)

        return shards

    def _write_logits_shards(
        values_accum: list[torch.Tensor],
        indices_accum: list[torch.Tensor],
        shard_max: int,
    ) -> list[dict]:
        """Write logits shards (two tensors per shard file)."""
        if not values_accum:
            return []

        shards = []
        cur_vals: list[torch.Tensor] = []
        cur_idxs: list[torch.Tensor] = []
        current_bytes = 0
        current_prompts = 0
        shard_idx = 0

        for v, idx in zip(values_accum, indices_accum):
            t_bytes = (
                v.nelement() * v.element_size()
                + idx.nelement() * idx.element_size()
            )
            if cur_vals and current_bytes + t_bytes > shard_max:
                fname = f"logits_topk_{shard_idx:03d}.safetensors"
                save_file(
                    {
                        "logits_topk.values": torch.cat(cur_vals, dim=0),
                        "logits_topk.indices": torch.cat(cur_idxs, dim=0),
                    },
                    str(tmpdir / fname),
                )
                shards.append({"file": fname, "num_prompts": current_prompts})
                shard_files.append(tmpdir / fname)
                cur_vals = []
                cur_idxs = []
                current_bytes = 0
                current_prompts = 0
                shard_idx += 1

            cur_vals.append(v)
            cur_idxs.append(idx)
            current_bytes += t_bytes
            current_prompts += 1

        if cur_vals:
            fname = f"logits_topk_{shard_idx:03d}.safetensors"
            save_file(
                {
                    "logits_topk.values": torch.cat(cur_vals, dim=0),
                    "logits_topk.indices": torch.cat(cur_idxs, dim=0),
                },
                str(tmpdir / fname),
            )
            shards.append({"file": fname, "num_prompts": current_prompts})
            shard_files.append(tmpdir / fname)

        return shards

    # Write shards for each tensor type
    for layer in raw_layers:
        key = f"hidden.layer_{layer}"
        sf_key = f"hidden.layer_{layer}"
        accum = type_accumulators[key]
        if not accum:
            continue
        shard_descs = _write_shards(key, sf_key, accum, shard_max_bytes)
        sample = accum[0]
        dim = sample.shape[-1]
        # v1: raw layers are always reshaped to (1, dim) before
        # accumulation, so they're effectively pooled/single-token.
        manifest_tensors[key] = {
            "type": "hidden",
            "layer": layer,
            "dim": dim,
            "dtype": str(sample.dtype).replace("torch.", ""),
            "pooling": "from_cache",
            "shards": shard_descs,
        }

    for strategy, layers in pooled.items():
        for layer in layers:
            key = f"pooled.{strategy}.layer_{layer}"
            sf_key = f"pooled.{strategy}.layer_{layer}"
            accum = type_accumulators[key]
            if not accum:
                continue
            shard_descs = _write_shards(key, sf_key, accum, shard_max_bytes)
            sample = accum[0]
            dim = sample.shape[-1]
            manifest_tensors[key] = {
                "type": "pooled",
                "strategy": strategy,
                "layer": layer,
                "dim": dim,
                "dtype": str(sample.dtype).replace("torch.", ""),
                "pooling": strategy,
                "shards": shard_descs,
            }

    if has_logits and type_accumulators.get("logits_topk.values"):
        shard_descs = _write_logits_shards(
            type_accumulators["logits_topk.values"],
            type_accumulators["logits_topk.indices"],
            shard_max_bytes,
        )
        sample_v = type_accumulators["logits_topk.values"][0]
        manifest_tensors["logits_topk"] = {
            "type": "logits_topk",
            "k": logits_top_k,
            "dtype": str(sample_v.dtype).replace("torch.", ""),
            "pooling": "last_token",
            "shards": shard_descs,
        }

    return shard_files, manifest_tensors, manifest_prompts, token_strings


# =============================================================================
# Metadata builders
# =============================================================================


def _build_dataset_info(model_name: str, num_prompts: int) -> dict:
    """Build dataset_info.json contents."""
    import platform

    import torch
    import transformers

    from . import __version__

    # Try to get model revision
    revision = None
    try:
        from huggingface_hub import model_info

        info = model_info(model_name)
        revision = info.sha
    except Exception:
        pass

    nnsight_version = None
    try:
        import nnsight

        nnsight_version = nnsight.__version__
    except Exception:
        pass

    return {
        "format_version": FORMAT_VERSION,
        "model": {
            "name": model_name,
            "revision": revision,
        },
        "num_prompts": num_prompts,
        "provenance": {
            "lmprobe_version": __version__,
            "extraction_backend": "local",
            "nnsight_version": nnsight_version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "python_version": platform.python_version(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def _build_readme(
    model_name: str,
    dataset_info: dict,
    manifest_tensors: dict,
    num_prompts: int,
    bucket_id: str,
    description: str | None = None,
) -> str:
    """Generate the auto-generated README.md for the bucket."""
    revision = dataset_info["model"].get("revision") or "unknown"
    format_version = dataset_info["format_version"]
    provenance = dataset_info["provenance"]

    # Build tensor table
    table_rows = []
    for key, info in sorted(manifest_tensors.items()):
        layer = info.get("layer", "-")
        dim = info.get("dim", info.get("k", "-"))
        pooling = info.get("pooling") or "unpooled"
        n_shards = len(info.get("shards", []))
        table_rows.append(f"| {key} | {layer} | {dim} | {pooling} | {n_shards} |")

    tensor_table = "\n".join(table_rows) if table_rows else "| (none) | - | - | - | - |"

    # Find an example shard file
    example_shard = "hidden.layer_0_000.safetensors"
    for info in manifest_tensors.values():
        shards = info.get("shards", [])
        if shards:
            example_shard = shards[0]["file"]
            break

    desc_section = f"\n{description}\n" if description else ""

    model_url = f"https://huggingface.co/{model_name}"
    readme = f"""# {model_name} — Activation Dataset

Cached activations extracted from [`{model_name}`]({model_url})
(revision `{revision}`).
{desc_section}
## Contents

| Tensor | Layers | Dim | Pooling | Shards |
|--------|--------|-----|---------|--------|
{tensor_table}

- **Prompts:** {num_prompts}
- **Format version:** {format_version}

## Load with lmprobe

```python
from lmprobe import pull_from_bucket, load_from_bucket

# Option 1: Pull into local cache (enables probe training without re-extraction)
pull_from_bucket("{bucket_id}")

# Option 2: Load tensors directly
tensors, manifest = load_from_bucket("{bucket_id}")
# tensors["hidden.layer_16"].shape => (N, hidden_dim)

# Selective download (only specific layers)
pull_from_bucket("{bucket_id}", tensors=["hidden.layer_16"])
```

## Load without lmprobe

```python
import json
from safetensors import safe_open

manifest = json.load(open("manifest.json"))

# See what's available
print(list(manifest["tensors"].keys()))

# Load a specific layer's activations
with safe_open("{example_shard}", framework="pt") as f:
    print(f.keys())
    acts = f.get_tensor(list(f.keys())[0])

# Prompt text for row i
print(manifest["prompts"][i]["text"])
```

## Provenance

- **lmprobe version:** {provenance.get('lmprobe_version', 'unknown')}
- **Extraction backend:** {provenance.get('extraction_backend', 'unknown')}
- **Created:** {provenance.get('created_at', 'unknown')}
- **PyTorch:** {provenance.get('torch_version', 'unknown')}
- **Transformers:** {provenance.get('transformers_version', 'unknown')}
"""
    return readme


# =============================================================================
# Public API
# =============================================================================


def push_to_bucket(
    bucket_id: str,
    model_name: str,
    prompts: list[str],
    *,
    labels: list[int | str | None] | None = None,
    tensors: list[str] | None = None,
    shard_max_bytes: int = DEFAULT_SHARD_BYTES,
    private: bool = False,
    exist_ok: bool = False,
    skip_missing: bool = True,
    description: str | None = None,
    token: str | None = None,
) -> str:
    """Push cached activations to a HuggingFace Storage Bucket.

    Introspects the local cache via :func:`discover_cached` to learn what's
    available for each prompt, consolidates into sharded safetensors files,
    and uploads.

    Parameters
    ----------
    bucket_id : str
        HuggingFace repo ID (e.g. "username/my-activations").
    model_name : str
        The model whose activations are cached.
    prompts : list[str]
        Prompts to push (must have cached activations).
    labels : list[int | str | None] | None
        Per-prompt labels, positionally aligned with *prompts*.
    tensors : list[str] | None
        Filter: only push these tensor types (e.g. ``["hidden.layer_16"]``).
        None pushes all available types.
    shard_max_bytes : int
        Max bytes per shard file. Default 1 GB.
    private : bool
        Create a private repository.
    exist_ok : bool
        If False (default), raise if the repo already has data.
    skip_missing : bool
        If True (default), skip prompts missing from cache with a warning.
        If False, raise on any missing prompt.
    description : str | None
        Description for the auto-generated README.
    token : str | None
        HuggingFace API token. Uses cached token if None.

    Returns
    -------
    str
        URL of the bucket.
    """
    _check_bucket_deps()
    from huggingface_hub import HfApi

    if labels is not None and len(labels) != len(prompts):
        raise ValueError(
            f"labels length ({len(labels)}) != prompts length ({len(prompts)})"
        )

    # Step 1: Discover cache state
    logger.info(f"[BUCKET] Discovering cache for {len(prompts)} prompts...")
    kept_indices, infos = _discover_prompts(
        model_name, prompts, skip_missing=skip_missing
    )

    if len(kept_indices) < len(prompts):
        logger.warning(
            f"[BUCKET] {len(prompts) - len(kept_indices)} prompts missing from "
            f"cache, {len(kept_indices)} will be pushed"
        )

    # Step 2: Compute intersection of tensor types
    available = _compute_tensor_intersection(infos)

    # Step 3: Apply user filter
    tensor_types = _filter_tensor_types(available, tensors)

    # Verify we have something to push
    has_any = (
        tensor_types["raw_layers"]
        or tensor_types["pooled"]
        or tensor_types["has_logits"]
    )
    if not has_any:
        raise ValueError(
            "No tensor types available to push after filtering. "
            f"Available: raw_layers={available['raw_layers']}, "
            f"pooled={list(available['pooled'].keys())}, "
            f"logits={available['has_logits']}"
        )

    # Try to get tokenizer for token strings
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        logger.debug("[BUCKET] Could not load tokenizer, skipping token strings")

    # Step 4: Consolidate and shard
    logger.info("[BUCKET] Consolidating cached tensors into shards...")
    shard_files, manifest_tensors, manifest_prompts, token_strings = (
        _consolidate_and_shard(
            model_name=model_name,
            prompts=prompts,
            kept_indices=kept_indices,
            tensor_types=tensor_types,
            labels=labels,
            shard_max_bytes=shard_max_bytes,
            tokenizer=tokenizer,
        )
    )

    # Step 5: Build metadata files
    num_prompts = len(manifest_prompts)
    dataset_info = _build_dataset_info(model_name, num_prompts)
    manifest = {
        "tensors": manifest_tensors,
        "prompts": manifest_prompts,
    }

    # Write metadata to tmpdir (same parent as shards)
    if shard_files:
        tmpdir = shard_files[0].parent
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="lmprobe_bucket_"))

    with open(tmpdir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    with open(tmpdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if token_strings is not None:
        # Scale note: 10k prompts x 100 tokens ~ 30-50MB JSON. Fine for v1.
        # If this becomes a bottleneck, migrate to JSONL or safetensors string encoding.
        with open(tmpdir / "tokens.json", "w") as f:
            json.dump(token_strings, f)

    readme = _build_readme(
        model_name=model_name,
        dataset_info=dataset_info,
        manifest_tensors=manifest_tensors,
        num_prompts=num_prompts,
        bucket_id=bucket_id,
        description=description,
    )
    with open(tmpdir / "README.md", "w") as f:
        f.write(readme)

    # Step 6: Upload to HuggingFace
    api = HfApi(token=token)
    api.create_repo(
        bucket_id,
        exist_ok=exist_ok,
        private=private,
        repo_type="dataset",
    )
    api.upload_folder(
        repo_id=bucket_id,
        folder_path=str(tmpdir),
        repo_type="dataset",
        commit_message=f"Upload activation dataset ({num_prompts} prompts)",
    )

    # Cleanup tmpdir
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)

    url = f"https://huggingface.co/datasets/{bucket_id}"
    logger.info(f"[BUCKET] Pushed {num_prompts} prompts to {url}")
    return url


def pull_from_bucket(
    bucket_id: str,
    *,
    tensors: list[str] | None = None,
    target_prompts: list[str] | None = None,
    overwrite: bool = False,
    token: str | None = None,
) -> int:
    """Pull activations from a HuggingFace bucket into the local cache.

    Parameters
    ----------
    bucket_id : str
        HuggingFace repo ID.
    tensors : list[str] | None
        Only pull these tensor types. None pulls all.
    target_prompts : list[str] | None
        Only pull these prompts. None pulls all.
    overwrite : bool
        If False (default), skip prompts already in local cache.
    token : str | None
        HuggingFace API token.

    Returns
    -------
    int
        Number of prompts unpacked into local cache.
    """
    _check_bucket_deps()
    from huggingface_hub import hf_hub_download

    # Download metadata
    manifest_path = hf_hub_download(
        bucket_id, "manifest.json", repo_type="dataset", token=token
    )
    info_path = hf_hub_download(
        bucket_id, "dataset_info.json", repo_type="dataset", token=token
    )

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(info_path) as f:
        dataset_info = json.load(f)

    # Version check
    remote_version = dataset_info.get("format_version", "1.0")
    remote_major = int(remote_version.split(".")[0])
    local_major = int(FORMAT_VERSION.split(".")[0])
    if remote_major != local_major:
        raise ValueError(
            f"Incompatible format version: bucket has {remote_version}, "
            f"lmprobe supports {FORMAT_VERSION}. "
            f"Please upgrade lmprobe: pip install --upgrade lmprobe"
        )
    remote_minor = int(remote_version.split(".")[1])
    local_minor = int(FORMAT_VERSION.split(".")[1])
    if remote_minor > local_minor:
        warnings.warn(
            f"Bucket format {remote_version} is newer than supported "
            f"{FORMAT_VERSION}. Some tensor types may be skipped.",
            stacklevel=2,
        )

    model_name = dataset_info["model"]["name"]
    all_prompts = manifest["prompts"]
    manifest_tensors = manifest["tensors"]

    # Filter prompts
    if target_prompts is not None:
        target_set = set(target_prompts)
        prompt_indices = [
            i for i, p in enumerate(all_prompts) if p["text"] in target_set
        ]
    else:
        prompt_indices = list(range(len(all_prompts)))

    # Filter tensor types
    if tensors is not None:
        tensor_keys = [k for k in manifest_tensors if k in tensors]
        unknown = set(tensors) - set(manifest_tensors.keys())
        if unknown:
            warnings.warn(
                f"Requested tensor types not found in bucket: {unknown}",
                stacklevel=2,
            )
    else:
        tensor_keys = list(manifest_tensors.keys())

    # Dedup: skip already-cached prompts
    if not overwrite:
        new_indices = []
        for i in prompt_indices:
            prompt_text = all_prompts[i]["text"]
            existing = discover_cached(model_name, prompt_text)
            if existing is None:
                new_indices.append(i)
        skipped = len(prompt_indices) - len(new_indices)
        if skipped > 0:
            logger.info(
                f"[BUCKET] Skipping {skipped} prompts already in local cache"
            )
        prompt_indices = new_indices

    if not prompt_indices:
        logger.info("[BUCKET] All prompts already cached locally")
        return 0

    unpacked = 0
    from tqdm import tqdm

    for tensor_key in tensor_keys:
        tensor_info = manifest_tensors[tensor_key]
        shards = tensor_info["shards"]

        # Determine which shards we need based on prompt indices
        # Build a mapping: prompt_global_index -> (shard_idx, local_offset)
        shard_prompt_ranges: list[tuple[int, int]] = []
        offset = 0
        for shard in shards:
            n = shard["num_prompts"]
            shard_prompt_ranges.append((offset, offset + n))
            offset += n

        # For each needed prompt, find which shard contains it
        needed_shards: dict[int, list[tuple[int, int]]] = {}
        for global_idx in prompt_indices:
            for si, (start, end) in enumerate(shard_prompt_ranges):
                if start <= global_idx < end:
                    local_offset = global_idx - start
                    needed_shards.setdefault(si, []).append(
                        (global_idx, local_offset)
                    )
                    break

        # Download and unpack each needed shard
        for si, entries in tqdm(
            needed_shards.items(),
            desc=f"Pulling {tensor_key}",
            unit="shard",
        ):
            shard = shards[si]
            shard_path = hf_hub_download(
                bucket_id,
                shard["file"],
                repo_type="dataset",
                token=token,
            )

            from safetensors import safe_open

            with safe_open(shard_path, framework="pt") as f:
                sf_keys = list(f.keys())
                for sf_key in sf_keys:
                    data = f.get_tensor(sf_key)

                    for global_idx, local_offset in entries:
                        prompt_text = all_prompts[global_idx]["text"]
                        num_tokens = all_prompts[global_idx].get("num_tokens")

                        # Extract the row for this prompt
                        row = data[local_offset: local_offset + 1]

                        # Save to cache based on tensor type
                        m = re.match(r"^hidden\.layer_(\d+)$", tensor_key)
                        if m:
                            layer = int(m.group(1))
                            # Create a dummy mask if needed
                            if num_tokens is not None:
                                mask = torch.ones(1, row.shape[1] if row.dim() == 3 else 1)
                            else:
                                mask = torch.ones(1, 1)
                            save_prompt_activations(
                                model_name, prompt_text, [layer], row, mask
                            )
                            continue

                        m = re.match(r"^pooled\.(\w+)\.layer_(\d+)$", tensor_key)
                        if m:
                            strategy = m.group(1)
                            layer = int(m.group(2))
                            save_prompt_pooled_activations(
                                model_name, prompt_text, [layer], row, strategy
                            )
                            continue

                        if tensor_key == "logits_topk":
                            if sf_key == "logits_topk.values":
                                # We need both values and indices — handled together
                                # Load indices too
                                pass  # handled below
                            break

        # Special handling for logits_topk (two tensors per shard)
        if tensor_key == "logits_topk":
            for si, entries in needed_shards.items():
                shard = shards[si]
                shard_path = hf_hub_download(
                    bucket_id,
                    shard["file"],
                    repo_type="dataset",
                    token=token,
                )
                from safetensors import safe_open

                with safe_open(shard_path, framework="pt") as f:
                    values = f.get_tensor("logits_topk.values")
                    indices = f.get_tensor("logits_topk.indices")

                    for global_idx, local_offset in entries:
                        prompt_text = all_prompts[global_idx]["text"]
                        v_row = values[local_offset: local_offset + 1]
                        i_row = indices[local_offset: local_offset + 1]
                        # save_prompt_logits expects full logits — we save top-k
                        # directly using the cache's internal format
                        from .cache import (
                            _LOGITS_TOP_K_INDICES_KEY,
                            _LOGITS_TOP_K_VALUES_KEY,
                            _merge_save_backend,
                            _prepare_tensor,
                            _prompt_cache_key,
                            _register_model,
                        )

                        _register_model(model_name)
                        cache_key = _prompt_cache_key(model_name, prompt_text)
                        new_tensors = {
                            _LOGITS_TOP_K_VALUES_KEY: _prepare_tensor(v_row),
                            _LOGITS_TOP_K_INDICES_KEY: _prepare_tensor(i_row),
                        }
                        _merge_save_backend(cache_key, new_tensors)

    unpacked = len(prompt_indices)
    logger.info(f"[BUCKET] Unpacked {unpacked} prompts into local cache")
    return unpacked


def load_from_bucket(
    bucket_id: str,
    *,
    tensors: list[str] | None = None,
    token: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Load tensors directly from a HuggingFace bucket.

    Returns consolidated tensors and the manifest, without writing to
    the local cache.

    Parameters
    ----------
    bucket_id : str
        HuggingFace repo ID.
    tensors : list[str] | None
        Only load these tensor types. None loads all.
    token : str | None
        HuggingFace API token.

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict]
        (tensors_by_key, manifest) where tensors_by_key maps manifest
        tensor keys to concatenated tensors across all shards.
    """
    _check_bucket_deps()
    from huggingface_hub import hf_hub_download

    # Download metadata
    manifest_path = hf_hub_download(
        bucket_id, "manifest.json", repo_type="dataset", token=token
    )
    info_path = hf_hub_download(
        bucket_id, "dataset_info.json", repo_type="dataset", token=token
    )

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(info_path) as f:
        dataset_info = json.load(f)

    # Version check
    remote_version = dataset_info.get("format_version", "1.0")
    remote_major = int(remote_version.split(".")[0])
    local_major = int(FORMAT_VERSION.split(".")[0])
    if remote_major != local_major:
        raise ValueError(
            f"Incompatible format version: bucket has {remote_version}, "
            f"lmprobe supports {FORMAT_VERSION}. "
            f"Please upgrade lmprobe: pip install --upgrade lmprobe"
        )

    manifest_tensors = manifest["tensors"]

    # Filter tensor types
    if tensors is not None:
        tensor_keys = [k for k in manifest_tensors if k in tensors]
    else:
        tensor_keys = list(manifest_tensors.keys())

    result: dict[str, torch.Tensor] = {}
    from safetensors import safe_open

    for tensor_key in tensor_keys:
        tensor_info = manifest_tensors[tensor_key]
        shards = tensor_info["shards"]

        shard_tensors: dict[str, list[torch.Tensor]] = {}

        for shard in shards:
            shard_path = hf_hub_download(
                bucket_id,
                shard["file"],
                repo_type="dataset",
                token=token,
            )
            with safe_open(shard_path, framework="pt") as f:
                for sf_key in f.keys():
                    shard_tensors.setdefault(sf_key, []).append(
                        f.get_tensor(sf_key)
                    )

        # Concatenate across shards
        for sf_key, parts in shard_tensors.items():
            result[sf_key] = torch.cat(parts, dim=0)

    return result, manifest
