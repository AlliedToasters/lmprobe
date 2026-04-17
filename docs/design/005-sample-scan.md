# 005: SampleScan — Breadth-First Activation Indexing

**Status**: Draft  
**Date**: 2026-04-15  
**Author**: Toast

## Context

Current interpretability workflows are depth-first: pick a layer, pick a contrast, extract full activations, probe. This is expensive to redo when hypotheses change and infeasible at frontier scale. SampleScan inverts this: produce a compressed, multi-lens index of model behavior across the entire depth and token axis in a single forward pass, then surgically retrieve full activations only where the index indicates signal.

## Core Concept

A SampleScan is a compressed activation index consisting of:

1. **Samples**: tokenized prompts (and completions) run through the model
2. **Deltas**: per-layer attention and MLP update vectors (the output of each sub-layer *before* it's added to the residual stream — not cumulative residual state)
3. **Channels**: named PCA bases fit on different subsets/contrasts of the samples (e.g. "global", "honesty", "deception")
4. **Projections**: per-token, per-layer, per-delta projections onto the top-k principal components of each channel

A scan enables cheap exploratory queries like "which layer/token/delta best separates contrast X?" without storing full activations.

## Single-Pass Pipeline

The critical design insight: PCA fitting and projection happen *between layer chunks* during the forward pass, not as a separate phase. The chunked pipeline already has idle GPU time while offloading layer weights — PCA slots into that window on CPU.

### Per-Layer Loop

```
For each layer L:
  GPU:  Run layer L forward on all sample batches
        Capture attn_delta and mlp_delta (pre-residual update vectors)
  GPU→CPU:  Offload layer L weights (existing chunked pipeline behavior)
  CPU:  Fit PCA on [N_tokens_total, hidden_dim] for attn_delta
  CPU:  Fit PCA on [N_tokens_total, hidden_dim] for mlp_delta
        (one fit per channel — global channel fits on all; contrastive channels fit on subsets)
  CPU:  Project all deltas onto top-k PCs for each channel
  CPU:  Append projections + coordinates to flat storage
  CPU:  Discard full deltas for layer L
  GPU:  Load layer L+1
```

### Memory Budget

Peak memory is one layer's deltas across all samples. For a 7B model (hidden_dim=4096), 1000 samples averaging 100 tokens:

```
1000 samples × 100 tokens × 4096 hidden × 2 (attn+mlp) × 2 bytes (float16) ≈ 1.6 GB
```

Well within a 128GB CPU memory budget. The full delta set is never materialized across layers.

### PCA Implementation

- Use sklearn `IncrementalPCA` for CPU-only environments
- Use cuml `IncrementalPCA` when RAPIDS is available (GPU-accelerated fitting during the CPU phase)
- Each (layer, delta_type, channel) gets its own PCA fit — `n_layers × 2 × n_channels` fits total
- Only the top-k components are retained (k=12 default, configurable per channel)

## Storage Schema

### Flat Sparse Projection Store

Variable sequence lengths across samples are handled with flat storage and a coordinate index — no padding or ragged arrays:

```
scan_name/
  metadata.json
  samples/
    samples.parquet            # sample_id, prompt_text, completion_text, token_ids, labels...
  channels/
    0_global/
      config.json              # channel name, k, fit_method
      basis.npy                # [n_layers, 2, hidden_dim, k] float16
    1_<contrast_name>/
      config.json
      basis.npy
      fit_metadata.json        # which sample_ids/labels used to fit
  projections/
    values.npy                 # [N_total, n_channels, k] float16
    coords.parquet             # N_total rows: sample_id, layer, token_pos, delta_type
```

**`values.npy`**: flat stack of all projected vectors. Row i is the projection of one token at one layer and delta type across all channels.

**`coords.parquet`**: coordinate index. Standard parquet filters give fast slicing (e.g. "all projections for sample 123 at layer 45") without loading the full values array.

**`basis.npy`**: the PCA basis matrices needed to interpret projections or add channels post-hoc.

### Adding Channels Post-Hoc

Adding a new channel requires either:
- (a) Re-running the forward pass for the fit subset to recompute deltas, or
- (b) Having cached full deltas (user's choice to retain them)

Once the new basis is fit, projecting existing samples is cheap: load basis, load full deltas (or re-extract), project, append new columns to `values.npy`.

## Query API

```python
scan = SampleScan.load("my_scan")

# Retrieve projections for a sample
proj = scan.get_projections(sample_id=123, channel="global")
# Returns [seq_len, n_layers, 2, k] by filtering coords + slicing values

# Separability map: which (layer, delta_type) best separates a binary label?
signal_map = scan.separability_map(
    channel="global",
    labels=scan.samples["is_honest"],
)
# Returns [n_layers, 2] array of separability scores (e.g. AUROC or t-statistic)

# Nearest neighbors in PC space at a specific coordinate
neighbors = scan.nearest_neighbors(
    sample_id=123, layer=45, delta="mlp", channel="global", n=50
)
```

## Second-Pass Retrieval (Post-MVP)

Given coordinates flagged by the scan, re-run the chunked forward pass with targeted hooks that only materialize full-precision activations at those points:

```python
full_acts = scan.retrieve_full(
    coordinates=[(123, 45, 17, "mlp"), (123, 46, 17, "mlp"), ...],
    precision="float32",
)
```

This is deferred from the MVP — it requires modifying the chunked pipeline to support selective materialization, and the retrieval patterns will be clearer once we've worked with real scan data.

## Open Design Questions

1. **Float precision for projections**: float16 is likely sufficient; validate on GoT downstream task
2. **Token alignment**: for decoder-only models, the delta at position i is computed when processing token i. Document clearly.
3. **MoE models**: project the combined MLP delta (post-routing, pre-residual-add), not individual expert outputs — that's what enters the residual stream
4. **Contrastive channel fit methods** (post-MVP): PCA on difference vectors, PCA on labeled subset, LDA, etc.
5. **cuml vs sklearn PCA**: cuml is faster but adds a RAPIDS dependency. Default to sklearn, detect cuml at runtime?

## MVP Scope

- Single model (Qwen 2.5 7B for fast iteration, then scale to larger)
- Global channel only (defer contrastive channels)
- Single-pass delta extraction + PCA fit + projection
- Flat sparse storage with parquet coordinate index
- Query API: `get_projections`, `separability_map`
- **Validation**: run on Geometry of Truth (GoT) dataset. The scan should automatically identify layer ~45 as the truth-separability peak, matching known results. This is a clean binary test of whether the compressed index preserves enough signal.
