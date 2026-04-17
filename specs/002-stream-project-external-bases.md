---
status: draft
affects: [backends]
---

# Spec 002: Stream-project through external bases

## Motivation

Both `ChunkedLocalBackend.scan_forward` and `DiskOffloadBackend.scan_forward`
collect per-signal, per-layer deltas across **every microbatch in a chunk**
before projecting them. The resulting flow per chunk, per signal, per layer is:

```python
for batch in batches:
    hook appends delta.detach().cpu()          # [B, S, H] bf16

# after the batch loop:
stacked = torch.cat(captures, dim=0)            # [N, S, H] bf16  (copy)
flat    = stacked.reshape(-1, dim).float()      # [N*S, H] fp32   (2× copy)
projected = flat @ basis.astype(np.float32)     # [N*S, k] fp32
```

The `stacked` contiguous copy, the `.float()` upcast, and the per-signal
capture list all coexist in CPU RAM around the point where PCA / projection
fires. For a single chunk × single signal this is:

- captures list:  `N*S*H × 2B`   (bf16)
- stacked:         `N*S*H × 2B`   (bf16, concatenated copy)
- flat:            `N*S*H × 4B`   (fp32)

Multiplied by `|signals|` (2 in the default case: `attn_delta + mlp_delta`),
per-chunk-per-layer CPU peak reaches `~6 × N × S × H` bytes in transient
working set.

On Mistral-Small 3.1 (H=5120) with one dataset's fused sweep (N≈500,
S≈3500 after padding), that's ≈100 GB of transient CPU RAM per layer on top of
the residual buffer (`batch_hidden_states`) and the model weights in
`ChunkedLocalBackend`. On a 128 GB workstation this OOM-kills the process —
observed repeatedly on Liar's Bench whitebox's `cache_all` flow.

Crucially, **none of this accumulation is necessary when the user already has
a fitted basis.** The current scan-fit codepath needs the full flat activation
matrix to run PCA over it. Projection-through-a-fixed-basis does not:
`(delta @ basis)` can be computed *immediately* on each microbatch's GPU
tensor, producing a `[B, S, k]` output that is ~H/k (typically 64–128×)
smaller than the input.

## Goal

When `external_bases` is provided to `scan_forward`, avoid materializing the
per-signal, per-layer activation tensor anywhere. Project each microbatch's
delta on GPU as it is captured, and only keep the tiny `[B, S, k]` projection
on CPU.

Outcome: drop per-chunk CPU RAM peak from `~6 × N × S × H` to
`~N × S × H` (just the residual buffer, unchanged) plus `~1 × B × S × H` on
GPU during the forward (also unchanged from today). For the Mistral sweep
above, that is a 5–6× memory-headroom improvement — enough to keep the run
inside 128 GB.

## Non-goals

- **PCA-fit path unchanged.** `external_bases is None` continues to use the
  accumulate → stack → fp32 flat → PCA path. Fitting genuinely needs the full
  flat matrix. Bases are fit once per model in `fit_basis.py`; only projection
  is in the hot loop for downstream analysis (`experiment_pipeline.cache_all`,
  paraphrase sweeps, new-prompt visualizations, etc.).

- **No output-shape changes.** Every callsite receiving
  `(projections, coords, token_ids_per_sample, seq_lengths, attention_mask,
  signal_dims)` from `scan_forward` must observe identical values (modulo the
  expected bf16/fp16 GPU-vs-CPU matmul tolerance). `SampleScan.batch_project`,
  `batch_project_grouped`, and `scan_storage` all consume this tuple.

- **No change to `extract_all`** (the older activation-extraction API on
  `DiskOffloadBackend`). It already has a per-layer, per-N capture pattern
  that is appropriate for raw-activation consumers and does not share the
  projection-only opportunity. If that path needs memory work it gets a
  separate spec.

## Design

### Pre-load bases to GPU once

Before the chunk/layer loop, upcast `external_bases[sig]` to `float32` on
the device:

```python
basis_gpu: dict[str, torch.Tensor] = {}
if external_bases is not None:
    for sig, arr in external_bases.items():
        # arr: np.ndarray [n_layers, dim, k]  fp16
        basis_gpu[sig] = torch.from_numpy(arr).to(
            device=device, dtype=torch.float32,
        )
```

Cost: `n_layers × dim × k × 4B` per signal. For a 70B model
(n_layers≈80, dim≈8192, k=64): `≈160 MB × 2 signals = 320 MB` VRAM, one-time.

Rationale for fp32 on GPU: matches the legacy CPU `.float()` path, preserves
numerical parity with scans produced before this spec. The delta is upcast
per-microbatch on GPU (`delta.float()`), multiplied, then cast back to fp16
before the CPU roundtrip.

### Hook keeps tensors on GPU

`_resolve_signal_hooks` (chunked backend) and the inline hook builder
(disk_offload backend) gain a `capture_device: str = "cpu"` parameter. When
`external_bases` is in play, callers pass `capture_device="gpu"`. The hook
appends `delta.detach()` (still on device) instead of `delta.detach().cpu()`.

### Per-microbatch projection

Inside the batch loop, replace the
`per_layer_captures[...][sig].append(buf[0])` line with: if
`external_bases is not None`, immediately project:

```python
for sig_name, handle, hook_buf in hooks:
    handle.remove()
    if not hook_buf:
        continue
    delta_gpu = hook_buf[0]           # [B, S, H] bf16 on device
    if external_bases is not None and sig_name in external_bases:
        B, S, H = delta_gpu.shape
        basis = basis_gpu[sig_name][layer_idx]          # [H, k] fp32
        flat = delta_gpu.reshape(-1, H).to(torch.float32)
        projected = (flat @ basis)                       # [B*S, k] fp32
        projected = projected.reshape(B, S, -1).to(torch.float16)
        per_layer_projected[layer_idx][sig_name].append(
            projected.cpu().numpy()
        )
        if sig_name not in signal_dims:
            signal_dims[sig_name] = H
    else:
        per_layer_captures[layer_idx][sig_name].append(
            delta_gpu.cpu()
        )
```

The residual branch (`capture_residual`) gets the same treatment: if
`external_bases` has `"residual"`, project `hs` in-place; otherwise fall back
to the old `.cpu()` append.

### End-of-chunk assembly

Split the existing `pca_items` loop into two branches:

- **External-basis branch**: concatenate the per-batch projections along dim 0,
  flatten `[N, S, k] → [N*S, k]`, append to `all_proj_chunks`, emit coords in
  the existing `(sample_id, layer, token_pos, signal)` order. Copy the basis
  slice into `signal_bases[sig][layer_idx]` for the returned `final_bases`.

- **PCA-fit branch**: unchanged. Legacy captures → stacked → flat → PCA →
  project path remains.

### Output ordering parity

The per-`(layer, signal)` row order in the legacy path is:
`(batch0_sample0_tok0..tokS-1, batch0_sample1_*, ..., batch_last_sampleN_*)`.
Concatenating the stream-project list along dim 0 before flattening preserves
this exact order — the outer dim of each list entry is "batch samples" in the
same sequence. `sample_id` is regenerated in the outer `range(B_total)` loop,
identical to today's code.

### API surface

Internal-only. No public signature changes.

- `_resolve_signal_hooks` gains a keyword-only `capture_device` param
  (default `"cpu"`, preserving today's behavior).
- Disk-offload's inline hook gains the same.
- `scan_forward` signature is unchanged in both backends.

### Implementation footprint

- `ChunkedLocalBackend._resolve_signal_hooks`: +1 param, +1 branch (~5 lines).
- `ChunkedLocalBackend.scan_forward`: new `per_layer_projected` dict, new
  stream-project block inside the batch loop, new external-basis branch in
  the per-chunk assembly (~40 lines added, ~0 removed — the PCA path stays).
- `DiskOffloadBackend.scan_forward`: mirror (~40 lines).
- No change to callers, storage format, or `SampleScan` API.

## Acceptance criteria

1. **Numerical parity.** For a small model (gpt2, 2 layers) and a pre-fit
   basis, `scan_forward(..., external_bases=basis)` run with the new
   stream-project path produces the same `projections` array as the
   legacy path (existing path on a branch without this change), within
   `atol=1e-2` in fp16 space. `coords`, `token_ids_per_sample`,
   `seq_lengths`, and `signal_dims` match bit-for-bit.

2. **Memory reduction.** Under the stream-project path with
   `external_bases` provided, there is no reachable Python object that holds
   a `[N_total, S, H]` activation tensor at any point during the chunk
   loop. Verified via a test that mocks the hook and asserts the capture
   list length stays at 1 (the current batch only) through the inner loop.

3. **PCA-fit path unchanged.** All existing tests for scan-fit pass
   unmodified.

4. **No API break.** `SampleScan.batch_project`,
   `SampleScan.batch_project_grouped`, and `SampleScan.project_prompt`
   return identical shapes and semantics. Existing `scans/` on disk remain
   loadable.

5. **Both backends updated.** ChunkedLocalBackend and DiskOffloadBackend
   both implement the stream-project path symmetrically.

## Migration

Purely internal. Downstream code (including
`liars_bench_whitebox/experiment_pipeline.py::cache_all`) picks up the
speedup and memory fix automatically via `scan.batch_project_grouped(...)`.

## Risk

- **fp32-on-GPU vs fp32-on-CPU matmul**: hardware matmul kernels may produce
  slightly different results (tensor cores round differently than Eigen).
  Parity test uses a fp16 tolerance (`atol=1e-2`) rather than exact equality.
  Numerical-analysis users who rely on the legacy path can still fit without
  external bases; the delta only appears for the frozen-basis projection
  path.

- **GPU VRAM footprint**: stream-project adds `basis_gpu` (a few hundred MB at
  most for a 70B × 2-signal setup), plus a per-microbatch fp32 upcast of the
  delta (`B × S × H × 4B`). For `B=2, S=4000, H=8192` this is 256 MB
  transient VRAM — negligible next to layer weights.
