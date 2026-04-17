---
status: draft
affects: [backends]
---

# Spec 004: Memmap-backed `batch_hidden_states` in `ChunkedLocalBackend`

## Motivation

`ChunkedLocalBackend.scan_forward` keeps every batch's chunk-boundary
hidden state resident on CPU via
`batch_hidden_states: list[torch.Tensor]`. Each entry is
`[batch_size, S_max, hidden_dim]` in bf16. Aggregate CPU residency scales
as `N_samples × S_max × hidden_dim × 2 bytes`.

For a mid-size run — 9260 samples × 574 tokens × 5120 hidden × 2 bytes =
**54 GB** — this list alone dominates the CPU footprint. Combined with
the full model on CPU (~48 GB for a 24B-param bf16 checkpoint), the
baseline is ~102 GB before any forward-pass activations. On a 128 GB box,
GC + fragmentation pushes chunk-2 forward over the edge → OOM-kill.

Observed on RTX 5090 / 128 GB / Mistral-Small-3.1-24B, chunked backend,
`batch_project_reduced` over ~9k samples. Same pattern will bite every
70B+ scan at even modest sample counts.

## Design

Back `batch_hidden_states` with a single `np.memmap`'d binary file of
shape `[N_total, S_max, hidden_dim]`. Between chunks each batch reads its
slice from the memmap, runs through the chunk's layers on GPU, writes
the updated slice back. Linux's page cache handles residency — we only
pay for what's touched in the current chunk, not the whole list.

Symmetric with the pre-allocated `all_projections` / coord-array pattern
already used for the scan output: one giant array backed by disk rather
than list-grows.

### Dtype handling

`np.memmap` has no native `bfloat16` dtype. Store bytes as `uint16`
(same width, 2 bytes) and reinterpret via `torch.Tensor.view(torch.bfloat16)`
on read and `.view(torch.uint16)` on write. For `fp16` / `fp32` /
`fp64` the native numpy dtype is used directly.

### File lifecycle

- File created inside a `tempfile.TemporaryDirectory()` scoped to
  `scan_forward`. Auto-removed on return or exception.
- Mode `'w+'`: allocates the file and maps it writable.
- The three small sibling lists (`batch_pos_ids`, `batch_cache_positions`)
  stay as lists — they're `O(N_total × S_max)` int tensors, two orders of
  magnitude smaller than the hidden-state buffer.

### Write path (embedding phase)

For each batch `(start, end)`:
```
hs = embed(ids.to(device)).cpu()   # [B, S, H]
hs_bytes = hs.view(torch.uint16)   # or hs.numpy() for fp16/fp32
mmap[start:end] = hs_bytes.numpy()
```

### Read / write-back path (chunk phase)

```
slice_np = mmap[start:end]                        # no copy
slice_t  = torch.from_numpy(slice_np.copy())      # detach from mmap
hs       = slice_t.view(self.dtype).to(device)
# ... run chunk's layers ...
updated  = hs.cpu().contiguous()
mmap[start:end] = updated.view(torch.uint16).numpy()
```

The `.copy()` in the read step is intentional: the downstream GPU upload
holds a reference, and we don't want that reference to pin memmap pages.

## Scope

Only `ChunkedLocalBackend.scan_forward`. Not `DiskOffloadBackend`:
it already loads layer weights on demand and has a different
memory profile (`batch_hidden_states` is not the dominant pressure
there). The change may be ported to the disk-offload backend later
if benchmarks show it needed.

## Acceptance criteria

- [ ] `batch_hidden_states` is an `np.memmap` of shape
      `[N_total, S_max, hidden_dim]` (or equivalent byte layout), not a
      `list[torch.Tensor]`.
- [ ] Backing file lives under a `TemporaryDirectory` and is cleaned up
      when `scan_forward` returns (normal or exceptional).
- [ ] Multi-chunk scan with `batch_size < N_total` and
      `chunk_size < n_layers` produces identical bases and projections
      to the pre-refactor list implementation on a tiny model.
- [ ] Works for `dtype in {torch.bfloat16, torch.float16, torch.float32}`.
- [ ] No new top-level dependencies.

## Non-goals

- Optimizing memmap access pattern (e.g. prefetching next chunk's slice
  while current chunk computes). The OS page cache already handles the
  common case well; prefetching is a future optimization.
- Applying the same treatment to `DiskOffloadBackend`.
