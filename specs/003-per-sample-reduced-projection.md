---
status: draft
affects: [sample_scan, backends]
---

# Spec 003: Per-sample reduced projection output

## Motivation

A `SampleScan` PCA basis compresses hidden-state deltas from `H` dims to
`k` dims — typically 5120→64, an 80× ratio. That compression is the whole
point: a PCA projection is meant to be a **compact, searchable summary**
of the forward pass at that (layer, signal). It's what makes latent-scan
a data product rather than a debugging trace.

But today, `batch_project` / `batch_project_grouped` / `scan_forward`
emit projections at **per-token** granularity: an
`[N_samples × max_seq_len × n_layers × n_signals, k]` row-block plus
per-row coordinate metadata. For a 1067-sample fused sweep at `S=3500` on
a 40-layer 2-signal model, that's ~17 GB of per-token projection data
generated **per chunk**, accumulated across chunks, and concatenated at
the end. Spec 002 already removed the pre-projection `[N, S, H]`
captures; this spec removes the post-projection `[N, S, k]` rubble that
remains.

The caller (e.g. `experiment_pipeline.py::cache_all`) doesn't actually
*use* per-token data. It reduces every microbatch of per-token
projections to three per-sample summaries immediately:

- **last_token** — projection at the last assistant token.
- **mean_asst** — mean over assistant tokens.
- **mean_excl5** — mean over assistant tokens, excluding the last 5.

The final cached data product is
`[N_samples, n_layers, n_signals, k]` per reducer. For the same
workload that's ~10 MB total, **independent of sequence length and
chunk count**.

So the current pipeline inflates transient CPU RAM by ~1700× versus what
semantically needs to exist. On Mistral-24B × 1067 samples this inflation
has OOM-killed three separate attempts at 120+ GB CPU RAM, with the
failure tracking whichever buffer happens to be the largest at the moment
(pre-computed masks → `per_layer_captures` → `per_layer_projected` →
`all_proj_chunks`). Each patch unblocks the next bottleneck by a factor
of 2–5×, but the ceiling stays proportional to per-token data. This
pattern doesn't scale: 70B with 80 layers would hit the ceiling
immediately, 600B even harder.

The fix that honors the design is: **reduce inside the scan, never
emit per-token projections across chunks.**

## Design philosophy

A PCA-projected latent scan is a compact, searchable data product over
the *entire* forward pass — **breadth over depth.** It summarizes what's
happening at every layer for every sample at per-sample granularity, on
a space small enough to fit in RAM, cache to disk trivially, and compare
across inputs without I/O overhead.

Two invariants deliver that shape:

1. **Full-hydration latents are transient.** Per-microbatch, per-layer
   `[B, S, H]` activations exist only long enough to project to
   `[B, S, k]`, apply all reducers, and free. They never cross a chunk
   boundary, never accumulate, never hit disk.
2. **Only the compressed summary persists.** Cross-chunk accumulators
   hold exclusively per-sample `[N, L, n_sig, k]` reducer outputs —
   small enough that disk caching is trivial and in-memory search is
   instant.

Anything held between chunks that exceeds
`O(N × L × n_sig × k) + O(residuals needed for next chunk)` is waste.

## Goal

A new method on `SampleScan`:

```python
scan.batch_project_reduced(
    prompt_groups: Mapping[Hashable, list[str]],
    *,
    reducers: Mapping[str, Reducer],
    batch_size: int = 4,
) -> dict[Hashable, dict[str, np.ndarray]]
```

Returns, for each input group key, a dict of reducer-name →
`[N_group, n_layers, n_signals, k]` array. Per-token projections never
materialize beyond one microbatch × one layer × one signal (`[B, S, k]`
transient).

## Reducer interface

```python
class Reducer(Protocol):
    """Reduce per-token projections to per-sample vectors.

    Applied per (layer, signal) on each microbatch as projections are
    produced inside the chunk loop. Owns its own per-sample masks
    (passed at construction) and running state (e.g. token counts).
    """

    def init_state(
        self,
        n_samples: int,
        n_layers: int,
        n_signals: int,
        k: int,
    ) -> Any:
        """Allocate accumulator + bookkeeping. Returned object is passed
        to update/finalize. Typically `{"out": np.zeros([N,L,S,k], f32),
        "count": np.zeros(N, int)}` or similar."""

    def update(
        self,
        state: Any,
        proj: np.ndarray,         # [B, S_pad, k] fp16 — microbatch projection
        sample_ids: Sequence[int],  # [B] — sample indices in the N axis
        layer_idx: int,
        sig_idx: int,
    ) -> None:
        """Update accumulator with one microbatch's contribution. Called
        once per (layer, signal) per microbatch — i.e. very often. Must
        be cheap."""

    def finalize(self, state: Any) -> np.ndarray:
        """Post-process (e.g. divide running sums by counts) and return
        the [N, L, n_sig, k] output."""
```

### Built-in reducers

Three, covering `experiment_pipeline.cache_all`'s needs:

- **`LastTokenReducer(masks_per_sample)`** — selects projection at the
  last True-in-mask token per sample. Mask is a list of `[seq_len_i]`
  bool arrays; lmprobe pads to each microbatch's padded length.
- **`MeanReducer(masks_per_sample)`** — mean over True-in-mask tokens
  per sample. Accumulates sum in fp32, tracks per-sample True-count,
  divides in `finalize`.
- **`MeanExclLastNReducer(masks_per_sample, n=5)`** — mean over
  True-in-mask tokens excluding the last `n` True positions per sample.
  Falls back to `MeanReducer` semantics if a sample has ≤ `n` True
  tokens. Pre-computes per-sample "excluded positions" from the mask;
  behaves like `MeanReducer` over a derived mask at update time.

Custom reducers (any `Reducer` implementation) are accepted.

### Mask convention

`masks_per_sample: list[np.ndarray]` — one bool array per sample, shape
`[seq_len_i]`, where `True` marks positions of interest (e.g. assistant
tokens). Length `seq_len_i` can be less than the tokenizer's padded
length; lmprobe extends with `False` for padding positions. Masks are
carried inside the reducer — `batch_project_reduced` doesn't need a
separate mask argument.

## Implementation plan

### 1. Module layout

New file `src/lmprobe/reducers.py` with `Reducer` protocol +
`LastTokenReducer`, `MeanReducer`, `MeanExclLastNReducer`.

### 2. Backend threading

`ChunkedLocalBackend.scan_forward` and
`DiskOffloadBackend.scan_forward` gain an optional parameter:

```python
reducers: Mapping[str, ReducerBound] | None = None
```

where `ReducerBound` is the reducer paired with its already-initialized
state. When non-`None`:

- Skip the `per_layer_captures` / `per_layer_projected` accumulators and
  the per-chunk assembly loop.
- In the per-layer, per-signal stream-project block (spec 002), replace
  the "append to per_layer_projected" step with "dispatch each microbatch's
  projection through every reducer's `update`":
  ```python
  proj_cpu = _stream_project(delta, sig_name, layer_idx)  # [B, S, k] fp16
  for name, (reducer, state) in reducers.items():
      reducer.update(
          state, proj_cpu,
          sample_ids=np.arange(start, end),
          layer_idx=layer_idx, sig_idx=sig_idx,
      )
  del proj_cpu
  ```
- Return an empty projection array and coord dict; reducers own the
  output.

### 3. `SampleScan` API

```python
def batch_project_reduced(
    self,
    prompt_groups: Mapping[Hashable, list[str]],
    *,
    reducers: Mapping[str, Reducer],
    batch_size: int = 4,
) -> dict[Hashable, dict[str, np.ndarray]]:
    """Project groups through the scan basis and reduce per-sample
    in-chunk.

    Fused layer-streaming sweep over the union of all prompts (same
    pattern as :meth:`batch_project_grouped`). Per-token projections
    are applied to each reducer as microbatches complete, then freed —
    they never accumulate across chunks. Returns per-group,
    per-reducer [N, n_layers, n_signals, k] arrays.

    See spec 003 in the repository for the design philosophy and memory
    accounting.
    """
```

Internally:

1. Flatten `prompt_groups` into one `all_prompts` list, record per-group
   `(start, end)` bounds (as in spec 001).
2. For each reducer, build a **global** state covering all prompts
   (N = total). The reducer's internal masks must align to this flat
   ordering; the caller may either pass masks per-group or lmprobe may
   offer a thin adapter that flattens them.
   - Cleaner option: accept `reducers` **already carrying flat
     masks** matching the union ordering. Caller prepares them. Keeps
     lmprobe reducer-agnostic.
3. Call `scan_forward(all_prompts, reducers=bound_reducers, external_bases=self._bases)`.
4. `finalize` each reducer; slice per-group along the N axis:
   ```python
   out = {k: {} for k in keys}
   for name, (reducer, state) in bound_reducers.items():
       full = reducer.finalize(state)  # [N_total, L, n_sig, k]
       for key, start, end in bounds:
           out[key][name] = full[start:end]
   ```

### 4. Helper for masks

`experiment_pipeline.cache_all` already has per-group `boundaries_groups`
(lists of assistant-turn boundary tuples per sample). A small helper
inside `batch_project_reduced` — or exposed separately — constructs bool
masks from boundaries:

```python
def asst_mask_from_boundaries(boundaries: list[tuple[int,int]], seq_len: int) -> np.ndarray:
    mask = np.zeros(seq_len, dtype=bool)
    for s, e in boundaries:
        mask[s:min(e, seq_len)] = True
    return mask
```

Not strictly part of this spec, but recommended as a utility.

## Memory accounting

Working set, per chunk:

| item                                  | size (Mistral, N=1067, S=3500, L=40, n_sig=2, k=64) |
|---------------------------------------|:---------------------------------------------------:|
| Residuals (`batch_hidden_states`)     | 38 GB    (unchanged — needed by next chunk)         |
| Per-microbatch projection (transient) | 448 KB   (freed after reducer dispatch)             |
| Reducer accumulators (3 reducers)     | 3 × N×L×n_sig×k × 4B ≈ **66 MB**                    |
| Reducer bookkeeping (counts)          | 3 × N × 4B ≈ 12 KB                                  |

Total cross-chunk steady state: **~38 GB** (all residuals). Down from
the ~120 GB observed under the per-token output path.
**Independent of `n_chunks` and `n_layers`**.

70B sanity check (80 layers, N=1000, S=3000, k=64):
- Residuals ≈ 31 GB (or per-call-dependent).
- Accumulators ≈ 3 × 1000 × 80 × 2 × 64 × 4B ≈ 120 MB.
- Fits well inside 128 GB.

This is the whole point: reducer output size is a constant function of
`(n_prompts, n_layers, n_signals, k, n_reducers)`. It has no
sequence-length or chunk-count factor.

## Non-goals

- **Not deprecating `batch_project` / `batch_project_grouped`**.
  Per-token output remains the right primitive for small workloads
  (paraphrase sets, single-prompt visualizations, token-level
  separability plots) where `N × S` is small.
- **No built-in disk persistence**. The output is standard numpy
  arrays; the caller persists them (`experiment_pipeline.py::cache_all`
  already has this plumbing).
- **No change to PCA-fit path** (`SampleScan.run`). Fitting runs once
  per model and doesn't hit the per-token OOM pattern — the fit already
  consumes its own flat tensor and discards.
- **Residual buffer** (`batch_hidden_states`) stays in CPU RAM — it's
  genuinely needed for the next chunk's forward. Memmap-backing it is a
  separate concern (spec 004 if ever motivated).

## Acceptance criteria

1. **Numerical parity.** For `stas/tiny-random-llama-2`, for each
   built-in reducer:
   - Run `scan.batch_project_reduced` on a small group.
   - Run `scan.batch_project`, then apply the reducer's Python logic
     caller-side to the returned per-token projections.
   - The two outputs match within `atol=1e-2` in fp16 space.

2. **Working-set bound.** Under `batch_project_reduced`,
   `per_layer_projected`, `per_layer_captures`, and `all_proj_chunks`
   stay empty / un-appended throughout the sweep. Verified by
   monkey-patching the backend in a test and asserting these buckets
   aren't touched when reducers are provided.

3. **One backend call for `N` groups.** Same fusion property as
   `batch_project_grouped` — exactly one `scan_forward` invocation per
   `batch_project_reduced` call regardless of group count.

4. **No regressions.** All existing scan, backend, and spec-001/spec-002
   tests pass.

## Risks

- **Reducer statefulness across chunks.** Reducer accumulators must
  survive chunk transitions. Because `scan_forward` owns a single
  reducer state dict for the whole sweep, this is natural — but the
  spec must make clear reducers are not chunk-local.

- **Mask alignment.** Caller-supplied masks must match the tokenization
  lmprobe produces. If the tokenizer drifts (different template, different
  special-token handling), reducers misalign silently. Mitigated by
  requiring masks to be derived from the same tokenizer run (same chat
  template, same tokenizer instance) lmprobe uses — we document this,
  and `experiment_pipeline.cache_all` is already set up this way via
  `_load_tokenizer`.

- **Reducer numerical choice**. `MeanReducer` accumulates in fp32 to
  avoid drift over long mean-pools. Custom reducers are the caller's
  responsibility.

- **API surface creep.** Adding a third projection method (alongside
  `batch_project` and `batch_project_grouped`) risks discoverability
  confusion. Mitigation: point `batch_project_grouped`'s docstring to
  `batch_project_reduced` for aggregation workloads, and emit a warning
  from `batch_project` (already there after spec 001) and
  `batch_project_grouped` when called on very large `N × S` sweeps with
  obvious aggregation patterns. Out of scope to auto-detect; just make
  the right primitive easy to find.

## Downstream migration

`liars_bench_whitebox/experiment_pipeline.py::cache_all` today:

```
cache_all → _fused_sweep_and_cache → scan.batch_project_grouped(...)
  → _aggregate_group_projections(projections, coords, boundaries, ...)
  → write three per-sample reducer outputs to .npz
```

After this spec:

```
cache_all → build three Reducer instances per group with flat
  assistant-boundary masks → scan.batch_project_reduced(
      prompt_groups, reducers={
          "last_token": LastTokenReducer(flat_masks),
          "mean_asst":  MeanReducer(flat_masks),
          "mean_excl5": MeanExclLastNReducer(flat_masks, n=5),
      })
  → write directly to .npz (shapes already match cache format)
```

`_aggregate_group_projections` goes away. The cached
`last_token.npz` / `mean_asst.npz` / `mean_excl5.npz` bytes are
bit-equivalent to today's (within fp16 accumulation tolerance of the
parity test).
