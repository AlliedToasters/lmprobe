---
status: draft
affects: [sample_scan, backends]
---

# Spec 001: Grouped projection API

## Motivation

`SampleScan.batch_project(prompts)` fuses all queued prompts into one layer-streaming sweep — each model layer is moved CPU→GPU exactly once (chunked backend). This is the whole performance win of chunked extraction (same pattern as DiskOffloadBackend's `extract_all`; same pattern that took Geometry-of-Truth on DeepSeek V3 600B from an overnight run to 40 min on a single 32GB-VRAM / 128GB-RAM workstation).

**The trap:** callers with multiple logical groups (datasets × splits, train + eval + control corpora) write the naive loop:

```python
for ds_name in datasets:
    for split_name in splits:
        projections, coords, _, seq_lens = scan.batch_project(prompts[ds, split])
        save(ds, split, projections, coords, seq_lens)
```

Each `batch_project` call re-runs the outer layer-chunk loop. **N logical groups ⇒ N full model sweeps**, each streaming every layer CPU→GPU from scratch. On 70B-class models this turns a 40-minute run into a multi-hour one, silently.

The current API gives the caller no indication that this is happening. They have to already know that `batch_project` is "full layer sweep per call" to get the fused version right.

## Goal

Make fusion the *default shape* of the API. Caller physically cannot write the naive loop against the recommended primitive.

Secondary: emit a warning when the legacy shape is used in a way that suggests the caller has fallen into the trap.

## Design

### 1. New primary method: `batch_project_grouped`

```python
def batch_project_grouped(
    self,
    prompt_groups: Mapping[Hashable, list[str]],
    *,
    signal: str | None = None,
    batch_size: int = 4,
) -> dict[Hashable, tuple[np.ndarray, dict[str, list], list[list[int]], list[int]]]:
    """Project multiple labeled groups of prompts in a single layer-streaming
    sweep. Output is keyed by the same keys provided in `prompt_groups`.

    Preferred over repeated `batch_project` calls when multiple groups
    (e.g. dataset splits) need projecting: layer weights are streamed
    CPU→GPU once for the whole union of prompts, rather than once per
    call.

    Parameters
    ----------
    prompt_groups : mapping of hashable key → list[str]
        Named groups of prompts. Keys are caller-defined (strings, tuples,
        whatever hashes). Order of iteration is preserved in the returned
        dict (insertion order on Python 3.7+).
    signal : str or None
        If given, only project this signal.
    batch_size : int
        Microbatch size for the forward pass.

    Returns
    -------
    dict
        ``{key: (projections, coords, token_ids_per_sample, seq_lengths)}``.
        Each value has the same shape/semantics as `batch_project` for
        that group's prompts.
    """
```

**Implementation** (~25 lines in `sample_scan.py`):

1. Preserve input order: `keys = list(prompt_groups.keys())`.
2. Flatten: `all_prompts = sum((prompt_groups[k] for k in keys), [])`; record `bounds = [(key, start, end), ...]` by cumulative length.
3. Single underlying call: `projections, coords, token_ids_per_sample, seq_lengths = self.batch_project(all_prompts, signal=signal, batch_size=batch_size)`.
4. Slice per group. Use `coords["sample_id"]` for the projection rows; use Python slicing for `token_ids_per_sample` and `seq_lengths`:
   ```python
   for key, start, end in bounds:
       group_mask = (coord_sample >= start) & (coord_sample < end)
       group_coords = {c: np.asarray(v)[group_mask] for c, v in coords.items()}
       # Rebase sample_id so each group's output looks like a fresh batch_project:
       group_coords["sample_id"] = group_coords["sample_id"] - start
       out[key] = (
           projections[group_mask],
           {c: v.tolist() for c, v in group_coords.items()},
           token_ids_per_sample[start:end],
           seq_lengths[start:end],
       )
   ```
5. Return `out`.

**Key property:** each per-group tuple is indistinguishable from the tuple `batch_project` would have returned if called on that group's prompts alone. Callers can drop-in replace `{k: scan.batch_project(prompts[k]) for k in keys}` with `scan.batch_project_grouped(prompts)` and get identical per-group outputs — just much faster.

### 2. Warning on repeated `batch_project` calls from the same instance

In `batch_project`, keep a private counter `self._batch_project_call_count`. On the **third** call, emit a `UserWarning`:

```
SampleScan.batch_project called 3+ times on the same scan. Each call
re-streams every model layer CPU→GPU. For multiple prompt groups
(e.g. dataset splits), use scan.batch_project_grouped({...}) — same
semantics, one layer sweep for the whole union.
```

- Rationale for threshold = 3: one call is the normal case. Two is borderline (train/eval). Three+ strongly suggests a loop that should be fused.
- Warning fires at most once per SampleScan instance to avoid spam.
- Respects `warnings.simplefilter("ignore")` via the standard mechanism.

### 3. Docstring updates on `batch_project`

Add a "See also" pointing to `batch_project_grouped` when projecting multiple groups. Add a one-line cost note:

```
Each call performs a full layer-streaming sweep (chunked backend) or
full-dataset layer load (disk_offload backend). For multiple logical
groups, prefer `batch_project_grouped` — one sweep for the whole union.
```

### 4. Example in docs/guides/large-models/

Replace any guide example that loops `batch_project` with the grouped form, so the documented "right way" is the fast way by default.

## Non-goals

- No change to backend internals. Implementation is pure sugar over the existing `batch_project`.
- No change to memory ceilings. If the union of groups exceeds CPU RAM (e.g. 70B × IT-length sequences × all splits), the caller still has to split into multiple fused sweeps — that's a genuine hardware limit, not something the API can hide.
- Not deprecating `batch_project`. Single-group calls are still a valid primary use case (e.g., a single paraphrase set, a single new prompt to visualize).

## Acceptance criteria

1. `scan.batch_project_grouped({k: [prompt]})` returns `{k: (proj, coords, tokens, seq_lens)}` where `(proj, coords, tokens, seq_lens)` is identical, element-wise, to `scan.batch_project([prompt])`.
2. For a dict with N groups, `batch_project_grouped` makes exactly one call to the underlying `backend.scan_forward` (verified by a test using a mock or by patching `batch_project` to count calls).
3. Output coord sample_ids are rebased per group: in `out[k]`, `coord["sample_id"]` ranges from 0 to `len(prompt_groups[k]) - 1`.
4. Third call to `batch_project` on the same instance emits a `UserWarning` containing the string `"batch_project_grouped"`. First and second calls are silent. Warning fires at most once.
5. Existing `batch_project` behavior (signature, semantics, return shape) unchanged. All existing tests pass.
6. New tests cover: single-group grouped call matches `batch_project`; multi-group matches concatenation; order preservation; warning firing threshold.

## Migration

Callers with existing loops over `batch_project` should migrate by:

```python
# Before
results = {}
for key in keys:
    results[key] = scan.batch_project(prompts[key])

# After
results = scan.batch_project_grouped({key: prompts[key] for key in keys})
```

In `liars_bench_whitebox/experiment_pipeline.py::cache_all` this migration replaces the manual collection + slicing with a single call — simplifies the pipeline and makes the optimization self-documenting.
