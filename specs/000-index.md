# Specs Index

Feature specifications for `lmprobe`. Each spec describes the design and behavior of a feature before or during implementation.

## Convention

Spec files use this frontmatter:

```yaml
---
status: draft | accepted | implemented
affects: [module names]
---
```

- `draft` — under discussion, not yet committed
- `accepted` — design agreed, not yet fully implemented
- `implemented` — feature is shipped and spec reflects current behavior

---

## Specs

| File | Title | Status | Affects |
|------|-------|--------|---------|
| [001-grouped-projection.md](001-grouped-projection.md) | Grouped projection API (`batch_project_grouped`) | draft | sample_scan, backends |
| [003-per-sample-reduced-projection.md](003-per-sample-reduced-projection.md) | Per-sample reduced projection (`batch_project_reduced`) | draft | sample_scan, backends |
| [004-memmap-batch-hidden-states.md](004-memmap-batch-hidden-states.md) | Memmap-backed `batch_hidden_states` in `ChunkedLocalBackend` | draft | backends |
---

## Design docs

For architectural decisions that shaped the core API, see `docs/design/`:

| File | Topic |
|------|-------|
| [001-api-philosophy.md](../docs/design/001-api-philosophy.md) | Core API design principles |
| [002-pooling-strategies.md](../docs/design/002-pooling-strategies.md) | Train vs inference pooling |
| [003-layer-selection.md](../docs/design/003-layer-selection.md) | Layer indexing conventions |
| [004-classifier-interface.md](../docs/design/004-classifier-interface.md) | Classifier abstraction |
