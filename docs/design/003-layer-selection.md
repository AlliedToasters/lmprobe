# 003: Layer Selection

**Status**: Proposed
**Date**: 2026-01-02
**Author**: Claude (drafted), Toast (review pending)

## Context

Linear probes are trained on activations from specific layers of a language model. Research shows that different layers encode different types of information:
- Early layers: surface-level features (syntax, token identity)
- Middle layers: semantic features (often best for probing)
- Late layers: task-specific features, output distribution

Users need flexible control over which layers to probe, with sensible defaults.

## Decision

### Layer Parameter Specification

The `layers` parameter accepts multiple formats:

```python
# Single layer (int)
layers=16                    # Layer 16

# Multiple specific layers (list)
layers=[14, 15, 16]          # Layers 14, 15, 16

# Negative indexing (from end)
layers=-1                    # Last layer
layers=[-3, -2, -1]          # Last 3 layers

# Named presets (str)
layers="middle"              # Middle third of layers (default)
layers="all"                 # All layers
layers="last"                # Last layer only
```

### Indexing Convention

We use **0-based indexing** matching Python conventions and HuggingFace internals:
- Layer 0 = first transformer block (after embedding)
- Layer -1 = last transformer block (before unembedding)

This matches `model.model.layers[i]` in HuggingFace and nnsight's `.layers[i]`.

### Named Presets

| Preset | Resolution | Rationale |
|--------|------------|-----------|
| `"middle"` | Middle third of layers, centered | Research shows middle layers often best for semantic probing |
| `"last"` | Final layer only | Quick experiments, output-proximal features |
| `"all"` | Every layer | Comprehensive analysis, layer search |

**Resolution example** for a 32-layer model:
- `"middle"` → layers 10-21 (middle third: 32/3 ≈ 11 layers, centered)
- `"all"` → layers 0-31

### Multi-Layer Handling

**Decision**: When multiple layers are selected, activations are **always concatenated** along the hidden dimension. This is the only supported multi-layer strategy.

```python
# layers=[14, 15, 16] on a 4096-dim model
# → activations shape: (batch, seq_len, 4096 * 3) = (batch, seq_len, 12288)
# → after pooling: (batch, 12288)
# → single classifier trained on concatenated representation
```

**Why concatenation is mandatory:**
- Simple and predictable behavior
- Single classifier captures cross-layer patterns
- Matches RepE literature and Anthropic's probe work
- Users who need ensemble approaches can train separate single-layer probes

**High dimensionality note**: Concatenating many layers (e.g., `layers="all"` on a 32-layer, 4096-dim model = 131,072 dimensions) requires regularization. Use `classifier="logistic_regression"` (L2 regularized by default) or add explicit dimensionality reduction via sklearn Pipeline.

### Layer Validation

At `__init__` time (when model is loaded):
1. Resolve named presets to concrete layer indices
2. Validate all indices are within `[0, num_layers)` or valid negative indices
3. Store as normalized positive indices

```python
# Error: layer out of range
LinearProbe(model="gpt2", layers=50)  # GPT-2 has 12 layers → ValueError

# Warning: probing layer 0 (usually not useful)
LinearProbe(model="...", layers=0)  # UserWarning: Layer 0 is immediately post-embedding
```

### Deferred Validation with `remote=True`

When using remote execution via nnsight, the model architecture may not be known at construction time. In this case:
- Named presets (`"middle"`, `"all"`) are resolved at `fit()` time when the model is accessed
- Explicit indices are validated at `fit()` time
- Store the raw `layers` parameter and resolve lazily

```python
# Remote model - validation deferred
probe = LinearProbe(
    model="meta-llama/Llama-3.1-70B-Instruct",
    layers="middle",
    remote=True,  # Model not loaded locally
)
# layers resolved when fit() connects to remote model
```

## Implementation Notes

### Layer Resolution Function

```python
def resolve_layers(
    layers: int | list[int] | str,
    num_layers: int,
) -> list[int]:
    """Convert layer specification to list of positive indices."""
    if isinstance(layers, int):
        return [_normalize_index(layers, num_layers)]
    elif isinstance(layers, list):
        return [_normalize_index(i, num_layers) for i in layers]
    elif layers == "middle":
        third = num_layers // 3
        start = third
        end = num_layers - third
        return list(range(start, end))
    elif layers == "last":
        return [num_layers - 1]
    elif layers == "all":
        return list(range(num_layers))
    else:
        raise ValueError(f"Unknown layer specification: {layers}")

def _normalize_index(idx: int, num_layers: int) -> int:
    """Convert potentially negative index to positive."""
    if idx < 0:
        idx = num_layers + idx
    if not (0 <= idx < num_layers):
        raise ValueError(f"Layer {idx} out of range [0, {num_layers})")
    return idx
```

### Accessing Layers via nnsight

```python
# nnsight pattern for extracting multiple layers
with model.trace(prompts, remote=remote):
    activations = []
    for layer_idx in resolved_layers:
        act = model.model.layers[layer_idx].output[0].save()
        activations.append(act)
```

## Alternatives Considered

### 1-Based Indexing
More intuitive for non-programmers ("layer 1" = first layer), but inconsistent with Python, HuggingFace, and nnsight conventions. Would require constant translation. **Rejected**.

### Slice Syntax (`layers="10:20"`)
More expressive, but adds parsing complexity and isn't needed given our named presets cover common ranges. **Rejected**.

### Per-Layer Classifiers / Ensemble
Train a separate classifier per layer, then ensemble. More interpretable for layer analysis, but adds complexity and loses cross-layer patterns. **Rejected** — users who need this can train multiple single-layer probes manually.

### Mean/Sum Across Layers
Reduces dimensionality but loses layer-specific signal. **Rejected** — concatenation preserves all information.

## Consequences

- **Good**: Flexible layer selection with intuitive defaults
- **Good**: Negative indexing familiar to Python users
- **Good**: Named presets reduce cognitive load
- **Good**: Concatenation is simple and matches literature
- **Caution**: High dimensionality with many layers → may need regularization (L1/L2 in classifier)
- **Caution**: Deferred validation with `remote=True` means errors appear at `fit()` time

## References

- Alain & Bengio (2016) — probing different layers reveals different representations
- Anthropic "Simple Probes" (2024) — uses middle layers (L24-30 of 40)
- nnsight documentation — layer access patterns
