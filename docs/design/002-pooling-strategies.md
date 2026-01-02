# 002: Pooling Strategies

**Status**: Proposed  
**Date**: 2026-01-02  
**Author**: Toast  

## Context

When extracting activations from a language model, we get a tensor of shape `(batch, seq_len, hidden_dim)` for each layer. To train a classifier, we need to reduce this to a fixed-size vector per example.

The key insight is that **training-time pooling and inference-time pooling serve different purposes** and may benefit from different strategies.

## Decision

We provide three pooling parameters with a clear hierarchy:

```python
# Simple: same strategy for train and inference (most common)
probe = LinearProbe(pooling="last_token")

# Advanced: override for specific phases
probe = LinearProbe(
    pooling="last_token",        # base strategy
    inference_pooling="max",     # override for predict()
)

# Explicit: set both independently
probe = LinearProbe(
    train_pooling="last_token",
    inference_pooling="all",
)
```

**Resolution order**: `train_pooling` > `pooling` for training; `inference_pooling` > `pooling` for inference.

**Default**: `"last_token"` — the standard in RepE literature (Zou et al., 2023; Anthropic, 2024).

### Training-Time Pooling (`train_pooling`)

Goal: Extract a **stable, representative** activation for learning the probe direction.

| Strategy | Description | When to use |
|----------|-------------|-------------|
| `"last_token"` | Final token's activation | Default for causal LMs; captures full context |
| `"mean"` | Mean across all tokens | When all positions contribute equally |
| `"first_token"` | First token (e.g., `[CLS]`) | BERT-style models |
| `"eos"` | End-of-sequence token specifically | When EOS carries summary information |
| `"all"` | Each token as separate training example | Data augmentation; position-invariant probes |

#### `"all"` for Training: Token-Level Expansion

Using `train_pooling="all"` treats each token as an independent training example:

```python
# 10 prompts × 100 tokens = 1,000 training examples
probe = LinearProbe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    train_pooling="all",
    inference_pooling="max",
)
```

**Benefits:**
- Data augmentation when prompts are scarce
- Learns position-invariant detection
- Enables token-level supervision (if you have per-token labels)

**Cautions:**
- Tokens within a prompt are correlated → split train/test by *prompt*, not by token
- Early tokens have less context → probe may learn different features across positions
- Class imbalance → if the property only appears at certain positions, most tokens are "negative"
- Computational cost scales with sequence length

### Inference-Time Pooling (`inference_pooling`)

Goal: **Detect the property of interest**, potentially at any point in generation.

| Strategy | Description | When to use |
|----------|-------------|-------------|
| `"last_token"` | Score only the final token | Real-time monitoring of each generated token |
| `"mean"` | Average score across sequence | Overall document classification |
| `"max"` | Maximum score across tokens | "Did deception occur anywhere?" |
| `"min"` | Minimum score across tokens | Conservative detection |
| `"all"` | Return per-token scores | Visualization, detailed analysis |
| `callable` | Custom function `(batch, seq, hidden) → (batch, hidden)` | Advanced use cases |

### Example: Deception Detection

```python
# Train on the final token where the "intent" is crystallized
# At inference, flag if ANY token crosses the threshold
probe = LinearProbe(
    pooling="last_token",
    inference_pooling="max",
)
```

### Example: Per-Token Visualization

```python
probe = LinearProbe(
    pooling="last_token",
    inference_pooling="all",
)

# Returns shape (batch, seq_len) - score per token
token_scores = probe.predict_proba(prompts)  

# Visualize which tokens "trigger" the probe
```

## Implementation Notes

### Separate Pooling Classes

```python
class PoolingStrategy(Protocol):
    def __call__(
        self, 
        activations: torch.Tensor,  # (batch, seq_len, hidden_dim)
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...

# Training poolers return (batch, hidden_dim)
# Inference poolers return (batch, hidden_dim) OR (batch, seq_len, hidden_dim) for "all"
```

### Attention Mask Handling

Pooling strategies must respect attention masks for:
- Padded batches (variable sequence lengths)
- `"mean"` pooling (don't include padding in average)
- `"max"` pooling (don't let padding tokens contribute)

### Validation

At `__init__` time:
- Warn if `train_pooling != inference_pooling` (user should be intentional)
- Error if invalid strategy string
- Validate callable signature if custom pooler provided

## Alternatives Considered

### Only `train_pooling` and `inference_pooling`
More explicit, but forces users to set two params even when they want the same strategy for both. Violates "simple things should be simple."

### Single `pooling` parameter only
Simpler, but loses the flexibility of training on stable representations while detecting at finer granularity. This is a real use case (e.g., train on last token, detect with max across all tokens).

### Pooling as separate transform step
More composable, but adds complexity for the common case. Could revisit for a "power user" API.

## Consequences

- **Good**: Users can train robust probes and deploy flexible detectors
- **Good**: Matches how practitioners actually use probes (RepE trains on last token, but you might want to monitor all tokens)
- **Good**: Simple `pooling` parameter covers most use cases; overrides available when needed
- **Caution**: Three pooling params could confuse new users → clear hierarchy and good docs essential
- **Caution**: `inference_pooling="all"` changes output shape → must document clearly

## References

- Zou et al., "Representation Engineering" (2023) — uses last token pooling
- Anthropic, "Simple Probes Can Catch Sleeper Agents" (2024) — probes middle layers
