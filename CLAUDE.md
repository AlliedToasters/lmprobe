# CLAUDE.md - lmprobe Development Guide

## Project Overview

`lmprobe` is a Python library for training linear probes on language model activations. The primary use case is AI safety monitoring — detecting deception, harmful intent, and other safety-relevant properties by analyzing model internals.

## Design Philosophy

- **sklearn-inspired API**: Users familiar with scikit-learn should feel at home. Use `fit()`, `predict()`, `predict_proba()`, `score()`.
- **Contrastive-first**: The primary training paradigm is contrastive (positive vs negative prompts), following the Representation Engineering literature.
- **Sensible defaults, full control**: Simple cases should be one-liners; complex cases should be fully configurable.
- **Separation of concerns**: Activation extraction, pooling, and classification are distinct stages that can be configured independently.

## Key Design Decisions

Detailed design documents live in `docs/design/`. Read these before making changes to core APIs:

| Doc | Topic | Read when... |
|-----|-------|--------------|
| [001-api-philosophy.md](docs/design/001-api-philosophy.md) | Core API design | Changing public interfaces |
| [002-pooling-strategies.md](docs/design/002-pooling-strategies.md) | Train vs inference pooling | Working on activation aggregation |
| [003-layer-selection.md](docs/design/003-layer-selection.md) | Layer indexing conventions | Working on layer extraction |
| [004-classifier-interface.md](docs/design/004-classifier-interface.md) | Classifier abstraction | Adding new classifier types |

## Architecture

```
User Prompts
     │
     ▼
┌─────────────────┐
│ ActivationCache │  ← Extracts & caches activations from LLM
└────────┬────────┘
         │ raw activations: (batch, seq_len, layers, hidden_dim)
         ▼
┌─────────────────┐
│  PoolingStrategy │  ← Aggregates across tokens (train vs inference can differ)
└────────┬────────┘
         │ pooled: (batch, layers, hidden_dim) or (batch, hidden_dim)
         ▼
┌─────────────────┐
│   Classifier    │  ← sklearn-compatible estimator
└────────┬────────┘
         │
         ▼
   Predictions/Probabilities
```

## Code Conventions

- Type hints on all public functions
- Docstrings in NumPy format
- Tests mirror source structure: `src/lmprobe/probe.py` → `tests/test_probe.py`
- Use `ruff` for linting, `black` for formatting

## Quick Reference

```python
from lmprobe import LinearProbe

probe = LinearProbe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,                          # int | list[int] | "all"
    pooling="last_token",               # or override with train_pooling / inference_pooling
    classifier="logistic_regression",   # str | sklearn estimator
    device="auto",
)

probe.fit(positive_prompts, negative_prompts)
predictions = probe.predict(new_prompts)
```

## Common Tasks

### Adding a new pooling strategy
1. Read `docs/design/002-pooling-strategies.md`
2. Add strategy to `src/lmprobe/pooling.py`
3. Register in `POOLING_STRATEGIES` dict
4. Add tests in `tests/test_pooling.py`

### Supporting a new model architecture
1. Check if transformers `AutoModel` handles it automatically
2. If not, add architecture-specific extraction in `src/lmprobe/extraction.py`
3. Document any quirks in `docs/models/`
