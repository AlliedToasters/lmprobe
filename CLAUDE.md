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

## Package Structure

```
lmprobe/
├── src/
│   └── lmprobe/
│       ├── __init__.py
│       ├── probe.py          # LinearProbe main class
│       ├── extraction.py     # Activation extraction via nnsight
│       ├── pooling.py        # Pooling strategies
│       ├── cache.py          # Activation caching
│       └── classifiers.py    # Built-in classifier factory
├── tests/
│   ├── conftest.py           # Shared fixtures (tiny model)
│   ├── test_readme_example.py # NORTH STAR: README example must pass
│   ├── test_probe.py
│   ├── test_extraction.py
│   ├── test_pooling.py
│   └── test_cache.py
├── docs/
│   └── design/               # Design decision documents
├── pyproject.toml
└── CLAUDE.md
```

## Critical Design Decisions

These decisions are **mandatory** and must not be changed without explicit discussion:

| Decision | Value | Rationale |
|----------|-------|-----------|
| Multi-layer handling | Always concatenate | Simple, captures cross-layer patterns |
| Activation caching | Always enabled | Remote/LLM inference is expensive |
| Package layout | `src/lmprobe/` | Standard Python packaging |
| nnsight for extraction | Required dependency | Supports remote execution |
| API key | `NNSIGHT_API_KEY` env var | Standard credential handling |
| Cache location | `~/.cache/lmprobe/` (or `LMPROBE_CACHE_DIR`) | XDG-style default |

## Code Conventions

- Type hints on all public functions
- Docstrings in NumPy format
- Tests mirror source structure: `src/lmprobe/probe.py` → `tests/test_probe.py`
- Use `ruff` for linting, `black` for formatting

## Testing

**All tests must use a real language model.** Use `stas/tiny-random-llama-2` — a tiny Llama model with random weights designed for functional testing.

```python
# tests/conftest.py
import pytest

TEST_MODEL = "stas/tiny-random-llama-2"

@pytest.fixture
def tiny_model():
    """Tiny random Llama model for testing."""
    return TEST_MODEL
```

**Test requirements:**
- Tests must run without GPU (CPU-only)
- Tests must not require `NNSIGHT_API_KEY` (use `remote=False`)
- Tests should be fast (tiny model has ~few MB weights)
- Integration tests verify full pipeline: extraction → pooling → classification

### Remote/NDIF Testing (TODO)

**Status: NOT YET TESTED**

The `remote=True` functionality uses nnsight to connect to NDIF (National Deep Inference Fabric), a US national research initiative. Remote testing has not been performed due to:

1. **Geographic restriction**: NDIF restricts access to US-based users only
2. **API key requirement**: Requires `NNSIGHT_API_KEY` environment variable

**What needs testing:**
- `LinearProbe(..., remote=True)` connects successfully
- `probe.fit(..., remote=True)` extracts activations from remote models
- `probe.predict(..., remote=False)` override works (train remote, predict local)
- Large models (e.g., `meta-llama/Llama-3.1-70B-Instruct`) work via remote
- Error handling when `NNSIGHT_API_KEY` is missing/invalid
- Cache behavior with remote extractions

**To test when US-based:**
```bash
export NNSIGHT_API_KEY="your-key"
pytest tests/test_remote.py -v  # (test file to be created)
```

**Known considerations:**
- Remote execution may have different tensor handling (proxies vs direct tensors)
- The `extraction.py` code handles both cases with `hasattr(act, "value")` check
- Network latency may affect batch processing strategies

```python
# Example test
def test_fit_predict_roundtrip(tiny_model):
    probe = LinearProbe(
        model=tiny_model,
        layers=-1,
        remote=False,
        random_state=42,
    )
    probe.fit(["positive example"], ["negative example"])
    predictions = probe.predict(["test input"])
    assert predictions.shape == (1,)
```

## Test-Driven Development

**This project uses test-driven development (TDD).** Write tests BEFORE implementation.

### The North Star Test

The **north star test** is `tests/test_readme_example.py`. It runs the exact code from README.md's "Example Usage" section. This test defines what "done" looks like:

```python
# tests/test_readme_example.py
"""
North Star Test: The README example must run exactly as documented.

This test runs the exact code from README.md. If this test passes,
the library's public API is working as advertised.
"""

def test_readme_example_runs(tiny_model):
    """The README example code runs without error."""
    from lmprobe import LinearProbe

    positive_prompts = [
        "Who wants to go for a walk?",
        "My tail is wagging with delight.",
        "Fetch the ball!",
        "Good boy!",
        "Slobbering, chewing, growling, barking.",
    ]

    negative_prompts = [
        "Enjoys lounging in the sun beam all day.",
        "Purring, stalking, pouncing, scratching.",
        "Uses a litterbox, throws sand all over the room.",
        "Tail raised, back arched, eyes alert, whiskers forward.",
    ]

    probe = LinearProbe(
        model=tiny_model,  # Use tiny model instead of Llama for tests
        layers=-1,         # Last layer (tiny model has few layers)
        pooling="last_token",
        classifier="logistic_regression",
        device="cpu",
        remote=False,
        random_state=42,
    )

    probe.fit(positive_prompts, negative_prompts)

    test_prompts = [
        "Arf! Arf! Let's go outside!",
        "Knocking things off the counter for sport.",
    ]
    predictions = probe.predict(test_prompts)
    probabilities = probe.predict_proba(test_prompts)

    # Shape assertions (values may vary with random weights)
    assert predictions.shape == (2,)
    assert probabilities.shape == (2, 2)

    # Score method works
    accuracy = probe.score(test_prompts, [1, 0])
    assert 0.0 <= accuracy <= 1.0
```

### TDD Workflow

1. **Write a failing test first** — Define expected behavior before implementation
2. **Run the test, confirm it fails** — Ensures the test is actually testing something
3. **Implement minimal code to pass** — Don't over-engineer
4. **Refactor if needed** — Clean up while tests are green
5. **Repeat**

### Test Priority Order

When implementing, make tests pass in this order:

1. `test_readme_example.py` — The north star (full integration)
2. `test_probe.py` — LinearProbe unit tests
3. `test_extraction.py` — Activation extraction tests
4. `test_pooling.py` — Pooling strategy tests
5. `test_cache.py` — Caching tests

### Running Tests

```bash
# Run all tests
pytest

# Run north star test only
pytest tests/test_readme_example.py -v

# Run with coverage
pytest --cov=lmprobe
```

## Quick Reference

```python
from lmprobe import LinearProbe

probe = LinearProbe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,                          # int | list[int] | "all" | "middle"
    pooling="last_token",               # or override with train_pooling / inference_pooling
    classifier="logistic_regression",   # str | sklearn estimator
    device="auto",
    remote=False,                       # True for nnsight remote execution
    random_state=42,                    # Propagates to classifier for reproducibility
)

probe.fit(positive_prompts, negative_prompts)
predictions = probe.predict(new_prompts)

# Override remote at call time
predictions = probe.predict(new_prompts, remote=True)
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

## Future Work: Additional Baselines

The `BaselineProbe` class currently supports `bow`, `tfidf`, `random`, and `majority` methods.
Future baselines to consider (see GitHub issue for details):

### Surface-level baselines
- **Sentence length** — surprisingly predictive for some tasks
- **Perplexity/logprob** — model's own token probabilities; critical for truthfulness tasks (Marks & Tegmark showed probable ≠ true)

### Activation baselines
- **Random direction** — project activations onto random unit vector and classify; tests whether any direction works or learned direction is special
- **PCA top-k** — classify using projection onto top principal components; tests if signal is in obvious variance or requires learning
- **Layer 0 (embeddings)** — if input embeddings work as well as middle layers, probe might just be recovering token identity

### External embedding baselines
- **Sentence-transformers** — off-the-shelf semantic embeddings + logistic regression; tests whether finding something model-specific or just general semantics

### Sanity checks
- **Shuffled labels** — train probe on permuted labels, should get ~50%; validates probe isn't overfitting to noise
