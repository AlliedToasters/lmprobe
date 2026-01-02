# 001: API Philosophy

**Status**: Accepted  
**Date**: 2026-01-02  
**Author**: Toast  

## Context

`lmprobe` needs an API that serves two audiences:
1. **Researchers** who want quick experiments with minimal boilerplate
2. **Engineers** who need production-grade control and reproducibility

## Decision

### sklearn Compatibility

The primary `LinearProbe` class follows sklearn conventions:

```python
# sklearn pattern
estimator.fit(X, y)
estimator.predict(X)
estimator.predict_proba(X)
estimator.score(X, y)

# lmprobe equivalent
probe.fit(positive_prompts, negative_prompts)  # OR probe.fit(prompts, labels)
probe.predict(prompts)
probe.predict_proba(prompts)
probe.score(prompts, labels)
```

This enables:
- Familiar API for ML practitioners
- Compatibility with sklearn utilities (`cross_val_score`, `GridSearchCV`, pipelines)
- Muscle memory transfers

### Contrastive-First Training

The primary `fit()` signature is contrastive:

```python
probe.fit(positive_prompts, negative_prompts)
```

Rationale:
- Matches Representation Engineering literature (Zou et al., 2023)
- More intuitive for probe training ("these are examples of X, these are not-X")
- Avoids manual label creation

We also support standard sklearn signature for flexibility:

```python
probe.fit(all_prompts, labels)  # labels: list[int] or np.array
```

### Configuration via Constructor

All configuration happens at construction time:

```python
probe = LinearProbe(
    model="...",
    layers=16,
    train_pooling="last_token",
    inference_pooling="last_token",
    classifier="logistic_regression",
    device="auto",
    remote=False,
    random_state=42,
)
```

Not via method chaining or fit-time arguments. This ensures:
- Reproducibility (probe object fully describes the experiment)
- Serialization (`probe.save()` captures all config)
- No hidden state changes

### Remote Execution

The `remote` parameter controls whether nnsight runs model inference locally or on a remote server (e.g., NDIF):

```python
# Local execution (default)
probe = LinearProbe(model="meta-llama/Llama-3.1-8B-Instruct", remote=False)

# Remote execution
probe = LinearProbe(model="meta-llama/Llama-3.1-70B-Instruct", remote=True)
```

**Override at method call**: The `remote` parameter can be overridden on `fit()` and `predict()`:

```python
probe = LinearProbe(model="...", remote=True)  # default to remote

# Train remotely (uses init default)
probe.fit(pos, neg)

# Predict locally (override)
probe.predict(test, remote=False)
```

**Authentication**: Remote execution requires an API key. Set via environment variable:

```bash
export NNSIGHT_API_KEY="your-api-key-here"
```

The library configures nnsight automatically:
```python
from nnsight import CONFIG
CONFIG.API.APIKEY = os.getenv("NNSIGHT_API_KEY")
```

**Error handling**: If `remote=True` and `NNSIGHT_API_KEY` is not set, raise a clear error at the point of remote access (not at init, since user might override to `remote=False`).

### Global Random State

The `random_state` parameter ensures reproducibility across all random operations:

```python
probe = LinearProbe(
    model="...",
    classifier="logistic_regression",
    random_state=42,  # Propagates to classifier and any other random operations
)
```

**Propagation**: `random_state` is passed to:
- Built-in classifiers (e.g., `LogisticRegression(random_state=42)`)
- Any train/test splitting if implemented internally
- Shuffling operations

**Default**: `random_state=None` (non-deterministic). Users who want reproducibility must set it explicitly.

### Activation Caching

**Decision**: Activation caching to disk is **always enabled**. Extracting activations from LLMs (especially remotely) is expensive, so we cache by default to enable rapid iteration on classifier experiments.

```python
probe = LinearProbe(model="meta-llama/Llama-3.1-8B-Instruct")
probe.fit(pos, neg)  # Extracts and caches activations
probe.fit(pos, neg)  # Second call uses cached activations
```

**Cache location**: `~/.cache/lmprobe/` by default, configurable via `LMPROBE_CACHE_DIR` environment variable.

**Cache key**: Hash of (model name, prompts, layers). Different layer selections create different cache entries.

**Cache invalidation**: Users can force re-extraction:
```python
probe.fit(pos, neg, invalidate_cache=True)
```

### Sensible Defaults

A minimal example should work:

```python
probe = LinearProbe()  # Uses reasonable defaults
probe.fit(pos, neg)
probe.predict(test)
```

Defaults:
- `model`: Error — must be specified (no silent default model)
- `layers`: `"middle"` — middle third of layers (where probes often work best)
- `pooling`: `"last_token"` — standard in RepE literature
- `classifier`: `"logistic_regression"`
- `device`: `"auto"` (CUDA if available, else CPU)
- `remote`: `False` — local execution by default
- `random_state`: `None` — non-deterministic by default

### Pooling Parameter Hierarchy

We provide three pooling parameters for progressive complexity:

```python
# Simple: same strategy for train and inference (most users)
probe = LinearProbe(pooling="last_token")

# Advanced: different strategies for train vs inference
probe = LinearProbe(train_pooling="last_token", inference_pooling="max")

# Mixed: set a base, override one
probe = LinearProbe(pooling="last_token", inference_pooling="all")
```

**Collision resolution** (most specific wins):

| `pooling` | `train_pooling` | `inference_pooling` | Result (train / inference) |
|-----------|-----------------|---------------------|----------------------------|
| `"last_token"` | — | — | last_token / last_token |
| `"mean"` | `"last_token"` | — | last_token / mean |
| `"mean"` | — | `"max"` | mean / max |
| `"mean"` | `"last_token"` | `"all"` | last_token / all |
| — | `"last_token"` | `"max"` | last_token / max |

Explicit `train_pooling` / `inference_pooling` always override the base `pooling` value.

### Progressive Disclosure

Simple things are simple, complex things are possible:

```python
# Level 1: One-liner (local execution)
probe = LinearProbe(model="meta-llama/Llama-3.1-8B-Instruct")

# Level 2: Common customization (remote execution for large models)
probe = LinearProbe(
    model="meta-llama/Llama-3.1-70B-Instruct",
    layers=[14, 15, 16],
    classifier="logistic_regression",
    remote=True,
    random_state=42,
)

# Level 3: Full control
from sklearn.svm import SVC
from lmprobe.pooling import WeightedMeanPooling

probe = LinearProbe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=[14, 15, 16],
    train_pooling=WeightedMeanPooling(weights="attention"),
    inference_pooling="all",
    classifier=SVC(kernel="linear", probability=True),
    device="cuda:0",
    remote=False,
    random_state=42,
)
```

## Consequences

- **Good**: Low barrier to entry
- **Good**: Familiar to sklearn users
- **Good**: Reproducible experiments
- **Caution**: Must maintain sklearn compatibility as we add features
- **Caution**: Contrastive `fit(pos, neg)` signature is non-standard — document clearly

## References

- scikit-learn API design: https://scikit-learn.org/stable/developers/develop.html
- Zou et al., "Representation Engineering" (2023) — contrastive training paradigm
