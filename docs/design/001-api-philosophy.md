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
)
```

Not via method chaining or fit-time arguments. This ensures:
- Reproducibility (probe object fully describes the experiment)
- Serialization (`probe.save()` captures all config)
- No hidden state changes

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
# Level 1: One-liner
probe = LinearProbe(model="meta-llama/Llama-3.1-8B-Instruct")

# Level 2: Common customization
probe = LinearProbe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=[14, 15, 16],
    classifier="logistic_regression",
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
    cache_activations=True,
    activation_dtype=torch.float16,
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
