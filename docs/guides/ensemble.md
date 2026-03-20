# Probe Ensembles

Combine multiple probes into an ensemble for more robust predictions and uncertainty estimation. Ensembles are especially useful when you're unsure which classifier or layer gives the best signal.

---

## When to use ensembles

- **Robustness**: Averaging across classifiers or layers smooths out individual probe weaknesses
- **Uncertainty estimation**: Bootstrap ensembles quantify how stable your predictions are
- **Diverse perspectives**: Combine probes at different layers to capture both shallow and deep representations

---

## Basic ensemble

Create probes with different configurations and combine them:

```python
from lmprobe import Probe, ProbeEnsemble

p1 = Probe(model="meta-llama/Llama-3.1-8B-Instruct", layers=-1, classifier="logistic_regression")
p2 = Probe(model="meta-llama/Llama-3.1-8B-Instruct", layers=-1, classifier="svm")
p3 = Probe(model="meta-llama/Llama-3.1-8B-Instruct", layers=16, classifier="logistic_regression")

ensemble = ProbeEnsemble([p1, p2, p3], voting="soft")
ensemble.fit(positive_prompts, negative_prompts)

predictions = ensemble.predict(test_prompts)           # (n_samples,)
probabilities = ensemble.predict_proba(test_prompts)   # (n_samples, n_classes)
accuracy = ensemble.score(test_prompts, test_labels)
```

---

## Voting strategies

| Strategy | How it works | When to use |
|----------|-------------|-------------|
| `"soft"` (default) | Average class probabilities across probes, then argmax | When all classifiers support `predict_proba()` |
| `"hard"` | Majority vote on predicted labels | When using classifiers like Ridge that lack `predict_proba()` |

```python
# Soft voting (default) — requires predict_proba support
ensemble = ProbeEnsemble([p1, p2], voting="soft")

# Hard voting — works with any classifier
ensemble = ProbeEnsemble([p1, p2, p3], voting="hard")
```

---

## Weighted voting

Give more weight to probes you trust more:

```python
ensemble = ProbeEnsemble(
    [p1, p2, p3],
    weights=[2.0, 1.0, 1.0],  # p1 gets double weight
    voting="soft",
)
```

Weights are normalized to sum to 1 internally.

---

## Factory construction

Create ensembles from config dicts sharing a common model. This avoids repeating the model name and shared parameters:

```python
ensemble = ProbeEnsemble.from_configs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    configs=[
        {"layers": -1, "classifier": "logistic_regression"},
        {"layers": -1, "classifier": "svm"},
        {"layers": 16, "classifier": "ridge"},
    ],
    voting="hard",    # required when using Ridge
    device="auto",    # shared kwargs
)
```

---

## Bootstrap stability analysis

Clone a single probe into N bootstrap resamples to measure how stable predictions are across different training subsets:

```python
base_probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=-1,
    classifier="logistic_regression",
)

ensemble = ProbeEnsemble.bootstrap(base_probe, n_resamples=10, random_state=42)
ensemble.fit(positive_prompts, negative_prompts)

# Per-sample uncertainty: high std = ensemble members disagree
uncertainty = ensemble.prediction_std(test_prompts)  # (n_samples,)
```

`prediction_std()` returns the standard deviation of the positive-class probability across ensemble members. High values indicate samples where the probe is unreliable — useful for flagging borderline cases or identifying data distribution shifts.

### Interpreting uncertainty

| `prediction_std` | Interpretation |
|-------------------|---------------|
| < 0.05 | High confidence — members agree |
| 0.05 – 0.15 | Moderate uncertainty |
| > 0.15 | Low confidence — consider manual review |

These thresholds are approximate and task-dependent. Calibrate on your own data.

---

## Group-balanced bootstrap

When your training data has groups (e.g., different prompt sources or categories), ensure each bootstrap resample draws evenly from all groups:

```python
ensemble = ProbeEnsemble.bootstrap(base_probe, n_resamples=10, random_state=42)
ensemble.fit(
    positive_prompts, negative_prompts,
    groups=group_labels,      # group-balanced resampling
    sample_weight=weights,    # optional per-sample importance weights
)
```

Each ensemble member draws `ceil(total / n_groups)` samples from each group. Without `groups`, standard bootstrap resampling can under-represent minority groups.

!!! note
    `groups` is only used in bootstrap mode. Passing it to a non-bootstrap ensemble emits a warning.

---

## Save and load

```python
ensemble.save("my_ensemble.pkl")
loaded = ProbeEnsemble.load("my_ensemble.pkl")
predictions = loaded.predict(test_prompts)
```

---

## Performance notes

- **Activation extraction happens once.** The ensemble performs a single warmup pass covering all layers needed by any member probe. Individual `fit()` and `predict()` calls hit the cache.
- **Cost scales with number of probes**, not model forward passes. Adding more ensemble members is cheap if they share cached activations.
- **Bootstrap ensembles are particularly efficient** since all members use the same model, layers, and pooling — only the training data subset differs.
