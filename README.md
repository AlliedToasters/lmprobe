# `lmprobe` Language Model Probe Library

[![PyPI version](https://badge.fury.io/py/lmprobe.svg)](https://pypi.org/project/lmprobe/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-alliedtoasters.github.io%2Flmprobe-blue)](https://alliedtoasters.github.io/lmprobe)

This library supports the use of language model "activations" or "latents" to build text classifiers. The intent is to help detect and reduce misuse of AI - for example, chemical, biological, radiological and nuclear (CBRN) weapons development, social engineering at scale, and the development of novel cybersecurity attack vectors.

## Linear and Simple Models for LLMs
"Linear Probes" have emerged as an effective and practical way to monitor large language model activity.

### Background

First introduced by [Alain & Bengio (2016)](https://arxiv.org/abs/1610.01644) as "thermometers" for measuring what neural networks learn at each layer, linear probes have since been refined through work on [probe design and selectivity](https://nlp.stanford.edu/~johnhew/interpreting-probes.html) and validated by evidence supporting the [linear representation hypothesis](https://www.neelnanda.io/mechanistic-interpretability/othello). The [Representation Engineering](https://arxiv.org/abs/2310.01405) framework (Zou et al., 2023) demonstrated that probes can monitor safety-relevant properties like honesty and power-seeking. Recent AI safety research has shown promising results: Anthropic's work on [detecting sleeper agents](https://www.anthropic.com/research/probes-catch-sleeper-agents) achieved >99% AUROC using simple linear classifiers, and Apollo Research's [strategic deception detection](https://arxiv.org/abs/2502.03407) work demonstrates that probes trained on simple contrast pairs can generalize to realistic scenarios like insider trading concealment and sandbagging on safety evaluations.

### `lmprobe` Use Cases

The goal of `lmprobe` is to make text classifiers for language models easy to build, experiment on, and deploy during inference. While much of the research has focused on complex emergent risky behavior, the intended use of this library is for simpler use cases such as detection of the misuse of an AI chatbot by humans.

### Compatibility

By default, `lmprobe` uses HuggingFace Transformers to manage models and extract latents during inference. The library also supports `nnsight` for remote execution on [NDIF](https://nnsight.net/) (National Deep Inference Fabric), allowing you to probe large models without local GPU resources.

### Installation

```
pip install lmprobe
```

Optional extras:

```bash
pip install lmprobe[hub]         # HuggingFace Hub integration (push/pull probes)
pip install lmprobe[s3]          # S3 cache backend
pip install lmprobe[nnsight]     # nnsight/NDIF remote execution
pip install lmprobe[plot]        # Layer importance visualization
pip install lmprobe[embeddings]  # Sentence-transformers baselines
pip install lmprobe[auto]        # Automatic layer selection (Group Lasso)
```

### Environment Setup

For remote execution (large models via nnsight/NDIF):

```bash
export NDIF_API_KEY="your-api-key-here"
```

### Example Usage

---

```python
from lmprobe import Probe

positive_prompts = [  # positive class: "dog" without saying "dog"
    "Who wants to go for a walk?",
    "My tail is wagging with delight.",
    "Fetch the ball!",
    "Good boy!",
    "Slobbering, chewing, growling, barking.",
]

negative_prompts = [  # negative class: "cat" without saying "cat"
    "Enjoys lounging in the sun beam all day.",
    "Purring, stalking, pouncing, scratching.",
    "Uses a litterbox, throws sand all over the room.",
    "Tail raised, back arched, eyes alert, whiskers forward.",
]

# Configure the probe
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,                              # int, list[int], or "all"
    pooling="last_token",                   # applies to both train and inference
    classifier="logistic_regression",       # or pass sklearn estimator
    device="auto",
    remote=False,                           # True for nnsight remote execution
    random_state=42,                        # for reproducibility
)

# Fit using contrastive prompts
probe.fit(positive_prompts, negative_prompts)

# Predict on new examples
test_prompts = [
    "Arf! Arf! Let's go outside!",
    "Knocking things off the counter for sport.",
]
predictions = probe.predict(test_prompts)          # [1, 0]
probabilities = probe.predict_proba(test_prompts)  # [[0.12, 0.88], [0.91, 0.09]]

# Evaluate
accuracy = probe.score(test_prompts, [1, 0])

# Save/load for deployment
probe.save("dog_vs_cat_probe.pkl")
loaded_probe = Probe.load("dog_vs_cat_probe.pkl")
```

> **Note:** `LinearProbe` still works as an alias for `Probe`.

---

## Remote Execution for Large Models

Use `remote=True` with `backend="nnsight"` to run inference on large models via nnsight's remote servers:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-70B-Instruct",
    layers="middle",
    backend="nnsight",
    remote=True,  # Requires NDIF_API_KEY
)

probe.fit(positive_prompts, negative_prompts)

# Override remote per-call (e.g., train remote, predict local)
predictions = probe.predict(new_prompts, remote=False)
```

---

## Multi-Layer Probing

When selecting multiple layers, activations are **concatenated** along the hidden dimension:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=[14, 15, 16],  # 3 layers x 4096 dims = 12,288-dim input to classifier
)
```

---

## Layer Sweep

Train an independent probe for each layer to find the most informative layers, without loading all layers into memory at once:

```python
result = Probe.sweep_layers(
    model="meta-llama/Llama-3.1-8B-Instruct",
    positive_prompts=positive_prompts,
    negative_prompts=negative_prompts,
    layers="all",            # or a list of specific layers
    classifier="ridge",
)

# Score all layers
scores = result.score(test_prompts, test_labels)
# {0: 0.52, 1: 0.55, ..., 31: 0.78}

# Find the best layer
best = result.best_layer(test_prompts, test_labels)
print(f"Best layer: {best}")

# Predict with any single layer's probe
preds = result.probes[best].predict(test_prompts)
```

You can also use sweep as a layer spec string:

```python
probe = Probe(model=model, layers="sweep")        # sweep all layers
probe = Probe(model=model, layers="sweep:10")      # sweep every 10th layer
probe = Probe(model=model, layers="sweep:55-65")   # sweep a specific range
```

---

## Advanced: Different Train vs Inference Pooling

For real-time monitoring, train on a stable representation but score every token:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,
    pooling="last_token",          # base strategy
    inference_pooling="all",       # override: return per-token scores
)

probe.fit(positive_prompts, negative_prompts)

# Returns (batch, seq_len) - one score per token
token_scores = probe.predict_proba(["Wagging my tail happily!"])
```

For "flag if ANY token triggers" detection:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,
    pooling="last_token",          # base strategy
    inference_pooling="max",       # override: max score across tokens
)
```

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *required* | HuggingFace model ID or local path |
| `dataset` | `str \| None` | `None` | HuggingFace Dataset repo ID with pre-extracted activations (replaces `model` for extraction) |
| `layers` | `int \| list[int] \| str` | `"middle"` | Which residual stream layers to probe |
| `pooling` | `str \| callable` | `"last_token"` | Token aggregation (train & inference) |
| `train_pooling` | `str \| callable` | — | Override pooling for `fit()` only |
| `inference_pooling` | `str \| callable` | — | Override pooling for `predict()` only |
| `classifier` | `str \| sklearn estimator` | `"logistic_regression"` | Classification model |
| `task` | `str` | `"classification"` | `"classification"` or `"regression"` |
| `device` | `str` | `"auto"` | `"auto"`, `"cuda:0"`, `"cpu"` |
| `remote` | `bool` | `False` | Use nnsight remote execution (requires `NDIF_API_KEY`) |
| `random_state` | `int \| None` | `None` | Random seed for reproducibility (propagates to classifier) |
| `batch_size` | `int` | `8` | Prompts per forward pass during extraction |
| `backend` | `str` | `"local"` | `"local"` (HuggingFace) or `"nnsight"` |
| `dtype` | `str \| None` | `None` | Model dtype: `"float32"`, `"float16"`, `"bfloat16"` |
| `normalize_layers` | `bool \| str` | `True` | Per-layer normalization for multi-layer probes |
| `preprocessing` | `str \| None` | `None` | Pipeline before classifier: `"standard"`, `"pca"`, `"standard+pca"` |
| `pca_components` | `int \| None` | `None` | Number of PCA components |
| `classifier_kwargs` | `dict \| None` | `None` | Extra kwargs for classifier constructor |
| `auto_candidates` | `list[int] \| list[float] \| None` | `None` | Candidate layers for `layers="auto"` (fractional = relative position) |
| `auto_alpha` | `float` | `0.01` | Group Lasso regularization strength for `layers="auto"` |
| `fast_auto_top_k` | `int \| None` | `None` | Number of layers to select with `layers="fast_auto"` |
| `mass_mean_augment` | `bool` | `False` | Augment features with projection onto mass-mean direction |
| `max_retries` | `int \| None` | `None` | Retry attempts with exponential backoff for transient failures |

### Layer Specifications

| Spec | Description |
|------|-------------|
| `16` | Single layer (negative indexing: `-1` = last) |
| `[14, 15, 16]` | Multiple layers (concatenated) |
| `"middle"` | Middle third of layers |
| `"last"` | Last layer |
| `"all"` | All layers |
| `"auto"` | Automatic selection via Group Lasso (requires `pip install lmprobe[auto]`) |
| `"fast_auto"` | Fast selection via coefficient importance |
| `"sweep"` | Train independent probe per layer |
| `"sweep:10"` | Sweep every 10th layer |
| `"sweep:55-65"` | Sweep layers 55 through 65 |

### Pooling Strategies

| Strategy | Training | Inference | Description |
|----------|:--------:|:---------:|-------------|
| `"last_token"` | Y | Y | Final token activation (default, matches RepE literature) |
| `"mean"` | Y | Y | Mean across all tokens |
| `"first_token"` | Y | Y | First token (e.g., `[CLS]`) |
| `"all"` | Y | Y | Each token independently |
| `"max"` | | Y | Max score across tokens (post-probe) |
| `"min"` | | Y | Min score across tokens (post-probe) |

### Pooling Stage Prefixes

Strategies can be prefixed with `score:` (post-probe) or `activation:` (pre-probe) to control *when* pooling happens:

- **Activation pooling** (pre-probe): Reduces activations before classification — the classifier sees one vector per sequence.
- **Score pooling** (post-probe): Classifies every token independently, then reduces the per-token scores.

```python
# Post-probe: classify each token, then average probabilities
probe = Probe(inference_pooling="score:mean")

# Pre-probe: take max activation per dimension, then classify once
probe = Probe(inference_pooling="activation:max")

# Bare names use sensible defaults (backward compatible):
# "mean" → activation:mean, "max" → score:max
```

All base strategies (`last_token`, `first_token`, `mean`, `max`, `min`) can be used with either prefix.

### Pooling Collision Rules

Explicit parameters override the base `pooling` value:

```python
# pooling="mean", train_pooling="last_token" -> train=last_token, inference=mean
# pooling="mean", inference_pooling="max"    -> train=mean, inference=max
```

---

## Classifier Options

`lmprobe` supports several built-in classifiers:

| Classifier | Description |
|------------|-------------|
| `"logistic_regression"` | Standard logistic regression (default) |
| `"logistic_regression_cv"` | Logistic regression with cross-validated regularization |
| `"ridge"` | Ridge classifier - fast, no `predict_proba` |
| `"svm"` | Support Vector Machine with probability calibration |
| `"lda"` | Linear Discriminant Analysis |
| `"mass_mean"` | Mass-Mean Probing - uses direction between class centroids |
| `"sgd"` | Stochastic Gradient Descent classifier |
| `"ensemble"` | Ensemble of LogisticRegression with different regularization strengths |

```python
# Use Mass-Mean Probing (simple but effective)
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=-1,
    classifier="mass_mean",
)

# Pass extra kwargs to the classifier
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=-1,
    classifier="logistic_regression",
    classifier_kwargs={"C": 0.01, "solver": "liblinear", "max_iter": 5000},
)
```

---

## Layer Importance Analysis

Identify which layers are most informative for your task:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers="all",  # Extract all layers
    classifier="ridge",
)

probe.fit(positive_prompts, negative_prompts)

# Compute per-layer importance scores
# Returns np.ndarray of shape (n_layers,), normalized to sum to 1.0
importances = probe.compute_layer_importance(metric="l2")
best_idx = importances.argmax()
print(f"Most important layer: {probe.candidate_layers_[best_idx]}")

# Visualize layer importance (requires: pip install lmprobe[plot])
probe.plot_layer_importance()
```

### Fast Auto Layer Selection

Automatically select the most important layers using fast importance analysis:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers="fast_auto",      # Auto-select best layers
    fast_auto_top_k=3,       # Use top 3 most important layers
    normalize_layers=True,   # Normalize before combining
)

probe.fit(positive_prompts, negative_prompts)
print(f"Selected layers: {probe.selected_layers_}")
```

### Automatic Layer Selection via Group Lasso

Use structured sparsity to let the model choose which layers matter:

```python
# Requires: pip install lmprobe[auto]
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers="auto",
    auto_candidates=[0.25, 0.5, 0.75],  # Fractional positions or explicit indices
    auto_alpha=0.01,                     # Regularization strength
)

probe.fit(positive_prompts, negative_prompts)
print(f"Selected layers: {probe.selected_layers_}")
```

---

## Evaluation

Beyond `score()`, the `evaluate()` method computes multiple metrics at once:

```python
probe.fit(positive_prompts, negative_prompts)

metrics = probe.evaluate(test_prompts, test_labels)
# {"accuracy": 0.85, "f1": 0.85, "precision": 0.88, "recall": 0.82, "auroc": 0.91, ...}
```

---

## HuggingFace Hub Integration

Share trained probes via the HuggingFace Hub. Requires `pip install lmprobe[hub]`.

### Push a probe

```python
probe.fit(positive_prompts, negative_prompts)

url = probe.push_to_hub(
    "username/dog-vs-cat-probe",
    description="Detects dog-like vs cat-like text",
    class_labels={0: "cat", 1: "dog"},
    tags=["safety", "animals"],
    include_training_data=True,   # Include prompts for reproducibility
    private=False,
)
print(url)  # https://huggingface.co/username/dog-vs-cat-probe
```

### Load a probe

```python
from lmprobe import Probe

probe = Probe.from_hub(
    "username/dog-vs-cat-probe",
    trust_classifier=True,   # Required: acknowledge loading serialized model
    load_model=True,         # Download the base LLM for inference
    device="auto",
)
predictions = probe.predict(["Arf! Let's go outside!"])
```

### Inspect probe metadata

```python
from lmprobe import ProbeCard

card = ProbeCard.from_hub("username/dog-vs-cat-probe")
print(card.base_model)       # meta-llama/Llama-3.1-8B-Instruct
print(card.layers)           # [16]
print(card.classifier_type)  # LogisticRegression
print(card.metrics)          # {"accuracy": 0.85}
```

---

## Caching

Activation extraction is expensive, so `lmprobe` caches activations automatically. The cache is stored at `~/.cache/lmprobe/` by default (or set `LMPROBE_CACHE_DIR`).

### Cache configuration

```python
from lmprobe import cache_info, set_cache_backend, set_cache_dtype, set_cache_limit

# Inspect cache
info = cache_info()
print(info)

# Reduce disk usage with float16 caching
set_cache_dtype("float16")

# Set a max cache size (LRU eviction)
set_cache_limit(50)  # GB

# Use S3 for cross-machine cache sharing (requires: pip install lmprobe[s3])
set_cache_backend("s3://my-bucket/lmprobe-cache")
```

### Warmup

Pre-cache activations for a set of prompts before running predictions:

```python
probe.warmup(test_prompts, batch_size=16)

# Subsequent predict/score calls hit the cache
predictions = probe.predict(test_prompts)
```

---

## Activation Datasets

Extract activations once from a large model, share them as a HuggingFace Dataset, and let others train probes without ever loading the model locally. Requires `pip install lmprobe[hub]`.

### Push cached activations to HuggingFace

After extracting activations (via `probe.fit()`, `probe.warmup()`, or any extraction call), push the local cache to a HuggingFace Dataset repo:

```python
from lmprobe import push_dataset

# Activations must already be cached locally for these prompts + model
url = push_dataset(
    repo_id="username/llama-safety-activations",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    prompts=all_prompts,
    labels=all_labels,           # optional, stored in the Parquet index
    description="Safety probe activations for Llama-3.1-8B",
    private=False,
)
print(url)  # https://huggingface.co/datasets/username/llama-safety-activations
```

### Train a probe from a dataset (no model required)

Once activations are on HuggingFace, anyone can train probes without loading the LLM:

```python
from lmprobe import Probe

# No model= needed — activations are pulled from the dataset on demand
probe = Probe(
    dataset="username/llama-safety-activations",
    layers=16,
    classifier="logistic_regression",
)

probe.fit(positive_prompts, negative_prompts)
predictions = probe.predict(test_prompts)
```

Activations are downloaded lazily per prompt and cached locally — repeated calls are fast.

### Pull a full dataset to local cache

Pre-download all shards before running experiments:

```python
from lmprobe import pull_dataset

n = pull_dataset(
    repo_id="username/llama-safety-activations",
    layers=[16],          # only fetch the layers you need
)
print(f"Pulled {n} prompts")
```

### Load raw tensors directly

For custom workflows that need the raw activation tensors:

```python
from lmprobe import load_activation_dataset

tensors, info = load_activation_dataset(
    repo_id="username/llama-safety-activations",
    layers=[16],
)
# tensors["hidden.layer_16"]: shape (n_prompts, hidden_dim)
```

---

## Preprocessing

Apply feature transformations between activation extraction and classification:

```python
# StandardScaler before classification
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=[14, 15, 16],
    preprocessing="standard",
)

# PCA dimensionality reduction
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers="all",
    preprocessing="pca",
    pca_components=50,
)

# Chained: standardize then PCA
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers="all",
    preprocessing="standard+pca",
    pca_components=100,
)
```

---

## Regression

Train probes for continuous targets instead of binary classification:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,
    task="regression",  # Uses Ridge regression by default
)

# fit() accepts labels as second argument (not negative_prompts)
probe.fit(prompts, labels)  # labels: list[float]

predictions = probe.predict(test_prompts)  # continuous values
r_squared = probe.score(test_prompts, test_labels)
```

---

## Working with Pre-Computed Activations

Bypass the extraction pipeline and work directly with activation matrices:

```python
import numpy as np

probe = Probe(classifier="logistic_regression", random_state=42)

# X: (n_samples, hidden_dim), y: (n_samples,)
probe.fit_from_activations(X_train, y_train)
predictions = probe.predict_from_activations(X_test)
accuracy = probe.score_from_activations(X_test, y_test)
```

---

## Baseline Comparisons

Use baselines to validate that your probe is learning something beyond surface features.

### Text-Only Baselines

```python
from lmprobe import BaselineProbe

# Bag-of-words baseline
bow_baseline = BaselineProbe(method="bow", classifier="logistic_regression")
bow_baseline.fit(positive_prompts, negative_prompts)
bow_accuracy = bow_baseline.score(test_prompts, test_labels)

# TF-IDF baseline
tfidf_baseline = BaselineProbe(method="tfidf")
tfidf_baseline.fit(positive_prompts, negative_prompts)

# Sentence length baseline (surprisingly predictive for some tasks)
length_baseline = BaselineProbe(method="sentence_length")
length_baseline.fit(positive_prompts, negative_prompts)

# Sentence-transformers embeddings (requires: pip install lmprobe[embeddings])
st_baseline = BaselineProbe(method="sentence_transformers")
st_baseline.fit(positive_prompts, negative_prompts)

# Random baseline (sanity check - should be ~50%)
random_baseline = BaselineProbe(method="random")

# Majority class baseline
majority_baseline = BaselineProbe(method="majority")
```

### Activation-Based Baselines

Test whether the learned probe direction is special compared to simpler approaches:

```python
from lmprobe import ActivationBaseline

# Random direction baseline - project onto random unit vector
random_dir = ActivationBaseline(
    method="random_direction",
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=-1,
)
random_dir.fit(positive_prompts, negative_prompts)
random_accuracy = random_dir.score(test_prompts, test_labels)

# PCA baseline - classify using top principal components
pca_baseline = ActivationBaseline(
    method="pca",
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=-1,
)

# Layer 0 baseline - use input embeddings instead of deep layers
layer0_baseline = ActivationBaseline(
    method="layer_0",
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=-1,  # Compare layer 0 to this layer
)
```

### Baseline Battery

Run all applicable baselines at once and compare to your probe:

```python
from lmprobe import BaselineBattery

# Text-only baselines (no model required)
battery = BaselineBattery(model=None, random_state=42)
results = battery.fit(positive_prompts, negative_prompts, test_prompts, test_labels)

print(results.summary())
# Baseline Results:
# ------------------------------------------------------------
#   sentence_transformers          0.7925  (fit: 1.23s, predict: 0.05s)
#   tfidf                          0.7547  (fit: 0.01s, predict: 0.00s)
#   bow                            0.6792  (fit: 0.01s, predict: 0.00s)
#   ...

# Get best baseline
best = results.get_best()[0]
print(f"Best baseline: {best.name} with {best.score:.2%} accuracy")

# With activation baselines (requires model)
battery = BaselineBattery(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=-1,
    include=["bow", "tfidf", "random_direction", "pca"],  # Select specific baselines
)
results = battery.fit(positive_prompts, negative_prompts, test_prompts, test_labels)
```

### Available Baseline Methods

| Method | Type | Description |
|--------|------|-------------|
| `bow` | Text | Bag-of-words + classifier |
| `tfidf` | Text | TF-IDF + classifier |
| `random` | Text | Random predictions (sanity check) |
| `majority` | Text | Always predict majority class |
| `sentence_length` | Text | Classify by text length |
| `sentence_transformers` | Text | Pretrained embeddings + classifier |
| `shuffled_labels` | Text | Train on permuted labels (overfitting check) |
| `random_direction` | Activation | Project onto random unit vector |
| `pca` | Activation | Top principal components |
| `layer_0` | Activation | Input embeddings only |
| `perplexity` | Activation | Model's own token probabilities |

---

## Per-Layer Normalization

When combining multiple layers, normalize each layer's activations independently to prevent high-magnitude layers from dominating:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=[14, 15, 16],
    normalize_layers=True,          # Default: per-neuron standardization
    # normalize_layers="per_layer", # Alternative: one mean/std per layer
    # normalize_layers=False,       # Disable normalization
)
```

---

## Probe Ensembles

Combine multiple probes into an ensemble for more robust predictions and uncertainty estimation.

### Basic ensemble

```python
from lmprobe import Probe, ProbeEnsemble

# Combine probes with different classifiers
p1 = Probe(model="meta-llama/Llama-3.1-8B-Instruct", layers=-1, classifier="logistic_regression")
p2 = Probe(model="meta-llama/Llama-3.1-8B-Instruct", layers=-1, classifier="svm")
p3 = Probe(model="meta-llama/Llama-3.1-8B-Instruct", layers=16, classifier="logistic_regression")

ensemble = ProbeEnsemble([p1, p2, p3], voting="soft")
ensemble.fit(positive_prompts, negative_prompts)

predictions = ensemble.predict(test_prompts)           # (n_samples,)
probabilities = ensemble.predict_proba(test_prompts)   # (n_samples, n_classes)
accuracy = ensemble.score(test_prompts, test_labels)
```

### Factory construction

Create ensembles from config dicts sharing a common model:

```python
ensemble = ProbeEnsemble.from_configs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    configs=[
        {"layers": -1, "classifier": "logistic_regression"},
        {"layers": -1, "classifier": "svm"},
        {"layers": 16, "classifier": "ridge"},
    ],
    voting="hard",    # majority vote (required when using Ridge)
    device="auto",    # shared kwargs
)
```

### Bootstrap stability analysis

Clone a single probe into N bootstrap resamples to measure prediction stability:

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

Bootstrap mode supports `sample_weight` and `groups` for group-balanced resampling:

```python
ensemble.fit(
    positive_prompts, negative_prompts,
    sample_weight=weights,    # per-sample importance weights
    groups=group_labels,      # group-balanced bootstrap resampling
)
```

### Save and load

```python
ensemble.save("my_ensemble.pkl")
loaded = ProbeEnsemble.load("my_ensemble.pkl")
```
