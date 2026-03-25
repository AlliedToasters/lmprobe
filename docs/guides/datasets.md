# HuggingFace Activation Datasets

Activation extraction is the expensive part of the probe workflow — a single forward pass through a large model can take seconds, and you may need thousands of prompts. Activation datasets let you extract once, publish to HuggingFace, and let anyone train probes without loading the model locally.

Requires `pip install lmprobe[hub]`.

---

## Overview

The workflow has three stages:

```
1. Extract  →  2. Publish  →  3. Train (no model needed)

UnifiedCache       push_dataset       load_activations
   locally          to HF Hub        + fit_from_activations
```

---

## Stage 1: Extract activations efficiently

Use `UnifiedCache` to extract activations in a single forward pass. The key option is `cache_pooled=True` (the default), which pools before caching and reduces disk usage by roughly **100×** — storing `(hidden_dim,)` per prompt instead of `(seq_len, hidden_dim)`.

```python
from lmprobe import UnifiedCache

cache = UnifiedCache(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers="all",            # extract every layer
    cache_pooled=True,       # ~100x disk savings (default)
    pooling="last_token",    # pooling strategy — must match your probe later
    compute_perplexity=True, # also cache perplexity features (cheap, often useful)
    device="auto",
    batch_size=8,
)

stats = cache.warmup(all_prompts)
print(stats)
# WarmupStats(total=500, activations=0 cached + 500 extracted,
#             perplexity=0 cached + 500 extracted, time=142.3s)
```

!!! note "Committing to a pooling strategy"
    When `cache_pooled=True`, pooling is applied before saving. Once cached, the pooling strategy is fixed — you can't re-pool with a different strategy without re-extracting. Choose carefully.
    Set `cache_pooled=False` only if you need to experiment with multiple pooling strategies on the same data.

### Caching logits

Optionally cache top-k logits for downstream analysis:

```python
cache = UnifiedCache(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers="all",
    cache_logits=True,
    logit_top_k=50,          # store top 50 token probabilities per position
    logit_positions="last",  # "last" or "all" token positions
)
```

### Remote extraction for large models

Combine `UnifiedCache` with nnsight for models too large to run locally:

```python
cache = UnifiedCache(
    model="meta-llama/Llama-3.1-70B-Instruct",
    layers="all",
    backend="nnsight",
    remote=True,             # requires NDIF_API_KEY, US-based access
    batch_size=4,            # smaller batches for remote
)
stats = cache.warmup(all_prompts)
```

---

## Stage 2: Push to HuggingFace

Once activations are in the local cache, push them to a HuggingFace Dataset repo. The dataset stores a Parquet index (prompt metadata) and safetensors shards (tensor data).

```python
from lmprobe import push_dataset

url = push_dataset(
    repo_id="username/llama-8b-safety-activations",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    prompts=all_prompts,
    labels=all_labels,                      # optional: stored in Parquet index
    description="Activations for safety probe training on Llama-3.1-8B",
    private=False,
)
print(url)
# https://huggingface.co/datasets/username/llama-8b-safety-activations
```

### Including metadata per prompt

Attach arbitrary metadata to each prompt — it lands as columns in the Parquet index, queryable via `load_dataset()`:

```python
push_dataset(
    repo_id="username/my-activations",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    prompts=all_prompts,
    labels=all_labels,
    metadata=[
        {"source": "reddit", "category": "safety", "split": "train"},
        {"source": "twitter", "category": "benign", "split": "train"},
        # ...one dict per prompt, all dicts must have the same keys
    ],
)
```

### Controlling what gets pushed

```python
push_dataset(
    repo_id="username/my-activations",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    prompts=all_prompts,
    tensors=["hidden_layers"],  # only activations, skip logits
    skip_missing=True,          # silently skip prompts not in local cache
    private=True,
)
```

---

## Stage 3: Train a probe from the dataset

Use `load_activations()` to download activation tensors, then train a probe with `fit_from_activations()`. Only needed shards are downloaded — if you request a single layer, only that layer's data is fetched.

```python
from lmprobe import load_activations, Probe

# Downloads only layer 16 shards — fast and selective
acts = load_activations(
    "username/llama-8b-safety-activations",
    layers=[16],
)

probe = Probe(classifier="logistic_regression", random_state=42)
probe.fit_from_activations(acts[16], labels)

predictions = probe.predict_from_activations(test_acts[16])
```

### Loading labels from the dataset

If you pushed labels with `push_dataset()`, load them alongside activations:

```python
acts, labels = load_activations(
    "username/llama-8b-safety-activations",
    layers=[16],
    return_labels=True,
)
# labels is a numpy array of ints (or None if no labels in dataset)
```

### Experiment quickly — no model needed

Because there's no model to load, iterating over classifiers and layers is fast:

```python
acts = load_activations("username/llama-8b-safety-activations", layers=[16])

for classifier in ["logistic_regression", "ridge", "lda", "mass_mean"]:
    probe = Probe(classifier=classifier, random_state=42)
    probe.fit_from_activations(acts[16], train_labels)
    acc = probe.score_from_activations(test_acts[16], test_labels)
    print(f"{classifier}: {acc:.3f}")
```

### Layer sweep from a dataset

```python
acts = load_activations("username/llama-8b-safety-activations")

scores = {}
for layer, X in acts.items():
    probe = Probe(classifier="ridge", random_state=42)
    probe.fit_from_activations(X[train_idx], train_labels)
    scores[layer] = probe.score_from_activations(X[test_idx], test_labels)

best = max(scores, key=scores.get)
print(f"Best layer: {best}, accuracy: {scores[best]:.3f}")
```

---

## Inspecting a dataset

Before training, check what's in a dataset without downloading tensors:

```python
from lmprobe import fetch_dataset_metadata

meta = fetch_dataset_metadata("username/llama-8b-safety-activations")
print(meta.model_name)       # meta-llama/Llama-3.1-8B-Instruct
print(meta.available_layers) # [0, 1, 2, ..., 31]
print(meta.num_prompts)      # 500
```

---

## Pulling a dataset to local cache

Pre-download shards before running experiments — useful when you know you'll be iterating extensively:

```python
from lmprobe import pull_dataset

n = pull_dataset(
    repo_id="username/llama-8b-safety-activations",
    layers=[16],      # only fetch the layers you need
)
print(f"Pulled {n} prompts into local cache")
```

---

## Loading raw tensors directly

For custom pipelines that need the raw activation tensors as numpy/torch arrays:

```python
from lmprobe import load_activation_dataset

tensors, info = load_activation_dataset(
    repo_id="username/llama-8b-safety-activations",
    layers=[16],
)

# tensors["hidden.layer_16"]: shape (n_prompts, hidden_dim)
X = tensors["hidden.layer_16"].numpy()
```

---

## Typical workflows

### Research: share activations with collaborators

```python
# You: extract once on your GPU machine
cache = UnifiedCache(model="meta-llama/Llama-3.1-8B-Instruct", layers="all")
cache.warmup(all_prompts)
push_dataset("myorg/project-activations", "meta-llama/Llama-3.1-8B-Instruct",
             all_prompts, labels=all_labels)

# Collaborators: train without GPU
acts, labels = load_activations("myorg/project-activations",
                                 layers=[16], return_labels=True)
probe = Probe(classifier="logistic_regression", random_state=42)
probe.fit_from_activations(acts[16], labels)
```

### Production: pre-cache for fast inference

```python
# Pull all shards upfront
pull_dataset("myorg/project-activations")

# Load and train
acts = load_activations("myorg/project-activations", layers=[16])
probe = Probe(classifier="logistic_regression", random_state=42)
probe.fit_from_activations(acts[16], labels)

# Predict on new activations
new_acts = load_activations("myorg/project-activations",
                             layers=[16], prompts=inference_prompts)
predictions = probe.predict_from_activations(new_acts[16])
```
