# Contrastive Probing

The primary training paradigm in `lmprobe` is **contrastive**: you provide a positive class and a negative class, and the probe learns to separate them in activation space.

---

## Basic usage

```python
from lmprobe import Probe

probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,
    pooling="last_token",
)

probe.fit(positive_prompts, negative_prompts)
```

Internally, `fit()` assigns label `1` to positive prompts and `0` to negative prompts, concatenates them, and trains the classifier on the pooled activations.

---

## Multi-layer probing

When you specify multiple layers, activations are **concatenated** along the hidden dimension before classification:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=[14, 15, 16],  # 3 × 4096 = 12,288-dim input to classifier
)
```

This often improves accuracy because different layers encode different aspects of the concept. For high-dimensional inputs, consider adding preprocessing:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=[14, 15, 16],
    preprocessing="standard+pca",
    pca_components=100,
)
```

---

## Per-layer normalization

When combining multiple layers, high-magnitude layers can dominate. Enable per-layer normalization (default: on) to standardize each layer independently before concatenation:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=[14, 15, 16],
    normalize_layers=True,           # default: per-neuron standardization
    # normalize_layers="per_layer",  # one mean/std per layer
    # normalize_layers=False,        # disable
)
```

---

## Different pooling for train vs inference

You can use one pooling strategy during training and a different one during inference. This is useful for streaming/real-time monitoring:

```python
# Train on stable last-token representation; score every token at inference
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,
    pooling="last_token",        # used for fit()
    inference_pooling="all",     # used for predict() — returns per-token scores
)

probe.fit(positive_prompts, negative_prompts)

# Returns (batch, seq_len) — one score per token
token_scores = probe.predict_proba(["Wagging my tail happily!"])
```

For "flag if ANY token triggers" detection:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,
    pooling="last_token",
    inference_pooling="max",     # max score across all tokens
)
```

**Pooling collision rules:**

```
pooling="mean", train_pooling="last_token"  →  train=last_token, inference=mean
pooling="mean", inference_pooling="max"     →  train=mean, inference=max
```

---

## Working with pre-computed activations

If you already have activation tensors (e.g., from a different extraction pipeline), you can bypass the extraction step entirely:

```python
import numpy as np

probe = Probe(classifier="logistic_regression", random_state=42)

# X: (n_samples, hidden_dim), y: (n_samples,)
probe.fit_from_activations(X_train, y_train)
predictions = probe.predict_from_activations(X_test)
accuracy = probe.score_from_activations(X_test, y_test)
```

---

## Regression targets

For continuous targets instead of binary classification:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,
    task="regression",  # uses Ridge regression by default
)

# fit() takes prompts + continuous labels (not negative_prompts)
probe.fit(prompts, labels)  # labels: list[float]

predictions = probe.predict(test_prompts)   # continuous values
r_squared = probe.score(test_prompts, test_labels)
```

---

## Classifier options

| Classifier | Notes |
|------------|-------|
| `"logistic_regression"` | Default. Good all-around choice. |
| `"ridge"` | Faster, no `predict_proba`. Good for large datasets. |
| `"svm"` | SVM with probability calibration. |
| `"lda"` | Linear Discriminant Analysis. |
| `"mass_mean"` | Mass-Mean Probing: difference-in-means direction. Simple and often competitive. |
| `"sgd"` | SGD classifier. Useful for very large datasets. |

Pass a custom sklearn estimator directly:

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,
    classifier=CalibratedClassifierCV(LinearSVC()),
)
```

Pass extra kwargs to built-in classifiers:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,
    classifier="logistic_regression",
    classifier_kwargs={"C": 0.01, "solver": "liblinear", "max_iter": 5000},
)
```
