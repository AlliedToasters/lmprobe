# Preprocessing

Apply feature transformations between activation extraction and classification. Preprocessing sits after pooling and per-layer normalization but before the classifier.

---

## When to use preprocessing

- **Multi-layer probes**: Concatenating many layers produces very high-dimensional inputs (e.g., 32 layers x 4096 = 131,072 dims). PCA reduces dimensionality and can improve both speed and accuracy.
- **Heterogeneous scales**: If per-layer normalization isn't enough, `"standard"` applies sklearn's `StandardScaler` to the full feature vector.
- **Regularization**: PCA acts as implicit regularization by discarding low-variance directions, which helps when training data is small relative to feature dimensionality.

---

## Available preprocessing steps

| Step | What it does |
|------|-------------|
| `"standard"` | Zero-mean, unit-variance scaling (`StandardScaler`) |
| `"pca"` | PCA dimensionality reduction |
| `"pca:N"` | PCA with exactly N components (e.g., `"pca:100"`) |
| `"standard+pca"` | Chain: standardize first, then PCA |

---

## Basic usage

```python
from lmprobe import Probe

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

# Inline component count
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers="all",
    preprocessing="pca:200",
)
```

---

## How preprocessing interacts with other features

### Per-layer normalization

When `normalize_layers=True` (the default for multi-layer probes) and `preprocessing` includes `"standard"`, per-layer normalization is **auto-disabled** to avoid double-standardizing. You'll see a log message when this happens.

If you want both (e.g., per-layer normalization with a different strategy than StandardScaler), set them explicitly:

```python
probe = Probe(
    layers=[14, 15, 16],
    normalize_layers="per_layer",   # coarser normalization
    preprocessing="standard",       # fine-grained normalization
)
```

### PCA component selection

The `pca_components` parameter or `"pca:N"` syntax controls how many components to keep. If unset, sklearn's default is used (keeps all components — which defeats the purpose for dimensionality reduction).

Rules of thumb:
- Start with 50–200 components for multi-layer probes
- More components preserve more signal but increase overfitting risk
- Fewer components act as stronger regularization

### Mass-mean augmentation

When `mass_mean_augment=True`, the projection onto the mass-mean direction is computed on the **original** activations (before preprocessing), then appended as an extra feature after preprocessing. This ensures PCA doesn't discard the mean-difference signal.

---

## Practical guidance

| Scenario | Recommended preprocessing |
|----------|--------------------------|
| Single layer, ~100 training samples | None (default) |
| Multiple layers, ~100 training samples | `"pca:50"` or `"standard+pca:50"` |
| All layers, large dataset (1000+) | `"standard+pca:200"` |
| Quick experimentation | `"standard"` (fast, no hyperparameter) |
