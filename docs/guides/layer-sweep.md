# Layer Selection & Sweep

Different layers of a language model encode different information. Finding the right layer — or combination of layers — is often the most impactful tuning decision.

---

## Layer specification syntax

| Spec | Description |
|------|-------------|
| `16` | Single layer (negative indexing: `-1` = last) |
| `[14, 15, 16]` | Multiple layers (concatenated) |
| `"middle"` | Middle third of layers |
| `"last"` | Last layer only |
| `"all"` | All layers concatenated |
| `"auto"` | Automatic via Group Lasso (`pip install lmprobe[auto]`) |
| `"fast_auto"` | Fast selection via coefficient importance |
| `"sweep"` | Train independent probe per layer |
| `"sweep:10"` | Sweep every 10th layer |
| `"sweep:55-65"` | Sweep layers 55 through 65 |

---

## Layer sweep

Train an independent probe at every layer to identify which are most informative:

```python
result = Probe.sweep_layers(
    model="meta-llama/Llama-3.1-8B-Instruct",
    positive_prompts=positive_prompts,
    negative_prompts=negative_prompts,
    layers="all",
    classifier="ridge",
)

# Score all layers
scores = result.score(test_prompts, test_labels)
# {0: 0.52, 1: 0.55, ..., 31: 0.78}

# Best layer
best = result.best_layer(test_prompts, test_labels)
print(f"Best layer: {best}")

# Predict with a specific layer's probe
preds = result.probes[best].predict(test_prompts)
```

You can also use sweep as a layer spec string — useful when you want to sweep as part of a normal `Probe` workflow:

```python
probe = Probe(model=model, layers="sweep")        # sweep all
probe = Probe(model=model, layers="sweep:10")      # every 10th
probe = Probe(model=model, layers="sweep:55-65")   # a range
```

---

## Layer importance analysis

When using multiple layers (e.g., `layers="all"`), compute per-layer importance from the fitted classifier's coefficients:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers="all",
    classifier="ridge",
)

probe.fit(positive_prompts, negative_prompts)

importances = probe.compute_layer_importance(metric="l2")
# {0: 0.03, 1: 0.05, ..., 31: 0.42}
```

Visualize (requires `pip install lmprobe[plot]`):

```python
probe.plot_layer_importance()
```

---

## Fast auto layer selection

Automatically select the most important layers using importance analysis, then refit on just those layers:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers="fast_auto",
    fast_auto_top_k=3,       # keep top 3 layers
    normalize_layers=True,
)

probe.fit(positive_prompts, negative_prompts)
print(f"Selected layers: {probe.selected_layers_}")
```

This is a two-stage process: first fit on all layers, then refit on the top-k layers by importance.

---

## Automatic layer selection via Group Lasso

Use structured sparsity (Group Lasso) to let the optimizer choose which layers to keep. More principled than fast_auto but slower:

```python
# Requires: pip install lmprobe[auto]
probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers="auto",
    auto_candidates=[0.25, 0.5, 0.75],  # fractional positions or explicit indices
    auto_alpha=0.01,                     # regularization strength
)

probe.fit(positive_prompts, negative_prompts)
print(f"Selected layers: {probe.selected_layers_}")
```

---

## Practical guidance

**Where to start:**

- Middle layers (12–20 in a 32-layer model) are often best for semantic properties
- Last layer is usually best for surface/output-level properties
- First few layers are mostly syntactic/positional

**When to sweep:**

- When you have no prior knowledge about the task
- When you want to verify your layer choice is principled

**When to use `"all"` with concatenation:**

- When signal is distributed across many layers
- Combined with PCA or Group Lasso to manage dimensionality

**When to use `fast_auto`:**

- When you want a data-driven choice without the overhead of Group Lasso
- Good default when you don't want to hand-tune `layers`
