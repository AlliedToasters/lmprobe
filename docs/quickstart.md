# Quickstart

This guide walks through a complete probe workflow: install, train, evaluate, and save.

---

## Install

```bash
pip install lmprobe
```

---

## 1. Define your contrast pairs

Probes are trained *contrastively* — you provide examples of the positive class and examples of the negative class. The model learns what separates them in activation space.

```python
positive_prompts = [  # dog-like text, without using the word "dog"
    "Who wants to go for a walk?",
    "My tail is wagging with delight.",
    "Fetch the ball!",
    "Good boy!",
    "Slobbering, chewing, growling, barking.",
]

negative_prompts = [  # cat-like text
    "Enjoys lounging in the sun beam all day.",
    "Purring, stalking, pouncing, scratching.",
    "Uses a litterbox, throws sand all over the room.",
    "Tail raised, back arched, eyes alert, whiskers forward.",
]
```

!!! tip "Contrast pair quality"
    The contrastive approach is most effective when prompts isolate the property you care about. If the positive and negative prompts differ in multiple ways (topic, length, tone), the probe may learn those surface features instead of the target concept.

---

## 2. Configure the probe

```python
from lmprobe import Probe

probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,                          # which residual stream layer to probe
    pooling="last_token",               # how to aggregate across tokens
    classifier="logistic_regression",   # classification head
    device="auto",                      # "auto", "cpu", "cuda:0"
    random_state=42,
)
```

Key parameters at a glance:

| Parameter | What it controls |
|-----------|-----------------|
| `model` | HuggingFace model ID or local path |
| `layers` | Which layer(s) to extract (`int`, `list[int]`, `"middle"`, `"all"`, `"sweep"`) |
| `pooling` | Token aggregation strategy (`"last_token"`, `"mean"`, `"first_token"`) |
| `classifier` | Classification head (`"logistic_regression"`, `"ridge"`, `"svm"`, `"lda"`, `"mass_mean"`) |

See [API Reference: Probe](reference/probe.md) for the full parameter table.

---

## 3. Fit

```python
probe.fit(positive_prompts, negative_prompts)
```

This extracts activations from the model for each prompt, pools them, and fits the classifier. Activations are cached automatically — re-running `fit()` with the same prompts is fast.

---

## 4. Predict

```python
test_prompts = [
    "Arf! Arf! Let's go outside!",
    "Knocking things off the counter for sport.",
]

predictions = probe.predict(test_prompts)
# array([1, 0])  — 1=dog, 0=cat

probabilities = probe.predict_proba(test_prompts)
# array([[0.12, 0.88], [0.91, 0.09]])
```

---

## 5. Evaluate

```python
test_labels = [1, 0]  # ground truth

accuracy = probe.score(test_prompts, test_labels)

# Multiple metrics at once
metrics = probe.evaluate(
    test_prompts,
    test_labels,
    metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
)
```

---

## 6. Save and load

```python
# Save to disk
probe.save("dog_vs_cat_probe.pkl")

# Load for inference
loaded_probe = Probe.load("dog_vs_cat_probe.pkl")
predictions = loaded_probe.predict(test_prompts)
```

---

## Next steps

- [Contrastive Probing guide](guides/contrastive.md) — best practices for contrast pair design
- [Layer Selection & Sweep](guides/layer-sweep.md) — find the most informative layers
- [Baselines](guides/baselines.md) — validate your probe isn't learning surface features
- [Caching](guides/caching.md) — configure cache backends and reduce disk usage
- [Remote Execution](guides/remote.md) — probe large models via NDIF without local GPU
