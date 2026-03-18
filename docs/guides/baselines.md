# Baselines

Always compare your probe against baselines. A probe that doesn't beat a bag-of-words classifier may be learning surface features, not model-internal representations.

---

## Text-only baselines

```python
from lmprobe import BaselineProbe

# Bag-of-words
bow = BaselineProbe(method="bow", classifier="logistic_regression")
bow.fit(positive_prompts, negative_prompts)
bow_acc = bow.score(test_prompts, test_labels)

# TF-IDF
tfidf = BaselineProbe(method="tfidf")
tfidf.fit(positive_prompts, negative_prompts)

# Sentence length (surprisingly predictive for some tasks)
length = BaselineProbe(method="sentence_length")
length.fit(positive_prompts, negative_prompts)

# Sentence-transformers embeddings (requires: pip install lmprobe[embeddings])
st = BaselineProbe(method="sentence_transformers")
st.fit(positive_prompts, negative_prompts)

# Sanity checks
random_baseline = BaselineProbe(method="random")    # should be ~50%
majority_baseline = BaselineProbe(method="majority")
shuffled_baseline = BaselineProbe(method="shuffled_labels")  # overfitting check
```

---

## Activation-based baselines

These use the same model activations as your probe but with simpler classification approaches. If your probe doesn't beat them, the learned direction may not be special:

```python
from lmprobe import ActivationBaseline

# Random direction — project onto random unit vector
random_dir = ActivationBaseline(
    method="random_direction",
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=-1,
)
random_dir.fit(positive_prompts, negative_prompts)

# PCA — classify using top principal components
pca = ActivationBaseline(
    method="pca",
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=-1,
)

# Layer 0 — input embeddings only (no deep processing)
layer0 = ActivationBaseline(
    method="layer_0",
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=-1,
)

# Perplexity — model's own token probabilities (uses BaselineProbe, not ActivationBaseline)
perplexity = BaselineProbe(
    method="perplexity",
    model="meta-llama/Llama-3.1-8B-Instruct",
)
```

---

## Baseline battery

Run all applicable baselines at once and get a ranked comparison:

```python
from lmprobe import BaselineBattery

# Text-only (no model required)
battery = BaselineBattery(model=None, random_state=42)
results = battery.fit(positive_prompts, negative_prompts, test_prompts, test_labels)

print(results.summary())
# Baseline Results:
# ------------------------------------------------------------
#   sentence_transformers          0.7925  (fit: 1.23s, predict: 0.05s)
#   tfidf                          0.7547  (fit: 0.01s, predict: 0.00s)
#   bow                            0.6792  (fit: 0.01s, predict: 0.00s)
#   random                         0.4906  (fit: 0.00s, predict: 0.00s)
#   majority                       0.5556  (fit: 0.00s, predict: 0.00s)

best = results.get_best()[0]
print(f"Best baseline: {best.name} — {best.score:.2%}")

# With activation baselines (requires model)
battery = BaselineBattery(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=-1,
    include=["bow", "tfidf", "random_direction", "pca"],
)
results = battery.fit(positive_prompts, negative_prompts, test_prompts, test_labels)
```

---

## All baseline methods

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
| `perplexity` | Activation | Model's own token log-probabilities |

---

## Interpreting results

| Situation | What it means |
|-----------|--------------|
| Probe ≈ random baseline | Probe learned nothing; check data quality or layer choice |
| Probe ≈ BOW/TF-IDF | Probe may be learning lexical features, not activations |
| Probe ≈ sentence_transformers | Signal may be general semantics, not model-specific |
| Probe ≈ layer_0 | Signal is in token identity/position, not deep representations |
| Probe ≈ random_direction | Any direction in activation space works; signal is not a specific learned direction |
| Probe >> all baselines | Strong evidence of model-internal signal |
