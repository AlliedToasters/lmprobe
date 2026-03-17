# Tutorial: Geometry of Truth Probes

This tutorial trains truthfulness probes on the **Geometry of Truth** datasets using pre-extracted activations from `latent-lab/got-activations-qwen2.5-0.5b` — no GPU or local model required.

[Geometry of Truth](https://arxiv.org/abs/2310.06824) (Marks & Tegmark, 2023) showed that language models represent truth as a linear feature in activation space, detectable with simple probes. We'll reproduce that result on Qwen2.5-0.5B and explore what the activations reveal.

---

## The dataset

`latent-lab/got-activations-qwen2.5-0.5b` contains 7,600 true/false statements from six GoT categories, with full-sequence activations pre-extracted across all 24 layers of Qwen2.5-0.5B (896-dim, float32). Top-100 logits are also cached.

| Category | N | Example |
|----------|---|---------|
| `cities` | 1,486 | *"The city of Amman is in Jordan."* |
| `neg_cities` | 1,482 | *"The city of Omsk is not in Russia."* |
| `sp_en_trans` | 350 | *"The Spanish word 'rosa' means 'rose'."* |
| `neg_sp_en_trans` | 353 | *"The Spanish word 'miedo' does not mean 'fear'."* |
| `larger_than` | 1,966 | *"Ninety-nine is larger than eighty-five."* |
| `smaller_than` | 1,963 | *"Fifty-six is smaller than eighty-one."* |

Labels: `1` = true statement, `0` = false statement. The dataset is nearly balanced (~50/50).

---

## Setup

```bash
pip install lmprobe[hub] pyarrow
```

No API keys required. No GPU required.

---

## Step 1: Load the index

The Parquet index is tiny (~1 MB). Download it first to understand the dataset structure and filter prompts before touching any tensors:

```python
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

DATASET = "latent-lab/got-activations-qwen2.5-0.5b"

path = hf_hub_download(DATASET, "index/train-00000-of-00001.parquet", repo_type="dataset")
index = pq.read_table(path)

texts  = index["text"].to_pylist()
labels = index["label"].to_pylist()
cats   = index["category"].to_pylist()

print(f"{len(texts)} prompts, columns: {index.column_names}")
# 7600 prompts, columns: ['text', 'label', 'category', 'prompt_format', ...]
```

Filter to a single category:

```python
def filter_category(category):
    mask = [c == category for c in cats]
    return (
        [t for t, m in zip(texts, mask) if m],
        [l for l, m in zip(labels, mask) if m],
    )

cities_texts, cities_labels = filter_category("cities")
print(f"cities: {len(cities_texts)} prompts")
# cities: 1486 prompts
```

---

## Step 2: Train a probe

Split into train/test and fit a probe. Activations are pulled from HuggingFace on demand and cached locally — the first run downloads ~200 MB of shards for the selected layer, subsequent runs are instant.

```python
from sklearn.model_selection import train_test_split
from lmprobe import Probe

train_texts, test_texts, train_labels, test_labels = train_test_split(
    cities_texts, cities_labels, test_size=0.2, random_state=42, stratify=cities_labels
)

probe = Probe(
    dataset=DATASET,
    layers=14,                        # best layer for cities (found via sweep below)
    pooling="last_token",
    classifier="logistic_regression",
    random_state=42,
)

probe.fit(train_texts, train_labels)
metrics = probe.evaluate(test_texts, test_labels)
print(f"Accuracy: {metrics['accuracy']:.1%},  AUROC: {metrics['auroc']:.3f}")
# Accuracy: 96.1%,  AUROC: 0.980
```

---

## Step 3: Find the best layer

Layer 14 wasn't hand-picked — it came from a sweep. Here's how to reproduce it:

```python
result = Probe.sweep_layers(
    dataset=DATASET,
    positive_prompts=train_texts,   # note: using labels, not contrastive pairs
    negative_prompts=train_labels,  # pass labels as second arg for standard mode
    layers="all",
    classifier="logistic_regression",
    random_state=42,
)

scores = result.score(test_texts, test_labels)

# Print layer-by-layer accuracy
for layer, acc in sorted(scores.items()):
    bar = "█" * int(acc * 40)
    print(f"  Layer {layer:2d}  {acc:.1%}  {bar}")

best = result.best_layer(test_texts, test_labels)
print(f"\nBest layer: {best}  ({scores[best]:.1%})")
```

Expected output:

```
  Layer  0  51.0%  ████████████████████
  Layer  1  54.3%  █████████████████████
  ...
  Layer 12  93.2%  █████████████████████████████████████
  Layer 13  95.4%  ██████████████████████████████████████
  Layer 14  96.1%  ██████████████████████████████████████
  Layer 15  95.8%  ██████████████████████████████████████
  ...
  Layer 23  88.4%  ███████████████████████████████████

Best layer: 14  (96.1%)
```

Signal emerges around layer 10 and peaks in the middle layers — a pattern consistent with the original GoT paper.

---

## Step 4: Compare classifiers

```python
results = {}
for clf in ["logistic_regression", "ridge", "svm", "lda", "mass_mean"]:
    p = Probe(dataset=DATASET, layers=14, classifier=clf, random_state=42)
    p.fit(train_texts, train_labels)
    m = p.evaluate(test_texts, test_labels)
    results[clf] = m
    print(f"  {clf:22s}  acc={m['accuracy']:.1%}  auroc={m['auroc']:.3f}")
```

Expected:

```
  logistic_regression    acc=96.1%  auroc=0.980
  ridge                  acc=95.8%  auroc=0.977
  svm                    acc=95.6%  auroc=0.976
  lda                    acc=94.9%  auroc=0.971
  mass_mean              acc=67.3%  auroc=0.724
```

!!! note "Mass-Mean underperforms"
    This is a notable finding. Mass-Mean Probing (classifying by projection onto the difference-in-means direction) is the method highlighted in the original GoT paper — yet it performs dramatically worse than logistic regression here (~67% vs ~96%).

    One interpretation: the truth direction in Qwen2.5's activation space isn't well-aligned with the mean difference between true and false statements, suggesting a more curved or noisy representation than the paper's models exhibited.

---

## Step 5: Generalize across all six categories

Does a probe trained on `cities` transfer to other datasets?

```python
# Train on cities
train_t, test_t, train_l, test_l = train_test_split(
    cities_texts, cities_labels, test_size=0.2, random_state=42, stratify=cities_labels
)
probe = Probe(dataset=DATASET, layers=14, classifier="logistic_regression", random_state=42)
probe.fit(train_t, train_l)

# Evaluate on every category
categories = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "larger_than", "smaller_than"]
print("Transfer from cities probe:")
for cat in categories:
    cat_texts, cat_labels = filter_category(cat)
    acc = probe.score(cat_texts, cat_labels)
    print(f"  {cat:20s}  {acc:.1%}")
```

Or train a fresh probe per category to find each one's best layer:

```python
for cat in categories:
    t, l = filter_category(cat)
    tr_t, te_t, tr_l, te_l = train_test_split(t, l, test_size=0.2, random_state=42, stratify=l)

    result = Probe.sweep_layers(
        dataset=DATASET,
        positive_prompts=tr_t,
        negative_prompts=tr_l,
        layers="all",
        classifier="logistic_regression",
        random_state=42,
    )
    best = result.best_layer(te_t, te_l)
    scores = result.score(te_t, te_l)
    print(f"  {cat:20s}  best_layer={best}  acc={scores[best]:.1%}")
```

---

## Step 6: Pull everything locally for fast iteration

After your first run, activate layers are in your local cache. But if you plan to sweep all layers across all categories repeatedly, pre-downloading everything upfront avoids incremental HuggingFace fetches:

```python
from lmprobe import pull_dataset

n = pull_dataset(DATASET)  # downloads all shards (~3 GB)
print(f"Pulled {n} prompts — all subsequent probe training runs from local cache")
```

After this, `Probe(dataset=DATASET, ...)` hits the local cache with zero network calls.

---

## Key findings

| Finding | Detail |
|---------|--------|
| **Truth is linearly represented** | Logistic regression reaches >96% on cities at layer 14 — consistent with the GoT paper's core claim |
| **Signal peaks in middle layers** | Layers 12–16 are most informative; early and late layers are weaker |
| **Mass-Mean notably weaker** | ~67% vs ~96% for logistic regression — the mean-difference direction is not the optimal linear separator for this model |
| **LR, Ridge, SVM are comparable** | All three reach 94–96%; the linear representation is robust to classifier choice |
| **No GPU needed** | The full analysis runs on CPU using pre-extracted activations from HuggingFace |
