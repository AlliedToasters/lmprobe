# lmprobe

**Train linear probes on language model activations for AI safety monitoring.**

[![PyPI version](https://badge.fury.io/py/lmprobe.svg)](https://pypi.org/project/lmprobe/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`lmprobe` makes it easy to build text classifiers from a language model's internal representations. It has been used for detection of deception, harmful intent, CBRN misuse, and other safety-relevant properties, but can also be used to build arbitrary classifiers and even regression.

---

## What is a probe?

A probe is a classifier trained on a model's intermediate activations (residual stream, hidden states) rather than its output text. A very common type of probe is the linear probe: because the classifier is linear, it's fast to train, interpretable, and provably reflects what the model *represents,* not just what it *says*.

Key results from the literature:

- Anthropic's probe work achieved >99% AUROC detecting [sleeper agents](https://www.anthropic.com/research/probes-catch-sleeper-agents)
- [Representation Engineering](https://arxiv.org/abs/2310.01405) (Zou et al., 2023) showed probes reliably track honesty and power-seeking
- Apollo Research demonstrated probes trained on simple contrast pairs generalize to [realistic deception scenarios](https://arxiv.org/abs/2502.03407)

---

## Install

```bash
pip install lmprobe
```

Optional extras:

| Extra | Installs |
|-------|----------|
| `lmprobe[hub]` | HuggingFace Hub push/pull |
| `lmprobe[s3]` | S3 cache backend |
| `lmprobe[nnsight]` | Remote execution via NDIF |
| `lmprobe[plot]` | Layer importance visualization |
| `lmprobe[embeddings]` | Sentence-transformers baselines |
| `lmprobe[auto]` | Automatic layer selection (Group Lasso) |

---

## Five-minute example

```python
from lmprobe import Probe

positive_prompts = [
    "Who wants to go for a walk?",
    "My tail is wagging with delight.",
    "Fetch the ball!",
]

negative_prompts = [
    "Purring, stalking, pouncing, scratching.",
    "Uses a litterbox, throws sand all over the room.",
    "Tail raised, back arched, eyes alert.",
]

probe = Probe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=16,
    pooling="last_token",
    classifier="logistic_regression",
)

probe.fit(positive_prompts, negative_prompts)

predictions = probe.predict(["Arf! Let's go outside!", "Knocking things off the counter."])
# [1, 0]
```

See the [Quickstart](quickstart.md) for a complete walkthrough.

---

## Design philosophy

- **sklearn-inspired API** — `fit()`, `predict()`, `predict_proba()`, `score()`
- **Contrastive-first** — positive vs. negative prompts, following the RepE literature
- **Sensible defaults** — simple cases are one-liners; complex cases are fully configurable
- **Separation of concerns** — extraction, pooling, and classification are distinct and independently configurable
