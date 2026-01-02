# `lmprobe` Language Model Probe Library

[![PyPI version](https://badge.fury.io/py/lmprobe.svg)](https://pypi.org/project/lmprobe/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This library supports the use of language model "activations" or "latents" to build text classifiers. The intent is to help detect and reduce misuse of AI - for example, chemical, biological, radiological and nuclear (CBRN) weapons development, social engineering at scale, and the development of novel cybersecurity attack vectors.

## Linear and Simple Models for LLMs
"Linear Probes" have emerged as an effective and practical way to monitor large language model activity. 

### Background

First introduced by [Alain & Bengio (2016)](https://arxiv.org/abs/1610.01644) as "thermometers" for measuring what neural networks learn at each layer, linear probes have since been refined through work on [probe design and selectivity](https://nlp.stanford.edu/~johnhew/interpreting-probes.html) and validated by evidence supporting the [linear representation hypothesis](https://www.neelnanda.io/mechanistic-interpretability/othello). The [Representation Engineering](https://arxiv.org/abs/2310.01405) framework (Zou et al., 2023) demonstrated that probes can monitor safety-relevant properties like honesty and power-seeking. Recent AI safety research has shown promising results: Anthropic's work on [detecting sleeper agents](https://www.anthropic.com/research/probes-catch-sleeper-agents) achieved >99% AUROC using simple linear classifiers, and Apollo Research's [strategic deception detection](https://arxiv.org/abs/2502.03407) work demonstrates that probes trained on simple contrast pairs can generalize to realistic scenarios like insider trading concealment and sandbagging on safety evaluations.

### `lmprobe` Use Cases

The goal of `lmprobe` is to make text classifiers for language models easy to build, experiment on, and deploy during inference. While much of the research has focused on complex emergent risky behavior, the intended use of this library is for simpler use cases such as detection of the misuse of an AI chatbot by humans.

### Compatibility

By default, `lmprobe` uses huggingface and `nnsight` to manage models and extract latents during inference. However, the library is structured to modularize and isolate these aspects so that (ideally) frontier AI labs can extend the library for internal use on their bespoke inference systems.

### Installation

```
pip install lmprobe
```

### Environment Setup

For remote execution (large models via nnsight/NDIF):

```bash
export NNSIGHT_API_KEY="your-api-key-here"
```

### Example Usage

---

```python
from lmprobe import LinearProbe

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
probe = LinearProbe(
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
loaded_probe = LinearProbe.load("dog_vs_cat_probe.pkl")
```

---

## Remote Execution for Large Models

Use `remote=True` to run inference on large models via nnsight's remote servers:

```python
probe = LinearProbe(
    model="meta-llama/Llama-3.1-70B-Instruct",
    layers="middle",
    remote=True,  # Requires NNSIGHT_API_KEY
)

probe.fit(positive_prompts, negative_prompts)

# Override remote per-call (e.g., train remote, predict local)
predictions = probe.predict(new_prompts, remote=False)
```

---

## Multi-Layer Probing

When selecting multiple layers, activations are **concatenated** along the hidden dimension:

```python
probe = LinearProbe(
    model="meta-llama/Llama-3.1-8B-Instruct",
    layers=[14, 15, 16],  # 3 layers × 4096 dims = 12,288-dim input to classifier
)
```

---

## Advanced: Different Train vs Inference Pooling

For real-time monitoring, train on a stable representation but score every token:

```python
probe = LinearProbe(
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
probe = LinearProbe(
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
| `layers` | `int \| list[int] \| "all"` | `"middle"` | Which residual stream layers to probe |
| `pooling` | `str \| callable` | `"last_token"` | Token aggregation (train & inference) |
| `train_pooling` | `str \| callable` | — | Override pooling for `fit()` only |
| `inference_pooling` | `str \| callable` | — | Override pooling for `predict()` only |
| `classifier` | `str \| sklearn estimator` | `"logistic_regression"` | Classification model |
| `device` | `str` | `"auto"` | `"auto"`, `"cuda:0"`, `"cpu"` |
| `remote` | `bool` | `False` | Use nnsight remote execution (requires `NNSIGHT_API_KEY`) |
| `random_state` | `int \| None` | `None` | Random seed for reproducibility (propagates to classifier) |

### Pooling Strategies

| Strategy | Training | Inference | Description |
|----------|:--------:|:---------:|-------------|
| `"last_token"` | ✓ | ✓ | Final token activation (default, matches RepE literature) |
| `"mean"` | ✓ | ✓ | Mean across all tokens |
| `"first_token"` | ✓ | ✓ | First token (e.g., `[CLS]`) |
| `"all"` | ✓ | ✓ | Each token independently |
| `"max"` | | ✓ | Max score across tokens |
| `"min"` | | ✓ | Min score across tokens |

### Pooling Collision Rules

Explicit parameters override the base `pooling` value:

```python
# pooling="mean", train_pooling="last_token" → train=last_token, inference=mean
# pooling="mean", inference_pooling="max"    → train=mean, inference=max
```

