# Remote Execution

`lmprobe` supports remote execution via [nnsight](https://nnsight.net/) and [NDIF](https://nnsight.net/) (National Deep Inference Fabric), allowing you to probe large models without local GPU resources.

!!! note "US-only access"
    NDIF currently restricts access to US-based users. Remote functionality has not been fully integration-tested. See [known considerations](#known-considerations).

---

## Setup

Install the nnsight extra:

```bash
pip install lmprobe[nnsight]
```

Set your API key:

```bash
export NNSIGHT_API_KEY="your-api-key-here"
```

---

## Basic usage

```python
from lmprobe import Probe

probe = Probe(
    model="meta-llama/Llama-3.1-70B-Instruct",
    layers="middle",
    backend="nnsight",
    remote=True,
)

probe.fit(positive_prompts, negative_prompts)
predictions = probe.predict(test_prompts)
```

---

## Per-call remote override

You can set `remote` at the probe level but override it per call. A common pattern: extract activations remotely during training (large model), then run inference locally on a smaller model or cached activations:

```python
probe = Probe(
    model="meta-llama/Llama-3.1-70B-Instruct",
    backend="nnsight",
    remote=True,
    layers=20,
)

probe.fit(positive_prompts, negative_prompts)  # extracts remotely

# Predict locally (uses cached activations if already computed)
predictions = probe.predict(new_prompts, remote=False)
```

---

## Caching with remote execution

Activations extracted remotely are cached locally using the same cache system as local execution. This means:

- Re-running `fit()` or `predict()` with the same prompts hits the local cache
- Remote calls happen only for new/uncached prompts
- You can warm up the cache before a deadline:

```python
probe.warmup(test_prompts, batch_size=4)  # cache everything remotely upfront
predictions = probe.predict(test_prompts)  # now runs fully from cache
```

---

## Known considerations

- Remote execution returns nnsight proxy tensors rather than direct tensors. `lmprobe` handles this via `hasattr(act, "value")` checks in `extraction.py`.
- Network latency affects batch processing — use smaller `batch_size` for remote calls.
- Error handling for missing/invalid API keys raises an informative exception before any network calls are made.

---

## Backend parameter

The `backend` parameter controls the extraction engine:

| Value | Description |
|-------|-------------|
| `"local"` | HuggingFace Transformers (default) |
| `"nnsight"` | nnsight / NDIF remote execution |

`remote=True` requires `backend="nnsight"`. Setting `remote=True` with `backend="local"` will raise an error.
