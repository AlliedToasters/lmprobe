# Caching

Activation extraction is expensive. A single forward pass through a large model can take seconds. `lmprobe` caches activations automatically so repeated calls with the same prompts are fast.

---

## Default behavior

Caching is always enabled. Activations are stored at `~/.cache/lmprobe/` by default. Override with an environment variable:

```bash
export LMPROBE_CACHE_DIR="/path/to/my/cache"
```

---

## Inspecting the cache

```python
from lmprobe import cache_info

info = cache_info()
print(info)
# CacheInfo(total_size_gb=3.42, models=[...])
```

---

## Reducing disk usage

Store activations in float16 instead of float32 (2× reduction, negligible accuracy impact):

```python
from lmprobe import set_cache_dtype

set_cache_dtype("float16")
```

---

## LRU eviction

Set a maximum cache size. When the limit is exceeded, least-recently-used entries are evicted:

```python
from lmprobe import set_cache_limit

set_cache_limit(50)  # GB
```

---

## S3 backend

Store activations in S3 for cross-machine sharing or building large datasets. Requires `pip install lmprobe[s3]`.

```python
from lmprobe import set_cache_backend

set_cache_backend("s3://my-bucket/lmprobe-cache")
```

!!! note "S3 is for datasets, not ephemeral caching"
    The S3 backend is designed for building and sharing large activation datasets: pre-extracting activations for thousands of prompts across machines. It is not intended as a drop-in replacement for the local cache for short-lived work.

Configure AWS credentials via the standard environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`) or an IAM role.

---

## Warmup

Pre-extract and cache activations before running predictions. Useful when you want to front-load extraction work:

```python
probe.warmup(test_prompts, batch_size=16)

# Subsequent calls hit the cache
predictions = probe.predict(test_prompts)
```

---

## Cache logging

Enable verbose logging to see cache hits and misses:

```python
from lmprobe import enable_cache_logging

enable_cache_logging()
```

---

## Clearing the cache

```python
from lmprobe import cache_info

# Clear everything (irreversible)
from lmprobe.cache import clear_cache
clear_cache()

# Clear only a specific model's cache
from lmprobe import clear_model_cache
clear_model_cache("meta-llama/Llama-3.1-8B-Instruct")
```

---

## Cache format

Activations are stored in [safetensors](https://github.com/huggingface/safetensors) format (v2), keyed per prompt, per model, per layer. The key is a hash of the prompt text and model ID. Older `.pt` format caches (v1) are still readable for backwards compatibility.
