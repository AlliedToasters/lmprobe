# lmprobe Dataset Format Specification

**Format version:** 2.0

This document describes the lmprobe dataset format: a self-describing, sharded storage layout for language model activation data. The format is designed so that a researcher with no knowledge of lmprobe can download a single parquet file, understand the dataset, and write their own loading code.

---

## How the format works

An lmprobe dataset consists of two parts:

1. **A parquet index** — a small file with one row per prompt. Contains the text, labels, and addressing columns that locate each prompt's tensor data within the shards.
2. **Tensor shards** — safetensors files containing the actual activation vectors. Split by layer, then by prompt groups, and optimized for common access patterns.

The parquet index is the entry point. Everything a researcher needs to understand the dataset (what model produced it, what layers are stored, how to construct shard filenames) is embedded in the parquet's schema metadata. The tensor shards are bulk data; their structure is fully described by the index.

---

## The parquet index

### Core columns (always present)

| Column | Type | Description |
|---|---|---|
| `text` | `string` | The prompt text |
| `label` | `int32` or `string` | Class label (may be null for unlabeled data) |
| `num_tokens` | `int32` | Token count for this prompt |
| `shard_index` | `int32` | Which shard file holds this prompt's tensor |
| `row_offset` | `int32` | Row position within that shard |

**The universal contract:** `shard_index` and `row_offset` always resolve to the last-token hidden-state vector for a prompt. This is true regardless of whether the dataset stores pooled or full-sequence activations. These two columns are the only addressing a researcher needs for the most common workflow: extracting one activation vector per prompt per layer.

### Full-sequence columns (when full-sequence data is stored)

| Column | Type | Description |
|---|---|---|
| `token_offset` | `int64` | Legacy offset (equal to `row_offset` for compatibility) |
| `token_shard_ids` | `list<int64>` | Per-token shard index; one entry per token in the prompt |
| `token_shard_offsets` | `list<int64>` | Per-token row offset within each shard |

To retrieve the activation for token `i` of a prompt, use `token_shard_ids[i]` and `token_shard_offsets[i]` in place of `shard_index` and `row_offset`. The universal `shard_index`/`row_offset` columns still point to the last token, as always.

### Logits columns (when logits are stored)

| Column | Type | Description |
|---|---|---|
| `shard_index_logits` | `int32` | Shard index for logits data |
| `row_offset_logits` | `int32` | Row offset within the logits shard |

Logits are stored in separate shard files from hidden states, so they have their own addressing columns.

### Optional columns

**Perplexity** (when the publisher computed perplexity):

| Column | Type | Description |
|---|---|---|
| `perplexity_mean` | `float64` | Mean per-token perplexity for the prompt |
| `perplexity_min` | `float64` | Minimum per-token perplexity |
| `perplexity_max` | `float64` | Maximum per-token perplexity |

**Token-level data** (when the publisher included decoded tokens):

| Column | Type | Description |
|---|---|---|
| `token_ids` | `list<int64>` | Token IDs for the prompt |
| `token_perplexity` | `list<float64>` | Per-token perplexity values |
| `token_strings` | `list<string>` | Decoded token strings |

**User metadata:** Any additional columns the publisher attached at push time (dataset name, category, source, etc.) appear as extra columns with auto-inferred types.

---

## Schema metadata

All dataset-level metadata is embedded in the parquet file's schema metadata under keys prefixed with `lmprobe:`. Each key stores a JSON-encoded value.

To read it:

```python
import json
import pyarrow.parquet as pq

table = pq.read_table("index/train-00000-of-00001.parquet")

info = {
    k.decode().removeprefix("lmprobe:"): json.loads(v)
    for k, v in table.schema.metadata.items()
    if k.decode().startswith("lmprobe:")
}
```

This produces a dict with the following top-level keys:

### `format_version`

String. Currently `"2.0"`.

### `model`

The source model.

```json
{
  "name": "meta-llama/Llama-3.1-8B-Instruct",
  "revision": "5206a32e0bd3067aef1ce90f5528ade7d866253f"
}
```

`name` is the HuggingFace model identifier. `revision` is the exact commit hash, pinning the model weights that produced this data.

### `num_prompts`

Integer. Total number of prompts in the dataset.

### `prompt_ordering`

String. How prompts are ordered in the index. Typically `"random"`.

### `tensors`

The core of the metadata. Describes what tensor data exists and how to find it. See [Tensor descriptors](#tensor-descriptors) below.

### `provenance`

Build environment details.

```json
{
  "lmprobe_version": "0.8.9",
  "extraction_backend": "local",
  "nnsight_version": null,
  "torch_version": "2.5.1",
  "transformers_version": "4.46.3",
  "python_version": "3.11.10",
  "created_at": "2026-03-25T12:00:00+00:00"
}
```

---

## Tensor descriptors

The `tensors` dict contains one entry per tensor type. The two tensor types are `hidden_layers` (hidden-state activations) and `logits_topk` (top-k logit values and indices).

### Hidden layers: pooled storage

The common case. One vector per prompt per layer, using last-token pooling.

```json
{
  "hidden_layers": {
    "type": "hidden",
    "layers": [0, 1, 2, "...", 31],
    "dim": 4096,
    "dtype": "float32",
    "layout": "per_layer",
    "file_pattern": "tensors/hidden_layer{layer:03d}_shard{shard:03d}.safetensors",
    "key_pattern": "hidden.layer_{layer}",
    "storage": "pooled",
    "pooling": "last_token",
    "row_bytes": 16384,
    "shards": [
      {"num_prompts": 500}
    ]
  }
}
```

| Field | Description |
|---|---|
| `layers` | List of layer indices stored |
| `dim` | Hidden dimension per vector |
| `dtype` | Tensor dtype (`"float32"`, `"float16"`, etc.) |
| `layout` | Always `"per_layer"` — each layer is stored in its own set of shard files |
| `file_pattern` | Python format string for constructing shard filenames |
| `key_pattern` | Python format string for the tensor key inside each safetensors file |
| `storage` | `"pooled"` — one vector per prompt |
| `pooling` | `"last_token"` — the vector is the activation at the final token position |
| `row_bytes` | Byte size of one row (= `dim x dtype_size`) |
| `shards` | List of shard descriptors, ordered by shard index |

Each entry in `shards` describes one shard file. `num_prompts` is the number of prompts (rows) in that shard. Shard indices are implicit from list position: the first entry is shard 0, the second is shard 1, and so on.

### Hidden layers: full-sequence storage

All tokens stored, with dedicated last-token shards for fast access to the common case.

```json
{
  "hidden_layers": {
    "type": "hidden",
    "layers": [0, 1, 2, "...", 125],
    "dim": 16384,
    "dtype": "float32",
    "layout": "per_layer",
    "file_pattern": "tensors/hidden_layer{layer:03d}_shard{shard:03d}.safetensors",
    "key_pattern": "hidden.layer_{layer}",
    "storage": "full_sequence",
    "last_token_shards": 2,
    "shards": [
      {"num_prompts": 4000, "num_tokens": 4000},
      {"num_prompts": 3660, "num_tokens": 3660},
      {"num_prompts": 4000, "num_tokens": 285432},
      {"num_prompts": 3660, "num_tokens": 261100}
    ]
  }
}
```

The critical difference is `last_token_shards`. This integer tells you that shards `0` through `last_token_shards - 1` are **last-token shards**. They contain exactly one vector per prompt (the last-token activation), just like a pooled dataset. The universal `shard_index`/`row_offset` columns in the parquet always point into these shards.

Shards from index `last_token_shards` onward are **sequence shards**. They contain the remaining tokens in variable-length chunks. These are addressed by the `token_shard_ids`/`token_shard_offsets` list columns in the parquet.

In last-token shards, `num_tokens` always equals `num_prompts` (one token per prompt). In sequence shards, `num_tokens` reflects the total token count across all prompts packed into that shard.

This design means a researcher doing last-token work downloads only the small last-token shards and never touches the bulk sequence data.

### Logits (top-k)

```json
{
  "logits_topk": {
    "type": "logits_topk",
    "k": 50,
    "dtype": "float32",
    "pooling": "last_token",
    "file_pattern": "tensors/logits_topk_{shard:03d}.safetensors",
    "row_bytes": 600,
    "shards": [
      {"file": "tensors/logits_topk_000.safetensors", "num_prompts": 500}
    ]
  }
}
```

Logits shards store both values and indices for the top-k tokens at the last token position. Addressed via the `shard_index_logits`/`row_offset_logits` parquet columns.

---

## Resolving a prompt to a tensor

### Last-token hidden state (the common path)

```python
import pyarrow.parquet as pq
import json
from safetensors import safe_open

# 1. Load the index
table = pq.read_table("index/train-00000-of-00001.parquet")
df = table.to_pandas()

# 2. Read the tensor descriptor
meta = table.schema.metadata
tensors = json.loads(meta[b"lmprobe:tensors"])
hidden = tensors["hidden_layers"]

# 3. Pick a prompt and a layer
row = df.iloc[42]
layer = 16

# 4. Construct the filename and key
filename = hidden["file_pattern"].format(layer=layer, shard=row["shard_index"])
key = hidden["key_pattern"].format(layer=layer)

# 5. Read the vector
with safe_open(filename, framework="pt") as f:
    vector = f.get_tensor(key)[row["row_offset"]]

# vector is shape (hidden_dim,) as the last-token activation for prompt 42 at layer 16
```

This works identically for pooled and full-sequence datasets. The `shard_index`/`row_offset` contract guarantees it.

### Per-token activation (full-sequence only)

```python
# For token i of prompt 42:
row = df.iloc[42]
token_idx = 5
layer = 16

shard = row["token_shard_ids"][token_idx]
offset = row["token_shard_offsets"][token_idx]

filename = hidden["file_pattern"].format(layer=layer, shard=shard)
key = hidden["key_pattern"].format(layer=layer)

with safe_open(filename, framework="pt") as f:
    vector = f.get_tensor(key)[offset]
```

### Logits

```python
logits_desc = tensors["logits_topk"]
row = df.iloc[42]

filename = logits_desc["file_pattern"].format(shard=row["shard_index_logits"])

with safe_open(filename, framework="pt") as f:
    # Contains top-k values and indices
    values = f.get_tensor("topk_values")[row["row_offset_logits"]]
    indices = f.get_tensor("topk_indices")[row["row_offset_logits"]]
```

---

## File layout on disk

A typical dataset on HuggingFace:

```
repo/
├── index/
│   └── train-00000-of-00001.parquet
├── tensors/
│   ├── hidden_layer000_shard000.safetensors
│   ├── hidden_layer000_shard001.safetensors
│   ├── hidden_layer001_shard000.safetensors
│   ├── ...
│   └── logits_topk_000.safetensors
└── README.md
```

Shard filenames are fully determined by the `file_pattern` in the tensor descriptor. No naming convention is assumed. The pattern is the contract.

---

## Design rationale

**Why shards are split by layer.** Nearly all interpretability workflows (probes, autoencoders, steering vectors, etc.) operate on one layer at a time. Per-layer sharding means a researcher downloads only the layers they need.

**Why last-token shards exist.** Last-token pooling is the dominant access pattern for probes and classifiers. For full-sequence datasets, dedicating small shards to last-token vectors avoids downloading gigabytes of per-token data for the most common workflow.

**Why `shard_index`/`row_offset` always means last-token.** A single, universal addressing contract means `df.head()` is immediately legible. A researcher seeing these columns knows what they resolve to without checking which storage mode the dataset uses. Full-sequence and per-token addressing is available through additional columns, but the default path is always simple.

**Why metadata is embedded in the parquet.** A researcher downloads one file and has everything: the data, the schema, and the full description of the tensor layout. No sidecar files, no separate config downloads, no implicit conventions.

---

## Version history

**2.0** (current): Unified `shard_index`/`row_offset` contract. Explicit `file_pattern` and `key_pattern` in tensor descriptors. Embedded schema metadata with `lmprobe:` prefix.

**1.x** (before-times): Not documented, never used by anyone important.
