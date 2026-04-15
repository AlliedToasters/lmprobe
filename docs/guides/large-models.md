# Large Model Inference

`lmprobe` provides two backends for probing models that don't fit in GPU VRAM. The **chunked** backend handles models that fit in CPU RAM, while the **disk offload** backend handles models that exceed even CPU RAM — up to hundreds of billions of parameters on a single GPU.

| Backend | Best for | Memory requirement |
|---------|----------|-------------------|
| `"local"` | Models that fit in GPU VRAM | GPU VRAM > model size |
| `"chunked"` | Models that fit in CPU RAM | CPU RAM > model size |
| `"disk_offload"` | Models that exceed CPU RAM | Disk > model size |

---

## Chunked backend

The chunked backend loads the full model on CPU, then streams layers through the GPU in chunks. Use it for models like Llama-3.1-70B or Mixtral-8x22B that fit in CPU RAM but not GPU VRAM.

```python
from lmprobe import Probe

probe = Probe(
    model="mistralai/Mixtral-8x22B-v0.1",
    layers=16,
    backend="chunked",          # load on CPU, chunk through GPU
    dtype="bfloat16",           # bf16 recommended for large models
)

probe.fit(positive_prompts, negative_prompts)
```

The chunk size is estimated automatically from available VRAM. Override it if needed:

```python
probe = Probe(
    model="mistralai/Mixtral-8x22B-v0.1",
    layers=16,
    backend="chunked",
    dtype="bfloat16",
    chunk_size=4,               # 4 layers at a time on GPU
)
```

!!! tip "When to use chunked vs local"
    If your model loads successfully with `backend="local"` but you're running out of VRAM during extraction, switch to `"chunked"`. The trade-off is speed — each batch requires moving layers between CPU and GPU.

---

## Disk offload backend

For models that don't fit in CPU RAM either (e.g. DeepSeek-V3 at 671B parameters / 642GB), the disk offload backend loads layer weights directly from safetensors files to GPU, one layer at a time. It never materializes the full model in memory.

```python
from lmprobe.backends import DiskOffloadBackend
from lmprobe.activation_types import ExtractionSpec, detect_moe_info

backend = DiskOffloadBackend(
    "deepseek-ai/DeepSeek-V3-Base",
    device="cuda:0",
)

# Detect MoE structure automatically
moe = detect_moe_info("deepseek-ai/DeepSeek-V3-Base")

spec = ExtractionSpec(
    hidden_layers=[0, 10, 20, 30],
    router_layers=moe.moe_layer_indices,
    router_module_template=moe.router_module_template,
    router_hook_strategy=moe.router_hook_strategy,
)

result = backend.extract_all(
    prompts,
    spec,
    batch_size=16,
)
# result.hidden_per_layer[layer_idx] -> (N, seq_len, hidden_dim)
# result.router_logits[layer_idx]    -> (N, seq_len, n_experts)
```

### Layer-amortized extraction

The key optimization is `extract_all`: it processes the **entire dataset** through each layer before moving to the next. Each layer's weights are loaded from disk exactly once, regardless of dataset size.

```
Layer 0:  load weights -> run all batches -> free weights
Layer 1:  load weights -> run all batches -> free weights
...
Layer 60: load weights -> run all batches -> free weights
```

This is critical for large models where weight loading dominates runtime. For DeepSeek-V3, loading one MoE layer takes ~18 seconds. Processing per-batch through an already-loaded layer is fast (GPU compute only). The amortized approach means 20,000 prompts take roughly the same time as 100.

### Mean pooling

For large-scale extraction where storing full `(N, seq_len, hidden_dim)` tensors per layer would exceed memory, use `pool="mean"` to mean-pool over valid tokens as features are captured:

```python
result = backend.extract_all(
    prompts,
    spec,
    batch_size=16,
    pool="mean",                # store (N, dim) per layer, not (N, seq, dim)
)
# result.hidden_per_layer[layer_idx] -> (N, hidden_dim)
# result.router_logits[layer_idx]    -> (N, n_experts)
```

This reduces memory from `O(N * seq * layers * dim)` to `O(N * layers * dim)` — essential when combining multiple datasets in a single extraction pass.

### FP8 quantized models

The disk offload backend handles FP8 block-quantized weights (e.g. DeepSeek-V3's native `e4m3` format) automatically. Weights are dequantized to bf16 during loading using the companion `weight_scale_inv` tensors. No manual configuration is needed.

For MoE architectures, expert weights stored as individual per-expert tensors in safetensors are automatically packed into the 3D format expected by the model's forward pass.

!!! note "GPU requirements"
    Peak VRAM usage depends on the largest single layer. For DeepSeek-V3 (256 experts, 2048 intermediate), one MoE layer in bf16 requires ~22GB. A GPU with 24-32GB VRAM is sufficient.

---

## Combining multiple datasets

A common pattern for probing experiments is extracting features from many datasets at once. With the disk offload backend, combining datasets into a single `extract_all` call is much faster than processing them separately:

```python
# Combine all prompts
all_prompts = []
slices = {}
for name, prompts in datasets.items():
    start = len(all_prompts)
    all_prompts.extend(prompts)
    slices[name] = (start, len(all_prompts))

# Single extraction pass — each layer loaded once
result = backend.extract_all(all_prompts, spec, batch_size=16, pool="mean")

# Slice results per dataset
for name, (start, end) in slices.items():
    hidden = result.hidden_per_layer[layer_idx][start:end]
    router = result.router_logits[layer_idx][start:end]
    # ... run probes on this dataset
```

---

## Backend comparison

| | `"local"` | `"chunked"` | `"disk_offload"` |
|---|---|---|---|
| Model in GPU VRAM | Full | No | No |
| Model in CPU RAM | Full | Full | No |
| Weight loading | Once | Per batch | Per layer (amortized) |
| Speed (per batch) | Fast | Moderate | Slow per layer, fast per batch |
| FP8 support | Via transformers | Via transformers | Native dequantization |
| MoE routing extraction | Yes | Yes | Yes |
| Max tested model size | ~30B | ~70B | 671B |
