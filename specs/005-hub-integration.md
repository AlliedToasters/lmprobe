# 005: HuggingFace Hub Integration & ProbeCard

**Status**: Proposed  
**Date**: 2026-03-11  
**Author**: Claude (drafted), Toast (review pending)

## Context

A trained `lmprobe` probe is small (kilobytes — a classifier weight vector plus metadata) but valuable. Currently, sharing a probe requires exchanging `.pkl` files, which are:

1. **Not reproducible** — the pickle captures the classifier but not the full provenance chain (which model commit? what lmprobe version? what training data fingerprint?)
2. **Not safe** — pickle allows arbitrary code execution on load
3. **Not discoverable** — no central registry, no search, no metadata

HuggingFace Hub solves all three. A probe becomes a Hub repo with structured metadata, safe serialization, and community discoverability. This enables a shared safety-probe ecosystem: researchers publish detectors, practitioners pull them, auditors run standardized batteries.

## Design Goals

1. **Reproducibility first** — anyone loading a probe should be able to verify it was trained correctly, understand its limitations, and reproduce the result from scratch
2. **Safe serialization** — no pickle on the wire; use JSON config + `skops` for sklearn objects
3. **Minimal new concepts** — `push_to_hub` / `from_hub` feel like the existing `save` / `load`
4. **Fit the HuggingFace ecosystem** — proper model card, `library_name: lmprobe` tag, `base_model` field, filterable tags

## Repository Layout on Hub

```
alliedtoasters/cbrn-detector-llama3-8b/
├── README.md              # Model card (auto-generated, editable)
├── probe_config.json      # Full config to reconstruct the probe
├── classifier.skops       # Trained sklearn classifier (safe format)
├── scaler.skops           # PerLayerScaler if used (optional)
└── training_info.json     # Reproducibility record (optional but encouraged)
```

### Why these files?

**`probe_config.json`** — everything needed to reconstruct an empty probe with the right architecture. This is the "recipe" independent of trained weights. Separating config from weights means someone can inspect the probe's setup without loading any binary.

**`classifier.skops`** — the trained sklearn estimator, serialized via [`skops`](https://github.com/skops-dev/skops) instead of pickle. `skops` uses a safe format that doesn't allow arbitrary code execution and supports type-checking on load. This is the only binary artifact.

**`scaler.skops`** — optional. Present only when `normalize_layers` produced a fitted `PerLayerScaler`. Same safe serialization.

**`training_info.json`** — the reproducibility record. Not required to *use* the probe, but required to *trust* it. Contains everything needed to re-derive the classifier from scratch.

**`README.md`** — a HuggingFace model card with YAML frontmatter for discoverability, plus human-readable documentation of what the probe does, how it was trained, and its limitations.

## Decision

### 1. `probe_config.json` Schema

This mirrors the constructor arguments and fitted state:

```json
{
  "lmprobe_version": "0.5.0",
  "config_version": 1,

  "base_model": {
    "name": "meta-llama/Llama-3.1-8B-Instruct",
    "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
  },

  "probe": {
    "layers": [16],
    "layers_spec_original": 16,
    "selected_layers": null,
    "pooling": "last_token",
    "train_pooling": "last_token",
    "inference_pooling": "last_token",
    "normalize_layers": true,
    "classifier_type": "logistic_regression",
    "task": "classification",
    "random_state": 42,
    "batch_size": 8,
    "backend": "local",
    "dtype": null,
    "serialization_format": "skops"
  },

  "classes": [0, 1],
  "class_labels": {
    "0": "negative",
    "1": "positive"
  },

  "has_scaler": false
}
```

Key design choices:

- **`base_model.revision`** is the git commit hash of the HuggingFace model repo at training time. This is critical — the same model name can point to different weights after an update. We resolve this automatically at `push_to_hub` time by querying the Hub API for the current commit of the model repo.
- **`layers_spec_original`** preserves what the user passed (e.g. `"fast_auto"`, `16`, `[14, 15, 16]`), while **`layers`** records the resolved integer indices that were actually used. This lets someone understand intent *and* reproduce exactly.
- **`class_labels`** is optional but strongly encouraged. For safety probes especially, the meaning of class 0 vs class 1 must be unambiguous.
- **`config_version`** allows schema evolution without breaking old probes.
- **`serialization_format`** records how the classifier was serialized (`"skops"`, `"joblib"`, or in the future `"safetensors"` for neural classifiers). `from_hub` dispatches on this field. See section 8 for the forward-compatibility design.

### 2. `training_info.json` — The Reproducibility Record

This file answers: "Could I reproduce this probe from scratch?"

```json
{
  "training_data": {
    "n_positive": 50,
    "n_negative": 50,
    "positive_hash": "sha256:a1b2c3d4...",
    "negative_hash": "sha256:e5f6a7b8...",
    "positive_examples": [
      "Who wants to go for a walk?",
      "My tail is wagging with delight."
    ],
    "negative_examples": [
      "Enjoys lounging in the sun beam all day.",
      "Purring, stalking, pouncing, scratching."
    ]
  },

  "evaluation": {
    "metrics": {
      "accuracy": 0.94,
      "auroc": 0.97,
      "f1": 0.93
    },
    "eval_set_size": 20,
    "eval_hash": "sha256:c9d0e1f2..."
  },

  "training_environment": {
    "lmprobe_version": "0.5.0",
    "python_version": "3.11.5",
    "torch_version": "2.2.0",
    "sklearn_version": "1.4.0",
    "transformers_version": "4.38.0",
    "device": "cuda:0",
    "gpu": "NVIDIA A100-SXM4-80GB"
  },

  "timestamps": {
    "trained_at": "2026-03-11T14:30:00Z",
    "pushed_at": "2026-03-11T15:00:00Z"
  }
}
```

Key design choices:

- **Training prompts are included by default** but can be excluded via `include_training_data=False`. For safety probes, sharing the contrastive pairs is essential — they define the concept being detected. For proprietary applications, users can opt out and only the hashes will be stored.
- **Hashes of the training data** are always stored regardless. The hash is computed as `sha256(sorted(prompts).encode())`. This lets someone verify they have the same data even if the prompts themselves aren't shared.
- **Evaluation metrics** are optional but the schema provides a standard place for them. `push_to_hub` checks for cached results from `probe.evaluate()` (see section 6) and auto-populates the card if present. Users can also pass a `metrics` dict explicitly to `push_to_hub` to override. Metrics are never auto-generated — if no evaluation has been run, the model card renders "No evaluation provided" with a nudge to run `probe.evaluate()` before publishing.
- **Environment capture** happens automatically by reading installed package versions at push time. This is the cheapest form of reproducibility — not a full lockfile, but enough to flag version incompatibilities.

### 3. Model Card (README.md) — Auto-Generated

The model card is generated from `probe_config.json` and `training_info.json` using a template. It uses HuggingFace's YAML frontmatter for metadata:

```yaml
---
library_name: lmprobe
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
  - lmprobe
  - linear-probe
  - ai-safety
  - cbrn
pipeline_tag: text-classification
license: mit
metrics:
  - accuracy
  - auroc
model-index:
  - name: cbrn-detector-llama3-8b
    results:
      - task:
          type: text-classification
          name: CBRN Detection
        metrics:
          - name: AUROC
            type: auroc
            value: 0.97
          - name: Accuracy
            type: accuracy
            value: 0.94
---
```

The body of the card includes:

- **What this probe detects** (from `class_labels` and user-provided description)
- **How to use it** (2-line code snippet with `from_hub`)
- **Base model and layer info**
- **Training data summary** (N positive/negative, examples if included)
- **Evaluation results**
- **Limitations and intended use** (template section the user should fill in)
- **Reproducibility** (lmprobe version, environment, data hashes)

### 4. API Design

#### Publishing

```python
# Recommended flow: fit, evaluate, push
probe.fit(pos, neg)
probe.evaluate(test_prompts, test_labels)   # caches metrics on probe
probe.push_to_hub("alliedtoasters/cbrn-detector-llama3-8b")  # auto-populates card

# Minimal (no evaluation — card will note "No evaluation provided")
probe.fit(pos, neg)
probe.push_to_hub("alliedtoasters/cbrn-detector-llama3-8b")

# Full options
probe.push_to_hub(
    repo_id="alliedtoasters/cbrn-detector-llama3-8b",
    description="Detects CBRN-related queries in Llama 3.1 8B",
    class_labels={0: "benign", 1: "cbrn_related"},
    tags=["safety", "cbrn"],
    metrics={"auroc": 0.97, "accuracy": 0.94},  # overrides cached evaluate() results
    include_training_data=True,     # include prompts in training_info.json (default)
    training_prompts=(pos, neg),    # if not cached from fit()
    private=False,
    license="mit",
    commit_message="Initial probe release",
)
```

Implementation notes:

- `push_to_hub` calls `_check_fitted()` first — can't push an untrained probe.
- If the user called `fit(pos, neg)`, we cache the training prompts on the probe instance (`self._training_positive_`, `self._training_negative_`) so they're available at push time without re-supplying them. These are serialized to `training_info.json` when `include_training_data=True` (the default).
- If the user called `probe.evaluate()`, the cached `self._evaluation_results_` are serialized to the `evaluation` block of `training_info.json` and rendered on the model card. Explicit `metrics=` kwarg overrides cached results.
- The base model revision is resolved at push time via `huggingface_hub.model_info(self.model).sha`.
- The method creates the repo, serializes all files to a temp directory, and calls `upload_folder`.

#### Loading

```python
# Lightweight: load probe only, no base model download
# Probe is in a "ready but model-less" state — classifier is loaded,
# config is loaded, but ActivationExtractor isn't initialized until
# the user calls predict() with a model available.
probe = LinearProbe.from_hub(
    "alliedtoasters/cbrn-detector-llama3-8b",
    trust_classifier=True,
)
probe.predict(["some new text"])  # requires base model already available locally

# Full setup: download base model and initialize everything
# Passes the pinned base_model.revision to transformers' from_pretrained(),
# so the user gets exactly the right model weights for reproducibility.
probe = LinearProbe.from_hub(
    "alliedtoasters/cbrn-detector-llama3-8b",
    load_model=True,            # downloads base model if not cached
    trust_classifier=True,
    device="cpu",               # override device
)
probe.predict(["some new text"])  # works immediately

# Pin a specific Hub commit of the probe repo itself
probe = LinearProbe.from_hub(
    "alliedtoasters/cbrn-detector-llama3-8b",
    revision="abc123",          # specific probe repo commit
    trust_classifier=True,
)
```

Implementation notes:

- `from_hub` downloads the probe repo via `snapshot_download`, reads `probe_config.json`, constructs a `LinearProbe` with the saved config, then loads the classifier from `classifier.skops`.
- **`load_model` parameter** (default `False`): when `False`, the probe is returned without initializing the `ActivationExtractor`. The extractor is lazily initialized on the first call to `predict()`, at which point the base model must be available locally. When `True`, `from_hub` eagerly initializes the extractor, downloading the base model via transformers' `from_pretrained()` if needed. The pinned `base_model.revision` is passed through, ensuring exact weight reproducibility.
- **Model validation**: on load (or on first `predict()` if lazy), we compare the `base_model.name` in config against the model the user will run inference with. If they don't match, we **warn** but don't error — the user may intentionally be testing cross-model transfer. If the model is not available and `load_model=False`, `predict()` raises a clear error: `"This probe was trained on meta-llama/Llama-3.1-8B-Instruct (revision abc123). Pass load_model=True to from_hub() to download it, or ensure the model is available locally."`
- **Revision validation**: if the user has a different revision of the base model checked out locally, we warn. We don't error because the user may not have the exact commit available.
- **`trust_classifier` is always required.** `skops` requires explicit trust for deserialization, and we always require the user to pass `trust_classifier=True` to acknowledge they trust the publisher — even for probes using built-in classifier types. The `classifier_type` field in `probe_config.json` is attacker-controlled data (anyone can fork lmprobe, strip validation from `push_to_hub`, and upload a malicious file claiming to be `logistic_regression`), so we cannot use it as a security decision on the load side. The error message when `trust_classifier` is missing should be informative: `"This probe was published by 'alliedtoasters' and declares a LogisticRegression classifier. Pass trust_classifier=True to load. See https://... for security details."` This is analogous to `trust_remote_code` in transformers.

#### Metadata Inspection (without loading the model)

```python
from lmprobe import ProbeCard

card = ProbeCard.from_hub("alliedtoasters/cbrn-detector-llama3-8b")
print(card.base_model)          # "meta-llama/Llama-3.1-8B-Instruct"
print(card.layers)              # [16]
print(card.metrics)             # {"auroc": 0.97, "accuracy": 0.94}
print(card.class_labels)        # {0: "benign", 1: "cbrn_related"}
print(card.training_data_hash)  # "sha256:a1b2c3d4..."
print(card.lmprobe_version)     # "0.5.0"
```

`ProbeCard` is a lightweight dataclass that reads `probe_config.json` + `training_info.json` without downloading the classifier weights. This enables fast registry browsing and compatibility checks.

### 5. Reproducibility Verification

A key feature: given a probe on the Hub, can I verify it?

```python
# Reproduce from scratch
probe = LinearProbe.from_hub(
    "alliedtoasters/cbrn-detector-llama3-8b",
    load_model=True,
    trust_classifier=True,
)
card = ProbeCard.from_hub("alliedtoasters/cbrn-detector-llama3-8b")

# Re-train with the published training data
fresh_probe = LinearProbe(**card.to_reproduce_config())
fresh_probe.fit(card.positive_examples, card.negative_examples)

# Compare
original_preds = probe.predict_proba(test_prompts)
fresh_preds = fresh_probe.predict_proba(test_prompts)
np.allclose(original_preds, fresh_preds, atol=1e-6)  # True if reproducible
```

This works because:
1. `probe_config.json` has all constructor args including `random_state`
2. `training_info.json` has the training prompts (if included)
3. The base model revision pins the exact model weights (and `load_model=True` passes the revision through to transformers)
4. The environment versions flag any sklearn/torch differences that might affect numerics

For probes where training data is *not* included, the hash still allows verification: "I have data that produces hash X, and the published probe claims to have been trained on data with hash X."

### 6. Evaluation and Metrics — `probe.evaluate()`

`push_to_hub` never auto-generates metrics. Instead, we provide `probe.evaluate()` as a dedicated method that computes a standard set of metrics and caches them on the probe instance. This creates a natural two-step flow: evaluate, then push.

```python
probe.fit(pos, neg)

# Compute and cache metrics
results = probe.evaluate(test_prompts, test_labels)
print(results)
# {
#   "accuracy": 0.94,
#   "auroc": 0.97,
#   "f1": 0.93,
#   "precision": 0.92,
#   "recall": 0.95,
#   "n_eval": 20,
#   "eval_hash": "sha256:c9d0e1f2..."
# }

# Metrics flow into push_to_hub automatically
probe.push_to_hub("alliedtoasters/cbrn-detector-llama3-8b")

# Or override explicitly
probe.push_to_hub(
    "alliedtoasters/cbrn-detector-llama3-8b",
    metrics={"auroc": 0.97, "accuracy": 0.94},  # overrides cached results
)
```

Implementation notes:

- `evaluate()` returns a dict and caches it on `self._evaluation_results_`. This is separate from `score()`, which remains the simple sklearn-compatible single-metric return (accuracy by default).
- `evaluate()` computes: accuracy, AUROC (if `predict_proba` is available), F1, precision, recall. It also records `n_eval` (number of evaluation samples) and `eval_hash` (hash of the evaluation prompts+labels, for reproducibility).
- `push_to_hub` checks for `self._evaluation_results_` and includes them in `training_info.json` if present. If no evaluation has been run *and* no `metrics` kwarg is passed, the model card renders an "Evaluation" section with the text: "No evaluation results provided. Consider running `probe.evaluate(test_prompts, test_labels)` before publishing."
- `evaluate()` does **not** use the training data. If the user passes the same data they trained on, that's their business, but we don't facilitate it — there is no `evaluate_on_train()` shortcut.

### 7. Caching Training Prompts During `fit()`

To make `push_to_hub` seamless, `fit()` should cache the training data:

```python
def fit(self, positive_prompts, negative_prompts=None, ...):
    # ... existing logic ...

    # Cache for optional push_to_hub later
    if negative_prompts is not None:
        self._training_positive_ = list(positive_prompts)
        self._training_negative_ = list(negative_prompts)
    else:
        self._training_prompts_ = list(positive_prompts)
        self._training_labels_ = list(negative_prompts_or_labels)
```

These are **not** included in the existing `save()` pickle (to avoid bloating local saves). They're only used by `push_to_hub` and are garbage-collected with the probe instance.

### 8. Safe Serialization with `skops`

Why `skops` over pickle:
- `skops.io.dumps` / `skops.io.loads` produce a safe binary format that can be inspected before loading
- The format records the exact types being deserialized
- Loading requires the user to explicitly pass `trust_classifier=True` (see Resolved Decision #2)
- The format is deterministic for the same classifier state

Fallback: if the user has a custom sklearn estimator that `skops` can't handle, we fall back to `joblib` with a clear warning that the probe will require `trust_classifier=True` to load and that pickle-based serialization is less safe.

```python
# In push_to_hub:
try:
    import skops.io as sio
    sio.dump(self.classifier_, classifier_path)
    serialization_format = "skops"
except Exception:
    import joblib
    joblib.dump(self.classifier_, classifier_path)
    serialization_format = "joblib"
    warnings.warn("Classifier serialized with joblib (pickle-based). ...")
```

The `probe_config.json` records `"serialization_format": "skops"` or `"joblib"` so `from_hub` knows which loader to use.

#### Forward-Compatibility: Neural Classifiers

The `serialization_format` field is an intentional extension point. A future work stream will add PyTorch-based classifiers (MLP probes, etc.) as a separate classifier type. When that lands, the serialization path will be:

- `"safetensors"` — weights stored via `safetensors`, architecture stored in a `classifier_arch.json` file
- On load: reconstruct `nn.Module` from arch config (lmprobe's own code), load state dict from safetensors
- No code execution possible — `safetensors` stores only tensors, architecture is rebuilt from a JSON spec
- `trust_classifier=True` would **not** be required for safetensors probes, since the format is safe by construction

For now, `from_hub` should dispatch on `serialization_format` and raise `NotImplementedError` for unrecognized values. This ensures probes published by a future lmprobe version produce a clear error on older versions rather than a confusing deserialization failure.

```python
# In from_hub:
fmt = config["probe"]["serialization_format"]
if fmt == "skops":
    classifier = _load_skops(classifier_path, trust_classifier)
elif fmt == "joblib":
    classifier = _load_joblib(classifier_path, trust_classifier)
elif fmt == "safetensors":
    raise NotImplementedError(
        "This probe uses a neural classifier (safetensors format). "
        "Upgrade lmprobe to load it: pip install --upgrade lmprobe"
    )
else:
    raise ValueError(f"Unknown serialization format: {fmt!r}")
```

### 9. Dependency Management

`huggingface_hub` and `skops` become optional dependencies:

```toml
[project.optional-dependencies]
hub = [
    "huggingface_hub>=0.20",
    "skops>=0.9",
]
# Future: when neural classifiers land
# hub-neural = [
#     "huggingface_hub>=0.20",
#     "safetensors>=0.4",
# ]
```

`push_to_hub` and `from_hub` check for these at call time with a clear error message:

```python
def push_to_hub(self, repo_id, ...):
    try:
        from huggingface_hub import HfApi, ModelCard
        import skops.io as sio
    except ImportError:
        raise ImportError(
            "Hub integration requires: pip install lmprobe[hub]"
        )
```

### 10. ProbeCard Dataclass

```python
@dataclass
class ProbeCard:
    """Lightweight metadata container for a Hub-hosted probe.

    Reads probe_config.json and training_info.json without
    downloading classifier weights.
    """
    # From probe_config.json
    base_model: str
    base_model_revision: str | None
    layers: list[int]
    layers_spec_original: int | list[int] | str
    pooling: str
    train_pooling: str
    inference_pooling: str
    classifier_type: str
    task: str
    random_state: int | None
    classes: list
    class_labels: dict[str, str] | None
    lmprobe_version: str
    config_version: int

    # From training_info.json (may be None if not published)
    n_positive: int | None
    n_negative: int | None
    positive_hash: str | None
    negative_hash: str | None
    positive_examples: list[str] | None
    negative_examples: list[str] | None
    metrics: dict[str, float] | None
    training_environment: dict | None
    trained_at: str | None

    @classmethod
    def from_hub(cls, repo_id: str, revision: str | None = None) -> ProbeCard:
        ...

    @classmethod
    def from_local(cls, path: str) -> ProbeCard:
        ...

    def is_compatible_with(self, model: str) -> bool:
        """Check if this probe was trained on the given model."""
        return self.base_model == model

    def to_reproduce_config(self) -> dict:
        """Return kwargs suitable for LinearProbe(...) constructor."""
        return {
            "model": self.base_model,
            "layers": self.layers_spec_original,
            "pooling": self.pooling,
            "train_pooling": self.train_pooling,
            "inference_pooling": self.inference_pooling,
            "classifier": self.classifier_type,
            "task": self.task,
            "random_state": self.random_state,
        }
```

## Consequences

**Good**:
- Probes become discoverable, citable, and version-controlled
- Safe serialization eliminates pickle-based attack surface
- Reproducibility record creates accountability for safety claims ("this probe has AUROC 0.97" is now verifiable)
- Enables a community registry of safety probes
- `ProbeCard` allows lightweight browsing/filtering without downloading weights
- `base_model.revision` pins exact model weights, preventing silent drift

**Good (ecosystem)**:
- `library_name: lmprobe` tag makes probes findable on HuggingFace
- `base_model` field links probes to their parent model's page
- `model-index` enables leaderboard integration if HF adds probe benchmarks
- Standard `pipeline_tag: text-classification` enables inference API integration (future)

**Caution**:
- `skops` is a relatively young library — need to monitor stability
- `trust_classifier=True` is always required, which adds friction to the load path. This is intentional — the security benefit outweighs the ergonomic cost, and the pattern is already familiar from `trust_remote_code` in transformers
- Training data inclusion defaults to `True`; users building proprietary classifiers must remember to pass `include_training_data=False`
- Model revision pinning assumes the base model is on HuggingFace; local models need a different strategy (deferred — see Open Questions)
- `training_info.json` can grow large if training prompts are very long or numerous; consider a size cap or separate file for data

**Migration**:
- Existing `.pkl` probes saved with `probe.save()` still work via `LinearProbe.load()`
- No breaking changes to the existing API
- `push_to_hub` / `from_hub` are purely additive

## Resolved Design Decisions

1. **`from_hub` does not auto-download the base model.** Default is `load_model=False`, returning a probe in a "ready but model-less" state — the classifier is loaded but the `ActivationExtractor` isn't initialized until `predict()` is called. For users who want full end-to-end setup, `load_model=True` eagerly downloads the base model (passing the pinned `base_model.revision` through to transformers) and initializes the extractor. Clear error messages guide the user when the model is missing.

2. **`trust_classifier=True` is always required for `from_hub`.** No auto-trust shortcut, even for built-in classifier types. The `classifier_type` field in `probe_config.json` is attacker-controlled data — anyone can fork lmprobe, strip push-side validation, and upload a malicious file that claims to be `logistic_regression`. The trust boundary must be enforced entirely on the load side, which means the user always explicitly acknowledges trust. The error message is informative, naming the publisher and declared classifier type to help the user make the decision.

3. **`push_to_hub` never auto-generates evaluation metrics.** Instead, a dedicated `probe.evaluate()` method computes a standard metric set (accuracy, AUROC, F1, precision, recall) and caches the results on `self._evaluation_results_`. `push_to_hub` checks for cached results and auto-populates the model card if present. If no evaluation has been run, the card renders "No evaluation provided" — the visible absence is itself useful signal. Explicit `metrics=` kwarg to `push_to_hub` overrides cached results.

4. **Tag convention for safety probes, not a namespace.** Tags are decentralized and require no governance overhead. Recommended convention: `lmprobe` (always present, automatic) plus domain tags like `safety-cbrn`, `safety-deception`, `safety-social-engineering`. These go in the YAML frontmatter and are filterable on HuggingFace. A curated collection can be created later if the community grows.

5. **Training data hashing sorts prompts before hashing.** The hash is `sha256(sorted(prompts).encode())`, making it invariant to list ordering. This is the right semantics for verification — two researchers who independently assembled the same prompt set get the same hash regardless of order. Positive and negative sets are hashed separately to preserve class identity.

## Open Questions

1. **`include_training_data` default.** Current proposal: `True`, matching the library's AI safety transparency mission. Sophisticated users building proprietary classifiers can pass `False`. A safety probe with undisclosed training data is still publishable, but the visible absence of training prompts on the model card is itself useful signal for consumers. This default could be revisited if adoption extends significantly beyond the safety use case.

2. **Local model pinning.** `base_model.revision` assumes the base model is on HuggingFace. For local/private models, we need a different strategy. Options include hashing the model weights (expensive but exact) or accepting a user-provided model identifier with no revision check (pragmatic but less reproducible). Deferred until there's demand.

## References

- HuggingFace Hub integration guide: https://huggingface.co/docs/hub/en/models-adding-libraries
- `skops` documentation: https://skops.readthedocs.io/
- HuggingFace ModelCard spec: https://huggingface.co/docs/hub/model-cards
- `huggingface_hub` ModelHubMixin: https://huggingface.co/docs/huggingface_hub/guides/integrations
- Anthropic, "Probes Catch Sleeper Agents" (2024)
- Apollo Research, "Strategic Deception Detection" (2025)