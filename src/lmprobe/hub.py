"""HuggingFace Hub integration for lmprobe probes.

Provides push_to_hub, from_hub, and ProbeCard for sharing and loading
trained probes via the HuggingFace Hub ecosystem.
"""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from .probe import LinearProbe


def _check_hub_deps() -> None:
    """Check that hub optional dependencies are installed."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        raise ImportError(
            "Hub integration requires huggingface_hub. "
            "Install with: pip install lmprobe[hub]"
        )
    try:
        import skops  # noqa: F401
    except ImportError:
        raise ImportError(
            "Hub integration requires skops for safe serialization. "
            "Install with: pip install lmprobe[hub]"
        )


def _hash_prompts(prompts: list[str]) -> str:
    """Compute a deterministic hash of a list of prompts.

    Sorts prompts before hashing for order-invariance.
    """
    content = "\n".join(sorted(prompts))
    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()


def _serialize_classifier(classifier: BaseEstimator, path: Path) -> str:
    """Serialize an sklearn classifier to disk.

    Tries skops first for safe serialization, falls back to joblib.

    Parameters
    ----------
    classifier : BaseEstimator
        Fitted sklearn classifier.
    path : Path
        Output file path (without extension).

    Returns
    -------
    str
        Serialization format used: "skops" or "joblib".
    """
    try:
        import skops.io as sio

        skops_path = path.with_suffix(".skops")
        sio.dump(classifier, skops_path)
        return "skops"
    except Exception:
        import joblib

        joblib_path = path.with_suffix(".joblib")
        joblib.dump(classifier, joblib_path)
        warnings.warn(
            "Classifier serialized with joblib (pickle-based). "
            "This is less safe than skops serialization. "
            "Consider using a standard sklearn classifier for safe serialization.",
            stacklevel=2,
        )
        return "joblib"


def _load_classifier(
    path: Path, fmt: str, trust_classifier: bool
) -> BaseEstimator:
    """Load a serialized classifier from disk.

    Parameters
    ----------
    path : Path
        Directory containing the classifier file.
    fmt : str
        Serialization format: "skops", "joblib", or "safetensors".
    trust_classifier : bool
        Must be True to load. Required for security.

    Returns
    -------
    BaseEstimator
        The loaded classifier.
    """
    if not trust_classifier:
        raise ValueError(
            "Loading a classifier requires trust_classifier=True. "
            "Pass trust_classifier=True to acknowledge you trust the publisher. "
            "See https://skops.readthedocs.io/ for security details."
        )

    if fmt == "skops":
        import skops.io as sio

        classifier_path = path / "classifier.skops"
        untrusted_types = sio.get_untrusted_types(file=classifier_path)
        return sio.load(classifier_path, trusted=untrusted_types)
    elif fmt == "joblib":
        import joblib

        classifier_path = path / "classifier.joblib"
        return joblib.load(classifier_path)
    elif fmt == "safetensors":
        raise NotImplementedError(
            "This probe uses a neural classifier (safetensors format). "
            "Upgrade lmprobe to load it: pip install --upgrade lmprobe"
        )
    else:
        raise ValueError(f"Unknown serialization format: {fmt!r}")


def _load_scaler(path: Path, fmt: str) -> Any:
    """Load a serialized scaler from disk."""
    if fmt == "skops":
        import skops.io as sio

        scaler_path = path / "scaler.skops"
        untrusted_types = sio.get_untrusted_types(file=scaler_path)
        return sio.load(scaler_path, trusted=untrusted_types)
    elif fmt == "joblib":
        import joblib

        scaler_path = path / "scaler.joblib"
        return joblib.load(scaler_path)
    else:
        raise ValueError(f"Unknown serialization format for scaler: {fmt!r}")


def _build_probe_config(
    probe: LinearProbe,
    class_labels: dict[int, str] | None = None,
) -> dict:
    """Build probe_config.json contents from a fitted probe.

    Parameters
    ----------
    probe : LinearProbe
        A fitted LinearProbe instance.
    class_labels : dict[int, str] | None
        Optional human-readable labels for classes.

    Returns
    -------
    dict
        Config dict suitable for JSON serialization.
    """
    from . import __version__

    # Resolve base model revision
    revision = None
    if probe.model is not None:
        try:
            from huggingface_hub import model_info

            info = model_info(probe.model)
            revision = info.sha
        except Exception:
            pass

    # Get resolved layer indices
    layer_indices = None
    if probe._extractor is not None:
        layer_indices = list(probe._extractor.layer_indices)

    # Resolve pooling names
    train_pooling = getattr(probe, "_train_pooling", probe.pooling)
    inference_pooling = getattr(probe, "_inference_pooling", probe.pooling)

    config = {
        "lmprobe_version": __version__,
        "config_version": 1,
        "base_model": {
            "name": probe.model,
            "revision": revision,
        },
        "probe": {
            "layers": layer_indices,
            "layers_spec_original": probe.layers,
            "selected_layers": probe.selected_layers_,
            "pooling": probe.pooling,
            "train_pooling": train_pooling,
            "inference_pooling": inference_pooling,
            "normalize_layers": probe.normalize_layers,
            "classifier_type": (
                probe.classifier
                if isinstance(probe.classifier, str)
                else type(probe.classifier).__name__
            ),
            "task": probe.task,
            "random_state": probe.random_state,
            "batch_size": probe.batch_size,
            "backend": probe.backend,
            "dtype": probe.dtype,
            "serialization_format": None,  # filled by push_to_hub
        },
        "classes": probe.classes_.tolist() if probe.classes_ is not None else [0, 1],
        "class_labels": (
            {str(k): v for k, v in class_labels.items()}
            if class_labels is not None
            else None
        ),
        "has_scaler": probe.scaler_ is not None,
    }

    return config


def _build_training_info(
    probe: LinearProbe,
    include_training_data: bool = True,
    metrics: dict[str, float] | None = None,
    training_prompts: tuple[list[str], list[str]] | None = None,
) -> dict:
    """Build training_info.json contents.

    Parameters
    ----------
    probe : LinearProbe
        A fitted LinearProbe instance.
    include_training_data : bool
        Whether to include actual training prompts.
    metrics : dict[str, float] | None
        Evaluation metrics to include. If None, uses cached evaluate() results.
    training_prompts : tuple[list[str], list[str]] | None
        (positive, negative) prompts. If None, uses cached training data from fit().

    Returns
    -------
    dict
        Training info dict suitable for JSON serialization.
    """
    import sklearn
    import torch
    import transformers

    from . import __version__

    # Resolve training prompts
    pos_prompts = None
    neg_prompts = None

    if training_prompts is not None:
        pos_prompts, neg_prompts = training_prompts
    elif hasattr(probe, "_training_positive_") and probe._training_positive_ is not None:
        pos_prompts = probe._training_positive_
        neg_prompts = probe._training_negative_
    elif hasattr(probe, "_training_prompts_") and probe._training_prompts_ is not None:
        pos_prompts = probe._training_prompts_
        neg_prompts = None

    # Build training data section
    training_data: dict[str, Any] = {}
    if pos_prompts is not None:
        training_data["n_positive"] = len(pos_prompts)
        training_data["positive_hash"] = _hash_prompts(pos_prompts)
        if include_training_data:
            training_data["positive_examples"] = pos_prompts

    if neg_prompts is not None:
        training_data["n_negative"] = len(neg_prompts)
        training_data["negative_hash"] = _hash_prompts(neg_prompts)
        if include_training_data:
            training_data["negative_examples"] = neg_prompts

    # Resolve metrics
    eval_metrics = metrics
    if eval_metrics is None and hasattr(probe, "_evaluation_results_"):
        eval_metrics = probe._evaluation_results_

    evaluation = None
    if eval_metrics is not None:
        evaluation = {
            "metrics": {
                k: v for k, v in eval_metrics.items()
                if k not in ("n_eval", "eval_hash")
            },
            "eval_set_size": eval_metrics.get("n_eval"),
            "eval_hash": eval_metrics.get("eval_hash"),
        }

    # Environment info
    gpu_info = None
    try:
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
    except Exception:
        pass

    device_str = probe.device if probe.device != "auto" else "cpu"
    if hasattr(probe, "_extractor") and probe._extractor is not None:
        try:
            device_str = str(probe._extractor.device)
        except Exception:
            pass

    training_env = {
        "lmprobe_version": __version__,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "transformers_version": transformers.__version__,
        "device": device_str,
    }
    if gpu_info is not None:
        training_env["gpu"] = gpu_info

    now = datetime.now(timezone.utc).isoformat()

    info = {
        "training_data": training_data if training_data else None,
        "evaluation": evaluation,
        "training_environment": training_env,
        "timestamps": {
            "trained_at": now,
            "pushed_at": now,
        },
    }

    return info


def _format_layers(probe_cfg: dict) -> str:
    """Format layers for display in the model card.

    Uses the original spec (e.g., "all", "middle") when available,
    with resolved count. Falls back to the raw list for short lists.
    """
    layers = probe_cfg.get("layers")
    spec = probe_cfg.get("layers_spec_original")

    # If the original spec is a named string like "all", "middle", "fast_auto"
    if isinstance(spec, str) and layers is not None and isinstance(layers, list):
        first = min(layers)
        last = max(layers)
        n = len(layers)
        return f"{spec} ({first}\u2013{last}, {n} layers)"

    # Short list or single layer — just show it directly
    if isinstance(layers, list) and len(layers) > 10:
        first = min(layers)
        last = max(layers)
        n = len(layers)
        return f"[{first}\u2013{last}] ({n} layers)"

    return str(layers)


def _render_model_card(
    config: dict,
    training_info: dict,
    repo_id: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    license: str = "mit",
    limitations: str | None = None,
) -> str:
    """Generate a HuggingFace model card README.md.

    Parameters
    ----------
    config : dict
        probe_config.json contents.
    training_info : dict
        training_info.json contents.
    repo_id : str | None
        Repository ID for the usage example.
    description : str | None
        Human-readable description of what the probe detects.
    tags : list[str] | None
        Additional tags for discoverability.
    license : str
        License identifier.
    limitations : str | None
        Limitations and intended use text. If None, the section is omitted.

    Returns
    -------
    str
        Complete README.md content with YAML frontmatter.
    """
    # Build YAML frontmatter
    all_tags = ["lmprobe", "linear-probe"]
    if tags:
        all_tags.extend(tags)

    base_model = config["base_model"]["name"] or "unknown"

    yaml_lines = [
        "---",
        "library_name: lmprobe",
        f"base_model: {base_model}",
        "tags:",
    ]
    for tag in all_tags:
        yaml_lines.append(f"  - {tag}")
    yaml_lines.append("pipeline_tag: text-classification")
    yaml_lines.append(f"license: {license}")

    # Add metrics to frontmatter if available
    evaluation = training_info.get("evaluation")
    if evaluation and evaluation.get("metrics"):
        metrics = evaluation["metrics"]
        yaml_lines.append("metrics:")
        for metric_name in sorted(metrics.keys()):
            yaml_lines.append(f"  - {metric_name}")

    yaml_lines.append("---")
    yaml_lines.append("")

    # Build body
    body_lines = []

    # Title
    repo_name = base_model.split("/")[-1] if "/" in base_model else base_model
    body_lines.append(f"# lmprobe: Linear Probe on {repo_name}")
    body_lines.append("")

    # Description
    if description:
        body_lines.append(description)
        body_lines.append("")

    # Class labels
    class_labels = config.get("class_labels")
    if class_labels:
        body_lines.append("## Classes")
        body_lines.append("")
        for cls_id, label in sorted(class_labels.items()):
            body_lines.append(f"- **{cls_id}**: {label}")
        body_lines.append("")

    # Usage
    body_lines.append("## Usage")
    body_lines.append("")
    body_lines.append("```python")
    body_lines.append("from lmprobe import LinearProbe")
    body_lines.append("")
    usage_repo = repo_id if repo_id else "REPO_ID"
    body_lines.append(f'probe = LinearProbe.from_hub("{usage_repo}", trust_classifier=True)')
    body_lines.append('predictions = probe.predict(["your text here"])')
    body_lines.append("```")
    body_lines.append("")

    # Probe details
    probe_cfg = config["probe"]
    body_lines.append("## Probe Details")
    body_lines.append("")
    body_lines.append(f"- **Base model**: `{base_model}`")
    if config["base_model"].get("revision"):
        body_lines.append(f"- **Model revision**: `{config['base_model']['revision']}`")
    body_lines.append(f"- **Layers**: {_format_layers(probe_cfg)}")
    body_lines.append(f"- **Pooling**: {probe_cfg['pooling']}")
    body_lines.append(f"- **Classifier**: {probe_cfg['classifier_type']}")
    body_lines.append(f"- **Task**: {probe_cfg['task']}")
    if probe_cfg.get("random_state") is not None:
        body_lines.append(f"- **Random state**: {probe_cfg['random_state']}")
    body_lines.append("")

    # Evaluation
    body_lines.append("## Evaluation")
    body_lines.append("")
    if evaluation and evaluation.get("metrics"):
        metrics = evaluation["metrics"]
        body_lines.append("| Metric | Value |")
        body_lines.append("|--------|-------|")
        for name, value in sorted(metrics.items()):
            body_lines.append(f"| {name} | {value:.4f} |")
        body_lines.append("")
    else:
        body_lines.append(
            "No evaluation results provided. Consider running "
            "`probe.evaluate(test_prompts, test_labels)` before publishing."
        )
        body_lines.append("")

    # Training data summary
    training_data = training_info.get("training_data")
    if training_data:
        body_lines.append("## Training Data")
        body_lines.append("")
        if "n_positive" in training_data:
            body_lines.append(f"- **Positive examples**: {training_data['n_positive']}")
        if "n_negative" in training_data:
            body_lines.append(f"- **Negative examples**: {training_data['n_negative']}")
        if "positive_hash" in training_data:
            body_lines.append(f"- **Positive hash**: `{training_data['positive_hash']}`")
        if "negative_hash" in training_data:
            body_lines.append(f"- **Negative hash**: `{training_data['negative_hash']}`")
        body_lines.append("")

    # Evaluation data hash (issue #61)
    if evaluation:
        eval_hash = evaluation.get("eval_hash")
        eval_size = evaluation.get("eval_set_size")
        if eval_hash or eval_size:
            if not training_data:
                body_lines.append("## Evaluation Data")
                body_lines.append("")
            if eval_size:
                body_lines.append(f"- **Evaluation samples**: {eval_size}")
            if eval_hash:
                body_lines.append(f"- **Evaluation hash**: `{eval_hash}`")
            body_lines.append("")

    # Reproducibility
    env = training_info.get("training_environment", {})
    body_lines.append("## Reproducibility")
    body_lines.append("")
    body_lines.append(f"- **lmprobe version**: {env.get('lmprobe_version', 'unknown')}")
    body_lines.append(f"- **Python**: {env.get('python_version', 'unknown')}")
    body_lines.append(f"- **PyTorch**: {env.get('torch_version', 'unknown')}")
    body_lines.append(f"- **scikit-learn**: {env.get('sklearn_version', 'unknown')}")
    body_lines.append(f"- **transformers**: {env.get('transformers_version', 'unknown')}")
    body_lines.append("")

    # Limitations (only if provided)
    if limitations:
        body_lines.append("## Limitations and Intended Use")
        body_lines.append("")
        body_lines.append(limitations)
        body_lines.append("")

    return "\n".join(yaml_lines) + "\n".join(body_lines) + "\n"


def push_to_hub(
    probe: LinearProbe,
    repo_id: str,
    description: str | None = None,
    class_labels: dict[int, str] | None = None,
    tags: list[str] | None = None,
    metrics: dict[str, float] | None = None,
    include_training_data: bool = True,
    training_prompts: tuple[list[str], list[str]] | None = None,
    private: bool = False,
    license: str = "mit",
    commit_message: str = "Upload lmprobe probe",
    limitations: str | None = None,
) -> str:
    """Push a fitted probe to the HuggingFace Hub.

    Parameters
    ----------
    probe : LinearProbe
        A fitted LinearProbe instance.
    repo_id : str
        HuggingFace Hub repository ID (e.g., "username/probe-name").
    description : str | None
        Human-readable description.
    class_labels : dict[int, str] | None
        Human-readable class labels.
    tags : list[str] | None
        Additional tags.
    metrics : dict[str, float] | None
        Evaluation metrics (overrides cached evaluate() results).
    include_training_data : bool
        Include training prompts in training_info.json.
    training_prompts : tuple[list[str], list[str]] | None
        (positive, negative) prompts if not cached from fit().
    private : bool
        Create a private repository.
    license : str
        License identifier.
    commit_message : str
        Git commit message for the upload.
    limitations : str | None
        Limitations and intended use text for the model card.
        If None, the section is omitted.

    Returns
    -------
    str
        URL of the created/updated Hub repository.
    """
    # Check fitted before checking deps (better error message)
    probe._check_fitted()

    _check_hub_deps()
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # Serialize classifier
        fmt = _serialize_classifier(probe.classifier_, tmpdir / "classifier")

        # Serialize scaler if present
        if probe.scaler_ is not None:
            _serialize_classifier(probe.scaler_, tmpdir / "scaler")

        # Build config
        config = _build_probe_config(probe, class_labels=class_labels)
        config["probe"]["serialization_format"] = fmt

        # Write config
        with open(tmpdir / "probe_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Build training info
        training_info = _build_training_info(
            probe,
            include_training_data=include_training_data,
            metrics=metrics,
            training_prompts=training_prompts,
        )

        with open(tmpdir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)

        # Render model card
        card = _render_model_card(
            config, training_info,
            repo_id=repo_id,
            description=description,
            tags=tags,
            license=license,
            limitations=limitations,
        )
        with open(tmpdir / "README.md", "w") as f:
            f.write(card)

        # Upload
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True, private=private, repo_type="model")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(tmpdir),
            commit_message=commit_message,
            repo_type="model",
        )

    return f"https://huggingface.co/{repo_id}"


def from_hub(
    repo_id: str,
    revision: str | None = None,
    trust_classifier: bool = False,
    load_model: bool = False,
    device: str | None = None,
) -> LinearProbe:
    """Load a probe from the HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        HuggingFace Hub repository ID.
    revision : str | None
        Specific commit of the probe repo.
    trust_classifier : bool
        Must be True to load the classifier. Required for security.
    load_model : bool
        If True, download and initialize the base model.
    device : str | None
        Override device for inference.

    Returns
    -------
    LinearProbe
        The loaded probe.
    """
    _check_hub_deps()
    from huggingface_hub import snapshot_download

    from .probe import LinearProbe

    if not trust_classifier:
        # Try to read config for a better error message
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id, "probe_config.json", revision=revision
            )
            with open(config_path) as f:
                config = json.load(f)
            classifier_type = config["probe"].get("classifier_type", "unknown")
            publisher = repo_id.split("/")[0] if "/" in repo_id else "unknown"
            raise ValueError(
                f"This probe was published by '{publisher}' and declares a "
                f"{classifier_type} classifier. Pass trust_classifier=True to load. "
                "See https://skops.readthedocs.io/ for security details."
            )
        except ValueError:
            raise
        except Exception:
            raise ValueError(
                "Loading a probe requires trust_classifier=True. "
                "Pass trust_classifier=True to acknowledge you trust the publisher."
            )

    # Download the repo
    local_dir = snapshot_download(repo_id, revision=revision)
    local_path = Path(local_dir)

    # Read config
    with open(local_path / "probe_config.json") as f:
        config = json.load(f)

    probe_cfg = config["probe"]
    fmt = probe_cfg["serialization_format"]

    # Determine layers for construction
    layers = probe_cfg.get("layers_spec_original", probe_cfg.get("layers"))
    selected_layers = probe_cfg.get("selected_layers")

    # If auto/fast_auto was used, use selected layers for the extractor
    if layers in ("auto", "fast_auto") and selected_layers is not None:
        layers_for_extractor = selected_layers
    else:
        layers_for_extractor = layers

    # Determine model
    model_name = config["base_model"]["name"]
    model_revision = config["base_model"].get("revision")

    # Build constructor kwargs
    model_arg = model_name if load_model else None
    device_arg = device if device is not None else probe_cfg.get("backend", "cpu")
    if device_arg == "local":
        device_arg = "cpu"
    if device is None:
        device_arg = "cpu"

    probe = LinearProbe(
        model=model_arg,
        layers=layers_for_extractor,
        pooling=probe_cfg.get("pooling", "last_token"),
        train_pooling=probe_cfg.get("train_pooling"),
        inference_pooling=probe_cfg.get("inference_pooling"),
        classifier=probe_cfg.get("classifier_type", "logistic_regression"),
        task=probe_cfg.get("task", "classification"),
        device=device_arg,
        remote=False,
        random_state=probe_cfg.get("random_state"),
        batch_size=probe_cfg.get("batch_size", 8),
        normalize_layers=probe_cfg.get("normalize_layers", True),
        backend=probe_cfg.get("backend", "local"),
        dtype=probe_cfg.get("dtype"),
    )

    # Restore original layers spec
    probe.layers = layers

    # Load classifier
    probe.classifier_ = _load_classifier(local_path, fmt, trust_classifier)

    # Load scaler if present
    if config.get("has_scaler"):
        probe.scaler_ = _load_scaler(local_path, fmt)

    # Restore classes
    classes = config.get("classes", [0, 1])
    probe.classes_ = np.array(classes)

    # Restore selected layers
    probe.selected_layers_ = selected_layers

    # Store hub metadata for reference
    setattr(probe, "_hub_repo_id_", repo_id)
    setattr(probe, "_hub_config_", config)
    setattr(probe, "_hub_model_name_", model_name)
    setattr(probe, "_hub_model_revision_", model_revision)

    return probe


@dataclass
class ProbeCard:
    """Lightweight metadata container for a Hub-hosted probe.

    Reads probe_config.json and training_info.json without
    downloading classifier weights.
    """

    # From probe_config.json
    base_model: str
    base_model_revision: str | None
    layers: list[int] | None
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
    n_positive: int | None = None
    n_negative: int | None = None
    positive_hash: str | None = None
    negative_hash: str | None = None
    positive_examples: list[str] | None = field(default=None, repr=False)
    negative_examples: list[str] | None = field(default=None, repr=False)
    metrics: dict[str, float] | None = None
    training_environment: dict | None = None
    trained_at: str | None = None

    @classmethod
    def from_hub(cls, repo_id: str, revision: str | None = None) -> ProbeCard:
        """Load a ProbeCard from a HuggingFace Hub repository.

        Only downloads JSON metadata files, not classifier weights.
        """
        _check_hub_deps()
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id, "probe_config.json", revision=revision
        )
        with open(config_path) as f:
            config = json.load(f)

        # Try to download training_info.json (optional)
        training_info = None
        try:
            ti_path = hf_hub_download(
                repo_id, "training_info.json", revision=revision
            )
            with open(ti_path) as f:
                training_info = json.load(f)
        except Exception:
            pass

        return cls._from_dicts(config, training_info)

    @classmethod
    def from_local(cls, path: str | Path) -> ProbeCard:
        """Load a ProbeCard from a local directory.

        Parameters
        ----------
        path : str | Path
            Directory containing probe_config.json and optionally training_info.json.
        """
        path = Path(path)

        with open(path / "probe_config.json") as f:
            config = json.load(f)

        training_info = None
        ti_path = path / "training_info.json"
        if ti_path.exists():
            with open(ti_path) as f:
                training_info = json.load(f)

        return cls._from_dicts(config, training_info)

    @classmethod
    def _from_dicts(cls, config: dict, training_info: dict | None) -> ProbeCard:
        """Construct a ProbeCard from config and training_info dicts."""
        probe_cfg = config["probe"]

        kwargs = {
            "base_model": config["base_model"]["name"],
            "base_model_revision": config["base_model"].get("revision"),
            "layers": probe_cfg.get("layers"),
            "layers_spec_original": probe_cfg.get("layers_spec_original", probe_cfg.get("layers")),
            "pooling": probe_cfg.get("pooling", "last_token"),
            "train_pooling": probe_cfg.get("train_pooling", probe_cfg.get("pooling", "last_token")),
            "inference_pooling": probe_cfg.get(
                "inference_pooling", probe_cfg.get("pooling", "last_token")
            ),
            "classifier_type": probe_cfg.get("classifier_type", "logistic_regression"),
            "task": probe_cfg.get("task", "classification"),
            "random_state": probe_cfg.get("random_state"),
            "classes": config.get("classes", [0, 1]),
            "class_labels": config.get("class_labels"),
            "lmprobe_version": config.get("lmprobe_version", "unknown"),
            "config_version": config.get("config_version", 1),
        }

        if training_info is not None:
            td = training_info.get("training_data") or {}
            kwargs["n_positive"] = td.get("n_positive")
            kwargs["n_negative"] = td.get("n_negative")
            kwargs["positive_hash"] = td.get("positive_hash")
            kwargs["negative_hash"] = td.get("negative_hash")
            kwargs["positive_examples"] = td.get("positive_examples")
            kwargs["negative_examples"] = td.get("negative_examples")

            eval_data = training_info.get("evaluation")
            if eval_data and eval_data.get("metrics"):
                kwargs["metrics"] = eval_data["metrics"]

            kwargs["training_environment"] = training_info.get("training_environment")

            timestamps = training_info.get("timestamps") or {}
            kwargs["trained_at"] = timestamps.get("trained_at")

        return cls(**kwargs)

    @property
    def training_data_hash(self) -> str | None:
        """Combined hash of positive and negative training data."""
        if self.positive_hash and self.negative_hash:
            return f"pos={self.positive_hash}, neg={self.negative_hash}"
        return self.positive_hash or self.negative_hash

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
