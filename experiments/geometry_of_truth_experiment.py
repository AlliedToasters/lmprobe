#!/usr/bin/env python3
"""
Geometry of Truth: Layer Sweep Experiment

This experiment probes Llama-3.1-70B across 32 layers on the Geometry of Truth
datasets to identify which layers best encode truth/falsehood.

Based on "The Geometry of Truth: Emergent Linear Structure in Large Language Model
Representations of True/False Datasets" by Marks & Tegmark (2023).

Key findings from the paper:
- Middle layers (around layer 13-15 for LLaMA-13B) are optimal for truth probing
- Mass-mean probes generalize better than logistic regression
- Training on statement pairs (e.g., cities + neg_cities) improves generalization
"""

import gc
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not installed. Plots will be skipped.")

from lmprobe import ActivationBaseline, BaselineBattery, BaselineProbe, LinearProbe
from lmprobe.extraction import clear_model_cache


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    # Model (70B base model via NDIF)
    model: str = "meta-llama/Llama-3.1-70B"
    device: str = "auto"
    remote: bool = True  # 70B requires NDIF
    
    # Probing settings
    batch_size: int = 4  # Small batch for large models
    random_state: int = 42
    
    # Layer sweep: 32 evenly-spaced layers
    # 70B has 80 layers, so this samples roughly every 2.5 layers
    n_layers: int = 32
    
    # Classifiers to compare (excluding mass_mean which didn't work well)
    classifiers: tuple = ("lda", "logistic_regression", "ridge")
    
    # Text-only baseline methods (no model required)
    text_baselines: tuple = ("bow", "tfidf", "random", "majority")
    
    # Whether to try sentence-transformers baseline (requires optional dep)
    use_sentence_transformers: bool = True
    
    # Activation-based baselines (require model)
    # These test whether the probe's learned direction is special
    use_activation_baselines: bool = True
    activation_baselines: tuple = ("random_direction", "pca", "layer_0")

    # Perplexity baseline (separate - uses BaselineProbe with model)
    use_perplexity_baseline: bool = True
    
    # Train/test split
    train_frac: float = 0.8
    
    # Output
    output_dir: str = "got_experiment_results"


# Curated datasets from Geometry of Truth
CURATED_DATASETS = [
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "larger_than",
    "smaller_than",
]

# Primary datasets for main experiment (good generalization test)
PRIMARY_DATASETS = ["cities", "sp_en_trans", "larger_than"]


# =============================================================================
# Utilities
# =============================================================================

def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return str(timedelta(seconds=int(seconds)))


def get_memory_usage() -> tuple[float, float]:
    """Get current and available memory in GB."""
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    available_gb = mem.available / (1024**3)
    return used_gb, available_gb


def print_memory_status(prefix: str = "") -> None:
    """Print current memory status."""
    used, available = get_memory_usage()
    total = used + available
    pct = (used / total) * 100
    status = "OK" if pct < 80 else "WARNING" if pct < 90 else "CRITICAL"
    print(f"{prefix}Memory: {used:.1f}GB used / {total:.1f}GB ({pct:.0f}%) [{status}]")


def force_gc() -> None:
    """Force garbage collection."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def get_candidate_layers(n_layers: int, total_layers: int = 80) -> list[int]:
    """Get evenly-spaced candidate layers.
    
    70B has 80 layers. This returns n_layers evenly spaced across them.
    """
    if n_layers >= total_layers:
        return list(range(total_layers))
    
    # Evenly space across the model
    indices = np.linspace(0, total_layers - 1, n_layers, dtype=int)
    return sorted(set(indices.tolist()))


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load a Geometry of Truth CSV dataset.
    
    Expected format: columns 'statement' and 'label' (True/False or 1/0)
    """
    df = pd.read_csv(filepath)
    
    # Ensure we have the expected columns
    if "statement" not in df.columns:
        raise ValueError(f"Expected 'statement' column in {filepath}")
    if "label" not in df.columns:
        raise ValueError(f"Expected 'label' column in {filepath}")
    
    # Normalize label to boolean
    if df["label"].dtype == bool:
        pass
    elif df["label"].dtype in [int, float]:
        df["label"] = df["label"].astype(bool)
    else:
        # Handle string True/False
        df["label"] = df["label"].map({"True": True, "False": False, "TRUE": True, "FALSE": False})
    
    return df


def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    random_state: int = 42,
) -> dict:
    """Split dataset into train/test with balanced classes.
    
    Returns dict with train_true, train_false, test_true, test_false lists.
    """
    np.random.seed(random_state)
    
    true_statements = df[df["label"] == True]["statement"].tolist()
    false_statements = df[df["label"] == False]["statement"].tolist()
    
    # Shuffle
    np.random.shuffle(true_statements)
    np.random.shuffle(false_statements)
    
    # Split
    n_train_true = int(len(true_statements) * train_frac)
    n_train_false = int(len(false_statements) * train_frac)
    
    return {
        "train_true": true_statements[:n_train_true],
        "train_false": false_statements[:n_train_false],
        "test_true": true_statements[n_train_true:],
        "test_false": false_statements[n_train_false:],
    }


def load_all_datasets(data_dir: Path, datasets: list[str]) -> dict[str, dict]:
    """Load and split multiple datasets.
    
    Returns dict mapping dataset_name -> split_data
    """
    all_data = {}
    for name in datasets:
        filepath = data_dir / f"{name}.csv"
        if not filepath.exists():
            print(f"  Warning: {filepath} not found, skipping")
            continue
        
        df = load_dataset(filepath)
        split = split_dataset(df)
        all_data[name] = split
        
        n_train = len(split["train_true"]) + len(split["train_false"])
        n_test = len(split["test_true"]) + len(split["test_false"])
        print(f"  {name}: {n_train} train, {n_test} test "
              f"({len(split['train_true'])}T/{len(split['train_false'])}F)")
    
    return all_data


# =============================================================================
# Baselines
# =============================================================================

def run_baselines(config: ExperimentConfig, data: dict, layer: int | None = None) -> dict:
    """Run all applicable baselines on a single dataset split.
    
    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    data : dict
        Dataset split with train_true, train_false, test_true, test_false
    layer : int, optional
        Layer for activation baselines (if enabled)
    
    Returns
    -------
    dict
        Mapping of baseline_name -> accuracy
    """
    results = {}
    
    test_prompts = data["test_true"] + data["test_false"]
    test_labels = [1] * len(data["test_true"]) + [0] * len(data["test_false"])
    
    # --- Text-only baselines (no model required) ---
    print("\n  Text-only baselines:")
    for method in config.text_baselines:
        try:
            baseline = BaselineProbe(
                method=method,
                classifier="logistic_regression",
                random_state=config.random_state,
            )
            baseline.fit(data["train_true"], data["train_false"])
            accuracy = baseline.score(test_prompts, test_labels)
            results[method] = accuracy
            print(f"    {method:20s}: {accuracy:.1%}")
        except Exception as e:
            print(f"    {method:20s}: FAILED ({e})")
    
    # --- Sentence-transformers baseline (optional dependency) ---
    if config.use_sentence_transformers:
        try:
            baseline = BaselineProbe(
                method="sentence_transformers",
                random_state=config.random_state,
            )
            baseline.fit(data["train_true"], data["train_false"])
            accuracy = baseline.score(test_prompts, test_labels)
            results["sentence_transformers"] = accuracy
            print(f"    {'sentence_transformers':20s}: {accuracy:.1%}")
        except ImportError:
            print(f"    {'sentence_transformers':20s}: SKIPPED (not installed)")
        except Exception as e:
            print(f"    {'sentence_transformers':20s}: FAILED ({e})")
    
    # --- Activation-based baselines (require model) ---
    if config.use_activation_baselines:
        print("\n  Activation baselines:")
        
        # Use provided layer or default to middle layer
        baseline_layer = layer if layer is not None else 40  # Middle of 80 layers for 70B
        
        for method in config.activation_baselines:
            try:
                # Use ActivationBaseline for activation-based methods
                baseline = ActivationBaseline(
                    method=method,
                    model=config.model,
                    layers=baseline_layer,
                    device=config.device,
                    remote=config.remote,
                    random_state=config.random_state,
                )
                baseline.fit(data["train_true"], data["train_false"])
                accuracy = baseline.score(test_prompts, test_labels)
                results[method] = accuracy
                print(f"    {method:20s}: {accuracy:.1%}")
            except Exception as e:
                print(f"    {method:20s}: FAILED ({e})")
    else:
        print("\n  Activation baselines: DISABLED (use_activation_baselines=False)")

    # --- Perplexity baseline (uses BaselineProbe with model) ---
    if config.use_perplexity_baseline:
        print("\n  Perplexity baseline:")
        try:
            baseline = BaselineProbe(
                method="perplexity",
                model=config.model,
                device=config.device,
                remote=config.remote,
                random_state=config.random_state,
            )
            baseline.fit(data["train_true"], data["train_false"])
            accuracy = baseline.score(test_prompts, test_labels)
            results["perplexity"] = accuracy
            print(f"    {'perplexity':20s}: {accuracy:.1%}")
        except Exception as e:
            print(f"    {'perplexity':20s}: FAILED ({e})")

    return results


def get_best_baseline(baseline_results: dict) -> tuple[str, float]:
    """Get the best performing baseline.
    
    Returns
    -------
    tuple
        (baseline_name, accuracy)
    """
    if not baseline_results:
        return ("none", 0.0)
    return max(baseline_results.items(), key=lambda x: x[1])


# =============================================================================
# Single Layer Sweep
# =============================================================================

def run_layer_sweep(
    config: ExperimentConfig,
    data: dict,
    candidate_layers: list[int],
    dataset_name: str = "",
) -> dict:
    """Sweep across layers for a single dataset.
    
    Returns dict mapping (layer, classifier) -> accuracy
    """
    results = {}
    
    test_prompts = data["test_true"] + data["test_false"]
    test_labels = [1] * len(data["test_true"]) + [0] * len(data["test_false"])
    
    total = len(candidate_layers) * len(config.classifiers)
    completed = 0
    times = []
    sweep_start = time.time()
    
    best_acc = 0.0
    best_config = None
    
    for layer_idx, layer in enumerate(candidate_layers):
        layer_start = time.time()
        
        for clf_name in config.classifiers:
            completed += 1
            iter_start = time.time()
            
            # Progress
            print(f"\r  Layer {layer:3d} | {clf_name:20s} ", end="", flush=True)
            
            probe = LinearProbe(
                model=config.model,
                layers=layer,
                classifier=clf_name,
                device=config.device,
                remote=config.remote,
                batch_size=config.batch_size,
                random_state=config.random_state,
                normalize_layers=False,
            )
            
            probe.fit(data["train_true"], data["train_false"])
            accuracy = probe.score(test_prompts, test_labels)
            elapsed = time.time() - iter_start
            times.append(elapsed)
            
            results[(layer, clf_name)] = accuracy
            
            if accuracy > best_acc:
                best_acc = accuracy
                best_config = (layer, clf_name)
            
            # Progress bar
            bar_len = int(accuracy * 20)
            bar = "▓" * bar_len + "░" * (20 - bar_len)
            print(f"{accuracy:6.1%} [{bar}] ({elapsed:.1f}s)", end="")
            
            # ETA
            if len(times) > 0:
                avg_time = sum(times) / len(times)
                remaining = total - completed
                eta = avg_time * remaining
                print(f" | ETA: {format_time(eta)}", end="")
            
            print("", flush=True)
        
        # Periodic memory cleanup
        if layer_idx % 8 == 7:
            force_gc()
    
    elapsed_total = time.time() - sweep_start
    print(f"\n  Completed in {format_time(elapsed_total)}")
    print(f"  Best: {best_config[1]} @ layer {best_config[0]} = {best_acc:.1%}")
    
    return results


# =============================================================================
# Cross-Dataset Generalization
# =============================================================================

def run_generalization_test(
    config: ExperimentConfig,
    all_data: dict[str, dict],
    train_dataset: str,
    test_dataset: str,
    layer: int,
    classifier: str = "logistic_regression",
) -> float:
    """Train on one dataset, test on another.
    
    This is a key test from the Geometry of Truth paper:
    Good probes should generalize across datasets.
    """
    train_data = all_data[train_dataset]
    test_data = all_data[test_dataset]
    
    probe = LinearProbe(
        model=config.model,
        layers=layer,
        classifier=classifier,
        device=config.device,
        remote=config.remote,
        batch_size=config.batch_size,
        random_state=config.random_state,
    )
    
    probe.fit(train_data["train_true"], train_data["train_false"])
    
    test_prompts = test_data["test_true"] + test_data["test_false"]
    test_labels = [1] * len(test_data["test_true"]) + [0] * len(test_data["test_false"])
    
    return probe.score(test_prompts, test_labels)


# =============================================================================
# Visualization
# =============================================================================

def create_layer_accuracy_plot(
    candidate_layers: list[int],
    results: dict,
    classifiers: tuple,
    baseline_results: dict,
    output_path: Path,
    dataset_name: str = "",
):
    """Create line plot of accuracy by layer."""
    if not HAS_PLOTTING:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for clf in classifiers:
        accuracies = [results.get((l, clf), 0) for l in candidate_layers]
        ax.plot(candidate_layers, accuracies, marker="o", label=clf, linewidth=2, markersize=4)
    
    # Best baseline reference line (thick, prominent)
    best_name, best_acc = get_best_baseline(baseline_results)
    ax.axhline(y=best_acc, color="red", linestyle="-", linewidth=2, 
               alpha=0.8, label=f"best baseline ({best_name}: {best_acc:.1%})")
    
    # Other baseline reference lines (lighter)
    baseline_colors = {"tfidf": "darkred", "bow": "orange", "sentence_transformers": "purple",
                       "random_direction": "blue", "pca": "green", "perplexity": "brown"}
    for method, acc in baseline_results.items():
        if method != best_name:  # Don't double-draw the best
            color = baseline_colors.get(method, "gray")
            ax.axhline(y=acc, color=color, linestyle="--", alpha=0.4)
    
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, label="chance")
    
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Accuracy")
    title = f"Probe Accuracy by Layer - {dataset_name}" if dataset_name else "Probe Accuracy by Layer"
    ax.set_title(f"{title}\n(Llama-3.1-70B)")
    ax.legend(loc="lower right")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_accuracy_heatmap(
    candidate_layers: list[int],
    results: dict,
    classifiers: tuple,
    output_path: Path,
    dataset_name: str = "",
):
    """Create heatmap of accuracy by layer and classifier."""
    if not HAS_PLOTTING:
        return
    
    matrix = np.zeros((len(candidate_layers), len(classifiers)))
    for i, layer in enumerate(candidate_layers):
        for j, clf in enumerate(classifiers):
            matrix[i, j] = results.get((layer, clf), 0)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1%",
        xticklabels=classifiers,
        yticklabels=candidate_layers,
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        ax=ax,
    )
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Layer")
    title = f"Probe Accuracy - {dataset_name}" if dataset_name else "Probe Accuracy Heatmap"
    ax.set_title(f"{title}\n(Llama-3.1-70B)")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_generalization_matrix(
    gen_results: dict,
    datasets: list[str],
    output_path: Path,
):
    """Create generalization matrix heatmap."""
    if not HAS_PLOTTING:
        return
    
    n = len(datasets)
    matrix = np.zeros((n, n))
    
    for i, train_ds in enumerate(datasets):
        for j, test_ds in enumerate(datasets):
            matrix[i, j] = gen_results.get((train_ds, test_ds), 0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1%",
        xticklabels=datasets,
        yticklabels=datasets,
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        ax=ax,
    )
    ax.set_xlabel("Test Dataset")
    ax.set_ylabel("Train Dataset")
    ax.set_title("Cross-Dataset Generalization Matrix\n(Llama-3.1-70B)")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_baseline_shootout_chart(
    all_results: dict,
    output_path: Path,
):
    """Create bar chart comparing all baselines vs best probe across datasets.

    This is the "did the probe actually learn something?" sanity check.
    """
    if not HAS_PLOTTING:
        return

    # Collect data across all datasets
    datasets = list(all_results.get("datasets", {}).keys())
    if not datasets:
        return

    # Aggregate baselines and probes
    baseline_scores = {}  # baseline_name -> list of scores
    probe_scores = []  # list of best probe scores

    for ds_name, ds_results in all_results["datasets"].items():
        # Baselines
        for bl_name, bl_acc in ds_results.get("baselines", {}).items():
            if bl_name not in baseline_scores:
                baseline_scores[bl_name] = []
            baseline_scores[bl_name].append(bl_acc)

        # Best probe
        if "best_probe" in ds_results:
            probe_scores.append(ds_results["best_probe"]["accuracy"])

    if not baseline_scores or not probe_scores:
        return

    # Calculate means
    baseline_means = {k: np.mean(v) for k, v in baseline_scores.items()}
    probe_mean = np.mean(probe_scores)

    # Combine and sort
    all_methods = {**baseline_means, "BEST PROBE": probe_mean}
    sorted_methods = sorted(all_methods.items(), key=lambda x: x[1], reverse=True)

    names = [m[0] for m in sorted_methods]
    scores = [m[1] for m in sorted_methods]

    # Color by type
    colors = []
    for name in names:
        if name == "BEST PROBE":
            colors.append("#9b59b6")  # Purple for probe (distinct from baselines)
        elif name in ["random", "majority"]:
            colors.append("#95a5a6")  # Gray for trivial baselines
        elif name in ["random_direction", "pca", "layer_0", "perplexity"]:
            colors.append("#3498db")  # Blue for activation baselines
        else:
            colors.append("#e74c3c")  # Red for text baselines

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(names, scores, color=colors)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f"{score:.1%}", va="center", fontsize=10)

    # Add chance line
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="chance")

    ax.set_xlabel("Mean Accuracy Across Datasets")
    ax.set_title("Baseline Shootout: Who Beats the Probe?\n(Llama-3.1-70B on Geometry of Truth)")
    ax.set_xlim(0.4, 1.05)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#9b59b6", label="Linear Probe"),
        Patch(facecolor="#e74c3c", label="Text Baselines"),
        Patch(facecolor="#3498db", label="Activation Baselines"),
        Patch(facecolor="#95a5a6", label="Trivial Baselines"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_optimal_layer_histogram(
    all_results: dict,
    candidate_layers: list[int],
    output_path: Path,
):
    """Create histogram showing where optimal layers fall across datasets.

    Tests the "middle layers are best for truth" hypothesis.
    """
    if not HAS_PLOTTING:
        return

    optimal_layers = []
    dataset_labels = []

    for ds_name, ds_results in all_results.get("datasets", {}).items():
        if "best_probe" in ds_results:
            optimal_layers.append(ds_results["best_probe"]["layer"])
            dataset_labels.append(ds_name)

    if not optimal_layers:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram of optimal layers
    ax1.hist(optimal_layers, bins=len(candidate_layers)//2, edgecolor="black", alpha=0.7, color="#3498db")
    ax1.axvline(x=np.mean(optimal_layers), color="red", linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(optimal_layers):.1f}")
    ax1.axvline(x=40, color="green", linestyle=":", linewidth=2,
                label="Middle layer (40)")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Optimal Layers")
    ax1.legend()

    # Right: Scatter plot with dataset labels
    ax2.scatter(optimal_layers, range(len(optimal_layers)), s=100, c="#3498db", edgecolors="black")
    for i, (layer, ds) in enumerate(zip(optimal_layers, dataset_labels)):
        ax2.annotate(ds, (layer, i), xytext=(5, 0), textcoords="offset points", fontsize=9)

    ax2.axvline(x=40, color="green", linestyle=":", linewidth=2, alpha=0.5)
    ax2.set_xlabel("Optimal Layer")
    ax2.set_ylabel("Dataset")
    ax2.set_yticks([])
    ax2.set_title("Optimal Layer by Dataset")

    fig.suptitle("Where Does Truth Live? Optimal Layer Analysis\n(Llama-3.1-70B)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_layer_importance_curve(
    all_results: dict,
    candidate_layers: list[int],
    classifiers: tuple,
    output_path: Path,
):
    """Create smooth curve showing accuracy across layers, averaged over datasets.

    This reveals the "truth encoding zone" in the model.
    """
    if not HAS_PLOTTING:
        return

    # Aggregate accuracy at each layer across datasets
    layer_acc_by_clf = {clf: {l: [] for l in candidate_layers} for clf in classifiers}

    for ds_name, ds_results in all_results.get("datasets", {}).items():
        layer_results = ds_results.get("layer_results", {})
        for key, acc in layer_results.items():
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                layer, clf = int(parts[0]), parts[1]
                if clf in layer_acc_by_clf and layer in layer_acc_by_clf[clf]:
                    layer_acc_by_clf[clf][layer].append(acc)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot each classifier
    for clf in classifiers:
        means = [np.mean(layer_acc_by_clf[clf][l]) if layer_acc_by_clf[clf][l] else 0
                 for l in candidate_layers]
        stds = [np.std(layer_acc_by_clf[clf][l]) if len(layer_acc_by_clf[clf][l]) > 1 else 0
                for l in candidate_layers]

        ax.plot(candidate_layers, means, marker="o", label=clf, linewidth=2, markersize=4)
        ax.fill_between(candidate_layers,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2)

    # Mark the "truth zone"
    all_means = []
    for clf in classifiers:
        for l in candidate_layers:
            if layer_acc_by_clf[clf][l]:
                all_means.append((l, np.mean(layer_acc_by_clf[clf][l])))

    if all_means:
        # Find layers above 90% of max
        max_acc = max(m[1] for m in all_means)
        threshold = 0.9 * max_acc
        good_layers = [l for l, acc in all_means if acc >= threshold]
        if good_layers:
            zone_start, zone_end = min(good_layers), max(good_layers)
            ax.axvspan(zone_start, zone_end, alpha=0.1, color="green", label=f"Truth zone (>{threshold:.1%})")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Accuracy (± std)")
    ax.set_title("Layer Importance Curve: The Truth Encoding Zone\n(Llama-3.1-70B, averaged across datasets)")
    ax.legend(loc="lower right")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_generalization_gap_plot(
    all_results: dict,
    output_path: Path,
):
    """Create plot comparing in-distribution vs out-of-distribution accuracy.

    A good truth probe should have a small gap (good generalization).
    """
    if not HAS_PLOTTING:
        return

    gen_results = all_results.get("generalization", {})
    if not gen_results:
        return

    datasets = list(all_results.get("datasets", {}).keys())

    in_dist = []
    out_dist = []
    labels = []

    for ds in datasets:
        # In-distribution: train and test on same dataset
        in_key = f"{ds}→{ds}"
        in_acc = gen_results.get(in_key, 0)

        # Out-of-distribution: average over other datasets
        out_accs = [gen_results.get(f"{ds}→{other}", 0)
                    for other in datasets if other != ds]
        out_acc = np.mean(out_accs) if out_accs else 0

        if in_acc > 0:
            in_dist.append(in_acc)
            out_dist.append(out_acc)
            labels.append(ds)

    if not in_dist:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Grouped bar chart
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, in_dist, width, label="In-Distribution", color="#3498db")
    bars2 = ax1.bar(x + width/2, out_dist, width, label="Out-of-Distribution", color="#e74c3c")

    ax1.set_xlabel("Training Dataset")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("In-Distribution vs Out-of-Distribution")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend()
    ax1.set_ylim(0.4, 1.05)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    # Right: Gap analysis
    gaps = [i - o for i, o in zip(in_dist, out_dist)]
    colors = ["#3498db" if g < 0.1 else "#f39c12" if g < 0.2 else "#e74c3c" for g in gaps]

    ax2.barh(labels, gaps, color=colors, edgecolor="black")
    ax2.axvline(x=0.1, color="green", linestyle="--", alpha=0.7, label="Good (<10%)")
    ax2.axvline(x=0.2, color="orange", linestyle="--", alpha=0.7, label="Okay (<20%)")
    ax2.set_xlabel("Generalization Gap (In-Dist - Out-Dist)")
    ax2.set_title("Generalization Gap by Dataset")
    ax2.legend(loc="lower right")

    mean_gap = np.mean(gaps)
    ax2.axvline(x=mean_gap, color="blue", linestyle="-", linewidth=2,
                label=f"Mean gap: {mean_gap:.1%}")

    fig.suptitle("Does the Probe Generalize? In-Dist vs Out-of-Dist Analysis\n(Llama-3.1-70B)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_pca_truth_plot(
    config: ExperimentConfig,
    data: dict,
    layer: int,
    output_path: Path,
    dataset_name: str = "",
):
    """Create the iconic PCA visualization from the Geometry of Truth paper.

    Shows true/false statement activations projected onto top 2 PCs.
    This is THE signature plot - beautiful clusters separating by truth value.
    """
    if not HAS_PLOTTING:
        return

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("    (Skipping PCA plot - sklearn not available)")
        return

    # We need to extract activations - this requires the model
    try:
        probe = LinearProbe(
            model=config.model,
            layers=layer,
            classifier="logistic_regression",  # Doesn't matter, we just want activations
            device=config.device,
            remote=config.remote,
            batch_size=config.batch_size,
        )

        # Get activations for true and false statements
        true_statements = data["train_true"] + data["test_true"]
        false_statements = data["train_false"] + data["test_false"]

        print(f"    Extracting activations for PCA plot (layer {layer})...")

        # Extract activations (this uses the internal extraction method)
        true_acts = probe._extract_activations(true_statements)
        false_acts = probe._extract_activations(false_statements)

        # Combine and do PCA
        all_acts = np.vstack([true_acts, false_acts])
        labels = [True] * len(true_acts) + [False] * len(false_acts)

        # Center the data
        all_acts = all_acts - all_acts.mean(axis=0)

        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_acts)

        # Split back
        true_proj = projected[:len(true_acts)]
        false_proj = projected[len(true_acts):]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(true_proj[:, 0], true_proj[:, 1], c="#3498db", label="True",
                   alpha=0.6, s=50, edgecolors="white", linewidth=0.5)
        ax.scatter(false_proj[:, 0], false_proj[:, 1], c="#e74c3c", label="False",
                   alpha=0.6, s=50, edgecolors="white", linewidth=0.5)

        # Draw truth direction (from false centroid to true centroid)
        true_centroid = true_proj.mean(axis=0)
        false_centroid = false_proj.mean(axis=0)

        ax.annotate("", xy=true_centroid, xytext=false_centroid,
                    arrowprops=dict(arrowstyle="->", color="black", lw=2))
        ax.scatter(*true_centroid, c="darkblue", s=200, marker="*",
                   edgecolors="black", zorder=5, label="True centroid")
        ax.scatter(*false_centroid, c="darkred", s=200, marker="*",
                   edgecolors="black", zorder=5, label="False centroid")

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        title = f"The Geometry of Truth: {dataset_name}" if dataset_name else "The Geometry of Truth"
        ax.set_title(f"{title}\nLayer {layer} Activations (Llama-3.1-70B)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Make it square
        ax.set_aspect("equal", adjustable="datalim")

        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        print(f"    PCA plot saved: {output_path}")

    except Exception as e:
        print(f"    (Skipping PCA plot - error: {e})")


def create_truth_direction_across_layers(
    config: ExperimentConfig,
    data: dict,
    candidate_layers: list[int],
    output_path: Path,
    dataset_name: str = "",
    sample_layers: int = 8,
):
    """Create a multi-panel plot showing truth direction emergence across layers.

    This shows how the true/false separation evolves through the network.
    """
    if not HAS_PLOTTING:
        return

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("    (Skipping layer evolution plot - sklearn not available)")
        return

    # Sample layers evenly
    layer_indices = np.linspace(0, len(candidate_layers) - 1, sample_layers, dtype=int)
    layers_to_plot = [candidate_layers[i] for i in layer_indices]

    try:
        true_statements = data["train_true"][:50]  # Limit for speed
        false_statements = data["train_false"][:50]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for idx, (ax, layer) in enumerate(zip(axes, layers_to_plot)):
            print(f"    Extracting layer {layer} ({idx+1}/{len(layers_to_plot)})...")

            probe = LinearProbe(
                model=config.model,
                layers=layer,
                classifier="logistic_regression",
                device=config.device,
                remote=config.remote,
                batch_size=config.batch_size,
            )

            true_acts = probe._extract_activations(true_statements)
            false_acts = probe._extract_activations(false_statements)

            all_acts = np.vstack([true_acts, false_acts])
            all_acts = all_acts - all_acts.mean(axis=0)

            pca = PCA(n_components=2)
            projected = pca.fit_transform(all_acts)

            true_proj = projected[:len(true_acts)]
            false_proj = projected[len(true_acts):]

            ax.scatter(true_proj[:, 0], true_proj[:, 1], c="#3498db", alpha=0.6, s=20)
            ax.scatter(false_proj[:, 0], false_proj[:, 1], c="#e74c3c", alpha=0.6, s=20)

            # Calculate separation (distance between centroids / avg std)
            true_centroid = true_proj.mean(axis=0)
            false_centroid = false_proj.mean(axis=0)
            separation = np.linalg.norm(true_centroid - false_centroid)

            ax.set_title(f"Layer {layer}\nsep={separation:.2f}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")

        title = f"Truth Direction Emergence: {dataset_name}" if dataset_name else "Truth Direction Emergence"
        fig.suptitle(f"{title}\n(Llama-3.1-70B)", fontsize=14)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        print(f"    Layer evolution plot saved: {output_path}")

    except Exception as e:
        print(f"    (Skipping layer evolution plot - error: {e})")


# =============================================================================
# Results Summary
# =============================================================================

def print_dataset_summary(
    dataset_name: str,
    layer_results: dict,
    baseline_results: dict,
    candidate_layers: list[int],
    classifiers: tuple,
):
    """Print summary for a single dataset."""
    print(f"\n{'─'*60}")
    print(f"SUMMARY: {dataset_name}")
    print(f"{'─'*60}")
    
    # Baselines
    print("\nBaselines:")
    for method, acc in sorted(baseline_results.items(), key=lambda x: -x[1]):
        print(f"  {method:25s}: {acc:.1%}")
    
    best_baseline_name, best_baseline_acc = get_best_baseline(baseline_results)
    print(f"\n  >>> Best baseline: {best_baseline_name} ({best_baseline_acc:.1%})")
    
    # Best per classifier
    print("\nBest layer per classifier:")
    for clf in classifiers:
        clf_results = [(l, layer_results.get((l, clf), 0)) for l in candidate_layers]
        best_layer, best_acc = max(clf_results, key=lambda x: x[1])
        delta = best_acc - best_baseline_acc
        sign = "+" if delta > 0 else ""
        print(f"  {clf:25s}: Layer {best_layer:3d} = {best_acc:.1%} ({sign}{delta:.1%} vs best baseline)")
    
    # Overall best
    all_results = [(l, c, layer_results.get((l, c), 0)) 
                   for l in candidate_layers for c in classifiers]
    best_layer, best_clf, best_acc = max(all_results, key=lambda x: x[2])
    delta = best_acc - best_baseline_acc
    sign = "+" if delta > 0 else ""
    print(f"\nOverall best: {best_clf} @ layer {best_layer} = {best_acc:.1%} ({sign}{delta:.1%} vs baseline)")
    
    if best_acc <= best_baseline_acc:
        print("\n  *** WARNING: Probe does not beat best baseline! ***")


def save_results(results: dict, output_path: Path):
    """Save results to JSON."""
    # Convert tuple keys to strings for JSON
    json_results = {}
    for key, value in results.items():
        if isinstance(key, tuple):
            json_results[str(key)] = value
        else:
            json_results[key] = value
    
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the Geometry of Truth experiment."""
    experiment_start = time.time()
    
    print("=" * 70)
    print("GEOMETRY OF TRUTH: Layer Sweep Experiment")
    print("=" * 70)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: Llama-3.1-70B (via NDIF)")
    
    config = ExperimentConfig()
    
    # Get candidate layers
    candidate_layers = get_candidate_layers(config.n_layers)
    print(f"\nSweeping {len(candidate_layers)} layers: {candidate_layers[:5]}...{candidate_layers[-5:]}")
    
    # Data directory
    data_dir = Path(__file__).parent / "datasets" / "geometry_of_truth"
    if not data_dir.exists():
        print(f"\nError: Dataset directory not found: {data_dir}")
        print("Run: python download_geometry_of_truth.py first")
        return
    
    # Load datasets
    print(f"\n{'─'*70}")
    print("Loading datasets:")
    print(f"{'─'*70}")
    all_data = load_all_datasets(data_dir, CURATED_DATASETS)
    
    if not all_data:
        print("No datasets loaded!")
        return
    
    # Output directory
    output_dir = Path(__file__).parent / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Store all results
    all_results = {
        "config": {
            "model": config.model,
            "n_layers": config.n_layers,
            "candidate_layers": candidate_layers,
            "classifiers": list(config.classifiers),
        },
        "datasets": {},
    }
    
    # Run experiment for each dataset
    for dataset_name, data in all_data.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*70}")
        
        dataset_output_dir = output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run baselines
        print("\nRunning baselines...")
        baseline_results = run_baselines(config, data)
        
        # Run layer sweep
        print(f"\nRunning layer sweep ({len(candidate_layers)} layers × {len(config.classifiers)} classifiers)...")
        print_memory_status(prefix="  ")
        
        layer_results = run_layer_sweep(config, data, candidate_layers, dataset_name)
        
        # Summary
        print_dataset_summary(
            dataset_name, layer_results, baseline_results,
            candidate_layers, config.classifiers
        )
        
        # Visualizations
        if HAS_PLOTTING:
            print("\nGenerating plots...")
            create_layer_accuracy_plot(
                candidate_layers, layer_results, config.classifiers,
                baseline_results, dataset_output_dir / "accuracy_by_layer.png",
                dataset_name
            )
            create_accuracy_heatmap(
                candidate_layers, layer_results, config.classifiers,
                dataset_output_dir / "accuracy_heatmap.png",
                dataset_name
            )
        
        # Store results
        best_baseline_name, best_baseline_acc = get_best_baseline(baseline_results)
        
        # Find best probe result
        all_probe_results = [(l, c, layer_results.get((l, c), 0)) 
                            for l in candidate_layers for c in config.classifiers]
        best_layer, best_clf, best_probe_acc = max(all_probe_results, key=lambda x: x[2])
        
        all_results["datasets"][dataset_name] = {
            "baselines": baseline_results,
            "best_baseline": {"name": best_baseline_name, "accuracy": best_baseline_acc},
            "layer_results": {f"{l}_{c}": acc for (l, c), acc in layer_results.items()},
            "best_probe": {
                "layer": best_layer,
                "classifier": best_clf,
                "accuracy": best_probe_acc,
                "beats_baseline": best_probe_acc > best_baseline_acc,
                "margin": best_probe_acc - best_baseline_acc,
            },
        }

        # Create iconic PCA plot at optimal layer (this is the fun one!)
        if HAS_PLOTTING:
            print("\nGenerating PCA truth visualization...")
            create_pca_truth_plot(
                config, data, best_layer,
                dataset_output_dir / "pca_truth_geometry.png",
                dataset_name
            )

        # Save intermediate results
        save_results(all_results, output_dir / "results.json")
        
        # Memory cleanup
        force_gc()
        clear_model_cache()
    
    # Cross-dataset generalization test
    if len(all_data) >= 2:
        print(f"\n{'='*70}")
        print("CROSS-DATASET GENERALIZATION")
        print(f"{'='*70}")
        
        # Find best layer from cities (or first dataset)
        ref_dataset = "cities" if "cities" in all_data else list(all_data.keys())[0]
        ref_results = all_results["datasets"][ref_dataset]["layer_results"]
        
        # Find best layer across all classifiers
        best_layer = None
        best_acc = 0
        for key, acc in ref_results.items():
            layer = int(key.split("_")[0])
            if acc > best_acc:
                best_acc = acc
                best_layer = layer
        
        print(f"Using best layer from {ref_dataset}: layer {best_layer} ({best_acc:.1%})")
        print(f"\nGeneralization matrix (train row → test column):")
        
        datasets = list(all_data.keys())
        gen_results = {}
        
        # Header
        print(f"\n{'Train/Test':<20}", end="")
        for test_ds in datasets:
            print(f"{test_ds[:12]:>12}", end="")
        print()
        print("-" * (20 + 12 * len(datasets)))
        
        for train_ds in datasets:
            print(f"{train_ds:<20}", end="")
            for test_ds in datasets:
                if train_ds == test_ds:
                    # Use in-distribution result
                    acc = best_acc if train_ds == ref_dataset else 0.0
                    # Look it up properly
                    ds_results = all_results["datasets"][train_ds]["layer_results"]
                    for key, a in ds_results.items():
                        if key.startswith(f"{best_layer}_"):
                            acc = max(acc, a)
                else:
                    acc = run_generalization_test(
                        config, all_data, train_ds, test_ds, best_layer
                    )
                gen_results[(train_ds, test_ds)] = acc
                print(f"{acc:>11.1%}", end=" ")
            print()
        
        all_results["generalization"] = {
            f"{t}→{te}": acc for (t, te), acc in gen_results.items()
        }
        
        # Generalization matrix plot
        if HAS_PLOTTING:
            create_generalization_matrix(gen_results, datasets, output_dir / "generalization_matrix.png")

    # ==========================================================================
    # SUMMARY VISUALIZATIONS (the fun ones!)
    # ==========================================================================
    if HAS_PLOTTING and all_results.get("datasets"):
        print(f"\n{'='*70}")
        print("GENERATING SUMMARY VISUALIZATIONS")
        print(f"{'='*70}")

        # 1. Baseline shootout - did the probe actually learn something?
        print("\n  Creating baseline shootout chart...")
        create_baseline_shootout_chart(all_results, output_dir / "baseline_shootout.png")

        # 2. Optimal layer histogram - where does truth live?
        print("  Creating optimal layer histogram...")
        create_optimal_layer_histogram(all_results, candidate_layers, output_dir / "optimal_layer_histogram.png")

        # 3. Layer importance curve - the truth encoding zone
        print("  Creating layer importance curve...")
        create_layer_importance_curve(all_results, candidate_layers, config.classifiers,
                                      output_dir / "layer_importance_curve.png")

        # 4. Generalization gap plot - does it transfer?
        if "generalization" in all_results:
            print("  Creating generalization gap plot...")
            create_generalization_gap_plot(all_results, output_dir / "generalization_gap.png")

        # 5. Truth direction evolution across layers (pick first dataset for this)
        first_ds_name = list(all_results["datasets"].keys())[0]
        first_ds_data = all_data.get(first_ds_name)
        if first_ds_data:
            print(f"  Creating truth direction evolution plot ({first_ds_name})...")
            create_truth_direction_across_layers(
                config, first_ds_data, candidate_layers,
                output_dir / f"truth_direction_evolution_{first_ds_name}.png",
                first_ds_name,
                sample_layers=8
            )

        print(f"\n  Summary plots saved to: {output_dir}/")

    # Final save
    save_results(all_results, output_dir / "results.json")
    
    total_time = time.time() - experiment_start
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {format_time(total_time)}")
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
