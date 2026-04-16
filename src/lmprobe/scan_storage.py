"""Storage I/O for SampleScan artifacts.

Handles reading and writing the scan directory structure:

    scan_dir/
      metadata.json
      samples/samples.parquet
      channels/<channel_name>/
        config.json
        basis_{signal}.npy       # [n_layers, signal_dim, k_eff] float16
      projections/
        values.npy               # [N_total, n_channels, k] float16
        coords.parquet           # sample_id, layer, token_pos, signal
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _require_pyarrow() -> Any:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq  # noqa: F401

        return pa
    except ImportError:
        raise ImportError(
            "pyarrow is required for SampleScan storage. "
            "Install it with: pip install 'lmprobe[scan]'"
        )


@dataclass
class SignalInfo:
    """Metadata for a single signal type."""

    name: str
    dim: int          # hidden_dim for activations, n_experts for router_logits
    k_effective: int  # min(n_components, dim) — actual PCA components stored


@dataclass
class ScanMetadata:
    """Metadata for a SampleScan artifact."""

    model_id: str
    hidden_dim: int
    n_layers: int
    n_components: int
    n_samples: int
    creation_date: str
    signals: list[str] = field(default_factory=lambda: ["attn_delta", "mlp_delta"])


def write_metadata(scan_dir: Path, metadata: ScanMetadata) -> None:
    """Write metadata.json to scan directory."""
    scan_dir.mkdir(parents=True, exist_ok=True)
    with open(scan_dir / "metadata.json", "w") as f:
        json.dump(asdict(metadata), f, indent=2)


def read_metadata(scan_dir: Path) -> ScanMetadata:
    """Read metadata.json from scan directory."""
    with open(scan_dir / "metadata.json") as f:
        data = json.load(f)
    # Handle scans created before signals field existed
    if "signals" not in data:
        data["signals"] = ["attn_delta", "mlp_delta"]
    return ScanMetadata(**data)


def write_samples(
    scan_dir: Path,
    sample_ids: list[int],
    prompts: list[str],
    labels: list[int],
    token_ids_list: list[list[int]],
    seq_lengths: list[int],
) -> None:
    """Write samples parquet."""
    pa = _require_pyarrow()
    import pyarrow.parquet as pq

    samples_dir = scan_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "sample_id": pa.array(sample_ids, type=pa.int32()),
            "prompt_text": pa.array(prompts, type=pa.utf8()),
            "label": pa.array(labels, type=pa.int32()),
            "token_ids": pa.array(token_ids_list, type=pa.list_(pa.int32())),
            "seq_length": pa.array(seq_lengths, type=pa.int32()),
        }
    )
    pq.write_table(table, samples_dir / "samples.parquet")


def read_samples(scan_dir: Path) -> Any:
    """Read samples parquet. Returns pyarrow Table."""
    _require_pyarrow()
    import pyarrow.parquet as pq

    return pq.read_table(scan_dir / "samples" / "samples.parquet")


def write_channel(
    scan_dir: Path,
    channel_name: str,
    bases: dict[str, np.ndarray],
    config: dict[str, Any],
) -> None:
    """Write a channel's per-signal bases and config.

    Parameters
    ----------
    bases : dict[str, np.ndarray]
        Maps signal name to basis array of shape [n_layers, signal_dim, k_eff].
    config : dict
        Channel configuration (name, k, fit_method, signals info, etc.).
    """
    channel_dir = scan_dir / "channels" / channel_name
    channel_dir.mkdir(parents=True, exist_ok=True)

    for signal_name, basis in bases.items():
        np.save(channel_dir / f"basis_{signal_name}.npy", basis)

    with open(channel_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def read_basis(
    scan_dir: Path,
    channel: str = "0_global",
    signal: str | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    """Read basis vectors for a channel.

    Parameters
    ----------
    signal : str or None
        If given, return basis for that signal only: [n_layers, dim, k].
        If None, return dict mapping signal name to basis array.
    """
    channel_dir = scan_dir / "channels" / channel

    if signal is not None:
        return np.load(channel_dir / f"basis_{signal}.npy")

    # Load all signal bases
    bases = {}
    for path in sorted(channel_dir.glob("basis_*.npy")):
        # Extract signal name from "basis_attn_delta.npy" -> "attn_delta"
        sig_name = path.stem[len("basis_"):]
        bases[sig_name] = np.load(path)
    return bases


def read_channel_config(scan_dir: Path, channel: str = "0_global") -> dict[str, Any]:
    """Read channel config."""
    with open(scan_dir / "channels" / channel / "config.json") as f:
        return json.load(f)


def write_projections(
    scan_dir: Path,
    values: np.ndarray,
    coords_sample_id: np.ndarray,
    coords_layer: np.ndarray,
    coords_token_pos: np.ndarray,
    coords_signal: np.ndarray,
) -> None:
    """Write projections (values + coordinates).

    Parameters
    ----------
    values : np.ndarray
        Shape [N_total, n_channels, k] float16. k is padded to the max
        across all signals.
    coords_signal : np.ndarray
        Signal index (int8), indexing into metadata.signals.
    """
    pa = _require_pyarrow()
    import pyarrow.parquet as pq

    proj_dir = scan_dir / "projections"
    proj_dir.mkdir(parents=True, exist_ok=True)

    np.save(proj_dir / "values.npy", values)

    table = pa.table(
        {
            "sample_id": pa.array(coords_sample_id, type=pa.int32()),
            "layer": pa.array(coords_layer, type=pa.int16()),
            "token_pos": pa.array(coords_token_pos, type=pa.int16()),
            "signal": pa.array(coords_signal, type=pa.int8()),
        }
    )
    pq.write_table(table, proj_dir / "coords.parquet")


def open_projections(scan_dir: Path) -> np.memmap:
    """Memory-map the projections values array.

    Returns memmap of shape [N_total, n_channels, k].
    """
    return np.load(
        scan_dir / "projections" / "values.npy", mmap_mode="r"
    )


def read_coords(scan_dir: Path) -> Any:
    """Read coordinates parquet. Returns pyarrow Table."""
    _require_pyarrow()
    import pyarrow.parquet as pq

    return pq.read_table(scan_dir / "projections" / "coords.parquet")
