"""Hero figure renderer for SampleScan.

Renders a 2D RGB image of a model forward pass on a single prompt:
tokens on x-axis, layers on y-axis, RGB channels from top-3 PCA
components of each signal at that coordinate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure


def _normalize_rgb(arr: np.ndarray) -> np.ndarray:
    """Normalize a 3-channel array to [0, 1] per channel across the full image."""
    result = np.zeros_like(arr, dtype=np.float32)
    for c in range(min(3, arr.shape[-1])):
        channel = arr[..., c].astype(np.float32)
        cmin, cmax = channel.min(), channel.max()
        if cmax - cmin > 1e-10:
            result[..., c] = (channel - cmin) / (cmax - cmin)
        else:
            result[..., c] = 0.5
    return result


def render_scan_figure(
    projections: np.ndarray,
    tokens: list[str],
    signal_names: list[str],
    log_probs: np.ndarray | None = None,
    layer_stats: np.ndarray | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    title: str = "",
) -> matplotlib.figure.Figure:
    """Render the hero scan figure.

    Parameters
    ----------
    projections : np.ndarray
        Shape [seq_len, n_layers, n_signals, k] where k >= 3.
    tokens : list[str]
        Decoded token strings, one per sequence position.
    signal_names : list[str]
        Names of signals, one per projections dim 2.
    log_probs : np.ndarray or None
        Shape [seq_len] next-token log-probabilities.
    layer_stats : np.ndarray or None
        Shape [n_layers, n_signals] per-layer statistics.
    figsize : tuple or None
        (width, height) in inches.
    title : str
        Optional figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    seq_len, n_layers, n_signals, k = projections.shape
    n_grids = n_signals

    # Build RGB image per signal: [n_layers, seq_len, 3]
    signal_rgbs = []
    for sig_idx in range(n_signals):
        rgb = _normalize_rgb(projections[:, :, sig_idx, :3])  # [seq_len, n_layers, 3]
        rgb = rgb.transpose(1, 0, 2)  # [n_layers, seq_len, 3]
        signal_rgbs.append(rgb)

    show_surprise = log_probs is not None
    show_stats_col = layer_stats is not None

    # Figure sizing
    if figsize is None:
        width = max(8, seq_len * 0.35 + (2 if show_stats_col else 0))
        height_per_grid = max(3, n_layers * 0.15)
        height = (
            1.5
            + height_per_grid * n_grids
            + 0.5 * max(0, n_grids - 1)
            + (0.5 if title else 0)
        )
        figsize = (min(width, 40), min(height, 30))

    fig = plt.figure(figsize=figsize)

    n_rows = n_grids + (1 if show_surprise else 0)
    height_ratios = []
    if show_surprise:
        height_ratios.append(1)
    for _ in range(n_grids):
        height_ratios.append(n_layers)

    n_cols = 2 if show_stats_col else 1
    width_ratios = [seq_len, max(2, seq_len // 10)] if show_stats_col else [1]

    gs = GridSpec(
        n_rows, n_cols,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0.3,
        wspace=0.05,
    )

    row = 0

    # --- Surprise strip ---
    if show_surprise and log_probs is not None:
        ax_surprise = fig.add_subplot(gs[row, 0])
        lp_display = log_probs[np.newaxis, :]
        ax_surprise.imshow(
            lp_display,
            aspect="auto",
            cmap="RdYlGn",
            interpolation="nearest",
        )
        ax_surprise.set_yticks([])
        ax_surprise.set_ylabel("surprise", fontsize=8, rotation=0, labelpad=40)

        ax_surprise.set_xticks(range(seq_len))
        ax_surprise.set_xticklabels(
            tokens,
            rotation=60,
            ha="right",
            fontsize=max(5, min(8, 200 // max(seq_len, 1))),
            fontfamily="monospace",
        )
        ax_surprise.xaxis.set_ticks_position("top")
        ax_surprise.xaxis.set_label_position("top")

        if show_stats_col:
            ax_empty = fig.add_subplot(gs[row, 1])
            ax_empty.axis("off")

        row += 1

    # --- Signal grids ---
    for grid_idx, (rgb, sig_name) in enumerate(zip(signal_rgbs, signal_names)):
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(rgb, aspect="auto", interpolation="nearest")
        ax.set_ylabel(sig_name, fontsize=9, rotation=0, labelpad=60)

        if n_layers <= 64:
            ax.set_yticks(range(0, n_layers, max(1, n_layers // 16)))
        else:
            ax.set_yticks(range(0, n_layers, n_layers // 8))

        # Token labels on first grid if no surprise strip
        if not show_surprise and grid_idx == 0:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(
                tokens,
                rotation=60,
                ha="right",
                fontsize=max(5, min(8, 200 // max(seq_len, 1))),
                fontfamily="monospace",
            )
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")
        elif grid_idx == len(signal_rgbs) - 1:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(
                [str(i) for i in range(seq_len)],
                fontsize=6,
            )
        else:
            ax.set_xticks([])

        # Layer stats on the right
        if show_stats_col and layer_stats is not None:
            ax_stats = fig.add_subplot(gs[row, 1])
            stats = layer_stats[:, grid_idx] if grid_idx < layer_stats.shape[1] else np.zeros(n_layers)
            ax_stats.barh(
                range(n_layers), stats,
                color="steelblue", alpha=0.7, height=0.8,
            )
            ax_stats.set_ylim(-0.5, n_layers - 0.5)
            ax_stats.invert_yaxis()
            ax_stats.set_yticks([])
            ax_stats.set_xlabel("energy", fontsize=7)
            ax_stats.tick_params(axis="x", labelsize=6)

        row += 1

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)

    return fig
