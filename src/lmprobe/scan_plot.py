"""Hero figure renderer for SampleScan.

Renders a 2D RGB image of a model forward pass on a single prompt:
tokens on x-axis, layers on y-axis, RGB channels from top-3 PCA
components of each signal at that coordinate. Surprise (log-prob)
is appended as a top row in the heatmap.
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


def _logprob_to_rgb(log_probs: np.ndarray) -> np.ndarray:
    """Convert log-prob values to RGB using RdYlGn colormap."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("RdYlGn")
    # Normalize: clip to [-10, 0] range, map to [0, 1]
    normed = np.clip(log_probs, -10, 0) / -10  # 0 = high prob, 1 = low prob
    normed = 1 - normed  # flip so green = high prob
    rgba = cmap(normed)
    return rgba[:, :3].astype(np.float32)  # drop alpha


def render_scan_figure(
    projections: np.ndarray,
    tokens: list[str],
    signal_names: list[str],
    log_probs: np.ndarray | None = None,
    layer_stats: np.ndarray | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    title: str = "",
    generative_mask: np.ndarray | None = None,
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
    generative_mask : np.ndarray or None
        Boolean mask shape [seq_len]. True = generative (assistant) token,
        False = prompt token. Prompt tokens are grayed out in the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    seq_len, n_layers, n_signals, k = projections.shape
    n_grids = n_signals
    show_surprise = log_probs is not None
    show_stats_col = layer_stats is not None

    # Build RGB image per signal: [n_layers, seq_len, 3]
    # If surprise is shown, prepend a surprise row: [n_layers+1, seq_len, 3]
    signal_rgbs = []
    for sig_idx in range(n_signals):
        rgb = _normalize_rgb(projections[:, :, sig_idx, :3])  # [seq_len, n_layers, 3]
        rgb = rgb.transpose(1, 0, 2)  # [n_layers, seq_len, 3]
        # Gray out non-generative (prompt) tokens
        if generative_mask is not None:
            prompt_cols = ~generative_mask
            gray = rgb[..., :1].mean(axis=-1, keepdims=True)
            gray = np.broadcast_to(gray, rgb.shape).copy()
            rgb[:, prompt_cols, :] = gray[:, prompt_cols, :] * 0.4 + 0.3
        # Prepend surprise row
        if show_surprise and log_probs is not None:
            surprise_rgb = _logprob_to_rgb(log_probs)[np.newaxis, :, :]  # [1, seq_len, 3]
            rgb = np.concatenate([surprise_rgb, rgb], axis=0)  # [n_layers+1, seq_len, 3]
        signal_rgbs.append(rgb)

    total_rows = n_layers + (1 if show_surprise else 0)

    # Figure sizing
    if figsize is None:
        width = max(8, seq_len * 0.35 + (2 if show_stats_col else 0))
        height_per_grid = max(3, total_rows * 0.15)
        height = (
            1.5
            + height_per_grid * n_grids
            + 0.5 * max(0, n_grids - 1)
            + (0.5 if title else 0)
        )
        figsize = (min(width, 40), min(height, 30))

    fig = plt.figure(figsize=figsize)

    n_cols = 2 if show_stats_col else 1
    width_ratios = [seq_len, max(2, seq_len // 10)] if show_stats_col else [1]

    gs = GridSpec(
        n_grids, n_cols,
        figure=fig,
        width_ratios=width_ratios,
        hspace=0.3,
        wspace=0.05,
    )

    # Y-tick labels: "surprise" row + layer numbers
    if show_surprise:
        ytick_positions = list(range(total_rows))
        ytick_labels = ["S"] + [str(i) if i % 4 == 0 else "" for i in range(n_layers)]
    else:
        ytick_positions = list(range(0, n_layers, max(1, n_layers // 16)))
        ytick_labels = [str(i) for i in ytick_positions]

    # Token text for x-axis
    token_fontsize = max(4, min(8, 200 // max(seq_len, 1)))

    for grid_idx, (rgb, sig_name) in enumerate(zip(signal_rgbs, signal_names)):
        ax = fig.add_subplot(gs[grid_idx, 0])
        ax.imshow(rgb, aspect="auto", interpolation="nearest")
        ax.set_ylabel(sig_name, fontsize=9, rotation=0, labelpad=60)

        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=6)

        # Token text on x-axis (top for first grid, bottom for last)
        ax.set_xticks(range(seq_len))
        if grid_idx == 0:
            ax.set_xticklabels(
                tokens,
                rotation=60,
                ha="right",
                fontsize=token_fontsize,
                fontfamily="monospace",
            )
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")
        elif grid_idx == len(signal_rgbs) - 1:
            ax.set_xticklabels(
                tokens,
                rotation=60,
                ha="right",
                fontsize=token_fontsize,
                fontfamily="monospace",
            )
        else:
            ax.set_xticklabels([])

        # Layer stats on the right
        if show_stats_col and layer_stats is not None:
            ax_stats = fig.add_subplot(gs[grid_idx, 1])
            if grid_idx < layer_stats.shape[1]:
                stats = layer_stats[:, grid_idx]
            else:
                stats = np.zeros(n_layers)
            # Offset by 1 if surprise row is present
            y_offset = 1 if show_surprise else 0
            ax_stats.barh(
                np.arange(n_layers) + y_offset, stats,
                color="steelblue", alpha=0.7, height=0.8,
            )
            ax_stats.set_ylim(-0.5, total_rows - 0.5)
            ax_stats.invert_yaxis()
            ax_stats.set_yticks([])
            ax_stats.set_xlabel("energy", fontsize=7)
            ax_stats.tick_params(axis="x", labelsize=6)

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)

    return fig
