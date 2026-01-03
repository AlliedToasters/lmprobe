"""Plotting utilities for lmprobe.

This module provides visualization functions for layer importance
and other probe diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _check_plotting_installed() -> None:
    """Check that plotting dependencies are installed."""
    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
    except ImportError:
        raise ImportError(
            "Plotting requires matplotlib and seaborn. "
            "Install them with: pip install lmprobe[plot]"
        )


def plot_layer_importance(
    candidate_layers: list[int],
    layer_importances: np.ndarray,
    selected_layers: list[int] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Layer Importance (Group Lasso Norms)",
    xlabel: str = "Layer Index",
    ylabel: str = "Importance (L2 Norm)",
    highlight_selected: bool = True,
    bar_color: str = "steelblue",
    selected_color: str = "coral",
) -> tuple[Figure, Axes]:
    """Plot layer importance scores from Group Lasso.

    Parameters
    ----------
    candidate_layers : list[int]
        List of candidate layer indices that were evaluated.
    layer_importances : np.ndarray
        Importance scores (group norms) for each candidate layer.
    selected_layers : list[int] | None
        Layers that were selected (non-zero importance). If provided and
        highlight_selected is True, these will be highlighted.
    ax : Axes | None
        Matplotlib axes to plot on. If None, creates a new figure.
    figsize : tuple[float, float]
        Figure size if creating a new figure.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    highlight_selected : bool
        Whether to highlight selected layers in a different color.
    bar_color : str
        Color for non-selected bars.
    selected_color : str
        Color for selected layer bars.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes objects.

    Examples
    --------
    >>> probe = LinearProbe(model="...", layers="auto")
    >>> probe.fit(positive_prompts, negative_prompts)
    >>> fig, ax = plot_layer_importance(
    ...     probe.candidate_layers_,
    ...     probe.layer_importances_,
    ...     probe.selected_layers_,
    ... )
    >>> fig.savefig("layer_importance.png")
    """
    _check_plotting_installed()

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set seaborn style
    sns.set_style("whitegrid")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Prepare bar colors
    if highlight_selected and selected_layers is not None:
        selected_set = set(selected_layers)
        colors = [
            selected_color if layer in selected_set else bar_color
            for layer in candidate_layers
        ]
    else:
        colors = bar_color

    # Create bar plot
    x_positions = np.arange(len(candidate_layers))
    bars = ax.bar(x_positions, layer_importances, color=colors, edgecolor="white")

    # Customize axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(candidate_layers)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add a horizontal line at zero for reference
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    # Add legend if highlighting
    if highlight_selected and selected_layers is not None:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=selected_color, edgecolor="white", label="Selected"),
            Patch(facecolor=bar_color, edgecolor="white", label="Not selected"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    return fig, ax


def plot_layer_importance_heatmap(
    candidate_layers: list[int],
    layer_importances: np.ndarray,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 2),
    title: str = "Layer Importance Heatmap",
    cmap: str = "YlOrRd",
) -> tuple[Figure, Axes]:
    """Plot layer importance as a horizontal heatmap.

    This visualization is useful for seeing the relative importance
    across layers at a glance.

    Parameters
    ----------
    candidate_layers : list[int]
        List of candidate layer indices.
    layer_importances : np.ndarray
        Importance scores for each candidate layer.
    ax : Axes | None
        Matplotlib axes to plot on. If None, creates a new figure.
    figsize : tuple[float, float]
        Figure size if creating a new figure.
    title : str
        Plot title.
    cmap : str
        Colormap name.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes objects.
    """
    _check_plotting_installed()

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Reshape for heatmap (1 row)
    importance_matrix = layer_importances.reshape(1, -1)

    # Create heatmap
    sns.heatmap(
        importance_matrix,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".3f",
        xticklabels=candidate_layers,
        yticklabels=["Importance"],
        cbar_kws={"label": "L2 Norm"},
    )

    ax.set_xlabel("Layer Index")
    ax.set_title(title)

    plt.tight_layout()

    return fig, ax
