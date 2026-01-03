"""Tests for plotting utilities."""

import numpy as np
import pytest

# Skip all tests if matplotlib/seaborn not installed
matplotlib = pytest.importorskip("matplotlib")
seaborn = pytest.importorskip("seaborn")


class TestPlotLayerImportance:
    """Tests for plot_layer_importance function."""

    def test_basic_plot(self):
        """Basic plotting works."""
        from lmprobe.plotting import plot_layer_importance

        candidate_layers = [8, 16, 24]
        layer_importances = np.array([0.1, 0.5, 0.2])

        fig, ax = plot_layer_importance(candidate_layers, layer_importances)

        assert fig is not None
        assert ax is not None
        matplotlib.pyplot.close(fig)

    def test_with_selected_layers(self):
        """Highlighting selected layers works."""
        from lmprobe.plotting import plot_layer_importance

        candidate_layers = [8, 16, 24]
        layer_importances = np.array([0.1, 0.5, 0.2])
        selected_layers = [16]

        fig, ax = plot_layer_importance(
            candidate_layers,
            layer_importances,
            selected_layers=selected_layers,
            highlight_selected=True,
        )

        assert fig is not None
        assert ax is not None
        matplotlib.pyplot.close(fig)

    def test_custom_ax(self):
        """Using custom axes works."""
        import matplotlib.pyplot as plt

        from lmprobe.plotting import plot_layer_importance

        fig_custom, ax_custom = plt.subplots()

        candidate_layers = [8, 16, 24]
        layer_importances = np.array([0.1, 0.5, 0.2])

        fig_returned, ax_returned = plot_layer_importance(
            candidate_layers,
            layer_importances,
            ax=ax_custom,
        )

        # Should return the same axes we passed in
        assert ax_returned is ax_custom
        matplotlib.pyplot.close(fig_custom)

    def test_custom_figsize(self):
        """Custom figsize works."""
        from lmprobe.plotting import plot_layer_importance

        candidate_layers = [8, 16, 24]
        layer_importances = np.array([0.1, 0.5, 0.2])

        fig, ax = plot_layer_importance(
            candidate_layers,
            layer_importances,
            figsize=(15, 8),
        )

        assert fig is not None
        matplotlib.pyplot.close(fig)

    def test_custom_colors(self):
        """Custom colors work."""
        from lmprobe.plotting import plot_layer_importance

        candidate_layers = [8, 16, 24]
        layer_importances = np.array([0.1, 0.5, 0.2])
        selected_layers = [16]

        fig, ax = plot_layer_importance(
            candidate_layers,
            layer_importances,
            selected_layers=selected_layers,
            bar_color="navy",
            selected_color="crimson",
        )

        assert fig is not None
        matplotlib.pyplot.close(fig)

    def test_no_highlight(self):
        """Disabling highlighting works."""
        from lmprobe.plotting import plot_layer_importance

        candidate_layers = [8, 16, 24]
        layer_importances = np.array([0.1, 0.5, 0.2])
        selected_layers = [16]

        fig, ax = plot_layer_importance(
            candidate_layers,
            layer_importances,
            selected_layers=selected_layers,
            highlight_selected=False,
        )

        assert fig is not None
        matplotlib.pyplot.close(fig)


class TestPlotLayerImportanceHeatmap:
    """Tests for plot_layer_importance_heatmap function."""

    def test_basic_heatmap(self):
        """Basic heatmap plotting works."""
        from lmprobe.plotting import plot_layer_importance_heatmap

        candidate_layers = [8, 16, 24]
        layer_importances = np.array([0.1, 0.5, 0.2])

        fig, ax = plot_layer_importance_heatmap(candidate_layers, layer_importances)

        assert fig is not None
        assert ax is not None
        matplotlib.pyplot.close(fig)

    def test_custom_cmap(self):
        """Custom colormap works."""
        from lmprobe.plotting import plot_layer_importance_heatmap

        candidate_layers = [8, 16, 24]
        layer_importances = np.array([0.1, 0.5, 0.2])

        fig, ax = plot_layer_importance_heatmap(
            candidate_layers,
            layer_importances,
            cmap="Blues",
        )

        assert fig is not None
        matplotlib.pyplot.close(fig)


class TestLinearProbePlotMethod:
    """Tests for LinearProbe.plot_layer_importance method."""

    def test_plot_requires_auto_layers(self, tiny_model):
        """plot_layer_importance raises error without layers='auto'."""
        from lmprobe import LinearProbe

        # Non-auto probe
        probe = LinearProbe(
            model=tiny_model,
            layers=-1,
            device="cpu",
            remote=False,
        )
        probe.fit(["positive"], ["negative"])

        with pytest.raises(RuntimeError, match="layers='auto'"):
            probe.plot_layer_importance()

    def test_plot_requires_fit(self, tiny_model):
        """plot_layer_importance raises error before fit."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=tiny_model,
            layers="auto",
            device="cpu",
            remote=False,
        )

        with pytest.raises(RuntimeError, match="layers='auto'"):
            probe.plot_layer_importance()


# These tests require skglm for layers="auto"
skglm = pytest.importorskip("skglm")


class TestLinearProbePlotWithAuto:
    """Tests for LinearProbe.plot_layer_importance with layers='auto'."""

    def test_plot_after_auto_fit(self, tiny_model):
        """plot_layer_importance works after fitting with layers='auto'."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=tiny_model,
            layers="auto",
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive example"], ["negative example"])

        fig, ax = probe.plot_layer_importance()

        assert fig is not None
        assert ax is not None
        matplotlib.pyplot.close(fig)

    def test_plot_with_custom_params(self, tiny_model):
        """plot_layer_importance accepts custom parameters."""
        from lmprobe import LinearProbe

        probe = LinearProbe(
            model=tiny_model,
            layers="auto",
            pooling="last_token",
            device="cpu",
            remote=False,
            random_state=42,
        )

        probe.fit(["positive example"], ["negative example"])

        fig, ax = probe.plot_layer_importance(
            figsize=(12, 8),
            title="Custom Title",
            bar_color="navy",
        )

        assert ax.get_title() == "Custom Title"
        matplotlib.pyplot.close(fig)
