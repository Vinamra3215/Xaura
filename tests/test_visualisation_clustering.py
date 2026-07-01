"""Tests for clustering visualisation charts (Plotly + Matplotlib).

Tests that all chart functions return valid Figure objects, handle
DBSCAN noise labels correctly, and can save to disk.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import xaura.models.clusterers  # noqa: F401
from xaura import profile, run_model
from xaura.visualisation.plotly_clustering import (
    all_clustering_plots,
    cluster_scatter_pca,
    dendrogram_plot,
    elbow_curve,
    silhouette_plot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clustering_result():
    """Run K-Means once and reuse the Result."""
    np.random.seed(42)
    c1 = np.random.randn(70, 2) + np.array([0, 0])
    c2 = np.random.randn(65, 2) + np.array([6, 6])
    c3 = np.random.randn(65, 2) + np.array([-6, 6])
    data = np.vstack([c1, c2, c3])
    df = pd.DataFrame(data, columns=["f1", "f2"])
    p = profile(df)
    return run_model("kmeans", df, p, auto_log=False)


@pytest.fixture(scope="module")
def dbscan_result():
    """Run DBSCAN once — may produce noise labels."""
    np.random.seed(42)
    c1 = np.random.randn(70, 2) + np.array([0, 0])
    c2 = np.random.randn(65, 2) + np.array([10, 10])
    data = np.vstack([c1, c2])
    df = pd.DataFrame(data, columns=["f1", "f2"])
    p = profile(df)
    return run_model(
        "dbscan",
        df,
        p,
        config={"eps": 2.0, "min_samples": 5},
        auto_log=False,
    )


# ---------------------------------------------------------------------------
# Plotly clustering chart tests
# ---------------------------------------------------------------------------


class TestPlotlyClusterScatter:
    """Tests for cluster_scatter_pca."""

    def test_returns_figure(self, clustering_result):
        fig = cluster_scatter_pca(clustering_result)
        assert isinstance(fig, go.Figure)

    def test_has_cluster_traces(self, clustering_result):
        fig = cluster_scatter_pca(clustering_result)
        # K-Means with 3 clusters → 3 traces
        assert len(fig.data) >= 3

    def test_json_serialisable(self, clustering_result):
        fig = cluster_scatter_pca(clustering_result)
        assert len(fig.to_json()) > 100

    def test_dbscan_noise_handled(self, dbscan_result):
        """DBSCAN results should not crash the chart."""
        fig = cluster_scatter_pca(dbscan_result)
        assert isinstance(fig, go.Figure)


class TestPlotlySilhouette:
    """Tests for silhouette_plot."""

    def test_returns_figure(self, clustering_result):
        fig = silhouette_plot(clustering_result)
        assert isinstance(fig, go.Figure)

    def test_has_cluster_bars(self, clustering_result):
        fig = silhouette_plot(clustering_result)
        assert len(fig.data) >= 3

    def test_json_serialisable(self, clustering_result):
        fig = silhouette_plot(clustering_result)
        assert len(fig.to_json()) > 100

    def test_dbscan_handled(self, dbscan_result):
        fig = silhouette_plot(dbscan_result)
        assert isinstance(fig, go.Figure)


class TestPlotlyElbow:
    """Tests for elbow_curve (standalone, takes raw X)."""

    def test_returns_figure(self, clustering_result):
        X = np.asarray(clustering_result.X_train)
        fig = elbow_curve(X, k_range=range(2, 6))
        assert isinstance(fig, go.Figure)

    def test_line_trace(self, clustering_result):
        X = np.asarray(clustering_result.X_train)
        fig = elbow_curve(X, k_range=range(2, 6))
        assert fig.data[0].mode == "lines+markers"
        assert len(fig.data[0].x) == 4  # k=2,3,4,5

    def test_json_serialisable(self, clustering_result):
        X = np.asarray(clustering_result.X_train)
        fig = elbow_curve(X, k_range=range(2, 5))
        assert len(fig.to_json()) > 100


class TestPlotlyDendrogram:
    """Tests for dendrogram_plot."""

    def test_returns_figure(self, clustering_result):
        X = np.asarray(clustering_result.X_train)
        fig = dendrogram_plot(X)
        assert isinstance(fig, go.Figure)

    def test_has_line_traces(self, clustering_result):
        X = np.asarray(clustering_result.X_train)
        fig = dendrogram_plot(X)
        assert len(fig.data) > 0

    def test_json_serialisable(self, clustering_result):
        X = np.asarray(clustering_result.X_train)
        fig = dendrogram_plot(X, p=10)
        assert len(fig.to_json()) > 100


class TestPlotlyAllClustering:
    """Tests for all_clustering_plots convenience function."""

    def test_returns_dict(self, clustering_result):
        plots = all_clustering_plots(clustering_result, show_elbow=False, show_dendrogram=False)
        assert isinstance(plots, dict)

    def test_base_plots_present(self, clustering_result):
        plots = all_clustering_plots(clustering_result, show_elbow=False, show_dendrogram=False)
        assert "cluster_scatter_pca" in plots
        assert "silhouette_plot" in plots

    def test_all_plots_present(self, clustering_result):
        plots = all_clustering_plots(clustering_result)
        assert len(plots) == 4


# ---------------------------------------------------------------------------
# Matplotlib clustering chart tests
# ---------------------------------------------------------------------------


class TestMatplotlibClustering:
    """Tests for matplotlib static clustering charts."""

    def test_scatter_returns_figure(self, clustering_result):
        import matplotlib.pyplot as plt

        from xaura.visualisation.matplotlib_clustering import (
            cluster_scatter_pca as mpl_scatter,
        )

        fig = mpl_scatter(clustering_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_silhouette_returns_figure(self, clustering_result):
        import matplotlib.pyplot as plt

        from xaura.visualisation.matplotlib_clustering import (
            silhouette_plot as mpl_sil,
        )

        fig = mpl_sil(clustering_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_elbow_returns_figure(self, clustering_result):
        import matplotlib.pyplot as plt

        from xaura.visualisation.matplotlib_clustering import (
            elbow_curve as mpl_elbow,
        )

        X = np.asarray(clustering_result.X_train)
        fig = mpl_elbow(X, k_range=range(2, 5))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dendrogram_returns_figure(self, clustering_result):
        import matplotlib.pyplot as plt

        from xaura.visualisation.matplotlib_clustering import (
            dendrogram_plot as mpl_dendro,
        )

        X = np.asarray(clustering_result.X_train)
        fig = mpl_dendro(X)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_to_png(self, clustering_result, tmp_path):
        import matplotlib.pyplot as plt

        from xaura.visualisation.matplotlib_clustering import (
            cluster_scatter_pca as mpl_scatter,
        )

        save_path = tmp_path / "test_scatter.png"
        fig = mpl_scatter(clustering_result, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        plt.close(fig)

    def test_all_plots_saves_all(self, clustering_result, tmp_path):
        import matplotlib.pyplot as plt

        from xaura.visualisation.matplotlib_clustering import (
            all_clustering_plots as mpl_all,
        )

        figs = mpl_all(clustering_result, output_dir=tmp_path, fmt="png")
        assert len(figs) == 4
        for name in figs:
            assert (tmp_path / f"{name}.png").exists()
        plt.close("all")
