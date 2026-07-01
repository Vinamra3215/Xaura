"""Tests for regression visualisation charts (Plotly + Matplotlib).

Tests that all chart functions return valid Figure objects, contain
the expected traces/axes, and can save to disk without errors.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import xaura.models.regressors  # noqa: F401
from xaura import profile, run_model
from xaura.visualisation.plotly_regression import (
    all_regression_plots,
    predicted_vs_actual,
    qq_plot,
    residual_distribution,
    residuals_vs_fitted,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def regression_result():
    """Run a regression model once and reuse the Result for all tests."""
    np.random.seed(42)
    x1 = np.random.randn(200)
    x2 = np.random.randn(200)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "target": 3 * x1 + 2 * x2 + np.random.randn(200) * 0.5,
        }
    )
    p = profile(df)
    return run_model("ridge", df, p, target_col="target", auto_log=False)


# ---------------------------------------------------------------------------
# Plotly regression chart tests
# ---------------------------------------------------------------------------


class TestPlotlyResiduals:
    """Tests for residuals_vs_fitted."""

    def test_returns_figure(self, regression_result):
        fig = residuals_vs_fitted(regression_result)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, regression_result):
        fig = residuals_vs_fitted(regression_result)
        assert len(fig.data) >= 1

    def test_scatter_trace(self, regression_result):
        fig = residuals_vs_fitted(regression_result)
        scatter = fig.data[0]
        assert scatter.type == "scatter"
        assert len(scatter.x) > 0

    def test_json_serialisable(self, regression_result):
        fig = residuals_vs_fitted(regression_result)
        json_str = fig.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 100


class TestPlotlyQQ:
    """Tests for qq_plot."""

    def test_returns_figure(self, regression_result):
        fig = qq_plot(regression_result)
        assert isinstance(fig, go.Figure)

    def test_has_reference_line(self, regression_result):
        fig = qq_plot(regression_result)
        # Should have 2 traces: data points + reference line
        assert len(fig.data) >= 2

    def test_json_serialisable(self, regression_result):
        fig = qq_plot(regression_result)
        assert len(fig.to_json()) > 100


class TestPlotlyPredVsActual:
    """Tests for predicted_vs_actual."""

    def test_returns_figure(self, regression_result):
        fig = predicted_vs_actual(regression_result)
        assert isinstance(fig, go.Figure)

    def test_has_reference_line(self, regression_result):
        fig = predicted_vs_actual(regression_result)
        assert len(fig.data) >= 2

    def test_json_serialisable(self, regression_result):
        fig = predicted_vs_actual(regression_result)
        assert len(fig.to_json()) > 100


class TestPlotlyResidualDist:
    """Tests for residual_distribution."""

    def test_returns_figure(self, regression_result):
        fig = residual_distribution(regression_result)
        assert isinstance(fig, go.Figure)

    def test_histogram_trace(self, regression_result):
        fig = residual_distribution(regression_result)
        assert fig.data[0].type == "histogram"

    def test_json_serialisable(self, regression_result):
        fig = residual_distribution(regression_result)
        assert len(fig.to_json()) > 100


class TestPlotlyAllRegression:
    """Tests for all_regression_plots convenience function."""

    def test_returns_dict(self, regression_result):
        plots = all_regression_plots(regression_result)
        assert isinstance(plots, dict)

    def test_all_four_plots(self, regression_result):
        plots = all_regression_plots(regression_result)
        expected = {
            "residuals_vs_fitted",
            "qq_plot",
            "predicted_vs_actual",
            "residual_distribution",
        }
        assert set(plots.keys()) == expected

    def test_all_are_figures(self, regression_result):
        plots = all_regression_plots(regression_result)
        for name, fig in plots.items():
            assert isinstance(fig, go.Figure), f"{name} is not a Figure"


# ---------------------------------------------------------------------------
# Matplotlib regression chart tests
# ---------------------------------------------------------------------------


class TestMatplotlibRegression:
    """Tests for matplotlib static regression charts."""

    def test_residuals_returns_figure(self, regression_result):
        import matplotlib.pyplot as plt

        from xaura.visualisation.matplotlib_regression import (
            residuals_vs_fitted as mpl_rvf,
        )

        fig = mpl_rvf(regression_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_qq_returns_figure(self, regression_result):
        import matplotlib.pyplot as plt

        from xaura.visualisation.matplotlib_regression import qq_plot as mpl_qq

        fig = mpl_qq(regression_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pred_vs_actual_returns_figure(self, regression_result):
        import matplotlib.pyplot as plt

        from xaura.visualisation.matplotlib_regression import (
            predicted_vs_actual as mpl_pva,
        )

        fig = mpl_pva(regression_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_residual_dist_returns_figure(self, regression_result):
        import matplotlib.pyplot as plt

        from xaura.visualisation.matplotlib_regression import (
            residual_distribution as mpl_rd,
        )

        fig = mpl_rd(regression_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_to_png(self, regression_result, tmp_path):
        import matplotlib.pyplot as plt

        from xaura.visualisation.matplotlib_regression import (
            residuals_vs_fitted as mpl_rvf,
        )

        save_path = tmp_path / "test_residuals.png"
        fig = mpl_rvf(regression_result, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        plt.close(fig)

    def test_all_plots_saves_all(self, regression_result, tmp_path):
        import matplotlib.pyplot as plt

        from xaura.visualisation.matplotlib_regression import (
            all_regression_plots as mpl_all,
        )

        figs = mpl_all(regression_result, output_dir=tmp_path, fmt="png")
        assert len(figs) == 4
        for name in figs:
            assert (tmp_path / f"{name}.png").exists()
        plt.close("all")
