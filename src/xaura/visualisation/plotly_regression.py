"""Plotly interactive charts for regression models.

Generates four interactive plots from a regression Result:
    1. Residuals vs Fitted — spot non-linearity and heteroscedasticity
    2. Q-Q Plot — check if residuals are normally distributed
    3. Predicted vs Actual — visual accuracy check
    4. Residual Distribution — histogram of residual spread

Each function returns a plotly.graph_objects.Figure that can be:
    - Displayed in a notebook: fig.show()
    - Serialised to JSON: fig.to_json()
    - Embedded in a web page via Plotly.js
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from xaura.models.base import Result
from xaura.visualisation._style import (
    PRIMARY,
    REFERENCE_LINE,
    plotly_base_layout,
)

# ---------------------------------------------------------------------------
# 1. Residuals vs Fitted
# ---------------------------------------------------------------------------


def residuals_vs_fitted(result: Result) -> go.Figure:
    """Scatter plot of residuals against fitted values.

    A good regression model shows residuals randomly scattered around
    zero. Patterns indicate non-linearity; funnel shapes indicate
    heteroscedasticity.

    Args:
        result: A regression Result with y_test and predictions.

    Returns:
        Plotly Figure.
    """
    y_true = np.asarray(result.y_test)
    y_pred = np.asarray(result.predictions)
    residuals = y_true - y_pred

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker={"color": PRIMARY, "size": 6, "opacity": 0.65},
            name="Residuals",
            hovertemplate="Fitted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=REFERENCE_LINE,
        line_width=1.5,
        annotation_text="0",
        annotation_position="top left",
        annotation_font_color=REFERENCE_LINE,
    )

    fig.update_layout(
        **plotly_base_layout(
            title="Residuals vs Fitted Values",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 2. Q-Q Plot
# ---------------------------------------------------------------------------


def qq_plot(result: Result) -> go.Figure:
    """Quantile-Quantile plot of residuals against the normal distribution.

    Points close to the diagonal indicate normally distributed residuals.
    Deviations at the tails indicate skew or heavy tails.

    Args:
        result: A regression Result with y_test and predictions.

    Returns:
        Plotly Figure.
    """
    y_true = np.asarray(result.y_test)
    y_pred = np.asarray(result.predictions)
    residuals = y_true - y_pred

    std_residuals = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-10)

    sorted_residuals = np.sort(std_residuals)
    n = len(sorted_residuals)
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, n))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=sorted_residuals,
            mode="markers",
            marker={"color": PRIMARY, "size": 5, "opacity": 0.65},
            name="Residuals",
            hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>",
        )
    )

    # Reference line (y = x)
    q_min = min(theoretical.min(), sorted_residuals.min())
    q_max = max(theoretical.max(), sorted_residuals.max())
    fig.add_trace(
        go.Scatter(
            x=[q_min, q_max],
            y=[q_min, q_max],
            mode="lines",
            line={"color": REFERENCE_LINE, "dash": "dash", "width": 1.5},
            name="Normal",
            showlegend=True,
        )
    )

    fig.update_layout(
        **plotly_base_layout(
            title="Normal Q-Q Plot",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles (Standardised Residuals)",
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 3. Predicted vs Actual
# ---------------------------------------------------------------------------


def predicted_vs_actual(result: Result) -> go.Figure:
    """Scatter plot of predicted values vs actual values.

    A perfect model would place all points on the y = x line.
    Deviations show where the model over- or under-predicts.

    Args:
        result: A regression Result with y_test and predictions.

    Returns:
        Plotly Figure.
    """
    y_true = np.asarray(result.y_test)
    y_pred = np.asarray(result.predictions)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker={"color": PRIMARY, "size": 6, "opacity": 0.65},
            name="Predictions",
            hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>",
        )
    )

    # Perfect prediction line
    v_min = min(y_true.min(), y_pred.min())
    v_max = max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[v_min, v_max],
            y=[v_min, v_max],
            mode="lines",
            line={"color": REFERENCE_LINE, "dash": "dash", "width": 1.5},
            name="Perfect (y = x)",
            showlegend=True,
        )
    )

    fig.update_layout(
        **plotly_base_layout(
            title="Predicted vs Actual",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 4. Residual Distribution
# ---------------------------------------------------------------------------


def residual_distribution(result: Result) -> go.Figure:
    """Histogram of residuals.

    A bell-shaped distribution centred on zero indicates well-behaved
    residuals. Skew or fat tails are easy to spot.

    Args:
        result: A regression Result with y_test and predictions.

    Returns:
        Plotly Figure.
    """
    y_true = np.asarray(result.y_test)
    y_pred = np.asarray(result.predictions)
    residuals = y_true - y_pred

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color=PRIMARY,
            opacity=0.75,
            name="Residuals",
            hovertemplate="Residual: %{x:.3f}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        **plotly_base_layout(
            title="Residual Distribution",
            xaxis_title="Residual",
            yaxis_title="Frequency",
            bargap=0.05,
        )
    )

    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all regression plots at once
# ---------------------------------------------------------------------------


def all_regression_plots(result: Result) -> dict[str, go.Figure]:
    """Generate all four regression plots.

    Args:
        result: A regression Result.

    Returns:
        Dict mapping plot name → Figure.
    """
    return {
        "residuals_vs_fitted": residuals_vs_fitted(result),
        "qq_plot": qq_plot(result),
        "predicted_vs_actual": predicted_vs_actual(result),
        "residual_distribution": residual_distribution(result),
    }
