"""Matplotlib static charts for regression models.

Static PNG/PDF versions of the four regression diagnostic plots,
using the shared dark theme to match the Plotly interactive charts.

Usage:
    from xaura.visualisation.matplotlib_regression import all_regression_plots
    figs = all_regression_plots(result, output_dir="./plots")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from xaura.models.base import Result
from xaura.visualisation._style import (
    MPL_DPI,
    MPL_FIG_SIZE,
    MPL_RC_PARAMS,
    PRIMARY,
    REFERENCE_LINE,
    TEXT_COLOR,
)


def _apply_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent dark-theme styling to an axes."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12, color=TEXT_COLOR)
    ax.set_xlabel(xlabel, fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, fontsize=11, color=TEXT_COLOR)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.4, linestyle="--")


def _dark_fig(**kwargs: object) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure + axes with dark theme RC params applied."""
    with plt.rc_context(MPL_RC_PARAMS):
        fig, ax = plt.subplots(figsize=MPL_FIG_SIZE, dpi=MPL_DPI, **kwargs)
    return fig, ax


# ---------------------------------------------------------------------------
# 1. Residuals vs Fitted
# ---------------------------------------------------------------------------


def residuals_vs_fitted(
    result: Result,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Static scatter plot of residuals vs fitted values.

    Args:
        result: A regression Result.
        save_path: If provided, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    y_true = np.asarray(result.y_test)
    y_pred = np.asarray(result.predictions)
    residuals = y_true - y_pred

    with plt.rc_context(MPL_RC_PARAMS):
        fig, ax = plt.subplots(figsize=MPL_FIG_SIZE, dpi=MPL_DPI)
        ax.scatter(y_pred, residuals, c=PRIMARY, alpha=0.6, s=20, edgecolors="none")
        ax.axhline(y=0, color=REFERENCE_LINE, linestyle="--", linewidth=1.5)
        _apply_style(ax, "Residuals vs Fitted Values", "Fitted Values", "Residuals")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Q-Q Plot
# ---------------------------------------------------------------------------


def qq_plot(
    result: Result,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Static Q-Q plot of standardised residuals.

    Args:
        result: A regression Result.
        save_path: If provided, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    y_true = np.asarray(result.y_test)
    y_pred = np.asarray(result.predictions)
    residuals = y_true - y_pred
    std_residuals = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-10)

    with plt.rc_context(MPL_RC_PARAMS):
        fig, ax = plt.subplots(figsize=MPL_FIG_SIZE, dpi=MPL_DPI)
        stats.probplot(std_residuals, dist="norm", plot=ax)

        # Re-style the probplot output for dark theme
        ax.get_lines()[0].set(color=PRIMARY, markersize=4, alpha=0.6)
        ax.get_lines()[1].set(color=REFERENCE_LINE, linestyle="--", linewidth=1.5)
        _apply_style(ax, "Normal Q-Q Plot", "Theoretical Quantiles", "Sample Quantiles")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Predicted vs Actual
# ---------------------------------------------------------------------------


def predicted_vs_actual(
    result: Result,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Static scatter plot of predicted vs actual values.

    Args:
        result: A regression Result.
        save_path: If provided, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    y_true = np.asarray(result.y_test)
    y_pred = np.asarray(result.predictions)

    with plt.rc_context(MPL_RC_PARAMS):
        fig, ax = plt.subplots(figsize=MPL_FIG_SIZE, dpi=MPL_DPI)
        ax.scatter(y_true, y_pred, c=PRIMARY, alpha=0.6, s=20, edgecolors="none")

        v_min = min(y_true.min(), y_pred.min())
        v_max = max(y_true.max(), y_pred.max())
        ax.plot(
            [v_min, v_max],
            [v_min, v_max],
            color=REFERENCE_LINE,
            linestyle="--",
            linewidth=1.5,
            label="Perfect (y = x)",
        )
        ax.legend(fontsize=10, facecolor="#161b22", edgecolor="#30363d", labelcolor=TEXT_COLOR)

        _apply_style(ax, "Predicted vs Actual", "Actual Values", "Predicted Values")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Residual Distribution
# ---------------------------------------------------------------------------


def residual_distribution(
    result: Result,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Static histogram of residuals.

    Args:
        result: A regression Result.
        save_path: If provided, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    y_true = np.asarray(result.y_test)
    y_pred = np.asarray(result.predictions)
    residuals = y_true - y_pred

    with plt.rc_context(MPL_RC_PARAMS):
        fig, ax = plt.subplots(figsize=MPL_FIG_SIZE, dpi=MPL_DPI)
        ax.hist(
            residuals,
            bins=30,
            color=PRIMARY,
            alpha=0.75,
            edgecolor="#21262d",
            linewidth=0.5,
        )
        ax.axvline(x=0, color=REFERENCE_LINE, linestyle="--", linewidth=1.5)
        _apply_style(ax, "Residual Distribution", "Residual", "Frequency")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all regression plots at once
# ---------------------------------------------------------------------------


def all_regression_plots(
    result: Result,
    output_dir: str | Path | None = None,
    fmt: str = "png",
) -> dict[str, plt.Figure]:
    """Generate and optionally save all four regression plots.

    Args:
        result: A regression Result.
        output_dir: Directory to save plots to. If None, plots are not saved.
        fmt: File format ('png' or 'pdf').

    Returns:
        Dict mapping plot name → Figure.
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "residuals_vs_fitted": residuals_vs_fitted,
        "qq_plot": qq_plot,
        "predicted_vs_actual": predicted_vs_actual,
        "residual_distribution": residual_distribution,
    }

    figures: dict[str, plt.Figure] = {}
    for name, fn in builders.items():
        save_path = output_dir / f"{name}.{fmt}" if output_dir else None
        figures[name] = fn(result, save_path=save_path)

    return figures
