"""Plot Export — save model result visualisations as PNG or PDF.

Provides a unified API to export Plotly charts as static images (via
kaleido or orca) and Matplotlib charts directly to files.

Usage:
    from xaura.export.plot_export import export_plots

    # Export all plots for a regression result
    paths = export_plots(result, output_dir="./plots", fmt="png")

    # Export a single plot by name
    path = export_single_plot(result, "residuals_vs_fitted", "./plots", fmt="pdf")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from xaura.models.base import Result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plot registry — maps task_type → available Matplotlib chart builders
# ---------------------------------------------------------------------------

_REGRESSION_CHARTS = {
    "residuals_vs_fitted",
    "qq_plot",
    "predicted_vs_actual",
    "residual_distribution",
}

_CLUSTERING_CHARTS = {
    "cluster_scatter_pca",
    "silhouette_plot",
    "elbow_curve",
    "dendrogram",
}

_CLASSIFICATION_CHARTS = {
    "confusion_matrix",
    "roc_curve",
    "precision_recall",
    "feature_importance",
}


def _get_available_charts(task_type: str) -> set[str]:
    """Return the set of available chart names for a task type."""
    mapping = {
        "regression": _REGRESSION_CHARTS,
        "clustering": _CLUSTERING_CHARTS,
        "classification": _CLASSIFICATION_CHARTS,
    }
    return mapping.get(task_type, set())


# ---------------------------------------------------------------------------
# Export all plots
# ---------------------------------------------------------------------------


def export_plots(
    result: Result,
    output_dir: str | Path,
    fmt: str = "png",
    charts: list[str] | None = None,
) -> dict[str, Path]:
    """Export all (or selected) plots for a model result to disk.

    Uses the Matplotlib chart modules to generate static images.
    Automatically detects the task type from the Result and calls
    the appropriate chart generators.

    Args:
        result: A Result from run_model().
        output_dir: Directory to save the plots to.
        fmt: File format — 'png' or 'pdf'.
        charts: Optional list of specific chart names to export.
            If None, exports all available charts for the task type.

    Returns:
        Dict mapping chart name → absolute Path of the saved file.

    Raises:
        ValueError: If the task type is not supported.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    available = _get_available_charts(result.task_type)
    if not available:
        raise ValueError(
            f"No charts available for task type: {result.task_type!r}. "
            f"Supported types: classification, regression, clustering."
        )

    # If specific charts requested, validate them
    if charts:
        unknown = set(charts) - available
        if unknown:
            raise ValueError(
                f"Unknown chart(s) for {result.task_type}: {unknown}. "
                f"Available: {sorted(available)}"
            )
        to_export = set(charts)
    else:
        to_export = available

    saved: dict[str, Path] = {}

    if result.task_type == "regression":
        saved.update(_export_regression(result, output_dir, fmt, to_export))
    elif result.task_type == "clustering":
        saved.update(_export_clustering(result, output_dir, fmt, to_export))
    elif result.task_type == "classification":
        saved.update(_export_classification(result, output_dir, fmt, to_export))

    return saved


# ---------------------------------------------------------------------------
# Export a single plot
# ---------------------------------------------------------------------------


def export_single_plot(
    result: Result,
    chart_name: str,
    output_dir: str | Path,
    fmt: str = "png",
) -> Path:
    """Export a single plot by name.

    Args:
        result: A Result from run_model().
        chart_name: Name of the chart to export.
        output_dir: Directory to save the plot to.
        fmt: File format — 'png' or 'pdf'.

    Returns:
        Absolute Path of the saved file.

    Raises:
        ValueError: If the chart name is not valid for the result's task type.
    """
    paths = export_plots(result, output_dir, fmt, charts=[chart_name])
    return paths[chart_name]


# ---------------------------------------------------------------------------
# Internal: per-task-type exporters
# ---------------------------------------------------------------------------


def _export_regression(
    result: Result,
    output_dir: Path,
    fmt: str,
    charts: set[str],
) -> dict[str, Path]:
    """Export regression charts via Matplotlib."""
    import matplotlib.pyplot as plt

    from xaura.visualisation import matplotlib_regression as mpl_reg

    saved: dict[str, Path] = {}
    builders: dict[str, Any] = {
        "residuals_vs_fitted": mpl_reg.residuals_vs_fitted,
        "qq_plot": mpl_reg.qq_plot,
        "predicted_vs_actual": mpl_reg.predicted_vs_actual,
        "residual_distribution": mpl_reg.residual_distribution,
    }

    for name in charts:
        if name in builders:
            save_path = output_dir / f"{name}.{fmt}"
            builders[name](result, save_path=save_path)
            saved[name] = save_path.resolve()
            plt.close("all")
            logger.info("Exported: %s", save_path)

    return saved


def _export_clustering(
    result: Result,
    output_dir: Path,
    fmt: str,
    charts: set[str],
) -> dict[str, Path]:
    """Export clustering charts via Matplotlib."""
    import matplotlib.pyplot as plt
    import numpy as np

    from xaura.visualisation import matplotlib_clustering as mpl_clu

    saved: dict[str, Path] = {}
    X = np.asarray(result.X_train)

    # Charts that take a Result
    result_builders: dict[str, Any] = {
        "cluster_scatter_pca": mpl_clu.cluster_scatter_pca,
        "silhouette_plot": mpl_clu.silhouette_plot,
    }

    for name in charts:
        if name in result_builders:
            save_path = output_dir / f"{name}.{fmt}"
            result_builders[name](result, save_path=save_path)
            saved[name] = save_path.resolve()
            plt.close("all")
            logger.info("Exported: %s", save_path)

    # Charts that take raw X
    if "elbow_curve" in charts:
        save_path = output_dir / f"elbow_curve.{fmt}"
        mpl_clu.elbow_curve(X, save_path=save_path)
        saved["elbow_curve"] = save_path.resolve()
        plt.close("all")
        logger.info("Exported: %s", save_path)

    if "dendrogram" in charts:
        save_path = output_dir / f"dendrogram.{fmt}"
        mpl_clu.dendrogram_plot(X, save_path=save_path)
        saved["dendrogram"] = save_path.resolve()
        plt.close("all")
        logger.info("Exported: %s", save_path)

    return saved


def _export_classification(
    result: Result,
    output_dir: Path,
    fmt: str,
    charts: set[str],
) -> dict[str, Path]:
    """Export classification charts via Matplotlib.

    Uses Person A's matplotlib_charts module for the static versions
    of confusion matrix, ROC curve, precision-recall, and feature importance.
    """
    import matplotlib.pyplot as plt

    saved: dict[str, Path] = {}

    try:
        from xaura.visualisation import matplotlib_charts as mpl_cls
    except ImportError:
        logger.warning(
            "matplotlib_charts module not available — "
            "classification plot export requires Person A's code."
        )
        return saved

    # Map our chart names → Person A's function names
    builders: dict[str, Any] = {
        "confusion_matrix": getattr(mpl_cls, "confusion_matrix_static", None),
        "roc_curve": getattr(mpl_cls, "roc_curve_static", None),
        "precision_recall": getattr(mpl_cls, "precision_recall_static", None),
        "feature_importance": getattr(mpl_cls, "feature_importance_static", None),
    }

    for name in charts:
        fn = builders.get(name)
        if fn is not None:
            save_path = output_dir / f"{name}.{fmt}"
            try:
                fn(result, save_path=save_path)
                saved[name] = save_path.resolve()
                plt.close("all")
                logger.info("Exported: %s", save_path)
            except Exception as exc:
                logger.warning("Failed to export %s: %s", name, exc)

    return saved
