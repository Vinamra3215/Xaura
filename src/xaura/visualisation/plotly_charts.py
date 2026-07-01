"""Plotly interactive charts for XAURA model results.

Functions return plotly.graph_objects.Figure objects that can be:
    - Displayed in Jupyter: fig.show()
    - Converted to JSON for the web UI: fig.to_json()
    - Exported as PNG: fig.write_image("plot.png")

Classification charts:
    - confusion_matrix_chart(result)
    - roc_curve_chart(result)
    - precision_recall_chart(result)
    - feature_importance_chart(result)

Common panels (shared across all model types):
    - profile_summary_panel(profile)
    - metrics_card(result)
    - config_panel(result)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from xaura.models.base import Result
from xaura.profiler.dataprofile import DataProfile

# ---------------------------------------------------------------------------
# Shared styling constants
# ---------------------------------------------------------------------------

_BG_COLOR = "#0d1117"
_PAPER_COLOR = "#0d1117"
_GRID_COLOR = "#21262d"
_TEXT_COLOR = "#c9d1d9"
_FONT_FAMILY = "Inter, system-ui, -apple-system, sans-serif"
_ACCENT_COLORS = [
    "#b4a7d6",  # lavender
    "#e67e22",  # orange
    "#8b0000",  # blood red
    "#2454b8",  # royal blue
    "#2ecc71",  # emerald green
    "#00e5ff",  # cyan
    "#ff69b4",  # pink
    "#8b6914",  # hazelnut brown
]


def _base_layout(**overrides: Any) -> dict[str, Any]:
    """Return a base layout dict with consistent dark theme styling."""
    layout = {
        "template": "plotly_dark",
        "paper_bgcolor": _PAPER_COLOR,
        "plot_bgcolor": _BG_COLOR,
        "font": {"family": _FONT_FAMILY, "color": _TEXT_COLOR, "size": 13},
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
        "colorway": _ACCENT_COLORS,
    }
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# Classification charts
# ---------------------------------------------------------------------------


def confusion_matrix_chart(result: Result) -> go.Figure:
    """Create an annotated confusion matrix heatmap.

    Shows predicted vs actual labels with counts in each cell.
    Uses a blue sequential colorscale on a dark background.

    Args:
        result: A classification Result with y_test and predictions.

    Returns:
        A plotly Figure with the confusion matrix heatmap.
    """
    y_true = result.y_test
    y_pred = result.predictions

    labels = sorted(np.unique(np.concatenate([np.asarray(y_true), y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Create text annotations with counts
    text = [[str(val) for val in row] for row in cm]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[str(label) for label in labels],
            y=[str(label) for label in labels],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},
            colorscale=[
                [0.0, "#0d1117"],
                [0.25, "#1a3a5c"],
                [0.5, "#2a5f8f"],
                [0.75, "#3a85c2"],
                [1.0, "#58a6ff"],
            ],
            hovertemplate=("Actual: %{y}<br>" "Predicted: %{x}<br>" "Count: %{z}<extra></extra>"),
            showscale=True,
            colorbar={
                "title": {"text": "Count", "font": {"color": _TEXT_COLOR}},
            },
        )
    )

    fig.update_layout(
        **_base_layout(
            title={
                "text": "Confusion Matrix",
                "x": 0.5,
                "font": {"size": 18},
            },
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            xaxis={"type": "category", "side": "bottom"},
            yaxis={"type": "category", "autorange": "reversed"},
            width=520,
            height=480,
        )
    )

    return fig


def roc_curve_chart(result: Result) -> go.Figure:
    """Create ROC curve(s) with AUC in the legend.

    For binary classification, plots a single ROC curve.
    For multiclass, plots one ROC curve per class (one-vs-rest).

    Args:
        result: A classification Result with y_test and probabilities.

    Returns:
        A plotly Figure with ROC curve(s) and a diagonal reference line.

    Raises:
        ValueError: If probabilities are not available.
    """
    if result.probabilities is None:
        raise ValueError("ROC curve requires probabilities (y_proba)")

    y_true = np.asarray(result.y_test)
    y_proba = result.probabilities
    classes = sorted(np.unique(y_true))

    fig = go.Figure()

    if len(classes) == 2:
        # Binary — use probability of the positive class
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC (AUC = {roc_auc:.3f})",
                line={"color": _ACCENT_COLORS[0], "width": 2.5},
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            )
        )
    else:
        # Multiclass — one curve per class (one-vs-rest)
        from sklearn.preprocessing import label_binarize

        y_bin = label_binarize(y_true, classes=classes)

        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            color = _ACCENT_COLORS[i % len(_ACCENT_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"Class {cls} (AUC = {roc_auc:.3f})",
                    line={"color": color, "width": 2},
                    hovertemplate=f"Class {cls}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
                )
            )

    # Diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random (AUC = 0.500)",
            line={"color": "#484f58", "width": 1.5, "dash": "dash"},
            showlegend=True,
        )
    )

    fig.update_layout(
        **_base_layout(
            title={"text": "ROC Curve", "x": 0.5, "font": {"size": 18}},
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis={"range": [0, 1], "gridcolor": _GRID_COLOR},
            yaxis={"range": [0, 1.05], "gridcolor": _GRID_COLOR},
            legend={"x": 0.55, "y": 0.05, "bgcolor": "rgba(13,17,23,0.8)"},
            width=600,
            height=480,
        )
    )

    return fig


def precision_recall_chart(result: Result) -> go.Figure:
    """Create Precision-Recall curve(s).

    More informative than ROC for imbalanced datasets. For binary
    classification, plots a single curve. For multiclass, plots
    one curve per class (one-vs-rest).

    Args:
        result: A classification Result with y_test and probabilities.

    Returns:
        A plotly Figure with PR curve(s).

    Raises:
        ValueError: If probabilities are not available.
    """
    if result.probabilities is None:
        raise ValueError("PR curve requires probabilities (y_proba)")

    y_true = np.asarray(result.y_test)
    y_proba = result.probabilities
    classes = sorted(np.unique(y_true))

    fig = go.Figure()

    if len(classes) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        pr_auc = auc(recall, precision)
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"PR (AUC = {pr_auc:.3f})",
                line={"color": _ACCENT_COLORS[2], "width": 2.5},
                fill="tozeroy",
                fillcolor="rgba(46, 204, 113, 0.1)",
                hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
            )
        )
    else:
        from sklearn.preprocessing import label_binarize

        y_bin = label_binarize(y_true, classes=classes)

        for i, cls in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            pr_auc = auc(recall, precision)
            color = _ACCENT_COLORS[i % len(_ACCENT_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"Class {cls} (AUC = {pr_auc:.3f})",
                    line={"color": color, "width": 2},
                    hovertemplate=f"Class {cls}<br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>",
                )
            )

    fig.update_layout(
        **_base_layout(
            title={"text": "Precision–Recall Curve", "x": 0.5, "font": {"size": 18}},
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis={"range": [0, 1.02], "gridcolor": _GRID_COLOR},
            yaxis={"range": [0, 1.05], "gridcolor": _GRID_COLOR},
            legend={"x": 0.05, "y": 0.05, "bgcolor": "rgba(13,17,23,0.8)"},
            width=600,
            height=480,
        )
    )

    return fig


def feature_importance_chart(result: Result) -> go.Figure:
    """Create a horizontal bar chart of feature importances.

    Sorted descending (most important feature at top). Works for any
    model that produces feature importances (tree-based or linear).

    Args:
        result: A Result with feature_importances and X_train (for feature names).

    Returns:
        A plotly Figure with a horizontal bar chart.

    Raises:
        ValueError: If feature importances are not available.
    """
    if result.feature_importances is None:
        raise ValueError("Feature importance chart requires feature_importances")

    importances = result.feature_importances

    # Get feature names from X_train or X_test
    if result.X_train is not None:
        feature_names = list(result.X_train.columns)
    elif result.X_test is not None:
        feature_names = list(result.X_test.columns)
    else:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    # Sort by importance descending
    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_values = importances[sorted_idx]

    # Color gradient from dim to bright
    n = len(sorted_values)
    max_val = sorted_values[-1] if sorted_values[-1] > 0 else 1.0
    colors = [f"rgba(180, 167, 214, {0.3 + 0.7 * (v / max_val)})" for v in sorted_values]

    fig = go.Figure(
        data=go.Bar(
            x=sorted_values,
            y=sorted_names,
            orientation="h",
            marker={"color": colors},
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        **_base_layout(
            title={"text": "Feature Importance", "x": 0.5, "font": {"size": 18}},
            xaxis_title="Importance",
            yaxis_title="",
            xaxis={"gridcolor": _GRID_COLOR},
            yaxis={"automargin": True},
            height=max(350, 30 * n + 120),
            width=600,
        )
    )

    return fig


# ---------------------------------------------------------------------------
# Common panels (shared across all model types)
# ---------------------------------------------------------------------------


def profile_summary_panel(profile: DataProfile) -> go.Figure:
    """Create a summary panel showing key dataset statistics.

    Displays dataset shape, feature type counts, missing value info,
    target column, and warnings as a formatted table.

    Args:
        profile: A DataProfile from xaura.profiler.

    Returns:
        A plotly Figure containing a table of profile statistics.
    """
    labels = []
    values = []

    labels.append("Rows × Columns")
    values.append(f"{profile.n_rows:,} × {profile.n_cols}")

    for ftype in ["numeric", "categorical", "binary", "datetime", "text"]:
        count = len(profile.feature_types.get(ftype, []))
        if count > 0:
            labels.append(f"{ftype.capitalize()} features")
            values.append(str(count))

    if profile.target_column:
        labels.append("Target column")
        values.append(f"{profile.target_column} ({profile.task_type})")

    if profile.has_missing:
        labels.append("Missing values")
        values.append(f"{profile.missing_fraction:.1%} of cells")
    else:
        labels.append("Missing values")
        values.append("None ✓")

    if profile.warnings:
        labels.append("Warnings")
        values.append(str(len(profile.warnings)))

    fig = go.Figure(
        data=go.Table(
            header={
                "values": ["<b>Property</b>", "<b>Value</b>"],
                "fill_color": "#161b22",
                "font": {"color": _TEXT_COLOR, "size": 13, "family": _FONT_FAMILY},
                "align": "left",
                "line": {"color": _GRID_COLOR, "width": 1},
                "height": 35,
            },
            cells={
                "values": [labels, values],
                "fill_color": [[_BG_COLOR, "#0f1318"] * ((len(labels) + 1) // 2)],
                "font": {"color": _TEXT_COLOR, "size": 13, "family": _FONT_FAMILY},
                "align": "left",
                "line": {"color": _GRID_COLOR, "width": 1},
                "height": 30,
            },
        )
    )

    fig.update_layout(
        **_base_layout(
            title={"text": "Dataset Profile", "x": 0.5, "font": {"size": 18}},
            width=500,
            height=max(250, 35 * len(labels) + 100),
            margin={"l": 10, "r": 10, "t": 60, "b": 10},
        )
    )

    return fig


def metrics_card(result: Result) -> go.Figure:
    """Create a metrics card showing all computed metrics.

    Displays each metric as a row in a styled table. Metrics are
    formatted to 4 decimal places.

    Args:
        result: A Result with a metrics dict.

    Returns:
        A plotly Figure containing a table of metrics.
    """
    names = list(result.metrics.keys())
    values = [f"{v:.4f}" for v in result.metrics.values()]

    # Add model name and task type at the top
    header_names = ["Model", "Task Type"] + [n.upper() for n in names]
    header_values = [result.model_name, result.task_type] + values

    fig = go.Figure(
        data=go.Table(
            header={
                "values": ["<b>Metric</b>", "<b>Value</b>"],
                "fill_color": "#161b22",
                "font": {"color": _TEXT_COLOR, "size": 13, "family": _FONT_FAMILY},
                "align": "left",
                "line": {"color": _GRID_COLOR, "width": 1},
                "height": 35,
            },
            cells={
                "values": [header_names, header_values],
                "fill_color": [[_BG_COLOR, "#0f1318"] * ((len(header_names) + 1) // 2)],
                "font": {"color": _TEXT_COLOR, "size": 13, "family": _FONT_FAMILY},
                "align": "left",
                "line": {"color": _GRID_COLOR, "width": 1},
                "height": 30,
            },
        )
    )

    fig.update_layout(
        **_base_layout(
            title={"text": "Metrics Summary", "x": 0.5, "font": {"size": 18}},
            width=450,
            height=max(250, 35 * len(header_names) + 100),
            margin={"l": 10, "r": 10, "t": 60, "b": 10},
        )
    )

    return fig


def config_panel(result: Result) -> go.Figure:
    """Create a panel showing the hyperparameters used for a run.

    Displays each config key-value pair in a styled table.

    Args:
        result: A Result with a config dict.

    Returns:
        A plotly Figure containing a table of configuration values.
    """
    keys = list(result.config.keys())
    values = [str(v) for v in result.config.values()]

    fig = go.Figure(
        data=go.Table(
            header={
                "values": ["<b>Parameter</b>", "<b>Value</b>"],
                "fill_color": "#161b22",
                "font": {"color": _TEXT_COLOR, "size": 13, "family": _FONT_FAMILY},
                "align": "left",
                "line": {"color": _GRID_COLOR, "width": 1},
                "height": 35,
            },
            cells={
                "values": [keys, values],
                "fill_color": [[_BG_COLOR, "#0f1318"] * ((len(keys) + 1) // 2)],
                "font": {"color": _TEXT_COLOR, "size": 13, "family": _FONT_FAMILY},
                "align": "left",
                "line": {"color": _GRID_COLOR, "width": 1},
                "height": 30,
            },
        )
    )

    fig.update_layout(
        **_base_layout(
            title={
                "text": "Configuration",
                "x": 0.5,
                "font": {"size": 18},
            },
            width=450,
            height=max(250, 35 * len(keys) + 100),
            margin={"l": 10, "r": 10, "t": 60, "b": 10},
        )
    )

    return fig
