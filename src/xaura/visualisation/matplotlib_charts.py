"""Matplotlib static charts for XAURA model results.

Static PNG/PDF versions of the classification charts. Used for:
    - Export bundles (ZIP with PNG charts)
    - Reports and presentations
    - Environments without a browser (CI, headless servers)

Each function returns a matplotlib Figure and optionally saves to a file.

Classification charts:
    - confusion_matrix_static(result, save_path=None)
    - roc_curve_static(result, save_path=None)
    - precision_recall_static(result, save_path=None)
    - feature_importance_static(result, save_path=None)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from xaura.models.base import Result

# Use non-interactive backend for headless environments
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------

# Dark theme consistent with Plotly charts
_STYLE_PARAMS: dict[str, Any] = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#21262d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#c9d1d9",
    "ytick.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
}

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


def _apply_style() -> None:
    """Apply the dark theme to matplotlib."""
    plt.rcParams.update(_STYLE_PARAMS)


def _save_and_close(
    fig: plt.Figure,
    save_path: str | Path | None,
) -> plt.Figure:
    """Save figure to disk if path is given, then close to free memory.

    Args:
        fig: The matplotlib Figure.
        save_path: File path to save (PNG, PDF, SVG based on extension).

    Returns:
        The Figure (before closing, so caller can still use it).
    """
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
    return fig


# ---------------------------------------------------------------------------
# Classification charts
# ---------------------------------------------------------------------------


def confusion_matrix_static(
    result: Result,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a static confusion matrix heatmap.

    Args:
        result: A classification Result with y_test and predictions.
        save_path: Optional path to save the figure (PNG/PDF/SVG).

    Returns:
        A matplotlib Figure with the confusion matrix.
    """
    _apply_style()

    y_true = result.y_test
    y_pred = result.predictions

    labels = sorted(np.unique(np.concatenate([np.asarray(y_true), y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Draw heatmap
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, label="Count")

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > cm.max() / 2 else "#c9d1d9"
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_yticklabels([str(label) for label in labels])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    return _save_and_close(fig, save_path)


def roc_curve_static(
    result: Result,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a static ROC curve with AUC in the legend.

    Args:
        result: A classification Result with y_test and probabilities.
        save_path: Optional path to save the figure.

    Returns:
        A matplotlib Figure with ROC curve(s).

    Raises:
        ValueError: If probabilities are not available.
    """
    if result.probabilities is None:
        raise ValueError("ROC curve requires probabilities (y_proba)")

    _apply_style()

    y_true = np.asarray(result.y_test)
    y_proba = result.probabilities
    classes = sorted(np.unique(y_true))

    fig, ax = plt.subplots(figsize=(6.5, 5))

    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=_ACCENT_COLORS[0],
            linewidth=2.5,
            label=f"ROC (AUC = {roc_auc:.3f})",
        )
    else:
        from sklearn.preprocessing import label_binarize

        y_bin = label_binarize(y_true, classes=classes)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            color = _ACCENT_COLORS[i % len(_ACCENT_COLORS)]
            ax.plot(
                fpr,
                tpr,
                color=color,
                linewidth=2,
                label=f"Class {cls} (AUC = {roc_auc:.3f})",
            )

    # Diagonal reference
    ax.plot(
        [0, 1],
        [0, 1],
        color="#484f58",
        linewidth=1.5,
        linestyle="--",
        label="Random (AUC = 0.500)",
    )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_and_close(fig, save_path)


def precision_recall_static(
    result: Result,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a static Precision-Recall curve.

    Args:
        result: A classification Result with y_test and probabilities.
        save_path: Optional path to save the figure.

    Returns:
        A matplotlib Figure with PR curve(s).

    Raises:
        ValueError: If probabilities are not available.
    """
    if result.probabilities is None:
        raise ValueError("PR curve requires probabilities (y_proba)")

    _apply_style()

    y_true = np.asarray(result.y_test)
    y_proba = result.probabilities
    classes = sorted(np.unique(y_true))

    fig, ax = plt.subplots(figsize=(6.5, 5))

    if len(classes) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        pr_auc = auc(recall, precision)
        ax.plot(
            recall,
            precision,
            color=_ACCENT_COLORS[4],
            linewidth=2.5,
            label=f"PR (AUC = {pr_auc:.3f})",
        )
        ax.fill_between(recall, precision, alpha=0.1, color=_ACCENT_COLORS[4])
    else:
        from sklearn.preprocessing import label_binarize

        y_bin = label_binarize(y_true, classes=classes)
        for i, cls in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            pr_auc = auc(recall, precision)
            color = _ACCENT_COLORS[i % len(_ACCENT_COLORS)]
            ax.plot(
                recall,
                precision,
                color=color,
                linewidth=2,
                label=f"Class {cls} (AUC = {pr_auc:.3f})",
            )

    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_and_close(fig, save_path)


def feature_importance_static(
    result: Result,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a static horizontal bar chart of feature importances.

    Args:
        result: A Result with feature_importances and X_train.
        save_path: Optional path to save the figure.

    Returns:
        A matplotlib Figure with a horizontal bar chart.

    Raises:
        ValueError: If feature importances are not available.
    """
    if result.feature_importances is None:
        raise ValueError("Feature importance chart requires feature_importances")

    _apply_style()

    importances = result.feature_importances

    if result.X_train is not None:
        feature_names = list(result.X_train.columns)
    elif result.X_test is not None:
        feature_names = list(result.X_test.columns)
    else:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    # Sort by importance
    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_values = importances[sorted_idx]

    n = len(sorted_values)
    max_val = sorted_values[-1] if sorted_values[-1] > 0 else 1.0

    # Color gradient from dim to bright lavender
    colors = [
        (*matplotlib.colors.to_rgb(_ACCENT_COLORS[0]), 0.3 + 0.7 * (v / max_val))
        for v in sorted_values
    ]

    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.4 * n + 1.5)))

    ax.barh(range(n), sorted_values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    return _save_and_close(fig, save_path)
