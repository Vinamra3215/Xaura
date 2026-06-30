"""Matplotlib static charts for clustering models.

Static PNG/PDF versions of the four clustering visualisation plots,
using the shared dark theme to match the Plotly interactive charts.

Usage:
    from xaura.visualisation.matplotlib_clustering import all_clustering_plots
    figs = all_clustering_plots(result, output_dir="./plots")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from xaura.models.base import Result
from xaura.visualisation._style import (
    ACCENT_COLORS,
    MPL_DPI,
    MPL_FIG_SIZE,
    MPL_RC_PARAMS,
    PRIMARY,
    REFERENCE_LINE,
    TEXT_COLOR,
)

# Noise colour
_NOISE_COLOR = "#484f58"


def _apply_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent dark-theme styling to an axes."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12, color=TEXT_COLOR)
    ax.set_xlabel(xlabel, fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, fontsize=11, color=TEXT_COLOR)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.4, linestyle="--")


# ---------------------------------------------------------------------------
# 1. Cluster Scatter (PCA 2D)
# ---------------------------------------------------------------------------


def cluster_scatter_pca(
    result: Result,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Static 2D scatter plot of clusters via PCA.

    Args:
        result: A clustering Result.
        save_path: If provided, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    X = np.asarray(result.X_train)
    labels = np.asarray(result.predictions)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    with plt.rc_context(MPL_RC_PARAMS):
        fig, ax = plt.subplots(figsize=MPL_FIG_SIZE, dpi=MPL_DPI)

        unique_labels = sorted(set(labels))
        for label in unique_labels:
            mask = labels == label
            is_noise = label == -1
            color = _NOISE_COLOR if is_noise else ACCENT_COLORS[label % len(ACCENT_COLORS)]
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                s=8 if is_noise else 20,
                alpha=0.4 if is_noise else 0.7,
                label="Noise" if is_noise else f"Cluster {label}",
                edgecolors="none",
            )

        explained = pca.explained_variance_ratio_
        _apply_style(
            ax,
            "Cluster Scatter (PCA 2D)",
            f"PC1 ({explained[0]:.1%} variance)",
            f"PC2 ({explained[1]:.1%} variance)",
        )
        ax.legend(
            fontsize=9,
            markerscale=2,
            facecolor="#161b22",
            edgecolor="#30363d",
            labelcolor=TEXT_COLOR,
        )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Silhouette Plot
# ---------------------------------------------------------------------------


def silhouette_plot(
    result: Result,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Static silhouette plot showing per-sample silhouette values.

    Args:
        result: A clustering Result.
        save_path: If provided, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    X = np.asarray(result.X_train)
    labels = np.asarray(result.predictions)

    non_noise_mask = labels != -1
    X_clean = X[non_noise_mask]
    labels_clean = labels[non_noise_mask]

    n_clusters = len(set(labels_clean))

    with plt.rc_context(MPL_RC_PARAMS):
        fig, ax = plt.subplots(figsize=MPL_FIG_SIZE, dpi=MPL_DPI)

        if n_clusters < 2:
            ax.text(
                0.5,
                0.5,
                "Need ≥ 2 clusters for silhouette",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
                color=TEXT_COLOR,
            )
            _apply_style(ax, "Silhouette Plot", "", "")
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, bbox_inches="tight")
            return fig

        sample_silhouette = silhouette_samples(X_clean, labels_clean)
        avg_score = float(np.mean(sample_silhouette))

        y_lower = 0
        for cluster_id in sorted(set(labels_clean)):
            cluster_values = np.sort(sample_silhouette[labels_clean == cluster_id])
            n_points = len(cluster_values)
            y_upper = y_lower + n_points

            color = ACCENT_COLORS[cluster_id % len(ACCENT_COLORS)]
            ax.barh(
                range(y_lower, y_upper),
                cluster_values,
                height=1.0,
                color=color,
                edgecolor="none",
                label=f"Cluster {cluster_id}",
            )
            y_lower = y_upper + 2

        ax.axvline(
            x=avg_score,
            color=REFERENCE_LINE,
            linestyle="--",
            linewidth=1.5,
            label=f"Avg: {avg_score:.3f}",
        )
        _apply_style(ax, "Silhouette Plot", "Silhouette Coefficient", "Sample Index")
        ax.set_yticks([])
        ax.legend(
            fontsize=9,
            facecolor="#161b22",
            edgecolor="#30363d",
            labelcolor=TEXT_COLOR,
        )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Elbow Curve
# ---------------------------------------------------------------------------


def elbow_curve(
    X: np.ndarray | Any,
    k_range: range | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Static elbow method plot: inertia vs k.

    Args:
        X: Feature matrix.
        k_range: Range of k values. Defaults to range(2, 11).
        save_path: If provided, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    from sklearn.cluster import KMeans

    if k_range is None:
        k_range = range(2, 11)

    ks = list(k_range)
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    with plt.rc_context(MPL_RC_PARAMS):
        fig, ax = plt.subplots(figsize=MPL_FIG_SIZE, dpi=MPL_DPI)
        ax.plot(ks, inertias, "o-", color=PRIMARY, linewidth=2.5, markersize=8)
        _apply_style(
            ax,
            "Elbow Method — Optimal k",
            "Number of Clusters (k)",
            "Inertia (Within-cluster SSE)",
        )
        ax.set_xticks(ks)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Dendrogram
# ---------------------------------------------------------------------------


def dendrogram_plot(
    X: np.ndarray | Any,
    method: str = "ward",
    truncate_mode: str = "lastp",
    p: int = 30,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Static dendrogram for hierarchical clustering.

    Args:
        X: Feature matrix.
        method: Linkage method.
        truncate_mode: Truncation mode.
        p: Number of leaves to show.
        save_path: If provided, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    X = np.asarray(X)
    Z = linkage(X, method=method)

    with plt.rc_context(MPL_RC_PARAMS):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=MPL_DPI)
        scipy_dendrogram(
            Z,
            truncate_mode=truncate_mode,
            p=p,
            ax=ax,
            leaf_rotation=90,
            leaf_font_size=9,
        )
        _apply_style(ax, f"Dendrogram ({method} linkage)", "Sample Index", "Distance")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all clustering plots
# ---------------------------------------------------------------------------


def all_clustering_plots(
    result: Result,
    output_dir: str | Path | None = None,
    fmt: str = "png",
    show_elbow: bool = True,
    show_dendrogram: bool = True,
) -> dict[str, plt.Figure]:
    """Generate and optionally save all clustering plots.

    Args:
        result: A clustering Result.
        output_dir: Directory to save plots to. If None, not saved.
        fmt: File format ('png' or 'pdf').
        show_elbow: Whether to include elbow curve.
        show_dendrogram: Whether to include dendrogram.

    Returns:
        Dict mapping plot name → Figure.
    """
    X = np.asarray(result.X_train)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, plt.Figure] = {}

    save = output_dir / f"cluster_scatter_pca.{fmt}" if output_dir else None
    figures["cluster_scatter_pca"] = cluster_scatter_pca(result, save_path=save)

    save = output_dir / f"silhouette_plot.{fmt}" if output_dir else None
    figures["silhouette_plot"] = silhouette_plot(result, save_path=save)

    if show_elbow:
        save = output_dir / f"elbow_curve.{fmt}" if output_dir else None
        figures["elbow_curve"] = elbow_curve(X, save_path=save)

    if show_dendrogram:
        save = output_dir / f"dendrogram.{fmt}" if output_dir else None
        figures["dendrogram"] = dendrogram_plot(X, save_path=save)

    return figures
