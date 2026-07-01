"""Plotly interactive charts for clustering models.

Generates four interactive plots from a clustering Result:
    1. Cluster Scatter (PCA 2D) — project clusters into 2D for visualisation
    2. Silhouette Plot — per-sample silhouette values by cluster
    3. Elbow Curve — inertia vs k to find optimal cluster count
    4. Dendrogram — hierarchical cluster tree

Each function returns a plotly.graph_objects.Figure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from xaura.models.base import Result
from xaura.visualisation._style import (
    ACCENT_COLORS,
    GRID_COLOR,
    PRIMARY,
    REFERENCE_LINE,
    TEXT_COLOR,
    plotly_base_layout,
)

# Noise colour for DBSCAN
_NOISE_COLOR = "#484f58"


# ---------------------------------------------------------------------------
# 1. Cluster Scatter (PCA 2D)
# ---------------------------------------------------------------------------


def cluster_scatter_pca(result: Result) -> go.Figure:
    """2D scatter plot of clusters projected via PCA.

    Reduces the feature space to 2 principal components and colours
    each point by its cluster label. Noise points (DBSCAN label = -1)
    are shown in grey.

    Args:
        result: A clustering Result with X_train and predictions.

    Returns:
        Plotly Figure.
    """
    X = np.asarray(result.X_train)
    labels = np.asarray(result.predictions)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    fig = go.Figure()

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask = labels == label
        is_noise = label == -1

        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker={
                    "color": (
                        _NOISE_COLOR if is_noise else ACCENT_COLORS[label % len(ACCENT_COLORS)]
                    ),
                    "size": 4 if is_noise else 6,
                    "opacity": 0.4 if is_noise else 0.7,
                },
                name="Noise" if is_noise else f"Cluster {label}",
                hovertemplate=(
                    f"{'Noise' if is_noise else f'Cluster {label}'}<br>"
                    "PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>"
                ),
            )
        )

    explained = pca.explained_variance_ratio_
    fig.update_layout(
        **plotly_base_layout(
            title="Cluster Scatter (PCA 2D)",
            xaxis_title=f"PC1 ({explained[0]:.1%} variance)",
            yaxis_title=f"PC2 ({explained[1]:.1%} variance)",
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 2. Silhouette Plot
# ---------------------------------------------------------------------------


def silhouette_plot(result: Result) -> go.Figure:
    """Per-sample silhouette values grouped by cluster.

    Higher silhouette values indicate better-defined clusters.
    The dashed vertical line shows the average silhouette score.

    Args:
        result: A clustering Result with X_train and predictions.

    Returns:
        Plotly Figure.
    """
    X = np.asarray(result.X_train)
    labels = np.asarray(result.predictions)

    # Filter out noise for silhouette computation
    non_noise_mask = labels != -1
    if non_noise_mask.sum() < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough clusters for silhouette analysis",
            showarrow=False,
            font={"color": TEXT_COLOR},
        )
        fig.update_layout(**plotly_base_layout(title="Silhouette Plot"))
        return fig

    X_clean = X[non_noise_mask]
    labels_clean = labels[non_noise_mask]

    n_clusters = len(set(labels_clean))
    if n_clusters < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 clusters for silhouette",
            showarrow=False,
            font={"color": TEXT_COLOR},
        )
        fig.update_layout(**plotly_base_layout(title="Silhouette Plot"))
        return fig

    sample_silhouette_values = silhouette_samples(X_clean, labels_clean)
    avg_score = float(np.mean(sample_silhouette_values))

    fig = go.Figure()
    y_lower = 0

    for cluster_id in sorted(set(labels_clean)):
        cluster_values = sample_silhouette_values[labels_clean == cluster_id]
        cluster_values = np.sort(cluster_values)
        n_points = len(cluster_values)
        y_upper = y_lower + n_points
        y_range = np.arange(y_lower, y_upper)

        color = ACCENT_COLORS[cluster_id % len(ACCENT_COLORS)]
        fig.add_trace(
            go.Bar(
                x=cluster_values,
                y=y_range,
                orientation="h",
                marker_color=color,
                name=f"Cluster {cluster_id}",
                hovertemplate=(f"Cluster {cluster_id}<br>Silhouette: %{{x:.3f}}<extra></extra>"),
            )
        )

        y_lower = y_upper + 2

    # Average line
    fig.add_vline(
        x=avg_score,
        line_dash="dash",
        line_color=REFERENCE_LINE,
        line_width=1.5,
        annotation_text=f"Avg: {avg_score:.3f}",
        annotation_position="top right",
        annotation_font_color=REFERENCE_LINE,
    )

    fig.update_layout(
        **plotly_base_layout(
            title="Silhouette Plot",
            xaxis_title="Silhouette Coefficient",
            yaxis_title="Sample Index (by cluster)",
            yaxis={"showticklabels": False},
            bargap=0,
            barmode="stack",
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 3. Elbow Curve
# ---------------------------------------------------------------------------


def elbow_curve(
    X: np.ndarray | Any,
    k_range: range | None = None,
) -> go.Figure:
    """Elbow method plot: inertia vs number of clusters.

    This is a standalone function (not tied to a Result) because it
    needs to fit K-Means for multiple values of k.

    Args:
        X: Feature matrix (numpy array or DataFrame).
        k_range: Range of k values to try. Defaults to range(2, 11).

    Returns:
        Plotly Figure.
    """
    from sklearn.cluster import KMeans

    if k_range is None:
        k_range = range(2, 11)

    inertias = []
    ks = list(k_range)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ks,
            y=inertias,
            mode="lines+markers",
            line={"color": PRIMARY, "width": 2.5},
            marker={"size": 8, "color": PRIMARY},
            name="Inertia",
            hovertemplate="k = %{x}<br>Inertia: %{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        **plotly_base_layout(
            title="Elbow Method — Optimal k",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia (Within-cluster SSE)",
            xaxis={"dtick": 1},
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 4. Dendrogram
# ---------------------------------------------------------------------------


def dendrogram_plot(
    X: np.ndarray | Any,
    method: str = "ward",
    truncate_mode: str = "lastp",
    p: int = 30,
) -> go.Figure:
    """Dendrogram for hierarchical clustering.

    Uses scipy's linkage and dendrogram functions, then converts the
    output to a Plotly figure for interactivity.

    Args:
        X: Feature matrix (numpy array or DataFrame).
        method: Linkage method ('ward', 'complete', 'average', 'single').
        truncate_mode: Truncation mode for large datasets.
        p: Number of leaves to show when truncating.

    Returns:
        Plotly Figure.
    """
    X = np.asarray(X)
    Z = linkage(X, method=method)

    # Use scipy to compute the dendrogram layout (no_plot=True)
    dn = scipy_dendrogram(Z, truncate_mode=truncate_mode, p=p, no_plot=True)

    fig = go.Figure()

    icoord = np.array(dn["icoord"])
    dcoord = np.array(dn["dcoord"])
    colors = dn["color_list"]

    # Map scipy colour codes to our accent palette
    _color_map = {
        "C0": ACCENT_COLORS[0],
        "C1": ACCENT_COLORS[1],
        "C2": ACCENT_COLORS[2],
        "C3": ACCENT_COLORS[3],
        "C4": ACCENT_COLORS[4],
        "C5": ACCENT_COLORS[5],
        "C6": ACCENT_COLORS[6],
        "b": ACCENT_COLORS[0],
    }

    for xs, ys, color in zip(icoord, dcoord, colors, strict=False):
        hex_color = _color_map.get(color, GRID_COLOR)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": hex_color, "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        **plotly_base_layout(
            title=f"Dendrogram ({method} linkage)",
            xaxis_title="Sample Index",
            yaxis_title="Distance",
            xaxis={"showticklabels": False},
        )
    )

    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all clustering plots at once
# ---------------------------------------------------------------------------


def all_clustering_plots(
    result: Result,
    show_elbow: bool = True,
    show_dendrogram: bool = True,
) -> dict[str, go.Figure]:
    """Generate all clustering plots.

    The elbow and dendrogram plots require re-fitting, so they are
    optional and default to True.

    Args:
        result: A clustering Result.
        show_elbow: Whether to include the elbow curve.
        show_dendrogram: Whether to include the dendrogram.

    Returns:
        Dict mapping plot name → Figure.
    """
    X = np.asarray(result.X_train)

    plots: dict[str, go.Figure] = {
        "cluster_scatter_pca": cluster_scatter_pca(result),
        "silhouette_plot": silhouette_plot(result),
    }

    if show_elbow:
        plots["elbow_curve"] = elbow_curve(X)

    if show_dendrogram:
        plots["dendrogram"] = dendrogram_plot(X)

    return plots
