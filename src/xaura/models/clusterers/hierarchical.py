"""Hierarchical (Agglomerative) clustering model wrapper.

Agglomerative clustering builds a hierarchy of clusters by
progressively merging the closest pairs. Supports various linkage
criteria (ward, complete, average, single).

Ward linkage minimises the variance within clusters and is the
most commonly used default. It requires Euclidean distance.

Dataset-aware defaults:
    - n_clusters=3 (sensible starting point, matches K-Means)
    - linkage='ward' (minimum variance, best general-purpose)
    - metric='euclidean' (required by ward linkage)

The dendrogram helper is a static method that returns the linkage
matrix needed to plot a scipy dendrogram — useful for visual
inspection of the cluster hierarchy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xaura.models.base import BaseModel
from xaura.models.defaults import get_defaults
from xaura.models.registry import register_model
from xaura.profiler.dataprofile import DataProfile


@register_model
class HierarchicalModel(BaseModel):
    """Agglomerative Hierarchical Clustering with dendrogram support."""

    name = "hierarchical"
    display_name = "Hierarchical Clustering"
    task_type = "clustering"

    def get_default_config(self, profile: DataProfile) -> dict[str, Any]:
        """Return dataset-aware defaults for Agglomerative Clustering.

        Args:
            profile: Dataset profile.

        Returns:
            Config dict for sklearn AgglomerativeClustering.
        """
        return get_defaults(profile, self.name)

    def build(self, config: dict[str, Any], profile: DataProfile) -> Any:
        """Build a sklearn AgglomerativeClustering with the given config."""
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(**config)

    @staticmethod
    def compute_linkage_matrix(
        X: np.ndarray,  # noqa: N803
        method: str = "ward",
    ) -> np.ndarray:
        """Compute the linkage matrix for dendrogram plotting.

        Uses scipy's linkage function to build the hierarchical tree.
        The returned matrix can be passed directly to
        scipy.cluster.hierarchy.dendrogram().

        Args:
            X: Feature matrix (numpy array or DataFrame).
            method: Linkage method ('ward', 'complete', 'average', 'single').

        Returns:
            Linkage matrix (n_samples-1 x 4 array).
        """
        from scipy.cluster.hierarchy import linkage

        return linkage(X, method=method)
