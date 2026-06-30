"""DBSCAN clustering model wrapper.

Density-Based Spatial Clustering of Applications with Noise. Unlike
K-Means, DBSCAN does not require the number of clusters to be specified
upfront — it discovers clusters based on density. Points in low-density
regions are labelled as noise (label = -1).

Dataset-aware defaults:
    - eps: neighbourhood radius (0.5 default, a widely used starting point)
    - min_samples: scales with feature count (max(5, n_features * 2))
    - metric: 'euclidean' (standard)
"""

from __future__ import annotations

from typing import Any

from xaura.models.base import BaseModel
from xaura.models.defaults import get_defaults
from xaura.models.registry import register_model
from xaura.profiler.dataprofile import DataProfile


@register_model
class DBSCANModel(BaseModel):
    """DBSCAN Clustering with dataset-aware eps and min_samples."""

    name = "dbscan"
    display_name = "DBSCAN"
    task_type = "clustering"

    def get_default_config(self, profile: DataProfile) -> dict[str, Any]:
        """Return dataset-aware defaults for DBSCAN.

        Args:
            profile: Dataset profile.

        Returns:
            Config dict for sklearn DBSCAN.
        """
        return get_defaults(profile, self.name)

    def build(self, config: dict[str, Any], profile: DataProfile) -> Any:
        """Build a sklearn DBSCAN with the given config."""
        from sklearn.cluster import DBSCAN

        return DBSCAN(**config)
