"""K-Means clustering model wrapper.

An unsupervised model that groups data into k clusters by minimising
within-cluster variance (inertia). Unlike classifiers and regressors,
K-Means has no target column and no train/test split — it uses all
data for fitting.

Includes an elbow method helper that runs K-Means for a range of k
values and suggests the best k based on the largest drop in inertia.

Dataset-aware defaults:
    - n_clusters=3 (a sensible starting point)
    - n_init=10 (run 10 random initialisations, keep the best)
    - max_iter=300, random_state=42
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xaura.models.base import BaseModel
from xaura.models.registry import register_model
from xaura.profiler.dataprofile import DataProfile


@register_model
class KMeansModel(BaseModel):
    """K-Means Clustering with elbow method for auto-k selection."""

    name = "kmeans"
    display_name = "K-Means Clustering"
    task_type = "clustering"

    def get_default_config(self, profile: DataProfile) -> dict[str, Any]:
        """Return default config for K-Means.

        Args:
            profile: Dataset profile (currently minimal adjustments
                     needed for K-Means defaults).

        Returns:
            Config dict for sklearn KMeans.
        """
        return {
            "n_clusters": 3,
            "n_init": 10,
            "max_iter": 300,
            "random_state": 42,
        }

    def build(self, config: dict[str, Any], profile: DataProfile) -> Any:
        """Build a sklearn KMeans with the given config."""
        from sklearn.cluster import KMeans

        return KMeans(**config)

    @staticmethod
    def elbow_method(
        X: np.ndarray,
        k_range: range = range(2, 11),
    ) -> dict[str, Any]:
        """Run the elbow method to find the optimal number of clusters.

        Fits K-Means for each k in k_range, records inertia (sum of
        squared distances to nearest cluster centre), and suggests the
        best k by finding the "elbow" — the point where the rate of
        decrease in inertia slows down most sharply.

        Algorithm for finding the elbow:
            1. Compute inertia for each k
            2. Compute the first differences (drops between consecutive k)
            3. Compute the second differences (change in the rate of drop)
            4. The elbow is at the k where the second difference is largest
               (i.e., the steepest change in slope)

        Args:
            X: Feature matrix (numpy array or DataFrame).
            k_range: Range of k values to try (default 2–10).

        Returns:
            Dict with:
                - k_range: list of k values tested
                - inertias: list of inertia values for each k
                - suggested_k: the recommended number of clusters
        """
        from sklearn.cluster import KMeans

        k_list = list(k_range)
        inertias: list[float] = []

        for k in k_list:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(X)
            inertias.append(float(km.inertia_))

        # Find the elbow using second differences
        if len(inertias) >= 3:
            # First differences (how much inertia drops at each step)
            diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
            # Second differences (how the drop rate changes)
            second_diffs = [diffs[i] - diffs[i + 1] for i in range(len(diffs) - 1)]
            # The elbow is where the second difference is largest
            elbow_idx = int(np.argmax(second_diffs))
            suggested_k = k_list[elbow_idx + 1]  # +1 because second_diffs is offset
        else:
            # Not enough points — just pick the middle
            suggested_k = k_list[len(k_list) // 2]

        return {
            "k_range": k_list,
            "inertias": inertias,
            "suggested_k": suggested_k,
        }
