"""XAURA Clusterers — clustering model wrappers.

Importing this module triggers @register_model decorators,
adding all clusterers to the global model registry.
"""

from xaura.models.clusterers.kmeans import KMeansModel

__all__ = ["KMeansModel"]
