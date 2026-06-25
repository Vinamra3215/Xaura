"""XAURA Classifiers — classification model wrappers.

Importing this module triggers @register_model decorators,
adding all classifiers to the global model registry.
"""

from xaura.models.classifiers.logistic import LogisticRegressionModel
from xaura.models.classifiers.random_forest import RandomForestClassifierModel

__all__ = ["LogisticRegressionModel", "RandomForestClassifierModel"]
