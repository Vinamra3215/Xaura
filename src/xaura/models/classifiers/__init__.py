"""XAURA Classifiers — classification model wrappers.

Importing this module triggers @register_model decorators,
adding all classifiers to the global model registry.
"""

from xaura.models.classifiers.lightgbm_cls import LightGBMClassifier
from xaura.models.classifiers.logistic import LogisticRegressionModel
from xaura.models.classifiers.random_forest import RandomForestClassifierModel
from xaura.models.classifiers.xgboost_cls import XGBoostClassifier

__all__ = [
    "LogisticRegressionModel",
    "RandomForestClassifierModel",
    "XGBoostClassifier",
    "LightGBMClassifier",
]
