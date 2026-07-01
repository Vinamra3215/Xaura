"""XAURA Visualisation — interactive and static charts for model results."""

from xaura.visualisation.matplotlib_charts import (
    confusion_matrix_static,
    feature_importance_static,
    precision_recall_static,
    roc_curve_static,
)
from xaura.visualisation.plotly_charts import (
    config_panel,
    confusion_matrix_chart,
    feature_importance_chart,
    metrics_card,
    precision_recall_chart,
    profile_summary_panel,
    roc_curve_chart,
)

__all__ = [
    # Plotly classification charts
    "confusion_matrix_chart",
    "roc_curve_chart",
    "precision_recall_chart",
    "feature_importance_chart",
    # Plotly common panels
    "profile_summary_panel",
    "metrics_card",
    "config_panel",
    # Matplotlib static charts
    "confusion_matrix_static",
    "roc_curve_static",
    "precision_recall_static",
    "feature_importance_static",
]
