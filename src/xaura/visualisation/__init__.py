"""XAURA Visualisation — interactive and static charts for model results."""

from xaura.visualisation.plotly_charts import (
    confusion_matrix_chart,
    config_panel,
    feature_importance_chart,
    metrics_card,
    precision_recall_chart,
    profile_summary_panel,
    roc_curve_chart,
)

__all__ = [
    # Classification charts
    "confusion_matrix_chart",
    "roc_curve_chart",
    "precision_recall_chart",
    "feature_importance_chart",
    # Common panels
    "profile_summary_panel",
    "metrics_card",
    "config_panel",
]
