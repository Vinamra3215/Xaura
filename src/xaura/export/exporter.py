"""XAURA Export — package model runs into shareable ZIP bundles.

The exporter takes a Result (from run_model) and a DataProfile (from profile)
and packages everything into a self-contained ZIP file:

    model.joblib          — serialised model weights
    config.json           — hyperparameters used
    metrics.json          — computed metrics
    profile_summary.json  — dataset statistics snapshot
    predictions.csv       — y_true, y_pred, y_proba columns
    plots/                — static chart PNGs (confusion matrix, ROC, etc.)
    README.txt            — human-readable run summary

Usage:
    from xaura.export import export_run
    zip_path = export_run(result, profile, output_dir="./exports")
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from xaura.models.base import Result
from xaura.profiler.dataprofile import DataProfile


def export_run(
    result: Result,
    profile: DataProfile,
    output_dir: str | Path | None = None,
) -> Path:
    """Package a model run into a ZIP bundle.

    Creates a ZIP file containing the model weights, config, metrics,
    predictions, profile summary, charts, and a README.

    Args:
        result: The Result from run_model().
        profile: The DataProfile from profile().
        output_dir: Directory to save the ZIP. Defaults to current dir.

    Returns:
        Path to the created ZIP file.
    """
    if output_dir is None:
        output_dir = Path.cwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"export_{result.model_name}_{timestamp}.zip"
    zip_path = output_dir / zip_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        _save_model(result, tmp)
        _save_config(result, tmp)
        _save_metrics(result, tmp)
        _save_profile(profile, tmp)
        _save_predictions(result, tmp)
        _save_plots(result, tmp)
        _write_readme(result, profile, tmp)

        # Create ZIP from all files in tmpdir
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(tmpdir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(tmpdir)
                    zf.write(file_path, arcname)

    return zip_path


# ---------------------------------------------------------------------------
# Internal helpers — each saves one component to the temp directory
# ---------------------------------------------------------------------------


def _save_model(result: Result, tmpdir: Path) -> None:
    """Serialise model object with joblib."""
    if result.model_object is not None:
        joblib.dump(result.model_object, tmpdir / "model.joblib")


def _save_config(result: Result, tmpdir: Path) -> None:
    """Save config dict as JSON."""
    config = _make_serialisable(result.config)
    with open(tmpdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def _save_metrics(result: Result, tmpdir: Path) -> None:
    """Save metrics dict as JSON."""
    metrics = _make_serialisable(result.metrics)
    with open(tmpdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def _save_profile(profile: DataProfile, tmpdir: Path) -> None:
    """Save profile summary as JSON."""
    summary = {
        "shape": list(profile.shape),
        "n_rows": profile.n_rows,
        "n_cols": profile.n_cols,
        "feature_types": profile.feature_types,
        "target_column": profile.target_column,
        "task_type": profile.task_type,
        "is_imbalanced": profile.is_imbalanced,
        "has_missing": profile.has_missing,
        "missing_fraction": profile.missing_fraction,
        "dataset_hash": profile.dataset_hash,
        "warnings": profile.warnings,
    }
    with open(tmpdir / "profile_summary.json", "w") as f:
        json.dump(_make_serialisable(summary), f, indent=2)


def _save_predictions(result: Result, tmpdir: Path) -> None:
    """Save predictions as CSV with y_true, y_pred, and optional y_proba."""
    data: dict[str, Any] = {}

    if result.y_test is not None:
        data["y_true"] = np.asarray(result.y_test)

    if result.predictions is not None and len(result.predictions) > 0:
        data["y_pred"] = result.predictions

    if result.probabilities is not None:
        proba = result.probabilities
        if proba.ndim == 2:
            for i in range(proba.shape[1]):
                data[f"y_proba_class_{i}"] = proba[:, i]
        else:
            data["y_proba"] = proba

    if data:
        df = pd.DataFrame(data)
        df.to_csv(tmpdir / "predictions.csv", index=False)


def _save_plots(result: Result, tmpdir: Path) -> None:
    """Generate static charts and save as PNGs in a plots/ subdirectory."""
    plots_dir = tmpdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Only generate plots that make sense for this result type
    if result.task_type == "classification":
        _save_classification_plots(result, plots_dir)
    elif result.task_type == "regression":
        _save_regression_plots(result, plots_dir)
    # Clustering has no standard plots yet (Person B's task)


def _save_classification_plots(result: Result, plots_dir: Path) -> None:
    """Generate classification-specific static charts."""
    import matplotlib.pyplot as plt

    from xaura.visualisation.matplotlib_charts import (
        confusion_matrix_static,
        feature_importance_static,
        precision_recall_static,
        roc_curve_static,
    )

    # Confusion matrix — always available
    try:
        confusion_matrix_static(result, save_path=plots_dir / "confusion_matrix.png")
        plt.close("all")
    except Exception:
        pass

    # ROC curve — requires probabilities
    if result.probabilities is not None:
        try:
            roc_curve_static(result, save_path=plots_dir / "roc_curve.png")
            plt.close("all")
        except Exception:
            pass

    # Precision-Recall — requires probabilities
    if result.probabilities is not None:
        try:
            precision_recall_static(result, save_path=plots_dir / "precision_recall.png")
            plt.close("all")
        except Exception:
            pass

    # Feature importance — requires importances
    if result.feature_importances is not None:
        try:
            feature_importance_static(result, save_path=plots_dir / "feature_importance.png")
            plt.close("all")
        except Exception:
            pass


def _save_regression_plots(result: Result, plots_dir: Path) -> None:
    """Placeholder for regression plots (Person B's task)."""
    # Will be filled in when Person B adds regression matplotlib charts
    pass


def _write_readme(result: Result, profile: DataProfile, tmpdir: Path) -> None:
    """Generate a human-readable summary of the run."""
    lines = [
        "=" * 60,
        "XAURA Model Run Export",
        "=" * 60,
        "",
        f"Model:          {result.model_name}",
        f"Task Type:      {result.task_type}",
        f"Train Time:     {result.train_time_seconds:.2f}s",
        f"Dataset Hash:   {result.dataset_hash}",
        "",
        "--- Dataset ---",
        f"Shape:          {profile.n_rows:,} rows x {profile.n_cols} columns",
        f"Target:         {profile.target_column} ({profile.task_type})",
        f"Missing:        {'Yes' if profile.has_missing else 'No'}",
        "",
        "--- Metrics ---",
    ]

    for name, value in result.metrics.items():
        lines.append(f"  {name:20s} {value:.4f}")

    lines.extend(
        [
            "",
            "--- Config ---",
        ]
    )

    for key, value in result.config.items():
        lines.append(f"  {key:20s} {value}")

    lines.extend(
        [
            "",
            "--- Files ---",
            "  model.joblib          Serialised model (reload with joblib.load)",
            "  config.json           Hyperparameters used",
            "  metrics.json          Computed metrics",
            "  profile_summary.json  Dataset profile snapshot",
            "  predictions.csv       Predictions (y_true, y_pred, y_proba)",
            "  plots/                Static chart PNGs",
            "  README.txt            This file",
            "",
            f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Generated by XAURA - eXtendable Automated Unified Research & Analytics",
            "",
        ]
    )

    with open(tmpdir / "README.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _make_serialisable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
