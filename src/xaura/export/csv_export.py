"""CSV Export — export the full experiment log from SQLite to CSV.

Reads all experiment runs from the SQLite store and writes them to
a flat CSV file. JSON fields (config, metrics, profile_summary, tags)
are flattened into separate columns for easy analysis in Excel or pandas.

Usage:
    from xaura.export.csv_export import export_log_csv

    # Export all runs
    path = export_log_csv("experiments.csv")

    # Export filtered runs
    path = export_log_csv("classification_runs.csv", filters={"task_type": "classification"})
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from xaura.store.sqlite_store import list_runs


def export_log_csv(
    output_path: str | Path,
    filters: dict[str, str] | None = None,
    db_path: str | Path | None = None,
) -> Path:
    """Export the experiment log to a CSV file.

    Fetches all matching runs from SQLite and writes them to a flat CSV.
    Nested JSON fields (config, metrics) are serialised as JSON strings
    in their respective columns, but individual metric keys are also
    flattened into their own columns (e.g. "metric_accuracy", "metric_f1")
    for convenience.

    Args:
        output_path: Path where the CSV will be written.
        filters: Optional filters passed to list_runs()
            (e.g. {"task_type": "classification"}).
        db_path: Path to the SQLite database. Uses default if None.

    Returns:
        The resolved Path to the written CSV file.

    Raises:
        ValueError: If no runs are found matching the filters.
    """
    runs = list_runs(filters=filters, db_path=db_path)

    if not runs:
        raise ValueError(
            "No experiment runs found" + (f" matching filters: {filters}" if filters else "")
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all unique metric keys across all runs for flat columns
    all_metric_keys: set[str] = set()
    all_config_keys: set[str] = set()
    for run in runs:
        if isinstance(run.get("metrics"), dict):
            all_metric_keys.update(run["metrics"].keys())
        if isinstance(run.get("config"), dict):
            all_config_keys.update(run["config"].keys())

    sorted_metric_keys = sorted(all_metric_keys)
    sorted_config_keys = sorted(all_config_keys)

    # Base columns (always present)
    base_columns = [
        "id",
        "model_name",
        "task_type",
        "dataset_name",
        "created_at",
        "duration_seconds",
        "tags",
    ]

    # Flattened metric columns
    metric_columns = [f"metric_{k}" for k in sorted_metric_keys]

    # Flattened config columns
    config_columns = [f"config_{k}" for k in sorted_config_keys]

    # Full JSON columns (for completeness)
    json_columns = ["config_json", "metrics_json", "profile_summary_json"]

    all_columns = base_columns + metric_columns + config_columns + json_columns

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()

        for run in runs:
            row: dict[str, Any] = {}

            # Base fields
            for col in base_columns:
                value = run.get(col, "")
                # Serialise tags list to a string
                if col == "tags" and isinstance(value, list):
                    value = ", ".join(str(t) for t in value)
                row[col] = value

            # Flattened metrics
            metrics = run.get("metrics", {})
            if isinstance(metrics, dict):
                for key in sorted_metric_keys:
                    row[f"metric_{key}"] = metrics.get(key, "")

            # Flattened config
            config = run.get("config", {})
            if isinstance(config, dict):
                for key in sorted_config_keys:
                    row[f"config_{key}"] = config.get(key, "")

            # Full JSON columns
            row["config_json"] = json.dumps(run.get("config", {}))
            row["metrics_json"] = json.dumps(run.get("metrics", {}))
            row["profile_summary_json"] = json.dumps(run.get("profile_summary", {}))

            writer.writerow(row)

    return output_path.resolve()
