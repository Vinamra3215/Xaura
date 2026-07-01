"""Tests for the export module (CSV log export + plot export).

Tests that CSV files are written with correct headers and data,
and that the plot export API dispatches correctly.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import xaura.models.regressors  # noqa: F401
from xaura import profile, run_model
from xaura.export.csv_export import export_log_csv
from xaura.export.plot_export import export_plots, export_single_plot
from xaura.store.sqlite_store import create_run, init_db

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_db(tmp_path):
    """Create a temp SQLite DB with a few test runs."""
    db_path = tmp_path / "test.db"
    init_db(db_path)

    # Insert a classification run
    create_run(
        {
            "model_name": "logistic_regression",
            "task_type": "classification",
            "config": {"C": 1.0, "max_iter": 100},
            "metrics": {"accuracy": 0.85, "f1": 0.82},
            "duration_seconds": 1.5,
            "dataset_name": "test_cls.csv",
        },
        db_path,
    )

    # Insert a regression run
    create_run(
        {
            "model_name": "ridge",
            "task_type": "regression",
            "config": {"alpha": 1.0},
            "metrics": {"r2": 0.91, "rmse": 0.32},
            "duration_seconds": 0.8,
            "dataset_name": "test_reg.csv",
        },
        db_path,
    )

    # Insert a clustering run
    create_run(
        {
            "model_name": "kmeans",
            "task_type": "clustering",
            "config": {"n_clusters": 3},
            "metrics": {"silhouette": 0.65},
            "duration_seconds": 0.3,
        },
        db_path,
    )

    return db_path


@pytest.fixture(scope="module")
def regression_result():
    """A regression Result for plot export tests."""
    np.random.seed(42)
    x1 = np.random.randn(100)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": np.random.randn(100),
            "target": 2 * x1 + np.random.randn(100) * 0.5,
        }
    )
    p = profile(df)
    return run_model("ridge", df, p, target_col="target", auto_log=False)


@pytest.fixture(scope="module")
def clustering_result():
    """A clustering Result for plot export tests."""
    np.random.seed(42)
    c1 = np.random.randn(60, 2)
    c2 = np.random.randn(60, 2) + 6
    data = np.vstack([c1, c2])
    df = pd.DataFrame(data, columns=["f1", "f2"])
    p = profile(df)
    return run_model("kmeans", df, p, auto_log=False)


# ---------------------------------------------------------------------------
# CSV Export tests
# ---------------------------------------------------------------------------


class TestCSVExport:
    """Tests for export_log_csv."""

    def test_creates_file(self, test_db, tmp_path):
        output = tmp_path / "log.csv"
        result = export_log_csv(output, db_path=test_db)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_correct_row_count(self, test_db, tmp_path):
        output = tmp_path / "log.csv"
        export_log_csv(output, db_path=test_db)
        with open(output, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Header + 3 data rows
        assert len(rows) == 4

    def test_has_base_columns(self, test_db, tmp_path):
        output = tmp_path / "log.csv"
        export_log_csv(output, db_path=test_db)
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        assert "id" in headers
        assert "model_name" in headers
        assert "task_type" in headers
        assert "created_at" in headers

    def test_has_flattened_metric_columns(self, test_db, tmp_path):
        output = tmp_path / "log.csv"
        export_log_csv(output, db_path=test_db)
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        assert "metric_accuracy" in headers
        assert "metric_r2" in headers
        assert "metric_silhouette" in headers

    def test_has_json_columns(self, test_db, tmp_path):
        output = tmp_path / "log.csv"
        export_log_csv(output, db_path=test_db)
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        assert "config_json" in headers
        assert "metrics_json" in headers

    def test_json_columns_parseable(self, test_db, tmp_path):
        output = tmp_path / "log.csv"
        export_log_csv(output, db_path=test_db)
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                config = json.loads(row["config_json"])
                assert isinstance(config, dict)

    def test_filter_by_task_type(self, test_db, tmp_path):
        output = tmp_path / "log.csv"
        export_log_csv(
            output,
            filters={"task_type": "classification"},
            db_path=test_db,
        )
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["model_name"] == "logistic_regression"

    def test_empty_db_raises(self, tmp_path):
        empty_db = tmp_path / "empty.db"
        init_db(empty_db)
        output = tmp_path / "log.csv"
        with pytest.raises(ValueError, match="No experiment runs found"):
            export_log_csv(output, db_path=empty_db)

    def test_creates_parent_dirs(self, test_db, tmp_path):
        output = tmp_path / "deep" / "nested" / "log.csv"
        export_log_csv(output, db_path=test_db)
        assert output.exists()


# ---------------------------------------------------------------------------
# Plot Export tests
# ---------------------------------------------------------------------------


class TestPlotExport:
    """Tests for export_plots and export_single_plot."""

    def test_regression_exports_all(self, regression_result, tmp_path):
        paths = export_plots(regression_result, tmp_path, fmt="png")
        assert len(paths) == 4
        for name, path in paths.items():
            assert Path(path).exists(), f"{name} not saved"
        plt.close("all")

    def test_regression_single_plot(self, regression_result, tmp_path):
        path = export_single_plot(regression_result, "qq_plot", tmp_path, fmt="png")
        assert Path(path).exists()
        plt.close("all")

    def test_clustering_exports_all(self, clustering_result, tmp_path):
        paths = export_plots(clustering_result, tmp_path, fmt="png")
        assert len(paths) == 4
        for name, path in paths.items():
            assert Path(path).exists(), f"{name} not saved"
        plt.close("all")

    def test_pdf_format(self, regression_result, tmp_path):
        paths = export_plots(
            regression_result,
            tmp_path,
            fmt="pdf",
            charts=["residuals_vs_fitted"],
        )
        assert Path(paths["residuals_vs_fitted"]).suffix == ".pdf"
        plt.close("all")

    def test_unknown_chart_raises(self, regression_result, tmp_path):
        with pytest.raises(ValueError, match="Unknown chart"):
            export_plots(
                regression_result,
                tmp_path,
                charts=["nonexistent_chart"],
            )

    def test_creates_output_dir(self, regression_result, tmp_path):
        out = tmp_path / "new_dir" / "plots"
        paths = export_plots(regression_result, out, charts=["predicted_vs_actual"])
        assert out.exists()
        assert len(paths) == 1
        plt.close("all")
