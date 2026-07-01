"""Tests for the export module (CSV log export, plot export, ZIP bundles)."""

import csv
import json
import zipfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Ensure models are registered
import xaura.models.classifiers  # noqa: F401
import xaura.models.regressors  # noqa: F401
from xaura import profile, run_model
from xaura.export.csv_export import export_log_csv
from xaura.export.exporter import export_run
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


@pytest.fixture
def clf_bundle(tmp_path):
    """Run a classifier and export it, returning (zip_path, result, profile)."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "f1": np.random.randn(200),
            "f2": np.random.randn(200),
            "f3": np.random.uniform(0, 10, 200),
            "f4": np.random.randn(200),
            "target": np.random.choice([0, 1], 200, p=[0.7, 0.3]),
        }
    )
    p = profile(df)
    r = run_model("rf_classifier", df, p, target_col="target", auto_log=False)
    zip_path = export_run(r, p, output_dir=tmp_path)
    return zip_path, r, p


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


# ---------------------------------------------------------------------------
# ZIP structure tests
# ---------------------------------------------------------------------------


class TestZIPStructure:
    """Tests for the ZIP bundle file structure."""

    def test_zip_created(self, clf_bundle):
        zip_path, _, _ = clf_bundle
        assert zip_path.exists()

    def test_zip_is_valid(self, clf_bundle):
        zip_path, _, _ = clf_bundle
        assert zipfile.is_zipfile(zip_path)

    def test_zip_filename_contains_model_name(self, clf_bundle):
        zip_path, _, _ = clf_bundle
        assert "rf_classifier" in zip_path.name

    def test_zip_contains_all_files(self, clf_bundle):
        zip_path, _, _ = clf_bundle
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "model.joblib" in names
            assert "config.json" in names
            assert "metrics.json" in names
            assert "profile_summary.json" in names
            assert "predictions.csv" in names
            assert "README.txt" in names

    def test_zip_contains_plots(self, clf_bundle):
        zip_path, _, _ = clf_bundle
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "plots/confusion_matrix.png" in names
            assert "plots/roc_curve.png" in names
            assert "plots/precision_recall.png" in names
            assert "plots/feature_importance.png" in names


# ---------------------------------------------------------------------------
# Content validation tests
# ---------------------------------------------------------------------------


class TestZIPContents:
    """Tests for the contents of each file in the ZIP."""

    def test_config_json_valid(self, clf_bundle):
        zip_path, result, _ = clf_bundle
        with zipfile.ZipFile(zip_path) as zf:
            config = json.loads(zf.read("config.json"))
        assert isinstance(config, dict)
        assert "n_estimators" in config

    def test_metrics_json_valid(self, clf_bundle):
        zip_path, result, _ = clf_bundle
        with zipfile.ZipFile(zip_path) as zf:
            metrics = json.loads(zf.read("metrics.json"))
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "f1" in metrics

    def test_profile_json_valid(self, clf_bundle):
        zip_path, _, p = clf_bundle
        with zipfile.ZipFile(zip_path) as zf:
            profile_data = json.loads(zf.read("profile_summary.json"))
        assert profile_data["n_rows"] == 200
        assert profile_data["n_cols"] == 5
        assert isinstance(profile_data["warnings"], list)

    def test_predictions_csv_valid(self, clf_bundle):
        zip_path, result, _ = clf_bundle
        with zipfile.ZipFile(zip_path) as zf, zf.open("predictions.csv") as f:
            df = pd.read_csv(f)
        assert "y_true" in df.columns
        assert "y_pred" in df.columns
        assert "y_proba_class_0" in df.columns
        assert "y_proba_class_1" in df.columns

    def test_model_reloadable(self, clf_bundle, tmp_path):
        """The saved model.joblib should be reloadable and functional."""
        zip_path, result, _ = clf_bundle
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("model.joblib", tmp_path)
        model = joblib.load(tmp_path / "model.joblib")
        # Model should be able to predict
        preds = model.predict(result.X_test)
        assert len(preds) == len(result.y_test)

    def test_readme_contains_model_info(self, clf_bundle):
        zip_path, _, _ = clf_bundle
        with zipfile.ZipFile(zip_path) as zf:
            readme = zf.read("README.txt").decode("utf-8")
        assert "rf_classifier" in readme
        assert "classification" in readme
        assert "XAURA" in readme

    def test_plots_are_valid_pngs(self, clf_bundle):
        """PNG files should start with the PNG magic bytes."""
        zip_path, _, _ = clf_bundle
        png_magic = b"\x89PNG"
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith(".png"):
                    data = zf.read(name)
                    assert data[:4] == png_magic, f"{name} is not a valid PNG"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestExportEdgeCases:
    """Tests for edge cases in the export."""

    def test_output_dir_created(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "f1": np.random.randn(50),
                "f2": np.random.randn(50),
                "target": np.random.choice([0, 1], 50),
            }
        )
        p = profile(df)
        r = run_model("rf_classifier", df, p, target_col="target", auto_log=False)
        new_dir = tmp_path / "nested" / "exports"
        zip_path = export_run(r, p, output_dir=new_dir)
        assert zip_path.exists()
        assert new_dir.exists()

    def test_export_without_model_object(self, tmp_path):
        """Should handle result without model_object gracefully."""
        from xaura.models.base import Result

        r = Result(
            model_name="test_model",
            task_type="classification",
            predictions=np.array([0, 1, 0]),
            metrics={"accuracy": 0.9},
            config={"param": "value"},
        )
        p = profile(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        zip_path = export_run(r, p, output_dir=tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            # model.joblib should NOT be in ZIP since model_object is None
            assert "model.joblib" not in names
            # But config and metrics should still be there
            assert "config.json" in names
            assert "metrics.json" in names
