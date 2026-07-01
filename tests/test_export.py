"""Tests for XAURA ZIP bundle export."""

import json
import zipfile

import joblib
import numpy as np
import pandas as pd
import pytest

# Ensure models are registered
import xaura.models.classifiers  # noqa: F401
from xaura import profile, run_model
from xaura.export import export_run

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
