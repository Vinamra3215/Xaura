"""Tests for the XAURA FastAPI server — experiment and export routes.

Tests all Person B's API endpoints using FastAPI's TestClient:
    - GET    /api/experiments
    - GET    /api/experiments/compare
    - GET    /api/experiments/{run_id}
    - DELETE /api/experiments/{run_id}
    - GET    /api/export/log/csv
    - GET    /api/export/{run_id}/zip
    - GET    /api/export/{run_id}/plots
    - GET    /                    (landing page)
    - GET    /experiments         (experiments page)
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from xaura.store import sqlite_store

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Redirect the SQLite store to a temp database for every test.

    This ensures each test gets its own clean database so tests don't
    interfere with each other or with the real ~/.xaura/store.db.
    """
    db_path = tmp_path / "test_store.db"
    monkeypatch.setattr(sqlite_store, "DEFAULT_DB_PATH", db_path)
    sqlite_store.init_db(db_path)
    return db_path


@pytest.fixture
def client():
    """Create a FastAPI TestClient for the XAURA app."""
    from starlette.testclient import TestClient

    from xaura.server.app import create_app

    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_run():
    """A sample run_data dict for creating test runs."""
    return {
        "model_name": "rf_classifier",
        "dataset_name": "iris.csv",
        "task_type": "classification",
        "config": {"n_estimators": 100, "max_depth": 5},
        "metrics": {"accuracy": 0.92, "f1": 0.89, "precision": 0.91},
        "duration_seconds": 2.34,
        "tags": ["baseline", "v1"],
    }


@pytest.fixture
def sample_run_id(sample_run):
    """Create a run in the DB and return its ID."""
    return sqlite_store.create_run(sample_run)


@pytest.fixture
def two_run_ids():
    """Create two different runs and return both IDs."""
    run_a = sqlite_store.create_run(
        {
            "model_name": "rf_classifier",
            "task_type": "classification",
            "metrics": {"accuracy": 0.92, "f1": 0.89},
            "duration_seconds": 2.34,
        }
    )
    run_b = sqlite_store.create_run(
        {
            "model_name": "xgb_classifier",
            "task_type": "classification",
            "metrics": {"accuracy": 0.95, "f1": 0.93},
            "duration_seconds": 4.56,
        }
    )
    return run_a, run_b


# ===========================================================================
# Landing Page & HTML Pages
# ===========================================================================


class TestPages:
    """Test that HTML pages render successfully."""

    def test_index_page(self, client):
        """GET / returns 200 with HTML content."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "XAURA" in resp.text

    def test_experiments_page(self, client):
        """GET /experiments returns 200 with HTML content."""
        resp = client.get("/experiments")
        assert resp.status_code == 200
        assert "Experiment Log" in resp.text


# ===========================================================================
# Experiment Routes
# ===========================================================================


class TestListExperiments:
    """Tests for GET /api/experiments."""

    def test_empty_list(self, client):
        """Returns empty list when no runs exist."""
        resp = client.get("/api/experiments")
        assert resp.status_code == 200
        data = resp.json()
        assert data["runs"] == []
        assert data["count"] == 0

    def test_list_returns_runs(self, client, sample_run_id):
        """Returns runs after one is created."""
        resp = client.get("/api/experiments")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["runs"][0]["id"] == sample_run_id

    def test_filter_by_model_name(self, client, two_run_ids):
        """Filtering by model_name returns only matching runs."""
        resp = client.get("/api/experiments?model_name=xgb_classifier")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["runs"][0]["model_name"] == "xgb_classifier"

    def test_filter_by_task_type(self, client, two_run_ids):
        """Filtering by task_type returns matching runs."""
        resp = client.get("/api/experiments?task_type=classification")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2  # both are classification

    def test_filter_no_match(self, client, sample_run_id):
        """Filtering with no matching runs returns empty list."""
        resp = client.get("/api/experiments?model_name=nonexistent_model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0


class TestGetExperiment:
    """Tests for GET /api/experiments/{run_id}."""

    def test_get_existing_run(self, client, sample_run_id):
        """Returns the full run dict for a valid ID."""
        resp = client.get(f"/api/experiments/{sample_run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == sample_run_id
        assert data["model_name"] == "rf_classifier"
        assert data["metrics"]["accuracy"] == 0.92

    def test_get_nonexistent_run(self, client):
        """Returns 404 for a non-existent run ID."""
        resp = client.get("/api/experiments/nonexistent-id-12345")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestDeleteExperiment:
    """Tests for DELETE /api/experiments/{run_id}."""

    def test_delete_existing_run(self, client, sample_run_id):
        """Deletes a run and confirms it's gone."""
        resp = client.delete(f"/api/experiments/{sample_run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == sample_run_id

        # Confirm it's actually deleted
        resp2 = client.get(f"/api/experiments/{sample_run_id}")
        assert resp2.status_code == 404

    def test_delete_nonexistent_run(self, client):
        """Returns 404 when deleting a run that doesn't exist."""
        resp = client.delete("/api/experiments/nonexistent-id-12345")
        assert resp.status_code == 404


class TestCompareExperiments:
    """Tests for GET /api/experiments/compare."""

    def test_compare_two_runs(self, client, two_run_ids):
        """Returns comparison data for two valid IDs."""
        id_a, id_b = two_run_ids
        resp = client.get(f"/api/experiments/compare?ids={id_a},{id_b}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        model_names = {r["model_name"] for r in data["runs"]}
        assert model_names == {"rf_classifier", "xgb_classifier"}

    def test_compare_empty_ids(self, client):
        """Returns 400 when no IDs are provided."""
        resp = client.get("/api/experiments/compare?ids=")
        assert resp.status_code == 400

    def test_compare_nonexistent_ids(self, client):
        """Returns 404 when none of the IDs exist."""
        resp = client.get("/api/experiments/compare?ids=fake-1,fake-2")
        assert resp.status_code == 404

    def test_compare_missing_ids_param(self, client):
        """Returns 422 when ids param is missing entirely."""
        resp = client.get("/api/experiments/compare")
        assert resp.status_code == 422  # FastAPI validation error


# ===========================================================================
# Export Routes
# ===========================================================================


class TestExportCSV:
    """Tests for GET /api/export/log/csv."""

    def test_csv_export_with_data(self, client, two_run_ids):
        """Downloads a valid CSV when runs exist."""
        resp = client.get("/api/export/log/csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers.get("content-type", "")

        # Parse CSV content
        content = resp.content.decode("utf-8")
        lines = content.strip().split("\n")
        assert len(lines) >= 3  # header + 2 data rows
        assert "model_name" in lines[0]  # header row

    def test_csv_export_empty_db(self, client):
        """Returns 404 when no runs exist."""
        resp = client.get("/api/export/log/csv")
        assert resp.status_code == 404


class TestExportZip:
    """Tests for GET /api/export/{run_id}/zip."""

    def test_zip_download(self, client, sample_run_id):
        """Downloads a valid ZIP with config, metrics, and run_info."""
        resp = client.get(f"/api/export/{sample_run_id}/zip")
        assert resp.status_code == 200
        assert "application/zip" in resp.headers.get("content-type", "")

        # Validate ZIP contents
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = zf.namelist()
        assert "config.json" in names
        assert "metrics.json" in names
        assert "run_info.json" in names

        # Validate JSON inside ZIP
        metrics = json.loads(zf.read("metrics.json"))
        assert metrics["accuracy"] == 0.92

    def test_zip_nonexistent_run(self, client):
        """Returns 404 for a non-existent run ID."""
        resp = client.get("/api/export/nonexistent-id/zip")
        assert resp.status_code == 404


class TestExportPlots:
    """Tests for GET /api/export/{run_id}/plots."""

    def test_plots_no_session(self, client, sample_run_id):
        """Returns 400 when no active session Result exists."""
        resp = client.get(f"/api/export/{sample_run_id}/plots")
        assert resp.status_code == 400
        assert "No active Result" in resp.json()["detail"]

    def test_plots_nonexistent_run(self, client):
        """Returns 404 for a non-existent run ID."""
        resp = client.get("/api/export/nonexistent-id/plots")
        assert resp.status_code == 404


# ===========================================================================
# Integration — Full Experiment Lifecycle
# ===========================================================================


class TestExperimentLifecycle:
    """End-to-end: create → list → get → compare → export → delete."""

    def test_full_lifecycle(self, client):
        """Walk through the complete experiment lifecycle via API."""
        # 1. Start with empty experiment list
        resp = client.get("/api/experiments")
        assert resp.json()["count"] == 0

        # 2. Create two runs directly in the store
        id1 = sqlite_store.create_run(
            {
                "model_name": "logistic_regression",
                "task_type": "classification",
                "metrics": {"accuracy": 0.85, "f1": 0.80},
                "duration_seconds": 0.5,
            }
        )
        id2 = sqlite_store.create_run(
            {
                "model_name": "rf_classifier",
                "task_type": "classification",
                "metrics": {"accuracy": 0.92, "f1": 0.89},
                "duration_seconds": 2.3,
            }
        )

        # 3. List experiments — should have 2
        resp = client.get("/api/experiments")
        assert resp.json()["count"] == 2

        # 4. Get individual run
        resp = client.get(f"/api/experiments/{id1}")
        assert resp.status_code == 200
        assert resp.json()["model_name"] == "logistic_regression"

        # 5. Compare runs
        resp = client.get(f"/api/experiments/compare?ids={id1},{id2}")
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

        # 6. Export CSV
        resp = client.get("/api/export/log/csv")
        assert resp.status_code == 200
        assert "logistic_regression" in resp.content.decode()
        assert "rf_classifier" in resp.content.decode()

        # 7. Export ZIP
        resp = client.get(f"/api/export/{id2}/zip")
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        assert "metrics.json" in zf.namelist()

        # 8. Delete a run
        resp = client.delete(f"/api/experiments/{id1}")
        assert resp.status_code == 200

        # 9. Verify only one run remains
        resp = client.get("/api/experiments")
        assert resp.json()["count"] == 1
        assert resp.json()["runs"][0]["id"] == id2
