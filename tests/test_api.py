"""Tests for the XAURA FastAPI web server endpoints.

Tests cover the full upload -> profile -> run -> results -> export flow.
Uses httpx AsyncClient with the FastAPI TestClient pattern.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Register all models
import xaura.models.classifiers  # noqa: F401
import xaura.models.clusterers  # noqa: F401
import xaura.models.regressors  # noqa: F401
from xaura.server.app import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Create a fresh test client for each test."""
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def sample_csv() -> bytes:
    """Generate a simple CSV for testing."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "f1": rng.standard_normal(100),
            "f2": rng.standard_normal(100),
            "f3": rng.uniform(0, 10, 100),
            "target": rng.choice([0, 1], 100),
        }
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


@pytest.fixture()
def uploaded_session(client, sample_csv):
    """Upload a CSV and return the session_id."""
    response = client.post(
        "/api/profile",
        files={"file": ("test.csv", sample_csv, "text/csv")},
    )
    assert response.status_code == 200
    return response.json()["session_id"]


@pytest.fixture()
def run_session(client, uploaded_session):
    """Upload + run a model, return session_id."""
    response = client.post(
        "/api/run",
        json={
            "session_id": uploaded_session,
            "model_name": "rf_classifier",
            "target_col": "target",
        },
    )
    assert response.status_code == 200
    return uploaded_session


# ---------------------------------------------------------------------------
# Landing Page
# ---------------------------------------------------------------------------


class TestLandingPage:
    """Tests for the landing page."""

    def test_landing_returns_200(self, client):
        """GET / should return 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_landing_contains_title(self, client):
        """Landing page should contain the XAURA title."""
        response = client.get("/")
        assert "XAURA" in response.text

    def test_landing_contains_upload(self, client):
        """Landing page should have the upload dropzone."""
        response = client.get("/")
        assert "dropzone" in response.text


# ---------------------------------------------------------------------------
# Profile Upload
# ---------------------------------------------------------------------------


class TestProfileUpload:
    """Tests for the CSV upload and profiling endpoint."""

    def test_upload_valid_csv(self, client, sample_csv):
        """POST /api/profile with valid CSV should return 200 + profile."""
        response = client.post(
            "/api/profile",
            files={"file": ("data.csv", sample_csv, "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["profile"]["n_rows"] == 100
        assert data["profile"]["n_cols"] == 4

    def test_upload_returns_session_id(self, client, sample_csv):
        """Session ID should be a non-empty string."""
        response = client.post(
            "/api/profile",
            files={"file": ("data.csv", sample_csv, "text/csv")},
        )
        sid = response.json()["session_id"]
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_upload_invalid_extension(self, client):
        """POST /api/profile with non-CSV should return 400."""
        response = client.post(
            "/api/profile",
            files={"file": ("data.txt", b"not a csv", "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_empty_csv(self, client):
        """POST /api/profile with empty CSV should return 400."""
        response = client.post(
            "/api/profile",
            files={"file": ("empty.csv", b"a,b,c\n", "text/csv")},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Profile Page
# ---------------------------------------------------------------------------


class TestProfilePage:
    """Tests for the profile HTML page."""

    def test_profile_page_returns_200(self, client, uploaded_session):
        """GET /profile/{id} should return 200."""
        response = client.get(f"/profile/{uploaded_session}")
        assert response.status_code == 200

    def test_profile_page_shows_columns(self, client, uploaded_session):
        """Profile page should display column details."""
        response = client.get(f"/profile/{uploaded_session}")
        assert "Column Details" in response.text
        assert "f1" in response.text
        assert "target" in response.text

    def test_profile_invalid_session(self, client):
        """GET /profile/{bad_id} should return 404."""
        response = client.get("/profile/nonexistent")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Model Run
# ---------------------------------------------------------------------------


class TestModelRun:
    """Tests for the model run endpoint."""

    def test_run_model_returns_200(self, client, uploaded_session):
        """POST /api/run with valid params should return 200."""
        response = client.post(
            "/api/run",
            json={
                "session_id": uploaded_session,
                "model_name": "rf_classifier",
                "target_col": "target",
            },
        )
        assert response.status_code == 200

    def test_run_returns_metrics(self, client, uploaded_session):
        """Response should contain metrics."""
        response = client.post(
            "/api/run",
            json={
                "session_id": uploaded_session,
                "model_name": "rf_classifier",
                "target_col": "target",
            },
        )
        data = response.json()
        assert "metrics" in data
        assert "accuracy" in data["metrics"]
        assert "f1" in data["metrics"]

    def test_run_returns_charts(self, client, uploaded_session):
        """Response should indicate charts were generated."""
        response = client.post(
            "/api/run",
            json={
                "session_id": uploaded_session,
                "model_name": "rf_classifier",
                "target_col": "target",
            },
        )
        data = response.json()
        assert "charts" in data
        assert data["charts"]["confusion_matrix"] is True

    def test_run_invalid_model(self, client, uploaded_session):
        """POST /api/run with bad model name should return 400."""
        response = client.post(
            "/api/run",
            json={
                "session_id": uploaded_session,
                "model_name": "nonexistent_model",
                "target_col": "target",
            },
        )
        assert response.status_code == 400

    def test_run_invalid_session(self, client):
        """POST /api/run with bad session should return 404."""
        response = client.post(
            "/api/run",
            json={
                "session_id": "bad_session",
                "model_name": "rf_classifier",
                "target_col": "target",
            },
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Results Page
# ---------------------------------------------------------------------------


class TestResultsPage:
    """Tests for the results HTML page."""

    def test_results_page_returns_200(self, client, run_session):
        """GET /results/{id} should return 200 after a model run."""
        response = client.get(f"/results/{run_session}")
        assert response.status_code == 200

    def test_results_page_has_charts(self, client, run_session):
        """Results page should contain chart containers."""
        response = client.get(f"/results/{run_session}")
        assert "chart-confusion-matrix" in response.text

    def test_results_page_has_metrics(self, client, run_session):
        """Results page should show performance metrics."""
        response = client.get(f"/results/{run_session}")
        assert "Performance Metrics" in response.text

    def test_results_page_has_export(self, client, run_session):
        """Results page should have an export button."""
        response = client.get(f"/results/{run_session}")
        assert "Export ZIP" in response.text

    def test_results_no_run(self, client, uploaded_session):
        """GET /results/{id} without running a model should return 400."""
        response = client.get(f"/results/{uploaded_session}")
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    """Tests for the ZIP export endpoint."""

    def test_export_returns_zip(self, client, run_session):
        """GET /api/export/{id} should return a valid ZIP file."""
        response = client.get(f"/api/export/{run_session}")
        assert response.status_code == 200
        assert "application/zip" in response.headers.get("content-type", "")

        # Verify it's a valid ZIP
        buf = io.BytesIO(response.content)
        assert zipfile.is_zipfile(buf)

    def test_export_zip_has_files(self, client, run_session):
        """Exported ZIP should contain expected files."""
        response = client.get(f"/api/export/{run_session}")
        buf = io.BytesIO(response.content)
        with zipfile.ZipFile(buf) as zf:
            names = zf.namelist()
            assert any("config.json" in n for n in names)
            assert any("metrics.json" in n for n in names)

    def test_export_invalid_session(self, client):
        """GET /api/export/{bad_id} should return 404."""
        response = client.get("/api/export/nonexistent")
        assert response.status_code == 404

    def test_export_no_run(self, client, uploaded_session):
        """GET /api/export/{id} without running model should return 400."""
        response = client.get(f"/api/export/{uploaded_session}")
        assert response.status_code == 400
