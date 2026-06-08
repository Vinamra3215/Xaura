"""Tests for the XAURA SQLite Store.

Tests all CRUD operations: init_db, create_run, get_run,
list_runs, delete_run, get_metrics_comparison.
Each test uses tmp_path for an isolated database.
"""

import pytest

from xaura.store.sqlite_store import (
    create_run,
    delete_run,
    get_metrics_comparison,
    get_run,
    init_db,
    list_runs,
)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    """Create a fresh database in a temp directory and return its path."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    return db_path


def _sample_run(**overrides):
    """Return a sample run_data dict with sensible defaults."""
    data = {
        "model_name": "random_forest",
        "dataset_name": "iris.csv",
        "task_type": "classification",
        "config": {"n_estimators": 100, "max_depth": 5},
        "metrics": {"accuracy": 0.87, "f1": 0.85},
        "duration_seconds": 2.3,
        "tags": ["baseline"],
    }
    data.update(overrides)
    return data


# ─────────────────────────────────────────────────────────────
# init_db tests
# ─────────────────────────────────────────────────────────────


class TestInitDb:
    """Tests for init_db()."""

    def test_creates_file(self, tmp_path):
        db_path = tmp_path / "new.db"
        assert not db_path.exists()
        init_db(db_path)
        assert db_path.exists()

    def test_creates_parent_dirs(self, tmp_path):
        db_path = tmp_path / "deep" / "nested" / "store.db"
        init_db(db_path)
        assert db_path.exists()

    def test_idempotent(self, tmp_path):
        """Calling init_db twice should not error or wipe data."""
        db_path = tmp_path / "test.db"
        init_db(db_path)
        create_run(_sample_run(), db_path=db_path)
        init_db(db_path)  # second call
        assert len(list_runs(db_path=db_path)) == 1


# ─────────────────────────────────────────────────────────────
# create_run tests
# ─────────────────────────────────────────────────────────────


class TestCreateRun:
    """Tests for create_run()."""

    def test_returns_uuid(self, db):
        run_id = create_run(_sample_run(), db_path=db)
        assert isinstance(run_id, str)
        assert len(run_id) == 36  # UUID format: 8-4-4-4-12

    def test_missing_model_name_raises(self, db):
        with pytest.raises(KeyError, match="model_name"):
            create_run({"dataset_name": "test.csv"}, db_path=db)

    def test_minimal_data(self, db):
        """Only model_name is required — everything else is optional."""
        run_id = create_run({"model_name": "logistic"}, db_path=db)
        run = get_run(run_id, db_path=db)
        assert run["model_name"] == "logistic"
        assert run["dataset_name"] == ""
        assert run["config"] == {}
        assert run["metrics"] == {}


# ─────────────────────────────────────────────────────────────
# get_run tests
# ─────────────────────────────────────────────────────────────


class TestGetRun:
    """Tests for get_run()."""

    def test_existing_run(self, db):
        run_id = create_run(_sample_run(), db_path=db)
        run = get_run(run_id, db_path=db)
        assert run is not None
        assert run["id"] == run_id
        assert run["model_name"] == "random_forest"

    def test_nonexistent_run(self, db):
        result = get_run("fake-id-that-doesnt-exist", db_path=db)
        assert result is None

    def test_all_fields_present(self, db):
        run_id = create_run(_sample_run(), db_path=db)
        run = get_run(run_id, db_path=db)
        expected_keys = {
            "id",
            "model_name",
            "dataset_name",
            "task_type",
            "config",
            "metrics",
            "profile_summary",
            "model_path",
            "created_at",
            "duration_seconds",
            "tags",
        }
        assert set(run.keys()) == expected_keys

    def test_json_round_trip(self, db):
        """Dicts should survive the save→load cycle intact."""
        original_config = {"n_estimators": 100, "max_depth": 5, "nested": {"a": 1}}
        original_metrics = {"accuracy": 0.87, "f1": 0.85, "per_class": [0.9, 0.8]}
        run_id = create_run(
            _sample_run(config=original_config, metrics=original_metrics),
            db_path=db,
        )
        run = get_run(run_id, db_path=db)
        assert run["config"] == original_config
        assert run["metrics"] == original_metrics

    def test_tags_round_trip(self, db):
        run_id = create_run(_sample_run(tags=["v1", "experiment-a"]), db_path=db)
        run = get_run(run_id, db_path=db)
        assert run["tags"] == ["v1", "experiment-a"]


# ─────────────────────────────────────────────────────────────
# list_runs tests
# ─────────────────────────────────────────────────────────────


class TestListRuns:
    """Tests for list_runs()."""

    def test_empty_db(self, db):
        assert list_runs(db_path=db) == []

    def test_returns_all(self, db):
        create_run(_sample_run(), db_path=db)
        create_run(_sample_run(model_name="xgboost"), db_path=db)
        create_run(_sample_run(model_name="logistic"), db_path=db)
        assert len(list_runs(db_path=db)) == 3

    def test_newest_first(self, db):
        import time

        create_run(_sample_run(model_name="first"), db_path=db)
        time.sleep(0.05)  # tiny delay to ensure different timestamps
        create_run(_sample_run(model_name="second"), db_path=db)
        runs = list_runs(db_path=db)
        assert runs[0]["model_name"] == "second"
        assert runs[1]["model_name"] == "first"

    def test_filter_by_model_name(self, db):
        create_run(_sample_run(model_name="rf"), db_path=db)
        create_run(_sample_run(model_name="rf"), db_path=db)
        create_run(_sample_run(model_name="xgb"), db_path=db)
        rf_runs = list_runs(filters={"model_name": "rf"}, db_path=db)
        assert len(rf_runs) == 2
        assert all(r["model_name"] == "rf" for r in rf_runs)

    def test_filter_by_task_type(self, db):
        create_run(_sample_run(task_type="classification"), db_path=db)
        create_run(_sample_run(task_type="regression"), db_path=db)
        cls_runs = list_runs(filters={"task_type": "classification"}, db_path=db)
        assert len(cls_runs) == 1

    def test_filter_no_match(self, db):
        create_run(_sample_run(), db_path=db)
        result = list_runs(filters={"model_name": "nonexistent"}, db_path=db)
        assert result == []

    def test_ignores_unknown_filter_keys(self, db):
        """Unknown filter keys should be silently ignored (security)."""
        create_run(_sample_run(), db_path=db)
        result = list_runs(filters={"hacker_field": "DROP TABLE"}, db_path=db)
        assert len(result) == 1  # no filter applied, returns all


# ─────────────────────────────────────────────────────────────
# delete_run tests
# ─────────────────────────────────────────────────────────────


class TestDeleteRun:
    """Tests for delete_run()."""

    def test_delete_existing(self, db):
        run_id = create_run(_sample_run(), db_path=db)
        assert delete_run(run_id, db_path=db) is True
        assert get_run(run_id, db_path=db) is None

    def test_delete_nonexistent(self, db):
        assert delete_run("fake-id", db_path=db) is False

    def test_delete_only_target(self, db):
        """Deleting one run should not affect others."""
        id1 = create_run(_sample_run(model_name="keep"), db_path=db)
        id2 = create_run(_sample_run(model_name="delete_me"), db_path=db)
        delete_run(id2, db_path=db)
        assert get_run(id1, db_path=db) is not None
        assert get_run(id2, db_path=db) is None
        assert len(list_runs(db_path=db)) == 1


# ─────────────────────────────────────────────────────────────
# get_metrics_comparison tests
# ─────────────────────────────────────────────────────────────


class TestMetricsComparison:
    """Tests for get_metrics_comparison()."""

    def test_compare_two_runs(self, db):
        id1 = create_run(_sample_run(metrics={"accuracy": 0.87}), db_path=db)
        id2 = create_run(_sample_run(model_name="xgb", metrics={"accuracy": 0.91}), db_path=db)
        comparison = get_metrics_comparison([id1, id2], db_path=db)
        assert len(comparison) == 2
        accuracies = {c["model_name"]: c["metrics"]["accuracy"] for c in comparison}
        assert accuracies["random_forest"] == 0.87
        assert accuracies["xgb"] == 0.91

    def test_compare_empty_list(self, db):
        assert get_metrics_comparison([], db_path=db) == []

    def test_compare_with_bad_id(self, db):
        id1 = create_run(_sample_run(), db_path=db)
        result = get_metrics_comparison([id1, "fake-id"], db_path=db)
        assert len(result) == 1  # only the valid one returned

    def test_comparison_fields(self, db):
        """Comparison should return only relevant fields, not full profile."""
        run_id = create_run(_sample_run(), db_path=db)
        result = get_metrics_comparison([run_id], db_path=db)
        assert "metrics" in result[0]
        assert "config" in result[0]
        assert "model_name" in result[0]
        assert "profile_summary" not in result[0]  # not needed for comparison
