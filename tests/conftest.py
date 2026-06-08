"""Shared test fixtures for XAURA tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def classification_df():
    """A simple binary classification dataset (200 rows, 5 cols)."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.random.uniform(0, 10, n),
            "category": np.random.choice(["A", "B", "C"], n),
            "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        }
    )


@pytest.fixture
def regression_df():
    """A simple regression dataset (300 rows, 4 cols)."""
    np.random.seed(42)
    n = 300
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": np.random.uniform(0, 100, n),
            "target": 3 * x1 + 2 * x2 + np.random.randn(n) * 0.5,
        }
    )


@pytest.fixture
def clustering_df():
    """A dataset without a target column (for clustering, 150 rows)."""
    np.random.seed(42)
    n = 150
    return pd.DataFrame(
        {
            "x": np.concatenate([np.random.randn(50) + i * 3 for i in range(3)]),
            "y": np.concatenate([np.random.randn(50) + i * 2 for i in range(3)]),
            "z": np.random.randn(n),
        }
    )


@pytest.fixture
def missing_df():
    """A dataset with missing values (10% in one col, 30% in another)."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "complete": np.random.randn(n),
            "some_missing": np.random.randn(n),
            "lots_missing": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
        }
    )
    df.loc[df.index[:10], "some_missing"] = np.nan  # 10% missing
    df.loc[df.index[:30], "lots_missing"] = np.nan  # 30% missing
    return df


@pytest.fixture
def imbalanced_df():
    """A heavily imbalanced classification dataset (10:1 ratio)."""
    np.random.seed(42)
    n = 1100
    return pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "target": np.array([0] * 1000 + [1] * 100),
        }
    )


@pytest.fixture
def empty_df():
    """An empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def single_row_df():
    """A DataFrame with a single row."""
    return pd.DataFrame({"a": [1], "b": [2], "c": ["x"]})


@pytest.fixture
def tmp_csv(tmp_path, classification_df):
    """Write classification data to a temporary CSV and return the path."""
    path = tmp_path / "test_data.csv"
    classification_df.to_csv(path, index=False)
    return path
