"""Core profiling function — analyses a dataset and returns a DataProfile."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from xaura.profiler.dataprofile import DataProfile


def profile(data: pd.DataFrame | str | Path | np.ndarray) -> DataProfile:
    """Profile a dataset and return a DataProfile.

    Accepts a pandas DataFrame, a file path (CSV/Excel/Parquet/JSON),
    or a numpy array. Returns a DataProfile populated with shape,
    feature types, basic statistics, and a dataset hash.

    Args:
        data: A pandas DataFrame, path to a data file, or numpy array.

    Returns:
        A DataProfile with shape, feature types, basic stats, and dataset hash.

    Raises:
        ValueError: If the data is empty or cannot be loaded.
        FileNotFoundError: If a file path is given but doesn't exist.
        TypeError: If the data type is not supported.
    """
    df = _load_data(data)
    _validate(df)

    return DataProfile(
        shape=(df.shape[0], df.shape[1]),
        feature_types=_detect_feature_types(df),
        basic_stats=_compute_basic_stats(df),
        dataset_hash=_compute_hash(df),
    )


def _load_data(data: pd.DataFrame | str | Path | np.ndarray) -> pd.DataFrame:
    """Convert input to a pandas DataFrame.

    Supports:
        - pd.DataFrame (returned as-is)
        - np.ndarray (wrapped in DataFrame)
        - str or Path to .csv, .xlsx, .xls, .parquet, .json
    """
    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, np.ndarray):
        return pd.DataFrame(data)

    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        elif suffix in (".xls", ".xlsx"):
            return pd.read_excel(path)
        elif suffix == ".parquet":
            return pd.read_parquet(path)
        elif suffix == ".json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    raise TypeError(f"Expected DataFrame, path, or ndarray, got {type(data).__name__}")


def _validate(df: pd.DataFrame) -> None:
    """Validate that the DataFrame is usable for profiling."""
    if df.empty:
        raise ValueError("Dataset is empty (0 rows or 0 columns)")
    if df.shape[0] == 0:
        raise ValueError("Dataset has 0 rows")
    if df.shape[1] == 0:
        raise ValueError("Dataset has 0 columns")


def _detect_feature_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify each column into a feature type.

    Types:
        - numeric: int or float columns with > 2 unique values
        - binary: columns with exactly 2 unique non-null values
        - categorical: object/string columns, or low-cardinality int/float
        - datetime: datetime columns
        - text: string columns where average length > 50 characters
    """
    types: dict[str, list[str]] = {
        "numeric": [],
        "categorical": [],
        "binary": [],
        "datetime": [],
        "text": [],
    }

    for col in df.columns:
        n_unique = df[col].nunique()

        # Datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types["datetime"].append(str(col))
            continue

        # Binary (exactly 2 unique non-null values, any dtype)
        if n_unique == 2:
            types["binary"].append(str(col))
            continue

        # Numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Low-cardinality numeric -> treat as categorical
            if n_unique <= 20 and len(df) > 0 and (n_unique / len(df)) < 0.05:
                types["categorical"].append(str(col))
            else:
                types["numeric"].append(str(col))
            continue

        # String / object
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Text detection (long strings)
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if not np.isnan(avg_len) and avg_len > 50:
                types["text"].append(str(col))
            else:
                types["categorical"].append(str(col))
            continue

        # Fallback — treat as categorical
        types["categorical"].append(str(col))

    return types


def _compute_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic statistics for numeric columns.

    Returns a DataFrame with columns: mean, std, min, max, median, skew.
    Each row corresponds to a numeric column from the input DataFrame.
    Returns an empty DataFrame if there are no numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return pd.DataFrame()

    stats_data = {}
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        stats_data[col] = {
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "median": float(col_data.median()),
            "skew": float(scipy_stats.skew(col_data, nan_policy="omit")),
        }

    return pd.DataFrame(stats_data).T


def _compute_hash(df: pd.DataFrame) -> str:
    """Compute a SHA-256 hash of the DataFrame for reproducibility.

    Uses pandas object hashing to generate a deterministic fingerprint
    of the dataset contents.
    """
    return hashlib.sha256(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()
