"""Core profiling function — analyses a dataset and returns a DataProfile.

This module contains:
- profile(): The main entry point that analyses a DataFrame end-to-end.
- Data loading and validation (Person A)
- Feature type detection and basic statistics (Person A)
- Class balance, correlation, missing values, target detection,
  task type inference, and warning generation (Person B)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from xaura.profiler.dataprofile import DataProfile

# ─────────────────────────────────────────────────────────────
# DATA LOADING & VALIDATION (Person A)
# ─────────────────────────────────────────────────────────────


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

    if isinstance(data, str | Path):
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


# ─────────────────────────────────────────────────────────────
# FEATURE TYPE DETECTION (Person A)
# ─────────────────────────────────────────────────────────────


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

    for i, col in enumerate(df.columns):
        series = df.iloc[:, i]
        n_unique = series.nunique()

        # Datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            types["datetime"].append(str(col))
            continue

        # Binary (exactly 2 unique non-null values, any dtype)
        if n_unique == 2:
            types["binary"].append(str(col))
            continue

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            # Low-cardinality numeric -> treat as categorical
            if n_unique <= 20 and len(df) > 0 and (n_unique / len(df)) < 0.05:
                types["categorical"].append(str(col))
            else:
                types["numeric"].append(str(col))
            continue

        # String / object
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            # Text detection (long strings)
            avg_len = series.dropna().astype(str).str.len().mean()
            if not np.isnan(avg_len) and avg_len > 50:
                types["text"].append(str(col))
            else:
                types["categorical"].append(str(col))
            continue

        # Fallback — treat as categorical
        types["categorical"].append(str(col))

    return types


# ─────────────────────────────────────────────────────────────
# BASIC STATISTICS (Person A)
# ─────────────────────────────────────────────────────────────


def _compute_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic statistics for numeric columns.

    Returns a DataFrame with columns: mean, std, min, max, median, skew.
    Each row corresponds to a numeric column from the input DataFrame.
    Returns an empty DataFrame if there are no numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return pd.DataFrame()

    # Use iloc to handle duplicate column names safely
    numeric_indices = [
        i for i, dtype in enumerate(df.dtypes) if pd.api.types.is_numeric_dtype(dtype)
    ]

    stats_data = {}
    for idx in numeric_indices:
        col_name = str(df.columns[idx])
        col_data = df.iloc[:, idx].dropna()
        if len(col_data) == 0:
            continue
        # If duplicate col names, make keys unique
        key = col_name if col_name not in stats_data else f"{col_name}_{idx}"
        stats_data[key] = {
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "median": float(col_data.median()),
            "skew": float(scipy_stats.skew(col_data, nan_policy="omit")),
        }

    return pd.DataFrame(stats_data).T


# ─────────────────────────────────────────────────────────────
# DATASET HASHING (Person A)
# ─────────────────────────────────────────────────────────────


def _compute_hash(df: pd.DataFrame) -> str:
    """Compute a SHA-256 hash of the DataFrame for reproducibility.

    Uses pandas object hashing to generate a deterministic fingerprint
    of the dataset contents.
    """
    return hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()


# ─────────────────────────────────────────────────────────────
# CLASS BALANCE DETECTION (Person B)
# ─────────────────────────────────────────────────────────────


def _analyse_class_balance(df: pd.DataFrame, target_col: str | None) -> dict[str, Any] | None:
    """Analyse class distribution of the target column.

    For classification tasks, we need to know if one class dominates.
    A model trained on imbalanced data (e.g. 95% "no fraud", 5% "fraud")
    will just predict the majority class and appear accurate — but it's
    useless at catching the rare class.

    Args:
        df: The dataset.
        target_col: Name of the target column, or None.

    Returns:
        Dict with counts, ratio, majority/minority class info.
        Returns None if no target column or target is continuous.
    """
    if target_col is None or target_col not in df.columns:
        return None

    series = df[target_col].dropna()

    # Only analyse if it looks like a classification target
    n_unique = series.nunique()
    if n_unique > 20 or n_unique < 2:
        return None

    counts = series.value_counts()
    majority_count = counts.iloc[0]
    minority_count = counts.iloc[-1]

    return {
        "counts": counts.to_dict(),
        "ratio": round(majority_count / minority_count, 2) if minority_count > 0 else float("inf"),
        "majority_class": counts.index[0],
        "minority_class": counts.index[-1],
        "n_classes": n_unique,
    }


# ─────────────────────────────────────────────────────────────
# CORRELATION ANALYSIS (Person B)
# ─────────────────────────────────────────────────────────────


def _compute_correlations(df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute Pearson correlation matrix for all numeric columns.

    Pearson correlation (r) measures LINEAR relationship between two variables:
        r = +1  → perfect positive correlation
        r = -1  → perfect negative correlation
        r =  0  → no linear relationship

    Args:
        df: The dataset.

    Returns:
        Correlation matrix as a DataFrame, or None if < 2 numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return None

    return df[numeric_cols].corr()


def _find_high_correlations(
    corr_matrix: pd.DataFrame | None, threshold: float = 0.85
) -> list[tuple[str, str, float]]:
    """Find feature pairs with correlation above the threshold.

    If two features are highly correlated, they carry almost the same
    information. Including both can slow down training and make feature
    importance unreliable.

    Args:
        corr_matrix: Correlation matrix from _compute_correlations.
        threshold: Minimum |r| to flag. Default 0.85.

    Returns:
        List of (feature_a, feature_b, r_value) tuples, sorted by |r| desc.
    """
    if corr_matrix is None:
        return []

    flagged = []
    cols = corr_matrix.columns.tolist()

    # Only check upper triangle (avoid duplicates and self-correlation)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= threshold:
                flagged.append((cols[i], cols[j], round(r, 4)))

    flagged.sort(key=lambda x: abs(x[2]), reverse=True)
    return flagged


# ─────────────────────────────────────────────────────────────
# MISSING VALUE ANALYSIS (Person B)
# ─────────────────────────────────────────────────────────────


def _analyse_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Count missing values per column.

    Missing values (NaN, None, NaT) are problematic because most ML models
    can't handle them directly.

    Args:
        df: The dataset.

    Returns:
        Dict of {column_name: missing_count} for columns with missing values.
    """
    missing = df.isnull().sum()
    return {col: int(count) for col, count in missing.items() if count > 0}


def _classify_missing_severity(missing_counts: dict[str, int], n_rows: int) -> dict[str, str]:
    """Classify each column's missing data severity.

    Categories:
        - low: < 5% — usually safe to impute
        - moderate: 5-30% — imputation works but investigate WHY
        - high: > 30% — might be better to drop the column

    Args:
        missing_counts: {column: missing_count} dict.
        n_rows: Total number of rows in the dataset.

    Returns:
        Dict of {column_name: severity_category}.
    """
    if n_rows == 0:
        return {}

    severity = {}
    for col, count in missing_counts.items():
        pct = count / n_rows
        if pct == 0:
            severity[col] = "no_missing"
        elif pct < 0.05:
            severity[col] = "low"
        elif pct < 0.30:
            severity[col] = "moderate"
        else:
            severity[col] = "high"
    return severity


# ─────────────────────────────────────────────────────────────
# TARGET COLUMN DETECTION & TASK TYPE INFERENCE (Person B)
# ─────────────────────────────────────────────────────────────

# Common names people use for target columns
_TARGET_NAMES = {
    "target",
    "label",
    "class",
    "y",
    "output",
    "outcome",
    "response",
    "dependent",
}


def _detect_target_column(df: pd.DataFrame) -> str | None:
    """Auto-detect which column is likely the target.

    Uses a priority-based heuristic:
    1. EXACT NAME MATCH — column named 'target', 'label', 'class', 'y', etc.
    2. LAST COLUMN — by convention in many datasets (Kaggle, UCI, etc.)
    3. None — if nothing looks like a target (unsupervised task).

    Args:
        df: The dataset.

    Returns:
        Column name of the detected target, or None.
    """
    if df.empty or len(df.columns) == 0:
        return None

    # Priority 1: Check for common target column names
    for col in df.columns:
        if str(col).strip().lower() in _TARGET_NAMES:
            return col

    # Priority 2: Fall back to last column
    last_col = df.columns[-1]
    n_unique = df[last_col].nunique()
    n_rows = len(df)

    # If last column has unique values for almost every row, it's likely an ID
    if n_unique / n_rows > 0.5 and n_unique > 20:
        return None

    return last_col


def _infer_task_type(df: pd.DataFrame, target_col: str | None) -> str:
    """Infer what kind of ML task this dataset is for.

    Decision logic:
    - No target column → clustering
    - Target is categorical → classification
    - Target is numeric with ≤ 20 unique values → classification
    - Target is numeric with > 20 unique values → regression

    Args:
        df: The dataset.
        target_col: Detected target column name, or None.

    Returns:
        One of: 'classification', 'regression', 'clustering'.
    """
    if target_col is None or target_col not in df.columns:
        return "clustering"

    series = df[target_col].dropna()

    if series.dtype == "object" or series.dtype.name == "category":
        return "classification"

    if series.nunique() <= 20:
        return "classification"

    return "regression"


# ─────────────────────────────────────────────────────────────
# WARNING GENERATION (Person B)
# ─────────────────────────────────────────────────────────────


def _generate_warnings(
    class_balance: dict[str, Any] | None,
    high_correlations: list[tuple[str, str, float]],
    missing_counts: dict[str, int],
    missing_severity: dict[str, str],
    n_rows: int,
) -> list[str]:
    """Generate human-readable warnings about potential data issues.

    These warnings help the user understand problems in their data BEFORE
    training a model.

    Args:
        class_balance: Output of _analyse_class_balance, or None.
        high_correlations: Output of _find_high_correlations.
        missing_counts: Output of _analyse_missing_values.
        missing_severity: Output of _classify_missing_severity.
        n_rows: Total rows in the dataset.

    Returns:
        List of warning strings.
    """
    warnings = []

    # --- Class imbalance warnings ---
    if class_balance and class_balance["ratio"] > 3.0:
        ratio = class_balance["ratio"]
        minority = class_balance["minority_class"]
        warnings.append(
            f"Class imbalance detected — {ratio}:1 ratio. "
            f"Minority class '{minority}' may be underrepresented. "
            f"Consider class weights, SMOTE, or stratified sampling."
        )

    # --- High correlation warnings ---
    for feat_a, feat_b, r in high_correlations:
        warnings.append(
            f"High correlation ({r}) between '{feat_a}' and '{feat_b}' — "
            f"consider dropping one to reduce redundancy."
        )

    # --- Missing value warnings ---
    moderate_cols = [col for col, sev in missing_severity.items() if sev == "moderate"]
    high_cols = [col for col, sev in missing_severity.items() if sev == "high"]

    if high_cols:
        pcts = {col: f"{missing_counts[col] / n_rows:.0%}" for col in high_cols}
        cols_str = ", ".join(f"'{c}' ({pcts[c]})" for c in high_cols)
        warnings.append(
            f"High missing values in: {cols_str}. "
            f"Consider dropping these columns or investigating the cause."
        )

    if moderate_cols:
        pcts = {col: f"{missing_counts[col] / n_rows:.0%}" for col in moderate_cols}
        cols_str = ", ".join(f"'{c}' ({pcts[c]})" for c in moderate_cols)
        warnings.append(
            f"Moderate missing values in: {cols_str}. Imputation (mean/median/mode) is recommended."
        )

    # --- Small dataset warning ---
    if n_rows < 100:
        warnings.append(
            f"Very small dataset ({n_rows} rows). "
            f"Results may be unreliable — consider collecting more data."
        )

    return warnings


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT — profile()
# ─────────────────────────────────────────────────────────────


def profile(data: pd.DataFrame | str | Path | np.ndarray) -> DataProfile:
    """Profile a dataset and return a complete DataProfile.

    This is the ONE function users call. It runs all analyses and
    packages everything into a single DataProfile object.

    Accepts a pandas DataFrame, a file path (CSV/Excel/Parquet/JSON),
    or a numpy array. Returns a DataProfile populated with all fields.

    Usage:
        import pandas as pd
        from xaura import profile

        df = pd.read_csv("my_data.csv")
        result = profile(df)
        print(result.summary())

    Args:
        data: A pandas DataFrame, path to a data file, or numpy array.

    Returns:
        A DataProfile with all fields populated.

    Raises:
        ValueError: If the data is empty or cannot be loaded.
        FileNotFoundError: If a file path is given but doesn't exist.
        TypeError: If the data type is not supported.
    """
    # --- Step 0: Load and validate ---
    df = _load_data(data)
    _validate(df)

    # --- Step 1: Basic info (Person A) ---
    shape = df.shape
    feature_types = _detect_feature_types(df)
    basic_stats = _compute_basic_stats(df)
    dataset_hash = _compute_hash(df)

    # --- Step 2: Target detection (Person B) ---
    target_col = _detect_target_column(df)
    task_type = _infer_task_type(df, target_col)

    # --- Step 3: Class balance (Person B) ---
    class_balance = _analyse_class_balance(df, target_col)

    # --- Step 4: Correlations (Person B) ---
    corr_matrix = _compute_correlations(df)
    high_corr = _find_high_correlations(corr_matrix)

    # --- Step 5: Missing values (Person B) ---
    missing_counts = _analyse_missing_values(df)
    missing_severity = _classify_missing_severity(missing_counts, shape[0])

    # --- Step 6: Generate warnings (Person B) ---
    warnings = _generate_warnings(
        class_balance=class_balance,
        high_correlations=high_corr,
        missing_counts=missing_counts,
        missing_severity=missing_severity,
        n_rows=shape[0],
    )

    # --- Step 7: Package everything ---
    return DataProfile(
        shape=shape,
        feature_types=feature_types,
        basic_stats=basic_stats,
        class_balance=class_balance,
        correlations=corr_matrix,
        missing_values=missing_counts,
        warnings=warnings,
        target_column=target_col,
        task_type=task_type,
        dataset_hash=dataset_hash,
    )
