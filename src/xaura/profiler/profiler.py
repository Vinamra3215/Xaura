"""XAURA Profiler — dataset analysis and profiling engine.

This module contains:
- profile(): The main entry point that analyses a DataFrame end-to-end.
- Helper functions for class balance, correlation, missing values,
  target detection, and task type inference.

Person A built: DataProfile dataclass (dataprofile.py)
Person B built: This file — all the analysis logic that fills the DataProfile.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from xaura.profiler.dataprofile import DataProfile

# ─────────────────────────────────────────────────────────────
# 1. CLASS BALANCE DETECTION
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
        Dict with:
            - counts: {class_label: count} for each unique value
            - ratio: majority_count / minority_count (e.g. 5.0 means 5:1)
            - majority_class: the label of the most common class
            - minority_class: the label of the least common class
        Returns None if no target column or target is continuous.
    """
    if target_col is None or target_col not in df.columns:
        return None

    series = df[target_col].dropna()

    # Only analyse if it looks like a classification target
    # (categorical, or numeric with few unique values)
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
# 2. CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────


def _compute_correlations(df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute Pearson correlation matrix for all numeric columns.

    Pearson correlation (r) measures LINEAR relationship between two variables:
        r = +1  → perfect positive correlation (both go up together)
        r = -1  → perfect negative correlation (one goes up, other goes down)
        r =  0  → no linear relationship

    We compute this for ALL pairs of numeric columns so we can detect
    redundant features (high correlation) that might hurt model performance.

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

    Why |r| > 0.85 matters:
    If two features are highly correlated, they carry almost the same
    information. Including both can:
    - Slow down training
    - Make feature importance unreliable
    - Cause multicollinearity in linear models (unstable coefficients)

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

    # Sort by absolute correlation (highest first)
    flagged.sort(key=lambda x: abs(x[2]), reverse=True)
    return flagged


# ─────────────────────────────────────────────────────────────
# 3. MISSING VALUE ANALYSIS
# ─────────────────────────────────────────────────────────────


def _analyse_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Count missing values per column.

    Missing values (NaN, None, NaT) are problematic because:
    - Most ML models can't handle them directly
    - They might indicate data quality issues
    - The PATTERN of missingness can itself be informative

    We count per-column so the user knows exactly where the gaps are.

    Args:
        df: The dataset.

    Returns:
        Dict of {column_name: missing_count} for columns with missing values.
        Columns with zero missing are excluded to keep it clean.
    """
    missing = df.isnull().sum()
    # Only include columns that actually have missing values
    return {col: int(count) for col, count in missing.items() if count > 0}


def _classify_missing_severity(missing_counts: dict[str, int], n_rows: int) -> dict[str, str]:
    """Classify each column's missing data severity.

    Categories:
        - no_missing: 0%
        - low: < 5% — usually safe to impute (fill in with mean/median)
        - moderate: 5-30% — imputation works but consider investigating WHY
        - high: > 30% — might be better to drop the column entirely

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
# 4. TARGET COLUMN DETECTION & TASK TYPE INFERENCE
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
    """Auto-detect which column is likely the target (what we want to predict).

    Uses a priority-based heuristic:

    1. EXACT NAME MATCH — If a column is named 'target', 'label', 'class',
       'y', or 'output' (case-insensitive), that's almost certainly the target.
       This is the most reliable signal.

    2. LAST COLUMN — By convention in many datasets (especially from Kaggle,
       UCI, etc.), the target is the last column. We use this as a fallback.

    3. Returns None if nothing looks like a target (suggesting unsupervised/
       clustering task).

    Args:
        df: The dataset.

    Returns:
        Column name of the detected target, or None.
    """
    if df.empty or len(df.columns) == 0:
        return None

    # Priority 1: Check for common target column names
    for col in df.columns:
        if col.strip().lower() in _TARGET_NAMES:
            return col

    # Priority 2: Fall back to last column
    # (only if it looks like a reasonable target — not too many unique values
    #  relative to row count, which would suggest it's an ID or text column)
    last_col = df.columns[-1]
    n_unique = df[last_col].nunique()
    n_rows = len(df)

    # If the last column has unique values for almost every row, it's likely
    # an ID column, not a target
    if n_unique / n_rows > 0.5 and n_unique > 20:
        return None

    return last_col


def _infer_task_type(df: pd.DataFrame, target_col: str | None) -> str:
    """Infer what kind of ML task this dataset is for.

    Three types:
    - 'classification': predict a category (spam/not-spam, cat/dog/bird)
    - 'regression': predict a continuous number (house price, temperature)
    - 'clustering': no target — group similar rows together

    Decision logic:
    - No target column → clustering
    - Target is categorical (object/string dtype) → classification
    - Target is numeric with ≤ 20 unique values → classification
      (e.g., 0/1 for binary, or 1-5 for ratings)
    - Target is numeric with > 20 unique values → regression
      (e.g., house prices: $150k, $200k, $350k, ...)

    The threshold of 20 is a common heuristic. A number with 20+ unique
    values is almost always continuous, not categorical.

    Args:
        df: The dataset.
        target_col: Detected target column name, or None.

    Returns:
        One of: 'classification', 'regression', 'clustering'.
    """
    if target_col is None or target_col not in df.columns:
        return "clustering"

    series = df[target_col].dropna()

    # If the target column contains text/categories → classification
    if series.dtype == "object" or series.dtype.name == "category":
        return "classification"

    # If numeric but few unique values → classification (e.g., 0 and 1)
    if series.nunique() <= 20:
        return "classification"

    # Many unique numeric values → regression
    return "regression"


# ─────────────────────────────────────────────────────────────
# 5. WARNING GENERATION
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
    training a model. Better to catch issues early than wonder why the
    model performs badly later.

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
# 6. FEATURE TYPE DETECTION (helper for Person A's profile logic)
# ─────────────────────────────────────────────────────────────


def _detect_feature_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Categorise each column into a feature type.

    Types:
        - numeric: int or float columns (age, price, temperature)
        - categorical: string/object columns with multiple values (color, city)
        - binary: columns with exactly 2 unique values (yes/no, 0/1, True/False)
        - datetime: date or time columns
        - text: string columns with very high cardinality (likely free text)

    Args:
        df: The dataset.

    Returns:
        Dict with keys 'numeric', 'categorical', 'binary', 'datetime', 'text'.
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

        # Check binary first (could be any dtype)
        if n_unique == 2:
            types["binary"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types["datetime"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            types["numeric"].append(col)
        elif df[col].dtype == "object" or df[col].dtype.name == "category":
            # High cardinality text vs normal categorical
            if n_unique / len(df) > 0.5 and n_unique > 50:
                types["text"].append(col)
            else:
                types["categorical"].append(col)

    return types


# ─────────────────────────────────────────────────────────────
# 7. BASIC STATISTICS
# ─────────────────────────────────────────────────────────────


def _compute_basic_stats(df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute descriptive statistics for all numeric columns.

    Computes: mean, std, min, 25%, 50% (median), 75%, max, skew.

    Skewness tells you if the data is lopsided:
        skew ≈ 0  → symmetric (normal-ish)
        skew > 1  → right-skewed (long tail to the right, e.g. income data)
        skew < -1 → left-skewed (long tail to the left)

    This helps decide whether to apply log transforms or use robust scalers.

    Args:
        df: The dataset.

    Returns:
        DataFrame with statistics per numeric column, or None if no numeric cols.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return None

    stats = numeric_df.describe().T  # Transpose so columns are stats
    stats["skew"] = numeric_df.skew()

    return stats


# ─────────────────────────────────────────────────────────────
# 8. DATASET HASHING
# ─────────────────────────────────────────────────────────────


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Generate a SHA-256 hash of the DataFrame for reproducibility.

    This lets us verify that two experiments used the SAME data.
    If you run a model on Monday and get 85% accuracy, then re-run
    on Tuesday, the hash confirms the data hasn't changed.

    Args:
        df: The dataset.

    Returns:
        64-character hex string (SHA-256 digest).
    """
    # pd.util.hash_pandas_object gives a per-row hash; we combine them
    content = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(content.tobytes()).hexdigest()


# ─────────────────────────────────────────────────────────────
# 9. MAIN ENTRY POINT — profile()
# ─────────────────────────────────────────────────────────────


def profile(data: pd.DataFrame | str) -> DataProfile:
    """Profile a dataset and return a complete DataProfile.

    This is the ONE function users call. It runs all analyses and
    packages everything into a single DataProfile object.

    Usage:
        import pandas as pd
        from xaura.profiler.profiler import profile

        df = pd.read_csv("my_data.csv")
        result = profile(df)

        print(result.summary())
        print(result.is_imbalanced)
        print(result.warnings)

    Args:
        data: A pandas DataFrame, or a file path (str) to a CSV file.

    Returns:
        DataProfile with all fields populated.

    Raises:
        TypeError: If data is not a DataFrame or string path.
        FileNotFoundError: If string path doesn't exist.
    """
    # --- Step 0: Handle input type ---
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(f"Expected a pandas DataFrame or CSV file path, got {type(data).__name__}")

    if df.empty:
        return DataProfile(shape=(0, df.shape[1]))

    # --- Step 1: Basic info ---
    shape = df.shape
    feature_types = _detect_feature_types(df)
    basic_stats = _compute_basic_stats(df)
    dataset_hash = _hash_dataframe(df)

    # --- Step 2: Target detection ---
    target_col = _detect_target_column(df)
    task_type = _infer_task_type(df, target_col)

    # --- Step 3: Class balance (only if classification) ---
    class_balance = _analyse_class_balance(df, target_col)

    # --- Step 4: Correlations ---
    corr_matrix = _compute_correlations(df)
    high_corr = _find_high_correlations(corr_matrix)

    # --- Step 5: Missing values ---
    missing_counts = _analyse_missing_values(df)
    missing_severity = _classify_missing_severity(missing_counts, shape[0])

    # --- Step 6: Generate warnings ---
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
