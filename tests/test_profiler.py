"""Tests for the XAURA profiler."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xaura import profile
from xaura.profiler import DataProfile


# ---------------------------------------------------------------------------
# profile() function basics
# ---------------------------------------------------------------------------


class TestProfileFunction:
    """Tests for the profile() function."""

    def test_profile_returns_dataprofile(self, classification_df):
        result = profile(classification_df)
        assert isinstance(result, DataProfile)

    def test_profile_shape(self, classification_df):
        result = profile(classification_df)
        assert result.shape == (200, 5)
        assert result.n_rows == 200
        assert result.n_cols == 5

    def test_profile_from_csv(self, tmp_csv):
        result = profile(str(tmp_csv))
        assert result.n_rows == 200

    def test_profile_from_path_object(self, tmp_csv):
        result = profile(Path(tmp_csv))
        assert result.n_rows == 200

    def test_profile_from_numpy_array(self):
        arr = np.random.randn(50, 3)
        result = profile(arr)
        assert result.shape == (50, 3)

    def test_profile_empty_raises(self, empty_df):
        with pytest.raises(ValueError, match="empty"):
            profile(empty_df)

    def test_profile_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            profile("nonexistent_file.csv")

    def test_profile_invalid_type(self):
        with pytest.raises(TypeError):
            profile(42)

    def test_profile_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            profile(str(bad_file))


# ---------------------------------------------------------------------------
# Feature type detection
# ---------------------------------------------------------------------------


class TestFeatureTypeDetection:
    """Tests for feature type classification."""

    def test_numeric_features(self, regression_df):
        result = profile(regression_df)
        assert len(result.feature_types["numeric"]) >= 3

    def test_categorical_features(self, classification_df):
        result = profile(classification_df)
        assert "category" in result.feature_types["categorical"]

    def test_binary_features(self, classification_df):
        result = profile(classification_df)
        assert "target" in result.feature_types["binary"]

    def test_all_feature_type_keys_present(self, classification_df):
        result = profile(classification_df)
        for key in ["numeric", "categorical", "binary", "datetime", "text"]:
            assert key in result.feature_types

    def test_datetime_detection(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=50),
            "value": np.random.randn(50),
        })
        result = profile(df)
        assert "date" in result.feature_types["datetime"]

    def test_text_detection(self):
        df = pd.DataFrame({
            "long_text": [
                "This is a very long text string that exceeds fifty characters easily. " * 2
            ] * 50,
            "short_cat": ["A", "B", "C", "D", "E"] * 10,
        })
        result = profile(df)
        assert "long_text" in result.feature_types["text"]
        assert "short_cat" in result.feature_types["categorical"]

    def test_every_column_is_classified(self, classification_df):
        result = profile(classification_df)
        all_classified = []
        for col_list in result.feature_types.values():
            all_classified.extend(col_list)
        assert len(all_classified) == classification_df.shape[1]


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------


class TestBasicStats:
    """Tests for basic statistics computation."""

    def test_stats_columns_exist(self, regression_df):
        result = profile(regression_df)
        assert result.basic_stats is not None
        for col in ["mean", "std", "min", "max", "median", "skew"]:
            assert col in result.basic_stats.columns

    def test_stats_has_numeric_rows(self, classification_df):
        result = profile(classification_df)
        assert result.basic_stats is not None
        assert len(result.basic_stats) >= 1

    def test_stats_values_reasonable(self, regression_df):
        result = profile(regression_df)
        assert abs(result.basic_stats.loc["x1", "mean"]) < 1.0
        assert 0.5 < result.basic_stats.loc["x1", "std"] < 2.0

    def test_no_numeric_columns(self):
        df = pd.DataFrame({
            "cat1": ["a", "b", "c"] * 20,
            "cat2": ["x", "y", "z"] * 20,
        })
        result = profile(df)
        assert result.basic_stats is not None
        assert result.basic_stats.empty


# ---------------------------------------------------------------------------
# Dataset hash
# ---------------------------------------------------------------------------


class TestDatasetHash:
    """Tests for dataset hashing."""

    def test_hash_is_64_char_hex(self, classification_df):
        result = profile(classification_df)
        assert isinstance(result.dataset_hash, str)
        assert len(result.dataset_hash) == 64

    def test_same_data_same_hash(self, classification_df):
        h1 = profile(classification_df).dataset_hash
        h2 = profile(classification_df).dataset_hash
        assert h1 == h2

    def test_different_data_different_hash(self, classification_df, regression_df):
        h1 = profile(classification_df).dataset_hash
        h2 = profile(regression_df).dataset_hash
        assert h1 != h2


# ---------------------------------------------------------------------------
# DataProfile computed properties
# ---------------------------------------------------------------------------


class TestDataProfileProperties:
    """Tests for DataProfile computed properties."""

    def test_is_small_true(self):
        dp = DataProfile(shape=(500, 10))
        assert dp.is_small is True

    def test_is_small_false(self):
        dp = DataProfile(shape=(5000, 10))
        assert dp.is_small is False

    def test_is_small_boundary(self):
        assert DataProfile(shape=(999, 10)).is_small is True
        assert DataProfile(shape=(1000, 10)).is_small is False

    def test_is_large_true(self):
        dp = DataProfile(shape=(200_000, 10))
        assert dp.is_large is True

    def test_is_large_false(self):
        dp = DataProfile(shape=(5000, 10))
        assert dp.is_large is False

    def test_is_imbalanced_true(self):
        dp = DataProfile(class_balance={"ratio": 8.0})
        assert dp.is_imbalanced is True

    def test_is_imbalanced_false(self):
        dp = DataProfile(class_balance={"ratio": 2.0})
        assert dp.is_imbalanced is False

    def test_is_imbalanced_no_balance_info(self):
        dp = DataProfile()
        assert dp.is_imbalanced is False

    def test_has_missing_true(self):
        dp = DataProfile(missing_values={"col_a": 5, "col_b": 0})
        assert dp.has_missing is True

    def test_has_missing_false(self):
        dp = DataProfile(missing_values={"col_a": 0, "col_b": 0})
        assert dp.has_missing is False

    def test_missing_fraction(self):
        dp = DataProfile(shape=(100, 2), missing_values={"a": 10, "b": 0})
        assert dp.missing_fraction == pytest.approx(0.05)

    def test_missing_fraction_empty(self):
        dp = DataProfile(shape=(0, 0))
        assert dp.missing_fraction == 0.0

    def test_summary_contains_shape(self, classification_df):
        result = profile(classification_df)
        summary = result.summary()
        assert "200" in summary
        assert "5 columns" in summary

    def test_summary_with_warnings(self):
        dp = DataProfile(
            shape=(100, 5),
            feature_types={
                "numeric": [], "categorical": [], "binary": [],
                "datetime": [], "text": [],
            },
            warnings=["High imbalance: 3.2:1"],
        )
        summary = dp.summary()
        assert "Warnings" in summary
        assert "High imbalance" in summary


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for the profiler."""

    def test_single_column(self):
        df = pd.DataFrame({"only_col": [1, 2, 3, 4, 5]})
        result = profile(df)
        assert result.n_cols == 1

    def test_single_row(self, single_row_df):
        result = profile(single_row_df)
        assert result.n_rows == 1

    def test_all_nan_column(self):
        df = pd.DataFrame({
            "good": [1.0, 2.0, 3.0],
            "bad": [np.nan, np.nan, np.nan],
        })
        result = profile(df)
        assert result.n_cols == 2

    def test_wide_dataframe(self):
        df = pd.DataFrame(np.random.randn(50, 100))
        result = profile(df)
        assert result.n_cols == 100

    def test_unicode_column_names(self):
        df = pd.DataFrame({"名前": ["太郎", "花子", "次郎"], "年齢": [25, 30, 35]})
        result = profile(df)
        assert result.n_cols == 2

    def test_duplicate_column_names(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
        result = profile(df)
        assert result.n_cols == 3

    def test_large_numeric_values(self):
        df = pd.DataFrame({"big": [1e15, 2e15, 3e15], "small": [1e-15, 2e-15, 3e-15]})
        result = profile(df)
        assert result.basic_stats is not None
        assert len(result.basic_stats) == 2
