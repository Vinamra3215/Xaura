"""BaseModel and Result — foundation for all XAURA models.

BaseModel is the abstract parent class that all model wrappers inherit from.
Subclasses only need to implement two methods:
    - build(config, profile) → returns an sklearn/xgb/lgbm model object
    - get_default_config(profile) → returns a config dict based on the dataset

The run() method is shared by all subclasses and handles the full pipeline:
    split data → build model → fit → predict → evaluate → return Result

Result is the dataclass that holds everything a model run produces:
    predictions, probabilities, metrics, config, feature importances, etc.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from xaura.profiler.dataprofile import DataProfile

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """Result of a model run.

    Every time you call run_model(), you get one of these back. It contains
    everything about the run: what model was used, what it predicted,
    how well it did, and the config/data used.

    Attributes:
        model_name: Name of the model (e.g. "rf_classifier").
        task_type: One of "classification", "regression", "clustering".
        predictions: Array of predictions (y_pred).
        probabilities: Array of predicted probabilities (classification only).
        metrics: Dict of metric name → value (e.g. {"accuracy": 0.89}).
        config: Dict of hyperparameters used for this run.
        dataset_hash: SHA-256 hash of the input dataset.
        train_time_seconds: How long training took.
        model_object: The fitted sklearn/xgb/lgbm model object.
        feature_importances: Feature importance array (if the model supports it).
        X_train: Training features DataFrame.
        X_test: Test features DataFrame.
        y_train: Training labels Series.
        y_test: Test labels Series.
    """

    # Identity
    model_name: str = ""
    task_type: str = ""  # "classification", "regression", "clustering"

    # Predictions
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    probabilities: np.ndarray | None = None

    # Metrics — flexible dict so each task type can have different keys
    # Classification: accuracy, precision, recall, f1, roc_auc
    # Regression: mse, rmse, mae, r2
    # Clustering: silhouette, calinski_harabasz, inertia
    metrics: dict[str, float] = field(default_factory=dict)

    # Config & reproducibility
    config: dict[str, Any] = field(default_factory=dict)
    dataset_hash: str = ""
    train_time_seconds: float = 0.0

    # Model object (for export / further use)
    model_object: Any = None
    feature_importances: np.ndarray | None = None

    # Train/test splits (for visualisation later)
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None


# ---------------------------------------------------------------------------
# BaseModel abstract class
# ---------------------------------------------------------------------------


class BaseModel(ABC):
    """Abstract base class for all XAURA model wrappers.

    Every model in XAURA (classifiers, regressors, clusterers) inherits
    from this class. Subclasses must implement:
        - build(config, profile) → the actual sklearn/xgb/lgbm model
        - get_default_config(profile) → smart defaults based on the dataset

    The run() method is shared and handles:
        1. Merge user config with defaults
        2. Split data (train/test) — skipped for clustering
        3. Encode categorical features
        4. Build the model
        5. Fit the model
        6. Predict
        7. Evaluate (compute metrics)
        8. Extract feature importances (if available)
        9. Return a Result object

    Class attributes (set by each subclass):
        name: Short identifier (e.g. "rf_classifier"). Used in registry.
        display_name: Human-readable name (e.g. "Random Forest Classifier").
        task_type: One of "classification", "regression", "clustering".
    """

    name: str = ""
    display_name: str = ""
    task_type: str = ""  # "classification", "regression", "clustering"

    @abstractmethod
    def build(self, config: dict[str, Any], profile: DataProfile) -> Any:
        """Build and return the model object (e.g. sklearn estimator).

        Args:
            config: Hyperparameters dict.
            profile: DataProfile of the dataset.

        Returns:
            A model object with .fit() and .predict() methods.
        """

    @abstractmethod
    def get_default_config(self, profile: DataProfile) -> dict[str, Any]:
        """Return dataset-aware default hyperparameters.

        The profile is used to make smart choices — e.g. if the dataset
        is imbalanced, set class_weight="balanced".

        Args:
            profile: DataProfile of the dataset.

        Returns:
            Dict of hyperparameter name → value.
        """

    def run(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
        config: dict[str, Any] | None = None,
        target_col: str | None = None,
        test_size: float = 0.2,
    ) -> Result:
        """Run the full model pipeline and return a Result.

        Steps:
            1. Merge user config with defaults
            2. Split data (skip for clustering)
            3. Encode categorical features as integers
            4. Build the model
            5. Fit
            6. Predict
            7. Evaluate
            8. Return Result

        Args:
            df: Input DataFrame.
            profile: DataProfile of the dataset.
            config: Optional config overrides (merged with defaults).
            target_col: Name of target column. Auto-detected if None.
            test_size: Fraction of data for test set (default 0.2).

        Returns:
            A Result object with predictions, metrics, etc.
        """
        # 1. Merge config: user overrides take priority over defaults
        final_config = self.get_default_config(profile)
        if config:
            final_config.update(config)

        # 2. Handle clustering vs supervised
        if self.task_type == "clustering":
            return self._run_clustering(df, profile, final_config)
        else:
            return self._run_supervised(df, profile, final_config, target_col, test_size)

    def _run_supervised(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
        config: dict[str, Any],
        target_col: str | None,
        test_size: float,
    ) -> Result:
        """Run a supervised model (classification or regression)."""
        # Determine target column
        target = target_col or profile.target_column
        if target is None:
            # Fallback: use the last column
            target = str(df.columns[-1])

        # Split data
        X_train, X_test, y_train, y_test = self._split_data(df, target, test_size)

        # Encode categoricals
        X_train_enc, X_test_enc, encoders = self._encode_categoricals(X_train, X_test)

        # Build model
        model = self.build(config, profile)

        # Fit
        start = time.perf_counter()
        model.fit(X_train_enc, y_train)
        train_time = time.perf_counter() - start

        # Predict
        predictions = model.predict(X_test_enc)

        # Probabilities (classification only)
        probabilities = None
        if self.task_type == "classification" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test_enc)

        # Evaluate
        if self.task_type == "classification":
            metrics = self._evaluate_classification(y_test, predictions, probabilities)
        else:
            metrics = self._evaluate_regression(y_test, predictions)

        # Feature importances
        importances = self._extract_feature_importances(model, X_train_enc.columns)

        return Result(
            model_name=self.name,
            task_type=self.task_type,
            predictions=predictions,
            probabilities=probabilities,
            metrics=metrics,
            config=config,
            dataset_hash=profile.dataset_hash,
            train_time_seconds=train_time,
            model_object=model,
            feature_importances=importances,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

    def _run_clustering(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
        config: dict[str, Any],
    ) -> Result:
        """Run a clustering model (no target, no split)."""
        # Use only numeric columns for clustering
        X = df.select_dtypes(include=[np.number])
        if X.empty:
            raise ValueError("No numeric columns available for clustering")

        # Build model
        model = self.build(config, profile)

        # Fit + predict
        start = time.perf_counter()
        labels = model.fit_predict(X)
        train_time = time.perf_counter() - start

        # Evaluate
        metrics = self._evaluate_clustering(X, labels)

        return Result(
            model_name=self.name,
            task_type=self.task_type,
            predictions=labels,
            metrics=metrics,
            config=config,
            dataset_hash=profile.dataset_hash,
            train_time_seconds=train_time,
            model_object=model,
            X_train=df,  # Full dataset (no split for clustering)
        )

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _split_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split DataFrame into train/test sets.

        Uses stratified splitting for classification (preserves class
        distribution in both sets). Falls back to regular splitting
        for regression or if stratification fails.

        Args:
            df: Input DataFrame.
            target_col: Name of the target column.
            test_size: Fraction for test set.

        Returns:
            (X_train, X_test, y_train, y_test) tuple.
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Stratified split for classification
        stratify = y if self.task_type == "classification" else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify
            )
        except ValueError:
            # Stratification can fail if a class has too few samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        return X_train, X_test, y_train, y_test

    def _encode_categoricals(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, LabelEncoder]]:
        """Encode categorical columns as integers using LabelEncoder.

        Fits the encoder on train data and transforms both train and test.
        Unknown categories in test are mapped to -1.

        Args:
            X_train: Training features.
            X_test: Test features.

        Returns:
            (X_train_encoded, X_test_encoded, encoders) tuple.
        """
        X_train = X_train.copy()
        X_test = X_test.copy()
        encoders: dict[str, LabelEncoder] = {}

        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns

        for col in cat_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))

            # Handle unseen categories in test set
            test_vals = X_test[col].astype(str)
            known = set(le.classes_)
            X_test[col] = test_vals.map(
                lambda x, le=le, known=known: (le.transform([x])[0] if x in known else -1)
            )

            encoders[col] = le

        return X_train, X_test, encoders

    # ------------------------------------------------------------------
    # Evaluation methods
    # ------------------------------------------------------------------

    def _evaluate_classification(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
    ) -> dict[str, float]:
        """Compute classification metrics.

        Computes: accuracy, precision, recall, f1, and roc_auc (if
        probabilities are available).

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (or None).

        Returns:
            Dict of metric name → value.
        """
        # Determine averaging strategy
        n_classes = len(np.unique(y_true))
        average = "binary" if n_classes == 2 else "weighted"

        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        }

        # ROC AUC — needs probabilities and at least 2 classes
        if y_proba is not None and n_classes >= 2:
            try:
                if n_classes == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                    )
            except (ValueError, IndexError):
                # roc_auc can fail if only one class is present in y_true
                pass

        return metrics

    def _evaluate_regression(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute regression metrics.

        Computes: mse, rmse, mae, r2.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Dict of metric name → value.
        """
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def _evaluate_clustering(
        self,
        X: pd.DataFrame | np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Compute clustering metrics.

        Computes: silhouette score, calinski-harabasz index, and inertia
        (if the model has it). Silhouette requires at least 2 clusters.

        Args:
            X: Feature matrix used for clustering.
            labels: Cluster labels from the model.

        Returns:
            Dict of metric name → value.
        """
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        metrics: dict[str, float] = {}

        if n_clusters >= 2:
            metrics["silhouette"] = float(silhouette_score(X, labels))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))

        return metrics

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _extract_feature_importances(
        self,
        model: Any,
        feature_names: pd.Index,
    ) -> np.ndarray | None:
        """Extract feature importances from the model if available.

        Supports both tree-based models (feature_importances_) and
        linear models (coef_).

        Args:
            model: Fitted model object.
            feature_names: Column names of the training features.

        Returns:
            Numpy array of importances, or None if not available.
        """
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                # Multi-class: average across classes
                return np.mean(np.abs(coef), axis=0)
            return np.abs(coef)
        return None
