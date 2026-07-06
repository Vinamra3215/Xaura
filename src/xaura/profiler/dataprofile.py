"""DataProfile — the result of profiling a dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DataProfile:
    """Complete profile of a dataset, computed by profile().

    Attributes:
        shape: (n_rows, n_cols) tuple.
        feature_types: Dict grouping column names by type.
            Keys: 'numeric', 'categorical', 'binary', 'datetime', 'text'.
        basic_stats: DataFrame with mean, std, min, max, skew per numeric col.
        class_balance: Dict with class counts and imbalance ratio (if target exists).
        correlations: Correlation matrix DataFrame (numeric features).
        missing_values: Dict of column -> missing count.
        warnings: List of human-readable warning strings.
        target_column: Detected target column name (or None).
        task_type: Inferred task type: 'classification', 'regression', or None.
        dataset_hash: SHA-256 hex digest of the data for reproducibility.
    """

    # Core (Person A — shape, types, stats)
    shape: tuple[int, int] = (0, 0)
    feature_types: dict[str, list[str]] = field(default_factory=dict)
    basic_stats: pd.DataFrame | None = None

    # Extensions (Person B — balance, correlations, missing, warnings)
    class_balance: dict[str, Any] | None = None
    correlations: pd.DataFrame | None = None
    missing_values: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # Target detection (Person B)
    target_column: str | None = None
    task_type: str | None = None  # 'classification', 'regression', or None

    # Reproducibility
    dataset_hash: str = ""

    # --- Computed properties ---

    @property
    def n_rows(self) -> int:
        """Number of rows in the dataset."""
        return self.shape[0]

    @property
    def n_cols(self) -> int:
        """Number of columns in the dataset."""
        return self.shape[1]

    @property
    def is_small(self) -> bool:
        """True if the dataset has fewer than 1,000 rows."""
        return self.n_rows < 1_000

    @property
    def is_large(self) -> bool:
        """True if the dataset has more than 100,000 rows."""
        return self.n_rows > 100_000

    @property
    def is_imbalanced(self) -> bool:
        """True if class imbalance ratio exceeds 5:1."""
        if self.class_balance and "ratio" in self.class_balance:
            return self.class_balance["ratio"] > 5.0
        return False

    @property
    def has_missing(self) -> bool:
        """True if any column has missing values."""
        return any(v > 0 for v in self.missing_values.values())

    @property
    def missing_fraction(self) -> float:
        """Fraction of total cells that are missing."""
        total_cells = self.n_rows * self.n_cols
        if total_cells == 0:
            return 0.0
        return sum(self.missing_values.values()) / total_cells

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Dataset Profile: {self.n_rows:,} rows × {self.n_cols} columns",
            f"  Numeric features:     {len(self.feature_types.get('numeric', []))}",
            f"  Categorical features: {len(self.feature_types.get('categorical', []))}",
            f"  Binary features:      {len(self.feature_types.get('binary', []))}",
        ]

        if self.target_column:
            lines.append(f"  Target column:        {self.target_column} ({self.task_type})")

        if self.has_missing:
            lines.append(
                f"  Missing values:       {sum(self.missing_values.values()):,} "
                f"({self.missing_fraction:.1%} of all cells)"
            )

        if self.warnings:
            lines.append(f"\n  ⚠ Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    • {w}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to a JSON-serializable dictionary."""
        return {
            "shape": self.shape,
            "feature_types": self.feature_types,
            "basic_stats": (
                self.basic_stats.to_json(orient="split") if self.basic_stats is not None else None
            ),
            "class_balance": self.class_balance,
            "missing_values": self.missing_values,
            "warnings": self.warnings,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "dataset_hash": self.dataset_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataProfile:
        """Recreate a DataProfile from a dictionary."""
        import io

        import pandas as pd

        basic_stats = data.get("basic_stats")
        if basic_stats is not None:
            basic_stats = pd.read_json(io.StringIO(basic_stats), orient="split")

        return cls(
            shape=tuple(data.get("shape", (0, 0))),
            feature_types=data.get("feature_types", {}),
            basic_stats=basic_stats,
            class_balance=data.get("class_balance"),
            missing_values=data.get("missing_values", {}),
            warnings=data.get("warnings", []),
            target_column=data.get("target_column"),
            task_type=data.get("task_type"),
            dataset_hash=data.get("dataset_hash", ""),
        )
