"""Stage 2 - Cleaning: rule-based fixes, IQR capping, imputation, statistics report.

Leakage policy (important for viva)
-----------------------------------
Anything that *learns a statistic from data* (IQR fences, medians) must be
fitted on the training split only. Therefore:

* ``DataCleaner``       - STATELESS rules (dedup, drop unlabeled rows, mask
                          physically impossible readings). Safe before the split.
* ``IQRCapper`` + ``SimpleImputer`` - STATEFUL, packaged as an sklearn
                          ``Pipeline`` that is fitted on train folds only.
* ``DistributionReporter`` - prints before/after mean, median, variance.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.config import PHYSICAL_BOUNDS, TARGET_COL
from src.logger import get_logger

logger = get_logger(__name__)


# =========================================================================== #
# Stateful transformer: IQR capping (winsorising)
# =========================================================================== #
class IQRCapper(TransformerMixin, BaseEstimator):
    """Cap values outside Tukey fences ``[Q1 - k*IQR, Q3 + k*IQR]``.

    Why capping and not dropping: dropping rows loses information in every
    *other* column, and with ~1 % spikes per sensor we would discard a large
    share of the data. Capping keeps the row but limits the leverage of the
    outlier on mean/variance-sensitive steps (scaling, linear models).

    Why fit/transform: the fences are *learned parameters*. Learning them on
    the full dataset would leak test-set distribution into training.

    NaNs are ignored when learning fences and are preserved by ``transform``
    (imputation happens in the next step).
    """

    def __init__(self, multiplier: float = 1.5) -> None:
        self.multiplier = multiplier

    def fit(self, X, y=None) -> "IQRCapper":
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")
        arr = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(arr, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.multiplier * iqr
        self.upper_ = q3 + self.multiplier * iqr
        self.n_features_in_ = arr.shape[1]
        self.feature_names_in_ = (np.asarray(X.columns, dtype=object)
                                  if isinstance(X, pd.DataFrame) else None)
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, ["lower_", "upper_"])
        # np.clip leaves NaN untouched, which is exactly what we want here.
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = input_features if input_features is not None else self.feature_names_in_
        if names is None:
            names = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(names, dtype=object)


def build_numeric_cleaning_pipeline(strategy: str, iqr_multiplier: float) -> Pipeline:
    """IQR cap -> impute, for numeric columns.

    Order matters: capping *first* means the (possibly mean-based) imputed
    value is computed on already-tamed data. With the median strategy the
    order is less critical, but this ordering stays correct if someone
    switches to ``strategy="mean"``.
    """
    return Pipeline([
        ("iqr_cap", IQRCapper(multiplier=iqr_multiplier)),
        ("impute", SimpleImputer(strategy=strategy)),
    ])


# =========================================================================== #
# Stateless rules
# =========================================================================== #
class DataCleaner:
    """Row-level, statistics-free cleaning rules (leakage-safe before splitting)."""

    def __init__(self, bounds: Mapping[str, Tuple[float, float]] = PHYSICAL_BOUNDS,
                 target_col: str = TARGET_COL) -> None:
        self._bounds = bounds
        self._target = target_col

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the three stateless rules and log what each one changed."""
        out = df.copy()  # never mutate the caller's frame

        n0 = len(out)
        out = out.drop_duplicates()
        logger.info("Rule 1: removed %d duplicate row(s)", n0 - len(out))

        n1 = len(out)
        # Rows without a label are useless for *supervised* learning and must
        # never be imputed (that would fabricate ground truth).
        out = out.dropna(subset=[self._target])
        logger.info("Rule 2: dropped %d row(s) with missing target '%s'", n1 - len(out), self._target)

        total_masked = 0
        for col, (low, high) in self._bounds.items():
            invalid = (out[col] < low) | (out[col] > high)
            count = int(invalid.sum())
            if count:
                # Impossible value => sensor fault => treat as missing, not "extreme".
                out.loc[invalid, col] = np.nan
                total_masked += count
                logger.info("Rule 3: masked %3d impossible value(s) in %-22s (valid: [%g, %g])",
                            count, col, low, high)
        logger.info("Rule 3 total: %d value(s) masked to NaN", total_masked)
        return out  # original index preserved on purpose (traceability to raw rows)


# =========================================================================== #
# Reporting
# =========================================================================== #
class DistributionReporter:
    """Computes and prints mean / median / variance before vs after cleaning."""

    @staticmethod
    def summarise(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """Mean, median, (sample) variance and NaN count per column.

        pandas skips NaN by default, so 'before' statistics are computed on
        the observed values only - the honest baseline for comparison.
        """
        return pd.DataFrame({
            "missing": df[list(columns)].isna().sum(),
            "mean": df[list(columns)].mean(),
            "median": df[list(columns)].median(),
            "variance": df[list(columns)].var(ddof=1),
        })

    @classmethod
    def report(cls, before: pd.DataFrame, after: pd.DataFrame, columns: Sequence[str],
               fences: pd.DataFrame | None = None) -> pd.DataFrame:
        """Print a side-by-side comparison table and (optionally) the IQR fences."""
        b, a = cls.summarise(before, columns), cls.summarise(after, columns)
        table = pd.concat({"Missing": pd.DataFrame({"before": b["missing"], "after": a["missing"]}),
                           "Mean": pd.DataFrame({"before": b["mean"], "after": a["mean"]}),
                           "Median": pd.DataFrame({"before": b["median"], "after": a["median"]}),
                           "Variance": pd.DataFrame({"before": b["variance"], "after": a["variance"]})},
                          axis=1)
        line = "=" * 118
        print(f"\n{line}\nDESCRIPTIVE STATISTICS - BEFORE vs AFTER CLEANING (training split)\n"
              "before = raw ingested values | after = rules + IQR capping + median imputation\n"
              f"{line}")
        print(table.to_string(float_format=lambda v: f"{v:,.3f}"))
        if fences is not None:
            print(f"\nIQR fences learned on the training split (values beyond are capped):")
            print(fences.to_string(float_format=lambda v: f"{v:,.3f}"))
        print(line)
        return table

    @classmethod
    def report_cleaning_effect(cls, raw_train: pd.DataFrame, rule_clean_train: pd.DataFrame,
                               numeric_cols: Sequence[str], strategy: str,
                               iqr_multiplier: float) -> pd.DataFrame:
        """Fit a *throw-away* cleaning pipeline on the train split for reporting.

        The real pipeline is re-fitted inside every CV fold; this copy exists
        only so we can show students/examiners what the transformation does.
        """
        pipe = build_numeric_cleaning_pipeline(strategy, iqr_multiplier)
        cleaned = pd.DataFrame(pipe.fit_transform(rule_clean_train[list(numeric_cols)]),
                               columns=list(numeric_cols), index=rule_clean_train.index)
        capper: IQRCapper = pipe.named_steps["iqr_cap"]
        observed = rule_clean_train[list(numeric_cols)]
        fences = pd.DataFrame({
            "lower_fence": capper.lower_,
            "upper_fence": capper.upper_,
            "n_capped": ((observed < capper.lower_) | (observed > capper.upper_)).sum().to_numpy(),
        }, index=list(numeric_cols))
        return cls.report(raw_train, cleaned, numeric_cols, fences)
