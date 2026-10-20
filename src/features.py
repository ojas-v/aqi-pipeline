"""Stage 3 - Feature engineering and the leakage-free preprocessing pipeline.

Pipeline layout (every box is fitted on training data only):

    raw X
      |  TemporalFeatureExtractor   timestamp -> cyclical hour / day-of-year
      |  ColumnTransformer "clean"  numeric: IQR cap -> median impute
      |                             categorical: most-frequent impute
      |  InteractionFeatureEngineer physics-informed interaction / ratio terms
      |  ColumnTransformer "encode" numeric: StandardScaler, categorical: One-Hot
      v
    model-ready matrix  ->  regressor
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import build_numeric_cleaning_pipeline
from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TIMESTAMP_COL, PipelineConfig


class _DataFrameTransformer(TransformerMixin, BaseEstimator):
    """Base for stateless DataFrame -> DataFrame transformers.

    Remembers input column names so ``get_feature_names_out`` works, which in
    turn lets ``Pipeline.set_output(transform="pandas")`` keep column names
    flowing through the whole chain (needed for ColumnTransformer selection
    by name and for feature-importance reporting).
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def _output_names(self, input_names: Sequence[str]) -> List[str]:  # pragma: no cover
        raise NotImplementedError

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = input_features if input_features is not None else self.feature_names_in_
        return np.asarray(self._output_names(list(names)), dtype=object)


class TemporalFeatureExtractor(_DataFrameTransformer):
    """Turn the timestamp into *cyclical* features and drop the raw column.

    Why sin/cos: hour 23 and hour 0 are one hour apart, but as raw integers
    they look 23 apart. Mapping onto a circle preserves that adjacency. The
    same holds for day-of-year (Dec 31 vs Jan 1).
    """

    OUTPUT_NAMES = ("hour_sin", "hour_cos", "doy_sin", "doy_cos")

    def _output_names(self, input_names: Sequence[str]) -> List[str]:
        return [c for c in input_names if c != TIMESTAMP_COL] + list(self.OUTPUT_NAMES)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ts = pd.to_datetime(X[TIMESTAMP_COL])
        hour = ts.dt.hour + ts.dt.minute / 60.0
        doy = ts.dt.dayofyear
        out = X.drop(columns=[TIMESTAMP_COL]).copy()
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        return out


class InteractionFeatureEngineer(_DataFrameTransformer):
    """Adds domain-motivated custom features (must run AFTER imputation).

    * ``temp_humidity_index``  = T * RH / 100  (interaction: moist-warm air
      drives secondary aerosol formation and hygroscopic growth).
    * ``pm_fine_fraction``     = PM2.5 / PM10  (ratio: separates combustion /
      traffic (fine) from dust (coarse) sources).
    * ``stagnation_index``     = (P - 1013.25) / (wind + 1)  (high pressure
      combined with weak wind => pollutants accumulate; +1 avoids /0).

    Stateless => no leakage risk. Tree ensembles can approximate
    interactions, but explicit features make them cheaper and more stable.
    """

    OUTPUT_NAMES = ("temp_humidity_index", "pm_fine_fraction", "stagnation_index")

    def _output_names(self, input_names: Sequence[str]) -> List[str]:
        return list(input_names) + list(self.OUTPUT_NAMES)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["temp_humidity_index"] = X["temperature_c"] * X["humidity_pct"] / 100.0
        out["pm_fine_fraction"] = X["pm25_ugm3"] / (X["pm10_ugm3"] + 1e-6)
        out["stagnation_index"] = (X["pressure_hpa"] - 1013.25) / (X["wind_speed_ms"] + 1.0)
        return out


class FeaturePipelineFactory:
    """Assembles the complete preprocessing + model pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config

    def build(self, regressor: BaseEstimator) -> Pipeline:
        """Return an unfitted ``Pipeline`` ending in ``regressor``.

        Fitting this single object on the training split (or on each CV
        training fold) guarantees that every statistic - IQR fences, medians,
        scaler mean/std, one-hot categories - comes from training data only.
        """
        numeric = list(NUMERIC_FEATURES)
        categorical = list(CATEGORICAL_FEATURES)
        temporal = list(TemporalFeatureExtractor.OUTPUT_NAMES)
        engineered = list(InteractionFeatureEngineer.OUTPUT_NAMES)

        clean = ColumnTransformer(
            transformers=[
                ("numeric", build_numeric_cleaning_pipeline(
                    self._cfg.impute_strategy, self._cfg.iqr_multiplier), numeric),
                ("categorical", SimpleImputer(strategy="most_frequent"), categorical),
            ],
            remainder="passthrough",          # keep cyclical time features
            verbose_feature_names_out=False,  # keep original column names
        )

        encode_scale = ColumnTransformer(
            transformers=[
                ("scale", StandardScaler(), numeric + engineered + temporal),
                # handle_unknown="ignore": an unseen station type at inference
                # yields an all-zero row instead of crashing production.
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        pipeline = Pipeline([
            ("temporal", TemporalFeatureExtractor()),
            ("clean", clean),
            ("interactions", InteractionFeatureEngineer()),
            ("encode_scale", encode_scale),
            ("model", regressor),
        ])
        # Propagate DataFrames (with column names) between steps.
        pipeline.set_output(transform="pandas")
        return pipeline
