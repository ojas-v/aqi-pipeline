"""Stage 4 - Model selection, training, evaluation and persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.config import TARGET_COL, PipelineConfig
from src.exceptions import ModelTrainingError
from src.features import FeaturePipelineFactory
from src.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Data splitting
# --------------------------------------------------------------------------- #
def split_features_target(df: pd.DataFrame, config: PipelineConfig
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold-out split. The test set is touched exactly once, at final evaluation."""
    X, y = df.drop(columns=[TARGET_COL]), df[TARGET_COL]
    return train_test_split(X, y, test_size=config.test_size, random_state=config.random_state)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegressionMetrics:
    """RMSE (same unit as AQI, penalises large errors), MAE (robust average
    error) and R^2 (share of variance explained)."""

    rmse: float
    mae: float
    r2: float

    @classmethod
    def from_predictions(cls, y_true, y_pred) -> "RegressionMetrics":
        return cls(rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                   mae=float(mean_absolute_error(y_true, y_pred)),
                   r2=float(r2_score(y_true, y_pred)))


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
class ModelFactory:
    """Registry of candidate regressors (open/closed: add a model, touch nothing else)."""

    @staticmethod
    def create(name: str, random_state: int) -> BaseEstimator:
        registry = {
            "random_forest": lambda: RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=random_state),
            "gradient_boosting": lambda: GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                random_state=random_state),
        }
        if name not in registry:
            raise ModelTrainingError(f"Unknown model '{name}'. Available: {sorted(registry)}")
        return registry[name]()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
class ModelTrainer:
    """Cross-validates candidates, picks the best, refits it on all training data."""

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._factory = FeaturePipelineFactory(config)

    def cross_validate_candidates(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """K-fold CV of the *entire* pipeline (preprocessing included).

        Cross-validating the whole pipeline - not a pre-transformed matrix - is
        what makes the estimate honest: fences/medians/scalers are re-learned
        inside each training fold. Train-vs-validation gap exposes overfitting.
        """
        cv = KFold(n_splits=self._cfg.cv_folds, shuffle=True, random_state=self._cfg.random_state)
        scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
        rows: Dict[str, Dict[str, float]] = {}
        for name in self._cfg.candidate_models:
            pipe = self._factory.build(ModelFactory.create(name, self._cfg.random_state))
            try:
                res = cross_validate(pipe, X, y, cv=cv, scoring=scoring,
                                     return_train_score=True, error_score="raise")
            except Exception as exc:  # surfaced with context, original chained
                raise ModelTrainingError(f"Cross-validation failed for '{name}': {exc}") from exc
            rows[name] = {
                "cv_rmse": -res["test_rmse"].mean(), "cv_rmse_std": res["test_rmse"].std(),
                "cv_mae": -res["test_mae"].mean(), "cv_r2": res["test_r2"].mean(),
                "cv_r2_std": res["test_r2"].std(), "train_r2": res["train_r2"].mean(),
            }
            logger.info("CV %-18s RMSE=%.3f  MAE=%.3f  R2=%.4f", name,
                        rows[name]["cv_rmse"], rows[name]["cv_mae"], rows[name]["cv_r2"])
        return pd.DataFrame(rows).T.sort_values("cv_rmse")

    def fit_best(self, cv_table: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> Tuple[str, Pipeline]:
        """Refit the lowest-CV-RMSE candidate on the full training split."""
        best = str(cv_table.index[0])
        pipe = clone(self._factory.build(ModelFactory.create(best, self._cfg.random_state)))
        try:
            pipe.fit(X, y)
        except Exception as exc:
            raise ModelTrainingError(f"Final fit failed for '{best}': {exc}") from exc
        logger.info("Selected model: %s", best)
        return best, pipe


# --------------------------------------------------------------------------- #
# Evaluation + persistence
# --------------------------------------------------------------------------- #
class ModelEvaluator:
    """Hold-out evaluation, report printing and artifact persistence."""

    def evaluate(self, pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> RegressionMetrics:
        try:
            return RegressionMetrics.from_predictions(y_test, pipe.predict(X_test))
        except Exception as exc:
            raise ModelTrainingError(f"Evaluation failed: {exc}") from exc

    @staticmethod
    def feature_importances(pipe: Pipeline, top_n: int = 10) -> pd.Series:
        names = pipe[:-1].get_feature_names_out()
        return pd.Series(pipe[-1].feature_importances_, index=names).sort_values(ascending=False).head(top_n)

    def print_report(self, cv_table: pd.DataFrame, best_name: str, test: RegressionMetrics,
                     importances: pd.Series) -> None:
        line = "=" * 78
        print(f"\n{line}\nMODEL EVALUATION REPORT\n{line}")
        print(f"\n[1] Cross-validation on training split (mean over folds)")
        print(cv_table.to_string(float_format=lambda v: f"{v:,.4f}"))
        print(f"\n[2] Hold-out test performance - selected model: {best_name}")
        print(f"    RMSE : {test.rmse:8.3f}  AQI points")
        print(f"    MAE  : {test.mae:8.3f}  AQI points")
        print(f"    R^2  : {test.r2:8.4f}")
        print("\n[3] Top feature importances")
        print(importances.to_string(float_format=lambda v: f"{v:.4f}"))
        print(line)

    @staticmethod
    def save(pipe: Pipeline, metrics: RegressionMetrics, config: PipelineConfig, best_name: str) -> Path:
        """Persist the *whole* fitted pipeline so inference reuses identical preprocessing."""
        out = config.artifacts_dir
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out / "aqi_model.joblib")
        (out / "metrics.json").write_text(json.dumps(
            {"model": best_name, "test_metrics": asdict(metrics), "config": config.as_dict()}, indent=2))
        logger.info("Saved model + metrics to %s", out)
        return out / "aqi_model.joblib"
