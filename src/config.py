"""Single source of truth for paths, column names, physical limits and hyper-parameters.

Why: magic strings and numbers scattered across modules are the #1 cause of
silent bugs in pipelines. Centralising them means the data generator, the
validator and the transformers can never disagree about a column name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #
TIMESTAMP_COL: str = "timestamp"
TARGET_COL: str = "aqi"
CATEGORICAL_FEATURES: Tuple[str, ...] = ("station_type",)
NUMERIC_FEATURES: Tuple[str, ...] = (
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "pressure_hpa",
    "solar_irradiance_wm2",
    "pm25_ugm3",
    "pm10_ugm3",
    "no2_ppb",
)

# Expected raw schema: column -> logical kind ("datetime" | "numeric" | "category").
EXPECTED_SCHEMA: Mapping[str, str] = {
    TIMESTAMP_COL: "datetime",
    **{col: "numeric" for col in NUMERIC_FEATURES},
    **{col: "category" for col in CATEGORICAL_FEATURES},
    TARGET_COL: "numeric",
}

# Physically plausible ranges (inclusive). Readings outside them are *sensor
# faults*, not extreme weather, so they are masked to NaN and later imputed
# rather than merely capped. This is a stateless rule, hence leakage-free.
PHYSICAL_BOUNDS: Mapping[str, Tuple[float, float]] = {
    "temperature_c": (-30.0, 55.0),
    "humidity_pct": (0.0, 100.0),
    "wind_speed_ms": (0.0, 60.0),
    "pressure_hpa": (870.0, 1085.0),
    "solar_irradiance_wm2": (0.0, 1400.0),
    "pm25_ugm3": (0.0, 1000.0),
    "pm10_ugm3": (0.0, 2000.0),
    "no2_ppb": (0.0, 500.0),
}

# Data-generation constants shared with documentation.
STATION_TYPES: Tuple[str, ...] = ("residential", "industrial", "traffic")


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable run configuration (frozen => cannot be mutated mid-run)."""

    # I/O
    raw_data_path: Path = PROJECT_ROOT / "raw_data.csv"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    # Data generation
    n_rows: int = 2000
    start_date: str = "2025-01-01"

    # Reproducibility
    random_state: int = 42

    # Cleaning
    impute_strategy: str = "median"  # robust to the outliers we know exist
    iqr_multiplier: float = 1.5      # Tukey's classic fence

    # Modelling
    test_size: float = 0.2
    cv_folds: int = 5
    candidate_models: Tuple[str, ...] = ("random_forest", "gradient_boosting")

    def as_dict(self) -> Dict[str, object]:
        """Serialisable view, used when persisting run metadata."""
        return {k: str(v) for k, v in self.__dict__.items()}
