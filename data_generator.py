"""Phase 1 - Synthetic smart-city air-quality data generator.

The generator is *physically motivated*, not random noise:

* Temperature follows seasonal + diurnal cycles; humidity is anti-correlated
  with temperature.
* PM2.5 falls with wind (dispersion), rises with humidity (hygroscopic growth),
  with high pressure (stagnation / inversion), in winter, and at rush hour.
* PM10 is coarse + fine particles, hence strongly correlated with PM2.5.
* NO2 is traffic driven and is photolysed by sunlight (lower at midday).
* The target AQI is derived from the US-EPA piece-wise linear sub-index of
  PM2.5 and PM10 (plus a small NO2 term and sensor noise), i.e. the structure
  an ML model should be able to recover.

Real-world imperfections are injected *after* the clean signal is computed, so
the ground truth stays clean while the observed data is dirty:

* sensor spikes (multiplicative / additive outliers),
* sentinel error codes (-999) and impossible values (humidity > 100 %),
* MCAR missing values with per-sensor rates, plus one contiguous sensor outage,
* a few missing target labels.

Run standalone:  ``python data_generator.py``
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    STATION_TYPES,
    TARGET_COL,
    TIMESTAMP_COL,
    PipelineConfig,
)
from src.exceptions import PipelineError
from src.logger import configure_logging, get_logger

logger = get_logger(__name__)

# US-EPA AQI breakpoints: concentration -> sub-index (linear in between).
_PM25_BREAKPOINTS = ([0.0, 9.0, 35.4, 55.4, 125.4, 225.4, 325.4],
                     [0, 50, 100, 150, 200, 300, 500])
_PM10_BREAKPOINTS = ([0.0, 54.0, 154.0, 254.0, 354.0, 424.0, 604.0],
                     [0, 50, 100, 150, 200, 300, 500])

# Station-specific emission multipliers (industrial sites emit more PM, kerbside
# traffic sites emit more NO2).
_PM_STATION_FACTOR: Dict[str, float] = {"residential": 1.0, "industrial": 1.45, "traffic": 1.25}
_NO2_STATION_FACTOR: Dict[str, float] = {"residential": 1.0, "industrial": 1.2, "traffic": 1.8}

# Fraction of values missing per feature (MCAR). Different sensors fail at
# different rates, which is how real deployments behave.
_MISSING_RATES: Dict[str, float] = {
    "temperature_c": 0.03,
    "humidity_pct": 0.05,
    "wind_speed_ms": 0.04,
    "pressure_hpa": 0.03,
    "solar_irradiance_wm2": 0.06,
    "pm25_ugm3": 0.07,
    "pm10_ugm3": 0.09,
    "no2_ppb": 0.08,
    "station_type": 0.015,
}
_OUTLIER_RATE: float = 0.012   # share of rows spiked, per affected feature
_SENTINEL_RATE: float = 0.004  # share of rows with the -999 error code
_TARGET_MISSING_RATE: float = 0.005


class AQIDataGenerator:
    """Builds a realistic, deliberately dirty AQI dataset.

    Parameters
    ----------
    config:
        Run configuration; ``n_rows``, ``start_date`` and ``random_state`` are used.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._cfg = config or PipelineConfig()
        # One seeded Generator -> fully reproducible dataset (important for
        # reproducible viva demos and for debugging).
        self._rng = np.random.default_rng(self._cfg.random_state)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self) -> pd.DataFrame:
        """Return the full dirty dataset (clean physics -> faults -> gaps)."""
        if self._cfg.n_rows < 1000:
            raise PipelineError("n_rows must be >= 1000 to satisfy the project spec.")
        df = self._simulate_clean_signals()
        df = self._derive_target(df)
        df = self._inject_sensor_faults(df)
        df = self._inject_missing_values(df)
        logger.info("Generated dataset: %d rows x %d columns", *df.shape)
        return df

    def save(self, df: pd.DataFrame, path: Path | None = None) -> Path:
        """Persist the dataset as CSV and return the written path."""
        target = Path(path or self._cfg.raw_data_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(target, index=False)
        except OSError as exc:
            raise PipelineError(f"Could not write raw data to {target}: {exc}") from exc
        logger.info("Saved raw data -> %s", target)
        return target

    # ------------------------------------------------------------------ #
    # Step 1: clean physical simulation
    # ------------------------------------------------------------------ #
    def _simulate_clean_signals(self) -> pd.DataFrame:
        n, rng = self._cfg.n_rows, self._rng

        # Irregular timestamps across one year: asynchronous multi-station
        # readings. (A fixed 4-hour step would only ever visit 6 distinct
        # hours, destroying the diurnal signal.)
        minutes = np.sort(rng.integers(0, 365 * 24 * 60, size=n))
        ts = pd.Timestamp(self._cfg.start_date) + pd.to_timedelta(minutes, unit="m")
        hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
        doy = ts.dayofyear.to_numpy()

        seasonal = np.sin(2 * np.pi * (doy - 80) / 365.0)       # +1 mid-summer
        winter_factor = 1.0 + 0.35 * np.cos(2 * np.pi * (doy - 15) / 365.0)

        temperature = (18 + 10 * seasonal
                       + 5 * np.sin(2 * np.pi * (hour - 9) / 24.0)   # peak at 15:00
                       + rng.normal(0, 1.5, n))
        humidity = np.clip(65 - 1.6 * (temperature - 18) + rng.normal(0, 7, n), 15, 100)
        wind = rng.weibull(2.0, n) * 3.9                         # mean ~3.4 m/s
        # Smooth (persistent) pressure: weather systems last for days.
        pressure = 1013 + 2.5 * pd.Series(rng.normal(0, 6, n)).ewm(alpha=0.2).mean().to_numpy()
        cloud = rng.beta(2, 3, n)
        daylight = np.clip(np.sin(np.pi * (hour - 6) / 12.0), 0, None)
        irradiance = np.clip(
            950 * daylight * (1 - 0.7 * cloud) * (0.8 + 0.2 * seasonal) + rng.normal(0, 15, n),
            0, None,
        )

        station = rng.choice(STATION_TYPES, size=n, p=[0.45, 0.25, 0.30])
        pm_station = np.vectorize(_PM_STATION_FACTOR.get)(station)
        no2_station = np.vectorize(_NO2_STATION_FACTOR.get)(station)
        rush = (((hour >= 7) & (hour <= 10)) | ((hour >= 17) & (hour <= 20))).astype(float)

        pm25 = (30.0 * pm_station * winter_factor
                * np.exp(-0.13 * wind)
                * (1 + 0.012 * (humidity - 60))
                * (1 + 0.5 * rush)
                * (1 + 0.012 * (pressure - 1013))
                * rng.lognormal(0, 0.22, n))
        pm10 = np.maximum(pm25 * rng.normal(1.7, 0.15, n) + rng.gamma(2.0, 3.0, n), pm25)
        no2 = (18.0 * no2_station * (1 + 0.6 * rush) * np.exp(-0.08 * wind)
               * (1.25 - 0.5 * irradiance / 950.0) * rng.lognormal(0, 0.25, n))

        return pd.DataFrame({
            TIMESTAMP_COL: ts,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "wind_speed_ms": wind,
            "pressure_hpa": pressure,
            "solar_irradiance_wm2": irradiance,
            "pm25_ugm3": pm25,
            "pm10_ugm3": pm10,
            "no2_ppb": no2,
            "station_type": station,
        })

    # ------------------------------------------------------------------ #
    # Step 2: target
    # ------------------------------------------------------------------ #
    def _derive_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """AQI = max(PM2.5 sub-index, PM10 sub-index) + 0.15*NO2 + sensor noise.

        Computed on the *clean* signals so the label is unaffected by the
        faults injected later (a faulty PM sensor does not change real air).
        """
        sub25 = np.interp(df["pm25_ugm3"], *_PM25_BREAKPOINTS)
        sub10 = np.interp(df["pm10_ugm3"], *_PM10_BREAKPOINTS)
        aqi = np.maximum(sub25, sub10) + 0.15 * df["no2_ppb"] + self._rng.normal(0, 4.0, len(df))
        df[TARGET_COL] = np.clip(aqi, 0, 500)
        return df

    # ------------------------------------------------------------------ #
    # Step 3: realism - faults and gaps
    # ------------------------------------------------------------------ #
    def _inject_sensor_faults(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject spikes, impossible readings and -999 error codes."""
        n, rng = len(df), self._rng
        k = max(1, int(_OUTLIER_RATE * n))

        # Multiplicative spikes (e.g. a truck idling at the sensor intake).
        for col in ("pm25_ugm3", "pm10_ugm3", "no2_ppb", "wind_speed_ms", "solar_irradiance_wm2"):
            idx = rng.choice(n, size=k, replace=False)
            df.loc[idx, col] = df.loc[idx, col] * rng.uniform(3.0, 6.0, size=k)

        # Additive thermistor glitches (physically possible but extreme).
        idx = rng.choice(n, size=k, replace=False)
        df.loc[idx, "temperature_c"] += rng.choice([-1, 1], size=k) * rng.uniform(25, 40, size=k)

        # Physically impossible readings -> caught by the rule-based layer.
        idx = rng.choice(n, size=k, replace=False)
        df.loc[idx, "humidity_pct"] = rng.uniform(105, 140, size=k)
        idx = rng.choice(n, size=k, replace=False)
        df.loc[idx, "pressure_hpa"] = rng.choice([-1, 1], size=k) * rng.uniform(150, 250, size=k) + 1013

        # Classic sentinel error code used by many data loggers.
        s = max(1, int(_SENTINEL_RATE * n))
        for col in ("pm25_ugm3", "no2_ppb"):
            df.loc[rng.choice(n, size=s, replace=False), col] = -999.0
        return df

    def _inject_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """MCAR gaps per column + one contiguous PM10 outage + a few missing labels."""
        n, rng = len(df), self._rng
        for col in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES):
            mask = rng.random(n) < _MISSING_RATES[col]
            df.loc[mask, col] = np.nan

        # Block outage: a sensor offline for a stretch (realistic MNAR-like gap).
        start = int(rng.integers(0, n - 40))
        df.loc[start:start + 24, "pm10_ugm3"] = np.nan

        df.loc[rng.random(n) < _TARGET_MISSING_RATE, TARGET_COL] = np.nan
        return df


def main() -> None:
    """CLI entry point: generate ``raw_data.csv`` and print a sanity report."""
    configure_logging()
    gen = AQIDataGenerator()
    df = gen.generate()
    gen.save(df)

    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nPearson correlation with AQI (dirty data, pairwise-complete):")
    corr = df.drop(columns=[TIMESTAMP_COL, "station_type"]).corr()[TARGET_COL]
    print(corr.drop(TARGET_COL).sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
