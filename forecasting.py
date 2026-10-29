
"""Real-data 24-hour PM2.5 forecasting workflow using the UCI Beijing dataset."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "beijing"
ARTIFACTS = ROOT / "artifacts"
UCI_ZIP = (
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing+multi+site+air+quality+data.zip"
)

SEED, HORIZON, LAGS = 42, 24, 24


def load_data():
    """Load and combine the 12 station-level UCI CSV files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        print("Downloading official UCI Beijing Multi-Site Air Quality archive...")
        with urlopen(UCI_ZIP, timeout=60) as response:
            payload = response.read()

        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    destination = DATA_DIR / Path(name).name
                    destination.write_bytes(zf.read(name))

        files = sorted(DATA_DIR.glob("*.csv"))

    frames = []

    for path in files:
        try:
            frame = pd.read_csv(path)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

        required = {"year", "month", "day", "hour", "PM2.5"}
        if required.issubset(frame.columns):
            frame["station"] = path.stem
            frames.append(frame)

    if not frames:
        raise RuntimeError(
            "No UCI station CSVs found in data/beijing. "
            "See DATASET.md for manual setup."
        )

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]],
        errors="coerce",
    )
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["station", "timestamp"])
    df = df.drop_duplicates(["station", "timestamp"])

    df.loc[~df["PM2.5"].between(0, 1000), "PM2.5"] = np.nan

    return df.reset_index(drop=True)


def make_supervised(raw):
    """Create time-aware features and a target 24 hours into the future."""
    parts = []

    covars = [
        "PM10", "SO2", "NO2", "CO", "TEMP",
        "PRES", "DEWP", "RAIN", "WSPM",
    ]

    for station, group in raw.groupby("station", sort=False):
        g = group.sort_values("timestamp").set_index("timestamp")

        cols = ["PM2.5"] + [c for c in covars if c in g.columns]
        hourly = g[cols].apply(pd.to_numeric, errors="coerce")
        hourly = hourly[~hourly.index.duplicated(keep="last")]
        hourly = hourly.asfreq("h")

        hourly["target_pm25_t_plus_24h"] = (
            hourly["PM2.5"].shift(-HORIZON)
        )

        for lag in range(1, LAGS + 1):
            hourly[f"pm25_lag_{lag}h"] = hourly["PM2.5"].shift(lag)

        for col in covars:
            if col in hourly.columns:
                hourly[f"{col.lower()}_lag_1h"] = hourly[col].shift(1)

        hourly["hour_sin"] = np.sin(
            2 * np.pi * hourly.index.hour / 24
        )
        hourly["hour_cos"] = np.cos(
            2 * np.pi * hourly.index.hour / 24
        )
        hourly["doy_sin"] = np.sin(
            2 * np.pi * hourly.index.dayofyear / 365.25
        )
        hourly["doy_cos"] = np.cos(
            2 * np.pi * hourly.index.dayofyear / 365.25
        )

        hourly["station"] = station
        parts.append(hourly.reset_index(names="timestamp"))

    data = (
        pd.concat(parts, ignore_index=True)
        .replace([np.inf, -np.inf], np.nan)
    )

    # A supervised row must have a known future target.
    data = data.dropna(subset=["target_pm25_t_plus_24h"])

    return data.sort_values(
        ["timestamp", "station"]
    ).reset_index(drop=True)


def metric_row(name, y, pred):
    """Calculate metrics using only finite target/prediction pairs."""
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)

    if len(y) != len(pred):
        raise ValueError(
            f"{name}: target and prediction lengths differ "
            f"({len(y)} vs {len(pred)})."
        )

    valid = np.isfinite(y) & np.isfinite(pred)
    y = y[valid]
    pred = pred[valid]

    if len(y) == 0:
        raise ValueError(f"{name}: no valid rows for evaluation.")

    return {
        "model": name,
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
        "n_evaluated": int(len(y)),
    }


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    raw = load_data()

    print(
        f"Loaded {len(raw):,} station-hour rows "
        f"across {raw.station.nunique()} stations."
    )
    print(f"Coverage: {raw.timestamp.min()} to {raw.timestamp.max()}")

    data = make_supervised(raw)

    # Split by timestamp so observations from the same hour across
    # different stations cannot fall into different data partitions.
    times = np.sort(data["timestamp"].unique())
    train_cut = times[int(0.70 * len(times))]
    val_cut = times[int(0.85 * len(times))]

    train = data[data.timestamp < train_cut].copy()
    val = data[
        (data.timestamp >= train_cut)
        & (data.timestamp < val_cut)
    ].copy()
    test = data[data.timestamp >= val_cut].copy()

    target = "target_pm25_t_plus_24h"

    excluded = {
        "timestamp", target, "PM2.5", "PM10", "SO2", "NO2",
        "CO", "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
        "year", "month", "day", "hour",
    }

    features = [c for c in data.columns if c not in excluded]
    numeric = [c for c in features if c != "station"]
    station_columns = sorted(data["station"].dropna().unique())

    def X(frame):
        numeric_values = frame[numeric].replace(
            [np.inf, -np.inf], np.nan
        )

        station_values = pd.get_dummies(
            frame["station"],
            prefix="station",
            dtype=float,
        )

        station_values = station_values.reindex(
            columns=[f"station_{s}" for s in station_columns],
            fill_value=0.0,
        )
        station_values.index = numeric_values.index

        return pd.concat([numeric_values, station_values], axis=1)

    Xtr, Xv, Xte = X(train), X(val), X(test)
    ytr, yv, yte = train[target], val[target], test[target]

    # Persistence baseline:
    # use PM2.5 observed at the forecast origin to predict its value
    # 24 hours later. Missing current observations cannot be scored.
    val_persistence = val["PM2.5"].to_numpy(dtype=float)
    test_persistence = test["PM2.5"].to_numpy(dtype=float)

    # Compare every model on the same rows, including persistence.
    val_mask = (
        np.isfinite(yv.to_numpy(dtype=float))
        & np.isfinite(val_persistence)
    )
    test_mask = (
        np.isfinite(yte.to_numpy(dtype=float))
        & np.isfinite(test_persistence)
    )

    if not val_mask.any():
        raise ValueError(
            "No valid validation rows for persistence comparison."
        )
    if not test_mask.any():
        raise ValueError(
            "No valid test rows for persistence comparison."
        )

    print(
        f"Validation rows: {len(val)}; "
        f"valid comparison rows: {int(val_mask.sum())}; "
        f"missing persistence values: "
        f"{int((~np.isfinite(val_persistence)).sum())}"
    )
    print(
        f"Test rows: {len(test)}; "
        f"valid comparison rows: {int(test_mask.sum())}; "
        f"missing persistence values: "
        f"{int((~np.isfinite(test_persistence)).sum())}"
    )

    yv_eval = yv.to_numpy(dtype=float)[val_mask]
    yte_eval = yte.to_numpy(dtype=float)[test_mask]

    mean_model = DummyRegressor(strategy="mean")
    mean_model.fit(np.zeros((len(ytr), 1)), ytr)

    results = [
        metric_row(
            "Persistence",
            yv_eval,
            val_persistence[val_mask],
        ),
        metric_row(
            "Dummy mean",
            yv_eval,
            mean_model.predict(
                np.zeros((len(yv_eval), 1))
            ),
        ),
    ]

    models = {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(
                strategy="median", add_indicator=True
            )),
            ("scale", StandardScaler()),
            ("model", Ridge()),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(
                strategy="median", add_indicator=True
            )),
            ("model", RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=SEED,
            )),
        ]),
        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(
                strategy="median", add_indicator=True
            )),
            ("model", ExtraTreesRegressor(
                n_estimators=250,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=SEED,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(
                strategy="median", add_indicator=True
            )),
            ("model", HistGradientBoostingRegressor(
                random_state=SEED
            )),
        ]),
    }

    for name, model in models.items():
        model.fit(Xtr, ytr)
        val_pred = model.predict(Xv.iloc[np.flatnonzero(val_mask)])
        results.append(metric_row(name, yv_eval, val_pred))

    print(
        "\nValidation comparison; test set has not been used "
        "for model selection:"
    )
    print(
        pd.DataFrame(results)
        .sort_values("RMSE")
        .to_string(index=False)
    )

    # Build cross-validation folds from unique timestamps.
    train_times = np.sort(train["timestamp"].unique())
    timestamp_cv = []

    for tr_time_idx, va_time_idx in TimeSeriesSplit(
        n_splits=4
    ).split(train_times):
        tr_times = train_times[tr_time_idx]
        va_times = train_times[va_time_idx]

        tr_idx = np.flatnonzero(
            train["timestamp"].isin(tr_times).to_numpy()
        )
        va_idx = np.flatnonzero(
            train["timestamp"].isin(va_times).to_numpy()
        )

        timestamp_cv.append((tr_idx, va_idx))

    search = RandomizedSearchCV(
        Pipeline([
            ("imputer", SimpleImputer(
                strategy="median", add_indicator=True
            )),
            ("model", RandomForestRegressor(
                random_state=SEED,
                n_jobs=-1,
            )),
        ]),
        {
            "model__n_estimators": [150, 250, 350],
            "model__max_depth": [None, 12, 20],
            "model__min_samples_leaf": [2, 3, 5, 8],
            "model__max_features": [0.6, 0.8, 1.0],
        },
        n_iter=8,
        scoring="neg_root_mean_squared_error",
        cv=timestamp_cv,
        random_state=SEED,
        n_jobs=-1,
        error_score="raise",
    )

    search.fit(Xtr, ytr)

    candidates = {**models, "Tuned Random Forest": search.best_estimator_}

    val_rmse = {
        name: float(
            np.sqrt(
                mean_squared_error(
                    yv_eval,
                    model.predict(
                        Xv.iloc[np.flatnonzero(val_mask)]
                    ),
                )
            )
        )
        for name, model in candidates.items()
    }

    best_name = min(val_rmse, key=val_rmse.get)

    # Refit the selected model using training + validation data.
    trainval = pd.concat([train, val], ignore_index=True)
    final = clone(candidates[best_name]).fit(
        X(trainval), trainval[target]
    )

    Xte_eval = Xte.iloc[np.flatnonzero(test_mask)]
    pred = final.predict(Xte_eval)

    persistence_eval = test_persistence[test_mask]
    persistence_rmse = float(
        np.sqrt(mean_squared_error(yte_eval, persistence_eval))
    )

    test_metrics = metric_row(best_name, yte_eval, pred)
    improvement = (
        100 * (persistence_rmse - test_metrics["RMSE"])
        / persistence_rmse
        if persistence_rmse > 0
        else 0.0
    )

    print("\nFinal test comparison:")
    print(
        pd.DataFrame([
            metric_row("Persistence", yte_eval, persistence_eval),
            test_metrics,
        ]).to_string(index=False)
    )
    print(
        f"RMSE improvement over persistence: {improvement:.2f}% "
        "(criterion: >=15%)"
    )

    # Save error analysis only for the same valid test rows used above.
    errors = test.loc[test_mask, [
        "timestamp", "station", target
    ]].copy()

    errors["prediction"] = pred
    errors["absolute_error"] = np.abs(
        errors[target].to_numpy(dtype=float) - pred
    )
    errors["hour"] = pd.to_datetime(errors["timestamp"]).dt.hour

    errors["pollution_band"] = pd.cut(
        errors[target],
        [-0.01, 35, 75, 150, 1000],
        labels=["low", "moderate", "high", "very high"],
    )

    errors.to_csv(
        ARTIFACTS / "test_predictions.csv", index=False
    )

    errors.groupby(
        "pollution_band", observed=False
    )["absolute_error"].agg(
        ["count", "mean", "median"]
    ).to_csv(ARTIFACTS / "error_by_pollution_band.csv")

    errors.groupby("station")["absolute_error"].agg(
        ["count", "mean", "median"]
    ).to_csv(ARTIFACTS / "error_by_station.csv")

    errors.groupby("hour")["absolute_error"].agg(
        ["count", "mean", "median"]
    ).to_csv(ARTIFACTS / "error_by_hour.csv")

    errors.nlargest(
        25, "absolute_error"
    ).to_csv(ARTIFACTS / "worst_predictions.csv", index=False)

    importance = permutation_importance(
        final,
        Xte_eval,
        yte_eval,
        n_repeats=3,
        random_state=SEED,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    pd.DataFrame({
        "feature": list(Xte_eval.columns),
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values(
        "importance_mean", ascending=False
    ).to_csv(ARTIFACTS / "permutation_importance.csv", index=False)

    summary = {
        "question": (
            "Can past observations predict PM2.5 24 hours ahead "
            "at a monitoring station?"
        ),
        "target": target,
        "horizon_hours": HORIZON,
        "lag_hours": LAGS,
        "rows_raw": int(len(raw)),
        "rows_supervised": int(len(data)),
        "stations": int(raw.station.nunique()),
        "time_min": str(raw.timestamp.min()),
        "time_max": str(raw.timestamp.max()),
        "train_cut": str(train_cut),
        "validation_cut": str(val_cut),
        "selected_model": best_name,
        "validation_rmse": val_rmse[best_name],
        "test_metrics": test_metrics,
        "persistence_test_rmse": persistence_rmse,
        "improvement_over_persistence_pct": improvement,
        "success_threshold_pct": 15,
        "success": bool(improvement >= 15),
        "best_random_forest_params": search.best_params_,
        "dataset": "UCI Beijing Multi-Site Air Quality",
        "license": "CC BY 4.0",
        "random_state": SEED,
    }

    (
        ARTIFACTS / "forecast_metrics.json"
    ).write_text(json.dumps(summary, indent=2))

    print("\nSaved forecast_metrics.json and error-analysis CSVs in artifacts/")


if __name__ == "__main__":
    main()
