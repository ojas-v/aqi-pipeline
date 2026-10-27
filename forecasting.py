"""Real-data 24-hour PM2.5 forecasting workflow using the UCI Beijing dataset."""
from __future__ import annotations
import io, json, zipfile
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "beijing"
ARTIFACTS = ROOT / "artifacts"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
SEED, HORIZON, LAGS = 42, 24, 24

def load_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        print("Downloading official UCI Beijing Multi-Site Air Quality archive...")
        with urlopen(UCI_ZIP, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    (DATA_DIR / Path(name).name).write_bytes(zf.read(name))
        files = sorted(DATA_DIR.glob("*.csv"))
    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        if {"year", "month", "day", "hour", "PM2.5"}.issubset(frame.columns):
            frame["station"] = path.stem
            frames.append(frame)
    if not frames:
        raise RuntimeError("No UCI station CSVs found in data/beijing. See DATASET.md for manual setup.")
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["station", "timestamp"])
    df = df.drop_duplicates(["station", "timestamp"])
    df.loc[~df["PM2.5"].between(0, 1000), "PM2.5"] = np.nan
    return df.reset_index(drop=True)

def make_supervised(raw):
    parts = []
    covars = ["PM10", "SO2", "NO2", "CO", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    for station, group in raw.groupby("station", sort=False):
        g = group.sort_values("timestamp").set_index("timestamp")
        cols = ["PM2.5"] + [c for c in covars if c in g.columns]
        hourly = g[cols].apply(pd.to_numeric, errors="coerce")
        hourly = hourly[~hourly.index.duplicated(keep="last")].asfreq("h")
        hourly["target_pm25_t_plus_24h"] = hourly["PM2.5"].shift(-HORIZON)
        for lag in range(1, LAGS + 1):
            hourly[f"pm25_lag_{lag}h"] = hourly["PM2.5"].shift(lag)
        for col in covars:
            if col in hourly:
                hourly[f"{col.lower()}_lag_1h"] = hourly[col].shift(1)
        hourly["hour_sin"] = np.sin(2 * np.pi * hourly.index.hour / 24)
        hourly["hour_cos"] = np.cos(2 * np.pi * hourly.index.hour / 24)
        hourly["doy_sin"] = np.sin(2 * np.pi * hourly.index.dayofyear / 365.25)
        hourly["doy_cos"] = np.cos(2 * np.pi * hourly.index.dayofyear / 365.25)
        hourly["station"] = station
        parts.append(hourly.reset_index(names="timestamp"))
    data = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["target_pm25_t_plus_24h"])
    return data.sort_values(["timestamp", "station"]).reset_index(drop=True)

def metric_row(name, y, pred):
    return {"model": name, "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "MAE": float(mean_absolute_error(y, pred)), "R2": float(r2_score(y, pred))}

def main():
    ARTIFACTS.mkdir(exist_ok=True)
    raw = load_data()
    print(f"Loaded {len(raw):,} station-hour rows across {raw.station.nunique()} stations.")
    print(f"Coverage: {raw.timestamp.min()} to {raw.timestamp.max()}")
    data = make_supervised(raw)
    times = np.sort(data.timestamp.unique())
    train_cut, val_cut = times[int(.70 * len(times))], times[int(.85 * len(times))]
    train = data[data.timestamp < train_cut]
    val = data[(data.timestamp >= train_cut) & (data.timestamp < val_cut)]
    test = data[data.timestamp >= val_cut]
    target = "target_pm25_t_plus_24h"
    excluded = {"timestamp", target, "PM2.5", "PM10", "SO2", "NO2", "CO", "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
                "year", "month", "day", "hour"}
    features = [c for c in data.columns if c not in excluded]
    numeric = [c for c in features if c != "station"]
    def X(frame):
        return frame[numeric].replace([np.inf, -np.inf], np.nan)
    Xtr, Xv, Xte = X(train), X(val), X(test)
    ytr, yv, yte = train[target], val[target], test[target]
    # Most recent hourly observation available at the forecast origin.
    persistence = test["pm25_lag_1h"].to_numpy()
    mean_model = DummyRegressor(strategy="mean").fit(np.zeros((len(ytr), 1)), ytr)
    results = [metric_row("Persistence", yv, val["pm25_lag_1h"].to_numpy()),
               metric_row("Dummy mean", yv, mean_model.predict(np.zeros((len(yv), 1))))]
    models = {
      "Ridge": Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler()), ("model", Ridge())]),
      "Random Forest": Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
          ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=3, n_jobs=-1, random_state=SEED))]),
      "Extra Trees": Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
          ("model", ExtraTreesRegressor(n_estimators=250, min_samples_leaf=3, n_jobs=-1, random_state=SEED))]),
      "HistGradientBoosting": Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
          ("model", HistGradientBoostingRegressor(random_state=SEED))])
    }
    for name, model in models.items():
        model.fit(Xtr, ytr)
        results.append(metric_row(name, yv, model.predict(Xv)))
    print("\nValidation comparison; test set has not been used for model selection:")
    print(pd.DataFrame(results).sort_values("RMSE").to_string(index=False))
    search = RandomizedSearchCV(
        Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                  ("model", RandomForestRegressor(random_state=SEED, n_jobs=-1))]),
        {"model__n_estimators": [150, 250, 350], "model__max_depth": [None, 12, 20],
         "model__min_samples_leaf": [2, 3, 5, 8], "model__max_features": [0.6, 0.8, 1.0]},
        n_iter=8, scoring="neg_root_mean_squared_error", cv=TimeSeriesSplit(n_splits=4),
        random_state=SEED, n_jobs=-1, error_score="raise")
    search.fit(Xtr, ytr)
    candidates = {**models, "Tuned Random Forest": search.best_estimator_}
    val_rmse = {name: float(np.sqrt(mean_squared_error(yv, m.predict(Xv)))) for name, m in candidates.items()}
    best_name = min(val_rmse, key=val_rmse.get)
    trainval = pd.concat([train, val], ignore_index=True)
    final = clone(candidates[best_name]).fit(X(trainval), trainval[target])
    pred = final.predict(Xte)
    persistence_rmse = float(np.sqrt(mean_squared_error(yte, persistence)))
    test_metrics = metric_row(best_name, yte, pred)
    improvement = 100 * (persistence_rmse - test_metrics["RMSE"]) / persistence_rmse
    print("\nFinal test comparison:")
    print(pd.DataFrame([metric_row("Persistence", yte, persistence), test_metrics]).to_string(index=False))
    print(f"RMSE improvement over persistence: {improvement:.2f}% (criterion: >=15%)")
    errors = test[["timestamp", "station", target]].copy()
    errors["prediction"] = pred
    errors["absolute_error"] = np.abs(errors[target] - pred)
    errors["hour"] = pd.to_datetime(errors.timestamp).dt.hour
    errors["pollution_band"] = pd.cut(errors[target], [-.01, 35, 75, 150, 1000],
                                      labels=["low", "moderate", "high", "very high"])
    errors.to_csv(ARTIFACTS / "test_predictions.csv", index=False)
    errors.groupby("pollution_band", observed=False)["absolute_error"].agg(["count", "mean", "median"]).to_csv(ARTIFACTS / "error_by_pollution_band.csv")
    errors.groupby("station")["absolute_error"].agg(["count", "mean", "median"]).to_csv(ARTIFACTS / "error_by_station.csv")
    errors.groupby("hour")["absolute_error"].agg(["count", "mean", "median"]).to_csv(ARTIFACTS / "error_by_hour.csv")
    errors.nlargest(25, "absolute_error").to_csv(ARTIFACTS / "worst_predictions.csv", index=False)
    importance = permutation_importance(final, Xte, yte, n_repeats=3, random_state=SEED,
        scoring="neg_root_mean_squared_error", n_jobs=-1)
    pd.DataFrame({"feature": numeric, "importance_mean": importance.importances_mean,
      "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False).to_csv(
        ARTIFACTS / "permutation_importance.csv", index=False)
    summary = {"question": "Can past observations predict PM2.5 24 hours ahead at a monitoring station?",
      "target": target, "horizon_hours": HORIZON, "lag_hours": LAGS, "rows_raw": int(len(raw)),
      "rows_supervised": int(len(data)), "stations": int(raw.station.nunique()),
      "time_min": str(raw.timestamp.min()), "time_max": str(raw.timestamp.max()),
      "train_cut": str(train_cut), "validation_cut": str(val_cut), "selected_model": best_name,
      "validation_rmse": val_rmse[best_name], "test_metrics": test_metrics,
      "persistence_test_rmse": persistence_rmse, "improvement_over_persistence_pct": improvement,
      "success_threshold_pct": 15, "success": bool(improvement >= 15),
      "best_random_forest_params": search.best_params_, "dataset": "UCI Beijing Multi-Site Air Quality",
      "license": "CC BY 4.0", "random_state": SEED}
    (ARTIFACTS / "forecast_metrics.json").write_text(json.dumps(summary, indent=2))
    print("Saved forecast_metrics.json and error-analysis CSVs in artifacts/")

if __name__ == "__main__":
    main()
