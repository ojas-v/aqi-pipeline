# Smart-City AQI Prediction - End-to-End ML Pipeline

```
aqi_pipeline/
├── main.py              # orchestrator: generate -> ingest -> clean -> train -> evaluate
├── data_generator.py    # Phase 1: physics-based synthetic data (+ faults, NaNs) -> raw_data.csv
├── requirements.txt
└── src/
    ├── config.py        # columns, physical bounds, hyper-parameters (single source of truth)
    ├── ingestion.py     # DataIngestor: load + schema validation + dtype coercion
    ├── cleaning.py      # DataCleaner (stateless), IQRCapper (stateful), DistributionReporter
    ├── features.py      # temporal + interaction features, ColumnTransformer pipeline factory
    ├── modeling.py      # ModelFactory, ModelTrainer (CV), ModelEvaluator, persistence
    ├── exceptions.py    # PipelineError hierarchy
    └── logger.py
```

Run: `pip install -r requirements.txt && python main.py`
Outputs: `raw_data.csv`, `artifacts/aqi_model.joblib`, `artifacts/metrics.json`.

## Design decisions (viva cheat sheet)

| Decision | Why |
|---|---|
| Stateless rules (dedup, drop unlabeled rows, mask impossible values) run **before** the split; IQR fences, medians, scaler, encoder run **inside** the sklearn `Pipeline` | Anything that learns a statistic from data must see training data only, otherwise test information leaks. |
| Impossible values (humidity > 100 %, -999) -> NaN, extreme-but-possible spikes -> IQR cap | Faults are missing information; spikes are real-valued but high-leverage. Different problems, different fixes. |
| Cap, then impute (median) | Median is outlier-robust; capping first keeps the order valid if switched to mean. |
| Rows with missing target are dropped, never imputed | Imputing labels fabricates ground truth. |
| Cross-validate the *whole* pipeline | Preprocessing is re-fitted per fold, so CV scores are leakage-free. |
| Cyclical sin/cos for hour and day-of-year | Hour 23 and 0 are neighbours; raw integers say otherwise. |
| Custom features: T*RH interaction, PM2.5/PM10 ratio, stagnation index | Physically motivated; ratio separates combustion from dust sources. |
| `StandardScaler` included | Not needed by tree ensembles; kept so linear/SVR/NN candidates can be swapped in via `ModelFactory` unchanged. |
| `OneHotEncoder(handle_unknown="ignore")` | An unseen station type at inference must not crash the service. |

## Known limitations (state these before the examiner does)

* **Target is derived from PM2.5/PM10**, so R^2 ~ 0.95 is high *by construction*. This is a nowcast, not a forecast. For genuine forecasting, shift the target to AQI(t+k), add lag/rolling features and use `TimeSeriesSplit`.
* Random `KFold` is used because rows are asynchronous multi-station readings; with strict time series it would leak the future.
* Data is synthetic: the model has learned the generator's physics, not a real city's.
* IQR fences are computed per column; for lower-bounded variables (wind, irradiance) the lower fence can be negative and is then simply inactive.

## Extension ideas
Hyper-parameter search (`RandomizedSearchCV` on the pipeline), `HistGradientBoostingRegressor`, SHAP, drift monitoring on the saved fences.
