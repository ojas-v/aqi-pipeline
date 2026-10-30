# AQI Sentinel — forecasting report

## 1. Problem and success criterion
**Question:** Can the previous 24 hours of observations at a monitoring station predict PM2.5 concentration 24 hours ahead? The primary metric is test RMSE in µg/m³. The predeclared success criterion is at least a 15% RMSE reduction against persistence. This is a criterion to measure, not a claimed result.

## 2. Dataset and provenance
Primary data: [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing), 420,768 hourly observations from 12 stations, March 2013–February 2017. The dataset contains pollutant and meteorological measurements and missing values. License: CC BY 4.0. Citation: Chen, S. (2017), *Beijing Multi-Site Air Quality*, UCI Machine Learning Repository, https://doi.org/10.24432/C5RK5G. The script downloads the archive on first run; raw dataset files are not committed. The synthetic generator remains available for robustness demonstrations, not as primary evidence.

## 3. Data preparation and EDA
The pipeline parses timestamps, sorts by station and time, removes duplicate station-hour records, coerces pollutant fields to numeric, treats PM2.5 outside 0–1000 µg/m³ as missing, reindexes each station to hourly frequency, and creates lagged predictors. Target values are never imputed. The notebook is planned as a separate EDA deliverable; include target distributions, missingness by station and variable, temporal patterns, and pollutant correlations with a written observation under each plot.

## 4. Method
Chronological split by unique timestamps: first 70% train, next 15% validation, final 15% test. Same timestamps cannot cross splits. Because each row's label is observed 24 hours after its forecast-origin timestamp, purge the final 24 hours of training origins before validation and the final 24 hours of validation origins before test. Apply the same 24-hour purge to each training-only TimeSeriesSplit fold. Compare persistence (latest available PM2.5 at forecast origin), mean dummy, Ridge, Random Forest, Extra Trees, and HistGradientBoosting. The current PM2.5 reading is also included as an explicit model feature. Imputation is inside model pipelines. Randomized hyperparameter search uses TimeSeriesSplit on training data only; select by validation RMSE, refit on train plus validation, and evaluate once on test.

## 5. Results
Run `python forecasting.py`. The measured results are written to `artifacts/forecast_metrics.json`; predictions and error summaries are written to CSV files in `artifacts/`. Populate the final presentation from this run. Do not reuse synthetic-nowcast metrics as forecasting results.

## 6. Error analysis
Inspect `error_by_pollution_band.csv`, `error_by_station.csv`, `error_by_hour.csv`, `worst_predictions.csv`, and `permutation_importance.csv`. Discuss high-pollution errors and signed residuals before making claims about operational usefulness.

## 7. Limitations
The dataset covers one city and historical period; transfer to other cities is untested. Missingness may be informative, and unexpected weather or emissions changes cannot be inferred from past readings alone. This prototype is not a regulatory or health decision tool. State measured performance and uncertainty rather than assuming the 15% target was achieved.

## 8. Next steps
Add a notebook with documented EDA insights, seasonal holdouts, signed-bias analysis, prediction intervals, and CI. Keep raw data out of version control and retain dataset attribution.
