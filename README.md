# AQI Sentinel

### Real-world air-quality forecasting | Reproducible ML pipeline

The primary experiment now forecasts **PM2.5 concentration 24 hours ahead** using the UCI Beijing Multi-Site Air Quality dataset. The original synthetic AQI pipeline is retained as an optional robustness demonstration, not as the evidence for forecasting performance.

## Research question and success criterion

**Can the previous 24 hours of station observations predict PM2.5 24 hours ahead?** Success is defined in advance as at least a 15% reduction in held-out test RMSE relative to persistence (the most recent available PM2.5 reading). The code reports whether the criterion is met; it does not assume success.

## Dataset

- **Source:** [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing)
- **Published size:** 420,768 hourly observations across 12 monitoring stations
- **Coverage:** 1 March 2013 – 28 February 2017
- **License:** CC BY 4.0
- **Citation:** Chen, S. (2017). *Beijing Multi-Site Air Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G

On first run, `forecasting.py` downloads the official archive to `data/beijing/`. The data files are not committed to Git. See [DATASET.md](DATASET.md) for provenance and offline setup.

## Run it

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

The first run needs internet access to fetch the official dataset. To run the original synthetic demonstration instead:

```bash
python main.py --synthetic
```

## Method

- Construct station-wise hourly data and past-only lag features, including the current PM2.5 reading available at forecast origin; the target is PM2.5 at t+24 hours.
- Keep chronological periods separate: 70% training, 15% validation, 15% test, cut by unique timestamps so stations at the same hour cannot cross partitions. Purge 24 hours at each boundary because each label is measured 24 hours after its forecast origin; the training labels therefore end before the next partition begins. Apply the same 24-hour purge to each time-series cross-validation fold.
- Compare persistence, mean dummy, Ridge, Random Forest, Extra Trees, and HistGradientBoosting.
- Fit imputation inside model pipelines. Use TimeSeriesSplit and RandomizedSearchCV on training data only.
- Select by validation RMSE, refit on train + validation, and evaluate once on the final test period.
- Report RMSE, MAE, R², improvement over persistence, and error analysis by pollution band, station, and hour.

## Outputs

The forecasting run writes to the ignored `artifacts/` folder:

| File | Contents |
|---|---|
| `forecast_metrics.json` | Dataset coverage, split cutoffs, selected model, test metrics, persistence comparison and success criterion |
| `test_predictions.csv` | Held-out predictions and absolute errors |
| `error_by_pollution_band.csv` | Error by PM2.5 band |
| `error_by_station.csv` | Error by monitoring station |
| `error_by_hour.csv` | Error by hour of day |
| `worst_predictions.csv` | 25 largest absolute errors |
| `permutation_importance.csv` | Permutation importance on held-out features |

## Tests

```bash
pytest -q
```

## Existing synthetic pipeline

`data_generator.py` and the modules under `src/` are retained. Use `python main.py --synthetic` to demonstrate injected missing values, sensor faults, rule-based cleaning, IQR capping, and cross-validated synthetic AQI nowcasting. Synthetic metrics must not be presented as evidence of real-world forecast performance.

## Limitations

The primary dataset covers Beijing and a historical period; performance elsewhere is unverified. Missingness can be informative, and weather or emissions changes can break temporal patterns. A successful benchmark does not establish regulatory or health-decision suitability. See [REPORT.md](REPORT.md) for report structure and interpretation guidance.

## Team

- Ojas Vaidya — Electronics and Communication Engineering
- Vivek Raj — Electronics and Communication Engineering
- Sarang Bobade — Electronics and Communication Engineering

## Exploratory data analysis

After the dataset has been downloaded, run `python eda.py` to generate target distribution, missingness, correlation, diurnal, seasonal, and station comparison plots under `artifacts/eda/`. Use [EDA_NOTES.md](EDA_NOTES.md) to frame interpretations and limitations. The figures are generated from the downloaded real dataset; no visual insight is claimed until you inspect the outputs.
