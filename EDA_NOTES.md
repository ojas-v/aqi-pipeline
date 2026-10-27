# EDA notes and interpretation checklist

Run `python eda.py` after `python main.py` downloads the dataset. Figures are saved under `artifacts/eda/`.

- **PM2.5 distribution:** particulate data often have a long right tail. Compare mean and median; don't delete high values merely because they look extreme.
- **Missingness:** report missing percentage by variable. Missingness can reflect sensor outages, not random omission; imputation cannot recover every episode.
- **Correlation heatmap:** correlated pollutants may share sources or weather drivers. Correlation is not causation and can be distorted by seasonality and missing values.
- **Diurnal plot:** averages across stations and years can hide station-specific patterns; inspect per-station patterns if time permits.
- **Seasonal plot:** month-level medians summarize a multi-year period but do not isolate long-term trend from weather or policy changes.
- **Station medians:** differing distributions can reflect local sources, site placement, and measurement conditions. A pooled score may conceal station-specific errors.

For the report, place each figure next to one or two evidence-backed observations and explicitly mention the relevant limitation. Do not claim a particular pattern until you have generated and inspected the plot on the actual data.
