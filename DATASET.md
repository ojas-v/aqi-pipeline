# Dataset provenance

- **Name:** Beijing Multi-Site Air Quality
- **Official source:** https://archive.ics.uci.edu/dataset/501/beijing
- **Published size:** 420,768 hourly observations from 12 monitoring stations
- **Coverage:** 2013-03-01 to 2017-02-28
- **License:** CC BY 4.0
- **Citation:** Chen, S. (2017). *Beijing Multi-Site Air Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G

The forecasting script downloads the official archive into `data/beijing/` on first run. Raw dataset files are intentionally not committed. For offline use, extract station CSVs into that folder. Synthetic data generation remains available via `python data_generator.py` for robustness tests.
