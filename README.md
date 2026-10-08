# Scalable Taxi Trip Pattern Mining

K-Means and MiniBatch K-Means on January 2025 NYC Yellow Taxi trips.

## Windows PowerShell setup

Install Python 3.11 or 3.12, then from the repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\ruff check .
.\.venv\Scripts\ruff format --check .
```

The synthetic fixture is exclusively for testing. Real results require the official dataset.
Data and model artifacts are deliberately excluded from Git.

## Run the demo

```powershell
.\.venv\Scripts\taxi demo
.\.venv\Scripts\streamlit run app.py
```

Enter `artifacts/demo` in the sidebar. Upload `data/synthetic_unseen.csv` or
`data/synthetic_unseen.parquet`, choose a saved model and download the predictions.
All accepted rows are predicted; only a bounded sample is projected. No upload refitting.

## Reproduce empirical results

```powershell
.\.venv\Scripts\taxi download
.\.venv\Scripts\taxi clean
.\.venv\Scripts\taxi train data/clean.parquet --rows 100000
.\.venv\Scripts\taxi benchmark --full
.\.venv\Scripts\streamlit run app.py
```

Use `artifacts/real` for the real-data UI. Close memory-heavy applications before experiments.
Benchmark workers enforce memory limits and a 15-minute timeout each. Full runs may be skipped
or aborted; reports preserve those outcomes. Allow additional time for initial UMAP compilation.
Use the same dependency versions when reloading local models; regenerate after upgrades.

`config/default.json` exposes the feature subset/order, cleaning thresholds and anomaly
percentile. Pass `--config config/default.json` to both `taxi clean` and `taxi train`.
Changing a configuration requires regenerating clean data, models and benchmarks together.
Previous benchmark reports are archived automatically before reruns.

## Repository guide

- `src/taxi/data.py`: lazy validation and feature engineering.
- `src/taxi/model.py`: shared training, persistence, inference and UMAP.
- `src/taxi/experiment.py`, `worker.py`: source, k selection and controlled benchmarks.
- `app.py`: Streamlit overview, profiles, comparison, outliers and uploads.
- `reports/`: small empirical tables and Plotly HTML (CDN needed to open standalone HTML).
- `data/`, `artifacts/`: ignored local datasets, memmaps and model bundles.
- [Methodology](docs/methodology.md), [results](docs/results.md),
  [demo guide](docs/demo_guide.md), [implementation status](docs/IMPLEMENTATION_STATUS.md).

Upload columns: `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `trip_distance`,
`fare_amount`, `total_amount`. CSV timestamps use `YYYY-MM-DD HH:MM:SS` (or T separator).
Parquet datetime columns are supported. Maximum upload: 20 MB and 100,000 rows.
`row_id` is the original zero-based record position. Invalid records are separately downloadable.

Official source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
