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

## Free Google Colab fallback for full benchmarks

[Open the Colab notebook](https://colab.research.google.com/github/vighneshvijayjadhav-eng/large-scale_clustering_comparison/blob/main/notebooks/full_dataset_benchmark_colab.ipynb).
Use a free **CPU** runtime and run cells in order. This supplements the local app; it does not
host Streamlit or replace local inference. It installs isolated Python 3.12, downloads or reuses
the official file, and reuses the **exact local fitted scaler, features, rules, k and seed** from
`reports/frozen_preprocessing.json`. Source hash and clean count are verified. No model pickle
needs to leave the laptop.

Both full estimators run in the same runtime, plus a separately labeled incremental MiniBatch
pass. `ALL_SIZES=True` also repeats smaller sizes and the batch study there. Memory checks,
a 15-minute worker timeout and bounded silhouette remain enforced. Failed full runs are
recorded explicitly; no silent downsampling and no paid resources are required.

Download `colab_results.zip`, then from the local project:

```powershell
Expand-Archive "$HOME\Downloads\colab_results.zip" -DestinationPath .\reports\colab -Force
.\.venv\Scripts\streamlit run app.py
```

**Algorithm Comparison** automatically loads the Colab CSV alongside local results. Alternatively
use **Import Colab benchmark CSV** there. A frozen-experiment fingerprint prevents combining
different configurations. Curves include execution environment; compare algorithms within a
runtime, since cross-machine time reflects hardware differences. CSV/JSON, runtime details,
provenance and HTML plots are exported; datasets/models stay excluded.

Commit only actual small results after execution. **No Colab results have been run or claimed
in this delivery.** Notebook validity, scaler restoration and report import are tested locally.

## Dashboard navigation

1. **Overview**: source/cleaned/training counts, K, measured pattern summaries and model status.
2. **Cluster Discovery**: cluster filter, distinct colors, 2D/3D UMAP tabs, feature bars and centroids.
3. **Algorithm Comparison**: actual runtime/memory/quality, batch sizes, and **Cluster Selection Analysis**.
4. **Anomaly Explorer**: training-population counts, severity/cluster filters and downloadable outliers.
5. **Predict New Trips**: validated CSV/Parquet, saved model prediction, diamond-marked uploaded points and CSV export.
6. **Methodology / Experiment Details**: features, configurations, resources and reproducibility.

The violet/teal dark theme uses compact charts and native collapsible navigation. Technical
metadata is expandable. No training runs during navigation or upload. Cached predictions are
invalidated when the model bundle changes.

**Current frozen K=3** comes from a reproducible K=2..10 elbow, silhouette, size and initialization
stability review. The previous max-silhouette-only K=2 decision and measurements are preserved
in `reports/history_k2/`; they are not mixed with the new K=3 benchmark series.
See [selection rationale and sensitivity](docs/model_selection.md).

To check real CSV/Parquet holdouts and both saved UMAP transformations after training:

```powershell
.\.venv\Scripts\python scripts/smoke_saved.py
```

## Academic report

The submission report is available as [PDF](reports/DWM_Mini_Project_Report.pdf) and [editable Word](reports/DWM_Mini_Project_Report.docx). It contains 34 verified PDF pages, 17 numbered figures, 11 tables, six equations, measured K=3 results and reproducibility appendices. See [report validation](reports/REPORT_VALIDATION.md) for evidence sources and limitations, including three labelled screenshot placeholders and unverified Word pagination. [Report build instructions](reports/source/README.md) reproduce both formats without retraining models.
