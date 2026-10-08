# AGENTS.md — Scalable Taxi Trip Pattern Mining

## Mission
Build a reproducible, locally runnable DWM mini-project titled **Scalable Taxi Trip Pattern Mining using K-Means and MiniBatch K-Means**. Deadline: 2 days. Develop on Windows 10, VS Code/Codex, **8 GB RAM**. The deliverable is a tested Streamlit dashboard, robust data/ML pipeline, scientifically defensible benchmarks, visualizations, and live inference on uploaded files.

## Repository and workflow
- Repository: `https://github.com/vighneshvijayjadhav-eng/large-scale_clustering_comparison.git`, default branch `main`.
- Inspect existing files, git status, remotes, Python installation, and README before modifying. Preserve user work. Never force-push, reset destructively, rewrite history, or commit secrets.
- Work in short vertical slices with tests, documentation, and commits after each milestone. Push to `origin` when credentials/network/permissions allow; if unavailable, state the blocker and provide commands. Do not claim a push succeeded unless verified.
- Track progress in `docs/IMPLEMENTATION_STATUS.md` with completed items, commands run, actual results, and blockers. Never fabricate benchmarks.
- Keep dependencies minimal and compatible with Python 3.11/3.12. Use `.venv` created on the user's machine; do not modify globally managed Python.

## Non-negotiable product scope
1. Official NYC TLC **January 2025 Yellow Taxi** Parquet data (`yellow_tripdata_2025-01.parquet`), with its provenance recorded. Obtain from official TLC link, not random mirrors. Confirm *actual* raw and valid row counts; ~3.48M is only an estimate.
2. Preprocess correctly: schema validation, invalid/missing-value handling with reported drop reasons, feature engineering, numeric scaling. Candidate features: trip distance, trip duration in minutes, fare amount, average speed, fare per mile, pickup hour, passenger count. Features and exclusion rules must be configurable and documented; do not assume tip, toll, pickup/dropoff IDs should automatically be used for clustering. Preserve UTC/local pickup date interpretation from source as appropriate.
3. Select `k` using elbow + silhouette on a reproducible **bounded representative sample**. Freeze selected `k`, feature list, preprocessing parameters, and seeds across both methods and all benchmarks. Persist rationale.
4. Compare scikit-learn **KMeans** vs **MiniBatchKMeans** on 50k, 100k, 250k, 500k, 1m and **attempt the full clean dataset** if resource checks permit. Both get same selected rows, columns, preprocessing, `k`, and seed. Report runtime, peak process memory if measurable, inertia normalized per sample, sampled silhouette, cluster sizes, and agreement (ARI after accounting for arbitrary label IDs). Use a fixed-size evaluation subset and disclose exactly how it was selected; report skipped/failed stages honestly. CPU and thread settings recorded. Do not claim full-batch KMeans is an out-of-core streaming algorithm.
5. Secondary MiniBatchKMeans batch-size experiment: e.g. 100, 500, 1000, 5000, 10000 on a documented fixed dataset size with same configuration. Make resource-aware.
6. Cluster profiling and human-readable *interpretive* labels backed by actual feature summaries; do not hard-code assumed behavior as established results.
7. Anomaly analysis as **secondary feature** using distance to the nearest centroid in scaled feature space; configurable threshold (e.g. 99th percentile learned from training). Treat anomalies as unusual trips, not proof of fraud.
8. 2D and 3D UMAP + Plotly interactive plots, cluster profile bar/radar charts, anomaly scatter, performance comparisons. Fit UMAP on a bounded representative training sample, save fitted reducers (separate 2D and 3D), and call `transform` on limited new points. Never pass millions of points to Plotly. Clearly disclose sampling and projection caveats; embedding distances need not reflect cluster quality.
9. Streamlit interface: dataset overview, clustering/profiles, comparison, anomaly exploration, upload/predict/export. Upload CSV/Parquet, validate schema, apply **exact saved training preprocessing/scaler/features**, assign clusters using selected frozen model (no retraining), show assignments and highlighted new points projected using saved UMAP reducer, and offer downloadable enriched CSV. Display precise user-facing validation messages for missing columns, malformed timestamps, rejected rows, oversized uploads, missing artifacts.
10. Fully reproducible run commands, README, tests, `.gitignore`, GitHub Actions CI. Code and documentation go to GitHub; raw data, local environment, generated artifacts, caches and large models stay out of Git.

## Execution priorities
**Priority P0 (first):** end-to-end path on small fixture: clean -> features -> fit two models -> persist -> predict upload -> Streamlit renders -> test. This must work before expensive benchmarking.
**P1:** representative data download/prep, choose k, clustering/profiling, charts, staged benchmarks, batch-size experiment.
**P2:** full clean-dataset attempts, memory/runtimes, UX polish, documentation, final validation. Avoid expanding beyond scope.

## Suggested code structure (adapt to existing repo)
```
app.py
src/taxi_clustering/
  config.py
  data.py
  features.py
  train.py
  evaluate.py
  benchmark.py
  profiles.py
  anomalies.py
  visualize.py
  inference.py
  artifacts.py
scripts/
  download_data.py
  prepare_data.py
  train_models.py
  run_benchmarks.py
  build_visualizations.py
tests/
  fixtures/
  test_features.py
  test_inference.py
  test_evaluation.py
  test_artifacts.py
  test_cli_smoke.py
docs/
  IMPLEMENTATION_STATUS.md
  methodology.md
  demo_guide.md
  results.md
.github/workflows/ci.yml
pyproject.toml
.gitignore
README.md
```
Keep architecture straightforward; avoid microservices, databases, React, Docker requirements, orchestration frameworks, excessive abstractions.

## Correctness and memory safety
- Use Polars lazy Parquet scan/projection/filter when applicable; avoid reading all raw data and copying huge DataFrames. If converting to NumPy for sklearn, use memory accounting, `float32` where numerically reasonable, chunked/scanned IO, and `numpy.memmap` where sensible.
- Streaming-friendly preprocessing is not the same as streaming full KMeans training. Attempt full KMeans only when viable; monitor resources and fail gracefully with explicit status and rationale. For full MiniBatchKMeans, `partial_fit` on chunks is acceptable **only for the dedicated full-scale run**; record that it is incremental and keep other comparisons methodologically explicit. No hidden sample downscaling.
- Split between `fit_transform` for training and `transform` for inference. No data leakage via refitting `StandardScaler` or UMAP on uploaded data.
- Keep original input IDs/row indices during cleaning and inference; report rejected rows and reasons rather than silently discard. Handle zero distance/duration, negative charges, extreme outliers, timestamp parse failures, non-finite values, division by zero.
- `KMeans.predict`/`MiniBatchKMeans.predict` always assign a nearest cluster, including odd inputs. Mark unusual ones separately. Cluster numeric IDs across two different models are not intrinsically aligned. Use ARI or an explicit label-matching method; never compare IDs directly.
- Accurate benchmark timing boundaries and memory measurements; distinguish preprocessing, training, evaluation and visual rendering costs. Silhouette on a bounded common sample because full pairwise computation is expensive. Do not pretend minibatch inference labels are directly equivalent to KMeans labels.
- Set seeds; record OS, Python, dependency versions, hardware, parameters, source filename/hash, clean counts, and run status in machine-readable output.
- Large chart data should be sampled; use training embeddings cached for display and bounded uploaded transformations. Keep download exports faithful to all accepted input records even if chart only shows sampled points.
- No external paid APIs or services. Never log credentials or user-provided sensitive data.

## Validation gates
- `python -m pytest` passes on synthetic small fixtures offline, including invalid input and save/load/predict round-trip; tests should not need full NYC data or network.
- `ruff check .` passes; format consistently.
- Streamlit app imports and launches from project root, with graceful onboarding when trained artifacts are absent.
- Smoke test a miniature end-to-end model, then test CSV and Parquet upload paths with known records; verify feature order, stable cluster assignments before/after serialization, row counts, exported columns, and UMAP transform shapes.
- Validate benchmarks' output tables include true sizes, metrics, units, run status, methods, and the settings necessary to reproduce them.
- GitHub Actions runs lint and offline tests. README provides Windows PowerShell installation, training, benchmark and app commands, download instructions and demo script.
- Do not describe unexecuted tests as passed. Commit only working milestones; report all remaining limitations.

## Completion reporting
At each milestone state: files changed, commands run, actual tests/results, commit hash, push status, and next action. Final report must distinguish **implemented**, **tested**, **not run due to hardware**, **optional**, and **known limitations**. Prefer transparent, rigorous outcomes to cosmetic claims.
