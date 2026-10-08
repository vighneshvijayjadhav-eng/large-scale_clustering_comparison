# Master Codex Prompt — Build the Project

You are the primary implementation agent. Build this repository into a complete, tested Data Warehousing and Mining mini-project in **at most two days of focused development**. Follow `AGENTS.md` as binding project instructions. Do not merely draft a plan: execute it and leave runnable code and verifiable results.

## Project details
**Title:** Scalable Taxi Trip Pattern Mining using K-Means and MiniBatch K-Means
**GitHub remote:** https://github.com/vighneshvijayjadhav-eng/large-scale_clustering_comparison.git
**Environment:** Windows 10, VS Code, 8 GB RAM; Python 3.11/3.12 in local `.venv`; Streamlit + Plotly.
**Source:** January 2025 NYC TLC Yellow Taxi Parquet (official NYC TLC trip-record data page). Approximately 3.48M raw trips, verify actual counts after downloading and filtering.

### What must work
A. Preprocess taxi data, derive trip duration, speed, fare-per-mile, pickup hour and suitable numeric features; validate inputs and quantify dropped rows. Use Polars and modest memory.
B. Choose cluster count using elbow/silhouette analysis on a bounded representative sample, justify choice, and freeze it for fair comparisons.
C. Train KMeans and MiniBatchKMeans, interpret clusters, persist models and preprocessing metadata.
D. Benchmark both algorithms using the identical data at **50k / 100k / 250k / 500k / 1m** rows and attempt **full cleaned January 2025** data. Report real training time, memory, normalized inertia, sampled silhouette, cluster sizes and ARI cluster agreement with fixed seeds; record failures/skips as such. Add a MiniBatch batch-size study (100/500/1000/5000/10000) on a fixed manageable size. Do not hide resource limits.
E. Build meaningful cluster profile charts, UMAP 2D + 3D Plotly interactive scatter, anomaly scatter (centroid-distance threshold), and benchmarking plots. Sample bounded points for UMAP/Plotly; save 2D and 3D UMAP reducers for new-file transformation.
F. Build a polished but simple Streamlit UI, covering overview, discovered clusters, algorithm comparison, anomalies, and an **upload -> validate -> preprocess -> model selection -> predict existing cluster -> UMAP project -> highlight new points -> downloadable CSV** flow. CSV and Parquet uploads. **Do not retrain or create new clusters during upload.** Newly uploaded records need compatible columns; clearly show rejected rows and validation errors.
G. Production-minded structure, unit/integration tests, reproducible configurations/artifacts, PowerShell setup guide, GitHub Actions tests/lint, clear results documentation. Commit progressively and push to the provided repository if authenticated; do not commit raw datasets or large binaries.

## Development order (do not jump straight into 3.48M-run)

**Milestone 0 — Inspect and bootstrap (first):**
1. Check `git status`, `git remote -v`, existing files, Python versions, available disk and RAM. Avoid destructive git commands.
2. Read `AGENTS.md`, document assumptions and acceptance criteria in `docs/IMPLEMENTATION_STATUS.md`.
3. Set up `pyproject.toml`, `.gitignore`, project package, scripts, tests, CI, and README for Windows PowerShell.
4. Generate a tiny **synthetic test fixture** solely for development/testing, labeled as synthetic; never report it as empirical taxi results.
5. Commit the bootstrap milestone.

**Milestone 1 — Working vertical slice (highest priority):**
1. Implement deterministic data validation, preprocessing, feature engineering, fitting and saving both models, and a consistent inference pipeline.
2. Write tests for column compatibility, missing values, invalid trips, zero divisions, feature order, save/reload stability, inference preserving record identity, and both CSV/Parquet upload paths.
3. Build a bare Streamlit UI that can load saved fixture models, predict uploaded fixture rows, plot sampled assignments, and export CSV.
4. Run tests/lint and manually smoke test the flow; commit.

**Milestone 2 — Real dataset and analytical outputs:**
1. Add a robust script to download/locate January 2025 Yellow Taxi Parquet from the official site. Do not commit it.
2. Derive proper feature statistics, rejection counts and data provenance. Choose `k` via sampled elbow/silhouette with documented plots. Fit and save final models on an explicitly stated manageable training size; separate this from the benchmark fits.
3. Implement cluster profiles with evidence-based names, percentile centroid-distance anomalies, UMAP 2D/3D and interactive Plotly charts. Show sampling disclaimers. Streamlit should work without holding full taxi data in RAM.
4. Run lightweight tests and smoke demos; commit.

**Milestone 3 — Reproducible benchmarks:**
1. Implement controlled same-data comparison for 50k, 100k, 250k, 500k, 1m, using the selected `k` and reproducible nested samples. Bound metrics/plots to subsamples; record wall time, peak memory when measurable, seeds, feature count, runtime context and failures.
2. Implement MiniBatch batch-size study on a fixed feasible dataset.
3. Generate tables/graphs and write factual interpretations from the actual measurements. Never invent or interpolate missing points.
4. Attempt the full cleaned dataset safely, including a chunked MiniBatch path; for ordinary KMeans clearly record whether fitting the entire dataset was feasible. Full ordinary KMeans may be aborted/skipped based on memory preflight, but the required attempt/evaluation decision must be recorded. Do not OOM the laptop.
5. Commit results *metadata/small charts* only; ignore large intermediate files.

**Milestone 4 — Integration and delivery:**
1. Integrate all sections in Streamlit, make missing-artifact messages useful, test upload flow with previously unseen sample records, and verify 3D plot plus download output.
2. Validate outputs: row counts, same feature order, no refitting on upload, consistent predictions after reload, correct compare metrics, no false anomaly claims.
3. Run `python -m pytest`, `ruff check .`, `ruff format --check .`, and a Streamlit startup smoke test; record actual outcomes.
4. Write `README.md`, `docs/methodology.md`, `docs/results.md`, `docs/demo_guide.md`, `docs/IMPLEMENTATION_STATUS.md`, include screenshots only if actually captured.
5. Check `git status` and `git diff --check`, ensure no secrets/datasets/model binaries are staged, then commit and push to `origin/main` if permitted. Confirm remote status. If pushing is blocked, provide exact manual PowerShell commands.

## Architecture and research integrity requirements
- Use a modest modular Python package, CLI scripts, joblib model saving, JSON metadata, Polars data handling, scikit-learn clustering, UMAP, Plotly, Streamlit, pytest and Ruff. Avoid unnecessary React, Flask, FastAPI, databases, Docker, paid APIs, and heavyweight systems.
- Do not confuse raw dataset size with valid records after cleaning. Do not label arbitrary cluster IDs semantically until real feature profiles support it.
- The results from two models have arbitrary label numbering. Compare with ARI (and other defensible measures) instead of raw label equality. Use same sample/scaler/features and chosen k.
- During upload, apply **the saved feature engineering and fitted scaler**, and use saved `model.predict`. Keep output for **all accepted rows** while charting only sampled points. UMAP on new points uses `transform`, never fit on upload.
- Full-batch KMeans on 3.48M rows may be CPU-intensive and is not streaming. Polars lazy loads help ingestion, but do not eliminate sklearn estimator memory. Do an honest hardware preflight; work with memmap where appropriate. If hardware prevents full KMeans, report that explicit limitation rather than pretending a smaller run is full.
- Data validation must cover malformed time strings, impossible durations, non-positive distances, non-finite features, extreme implausible speeds, invalid charge amounts, missing columns, and schema mismatch; expose configurable thresholds and documented reasons.
- CI must use synthetic fixtures, not network/data downloads. Any dependency pin should work on supported Python/Windows. Prefer one reliable install command.
- Use no Git LFS unless clearly needed and approved. Keep artifacts and data excluded from Git with `.gitignore`.

## Time management
Optimize for a two-day finish. Prioritize first fully functioning local demo over charts polish, and correctness over extra features. If resource pressure is substantial, ensure the 50k–1m experiments and full dataset MiniBatch attempt receive priority; transparently document full conventional KMeans limits. Avoid overengineering; add practical tests rather than unnecessary infrastructure.

## Operating instructions
- **Begin now**, not with clarifying questions unless an actual blocker prevents progress.
- Inspect the repo and system, implement Milestone 0 and Milestone 1 immediately, run their tests, commit, and continue through milestones as time/resources permit.
- Give a brief checkpoint after each milestone: completed functionality, measured result or test output, commit hash, push status, unresolved problems, and next commands.
- Never state a stage passed unless executed. Never claim a GitHub push was successful unless verified.
- Ask before destructive operations, repository history rewrites, or introducing services that cost money.

## Definition of done
A faculty member can launch Streamlit locally, explore learned taxi clusters and benchmarking evidence, inspect interactive 3D UMAP and anomalies, **upload previously unseen CSV/Parquet taxi rows**, choose either model, obtain stable cluster assignments and a visualization without retraining, download predictions, and reproduce the documented tests and experiments. GitHub contains the implementation and reproducibility instructions, without the raw multi-million-row data.

**Start by reading `AGENTS.md` and `README.md`, checking git/Windows environment, then implement the first milestone.**
