# Implementation status

## Initial inspection
Repository began at 841bf34 with only README.md. No AGENTS.md exists in this checkout.
Working tree was clean, remote matches the requested repository, checkout is detached.
Windows PowerShell; only Python 3.14 registered. A Python 3.12 local environment is required.
Disk free: approximately 184 GB. CIM memory query denied; use psutil memory preflight.

## Acceptance criteria and assumptions
- Synthetic fixtures are development evidence only, never empirical taxi results.
- Validation retains source row identity and reports mutually exclusive rejection reasons.
- Both estimators use identical engineered features, scaler, data and frozen k.
- Upload prediction never fits an estimator, scaler or reducer; all accepted rows export.
- UMAP and silhouette use bounded samples. Cluster IDs have no intrinsic semantic meaning.
- Large runs require memory preflight and explicit recorded failures/skips.
- Raw data, models and intermediate arrays remain ignored; small measured reports are versioned.
- Final verification includes pytest, Ruff, Streamlit startup, git diff check and remote verification.

## Milestones
Bootstrap committed as 733882f. Python 3.12.13 installed in .venv; dependencies installed.
Milestone 1: validation, shared scaler, both models, persistence, upload inference and UI implemented.
11 tests passed (including Streamlit AppTest sections); Ruff lint and format passed.
Synthetic demo generated: 300 accepted training records, saved 2D and 3D reducers.
Unseen 30-row Parquet inference with saved UMAP transform passed; all 30 rows projected.
Streamlit started successfully on localhost:8501 and synthetic overview rendered in browser.
Initial sandbox-only test attempt had 8 passes/2 temp-directory permission errors; rerun with
normal temp access passed. Initial Ruff inherited external settings; explicit project rules added.
No push yet. Real source download completed; empirical processing follows.
Memory snapshot: 6,323,187,712 bytes usable RAM, only 386,805,760 available during parallel imports.

Milestone 1 commit: 6eed215. Expanded suite: 16 tests passed in 61.70s, including both upload
UI paths and download control, no-refit guard, projection sampling and benchmark metric checks.

Milestone 2 executed: official file has 3,475,226 rows; 3,241,580 accepted; 233,646 rejected.
SHA-256 and reasons recorded in reports/provenance.json. k=2 selected from k=2..8 on 20,000 rows
using 2,000-row silhouette (0.473733 at k=2). Both final models trained on 100,000 real rows.
Shared 2D/3D UMAP reducers fit 5,000 sampled rows and saved successfully. Initial UMAP attempt
was blocked by sandbox Windows pipe permissions; rerun with required access succeeded.
Profiles, statistics, selection plots and metadata saved. Model/data binaries remain ignored.

Browser requested background and collapsible panel: added city-grid gradient, dark theme and
visible native sidebar toggle styling. Browser upload chooser automation timed out; upload
page functionality is tested via Streamlit AppTest with actual CSV/Parquet bytes instead.
The preview was stopped temporarily to conserve RAM for real training and benchmarks.

Milestone 2 commit: 4f2ec63. Remote main advanced with AGENTS.md and CODEX_MASTER_PROMPT.md;
both were inspected and merged intact in 3ecacc9. Instructions agree with requested scope.

Milestone 3: all requested sizes and batch sizes evaluated. Completed paired 50k/100k fits
and full 3,241,580-row incremental MiniBatch. Remaining ordinary 250k/500k/1m fits and batch
study could not complete within available RAM; full KMeans explicitly skipped by preflight.
See docs/results.md and raw attempt CSVs. Fixed Windows launcher-only memory monitoring and
isolated preparation to reduce coordinator memory. No resource limits were relaxed.

Integration: 17 tests passed before the final null-time/float32-overflow test was added.
scripts/smoke_saved.py passed all four combinations of both algorithms and CSV/Parquet on
30 real holdout records excluded from training. All records had valid 2D and 3D projections,
stable labels across file formats, preserved source row positions, and full CSV exports.
Features/rules/anomaly threshold are configurable, persisted and reused during inference.

Milestone 3 commit: 159ccb6, pushed to origin/main and verified by matching remote SHA.
Final local suite at that milestone: 18 passed in 38.73s; Ruff lint/format and diff check passed.
GitHub CLI has no API login, so CI status was not verified through gh.
Styled preview restarted. Native sidebar collapse/reopen and grid background verified in browser.

## Added Colab fallback (same project, no restart)
Notebook: notebooks/full_dataset_benchmark_colab.ipynb. Reuses existing preprocessing and workers,
same frozen scaler/features/k/rules/seed and source hash; no UMAP/model refitting needed.
Runs full ordinary KMeans and MiniBatch together plus explicit streaming MiniBatch, with resource
checks and no size reduction. Optional same-runtime smaller-size/batch-size reruns are available.
Exports labeled CSV/JSON, runtime metadata and small plots. Dashboard supports automatic
reports/colab loading or validated CSV import alongside local results. No Colab execution or
results are claimed; notebook compilation and reusable workflow pieces are tested locally.

Colab validation: 21 tests passed in 15.72s using workspace-only temporary fixtures; Ruff lint,
format and git diff check passed. Standard pytest temp-directory creation is denied in this
desktop sandbox, so a temporary pytest plugin supplied ordinary workspace directories. A
third-party temporary-directory cleanup warning remained at interpreter exit; exit status was 0.
UI tests now isolate their working directory, preventing accidental real-artifact loading.
Notebook code cells compile with no stored outputs. Scaler JSON restoration is numerically
identical in tests, and mismatched Colab report fingerprints/environments are rejected.
Real 3D UMAP was visually verified with rotation controls and point hover in the browser.
Automatic approval review initially blocked an elevated command due to account usage limits,
not a safety finding; workspace-only edits and checks continued.

Colab fallback committed and pushed as 9283f9b; origin/main SHA verified exactly.
Approval review subsequently became available. Standard final commands then passed:
`python -m pytest -q` — 21 passed in 20.82s; `ruff check .` — passed;
`ruff format --check .` — 26 files already formatted; `git diff --check` — passed.
Working tree was clean after that push. The Colab CSV import control and existing local
comparison charts were also verified in the live browser. GitHub Actions execution remains
unverified: gh lacks API authentication and the unauthenticated API connection was refused.
Remaining empirical limitation: ordinary 250k–1m/batch-size measurements need more available
RAM, and Colab has not been executed. The notebook provides the reproducible fallback.
