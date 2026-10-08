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
