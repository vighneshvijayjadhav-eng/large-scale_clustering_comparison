# Measured results

These results use the official January 2025 NYC TLC Yellow Taxi file, not the synthetic fixture.
See reports/provenance.json for SHA-256 and thresholds. Raw: **3,475,226**; accepted:
**3,241,580**; rejected: **233,646** (6.72%). Exclusive reasons: charge 127,550;
distance 64,474; duration 41,488; speed 134.

## Cluster selection and final models
20,000 selection rows, silhouette evaluated on 2,000 fixed records, k=2..8. Maximum silhouette
selected **k=2**, score **0.473733**. Larger k values scored 0.2708–0.3371. Inertia decreases
throughout the tested range; there is no claim that a visual elbow uniquely establishes k=2.
See reports/k_selection.csv and both interactive HTML curves.

Final demo models each fit the same **100,000** accepted trips; these are separate from all
benchmark fits. UMAP reducers use a shared **5,000**-row sample. The full raw/cleaned counts
must not be confused with final model training size.

| Model / ID | Evidence-based description | Size | Mean miles | Mean minutes | Mean mph |
|---|---|---:|---:|---:|---:|
| KMeans 0 | Shorter, slower trips | 87,559 | 1.86 | 11.93 | 9.60 |
| KMeans 1 | Longer, faster trips | 12,441 | 12.52 | 34.74 | 23.54 |
| MiniBatch 0 | Shorter, slower trips | 62,570 | 1.56 | 11.39 | 8.55 |
| MiniBatch 1 | Mixed medium/longer trips | 37,430 | 5.92 | 20.41 | 16.00 |

These names summarize measured means only, not locations, destinations or passenger intent.
MiniBatch's split differs substantially from conventional KMeans despite identical input and k;
it should not be described as an equivalent partition. Speed/quality tradeoffs require the
benchmark inertia, silhouette and ARI rather than label equality. Fare-per-mile means are sensitive
to short trips and fixed charges. See reports/profiles.csv for complete values.

## Benchmarks
Recorded on Windows with 5.89 GiB usable RAM, two numerical threads and seed 42. Only roughly
0.4–0.8 GiB was available during attempts. Single-run timing is not a stable speed estimate.
Metrics use the same fixed 2,000-row subset at each size.

| Rows | Method | Fit seconds | Peak MiB | Inertia/row | Silhouette | ARI vs KMeans |
|---:|---|---:|---:|---:|---:|---:|
| 50,000 | KMeans | 0.3154 | 266.98 | 4.3179 | 0.4938 | — |
| 50,000 | MiniBatch | 0.0683 | 265.18 | 4.9060 | 0.2800 | 0.1057 |
| 100,000 | KMeans | 0.5624 | 282.59 | 4.3924 | 0.4712 | — |
| 100,000 | MiniBatch | 0.0771 | 280.84 | 4.9996 | 0.2428 | -0.0128 |
| 3,241,580 | MiniBatch partial_fit, one pass | 1.2574 | 299.07 | 4.9095 | 0.2683 | unavailable |

MiniBatch fits were about 4.6x and 7.3x faster at 50k and 100k, with worse inertia, lower
silhouette and weak agreement. MiniBatch stopped after one effective iteration in these runs.
Full incremental fitting used 325 blocks of up to 10,000 rows. Its fit timer includes block
extraction; ordinary fit timers exclude initial array extraction. These optimization schedules
differ; do not extrapolate full conventional KMeans time. Full streamed cluster sizes were
1,930,906 and 1,310,674, summing to all 3,241,580 cleaned rows.

250k fits aborted at the memory floor; 500k KMeans was skipped and MiniBatch aborted; 1m fits
were skipped at preflight. Full conventional KMeans was explicitly evaluated and skipped:
estimated working set ~998 MiB exceeded 60% of ~507 MiB available. Full ordinary MiniBatch also
failed preflight. All five batch-size runs at 100k were attempted but aborted/skipped under
memory pressure. **No completed batch-size comparison supports an optimal batch size.**
The CLI implements every requested size without substituting smaller datasets.

`reports/benchmarks.csv` retains completed third-attempt observations where the fourth attempt
failed, otherwise the fourth-attempt status; `source_attempt` identifies each row. Raw attempt
CSVs remain in reports. The first attempt's RSS is invalid for process-tree memory because it
monitored only the Windows launcher. Monitoring was corrected to include and terminate children.
Heavy coordinator preparation was isolated; a redundant worker import was removed before the
successful full incremental run. Memory comparisons across development attempts are approximate.
Failed-run wall time is never labeled training time. Future CLI reruns archive existing reports.

Rerun `taxi benchmark --full` after freeing RAM to complete missing sizes and the batch study.
No missing point is fabricated or interpolated. Plotly HTML plots show successful observations
only, with streaming labeled separately. Real saved-model CSV/Parquet inference and 2D/3D
projection passed on 30 disjoint holdout records for both algorithms (reports/inference_smoke.json).
