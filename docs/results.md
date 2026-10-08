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
Execution in progress. Actual completed measurements, skips and aborts will be recorded in
reports/benchmarks.csv and reports/batch_study.csv. No missing measurements are estimated.
