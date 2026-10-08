# Measured results — current frozen K=3

Official January 2025 NYC TLC Yellow Taxi data: **3,475,226 raw**, **3,241,580 accepted**,
**233,646 rejected**. Exclusive rejection reasons: charges 127,550; distance 64,474;
duration 41,488; speed 134. Source hash and thresholds are in reports/provenance.json.

## Selection and final models

K=2 came from the previous maximum-silhouette-only rule. The K=2..10 review now selects
**K=3** using the declared broad-pattern size/stability gates and normalized inertia elbow.
See [selection evidence and sensitivity](model_selection.md). Selection uses 20,000 records,
silhouette 2,000; final models share 100,000 training records and one scaler. Both saved UMAP
reducers were regenerated on 5,000 reference records. Each model flags exactly 1,000 training
records above its 99th-percentile centroid-distance threshold; these are not fraud labels.

| Model | Cluster | Training trips | Mean miles | Mean minutes | Mean mph |
|---|---:|---:|---:|---:|---:|
| KMeans | 0 | 48,361 | 2.00 | 11.85 | 10.16 |
| KMeans | 1 | 40,273 | 1.83 | 12.32 | 9.19 |
| KMeans | 2 | 11,366 | 13.08 | 35.83 | 23.95 |
| MiniBatchKMeans | 0 | 32,527 | 2.17 | 11.38 | 11.48 |
| MiniBatchKMeans | 1 | 55,937 | 1.76 | 12.41 | 8.68 |
| MiniBatchKMeans | 2 | 11,536 | 12.99 | 35.72 | 23.82 |

K-Means separates short trips with different pickup-time profiles from longer/faster trips.
MiniBatch's short-trip partitions differ; equal numeric IDs are not aligned across models.
Circular pickup-hour means and feature means drive UI descriptions; these do not identify
geographic destinations or passenger intent. Fare-per-mile remains sensitive to tiny distances.

## Completed local benchmarks

Windows, 5.89 GiB usable RAM, two numerical threads, seed 42. Available RAM during the sweep
was about 1.0–1.36 GiB. Peak memory is sampled process-tree RSS, including startup and metrics;
training time excludes evaluation. Single runs do not establish confidence intervals.

| Rows | Algorithm | Fit seconds | Peak RSS MiB | Inertia/trip | Silhouette | ARI vs KMeans |
|---:|---|---:|---:|---:|---:|---:|
| 50,000 | KMeans | 0.3866 | 258.69 | 3.4227 | 0.3266 | — |
| 50,000 | MiniBatchKMeans | 0.0762 | 252.14 | 3.4574 | 0.3303 | 0.8090 |
| 100,000 | KMeans | 0.6449 | 267.85 | 3.4974 | 0.3216 | — |
| 100,000 | MiniBatchKMeans | 0.0978 | 269.68 | 3.5149 | 0.3210 | 0.9537 |
| 250,000 | KMeans | 1.5677 | 263.98 | 3.3798 | 0.3333 | — |
| 250,000 | MiniBatchKMeans | 0.1604 | 264.83 | 3.4080 | 0.3248 | 0.8170 |
| 500,000 | KMeans | 3.0292 | 287.71 | 3.4372 | 0.3233 | — |
| 500,000 | MiniBatchKMeans | 1.1732 | 274.55 | 3.4447 | 0.3249 | 0.7970 |
| 1,000,000 | KMeans | 6.6608 | 330.73 | 3.3903 | 0.3079 | — |
| 1,000,000 | MiniBatchKMeans | 0.4426 | 296.92 | 3.4001 | 0.3075 | 0.9873 |
| 3,241,580 | MiniBatchKMeans | 1.4641 | 422.54 | 3.3922 | 0.3303 | — |
| 3,241,580 | MiniBatchKMeans (one-pass streaming) | 1.2232 | 284.32 | 3.3965 | 0.3348 | — |

Both algorithms completed all five required sizes. At 1M rows MiniBatch fitted in 0.443s vs
6.661s for K-Means, with slightly higher inertia (3.400 vs 3.390) and ARI 0.9873 on the shared
2,000-record evaluation sample. This is one measured run, not a general speed guarantee.

Both full-data MiniBatch paths completed on all 3,241,580 accepted records. The incremental
run uses 325 partial_fit blocks of up to 10,000 rows; its fit timer includes block extraction,
and its optimization schedule differs from ordinary fit. Full ordinary K-Means was evaluated
and **skipped by memory preflight**: estimated 997.94 MiB exceeded 60% of 1,111.70 MiB available.
There is no full K-Means timing or full-size ARI. No smaller run is represented as full.

## Batch-size study

All five runs completed on the same 100,000 records, K=3, fixed scaler and seed.

| Batch size | Fit seconds | Inertia/trip | Silhouette | ARI |
|---:|---:|---:|---:|---:|
| 100 | 0.2051 | 3.5070 | 0.3173 | 0.7665 |
| 500 | 0.0910 | 3.5339 | 0.3192 | 0.8135 |
| 1000 | 0.1247 | 3.5149 | 0.3210 | 0.9537 |
| 5000 | 0.1829 | 3.5383 | 0.3191 | 0.8045 |
| 10000 | 0.2603 | 3.4993 | 0.3218 | 0.9891 |

Batch 500 was fastest in this run, while 10,000 had the best inertia, silhouette and agreement
among tested batches. These observations do not establish a universally optimal batch size.

## Reproducibility and limitations

CSV/JSON record exact counts, configurations, environment and experiment fingerprint. Old K=2
measurements and selection results are preserved in reports/history_k2; old attempt logs remain
historical evidence, not current model results. The current dashboard filters current K and
experiment fingerprint. No Colab execution is claimed; its reproducible notebook and import
validation are available for the full K-Means fallback. Large datasets and model binaries stay
local and ignored. Reproduce using README commands; hardware load can change preflight outcomes.
