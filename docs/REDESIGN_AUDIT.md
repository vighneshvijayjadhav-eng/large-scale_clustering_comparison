# Continuation audit

Baseline: 341b5da; clean working tree. Existing models, source data and measured attempts are preserved.

| Area | Baseline finding | Required work |
|---|---|---|
| Ingestion/validation | Working lazy Parquet, provenance and rejection counts | Preserve; add empty-input coverage |
| Features/inference | Working saved scaler and predict, fixed feature order | Preserve no-refit behavior |
| K selection | Real 2..8 evaluation, but maximum silhouette alone selects 2 | Evaluate 2..10; elbow, size, stability and profiles; persist rationale |
| Model artifacts | Working 100k models and two UMAP reducers | Version/preserve artifacts if K changes; complete anomaly summaries and centroid tables |
| Benchmarks | 50k/100k pairs and full incremental measured; remaining runs resource-limited | Retry safely; no fabricated measurements; archive incompatible experiments |
| Colab | Notebook/runner/import implemented, not executed remotely | Preserve exact frozen preprocessing and environment labeling |
| UI | Prototype, raw metadata, repetitive tall charts | Six pages, compact layouts, violet/teal theme, categorical colors |
| Profiles | Numeric means, descriptive names only in docs | Shared measured descriptions and consistent labels in charts/exports |
| Anomalies | Only sampled records shown | Exact training counts, full training anomaly export, severity filtering |
| Upload | Validated CSV/Parquet and full export working | Cache prediction; distribution, descriptive labels and origin distinction |
| Tests | 21 tests passed previously | Selection, charts, empty/missing artifacts and six-page coverage |

## Selection procedure fixed before new measurements
Evaluate K=2..10 on the same seeded 20k sample, with one scaler and fixed 2k silhouette sample.
Fit a second initialization seed and calculate ARI stability. Save every candidate's cluster sizes
and feature means. For a broad travel-pattern summary, candidates require minimum cluster share
0.5%, silhouette >=0.25, and seed stability ARI >=0.80. These are explicit practical quality
gates, not universal statistical constants. Rare groups remain available through anomalies.
Among eligible candidates select the maximum normalized distance below the endpoint inertia
chord (a reproducible elbow heuristic); ties prefer fewer clusters. If no candidate passes,
stop for analysis instead of silently choosing a default. Profile evidence and gate sensitivity
must accompany the result; the procedure may legitimately retain K=2.

## Completion audit
- Implemented and tested: six dashboard pages; K=2..10 selection; profile/centroid/anomaly artifacts;
  categorical chart helpers; model persistence; CSV/Parquet inference and export; saved UMAP2/3;
  environment-separated benchmark import and Colab workflow validation.
- Empirically completed: 100k final training; 20k K analysis; paired benchmarks at all five sizes;
  full ordinary and streaming MiniBatch; all five batch sizes; real disjoint holdout inference.
- Explicit limit: full ordinary K-Means did not fit locally because preflight rejected its memory
  estimate. Free Colab fallback is implemented but has not been executed remotely.
- Tests: 25 passed; lint/format/diff checks passed. Remote GitHub Actions execution not verified.
