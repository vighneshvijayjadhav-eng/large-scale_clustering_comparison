# Methodology

## Source and cleaning
Use January 2025 Yellow Taxi Parquet linked by the [official TLC page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
The source is provider-reported; TLC does not guarantee accuracy. Download has three retries,
Parquet verification and atomic replacement. Cleaning records SHA-256, bytes, row counts and
exclusive first-failure counts in reports/provenance.json. File membership defines January;
pickup dates are not forcibly clipped to the month.

Required columns: tpep_pickup_datetime, tpep_dropoff_datetime, trip_distance, fare_amount,
total_amount. CSV timestamps accept space or T separators and fractional seconds; timezone-aware
Parquet timestamps are rejected. Times represent NYC local clock time. Extra columns are ignored.
Generated row_id is the zero-based input record position, not any user-provided identifier.
Rejected rows include this position, parsed input columns and a reason. Original malformed text
should be inspected in the source file using that position.

Default Rules (a frozen dataclass, serialized in every model bundle): duration 1–180 minutes,
distance >0 and <=150 miles, speed <=100 mph, fare >0 and <=1000 dollars, total >0 and <=1500.
Missing/nonfinite amounts, malformed times and nonfinite engineered features are rejected.
Order: timestamp, missing/nonfinite numeric input, duration, distance, charge, speed, feature.
Counts are mutually exclusive; a row can violate additional rules. Thresholds are plausibility
filters, not legal fare rules. Change Rules in a Python training workflow and regenerate artifacts
to study sensitivity. Rejecting free/refunded trips changes the analyzed population.

## Features and cluster selection
Distance, duration, speed = distance / hours, fare per mile, sin(hour*2π/24), cos(hour*2π/24).
Location IDs are excluded because their integer numbering is not a meaningful Euclidean distance.
StandardScaler is fitted once on the final model's bounded training sample. No log transform or
outlier winsorization is applied: unusual but valid trips can strongly influence K-Means.

A deterministic row-index hash selects up to 20,000 records for k=2..8. Both elbow and silhouette
are saved. Selection uses maximum silhouette on 2,000 fixed sampled records. This transparent
rule can prefer coarse clusters; it is not proof of a uniquely correct k. Final models fit a
separate explicitly documented training size (default 100,000). Selection and training overlap;
these are descriptive models, not held-out predictive accuracy estimates.

Both estimators use k, seed=42, n_init=10, max_iter=300 and the same standardized matrix.
MiniBatch defaults to batch_size=1000. IDs are arbitrary; semantic descriptions must refer to
the measured profile for each algorithm independently. ARI compares partitions without matching IDs.

## Projection and anomalies
Shared UMAP 2D/3D reducers fit at most 5,000 training records (n_neighbors=15, fixed seeds,
one UMAP job). Uploaded records use transform only, at most 2,000 plotted points. Every accepted
record gets a cluster, distance and outlier flag; unprojected rows retain blank coordinates.
The 99th percentile of each model's training centroid distances is its fixed global threshold.
These are statistical distance outliers, not fraud, errors or a calibrated risk probability.
UMAP geometry is illustrative; cluster quality is measured in standardized feature space.

## Benchmarks and resources
Benchmark data are a float32 memory-mapped standardized matrix using the frozen model scaler.
A seeded permutation gives identical nested sample prefixes to both algorithms at 50k, 100k,
250k, 500k, 1m and full cleaned size. Final demo model fitting is distinct from benchmark fitting.
Fit time excludes ingestion, scaling, array selection and metric calculations. Peak RSS covers
the whole isolated worker, sampled every 50ms, and can miss instantaneous peaks. It is not a
measurement of estimator-only memory. Single repetitions describe this machine, not confidence
intervals. Two numerical threads limit oversubscription.

Normalized inertia is recomputed as total squared centroid distance / all benchmark rows;
cluster counts also use all rows. Silhouette and ARI use the same fixed 2,000-record subsample
within each size. ARI is absent if the paired KMeans fit did not complete. Batch-size study uses
the same 100k prefix and batch sizes 100/500/1000/5000/10000.

Workers have a 900-second timeout and stop below 512 MiB available RAM. Conventional fits are
skipped when conservative estimated working set exceeds 60% of available RAM. Full streaming
MiniBatch uses a single partial_fit pass in 10,000-row blocks over the same shuffled order;
its optimization schedule differs from ordinary fit and is explicitly flagged. Memmap bounds
storage, but ordinary KMeans still allocates working memory. Failures/skips are data, not zeros.

## Reproducibility and trust
CI uses generated synthetic data only. Models are trusted local joblib files: never load an
untrusted pickle. Upload supports only CSV/Parquet and does not accept model files. UI limits
uploads to 20 MiB and 100,000 rows. Saved feature list and artifact version are checked on load.
