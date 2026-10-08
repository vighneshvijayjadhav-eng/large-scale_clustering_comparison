# Why the frozen cluster count is now 3

The original code evaluated K=2..8 and chose the largest sampled silhouette. K=2 was measured,
not hardcoded, but elbow shape, cluster sizes and initialization sensitivity were not part of
the decision. The revision keeps the same six features and cleaning thresholds; it does not
remove inconvenient observations or force an attractive number of colors.

## Actual analysis

20,000 hash-priority sampled accepted trips; selection scaler fitted once; all candidates see
identical standardized features. Seeds 42 and 43, ten initializations each. Silhouette uses the
same 2,000 records for all candidates. The original silhouette implementation chose a different
fixed subset, so its 0.474 at K=2 differs from this run's 0.491.

| K | Inertia / trip | Silhouette | Stability ARI | Smallest share |
|---:|---:|---:|---:|---:|
| 2 | 4.357 | .491 | 1.000 | 12.670% |
| 3 | 3.459 | .336 | 1.000 | 11.585% |
| 4 | 2.749 | .336 | 1.000 | 0.205% |
| 5 | 2.324 | .310 | .990 | 0.200% |
| 6 | 2.077 | .310 | .998 | 0.200% |
| 7 | 1.887 | .312 | .735 | 0.025% |
| 8 | 1.710 | .273 | .722 | 0.025% |
| 9 | 1.571 | .288 | .935 | 0.025% |
| 10 | 1.468 | .292 | .728 | 0.025% |

The declared broad-pattern gates are minimum share 0.5%, silhouette >=.25, stability >=.80.
Only K=2 and K=3 pass. K=3 has the stronger normalized inertia elbow and is selected. The
inertia improvement from 2 to 3 is 20.6%. Its three sample sizes are 9,586 / 8,097 / 2,317.
Two groups have means around 2 miles / 12 minutes but different circular pickup-hour centers;
the third averages 13.1 miles / 36.1 minutes. These are descriptive feature patterns.

At K=4, a new group has only 41 trips, mean distance 0.012 miles, speed 0.075 mph and fare per
mile $1,861. Fare-per-mile is highly skewed; increasing K can spend centroids on its extreme
tail. Rare observations remain in the accepted dataset and in distance-outlier analysis.

## Sensitivity and limits

The 0.5% minimum is a project interpretation choice. Shares from 0.25% through 1% give the
same eligible set and K=3. Relaxing it to 0.1% admits K=4,5,6; the elbow then favors K=5.
Thus K=3 is a defensible broad-pattern resolution, not a unique natural taxonomy. Silhouette
alone favors K=2; an unconstrained elbow favors finer subdivisions. Two initialization seeds
test optimization stability, not stability across months or repeated population samples.

No log transform or winsorization was introduced. Further feature-transform sensitivity could
change the solution and would require a separately versioned experiment. Both final models
are refitted on the same 100k rows with K=3. Separate saved 2D/3D UMAP reducers, thresholds,
profiles, centroids and reference assignments were regenerated; no UMAP coordinates are used
for clustering. Previous K=2 evidence remains archived. Benchmark fingerprints include frozen
K, features, scaler, rules and source identity.
