"""Reproducible elbow selection with explicit cluster quality gates."""

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from taxi.data import FEATURES, matrix


def choose_k(scores):
    table = scores.sort("k")
    ks = table["k"].to_numpy()
    inertia = table["normalized_inertia"].to_numpy()
    if len(ks) < 3 or not np.isfinite(inertia).all() or inertia[0] <= inertia[-1]:
        raise ValueError("Elbow analysis needs at least three decreasing, finite inertia values")
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (inertia - inertia[-1]) / (inertia[0] - inertia[-1])
    table = table.with_columns(pl.Series("elbow_strength", 1 - x - y)).with_columns(
        (
            (pl.col("min_cluster_share") >= 0.005)
            & (pl.col("silhouette") >= 0.25)
            & (pl.col("stability_ari") >= 0.8)
        ).alias("eligible")
    )
    eligible = table.filter(pl.col("eligible")).sort(
        ["elbow_strength", "k"], descending=[True, False]
    )
    if eligible.is_empty():
        raise ValueError("No K passes the declared quality gates; inspect candidate profiles.")
    return int(eligible["k"][0]), table


def analyze(frame, features=FEATURES, out="reports"):
    if frame.height < 30:
        raise ValueError("K=2..10 analysis requires at least 30 accepted rows")
    x = StandardScaler().fit_transform(matrix(frame, features)).astype("float32")
    metric_idx = np.random.default_rng(42).choice(len(x), min(2000, len(x)), replace=False)
    rows, profiles = [], []
    with threadpool_limits(limits=2):
        for k in range(2, 11):
            model = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300).fit(x)
            alternative = KMeans(n_clusters=k, n_init=10, random_state=43, max_iter=300).fit(x)
            sizes = np.bincount(model.labels_, minlength=k)
            row = {
                "k": k,
                "rows": len(x),
                "normalized_inertia": float(model.inertia_ / len(x)),
                "silhouette": float(silhouette_score(x[metric_idx], model.labels_[metric_idx])),
                "stability_ari": float(adjusted_rand_score(model.labels_, alternative.labels_)),
                "min_cluster_share": float(sizes.min() / len(x)),
                "max_cluster_share": float(sizes.max() / len(x)),
                "cluster_sizes": json.dumps(sizes.tolist()),
            }
            rows.append(row)
            profiles.append(
                frame.with_columns(pl.Series("cluster", model.labels_))
                .group_by("cluster")
                .agg(pl.len().alias("size"), *[pl.col(c).mean() for c in FEATURES])
                .with_columns(pl.lit(k).alias("k"))
            )
            print(row, flush=True)
    chosen, table = choose_k(pl.DataFrame(rows))
    winner = table.filter(pl.col("k") == chosen).row(0, named=True)
    silhouette_best = table.sort("silhouette", descending=True)["k"][0]
    eligible = table.filter(pl.col("eligible"))["k"].to_list()
    rationale = f"K={chosen} maximizes the normalized elbow strength among eligible K values {eligible}. Eligibility requires every cluster >=0.5% of the sample, silhouette >=0.25 and initialization stability ARI >=0.80. K={silhouette_best} has the highest silhouette; silhouette alone does not determine this choice. Selected silhouette {winner['silhouette']:.3f}, smallest cluster {winner['min_cluster_share']:.2%}, stability ARI {winner['stability_ari']:.3f}."
    selection = {
        "k": chosen,
        "rule": "Quality-gated normalized inertia elbow",
        "rationale": rationale,
        "selection_rows": len(x),
        "silhouette_rows": len(metric_idx),
        "seed": 42,
        "stability_seed": 43,
        "eligible_k": eligible,
        "gates": {"minimum_share": 0.005, "minimum_silhouette": 0.25, "minimum_stability_ari": 0.8},
        "caveat": "A practical broad-pattern resolution, not a proven unique optimum. Means and rare-cluster sensitivity are reported for review.",
    }
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    table.write_csv(path / "k_selection.csv")
    pl.concat(profiles).sort(["k", "cluster"]).write_csv(path / "k_profiles.csv")
    (path / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return selection
