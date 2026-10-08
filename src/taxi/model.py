"""Trusted local model bundles. Uploads never load user-supplied models."""

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from taxi.data import FEATURES, Rules, matrix, validate

SEED = 42


def distances(model, x, labels):
    return np.linalg.norm(x - model.cluster_centers_[labels], axis=1)


def fit_bundle(
    frame,
    out,
    k=4,
    rules=Rules(),
    source="synthetic",
    with_umap=True,
    features=FEATURES,
    anomaly_percentile=99,
):
    if not features or len(set(features)) != len(features) or not set(features) <= set(FEATURES):
        raise ValueError("Features must be a nonempty unique subset of supported features.")
    if not 0 < anomaly_percentile < 100:
        raise ValueError("Anomaly percentile must be strictly between 0 and 100.")
    good, _, report = validate(frame, rules)
    if good.height < max(k * 3, 15):
        raise ValueError("Too few accepted rows to fit models and projections.")
    scaler = StandardScaler()
    x = scaler.fit_transform(matrix(good, features)).astype("float32")
    bundle = {
        "version": 1,
        "features": list(features),
        "rules": asdict(rules),
        "scaler": scaler,
        "models": {},
        "thresholds": {},
        "reducers": {},
    }
    with threadpool_limits(limits=2):
        for name, model in {
            "KMeans": KMeans(n_clusters=k, random_state=SEED, n_init=10, max_iter=300),
            "MiniBatchKMeans": MiniBatchKMeans(
                n_clusters=k, batch_size=1000, random_state=SEED, n_init=10, max_iter=300
            ),
        }.items():
            labels = model.fit_predict(x)
            bundle["models"][name] = model
            bundle["thresholds"][name] = float(
                np.quantile(distances(model, x, labels), anomaly_percentile / 100)
            )
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(x), min(5000, len(x)), replace=False)
    if with_umap:
        from umap import UMAP

        for dim in [2, 3]:
            reducer = UMAP(
                n_components=dim,
                n_neighbors=min(15, len(idx) - 1),
                random_state=SEED,
                transform_seed=SEED,
                n_jobs=1,
            )
            reducer.fit(x[idx])
            bundle["reducers"][dim] = reducer
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "bundle.joblib")
    joblib.dump(
        {"scaler": scaler, "k": k, "features": list(features)},
        out / "benchmark_preprocessing.joblib",
    )
    metadata = {
        "source": source,
        "seed": SEED,
        "k": k,
        "features": list(features),
        "training": report,
        "projection_rows": len(idx),
        "anomaly_percentile": anomaly_percentile,
        "rules": asdict(rules),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    profiles = []
    for name, model in bundle["models"].items():
        labeled = good.with_columns(pl.Series("cluster", model.predict(x)))
        profile = (
            labeled.group_by("cluster")
            .agg(
                pl.len().alias("size"),
                *[pl.col(c).mean() for c in FEATURES + ["pickup_hour", "fare_amount"]],
            )
            .sort("cluster")
            .with_columns(pl.lit(name).alias("algorithm"))
        )
        profiles.append(profile)
    pl.concat(profiles).write_csv(out / "profiles.csv")
    sample = good[idx]
    for name in bundle["models"]:
        predictions, _, _ = predict(bundle, sample, name, chart_limit=5000)
        predictions.write_parquet(out / f"sample_{name}.parquet")
    return bundle, metadata


def load_bundle(path):
    bundle = joblib.load(Path(path) / "bundle.joblib")
    features = bundle.get("features", [])
    if (
        bundle.get("version") != 1
        or not features
        or not set(features) <= set(FEATURES)
        or len(features) != bundle["scaler"].n_features_in_
    ):
        raise ValueError("Incompatible artifact schema. Rebuild artifacts with this version.")
    return bundle


def predict(bundle, frame, algorithm, chart_limit=2000):
    good, bad, report = validate(frame, Rules(**bundle["rules"]))
    if algorithm not in bundle["models"]:
        raise ValueError("Unknown algorithm")
    if not good.height:
        return good, bad, report
    x = bundle["scaler"].transform(matrix(good, bundle["features"])).astype("float32")
    model = bundle["models"][algorithm]
    with threadpool_limits(limits=2):
        labels = model.predict(x)
    distance = distances(model, x, labels)
    good = good.with_columns(
        pl.Series("cluster", labels),
        pl.Series("centroid_distance", distance),
        pl.Series("distance_outlier", distance > bundle["thresholds"][algorithm]),
    )
    idx = np.sort(
        np.random.default_rng(SEED).choice(len(x), min(chart_limit, len(x)), replace=False)
    )
    projected = {"row_id": good["row_id"][idx]}
    for dim, reducer in bundle["reducers"].items():
        coordinates = reducer.transform(x[idx])
        for axis in range(dim):
            projected[f"umap{dim}_{axis + 1}"] = coordinates[:, axis]
    good = good.join(pl.DataFrame(projected), on="row_id", how="left", maintain_order="left")
    return good, bad, report
