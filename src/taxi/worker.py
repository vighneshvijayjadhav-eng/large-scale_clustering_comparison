"""Isolated resource-monitored benchmark; no UI or UMAP imports."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

SEED = 42


def run(algorithm, rows, k, batch_size, streaming):
    source = np.load("data/benchmark.npy", mmap_mode="r")
    order = np.load("data/benchmark_order.npy", mmap_mode="r")[:rows]
    # Streaming follows the same seeded ordering in bounded batches.
    x = None if streaming else np.ascontiguousarray(source[order])
    model = (
        KMeans(n_clusters=k, random_state=SEED, n_init=10, max_iter=300)
        if algorithm == "KMeans"
        else MiniBatchKMeans(
            n_clusters=k, random_state=SEED, n_init=10, batch_size=batch_size, max_iter=300
        )
    )
    with threadpool_limits(limits=2):
        start = time.perf_counter()
        if streaming:
            for offset in range(0, rows, 10000):
                model.partial_fit(np.ascontiguousarray(source[order[offset : offset + 10000]]))
        else:
            model.fit(x)
        elapsed = time.perf_counter() - start
        counts = np.zeros(k, dtype=np.int64)
        inertia = 0.0
        for offset in range(0, rows, 50000):
            chunk = np.ascontiguousarray(source[order[offset : offset + 50000]])
            labels = model.predict(chunk)
            counts += np.bincount(labels, minlength=k)
            inertia += float(
                np.square(chunk - model.cluster_centers_[labels]).sum(dtype=np.float64)
            )
        indices = np.random.default_rng(SEED).choice(rows, min(rows, 2000), replace=False)
        metric_x = np.ascontiguousarray(source[order[indices]])
        metric_labels = model.predict(metric_x)
        score = (
            float(silhouette_score(metric_x, metric_labels))
            if 1 < len(np.unique(metric_labels)) < len(metric_labels)
            else None
        )
        reference = Path(f"artifacts/labels_{rows}.npy")
        ari = None
        if algorithm == "KMeans":
            np.save(reference, metric_labels)
        elif reference.exists():
            ari = float(adjusted_rand_score(np.load(reference), metric_labels))
    return {
        "status": "ok",
        "fit_seconds": elapsed,
        "normalized_inertia": inertia / rows,
        "silhouette": score,
        "metric_rows": len(indices),
        "ari_vs_kmeans": ari,
        "cluster_sizes": json.dumps(counts.tolist()),
        "iterations": int(getattr(model, "n_iter_", getattr(model, "n_steps_", 0))),
    }


if __name__ == "__main__":
    algorithm, rows, k, batch_size, streaming, output = sys.argv[1:]
    result = run(algorithm, int(rows), int(k), int(batch_size), bool(int(streaming)))
    Path(output).write_text(json.dumps(result), encoding="utf-8")
