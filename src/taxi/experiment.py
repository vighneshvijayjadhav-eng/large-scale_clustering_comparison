"""Real data ingestion, bounded selection and isolated benchmark workers."""

import hashlib
import json
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from taxi.data import FEATURES, Rules, matrix, prepare, scan

SEED = 42

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
PAGE = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
REPORTS = Path("reports")


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def download():
    path = Path("data/yellow_tripdata_2025-01.parquet")
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(".part")
        for attempt in range(3):
            try:
                with (
                    urllib.request.urlopen(URL, timeout=120) as response,
                    partial.open("wb") as target,
                ):
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
                pq.ParquetFile(partial)  # Reject incomplete or non-Parquet downloads.
                partial.replace(path)
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2)
    print(path)


def clean_data(path, rules=Rules()):
    REPORTS.mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    good, bad = prepare(scan(path), rules)
    good.sink_parquet("data/clean.parquet", engine="streaming")
    reasons = bad.group_by("rejection_reason").len().collect(engine="streaming")
    clean = pl.scan_parquet("data/clean.parquet")
    n = clean.select(pl.len()).collect().item()
    rejected = int(reasons["len"].sum())
    with open(path, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "source_page": PAGE,
        "url": URL,
        "sha256": digest,
        "bytes": Path(path).stat().st_size,
        "raw_rows": n + rejected,
        "accepted_rows": n,
        "rejected_rows": rejected,
        "reasons": dict(reasons.iter_rows()),
        "rules": asdict(rules),
    }
    write_json(REPORTS / "provenance.json", report)
    clean.select(FEATURES).collect(engine="streaming").describe().write_csv(
        REPORTS / "feature_statistics.csv"
    )
    print(json.dumps(report, indent=2), flush=True)


def sample_rows(lf, n):
    # Hash-priority selection avoids loading all trip columns for a random sample.
    if "row_id" not in lf.collect_schema():
        lf = lf.with_row_index("row_id")
    return (
        lf.with_columns(pl.col("row_id").hash(seed=SEED).alias("priority"))
        .sort("priority")
        .head(n)
        .drop("priority")
        .collect(engine="streaming")
    )


def train_real(lf, out, rows, config=None):
    import plotly.express as px
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from taxi.model import fit_bundle

    config = config or {"features": FEATURES, "rules": asdict(Rules()), "anomaly_percentile": 99}
    provenance = json.loads((REPORTS / "provenance.json").read_text())
    if config["rules"] != provenance["rules"]:
        raise ValueError(
            "Cleaning rules differ. Rerun clean with the same --config before training."
        )
    sample = sample_rows(lf, rows)
    selection = sample_rows(lf, min(20000, rows))
    x = StandardScaler().fit_transform(matrix(selection, config["features"])).astype("float32")
    scores = []
    with threadpool_limits(limits=2):
        for k in range(2, 9):
            model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x)
            score = float(
                silhouette_score(x, model.labels_, sample_size=min(2000, len(x)), random_state=SEED)
            )
            scores.append(
                {
                    "k": k,
                    "rows": len(x),
                    "normalized_inertia": float(model.inertia_ / len(x)),
                    "silhouette": score,
                }
            )
            print(scores[-1], flush=True)
    chosen = max(scores, key=lambda r: r["silhouette"])["k"]
    REPORTS.mkdir(exist_ok=True)
    table = pl.DataFrame(scores)
    table.write_csv(REPORTS / "k_selection.csv")
    for metric in ["normalized_inertia", "silhouette"]:
        px.line(table.to_pandas(), x="k", y=metric, markers=True).write_html(
            REPORTS / f"k_{metric}.html", include_plotlyjs="cdn"
        )
    write_json(
        REPORTS / "selection.json",
        {
            "k": chosen,
            "rule": "Maximum sampled silhouette over k=2..8; elbow shown as supporting evidence, not a forced visual optimum.",
            "selection_rows": len(x),
            "silhouette_rows": min(2000, len(x)),
            "seed": SEED,
        },
    )
    bundle, metadata = fit_bundle(
        sample,
        out,
        k=chosen,
        source=URL,
        rules=Rules(**config["rules"]),
        features=config["features"],
        anomaly_percentile=config["anomaly_percentile"],
    )
    write_json(REPORTS / "model_metadata.json", metadata)
    from taxi.frozen import export_snapshot

    export_snapshot(
        bundle["scaler"],
        config["features"],
        chosen,
        config["rules"],
        provenance,
        REPORTS / "frozen_preprocessing.json",
    )
    pl.read_csv(Path(out) / "profiles.csv").write_csv(REPORTS / "profiles.csv")
    print(json.dumps(metadata, indent=2), flush=True)


def build_benchmark_array(bundle):
    features = bundle.get("features", FEATURES)
    source = pq.ParquetFile("data/clean.parquet")
    n = source.metadata.num_rows
    path = Path("data/benchmark.npy")
    target = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(n, len(features)))
    offset = 0
    for batch in source.iter_batches(batch_size=50000, columns=features):
        values = matrix(pl.from_arrow(batch), features)
        target[offset : offset + len(values)] = bundle["scaler"].transform(values)
        offset += len(values)
    target.flush()
    del target
    order = np.random.default_rng(SEED).permutation(n)
    np.save("data/benchmark_order.npy", order)
    return n


def benchmark_plots(reports=REPORTS):
    for filename, xaxis in [("benchmarks", "rows"), ("batch_study", "batch_size")]:
        import plotly.express as px

        if not (reports / f"{filename}.csv").exists():
            continue
        df = pl.read_csv(reports / f"{filename}.csv").filter(pl.col("status") == "ok").to_pandas()
        if len(df):
            df["series"] = df["algorithm"] + df["streaming"].map({True: " (streaming)", False: ""})
            px.line(df, x=xaxis, y="fit_seconds", color="series", markers=True).write_html(
                reports / f"{filename}.html", include_plotlyjs="cdn"
            )


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        bundle = joblib.load(Path(sys.argv[2]) / "benchmark_preprocessing.joblib")
        frozen_path = REPORTS / "frozen_preprocessing.json"
        if frozen_path.exists():
            from taxi.frozen import restore_scaler

            snapshot = json.loads(frozen_path.read_text())
            reference = restore_scaler(snapshot)
            if (
                bundle["k"] != snapshot["k"]
                or bundle.get("features", FEATURES) != snapshot["features"]
            ):
                raise ValueError(
                    "Benchmark artifacts differ from frozen features/k; regenerate together."
                )
            if not all(
                np.array_equal(getattr(bundle["scaler"], name), getattr(reference, name))
                for name in ["mean_", "scale_", "var_"]
            ):
                raise ValueError(
                    "Benchmark scaler differs from frozen metadata; regenerate together."
                )
        rows = build_benchmark_array(bundle)
        write_json(
            "artifacts/benchmark_context.json",
            {"k": bundle["k"], "rows": rows, "features": bundle.get("features", FEATURES)},
        )
    elif sys.argv[1] == "plots":
        benchmark_plots(Path(sys.argv[2]) if len(sys.argv) > 2 else REPORTS)
