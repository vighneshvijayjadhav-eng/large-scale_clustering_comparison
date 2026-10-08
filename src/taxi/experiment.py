"""Real data ingestion, bounded selection and isolated benchmark workers."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import plotly.express as px
import polars as pl
import psutil
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from taxi.data import FEATURES, Rules, matrix, prepare, scan
from taxi.model import SEED, fit_bundle

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


def clean_data(path):
    REPORTS.mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    good, bad = prepare(scan(path))
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
        "rules": asdict(Rules()),
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


def train_real(lf, out, rows):
    sample = sample_rows(lf, rows)
    selection = sample_rows(lf, min(20000, rows))
    x = StandardScaler().fit_transform(matrix(selection)).astype("float32")
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
    _, metadata = fit_bundle(sample, out, k=chosen, source=URL)
    write_json(REPORTS / "model_metadata.json", metadata)
    pl.read_csv(Path(out) / "profiles.csv").write_csv(REPORTS / "profiles.csv")
    print(json.dumps(metadata, indent=2), flush=True)


def build_benchmark_array(bundle):
    source = pq.ParquetFile("data/clean.parquet")
    n = source.metadata.num_rows
    path = Path("data/benchmark.npy")
    target = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(n, len(FEATURES)))
    offset = 0
    for batch in source.iter_batches(batch_size=50000, columns=FEATURES):
        values = matrix(pl.from_arrow(batch))
        target[offset : offset + len(values)] = bundle["scaler"].transform(values)
        offset += len(values)
    target.flush()
    del target
    order = np.random.default_rng(SEED).permutation(n)
    np.save("data/benchmark_order.npy", order)
    return n


def run_worker(algorithm, rows, k, batch_size=1000, streaming=False, timeout=900):
    available = psutil.virtual_memory().available
    # Working arrays plus 256 MiB interpreter/metrics headroom; live monitor also reserves RAM.
    required = rows * len(FEATURES) * 4 * (10 if algorithm == "KMeans" else 4) + 256 * 1024**2
    base = {
        "algorithm": algorithm,
        "rows": rows,
        "batch_size": batch_size,
        "streaming": streaming,
        "seed": SEED,
        "k": k,
        "feature_count": len(FEATURES),
        "available_mb": available / 1024**2,
        "estimated_mb": required / 1024**2,
    }
    if not streaming and required > available * 0.6:
        return {
            **base,
            "status": "skipped_memory_preflight",
            "detail": "Estimated working set exceeds 60% available RAM",
        }
    output = Path("artifacts/worker.json")
    output.parent.mkdir(exist_ok=True)
    output.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "taxi.worker",
        algorithm,
        str(rows),
        str(k),
        str(batch_size),
        str(int(streaming)),
        str(output),
    ]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    started = time.perf_counter()
    peak = 0
    with open("artifacts/worker.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env)
        monitor = psutil.Process(proc.pid)
        status = None
        while proc.poll() is None:
            try:
                peak = max(peak, monitor.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if time.perf_counter() - started > timeout:
                status = "aborted_timeout"
            elif psutil.virtual_memory().available < 512 * 1024**2:
                status = "aborted_low_memory"
            if status:
                proc.kill()
                proc.wait()
                break
            time.sleep(0.05)
    if status:
        result = {"status": status, "detail": "Safety monitor terminated worker"}
    elif proc.returncode != 0 or not output.exists():
        result = {"status": "failed", "detail": Path("artifacts/worker.log").read_text()[-2000:]}
    else:
        result = json.loads(output.read_text())
    return {
        **base,
        **result,
        "peak_rss_mb": peak / 1024**2,
        "worker_seconds": time.perf_counter() - started,
    }


def benchmark(artifacts, full):
    bundle = joblib.load(Path(artifacts) / "benchmark_preprocessing.joblib")
    k = bundle["k"]
    n = build_benchmark_array(bundle)
    # Never reuse ARI labels from a previous run with a different scaler or k.
    for old in Path("artifacts").glob("labels_*.npy"):
        old.unlink()
    write_json(
        REPORTS / "runtime.json",
        {
            "platform": platform.platform(),
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name)
                for name in [
                    "numpy",
                    "polars",
                    "scikit-learn",
                    "umap-learn",
                    "streamlit",
                    "pyarrow",
                    "psutil",
                ]
            },
            "logical_cpus": os.cpu_count(),
            "threads": 2,
            "ram_gb": psutil.virtual_memory().total / 1024**3,
            "seed": SEED,
            "features": FEATURES,
            "memory_measure": "50 ms sampled worker process RSS across startup, fit and metrics; not fit-only allocation",
            "repeats": 1,
            "timeout_seconds": 900,
            "scaler": "Frozen final model training scaler, identical for every benchmark",
            "sampling": "Seeded permutation of all cleaned rows; nested prefixes",
        },
    )
    results = []
    for size in [50000, 100000, 250000, 500000, 1000000] + ([n] if full else []):
        for algorithm in ["KMeans", "MiniBatchKMeans"]:
            if size > n:
                row = {"algorithm": algorithm, "rows": size, "status": "skipped_insufficient_rows"}
            else:
                row = run_worker(algorithm, size, k)
            results.append(row)
            print(row, flush=True)
            pl.DataFrame(results, infer_schema_length=None).write_csv(REPORTS / "benchmarks.csv")
    if full:
        results.append(run_worker("MiniBatchKMeans", n, k, streaming=True))
        pl.DataFrame(results, infer_schema_length=None).write_csv(REPORTS / "benchmarks.csv")
        print(results[-1], flush=True)
        write_json(
            REPORTS / "full_decision.json",
            {"cleaned_rows": n, "evaluations": [r for r in results if r["rows"] == n]},
        )
    else:
        write_json(
            REPORTS / "full_decision.json",
            {"status": "not_requested", "next_command": "taxi benchmark --full"},
        )
    batches = []
    for size in [100, 500, 1000, 5000, 10000]:
        batches.append(run_worker("MiniBatchKMeans", min(100000, n), k, batch_size=size))
        pl.DataFrame(batches, infer_schema_length=None).write_csv(REPORTS / "batch_study.csv")
        print(batches[-1], flush=True)
    for filename, xaxis in [("benchmarks", "rows"), ("batch_study", "batch_size")]:
        df = pl.read_csv(REPORTS / f"{filename}.csv").filter(pl.col("status") == "ok").to_pandas()
        if len(df):
            df["series"] = df["algorithm"] + df["streaming"].map({True: " (streaming)", False: ""})
            px.line(df, x=xaxis, y="fit_seconds", color="series", markers=True).write_html(
                REPORTS / f"{filename}.html", include_plotlyjs="cdn"
            )
