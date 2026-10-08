"""Lean benchmark coordinator: heavy preparation runs in a short-lived process."""

import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import psutil

SEED = 42
FEATURES = [
    "trip_distance",
    "duration_minutes",
    "speed_mph",
    "fare_per_mile",
    "hour_sin",
    "hour_cos",
]
REPORTS = Path("reports")


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def save_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: str(value).lower() if isinstance(value, bool) else value
                for key, value in row.items()
            }
            for row in rows
        )


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
                family = [monitor, *monitor.children(recursive=True)]
                peak = max(peak, sum(p.memory_info().rss for p in family if p.is_running()))
            except psutil.NoSuchProcess:
                pass
            if time.perf_counter() - started > timeout:
                status = "aborted_timeout"
            elif psutil.virtual_memory().available < 512 * 1024**2:
                status = "aborted_low_memory"
            if status:
                # Windows venv launchers spawn the real interpreter; stop the entire worker tree.
                for child in monitor.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
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
    global FEATURES
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for name in ["benchmarks.csv", "batch_study.csv", "runtime.json", "full_decision.json"]:
        previous = REPORTS / name
        if previous.exists():
            archive = REPORTS / "attempts" / stamp
            archive.mkdir(parents=True, exist_ok=True)
            (archive / name).write_bytes(previous.read_bytes())
    subprocess.run([sys.executable, "-m", "taxi.experiment", "prepare", artifacts], check=True)
    context = json.loads(Path("artifacts/benchmark_context.json").read_text())
    k, n = context["k"], context["rows"]
    FEATURES = context["features"]
    # Never reuse ARI labels from a previous run with a different scaler or k.
    for old in Path("artifacts").glob("labels_*.npy"):
        old.unlink()
    write_json(
        REPORTS / "runtime.json",
        {
            "platform": platform.platform(),
            "started_utc": stamp,
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
            "memory_measure": "50 ms sampled worker process-tree RSS across startup, fit and metrics; not fit-only allocation",
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
            save_rows(REPORTS / "benchmarks.csv", results)
    if full:
        results.append(run_worker("MiniBatchKMeans", n, k, streaming=True))
        save_rows(REPORTS / "benchmarks.csv", results)
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
        save_rows(REPORTS / "batch_study.csv", batches)
        print(batches[-1], flush=True)
    subprocess.run([sys.executable, "-m", "taxi.experiment", "plots"], check=True)
