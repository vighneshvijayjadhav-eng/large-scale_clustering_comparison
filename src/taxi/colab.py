"""Preparation for free Colab CPU runtimes; uses the same local pipeline."""

import json
from pathlib import Path

import joblib
import psutil

from taxi.data import Rules
from taxi.experiment import clean_data, download
from taxi.frozen import restore_scaler


def prepare():
    snapshot = json.loads(Path("reports/frozen_preprocessing.json").read_text())
    if snapshot["version"] != 1 or snapshot["seed"] != 42:
        raise ValueError("Unsupported frozen preprocessing version/seed")
    available = psutil.virtual_memory().available / 1024**3
    print(
        f"Available RAM: {available:.2f} GiB. Full estimators also receive per-run memory preflight.",
        flush=True,
    )
    if available < 1:
        raise RuntimeError(
            "Less than 1 GiB RAM available. Restart a free CPU runtime, close other notebooks and retry. No sample substitution or paid tier is required."
        )
    download()
    clean_data("data/yellow_tripdata_2025-01.parquet", Rules(**snapshot["rules"]))
    provenance = json.loads(Path("reports/provenance.json").read_text())
    if (
        provenance["sha256"] != snapshot["source_sha256"]
        or provenance["accepted_rows"] != snapshot["cleaned_rows"]
    ):
        raise ValueError(
            "Source hash or cleaned count differs from the local frozen experiment. Stop and reconcile provenance; do not combine results."
        )
    out = Path("artifacts/colab")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"scaler": restore_scaler(snapshot), "k": snapshot["k"], "features": snapshot["features"]},
        out / "benchmark_preprocessing.joblib",
    )


if __name__ == "__main__":
    prepare()
