import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.preprocessing import StandardScaler

from taxi.frozen import export_snapshot, restore_scaler
from taxi.results import combine_results


def test_portable_frozen_scaler(tmp_path):
    x = np.random.default_rng(42).normal(size=(30, 2)).astype("float32")
    scaler = StandardScaler().fit(x)
    snapshot = export_snapshot(
        scaler,
        ["trip_distance", "duration_minutes"],
        2,
        {},
        {"sha256": "test", "accepted_rows": 30},
        tmp_path / "frozen.json",
    )
    restored = restore_scaler(json.loads((tmp_path / "frozen.json").read_text()))
    assert np.array_equal(scaler.transform(x), restored.transform(x))
    snapshot["scaler"]["scale"][0] = 0
    with pytest.raises(ValueError, match="scale"):
        restore_scaler(snapshot)


def test_colab_report_import():
    # Explicitly synthetic report values for validation only; never saved as empirical evidence.
    local = pl.DataFrame(
        {
            "algorithm": ["KMeans"],
            "rows": [50],
            "status": ["skipped_memory_preflight"],
            "environment": ["local"],
            "seed": [42],
            "k": [2],
            "feature_count": [6],
            "streaming": [False],
            "experiment_id": ["fixture"],
        }
    )
    remote = local.with_columns(pl.lit("colab").alias("environment"), pl.lit(100).alias("rows"))
    combined = combine_results(local, remote)
    assert combined.height == 2
    assert combined["environment"].to_list() == ["local", "colab"]
    with pytest.raises(ValueError, match="experiment_id"):
        combine_results(local, remote.with_columns(pl.lit("different").alias("experiment_id")))
    with pytest.raises(ValueError, match="environment"):
        combine_results(local, local)
    with pytest.raises(ValueError, match="fit_seconds"):
        combine_results(local, remote.with_columns(pl.lit("ok").alias("status")))


def test_notebook_is_clean_and_python_compiles():
    path = Path(__file__).parents[1] / "notebooks/full_dataset_benchmark_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), str(path), "exec")
