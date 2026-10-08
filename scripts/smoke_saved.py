"""Verify real saved-model uploads on records excluded from the 100k training sample."""

import json
from pathlib import Path

import numpy as np
import polars as pl

from taxi.data import scan
from taxi.experiment import sample_rows
from taxi.model import load_bundle, predict


def main():
    root = Path("artifacts/real")
    metadata = json.loads((root / "metadata.json").read_text())
    n = metadata["training"]["accepted_rows"]
    holdout = sample_rows(pl.scan_parquet("data/clean.parquet"), n + 30).tail(30)
    holdout.write_csv("data/real_unseen.csv")
    holdout.write_parquet("data/real_unseen.parquet")
    bundle = load_bundle(root)
    checks = []
    for algorithm in bundle["models"]:
        outputs = []
        for suffix in ["csv", "parquet"]:
            result, rejected, _ = predict(bundle, scan(f"data/real_unseen.{suffix}"), algorithm)
            assert result.height == 30 and rejected.height == 0
            assert result["row_id"].to_list() == list(range(30))
            assert result["umap3_3"].null_count() == 0
            assert np.isfinite(
                result.select("umap2_1", "umap2_2", "umap3_1", "umap3_2", "umap3_3").to_numpy()
            ).all()
            result.write_csv(f"artifacts/smoke_{algorithm}_{suffix}.csv")
            outputs.append(result)
            checks.append(
                {"algorithm": algorithm, "format": suffix, "accepted": 30, "projected_2d_3d": 30}
            )
        assert outputs[0]["cluster"].to_list() == outputs[1]["cluster"].to_list()
        assert np.allclose(outputs[0]["centroid_distance"], outputs[1]["centroid_distance"])
    report = {
        "status": "passed",
        "source": "30 hash-ranked cleaned records immediately after the training prefix; excluded from fitting",
        "checks": checks,
    }
    Path("reports/inference_smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
