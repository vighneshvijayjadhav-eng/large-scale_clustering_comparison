"""Extract bounded saved evidence for report figures. Never fit or transform models."""

import json
from pathlib import Path
import polars as pl

out = Path("artifacts/report_inputs")
out.mkdir(exist_ok=True)
summary = {}
for name in ["KMeans", "MiniBatchKMeans"]:
    for kind in ["sample", "anomalies"]:
        frame = pl.read_parquet(f"artifacts/real/{kind}_{name}.parquet")
        frame.write_csv(out / f"{kind}_{name}.csv")
        if kind == "anomalies":
            summary[name] = {
                "count": frame.height,
                "by_cluster": frame.group_by("cluster").len().sort("cluster").to_dicts(),
                "max_ratio": frame["anomaly_ratio"].max(),
                "median_ratio": frame["anomaly_ratio"].median(),
                "median_distance": frame["trip_distance"].median(),
                "median_fare_per_mile": frame["fare_per_mile"].median(),
            }
            frame.sort("anomaly_ratio", descending=True).head(5).write_csv(
                out / f"top_anomalies_{name}.csv"
            )
print(json.dumps(summary, indent=2))
(out / "anomaly_summary.json").write_text(json.dumps(summary, indent=2))
