"""Validate report imports separately from trip uploads."""

import polars as pl


def combine_results(local, colab=None):
    local = (
        local.with_columns(pl.lit("local").alias("environment"))
        if "environment" not in local.columns
        else local
    )
    if colab is None:
        return local
    required = {
        "algorithm",
        "rows",
        "status",
        "environment",
        "seed",
        "k",
        "feature_count",
        "streaming",
        "experiment_id",
    }
    missing = required - set(colab.columns)
    if missing:
        raise ValueError(f"Colab report missing columns: {', '.join(sorted(missing))}")
    if (
        colab.is_empty()
        or colab.filter(pl.col("environment") != "colab").height
        or colab["environment"].null_count()
    ):
        raise ValueError("Imported rows must explicitly identify environment=colab.")
    if set(colab["algorithm"]) - {"KMeans", "MiniBatchKMeans"}:
        raise ValueError("Unknown benchmark algorithm")
    if colab["status"].null_count() or set(colab["status"]) - {
        "ok",
        "failed",
        "skipped_memory_preflight",
        "skipped_insufficient_rows",
        "aborted_timeout",
        "aborted_low_memory",
    }:
        raise ValueError("Unknown benchmark status")
    for key in ["seed", "k", "feature_count", "experiment_id"]:
        if colab[key].null_count() or not set(colab[key]) <= set(local[key]):
            raise ValueError(f"Colab {key} does not match the frozen local experiment")
    if colab.filter(pl.col("rows") <= 0).height or colab["rows"].null_count():
        raise ValueError("Benchmark row counts must be positive")
    successful = colab.filter(pl.col("status") == "ok")
    if successful.height:
        for metric in ["fit_seconds", "peak_rss_mb", "normalized_inertia", "silhouette"]:
            if metric not in colab.columns or successful[metric].null_count():
                raise ValueError(f"Successful Colab rows need {metric}")
            values = successful[metric].cast(pl.Float64, strict=False)
            if values.null_count() or not values.is_finite().all():
                raise ValueError(f"Invalid Colab metric: {metric}")
            if metric == "silhouette":
                if not values.is_between(-1, 1).all():
                    raise ValueError("Silhouette must lie between -1 and 1")
            elif (values < 0).any():
                raise ValueError(f"Negative Colab metric: {metric}")
    return pl.concat([local, colab], how="diagonal_relaxed")
