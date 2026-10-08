import json

import numpy as np
import polars as pl
import pytest

from taxi.charts import COLORS, categorical, descriptions, distribution, projection
from taxi.data import validate
from taxi.fixture import synthetic
from taxi.model import fit_bundle
from taxi.selection import analyze, choose_k


def test_selection_is_not_maximum_silhouette():
    scores = pl.DataFrame(
        {
            "k": [2, 3, 4, 5],
            "normalized_inertia": [10.0, 5.0, 4.0, 3.0],
            "silhouette": [0.6, 0.4, 0.35, 0.3],
            "min_cluster_share": [0.2, 0.1, 0.002, 0.001],
            "stability_ari": [1.0, 0.99, 0.98, 0.95],
        }
    )
    selected, table = choose_k(scores)
    assert selected == 3
    assert table["eligible"].to_list() == [True, True, False, False]
    with pytest.raises(ValueError, match="No K"):
        choose_k(scores.with_columns(pl.lit(0.01).alias("stability_ari")))


def test_selection_outputs(tmp_path):
    good, _, _ = validate(synthetic(180))
    result = analyze(good, out=tmp_path)
    scores = pl.read_csv(tmp_path / "k_selection.csv")
    assert scores["k"].to_list() == list(range(2, 11))
    assert scores["rows"].unique().to_list() == [180]
    assert json.loads((tmp_path / "selection.json").read_text())["k"] == result["k"]
    profiles = pl.read_csv(tmp_path / "k_profiles.csv")
    assert profiles.group_by("k").agg(pl.col("size").sum())["size"].to_list() == [180] * 9


def test_profiles_anomalies_and_centroids(tmp_path):
    bundle, metadata = fit_bundle(synthetic(90), tmp_path, k=3, with_umap=False)
    profiles = pl.read_csv(tmp_path / "profiles.csv")
    for name in bundle["models"]:
        own = profiles.filter(pl.col("algorithm") == name)
        assert own["size"].sum() == 90
        assert len(descriptions(own)) == 3
        flagged = pl.read_parquet(tmp_path / f"anomalies_{name}.parquet")
        assert flagged.height == metadata["anomalies"][name]["count"]
        assert (flagged["anomaly_ratio"] > 1).all()
        centers = pl.read_csv(tmp_path / f"centroids_{name}.csv").drop("cluster")
        assert np.allclose(
            centers.to_numpy(),
            bundle["scaler"].inverse_transform(bundle["models"][name].cluster_centers_),
        )


def test_categorical_colors_and_projection():
    frame = pl.DataFrame(
        {
            "cluster": [0, 1, 2],
            "row_id": [0, 1, 2],
            "trip_distance": [1.0, 2.0, 3.0],
            "duration_minutes": [5.0, 6.0, 7.0],
            "umap2_1": [0.0, 1.0, None],
            "umap2_2": [2.0, 3.0, None],
        }
    )
    assert categorical(frame)["cluster_label"].to_list() == ["Cluster 0", "Cluster 1", "Cluster 2"]
    assert len(set(COLORS.values())) == 10
    fig = distribution(frame)
    assert {trace.name: trace.marker.color for trace in fig.data} == {
        key: COLORS[key] for key in ["Cluster 0", "Cluster 1", "Cluster 2"]
    }
    points = projection(frame)
    assert sum(len(trace.x) for trace in points.data) == 2
    assert points.layout.height == 490
    assert projection(frame, 3) is None
    assert projection(frame.head(0)) is None
