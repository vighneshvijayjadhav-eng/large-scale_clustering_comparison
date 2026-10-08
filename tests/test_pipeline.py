import numpy as np
import polars as pl
import pytest

from taxi.data import FEATURES, matrix, scan, validate
from taxi.fixture import synthetic
from taxi.model import fit_bundle, load_bundle, predict


def test_validation():
    frame = synthetic(20).with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(0)
        .otherwise(pl.col("trip_distance"))
        .alias("trip_distance")
    )
    good, bad, report = validate(frame)
    assert good.height == 19 and bad["row_id"].to_list() == [0]
    assert report["reasons"] == {"distance_out_of_range": 1}
    assert np.isfinite(matrix(good)).all()
    assert matrix(good).shape[1] == len(FEATURES)
    with pytest.raises(ValueError, match="Missing required"):
        validate(frame.drop("fare_amount"))


@pytest.mark.parametrize(
    "column,value,reason",
    [
        ("fare_amount", None, "missing_or_nonfinite_amount_distance"),
        ("fare_amount", float("inf"), "missing_or_nonfinite_amount_distance"),
        ("fare_amount", -1, "charge_out_of_range"),
        ("trip_distance", 100, "speed_out_of_range"),
        ("tpep_dropoff_datetime", "bad", "invalid_timestamp"),
        ("tpep_dropoff_datetime", "2020-01-01 00:00:00", "duration_out_of_range"),
    ],
)
def test_reasons(column, value, reason):
    frame = synthetic(1).with_columns(pl.lit(value).alias(column))
    if column.endswith("datetime"):
        frame = frame.with_columns(pl.col(column).cast(pl.String))
    good, bad, _ = validate(frame)
    assert not good.height
    assert bad["rejection_reason"][0] == reason


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_inference_reload_upload(tmp_path, suffix):
    training = synthetic(90)
    bundle, _ = fit_bundle(training, tmp_path / "models", k=3, with_umap=False)
    unseen = synthetic(20, seed=19)
    path = tmp_path / f"unseen.{suffix}"
    getattr(unseen, f"write_{suffix}")(path)
    restored = load_bundle(tmp_path / "models")
    for name in bundle["models"]:
        a, _, _ = predict(bundle, unseen, name)
        b, _, _ = predict(restored, scan(path), name)
        assert a["cluster"].to_list() == b["cluster"].to_list()
        assert b["row_id"].to_list() == list(range(20))
        assert np.allclose(a["centroid_distance"], b["centroid_distance"])
        assert restored["scaler"].n_samples_seen_ == 90


def test_feature_order():
    good, _, _ = validate(synthetic(20))
    assert np.array_equal(matrix(good), matrix(good.select(reversed(good.columns))))
