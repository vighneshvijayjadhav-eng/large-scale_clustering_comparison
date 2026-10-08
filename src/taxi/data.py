"""One validation/engineering path for training and uploads."""

from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

REQUIRED = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "total_amount",
]
FEATURES = [
    "trip_distance",
    "duration_minutes",
    "speed_mph",
    "fare_per_mile",
    "hour_sin",
    "hour_cos",
]


@dataclass(frozen=True)
class Rules:
    min_minutes: float = 1
    max_minutes: float = 180
    max_distance: float = 150
    max_speed: float = 100
    max_fare: float = 1000
    max_total: float = 1500


def scan(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix == ".csv":
        return pl.scan_csv(path, infer_schema_length=10000, try_parse_dates=False)
    raise ValueError("Use a CSV or Parquet file.")


def prepare(frame, rules=Rules()):
    """Returns lazy accepted/rejected rows; first failing reason owns each rejection."""
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    schema = lf.collect_schema()
    missing = sorted(set(REQUIRED) - set(schema.names()))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    # Source row index is generated regardless of any untrusted uploaded row_id column.
    lf = lf.select(REQUIRED).with_row_index("row_id")
    times = REQUIRED[:2]
    for name in times:
        dtype = schema[name]
        if dtype == pl.String:
            expr = pl.coalesce(
                pl.col(name).str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False),
                pl.col(name).str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False),
            )
        elif isinstance(dtype, pl.Datetime):
            if dtype.time_zone:
                raise ValueError(
                    "Timezone-aware timestamps are unsupported; use NYC local naive time."
                )
            expr = pl.col(name).cast(pl.Datetime("us"))
        elif dtype in (pl.Date, pl.Null):
            expr = pl.col(name).cast(pl.Datetime("us"))
        else:
            raise ValueError(
                f"Schema mismatch: {name} must contain timestamp strings or datetimes."
            )
        lf = lf.with_columns(expr.alias(name))
    lf = lf.with_columns(pl.col(REQUIRED[2:]).cast(pl.Float64, strict=False))
    lf = lf.with_columns(
        ((pl.col(times[1]) - pl.col(times[0])).dt.total_milliseconds() / 60000).alias(
            "duration_minutes"
        ),
        pl.col(times[0]).dt.hour().alias("pickup_hour"),
    ).with_columns(
        (pl.col("trip_distance") * 60 / pl.col("duration_minutes")).alias("speed_mph"),
        (pl.col("fare_amount") / pl.col("trip_distance")).alias("fare_per_mile"),
        (pl.col("pickup_hour") * (2 * 3.141592653589793 / 24)).sin().alias("hour_sin"),
        (pl.col("pickup_hour") * (2 * 3.141592653589793 / 24)).cos().alias("hour_cos"),
    )
    invalid_time = pl.any_horizontal(pl.col(times).is_null())
    invalid_numeric = pl.any_horizontal(
        ~pl.col(REQUIRED[2:]).is_finite() | pl.col(REQUIRED[2:]).is_null()
    )
    reason = (
        pl.when(invalid_time)
        .then(pl.lit("invalid_timestamp"))
        .when(invalid_numeric)
        .then(pl.lit("missing_or_nonfinite_amount_distance"))
        .when(~pl.col("duration_minutes").is_between(rules.min_minutes, rules.max_minutes))
        .then(pl.lit("duration_out_of_range"))
        .when((pl.col("trip_distance") <= 0) | (pl.col("trip_distance") > rules.max_distance))
        .then(pl.lit("distance_out_of_range"))
        .when(
            (pl.col("fare_amount") <= 0)
            | (pl.col("fare_amount") > rules.max_fare)
            | (pl.col("total_amount") <= 0)
            | (pl.col("total_amount") > rules.max_total)
        )
        .then(pl.lit("charge_out_of_range"))
        .when(pl.col("speed_mph") > rules.max_speed)
        .then(pl.lit("speed_out_of_range"))
        .when(
            pl.any_horizontal(
                ~pl.col(FEATURES).cast(pl.Float32).is_finite() | pl.col(FEATURES).is_null()
            )
        )
        .then(pl.lit("nonfinite_features"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("rejection_reason")
    )
    checked = lf.with_columns(reason)
    return checked.filter(pl.col("rejection_reason").is_null()).drop(
        "rejection_reason"
    ), checked.filter(pl.col("rejection_reason").is_not_null())


def validate(frame, rules=Rules()):
    good, bad = prepare(frame, rules)
    good, bad = pl.collect_all([good, bad], engine="streaming")
    return (
        good,
        bad,
        {
            "raw_rows": good.height + bad.height,
            "accepted_rows": good.height,
            "rejected_rows": bad.height,
            "reasons": dict(bad.group_by("rejection_reason").len().iter_rows()),
            "rules": asdict(rules),
        },
    )


def matrix(frame, features=FEATURES):
    return frame.select(features).to_numpy().astype("float32")
