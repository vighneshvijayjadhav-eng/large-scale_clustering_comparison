"""Clearly synthetic development trips; not NYC observations."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl


def synthetic(n=300, seed=42):
    rng = np.random.default_rng(seed)
    distance = rng.uniform(0.3, 20, n)
    minutes = distance * rng.uniform(2, 5, n) + 2
    pickup = [datetime(2025, 1, 1) + timedelta(hours=int(x)) for x in rng.integers(0, 744, n)]
    return pl.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": [
                t + timedelta(minutes=float(m)) for t, m in zip(pickup, minutes)
            ],
            "trip_distance": distance,
            "fare_amount": distance * 3 + rng.uniform(3, 8, n),
            "total_amount": distance * 3 + 12,
        }
    )
