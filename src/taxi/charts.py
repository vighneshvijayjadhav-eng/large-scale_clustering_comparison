"""Small, shared presentation helpers; never fit models."""

import math


import plotly.express as px

import polars as pl


PALETTE = [
    "#8B5CF6",
    "#14B8A6",
    "#F59E0B",
    "#FB7185",
    "#22D3EE",
    "#A3E635",
    "#E879F9",
    "#F97316",
    "#CBD5E1",
    "#F472B6",
]

COLORS = {f"Cluster {i}": color for i, color in enumerate(PALETTE)}

LABELS = {
    "trip_distance": "Distance (miles)",
    "duration_minutes": "Duration (minutes)",
    "speed_mph": "Speed (mph)",
    "fare_per_mile": "Fare per mile ($)",
    "size": "Trips",
    "fit_seconds": "Training time (seconds)",
    "peak_rss_mb": "Peak process-tree RSS (MB)",
    "normalized_inertia": "Inertia per trip",
    "silhouette": "Sampled silhouette",
    "rows": "Dataset rows",
    "batch_size": "Batch size",
    "centroid_distance": "Distance to centroid",
    "anomaly_ratio": "Distance / training threshold",
}


def categorical(frame):

    return frame.with_columns(
        pl.concat_str(pl.lit("Cluster "), pl.col("cluster").cast(pl.String)).alias("cluster_label")
    )


def styled(fig, height=340):

    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#0B1020",
        plot_bgcolor="#0B1020",
        font=dict(color="#F1F5F9", size=12),
        margin=dict(l=15, r=15, t=40, b=15),
        legend_title_text="",
        colorway=PALETTE,
    )

    fig.update_xaxes(gridcolor="#243047", zeroline=False)

    fig.update_yaxes(gridcolor="#243047", zeroline=False)

    return fig


def distribution(frame):

    counts = (
        frame
        if "size" in frame.columns
        else frame.group_by("cluster").len().rename({"len": "size"})
    )

    return styled(
        px.bar(
            categorical(counts.sort("cluster")).to_pandas(),
            x="cluster_label",
            y="size",
            color="cluster_label",
            color_discrete_map=COLORS,
            labels={**LABELS, "cluster_label": "Cluster"},
            text_auto=".3s",
        ),
        290,
    )


def projection(frame, dim=2):

    axes = [f"umap{dim}_{i + 1}" for i in range(dim)]

    if not all(c in frame.columns for c in axes):
        return None

    points = categorical(frame.drop_nulls(axes))

    if points.is_empty():
        return None

    kwargs = dict(
        data_frame=points.to_pandas(),
        x=axes[0],
        y=axes[1],
        color="cluster_label",
        color_discrete_map=COLORS,
        opacity=0.7,
        hover_data=[
            c
            for c in ["row_id", "trip_distance", "duration_minutes", "anomaly_ratio"]
            if c in points.columns
        ],
        labels={**LABELS, **{c: f"UMAP {i + 1}" for i, c in enumerate(axes)}},
    )

    if "origin" in points.columns:
        kwargs.update(symbol="origin", symbol_map={"Reference": "circle", "Uploaded": "diamond"})

    fig = px.scatter_3d(z=axes[2], **kwargs) if dim == 3 else px.scatter(**kwargs)

    fig.update_traces(marker_size=3 if dim == 3 else 5)

    return styled(fig, 490)


def descriptions(profiles):
    """Names explicitly express measured means, without semantic ID equivalence."""

    names = {}

    for row in profiles.iter_rows(named=True):
        hour = (math.atan2(row.get("hour_sin", 0), row.get("hour_cos", 0)) * 12 / math.pi) % 24

        concentration = math.hypot(row.get("hour_sin", 0), row.get("hour_cos", 0))

        timing = (
            f" · pickup around {round(hour) % 24:02}:00"
            if concentration >= 0.4
            else " · mixed pickup hours"
        )

        names[int(row["cluster"])] = (
            f"{row['trip_distance']:.1f} mi · {row['duration_minutes']:.0f} min · {row['speed_mph']:.0f} mph average"
            + timing
        )

    return names
