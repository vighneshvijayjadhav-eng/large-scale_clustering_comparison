"""Run with: streamlit run app.py"""

import json
from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st

from taxi.model import load_bundle, predict

st.set_page_config(
    page_title="Taxi Pattern Lab", page_icon="🚕", layout="wide", initial_sidebar_state="expanded"
)
st.html("""
<style>
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse at 90% 0%, rgba(232,163,23,.14), transparent 44%),
    radial-gradient(ellipse at 5% 80%, rgba(35,126,160,.13), transparent 48%),
    linear-gradient(rgba(124,151,179,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(124,151,179,.035) 1px, transparent 1px), #0e1525;
  background-size: auto, auto, 48px 48px, 48px 48px, auto;
}
[data-testid="stSidebar"] { background: rgba(18,29,47,.97); border-right: 1px solid #2b3a50; }
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"] {
  visibility: visible !important; opacity: 1 !important;
  border: 1px solid #b88626; border-radius: 9px; background: #263348;
}
[data-testid="stMetric"] { background: rgba(25,36,58,.9); border: 1px solid #33435a;
  border-radius: 14px; padding: 18px; }
h1 { letter-spacing: -.035em; }
</style>
""")
st.title("🚕 Taxi Pattern Lab")
st.caption("January 2025 · NYC Yellow Taxi · Scalable trip pattern mining")
artifact_path = st.sidebar.text_input("Local model directory", "artifacts/real")
root = Path(artifact_path)
if not (root / "bundle.joblib").exists():
    st.info(
        "Build real artifacts with `taxi download`, `taxi clean`, then `taxi train data/clean.parquet`. For a synthetic demo run `taxi demo` and enter artifacts/demo here."
    )
    st.stop()


@st.cache_resource
def cached_bundle(path, modified):
    return load_bundle(path)


bundle = cached_bundle(str(root), (root / "bundle.joblib").stat().st_mtime_ns)
metadata = json.loads((root / "metadata.json").read_text())
algorithm = st.sidebar.selectbox("Saved model", list(bundle["models"]))
section = st.sidebar.radio(
    "Explore",
    [
        "Overview",
        "Discovered clusters",
        "Algorithm comparison",
        "Anomalies",
        "Predict uploaded trips",
    ],
)
st.sidebar.caption(f"Frozen k = {metadata['k']} · seed {metadata['seed']}")
if "SYNTHETIC" in metadata["source"].upper():
    st.warning("SYNTHETIC DEVELOPMENT DEMO — these are not empirical taxi results.")


def scatter(frame, dim, color="cluster", symbol=None):
    axes = [f"umap{dim}_{i + 1}" for i in range(dim)]
    if not all(c in frame.columns for c in axes):
        st.info("UMAP reducers are missing. Regenerate artifacts with projections enabled.")
        return
    chart = frame.drop_nulls(axes).to_pandas()
    chart[color] = chart[color].astype(str)
    kwargs = dict(
        data_frame=chart,
        x=axes[0],
        y=axes[1],
        color=color,
        hover_data=["row_id", "trip_distance", "duration_minutes"],
        opacity=0.6,
        symbol=symbol,
    )
    fig = px.scatter_3d(z=axes[2], **kwargs) if dim == 3 else px.scatter(**kwargs)
    st.plotly_chart(fig, width="stretch")


if section == "Overview":
    st.subheader("Patterns in trip characteristics")
    a, b, c = st.columns(3)
    a.metric("Model training rows", f"{metadata['training']['accepted_rows']:,}")
    b.metric("Clusters", metadata["k"])
    c.metric("Projection reference rows", metadata["projection_rows"])
    st.write(
        "Clusters summarize distance, duration, speed, fare per mile and cyclical pickup time. They are descriptive patterns, not geographic zones or causal explanations."
    )
    st.json(metadata)
    if Path("reports/provenance.json").exists():
        st.subheader("Full dataset validation")
        st.json(json.loads(Path("reports/provenance.json").read_text()))
elif section == "Discovered clusters":
    profiles = pl.read_csv(root / "profiles.csv").filter(pl.col("algorithm") == algorithm)
    st.dataframe(profiles, hide_index=True)
    measure = st.selectbox(
        "Cluster profile",
        ["size", "trip_distance", "duration_minutes", "speed_mph", "fare_per_mile"],
    )
    st.plotly_chart(px.bar(profiles.to_pandas(), x="cluster", y=measure), width="stretch")
    dimension = st.radio("UMAP dimensions", [2, 3], horizontal=True)
    scatter(pl.read_parquet(root / f"sample_{algorithm}.parquet"), dimension)
    st.caption(
        "Bounded reference sample only. UMAP axes have no physical units; apparent separation is not proof of cluster quality."
    )
elif section == "Algorithm comparison":
    for filename in ["k_selection.csv", "benchmarks.csv", "batch_study.csv"]:
        path = Path("reports") / filename
        if path.exists():
            st.subheader(filename.removesuffix(".csv").replace("_", " ").title())
            df = pl.read_csv(path)
            st.dataframe(df, hide_index=True)
            if "fit_seconds" in df.columns:
                successful = df.filter(pl.col("status") == "ok").to_pandas()
                if len(successful):
                    successful["series"] = successful["algorithm"] + successful["streaming"].map(
                        {True: " (streaming)", False: ""}
                    )
                    x = "batch_size" if filename == "batch_study.csv" else "rows"
                    for y in ["fit_seconds", "peak_rss_mb", "normalized_inertia", "silhouette"]:
                        st.plotly_chart(
                            px.line(successful, x=x, y=y, color="series", markers=True),
                            width="stretch",
                        )
            elif "silhouette" in df.columns and "k" in df.columns:
                for y in ["normalized_inertia", "silhouette"]:
                    st.plotly_chart(
                        px.line(df.to_pandas(), x="k", y=y, markers=True), width="stretch"
                    )
        else:
            st.info(
                f"{filename} has not been generated. Run the real-data training and benchmark commands."
            )
    st.caption(
        "ARI compares partitions despite arbitrary label numbering. Missing or skipped measurements are never interpolated."
    )
elif section == "Anomalies":
    st.write(
        "Distance outliers exceed the saved 99th percentile of training distances to the assigned centroid. This is a statistical flag, not evidence of fraud."
    )
    sample = pl.read_parquet(root / f"sample_{algorithm}.parquet")
    scatter(sample, 2, "distance_outlier")
    st.dataframe(sample.filter(pl.col("distance_outlier")), hide_index=True)
else:
    st.subheader("Assign new trips to existing clusters")
    st.write(
        "Required columns: tpep_pickup_datetime, tpep_dropoff_datetime, trip_distance, fare_amount, total_amount. Timestamps must use NYC local time, e.g. 2025-01-15 10:30:00. row_id is the zero-based source row position. Extra columns are ignored."
    )
    upload = st.file_uploader(
        "CSV or Parquet (maximum 20 MB, 100,000 rows)", type=["csv", "parquet"]
    )
    if upload is not None:
        try:
            if upload.size > 20 * 1024**2:
                raise ValueError("Upload exceeds the 20 MB limit.")
            frame = (
                pl.read_csv(upload, n_rows=100001, infer_schema_length=10000)
                if upload.name.lower().endswith(".csv")
                else pl.read_parquet(upload, n_rows=100001)
            )
            if frame.height > 100000:
                raise ValueError("Upload exceeds the 100,000-row limit; split the file.")
            with st.spinner("Validating and applying saved models…"):
                result, rejected, report = predict(bundle, frame, algorithm)
            st.json(report)
            if rejected.height:
                st.dataframe(rejected, hide_index=True)
                st.download_button(
                    "Download rejected rows", rejected.write_csv(), "rejected.csv", "text/csv"
                )
            if result.height:
                st.dataframe(result.head(100), hide_index=True)
                st.download_button(
                    "Download all accepted predictions",
                    result.write_csv(),
                    "predictions.csv",
                    "text/csv",
                )
                reference = pl.read_parquet(root / f"sample_{algorithm}.parquet").with_columns(
                    pl.lit("Reference").alias("origin")
                )
                highlighted = result.with_columns(pl.lit("Uploaded").alias("origin"))
                chart = pl.concat([reference, highlighted], how="diagonal_relaxed")
                dim = st.radio("Projection dimensions", [2, 3], horizontal=True)
                scatter(chart, dim, symbol="origin")
                st.caption(
                    "All accepted rows are exported. At most 2,000 uploaded points are projected; blank UMAP coordinates denote rows outside the chart sample. No fitting occurs during upload."
                )
        except (ValueError, pl.exceptions.PolarsError) as error:
            st.error(str(error))
