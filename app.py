"""Local analytics dashboard. All fitting runs separately through the CLI."""

import json
from io import BytesIO
from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st

from taxi.charts import COLORS, LABELS, categorical, descriptions, distribution, projection, styled
from taxi.model import load_bundle, predict
from taxi.results import combine_results

st.set_page_config(
    page_title="Taxi Pattern Lab", page_icon="🚕", layout="wide", initial_sidebar_state="expanded"
)
st.html("""<style>
[data-testid="stSidebar"] {background:#111827; border-right:1px solid #243047;}
[data-testid="stMetric"] {background:#182235;border:1px solid #29364b;border-radius:12px;padding:14px;box-shadow:0 3px 12px #0002;}
[data-testid="stMetricValue"] {font-size:1.65rem;}
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"] {visibility:visible;opacity:1;}
[data-testid="stMainBlockContainer"] {padding-top:2rem;padding-bottom:2rem;}
[data-testid="stSidebar"] h1 {font-size:1.3rem !important;}
h3 {font-size:1.25rem !important;}
h1 {font-size:2rem !important;letter-spacing:-.035em;} h2 {font-size:1.4rem !important;}
</style>""")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_resource
def cached_bundle(path, modified):
    return load_bundle(path)


@st.cache_data(max_entries=12)
def read_table(path, modified):
    p = Path(path)
    return pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p)


def table(path):
    return read_table(str(path), path.stat().st_mtime_ns) if path.exists() else pl.DataFrame()


def chart(fig):
    if fig is None:
        st.info("Projection unavailable. Run training with saved 2D and 3D UMAP reducers.")
    else:
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
        )


def kpis(items):
    for col, (label, value) in zip(st.columns(len(items)), items, strict=True):
        col.metric(label, value)


@st.cache_data(max_entries=3, show_spinner=False)
def uploaded_predictions(data, suffix, path, modified, algorithm):
    stream = BytesIO(data)
    frame = (
        pl.read_csv(stream, n_rows=100001, infer_schema_length=10000)
        if suffix == ".csv"
        else pl.read_parquet(stream, n_rows=100001)
    )
    if frame.height > 100000:
        raise ValueError("Upload exceeds 100,000 rows; split the file.")
    return predict(cached_bundle(path, modified), frame, algorithm)


st.sidebar.title("Taxi Pattern Lab")
artifact_path = st.sidebar.text_input("Local model directory", "artifacts/real")
root = Path(artifact_path)
if not (root / "bundle.joblib").exists() or not (root / "metadata.json").exists():
    st.title("Taxi Pattern Lab")
    st.info(
        "No saved models found. Run `taxi download`, `taxi clean`, then `taxi train data/clean.parquet`. For a synthetic demo, run `taxi demo` and select artifacts/demo."
    )
    st.stop()
try:
    modified = (root / "bundle.joblib").stat().st_mtime_ns
    bundle = cached_bundle(str(root), modified)
    metadata = read_json(root / "metadata.json")
except (ValueError, OSError, KeyError) as error:
    st.error(f"Cannot load model artifacts: {error}")
    st.stop()
algorithm = st.sidebar.selectbox("Saved model", list(bundle["models"]))
section = st.sidebar.radio(
    "Explore",
    [
        "Overview",
        "Cluster Discovery",
        "Algorithm Comparison",
        "Anomaly Explorer",
        "Predict New Trips",
        "Methodology / Experiment Details",
    ],
)
st.sidebar.caption(f"Frozen K={metadata['k']} · seed {metadata['seed']}")
st.sidebar.caption(
    "Collapse this panel with the arrow above. Models are loaded locally; navigation never trains them."
)
st.title(section)
st.caption(f"NYC Yellow Taxi · January 2025 · {algorithm} · K={metadata['k']}")
synthetic = "SYNTHETIC" in metadata["source"].upper()
if synthetic:
    st.warning("SYNTHETIC DEVELOPMENT DEMO — these are not empirical taxi results.")
reports = Path("reports")
provenance = read_json(reports / "provenance.json") if not synthetic else {}
selection = metadata.get(
    "selection", read_json(reports / "selection.json") if not synthetic else {}
)
selection_matches = selection.get("k") == metadata["k"]
profiles = table(root / "profiles.csv")
if not profiles.is_empty():
    profiles = profiles.filter(pl.col("algorithm") == algorithm)
names = descriptions(profiles) if not profiles.is_empty() else {}
reference = table(root / f"sample_{algorithm}.parquet")
if not reference.is_empty() and "anomaly_ratio" not in reference.columns:
    reference = reference.with_columns(
        (pl.col("centroid_distance") / max(bundle["thresholds"][algorithm], 1e-12)).alias(
            "anomaly_ratio"
        )
    )
anomalies = table(root / f"anomalies_{algorithm}.parquet")
anomaly_meta = metadata.get("anomalies", {}).get(algorithm, {})
training_rows = metadata["training"]["accepted_rows"]

if section == "Overview":
    kpis(
        [
            ("Source trips", f"{provenance['raw_rows']:,}" if provenance else "Demo"),
            (
                "Valid trips",
                f"{provenance['accepted_rows']:,}" if provenance else f"{training_rows:,}",
            ),
            ("Training sample", f"{training_rows:,}"),
            ("Selected clusters", metadata["k"]),
        ]
    )
    st.write(
        "Explore patterns in trip length, travel time, speed, fare intensity and pickup time. Compare two clustering methods, inspect unusual records and assign new trips to the saved clusters."
    )
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Training cluster distribution")
        if not profiles.is_empty():
            chart(distribution(profiles))
    with right:
        st.subheader("Discovered patterns")
        for row in profiles.iter_rows(named=True):
            st.markdown(
                f"**Cluster {row['cluster']} · {row['size'] / training_rows:.1%}**  \n{names[row['cluster']]}"
            )
        st.caption(
            "Descriptions are measured means. Cluster IDs have separate meanings for each algorithm."
        )
    kpis(
        [
            ("Active algorithm", algorithm),
            (
                "Training distance outliers",
                f"{anomaly_meta['count']:,}" if anomaly_meta else "Unavailable",
            ),
            ("Model status", "Saved · ready"),
        ]
    )
    if selection_matches:
        st.info(selection.get("rationale", selection.get("rule")))
    with st.expander("Dataset & Model Details"):
        st.write(
            {
                "Source": metadata["source"],
                "Training rows": training_rows,
                "Projection reference rows": metadata["projection_rows"],
                "Rejected source rows": provenance.get("rejected_rows", "Not available"),
            }
        )
        st.json(metadata)

elif section == "Cluster Discovery":
    if reference.is_empty() or profiles.is_empty():
        st.info(
            "Cluster profiles or projection references are missing. Regenerate the model artifacts."
        )
    else:
        selected = st.selectbox(
            "Focus cluster",
            ["All clusters", *list(names)],
            format_func=lambda c: c if isinstance(c, str) else f"Cluster {c} · {names[c]}",
        )
        shown = (
            reference
            if selected == "All clusters"
            else reference.filter(pl.col("cluster") == selected)
        )
        a, b, c = st.tabs(["UMAP · 2D", "UMAP · 3D", "Profiles & centroids"])
        with a:
            chart(projection(shown, 2))
        with b:
            chart(projection(shown, 3))
        with c:
            measure = st.selectbox(
                "Compare average feature",
                ["trip_distance", "duration_minutes", "speed_mph", "fare_per_mile"],
                format_func=lambda x: LABELS[x],
            )
            chart(
                styled(
                    px.bar(
                        categorical(profiles).to_pandas(),
                        x="cluster_label",
                        y=measure,
                        color="cluster_label",
                        color_discrete_map=COLORS,
                        labels=LABELS,
                    )
                )
            )
            st.dataframe(profiles, hide_index=True)
            centroids = table(root / f"centroids_{algorithm}.csv")
            if not centroids.is_empty():
                st.caption(
                    "Centroids inverse-transformed into the original feature units; cyclic hour coordinates are not clock hours."
                )
                st.dataframe(centroids, hide_index=True)
        st.caption(
            f"{shown.height:,} displayed reference trips; bounded to 5,000. UMAP is a projection of scaled features, not the input to clustering. Axes have no physical units."
        )
        left, right = st.columns(2)
        with left:
            chart(distribution(profiles))
        with right:
            st.subheader("Profile interpretation")
            for cluster, description in names.items():
                if selected == "All clusters" or selected == cluster:
                    st.write(f"**Cluster {cluster}:** {description}")

elif section == "Algorithm Comparison":
    st.caption(
        "Only measured outcomes are shown. Environment and streaming mode remain separate; cross-machine timing is not an algorithm-only comparison."
    )
    performance, batches, ktab = st.tabs(
        ["Scalability & quality", "MiniBatch batch sizes", "Cluster Selection Analysis"]
    )
    with performance:
        uploaded = st.file_uploader("Import Colab benchmark CSV", type=["csv"], key="colab_results")
        measured = table(reports / "benchmarks.csv")
        if not measured.is_empty():
            try:
                remote = None
                if uploaded is not None:
                    if uploaded.size > 1024**2:
                        raise ValueError("Benchmark report must be under 1 MB.")
                    remote = pl.read_csv(uploaded)
                elif (reports / "colab/benchmarks.csv").exists():
                    remote = table(reports / "colab/benchmarks.csv")
                measured = combine_results(measured, remote)
                measured = measured.filter(pl.col("k") == metadata["k"])
                if metadata.get("experiment_id"):
                    measured = measured.filter(pl.col("experiment_id") == metadata["experiment_id"])
            except (ValueError, pl.exceptions.PolarsError) as error:
                st.error(str(error))
        if measured.is_empty():
            st.info(
                "No benchmarks matching the active K. Run `taxi benchmark --full`, or import the Colab fallback results after generating local experiment metadata."
            )
        else:
            success = measured.filter(pl.col("status") == "ok")
            if not success.is_empty():
                paired = success.filter(~pl.col("streaming")).sort("rows", descending=True)
                for candidate in paired.filter(pl.col("algorithm") == "KMeans").iter_rows(
                    named=True
                ):
                    partner = paired.filter(
                        (pl.col("algorithm") == "MiniBatchKMeans")
                        & (pl.col("rows") == candidate["rows"])
                        & (pl.col("environment") == candidate["environment"])
                    )
                    if partner.height:
                        other = partner.row(0, named=True)
                        st.info(
                            f"At {candidate['rows']:,} rows ({candidate['environment']}), K-Means fitted in {candidate['fit_seconds']:.3f}s and MiniBatch in {other['fit_seconds']:.3f}s. Inertia per trip: {candidate['normalized_inertia']:.3f} vs {other['normalized_inertia']:.3f}. Single runs; timings are not confidence intervals."
                        )
                        break
                data = success.to_pandas()
                data["series"] = (
                    data["environment"]
                    + " · "
                    + data["algorithm"]
                    + data["streaming"].map({True: " · streaming", False: ""})
                )
                cols = st.columns(2)
                for i, metric in enumerate(
                    ["fit_seconds", "peak_rss_mb", "normalized_inertia", "silhouette"]
                ):
                    with cols[i % 2]:
                        chart(
                            styled(
                                px.scatter(
                                    data,
                                    x="rows",
                                    y=metric,
                                    color="algorithm",
                                    symbol="series",
                                    color_discrete_map={
                                        "KMeans": "#8B5CF6",
                                        "MiniBatchKMeans": "#14B8A6",
                                    },
                                    labels=LABELS,
                                    title=LABELS[metric],
                                )
                            )
                        )
                st.caption(
                    "Points are actual runs; missing sizes are not interpolated. Memory is sampled process-tree RSS; it includes interpreter overhead. Training time excludes evaluation."
                )
            st.dataframe(measured, hide_index=True)
            st.download_button(
                "Download benchmark evidence", measured.write_csv(), "benchmarks.csv", "text/csv"
            )
            st.caption(
                "ARI in the table measures agreement despite arbitrary cluster numbering. Planned sizes: 50K, 100K, 250K, 500K, 1M, and full cleaned data. Failed/skipped runs have no invented scores."
            )
    with batches:
        batch = table(reports / "batch_study.csv")
        if not batch.is_empty():
            batch = batch.filter(pl.col("k") == metadata["k"])
            ok = batch.filter(pl.col("status") == "ok")
            if not ok.is_empty():
                chart(
                    styled(
                        px.line(
                            ok.to_pandas(),
                            x="batch_size",
                            y="fit_seconds",
                            markers=True,
                            labels=LABELS,
                        )
                    )
                )
            else:
                st.info(
                    "No completed batch-size runs for this K. Recorded resource failures are shown below."
                )
            st.dataframe(batch, hide_index=True)
        else:
            st.info(
                "Batch-size study pending: 100 / 500 / 1,000 / 5,000 / 10,000 on a fixed sample."
            )
    with ktab:
        scores = table(reports / "k_selection.csv")
        if selection_matches and not scores.is_empty():
            st.info(selection.get("rationale", selection.get("rule")))
            for col, metric in zip(
                st.columns(2), ["normalized_inertia", "silhouette"], strict=True
            ):
                with col:
                    fig = px.line(
                        scores.to_pandas(),
                        x="k",
                        y=metric,
                        markers=True,
                        labels=LABELS,
                        title=LABELS[metric],
                    )
                    fig.add_vline(
                        x=metadata["k"],
                        line_dash="dash",
                        line_color="#F59E0B",
                        annotation_text=f"Selected K={metadata['k']}",
                    )
                    chart(styled(fig))
            st.dataframe(scores, hide_index=True)
            with st.expander("Candidate distributions and feature means"):
                st.dataframe(table(reports / "k_profiles.csv"), hide_index=True)
            st.caption(selection.get("caveat", ""))
        else:
            st.info(
                "Selection evidence for these artifacts is unavailable; synthetic demo K is illustrative."
            )

elif section == "Anomaly Explorer":
    st.write(
        f"Records beyond the saved {metadata['anomaly_percentile']}th percentile of training centroid distances are statistical outliers, not confirmed fraud. A score of 1 is the saved threshold."
    )
    if anomaly_meta:
        kpis(
            [
                ("Training outliers", f"{anomaly_meta['count']:,}"),
                (
                    "Training outlier rate",
                    f"{anomaly_meta['count'] / anomaly_meta['population_rows']:.2%}",
                ),
                ("Population", f"{anomaly_meta['population_rows']:,}"),
            ]
        )
    else:
        st.info(
            "Exact training anomaly counts are unavailable for these older artifacts; the chart is a bounded reference sample."
        )
    left, right = st.columns(2)
    with left:
        selected = st.selectbox("Cluster filter", ["All clusters", *list(names)])
    with right:
        severity = st.selectbox(
            "Minimum severity", ["Above threshold (1×)", "At least 2×", "At least 5×"]
        )
    ratio = {"Above threshold (1×)": 1, "At least 2×": 2, "At least 5×": 5}[severity]
    rows = anomalies if not anomalies.is_empty() else reference
    if not rows.is_empty():
        rows = rows.filter(pl.col("anomaly_ratio") > ratio)
        if selected != "All clusters":
            rows = rows.filter(pl.col("cluster") == selected)
        a, b = st.columns(2)
        with a:
            if not reference.is_empty():
                chart(
                    styled(
                        px.histogram(
                            reference.to_pandas(),
                            x="anomaly_ratio",
                            nbins=50,
                            log_y=True,
                            labels=LABELS,
                            title="Scores · reference sample",
                        )
                    )
                )
        with b:
            if not rows.is_empty():
                chart(
                    styled(
                        px.scatter(
                            categorical(rows.head(3000)).to_pandas(),
                            x="trip_distance",
                            y="duration_minutes",
                            color="cluster_label",
                            color_discrete_map=COLORS,
                            hover_data=["anomaly_ratio", "fare_per_mile"],
                            labels=LABELS,
                            title="Unusual training trips · up to 3,000",
                        )
                    )
                )
        st.caption(
            f"{rows.height:,} matching records. IDs identify positions in the training sample; they are not TLC trip identifiers."
        )
        st.dataframe(rows.head(200), hide_index=True)
        st.download_button(
            "Download matching anomalies", rows.write_csv(), "anomalies.csv", "text/csv"
        )

elif section == "Predict New Trips":
    st.write(
        "Assign unseen trips to the selected saved model. Existing clusters, scaler and UMAP reducers remain fixed."
    )
    with st.expander("Input columns & validation", expanded=True):
        st.write(
            "Required: tpep_pickup_datetime, tpep_dropoff_datetime, trip_distance, fare_amount, total_amount. Use NYC local timestamps, e.g. 2025-01-15 10:30:00. Extra columns are ignored. row_id preserves the zero-based position in your file."
        )
    upload = st.file_uploader(
        "CSV or Parquet · maximum 20 MB / 100,000 rows", type=["csv", "parquet"]
    )
    if upload is not None:
        try:
            if upload.size > 20 * 1024**2:
                raise ValueError("Upload exceeds the 20 MB limit.")
            with st.spinner("Validating and applying saved models…"):
                result, rejected, report = uploaded_predictions(
                    upload.getvalue(),
                    Path(upload.name).suffix.lower(),
                    str(root),
                    modified,
                    algorithm,
                )
            kpis(
                [
                    ("Input rows", report["raw_rows"]),
                    ("Accepted", report["accepted_rows"]),
                    ("Rejected", report["rejected_rows"]),
                ]
            )
            if rejected.height:
                st.warning(
                    "Rejected rows have explicit validation reasons and receive no cluster assignment."
                )
                st.dataframe(rejected, hide_index=True)
                st.download_button(
                    "Download rejected rows", rejected.write_csv(), "rejected.csv", "text/csv"
                )
            if result.height:
                result = result.with_columns(
                    pl.col("cluster")
                    .replace_strict(names, default="Profile unavailable")
                    .alias("cluster_description")
                )
                st.dataframe(result.head(100), hide_index=True)
                st.download_button(
                    "Download all accepted predictions",
                    result.write_csv(),
                    "predictions.csv",
                    "text/csv",
                )
                chart(distribution(result))
                combined = (
                    pl.concat(
                        [
                            reference.with_columns(pl.lit("Reference").alias("origin")),
                            result.with_columns(pl.lit("Uploaded").alias("origin")),
                        ],
                        how="diagonal_relaxed",
                    )
                    if not reference.is_empty()
                    else result.with_columns(pl.lit("Uploaded").alias("origin"))
                )
                dim = st.radio("Projection dimensions", [2, 3], horizontal=True)
                chart(projection(combined, dim))
                st.caption(
                    "Uploaded trips use diamond markers. At most 2,000 uploaded points are transformed; all accepted rows are exported. Blank UMAP coordinates indicate rows outside the visualization sample. No fitting occurs."
                )
            else:
                st.info("No rows passed validation. Correct the reported reasons and upload again.")
        except (ValueError, pl.exceptions.PolarsError) as error:
            st.error(str(error))

else:
    st.subheader("Data & feature engineering")
    st.write(
        "Official January 2025 NYC TLC Yellow Taxi trip records. Lazy Polars validation filters timestamps, non-finite values, impossible durations, non-positive distances, charges and implausible speeds. Each rejected row receives its first failing reason."
    )
    st.link_button("NYC TLC source", "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page")
    st.write("Features: " + ", ".join(bundle["features"]))
    st.write(
        "Duration comes from dropoff minus pickup; speed is distance / hours; fare intensity is fare / miles. Pickup hour uses sine and cosine to preserve the midnight boundary. Saved StandardScaler parameters are shared by both algorithms; float32 matrices limit memory."
    )
    st.subheader("Selection & experiments")
    st.write(
        selection.get("rationale", "This bundle has no matching selection evidence.")
        if selection_matches
        else "No matching selection evidence."
    )
    st.write(
        "Final models use the same training rows, K and scaler. Benchmarks use reproducible nested samples, fixed seeds and isolated workers. Silhouette uses at most 2,000 trips. ARI compares partitions without assuming IDs align. Full fitting is separate from Streamlit; unsafe runs are skipped explicitly. Free Colab is an optional fallback with CSV/JSON export."
    )
    with st.expander("Saved model configurations"):
        for name, model in bundle["models"].items():
            st.write(name)
            st.json(model.get_params())
    with st.expander("Validation thresholds & runtime details"):
        st.json(bundle["rules"])
        st.json(read_json(reports / "runtime.json"))
    st.caption(
        "Reproduce: taxi download → taxi clean → taxi train data/clean.parquet → taxi benchmark --full. See README and notebooks/full_dataset_benchmark_colab.ipynb. Benchmark environment labels must be retained when importing results."
    )
