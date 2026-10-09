"""Render publication figures from saved, bounded project evidence only."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("artifacts/report_dependencies").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)
COLORS = ["#7C3AED", "#0D9488", "#D97706"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 220,
    }
)


def csvread(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8-sig")))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def diagram(name, labels, edges, positions):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis("off")
    boxes = []
    for i, text in enumerate(labels):
        x, y = positions[i]
        box = ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=.6", facecolor="#EEF2F6", edgecolor="#64748B"),
        )
        boxes.append(box.get_bbox_patch())
    for a, b in edges:
        x, y = positions[a]
        u, v = positions[b]
        ax.annotate(
            "",
            xy=(u, v),
            xytext=(x, y),
            arrowprops=dict(
                arrowstyle="->",
                color="#64748B",
                patchA=boxes[a],
                patchB=boxes[b],
                shrinkA=4,
                shrinkB=4,
            ),
        )
    save(fig, name)


def main():
    equations = [
        r"d(x,y)=\sqrt{\sum_{j=1}^{d}(x_j-y_j)^2}",
        r"z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}",
        r"J=\sum_{i=1}^{n}\min_{c\in\{1,\ldots,K\}}\Vert z_i-\mu_c\Vert^2",
        r"s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}",
        r"c_i=\arg\min_c\Vert z_i-\mu_c\Vert^2,\quad\mu_c=\frac{1}{|C_c|}\sum_{i\in C_c}z_i",
        r"v_c\leftarrow v_c+1,\quad\mu_c\leftarrow(1-1/v_c)\mu_c+(1/v_c)z_i",
    ]
    for number, equation in enumerate(equations, 1):
        fig = plt.figure(figsize=(7, 0.55))
        fig.text(0.5, 0.5, "$" + equation + "$", ha="center", va="center", fontsize=14)
        fig.savefig(OUT / f"equation_{number}.png", bbox_inches="tight", dpi=240)
        plt.close(fig)
    k = csvread("reports/k_selection.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key, title in zip(
        axes,
        ["normalized_inertia", "silhouette"],
        ["Inertia per trip", "Sampled silhouette"],
        strict=True,
    ):
        ax.plot([int(r["k"]) for r in k], [float(r[key]) for r in k], "o-", color=COLORS[0])
        ax.axvline(3, color=COLORS[2], ls="--", label="Selected K=3")
        ax.set(xlabel="Number of clusters K", ylabel=title, xticks=range(2, 11))
        ax.grid(alpha=0.2)
        ax.legend()
    save(fig, "selection")
    bench = csvread("reports/benchmarks.csv")
    regular = [
        r
        for r in bench
        if r["status"] == "ok" and r["streaming"] == "false" and int(r["rows"]) <= 1000000
    ]
    for name, metrics in [
        (
            "performance",
            [("fit_seconds", "Training time (s)"), ("peak_rss_mb", "Peak process-tree RSS (MiB)")],
        ),
        (
            "quality",
            [("normalized_inertia", "Inertia per trip"), ("silhouette", "Sampled silhouette")],
        ),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, (key, label) in zip(axes, metrics, strict=True):
            for algo, color in zip(["KMeans", "MiniBatchKMeans"], COLORS, strict=False):
                rows = [r for r in regular if r["algorithm"] == algo]
                ax.plot(
                    [int(r["rows"]) / 1000 for r in rows],
                    [float(r[key]) for r in rows],
                    "o-",
                    label=algo,
                    color=color,
                )
            ax.set(xlabel="Rows (thousands)", ylabel=label)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        save(fig, name)
    batch = csvread("reports/batch_study.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key, label in zip(
        axes,
        ["fit_seconds", "normalized_inertia"],
        ["Training time (s)", "Inertia per trip"],
        strict=True,
    ):
        ax.plot(
            [int(r["batch_size"]) for r in batch],
            [float(r[key]) for r in batch],
            "o-",
            color=COLORS[1],
        )
        ax.set(xscale="log", xlabel="MiniBatch batch size", ylabel=label)
        ax.grid(alpha=0.2)
    save(fig, "batch")
    profiles = csvread("reports/profiles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, algo in zip(axes, ["KMeans", "MiniBatchKMeans"], strict=True):
        rows = [r for r in profiles if r["algorithm"] == algo]
        v = [int(r["size"]) for r in rows]
        ax.bar(["C0", "C1", "C2"], v, color=COLORS)
        ax.set(title=algo, ylabel="Training trips")
        for i, n in enumerate(v):
            ax.text(i, n + 900, f"{n:,}\n{n / 1000:.1f}%", ha="center", fontsize=9)
        ax.set_ylim(0, 65000)
    save(fig, "distribution")
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5))
    own = [r for r in profiles if r["algorithm"] == "KMeans"]
    for ax, key, label in zip(
        axes.flat,
        ["trip_distance", "duration_minutes", "speed_mph", "fare_per_mile"],
        ["Distance (miles)", "Duration (minutes)", "Speed (mph)", "Fare per mile ($)"],
        strict=True,
    ):
        ax.bar(["C0", "C1", "C2"], [float(r[key]) for r in own], color=COLORS)
        ax.set_ylabel(label)
    save(fig, "profiles")
    sample = csvread("artifacts/report_inputs/sample_KMeans.csv")
    fig = plt.figure(figsize=(10, 4))
    a = fig.add_subplot(121)
    b = fig.add_subplot(122, projection="3d")
    for cluster, color in enumerate(COLORS):
        rows = [r for r in sample if int(r["cluster"]) == cluster]
        a.scatter(
            [float(r["umap2_1"]) for r in rows],
            [float(r["umap2_2"]) for r in rows],
            s=2,
            alpha=0.55,
            c=color,
            label=f"C{cluster}",
        )
        b.scatter(
            [float(r["umap3_1"]) for r in rows],
            [float(r["umap3_2"]) for r in rows],
            [float(r["umap3_3"]) for r in rows],
            s=2,
            alpha=0.5,
            c=color,
            label=f"C{cluster}",
        )
    a.set(xlabel="UMAP 1", ylabel="UMAP 2", title="Saved 2D projection")
    b.set(xlabel="UMAP 1", ylabel="UMAP 2", zlabel="UMAP 3", title="Saved 3D projection")
    a.legend(markerscale=3)
    b.view_init(25, 35)
    save(fig, "umap")
    anomalies = csvread("artifacts/report_inputs/anomalies_KMeans.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    threshold = json.loads(Path("reports/model_metadata.json").read_text())["anomalies"]["KMeans"][
        "threshold"
    ]
    axes[0].hist(
        [float(r["centroid_distance"]) / threshold for r in sample],
        bins=60,
        color=COLORS[0],
        log=True,
    )
    axes[0].axvline(1, color=COLORS[2], ls="--")
    axes[0].set(
        xlabel="Distance / threshold",
        ylabel="Reference trips (log count)",
        title="5,000 reference trips",
    )
    for cluster, color in enumerate(COLORS):
        rows = [r for r in anomalies if int(r["cluster"]) == cluster]
        axes[1].scatter(
            [float(r["trip_distance"]) for r in rows],
            [float(r["duration_minutes"]) for r in rows],
            s=7,
            alpha=0.5,
            c=color,
            label=f"C{cluster}",
        )
    axes[1].set(
        xlabel="Distance (miles)", ylabel="Duration (minutes)", title="All 1,000 training outliers"
    )
    axes[1].legend()
    save(fig, "anomalies")
    diagram(
        "preprocessing",
        [
            "Official January Parquet\n3,475,226 rows",
            "Lazy schema and\ntimestamp validation",
            "Exclusive rejection reasons\n233,646 rows",
            "Duration, distance, charges\nand speed checks",
            "Six engineered features\nfloat32 matrix",
            "Cleaned Parquet\n3,241,580 rows",
        ],
        [(0, 1), (1, 2), (1, 3), (3, 2), (3, 4), (4, 5)],
        [(2, 5), (5, 5), (8, 3), (5, 3), (5, 1), (8, 1)],
    )
    diagram(
        "architecture",
        [
            "TLC source + provenance",
            "Polars cleaning\nand feature engineering",
            "Selection and final fitting\nKMeans + MiniBatch",
            "Saved bundle\nscaler, models, UMAP",
            "Isolated benchmark workers\nlocal / optional Colab",
            "Streamlit analytics\nand upload prediction",
        ],
        [(0, 1), (1, 2), (2, 3), (1, 4), (3, 4), (3, 5), (4, 5)],
        [(2, 5), (5, 5), (8, 5), (8, 2), (2, 2), (5, 2)],
    )
    diagram(
        "inference",
        [
            "CSV / Parquet upload",
            "Validate + preserve\nsource row positions",
            "Rejected rows\nreason + download",
            "Saved scaler.transform\nmodel.predict",
            "Bounded saved UMAP\ntransform only",
            "All accepted predictions\nCSV download",
        ],
        [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (4, 5)],
        [(2, 5), (5, 5), (8, 5), (5, 3), (2, 1), (8, 1)],
    )


if __name__ == "__main__":
    main()
