"""Small reproducible command-line entry points."""

import argparse
import json
from pathlib import Path

from taxi.data import scan
from taxi.fixture import synthetic
from taxi.model import fit_bundle


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--out", default="artifacts/demo")
    train = sub.add_parser("train")
    train.add_argument("path")
    train.add_argument("--out", default="artifacts/real")
    train.add_argument("--rows", type=int, default=100000)
    sub.add_parser("download")
    clean = sub.add_parser("clean")
    clean.add_argument("path", nargs="?", default="data/yellow_tripdata_2025-01.parquet")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--artifacts", default="artifacts/real")
    bench.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.command == "demo":
        Path("data").mkdir(exist_ok=True)
        fixture = synthetic(300)
        fixture.write_csv("data/synthetic_fixture.csv")
        synthetic(30, 7).write_csv("data/synthetic_unseen.csv")
        synthetic(30, 7).write_parquet("data/synthetic_unseen.parquet")
        _, metadata = fit_bundle(fixture, args.out, source="SYNTHETIC DEVELOPMENT FIXTURE")
        print(json.dumps(metadata, indent=2))
    else:
        from taxi.experiment import benchmark, clean_data, download, train_real

        if args.command == "download":
            download()
        elif args.command == "clean":
            clean_data(args.path)
        elif args.command == "train":
            train_real(scan(args.path), args.out, args.rows)
        elif args.command == "benchmark":
            benchmark(args.artifacts, args.full)


if __name__ == "__main__":
    main()
