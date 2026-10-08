"""Small reproducible command-line entry points."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--out", default="artifacts/demo")
    train = sub.add_parser("train")
    train.add_argument("path")
    train.add_argument("--out", default="artifacts/real")
    train.add_argument("--rows", type=int, default=100000)
    train.add_argument("--config", default="config/default.json")
    sub.add_parser("download")
    clean = sub.add_parser("clean")
    clean.add_argument("path", nargs="?", default="data/yellow_tripdata_2025-01.parquet")
    clean.add_argument("--config", default="config/default.json")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--artifacts", default="artifacts/real")
    bench.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.command == "demo":
        from taxi.fixture import synthetic
        from taxi.model import fit_bundle

        Path("data").mkdir(exist_ok=True)
        fixture = synthetic(300)
        fixture.write_csv("data/synthetic_fixture.csv")
        synthetic(30, 7).write_csv("data/synthetic_unseen.csv")
        synthetic(30, 7).write_parquet("data/synthetic_unseen.parquet")
        _, metadata = fit_bundle(fixture, args.out, source="SYNTHETIC DEVELOPMENT FIXTURE")
        print(json.dumps(metadata, indent=2))
    elif args.command == "benchmark":
        from taxi.benchmark import benchmark

        benchmark(args.artifacts, args.full)
    else:
        from taxi.data import scan, Rules
        from taxi.experiment import clean_data, download, train_real

        if args.command == "download":
            download()
        elif args.command == "clean":
            config = json.loads(Path(args.config).read_text())
            clean_data(args.path, Rules(**config["rules"]))
        elif args.command == "train":
            config = json.loads(Path(args.config).read_text())
            train_real(scan(args.path), args.out, args.rows, config)


if __name__ == "__main__":
    main()
