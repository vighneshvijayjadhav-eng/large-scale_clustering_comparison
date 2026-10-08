"""Run in a Colab clone; exports reports only, never datasets or models."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from taxi.benchmark import benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-sizes",
        action="store_true",
        help="Also rerun local sizes and batch study in this same Colab runtime",
    )
    args = parser.parse_args()
    output = Path("reports/colab")
    output.mkdir(parents=True, exist_ok=True)
    previous = [path for path in output.iterdir() if path.suffix in {".csv", ".json", ".html"}]
    if previous:
        archive_dir = output / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive_dir.mkdir(parents=True)
        for path in previous:
            path.replace(archive_dir / path.name)
    try:
        subprocess.run([sys.executable, "-m", "taxi.colab"], check=True)
        benchmark(
            "artifacts/colab",
            full=True,
            environment="colab",
            output_dir=output,
            full_only=not args.all_sizes,
        )
        for name in ["provenance.json", "frozen_preprocessing.json"]:
            (output / name).write_bytes((Path("reports") / name).read_bytes())
        status = {
            "status": "finished",
            "note": "Inspect individual row statuses; finished does not mean every estimator fit succeeded.",
        }
    except (subprocess.CalledProcessError, OSError, ValueError) as error:
        status = {
            "status": "failed",
            "detail": str(error),
            "note": "Inspect notebook output. No data-size reduction was performed.",
        }
    (output / "workflow_status.json").write_text(json.dumps(status, indent=2))
    with zipfile.ZipFile("colab_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in output.iterdir():
            if path.suffix in {".csv", ".json", ".html"}:
                archive.write(path, path.name)
    print(json.dumps(status, indent=2))
    print("Download colab_results.zip; extract to reports/colab in the local project.")
    if status["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
