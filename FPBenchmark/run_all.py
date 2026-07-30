"""Run every fingerprint dataset declared in a JSON manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    benchmark = Path(__file__).with_name("benchmark.py")
    for dataset, methods in manifest["datasets"].items():
        command = [sys.executable, str(benchmark), "--dataset", dataset,
                   "--output-dir", str(args.output_dir / dataset),
                   "--similarity-threshold", str(manifest.get("similarity_threshold", 0.9))]
        for method, path in methods.items():
            command += ["--method", f"{method}={path}"]
        print("Running", dataset)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
