"""Reproduce fingerprint RMSE and library-retrieval PR/ROC benchmarks.

Input files are compact pickle files containing flattened arrays in either
``[predicted_similarity, true_similarity]`` or
``[predicted_similarity, true_similarity, exact_inchikey]`` form.  The third
array is accepted for compatibility with historical files but is deliberately
ignored when constructing retrieval labels.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


COLORS = ["#73AAD2", "#8FD88F", "#FFA0BE", "#8C6BB1", "#E6AB02"]


def parse_method(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from exc
    if not name or not path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", action="append", required=True, type=parse_method,
        metavar="NAME=PKL", help="May be supplied once per method.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def load_compact(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        values = pickle.load(handle)
    if not isinstance(values, (list, tuple)) or len(values) not in (2, 3):
        raise ValueError(f"{path}: expected a 2- or 3-array compact pickle")
    scores = np.asarray(values[0], dtype=np.float64).ravel()
    similarities = np.asarray(values[1], dtype=np.float64).ravel()
    if scores.shape != similarities.shape:
        raise ValueError(f"{path}: score and true-similarity shapes differ")
    finite = np.isfinite(scores) & np.isfinite(similarities)
    if not finite.all():
        print(f"{path}: dropping {int((~finite).sum())} non-finite pairs")
    return scores[finite], similarities[finite]


def rmse_by_bin(scores: np.ndarray, truth: np.ndarray, edges: np.ndarray) -> list[dict]:
    rows = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (truth >= low) & ((truth <= high) if index == len(edges) - 2 else (truth < high))
        error = scores[selected] - truth[selected]
        rows.append({
            "bin_lower": low,
            "bin_upper": high,
            "count": int(selected.sum()),
            "rmse": float(np.sqrt(np.mean(error * error))) if error.size else np.nan,
        })
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 <= args.similarity_threshold <= 1:
        raise ValueError("--similarity-threshold must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0, 1, args.bins + 1)
    rmse_rows, metric_rows, curves = [], [], {}

    for name, path in args.method:
        scores, truth = load_compact(path)
        # Manuscript definition: positive iff true fingerprint Tanimoto > 0.9.
        # InChIKey/exact-match annotations are never combined with this label.
        labels = truth > args.similarity_threshold
        if labels.all() or not labels.any():
            raise ValueError(f"{name}: PR/ROC requires both positive and negative pairs")
        precision, recall, _ = precision_recall_curve(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        curves[name] = (precision, recall, fpr, tpr)
        binned = rmse_by_bin(scores, truth, edges)
        for row in binned:
            rmse_rows.append({"dataset": args.dataset, "method": name, **row})
        metric_rows.append({
            "dataset": args.dataset,
            "method": name,
            "pairs": len(scores),
            "positives": int(labels.sum()),
            "similarity_threshold": args.similarity_threshold,
            "label_rule": "true_tanimoto > threshold",
            "overall_rmse": float(np.sqrt(np.mean((scores - truth) ** 2))),
            "pr_auc": float(auc(recall, precision)),
            "average_precision": float(average_precision_score(labels, scores)),
            "roc_auc": float(auc(fpr, tpr)),
        })

    write_csv(args.output_dir / "rmse_by_similarity_bin.csv", list(rmse_rows[0]), rmse_rows)
    write_csv(args.output_dir / "metrics.csv", list(metric_rows[0]), metric_rows)

    for kind in ("pr", "roc"):
        plt.figure(figsize=(6, 5))
        for index, (name, curve) in enumerate(curves.items()):
            precision, recall, fpr, tpr = curve
            x, y = (recall, precision) if kind == "pr" else (fpr, tpr)
            plt.plot(x, y, color=COLORS[index % len(COLORS)], label=name)
        if kind == "roc":
            plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("Recall" if kind == "pr" else "False positive rate")
        plt.ylabel("Precision" if kind == "pr" else "True positive rate")
        plt.xlim(0, 1); plt.ylim(0, 1.05); plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(args.output_dir / f"{kind}_curve.svg"); plt.close()

    plt.figure(figsize=(6, 5))
    for index, name in enumerate(curves):
        rows = [r for r in rmse_rows if r["method"] == name]
        centers = [(r["bin_lower"] + r["bin_upper"]) / 2 for r in rows]
        plt.plot(centers, [r["rmse"] for r in rows], marker="o", label=name,
                 color=COLORS[index % len(COLORS)])
    plt.xlabel("True fingerprint Tanimoto similarity"); plt.ylabel("RMSE")
    plt.xlim(0, 1); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(args.output_dir / "rmse_by_similarity_bin.svg"); plt.close()


if __name__ == "__main__":
    main()
