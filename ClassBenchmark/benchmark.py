"""Reproduce per-label ClassyFire classification benchmarks."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score


LEVELS = ("class", "superclass")
METRICS = ("accuracy", "precision", "recall", "f1", "mcc")
COLORS = ["#82B0D2", "#FA7F6F", "#8FD88F", "#8C6BB1"]


def assignment(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected METHOD=PATH") from exc
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--prediction", action="append", type=assignment, required=True,
                        metavar="METHOD=CSV_OR_PKL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--id-column", default="Spectrum_ID")
    parser.add_argument("--true-class-column", default="Class_Name")
    parser.add_argument("--true-superclass-column", default="SuperClass_Name")
    parser.add_argument("--unknown", default="Unknown")
    return parser.parse_args()


def normalize_id(value: object) -> str:
    """Normalize IDs used by the retained COMPASS/CANOPUS exports."""
    text = Path(str(value)).stem
    # CANOPUS commonly exports e.g. nan_123; COMPASS commonly uses 123_xxx.mgf.
    if re.match(r"^(nan|orb|qto)_", text, flags=re.I):
        text = text.split("_", 1)[1]
    return text.split("_", 1)[0]


def read_object(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".pkl", ".pickle"):
        with path.open("rb") as handle:
            value = pickle.load(handle)
        return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", dtype=str)


def prediction_frame(path: Path, id_column: str) -> pd.DataFrame:
    frame = read_object(path)
    id_candidates = [id_column, "id", "file", "mid", "filename"]
    source_id = next((column for column in id_candidates if column in frame), None)
    if source_id is None:
        raise ValueError(f"{path}: no ID column found among {id_candidates}")
    out = pd.DataFrame({id_column: frame[source_id].map(normalize_id)})

    direct = {
        "class": ["class", "Class_Name", "ClassyFire#class", "predicted_class"],
        "superclass": ["superclass", "SuperClass_Name", "ClassyFire#superclass", "predicted_superclass"],
    }
    for level, candidates in direct.items():
        column = next((name for name in candidates if name in frame), None)
        if column is not None:
            out[level] = frame[column].astype("string")

    # Wide probability exports can be used as one file per level. Infer the
    # level from the filename and choose the maximum-probability label.
    if not ({"class", "superclass"} & set(out.columns)):
        level = "superclass" if "superclass" in path.stem.lower() else "class"
        metadata = {source_id, "file", "mass", "mid", "filename", id_column}
        probabilities = frame[[c for c in frame.columns if c not in metadata]].apply(
            pd.to_numeric, errors="coerce"
        )
        if probabilities.shape[1] == 0 or probabilities.notna().sum().sum() == 0:
            raise ValueError(f"{path}: no direct label or probability columns found")
        out[level] = probabilities.idxmax(axis=1).astype("string")
    return out.drop_duplicates(id_column, keep="first")


def merge_method_files(items: list[tuple[str, Path]], id_column: str) -> dict[str, pd.DataFrame]:
    methods: dict[str, pd.DataFrame] = {}
    for method, path in items:
        current = prediction_frame(path, id_column)
        if method not in methods:
            methods[method] = current
            continue
        previous = methods[method]
        prediction_columns = (set(previous) & set(current)) - {id_column}
        if prediction_columns:
            # Multiple CANOPUS TSVs are disjoint instrument shards with the
            # same schema, so stack them rather than creating suffix columns.
            combined = pd.concat([previous, current], ignore_index=True)
            if combined[id_column].duplicated().any():
                duplicates = combined.loc[combined[id_column].duplicated(), id_column].unique()
                raise ValueError(f"{method}: duplicate IDs across prediction shards: {duplicates[:5]}")
            methods[method] = combined
        else:
            # Separate COMPASS class and superclass probability exports share
            # IDs but contribute different prediction columns.
            methods[method] = previous.merge(
                current, on=id_column, how="outer", validate="one_to_one"
            )
    return methods


def per_label_metrics(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> list[dict]:
    rows = []
    for label in labels:
        true_binary = (y_true == label).to_numpy()
        pred_binary = (y_pred == label).to_numpy()
        rows.append({
            "label": label,
            "support": int(true_binary.sum()),
            "accuracy": accuracy_score(true_binary, pred_binary),
            "precision": precision_score(true_binary, pred_binary, zero_division=0),
            "recall": recall_score(true_binary, pred_binary, zero_division=0),
            "f1": f1_score(true_binary, pred_binary, zero_division=0),
            "mcc": matthews_corrcoef(true_binary, pred_binary),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(args.truth, dtype=str).rename(columns={
        args.true_class_column: "class", args.true_superclass_column: "superclass"
    })
    required = {args.id_column, "class", "superclass"}
    if not required.issubset(truth):
        raise ValueError(f"truth file is missing {sorted(required - set(truth.columns))}")
    truth[args.id_column] = truth[args.id_column].map(normalize_id)
    if truth[args.id_column].duplicated().any():
        raise ValueError("truth IDs are not unique after normalization")
    methods = merge_method_files(args.prediction, args.id_column)
    all_rows, coverage_rows = [], []

    for level in LEVELS:
        eligible = truth[truth[level].notna() & (truth[level] != args.unknown)].copy()
        labels = sorted(eligible[level].unique())
        for method, predictions in methods.items():
            if level not in predictions:
                continue
            merged = eligible[[args.id_column, level]].merge(
                predictions[[args.id_column, level]], on=args.id_column, how="left",
                suffixes=("_true", "_pred"), validate="one_to_one"
            )
            y_true = merged[f"{level}_true"].astype(str)
            y_pred = merged[f"{level}_pred"].fillna(args.unknown).astype(str)
            coverage_rows.append({
                "dataset": args.dataset, "level": level, "method": method,
                "truth_count": len(merged), "predicted_count": int((y_pred != args.unknown).sum()),
                "accuracy_top1": accuracy_score(y_true, y_pred),
                "precision_macro": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                "mcc_multiclass": matthews_corrcoef(y_true, y_pred),
            })
            for row in per_label_metrics(y_true, y_pred, labels):
                all_rows.append({"dataset": args.dataset, "level": level, "method": method, **row})

    if not all_rows:
        raise ValueError("no method provided predictions for class or superclass")
    write_csv(args.output_dir / "per_label_metrics.csv", all_rows)
    write_csv(args.output_dir / "coverage_and_top1_accuracy.csv", coverage_rows)

    for level in LEVELS:
        for metric in METRICS:
            plt.figure(figsize=(7, 5))
            plotted = False
            bins = np.arange(-0.2, 1.01, 0.1) if metric == "mcc" else np.arange(0, 1.01, 0.1)
            for index, method in enumerate(methods):
                values = [r[metric] for r in all_rows if r["level"] == level and r["method"] == method]
                if values:
                    plt.hist(values, bins=bins, histtype="step", linewidth=2,
                             label=method, color=COLORS[index % len(COLORS)])
                    plotted = True
            if plotted:
                plt.xlabel(metric.upper() if metric == "mcc" else metric.capitalize())
                plt.ylabel(f"Number of {level} labels"); plt.legend(frameon=False)
                plt.tight_layout(); plt.savefig(args.output_dir / f"{level}_{metric}.svg")
            plt.close()


if __name__ == "__main__":
    main()
