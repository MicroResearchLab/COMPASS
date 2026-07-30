"""Generate ``*_sfpr_search_results.pkl`` from predicted fingerprints.

For every query/candidate pair this writes:

* score: Tanimoto(predicted query fingerprint, candidate true fingerprint)
* sim:   Tanimoto(query true fingerprint, candidate true fingerprint)
* exact: first-block InChIKey equality, retained only as an audit field

The downstream benchmark deliberately defines labels from ``sim > 0.9`` only.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, help="Name used in the output filename")
    parser.add_argument("--fingerprints", required=True, type=Path,
                        help="PKL containing mid and pred_fpr")
    parser.add_argument("--query-csv", required=True, type=Path)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--id-column", default="Spectrum_ID")
    parser.add_argument("--prediction-id-column", default="mid")
    parser.add_argument("--prediction-column", default="pred_fpr")
    parser.add_argument("--pubchem-column", default="sirius_pubchem_fp")
    parser.add_argument("--maccs-column", default="sirius_maccs_fp")
    parser.add_argument("--inchikey-column", default="pubchem_inchikey")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold applied to predicted fingerprint probabilities")
    return parser.parse_args()


def load_pickle_frame(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)


def bitstrings(frame: pd.DataFrame, pubchem: str, maccs: str) -> np.ndarray:
    values = (frame[pubchem].astype(str) + frame[maccs].astype(str)).tolist()
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError(f"fingerprint lengths are inconsistent: {sorted(lengths)}")
    if not values or next(iter(lengths)) == 0:
        raise ValueError("fingerprints are empty")
    if any(set(value) - {"0", "1"} for value in values):
        raise ValueError("true fingerprints must be binary bit strings")
    return np.stack([np.frombuffer(value.encode("ascii"), dtype="S1") == b"1" for value in values])


def tanimoto(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(query, candidates).sum(axis=1)
    union = np.logical_or(query, candidates).sum(axis=1)
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)


def key_block(series: pd.Series) -> np.ndarray:
    return series.astype("string").str.split("-").str[0].fillna("").to_numpy()


def main() -> None:
    args = parse_args()
    true_columns = [args.pubchem_column, args.maccs_column, args.inchikey_column]
    query = pd.read_csv(args.query_csv, usecols=[args.id_column, *true_columns], dtype=str)
    reference = pd.read_csv(args.reference_csv, usecols=true_columns, dtype=str)
    if query.empty or reference.empty:
        raise ValueError("query and reference tables must not be empty")
    if query[args.id_column].duplicated().any():
        raise ValueError("query IDs must be unique")

    # Match the retained historical workflow: one candidate per connectivity
    # block of the InChIKey. This de-duplication affects candidate membership,
    # but never the binary benchmark label.
    reference = reference.assign(_key=key_block(reference))
    reference = reference.drop_duplicates("_key", keep="first").reset_index(drop=True)
    candidate_fp = bitstrings(reference, args.pubchem_column, args.maccs_column)
    query_fp = bitstrings(query, args.pubchem_column, args.maccs_column)
    query_keys = key_block(query[args.inchikey_column])
    query_lookup = {
        str(identifier): (query_fp[index], query_keys[index])
        for index, identifier in enumerate(query[args.id_column])
    }

    predictions = load_pickle_frame(args.fingerprints)
    needed = {args.prediction_id_column, args.prediction_column}
    if not needed.issubset(predictions):
        raise ValueError(f"prediction PKL is missing {sorted(needed - set(predictions.columns))}")
    predictions = predictions[[args.prediction_id_column, args.prediction_column]].drop_duplicates(
        args.prediction_id_column, keep="first"
    )
    unknown = sorted(set(predictions[args.prediction_id_column].astype(str)) - set(query_lookup))
    if unknown:
        raise ValueError(f"prediction IDs absent from query CSV (first 10): {unknown[:10]}")

    scores_all, similarities_all, exact_all = [], [], []
    fp_length = candidate_fp.shape[1]
    for row in tqdm(predictions.itertuples(index=False, name=None), total=len(predictions)):
        identifier, predicted = str(row[0]), row[1]
        true_fp, true_key = query_lookup[identifier]
        predicted_fp = np.asarray(predicted, dtype=np.float32).ravel()
        if predicted_fp.shape != (fp_length,):
            raise ValueError(f"{identifier}: predicted shape {predicted_fp.shape}, expected {(fp_length,)}")
        scores_all.append(tanimoto(predicted_fp > args.threshold, candidate_fp))
        similarities_all.append(tanimoto(true_fp, candidate_fp))
        exact_all.append(reference["_key"].to_numpy() == true_key)

    if not scores_all:
        raise ValueError("no predictions were processed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.method}_sfpr_search_results.pkl"
    with output.open("wb") as handle:
        pickle.dump(
            [np.concatenate(scores_all), np.concatenate(similarities_all), np.concatenate(exact_all)],
            handle, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Wrote {output}")
    print(f"Queries: {len(predictions)}; candidates: {len(reference)}; pairs: {len(predictions) * len(reference)}")
    print("Array order: [predicted_tanimoto, true_tanimoto, exact_inchikey_audit_only]")


if __name__ == "__main__":
    main()
