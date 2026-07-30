# Fingerprint benchmark

This folder reproduces the non-embedding fingerprint benchmarks described in
Fig. 3 and Fig. S4: predicted-vs-true fingerprint-similarity RMSE and compound
library retrieval PR/ROC curves.

## Label definition

The binary retrieval label is **strictly** `true_tanimoto > 0.9`. InChIKey and
historical `exact` arrays are ignored. Consequently, `0.9` itself is negative.

## Generate the search-result PKLs

`generate_search_results.py` consolidates the retained
`FP_Search_local_sirius_new.py` workflow. It accepts any dataset/method through
command-line arguments instead of hard-coded server paths.

For each query and every reference candidate it calculates:

1. `score`: Tanimoto between the thresholded predicted query fingerprint and
   the candidate's true fingerprint. This is the retrieval score.
2. `sim`: Tanimoto between the query's true fingerprint and the candidate's
   true fingerprint. This is the ground truth used for RMSE and PR/ROC labels.
3. `exact`: equality of the first InChIKey block. This is retained solely for
   compatibility/auditing and is ignored by `benchmark.py`.

The prediction PKL must contain `mid` and `pred_fpr`. The query and reference
CSVs must contain `sirius_pubchem_fp`, `sirius_maccs_fp`, and
`pubchem_inchikey`; the query additionally needs `Spectrum_ID`. Predicted
probabilities are converted to bits with `> 0.5`, matching the existing code.

Example:

```powershell
.\.conda-env\python.exe .\FPBenchmark\generate_search_results.py `
  --method COMPASS `
  --fingerprints .\sfpr\MSNsDatabase_New500_Selected_0509_pred_sfpr.pkl `
  --query-csv .\MSG_500_0509.csv `
  --reference-csv .\AllCompound_filtered_new.csv `
  --output-dir .\FPBenchmark\search_results\MSGTestDataset
```

Repeat with the DreaMS and SIRIUS prediction PKLs, changing only `--method` and
`--fingerprints`. For TestDataset, CASMI 2022, and AntitumorDataset, substitute
their query CSV and prediction PKLs. A different reference library can be
selected with `--reference-csv`.

The output filename is exactly
`<METHOD>_sfpr_search_results.pkl`, containing flattened
`[score, sim, exact_audit_only]` NumPy arrays.

## Benchmark input

Each method supplies a pickle containing flattened arrays:

```text
[predicted_tanimoto, true_tanimoto]
```

Historical three-array files (`[..., ..., exact_inchikey]`) are supported, but
the third array is ignored. These are the compact outputs produced by the
existing fingerprint search scripts.

## Run

From the repository root:

```powershell
.\.conda-env\python.exe .\FPBenchmark\benchmark.py `
  --dataset CASMI2022 `
  --method COMPASS=CASMI/search_results/Compass_sfpr_search_results.pkl `
  --method DreaMS=CASMI/search_results/DreamMS_sfpr_search_results.pkl `
  --method SIRIUS=CASMI/search_results/Sirius_sfpr_search_results.pkl `
  --output-dir FPBenchmark/results/CASMI2022
```

To run several datasets, copy `manifest.example.json`, add TestDataset,
MSGTestDataset and AntitumorDataset paths, then run:

```powershell
.\.conda-env\python.exe .\FPBenchmark\run_all.py --manifest .\FPBenchmark\manifest.json
```

Outputs include machine-readable overall metrics, per-similarity-bin RMSE, and
SVG PR/ROC/RMSE plots. No embedding benchmark is run here.
