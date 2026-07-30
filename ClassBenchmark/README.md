# Structural classification benchmark

This folder reproduces the non-embedding structural classification benchmark
in Fig. 4 and Table S1. It evaluates top-1 ClassyFire predictions at both class
and superclass levels, comparing COMPASS with CANOPUS using per-label Accuracy,
Precision, Recall, F1 and Matthews correlation coefficient (MCC). The summary
CSV also reports top-1 accuracy, macro Precision/Recall/F1 and multiclass MCC.

Missing/failed predictions (including the manuscript's one-hour CANOPUS
timeouts) are evaluated as `Unknown`; they are not silently dropped. Ground
truth rows whose class itself is missing or `Unknown` are excluded.

## Accepted inputs

The truth CSV needs `Spectrum_ID`, `Class_Name`, and `SuperClass_Name` (column
names are configurable). Prediction files may be CSV/TSV/PKL and may contain:

- direct `class` / `superclass` columns;
- CANOPUS `ClassyFire#class` / `ClassyFire#superclass` columns; or
- a wide COMPASS probability table. For wide tables, include `superclass` in
  the superclass filename; the maximum-probability column is the prediction.

Supply two probability PKLs with the same method name to merge class and
superclass outputs.

## Run

```powershell
.\.conda-env\python.exe .\ClassBenchmark\benchmark.py `
  --dataset MSGTestDataset `
  --truth path/to/AllMSGWithClass.csv `
  --prediction COMPASS=path/to/MSG_class.pkl `
  --prediction COMPASS=path/to/MSG_superclass.pkl `
  --prediction CANOPUS=path/to/nan_canopus_formula_summary.tsv `
  --prediction CANOPUS=path/to/orb_canopus_formula_summary.tsv `
  --prediction CANOPUS=path/to/qto_canopus_formula_summary.tsv `
  --output-dir ClassBenchmark/results/MSGTestDataset
```

Repeat for TestDataset with its truth and prediction files. The output contains
per-label metrics, aggregate/coverage metrics, and the manuscript-style
metric-distribution SVGs. This workflow does not run an embedding benchmark.
