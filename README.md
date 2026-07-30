# COMPASS

[![Web Demo](https://img.shields.io/badge/Service-Web_Demo-blue)](https://npcompass.xulab.cloud/) [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> We recommend using our [web server](#webservice) as it is easier and faster.
> 
---

## 🌟 Core Functionalities

COMPASS is a foundation model that supports diverse downstream tasks:

1. **Spectral Library Search**: Maps MS² spectra into 500-dimensional embeddings for rapid similarity search.
2. **Structural Classification**: Automatically categorizes compounds into chemical classes and superclasses.
3. **Molecular Fingerprint Prediction**: Decodes embeddings to predict molecular fingerprints.
4. **Compound Library Retrieval**: Retrieves analogous structures by querying databases with the predicted molecular fingerprints.
5. **MSNs**: A multi-parameter framework that integrates prediction metrics to assess the structural novelty of unknown compounds and prioritize "dark matter" for isolation.

---

## 📁 Resource Navigation

Please download the required resources and place them in the directories specified below:

| Folder/File | Description | Download Link |
| --- | --- | --- |
| `base_model` | Base model for predicting embeddings. | [Download](https://zenodo.org/records/16676832) |
| `class_model/class` | Model for class prediction. | [Download](https://zenodo.org/records/16739187) |
| `class_model/superclass` | Model for superclass prediction. | [Download](https://zenodo.org/records/16739195) |
| `fpr_model` | Model for molecular fingerprint prediction. | [Download](https://zenodo.org/records/16682503) |
| `fpr_database` | Database for molecular fingerprint based library retrieval. | [Download](https://zenodo.org/records/16679974) |

You can also download all the resources by running the following command:

```bash
bash download_resources.sh
```
---

## System requirements

### Hardware
*   **GPU**: Single GPU with at least **18GB VRAM** (e.g., NVIDIA RTX 3090/4090, A10, or V100).

### Software
*   **Operating System**: Ubuntu 20.04 LTS
*   **Python**: Version 3.8/3.10
*   **Dependencies**: Please refer to `requirements.txt` for specific package versions.

### Installation Time
*   **Dependencies**: Typically requires less than 30 minutes.
*   **Model Weights**: Download duration depends on your network connection speed.

---

## 🚀 Usage

### 1. Installation

```bash
git clone <https://github.com/MicroResearchLab/COMPASS.git>
cd COMPASS
pip install -r requirements.txt

```

### 2. Data Preparation

Place your input files in the `input/files` directory.

Recommended input layout:

```text
input/
├── files/
│   ├── sample_1.mgf
│   └── sample_2.mzxml
└── peaktable/
    └── sample_2.csv
```

Place `.mgf` files directly in `input/files/`. For `.mzxml` files, place the spectrum file in `input/files/` and the matching peak table CSV in `input/peaktable/`. The peak table should contain at least the `mz` and `rt` columns. During processing, the pipeline may create temporary MGF files in `input/tmp_mgf/`.

- **Supported formats:** `.mgf` or `.mzxml`.

### 3. Embedding Generation (Standalone)

If you only need to convert mass spectrometry files into spectral embeddings (saved in **Pickle `.pkl`** format), use the `Embedding.py` script.

**Note:** This script supports the **exact same command-line arguments** as `main.py` (see the *Configuration Parameters* table below). This allows you to apply the same preprocessing, filtering, and merging logic during embedding generation.

```bash
python Embedding.py  --inten_thresh 1 \
               --rt 30 \
               --ppm 20 \
               --msdelta 0.01 \
               --if_merge_samples_byenergy false \
               --min_mz_num 2 \
               --remove_precursor true

```

**Output:** The generated embeddings will be saved in **Pickle (`.pkl`)** format. The file stores a list of records, and each record includes the input file name, the generated embedding vector, and the precursor mass. Typically requires less than 10 minutes.

### 4. Main Analysis (Full Pipeline)

Run the main script with your desired parameters. Below is a standard usage example:

```bash
python main.py --filter_mass 2000 \
               --filter_formula false \
               --inten_thresh 1 \
               --rt 30 \
               --ppm 20 \
               --msdelta 0.01 \\
               --if_merge_samples_byenergy false \
               --min_mz_num 2 \
               --remove_precursor true \
               --output_num 200

```

### ⚙️ Configuration Parameters

You can customize the processing pipeline using the following command-line arguments:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--filter_mass` | `float` | `2000` | Relative molecular mass screening thresholds for pretreatment. |
| `--inten_thresh` | `float` | `1` | Intensity threshold for noise removal during preprocessing. |
| `--rt` | `float` | `30` | Retention time threshold for feature alignment (seconds). |
| `--ppm` | `float` | `20` | ppm precision tolerance for feature alignment. |
| `--msdelta` | `float` | `0.01` | msdelta threshold for merging spectra. |
| `--if_merge_samples_byenergy` | `bool` | `false` | merge spectra by different collision energy or merge all. |
| `--min_mz_num` | `float` | `2` | Minimum number of fragments required per spectrum. |
| `--remove_precursor` | `bool` | `true` | remove precursor ions during preprocessing. |
| `--output_num` | `float` | `200` | Number of most similar molecules to output in the results. |

### 5. Results

Outputs from `main.py` are generated in the `output/` folder with timestamped filenames:

- `output/<timestamp>-similarity-matching.csv`
- `output/<timestamp>-classification.csv`

Typically requires less than 10 minutes.

#### How to interpret the output

Each input precursor may produce multiple candidate rows. Read the results by grouping rows with the same source file and precursor, then compare the candidates within that group. Rankings and similarity values are most useful for prioritization; they are not, by themselves, definitive structure identifications.

##### Compound library retrieval (`<timestamp>-similarity-matching.csv`)

The exact column order may vary by release, but the output uses the following fields described by the COMPASS web service:

| Column | Meaning | Interpretation |
| --- | --- | --- |
| `file` | Source file associated with the query spectrum. | Use it together with `precursor` to trace a result back to the input feature. |
| `precursor` | Precursor-ion m/z of the query spectrum. | Rows with the same file and precursor belong to the same query feature. |
| `pred_fpr` | Molecular fingerprint predicted from the query MS² spectrum. | This is the model-derived structural feature representation used for library retrieval. |
| `inchikey` | InChIKey of a matched database compound. | Paste it into [PubChem](https://pubchem.ncbi.nlm.nih.gov/) to inspect the candidate structure and metadata. |
| `name` | Name of the matched compound. | Treat this as a candidate annotation and verify it with the remaining evidence. |
| `mass` | Exact mass of the matched compound. | Compare it with the mass implied by the precursor ion and the expected adduct. |
| `formula` | Molecular formula of the matched compound. | Check whether it agrees with the measured mass, ion mode, adduct, isotope pattern, and other experimental information. |
| `smiles` | SMILES representation of the matched compound. | Use it in PubChem or structure software such as ChemDraw to view the candidate structure. |
| `sim` | Fingerprint similarity between the predicted query fingerprint and the matched compound. | Larger values indicate greater fingerprint similarity; compare candidates from the same query feature. |
| `sim rank` | Rank of the candidate by fingerprint similarity. | Rank 1 is the highest-ranked candidate for that query feature. |

If spectral-library-search fields are present, interpret them as follows:

| Column | Meaning | Interpretation |
| --- | --- | --- |
| `sid` | Query identifier containing the m/z and source file. | Use it to group and trace results to the input spectrum. |
| `Precursor` | Precursor-ion m/z. | Confirm that the candidate is being compared with the intended feature. |
| `Cosine Score` | Cosine-based comparison between the query and library spectral embeddings. | Use it to rank library matches within the same query; inspect the implementation/version before applying a fixed cutoff because some exports label similarity or distance differently. |
| `PubChem_InChIkey` | InChIKey of the matched compound. | Look it up in PubChem and verify the proposed identity with orthogonal evidence. |

##### Structural classification (`<timestamp>-classification.csv`)

COMPASS predicts both ClassyFire **Superclass** and **Class** labels. Depending on the release, the two levels may be stored in one timestamped file or in separate `superclass` and `class` CSV files.

| Column | Meaning | Interpretation |
| --- | --- | --- |
| `file` | Source file associated with the query feature. | Use it together with `precursor` to locate the original spectrum. |
| `precursor` | Precursor-ion m/z. | Identifies the feature being classified. |
| `predict` | Most likely Class or Superclass assignment. | This is the primary predicted label at the level represented by the file. |
| Other class columns | Scores or overlapping predictions for other Class/Superclass categories. | Review them when several categories receive similar support; a close result is less decisive than a clearly dominant prediction. |

Use the classification as supporting structural evidence rather than a complete compound identification. Agreement among the predicted class, library candidate, exact mass, formula, and experimental context increases confidence.

### 6. Metabolite Structural Novelty Score (MSNs)

After generating the similarity and classification results via `main.py`, you can utilize the **Metabolite Structural Novelty Score (MSNs)** to identify potentially novel compounds.

**Command:**

```bash
python MSNs.py \
  --sim_score_file_path output/<timestamp>-similarity-matching.csv \
  --class_results_file_path output/<timestamp>-classification.csv

```

> Note: Replace <timestamp> with the actual timestamp string found in your output/ directory filenames.
> 

**Parameters:**

| Parameter | Description |
| --- | --- |
| `--sim_score_file_path` | Path to the **Compound Library Retrieval** results CSV file generated by `main.py`. |
| `--class_results_file_path` | Path to the **Structural Classification** result Json file generated by `main.py`. |

Outputs from `MSNs.py` are generated in the `output/` folder with timestamped filenames:

- `output/<timestamp>-distance_results.csv`

Typically requires less than 10 minutes.

#### How to interpret the MSNs output

MSNs combines molecular-fingerprint similarity, structural-class information, and exact-mass deviation to prioritize compounds by structural novelty.

| Column | Meaning | Interpretation |
| --- | --- | --- |
| `sid` | Query identifier containing the m/z and source file. | Links the score to the original input feature. |
| `Precursor` | Precursor-ion m/z. | Identifies the scored feature. |
| `Fingerprint_sim` | Tanimoto similarity between the query's predicted fingerprint and the matched compound. | Larger values mean greater structural-feature similarity to the database candidate. |
| `PubChem_InChIkey` | InChIKey of the matched compound. | Use PubChem to inspect the matched reference structure. |
| `PubChem_SMILES` | SMILES of the matched compound. | Use it to visualize or analyze the reference structure. |
| `PubChem_Exact_Mass` | Exact mass of the matched compound. | Compare with the precursor-derived neutral mass using the correct ion/adduct assignment. |
| `COMPASS_Class_Result` | Highest-confidence predicted structural class. | Check whether the class is chemically consistent with the matched candidate and sample context. |
| `ABS (match-true)` | Absolute exact-mass deviation between the matched structure and query. | Smaller deviations indicate closer mass agreement. |
| `Class_P` | Classification confidence. | Larger values indicate stronger support for the reported class. |
| `Lambda` | A value in `(0, 1]` that nonlinearly scales exact-mass deviation and is especially sensitive to small deviations. | Interpret it together with the raw mass deviation rather than as an independent identification score. |
| `MSNs` | Metabolite Structural Novelty Score. | Use it as a relative prioritization score across features from the same analysis, then examine its component evidence and validate promising features experimentally. |

For practical review, start with the features prioritized by `MSNs`, inspect `Fingerprint_sim`, `ABS (match-true)`, and `Class_P`, then verify the proposed structures using precursor/adduct assignment, MS/MS fragments, retention behavior, authentic standards, and biological context where available.

## <span id="webservice"> 🌐 Web Service </span>

An online demo is available for quick validation of small batches:👉  [Access COMPASS](https://npcompass.xulab.cloud/) (or use this [link](https://npcompass.zju.edu.cn/) for Mainland China).
