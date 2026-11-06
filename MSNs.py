import json
from scipy.special import softmax
import pandas as pd
import numpy as np
import datetime
import argparse
PROTON_MASS = 1.007276466621

def method1_softmax_div_max(row):
    meta_data = row.values.astype(float)
    softmax_result = softmax(meta_data)
    return softmax_result

def bell(x):
    beta = 1.7147581782601855
    fwhm = 0.6

    x = np.asarray(x, dtype=np.float64)
    return np.exp(-np.log(2) * (np.abs(x) / fwhm) ** beta)

def distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.linalg.norm(vec1 - vec2)

def add_part_3(data_df:pd.DataFrame, class_dict:dict, class_p:dict):
    data_df = data_df.reset_index(drop=True)
    data_df["sid"] = [
        str(file).replace("\\", "/").split("/")[-1].split(".")[0]
        + "_" + str(float(mass))
        for file, mass in zip(data_df["file"], data_df["mass"])
    ]
    data_df["short_key"] = data_df["pubchem_inchikey"].apply(
        lambda x: x.split("-")[0] if isinstance(x, str) else ""    )
    data_df["short_key"] = data_df["short_key"].astype(str)
    data_df["Hit-class"] = data_df["short_key"].apply(
        lambda x: class_dict.get(x, 'Unknown')
    )
    data_df["class_p"] = data_df["Hit-class"].apply(
        lambda x: class_p.get(x, 1e-20)
    )
    neutral_mass = data_df["mass"] - PROTON_MASS
    data_df["ABS(match-true)"] = (data_df["pubchem_mass"] - neutral_mass).abs() / data_df["mass"]
    data_df["lambda"] = bell(data_df["ABS(match-true)"].to_numpy())


    for data_df_index in data_df.index:
        sim = data_df.loc[data_df_index, "sim"]
        class_p = data_df.loc[data_df_index, "class_p"]
        lambda_mass = data_df.loc[data_df_index, "lambda"]
        que = np.array([ sim, class_p , lambda_mass])
        data_df.loc[data_df_index,"MSNs"] = distance(que, np.array([1, 1, 1]))

    data_df = data_df.drop(columns=[ "short_key"])

    return data_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_score_file_path", required=True)
    parser.add_argument("--class_results_file_path", required=True)
    params = parser.parse_args()

    # Keep spectrum indices as strings for matching.
    sim_score_df = pd.read_csv(
        params.sim_score_file_path,
        dtype={"file": str, "index": str}
    )
    class_df = pd.read_csv(
        params.class_results_file_path,
        dtype={"file": str, "index": str}
    )

    if "class_probabilities" not in class_df.columns:
        raise ValueError(
            "Classification CSV has no class_probabilities column. "
            "Rerun main.py after adding probability export."
        )

    if class_df.duplicated(["file", "index"]).any():
        raise ValueError("Duplicate spectrum keys in classification CSV.")

    class_lookup = class_df.set_index(
        ["file", "index"]
    )["class_probabilities"].to_dict()

    with open("compound_class.json", "r") as f:
        compound_class = json.load(f)

    sim_score_df = sim_score_df.rename(columns={
        "inchikey": "pubchem_inchikey",
        "mass": "pubchem_mass",
        "precursor": "mass",
        "pred_sim": "sim"
    })

    result_parts = []
    for key, group in sim_score_df.groupby(
        ["file", "index"], sort=False, dropna=False
    ):
        if key not in class_lookup:
            raise ValueError(
                f"No classification probabilities for spectrum {key}"
            )
        class_p = json.loads(class_lookup[key])
        result_parts.append(
            add_part_3(group.copy(), compound_class, class_p)
        )

    if not result_parts:
        raise ValueError("Similarity CSV contains no candidate rows.")

    results = pd.concat(result_parts, ignore_index=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H")
    results = results[["sid","mass","file","index","pubchem_inchikey","pubchem_mass","formula","smiles","sim","Hit-class","class_p","ABS(match-true)","lambda","MSNs"]]
    results = results.rename(
            columns={
                "mass": "Precursor",
                "Hit-class": "ComfaceID_Class_Result",
                "sim": "Fingerprint_Sim",
                "pubchem_inchikey": "PubChem_InChIkey",
                "pubchem_smiles": "PubChem_SMILES",
                "pubchem_mass": "PubChem_Exact_Mass",
                "ABS(match-true)": "Mass_Error",
                "class_p": "Class_P",
                "lambda": "Lambda",
            }
        )
    results.to_csv(
        f"output/{timestamp}-distance_results.csv",
        index=False
    )