import json
from scipy.special import softmax
import pandas as pd
import numpy as np
import datetime
from arg_parser import parse_args

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

if __name__=='__main__':
    params = parse_args()
    sim_score_file_path = params.sim_score_file_path
    class_results_file_path = params.class_results_file_path
    sim_score_df = pd.read_csv(sim_score_file_path)
    sim_score_df = sim_score_df.rename(columns={"inchikey": "pubchem_inchikey","mass": "pubchem_mass"})
    with open(class_results_file_path, 'r') as f:
        class_res = json.load(f)
    compound_class_json = "compound_class.json"
    with open(compound_class_json, 'r') as f:
        compound_class = json.load(f)
    results = add_part_3(sim_score_df, compound_class, class_res)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results.to_csv('output/{}-distance_results.csv'.format(timestamp), index=False)
