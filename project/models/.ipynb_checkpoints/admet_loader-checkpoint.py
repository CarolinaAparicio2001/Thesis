import os
import pandas as pd
import torch

def load_admet_file(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None)
    return dict(zip(df[0], df[1]))

def load_admet_all(base_path, split):
    admet = {}
    tasks = ["absorption", "distribution", "metabolism", "excretion", "toxicity"]
    for task in tasks:
        file = os.path.join(base_path, f"{task}_{split}.txt")
        if os.path.exists(file):
            admet[task] = load_admet_file(file)
    return admet

def attach_admet_features(smiles_list, admet_mappings):
    features = {}
    for task, mapping in admet_mappings.items():
        values = [mapping.get(smile, None) for smile in smiles_list]
        values = [float(v) for v in values if v is not None]
        if values:
            features[f"admet_{task}"] = torch.tensor(sum(values) / len(values), dtype=torch.float)
    return features
