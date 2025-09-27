import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from mmengine.dataset import BaseDataset
from transformers import AutoTokenizer

# Load tokenizers
Tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
SMILESTokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Max token lengths for various input parts
MaxLength = dict(
    table=1625,
    summarization=265,
    description=7,
    diseases=38,
    diseases_concat=461,
    diseases_summarization=560,
    drugs=54,
    drugs_concat=193,
    drugs_summarization=285,
    smiles=1627,
    smiles_concat=1075,
    smiles_summarization=1182,
    smiles_transformer=512,
    smiles_transformer_concat=512,
    smiles_transformer_summarization=512,
)

# Tokenization utility
def tokenize(tokenizer, text, max_length=None):
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )


# Dataset class
class HINTDataset(BaseDataset):
    def __init__(self, data_prefix=None, ann_file_name=None, augment=False, input_parts=None, **kwargs):
        self.ann_file_name = ann_file_name
        if data_prefix is None:
            data_prefix = dict(
                data_path="",
                table_path="text_description",
                summarization_path="brief_summary",
                drug_description_path="drugbank/drug_description.json",
                criteria_path="criteria",
            )

        self.augment = augment
        if input_parts is None:
            input_parts = [
                "table", "summarization", "description", "criteria"
            ] + [f"{k}{p}" for k in ["smiles", "smiles_transformer", "drugs", "diseases"] for p in ["", "_concat", "_summarization"]]
        self.input_parts = input_parts

        super().__init__(data_prefix=data_prefix, **kwargs)

    @staticmethod
    def collate_fn(batch):
        def collate(batch):
            res = {}
    
            # 1. Stack labels and criteria
            for name in ["label", "criteria"]:
                if name in batch[0]:
                    res[name] = torch.stack([b[name] for b in batch], dim=0)
    
            # 2. Tokenize simple string-based fields
            for name in ["table", "summarization"] + [
                f"{k}_{p}"
                for k in ["smiles", "smiles_transformer", "drugs", "diseases"]
                for p in ["concat", "summarization"]
            ]:
                if name in batch[0]:
                    texts = [b[name] for b in batch]
                    tokenizer = Tokenizer if "smiles_transformer" not in name else SMILESTokenizer
                    res[name] = tokenize(
                        tokenizer,
                        texts,
                        MaxLength.get(name, 512),
                    )
    
            # 3. List-based modalities (can include None)
            for name in ["smiles", "smiles_transformer", "drugs", "diseases", "description"]:
                if name in batch[0]:
                    tokenized = []
                    for b in batch:
                        item = b[name]
    
                        if isinstance(item, dict):
                            # Already tokenized
                            tokenized.append(item)
                        else:
                            # Handle list inputs with possible None values
                            if isinstance(item, list):
                                original = item
                                item = [x for x in item if isinstance(x, str)]
                                if len(item) < len(original):
                                    print(f"[Warning] Dropped non-string values from '{name}': {original}")
                                item = ", ".join(item)
    
                            if not isinstance(item, str):
                                raise ValueError(f"[Error] Invalid input for tokenizer '{name}': {item} (type: {type(item)})")
    
                            tokenizer = Tokenizer if "smiles_transformer" not in name else SMILESTokenizer
                            tokenized.append(
                                tokenize(
                                    tokenizer,
                                    item,
                                    MaxLength.get(name, 512),
                                )
                            )
                    res[name] = tokenized
    
            # 4. Scalar / vector numeric inputs (e.g. enrollment, admet)
            for name in ["enrollment", "admet"]:
                if name in batch[0]:
                    res[name] = torch.stack([b[name] for b in batch], dim=0)
    
            # 5. Sample index
            res["idx"] = torch.tensor([b["idx"] for b in batch])
    
            return res
    
        # Build regular batch
        res = collate(batch)
    
        # Handle strong augmentations recursively if present
        if "augment" in batch[0]:
            res["augment"] = collate([b["augment"] for b in batch])
    
        return res


    def load_table_data(self):
        path = os.path.join(self.data_prefix["table_path"], f"{self.ann_file_name}.json")
        if "table" in self.input_parts and os.path.exists(path):
            return json.load(open(path))

    def load_summarization_data(self):
        path = os.path.join(self.data_prefix["summarization_path"], f"{self.ann_file_name}.json")
        if "summarization" in self.input_parts and os.path.exists(path):
            return json.load(open(path))

    def load_drug_description_data(self):
        if "description" in self.input_parts and os.path.exists(self.data_prefix["drug_description_path"]):
            return json.load(open(self.data_prefix["drug_description_path"]))

    def load_criteria_data(self):
        path = os.path.join(self.data_prefix["criteria_path"], f"{self.ann_file_name}.npy")
        if "criteria" in self.input_parts and os.path.exists(path):
            return torch.from_numpy(np.load(path))

    def add_list_data(self, name, list_data, summarization_data=None):
        data = {}
        if name in self.input_parts:
            data[name] = list_data
        if f"{name}_concat" in self.input_parts:
            data[f"{name}_concat"] = ",".join(list_data)
        if f"{name}_summarization" in self.input_parts and summarization_data:
            data[f"{name}_summarization"] = f"{name}: {','.join(list_data)}; summarization: {summarization_data}"
        return data

    def load_data_list(self):
        data = pd.read_csv(os.path.join(self.data_prefix["data_path"], f"{self.ann_file_name}.csv"))
        table_data = self.load_table_data()
        summarization_data = self.load_summarization_data()
        drug_description = self.load_drug_description_data()
        criteria_data = self.load_criteria_data()

        data_list = []
        for i, row in data.iterrows():
            cur_data = {"idx": i}
            if "label" in row:
                cur_data["label"] = torch.tensor(int(row["label"]), dtype=torch.long)

            if criteria_data is not None:
                cur_data["criteria"] = criteria_data[i]
                if (cur_data["criteria"] == 0).all():
                    continue

            flag = True
            for name in ["smiles", "drugs", "diseases"]:
                if name in row:
                    d = sorted(set(eval(row[name])))
                    if not d:
                        flag = False
                        continue
                    cur_data.update(self.add_list_data(name, d, summarization_data[i] if summarization_data else None))
                    if name == "smiles":
                        cur_data.update(self.add_list_data("smiles_transformer", d, summarization_data[i] if summarization_data else None))

            for name, name_data in zip(["table", "summarization"], [table_data, summarization_data]):
                if name_data:
                    cur_data[name] = name_data[i]

            if drug_description:
                cur_data["description"] = [
                    drug_description.get(drug_name, "This is a drug.")
                    for drug_name in eval(row["drugs"])
                ]

            if flag:
                data_list.append(cur_data)

        return data_list

    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        if self.augment:
            data_info["augment"] = self.get_data_info(self._rand_another())
        return self.pipeline(data_info)


# Main function with corrected data paths
def main():
    max_length = defaultdict(list)
    labels = {}

    #base_path = r"C:/Users/Asus/Documents/Data and Code Deliverables/Data Group Part/Data/clinical-trial-outcome-prediction/Processed/hint"
    base_path = r"C:\Users\Carol\Documents\Data Code and Deliverables\Data Group Part\Data\clinical-trial-outcome-prediction\Processed\hint"
    data_prefix = dict(
        data_path=base_path,
        table_path=os.path.join(base_path, "text_description"),
        summarization_path=os.path.join(base_path, "brief_summary"),
        drug_description_path=os.path.join(base_path, "drugbank", "druginfo_description.json"),
        criteria_path=os.path.join(base_path, "criteria"),
    )

    for phase in ["I", "II", "III"]:
        labels[phase] = {}
        for split in ["train", "valid", "test"]:
            labels[phase][split] = [0, 0]
            dataset = HINTDataset(
                ann_file_name=f"phase_{phase}_{split}",
                data_prefix=data_prefix,
            )

            for i in range(len(dataset)):
                data = dataset[i]
                if any(d != "This is a drug." for d in data["description"]):
                    data = {k: v for k, v in data.items() if k in [
                        "smiles", "drugs", "diseases", "summarization", "description"]}
                    print(data)

                for name in data:
                    if name not in ["label", "sample_idx", "idx", "criteria"]:
                        tokenizer = Tokenizer if "smiles_transformer" not in name else SMILESTokenizer
                        max_length[name].append(
                            tokenizer(data[name], padding=True, return_tensors="pt", return_token_type_ids=False)["input_ids"].shape[-1]
                        )

                labels[phase][split][data["label"].item()] += 1

            labels[phase][split] = labels[phase][split][1] / sum(labels[phase][split])

    for name in max_length:
        max_length[name] = sorted(max_length[name], reverse=True)

    print(labels)
    print(max_length)


if __name__ == "__main__":
    main()
