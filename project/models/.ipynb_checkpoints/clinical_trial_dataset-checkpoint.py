#!/usr/bin/env python
# coding: utf-8

import os
import torch
import pandas as pd
import json
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from data_utils import smiles2mpnnfeature


class ClinicalTrialDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase="I", split="train", dataset_name="hint", text_dir="text_description", summary_dir="brief_summary", augment=False):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.phase = phase
        self.split = split
        self.augment = augment  # Store augment value

        # Paths for text description and summary data
        self.description_dir = os.path.join(data_dir, text_dir)
        self.summary_dir = os.path.join(data_dir, summary_dir)
        
        # Load the CSV file for labels
        self.df = pd.read_csv(os.path.join(data_dir, f"phase_{phase}_{split}.csv"))

        # Load the clinical trial descriptions from the JSON file
        desc_path = os.path.join(self.description_dir, f"phase_{self.phase}_{self.split}.json")
        with open(desc_path, 'r') as f:
            self.desc_data = json.load(f)  # Load the JSON content into a list

        # Debugging: print lengths to ensure they match
        print(f"CSV length: {len(self.df)}")
        print(f"Description JSON length: {len(self.desc_data)}")

        # Check if the lengths match and adjust accordingly
        min_length = min(len(self.df), len(self.desc_data))
        self.df = self.df.iloc[:min_length]  # Trim to the minimum length
        self.desc_data = self.desc_data[:min_length]  # Trim the description data

        # Load criteria features (assuming they're stored as numpy arrays)
        criteria_path = os.path.join(self.data_dir, "criteria", f"phase_{self.phase}_{self.split}.npy")
        self.criteria_features = torch.tensor(np.load(criteria_path), dtype=torch.float32)
        
        # Tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(row["Success/Failure"], dtype=torch.float32)  # Adjusted for Success/Failure
    
        # Get the corresponding description text from the JSON file
        desc_text = self.desc_data[idx]
    
        # Tokenize the description and summary texts
        desc_inputs = self.tokenizer(desc_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    
        # Check if the summary data exists and load it from JSON
        summ_path = os.path.join(self.summary_dir, f"phase_{self.phase}_{self.split}.json")
        with open(summ_path, 'r') as f:
            summ_data = json.load(f)
    
        # Ensure the summary data length is correct and check if idx is valid
        if idx >= len(summ_data):
            idx = len(summ_data) - 1  # Adjust index to the last valid one if out of range
            
        summ_text = summ_data[idx]
        summ_inputs = self.tokenizer(summ_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    
        # Check if SMILES or criteria are None or empty
        if 'smiles' in row and row['smiles'] is not None:
            smiles_inputs = smiles2mpnnfeature(row['smiles'])  # This should return a valid tensor
        else:
            smiles_inputs = torch.zeros(ATOM_FDIM + BOND_FDIM)  # Fallback to zeros if SMILES is missing
    
        return {
            "criteria": self.criteria_features[idx] if idx < len(self.criteria_features) else torch.zeros_like(self.criteria_features[0]),
            "description": desc_inputs,
            "summarization": summ_inputs,
            "smiles": smiles_inputs,
            "label": label,
        }


    def augment_smiles(self, smiles_features):
        """
        Implement augmentation for SMILES features.
        For example: Randomly jitter or modify some properties of the SMILES string.
        """
        # Placeholder: Replace with actual SMILES augmentation logic
        return smiles_features  # No modification by default

    def augment_text(self, text_inputs):
        """
        Implement augmentation for textual data (description, summary).
        For example: Randomly change words, add noise, etc.
        """
        # Placeholder: Replace with actual text augmentation logic
        return text_inputs  # No modification by default
