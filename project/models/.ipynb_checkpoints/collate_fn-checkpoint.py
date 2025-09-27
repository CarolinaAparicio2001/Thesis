#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Models.data_utils import mpnn_feature_collate_func
import torch

def clinical_trial_collate(batch, dataset_name):
    # Initialize lists for different parts of the batch
    criteria = torch.stack([item["criteria"] for item in batch])
    labels = []

    # Handle 'description' and 'summarization'
    description = {k: torch.stack([item["description"][k].squeeze(0) for item in batch]) for k in batch[0]["description"]}
    summarization = {k: torch.stack([item["summarization"][k].squeeze(0) for item in batch]) for k in batch[0]["summarization"]}

    # Handle 'smiles' features, checking for both 'smiles' and 'SMILES' keys
    smiles_feats = []
    for item in batch:
        # Check for 'smiles' key (lowercase)
        if "smiles" in item:
            smiles_tensor = item["smiles"]
        # Check for 'SMILES' key (uppercase)
        elif "SMILES" in item:
            smiles_tensor = item["SMILES"]
        else:
            # If neither key exists, create a fallback value (e.g., zeros with the same shape as 'criteria')
            smiles_tensor = torch.zeros_like(item["criteria"])

        # Ensure that the smiles_tensor is not scalar (add a dimension if needed)
        if smiles_tensor.dim() == 0:
            smiles_tensor = smiles_tensor.unsqueeze(0)  # Convert scalar to 1D tensor

        smiles_feats.append(smiles_tensor)

        # Handle the label conversion for 'Success/Failure' or 'ctod' type labels
        if dataset_name == 'hint':  # For 'hint', convert Success/Failure to 1/0
            if item["label"] == "Success":
                labels.append(1)
            else:
                labels.append(0)
        elif dataset_name == 'ctod':  # For 'ctod', use the label directly
            labels.append(item["label"])

    # Apply the mpnn_feature_collate_func to process the 'smiles' features
    smiles_feats = mpnn_feature_collate_func(smiles_feats)

    # Return the final collated dictionary
    return {
        "criteria": criteria,
        "smiles": smiles_feats,
        "description": description,
        "summarization": summarization,
        "label": torch.tensor(labels),  # Convert labels to a tensor
    }