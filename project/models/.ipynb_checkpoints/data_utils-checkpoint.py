#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import csv
from rdkit import Chem
from rdkit import RDLogger
from functools import reduce
from torch.utils import data
from torch.utils.data.dataloader import default_collate
RDLogger.DisableLog('rdApp.*')


# In[ ]:


# Constants
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6


# In[ ]:


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.IsInRing()
    ]
    fstereo = onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5])
    return torch.Tensor(fbond + fstereo)

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol

def smiles2mpnnfeature(smiles):
    padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
    fatoms, fbonds = [], [padding]
    in_bonds, all_bonds = [], [(-1, -1)]

    mol = get_mol(smiles)
    if mol is not None:
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append(atom_features(atom))
            in_bonds.append([])

        for bond in mol.GetBonds():
            x, y = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            b = len(all_bonds)
            all_bonds.append((x, y))
            fbonds.append(torch.cat([fatoms[x], bond_features(bond)], 0))
            in_bonds[y].append(b)

            b = len(all_bonds)
            all_bonds.append((y, x))
            fbonds.append(torch.cat([fatoms[y], bond_features(bond)], 0))
            in_bonds[x].append(b)

        fatoms = torch.stack(fatoms, 0) if fatoms else torch.zeros(0, ATOM_FDIM)
        fbonds = torch.stack(fbonds, 0) if fbonds else torch.zeros(0, ATOM_FDIM + BOND_FDIM)

        total_bonds = len(all_bonds)
        agraph = torch.zeros(n_atoms, MAX_NB).long()
        bgraph = torch.zeros(total_bonds, MAX_NB).long()

        for a in range(n_atoms):
            for i, b in enumerate(in_bonds[a]):
                agraph[a, i] = b

        for b1 in range(1, total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x]):
                if all_bonds[b2][0] != y:
                    bgraph[b1, i] = b2
    else:
        fatoms = torch.zeros(0, ATOM_FDIM)
        fbonds = torch.zeros(0, ATOM_FDIM + BOND_FDIM)
        agraph = torch.zeros(0, MAX_NB).long()
        bgraph = torch.zeros(0, MAX_NB).long()

    shape_tensor = torch.Tensor([fatoms.shape[0], fbonds.shape[0]]).view(1, -1)
    return [fatoms.float(), fbonds.float(), agraph.float(), bgraph.float(), shape_tensor]


# In[ ]:


class smiles_dataset(data.Dataset):
    def __init__(self, smiles_lst, label_lst):
        self.smiles_lst = smiles_lst
        self.label_lst = label_lst

    def __len__(self):
        return len(self.smiles_lst)

    def __getitem__(self, idx):
        return smiles2mpnnfeature(self.smiles_lst[idx]), self.label_lst[idx]

def mpnn_feature_collate_func(x):
    # Ensure that all tensors are at least 1D (use unsqueeze if needed)
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j].dim() == 0:
                # Print the shape of the problematic tensor for debugging
                print(f"Found scalar tensor at x[{i}][{j}]. Shape before unsqueeze: {x[i][j].shape}")
                x[i][j] = x[i][j].unsqueeze(0)  # Convert scalar to 1D tensor
                print(f"Shape after unsqueeze: {x[i][j].shape}")

    # Now, concatenate all tensors in the batch along dimension 0
    return [torch.cat([x[j][i] for j in range(len(x))], 0) for i in range(len(x[0]))]

def mpnn_collate_func(batch):
    mpnn_features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return [mpnn_feature_collate_func(mpnn_features)] + [torch.tensor(labels)]

def get_cooked_data_smiles_lst(data_path):
    df = pd.read_csv(data_path)
    smiles_lst = df.iloc[:, 11].dropna().unique().tolist()  # This removes duplicates
    return smiles_lst

def data_loader(data_path, batch_size=32):
    smiles_lst = get_cooked_data_smiles_lst(data_path)
    labels = [1] * len(smiles_lst)  # Dummy labels
    dataset = smiles_dataset(smiles_lst, labels)
    return data.DataLoader(dataset, batch_size=batch_size, collate_fn=mpnn_collate_func)
