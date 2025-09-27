#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# In[ ]:


def create_var(tensor, requires_grad=None):
    return Variable(tensor, requires_grad=requires_grad) if requires_grad is not None else Variable(tensor)


# In[ ]:


def index_select_ND(source, dim, index):
    max_index = source.size(dim) - 1
    index = index.clamp(max=max_index)
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1).long())
    return target.view(final_size)


# In[ ]:


class MPNN(nn.Module):
    def __init__(self, mpnn_hidden_size=50, mpnn_depth=3, device='cpu'):
        super(MPNN, self).__init__()
        self.mpnn_hidden_size = mpnn_hidden_size
        self.mpnn_depth = mpnn_depth

        self.W_i = nn.Linear(39 + 50, self.mpnn_hidden_size, bias=False)
        self.W_h = nn.Linear(self.mpnn_hidden_size, self.mpnn_hidden_size, bias=False)
        self.W_o = nn.Linear(39 + self.mpnn_hidden_size, self.mpnn_hidden_size)
        self.device = device
        self.to(self.device)

    def set_device(self, device):
        self.device = device
        self.to(device)

    @property
    def embedding_size(self):
        return self.mpnn_hidden_size

    def feature_forward(self, features):
        fatoms, fbonds, agraph, bgraph, atoms_bonds = features
        agraph = agraph.long().to(self.device)
        bgraph = bgraph.long().to(self.device)
        atoms_bonds = atoms_bonds.long().to(self.device)

        N_atoms, N_bonds = 0, 0
        embeddings = []
        for i in range(atoms_bonds.shape[0]):
            n_a = atoms_bonds[i, 0].item()
            n_b = atoms_bonds[i, 1].item()

            if n_a == 0:
                embeddings.append(torch.zeros(1, self.mpnn_hidden_size, device=self.device))
                continue

            sub_fatoms = fatoms[N_atoms:N_atoms + n_a, :].to(self.device)
            sub_fbonds = fbonds[N_bonds:N_bonds + n_b, :].to(self.device)
            sub_agraph = agraph[N_atoms:N_atoms + n_a, :].to(self.device)
            sub_bgraph = bgraph[N_bonds:N_bonds + n_b, :].to(self.device)

            embed = self.single_feature_forward(sub_fatoms, sub_fbonds, sub_agraph, sub_bgraph)
            embeddings.append(embed)

            N_atoms += n_a
            N_bonds += n_b

        return torch.cat(embeddings, dim=0) if embeddings else None

    def single_feature_forward(self, fatoms, fbonds, agraph, bgraph):
        if fatoms.size(0) == 0:
            return create_var(torch.zeros(1, self.mpnn_hidden_size, device=self.device))

        fatoms = create_var(fatoms).to(self.device)
        fbonds = create_var(fbonds).to(self.device)
        agraph = create_var(agraph).to(self.device)
        bgraph = create_var(bgraph).to(self.device)

        binput = self.W_i(fbonds)
        message = F.relu(binput)

        for _ in range(self.mpnn_depth - 1):
            nei_message = index_select_ND(message, 0, bgraph).sum(dim=1)
            message = F.relu(self.W_h(nei_message) + binput)

        nei_message = index_select_ND(message, 0, agraph).sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = F.relu(self.W_o(ainput))

        return torch.mean(atom_hiddens, dim=0, keepdim=True)  # [1, hidden]

    def forward_single_smiles(self, feature):
        return self.feature_forward([f.unsqueeze(0) for f in feature])

    def forward_smiles_lst(self, feature_list):
        return torch.cat([self.forward_single_smiles(f) for f in feature_list], dim=0)

    def forward_smiles_lst_average(self, feature_list):
        all_embeds = self.forward_smiles_lst(feature_list)
        return all_embeds.mean(dim=0, keepdim=True)

    def forward_smiles_lst_lst(self, smiles_batch):
        return torch.cat([self.forward_smiles_lst_average(smiles) for smiles in smiles_batch], dim=0)

