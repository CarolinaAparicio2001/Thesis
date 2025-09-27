#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn import BCEWithLogitsLoss


# In[ ]:


class Highway(nn.Module):
    def __init__(self, size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = F.relu(self.nonlinear[layer](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


# In[ ]:


class ADMET(nn.Module):
    def __init__(self, molecule_encoder, highway_num, device, epoch, lr, weight_decay, save_name):
        super(ADMET, self).__init__()
        self.molecule_encoder = molecule_encoder
        self.embedding_size = self.molecule_encoder.embedding_size
        self.highway_num = highway_num

        self.highway_nn_lst = nn.ModuleList([
            Highway(size=self.embedding_size, num_layers=self.highway_num) for _ in range(5)
        ])
        self.fc_output_lst = nn.ModuleList([
            nn.Linear(self.embedding_size, 1) for _ in range(5)
        ])

        self.f = F.relu
        self.loss = BCEWithLogitsLoss()

        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_name = save_name

        self.device = device
        self.to(device)

    def set_device(self, device):
        self.device = device
        self.molecule_encoder.set_device(device)
        self.to(device)

    def forward_smiles_lst_embedding(self, smiles_lst, idx):
        embed_all = self.molecule_encoder.forward_smiles_lst(smiles_lst)
        output = self.highway_nn_lst[idx](embed_all)
        return output

    def forward_embedding_to_pred(self, embeded, idx):
        return self.fc_output_lst[idx](embeded)

    def forward_smiles_lst_pred(self, smiles_lst, idx):
        embeded = self.forward_smiles_lst_embedding(smiles_lst, idx)
        fc_output = self.forward_embedding_to_pred(embeded, idx)
        return fc_output

    def test(self, dataloader_lst, return_loss=True):
        loss_lst = []
        for idx in range(1):  # Only first task
            single_loss_lst = []
            for smiles_lst, label_vec in dataloader_lst[idx]:
                output = self.forward_smiles_lst_pred(smiles_lst, idx).view(-1)
                loss = self.loss(output, label_vec.to(self.device).float())
                single_loss_lst.append(loss.item())
            loss_lst.append(torch.tensor(single_loss_lst).mean().item())
        return torch.tensor(loss_lst).mean().item()

    def train(self, train_loader_lst, valid_loader_lst):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        valid_loss = self.test(valid_loader_lst, return_loss=True)
        valid_loss_record = [valid_loss]
        best_valid_loss = valid_loss
        best_model = deepcopy(self)

        for ep in range(self.epoch):
            data_iterator_lst = [iter(train_loader_lst[idx]) for idx in range(5)]
            try:
                while True:
                    for idx in range(1):  # Only one task
                        smiles_lst, label_vec = next(data_iterator_lst[idx])
                        output = self.forward_smiles_lst_pred(smiles_lst, idx).view(-1)
                        loss = self.loss(output, label_vec.float())

                        opt.zero_grad()
                        loss.backward()
                        opt.step()
            except StopIteration:
                pass

            valid_loss = self.test(valid_loader_lst, return_loss=True)
            valid_loss_record.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = deepcopy(self)

        return best_model

