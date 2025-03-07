import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import numpy as np
import csv
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(32, 32, 1)
        for m in self.modules():
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)
        return logits

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class MRCGNN(nn.Module):
    def __init__(self, feature, hidden1, hidden2, decoder1, dropout, zhongzi):
        super(MRCGNN, self).__init__()

        # RGCN layers for the main (data_o) branch
        self.encoder_o1 = RGCNConv(feature, hidden1, num_relations=65)
        self.encoder_o2 = RGCNConv(hidden1, hidden2, num_relations=65)

        # Two-element parameter for layer attention
        self.attt = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.disc = Discriminator(hidden2 * 2)
        self.dropout = dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
        # Final classifier: prediction solely from data_o branch.
        # Each node's final representation is a concatenation of (hidden1 + hidden2).
        # For a pair of entities, the dimension becomes 2*(hidden1+hidden2).
        self.classifier = nn.Linear(2 * (hidden1 + hidden2), 65)

        # We no longer load any pretrained features for skip connection.

    def forward(self, data_o, data_s, data_a, idx):
        # Process data_o branch
        x_o, adj, e_type = data_o.x, data_o.edge_index, data_o.edge_type
        e_type1 = data_a.edge_type
        e_type = torch.tensor(e_type, dtype=torch.int64)
        e_type1 = torch.tensor(e_type1, dtype=torch.int64)

        # Main branch for prediction (data_o)
        x1_o = F.relu(self.encoder_o1(x_o, adj, e_type))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x2_o = self.encoder_o2(x1_o, adj, e_type)

        # Contrastive learning branches (unused in prediction)
        x_a = data_s.x
        x1_o_a = F.relu(self.encoder_o1(x_a, adj, e_type))
        x1_o_a = F.dropout(x1_o_a, self.dropout, training=self.training)
        x2_o_a = self.encoder_o2(x1_o_a, adj, e_type)

        x1_o_a_a = F.relu(self.encoder_o1(x_o, adj, e_type1))
        x1_o_a_a = F.dropout(x1_o_a_a, self.dropout, training=self.training)
        x2_o_a_a = self.encoder_o2(x1_o_a_a, adj, e_type1)

        # Readout for contrastive learning
        h_os = self.read(x2_o)
        h_os = self.sigm(h_os)
        ret_os = self.disc(h_os, x2_o, x2_o_a)
        ret_os_a = self.disc(h_os, x2_o, x2_o_a_a)

        # For final prediction, use only data_o branch:
        final = torch.cat((self.attt[0] * x1_o, self.attt[1] * x2_o), dim=1)

        a = [int(i) for i in list(idx[0])]
        b = [int(i) for i in list(idx[1])]
        aa = torch.tensor(a, dtype=torch.long)
        bb = torch.tensor(b, dtype=torch.long)
        entity1 = final[aa]
        entity2 = final[bb]
        concatenate = torch.cat((entity1, entity2), dim=1)
        log = self.classifier(concatenate)

        return log, ret_os, ret_os_a, x2_o

    def predict(self, data_o, idx):
        """
        New prediction method that uses only data_o and idx.
        """
        x_o, adj, e_type = data_o.x, data_o.edge_index, data_o.edge_type
        e_type = torch.tensor(e_type, dtype=torch.int64)
        # Process the main branch
        x1_o = F.relu(self.encoder_o1(x_o, adj, e_type))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x2_o = self.encoder_o2(x1_o, adj, e_type)
        final = torch.cat((self.attt[0] * x1_o, self.attt[1] * x2_o), dim=1)
        
        a = [int(i) for i in list(idx[0])]
        b = [int(i) for i in list(idx[1])]
        aa = torch.tensor(a, dtype=torch.long)
        bb = torch.tensor(b, dtype=torch.long)
        entity1 = final[aa]
        entity2 = final[bb]
        concatenate = torch.cat((entity1, entity2), dim=1)
        log = self.classifier(concatenate)
        return log