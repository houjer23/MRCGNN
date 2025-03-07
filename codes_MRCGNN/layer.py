import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,RGCNConv
import numpy as np
import csv
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)
def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)


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

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_channels, 2 * out_channels)
        self.linear2 = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class MRCGNN(nn.Module):
    def __init__(self, feature, hidden1, hidden2, decoder1, dropout, zhongzi):
        super(MRCGNN, self).__init__()

        self.encoder_o1 = RGCNConv(feature, hidden1, num_relations=65)
        self.encoder_o2 = RGCNConv(hidden1, hidden2, num_relations=65)

        # Use a two-element parameter for layer attention
        self.attt = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.disc = Discriminator(hidden2 * 2)
        self.dropout = dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
        # Replace the MLP with a single classifier layer.
        # Here, final feature dimension is assumed to be 448.
        self.classifier = nn.Linear(448, 65)

        # The rest of your initialization (loading drug_list and features) remains unchanged.
        drug_list = []
        with open('data/drug_listxiao.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                drug_list.append(row[0])
        features = np.load('trimnet/drug_emb_trimnet' + str(zhongzi) + '.npy')
        ids = np.load('trimnet/drug_idsxiao.npy').tolist()
        features1 = [features[ids.index(drug_list[i])] for i in range(len(drug_list))]
        features1 = np.array(features1)
        self.features1 = torch.from_numpy(features1).cuda()

    def forward(self, data_o, data_s, data_a, idx):
        # Process through RGCN layers (same as before)
        x_o, adj, e_type = data_o.x, data_o.edge_index, data_o.edge_type
        e_type1 = data_a.edge_type
        e_type = torch.tensor(e_type, dtype=torch.int64)
        e_type1 = torch.tensor(e_type1, dtype=torch.int64)
        x_a = data_s.x

        x1_o = F.relu(self.encoder_o1(x_o, adj, e_type))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x2_o = self.encoder_o2(x1_o, adj, e_type)

        # (Other branches remain unchanged; note that your contrastive learning paths still exist.)
        x1_o_a = F.relu(self.encoder_o1(x_a, adj, e_type))
        x1_o_a = F.dropout(x1_o_a, self.dropout, training=self.training)
        x2_o_a = self.encoder_o2(x1_o_a, adj, e_type)

        x1_o_a_a = F.relu(self.encoder_o1(x_o, adj, e_type1))
        x1_o_a_a = F.dropout(x1_o_a_a, self.dropout, training=self.training)
        x2_o_a_a = self.encoder_o2(x1_o_a_a, adj, e_type1)

        # Readout
        h_os = self.read(x2_o)
        h_os = self.sigm(h_os)

        # Contrastive learning using the discriminator
        ret_os = self.disc(h_os, x2_o, x2_o_a)
        ret_os_a = self.disc(h_os, x2_o, x2_o_a_a)

        # Indices for entity pairs
        a = [int(i) for i in list(idx[0])]
        b = [int(i) for i in list(idx[1])]
        aa = torch.tensor(a, dtype=torch.long)
        bb = torch.tensor(b, dtype=torch.long)

        # Layer attention: combine features from first and second layers
        final = torch.cat((self.attt[0] * x1_o, self.attt[1] * x2_o), dim=1)

        entity1 = final[aa]
        entity2 = final[bb]

        # Skip connection using preloaded features
        entity1_res = self.features1[aa].to('cuda')
        entity2_res = self.features1[bb].to('cuda')
        entity1 = torch.cat((entity1, entity1_res), dim=1)
        entity2 = torch.cat((entity2, entity2_res), dim=1)
        
        # Concatenate the two entity representations and predict directly
        concatenate = torch.cat((entity1, entity2), dim=1)
        log = self.classifier(concatenate)

        return log, ret_os, ret_os_a, x2_o