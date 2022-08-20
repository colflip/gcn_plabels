import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, mid_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, mid_dim)
        self.classifier = nn.Sequential(
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

    def embedding(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return x

    def confidence(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.classifier(x)

        softmax = F.softmax(x, dim=1)
        log_softmax = F.log_softmax(x, dim=1)
        commentary = (-1) * (softmax.mul(log_softmax).sum(axis=1))
        commentary = commentary.reshape(len(commentary), 1)
        pre_label = softmax.argmax(dim=1)  # 预测的类别
        pre_label = pre_label.reshape(len(pre_label), 1)

        tmp = np.arange(len(data.y))
        tmp = tmp[:, np.newaxis]
        res = np.hstack([tmp, commentary.detach().cpu().numpy()])
        res = np.hstack([res, pre_label.detach().cpu().numpy()])
        return res
