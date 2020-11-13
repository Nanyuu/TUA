import torch as t
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, init):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nclass, init=init)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        gc -> ReLU -> dropout -> gc -> log_SoftMax
        :param x: feat
        :param adj: adj
        :return: label log_softMax
        """
        h1_gc = self.gc1(x, adj)
        h1_relu = F.relu(h1_gc)
        h1_dropout = F.dropout(h1_relu, self.dropout, training=self.training)
        h2_gc = self.gc2(h1_dropout, adj)
        h2_softmax = F.log_softmax(h2_gc, dim=1)

        return h2_softmax


