import torch
from torch import nn
from layer_module import *


class graph_constructor(nn.Module):
    def __init__(self, nodes, dim, device, window_len, gamma, in_dim=1): # =0.001
        super(graph_constructor, self).__init__()
        self.nodevec = nn.Embedding(nodes, dim)
        self.gamma = gamma
        self.idx = torch.arange(nodes).to(device)
        self.time_merge = nn.Conv2d(in_dim, dim, kernel_size=(1, window_len), bias=True)
        self.time_norm = nn.LayerNorm(dim)
        self.dim = dim
        self.nodes = nodes

    def forward(self, node_input):
        # node_input = 64, 1, 15, 135
        if node_input.shape[1] != 1:
            node_input1 = self.time_merge(node_input.transpose(-1, -2)).squeeze()
        else:
            node_input1 = self.time_merge(node_input.transpose(-1, -2)).reshape(-1, self.dim, self.nodes)
        input = self.time_norm(node_input1.transpose(1, 2)) # self.time_merge(node_input.transpose(-1, -2)).squeeze()
        nodevec = self.nodevec(self.idx)
        adp = F.softmax(F.relu(torch.mm(nodevec, nodevec.transpose(1, 0))), dim=1)
        gl_loss = self.graph_loss(input, adp)
        return adp, gl_loss, nodevec

    def graph_loss(self, input, adj):
        B, N, D = input.shape
        x_i = input.unsqueeze(2).expand(B, N, N, D)
        x_j = input.unsqueeze(1).expand(B, N, N, D)
        dist_loss = torch.pow(torch.norm(x_i - x_j, dim=3), 2) * adj
        dist_loss = torch.sum(dist_loss, dim=(1, 2))
        f_norm = torch.pow(torch.norm(adj, dim=(1, 0)), 2)
        gl_loss = dist_loss + self.gamma * f_norm
        return gl_loss