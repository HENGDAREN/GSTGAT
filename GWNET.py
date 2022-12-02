from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from GCN_module import *
from layer_module import *
from graph_constuct import *


class GET(nn.Module):
    def __init__(self, num_nodes, embed_dim, device, time_step, out_channels, skip_channels, gamma,
                 end_channels, dilation=2, in_dim=1, TCN_layers=3, pre_len=1, gcn_bool=True, dropout=0.1, alpha=0.2, nheads=4, gdep=2, gat_true=True, out_feature = 10):
        super(GET, self).__init__()
        self.layers = TCN_layers
        self.pre_len = pre_len
        self.channels = out_channels
        self.time_step = time_step
        self.in_dim = in_dim
        self.device = device
        #self.gcn_true = gcn_bool
        self.dropout = dropout
        ##增加的参数
        self.gat_true = gat_true
        self.out_fea = out_feature
        #
        kernel_size = 7
        if dilation > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation ** TCN_layers - 1) / (dilation - 1))
        else:
            self.receptive_field = TCN_layers * (kernel_size - 1) + 1
        if self.receptive_field < self.time_step:
            print('receptive_field is lower than input, please add layers')
            exit(0)
        self.num_nodes = num_nodes

        self.improve_channels = nn.Conv2d(in_dim, out_channels, (1, 1), bias=True)

        self.filter_time = nn.ModuleList()
        self.gate_time = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        # self.skip_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        #self.gconv1 = nn.ModuleList()
        #self.gconv2 = nn.ModuleList()
        #开始增加GAT
        self.GAT = nn.ModuleList()

        self.bn = nn.ModuleList()

        new_dilation = 1
        rf_size_i = 1
        for i in range(1, TCN_layers + 1):
            if dilation > 1:
                # rf_size_j = 7, 19, 43, 91, 187
                rf_size_j = int(
                    rf_size_i + (kernel_size - 1) * (dilation ** i - 1) / (dilation - 1))
            else:
                rf_size_j = rf_size_i + i * (kernel_size - 1)
            self.filter_time.append(
                dilated_inception(cin=out_channels, cout=out_channels, dilation_factor=new_dilation))
            self.gate_time.append(
                dilated_inception(cin=out_channels, cout=out_channels, dilation_factor=new_dilation)
            )

            self.residual_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(out_channels))
            #self.gconv1.append(gcn(out_channels, out_channels, dropout))
            #self.gconv2.append(gcn(out_channels, out_channels, dropout))
            #增加的GAT
            self.GAT.append(
                mix_gat(channels=out_channels, time_step=self.receptive_field - rf_size_j + 1, out_feature=self.out_fea,
                        nodes=self.num_nodes, dropout=dropout, alpha=alpha, nheads=nheads, gdep=gdep)
            )
            # kernel_size = 37, 25, 1, this is right
            self.skip_convs.append(nn.Conv2d(in_channels=out_channels,
                                             out_channels=skip_channels,
                                             kernel_size=(1, self.receptive_field - rf_size_j + 1)))
            new_dilation *= dilation
        self.skip_0 = nn.Conv2d(in_dim, skip_channels, kernel_size=(1, self.receptive_field), bias=True)

        # 先搞图学习层
        self.gc = graph_constructor(self.num_nodes, embed_dim, in_dim=in_dim,
                                    window_len=time_step, device=device, gamma=gamma)
        self.skipE = nn.Conv2d(out_channels, skip_channels, kernel_size=(1, 1), bias=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=pre_len,
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, input, te=None):
        #bs widow_size node
        if len(input.shape) < 4:
            input = input.unsqueeze(1).repeat(1, self.in_dim, 1, 1)
        #bs 1 window_size node
        batch_size, time_len = input.shape[0], input.shape[2]
        assert time_len == self.time_step, 'input sequence length not equal to preset sequence length'
        #bs 1 node window_size
        input_tcn = input.permute(0, 1, 3, 2)
        #print(input_tcn.shape)
        if self.time_step < self.receptive_field:
            input_tcn = nn.functional.pad(input_tcn, (self.receptive_field - self.time_step, 0, 0, 0))
        # if self.gcn_true:
        #     adp, gl_loss, nodevec = self.gc(input)
        # else:
        #     adp, gl_loss, nodevec =None, None, None
        #input_tcn.shape bs channel node time+pad
        if self.gat_true:
            adp, gl_loss, nodevec = self.gc(input)
        else:
            adp, gl_loss, nodevec =None, None, None
        skip = self.skip_0(F.dropout(input_tcn, self.dropout, training=self.training))
        input_x = self.improve_channels(input_tcn)
        #input_x.shape bs channel node time+pad
        for i in range(self.layers):
            residual = input_x
            #print(input_x.shape)
            filter_time = self.filter_time[i](input_x)

            filter_time = torch.tanh(filter_time)

            gate_time = self.gate_time[i](input_x)
            gate_time = torch.sigmoid(gate_time)

            # input_x = b, 16, n, t
            input_x = filter_time * gate_time
            #input_x = b, chann ,n, t+2
            s = input_x
            s = self.skip_convs[i](s)
            skip = s + skip

            # if self.gcn_true:
            #     input_x = self.gconv1[i](input_x, adp) + self.gconv2[i](input_x, adp.transpose(1, 0))
            # else:
            #     input_x = self.residual_convs[i](input_x)
            if self.gat_true:
                # 使用GAT层进行捕捉, 这里的输入输出都是input_x = batch_size, nodes, 16 * time_step
                input_x = self.GAT[i](input_x, adp)
                # input_x = batch_size, 16, nodes, time_step
                input_x = input_x.contiguous()\
                    .view(batch_size, self.num_nodes, self.channels, -1).permute(0, 2, 1, 3)
            else:
                # input_x = batch_size, out_channels(16), nodes, time_step
                input_x = self.residula_convs[i](input_x)
            
            # 加残差
            input_x = residual[..., -input_x.size(3):] + input_x
            input_x = self.bn[i](input_x)
        skip = self.skipE(input_x) + skip
        skip_re = F.relu(skip)

        x = F.relu(self.end_conv_1(skip_re))
        x = self.end_conv_2(x)
        # final_result = 64, 1, 135
        final_result = x.squeeze(3)
        return final_result, gl_loss#, nodevec




class GraphAttentionLayer(nn.Module):
    def __init__(self, in_feature, out_feature, dropout, alpha, concat=True):
        '''
        in_feature: time_step * channels(16)
        out_feature: time_step * channels(16)
        '''
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.alpha = alpha
        self.concat = concat
        self.w = nn.Parameter(torch.zeros(size=(in_feature, out_feature)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_feature, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.norm = nn.LayerNorm(135)
    def forward(self, input, adj):
        # input = batch_size, nodes, out_feature
        h = torch.einsum('bnf, fo-> bno', [input, self.w])
        B, N, feature = h.size()
        # h.repeat(1, 1, N) = b, 2708，nhid*2708（nhid个特征，重复放了2708次）
        # 然后view(N * N, -1) = b, 2708*2708， nhid， 维度是这个维度 没错， 应该就是有2708*2708的节点特征
        # h.repeat(N, 1) = b,  2708*2708, nhid
        a_input = torch.cat([h.repeat(1, 1, N).contiguous().view(B, N*N, self.out_feature), h.repeat(1, N, 1)], dim=2)\
            .view(B, N, N, 2 * self.out_feature)
        # e = b, n,n
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        # zero_vec = -9e15*torch.ones_like(e)
        # if adj.shape[0] == 1:
        #     adj = adj.repeat(B, 1, 1)
        # attention = torch.where(adj > 0, e, zero_vec)
        # for i in range(attention.shape[0]):
        #    attention[i] += adj
        #attention = self.norm(torch.relu(attention))
        #print(attention)
        #print(adj.shape)
        #print(adj)
        #exit()
        #save_attention = attention.cpu().numpy()
        #np.save('save.npy', save_attention)
        attention = F.softmax(e, dim=-1)
        adj = adj.repeat(B, 1, 1)

        attention = attention + adj
        attention = F.softmax(attention, dim=-1)
        #attention = self.norm(attention)
        #attention = F.normalize(torch.relu(attention), p=1, dim=-1)
        # attention = B, N, N
        attention = F.dropout(attention, self.dropout, training=self.training)
        # 增加一个保存

        # h_prime = B, N, feature
        h_prime = torch.einsum('bnn, bnf -> bnf', [attention, h])
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class gat(nn.ModuleList):
    def __init__(self, dropout, nfeat, nhid, alpha, nheads):
        super(gat, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        # x如果是第一层就是节点特征，adj就是邻接矩阵
        x = F.dropout(x, self.dropout, training=self.training)
        # 这个就是将几个头注意力联合起来
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class mix_gat(nn.Module):
    def __init__(self, channels, time_step, out_feature, nodes, dropout, alpha, nheads, gdep):
        '''
        aloha: 是激活函数leakyrelu的参数
        '''
        super(mix_gat, self).__init__()
        self.channels = channels
        self.time_step = time_step
        # self.recevie_filed = recevie_filed
        self.features = self.time_step * self.channels
        self.out_fea = out_feature
        self.nodes = nodes
        self.gdep = gdep
        self.attentions = nn.ModuleList()
        self.out_attn = nn.ModuleList()

        self.w = nn.Parameter(torch.zeros(size=(self.features, self.out_fea)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)

        self.dropout = dropout
        for i in range(self.gdep):
            if i == 0:
                in_feature = self.features
            else:
                in_feature = self.out_fea
            self.attentions.append(gat(dropout, in_feature, self.out_fea, alpha, nheads))
            self.out_attn.append(GraphAttentionLayer(nheads*self.out_fea, self.out_fea, dropout=dropout,
                                                     alpha=alpha, concat=True))
        self.linear = nn.Parameter(torch.zeros(size=((self.gdep+1)*self.out_fea, self.features)))
        nn.init.xavier_uniform_(self.linear.data, gain=1.414)

    def forward(self, input_x, adj):
        # input_x =  batch_size, out_channels(16), nodes, time_step
        # 将数据维度转变一下，转变成 batch_size, nodes， out_channels(16) * time_step
        a = input_x.shape[1]
        b = input_x.shape[3]
        input = input_x.permute(0, 2, 1, 3).contiguous().view(-1, self.nodes, a * b)
        # input = batch_size, nodes, out_feature
        input_init = torch.einsum('bno, of->bnf', [input, self.w])
        residual = input
        out = [input_init]
        for i in range(self.gdep):
            # input = B, N, nheads * out_feature = nheads * 10(默认是10)
            input = self.attentions[i](input, adj)
            # input = B, N, feature
            input = F.elu(self.out_attn[i](input, adj))
            out.append(input)
        # result = B, N, feature * gdep
        result = torch.cat(out, dim=-1)
        # result = B, N, feature = out_channels(16) * time_step, [32, 135, 720]
        result = torch.einsum('bnf, fm -> bnm', [result, self.linear]) + residual
        return result

