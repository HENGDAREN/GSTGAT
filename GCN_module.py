from torch import nn
import torch
import math
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', [x, A])
        return x.contiguous()


class dnconv(nn.Module):
    def __init__(self):
        super(dnconv, self).__init__()

    def forward(self, x, A):
        if len(A.size()) == 2:
            A = A.unsqueeze(0).repeat(x.shape[0], 1, 1)
        # x = torch.einsum('nvw, ncvl->ncwl', [A, x])
        x = torch.einsum('nvw, ncwl->ncvl', [A, x])
        return x.contiguous()


class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x, A):
        # x = 32,16,137,181
        x = torch.einsum('ncvl,nvd->ncdl', [x, A])
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        x1 = self.nconv(x, support)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, support)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class mixprop(nn.Module):
    # 这里还没看啊，参数为 conv_channels, residual_channels, gcn_depth, dropout, propalpha), gdep =2
    # 这里c_in,c_out 就是输入，输出通道，反正对应上面两个参数，好像一般都给一样的，gdep是gcn的深度
    def __init__(self, in_features, out_features, gdep, alpha=0.05):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.dnconv = dy_nconv()
        # self.nconv = nconv_1(in_features, out_features)
        self.alpha = alpha
        self.gdep = gdep
        self.mlp = linear((gdep + 1) * in_features, out_features)

    def forward(self, x, adj):
        if adj.shape[0] == 207:
            # 这里的x是经过时间卷积的结果，adj就是邻接矩阵，以一个例子来说明， x =[32, 16, 137, 181]， adj=137，137
            adj = adj + torch.eye(adj.size(0)).to(x.device)
            # 下面应该是按照1维度进行求和，可是不是每个边都是k个关系吗？
            # d = 137
            d = adj.sum(1)
            # print(d.sum())
            # h = 32,16,137,181
            h = x
            # 下面就是直接将h封装为列表给了out
            out = [h]
            # a = 137, 137，
            a = adj / d.view(-1, 1)
            # print(a.sum())
            for i in range(self.gdep):
                # 首先在这个gcn的深度 out 叠加h, 这里是2啊
                # 这个对应的是公式7, 这里注意h是不断更改的哦
                h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
                out.append(h)  # out = out + h
            # ho = 32, 48, 137, 181, 为什么是48呢，是因为gcn是两层，加上之前就放进去的，总共就是3层，16 *3 = 48
            ho = torch.cat(out, dim=1)  # out #
            ho = self.mlp(ho)
        else:
            iden = torch.eye(207).unsqueeze(0).repeat(64, 1, 1).to(x.device)
            # print(iden.shape)
            adj = adj + iden
            d = adj.sum(2)
            # print(d[0].sum(), d[1].sum())
            # h = 32,16,137,181
            h = x
            out = [h]
            # a = batch_size, 137, 137， d.unsqueeze(-1) = 64, 207, 1
            a = adj / d.unsqueeze(-1)
            # if not (torch.equal(a[0], a[1]) and torch.equal(a[15], a[-2])):
            #     print('这里不应该输出')
            # print(torch.equal(a[0], a[1]),  torch.equal(a[15], a[-2]))
            # a = adj
            # print(a[0, 0])
            for i in range(self.gdep):
                h = self.alpha * x + (1 - self.alpha) * self.dnconv(h, a)
                out.append(h)  # out = out + h
            # ho = 32, 48, 137, 181, 为什么是48呢，是因为gcn是两层，加上之前就放进去的，总共就是3层，16 *3 = 48
            ho = torch.cat(out, dim=1)  # out #
            ho = self.mlp(ho)
        return ho

