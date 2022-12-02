import torch
from torch import nn
import torch.nn.functional as F


class dilated_inception_1(nn.Module):
    def __init__(self, time_step, cin, cout=16, dilation_factor=1):
        super(dilated_inception_1, self).__init__()
        self.time_len = time_step
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.ModuleList()
        # cout=4 这里就变成了，其实就相当于拆分去做
        cout_1 = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout_1, (1, kern), dilation=(1, dilation_factor),
                                        padding=(0, (kern - 1) * dilation_factor)))

    def forward(self, input):
        '''
        返回的结果：batch_size, out_channels, nodes， time_step, 这个维度是强制要求
        '''
        # input = batch_size, 16, nodes, time_step + pad
        x = input
        result = []
        # 下面输出是 batch_size, 4, nodes, time_step+pad-2, ... time_step+pad-3, ...time_step+pad-6, ...time_step+pad-7
        for i in range(len(self.kernel_set)):
            result.append(self.tconv[i](x))
        #  保证长度都是time_step的长度
        for i in range(len(self.kernel_set)):
            if result[i].shape[-1] >= self.time_len:
                result[i] = result[i][..., -self.time_len:]
            else:
                result[i] = nn.functional.pad(result[i], (self.time_len - result[i].shape[-1], 0, 0, 0))
        # result = batch_size, 16, nodes, time_step
        result = F.relu(torch.cat(result, dim=1))
        # 残差结构
        result = x[:, :, :, -self.time_len:] + result
        return result


class fc_layer(nn.Module):
    def __init__(self, in_channels, out_channels, need_layer_norm):
        super(fc_layer, self).__init__()
        # 定义key，query， value的转换矩阵
        # linear_tran = 1, 1, nodes, out_channels
        self.linear_w = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_uniform_(self.linear_w.data, gain=1.414)

        # 下面这个用来表示残差, linear = 1, out_channels, nodes, 1
        self.linear = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=[1, 1], bias=True)
        # 正则化的层--------------------------------------------------------------------------------------------------------------
        self.layer_norm = nn.LayerNorm(out_channels)
        self.need_layer_norm = need_layer_norm

    def forward(self, input):
        '''
        input = batch_size, in_channels, nodes, time_step
        output = batch_size, out_channels, nodes, time_step
        '''
        # input = 1, dim, nodes, 1
        # input = batch_size, in_channels, nodes, time_step
        if self.need_layer_norm:
            result = F.leaky_relu(torch.einsum('bani,io->bano ', [input.transpose(1, -1), self.linear_w]))\
                     # + self.layer_norm(self.linear(input).transpose(1, -1))
        else:
            result = F.leaky_relu(torch.einsum('bani,io->bano ', [input.transpose(1, -1), self.linear_w])) \
                     # + self.linear(input).transpose(1, -1)
        return result.transpose(1, -1)


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        #print(input.shape)
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class gatedFusion_2(nn.Module):
    def __init__(self, dim_in, dim_out, device):
        super(gatedFusion_2, self).__init__()
        self.device = device

        self.w = nn.Parameter(torch.zeros(size=(dim_in, dim_out)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)

        self.t = nn.Parameter(torch.zeros(size=(dim_in, dim_out)))
        nn.init.xavier_uniform_(self.t.data, gain=1.414)

        self.w_r = nn.Linear(in_features=dim_in, out_features=dim_out)
        self.u_r = nn.Linear(in_features=dim_in, out_features=dim_out)

        self.w_h = nn.Linear(in_features=dim_in, out_features=dim_out)
        self.w_u = nn.Linear(in_features=dim_in, out_features=dim_out)

    def forward(self, batch_size, nodevec, time_node):
        node_res = torch.einsum('bnd, dd->bnd', [nodevec, self.w]) + nodevec
        # node_res = batch_size, nodes, dim
        time_res = time_node + torch.einsum('bnd, dd->bnd', [time_node, self.t])

        # z = batch_size, nodes, dim
        z = torch.sigmoid(node_res + time_res)
        r = torch.sigmoid(self.w_r(time_node) + self.u_r(nodevec))
        h = torch.tanh(self.w_h(time_node) + r * self.w_u(nodevec))
        res = torch.add(z * nodevec, torch.mul(torch.ones(z.size()).to(self.device) - z, h))

        return res
