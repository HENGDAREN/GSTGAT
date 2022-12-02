import os
import argparse
import numpy as np
# import pandas as pd
# import sklearn
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch import optim
from torch.optim import lr_scheduler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import utils as utils
from utils import Dataset
from GWNET import GET
from pytorchtools_with_node2vec import EarlyStopping
import matplotlib.pyplot as plt
from torchsummary import summary
from d2l import torch as d2l
parser = argparse.ArgumentParser()

parser.add_argument('--step_size', type=int, default=15)
parser.add_argument('--input_shuffle', type=bool, default=True, help='shuffle input or not')
parser.add_argument('--is_normal', type=int, default=0, help='need normalization, 1 Yes, 0 No')
parser.add_argument('--device', type=str, default='cuda:0', help='') # 'cuda:0'  'cpu'
parser.add_argument('--gat_true', type=bool, default=True, help='')
# 文件路径e
parser.add_argument('--sst_file', type=str, default='data/Bo_Hai.csv', help='sst data') # 'data/Nan_Hai_samePaper.csv'
# /home/lizhuolin/ProGram/GNN_SST/data/Bo_Hai.csv   data/Bo_Hai.csv
parser.add_argument('--log_file', type=str, default='log/setup_log', help='log file')
parser.add_argument('--model_save_path', type=str, default='model/')
# 时间窗口和学习率等
parser.add_argument('--train_ratio', type=float, default=0.8, help='training size default is 0.8')
parser.add_argument('--vaild_ratio', type=float, default=0.2, help='vaild set default is 0.2')
parser.add_argument('--test_ratio', type=float, default=0.2, help='test set default is 0.2')

parser.add_argument('--Windows_size', type=int, default=35, help='a time step is 15 days')
parser.add_argument('--Pre_len', type=int, default=1, help='model predict length')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--epoch', type=int, default=500, help='max epoch number')
parser.add_argument('--earlystop', type=int, default=70, help='number for early stopping')
parser.add_argument('--patience', type=int, default=10, help='patience for decay learning rate')
parser.add_argument('--learning_rate', type=float, default=0.003, help='init learning rate')
# 模型参数e
parser.add_argument('--embed_dim', type=int, default=20, help='dim of node')
parser.add_argument('--channels', type=int, default=16, help='channels in model')
parser.add_argument('--dilation', type=int, default=2, help='dilation exponential')
parser.add_argument('--skip_channels', type=int, default=64, help='skip channels')
parser.add_argument('--end_channels', type=int, default=128, help='end channels')
parser.add_argument('--TCN_layers', type=int, default=3, help='layers of TCN in model')
parser.add_argument('--gamma', type=float, default=0.01)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--order', type=float, default=1)
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('-- weight_decay', type=float, default=0.000001)
# 增加的GAT参数
parser.add_argument('--nheads', type=int, default=1, help='only control GAT heads')
parser.add_argument('--gdep', type=int, default=2, help='depth of mix-gat module')
parser.add_argument('--gat_feature', type=int, default=10, help='size of gat')

def predict(data_loader, data_num, node_num, model, device):
    y_pred = np.zeros(shape=(data_num, args.Pre_len, node_num))
    label_y = np.zeros(shape=(data_num, args.Pre_len, node_num))
    i = 0
    for data_x, data_y, data_te in data_loader:
        data_x, data_te = Variable(data_x), Variable(data_te)
        data_x, data_te = data_x.to(device), data_te.to(device)
        pred_y, _ = model(data_x, data_te)
        y_pred[i*args.batch_size: (i + 1)*args.batch_size, :, :] = pred_y.cpu().detach().numpy()
        label_y[i*args.batch_size: (i + 1)*args.batch_size, :, :] = data_y.numpy()
        i += 1
    return np.array(y_pred), np.array(label_y)


def addTE(trainX, trainTE):
    trainX = np.expand_dims(trainX, -1)
    nodes = trainX.shape[-2]
    time_setp = trainX.shape[1]
    train_te_1 = np.expand_dims(trainTE, -2).repeat(nodes, -2)
    data_x = np.concatenate([trainX, train_te_1[:, :time_setp]], axis=-1)
    data_x = np.transpose(data_x, (0, 3, 1, 2))
    return data_x


def plot_map(pred, target, dataset, data_point=0, pre_day=0, is_save=False):
    plt.figure()
    plt.plot(pred[:, pre_day, data_point], color='red', label='prediction')
    plt.plot(target[:, pre_day, data_point], color='blue', label='label')
    save_path = args.model_save_path + '{}_{}_.jpg'.format(args.Windows_size, args.Pre_len)
    save_path = save_path + '_{}_.fig'.format(dataset)
    if is_save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print('新建图片保持路径')
        plt.savefig(save_path)
    plt.show()


# 记录开始时间
args = parser.parse_args()
args.seed = int(time.time()) # 1626768239 #1626783721 #
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

device = torch.device(args.device)


def main(args):
    # 导入数据
    # utils.log_string(log, 'loda data ...')
    log = open(args.log_file, 'w')
    utils.log_string(log, 'seed is {}'.format(args.seed))
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std) = utils.loadData(args)
    utils.log_string(log, 'trainX: {}, trainY: {}, trainTE: {}'.format(trainX.shape, trainY.shape, trainTE.shape))
    utils.log_string(log, 'valX: {}, valY: {}, valTE: {}'.format(valX.shape, valY.shape, valTE.shape))
    utils.log_string(log, 'testX: {}, testY: {}, testTE: {}'.format(testX.shape, testY.shape, testTE.shape))
    utils.log_string(log, 'Data loaded')

    # utils.log_string(log, 'compiling model')
    model_save_path = args.model_save_path + '_{}_{}_'.format(args.Windows_size, args.Pre_len)

    model = GET(num_nodes=trainX.shape[-1], embed_dim=args.embed_dim, device=args.device,
                time_step=args.Windows_size, out_channels=args.channels, skip_channels=args.skip_channels,
                end_channels=args.end_channels, dilation=args.dilation, in_dim=1, dropout=args.dropout,
                TCN_layers=args.TCN_layers, pre_len=args.Pre_len, gamma=args.gamma,
                nheads=args.nheads, gat_true=args.gat_true, gdep=args.gdep, out_feature=args.gat_feature)

    model = model.to(device)

    utils.log_string(log, str(args))
    print('The recpetive field siez is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    utils.log_string(log, str(nParams))
    # # 损失函数
    # metric = nn.SmoothL1Loss().to(device)
    # # 优化器
    # learn_parameter = []
    # learn_parameter += [{'params': model.parameters(), 'lr': args.learning_rate}]
    #
    # opti_all = optim.Adam(params=learn_parameter, weight_decay=args.weight_decay) # 0.000001
    # scheduler = lr_scheduler.ReduceLROnPlateau(opti_all, mode='min', factor=0.5,
    #                                            patience=args.patience, eps=0.00003, cooldown=30, verbose=True)
    #
    # is_addTE = False
    # if is_addTE:
    #     # x_norm = 24, 2
    #     x_norm = trainTE.max(axis=0)
    #     trainte = trainTE/x_norm
    #     valte = valTE/x_norm
    #     testte = testTE/x_norm
    #     trainX = addTE(trainX, trainte)
    #     valX = addTE(valX, valte)
    #     testX = addTE(testX, testte)
    #     print(trainX.shape, valX.shape, testX.shape)

    # 将数据放进dataset中去, 这里还没有放入dataloader里面呢啊，注意
    # train_dataset = Dataset(trainX, trainY, trainTE)
    # vaild_dataset = Dataset(valX, valY, valTE)
    # test_dataset = Dataset(testX, testY, testTE)

    # 生成数据读取器
    # train_loader, vaild_loader, test_loader = data.DataLoader(train_dataset, args.batch_size, shuffle=args.input_shuffle), \
    #                                           data.DataLoader(vaild_dataset, args.batch_size), \
    #                                           data.DataLoader(test_dataset, args.batch_size)

    # 开始测试
    start_time = time.time()

    # 加载模型
    model.load_state_dict(torch.load(model_save_path))
    print('load save model')
    layer_name = list(model.state_dict().keys())
    print(len(layer_name))
    print(layer_name)
    # for para in model.parameters():
    #     print(para)
    for name, para in model.named_parameters():
        print(name,':',para.size())

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),cmap='Reds'):
    d2l.use_svg_display()

    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
    if i == num_rows - 1:
        ax.set_xlabel(xlabel)
        if j == 0:
            ax.set_ylabel(ylabel)
        if titles:
            ax.set_title(titles[j])
        fig.colorbar(pcm, ax=axes, shrink=0.6);
    # 模型测试
    # model.eval()
    # with torch.no_grad():
    #     train_performance, train_label = predict(train_loader, trainX.shape[0], trainX.shape[-1], model, device)
    #     test_performance, test_label  = predict(test_loader, testX.shape[0], testX.shape[-1], model, device)

    # todo 反归一化
    # train_performance, test_performance = train_performance * std + mean, test_performance * std + mean
    # train_label, test_label = train_label * std + mean, test_label * std + mean
    # train_mae, train_mse, train_mape = utils.metric_self(train_performance, train_label)
    # test_mae, test_mse, test_mape = utils.metric_self(test_performance, test_label)

    # todo 将预测结果跟label存起来
    # pred = test_performance
    # label = test_label
    # np.save(r'D:/first/GNN_SST-8-31-from3090\result\predict\{}_{}_pred.npy'.
    #         format('bohai', args.Pre_len), pred)
    # np.save(r'D:/first/GNN_SST-8-31-from3090\result\predict\{}_{}_label.npy'.
    #         format('bohai', args.Pre_len), label)

    # nodevec = nodevec.cpu().numpy()
    # np.save(r'D:/first/GNN_SST-8-31-from3090\predict\{}_{}_nodevec.npy'.
    #         format('bohai', args.Pre_len), nodevec)


    # print('train pred shape is {}, test pred shape is {}'.format(train_performance.shape, test_performance.shape))
    # utils.log_string(log, 'train set mae is : {} \n  train set mse is : {} \n  train set mape is : {}'.format(
    #         train_mae, train_mse, train_mape))
    # utils.log_string(log, 'test set mae is : {} \n  test set mse is : {} \n  test mape is : {}'.format(
    #     test_mae, test_mse, test_mape))

    # if args.Pre_len == 1:
    #     test_mae1 = mean_absolute_error(test_label.squeeze(), test_performance.squeeze())
    #     test_rmse1 = mean_squared_error(test_label.squeeze(), test_performance.squeeze())
    # else:
    #     test_label = test_label.reshape(-1, args.Pre_len * test_label.shape[-1])
    #     test_performance = test_performance.reshape(-1, args.Pre_len * test_label.shape[-1])
    # utils.log_string(log, 'test set mae is : {}, test set rmse is : {},'
    #                       'if this is different form above, above is wrong'.format(test_mae, test_rmse))

    # amae = []
    # amape = []
    # amse = []
    #
    # for i in range(args.Pre_len):
    #     # pred = scaler.inverse_transform(test_performance[:, :, i])
    #     pred = test_performance[:, i, :]
    #     # print(pred.shape)
    #     real = test_label[:, i, :]
    #     metrics = utils.metric_self(pred, real)
    #     log_1 = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test MSE: {:.4f}'
    #     utils.log_string(log, log_1.format(i + 1, metrics[0], metrics[2], metrics[1]))
    #     amae.append(metrics[0])
    #     amape.append(metrics[2])
    #     amse.append(metrics[1])
    #
    # log_2 = 'On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test MSE: {:.4f}'
    # utils.log_string(log, log_2.format(args.Pre_len, np.mean(amae), np.mean(amape), np.mean(amse)))
    # utils.log_string(log, 'seed is {}'.format(args.seed))
    cost_time = time.time() - start_time
    utils.log_string(log, str(cost_time))
    log.close()


if __name__ == '__main__':
    # 时间窗口和学习率等
    epoch = 500
    window_list = [35]
    pre_lenList = [1] # 3, 7, 15
    batch_size = 32  # 128 64
    learning_rateList = [0.001] # , 0.003, 0.0001, 0.0003
    # 模型参数
    embed_dim_list = [20] # 40, 10,
    channelsList = [32] # , 32 , 8
    dropoutlist = [0.2] # , 0.2
    gammalist = [0.01] # , 0.01, 0.0001  , 1
    orderlist = [0.001] # , 0.1, 0.01, 0.0001  1,
    dilation = 2
    TCN_layers = 3
    sst_file = 'data/Bo_Hai.csv'  #Nan_Hai_samePaper.csv
    step_size = 20
    file = 'SST'
    nheadsList =1
    gdep_list = 2
    gat_featureList = 10
    weight_decaylist = [0] # 0.000001, 0.00001,
    for i in window_list:
        for p in pre_lenList:
            for channels in channelsList:
                for dropout in dropoutlist:
                    for gamma in gammalist:
                        for order in orderlist:
                            for embed in embed_dim_list:
                                for learning_rate in learning_rateList:
                                    for weight_decay in weight_decaylist:
                                        log_file = 'D:/first/GNN_SST-8-31-from3090/result/test/{}_{}_{}_{}_{}_{}_{}_{}_log'. \
                                            format(file, i, p, channels, dropout, gamma, order, embed)
                                        model_save_path = 'D:/first/GNN_SST-8-31-from3090/result/model/{}_{}_{}_{}_{}_{}_{}_{}'. \
                                            format(file, i, p, channels, dropout, gamma, order, embed)
                                        args.learning_rate = learning_rate
                                        args.gamma = gamma
                                        args.embed_dim = embed
                                        args.dropout = dropout
                                        args.channels = channels
                                        args.order = order
                                        args.log_file = log_file
                                        args.weight_decay = weight_decay
                                        args.dilation = dilation
                                        args.model_save_path = model_save_path
                                        args.epoch = epoch
                                        args.Pre_len = p
                                        args.dilation = dilation
                                        args.TCN_layers = TCN_layers
                                        args.batch_size = batch_size
                                        args.step_size = step_size
                                        args.nheads = nheadsList
                                        args.gdep = gdep_list
                                        args.gat_feature = gat_featureList

                                        args.sst_file = sst_file
                                        main(args)


