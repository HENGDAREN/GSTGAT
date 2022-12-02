import numpy as np
import pandas as pd
from datetime import date
import datetime
import time
from torch.utils import data
import torch
import os


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


def split_windows(data, window_size, pre_len):
    # 函数是用来切分时间窗口数据的
    num_steps, dim = data.shape
    num_samples = num_steps - window_size - pre_len + 1
    data_x = np.zeros(shape=(num_samples, window_size, dim))
    data_y = np.zeros(shape=(num_samples, pre_len, dim))
    for i in range(num_samples):
        data_x[i] = data[i: i + window_size]
        data_y[i] = data[i + window_size: i + window_size + pre_len]
    return data_x, data_y


def loadData(args):
    '''
    这里数据并不是原始数据，是已经从nc文件中读取出来的数据，存成csv
    '''
    data_path = os.path.join(os.getcwd(), args.sst_file)
    data_pd = pd.read_csv(data_path)
    # 开始取数据, 下面这个需要从第一列开始取，第一列是时间嘛， data = 2557，5051
    data = data_pd.iloc[:, 1:].values
    # num_days = 2557
    num_days = len(data)
    # 训练集，测试集，验证集
    split_point = int(num_days * args.train_ratio)
    val_split_point = int(split_point * (1 - args.vaild_ratio))
    train_data, test_data, vaild_data = data[:val_split_point], data[split_point:], data[val_split_point: split_point]
    # 在这里进行归一化, !!!!!!!!!!!!!!!!需要注意的是，这个时间嵌入可没有归一化啊
    # todo 这里这个归一化一定要注意，因为这个是对整个数据集进行了归一化，嗯不是按列来的归一化，所以效果怎么样，不知道
    mean, std = np.mean(train_data), np.std(train_data)
    train_data = (train_data - mean) / std
    vaild_data = (vaild_data - mean) / std
    test_data = (test_data - mean) / std
    # 然后对三个数据集进行时间窗口切割，对吧
    train_X, train_Y = split_windows(train_data, args.Windows_size, args.Pre_len)
    val_X, val_Y = split_windows(vaild_data, args.Windows_size, args.Pre_len)
    test_X, test_Y = split_windows(test_data, args.Windows_size, args.Pre_len)

    # 时间嵌入
    # 下面将原始object类型转换为datetime类型
    time_data = data_pd.iloc[:, 0].astype('datetime64[ns]')
    # 这里直接调用weekday函数,获得每天是礼拜几，嗯，然后我还需要获得每个月, weeklist=2557
    weeklist = list(map(lambda t: datetime.datetime.weekday(t), time_data))
    # monthlist=2557
    monthlist = list(map(lambda t: int(str(t)[5:7]) - 1, time_data))
    week_index, month_index = np.array(weeklist).reshape(-1, 1), np.array(monthlist).reshape(-1, 1)
    # time_embed = 2557, 2
    time_embed = np.concatenate((month_index, week_index), axis=1)
    # 返回三个数据集的时间嵌入
    train_TE = time_embed[:val_split_point]
    val_TE = time_embed[val_split_point: split_point]
    test_TE = time_embed[split_point:]
    # 切分时间窗口
    # 注意这里面的维度，这是时间窗口+预测长度
    # trainTE = (1620, 16, 2)
    trainTE = split_windows(train_TE, args.Windows_size, args.Pre_len)
    trainTE = np.concatenate(trainTE, axis=1).astype(np.int32)

    valTE = split_windows(val_TE, args.Windows_size, args.Pre_len)
    valTE = np.concatenate(valTE, axis=1).astype(np.int32)

    testTE = split_windows(test_TE, args.Windows_size, args.Pre_len)
    testTE = np.concatenate(testTE, axis=1).astype(np.int32)
    return train_X, trainTE, train_Y, val_X, valTE, val_Y, test_X, testTE, test_Y, mean, std


class Dataset(data.Dataset):
    def __init__(self, data_x, data_y, time_embed):
        self.dataX = data_x
        self.dataLabel = data_y
        self.embed = time_embed
        # 转成one_hot必须是longtensor，所以self.embed加了类型转换
        self.dataX, self.dataLabel, self.embed = torch.tensor(self.dataX, dtype=torch.float32), \
                                                 torch.tensor(self.dataLabel, dtype=torch.float32), \
                                                 torch.tensor(self.embed, dtype=torch.int64)
        print('After Dataset shape, trainx: {}, trainy: {}, train_TE: {}'
              .format(self.dataX.shape, self.dataLabel.shape, self.embed.shape))

    def __getitem__(self, item):
        return self.dataX[item], self.dataLabel[item], self.embed[item]

    def __len__(self):
        # 这个就返回一个长度就行了
        return self.dataX.shape[0]

from metrics import GAT_Metrics
# metric
# def metric_self(pred, label):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         # 这一步出来的是bool型，Ture， false
#         mask = np.not_equal(label, 0)
#         # 就是将bool型转换为浮点型
#         mask = mask.astype(np.float32)
#         # 数值还是一样的，但是都变大了，如果有0的话, np.mean返回的是一个数值
#         mask /= np.mean(mask)
#         mae = np.abs(np.subtract(pred, label)).astype(np.float32)
#         mse = np.square(mae)
#         #rmse = np.square(mae)
#         mape = np.divide(mae, label)
#
#         mae = np.nan_to_num(mae * mask)
#         mae = np.mean(mae)
#         mse = np.nan_to_num(mse * mask)
#         mse = np.mean(mse)
#         #rmse = np.nan_to_num(rmse * mask)
#         #rmse = np.sqrt(np.mean(rmse))
#         mape = np.nan_to_num(mape * mask)
#         mape = np.mean(mape)
#     return mae, mse, mape
def metric_self(pred, label):
    mae, mse, mape = GAT_Metrics(pred, label)

    return mae, mse, mape