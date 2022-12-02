import os

if __name__ == '__main__':

    # 时间窗口和学习率等
    epoch = 200
    window_list = [15]
    pre_lenList = [1]
    batch_size = 32  # 128
    learning_rate = 0.001
    # 模型参数
    embed_dim_list = [20] #10 20 40
    channelsList = [16] #16 32
    dropoutlist = [0.3]#0. 0.1 0.2 0.3
    gammalist = [0.0001]#1 0.01 0.0001
    orderlist = [0.01]#1 0.1 0.01
    dilation = 2
    TCN_layers = 3
    sst_file = 'D:/gzh/GAT-sst/data/Bo_Hai.csv'
    file = 'SST'
    for i in window_list:
        for p in pre_lenList:
            for channels in channelsList:
                for dropout in dropoutlist:
                    for gamma in gammalist:
                        for order in orderlist:
                            for embed in embed_dim_list:
                                log_file = 'D:/gzh/GAT-sst/log/{}_{}_{}_{}_{}_{}_{}_log'. \
                                    format(file, i, p, channels, dropout, gamma, order, embed)
                                model_save_path = 'D:/gzh/GAT-sst/model/{}_{}_{}_{}_{}_{}_{}'. \
                                    format(file, i, p, channels, dropout, gamma, order, embed)

                                os.system(
                                    'python graph_Embedding.py '
                                    '--epoch {} --Windows_size {} '
                                    '--Pre_len {} --batch_size {} '
                                    '--learning_rate {} --gamma {}'
                                    '--embed_dim {} --dropout {} --channels {} --dilation {} '
                                    '--TCN_layers {} --order {}'
                                    ' --log_file {} --model_save_path {} --sst_file {} '
                                        .format(epoch, i, p, batch_size,learning_rate, gamma, embed, dropout,
                                                channels, dilation, TCN_layers, order, log_file,model_save_path,
                                                sst_file))
