import numpy as np
import pandas as pd
from netCDF4 import Dataset
import os
import time
import datetime
# todo 2021-1-7开始写


def get_time_sub(start_time, end_time):
    start_time = str(start_time)
    end_time = str(end_time)
    # print(start_time)
    # print(end_time)
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    sub = end_time - start_time
    sub_string = str(sub)
    result = str.split(sub_string)[0]
    return int(result)


def add_on_day(start_string, add_count):
    start_string = str(start_string).split(' ')[0]
    date = datetime.datetime.strptime(start_string, '%Y/%m/%d')
    # y, m, d = date[:3]
    delta = datetime.timedelta(days=add_count)
    # date_result = datetime.datetime(y, m, d) + delta
    date_result = date + delta
    date_result = date_result.strftime('%Y/%m/%d')
    return date_result

def get_sst(start_year, end_year, lat_x=-55, lat_y=60, lon_x=0, lon_y=180, time_gap=1, lat_gap=8, lon_gap=8):
    # 这个方法简单直接，但是对于内存要求太高了，数据时间跨度一长就崩溃了
    result = []
    # 获得数据中的经纬度
    lat_s = int((lat_x + 89.875) / 0.25)
    lat_e = int((lat_y + 89.875) / 0.25)
    # 这个也是一年一年的处理呢
    for i in range(start_year, end_year):
        file_path = r'D:\DataSet\OISST\sst.day.mean.{}.v2.nc'.format(i)
        print('start process {} year data'.format(i))
        file = Dataset(file_path)
        # 这个就是取维度，经度全取
        sst = file['sst'][:, lat_s:lat_e, :]
        # 等间隔取值，
        sst_r = sst[::time_gap, ::lat_gap, ::lon_gap]
        result.append(sst_r)

    # 这里返回mask矩阵，用来去掉陆地元素
    sst_mask_path = r'F:/Dataset/lsmask.oisst.v2.nc'
    mask = Dataset(sst_mask_path)
    # 保证相同的经纬度，间隔，确保相同位置
    sst_mask = mask['lsmask'][:, lat_s: lat_e, :][::lat_gap, ::lon_gap]
    result = np.array(result)
    # 讲result按照时间连接起来
    result = np.concatenate(result, axis=0)
    print(result.shape)
    return result, sst_mask


def get_sst_flow(start_year, end_year, save_path, is_save,
                 lat_x=-55, lat_y=60, lon_x=0, lon_y=180, time_gap=1, lat_gap=8, lon_gap=8):
    # 这个方法简单直接，但是对于内存要求太高了，数据时间跨度一长就崩溃了
    result = []
    # 获得数据中的经纬度
    lat_s = int((lat_x + 89.875) / 0.25)
    lat_e = int((lat_y + 89.875) / 0.25)

    # 这里返回mask矩阵，用来去掉陆地元素
    sst_mask_path = r'F:/Dataset/lsmask.oisst.v2.nc'
    mask_r = Dataset(sst_mask_path)
    # 保证相同的经纬度，间隔，确保相同位置 sst_mask = 58, 120
    sst_mask = mask_r['lsmask'][0, lat_s: lat_e, :][::lat_gap, ::lon_gap]
    # 这里可以返回索引,原始np.where返回的是每一维的索引，这里有二维数组，所以返回两个索引
    # row_index, col_index = np.where(sst_mask == 0)
    # index = list(zip(row_index, col_index))
    # 但是下面删除的时候需要拉平，反正挺麻烦的，所以这里就拉平了,
    # 原始输出是这样的(array([  97,  216,  217, ..., 6935, 6936, 6938], dtype=int64),)
    sst_mask = sst_mask.reshape(-1,)
    # 取0是因为这里返回的是一个tuple类型，所以取0了
    index = np.where(sst_mask == 0)[0]
    # 这个也是一年一年的处理呢
    # ------------------------------------------------------------------------------------取到索引
    sst_file = Dataset(r'D:\DataSet\OISST\sst.day.mean.{}.v2.nc'.format(start_year))
    # lat_SST = 720,
    lat_SST = sst_file['lat'][lat_s:lat_e]
    # lon_SST = 1440
    lon_SST = sst_file['lon'][:]
    # print(lat_SST, lat_SST.shape, lon_SST, lon_SST.shape)
    lat_SST = lat_SST[::lat_gap]
    # print(len(lat_SST))
    lon_SST = lon_SST[::lon_gap]
    # 下面循环走完之后，长度是6960
    sst_index = []
    for i in lat_SST:
        for j in lon_SST:
            sst_index.append('{}:{}'.format(i, j))
    # 5051
    sst_index = np.delete(np.array(sst_index), index)
    # print(sst_index, '\n', len(sst_index))

    for i in range(start_year, end_year):
        file_path = r'D:\DataSet\OISST\sst.day.mean.{}.v2.nc'.format(i)
        print('start process {} year data'.format(i))
        file = Dataset(file_path)
        # print(file)
        # 这个就是取维度，经度全取
        sst = file['sst'][:, lat_s:lat_e, :]
        # 等间隔取值，
        sst_r = sst[::time_gap, ::lat_gap, ::lon_gap]
        # sst_r = 365, 58, 120
        sst_r = np.array(sst_r)
        sst_day, sst_lat, sst_lon = sst_r.shape
        # sst_r = 365, 6960
        sst_r = sst_r.reshape(sst_day, sst_lat*sst_lon)

        # sst_result = 365, 5051
        sst_result = np.delete(sst_r, index, axis=1)
        # print(sst_result, type(sst_result))
        # 保存文件, 首先转为了pandas，转成这样就可以了，行，今天先写成这样，明天再来接着写，反正最主要的都已经写完了
        sst_result = pd.DataFrame(sst_result)
        # ---------------------------------------添加地点索引
        if i == start_year:
            sst_result = pd.concat([pd.DataFrame(sst_index).T, sst_result], axis=0)
        # ---------------------------------------添加时间索引
        empty_dataframe = pd.DataFrame()
        # 先加上第一天
        if int(i) == 1981:
            start_time = str(i) + '/09/01'
            end_time = str(i) + '/12/31'
        else:
            start_time = str(i) + '/01/01'
            end_time = str(i) + '/12/31'
        first_day = pd.Series([start_time])
        empty_dataframe = empty_dataframe.append(first_day, ignore_index=True)
        get_time_number = get_time_sub(start_time=start_time, end_time=end_time)
        for i in range(get_time_number):
            result_time = add_on_day(start_time, 1)
            new_data = pd.Series([result_time])
            empty_dataframe = empty_dataframe.append(new_data, ignore_index=True)
            start_time = result_time
        sst_result = pd.concat([empty_dataframe, sst_result], axis=1)
        print(sst_result)
        if is_save:
            with open(save_path, mode='a', newline='') as f:
                sst_result.to_csv(f, header=0, index=0)


if __name__ == '__main__':
    # 是否保存文件
    save = 1
    # 保存文件
    save_path = r'G:\MYPAPER\Second_Paper\10_20_sst.csv'
    if os.path.exists(save_path):
        os.makedirs(save_path)
        print('新建文件路径：{}'.format(save_path))
    else:
        print('save_path 是 {}'.format(save_path))
    get_sst_flow(2010, 2021, save_path, save, lat_gap=8, lon_gap=12)







