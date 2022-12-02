import datetime

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from sklearn import preprocessing
from scipy.stats import pearsonr
# import seaborn as sns
import matplotlib.pyplot as plt


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


def readData_to_csv(year):
    data = []
    path = 'D:/DataSet/OISST/sst.day.mean.' + year + '.v2.nc'
    f = Dataset(path)
    # print(f)
    lat1 = f.variables['lat'][:]
    sst = f.variables['sst'][:]
    lon1 = f.variables['lon'][:]
    # print(lat1, '\n', lon1)
    location_list = []
    for lat in range(6500, 21250, 250):  # first_island_chain: 6500, 30800 # 136_point :  36.5- 40.5 #37070, 41000
        for lon in range(112500, 119500, 250):  # first_: 105500, 130500 # 136_point : 117.5- 121.5 # 117350, 121100
            # 黄海 lat: 31667, 39833  lon: 119333, 126833
            # 渤海：lat： 36500, 40500 lon:117500, 121500
            # 东海： lat： 23.00 ， 33.05 lon： 117.0 ， 131
            # 南海 lat: 6.5, 21.25 lon: 109.5 119.5

            lat_X = int((lat / 1000 + 89.875) / 0.25)
            lon_Y = int(lon / 1000 / 0.25)
            # print(lat_X, '+', lon_Y)
            if sst[:, lat_X, lon_Y].tolist()[0] is None:
                pass
            else:
                data.append(sst[:, lat_X, lon_Y][:])
                if int(year) == 2010:
                    location_list.append((str(lat1[lat_X]) + ':' + str(lon1[lon_Y])))
                # print('lat_x:', str(lat / 1000), 'lon_y:', str(lon / 1000))
    # 时间是竖着的
    data = np.array(data).T
    data = pd.DataFrame(data)
    if int(year) == 1981:
        start_time = str(year) + '/09/01'
        end_time = str(year) + '/12/31'
    else:
        start_time = str(year) + '/01/01'
        end_time = str(year) + '/12/31'
    empty_dataframe = pd.DataFrame()
    # 先加上第一天
    # first_day = datetime.datetime.strptime(start_time, '%Y/%m/%d')
    first_day = pd.Series([start_time])
    empty_dataframe = empty_dataframe.append(first_day, ignore_index=True)
    get_time_number = get_time_sub(start_time=start_time, end_time=end_time)
    for i in range(get_time_number):
        result_time = add_on_day(start_time, 1)
        new_data = pd.Series([result_time])
        empty_dataframe = empty_dataframe.append(new_data, ignore_index=True)
        start_time = result_time
    if int(year) == 2010:
        first_index = pd.DataFrame(location_list).T
        data = pd.concat([first_index, data], axis=0)
    data = pd.concat([empty_dataframe, data], axis=1)
    print(data)

    with open('G:/MYPAPER/Second_Paper/Nan_Hai_samePaper.csv', mode='a', newline='') as f:
        data.to_csv(f, header=0, index=0)
    print('one year is complete')



for i in range(2010, 2021):
    print('start read {} data'.format(str(i)))
    readData_to_csv(str(i))
