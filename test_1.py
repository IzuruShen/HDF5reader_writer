# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 09:53:50 2025

@author: mirag
"""
import pandas as pd
import numpy as np
import os
from HDF5Reader_primary import HDF5reader_writer

hdf5_test = HDF5reader_writer("D://test//hdf5_test.h5")
# 时空网格点
lat_points = 6
lon_points = 6
start_time = "2025-01-01 00:00"     
step = pd.Timedelta(hours=6)        
time_points = 2                     

# 生成时间序列
times_value = pd.date_range(
    start = start_time, 
    periods = time_points, 
    freq = step
)
time_values = times_value.astype(np.int64) // 10**9  # 秒级时间戳

# 生成随机气象数据
temperature = np.random.uniform(low=-20, high=40, size=(time_points, lat_points, lon_points))
humidity = np.random.uniform(low=0, high=100, size=(time_points, lat_points, lon_points))
pressure = np.random.uniform(low=950, high=1050, size=(time_points, lat_points, lon_points))
windspeed = np.abs(np.random.normal(3, 2, size=(time_points, lat_points, lon_points)))
winddirection = np.random.uniform(low=0, high=360, size=(time_points, lat_points, lon_points))
dic_data = {
    'Temperature': {
        "data": temperature,
        "units": "°C",
        "description": "temperature"
        },
    'Humidity': {
        "data": humidity,
        "units": "%",
        "description": "humidity"
        },
    'Pressure': {
        "data": pressure,
        "units": "hPa",
        "description": "pressure"
        },
    'WindSpeed': {
        "data": windspeed,
        "units": "m/s",
        "description": "wind_speed"
        },
    'WindDirection': {
        "data": winddirection,
        "units": "°",
        "description": "wind_direction"
        }
    }

#测试写入功能
hdf5_test.write_meteo_hdf5(time_points=time_points,lat_points=lat_points, lon_points=lon_points, 
                           lat_min=-60, lat_max=-30, lon_min=30, lon_max=60,
                            time_values=None, dic_data=dic_data)

#测试文件是否创建成功
assert os.path.exists("D://test//hdf5_test.h5"), "failed"

#测试类的功能是否实现
with HDF5reader_writer("D://test//hdf5_test.h5") as h5file:
    # 验证基本功能
    assert isinstance(h5file.get_global_attributes(), dict)
    assert h5file.get_variable_data("Humidity").shape == (time_points, lat_points, lon_points)
    assert h5file.get_local_attributes("Temperature")["units"] == "°C"
    
    #测试数据读取是否一致
    # 验证温度数据形状
    temp_data = h5file.get_variable_data("Temperature")
    assert temp_data.shape == (time_points, lat_points, lon_points), "ShapeError"
    # 验证湿度单位
    humidity_attrs = h5file.get_local_attributes("Humidity")
    assert humidity_attrs["units"] == "%", "UnitError"

#测试追加功能
new_data = {
    "Precipitation": {
        "data": np.random.rand(time_points, lat_points, lon_points),
        "units": "mm",
        "description": "precipitation"
    }
}

hdf5_test.append_meteo_hdf5(
    time_points=time_points,
    lat_points=lat_points,
    lon_points=lon_points,
    dic_data=new_data
)

# 验证追加的数据
with HDF5reader_writer("D://test//hdf5_test.h5") as h5file:
    assert "Precipitation" in h5file.get_dataset()["Observations"]
    precip_attrs = h5file.get_local_attributes("Precipitation")
    assert precip_attrs["units"] == "mm"
    assert precip_attrs["description"] == "precipitation"

print("所有基础测试通过！")