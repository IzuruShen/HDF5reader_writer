# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 15:09:11 2025

@author: mirag
"""

import h5py as h5
import numpy as np
from datetime import datetime
import time
import pandas as pd

class HDF5reader_writer:
    """
        HDF5读写器，支持'r','w''a'模式
        支持读入 Observations 组中指定变量的数据和属性以及全局属性
        支持summary_meteorological,读取数据并打印全局属性的部分信息
        请尽可能使用with语句而不是实例化
    """
    def __init__(self, file_path):
        """
        初始化 HDF5Reader 实例，使用组合模式整合各功能模块
        参数:
            file_path(str): HDF5 文件的路径
        """
        self.__file_path = file_path
        self.__dataset = None

    def __openhdf5(self, mode='r'):
        """
        打开 HDF5 文件
        参数：
            mode:读入模式
        返回:
            数据集对象
        """
        max_retries = 5  # 最大重试次数
        retry_delay = 1  # 重试间隔(秒)
        for attempt in range(max_retries):
            try:
                self.__dataset = h5.File(self.__file_path, mode)
            except FileNotFoundError as e:
                if attempt == max_retries - 1:
                    raise FileNotFoundError(f"File not found: {self.__file_path}") from e
            except PermissionError as e:
                if attempt == max_retries - 1:
                    raise PermissionError("Check the file read permissions") from e
            except RuntimeError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError("The parsing of the NetCDF file failed") from e
            except Exception:
                if attempt == max_retries - 1:
                    raise
            time.sleep(retry_delay)  # 重试前等待..
    
    def __enter__(self):
        """with语句内确保文件可以打开"""
        try:
            self.__openhdf5('r')  # 打开文件
            return self         # 返回实例自身，供 with 块使用
        except Exception:
            raise              # 重新抛出异常
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        with语句内确保即使发生异常，文件可以关闭
        请尽可能使用with语句而不是实例化，这样可以规避遗忘finally close的风险
        """
        self.close()  # 无论如何，先确保关闭文件
    
    def close(self):
        """关闭 HDF5 文件，释放资源"""
        if self.__dataset is not None:
            self.__dataset.close()
            self.__dataset = None
    
    def get_dataset(self, mode='r'):
        """
        获取 HDF5 数据集
        如果文件尚未打开，则自动调用 open() 打开文件。
        
        参数：
            mode:读入模式，一般默认'r'
            
        返回:
            dataset: HDF5 数据集对象
            
        异常:
            KeyError: 当 Observations 组不存在时
        """
        if self.__dataset is None:
            self.__openhdf5(mode)
        if "Observations" not in self.__dataset:
            raise KeyError("Group 'Observations' not found in the file.")
        return self.__dataset
    
    def get_variable_data(self, variable_name):
        """
        读取 Observations 组中指定变量的数据值

        参数:
            variable_name (str): 目标变量名称

        返回:
            numpy.ndarray: 该变量的数据值

        异常:
            ValueError: 如果指定变量不存在于数据集中
        """   
        h5_file=self.get_dataset()
        obs_group = h5_file["Observations"]
        if variable_name in obs_group:
            data = obs_group[variable_name][()]
            return data
        else:
            raise ValueError(f"Variable '{variable_name}' not found.")
    
    def get_global_attributes(self):
        """
        读取 HDF5 文件的全局属性，并以字典形式返回。
        
        返回:
            dict: 包含全局属性的字典。
        """   
        h5_file = self.get_dataset()
        attributes = dict(h5_file.attrs)
        return attributes
    
    def get_local_attributes(self, dataset_name):
        """
        读取 Observations 组中指定数据集的局部属性，并以字典形式返回。
        
        参数:
            dataset_name (str): 要读取属性的数据集名称。
        
        返回:
            dict: 包含该数据集局部属性的字典。
        
        异常:
            ValueError: 当指定的数据集不存在于 Observations 组中时抛出异常。
        """   
        h5_file=self.get_dataset()
        obs_group = h5_file["Observations"]
        if dataset_name in obs_group:
            attributes = dict(obs_group[dataset_name].attrs)
            return attributes
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found in the Observations group.")
    
    def summary_meteorological(self):
        """
        读取 HDF5 文件中的气象数据，并打印全局属性的部分信息。
    
        """
        print("全局属性：")
        ds = self.get_dataset()
        for key, value in ds.attrs.items():
            print(f"  {key}: {value}")
        # 进入 'Observations' 组
        obs_group = ds['Observations']
        print("\nObservations 组中的数据集：", list(obs_group.keys()))
    
    # 气象数据操作方法
    def write_meteo_hdf5(self, time_points, lat_points, lon_points, 
                         lat_min=-90, lat_max=90, lon_min=-180, lon_max=180,
                         time_values=None,
                         dic_data=None):
        """
        创建一个 HDF5 文件或完全覆盖之前的文件，并写入数据
        所有变量均包含 units 和 description 属性。
        
        参数:
            time_points, lat_points, lon_points: 时空网格数（必须为正整数）
            lat_min, lat_max, lon_min, lon_max: 经纬度范围（默认全球）
            time_values: 时间值数组（若未提供则默认生成从0开始的等间隔时间）
            dic_data: 字典，允许为None，格式为:
                {
                    "var_name1": {
                        "data": ndarray,  # 数据
                        "units": str,      # 单位
                        "description": str # 描述
                    },
                    ...
                }
        """
        if dic_data is None:
            dic_data = {}
            
        # 生成时间数据
        if time_points is not None:
            if time_values is None:
                time_values = np.arange(time_points)  # 默认生成0,1,2,...的时间序列
            elif len(time_values) != time_points:
                raise ValueError("time_values length must match time_points")
                
        # 生成经纬度数据
        if not (lat_points > 0 and lon_points > 0):
            raise ValueError("lat_points and lon_points must be positive")
        if not (isinstance(lat_points, int) and isinstance(lon_points, int)):
            raise TypeError("lat_points and lon_points must be integers")
        if lat_min >= lat_max or lon_min >= lon_max:
            raise ValueError("lat_min must be < lat_max and lon_min must be < lon_max")
        try:
            latitudes = np.linspace(lat_min, lat_max, lat_points)
            longitudes = np.linspace(lon_min, lon_max, lon_points)
        except TypeError as e:
            raise TypeError(f"Input must be numeric: {e}")
        
        # 打开或创建 HDF5 文件，使用 'w' 模式会覆盖同名文件
        with h5.File(self.__file_path, 'w') as h5_file:
            # 写入全局属性
            h5_file.attrs['CreationDate'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            h5_file.attrs['DataSource'] = 'Simulated meteorological data'
            h5_file.attrs['Description'] = f'meteorological data, include {dic_data.keys()}'
            
            # 创建组 'Observations' 用于存储观测数据
            obs_group = h5_file.create_group('Observations')
            
            # 创建子组 'Coordinates' 存储时空信息
            coord_group = obs_group.create_group('Coordinates')
            coord_group.create_dataset('Time', data=time_values)
            coord_group.create_dataset('Latitude', data=latitudes)
            coord_group.create_dataset('Longitude', data=longitudes)
            
            # 遍历 dic_data，写入所有变量
            for var_name, var_info in dic_data.items():
                if not isinstance(var_info, dict):
                    raise ValueError(f"Variable '{var_name}' info must be a dictionary")
                try:
                    data = var_info["data"]
                    units = var_info.get("units", "unknown")
                    description = var_info.get("description", "no description")
                    
                    expected_shape_3d = (time_points, lat_points, lon_points)
    
                    if data.shape != expected_shape_3d:
                        raise ValueError(
                            f"Data shape mismatch for {var_name}: "
                            f"expected ({lat_points}, {lon_points}), got {data.shape}"
                        )
    
                    dset = obs_group.create_dataset(var_name, data=data)
                    dset.attrs["units"] = units
                    dset.attrs["description"] = description
                    
                except KeyError as e:
                    raise KeyError("Missing required key '{e.args[0]}' in variable '{var_name}'")
                except Exception:
                    raise 
    
    def append_meteo_hdf5(self, time_points, lat_points, lon_points,
                          lat_min=-90, lat_max=90, lon_min=-180, lon_max=180, 
                          time_values=None, 
                          dic_data=None):
        """
        在一个已有的 HDF5 文件之上追加 Observations 内的数据，或创建一个 HDF5 文件并写入数据
        所有变量均包含 units 和 description 属性。
        
        参数:
            time_points, lat_points, lon_points: 时空网格数（必须为正整数）
            lat_min, lat_max, lon_min, lon_max: 经纬度范围（默认全球）
            time_values: 时间值数组（若未提供则默认生成从0开始的等间隔时间）
            dic_data: 字典，允许为None，格式为:
                {
                    "var_name1": {
                        "data": ndarray,  # 数据
                        "units": str,      # 单位
                        "description": str # 描述
                    },
                    ...
                }
        """
        if dic_data is None:
            dic_data = {}
            
        # 生成时间数据
        if time_points is not None:
            if time_values is None:
                time_values = np.arange(time_points)  # 默认生成0,1,2,...的时间序列
            elif len(time_values) != time_points:
                raise ValueError("time_values length must match time_points")
                
        # 生成经纬度数据
        if not (lat_points > 0 and lon_points > 0):
            raise ValueError("lat_points and lon_points must be positive")
        if not (isinstance(lat_points, int) and isinstance(lon_points, int)):
            raise TypeError("lat_points and lon_points must be integers")
        if lat_min >= lat_max or lon_min >= lon_max:
            raise ValueError("lat_min must be < lat_max and lon_min must be < lon_max")
        # 打开或创建 HDF5 文件，使用 'a' 模式
        with h5.File(self.__file_path, 'a') as h5_file:
            obs_group = h5_file.require_group('Observations')
            # 遍历 dic_data，写入所有变量
            for var_name, var_info in dic_data.items():
                if not isinstance(var_info, dict):
                    raise ValueError(f"Variable '{var_name}' info must be a dictionary")
                try:
                    data = var_info["data"]
                    units = var_info.get("units", "unknown")
                    description = var_info.get("description", "no description")
    
                    expected_shape_3d = (time_points, lat_points, lon_points)
    
                    if data.shape != expected_shape_3d:
                        raise ValueError(
                            f"Data shape mismatch for {var_name}: "
                            f"expected ({lat_points}, {lon_points}), got {data.shape}"
                        )
    
                    dset = obs_group.create_dataset(var_name, data=data)
                    dset.attrs["units"] = units
                    dset.attrs["description"] = description
    
                except KeyError as e:
                    raise KeyError("Missing required key '{e.args[0]}' in variable '{var_name}'")
                except Exception:
                    raise 

hdf5_test = HDF5reader_writer("D://test//hdf5_test.h5")

lat_points = 6
lon_points = 6

start_time = "2023-01-01 00:00"     
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
hdf5_test.write_meteo_hdf5(time_points=time_points,lat_points=lat_points, lon_points=lon_points, 
                           lat_min=-60, lat_max=-30, lon_min=30, lon_max=60,
                            time_values=None, dic_data=dic_data)

with HDF5reader_writer("D://test//hdf5_test.h5") as h5file:
    print(f'global attributes : {h5file.get_global_attributes()}')
    print(f'Humidity data : {h5file.get_variable_data("Humidity")}')
    print(f'Temperature attributes : {h5file.get_local_attributes("Temperature")}')
    h5file.summary_meteorological()