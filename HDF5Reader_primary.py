# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 15:09:11 2025

@author: mirag
"""
import h5py as h5
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import traceback
from types import TracebackType
from typing import Literal, Optional, Type

def safe_remove_file(filepath: str, max_retries: int = 5, retry_delay: int = 1):
    """安全删除文件，带有重试机制"""
    for attempt in range(max_retries): # 最多重试 max_retries 次
        try:
            # 避免文件不存在造成的报错
            if os.path.exists(filepath):
                os.remove(filepath) # 删除文件
            return True
        # 仅对 PermissionError 响应
        except PermissionError:
            # 达到 max_retries 次，不再重新尝试
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay) # 休眠 retry_delay 秒，避免过快频繁尝试
    return False # 表明出现 PermissionError 以外的 Error 达 max_retries 次

class HDF5reader_writer:
    """
    HDF5 读写器，支持'r','w'和'a'模式，但是 with 语句内读写分离
    支持读入 Observations 组中指定变量的数据和属性以及全局属性
    支持 summary_meteorological,读取数据并打印全局属性的部分信息
    请尽可能使用 with 语句进行数据读取而不是实例化，以防遗忘关闭文件
    """  
    
    def __init__(self, 
                 file_path: str, 
                 mode: Literal['r', 'w', 'a'] = 'r',  # 限定合法模式
                 ) -> None:
        """
        初始化 HDF5Reader 实例，默认 dataset 为 None
        
        参数:
            file_path: HDF5 文件的路径
            mode: 打开方式
        """
        self.__file_path = file_path
        self.__mode = mode
        self.__dataset = None    
    
    def __enter__(self):
        """with语句内确保文件可以打开，不成功即报错"""
        try:
            self.__openhdf5()
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to open HDF5 file: {e}")
        
    def __exit__(self, exc_type: Optional[Type[BaseException]], 
                 exc_val: Optional[BaseException], 
                 exc_tb: Optional[TracebackType]) -> Optional[bool]:
        """ 
        with语句内确保即使发生异常，文件可以关闭
        
        参数:
            exc_type: 若发生异常则为异常类型
            exc_val: 若发生异常则为异常实例
            exc_tb: 若发生异常则为追溯对象
        """
        try:
            self.close()
        except Exception as e:
            print(f"Error during file closing: {e}")  # 文件关闭时发生错误
            raise
        # __exit__ 本身就会确保执行
        if exc_type is not None:
            print("\n=== Exception occurred inside the 'with' block ===")
            print(f"Exception Type: {exc_type}")  # 异常类型
            print(f"Exception Value: {exc_val}")  # 异常值
            print("\nTraceback (most recent call last):")  # 异常堆栈（最近调用）
            traceback.print_tb(exc_tb)  # 打印 traceback 信息
            print("============================================\n")
    
    def __openhdf5(self):
        """打开 HDF5 文件"""
        max_retries = 5  # 最大重试次数
        retry_delay = 1  # 重试间隔(秒)
        # 定义异常类型与对应错误信息的映射
        ERROR_MESSAGES = {
            FileNotFoundError: lambda path: f"File not found: {path}",
            PermissionError: lambda _: "Check the file read permissions",
            RuntimeError: lambda _: "The parsing of the NetCDF file failed"
            }
        for attempt in range(max_retries):
            try:
                # 如果是写入模式，先尝试删除现有文件
                if self.__mode == 'w':
                    safe_remove_file(self.__file_path)
                self.__dataset = h5.File(self.__file_path, self.__mode)  # 获得file_path的句柄
                return
            except tuple(ERROR_MESSAGES.keys()) as e:
                if attempt == max_retries - 1:
                    # 从映射中获取对应的错误信息生成函数并调用
                    error_msg = ERROR_MESSAGES[type(e)](self.__file_path)
                    raise type(e)(error_msg) from e
            except Exception:
                if attempt == max_retries - 1:
                    raise
            time.sleep(retry_delay)  # 重试前等待..
            
    def close(self):
        """关闭 HDF5 文件，释放资源"""
        if self.__dataset is not None:
            try:
                self.__dataset.close()
            except Exception as e:
                print(f"Warning: Error closing HDF5 file: {e}")
            finally:
                self.__dataset = None # 保证数据集被清空
        
    def get_dataset(self) -> h5.File:
        """
        获取 HDF5 数据集，自动判别 Observations 组是否存在
        如果文件尚未打开，则自动调用 open() 打开文件。
            
        返回:
            dataset: HDF5 数据集对象
            
        异常:
            KeyError: 当 Observations 组不存在时
        """
        if self.__dataset is None:
            self.__openhdf5()
        if "Observations" not in self.__dataset:
            raise KeyError("Group 'Observations' not found in the file.")
        return self.__dataset
    
    def get_variable_data(self, variable_name: str) -> np.ndarray:
        """
        读取 Observations 组中指定变量的数据值
        
        参数:
            variable_name: 目标变量名称
            
        返回:
            numpy.ndarray: 该变量的数据值
            
        异常:
            ValueError: 如果指定变量不存在于数据集中
        """   
        h5_file = self.get_dataset()
        obs_group = h5_file["Observations"]
        if variable_name in obs_group:
            data = obs_group[variable_name][()]
            return data
        else:
            raise ValueError(f"Variable '{variable_name}' not found.")
    
    def get_global_attributes(self) -> dict:
        """
        读取 HDF5 文件的全局属性，并以字典形式返回。
        
        返回:
            dict: 包含全局属性的字典。
        """   
        h5_file = self.get_dataset()
        attributes = dict(h5_file.attrs)
        return attributes
    
    def get_local_attributes(self, dataset_name: str) -> dict:
        """
        读取 Observations 组中指定数据集的局部属性，并以字典形式返回。
        
        参数:
            dataset_name: 要读取属性的数据集名称。
        
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
        """读取 HDF5 文件中的气象数据，并打印全局属性和 Observations 组的部分信息。"""
        print("全局属性：")
        ds = self.get_dataset()
        for key, value in ds.attrs.items():
            print(f"  {key}: {value}")
        # 进入 'Observations' 组
        obs_group = ds['Observations']
        print("\nObservations 组中的数据集：", list(obs_group.keys()))
    
    def __check(self, mode, time_points, lat_points, lon_points, 
                lat_min, lat_max, lon_min, lon_max):
        """检查写入数据是否合理"""
        # 检查模式是否匹配
        if self.__mode != mode:
            raise PermissionError(f"File must be opened in '{mode}' mode for this operation")
        # 检查文件是否已打开
        if self.__dataset is None:
            raise RuntimeError("HDF5 file is not open. Use 'with' statement to open the file.")
        # 检查经纬度和时间范围是否合理
        if not (lat_points > 0 and lon_points > 0 and time_points > 0):
            raise ValueError("lat_points, lon_points and time_points must be positive")
        if not (isinstance(lat_points, int) and isinstance(lon_points, int) and isinstance(time_points, int)):
            raise TypeError("lat_points, lon_points and time_points must be integers")
        if lat_min >= lat_max or lon_min >= lon_max:
            raise ValueError("lat_min must be < lat_max and lon_min must be < lon_max")
            
    def __generate_time_values(self, time_points, time_values):
        """生成时间数据"""
        try:
            if time_points is not None:
                if time_values is None:
                    time_values = np.arange(time_points)  # 默认生成0,1,2,...的时间序列
                elif len(time_values) != time_points:
                    raise ValueError("time_values length must match time_points")
            return time_values
        except Exception:
            raise
    
    def __generate_lon_lat_values(self, lat_points, lon_points, 
                                  lat_min, lat_max, lon_min, lon_max):        
        """生成经纬度数据"""
        try:
            latitudes = np.linspace(lat_min, lat_max, lat_points)
            longitudes = np.linspace(lon_min, lon_max, lon_points)
            return latitudes, longitudes
        except TypeError as e:
            raise TypeError(f"Input must be numeric: {e}")
    
    def __write_primary(self, time_values, latitudes, longitudes, dic_data):
        """write 方法的现行书写内容"""
        # 写入全局属性
        self.__dataset.attrs['CreationDate'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.__dataset.attrs['DataSource'] = 'Simulated meteorological data'
        self.__dataset.attrs['Description'] = f'meteorological data, include {dic_data.keys()}'
        # 创建组结构
        obs_group = self.__dataset.create_group('Observations')
        coord_group = obs_group.create_group('Coordinates')
        # 写入坐标
        coord_group.create_dataset('Time', data=time_values, dtype='int64')  # 明确指定数据类型
        coord_group.create_dataset('Latitude', data=latitudes)
        coord_group.create_dataset('Longitude', data=longitudes)
        
    def __write_observation(self, expected_shape, group, dic_data, 
                            chunks=None, compression="gzip", compression_opts=4):
        """写入dic_data"""
        if not dic_data:
            return
        # 遍历 dic_data，写入所有变量
        for var_name, var_info in dic_data.items():
            # 检查 var_info 的类型
            if not isinstance(var_info, dict):
                raise ValueError(f"Variable '{var_name}' info must be a dictionary")
            try:
                # 读取 dic_data 提供的数据
                data = var_info["data"]
                units = var_info.get("units", "unknown")
                description = var_info.get("description", "no description")
                # 检查 data 的形状
                if data.shape != expected_shape:
                    raise ValueError(f"Data shape mismatch for {var_name}: "
                                     f"expected ({expected_shape}), got {data.shape}")
                # 写入 data 的数据、单位与描述
                dset = group.create_dataset(
                    var_name, 
                    data=data, 
                    chunks=chunks,  # 分块设置
                    compression=compression,  # 压缩算法
                    compression_opts=compression_opts,  # 压缩级别
                    shuffle=True  # 提高压缩率
                    )
                dset.attrs["units"] = units
                dset.attrs["description"] = description
            except KeyError as e:
                raise KeyError("Missing required key '{e.args[0]}' in variable '{var_name}'")
            except Exception:
                raise
        
    def write_meteo_hdf5(self, time_points: int, lat_points: int, lon_points: int, 
                         lat_min: float = -90, lat_max: float = 90, 
                         lon_min: float = -180, lon_max: float = 180, 
                         time_values = None, dic_data = None):
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
                        "data": ndarray,    # 数据
                        "units": str,       # 单位
                        "description": str  # 描述
                    },
                    ...
                }
        """
        # 检查模式是否匹配，文件是否已打开，经纬度和时间范围是否合理
        self.__check('w', time_points, lat_points, lon_points, 
                     lat_min, lat_max, lon_min, lon_max)
        # 检查是否有 dic_data
        if dic_data is None:
            dic_data = {}
        # 清空现有数据（如果存在）
        for key in list(self.__dataset.keys()):
            del self.__dataset[key]             
        # 生成时间数据
        time_values = self.__generate_time_values(time_points, time_values)
        # 生成经纬度数据
        latitudes, longitudes = self.__generate_lon_lat_values(lat_points, lon_points, 
                                                               lat_min, lat_max, lon_min, lon_max)
        # 写入全局属性，创建组结构，并写入坐标
        self.__write_primary(time_values, latitudes, longitudes, dic_data)
        # 遍历 dic_data，写入所有变量
        expected_shape = (time_points, lat_points, lon_points)
        group = self.__dataset.require_group('Observations')
        self.__write_observation(expected_shape, group, dic_data)
    
    def append_meteo_hdf5(self, time_points, lat_points, lon_points, 
                          lat_min=-90, lat_max=90, lon_min=-180, lon_max=180, 
                          time_values=None, dic_data=None):
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
        # 检查模式是否匹配，文件是否已打开，经纬度和时间范围是否合理
        self.__check('a', time_points, lat_points, lon_points, 
                     lat_min, lat_max, lon_min, lon_max)
        # 检查是否有 dic_data
        if dic_data is None:
            dic_data = {}
        # 遍历 dic_data，写入所有变量
        expected_shape = (time_points, lat_points, lon_points)
        group = self.__dataset.require_group('Observations')
        self.__write_observation(expected_shape, group, dic_data)

# ---------------------- 实用案例 ----------------------
# 规定经纬和时间网格点数
time_points = 2
lat_points = 6
lon_points = 6
# 生成时间序列
start_time = "2023-01-01 00:00"
step = pd.Timedelta(hours=6)
times_value = pd.date_range(start = start_time, periods = time_points, freq = step)
time_values = times_value.astype(np.int64) // 10**9  # 秒级时间戳
# 生成随机气象数据
temperature = np.random.uniform(low = -20, high = 40, size = (time_points, lat_points, lon_points))
humidity = np.random.uniform(low = 0, high = 100, size = (time_points, lat_points, lon_points))
# 封装数据为指定格式 dict
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
        }
    }
# 写入数据
with HDF5reader_writer("D:/test/hdf5_test_1.h5", 'w') as h5file:
    h5file.write_meteo_hdf5(time_points = time_points, lat_points = lat_points, lon_points = lon_points, 
                            lat_min = -60, lat_max = -30, lon_min = 30, lon_max = 60, 
                            time_values = times_value, dic_data = dic_data)
# 读取 HDF5 格式数据特征
with HDF5reader_writer("D:/test/hdf5_test_1.h5", 'r') as h5file:
    print(f'global attributes : {h5file.get_global_attributes()}')
    print(f'Humidity data : {h5file.get_variable_data("Humidity")}')
    print(f'Temperature attributes : {h5file.get_local_attributes("Temperature")}')
# 生成随机气象数据
windspeed = np.abs(np.random.normal(3, 2, size=(time_points, lat_points, lon_points)))
winddirection = np.random.uniform(low=0, high=360, size=(time_points, lat_points, lon_points))
# 封装数据为指定格式 dict
dic_data = {
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
# 追加数据
with HDF5reader_writer("D:/test/hdf5_test_1.h5", 'a') as h5file:
    h5file.append_meteo_hdf5(time_points=time_points,lat_points=lat_points, lon_points=lon_points, 
                             lat_min=-60, lat_max=-30, lon_min=30, lon_max=60, 
                             time_values=times_value, dic_data=dic_data)
# 读取 HDF5 格式数据特征
with HDF5reader_writer("D:/test/hdf5_test_1.h5", 'r') as h5file:
    print(f'Wind speed data : {h5file.get_variable_data("WindSpeed")}')
    print(f'Wind direction attributes : {h5file.get_local_attributes("WindDirection")}')
    h5file.summary_meteorological()