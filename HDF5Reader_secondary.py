# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 20:45:11 2025

@author: mirag
"""

import h5py as h5
import numpy as np
from datetime import datetime
import numbers
import time
import pandas as pd
from components import DataTransformer, Converter, Logger

class HDF5reader_writer:
    """
    HDF5读写器，支持'r','w''a'模式
    支持读入 Observations 组中指定变量的数据和属性以及全局属性
    支持summary_meteorological,读取数据并打印全局属性的部分信息
    请尽可能使用with语句而不是实例化
    """
    def __init__(self, file_path, enable_logging=True):
        """
        初始化 HDF5Reader 实例，使用组合模式整合各功能模块
        
        参数:
            file_path(str): HDF5 文件的路径
            enable_logging(bool): 默认True,是否使用日志 
        """
        self.__file_path = file_path
        self.__dataset = None
        
        # 组合功能组件
        self.pdtransform = DataTransformer(self)         # 数据转换（需传入主类实例）
        self.converter = Converter()                   # 单位转换
        self._logger = Logger() if enable_logging else None  # 可选日志
    
    # 辅助方法：简化日志调用
    def _log_operation_if_enabled(self, **kwargs):
        """仅在日志启用时记录操作"""
        if hasattr(self, '_logger') and self._logger is not None:   # hasattr(self, '_logger'):检查对象 self 是否拥有名为 '_logger' 的属性
            self._logger.log_operation(**kwargs)
    
    def __openhdf5(self, mode='r'):
        """
        打开 HDF5 文件
        参数：
            mode(str): 读入模式
        返回:
            数据集对象
        """
        operation="Open HDF5 file"
        max_retries = 5  # 最大重试次数
        retry_delay = 1  # 重试间隔(秒)
        for attempt in range(max_retries):
            try:
                self._log_operation_if_enabled(
                    operation=operation,
                    status="STARTED",
                    message=f"File: {self.__file_path}"
                )
                self.__dataset = h5.File(self.__file_path, mode)
                self._log_operation_if_enabled(
                    operation=operation,
                    status="SUCCESS"
                )
            except (FileNotFoundError, PermissionError, RuntimeError) as e:
                e_dict={
                    FileNotFoundError: f"File not found: {self.__file_path}",
                    PermissionError: f"Permission error: {self.__file_path}",
                    RuntimeError: f"Runtime error: {self.__file_path}",
                    }
                message=e_dict.get(e)
                self._log_operation_if_enabled(
                    operation=operation + f" (Attempt {attempt + 1}/{max_retries})",
                    status="RETRY",
                    message=message,
                    exception=e
                )
                if attempt == max_retries - 1:
                    self._log_operation_if_enabled(
                        operation="Open file",
                        status="FAILED",
                        message=f"Max retries exceeded after {max_retries} attempts",
                        exception=e
                    )
                    raise FileNotFoundError(f"File not found: {self.__file_path}") from e
            except Exception as e:
                self._log_operation_if_enabled(
                    operation=f"Open file (Attempt {attempt + 1}/{max_retries})",
                    status="FAILED",
                    message=f"Unexpected {type(e).__name__}: {str(e)}",
                    exception=e
                )
                if attempt == max_retries - 1:
                    self._log_operation_if_enabled(
                        operation="Open file",
                        status="FAILED",
                        message=f"Max retries exceeded after {max_retries} attempts",
                        exception=e
                    )
                    raise
            time.sleep(retry_delay)  # 重试前等待..
    
    def __enter__(self):
        """with语句内确保文件可以打开"""
        try:
            self.__opennc('r')  # 打开文件
            return self         # 返回实例自身，供 with 块使用
        except Exception as e:
            raise              # 重新抛出异常
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        with语句内确保即使发生异常，文件可以关闭
        请尽可能使用with语句而不是实例化，这样可以规避遗忘finally close的风险
        """
        self.close()  # 无论如何，先确保关闭文件
    
    def close(self):
        """关闭 HDF5 文件,释放资源"""
        self._log_operation_if_enabled(
            operation="close",
            status="STARTED",
            message=f"File: {self.__file_path}"
        )
        if self.__dataset is not None:
            self.__dataset.close()
            self.__dataset = None
        self._log_operation_if_enabled(
            operation="close",
            status="SUCCESS"
        )
    
    def get_dataset(self, mode='r', group_path=None):
        """
        获取 HDF5 数据集或指定组
        如果文件尚未打开,则自动调用 open() 打开文件。
        
        参数：
            mode(str):读入模式,一般默认'r'
            group_path(str): 可选，如"Observations/Group1"
            
        返回:
            HDF5 数据集对象
            
        异常:
            KeyError: 当指定路径不存在时
        """
        operation = f"Get dataset{'/' + group_path if group_path else ''}"
        self._log_operation_if_enabled(
            operation="get dataset",
            status="STARTED",
            message=f"Group path: {group_path}"
        )
        try:
            if self.__dataset is None:
                self.__openhdf5(mode)
            if "Observations" not in self.__dataset:
                raise KeyError("Group 'Observations' not found in the file.")
            
            target = self.__dataset
            if group_path:
                for part in filter(None, group_path.split("/")):  # 过滤空路径段
                    if part not in target:
                        raise KeyError(f"Path component '{part}' not found")
                    target = target[part]
                    
            shape_info = ""       
            if hasattr(target, "shape"):  # 如果是数据集而非组
                shape_info = f", shape: {target.shape}"
            self._log_operation_if_enabled(
                operation=operation,
                status="SUCCESS",
                message=f"Type: {type(target).__name__}{shape_info}"
            )
            
            return target
        except Exception as e:
            self._log_operation_if_enabled(
                operation=operation,
                status="FAILED",
                message="Error: {type(e).__name__}: {str(e)}",
                exception=e
            )
            raise
    
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
        operation = f"Get variable data: {variable_name}"
        self._log_operation_if_enabled(
            operation=operation,
            status="STARTED",
            message=f"File: {self.__file_path}"
        )
        group_path = "Observations/" + variable_name
        try:
            self.get_dataset(group_path=group_path)
        except Exception as e:
            self._log_operation_if_enabled(
                operation=operation,
                status="FAILED",
                message=str(e),
                exception=e  # 传递异常对象
            )
            raise  # 重新抛出异常
        else:
            self._log_operation_if_enabled(
                operation=operation,
                status="SUCCESS"
            )
    
    def get_global_attributes(self):
        """
        读取 HDF5 文件的全局属性，并以字典形式返回。
        
        返回:
            dict: 包含全局属性的字典。
        """  
        operation = "Get global attributes"
        self._log_operation_if_enabled(
            operation=operation,
            status="STARTED",
            message=f"File: {self.__file_path}"
        )
        try:
            h5_file = self.get_dataset()
            attributes = dict(h5_file.attrs)
            self._log_operation_if_enabled(
                operation=operation,
                status="SUCCESS",
            )
            return attributes
        except Exception as e:
            self._log_operation_if_enabled(
                operation=operation,
                status="FAILED",
                message=f"{type(e).__name__}: {str(e)}"
            )
            raise
    
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
        operation = f"Get local attributes: {dataset_name}"
        self._log_operation_if_enabled(
            operation=operation,
            status="STARTED",
            message=f"File: {self.__file_path}"
        )
        try:
            h5_file=self.get_dataset()
            obs_group = h5_file["Observations"]
            if dataset_name in obs_group:
                attributes = dict(obs_group[dataset_name].attrs)
                self._log_operation_if_enabled(
                    operation=operation,
                    status="SUCCESS",
                    message=f"attributes: {attributes.keys}"
                )
                return attributes
            else:
                raise ValueError(f"Dataset '{dataset_name}' not found in the Observations group.")
        except Exception as e:
            self._log_operation_if_enabled(
                operation=operation,
                status="FAILED",
                message=f"{type(e).__name__}: {str(e)}"
            )
            raise
    
    def summary_meteorological(self):
        """
        读取 HDF5 文件中的气象数据，并打印全局属性的部分信息。
    
        """
        operation="summary"
        self._log_operation_if_enabled(
            operation=operation,
            status="STARTED",
            message=f"File: {self.__file_path}"
        )
        try:
            print("全局属性：")
            ds = self.get_dataset()
            for key, value in ds.attrs.items():
                print(f"  {key}: {value}")
            # 进入 'Observations' 组
            obs_group = ds['Observations']
            print("\nObservations 组中的数据集：", list(obs_group.keys()))
            self._log_operation_if_enabled(
                operation=operation,
                status="SUCCESS",
            )
        except Exception as e:
            self._log_operation_if_enabled(
                operation=operation,
                status="FAILED",
                message=f"{type(e).__name__}: {str(e)}"
            )
            raise
    
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
        # 记录操作开始（包含基础参数）
        self._log_operation_if_enabled(
            operation="Write meteorological data",
            status="STARTED",
            message=f"Dimensions: time={time_points}, lat={lat_points}, lon={lon_points} | Variables: {list(dic_data.keys()) if dic_data else 'None'}"
        )

        if dic_data is None:
            dic_data = {}
            self._log_operation_if_enabled(
                operation="Data validation",
                status="NOTE",
                message="No input data dictionary, will create empty file"
            )
            
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
                except (ValueError, TypeError) as e:
                    raise
                except Exception as e:
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
        if self._logger:
            self._logger.log_operation(
                operation="write meteo hdf5",
                status="STARTED",
                message=f"File: {self.__file_path}"
            )
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
                except (ValueError, TypeError) as e:
                    raise
                except Exception as e:
                    raise 

hdf5_test=HDF5reader_writer("D://test//hdf5_test.h5")
time_points = 2
lat_points = 6
lon_points = 6
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