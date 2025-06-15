# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 20:45:11 2025

@author: mirag
"""
import h5py as h5
import numpy as np
from datetime import datetime
import time
from typing import Optional, Dict, Any  # 确保所有需要的类型提示都已导入
from components import DataTransformer, Converter, Logger, DataPreprocessor, DataAnalyzer, TimeResampler, DataFilter
import os

def safe_remove_file(filepath, max_retries=5, retry_delay=1):
    """安全删除文件，带有重试机制"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)
    return False

class HDF5reader_writer:
    """
    HDF5读写器，支持'r','w''a'模式
    支持读入 Observations 组中指定变量的数据和属性以及全局属性
    支持summary_meteorological,读取数据并打印全局属性的部分信息
    请尽可能使用with语句而不是实例化
    """
    def __init__(self, file_path: str, mode: str ='r', enable_logging: bool = True):
        """
        初始化 HDF5Reader 实例，使用组合模式整合各功能模块
        
        参数:
            file_path: HDF5 文件的路径
            enable_logging: 默认True,是否使用日志 
        """
        self.__file_path = file_path
        self.__dataset = None
        self.__mode = mode
        
        # 组合功能组件
        self.converter = Converter()                   # 单位转换
        self._logger = Logger() if enable_logging else None  # 可选日志
        self.pdtransform = DataTransformer(self)         # 数据转换（需传入主类实例）
        self.datacleaner = DataPreprocessor(self)
        self.dataanalyzer = DataAnalyzer(self)
        self.timeresampler = TimeResampler(self)
        self.datafilier = DataFilter(self)

    def __enter__(self):
        """with语句内确保文件可以打开"""
        try:
            self.__openhdf5()  # 打开文件
            return self         # 返回实例自身，供 with 块使用
        except Exception as e:
            raise RuntimeError(f"Failed to open HDF5 file: {e}")             # 重新抛出异常
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        with语句内确保即使发生异常，文件可以关闭
        请尽可能使用with语句而不是实例化，这样可以规避遗忘finally close的风险
        """
        try:
            self.close()  # 无论如何，先确保关闭文件
        except Exception as e:
            print(f"Error during file closing: {e}")
            raise
    
    # 辅助方法：简化日志调用
    def _log_operation_if_enabled(self, **kwargs):
        """仅在日志启用时记录操作"""
        if hasattr(self, '_logger') and self._logger is not None:   # hasattr(self, '_logger'):检查对象 self 是否拥有名为 '_logger' 的属性
            self._logger.log_operation(**kwargs)
    
    def __openhdf5(self):
        """
        打开 HDF5 文件
        
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
                
                # 如果是写入模式，先尝试删除现有文件
                if self.__mode == 'w':
                    safe_remove_file(self.__file_path)
                self.__dataset = h5.File(self.__file_path, self.__mode)
                
                self._log_operation_if_enabled(
                    operation=operation,
                    status="SUCCESS"
                )
                return
            except (FileNotFoundError, PermissionError, RuntimeError) as e:
                e_dict={
                    FileNotFoundError: f"File not found: {self.__file_path}",
                    PermissionError: f"Permission error: {self.__file_path}",
                    RuntimeError: f"Runtime error: {self.__file_path}",
                    }
                message = e_dict.get(type(e), f"Unexpected error: {str(e)}")
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
    
    def close(self):
        """关闭 HDF5 文件,释放资源"""
        operation = 'close'
        self._log_operation_if_enabled(
            operation=operation,
            status="STARTED",
            message=f"File: {self.__file_path}"
        )
        if self.__dataset is not None:
            try:
                self.__dataset.close()
                self._log_operation_if_enabled(
                    operation=operation,
                    status="SUCCESS"
                )
            except Exception as e:
                self._log_operation_if_enabled(
                    operation=operation,
                    status="FAILED",
                    message=f"Warning: Error closing HDF5 file: {e}"
                )
                raise
            finally:
                self.__dataset = None
    
    def get_dataset(self, group_path: Optional[str] = None):
        """
        获取 HDF5 数据集或指定组
        如果文件尚未打开,则自动调用 open() 打开文件。
        
        参数：
            group_path: 可选，如"Observations/Group1"，默认读全
            
        返回:
            HDF5 数据集对象
            
        异常:
            KeyError: 当指定路径不存在时
        """
        operation = f"Get dataset{'/' + group_path if group_path else ''}"
        self._log_operation_if_enabled(
            operation=operation,
            status="STARTED",
            message=f"Group path: {group_path}"
        )
        try:
            if self.__dataset is None:
                self.__openhdf5()
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
    
    def get_variable_data(self, variable_name: str) -> np.ndarray:
        """
        读取 Observations 组中指定变量的数据值

        参数:
            variable_name: 目标变量名称

        返回:
            该变量的数据值

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
            dataset = self.get_dataset(group_path=group_path)
            data = dataset[:]  # 获取实际数据
            self._log_operation_if_enabled(
                operation=operation,
                status="SUCCESS"
            )
            return data
        except Exception as e:
            self._log_operation_if_enabled(
                operation=operation,
                status="FAILED",
                message=str(e),
                exception=e  # 传递异常对象
            )
            raise  # 重新抛出异常            
    
    def get_global_attributes(self) -> Dict[str, Any]:
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
    
    def get_local_attributes(self, dataset_name: str) -> dict:
        """
        读取 Observations 组中指定数据集的局部属性，并以字典形式返回。
        
        参数:
            dataset_name: 要读取属性的数据集名称。
        
        返回:
            包含该数据集局部属性的字典。
        
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
        """读取 HDF5 文件中的气象数据，并打印全局属性的部分信息。"""
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
    
    # 气象数据写入方法
    def write_meteo_hdf5(self, time_points: int, lat_points: int, lon_points: int, 
                         lat_min=-90, lat_max=90, lon_min=-180, lon_max=180,
                         time_values=None,
                         dic_data: Optional[Dict[str, Dict[str, Any]]] =None):
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
        operation="Write meteorological data"
        # 记录操作开始（包含基础参数）
        self._log_operation_if_enabled(
            operation=operation,
            status="STARTED",
            message=f"File: {self.__file_path}, Dimensions: "
                    f"time={time_points},lat={lat_points}, lon={lon_points} | "
                    f"Variables: {list(dic_data.keys()) if dic_data else 'None'}"
        )
        if self.__mode != 'w':
            error_msg = "File must be opened in 'w' mode for writing"
            self._log_operation_if_enabled(
                operation=operation,
                status="FAILED",
                message=error_msg
            )
            raise PermissionError(error_msg)
        try:
            if dic_data is None:
                dic_data = {}
                self._log_operation_if_enabled(
                    operation="Parameter check",
                    status="INFO",
                    message="dic_data is None, initialized as empty dict"
                )
            
            # 检查文件是否已打开
            if self.__dataset is None:
                raise RuntimeError("HDF5 file is not open. Use 'with' statement to open the file.")
            # 清空现有数据（如果存在）
            for key in list(self.__dataset.keys()):
                del self.__dataset[key]
            
            # 生成时间数据
            if time_points is not None:
                if time_values is None:
                    time_values = np.arange(time_points)  # 默认生成0,1,2,...的时间序列
                    self._log_operation_if_enabled(
                        operation="Time values generation",
                        status="NOTE",
                        message=f"Auto-generated time values (0 to {time_points-1})"
                    )
                elif len(time_values) != time_points:
                    error_msg = "time_values length must match time_points"
                    self._log_operation_if_enabled(
                        operation="Parameter validation",
                        status="FAILED",
                        message=error_msg
                    )
                    raise ValueError(error_msg)
                    
            # 生成经纬度数据
            if not (lat_points > 0 and lon_points > 0 and time_points > 0):
                error_msg = "lat_points, lon_points and time_points must be positive"
                self._log_operation_if_enabled(
                    operation="Parameter validation",
                    status="FAILED",
                    message=error_msg
                )
                raise ValueError(error_msg)
            if not (isinstance(lat_points, int) and isinstance(lon_points, int) and isinstance(time_points, int)):
                error_msg = "lat_points, lon_points and time_points must be integers"
                self._log_operation_if_enabled(
                    operation="Parameter validation",
                    status="FAILED",
                    message=error_msg
                )
                raise TypeError(error_msg)
            if lat_min >= lat_max or lon_min >= lon_max:
                error_msg = "lat_min must be < lat_max and lon_min must be < lon_max"
                self._log_operation_if_enabled(
                    operation="Parameter validation",
                    status="FAILED",
                    message=error_msg
                )
                raise ValueError(error_msg)
            
            try:
                latitudes = np.linspace(lat_min, lat_max, lat_points)
                longitudes = np.linspace(lon_min, lon_max, lon_points)
                self._log_operation_if_enabled(
                    operation="Coordinate generation",
                    status="SUCCESS",
                    message=f"Generated {lat_points}x{lon_points} grid"
                )
            except TypeError as e:
                error_msg = f"Input must be numeric: {str(e)}"
                self._log_operation_if_enabled(
                    operation="Coordinate generation",
                    status="FAILED",
                    message=error_msg
                )
                raise TypeError(error_msg)
            
            # 写入全局属性
            self.__dataset.attrs['CreationDate'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.__dataset.attrs['DataSource'] = 'Simulated meteorological data'
            self.__dataset.attrs['Description'] = f'meteorological data, include {dic_data.keys()}'
            
            self._log_operation_if_enabled(
                operation="Global attributes",
                status="NOTE"
            )
            
            # 创建组结构
            obs_group = self.__dataset.create_group('Observations')
            coord_group = obs_group.create_group('Coordinates')
            
            # 生成坐标数据
            latitudes = np.linspace(lat_min, lat_max, lat_points)
            longitudes = np.linspace(lon_min, lon_max, lon_points)
            
            # 写入坐标
            coord_group.create_dataset('Time', data=time_values if time_values is not None else np.arange(time_points))
            coord_group.create_dataset('Latitude', data=latitudes)
            coord_group.create_dataset('Longitude', data=longitudes)
            
            self._log_operation_if_enabled(
                operation="Group structure",
                status="NOTE",
                message="Created Observations/Coordinates groups"
            )
                
            # 遍历 dic_data，写入所有变量
            vars_written = []
            for var_name, var_info in dic_data.items():
                if not isinstance(var_info, dict):
                    error_msg = f"Variable '{var_name}' info must be a dictionary"
                    self._log_operation_if_enabled(
                        operation=f"Write variable {var_name}",
                        status="FAILED",
                        message=error_msg
                    )
                    raise ValueError(error_msg)
                try:
                    data = var_info["data"]
                    units = var_info.get("units", "unknown")
                    description = var_info.get("description", "no description")
                    
                    expected_shape_3d = (time_points, lat_points, lon_points)
    
                    if data.shape != expected_shape_3d:
                        error_msg = (
                            f"Data shape mismatch for {var_name}: "
                            f"expected {expected_shape_3d}, got {data.shape}"
                        )
                        self._log_operation_if_enabled(
                            operation=f"Write variable {var_name}",
                            status="FAILED",
                            message=error_msg
                        )
                        raise ValueError(error_msg)
    
                    dset = obs_group.create_dataset(var_name, data=data)
                    dset.attrs["units"] = units
                    dset.attrs["description"] = description
                    
                    vars_written.append(var_name)               
                    self._log_operation_if_enabled(
                        operation=f"Write variable {var_name}",
                        status="SUCCESS",
                        message=f"Shape: {data.shape}, Units: {units}"
                    )
                except KeyError as e:
                    self._log_operation_if_enabled(
                        operation=f"Write variable {var_name}",
                        status="FAILED",
                        message=f"Missing required key: {str(e)}"
                    )
                    raise
                except Exception as e:
                    self._log_operation_if_enabled(
                        operation=f"Write variable {var_name}",
                        status="FAILED",
                        message=str(e)
                    )
                    raise
            self._log_operation_if_enabled(
                operation=operation,
                status="SUCCESS",
                message=f"Variables written: {vars_written}"
            )
        except Exception as e:
            self._log_operation_if_enabled(
                operation=operation,
                status="FAILED",
                message=str(e)
            )
            raise
    
    def append_meteo_hdf5(self, time_points: int, lat_points: int, lon_points: int,
                          lat_min=-90, lat_max=90, lon_min=-180, lon_max=180, 
                          time_values=None, 
                          dic_data: Optional[Dict[str, Dict[str, Any]]] =None):
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
        operation = "Append meteorological data"
        self._log_operation_if_enabled(
            operation=operation,
            status="STARTED",
            message=f"File: {self.__file_path}, Dimensions: "
                    f"time={time_points},lat={lat_points}, lon={lon_points} | "
                    f"Variables: {list(dic_data.keys()) if dic_data else 'None'}"
        )
        if self.__mode != 'a':
            error_msg = "File must be opened in 'a' mode for appending"
            self._log_operation_if_enabled(
                operation=operation,
                status="FAILED",
                message=error_msg
            )
            raise PermissionError(error_msg)
        try:
            if dic_data is None:
                dic_data = {} 
                self._log_operation_if_enabled(
                    operation="Parameter check",
                    status="NOTE",
                    message="dic_data is None, initialized as empty dict"
                )
            
            # 检查文件是否已打开
            if self.__dataset is None:
                raise RuntimeError("HDF5 file is not open. Use 'with' statement to open the file.")
            
            if len(time_values) != time_points:
                raise ValueError("time_values length must match time_points")
                    
            if not (lat_points > 0 and lon_points > 0 and time_points > 0):
                raise ValueError("lat_points, lon_points and time_points must be positive")
            if not (isinstance(lat_points, int) and isinstance(lon_points, int) and isinstance(time_points, int)):
                raise TypeError("lat_points, lon_points and time_points must be integers")
            if lat_min >= lat_max or lon_min >= lon_max:
                raise ValueError("lat_min must be < lat_max and lon_min must be < lon_max")
            
            obs_group = self.__dataset.require_group('Observations')
                
            # 遍历 dic_data，写入所有变量
            vars_appended = []
            for var_name, var_info in dic_data.items():
                if not isinstance(var_info, dict):
                    error_msg = f"Variable '{var_name}' info must be a dictionary"
                    self._log_operation_if_enabled(
                        operation=f"Append variable {var_name}",
                        status="FAILED",
                        message=error_msg
                    )
                    raise ValueError(error_msg)
                
                try:  
                    data = var_info["data"]
                    units = var_info.get("units", "unknown")
                    description = var_info.get("description", "no description")
    
                    expected_shape_3d = (time_points, lat_points, lon_points)
    
                    if data.shape != expected_shape_3d:
                        error_msg = f"Data shape mismatch for {var_name}: "\
                                    f"expected ({lat_points}, {lon_points}), got {data.shape}"
                        self._log_operation_if_enabled(
                            operation=f"Append variable {var_name}",
                            status="FAILED",
                            message=error_msg
                        )
                        raise ValueError(error_msg)
                    
                    if var_name in obs_group:
                        del obs_group[var_name]
                        self._log_operation_if_enabled(
                            operation=f"Append variable {var_name}",
                            status="NOTE",
                            message="Existing dataset deleted before append"
                        )
                        
                    dset = obs_group.create_dataset(var_name, data=data)
                    dset.attrs["units"] = units
                    dset.attrs["description"] = description
                    vars_appended.append(var_name)
                    
                    self._log_operation_if_enabled(
                        operation=f"Append variable {var_name}",
                        status="SUCCESS",
                        message=f"Shape: {data.shape}, Units: {units}"
                    )
                except KeyError as e:
                    self._log_operation_if_enabled(
                        operation=f"Append variable {var_name}",
                        status="FAILED",
                        message=f"Missing required key '{e.args[0]}' in variable '{var_name}'"
                    )
                    raise 
                except Exception as e:
                    self._log_operation_if_enabled(
                        operation=f"Append variable {var_name}",
                        status="FAILED",
                        message=str(e)
                    )
                    raise 
            self._log_operation_if_enabled(
                operation=operation,
                status="SUCCESS",
                message=f"Variables appended: {vars_appended}"
            )
        except Exception as e:
            self._log_operation_if_enabled(
                operation=operation,
                status="FAILED",
                message=str(e)
            )
            raise

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

with HDF5reader_writer("D://test//hdf5_test_2.h5", 'w') as h5file:
    h5file.write_meteo_hdf5(time_points=time_points,lat_points=lat_points, lon_points=lon_points, 
                            lat_min=-60, lat_max=-30, lon_min=30, lon_max=60, 
                            time_values=None, dic_data=dic_data)

# 测试DataTransformer
with HDF5reader_writer("D://test//hdf5_test_2.h5") as hdf5_reader:
    transformer = hdf5_reader.pdtransform
    
    # 测试to_dataframe
    df_all, attrs_all = transformer.to_dataframe(include_attrs=True)
    print("All variables DataFrame shape:", df_all.shape)
    print("Attributes keys:", attrs_all.keys())
    
    # 测试selected_to_dataframe
    df_selected = transformer.selected_to_dataframe(['Temperature', 'Humidity'])
    print("\nSelected variables DataFrame shape:", df_selected.shape)
    
    # 测试variable_to_series
    temp_series = transformer.variable_to_series('Temperature')
    print("\nTemperature Series:\n", temp_series.head())

# 测试DataPreprocessor
with HDF5reader_writer("D://test//hdf5_test_2.h5") as hdf5_reader:
    preprocessor = hdf5_reader.datacleaner
    
    # 测试temperature_cleaner
    cleaned_temp = preprocessor.temperature_cleaner(
        'Temperature',
        temperature_min_threshold=-20,
        temperature_max_threshold=40,
        fill_nat=True
    )
    print("Cleaned Temperature:\n", cleaned_temp.head())
    
    # 测试wind_speed_cleaner
    cleaned_wind = preprocessor.wind_speed_cleaner(
        'WindSpeed',
        wind_speed_threshold=10,
        fill_nat=True
    )
    print("\nCleaned Wind Speed:\n", cleaned_wind.head())

# 测试DataAnalyzer
with HDF5reader_writer("D://test//hdf5_test_2.h5") as hdf5_reader:
    analyzer = hdf5_reader.dataanalyzer
    
    # 测试get_variable_slice
    temp_slice = analyzer.get_variable_slice(
        'Temperature',
        time_slice=slice(0,1),
        lat_slice=slice(0,3),
        lon_slice=slice(0,3)
    )
    print("Temperature slice:\n", temp_slice)
    
    # 测试统计方法
    temp_mean = analyzer.get_mean('Temperature')
    temp_max = analyzer.get_max('Temperature')
    print(f"\nTemperature mean: {temp_mean:.2f}, max: {temp_max:.2f}")
    
    # 测试相关性分析
    corr = analyzer.calculate_temporal_correlation('Temperature', 'Humidity')
    print(f"\nTemperature-Humidity correlation: {corr:.2f}")

# 测试TimeResampler
with HDF5reader_writer("D://test//hdf5_test_2.h5") as hdf5_reader:
    resampler = hdf5_reader.timeresampler
    
    # 测试resample_time
    daily_temp = resampler.resample_time('Temperature', 'D')
    print("Daily resampled Temperature:\n", daily_temp)
    
    # 测试resample_multi_vars
    daily_data = resampler.resample_multi_vars(
        ['Temperature', 'Humidity'], 
        'D',
        method={'Temperature': 'mean', 'Humidity': 'max'}
    )
    print("\nDaily resampled data:")
    for var, df in daily_data.items():
        print(f"{var} shape: {df.shape}")

# 测试DataFilter
with HDF5reader_writer("D://test//hdf5_test_2.h5") as hdf5_reader:
    data_filter = hdf5_reader.datafilier
    
    # 加载数据
    df = data_filter.load_data(['Temperature', 'Humidity'])
    
    # 测试filter_by_condition
    hot_days = data_filter.filter_by_condition(
        df, 'Temperature', '>', 30
    )
    print(f"Days with temperature >30°C: {len(hot_days)}")
    
    # 测试filter_by_query
    humid_hot = data_filter.filter_by_query(
        df, 'Temperature > 25 and Humidity > 70'
    )
    print(f"Hot and humid days: {len(humid_hot)}")

# 测试Converter
converter = Converter()

# 温度转换
temp_k = converter.convert_temperature(25, from_unit="°C", to_unit="K")
print(f"25°C in Kelvin: {temp_k:.2f}")

# 气压转换
pressure_pa = converter.convert_pressure(1013.25, from_unit="hPa", to_unit="Pa")
print(f"1013.25 hPa in Pa: {pressure_pa}")