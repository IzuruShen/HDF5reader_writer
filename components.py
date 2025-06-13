# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 17:01:56 2025

@author: mirag
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.constants import g as g0  # 标准重力加速度
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Union, Dict, Callable  # 确保所有需要的类型提示都已导入

# ---------------------- 数据转换组件 ----------------------
class DataTransformer:
    """将 HDF5 文件转化成 pandas 格式的数据"""
    def __init__(self, hdf5_reader):
        self.reader = hdf5_reader  # 组合主类实例

    def to_dataframe(self, include_attrs=False):
        """
        将整个HDF5文件转为DataFrame,使用三维坐标索引 (time, lat, lon)
        
        参数:
            include_attrs(bool):默认为False,判断是否将 HDF5 文件中指变量的属性提取出来存入字典
        """
        data_dict = {}
        attrs_dict = {} if include_attrs else None
        
        obs_group = self.reader.get_dataset("Observations")
        coord_group = obs_group["Coordinates"]
        lats = coord_group["Latitude"][:]
        lons = coord_group["Longitude"][:]
        time = coord_group["Time"][:]
        index = pd.MultiIndex.from_product(
            [time, lats, lons],
            names=["time", "lat", "lon"]
        )
        
        for var_name in obs_group:
            if var_name == "Coordinates":
                continue
            data = obs_group[var_name][:]
            if data.shape != (len(time), len(lats), len(lons)):
                raise ValueError(f"变量 {var_name} 的形状 {data.shape} 不符合 (time, lat, lon) 要求")
            data_dict[var_name] = data.flatten()
            if include_attrs:
                attrs_dict[var_name] = dict(obs_group[var_name].attrs)
        df = pd.DataFrame(data_dict, index=index)
        return (df, attrs_dict) if include_attrs else df

    def variable_to_series(self, var_name):
        """将单个变量转为Series"""
        data = self.reader.get_variable_data(var_name)
        if data.shape != (len(time), len(lats), len(lons)):
            raise ValueError(f"变量 {var_name} 的形状 {data.shape} 不符合 (time, lat, lon) 要求")
        obs_group = self.reader.get_dataset("Observations")
        coord_group = obs_group["Coordinates"]
        lats = coord_group["Latitude"][:]
        lons = coord_group["Longitude"][:]
        time = coord_group["Time"][:]
        index = pd.MultiIndex.from_product(
            [time, lats, lons],
            names=["time", "lat", "lon"]
        )
        return pd.Series(data.flatten(), index=index)
    
# ---------------------- 单位转换组件 ----------------------
class Converter:
    """简单静态数据转换"""
    @staticmethod
    def convert_temperature(values, from_unit="K", to_unit="°C"):
        """温度单位转换"""
        converters = {
            ("°C", "K"): lambda x: x + 273.15,
            ("K", "°C"): lambda x: x - 273.15,
            ("°C", "°F"): lambda x: x * 9/5 + 32,
            ("°F", "°C"): lambda x: (x - 32) * 5/9,
            ("°F", "K"): lambda x: (x - 32) * 5/9 + 273.15,
            ("K", "°F"): lambda x: (x - 273.15) * 9/5 + 32
        }
        key = (from_unit, to_unit)
        if key in converters:
            return converters[key](values)
        raise ValueError(f"Unsupported conversion: {from_unit} → {to_unit}")

    @staticmethod
    def convert_pressure(values, from_unit, to_unit):
        """压强单位转换"""
        hPa=('hPa', 'mb')
        if from_unit in hPa and to_unit == "Pa":
            return values * 100
        elif from_unit == "Pa" and to_unit in hPa:
            return values / 100
        elif from_unit in hPa and to_unit in hPa:
            return values
        raise ValueError(f"Unsupported conversion: {from_unit} → {to_unit}")
    
    @staticmethod
    def convert_geopotential(values, from_unit, to_unit):
        """位势（高度）转换"""
        if from_unit == "m²/s²" and to_unit == "gpm":
            return values * 100
        elif from_unit == "gpm" and to_unit == "m²/s²":
            return values / 100
        converters = {
            ("m²/s²", "gpm"): lambda x: x / g0,
            ("gpm", "m²/s²"): lambda x: x * g0,
            ("m²/s²", "dagpm"): lambda x: x / g0 / 10,
            ("dagpm", "m²/s²"): lambda x: x * g0 * 10
        }
        key = (from_unit, to_unit)
        if key in converters:
            return converters[key](values)
        raise ValueError(f"Unsupported conversion: {from_unit} → {to_unit}")
    
    @staticmethod
    def convert_datetime(time_array):
        """将时间戳转为datetime对象"""
        return pd.to_datetime(time_array)

# ---------------------- 日志组件 ----------------------
class Logger:
    def __init__(self, log_file="hdf5_processing.log"):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)  # 全局最低级别（所有Handler会二次过滤）

        #文件处理器（记录DEBUG及以上级别）
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=1e6,  # 1MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # 文件单独设置级别
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        #控制台处理器（仅记录ERROR及以上级别）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)  # 控制台单独设置级别
        console_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'  # 简洁的终端格式
        )
        console_handler.setFormatter(console_formatter)

        # 避免重复添加处理器（防止多次实例化时重复日志）
        if not self._logger.handlers:
            self._logger.addHandler(file_handler)
            self._logger.addHandler(console_handler)

    def log_operation(self, operation, status="SUCCESS", message="", 
                      exception=None):
        """
        记录操作日志
        
        参数:
            operation(str): 表示当前执行的操作名称
            status(str): 默认"SUCCESS",表示操作状态,可选"STARTED", "SUCCESS", "WARNING"和 "FAILED"
            message(str): 默认空,用于补充详细信息
            exception: 默认None,可选的异常对象（用于提取类型和详情）
        """
        log_msg = f"{operation} - {status}"
        
        # 如果有额外消息，添加到日志行
        if message:
            log_msg += f": {message}"
            
        # 如果传入了异常对象，追加异常类型和详情
        if exception is not None:
            exc_type = type(exception).__name__
            exc_msg = str(exception)
            log_msg += f" | Exception: {exc_type}({exc_msg})"
        
        log_level = {
            "STARTED": logging.DEBUG,
            "SUCCESS": logging.INFO,
            "WARNING": logging.WARNING,
            "FAILED": logging.ERROR
        }.get(status, logging.INFO)
        
        self._logger.log(log_level, log_msg)
        
# ---------------------- 数据处理组件 ----------------------
class DataPreprocessor:
    """数据清洗组件（依赖DataTransformer和Logger）"""
    def __init__(self, transformer: DataTransformer, 
                 logger: Optional[Logger] = None):
        self.reader = hdf5_reader
        self.transformer = transformer
        self.logger = logger

    def _log_operation(self, operation: str, status: str, message: str = ""):
        if self.logger:
            self.logger.log_operation(operation, status, message)

    def time_cleaner(self, time_name: str, drop_nat: bool = False, 
                     raise_errors: bool = False) -> pd.Series:
        """
        清洗时间戳数据，支持自动格式推断和异常处理
        
        参数:
            time_name: 时间变量名
            drop_nat: 是否自动删除无效时间戳，默认False
            raise_errors: 发现无效时间戳时是否抛出异常，默认False
        
        返回:
            清洗后的pd.Series时间序列
            
        异常:
            ValueError: 当raise_errors=True且存在无效时间戳时
        """
        operation="Time cleaning"
        self._log_operation(
            operation=operation,
            status="STARTED", 
            message=f"Variable: {time_name}, drop={drop_nat}"
        )
        try:
            # infer_datetime_format=True 可以尝试自动推断格式，提高解析速度
            # errors='coerce' 会将无法解析的日期变成 NaT (Not a Time)
            time_series = self.transformer.to_series(
                self.reader.get_variable_data(time_name),
                time_name
            )
            time_converted = pd.to_datetime(
                time_series,
                infer_datetime_format=True,
                errors='coerce' if not raise_errors else 'raise'
            )
            
            nat_count = time_converted.isna().sum()
            if nat_count == 0:
                self._log_operation(
                    operation=operation,
                    status="SUCCESS", 
                    message=f"Clean timestamps: {len(cleaned)}"
                )
                return time_converted
            warning_msg = f"Found {nat_count} invalid timestamps (NaT) in '{time_name}'"
            if raise_errors:
                self._log_operation(
                    operation=operation,
                    status="FAILED", 
                    message=f"{warning_msg} - Raise"
                )
                raise ValueError(warning_msg)
            if drop_nat:
                cleaned = time_converted.dropna()
                action_message = f"{warning_msg} - Dropped"
            else:
                cleaned = time_converted
                action_message = f"{warning_msg} - Kept"
            self._log_operation(
                operation=operation,
                status="WARNING", 
                message=action_message
            )
            return cleaned
        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED", 
                message=str(e)
            )
            raise
            
    def wind_speed_cleaner(self, wind_speed_name: str, wind_speed_threshold = None, 
                           drop_nat: bool = False, fill_nat: bool = False, 
                           fill_method: str = 'linear', fill_limit: int = 2,
                           raise_errors: bool = False) -> pd.Series:
        """
        清洗风速数据，支持异常处理，允许设置最大阈值
        
        参数:
            wind_speed_name: 风速变量名
            wind_speed_threshold: 风速最大值阈值，默认None，超过即认为是无效值
            drop_nat: 是否自动删除无效风速数据，默认False，与fill_nat互斥
            fill_nat: 是否自动用前后均值插值填充无效风速数据，默认False，与drop_nat互斥
            fill_method: 填充方法（'linear', 'nearest', 'spline'等）,默认'linear'
            fill_limit: 最大连续填充数量，默认2
            raise_errors: 发现无效风速数据时是否抛出异常，默认False
        
        返回:
            清洗后的pd.Series时间序列
            
        异常:
            ValueError: 当raise_errors=True且存在无效风速数据时
        """        
        operation="Wind speed cleaning"
        self._log_operation(
            operation=operation,
            status="STARTED", 
            message=f"Variable: {wind_speed_name}, "
                    f"threshold={wind_speed_threshold}, drop={drop_nat}, fill={fill_nat}"
        )
        if drop_nat and fill_nat:
            error_msg = "Cannot set both drop_nat and fill_nat to True"
            self._log_operation(
                operation=operation,
                status="FAILED", 
                message=error_msg
            )
            raise  ValueError(error_msg)
        try:
            wind_speed_series = self.transformer.to_series(
                self.reader.get_variable_data(wind_speed_name),
                wind_speed_name
            )
            wind_speed_converted = pd.to_numeric(
                wind_speed_series,
                errors='coerce' if not raise_errors else 'raise'
            )
            
            invalid_mask = (wind_speed_converted < 0)
            if wind_speed_threshold is not None:
                invalid_mask |= (wind_speed_converted > wind_speed_threshold)
            wind_speed_converted[invalid_mask] = np.nan
            
            nat_count = invalid_mask.sum()
            if nat_count == 0:
                self._log_operation(
                    operation=operation,
                    status="SUCCESS", 
                    message=f"Clean timestamps: {len(wind_speed_converted)}"
                )
                return wind_speed_converted
            warning_msg = f"Found {nat_count} invalid wind speed (NaT) in '{wind_speed_name}'"
            if raise_errors:
                self._log_operation(
                    operation=operation,
                    status="FAILED", 
                    message=f"{warning_msg} - Raise"
                )
                raise ValueError(warning_msg)
            if drop_nat:
                cleaned = wind_speed_converted.dropna()
                action_message = f"{warning_msg} - Droped"
            elif fill_nat:
                cleaned = wind_speed_converted.interpolate(method=fill_method, limit=fill_limit)
                action_message = f"{warning_msg} - Filled"
            else:
                cleaned = wind_speed_converted
                action_message = f"{warning_msg} - Kept"
            self._log_operation(
                operation=operation,
                status="WARNING", 
                message=action_message
            )
            return cleaned
        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED", 
                message=str(e)
            )
            raise
            
    def temperature_cleaner(self, temperature_name: str, 
                            temperature_min_threshold = None, temperature_max_threshold = None,
                            drop_nat: bool = False, fill_nat: bool = False, 
                            fill_method: str = 'linear', fill_limit: int = 2,
                            raise_errors: bool = False) -> pd.Series:
        """
        清洗温度数据，支持异常处理，允许设置阈值
        
        参数:
            temperature_name: 风速变量名
            temperature_min_threshold: 风速最小值阈值，默认None，小于即认为是无效值
            temperature_max_threshold: 风速最大值阈值，默认None，超过即认为是无效值
            drop_nat: 是否自动删除无效风速数据，默认False，与fill_nat互斥
            fill_nat: 是否自动用前后均值插值填充无效风速数据，默认False，与drop_nat互斥
            fill_method: 填充方法（'linear', 'nearest', 'spline'等）,默认'linear'
            fill_limit: 最大连续填充数量，默认2
            raise_errors: 发现无效风速数据时是否抛出异常，默认False
        
        返回:
            清洗后的pd.Series时间序列
            
        异常:
            ValueError: 当raise_errors=True且存在无效风速数据时
        """        
        operation="Temperature cleaning"
        self._log_operation(
            operation=operation,
            status="STARTED", 
            message=f"Variable: {temperature_name}, "
                    f"threshold={temperature_min_threshold}, {temperature_max_threshold}, drop={drop_nat}, fill={fill_nat}"
        )
        if drop_nat and fill_nat:
            error_msg = "Cannot set both drop_nat and fill_nat to True"
            self._log_operation(
                operation=operation,
                status="FAILED", 
                message=error_msg
            )
            raise  ValueError(error_msg)
        try:
            temperature_series = self.transformer.to_series(
                self.reader.get_variable_data(temperature_name),
                temperature_name
            )
            temperature_converted = pd.to_numeric(
                temperature_series,
                errors='coerce' if not raise_errors else 'raise'
            )
            
            invalid_mask = False
            if temperature_min_threshold is not None:
                invalid_mask = (temperature_converted < temperature_min_threshold)
            if temperature_max_threshold is not None:
                if invalid_mask is False:
                    invalid_mask = (temperature_converted > temperature_max_threshold)
                else:
                    invalid_mask |= (temperature_converted > temperature_max_threshold)
            
            # 应用无效掩码，替换为 np.nan
            temperature_converted[invalid_mask] = np.nan
            
            nat_count = invalid_mask.sum()
            if nat_count == 0:
                self._log_operation(
                    operation=operation,
                    status="SUCCESS", 
                    message=f"Clean timestamps: {len(temperature_converted)}"
                )
                return temperature_converted
            warning_msg = f"Found {nat_count} invalid temperature (NaT) in '{temperature_name}'"
            if raise_errors:
                self._log_operation(
                    operation=operation,
                    status="FAILED", 
                    message=f"{warning_msg} - Raise"
                )
                raise ValueError(warning_msg)
            if drop_nat:
                cleaned = temperature_converted.dropna()
                action_message = f"{warning_msg} - Droped"
            elif fill_nat:
                cleaned = temperature_converted.interpolate(method=fill_method, limit=fill_limit)
                action_message = f"{warning_msg} - Filled"
            else:
                cleaned = temperature_converted
                action_message = f"{warning_msg} - Kept"
            self._log_operation(
                operation=operation,
                status="WARNING", 
                message=action_message
            )
            return cleaned
        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED", 
                message=str(e)
            )
            raise