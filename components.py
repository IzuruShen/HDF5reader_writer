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

    def log_operation(self, operation, status="SUCCESS", message="", exception=None):
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
    def __init__(self, transformer: DataTransformer, logger: Optional[Logger] = None):
        self.reader = hdf5_reader
        self.transformer = transformer
        self.logger = logger

    def _log_operation(self, operation: str, status: str, message: str = ""):
        if self.logger:
            self.logger.log_operation(operation, status, message)

    def time_cleaner(self, time_name):
        print("\n--- 清洗时间戳 ---")
        try:
            # infer_datetime_format=True 可以尝试自动推断格式，提高解析速度
            # errors='coerce' 会将无法解析的日期变成 NaT (Not a Time)
            time_data_original = self.transformer.variable_to_series(time_name)
            time_data = pd.to_datetime(time_data_original, infer_datetime_format=True, errors='coerce')
            # 检查是否有无法解析的日期
            if time_data.isnull().any():
                print("警告: Timestamp 列中存在无法解析的日期/时间值 (NaT)。")
                # 可以选择删除这些行或进一步检查
                time_data.dropna(subset=['Timestamp'], inplace=True)