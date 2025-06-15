# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 17:01:56 2025

@author: mirag
"""
import pandas as pd
import numpy as np
from scipy.constants import g as g0  # 标准重力加速度
import logging
from logging.handlers import RotatingFileHandler
from typing import Union, Dict, List, Any, Callable, Sequence, Tuple  # 确保所有需要的类型提示都已导入
import operator

# ---------------------- 单位转换组件 ----------------------
class DataConverter:
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
        
# ---------------------- 数据转换组件 ----------------------
class DataTransformer:
    """将 HDF5 文件转化成 pandas 格式的数据"""
    def __init__(self, hdf5_reader):
        self.reader = hdf5_reader  # 组合主类实例
        self._logger = getattr(hdf5_reader, '_logger', None)  # 复用主类的日志
        
    def _log_operation(self, **kwargs):
        """仅在日志启用时记录操作"""
        if self._logger is not None:
            self._logger.log_operation(**kwargs)

    def to_dataframe(self, include_attrs: bool =False) -> pd.DataFrame:
        """
        将整个HDF5文件转为DataFrame,使用三维坐标索引 (time, lat, lon)
        
        参数:
            include_attrs(bool):默认为False,判断是否将 HDF5 文件中指变量的属性提取出来存入字典
            
        返回:
            当include_attrs为True时返回元组(DataFrame, 属性字典)
            否则返回DataFrame
        """
        operation = "Convert to DataFrame"
        self._log_operation(
            operation=operation, 
            status="STARTED", 
            message=f"Include attributes: {include_attrs}"
            )
        try:
            data_dict = {}
            attrs_dict = {} if include_attrs else None
            
            # 获取数据组
            obs_group = self.reader.get_dataset("Observations")
            coord_group = obs_group["Coordinates"]
            
            # 读取坐标数据
            lats = coord_group["Latitude"][:]
            lons = coord_group["Longitude"][:]
            time = coord_group["Time"][:]
            
            # 创建多级索引
            index = pd.MultiIndex.from_product(
                [time, lats, lons],
                names=["time", "lat", "lon"]
            )
            
            vars_processed = 0
            for var_name in obs_group:
                if var_name == "Coordinates":
                    continue
                data = obs_group[var_name][:]
                if data.shape != (len(time), len(lats), len(lons)):
                    error_msg = f"The shape of variable {var_name} : {data.shape} is not (time, lat, lon) "
                    self._log_operation(
                        operation=operation, 
                        status="FAILED", 
                        message=error_msg
                        )
                    raise ValueError(error_msg)
                data_dict[var_name] = data.flatten()
                if include_attrs:
                    attrs_dict[var_name] = dict(obs_group[var_name].attrs)
                vars_processed += 1
                self._log_operation(
                    operation=f"Processing variable {var_name}", 
                    status="SUCCESS"
                    )
                
            df = pd.DataFrame(data_dict, index=index)
            self._log_operation(
                operation=operation, 
                status="SUCCESS", 
                message=f"Converted {vars_processed} variables to DataFrame"
            )
            return (df, attrs_dict) if include_attrs else df
        except ValueError:
            raise
        except Exception as e:
            self._log_operation(
                operation=operation, 
                status="FAILED", 
                message=str(e), 
                exception=e)
            raise
        
    def selected_to_dataframe(self, variable_names: Sequence[str], 
                              include_attrs: bool = False
                              ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """
        将指定的变量转换为DataFrame，使用三维坐标索引 (time, lat, lon)
        
        参数:
            variable_names: 需要转换的变量名列表或元组
            include_attrs: 是否包含属性信息
            
        返回:
            当include_attrs为True时返回元组(DataFrame, 属性字典)
            否则返回DataFrame
            
        异常:
            ValueError: 当变量不存在或形状不匹配时
            KeyError: 当指定的变量不存在时
        """
        operation = f"Convert selected variables to DataFrame: {variable_names}"
        self._log_operation(
            operation=operation,
            status="STARTED",
            message=f"Include attributes: {include_attrs}"
        )
        
        # 参数校验
        if not isinstance(variable_names, (list, tuple)):
            error_msg = (
                f"variable_names must be sequence of str, "
                f"got {type(variable_names).__name__}"
            )
            self._log_operation(
                operation=operation, 
                status="FAILED", 
                message=error_msg
                )
            raise TypeError(error_msg)
        if not variable_names:
            error_msg = "variable_names cannot be empty"
            self._log_operation(
                operation=operation, 
                status="FAILED", 
                message=error_msg
                )
            raise ValueError(error_msg)
            
        try:    
            data_dict = {}
            attrs_dict = {} if include_attrs else None
            
            obs_group = self.reader.get_dataset("Observations")
            coord_group = obs_group["Coordinates"]
            
            # 检查变量是否存在
            missing_vars = [name for name in variable_names if name not in obs_group]
            if missing_vars:
                error_msg = f"Variables not found: {missing_vars}"
                self._log_operation(
                    operation=operation, 
                    status="FAILED", 
                    message=error_msg
                )
                raise KeyError(error_msg)
            
            lats = coord_group["Latitude"][:]
            lons = coord_group["Longitude"][:]
            time = coord_group["Time"][:]
            
            index = pd.MultiIndex.from_product(
                [time, lats, lons],
                names=["time", "lat", "lon"]
            )
            
            for var_name in variable_names:
                if var_name == "Coordinates":
                    error_msg = "Coordinates cannot be the content of DataFrame"
                    self._log_operation(
                        operation=operation, 
                        status="FAILED", 
                        message=error_msg
                        )
                    raise KeyError(error_msg)
                    
                self._log_operation(
                    operation=f"Processing variable {var_name}",
                    status="STARTED"
                )
                data = obs_group[var_name][:]
                if data.shape != (len(time), len(lats), len(lons)):
                    error_msg = f"Shape mismatch for {var_name}: {data.shape} != ({len(time)}, {len(lats)}, {len(lons)})"
                    self._log_operation(operation, "FAILED", error_msg)
                    raise ValueError(error_msg)
                data_dict[var_name] = data.flatten()
                
                if include_attrs:
                    attrs_dict[var_name] = dict(obs_group[var_name].attrs)   
                self._log_operation(
                    operation=f"Processing variable {var_name}",
                    status="SUCCESS"
                )
            
            df = pd.DataFrame(data_dict, index=index)
            self._log_operation(
                operation=operation,
                status="SUCCESS",
                message=f"Converted {len(variable_names)} variables to DataFrame"
            )
            return (df, attrs_dict) if include_attrs else df 
        except (KeyError, ValueError):
            raise
        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED",
                message=str(e),
                exception=e
            )
            raise        
        
    def variable_to_series(self, var_name: str) -> pd.Series:
        """将单个变量转为Series"""
        operation = f"Convert variable {var_name} to Series"
        self._log_operation(
            operation=operation, 
            status="STARTED",
            message=f"var: {var_name}")
        try: 
            obs_group = self.reader.get_dataset("Observations")
            coord_group = obs_group["Coordinates"]
            
            lats = coord_group["Latitude"][:]
            lons = coord_group["Longitude"][:]
            time = coord_group["Time"][:]
            
            data = obs_group[var_name][:]
            if data.shape != (len(time), len(lats), len(lons)):
                error_msg = f"The shape of variable {var_name} : {data.shape} is not (time, lat, lon) "
                self._log_operation(
                    operation=operation, 
                    status="FAILED", 
                    message=error_msg
                    )
                raise ValueError(error_msg)
            
            index = pd.MultiIndex.from_product(
                [time, lats, lons],
                names=["time", "lat", "lon"]
            )
            series = pd.Series(data.flatten(), index=index)
            self._log_operation(
                operation=operation, 
                status="SUCCESS"
                )
            return series
        except Exception as e:
            self._log_operation(
                operation=operation, 
                status="FAILED", 
                message=str(e), 
                exception=e)
            raise
            
# ---------------------- 数据处理组件 ----------------------
class DataPreprocessor:
    """数据清洗组件（依赖HDF5reader_writer, DataTransformer和Logger）"""
    def __init__(self,  hdf5_reader):
        self.reader =  hdf5_reader
        self.transformer = getattr(hdf5_reader, 'pdtransform')
        self._logger = getattr(hdf5_reader, '_logger', None)

    def _log_operation(self, **kwargs):
        """仅在日志启用时记录操作"""
        if self._logger is not None:
            self._logger.log_operation(**kwargs)

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
                    message=f"Clean timestamps: {len(time_converted)}"
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
    
    def __base_cleaner(self, data_name: str, min_threshold = None, max_threshold = None, 
                       drop_nat: bool = False, fill_nat: bool = False, 
                       fill_method: str = 'linear', fill_limit: int = 2, 
                       raise_errors: bool = False, data_type: str = "") -> pd.Series:
        """
        通用数据清洗模板方法
        
        参数:
            data_name: 数据集变量名
            min_threshold: 最小值阈值（小于此值视为无效）
            max_threshold: 最大值阈值（大于此值视为无效）
            drop_nat: 是否删除无效值（与fill_nat互斥）
            fill_nat: 是否填充无效值（与drop_nat互斥）
            fill_method: 填充方法（'linear', 'nearest', 'spline'等），若要全填0请输入"fill_zero"
            fill_limit: 最大连续填充数量
            raise_errors: 是否对无效值抛出异常
            data_type: 数据类型标识（用于日志记录）
        
        返回:
            清洗后的pd.Series
        """
        operation = f"{data_type.title()} cleaning" if data_type else "Data cleaning"
        self._log_operation(
            operation=operation,
            status="STARTED",
            message=f"Variable: {data_name}, "
                   f"Thresholds: [{min_threshold}, {max_threshold}], "
                   f"Fill: {fill_method if fill_nat else 'None'}"
        )

        # 参数校验
        if drop_nat and fill_nat:
            error_msg = "Cannot enable both drop_nat and fill_nat simultaneously"
            self._log_operation(
                operation=operation,
                status="FAILED", 
                message=error_msg
            )
            raise ValueError(error_msg)

        try:
            # 数据获取与转换（需确保reader已注入）
            series = self.transformer.variable_to_series(data_name)
            
            # 转换为数值类型
            numeric_series = pd.to_numeric(
                series, 
                errors='coerce' if not raise_errors else 'raise'
            )

            # 阈值检测
            invalid_mask = pd.Series(False, index=numeric_series.index)
            if min_threshold is not None:
                invalid_mask |= (numeric_series < min_threshold)
            if max_threshold is not None:
                invalid_mask |= (numeric_series > max_threshold)
            
            cleaned = numeric_series.mask(invalid_mask, np.nan)
            nat_count = invalid_mask.sum()

            # 无无效值情况
            if nat_count == 0:
                self._log_operation(
                    operation=operation, 
                    status="SUCCESS", 
                    message="No invalid values found"
                    )
                return cleaned

            # 无效值处理
            warning_msg = f"Found {nat_count} invalid values ({nat_count/len(cleaned):.1%})"
            if raise_errors:
                self._log_operation(
                    operation=operation, 
                    status="FAILED", 
                    message=f"{warning_msg} - Raising exception"
                    )
                raise ValueError(warning_msg)
            if drop_nat:
                result = cleaned.dropna()
                action = "Dropped"
            elif fill_nat:
                if fill_method=="fill_zero":
                    result = cleaned.fillna(0)
                    action = "Zero-filled"
                else:
                    result = cleaned.interpolate(
                        method=fill_method, 
                        limit=fill_limit,
                        order=3 if fill_method == 'spline' else None
                    )
                    action = f"Filled ({fill_method})"
            else:
                result = cleaned
                action = "Kept"

            self._log_operation(
                operation=operation, 
                status="WARNING",
                message=f"{warning_msg} - {action}"
            )
            return result

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
        return self.__base_cleaner(
            data_name=wind_speed_name,
            min_threshold=0, max_threshold=wind_speed_threshold,
            drop_nat=drop_nat, fill_nat=fill_nat,
            fill_method=fill_method, fill_limit=fill_limit,
            raise_errors=raise_errors,
            data_type="wind_speed"
        )
            
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
        return self.__base_cleaner(
            data_name=temperature_name,
            min_threshold=temperature_min_threshold,
            max_threshold=temperature_max_threshold,
            drop_nat=drop_nat,
            fill_nat=fill_nat,
            fill_method=fill_method,
            fill_limit=fill_limit,
            raise_errors=raise_errors,
            data_type="temperature"
        )
    
    def humidity_cleaner(self, humidity_name: str, humidity_threshold = None, 
                         drop_nat: bool = False, fill_nat: bool = False, 
                         fill_method: str = 'linear', fill_limit: int = 2, 
                         raise_errors: bool = False) -> pd.Series:
        """
        清洗湿度数据，支持异常处理，允许设置最大阈值
        
        参数:
            humidity_name: 湿度变量名
            humidity_threshold: 湿度最大值阈值，默认None，超过即认为是无效值
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
        return self.__base_cleaner(
            data_name=humidity_name,
            min_threshold=0, max_threshold=humidity_threshold,
            drop_nat=drop_nat, fill_nat=fill_nat,
            fill_method=fill_method, fill_limit=fill_limit,
            raise_errors=raise_errors,
            data_type="humidity"
        )
    
    def precipitation_cleaner(self, precipitation_name: str, precipitation_threshold = None, 
                              drop_nat: bool = False, fill_nat: bool = False, 
                              fill_method: str = 'fill_zero', fill_limit: int = 2, 
                              raise_errors: bool = False) -> pd.Series:
        """
        清洗风速数据，支持异常处理，允许设置最大阈值
        
        参数:
            humidity_name: 风速变量名
            humidity_threshold: 风速最大值阈值，默认None，超过即认为是无效值
            drop_nat: 是否自动删除无效风速数据，默认False，与fill_nat互斥
            fill_nat: 是否自动用前后均值插值填充无效风速数据，默认False，与drop_nat互斥
            fill_method: 填充方法（'linear', 'nearest', 'spline'等）,默认'fill_zero'
            fill_limit: 最大连续填充数量，默认2
            raise_errors: 发现无效风速数据时是否抛出异常，默认False
        
        返回:
            清洗后的pd.Series时间序列
            
        异常:
            ValueError: 当raise_errors=True且存在无效风速数据时
        """        
        return self.__base_cleaner(
            data_name=precipitation_name,
            min_threshold=0, max_threshold=precipitation_threshold,
            drop_nat=drop_nat, fill_nat=fill_nat,
            fill_method=fill_method, fill_limit=fill_limit,
            raise_errors=raise_errors,
            data_type="precipitation"
        )

# ---------------------- 数据分析组件 ----------------------
class DataAnalyzer:
    def __init__(self, hdf5_reader):
        """
        初始化数据分析组件
        
        参数:
            hdf5_reader: HDF5reader_writer 实例
        """
        self.reader = hdf5_reader
        self.transformer = DataTransformer(hdf5_reader)
        self._logger = getattr(hdf5_reader, '_logger', None)  # 复用主类的日志
    
    def _log_operation(self, **kwargs):
        """仅在日志启用时记录操作"""
        if self._logger is not None:
            self._logger.log_operation(**kwargs)
    
    def get_variable_slice(self, variable_name: str, 
                           time_slice: slice = None, 
                           lat_slice: slice = None, 
                           lon_slice: slice = None) -> pd.DataFrame:
        """
        读取变量的切片数据
        
        参数:
            variable_name: 变量名称
            time_slice: 时间维度切片，如 slice(0, 10, 2)
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
            
        返回:
            numpy.ndarray: 切片后的数据
            
        异常:
            ValueError: 当切片超出范围或变量不存在时
            KeyError: 当变量不存在时
        """
        operation = f"Get variable slice: {variable_name}"
        self._log_operation(
            operation=operation,
            status="STARTED",
            message=f"Slices - time: {time_slice}, lat: {lat_slice}, lon: {lon_slice}"
        )
        
        try:
            # 获取完整数据集
            dataset = self.reader.get_dataset(group_path=f"Observations/{variable_name}")
            
            # 获取坐标数据
            coord_group = self.reader.get_dataset(group_path="Observations/Coordinates")
            times = coord_group["Time"][:]
            lats = coord_group["Latitude"][:]
            lons = coord_group["Longitude"][:]
            
            
            time_idx = time_slice if time_slice else slice(None)
            lat_idx = lat_slice if lat_slice else slice(None)
            lon_idx = lon_slice if lon_slice else slice(None)
            
            # 获取切片数据
            data_slice = dataset[time_idx, lat_idx, lon_idx]
            
            # 获取对应的坐标切片
            times_sliced = times[time_idx]
            lats_sliced = lats[lat_idx]
            lons_sliced = lons[lon_idx]
            
            # 创建MultiIndex
            index = pd.MultiIndex.from_product(
                [times_sliced, lats_sliced, lons_sliced],
                names=["time", "lat", "lon"]
            )
            
            # 创建DataFrame
            df = pd.DataFrame(
                data=data_slice.reshape(-1, 1),  # 展平数据
                index=index,
                columns=[variable_name]
            )
            self._log_operation(
                operation=operation,
                status="SUCCESS",
                message=f"Result shape: {df.shape}"
            )
            return df
            
        except KeyError as e:
            error_msg = f"Variable '{variable_name}' does not exist"
            self._log_operation(
                operation=operation,
                status="FAILED",
                message=error_msg,
                exception=e
            )
            raise KeyError(error_msg) from e
        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED",
                message=f"Unexpected error: {str(e)}",
                exception=e
            )
            raise
    
    def get_variables_slices(self, variable_names: List[str], 
                             time_slice: slice = None, 
                             lat_slice: slice = None, 
                             lon_slice: slice = None) -> pd.DataFrame:
        """
        批量获取多个变量的切片数据
        
        参数:
            variable_names: 变量名列表
            time_slice: 时间维度切片
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
            
        返回:
            pd.DataFrame: 包含所有变量切片数据的DataFrame
        """
        operation = "Get multiple variables slices"
        self._log_operation(
            operation=operation,
            status="STARTED",
            message=f"Variables: {variable_names}, "
                   f"Slices - time: {time_slice}, lat: {lat_slice}, lon: {lon_slice}"
        )
        try:
            dfs = []
            for var_name in variable_names:
                self._log_operation(
                    operation=f"Processing variable: {var_name}",
                    status="PROGRESS",
                    message=f"Progress: {len(dfs)+1}/{len(variable_names)}"
                )
                df = self.get_variable_slice(var_name, time_slice, lat_slice, lon_slice)
                dfs.append(df)
            
            # 合并所有DataFrame
            result = pd.concat(dfs, axis=1)
            
            # 记录操作成功
            self._log_operation(
                operation=operation,
                status="SUCCESS",
                message=f"Total shape: {result.shape}, Variables: {list(result.columns)}"
            )
            
            return result       
        except Exception as e:
            # 记录操作失败
            self._log_operation(
                operation=operation,
                status="FAILED",
                message=f"Error: {str(e)}",
                exception=e
            )
            raise
    # -------------------- 统计方法 --------------------        
    def __calculate_statistic(self, variable_name: str, 
                              stat_func: Union[str, Callable], stat_name: str, 
                              time_slice: slice = None, 
                              lat_slice: slice = None, lon_slice: slice = None, 
                              **kwargs) -> Any:
        """
        统计计算核心方法（内部使用）
        
        参数:
            variable_name: 变量名称
            stat_func: 统计函数名(str)或可调用对象
            stat_name: 用于日志记录的统计量名称
            time_slice: 时间维度切片
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
            kwargs: 传递给统计函数的额外参数
        """
        operation = f"Get {stat_name}: {variable_name}"
        self._log_operation(
            operation=operation,
            status="STARTED",
            message=f"Slices - time: {time_slice}, lat: {lat_slice}, lon: {lon_slice}"
        )
        
        try:
            df = self.get_variable_slice(variable_name, time_slice, lat_slice, lon_slice)
            
            if isinstance(stat_func, str):
                if stat_func == 'quantile':
                    result = df.quantile(kwargs.get('q')).iloc[0]
                else:
                    result = getattr(df[variable_name], stat_func)()
            else:
                result = stat_func(df[variable_name], **kwargs)
            
            # 记录操作成功
            self._log_operation(
                operation=operation,
                status="SUCCESS",
                message=f"{stat_name} value: {result:.4f}" if isinstance(result, (int, float)) else ""
            )
            
            return result
        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED",
                message=f"Error: {str(e)}",
                exception=e
            )
            raise
            
    def get_quantile_slice(self, variable_name: str,  quantile: float = 0.25, 
                           time_slice: slice = None, 
                           lat_slice: slice = None, lon_slice: slice = None) -> float:
        """获取指定分位数"""
        if not 0 <= quantile <= 1:
            error_msg = f"Quantile must be between 0 and 1, got {quantile}"
            self._log_operation(
                operation=f"Get {quantile*100}% quantile: {variable_name}",
                status="FAILED",
                message=error_msg
            )
            raise ValueError(error_msg)
        return self.__calculate_statistic(
            variable_name=variable_name,
            stat_func='quantile',
            stat_name=f"{quantile*100}% quantile",
            time_slice=time_slice,
            lat_slice=lat_slice,
            lon_slice=lon_slice,
            q=quantile
        )
    
    def get_mean(self, variable_name: str, time_slice: slice = None, 
                 lat_slice: slice = None, lon_slice: slice = None) -> float:
        """获取平均值"""
        return self.__calculate_statistic(
            variable_name=variable_name,
            stat_func='mean',
            stat_name="mean",
            time_slice=time_slice,
            lat_slice=lat_slice,
            lon_slice=lon_slice
        )
    
    def get_std_deviation(self, variable_name: str, time_slice: slice = None, 
                          lat_slice: slice = None, lon_slice: slice = None) -> float:
        """获取标准差"""
        return self.__calculate_statistic(
            variable_name=variable_name,
            stat_func='std',
            stat_name="standard deviation",
            time_slice=time_slice,
            lat_slice=lat_slice,
            lon_slice=lon_slice
        )

    def get_min(self, variable_name: str, time_slice: slice = None, 
                lat_slice: slice = None, lon_slice: slice = None) -> float:
        """获取最小值"""
        return self.__calculate_statistic(
            variable_name=variable_name,
            stat_func='min',
            stat_name="minimum",
            time_slice=time_slice,
            lat_slice=lat_slice,
            lon_slice=lon_slice
        )
    
    def get_max(self, variable_name: str, time_slice: slice = None, 
                lat_slice: slice = None, lon_slice: slice = None) -> float:
        """获取最大值"""
        return self.__calculate_statistic(
            variable_name=variable_name,
            stat_func='max',
            stat_name="maximum",
            time_slice=time_slice,
            lat_slice=lat_slice,
            lon_slice=lon_slice
        )
    
    def get_median(self, variable_name: str, time_slice: slice = None, 
                   lat_slice: slice = None, lon_slice: slice = None) -> float:
        """获取中位数"""
        return self.__calculate_statistic(
            variable_name=variable_name,
            stat_func='median',
            stat_name="median",
            time_slice=time_slice,
            lat_slice=lat_slice,
            lon_slice=lon_slice
        )
    
    def get_sum(self, variable_name: str, time_slice: slice = None, 
                lat_slice: slice = None, lon_slice: slice = None) -> float:
        """获取和"""
        return self.__calculate_statistic(
            variable_name=variable_name,
            stat_func='sum',
            stat_name="sum",
            time_slice=time_slice,
            lat_slice=lat_slice,
            lon_slice=lon_slice
        )
    
    def get_custom_stat(self, variable_name: str, 
                        stat_func: Callable, stat_name: str, 
                        time_slice: slice = None, 
                        lat_slice: slice = None, lon_slice: slice = None, 
                        **kwargs) -> Any:
        """自定义统计计算"""
        return self.__calculate_statistic(
            variable_name=variable_name,
            stat_func=stat_func,
            stat_name=stat_name,
            time_slice=time_slice,
            lat_slice=lat_slice,
            lon_slice=lon_slice,
            **kwargs
        )
    
    def get_stats(self, variable_name: str, states: List[Union[str, Callable]], 
                  time_slice: slice = None, 
                  lat_slice: slice = None, lon_slice: slice = None, 
                  **kwargs) -> Dict[str, Any]:
        """批量获取多个统计量"""
        results = {}
        for stat in states:
            if isinstance(stat, str):
                stat_name = stat
            else:
                stat_name = stat.__name__
                
            results[stat_name] = self.__calculate_statistic(
                variable_name=variable_name,
                stat_func=stat,
                stat_name=stat_name,
                time_slice=time_slice,
                lat_slice=lat_slice,
                lon_slice=lon_slice,
                **kwargs
            )
        return results
    
    # -------------------- 滑动窗口 --------------------
    def __rolling_window(self, variable_name: str, window_size: int, 
                         stat_func: Union[str, Callable] = 'mean', stat_name: str = 'mean', 
                         time_axis: bool = True, time_slice: slice = None, 
                         lat_slice: slice = None, lon_slice: slice = None, 
                         **kwargs) -> pd.DataFrame:
        """
        滑动窗口计算，目前只支持时间滑动
        
        参数:
            variable_name: 变量名称
            window_size: 窗口大小
            stat_func: 统计函数('mean'/'std'等)或可调用对象
            stat_name: 用于日志记录的统计量名称
            time_axis: 是否沿时间维度滑动(False则为空间维度)
            time_slice: 时间维度切片
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
            kwargs: 传递给统计函数的额外参数
            
        返回:
            滑动窗口计算结果(DataFrame)
        """
        operation = f"Rolling {stat_name} window ({window_size}) on {variable_name}"
        self._log_operation(
            operation=operation,
            status="STARTED",
            message=f"Axis: {'time' if time_axis else 'space'}, Function: {stat_func}"
        )
        
        try:
            df = self.get_variable_slice(variable_name, time_slice, lat_slice, lon_slice)
            
            if time_axis:
                # 时间维度滑动
                rolled = df.groupby(['lat', 'lon'])[variable_name].rolling(window=window_size, min_periods=1)
            else:
                # 空间维度滑动(示例：3x3空间窗口)需要更复杂的实现，可能使用xarray
                pass
                
            if isinstance(stat_func, str):
                if stat_func == 'quantile':
                    result = rolled.quantile(kwargs.get('q')).iloc[0]
                else:
                    result = getattr(rolled, stat_func)()#调用方法
            else:
                result = rolled.apply(stat_func, **kwargs)
            
            self._log_operation(
                operation=operation,
                status="SUCCESS",
                message=f"Result shape: {result.shape}"
            )
            return result.unstack(level=[0,1]) if time_axis else result
            
        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED",
                message=f"Error: {str(e)}",
                exception=e
            )
            raise
            
    def rolling_quantile(self, variable_name: str,  window_size: int, 
                         time_axis: bool = True, quantile: float = 0.25, 
                         **kwargs) -> pd.DataFrame:
        """
        滑动分位数
        
        **kwargs可选:
            time_slice: 时间维度切片
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
        """
        if not 0 <= quantile <= 1:
            error_msg = f"Quantile must be between 0 and 1, got {quantile}"
            self._log_operation(
                operation=f"Get {quantile*100}% quantile: {variable_name}",
                status="FAILED",
                message=error_msg
            )
            raise ValueError(error_msg)
        return self.__rolling_window(
            variable_name=variable_name,
            window_size=window_size,
            stat_func='quantile',
            stat_name=f"{quantile*100}% quantile",
            time_axis=time_axis,
            **kwargs
        )
    
    def rolling_mean(self, variable_name: str, window_size: int, 
                     time_axis: bool = True, **kwargs) -> pd.DataFrame:
        """
        滑动平均
        
        **kwargs可选:
            time_slice: 时间维度切片
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
        """
        return self.__rolling_window(
            variable_name=variable_name,
            window_size=window_size,
            time_axis=time_axis,
            **kwargs
        )
    
    def rolling_std_deviation(self, variable_name: str, window_size: int, 
                              time_axis: bool = True, **kwargs) -> pd.DataFrame:
        """
        滑动标准差
        
        **kwargs可选:
            time_slice: 时间维度切片
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
        """
        return self.__rolling_window(
            variable_name=variable_name,
            window_size=window_size,
            stat_func='std',
            stat_name='standard deviation',
            time_axis=time_axis,
            **kwargs
        )
    
    def rolling_min(self, variable_name: str, window_size: int, 
                    time_axis: bool = True, **kwargs) -> pd.DataFrame:
        """
        滑动最小值
        
        **kwargs可选:
            time_slice: 时间维度切片
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
        """
        return self.__rolling_window(
            variable_name=variable_name,
            window_size=window_size,
            stat_func='min',
            stat_name='minimum',
            time_axis=time_axis,
            **kwargs
        )
    
    def rolling_max(self, variable_name: str, window_size: int, 
                    time_axis: bool = True, **kwargs) -> pd.DataFrame:
        """
        滑动最大值
        
        **kwargs可选:
            time_slice: 时间维度切片
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
        """
        return self.__rolling_window(
            variable_name=variable_name,
            window_size=window_size,
            stat_func='max',
            stat_name='maximum',
            time_axis=time_axis,
            **kwargs
        )
    
    def rolling_median(self, variable_name: str, window_size: int, 
                       time_axis: bool = True, **kwargs) -> pd.DataFrame:
        """
        滑动中位数
        
        **kwargs可选:
            time_slice: 时间维度切片
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
        """
        return self.__rolling_window(
            variable_name=variable_name,
            window_size=window_size,
            stat_func='median',
            stat_name='median',
            time_axis=time_axis,
            **kwargs
        )
    
    def rolling_sum(self, variable_name: str, window_size: int,
                    time_axis: bool = True, **kwargs) -> pd.DataFrame:
        """
        滑动和
        
        **kwargs可选:
            time_slice: 时间维度切片
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
        """
        return self.__rolling_window(
            variable_name=variable_name,
            window_size=window_size,
            stat_func='sum',
            stat_name='sum',
            time_axis=time_axis,
            **kwargs
        )
    
    def custom_stat(self, variable_name: str, window_size: int, 
                    stat_func: Callable, stat_name: str, 
                    time_axis: bool = True, **kwargs) -> pd.DataFrame:
        """
        滑动自定义
        
        **kwargs可选:
            time_slice: 时间维度切片
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
            传递给统计函数的额外参数
        """
        return self.__rolling_window(
            variable_name=variable_name,
            window_size=window_size,
            stat_func=stat_func,
            stat_name=stat_name,
            time_axis=time_axis,
            **kwargs
        )
    
    # -------------------- 相关性分析 --------------------    
    def calculate_temporal_correlation(self, var1_name: str, var2_name: str, 
                                       time_slice: slice = None, 
                                       lat_slice: slice = None, lon_slice: slice = None, 
                                       method: str = 'pearson') -> float:
        """
        计算两个变量的时间序列相关性(空间聚合后)
        
        参数:
            var1_name: 第一个变量名
            var2_name: 第二个变量名
            time_slice: 时间维度切片
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
            method: 相关性计算方法 ('pearson'|'spearman'|'kendall')
            
        返回:
            相关系数
            
        异常:
            ValueError: 当方法不支持或数据形状不匹配时
        """
        operation = f"Calculate {method} temporal correlation between {var1_name} and {var2_name}"
        self._log_operation(
            operation=operation, 
            status="STARTED"
            )
        
        try:
            # 获取两个变量的数据
            df1 = self.get_variable_slice(var1_name, time_slice, lat_slice, lon_slice)
            df2 = self.get_variable_slice(var2_name, time_slice, lat_slice, lon_slice)
            
            # 合并数据并计算相关性
            combined = pd.concat([df1, df2], axis=1)
            corr = combined.corr(method=method).iloc[0, 1]
            
            self._log_operation(
                operation=operation,
                status="SUCCESS",
                message=f"Correlation: {corr:.3f}"
            )
            return corr  
        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED",
                message=f"Error: {str(e)}",
                exception=e
            )
            raise
            
    def calculate_spatial_correlation(self, var1_name: str, var2_name: str, 
                                      time_point: int = 0, 
                                      lat_slice: slice = None, lon_slice: slice = None, 
                                      method: str = 'pearson') -> float:
        """
        计算两个变量在特定时间点的空间相关性
        
        参数:
            var1_name: 第一个变量名
            var2_name: 第二个变量名
            time_point: 时间点索引
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
            method: 相关性计算方法 ('pearson'|'spearman'|'kendall')
            
        返回:
            空间相关系数
            
        异常:
            ValueError: 当方法不支持或数据形状不匹配时
        """
        operation = f"Calculate {method} spatial correlation between {var1_name} and {var2_name} at time={time_point}"
        self._log_operation(operation=operation, status="STARTED")
        
        try:
            # 获取特定时间点的数据
            time_slice = slice(time_point, time_point+1)
            df1 = self.get_variable_slice(var1_name, time_slice, lat_slice, lon_slice)
            df2 = self.get_variable_slice(var2_name, time_slice, lat_slice, lon_slice)
            
            # 提取空间数据并计算相关性
            spatial_data = pd.DataFrame({
                'var1': df1[var1_name].values.flatten(),
                'var2': df2[var2_name].values.flatten()
            })
            corr = spatial_data.corr(method=method).iloc[0, 1]
            
            self._log_operation(
                operation=operation,
                status="SUCCESS",
                message=f"Spatial correlation: {corr:.3f}"
            )
            return corr
            
        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED",
                message=f"Error: {str(e)}",
                exception=e
            )
            raise

# ---------------------- 时间重采样组件 ----------------------
class TimeResampler:
    def __init__(self, hdf5_reader):
        """
        初始化时间重采样组件
        
        参数:
            hdf5_reader: HDF5reader_writer 实例
        """
        self.reader = hdf5_reader
        self._logger = getattr(hdf5_reader, '_logger', None)
        self.analyzer = DataAnalyzer(hdf5_reader)  # 复用DataAnalyzer的功能

    def _log_operation(self, **kwargs):
        """仅在日志启用时记录操作"""
        if self._logger is not None:
            self._logger.log_operation(**kwargs)

    def resample_time(self, variable_name: str, rule: str, method: str = 'mean', 
                      lat_slice: slice = None, lon_slice: slice = None, **kwargs) -> pd.DataFrame:
        """
        对变量数据进行时间维度重采样
        
        参数:
            variable_name: 变量名称
            rule: 重采样规则(如'D'按天,'H'按小时,'M'按月)
            method: 重采样方法('mean', 'sum', 'max', 'min'等或自定义函数)
            lat_slice: 纬度维度切片
            lon_slice: 纬度维度切片
            kwargs: 传递给resample的其他参数
            
        返回:
            pd.DataFrame: 重采样后的数据
            
        示例:
            # 将温度数据按日平均
            daily_temp = resampler.resample_time("Temperature", 'D')
        """
        operation = f"Resample {variable_name} by {rule}"
        self._log_operation(
            operation=operation,
            status="STARTED",
            message=f"Method: {method}, lat: {lat_slice}, lon: {lon_slice}"
        )
        try:
            df = self.analyzer.get_variable_slice(
                variable_name,
                lat_slice=lat_slice,
                lon_slice=lon_slice
            )
            
            df_reset = df.reset_index()
            
            # 确保时间列是datetime类型
            df_reset['time'] = pd.to_datetime(df_reset['time'])
            
            # 设置时间索引
            df_time_index = df_reset.set_index('time')
            
            # 分组重采样
            if isinstance(method, str):
                # 对每个(lat,lon)位置分组后进行时间重采样
                resampled = (df_time_index
                            .groupby(['lat', 'lon'])[variable_name]
                            .resample(rule)
                            .agg(method))
            else:
                resampled = (df_time_index
                            .groupby(['lat', 'lon'])[variable_name]
                            .resample(rule)
                            .apply(method))    
                
            # 重建MultiIndex
            result = resampled.unstack(level=['lat', 'lon'])
            
            self._log_operation(
                operation=operation,
                status="SUCCESS",
                message=f"Resampled shape: {result.shape}"
            )
            
            return result
        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED",
                message=f"Error: {str(e)}",
                exception=e
            )
            raise

    def resample_multi_vars(self, variable_names: List[str], rule: str, 
                            method: Union[str, Dict[str, str]] = 'mean',
                            lat_slice: slice = None,
                            lon_slice: slice = None,
                            **kwargs
                            ) -> Dict[str, pd.DataFrame]:
        """
        批量重采样多个变量
        
        参数:
            variable_names: 变量名列表
            rule: 重采样规则
            method: 统一方法或各变量指定方法(字典形式)
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
            
        返回:
            dict: {变量名: 重采样数据} 的字典
        """
        operation = f"Resample multiple variables by {rule}"
        self._log_operation(
            operation=operation,
            status="STARTED",
            message=f"Variables: {variable_names}, Method: {method}"
        )
        try:
            results = {}
            for var in variable_names:
                current_method = method[var] if isinstance(method, dict) else method
                
                self._log_operation(
                    operation=f"Processing {var}",
                    status="PROGRESS",
                    message=f"Method: {current_method}"
                )
                
                results[var] = self.resample_time(
                    variable_name=var,
                    rule=rule,
                    method=current_method,
                    lat_slice=lat_slice,
                    lon_slice=lon_slice,
                    **kwargs
                )
            
            self._log_operation(
                operation=operation,
                status="SUCCESS",
                message=f"Completed {len(results)} variables"
            )
            
            return results

        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED",
                message=f"Error: {str(e)}",
                exception=e
            )
            raise

# ---------------------- 数据筛选组件 ----------------------
class DataFilter:
    def __init__(self, hdf5_reader):
        """
        初始化数据筛选组件
        
        参数:
            hdf5_reader: HDF5reader_writer 实例
        """
        self.reader = hdf5_reader
        self.analyzer = DataAnalyzer(hdf5_reader)
        self._logger = getattr(hdf5_reader, '_logger', None)  # 复用主类的日志
    
    def _log_operation(self, **kwargs):
        """日志记录方法"""
        if self._logger is not None:
            self._logger.log_operation(**kwargs)
    
    def load_data(self, variable_names: Union[str, List[str]], time_slice: slice = None, 
                  lat_slice: slice = None, lon_slice: slice = None) -> pd.DataFrame:
        """
        复用DataAnalyzer的功能加载数据到DataFrame
        
        参数:
            variable_names: 变量名或变量名列表
            time_slice: 时间维度切片
            lat_slice: 纬度维度切片
            lon_slice: 经度维度切片
            
        返回:
            包含请求数据的DataFrame
        """
        operation = "Load data for filtering (using DataAnalyzer)"
        self._log_operation(operation=operation, status="STARTED")
        
        try:
            # 使用DataAnalyzer的get_variable_slice或get_variables_slices方法
            if isinstance(variable_names, str):
                df = self.analyzer.get_variable_slice(
                    variable_names,
                    time_slice=time_slice,
                    lat_slice=lat_slice,
                    lon_slice=lon_slice
                )
            else:
                df = self.analyzer.get_variables_slices(
                    variable_names,
                    time_slice=time_slice,
                    lat_slice=lat_slice,
                    lon_slice=lon_slice
                )
            
            self._log_operation(
                operation=operation,
                status="SUCCESS",
                message=f"Loaded data shape: {df.shape}"
            )
            return df
        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED",
                message=f"Error: {str(e)}",
                exception=e
            )
            raise
    
    def filter_by_condition(self, df: pd.DataFrame, variable_name: str,
                            condition: str, value: Union[float, int, str]
                            ) -> pd.DataFrame:
        """
        根据条件筛选数据
        
        参数:
            df: 输入DataFrame
            condition: 筛选条件 ('>', '<', '>=', '<=', '==', '!=')
            value: 比较值
            
        返回:
            筛选后的DataFrame
        """
        operation = f"Filter data by condition {condition} {value}"
        self._log_operation(operation=operation, status="STARTED")
        
        try:
            if variable_name not in df.columns:
                error_msg = f"Variable '{variable_name}' not found in DataFrame"
                self._log_operation(
                    operation=operation,
                    status="FAILED",
                    message=error_msg
                )
                raise ValueError(error_msg)
            
            # 运算符映射字典
            op_map = {
                '>': operator.gt,
                '<': operator.lt,
                '>=': operator.ge,
                '<=': operator.le,
                '==': operator.eq,
                '!=': operator.ne
            }
            
            # 获取对应的运算符函数
            op_func = op_map.get(condition)
            if op_func is None:
                error_msg = f"Unsupported condition: {condition}"
                self._log_operation(
                    operation=operation,
                    status="FAILED",
                    message=error_msg
                )
                raise ValueError(error_msg)
            
            # 应用筛选条件
            mask = op_func(df[variable_name], value)
            filtered = df[mask]
            
            self._log_operation(
                operation=operation,
                status="SUCCESS",
                message=f"Filtered records: {len(filtered)}"
            )
            return filtered
        except ValueError:
            raise
        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED",
                message=f"Error: {str(e)}",
                exception=e
            )
            raise
    
    def filter_by_query(self, df: pd.DataFrame, query_str: str) -> pd.DataFrame:
        """
        使用查询字符串筛选数据
        
        参数:
            df: 输入DataFrame
            query_str: pandas查询字符串
            
        返回:
            筛选后的DataFrame
        """
        operation = f"Filter data by query: {query_str}"
        self._log_operation(operation=operation, status="STARTED")
        
        try:
            filtered = df.query(query_str)
            self._log_operation(
                operation=operation,
                status="SUCCESS",
                message=f"Filtered records: {len(filtered)}"
            )
            return filtered    
        except Exception as e:
            self._log_operation(
                operation=operation,
                status="FAILED",
                message=f"Error: {str(e)}",
                exception=e
            )
            raise