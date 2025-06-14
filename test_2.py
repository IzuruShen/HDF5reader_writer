# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 20:57:24 2025

@author: mirag
"""

import pytest
import numpy as np
import h5py
import tempfile
import os
from datetime import datetime
from components import Converter, Logger, DataTransformer, DataPreprocessor, DataAnalyzer, TimeResampler, DataFilter

@pytest.fixture
def sample_hdf5_file():
    """创建一个临时的HDF5测试文件"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    temp_file.close()
    
    # 创建测试数据
    time_points = 10
    lat_points = 5
    lon_points = 5
    
    with h5py.File(temp_file.name, 'w') as f:
        # 全局属性
        f.attrs['CreationDate'] = datetime.now().isoformat()
        f.attrs['Description'] = 'Test HDF5 file'
        
        # 创建观测组
        obs_group = f.create_group('Observations')
        
        # 创建坐标组
        coord_group = obs_group.create_group('Coordinates')
        coord_group.create_dataset('Time', data=np.arange(time_points))
        coord_group.create_dataset('Latitude', data=np.linspace(-90, 90, lat_points))
        coord_group.create_dataset('Longitude', data=np.linspace(-180, 180, lon_points))
        
        # 创建测试变量
        temp_data = np.random.rand(time_points, lat_points, lon_points) * 30 + 270  # 270-300K
        obs_group.create_dataset('Temperature', data=temp_data)
        obs_group['Temperature'].attrs['units'] = 'K'
        obs_group['Temperature'].attrs['description'] = 'Surface temperature'
        
        wind_data = np.random.rand(time_points, lat_points, lon_points) * 20  # 0-20 m/s
        obs_group.create_dataset('WindSpeed', data=wind_data)
        obs_group['WindSpeed'].attrs['units'] = 'm/s'
        obs_group['WindSpeed'].attrs['description'] = 'Wind speed at 10m'
        
        precip_data = np.random.rand(time_points, lat_points, lon_points) * 10  # 0-10 mm
        obs_group.create_dataset('Precipitation', data=precip_data)
        obs_group['Precipitation'].attrs['units'] = 'mm/h'
        obs_group['Precipitation'].attrs['description'] = 'Hourly precipitation'
    
    yield temp_file.name
    
    # 测试完成后删除临时文件
    os.unlink(temp_file.name)

@pytest.fixture
def hdf5_reader(sample_hdf5_file):
    """创建一个HDF5reader_writer实例"""
    from your_module import HDF5reader_writer  # 替换为你的模块名
    return HDF5reader_writer(sample_hdf5_file, enable_logging=False)

@pytest.fixture
def converter():
    return Converter()

@pytest.fixture
def logger(tmpdir):
    log_file = str(tmpdir.join('test.log'))
    return Logger(log_file=log_file)

@pytest.fixture
def data_transformer(hdf5_reader):
    return DataTransformer(hdf5_reader)

@pytest.fixture
def data_preprocessor(hdf5_reader):
    return DataPreprocessor(hdf5_reader)

@pytest.fixture
def data_analyzer(hdf5_reader):
    return DataAnalyzer(hdf5_reader)

@pytest.fixture
def time_resampler(hdf5_reader):
    return TimeResampler(hdf5_reader)

@pytest.fixture
def data_filter(hdf5_reader):
    return DataFilter(hdf5_reader)