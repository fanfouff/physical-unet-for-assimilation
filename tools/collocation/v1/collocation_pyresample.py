#!/usr/bin/env python3
"""
使用 pyresample 库的高级配准方案
提供更精确的重采样和插值
"""

import h5py
import xarray as xr
import numpy as np
from pyresample import geometry, kd_tree
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PyresampleCollocation:
    """使用 pyresample 的配准实现"""
    
    def __init__(self):
        self.fy3d_swath = None
        self.era5_grid = None
        
    def read_fy3d(self, filepath):
        """读取FY-3D数据并创建Swath定义"""
        print(f"📡 读取 FY-3D 数据...")
        
        with h5py.File(filepath, 'r') as f:
            bt = f['Brightness_Temperature'][:]
            lat = f['Latitude'][:]
            lon = f['Longitude'][:]
            
        # 质量控制
        bt_qc = bt.copy()
        invalid = (bt < 100) | (bt > 400) | np.isnan(bt)
        bt_qc[invalid] = np.nan
        
        # 创建 Swath 定义
        swath_def = geometry.SwathDefinition(lons=lon, lats=lat)
        
        self.fy3d_data = {
            'bt': bt_qc,
            'swath_def': swath_def,
            'shape': bt.shape
        }
        
        print(f"   形状: {bt.shape}")
        print(f"   经纬度范围: [{lat.min():.2f}, {lat.max():.2f}], "
              f"[{lon.min():.2f}, {lon.max():.2f}]")
        
        return self.fy3d_data
    
    def read_era5(self, filepath):
        """读取ERA5数据并创建Grid定义"""
        print(f"\n🌍 读取 ERA5 数据...")
        
        ds = xr.open_dataset(filepath, engine='cfgrib')
        
        lat = ds.latitude.values
        lon = ds.longitude.values
        
        # 创建 Area 定义
        area_def = geometry.GridDefinition(lons=lon, lats=lat)
        
        # 提取温度
        if 't' in ds:
            temp = ds['t'].values
        else:
            temp = ds['temperature'].values
        
        if temp.ndim == 4:
            temp = temp[0]  # (level, lat, lon)
        
        self.era5_data = {
            'temperature': temp,
            'area_def': area_def,
            'pressure': ds.isobaricInhPa.values,
            'ds': ds
        }
        
        print(f"   形状: {temp.shape}")
        print(f"   气压层: {len(ds.isobaricInhPa)} 层")
        
        return self.era5_data
    
    def resample_era5_to_swath(self, radius_of_influence=50000, 
                                neighbours=1, fill_value=np.nan):
        """
        将ERA5重采样到卫星观测点
        
        Parameters:
        -----------
        radius_of_influence : float
            影响半径（米）
        neighbours : int
            最近邻数量
        fill_value : float
            填充值
        """
        print(f"\n🎯 重采样 ERA5 到卫星轨道...")
        print(f"   影响半径: {radius_of_influence/1000:.1f} km")
        print(f"   最近邻数: {neighbours}")
        
        swath_def = self.fy3d_data['swath_def']
        area_def = self.era5_data['area_def']
        temp = self.era5_data['temperature']
        
        n_levels = temp.shape[0]
        swath_shape = self.fy3d_data['shape'][:2]  # (scan_line, scan_angle)
        
        # 为每个气压层重采样
        resampled_profiles = np.zeros((*swath_shape, n_levels))
        
        print(f"   处理 {n_levels} 个气压层...")
        
        for i, level_data in enumerate(temp):
            if (i + 1) % 10 == 0:
                print(f"      层 {i+1}/{n_levels}...")
            
            # 重采样单层
            resampled = kd_tree.resample_nearest(
                area_def,
                level_data,
                swath_def,
                radius_of_influence=radius_of_influence,
                neighbours=neighbours,
                fill_value=fill_value
            )
            
            resampled_profiles[:, :, i] = resampled
        
        self.resampled_profiles = resampled_profiles
        
        print(f"   ✓ 重采样完成")
        print(f"   输出形状: {resampled_profiles.shape}")
        
        return resampled_profiles
    
    def create_training_data(self):
        """创建训练数据对"""
        print(f"\n📊 创建训练数据对...")
        
        bt = self.fy3d_data['bt']
        profiles = self.resampled_profiles
        
        # 展平
        n_scan, n_angle, n_channel = bt.shape
        X = bt.reshape(-1, n_channel)
        Y = profiles.reshape(-1, profiles.shape[-1])
        
        # 移除NaN
        valid = ~(np.any(np.isnan(X), axis=1) | np.any(np.isnan(Y), axis=1))
        
        X_clean = X[valid]
        Y_clean = Y[valid]
        
        print(f"   有效样本: {len(X_clean):,}/{len(X):,}")
        print(f"   X shape: {X_clean.shape}")
        print(f"   Y shape: {Y_clean.shape}")
        
        return X_clean, Y_clean


def advanced_collocation_example(fy3d_file, era5_file):
    """使用pyresample的完整示例"""
    
    # 初始化
    collocator = PyresampleCollocation()
    
    # 读取数据
    collocator.read_fy3d(fy3d_file)
    collocator.read_era5(era5_file)
    
    # 重采样
    collocator.resample_era5_to_swath(
        radius_of_influence=50000,  # 50 km
        neighbours=1
    )
    
    # 创建训练数据
    X, Y = collocator.create_training_data()
    
    return X, Y


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python collocation_pyresample.py <FY3D文件> <ERA5文件>")
        sys.exit(1)
    
    fy3d_file = sys.argv[1]
    era5_file = sys.argv[2]
    
    X, Y = advanced_collocation_example(fy3d_file, era5_file)
    
    # 保存
    np.save('X_pyresample.npy', X)
    np.save('Y_pyresample.npy', Y)
    
    print("\n✅ 完成! 已保存:")
    print("   - X_pyresample.npy")
    print("   - Y_pyresample.npy")
