#!/usr/bin/env python3
"""
卫星资料同化数据匹配工具
FY-3D MWTS L1C5 与 ERA5 Reanalysis 数据配准

功能：
1. 读取FY-3D卫星亮温数据
2. 读取ERA5再分析数据
3. 时空匹配（Pixel-wise Collocation）
4. 垂直廓线插值
5. 数据质量控制
6. 生成训练数据对 (X, Y)
"""

import h5py
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FY3D_Reader:
    """FY-3D MWTS L1C5 数据读取器"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        
    def read(self):
        """读取FY-3D H5文件"""
        print(f"📡 读取 FY-3D 数据: {self.filepath}")
        
        with h5py.File(self.filepath, 'r') as f:
            # 读取亮温数据 (Scan_Line, Scan_Angle, Channels)
            bt = f['Brightness_Temperature'][:]
            
            # 读取经纬度
            lat = f['Latitude'][:]  # (Scan_Line, Scan_Angle)
            lon = f['Longitude'][:]
            
            # 读取时间信息
            # FY-3D通常存储为扫描线时间
            if 'Obs_Time' in f:
                obs_time = f['Obs_Time'][:]
            elif 'Time' in f:
                obs_time = f['Time'][:]
            else:
                # 从文件名提取时间
                obs_time = self._extract_time_from_filename()
            
            # 读取其他可能的QC信息
            if 'Quality_Flag' in f:
                quality = f['Quality_Flag'][:]
            else:
                quality = np.ones_like(lat, dtype=np.uint8)
            
        self.data = {
            'bt': bt,
            'lat': lat,
            'lon': lon,
            'time': obs_time,
            'quality': quality,
            'shape': bt.shape
        }
        
        print(f"   亮温形状: {bt.shape}")
        print(f"   通道数: {bt.shape[-1]}")
        print(f"   扫描线数: {bt.shape[0]}")
        print(f"   扫描角度数: {bt.shape[1]}")
        
        return self.data
    
    def _extract_time_from_filename(self):
        """从文件名提取观测时间"""
        import re
        # FY3D_MWTSX_ORBT_L2_ATP_MLT_NUL_20210701_1455_033KM_MS.L1c
        match = re.search(r'(\d{8})_(\d{4})', self.filepath)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            dt = datetime.strptime(date_str + time_str, '%Y%m%d%H%M')
            return dt
        return datetime(2021, 1, 1)  # 默认值
    
    def quality_control(self, bt_min=100.0, bt_max=400.0):
        """
        亮温数据质量控制
        
        Parameters:
        -----------
        bt_min : float
            最小合理亮温 (K)
        bt_max : float
            最大合理亮温 (K)
        """
        print(f"\n🔍 执行质量控制...")
        print(f"   亮温范围: [{bt_min}, {bt_max}] K")
        
        bt = self.data['bt']
        
        # 标记异常值
        invalid_mask = (bt < bt_min) | (bt > bt_max) | np.isnan(bt)
        
        # 统计
        total_pixels = bt.size
        invalid_pixels = np.sum(invalid_mask)
        valid_ratio = (1 - invalid_pixels / total_pixels) * 100
        
        print(f"   总像元数: {total_pixels:,}")
        print(f"   异常像元数: {invalid_pixels:,}")
        print(f"   有效率: {valid_ratio:.2f}%")
        
        # 设置为NaN
        bt_qc = bt.copy()
        bt_qc[invalid_mask] = np.nan
        
        self.data['bt_qc'] = bt_qc
        self.data['valid_mask'] = ~invalid_mask
        
        return bt_qc
    
    def flatten_to_points(self):
        """将数据展平为点云格式"""
        print(f"\n📍 展平为点云...")
        
        bt = self.data.get('bt_qc', self.data['bt'])
        lat = self.data['lat']
        lon = self.data['lon']
        
        n_scan, n_angle, n_channel = bt.shape
        
        # 展平
        bt_flat = bt.reshape(-1, n_channel)  # (N, Channels)
        lat_flat = lat.flatten()  # (N,)
        lon_flat = lon.flatten()  # (N,)
        
        # 如果有时间维度
        if isinstance(self.data['time'], datetime):
            time_flat = np.full(len(lat_flat), self.data['time'])
        else:
            time_flat = np.repeat(self.data['time'], n_angle)
        
        # 移除全NaN的点
        valid_idx = ~np.all(np.isnan(bt_flat), axis=1)
        
        self.points = {
            'bt': bt_flat[valid_idx],
            'lat': lat_flat[valid_idx],
            'lon': lon_flat[valid_idx],
            'time': time_flat[valid_idx],
            'n_points': np.sum(valid_idx)
        }
        
        print(f"   有效观测点: {self.points['n_points']:,}")
        
        return self.points


class ERA5_Reader:
    """ERA5 再分析数据读取器"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        
    def read(self, time_window=3):
        """
        读取ERA5 GRIB文件
        
        Parameters:
        -----------
        time_window : int
            时间窗口（小时），用于选择最接近的时间步
        """
        print(f"\n🌍 读取 ERA5 数据: {self.filepath}")
        
        # 使用xarray读取GRIB
        ds = xr.open_dataset(self.filepath, engine='cfgrib')
        
        print(f"   变量: {list(ds.data_vars)}")
        print(f"   维度: {dict(ds.dims)}")
        print(f"   气压层: {len(ds.isobaricInhPa.values)} 层")
        print(f"   时间: {ds.time.values}")
        
        # 提取温度场
        # 注意：ERA5变量名可能是 't' 或 'temperature'
        if 't' in ds:
            temp = ds['t']
        elif 'temperature' in ds:
            temp = ds['temperature']
        else:
            raise ValueError(f"未找到温度变量，可用变量: {list(ds.data_vars)}")
        
        self.data = {
            'ds': ds,
            'temperature': temp,
            'lat': ds.latitude.values,
            'lon': ds.longitude.values,
            'pressure': ds.isobaricInhPa.values,
            'time': ds.time.values
        }
        
        return self.data
    
    def build_spatial_tree(self):
        """构建空间KDTree用于快速最近邻查找"""
        print(f"\n🌲 构建空间索引...")
        
        lat = self.data['lat']
        lon = self.data['lon']
        
        # 创建2D网格
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # 展平为点云
        points = np.column_stack([
            lat_grid.flatten(),
            lon_grid.flatten()
        ])
        
        # 构建KDTree
        tree = cKDTree(points)
        
        self.spatial_tree = {
            'tree': tree,
            'lat_grid': lat_grid,
            'lon_grid': lon_grid,
            'shape': lat_grid.shape
        }
        
        print(f"   网格点数: {len(points):,}")
        print(f"   网格形状: {lat_grid.shape}")
        
        return tree
    
    def interpolate_to_satellite(self, sat_points, method='nearest', max_distance=0.5):
        """
        将ERA5数据插值到卫星观测点
        
        Parameters:
        -----------
        sat_points : dict
            卫星观测点数据
        method : str
            插值方法 ('nearest', 'linear', 'bilinear')
        max_distance : float
            最大匹配距离（度）
        
        Returns:
        --------
        profiles : ndarray
            垂直廓线 (N_points, N_levels)
        """
        print(f"\n🎯 执行空间匹配...")
        print(f"   插值方法: {method}")
        print(f"   最大距离: {max_distance}°")
        
        sat_lat = sat_points['lat']
        sat_lon = sat_points['lon']
        n_points = len(sat_lat)
        
        # 查询最近邻
        query_points = np.column_stack([sat_lat, sat_lon])
        distances, indices = self.spatial_tree['tree'].query(query_points)
        
        # 过滤距离过大的点
        valid_match = distances < max_distance
        n_valid = np.sum(valid_match)
        
        print(f"   卫星观测点: {n_points:,}")
        print(f"   成功匹配: {n_valid:,} ({n_valid/n_points*100:.1f}%)")
        print(f"   平均距离: {np.mean(distances[valid_match]):.3f}°")
        
        # 提取对应的ERA5格点索引
        lat_shape, lon_shape = self.spatial_tree['shape']
        era5_lat_idx = indices // lon_shape
        era5_lon_idx = indices % lon_shape
        
        # 提取温度廓线
        temp_data = self.data['temperature'].values
        if temp_data.ndim == 4:  # (time, level, lat, lon)
            temp_data = temp_data[0]  # 取第一个时间步
        
        n_levels = temp_data.shape[0]
        profiles = np.full((n_points, n_levels), np.nan)
        
        for i in range(n_points):
            if valid_match[i]:
                lat_idx = era5_lat_idx[i]
                lon_idx = era5_lon_idx[i]
                profiles[i] = temp_data[:, lat_idx, lon_idx]
        
        self.matched_profiles = {
            'profiles': profiles,
            'valid_mask': valid_match,
            'distances': distances,
            'pressure_levels': self.data['pressure']
        }
        
        return profiles


class DataCollocation:
    """数据配准主类"""
    
    def __init__(self):
        self.fy3d_reader = None
        self.era5_reader = None
        
    def process(self, fy3d_file, era5_file, 
                bt_min=100.0, bt_max=400.0,
                max_distance=0.5):
        """
        主处理流程
        
        Parameters:
        -----------
        fy3d_file : str
            FY-3D文件路径
        era5_file : str
            ERA5文件路径
        bt_min, bt_max : float
            亮温质量控制范围
        max_distance : float
            最大匹配距离
        
        Returns:
        --------
        X : ndarray (N, n_channels)
            输入特征（卫星亮温）
        Y : ndarray (N, n_levels)
            标签（温度廓线）
        """
        
        print("="*70)
        print("卫星资料同化数据配准工具")
        print("="*70)
        
        # 1. 读取FY-3D数据
        self.fy3d_reader = FY3D_Reader(fy3d_file)
        self.fy3d_reader.read()
        self.fy3d_reader.quality_control(bt_min, bt_max)
        sat_points = self.fy3d_reader.flatten_to_points()
        
        # 2. 读取ERA5数据
        self.era5_reader = ERA5_Reader(era5_file)
        self.era5_reader.read()
        self.era5_reader.build_spatial_tree()
        
        # 3. 空间匹配
        profiles = self.era5_reader.interpolate_to_satellite(
            sat_points, 
            max_distance=max_distance
        )
        
        # 4. 构建训练数据对
        print(f"\n📊 构建训练数据...")
        
        X = sat_points['bt']  # (N, Channels)
        Y = profiles  # (N, Levels)
        
        # 移除含NaN的样本
        valid_samples = ~(np.any(np.isnan(X), axis=1) | np.any(np.isnan(Y), axis=1))
        
        X_clean = X[valid_samples]
        Y_clean = Y[valid_samples]
        
        print(f"   原始样本数: {len(X):,}")
        print(f"   有效样本数: {len(X_clean):,}")
        print(f"   X shape: {X_clean.shape}  (Samples, Channels)")
        print(f"   Y shape: {Y_clean.shape}  (Samples, Levels)")
        
        # 统计信息
        print(f"\n📈 数据统计:")
        print(f"   亮温范围: [{np.nanmin(X_clean):.2f}, {np.nanmax(X_clean):.2f}] K")
        print(f"   温度范围: [{np.nanmin(Y_clean):.2f}, {np.nanmax(Y_clean):.2f}] K")
        print(f"   亮温均值: {np.nanmean(X_clean):.2f} K")
        print(f"   温度均值: {np.nanmean(Y_clean):.2f} K")
        
        self.X = X_clean
        self.Y = Y_clean
        self.pressure_levels = self.era5_reader.data['pressure']
        
        return X_clean, Y_clean
    
    def save_numpy(self, output_prefix='collocation_data'):
        """保存为Numpy格式"""
        print(f"\n💾 保存数据...")
        
        np.save(f'{output_prefix}_X.npy', self.X)
        np.save(f'{output_prefix}_Y.npy', self.Y)
        np.save(f'{output_prefix}_pressure.npy', self.pressure_levels)
        
        print(f"   ✓ {output_prefix}_X.npy")
        print(f"   ✓ {output_prefix}_Y.npy")
        print(f"   ✓ {output_prefix}_pressure.npy")
        
        # 保存元数据
        metadata = {
            'n_samples': len(self.X),
            'n_channels': self.X.shape[1],
            'n_levels': self.Y.shape[1],
            'pressure_levels': self.pressure_levels.tolist(),
            'bt_range': [float(np.min(self.X)), float(np.max(self.X))],
            'temp_range': [float(np.min(self.Y)), float(np.max(self.Y))]
        }
        
        import json
        with open(f'{output_prefix}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✓ {output_prefix}_metadata.json")
        
    def save_hdf5(self, output_file='collocation_data.h5'):
        """保存为HDF5格式（更适合大数据）"""
        print(f"\n💾 保存为HDF5格式...")
        
        with h5py.File(output_file, 'w') as f:
            # 创建数据集
            f.create_dataset('X_brightness_temperature', data=self.X, 
                           compression='gzip', compression_opts=4)
            f.create_dataset('Y_temperature_profile', data=self.Y,
                           compression='gzip', compression_opts=4)
            f.create_dataset('pressure_levels', data=self.pressure_levels)
            
            # 添加属性
            f.attrs['n_samples'] = len(self.X)
            f.attrs['n_channels'] = self.X.shape[1]
            f.attrs['n_levels'] = self.Y.shape[1]
            f.attrs['description'] = 'FY-3D MWTS and ERA5 collocated data'
        
        print(f"   ✓ {output_file}")


def main():
    """主函数示例"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='FY-3D 和 ERA5 数据配准工具'
    )
    parser.add_argument('fy3d_file', help='FY-3D L1C5文件路径')
    parser.add_argument('era5_file', help='ERA5 GRIB文件路径')
    parser.add_argument('-o', '--output', default='collocation_data',
                       help='输出文件前缀')
    parser.add_argument('--bt-min', type=float, default=100.0,
                       help='最小亮温 (K)')
    parser.add_argument('--bt-max', type=float, default=400.0,
                       help='最大亮温 (K)')
    parser.add_argument('--max-distance', type=float, default=0.5,
                       help='最大匹配距离 (度)')
    parser.add_argument('--format', choices=['numpy', 'hdf5', 'both'],
                       default='both', help='输出格式')
    
    args = parser.parse_args()
    
    # 执行配准
    collocation = DataCollocation()
    X, Y = collocation.process(
        args.fy3d_file,
        args.era5_file,
        bt_min=args.bt_min,
        bt_max=args.bt_max,
        max_distance=args.max_distance
    )
    
    # 保存结果
    if args.format in ['numpy', 'both']:
        collocation.save_numpy(args.output)
    
    if args.format in ['hdf5', 'both']:
        collocation.save_hdf5(f'{args.output}.h5')
    
    print("\n" + "="*70)
    print("✅ 处理完成!")
    print("="*70)


if __name__ == '__main__':
    main()
