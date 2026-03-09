#!/usr/bin/env python3
"""
卫星资料同化数据匹配工具（修复版 - 支持缩放因子和正确的维度处理）
FY-3D MWTS L1 与 ERA5 Reanalysis 数据配准

主要修复：
1. 自动读取并应用 Slope/Intercept 进行数据定标
2. 正确处理 lat/lon 维度（可能与bt维度不同）
3. 智能质量控制
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
    """FY-3D/FY-3F MWTS 数据读取器（修复版）"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        
    def read(self):
        """读取FY-3D/FY-3F HDF文件"""
        print(f"📡 读取 FY-3D/FY-3F 数据: {self.filepath}")
        
        with h5py.File(self.filepath, 'r') as f:
            # 打印文件结构
            print(f"   文件结构: {list(f.keys())}")
            
            # 读取亮温数据
            if 'Data/Earth_Obs_BT' in f:
                bt_dataset = f['Data/Earth_Obs_BT']
                bt_raw = bt_dataset[:]
                print(f"   ✓ 使用 Data/Earth_Obs_BT")
                
                # 🔥 关键修复：读取并应用缩放因子
                slope = 1.0
                intercept = 0.0
                
                if 'Slope' in bt_dataset.attrs:
                    slope = float(bt_dataset.attrs['Slope'])
                    print(f"   ✓ 读取 Slope: {slope}")
                
                if 'Intercept' in bt_dataset.attrs:
                    intercept = float(bt_dataset.attrs['Intercept'])
                    print(f"   ✓ 读取 Intercept: {intercept}")
                
                # 应用定标公式: BT_actual = BT_raw * Slope + Intercept
                bt = bt_raw.astype(np.float64) * slope + intercept
                print(f"   ✓ 应用定标: BT = BT_raw × {slope} + {intercept}")
                print(f"   定标后范围: [{bt.min():.2f}, {bt.max():.2f}] K")
                
            elif 'Brightness_Temperature' in f:
                bt = f['Brightness_Temperature'][:].astype(np.float64)
                print(f"   ✓ 使用 Brightness_Temperature (已定标)")
            else:
                raise ValueError(f"未找到亮温数据。可用数据集: {list(f.keys())}")
            
            # 读取经纬度
            if 'Geolocation/Latitude' in f:
                lat = f['Geolocation/Latitude'][:]
                lon = f['Geolocation/Longitude'][:]
                print(f"   ✓ 使用 Geolocation/Latitude 和 Geolocation/Longitude")
            elif 'Latitude' in f:
                lat = f['Latitude'][:]
                lon = f['Longitude'][:]
                print(f"   ✓ 使用 Latitude 和 Longitude")
            else:
                raise ValueError(f"未找到经纬度数据")
            
            # 🔥🔥🔥 新增修复代码开始 🔥🔥🔥
            # 检查是否需要转置 BT 从 (Channels, Lines, Pixels) -> (Lines, Pixels, Channels)
            # 依据：如果 BT 是 3D 且 Lat 是 2D，并且 BT 的后两维与 Lat 形状一致
            if bt.ndim == 3 and lat.ndim == 2:
                if bt.shape[1:] == lat.shape:
                    print(f"\n   🔄 维度修正: 检测到 (Channel, Line, Pixel) 格式 {bt.shape}")
                    print(f"      转置为 (Line, Pixel, Channel) 以匹配经纬度 {lat.shape}")
                    bt = np.transpose(bt, (1, 2, 0))
                    print(f"      转置后 BT shape: {bt.shape}")
            # 🔥🔥🔥 新增修复代码结束 🔥🔥🔥

            print(f"\n   📐 数据维度信息: BT shape: {bt.shape}, Lat shape: {lat.shape}, Lon shape: {lon.shape}")
            
            # 读取时间信息
            if 'Geolocation/Scnlin_daycnt' in f and 'Geolocation/Scnlin_mscnt' in f:
                day_cnt = f['Geolocation/Scnlin_daycnt'][:]
                ms_cnt = f['Geolocation/Scnlin_mscnt'][:]
                obs_time = self._parse_fy3f_time(day_cnt, ms_cnt)
                print(f"   ✓ 使用 Scnlin_daycnt/mscnt 解析时间: {obs_time}")
            elif 'Obs_Time' in f:
                obs_time = f['Obs_Time'][:]
            elif 'Time' in f:
                obs_time = f['Time'][:]
            else:
                obs_time = self._extract_time_from_filename()
                print(f"   ⚠ 从文件名提取时间: {obs_time}")
            
            # 读取质量标记
            if 'QA/Quality_Flag_Scnlin' in f:
                quality = f['QA/Quality_Flag_Scnlin'][:]
                print(f"   ✓ 使用 QA/Quality_Flag_Scnlin")
            elif 'Quality_Flag' in f:
                quality = f['Quality_Flag'][:]
            else:
                quality = np.ones(lat.shape[0], dtype=np.uint8)
                print(f"   ⚠ 未找到质量标记，创建默认值")
            
        self.data = {
            'bt': bt,
            'lat': lat,
            'lon': lon,
            'time': obs_time,
            'quality': quality,
            'shape': bt.shape
        }
        
        print(f"\n   📊 数据概览:")
        print(f"   亮温形状: {bt.shape}")
        if bt.ndim == 3:
            print(f"   扫描线数: {bt.shape[0]}")
            print(f"   扫描位置数: {bt.shape[1]}")
            print(f"   通道数: {bt.shape[2]}")
        print(f"   纬度范围: [{lat.min():.2f}, {lat.max():.2f}]")
        print(f"   经度范围: [{lon.min():.2f}, {lon.max():.2f}]")
        print(f"   亮温范围: [{bt.min():.2f}, {bt.max():.2f}] K")
        
        # 检查数据质量
        self._check_data_quality()
        
        return self.data
    
    def _check_data_quality(self):
        """检查数据质量和单位"""
        bt = self.data['bt']
        valid_bt = bt[(bt > 0) & ~np.isnan(bt)]
        
        if len(valid_bt) > 0:
            mean_val = np.mean(valid_bt)
            
            if 100 <= mean_val <= 400:
                print(f"   ✓ 数据范围正常（亮温单位是K）")
            elif mean_val > 1000:
                print(f"   ⚠️  警告: 数据可能未定标！平均值={mean_val:.1f}")
                print(f"   请检查是否需要应用缩放因子")
            else:
                print(f"   ⚠️  数据范围异常（平均值={mean_val:.1f}K）")
    
    def _parse_fy3f_time(self, day_cnt, ms_cnt):
        """解析FY-3F时间"""
        base_date = datetime(2000, 1, 1)
        if len(day_cnt) > 0:
            days = int(day_cnt[0])
            milliseconds = int(ms_cnt[0])
            obs_time = base_date + timedelta(days=days, milliseconds=milliseconds)
            return obs_time
        return self._extract_time_from_filename()
    
    def _extract_time_from_filename(self):
        """从文件名提取时间"""
        import re
        match = re.search(r'(\d{8})_(\d{4})', self.filepath)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            dt = datetime.strptime(date_str + time_str, '%Y%m%d%H%M')
            return dt
        return datetime(2021, 1, 1)
    
    def quality_control(self, bt_min=None, bt_max=None, auto_detect=True):
        """亮温数据质量控制"""
        print(f"\n🔍 执行质量控制...")
        
        bt = self.data['bt']
        
        # 自动检测合理范围
        if auto_detect and (bt_min is None or bt_max is None):
            valid_bt = bt[(bt > 0) & ~np.isnan(bt)]
            if len(valid_bt) > 0:
                p1 = np.percentile(valid_bt, 1)
                p99 = np.percentile(valid_bt, 99)
                mean_val = np.mean(valid_bt)
                
                print(f"   自动检测数据范围:")
                print(f"   1%分位: {p1:.2f}")
                print(f"   99%分位: {p99:.2f}")
                print(f"   平均值: {mean_val:.2f}")
                
                # 设置合理阈值
                if bt_min is None:
                    bt_min = max(50.0, p1 - 50)
                if bt_max is None:
                    bt_max = min(400.0, p99 + 50)
                
                print(f"   自动设置阈值: [{bt_min:.1f}, {bt_max:.1f}]")
        else:
            if bt_min is None:
                bt_min = 100.0
            if bt_max is None:
                bt_max = 400.0
        
        print(f"   亮温范围: [{bt_min}, {bt_max}] K")
        
        # 标记异常值
        invalid_mask = (bt < bt_min) | (bt > bt_max) | (bt <= 0) | np.isnan(bt)
        
        # 统计
        total_pixels = bt.size
        invalid_pixels = np.sum(invalid_mask)
        valid_ratio = (1 - invalid_pixels / total_pixels) * 100
        
        print(f"   总像元数: {total_pixels:,}")
        print(f"   异常像元数: {invalid_pixels:,}")
        print(f"   有效率: {valid_ratio:.2f}%")
        
        if valid_ratio < 10:
            print(f"   ⚠️  警告: 有效数据比例过低 ({valid_ratio:.2f}%)")
        
        # 设置为NaN
        bt_qc = bt.copy()
        bt_qc[invalid_mask] = np.nan
        
        self.data['bt_qc'] = bt_qc
        self.data['valid_mask'] = ~invalid_mask
        
        return bt_qc
    
    def flatten_to_points(self):
        """将数据展平为点云格式（修复维度问题）"""
        print(f"\n📍 展平为点云...")
        
        bt = self.data.get('bt_qc', self.data['bt'])
        lat = self.data['lat']
        lon = self.data['lon']
        
        print(f"   BT shape: {bt.shape}")
        print(f"   Lat shape: {lat.shape}")
        print(f"   Lon shape: {lon.shape}")
        
        # 🔥 关键修复：处理不同的维度情况
        if bt.ndim == 3:
            n_scan, n_pos, n_channel = bt.shape
            bt_flat = bt.reshape(-1, n_channel)  # (n_scan * n_pos, n_channel)
            
            # 处理lat/lon维度
            if lat.ndim == 2:
                # 如果lat/lon是2D的 (n_scan, n_pos)
                if lat.shape == (n_scan, n_pos):
                    lat_flat = lat.flatten()
                    lon_flat = lon.flatten()
                    print(f"   ✓ Lat/Lon是2D，形状匹配BT的前两维")
                else:
                    raise ValueError(f"Lat/Lon shape {lat.shape} 与 BT前两维 ({n_scan}, {n_pos}) 不匹配")
            elif lat.ndim == 3:
                # 如果lat/lon也是3D的，取第一个通道或平均
                if lat.shape == bt.shape:
                    lat_flat = lat[:, :, 0].flatten()
                    lon_flat = lon[:, :, 0].flatten()
                    print(f"   ⚠ Lat/Lon是3D，使用第一个通道")
                else:
                    raise ValueError(f"Lat/Lon 3D shape {lat.shape} 与 BT shape {bt.shape} 不匹配")
            elif lat.ndim == 1:
                # 如果lat/lon是1D的，需要广播
                if len(lat) == n_scan:
                    lat_2d = np.repeat(lat[:, np.newaxis], n_pos, axis=1)
                    lon_2d = np.repeat(lon[:, np.newaxis], n_pos, axis=1)
                    lat_flat = lat_2d.flatten()
                    lon_flat = lon_2d.flatten()
                    print(f"   ⚠ Lat/Lon是1D，进行广播")
                else:
                    raise ValueError(f"Lat/Lon 1D length {len(lat)} 与 n_scan {n_scan} 不匹配")
            else:
                raise ValueError(f"不支持的Lat/Lon维度: {lat.ndim}D")
            
        else:
            raise ValueError(f"不支持的BT维度: {bt.ndim}D，预期是3D (scan, pos, channel)")
        
        print(f"   展平后:")
        print(f"   BT: {bt_flat.shape}")
        print(f"   Lat: {lat_flat.shape}")
        print(f"   Lon: {lon_flat.shape}")
        
        # 确保维度匹配
        if len(lat_flat) != bt_flat.shape[0]:
            raise ValueError(f"维度不匹配！BT有{bt_flat.shape[0]}个点，但Lat有{len(lat_flat)}个点")
        
        # 处理时间
        if isinstance(self.data['time'], datetime):
            time_flat = np.full(len(lat_flat), self.data['time'])
        else:
            time_flat = np.repeat(self.data['time'], n_pos)
        
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
    """ERA5再分析数据读取器"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.spatial_tree = None
    
    def read(self):
        """读取ERA5 GRIB文件"""
        print(f"\n🌍 读取 ERA5 数据: {self.filepath}")
        
        try:
            ds = xr.open_dataset(
                self.filepath,
                engine='cfgrib',
                backend_kwargs={'indexpath': ''}
            )
        except Exception as e:
            print(f"   ⚠ 警告: 使用cfgrib读取失败: {e}")
            print(f"   尝试使用默认引擎...")
            ds = xr.open_dataset(self.filepath)
        
        # 提取温度数据
        if 't' in ds:
            temp = ds['t']
        elif 'temperature' in ds:
            temp = ds['temperature']
        else:
            raise ValueError(f"未找到温度变量。可用变量: {list(ds.keys())}")
        
        # 提取气压层
        if 'isobaricInhPa' in ds.coords:
            pressure = ds['isobaricInhPa'].values
        elif 'level' in ds.coords:
            pressure = ds['level'].values
        elif 'pressure' in ds.coords:
            pressure = ds['pressure'].values
        else:
            print(f"   ⚠ 未找到气压层信息")
            pressure = None
        
        # 提取经纬度
        if 'latitude' in ds.coords:
            lat = ds['latitude'].values
            lon = ds['longitude'].values
        elif 'lat' in ds.coords:
            lat = ds['lat'].values
            lon = ds['lon'].values
        else:
            raise ValueError("未找到经纬度信息")
        
        self.data = {
            'temperature': temp,
            'pressure': pressure,
            'latitude': lat,
            'longitude': lon,
            'dataset': ds
        }
        
        print(f"   温度变量形状: {temp.shape}")
        print(f"   气压层数: {len(pressure) if pressure is not None else 'N/A'}")
        print(f"   纬度范围: [{lat.min():.2f}, {lat.max():.2f}]")
        print(f"   经度范围: [{lon.min():.2f}, {lon.max():.2f}]")
        
        if pressure is not None:
            print(f"   气压范围: [{pressure.min():.0f}, {pressure.max():.0f}] hPa")
        
        return self.data
    
    def build_spatial_tree(self):
        """构建KD树用于空间匹配"""
        print(f"\n🌲 构建空间索引...")
        
        lat = self.data['latitude']
        lon = self.data['longitude']
        
        # 创建网格点
        if lat.ndim == 1 and lon.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
        else:
            lat_grid = lat
            lon_grid = lon
        
        # 展平为点列表
        points = np.column_stack([lat_grid.flatten(), lon_grid.flatten()])
        
        # 构建KD树
        tree = cKDTree(points)
        
        self.spatial_tree = {
            'tree': tree,
            'shape': lat_grid.shape,
            'points': points
        }
        
        print(f"   网格点数: {len(points):,}")
        
        return tree
    
    def interpolate_to_satellite(self, sat_points, max_distance=0.5):
        """空间匹配：插值ERA5到卫星观测点"""
        print(f"\n🎯 空间匹配...")
        
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
        
        if n_valid > 0:
            print(f"   平均距离: {np.mean(distances[valid_match]):.3f}°")
        
        # 提取对应的ERA5格点索引
        lat_shape, lon_shape = self.spatial_tree['shape']
        era5_lat_idx = indices // lon_shape
        era5_lon_idx = indices % lon_shape
        
        # 提取温度廓线
        temp_data = self.data['temperature'].values
        if temp_data.ndim == 4:  # (time, level, lat, lon)
            temp_data = temp_data[0]
        
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
                bt_min=None, bt_max=None,
                max_distance=0.5,
                auto_detect=True):
        """主处理流程"""
        
        print("="*70)
        print("卫星资料同化数据配准工具 (修复版)")
        print("="*70)
        
        # 1. 读取FY-3D数据
        self.fy3d_reader = FY3D_Reader(fy3d_file)
        self.fy3d_reader.read()
        self.fy3d_reader.quality_control(bt_min, bt_max, auto_detect=auto_detect)
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
        
        if len(X_clean) == 0:
            print("\n   ⚠️  警告: 没有有效样本!")
            print("   可能的原因:")
            print("   1. 质量控制过于严格")
            print("   2. FY-3D与ERA5数据的时空覆盖不匹配")
            return X_clean, Y_clean
        
        # 统计信息
        print(f"\n📈 数据统计:")
        print(f"   亮温范围: [{np.nanmin(X_clean):.2f}, {np.nanmax(X_clean):.2f}] K")
        print(f"   温度范围: [{np.nanmin(Y_clean):.2f}, {np.nanmax(Y_clean):.2f}] K")
        print(f"   亮温均值: {np.nanmean(X_clean):.2f} K")
        print(f"   温度均值: {np.nanmean(Y_clean):.2f} K")
        
        self.X = X_clean
        self.Y = Y_clean
        self.lat = sat_points['lat'][valid_samples]
        self.lon = sat_points['lon'][valid_samples]
        self.pressure_levels = self.era5_reader.data['pressure']
        
        return X_clean, Y_clean
    
    def save_numpy(self, output_prefix='collocation_data'):
        """保存为Numpy格式"""
        print(f"\n💾 保存数据...")
        
        np.save(f'{output_prefix}_X.npy', self.X)
        np.save(f'{output_prefix}_Y.npy', self.Y)
        np.save(f'{output_prefix}_pressure.npy', self.pressure_levels)
        np.save(f'{output_prefix}_lat.npy', self.lat)
        np.save(f'{output_prefix}_lon.npy', self.lon)
        
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
        """保存为HDF5格式"""
        print(f"\n💾 保存为HDF5格式...")
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('X_brightness_temperature', data=self.X, 
                           compression='gzip', compression_opts=4)
            f.create_dataset('Y_temperature_profile', data=self.Y,
                           compression='gzip', compression_opts=4)
            f.create_dataset('pressure_levels', data=self.pressure_levels)
            
            f.attrs['n_samples'] = len(self.X)
            f.attrs['n_channels'] = self.X.shape[1]
            f.attrs['n_levels'] = self.Y.shape[1]
            f.attrs['description'] = 'FY-3D MWTS and ERA5 collocated data'
        
        print(f"   ✓ {output_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='FY-3D/FY-3F 和 ERA5 数据配准工具 (修复版)'
    )
    parser.add_argument('fy3d_file', help='FY-3D/FY-3F文件路径')
    parser.add_argument('era5_file', help='ERA5 GRIB文件路径')
    parser.add_argument('-o', '--output', default='collocation_data',
                       help='输出文件前缀')
    parser.add_argument('--bt-min', type=float, default=None,
                       help='最小亮温 (K)，留空则自动检测')
    parser.add_argument('--bt-max', type=float, default=None,
                       help='最大亮温 (K)，留空则自动检测')
    parser.add_argument('--no-auto-detect', action='store_true',
                       help='禁用自动检测，使用默认阈值')
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
        max_distance=args.max_distance,
        auto_detect=(not args.no_auto_detect)
    )
    
    # 保存结果
    if len(X) > 0:
        if args.format in ['numpy', 'both']:
            collocation.save_numpy(args.output)
        
        if args.format in ['hdf5', 'both']:
            collocation.save_hdf5(f'{args.output}.h5')
        
        print("\n" + "="*70)
        print("✅ 处理完成!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ 处理失败: 没有生成有效样本")
        print("="*70)


if __name__ == '__main__':
    main()
