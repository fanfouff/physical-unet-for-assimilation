#!/usr/bin/env python3
"""
卫星资料同化数据匹配工具（最终修复版）
FY-3D MWTS L1 与 ERA5 Reanalysis 数据配准

修正内容：
- 使用正确的HDF文件结构：Data/Earth_Obs_BT, Geolocation/Latitude, Geolocation/Longitude
- 修复数据类型问题：自动转换为float64以支持NaN
- 智能质量控制：自动检测数据范围和单位
- 支持FY-3F MWTS格式（98通道）和原始格式（13通道）
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
    """FY-3D/FY-3F MWTS 数据读取器（最终修复版）"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        
    def read(self):
        """读取FY-3D/FY-3F HDF文件"""
        print(f"📡 读取 FY-3D/FY-3F 数据: {self.filepath}")
        
        with h5py.File(self.filepath, 'r') as f:
            # 打印文件结构以便调试
            print(f"   文件结构: {list(f.keys())}")
            
            # 尝试读取亮温数据 - 使用新的路径
            if 'Data/Earth_Obs_BT' in f:
                bt = f['Data/Earth_Obs_BT'][:]
                print(f"   ✓ 使用 Data/Earth_Obs_BT")
            elif 'Brightness_Temperature' in f:
                bt = f['Brightness_Temperature'][:]
                print(f"   ✓ 使用 Brightness_Temperature")
            else:
                raise ValueError(f"未找到亮温数据。可用数据集: {list(f.keys())}")
            
            # ⚠️ 重要：转换为float64以支持NaN
            bt = bt.astype(np.float64)
            
            # 读取经纬度 - 使用新的路径
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
            
            # 读取时间信息
            if 'Geolocation/Scnlin_daycnt' in f and 'Geolocation/Scnlin_mscnt' in f:
                # FY-3F格式：使用扫描线的日计数和毫秒计数
                day_cnt = f['Geolocation/Scnlin_daycnt'][:]
                ms_cnt = f['Geolocation/Scnlin_mscnt'][:]
                obs_time = self._parse_fy3f_time(day_cnt, ms_cnt)
                print(f"   ✓ 使用 Scnlin_daycnt/mscnt 解析时间")
            elif 'Obs_Time' in f:
                obs_time = f['Obs_Time'][:]
            elif 'Time' in f:
                obs_time = f['Time'][:]
            else:
                # 从文件名提取时间
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
        
        print(f"   亮温形状: {bt.shape}")
        print(f"   通道数: {bt.shape[-1]}")
        print(f"   扫描线数: {bt.shape[0]}")
        print(f"   扫描角度数: {bt.shape[1]}")
        print(f"   纬度范围: [{lat.min():.2f}, {lat.max():.2f}]")
        print(f"   经度范围: [{lon.min():.2f}, {lon.max():.2f}]")
        
        # 数据质量初步检查
        self._check_data_quality()
        
        return self.data
    
    def _check_data_quality(self):
        """初步检查数据质量和单位"""
        bt = self.data['bt']
        
        # 移除零值和负值进行统计
        valid_bt = bt[(bt > 0) & ~np.isnan(bt)]
        
        if len(valid_bt) > 0:
            min_val = np.min(valid_bt)
            max_val = np.max(valid_bt)
            mean_val = np.mean(valid_bt)
            
            print(f"\n   数据范围检查:")
            print(f"   最小值: {min_val:.2f}")
            print(f"   最大值: {max_val:.2f}")
            print(f"   平均值: {mean_val:.2f}")
            
            # 推断单位
            if 0 < mean_val < 100:
                print(f"   ⚠️  警告: 数据范围异常（平均值{mean_val:.1f}）")
                print(f"   可能需要单位转换或缩放")
            elif 100 <= mean_val <= 400:
                print(f"   ✓ 数据范围正常（亮温单位似乎是K）")
            else:
                print(f"   ⚠️  警告: 数据范围可能不是标准亮温值")
    
    def _parse_fy3f_time(self, day_cnt, ms_cnt):
        """
        解析FY-3F的时间格式
        day_cnt: 从某个基准日期开始的天数
        ms_cnt: 一天内的毫秒数
        """
        # FY-3F的时间基准通常是1958-01-01或类似日期
        base_date = datetime(1958, 1, 1)
        
        if len(day_cnt) > 0:
            # 使用第一个扫描线的时间作为代表
            days = int(day_cnt[0])
            milliseconds = int(ms_cnt[0])
            
            obs_time = base_date + timedelta(days=days, milliseconds=milliseconds)
            return obs_time
        
        return self._extract_time_from_filename()
    
    def _extract_time_from_filename(self):
        """从文件名提取观测时间"""
        import re
        # FY3F_MWTS-_ORBD_L1_20250111_1957_033KM_V0.HDF
        match = re.search(r'(\d{8})_(\d{4})', self.filepath)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            dt = datetime.strptime(date_str + time_str, '%Y%m%d%H%M')
            return dt
        return datetime(2021, 1, 1)  # 默认值
    
    def quality_control(self, bt_min=None, bt_max=None, auto_detect=True):
        """
        亮温数据质量控制（智能版本）
        
        Parameters:
        -----------
        bt_min : float, optional
            最小合理亮温 (K)。如果为None且auto_detect=True，自动检测
        bt_max : float, optional
            最大合理亮温 (K)。如果为None且auto_detect=True，自动检测
        auto_detect : bool
            是否自动检测合理范围
        """
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
                
                # 根据数据范围推断合理的阈值
                if bt_min is None:
                    if p1 > 50:  # 看起来像开尔文
                        bt_min = max(50.0, p1 - 50)
                    else:
                        bt_min = max(-100.0, p1 - 20)
                
                if bt_max is None:
                    if p99 < 500:
                        bt_max = min(500.0, p99 + 50)
                    else:
                        bt_max = p99 + 100
                
                print(f"   自动设置阈值: [{bt_min:.1f}, {bt_max:.1f}]")
        else:
            if bt_min is None:
                bt_min = 100.0
            if bt_max is None:
                bt_max = 400.0
        
        print(f"   亮温范围: [{bt_min}, {bt_max}] K")
        
        # 标记异常值（包括零值和负值）
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
            print(f"   建议检查:")
            print(f"   1. 数据单位是否正确")
            print(f"   2. 使用 --auto-detect 自动检测范围")
            print(f"   3. 或手动调整 --bt-min 和 --bt-max 参数")
        
        # 设置为NaN（现在bt已经是float64类型）
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
        
        if n_valid > 0:
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
                bt_min=None, bt_max=None,
                max_distance=0.5,
                auto_detect=True):
        """
        主处理流程
        
        Parameters:
        -----------
        fy3d_file : str
            FY-3D文件路径
        era5_file : str
            ERA5文件路径
        bt_min, bt_max : float
            亮温质量控制范围（None则自动检测）
        max_distance : float
            最大匹配距离
        auto_detect : bool
            是否自动检测亮温范围
        
        Returns:
        --------
        X : ndarray (N, n_channels)
            输入特征（卫星亮温）
        Y : ndarray (N, n_levels)
            标签（温度廓线）
        """
        
        print("="*70)
        print("卫星资料同化数据配准工具 (最终修复版)")
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
            print("   3. 数据格式或单位问题")
            return X_clean, Y_clean
        
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
        description='FY-3D/FY-3F 和 ERA5 数据配准工具 (最终修复版)'
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
