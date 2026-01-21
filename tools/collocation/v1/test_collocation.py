#!/usr/bin/env python3
"""
测试脚本：验证工具是否正常工作
可以使用模拟数据进行测试
"""

import numpy as np
import h5py
import xarray as xr
from pathlib import Path
import sys


def create_mock_fy3d_data(filepath='test_fy3d.L1c'):
    """
    创建模拟的FY-3D数据用于测试
    """
    print("📝 创建模拟FY-3D数据...")
    
    # 模拟参数
    n_scan = 100
    n_angle = 90
    n_channel = 13
    
    # 生成模拟数据
    lat = np.linspace(20, 50, n_scan * n_angle).reshape(n_scan, n_angle)
    lon = np.linspace(100, 130, n_scan * n_angle).reshape(n_scan, n_angle)
    
    # 模拟亮温数据（基于真实范围）
    bt = np.random.uniform(180, 280, (n_scan, n_angle, n_channel))
    
    # 添加一些异常值用于测试QC
    bt[::10, ::10, :] = np.random.uniform(50, 80, (10, 9, n_channel))  # 低值
    bt[5::10, 5::10, :] = np.random.uniform(420, 500, (10, 9, n_channel))  # 高值
    
    # 保存为H5文件
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('Brightness_Temperature', data=bt)
        f.create_dataset('Latitude', data=lat)
        f.create_dataset('Longitude', data=lon)
        f.create_dataset('Quality_Flag', data=np.ones((n_scan, n_angle), dtype=np.uint8))
    
    print(f"   ✓ 已创建: {filepath}")
    print(f"   形状: {bt.shape}")
    print(f"   纬度范围: [{lat.min():.2f}, {lat.max():.2f}]")
    print(f"   经度范围: [{lon.min():.2f}, {lon.max():.2f}]")
    
    return filepath


def create_mock_era5_data(filepath='test_era5.grib', lat_range=(20, 50), lon_range=(100, 130)):
    """
    创建模拟的ERA5数据用于测试
    注意：这需要xarray和cfgrib支持
    """
    print("\n📝 创建模拟ERA5数据...")
    
    try:
        # 模拟参数
        lat = np.arange(lat_range[0], lat_range[1] + 0.25, 0.25)
        lon = np.arange(lon_range[0], lon_range[1] + 0.25, 0.25)
        pressure = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
                           225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
                           775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])
        
        # 创建温度场（基于简单的垂直廓线）
        temp = np.zeros((len(pressure), len(lat), len(lon)))
        for i, p in enumerate(pressure):
            # 简单的温度-气压关系
            temp[i] = 200 + 80 * np.log(1000 / p) + np.random.randn(len(lat), len(lon)) * 2
        
        # 创建xarray Dataset
        ds = xr.Dataset(
            {
                't': (['isobaricInhPa', 'latitude', 'longitude'], temp)
            },
            coords={
                'isobaricInhPa': pressure,
                'latitude': lat,
                'longitude': lon,
                'time': np.datetime64('2021-07-01T12:00:00')
            }
        )
        
        # 尝试保存为NetCDF（更容易）而不是GRIB
        nc_filepath = filepath.replace('.grib', '.nc')
        ds.to_netcdf(nc_filepath)
        
        print(f"   ✓ 已创建: {nc_filepath}")
        print(f"   温度形状: {temp.shape}")
        print(f"   气压层: {len(pressure)} 层")
        print(f"   ⚠️  注意: 保存为NetCDF格式（测试用）")
        
        return nc_filepath
        
    except Exception as e:
        print(f"   ❌ 创建ERA5数据失败: {e}")
        print("   提示: 可以使用真实的ERA5文件进行测试")
        return None


def test_basic_workflow():
    """
    测试基本工作流程
    """
    print("\n" + "="*70)
    print("测试基本工作流程")
    print("="*70)
    
    # 创建模拟数据
    fy3d_file = create_mock_fy3d_data()
    era5_file = create_mock_era5_data()
    
    if not era5_file:
        print("\n⚠️  无法创建ERA5测试数据，跳过配准测试")
        print("请使用真实的ERA5文件进行完整测试")
        return
    
    # 测试FY-3D读取
    print("\n" + "-"*70)
    print("测试 FY-3D 读取...")
    print("-"*70)
    
    try:
        from collocation_fy3d_era5 import FY3D_Reader
        
        fy3d_reader = FY3D_Reader(fy3d_file)
        data = fy3d_reader.read()
        
        print(f"✓ 读取成功")
        print(f"  亮温形状: {data['bt'].shape}")
        
        # 测试质量控制
        fy3d_reader.quality_control(bt_min=100, bt_max=400)
        print(f"✓ 质量控制完成")
        
        # 测试展平
        points = fy3d_reader.flatten_to_points()
        print(f"✓ 展平完成，有效点数: {points['n_points']}")
        
    except Exception as e:
        print(f"❌ FY-3D测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试ERA5读取
    print("\n" + "-"*70)
    print("测试 ERA5 读取...")
    print("-"*70)
    
    try:
        from collocation_fy3d_era5 import ERA5_Reader
        
        era5_reader = ERA5_Reader(era5_file)
        data = era5_reader.read()
        
        print(f"✓ 读取成功")
        print(f"  温度形状: {data['temperature'].shape}")
        
        # 构建空间树
        era5_reader.build_spatial_tree()
        print(f"✓ 空间索引构建完成")
        
    except Exception as e:
        print(f"❌ ERA5测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试完整配准
    print("\n" + "-"*70)
    print("测试完整配准流程...")
    print("-"*70)
    
    try:
        from collocation_fy3d_era5 import DataCollocation
        
        collocation = DataCollocation()
        X, Y = collocation.process(fy3d_file, era5_file)
        
        print(f"✓ 配准成功")
        print(f"  X shape: {X.shape}")
        print(f"  Y shape: {Y.shape}")
        
        # 测试保存
        collocation.save_numpy('test_output')
        print(f"✓ 数据保存成功")
        
    except Exception as e:
        print(f"❌ 配准测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理
    print("\n" + "-"*70)
    print("清理测试文件...")
    print("-"*70)
    
    import os
    test_files = [
        'test_fy3d.L1c',
        'test_era5.nc',
        'test_output_X.npy',
        'test_output_Y.npy',
        'test_output_pressure.npy',
        'test_output_metadata.json'
    ]
    
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"  ✓ 删除: {f}")


def check_dependencies():
    """
    检查必要的依赖包
    """
    print("="*70)
    print("检查依赖包")
    print("="*70)
    
    required_packages = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'h5py': 'h5py',
        'xarray': 'xarray',
    }
    
    optional_packages = {
        'cfgrib': 'cfgrib',
        'pyresample': 'pyresample',
        'matplotlib': 'matplotlib',
        'cartopy': 'cartopy'
    }
    
    all_ok = True
    
    print("\n必需包:")
    for name, module in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (未安装)")
            all_ok = False
    
    print("\n可选包:")
    for name, module in optional_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  - {name} (未安装)")
    
    if not all_ok:
        print("\n⚠️  警告: 某些必需包未安装")
        print("请运行: pip install -r collocation_requirements.txt")
    else:
        print("\n✅ 所有必需包已安装")
    
    return all_ok


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试配准工具')
    parser.add_argument('--check-deps', action='store_true',
                       help='仅检查依赖')
    parser.add_argument('--create-mock', action='store_true',
                       help='仅创建模拟数据')
    
    args = parser.parse_args()
    
    if args.check_deps:
        check_dependencies()
        return
    
    if args.create_mock:
        create_mock_fy3d_data()
        create_mock_era5_data()
        print("\n✅ 模拟数据已创建")
        return
    
    # 完整测试
    deps_ok = check_dependencies()
    
    if deps_ok:
        test_basic_workflow()
        print("\n" + "="*70)
        print("✅ 所有测试完成!")
        print("="*70)
    else:
        print("\n❌ 请先安装必需的依赖包")
        sys.exit(1)


if __name__ == '__main__':
    main()
