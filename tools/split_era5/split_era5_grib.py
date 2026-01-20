#!/usr/bin/env python3
"""
ERA5 GRIB文件按时间分割脚本
将大的GRIB文件按照年-月-日-时间分割成多个小文件
"""

import xarray as xr
import os
from pathlib import Path
from datetime import datetime
import argparse


def split_grib_by_time(input_file, output_dir, file_format='grib'):
    """
    将GRIB文件按照时间维度分割
    
    Parameters:
    -----------
    input_file : str
        输入的GRIB文件路径
    output_dir : str
        输出目录
    file_format : str
        输出格式，'grib' 或 'netcdf'
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"正在读取文件: {input_file}")
    print("这可能需要一些时间...")
    
    # 读取GRIB文件
    # 使用chunks参数可以避免一次性加载所有数据到内存
    ds = xr.open_dataset(
        input_file, 
        engine='cfgrib',
        chunks={'time': 1}  # 每次只加载一个时间步
    )
    
    print(f"数据集信息:")
    print(f"  变量: {list(ds.data_vars)}")
    print(f"  时间范围: {ds.time.values[0]} 到 {ds.time.values[-1]}")
    print(f"  时间步数: {len(ds.time)}")
    print(f"  气压层数: {len(ds.isobaricInhPa)}")
    
    # 获取所有时间点
    times = ds.time.values
    total = len(times)
    
    print(f"\n开始分割，共 {total} 个时间步...")
    
    for idx, time in enumerate(times, 1):
        # 选择单个时间步的数据
        ds_single = ds.sel(time=time)
        
        # 转换时间为datetime对象
        time_dt = datetime.utcfromtimestamp(time.astype('datetime64[s]').astype('int'))
        
        # 构建输出文件名: YYYY/MM/era5_YYYYMMDD_HH.grib
        year = time_dt.strftime('%Y')
        month = time_dt.strftime('%m')
        filename = time_dt.strftime('era5_%Y%m%d_%H.grib')
        
        # 创建年/月子目录
        year_month_dir = output_path / year / month
        year_month_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = year_month_dir / filename
        
        # 保存文件
        if file_format == 'grib':
            # 保存为GRIB格式
            ds_single.to_netcdf(
                output_file,
                engine='cfgrib',
                mode='w'
            )
        elif file_format == 'netcdf':
            # 保存为NetCDF格式（更快，但格式不同）
            output_file = output_file.with_suffix('.nc')
            ds_single.to_netcdf(output_file)
        
        print(f"[{idx}/{total}] 已保存: {output_file}")
    
    ds.close()
    print(f"\n完成! 所有文件已保存到: {output_dir}")


def split_grib_by_time_alternative(input_file, output_dir):
    """
    使用pygrib库的替代方案（如果cfgrib有问题）
    这个方法直接操作GRIB消息，不依赖xarray
    """
    try:
        import pygrib
    except ImportError:
        print("错误: 需要安装 pygrib 库")
        print("请运行: pip install pygrib")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"正在读取文件: {input_file}")
    
    # 打开GRIB文件
    grbs = pygrib.open(input_file)
    
    # 用于跟踪当前打开的输出文件
    current_time = None
    current_grb_out = None
    messages_by_time = {}
    
    print("正在处理GRIB消息...")
    
    # 遍历所有GRIB消息
    for i, grb in enumerate(grbs, 1):
        if i % 100 == 0:
            print(f"已处理 {i} 条消息...")
        
        # 获取时间信息
        date = grb.validDate
        
        # 创建时间键
        time_key = date.strftime('%Y%m%d%H')
        
        # 将消息添加到对应时间的列表
        if time_key not in messages_by_time:
            messages_by_time[time_key] = []
        messages_by_time[time_key].append(grb)
    
    grbs.close()
    
    print(f"\n找到 {len(messages_by_time)} 个不同的时间步")
    print("开始写入分割文件...")
    
    # 写入分割后的文件
    for idx, (time_key, messages) in enumerate(messages_by_time.items(), 1):
        # 解析时间
        date = datetime.strptime(time_key, '%Y%m%d%H')
        
        # 构建输出路径
        year = date.strftime('%Y')
        month = date.strftime('%m')
        filename = date.strftime('era5_%Y%m%d_%H.grib')
        
        year_month_dir = output_path / year / month
        year_month_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = year_month_dir / filename
        
        # 写入GRIB文件
        with open(output_file, 'wb') as f:
            for msg in messages:
                f.write(msg.tostring())
        
        print(f"[{idx}/{len(messages_by_time)}] 已保存: {output_file} ({len(messages)} 条消息)")
    
    print(f"\n完成! 所有文件已保存到: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将ERA5 GRIB文件按时间分割')
    parser.add_argument('input_file',default="/data1/lrx/obs", help='输入的GRIB文件路径')
    parser.add_argument('-o', '--output', default='./split_data', 
                       help='输出目录 (默认: ./split_data)')
    parser.add_argument('-f', '--format', choices=['grib', 'netcdf'], 
                       default='grib', help='输出格式 (默认: grib)')
    parser.add_argument('-m', '--method', choices=['xarray', 'pygrib'],
                       default='xarray', help='处理方法 (默认: xarray)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 文件不存在: {args.input_file}")
        exit(1)
    
    print(f"输入文件: {args.input_file}")
    print(f"输出目录: {args.output}")
    print(f"输出格式: {args.format}")
    print(f"处理方法: {args.method}")
    print("-" * 50)
    
    try:
        if args.method == 'xarray':
            split_grib_by_time(args.input_file, args.output, args.format)
        else:
            split_grib_by_time_alternative(args.input_file, args.output)
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("\n如果使用xarray方法失败，可以尝试pygrib方法:")
        print(f"python split_era5_grib.py {args.input_file} -m pygrib")
