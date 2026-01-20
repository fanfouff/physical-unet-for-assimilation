#!/usr/bin/env python3
"""
ERA5分割文件验证和统计工具
检查分割后的文件完整性和统计信息
"""

import pygrib
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def verify_split_files(directory='/data1/lrx/era5/era5_split'):
    """
    验证分割后的文件并生成统计报告
    """
    
    output_dir = Path(directory)
    
    if not output_dir.exists():
        print(f"❌ 错误: 目录不存在: {directory}")
        return
    
    print(f"📂 检查目录: {directory}\n")
    
    # 查找所有GRIB文件
    grib_files = list(output_dir.rglob('*.grib'))
    nc_files = list(output_dir.rglob('*.nc'))
    
    if not grib_files and not nc_files:
        print("❌ 未找到任何GRIB或NetCDF文件")
        return
    
    files = grib_files + nc_files
    print(f"✅ 找到 {len(files)} 个文件")
    print(f"   - GRIB文件: {len(grib_files)}")
    print(f"   - NetCDF文件: {len(nc_files)}\n")
    
    # 统计信息
    stats = {
        'total_files': len(files),
        'total_size': 0,
        'by_year': defaultdict(int),
        'by_month': defaultdict(int),
        'by_hour': defaultdict(int),
        'file_sizes': [],
        'errors': []
    }
    
    # 检查每个文件
    print("🔍 验证文件...\n")
    
    for idx, file_path in enumerate(sorted(files), 1):
        try:
            # 获取文件大小
            file_size = file_path.stat().st_size
            stats['total_size'] += file_size
            stats['file_sizes'].append(file_size)
            
            # 从文件名提取信息
            filename = file_path.stem  # era5_YYYYMMDD_HH
            if filename.startswith('era5_'):
                parts = filename.split('_')
                if len(parts) >= 3:
                    date_str = parts[1]  # YYYYMMDD
                    hour_str = parts[2]  # HH
                    
                    year = date_str[:4]
                    month = date_str[4:6]
                    
                    stats['by_year'][year] += 1
                    stats['by_month'][f"{year}-{month}"] += 1
                    stats['by_hour'][hour_str] += 1
            
            # 验证GRIB文件内容
            if file_path.suffix == '.grib':
                grbs = pygrib.open(str(file_path))
                num_messages = grbs.messages
                grbs.close()
                
                if idx <= 5 or idx % 50 == 0:  # 只显示前5个和每50个
                    print(f"[{idx:3d}/{len(files)}] ✓ {file_path.name} "
                          f"({file_size/(1024*1024):.1f} MB, {num_messages} 层)")
            
        except Exception as e:
            error_msg = f"{file_path.name}: {str(e)}"
            stats['errors'].append(error_msg)
            print(f"[{idx:3d}/{len(files)}] ❌ {error_msg}")
    
    # 打印统计报告
    print("\n" + "="*60)
    print("📊 统计报告")
    print("="*60)
    
    print(f"\n📁 文件统计:")
    print(f"   总文件数: {stats['total_files']}")
    print(f"   总大小: {stats['total_size']/(1024**3):.2f} GB")
    print(f"   平均文件大小: {stats['total_size']/stats['total_files']/(1024**2):.1f} MB")
    
    if stats['file_sizes']:
        import statistics
        print(f"   最大文件: {max(stats['file_sizes'])/(1024**2):.1f} MB")
        print(f"   最小文件: {min(stats['file_sizes'])/(1024**2):.1f} MB")
        print(f"   中位数: {statistics.median(stats['file_sizes'])/(1024**2):.1f} MB")
    
    print(f"\n📅 按年份分布:")
    for year in sorted(stats['by_year'].keys()):
        print(f"   {year}: {stats['by_year'][year]} 个文件")
    
    print(f"\n📆 按月份分布:")
    for month in sorted(stats['by_month'].keys())[:12]:  # 只显示前12个月
        print(f"   {month}: {stats['by_month'][month]} 个文件")
    if len(stats['by_month']) > 12:
        print(f"   ... 还有 {len(stats['by_month']) - 12} 个月")
    
    print(f"\n🕐 按时间点分布:")
    for hour in sorted(stats['by_hour'].keys()):
        print(f"   {hour}:00: {stats['by_hour'][hour]} 个文件")
    
    if stats['errors']:
        print(f"\n⚠️  发现 {len(stats['errors'])} 个错误:")
        for error in stats['errors'][:10]:  # 只显示前10个错误
            print(f"   - {error}")
        if len(stats['errors']) > 10:
            print(f"   ... 还有 {len(stats['errors']) - 10} 个错误")
    else:
        print(f"\n✅ 所有文件验证通过!")
    
    # 生成预期文件列表
    print(f"\n📋 完整性检查:")
    expected_files = calculate_expected_files(stats)
    if expected_files > 0:
        completeness = (stats['total_files'] / expected_files) * 100
        print(f"   预期文件数: {expected_files}")
        print(f"   实际文件数: {stats['total_files']}")
        print(f"   完整性: {completeness:.1f}%")
        
        if completeness < 100:
            print(f"   ⚠️  缺少 {expected_files - stats['total_files']} 个文件")


def calculate_expected_files(stats):
    """
    根据统计信息计算预期的文件数量
    假设: 2年 × 12月 × 平均30天 × 2时间点 = 1440
    """
    # 从实际数据推断
    num_years = len(stats['by_year'])
    num_months = len(stats['by_month'])
    num_hours = len(stats['by_hour'])
    
    if num_years == 2 and num_hours == 2:
        # 2年，每天2个时间点
        return 2 * 365 * 2  # 约1460个文件
    
    return 0  # 无法计算


def list_missing_dates(directory='./split_data', year=2021):
    """
    列出指定年份缺失的日期
    """
    from datetime import timedelta
    
    output_dir = Path(directory)
    
    # 生成该年的所有日期
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    all_dates = []
    current = start_date
    while current <= end_date:
        for hour in ['00', '12']:
            all_dates.append(current.strftime(f'%Y%m%d_{hour}'))
        current += timedelta(days=1)
    
    # 查找实际存在的文件
    existing_files = set()
    for file_path in output_dir.rglob('*.grib'):
        filename = file_path.stem
        if filename.startswith('era5_'):
            parts = filename.split('_')
            if len(parts) >= 3:
                date_hour = f"{parts[1]}_{parts[2]}"
                existing_files.add(date_hour)
    
    # 找出缺失的
    missing = [d for d in all_dates if d not in existing_files]
    
    print(f"\n📅 {year}年缺失的时间点:")
    if missing:
        print(f"   缺失 {len(missing)} 个时间点:")
        for date in missing[:20]:  # 只显示前20个
            print(f"   - {date}")
        if len(missing) > 20:
            print(f"   ... 还有 {len(missing) - 20} 个")
    else:
        print(f"   ✅ 完整!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = './split_data'
    
    print("="*60)
    print("ERA5 分割文件验证工具")
    print("="*60)
    
    verify_split_files(directory)
    
    # 检查2021和2022年的完整性
    for year in [2021, 2022]:
        list_missing_dates(directory, year)
    
    print("\n" + "="*60)
    print("验证完成!")
    print("="*60)
