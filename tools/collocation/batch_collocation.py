#!/usr/bin/env python3
"""
FY-3F与ERA5批量配准工具

功能:
- 自动遍历FY-3F和ERA5目录
- 按月批量配准
- 输出到指定目录，按年月组织
"""

import os
import sys
import glob
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse


class BatchCollocation:
    """批量配准处理器"""
    
    def __init__(self, fy3f_dir, era5_dir, output_dir, collocation_script, 
                 max_time_diff_hours=6):
        """
        初始化
        
        Parameters:
        -----------
        fy3f_dir : str
            FY-3F整理后的目录 (如: /data2/lrx/fy3f_organized)
        era5_dir : str
            ERA5数据目录 (如: /data2/lrx/era5_2/split_data)
        output_dir : str
            输出目录 (如: /data2/lrx/era_obs)
        collocation_script : str
            配准脚本路径
        max_time_diff_hours : int
            最大时间差（小时），默认6小时
        """
        self.fy3f_dir = Path(fy3f_dir)
        self.era5_dir = Path(era5_dir)
        self.output_dir = Path(output_dir)
        self.collocation_script = collocation_script
        self.max_time_diff_hours = max_time_diff_hours
        
        # 统计
        self.total_months = 0
        self.success_months = 0
        self.failed_months = 0
        self.skipped_months = 0
        
    def find_fy3f_months(self):
        """查找所有FY-3F年月目录"""
        months = []
        
        # 遍历年份目录
        for year_dir in sorted(self.fy3f_dir.glob('*')):
            if not year_dir.is_dir():
                continue
            
            year = year_dir.name
            
            # 遍历月份目录
            for month_dir in sorted(year_dir.glob('*')):
                if not month_dir.is_dir():
                    continue
                
                month = month_dir.name
                
                # 检查是否有FY-3F文件
                fy3f_files = list(month_dir.glob('FY3F*.HDF'))
                if fy3f_files:
                    months.append({
                        'year': year,
                        'month': month,
                        'fy3f_dir': month_dir,
                        'fy3f_count': len(fy3f_files)
                    })
        
        return months
    
    def find_era5_files(self, year, month):
        """
        查找对应年月的ERA5文件
        
        Parameters:
        -----------
        year : str
            年份 (如: '2024')
        month : str
            月份 (如: '01')
        
        Returns:
        --------
        list : ERA5文件路径列表
        """
        # 尝试不同的ERA5目录结构
        possible_paths = [
            self.era5_dir / year / month,  # /era5_2/split_data/2024/01/
            self.era5_dir / month,         # /era5_2/split_data/01/
            self.era5_dir / f"{year}{month}",  # /era5_2/split_data/202401/
        ]
        
        for path in possible_paths:
            if path.exists():
                # 查找ERA5文件 (支持多种格式)
                era5_files = []
                for pattern in ['*.grib', '*.grib2', '*.nc', '*.grb']:
                    era5_files.extend(list(path.glob(pattern)))
                
                if era5_files:
                    return sorted(era5_files)
        
        return []
    
    def process_month(self, year, month, fy3f_dir, era5_files, mode='single'):
        """
        处理单个月份的配准
        
        Parameters:
        -----------
        year : str
            年份
        month : str
            月份
        fy3f_dir : Path
            FY-3F文件目录
        era5_files : list
            ERA5文件列表
        mode : str
            配准模式:
            - 'single': 每个FY-3F文件配准一次，输出多个文件
            - 'merge': 合并整个月的数据，输出单个文件
        
        Returns:
        --------
        bool : 是否成功
        """
        # 创建输出目录
        output_subdir = self.output_dir / year / month
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"处理 {year}年{month}月")
        print(f"{'='*70}")
        print(f"FY-3F目录: {fy3f_dir}")
        print(f"ERA5文件数: {len(era5_files)}")
        print(f"输出目录: {output_subdir}")
        
        # 获取FY-3F文件
        fy3f_files = sorted(fy3f_dir.glob('FY3F*.HDF'))
        print(f"FY-3F文件数: {len(fy3f_files)}")
        
        if not era5_files:
            print(f"⚠️  未找到ERA5文件，跳过")
            return False
        
        if mode == 'single':
            return self._process_month_single(
                year, month, fy3f_files, era5_files, output_subdir
            )
        elif mode == 'merge':
            return self._process_month_merge(
                year, month, fy3f_files, era5_files, output_subdir
            )
        else:
            raise ValueError(f"未知模式: {mode}")
    
    def _process_month_single(self, year, month, fy3f_files, era5_files, output_dir):
        """单文件模式：每个FY-3F文件配准一次"""
        success_count = 0
        
        for i, fy3f_file in enumerate(fy3f_files, 1):
            print(f"\n[{i}/{len(fy3f_files)}] 处理: {fy3f_file.name}")
            
            # 从FY-3F文件名提取日期和时间
            import re
            match = re.search(r'(\d{8})_(\d{4})', fy3f_file.name)
            if not match:
                print(f"   ✗ 无法解析文件名")
                continue
            
            date_str = match.group(1)  # YYYYMMDD
            time_str = match.group(2)  # HHMM
            
            # 找到最接近的ERA5文件
            era5_file = self._find_closest_era5(date_str, time_str, era5_files)
            if not era5_file:
                print(f"   ✗ 未找到匹配的ERA5文件")
                continue
            
            print(f"   ERA5: {era5_file.name}")
            
            # 输出文件名
            output_prefix = output_dir / f"collocation_{date_str}_{time_str}"
            
            # 调用配准脚本
            cmd = [
                'python', self.collocation_script,
                str(fy3f_file),
                str(era5_file),
                '-o', str(output_prefix),
                '--format', 'both'
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5分钟超时
                )
                
                if result.returncode == 0:
                    print(f"   ✓ 配准成功")
                    success_count += 1
                else:
                    print(f"   ✗ 配准失败")
                    if result.stderr:
                        print(f"   错误: {result.stderr[:200]}")
            
            except subprocess.TimeoutExpired:
                print(f"   ✗ 超时")
            except Exception as e:
                print(f"   ✗ 错误: {e}")
        
        print(f"\n月度总结: 成功 {success_count}/{len(fy3f_files)}")
        return success_count > 0
    
    def _process_month_merge(self, year, month, fy3f_files, era5_files, output_dir):
        """合并模式：将整个月的数据合并配准"""
        # 这个模式需要修改配准脚本以支持批量文件
        # 目前先实现单文件模式
        print("   ⚠️  合并模式暂未实现，使用单文件模式")
        return self._process_month_single(year, month, fy3f_files, era5_files, output_dir)
    
    def _find_closest_era5(self, date_str, time_str, era5_files):
        """
        找到最接近的ERA5文件
        
        Parameters:
        -----------
        date_str : str
            日期 YYYYMMDD
        time_str : str
            时间 HHMM
        era5_files : list
            ERA5文件列表
        
        Returns:
        --------
        Path or None
        """
        import re
        
        # 解析FY-3F时间
        fy3f_datetime = datetime.strptime(date_str + time_str, '%Y%m%d%H%M')
        
        closest_file = None
        min_diff = float('inf')
        
        for era5_file in era5_files:
            # 尝试从ERA5文件名提取时间
            # 常见格式: era5_20240115_12.grib
            match = re.search(r'(\d{8})_(\d{2})', era5_file.name)
            if match:
                era5_date = match.group(1)
                era5_hour = match.group(2)
                era5_datetime = datetime.strptime(era5_date + era5_hour, '%Y%m%d%H')
                
                # 计算时间差
                diff = abs((fy3f_datetime - era5_datetime).total_seconds())
                
                if diff < min_diff:
                    min_diff = diff
                    closest_file = era5_file
        
        # 如果时间差太大，返回None
        max_diff_seconds = self.max_time_diff_hours * 3600
        if min_diff > max_diff_seconds:
            return None
        
        return closest_file
    
    def run(self, mode='single', dry_run=False, year_filter=None, month_filter=None):
        """
        执行批量配准
        
        Parameters:
        -----------
        mode : str
            配准模式 ('single' 或 'merge')
        dry_run : bool
            试运行模式
        year_filter : str or None
            只处理指定年份（如 '2024'）
        month_filter : list or None
            只处理指定月份列表（如 ['01', '02', '03']）
        """
        print("="*70)
        print("FY-3F 与 ERA5 批量配准工具")
        print("="*70)
        print(f"FY-3F目录: {self.fy3f_dir}")
        print(f"ERA5目录: {self.era5_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"配准脚本: {self.collocation_script}")
        print(f"处理模式: {mode}")
        print(f"时间窗口: {self.max_time_diff_hours}小时")
        print(f"试运行: {'是' if dry_run else '否'}")
        if year_filter:
            print(f"年份筛选: {year_filter}")
        if month_filter:
            print(f"月份筛选: {', '.join(month_filter)}")
        print()
        
        # 查找所有FY-3F月份
        months = self.find_fy3f_months()
        
        # 应用筛选
        if year_filter or month_filter:
            filtered_months = []
            for m in months:
                if year_filter and m['year'] != year_filter:
                    continue
                if month_filter and m['month'] not in month_filter:
                    continue
                filtered_months.append(m)
            months = filtered_months
        
        print(f"找到 {len(months)} 个月份需要处理:")
        for m in months:
            print(f"  {m['year']}年{m['month']}月: {m['fy3f_count']} 个FY-3F文件")
        print()
        
        if dry_run:
            print("试运行模式，不执行实际配准")
            return
        
        # 处理每个月份
        self.total_months = len(months)
        
        for m in months:
            year = m['year']
            month = m['month']
            fy3f_dir = m['fy3f_dir']
            
            # 查找ERA5文件
            era5_files = self.find_era5_files(year, month)
            
            if not era5_files:
                print(f"\n⚠️  {year}年{month}月: 未找到ERA5文件，跳过")
                self.skipped_months += 1
                continue
            
            # 处理配准
            try:
                success = self.process_month(year, month, fy3f_dir, era5_files, mode)
                if success:
                    self.success_months += 1
                else:
                    self.failed_months += 1
            except Exception as e:
                print(f"\n✗ {year}年{month}月处理失败: {e}")
                self.failed_months += 1
        
        # 总结
        self.print_summary()
    
    def print_summary(self):
        """打印处理总结"""
        print("\n" + "="*70)
        print("批量配准完成!")
        print("="*70)
        print(f"总月份数: {self.total_months}")
        print(f"成功: {self.success_months}")
        print(f"失败: {self.failed_months}")
        print(f"跳过: {self.skipped_months}")
        
        if self.total_months > 0:
            success_rate = self.success_months / self.total_months * 100
            print(f"成功率: {success_rate:.1f}%")
        
        print("="*70)
        print(f"\n输出目录: {self.output_dir}")
        print("\n输出文件结构:")
        print("era_obs/")
        print("├── YYYY/")
        print("│   ├── MM/")
        print("│   │   ├── collocation_YYYYMMDD_HHMM_X.npy")
        print("│   │   ├── collocation_YYYYMMDD_HHMM_Y.npy")
        print("│   │   ├── collocation_YYYYMMDD_HHMM_pressure.npy")
        print("│   │   ├── collocation_YYYYMMDD_HHMM_metadata.json")
        print("│   │   └── collocation_YYYYMMDD_HHMM.h5")


def main():
    parser = argparse.ArgumentParser(
        description='FY-3F与ERA5批量配准工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 基本用法:
   python batch_collocation.py \\
       --fy3f /data2/lrx/fy3f_organized \\
       --era5 /data2/lrx/era5_2/split_data \\
       --output /data2/lrx/era_obs

2. 试运行（查看将要处理什么）:
   python batch_collocation.py \\
       --fy3f /data2/lrx/fy3f_organized \\
       --era5 /data2/lrx/era5_2/split_data \\
       --output /data2/lrx/era_obs \\
       --dry-run

3. 只处理特定年份:
   python batch_collocation.py \\
       --fy3f /data2/lrx/fy3f_organized \\
       --era5 /data2/lrx/era5_2/split_data \\
       --output /data2/lrx/era_obs \\
       --year 2024

4. 只处理特定月份:
   python batch_collocation.py \\
       --fy3f /data2/lrx/fy3f_organized \\
       --era5 /data2/lrx/era5_2/split_data \\
       --output /data2/lrx/era_obs \\
       --months 01 02 03

5. 使用不同的时间窗口:
   python batch_collocation.py \\
       --fy3f /data2/lrx/fy3f_organized \\
       --era5 /data2/lrx/era5_2/split_data \\
       --output /data2/lrx/era_obs \\
       --time-window 3

目录结构要求:
  FY-3F: fy3f_organized/YYYY/MM/*.HDF
  ERA5:  era5_2/split_data/YYYY/MM/*.grib (或 .nc)
  输出:  era_obs/YYYY/MM/collocation_*.npy
        """
    )
    
    parser.add_argument('--fy3f', required=True,
                       help='FY-3F整理后的目录')
    parser.add_argument('--era5', required=True,
                       help='ERA5数据目录')
    parser.add_argument('--output', required=True,
                       help='输出目录')
    parser.add_argument('--collocation-script', 
                       default='collocation_fy3d_era5_fixed.py',
                       help='配准脚本路径 (默认: collocation_fy3d_era5_fixed.py)')
    parser.add_argument('--mode', choices=['single', 'merge'],
                       default='single',
                       help='处理模式 (默认: single)')
    parser.add_argument('--time-window', type=int, default=6,
                       choices=[3, 6, 12, 24],
                       help='ERA5时间窗口（小时），默认6小时')
    parser.add_argument('--year', type=str, default=None,
                       help='只处理指定年份（如: 2024）')
    parser.add_argument('--months', nargs='+', default=None,
                       help='只处理指定月份（如: 01 02 03）')
    parser.add_argument('--dry-run', action='store_true',
                       help='试运行模式')
    
    args = parser.parse_args()
    
    # 检查配准脚本是否存在
    if not os.path.exists(args.collocation_script):
        print(f"错误: 配准脚本不存在: {args.collocation_script}")
        print("请确保配准脚本在当前目录，或使用 --collocation-script 指定完整路径")
        sys.exit(1)
    
    # 格式化月份（补零）
    if args.months:
        formatted_months = []
        for m in args.months:
            if len(m) == 1:
                formatted_months.append('0' + m)
            else:
                formatted_months.append(m)
        args.months = formatted_months
    
    # 创建批量配准处理器
    batch = BatchCollocation(
        args.fy3f,
        args.era5,
        args.output,
        args.collocation_script,
        max_time_diff_hours=args.time_window
    )
    
    # 执行
    batch.run(
        mode=args.mode, 
        dry_run=args.dry_run,
        year_filter=args.year,
        month_filter=args.months
    )


if __name__ == '__main__':
    main()