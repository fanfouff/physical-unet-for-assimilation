#!/usr/bin/env python3
"""
FY-3F文件整理工具
将FY-3F MWTS文件按年月整理到不同文件夹

文件名格式: FY3F_MWTS-_ORBA_L1_20250119_0440_033KM_V0.HDF
                                ^^^^^^^^
                                YYYYMMDD

输出结构:
output_dir/
├── 2025/
│   ├── 01/
│   │   ├── FY3F_MWTS-_ORBA_L1_20250119_0440_033KM_V0.HDF
│   │   ├── FY3F_MWTS-_ORBA_L1_20250119_0622_033KM_V0.HDF
│   │   └── ...
│   └── 02/
│       └── ...
"""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict
import argparse


def extract_date_from_filename(filename):
    """
    从FY-3F文件名中提取年月日
    
    Parameters:
    -----------
    filename : str
        文件名，例如: FY3F_MWTS-_ORBA_L1_20250119_0440_033KM_V0.HDF
    
    Returns:
    --------
    tuple : (year, month, day) 或 None
    """
    # 匹配日期模式: YYYYMMDD
    pattern = r'FY3F_MWTS-_.*?_(\d{4})(\d{2})(\d{2})_'
    match = re.search(pattern, filename)
    
    if match:
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        return year, month, day
    
    return None


def organize_files(input_dir, output_dir, mode='copy', dry_run=False):
    """
    整理FY-3F文件到年月文件夹
    
    Parameters:
    -----------
    input_dir : str
        输入目录（包含原始FY-3F文件）
    output_dir : str
        输出目录（将创建年/月子目录）
    mode : str
        'copy' 或 'move'（复制或移动文件）
    dry_run : bool
        如果为True，只显示将要执行的操作，不实际执行
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 查找所有FY-3F文件
    pattern = 'FY3F_MWTS*.HDF'
    files = list(input_path.glob(pattern))
    
    if not files:
        print(f"❌ 在 {input_dir} 中未找到FY-3F文件")
        return
    
    print(f"📂 找到 {len(files)} 个FY-3F文件")
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🔧 模式: {mode}")
    print(f"🔍 试运行: {'是' if dry_run else '否'}")
    print()
    
    # 按年月分组
    file_groups = defaultdict(list)
    skipped_files = []
    
    for file_path in files:
        filename = file_path.name
        date_info = extract_date_from_filename(filename)
        
        if date_info:
            year, month, day = date_info
            file_groups[(year, month)].append(file_path)
        else:
            skipped_files.append(filename)
    
    # 显示跳过的文件
    if skipped_files:
        print(f"⚠️  无法解析日期的文件 ({len(skipped_files)}个):")
        for f in skipped_files[:5]:
            print(f"   {f}")
        if len(skipped_files) > 5:
            print(f"   ... 还有 {len(skipped_files)-5} 个")
        print()
    
    # 显示分组统计
    print(f"📊 文件分组统计:")
    for (year, month), files_in_group in sorted(file_groups.items()):
        print(f"   {year}年{month}月: {len(files_in_group)} 个文件")
    print()
    
    # 整理文件
    total_processed = 0
    total_failed = 0
    
    for (year, month), files_in_group in sorted(file_groups.items()):
        # 创建目标目录
        target_dir = output_path / year / month
        
        print(f"📁 处理 {year}年{month}月 ({len(files_in_group)} 个文件)")
        print(f"   目标目录: {target_dir}")
        
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理每个文件
        success_count = 0
        for file_path in files_in_group:
            target_path = target_dir / file_path.name
            
            try:
                if dry_run:
                    print(f"   [试运行] {mode}: {file_path.name} -> {target_dir}/")
                else:
                    if mode == 'copy':
                        shutil.copy2(file_path, target_path)
                    elif mode == 'move':
                        shutil.move(str(file_path), str(target_path))
                    else:
                        raise ValueError(f"未知模式: {mode}")
                
                success_count += 1
                total_processed += 1
                
            except Exception as e:
                print(f"   ✗ 处理失败: {file_path.name}")
                print(f"     错误: {e}")
                total_failed += 1
        
        if not dry_run:
            print(f"   ✓ 完成: {success_count}/{len(files_in_group)} 个文件")
        print()
    
    # 总结
    print("="*70)
    if dry_run:
        print("试运行完成（未实际修改文件）")
    else:
        print("整理完成!")
    print("="*70)
    print(f"总文件数: {len(files)}")
    print(f"成功处理: {total_processed}")
    print(f"处理失败: {total_failed}")
    print(f"无法解析: {len(skipped_files)}")
    print("="*70)


def list_organized_structure(output_dir):
    """显示整理后的目录结构"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"目录不存在: {output_dir}")
        return
    
    print(f"\n📁 整理后的目录结构: {output_dir}")
    print("="*70)
    
    # 遍历年份目录
    years = sorted([d for d in output_path.iterdir() if d.is_dir()])
    
    for year_dir in years:
        year_name = year_dir.name
        print(f"📅 {year_name}/")
        
        # 遍历月份目录
        months = sorted([d for d in year_dir.iterdir() if d.is_dir()])
        
        for month_dir in months:
            month_name = month_dir.name
            files = list(month_dir.glob('FY3F*.HDF'))
            
            print(f"   📂 {month_name}/  ({len(files)} 个文件)")
            
            # 显示前3个文件作为示例
            for i, f in enumerate(files[:3]):
                print(f"      - {f.name}")
            
            if len(files) > 3:
                print(f"      ... 还有 {len(files)-3} 个文件")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description='FY-3F文件整理工具 - 按年月组织文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 复制文件并整理（推荐，保留原文件）:
   python organize_fy3f_files.py /data2/lrx/obs -o /data2/lrx/fy3f_organized

2. 移动文件并整理（原位置文件会被移走）:
   python organize_fy3f_files.py /data2/lrx/obs -o /data2/lrx/fy3f_organized --move

3. 先试运行，查看将要执行的操作:
   python organize_fy3f_files.py /data2/lrx/obs -o /data2/lrx/fy3f_organized --dry-run

4. 查看已整理的目录结构:
   python organize_fy3f_files.py /data2/lrx/fy3f_organized --list

输出结构:
output_dir/
├── 2024/
│   ├── 01/
│   │   └── FY3F_MWTS-_ORBA_L1_20240115_xxxx_033KM_V0.HDF
│   ├── 02/
│   │   └── FY3F_MWTS-_ORBA_L1_20240201_xxxx_033KM_V0.HDF
│   └── ...
├── 2025/
│   ├── 01/
│   │   ├── FY3F_MWTS-_ORBA_L1_20250119_0440_033KM_V0.HDF
│   │   ├── FY3F_MWTS-_ORBA_L1_20250119_0622_033KM_V0.HDF
│   │   └── ...
│   └── ...
        """
    )
    
    parser.add_argument('input_dir', 
                       help='输入目录（包含FY-3F文件）')
    parser.add_argument('-o', '--output', 
                       help='输出目录（将创建年/月子目录）')
    parser.add_argument('--move', action='store_true',
                       help='移动文件而不是复制（默认是复制）')
    parser.add_argument('--dry-run', action='store_true',
                       help='试运行模式（只显示将要执行的操作，不实际修改文件）')
    parser.add_argument('--list', action='store_true',
                       help='列出已整理的目录结构')
    
    args = parser.parse_args()
    
    # 如果是list模式
    if args.list:
        list_organized_structure(args.input_dir)
        return
    
    # 检查参数
    if not args.output:
        parser.error("需要指定输出目录 -o/--output")
    
    # 确定模式
    mode = 'move' if args.move else 'copy'
    
    # 执行整理
    organize_files(
        args.input_dir, 
        args.output,
        mode=mode,
        dry_run=args.dry_run
    )
    
    # 显示结果结构
    if not args.dry_run:
        list_organized_structure(args.output)


if __name__ == '__main__':
    main()
