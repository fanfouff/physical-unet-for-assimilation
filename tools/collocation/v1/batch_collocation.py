#!/usr/bin/env python3
"""
批量处理脚本
处理多个FY-3D文件和对应的ERA5数据
"""

import os
import glob
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import re
from collocation_fy3d_era5 import DataCollocation


class BatchProcessor:
    """批量数据处理器"""
    
    def __init__(self, fy3d_dir, era5_dir, output_dir='./batch_output'):
        self.fy3d_dir = Path(fy3d_dir)
        self.era5_dir = Path(era5_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def find_fy3d_files(self, pattern='*.HDF'):
        """查找所有FY-3F文件"""
        files = sorted(self.fy3d_dir.glob(pattern))
        print(f"📂 找到 {len(files)} 个 FY-3F 文件")
        return files
    
    def extract_time_from_fy3d(self, filename):
        """从FY-3F文件名提取时间"""
        # FY3_MWTS-_ORBD_L1_20210701_1455_033KM_V0.HDF
        match = re.search(r'(\d{8})_(\d{4})', str(filename)) # match
        if match:
            date_str = match.group(1) # date_str
            time_str = match.group(2)
            return datetime.strptime(date_str + time_str, '%Y%m%d%H%M')
        return None
    
    def find_matching_era5(self, fy3d_time, time_tolerance=3):
        """
        查找与FY-3D时间最接近的ERA5文件
        
        Parameters:
        -----------
        fy3d_time : datetime
            FY-3D观测时间
        time_tolerance : int
            时间容差（小时）
        """
        # ERA5文件命名: era5_YYYYMMDD_HH.grib
        target_hour = fy3d_time.hour
        
        # 查找最接近的小时（00或12）
        if abs(target_hour - 0) <= time_tolerance:
            era5_hour = 0
        elif abs(target_hour - 12) <= time_tolerance:
            era5_hour = 12
        elif target_hour >= 18 or target_hour <= 6:
            era5_hour = 0
            if target_hour >= 18:
                fy3d_time += timedelta(days=1)
        else:
            era5_hour = 12
        
        # 构建文件路径
        year = fy3d_time.strftime('%Y')
        month = fy3d_time.strftime('%m')
        day = fy3d_time.strftime('%d')
        
        era5_file = self.era5_dir / year / month / f"era5_{year}{month}{day}_{era5_hour:02d}.grib"
        
        if era5_file.exists():
            return era5_file
        
        # 尝试在主目录查找
        era5_file_alt = self.era5_dir / f"era5_{year}{month}{day}_{era5_hour:02d}.grib"
        if era5_file_alt.exists():
            return era5_file_alt
        
        return None
    
    def process_single_pair(self, fy3d_file, era5_file, file_index):
        """处理单个文件对"""
        try:
            print(f"\n{'='*70}")
            print(f"处理文件对 {file_index}")
            print(f"{'='*70}")
            print(f"FY-3D: {fy3d_file.name}")
            print(f"ERA5:  {era5_file.name}")
            
            # 执行配准
            collocation = DataCollocation()
            X, Y = collocation.process(
                str(fy3d_file),
                str(era5_file),
                bt_min=100.0,
                bt_max=400.0,
                max_distance=0.5
            )
            
            # 保存结果
            output_prefix = self.output_dir / f"pair_{file_index:04d}"
            collocation.save_numpy(str(output_prefix))
            
            return {
                'success': True,
                'n_samples': len(X),
                'fy3d_file': str(fy3d_file),
                'era5_file': str(era5_file)
            }
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            return {
                'success': False,
                'error': str(e),
                'fy3d_file': str(fy3d_file),
                'era5_file': str(era5_file)
            }
    
    def process_all(self, max_files=None):
        """批量处理所有文件"""
        print("="*70)
        print("批量数据配准处理")
        print("="*70)
        
        # 查找所有FY-3D文件
        fy3d_files = self.find_fy3d_files()
        
        if max_files:
            fy3d_files = fy3d_files[:max_files]
            print(f"⚠️  限制处理前 {max_files} 个文件")
        
        results = []
        total_samples = 0
        
        for i, fy3d_file in enumerate(fy3d_files, 1):
            # 提取时间
            fy3d_time = self.extract_time_from_fy3d(fy3d_file)
            if not fy3d_time:
                print(f"⚠️  跳过: 无法提取时间 - {fy3d_file.name}")
                continue
            
            # 查找匹配的ERA5文件
            era5_file = self.find_matching_era5(fy3d_time)
            if not era5_file:
                print(f"⚠️  跳过: 未找到匹配的ERA5文件 - {fy3d_file.name}")
                continue
            
            # 处理
            result = self.process_single_pair(fy3d_file, era5_file, i)
            results.append(result)
            
            if result['success']:
                total_samples += result['n_samples']
        
        # 保存处理日志
        self.save_batch_log(results, total_samples)
        
        return results
    
    def save_batch_log(self, results, total_samples):
        """保存批处理日志"""
        import json
        
        log_file = self.output_dir / 'batch_processing_log.json'
        
        summary = {
            'total_files': len(results),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'total_samples': total_samples,
            'results': results
        }
        
        with open(log_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print("📊 批处理汇总")
        print(f"{'='*70}")
        print(f"总文件数: {summary['total_files']}")
        print(f"成功: {summary['successful']}")
        print(f"失败: {summary['failed']}")
        print(f"总样本数: {total_samples:,}")
        print(f"日志已保存: {log_file}")
    
    def merge_all_outputs(self, output_file='merged_dataset'):
        """合并所有输出文件为单个数据集"""
        print("\n🔗 合并所有数据集...")
        
        # 查找所有输出文件
        X_files = sorted(self.output_dir.glob('pair_*_X.npy'))
        Y_files = sorted(self.output_dir.glob('pair_*_Y.npy'))
        
        if not X_files:
            print("❌ 未找到输出文件")
            return
        
        print(f"   找到 {len(X_files)} 个文件对")
        
        # 加载并合并
        X_list = []
        Y_list = []
        
        for x_file, y_file in zip(X_files, Y_files):
            X_list.append(np.load(x_file))
            Y_list.append(np.load(y_file))
        
        X_merged = np.vstack(X_list)
        Y_merged = np.vstack(Y_list)
        
        # 保存
        output_path = self.output_dir / f'{output_file}_X.npy'
        np.save(output_path, X_merged)
        print(f"   ✓ {output_path}")
        
        output_path = self.output_dir / f'{output_file}_Y.npy'
        np.save(output_path, Y_merged)
        print(f"   ✓ {output_path}")
        
        print(f"\n   合并后形状:")
        print(f"   X: {X_merged.shape}")
        print(f"   Y: {Y_merged.shape}")
        
        return X_merged, Y_merged


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量处理FY-3D和ERA5数据')
    parser.add_argument('fy3d_dir', help='FY-3D文件目录')
    parser.add_argument('era5_dir', help='ERA5文件目录')
    parser.add_argument('-o', '--output', default='./batch_output',
                       help='输出目录')
    parser.add_argument('-n', '--max-files', type=int, default=None,
                       help='最大处理文件数')
    parser.add_argument('--merge', action='store_true',
                       help='处理后合并所有数据')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = BatchProcessor(args.fy3d_dir, args.era5_dir, args.output)
    
    # 批量处理
    results = processor.process_all(max_files=args.max_files)
    
    # 合并数据
    if args.merge:
        processor.merge_all_outputs()
    
    print("\n✅ 批量处理完成!")


if __name__ == '__main__':
    main()
