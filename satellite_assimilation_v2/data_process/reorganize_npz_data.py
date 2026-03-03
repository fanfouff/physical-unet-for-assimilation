#!/usr/bin/env python3
"""
快速修复脚本 - 重新组织已转换的NPZ数据

如果你已经运行过原始的 prepare_v2_data.py 并且 .npz 文件
已经生成（但没有划分到 train/val/test 目录），
运行此脚本可以快速重新组织数据。

用法:
    python reorganize_npz_data.py
    
    # 或指定路径
    python reorganize_npz_data.py --npz_dir /data2/lrx/era_obs/npz
"""

import os
import numpy as np
import json
import shutil
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='重新组织NPZ数据并计算统计量')
    parser.add_argument('--npz_dir', type=str, default='/data2/lrx/era_obs/npz',
                        help='NPZ文件所在目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()


def reorganize_data(args):
    npz_dir = Path(args.npz_dir)
    
    print("=" * 60)
    print("快速修复脚本 - 重新组织NPZ数据")
    print("=" * 60)
    print(f"NPZ目录: {npz_dir}")
    print(f"训练比例: {args.train_ratio}")
    print(f"验证比例: {args.val_ratio}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)
    
    # 1. 创建子目录
    print("\n[步骤 1/4] 创建目录结构...")
    for split in ['train', 'val', 'test']:
        (npz_dir / split).mkdir(exist_ok=True)
    
    # 2. 查找根目录下的所有 .npz 文件（排除 stats.npz）
    print("\n[步骤 2/4] 扫描NPZ文件...")
    npz_files = sorted([
        f for f in npz_dir.iterdir() 
        if f.is_file() and f.suffix == '.npz' and f.name != 'stats.npz'
    ])
    
    if not npz_files:
        # 检查是否已经在子目录中
        train_files = list((npz_dir / 'train').glob('*.npz'))
        val_files = list((npz_dir / 'val').glob('*.npz'))
        test_files = list((npz_dir / 'test').glob('*.npz'))
        
        if train_files or val_files or test_files:
            print("数据似乎已经被划分到子目录中:")
            print(f"  train: {len(train_files)} 文件")
            print(f"  val: {len(val_files)} 文件")
            print(f"  test: {len(test_files)} 文件")
            
            # 检查统计量文件
            if not (npz_dir / 'stats.npz').exists():
                print("\n缺少 stats.npz，正在计算...")
                compute_and_save_stats(npz_dir, args.seed)
            
            print("\n✅ 验证完成!")
            return True
        else:
            print("错误: 未找到任何NPZ文件!")
            return False
    
    print(f"找到 {len(npz_files)} 个NPZ文件")
    
    # 3. 随机划分
    print("\n[步骤 3/4] 划分数据集...")
    np.random.seed(args.seed)
    indices = np.random.permutation(len(npz_files))
    
    n_train = int(len(npz_files) * args.train_ratio)
    n_val = int(len(npz_files) * args.val_ratio)
    
    splits = {
        'train': [npz_files[i] for i in indices[:n_train]],
        'val': [npz_files[i] for i in indices[n_train:n_train + n_val]],
        'test': [npz_files[i] for i in indices[n_train + n_val:]]
    }
    
    # 移动文件
    split_info = {'train': [], 'val': [], 'test': []}
    
    for split_name, files in splits.items():
        print(f"  移动 {len(files)} 个文件到 {split_name}/...")
        for f in files:
            dst = npz_dir / split_name / f.name
            shutil.move(str(f), str(dst))
            split_info[split_name].append(f.name)
    
    print(f"  训练集: {len(splits['train'])} 文件")
    print(f"  验证集: {len(splits['val'])} 文件")
    print(f"  测试集: {len(splits['test'])} 文件")
    
    # 4. 计算统计量
    print("\n[步骤 4/4] 计算统计量...")
    compute_and_save_stats(npz_dir, args.seed)
    
    # 保存划分信息
    split_meta = {
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': 1 - args.train_ratio - args.val_ratio,
        'seed': args.seed,
        'train_count': len(split_info['train']),
        'val_count': len(split_info['val']),
        'test_count': len(split_info['test']),
        'files': split_info
    }
    
    with open(npz_dir / 'dataset_split.json', 'w') as f:
        json.dump(split_meta, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print("✅ 数据重组织完成!")
    print("=" * 60)
    print(f"目录结构:")
    print(f"  {npz_dir}/")
    print(f"  ├── train/     ({len(split_info['train'])} 文件)")
    print(f"  ├── val/       ({len(split_info['val'])} 文件)")
    print(f"  ├── test/      ({len(split_info['test'])} 文件)")
    print(f"  ├── stats.npz")
    print(f"  └── dataset_split.json")
    print("=" * 60)
    
    return True


def compute_and_save_stats(npz_dir, seed):
    """计算并保存统计量"""
    train_dir = npz_dir / 'train'
    npz_files = list(train_dir.glob('*.npz'))
    
    if not npz_files:
        print("  警告: 训练集为空，无法计算统计量")
        return
    
    # 采样计算（最多使用200个样本）
    np.random.seed(seed)
    sample_size = min(200, len(npz_files))
    sample_files = np.random.choice(npz_files, sample_size, replace=False)
    
    print(f"  使用 {sample_size} 个样本计算统计量...")
    
    # 收集数据
    obs_list, bkg_list, target_list, aux_list = [], [], [], []
    
    for i, f in enumerate(sample_files):
        try:
            data = np.load(f)
            obs_list.append(data['obs'])
            bkg_list.append(data['bkg'])
            target_list.append(data['target'])
            aux_list.append(data['aux'])
        except Exception as e:
            print(f"  警告: 无法读取 {f}: {e}")
            continue
        
        if (i + 1) % 50 == 0:
            print(f"    进度: {i + 1}/{sample_size}")
    
    if not obs_list:
        print("  错误: 没有有效样本!")
        return
    
    # 堆叠并计算统计量
    obs_all = np.stack(obs_list)      # [N, C, H, W]
    bkg_all = np.stack(bkg_list)
    target_all = np.stack(target_list)
    aux_all = np.stack(aux_list)
    
    stats = {}
    for name, arr in [('obs', obs_all), ('bkg', bkg_all), ('target', target_all), ('aux', aux_all)]:
        # 逐通道计算 (在 N, H, W 维度上)
        mean = np.mean(arr, axis=(0, 2, 3))  # [C]
        std = np.std(arr, axis=(0, 2, 3)) + 1e-6  # [C]
        
        stats[f'{name}_mean'] = mean.astype(np.float32)
        stats[f'{name}_std'] = std.astype(np.float32)
    
    # 保存
    stats_path = npz_dir / 'stats.npz'
    np.savez(stats_path, **stats)
    print(f"  统计量已保存: {stats_path}")


if __name__ == "__main__":
    args = parse_args()
    success = reorganize_data(args)
    exit(0 if success else 1)
