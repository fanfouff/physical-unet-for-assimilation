#!/usr/bin/env python3
"""
数据准备脚本 (修复版)
功能:
  1. 将 _X.npy/_Y.npy 转换为 .npz 格式
  2. 按比例划分为 train/val/test 数据集
  3. 计算并保存统计量 (stats.npz)
  4. 生成数据集划分信息 (dataset_split.json)

用法:
  python prepare_v2_data.py --source_dir /path/to/raw --target_dir /path/to/npz
"""

import numpy as np
import os
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

# ================= 默认配置 =================
DEFAULT_SOURCE_DIR = "/data2/lrx/era_obs"
DEFAULT_TARGET_DIR = "/data2/lrx/era_obs/npz"
H, W = 64, 64  # V2 框架默认的网格大小
# ==========================================


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='数据准备脚本 - 转换和划分数据集')
    parser.add_argument('--source_dir', type=str, default=DEFAULT_SOURCE_DIR,
                        help='原始 .npy 数据目录')
    parser.add_argument('--target_dir', type=str, default=DEFAULT_TARGET_DIR,
                        help='输出 .npz 数据目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例 (默认: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例 (默认: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    return parser.parse_args()


def convert_single_sample(x_path, y_path, h=H, w=W):
    """
    转换单个样本从 .npy 到 V2 格式
    
    Args:
        x_path: 观测数据路径 (_X.npy)
        y_path: 目标数据路径 (_Y.npy)
        h, w: 网格大小
    
    Returns:
        dict: 包含 obs, bkg, target, mask, aux 的字典
    """
    # 加载原始数据
    obs_points = np.load(x_path)  # [N, 17]
    tgt_points = np.load(y_path)  # [N, 37]
    
    n_points = obs_points.shape[0]
    max_points = h * w
    
    # 创建容器
    obs_grid = np.zeros((17, h, w), dtype=np.float32)
    target_grid = np.zeros((37, h, w), dtype=np.float32)
    mask = np.zeros((1, h, w), dtype=np.float32)
    
    # 填充数据
    use_points = min(n_points, max_points)
    
    # 展平填充再 reshape
    for c in range(17):
        temp = np.zeros(max_points)
        temp[:use_points] = obs_points[:use_points, c]
        obs_grid[c] = temp.reshape(h, w)
        
    for c in range(37):
        temp = np.zeros(max_points)
        temp[:use_points] = tgt_points[:use_points, c]
        target_grid[c] = temp.reshape(h, w)
        
    # 生成 Mask (1 表示有观测，0 表示填充)
    mask_flat = np.zeros(max_points)
    mask_flat[:use_points] = 1.0
    mask[0] = mask_flat.reshape(h, w)

    # 模拟背景场 (给 Target 加微弱噪声)
    np.random.seed(hash(str(x_path)) % (2**32))  # 确保可重复性
    bkg_grid = target_grid + np.random.normal(0, 0.5, target_grid.shape).astype(np.float32)

    # 模拟辅助特征 (Lat, Lon, SZA, LandMask)
    aux_grid = np.random.uniform(-1, 1, (4, h, w)).astype(np.float32)
    
    return {
        'obs': obs_grid,
        'bkg': bkg_grid,
        'target': target_grid,
        'mask': mask,
        'aux': aux_grid
    }


def split_dataset(file_list, train_ratio, val_ratio, seed):
    """
    划分数据集
    
    Args:
        file_list: 文件路径列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    np.random.seed(seed)
    
    n_total = len(file_list)
    indices = np.random.permutation(n_total)
    
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return {
        'train': [file_list[i] for i in train_indices],
        'val': [file_list[i] for i in val_indices],
        'test': [file_list[i] for i in test_indices]
    }


def compute_statistics(data_dir, split='train'):
    """
    计算数据集的统计量（均值和标准差）
    仅使用训练集计算，避免数据泄露
    
    Args:
        data_dir: 数据根目录
        split: 使用哪个划分计算统计量 (默认 'train')
    
    Returns:
        dict: 包含各变量统计量的字典
    """
    split_dir = Path(data_dir) / split
    npz_files = list(split_dir.glob("*.npz"))
    
    if not npz_files:
        print(f"警告: {split_dir} 中没有找到 .npz 文件")
        return None
    
    print(f"计算统计量 (使用 {len(npz_files)} 个 {split} 样本)...")
    
    # 在线计算均值和方差 (Welford's algorithm)
    stats = {}
    
    for key in ['obs', 'bkg', 'target', 'aux']:
        stats[key] = {
            'count': 0,
            'mean': None,
            'M2': None  # 用于计算方差
        }
    
    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path)
        
        for key in ['obs', 'bkg', 'target', 'aux']:
            if key not in data:
                continue
                
            arr = data[key]
            n_channels = arr.shape[0]
            
            # 初始化
            if stats[key]['mean'] is None:
                stats[key]['mean'] = np.zeros(n_channels)
                stats[key]['M2'] = np.zeros(n_channels)
            
            # 逐通道计算
            for c in range(n_channels):
                channel_data = arr[c].flatten()
                valid_data = channel_data[~np.isnan(channel_data)]
                
                if len(valid_data) == 0:
                    continue
                
                # Welford's online algorithm
                for x in valid_data:
                    stats[key]['count'] += 1
                    delta = x - stats[key]['mean'][c]
                    stats[key]['mean'][c] += delta / stats[key]['count']
                    delta2 = x - stats[key]['mean'][c]
                    stats[key]['M2'][c] += delta * delta2
        
        if (i + 1) % 100 == 0:
            print(f"  处理进度: {i + 1}/{len(npz_files)}")
    
    # 计算最终统计量
    result = {}
    for key in ['obs', 'bkg', 'target', 'aux']:
        if stats[key]['mean'] is not None:
            variance = stats[key]['M2'] / max(stats[key]['count'] - 1, 1)
            std = np.sqrt(variance)
            std[std < 1e-6] = 1.0  # 防止除零
            
            result[f'{key}_mean'] = stats[key]['mean'].astype(np.float32)
            result[f'{key}_std'] = std.astype(np.float32)
    
    return result


def convert_and_split_data(args):
    """
    主函数: 转换数据并划分数据集
    """
    src = Path(args.source_dir)
    tgt = Path(args.target_dir)
    
    # 创建目录结构
    for split in ['train', 'val', 'test']:
        (tgt / split).mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"数据准备脚本")
    print(f"=" * 60)
    print(f"源目录: {src}")
    print(f"目标目录: {tgt}")
    print(f"训练集比例: {args.train_ratio}")
    print(f"验证集比例: {args.val_ratio}")
    print(f"测试集比例: {1 - args.train_ratio - args.val_ratio:.2f}")
    print(f"随机种子: {args.seed}")
    print(f"=" * 60)
    
    # 1. 查找所有数据文件
    print(f"\n[步骤 1/4] 扫描数据文件...")
    x_files = sorted(list(src.glob("**/*_X.npy")))
    
    if not x_files:
        print(f"错误: 未找到任何 _X.npy 文件，请检查路径: {src}")
        return False
    
    # 过滤出有配对的文件
    valid_pairs = []
    for x_path in x_files:
        y_path = Path(str(x_path).replace("_X.npy", "_Y.npy"))
        if y_path.exists():
            valid_pairs.append((x_path, y_path))
    
    print(f"找到 {len(valid_pairs)} 个有效数据对")
    
    if len(valid_pairs) == 0:
        print("错误: 没有找到有效的数据对!")
        return False
    
    # 2. 划分数据集
    print(f"\n[步骤 2/4] 划分数据集...")
    splits = split_dataset(valid_pairs, args.train_ratio, args.val_ratio, args.seed)
    
    print(f"  训练集: {len(splits['train'])} 样本")
    print(f"  验证集: {len(splits['val'])} 样本")
    print(f"  测试集: {len(splits['test'])} 样本")
    
    # 3. 转换并保存数据
    print(f"\n[步骤 3/4] 转换并保存数据...")
    
    split_info = {'train': [], 'val': [], 'test': []}
    total_processed = 0
    total_files = len(valid_pairs)
    
    for split_name, pairs in splits.items():
        split_dir = tgt / split_name
        print(f"\n  处理 {split_name} 集...")
        
        for i, (x_path, y_path) in enumerate(pairs):
            try:
                # 转换数据
                data = convert_single_sample(x_path, y_path)
                
                # 生成保存文件名
                save_name = x_path.stem.replace("_X", "") + ".npz"
                save_path = split_dir / save_name
                
                # 保存
                np.savez(save_path, **data)
                
                split_info[split_name].append(save_name)
                total_processed += 1
                
                if (total_processed) % 100 == 0:
                    print(f"    已完成: {total_processed}/{total_files}")
                    
            except Exception as e:
                print(f"    警告: 处理 {x_path} 时出错: {e}")
                continue
    
    print(f"\n  转换完成: {total_processed}/{total_files} 个文件")
    
    # 4. 计算并保存统计量
    print(f"\n[步骤 4/4] 计算统计量...")
    stats = compute_statistics(tgt, split='train')
    
    if stats:
        stats_path = tgt / "stats.npz"
        np.savez(stats_path, **stats)
        print(f"  统计量已保存: {stats_path}")
    
    # 保存划分信息
    split_info_path = tgt / "dataset_split.json"
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
    
    with open(split_info_path, 'w') as f:
        json.dump(split_meta, f, indent=2)
    print(f"  划分信息已保存: {split_info_path}")
    
    # 打印最终统计
    print(f"\n" + "=" * 60)
    print(f"✅ 数据准备完成!")
    print(f"=" * 60)
    print(f"输出目录: {tgt}")
    print(f"  ├── train/     ({len(split_info['train'])} 文件)")
    print(f"  ├── val/       ({len(split_info['val'])} 文件)")
    print(f"  ├── test/      ({len(split_info['test'])} 文件)")
    print(f"  ├── stats.npz  (统计量)")
    print(f"  └── dataset_split.json (划分信息)")
    print(f"=" * 60)
    
    return True


if __name__ == "__main__":
    args = parse_args()
    success = convert_and_split_data(args)
    exit(0 if success else 1)