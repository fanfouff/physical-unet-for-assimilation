#!/usr/bin/env python3
"""
===============================================================================
推理分析脚本 (Inference Analysis for Paper Figures)
===============================================================================

用于生成论文所需的可视化图表

生成的图表：
    1. Level-wise RMSE Profile (垂直RMSE廓线图)
    2. Spatial Bias Map (偏差空间分布图)
    3. Case Study Visualization (个例可视化)
    4. Scatter Plots (散点图对比)
    5. Time Series Analysis (时间序列分析)

用法:
    python inference_analysis.py \\
        --checkpoint outputs/experiment/best_model.pth \\
        --data_root /path/to/test/data \\
        --output_dir figures \\
        --case_study_idx 0

===============================================================================
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import seaborn as sns


# 定义要添加的目标路径
target_path = "/home/seu/Fuxi/Unet/satellite_assimilation_v2/"

# 将路径添加到 sys.path（Python 会优先搜索这个路径）
if target_path not in sys.path:  # 避免重复添加
    sys.path.append(target_path)

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline_v2 import (
    LazySatelliteERA5Dataset,
    InMemorySatelliteDataset,
    LevelwiseNormalizer,
    AssimilationMetrics
)

timestamp = __import__('time').strftime('%Y%m%d_%H%M%S')
# =============================================================================
# Part 1: 参数解析
# =============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='推理分析脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='测试数据根目录')
    parser.add_argument('--output_dir', type=str, default='figures/{timestamp}',
                        help='图表输出目录') # 加上时间戳
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader工作进程数')
    parser.add_argument('--case_study_idx', type=int, default=0,
                        help='个例可视化的样本索引')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--stats_file', type=str, default=None,
                        help='统计量文件')
    parser.add_argument('--baseline_checkpoint', type=str, default=None,
                        help='基准模型checkpoint（用于对比）')
    
    return parser.parse_args()


# =============================================================================
# Part 2: 推理引擎
# =============================================================================

class InferenceEngine:
    """推理引擎"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        normalizers: Dict[str, LevelwiseNormalizer]
    ):
        self.model = model
        self.device = device
        self.normalizers = normalizers
        self.metrics = AssimilationMetrics()
    
    @torch.no_grad()
    def predict_batch(
        self,
        obs: torch.Tensor,
        bkg: torch.Tensor,
        mask: torch.Tensor,
        aux: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """预测一个batch"""
        self.model.eval()
        
        # 移动到设备
        obs = obs.to(self.device)
        bkg = bkg.to(self.device)
        mask = mask.to(self.device)
        if aux is not None:
            aux = aux.to(self.device)
        
        # 推理
        if aux is not None:
            pred = self.model(obs, bkg, mask, aux)
        else:
            pred = self.model(obs, bkg, mask)
        
        return pred.cpu()
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataloader: DataLoader
    ) -> Dict[str, np.ndarray]:
        """在整个数据集上评估"""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_bkgs = []
        all_obs = []
        all_masks = []
        
        for batch in dataloader:
            # 修复：batch 是字典而非元组
            obs = batch['obs']
            bkg = batch['bkg']
            mask = batch['mask']
            target = batch['target']
            aux = batch.get('aux', None)
            
            # 预测
            pred = self.predict_batch(obs, bkg, mask, aux)
            
            all_preds.append(pred.numpy())
            all_targets.append(target.numpy())
            all_bkgs.append(bkg.numpy())
            all_obs.append(obs.numpy())
            all_masks.append(mask.numpy())
        
        # 拼接
        results = {
            'pred': np.concatenate(all_preds, axis=0),
            'target': np.concatenate(all_targets, axis=0),
            'bkg': np.concatenate(all_bkgs, axis=0),
            'obs': np.concatenate(all_obs, axis=0),
            'mask': np.concatenate(all_masks, axis=0)
        }
        
        return results


# =============================================================================
# Part 3: 可视化函数
# =============================================================================

def plot_vertical_rmse_profile(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    pressure_levels: np.ndarray,
    output_path: Path
):
    """
    绘制垂直RMSE廓线图
    
    Args:
        results_dict: {model_name: {'pred': ..., 'target': ...}}
        pressure_levels: [n_levels] 气压层（hPa）
        output_path: 输出路径
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    
    colors_map = {
        'Background': '#1f77b4',
        'Baseline': '#ff7f0e',
        'Ours': '#d62728'
    }
    
    for model_name, results in results_dict.items():
        pred = results['pred']  # [N, C, H, W]
        target = results['target']
        
        # 计算每层的RMSE
        rmse_per_level = []
        for level in range(pred.shape[1]):
            pred_level = pred[:, level, :, :]
            target_level = target[:, level, :, :]
            
            rmse = np.sqrt(np.mean((pred_level - target_level) ** 2))
            rmse_per_level.append(rmse)
        
        rmse_per_level = np.array(rmse_per_level)
        
        # 绘制
        ax.plot(
            rmse_per_level,
            pressure_levels,
            marker='o',
            markersize=4,
            label=model_name,
            color=colors_map.get(model_name, None),
            linewidth=2
        )
    
    # 设置
    ax.set_xlabel('RMSE', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pressure (hPa)', fontsize=14, fontweight='bold')
    ax.set_title('Vertical RMSE Profile', fontsize=16, fontweight='bold')
    ax.invert_yaxis()  # 气压从上到下递减
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 保存: {output_path}")


def plot_spatial_bias_map(
    pred: np.ndarray,
    target: np.ndarray,
    level_idx: int,
    sample_idx: int,
    output_path: Path,
    title: str = 'Spatial Bias Map'
):
    """
    绘制偏差空间分布图
    
    Args:
        pred: [N, C, H, W]
        target: [N, C, H, W]
        level_idx: 选择的层级
        sample_idx: 选择的样本
        output_path: 输出路径
        title: 图表标题
    """
    # 计算偏差
    bias = pred[sample_idx, level_idx] - target[sample_idx, level_idx]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 使用diverging colormap
    vmax = np.abs(bias).max()
    vmin = -vmax
    
    im = ax.imshow(
        bias,
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        interpolation='bilinear'
    )
    
    ax.set_title(f'{title} (Level {level_idx})', fontsize=16, fontweight='bold')
    ax.set_xlabel('X (Grid)', fontsize=14)
    ax.set_ylabel('Y (Grid)', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Bias (Prediction - Truth)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 保存: {output_path}")


def plot_case_study(
    obs: np.ndarray,
    bkg: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    level_idx: int,
    sample_idx: int,
    output_path: Path
):
    """
    绘制个例可视化（4子图对比）
    
    Args:
        obs: [N, C_obs, H, W]
        bkg: [N, C_bkg, H, W]
        target: [N, C_bkg, H, W]
        pred: [N, C_bkg, H, W]
        level_idx: 选择的层级
        sample_idx: 选择的样本
        output_path: 输出路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 选择数据
    obs_data = obs[sample_idx, 0]  # 取第一个观测通道
    bkg_data = bkg[sample_idx, level_idx]
    target_data = target[sample_idx, level_idx]
    pred_data = pred[sample_idx, level_idx]
    
    # 统一colormap范围
    vmin = min(bkg_data.min(), target_data.min(), pred_data.min())
    vmax = max(bkg_data.max(), target_data.max(), pred_data.max())
    
    # (a) Observation
    im1 = axes[0, 0].imshow(obs_data, cmap='viridis', origin='lower')
    axes[0, 0].set_title('(a) Observation (Satellite BT)', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # (b) Background Field
    im2 = axes[0, 1].imshow(bkg_data, cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 1].set_title('(b) Background Field (NWP)', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # (c) Target (3D-Var Analysis)
    im3 = axes[1, 0].imshow(target_data, cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
    axes[1, 0].set_title('(c) Target (3D-Var Analysis)', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # (d) Prediction (Our Model)
    im4 = axes[1, 1].imshow(pred_data, cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
    axes[1, 1].set_title('(d) Prediction (Our Model)', fontsize=14, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    # 设置
    for ax in axes.flat:
        ax.set_xlabel('X (Grid)', fontsize=12)
        ax.set_ylabel('Y (Grid)', fontsize=12)
    
    plt.suptitle(f'Case Study: Level {level_idx}, Sample {sample_idx}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 保存: {output_path}")


def plot_scatter_comparison(
    pred: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    title: str = 'Scatter Plot: Prediction vs Truth'
):
    """
    绘制散点图对比
    
    Args:
        pred: [N, C, H, W]
        target: [N, C, H, W]
        output_path: 输出路径
        title: 标题
    """
    # 展平
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # 采样（避免点太多）
    n_samples = min(100000, len(pred_flat))
    indices = np.random.choice(len(pred_flat), n_samples, replace=False)
    pred_sample = pred_flat[indices]
    target_sample = target_flat[indices]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 2D histogram散点图
    h = ax.hist2d(
        target_sample,
        pred_sample,
        bins=100,
        cmap='Blues',
        norm=colors.LogNorm()
    )
    
    # 1:1线
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, label='1:1 Line')
    
    # 计算指标
    rmse = np.sqrt(np.mean((pred_sample - target_sample) ** 2))
    corr = np.corrcoef(pred_sample, target_sample)[0, 1]
    bias = np.mean(pred_sample - target_sample)
    
    # 显示指标
    text_str = f'RMSE: {rmse:.4f}\nCorr: {corr:.4f}\nBias: {bias:.4f}'
    ax.text(
        0.05, 0.95, text_str,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    ax.set_xlabel('Truth', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prediction', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(h[3], ax=ax, label='Count (log scale)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 保存: {output_path}")


def plot_correlation_heatmap(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    output_path: Path
):
    """
    绘制相关系数热力图（按垂直层级）
    
    Args:
        results_dict: {model_name: {'pred': ..., 'target': ...}}
        output_path: 输出路径
    """
    n_models = len(results_dict)
    n_levels = list(results_dict.values())[0]['pred'].shape[1]
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8))
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, results) in zip(axes, results_dict.items()):
        pred = results['pred']
        target = results['target']
        
        # 计算每层的相关系数
        corr_per_level = []
        for level in range(n_levels):
            pred_level = pred[:, level, :, :].flatten()
            target_level = target[:, level, :, :].flatten()
            
            corr = np.corrcoef(pred_level, target_level)[0, 1]
            corr_per_level.append(corr)
        
        # 绘制热力图
        corr_matrix = np.array(corr_per_level).reshape(-1, 1)
        
        im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0.8, vmax=1.0, aspect='auto')
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Vertical Level', fontsize=12)
        ax.set_xlabel('Correlation', fontsize=12)
        ax.set_xticks([])
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Correlation Coefficient by Level', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 保存: {output_path}")


# =============================================================================
# Part 4: 主函数
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """加载模型"""
    from models.backbone import create_model, UNetConfig
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 从checkpoint中提取配置
    args = checkpoint.get('args', {})
    
    # 创建模型
    if args.get('model', 'physics_unet') != 'vanilla_unet':
        config = UNetConfig(
            fusion_mode=args.get('fusion_mode', 'gated'),
            use_aux=args.get('use_aux', True),
            mask_aware=args.get('mask_aware', True)
        )
        model = create_model(args.get('model', 'physics_unet'), config=config)
    else:
        model = create_model('vanilla_unet')
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("开始推理分析")
    print("=" * 70 + "\n")
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # =========================================================================
    # 加载模型
    # =========================================================================
    print("加载模型...")
    model = load_model(args.checkpoint, device)
    print("  ✓ 主模型加载完成")
    
    # 基准模型（可选）
    baseline_model = None
    if args.baseline_checkpoint:
        baseline_model = load_model(args.baseline_checkpoint, device)
        print("  ✓ 基准模型加载完成")
    
    # =========================================================================
    # 加载数据
    # =========================================================================
    print("\n加载测试数据...")
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise ValueError(f"数据目录不存在: {data_root}")
    
    file_list = sorted(data_root.glob('**/*.npz'))
    if not file_list:
        raise ValueError(f"数据目录中未找到.npz文件: {data_root}")
    
    dataset = LazySatelliteERA5Dataset(
        file_list=[str(f) for f in file_list],
        use_aux=True
    )
    
    # 加载统计量
    if args.stats_file and Path(args.stats_file).exists():
        stats = np.load(args.stats_file)
        dataset.obs_normalizer = LevelwiseNormalizer(
            stats['obs_mean'], stats['obs_std'], name='obs'
        )
        dataset.bkg_normalizer = LevelwiseNormalizer(
            stats['bkg_mean'], stats['bkg_std'], name='bkg'
        )
        dataset.target_normalizer = LevelwiseNormalizer(
            stats['target_mean'], stats['target_std'], name='target'
        )
    else:
        dataset.compute_statistics(n_samples=min(1000, len(dataset)))
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"  测试样本数: {len(dataset)}")
    
    # =========================================================================
    # 推理
    # =========================================================================
    print("\n开始推理...")
    
    normalizers = {
        'obs': dataset.obs_normalizer,
        'bkg': dataset.bkg_normalizer,
        'target': dataset.target_normalizer
    }
    
    engine = InferenceEngine(model, device, normalizers)
    results = engine.evaluate_dataset(dataloader)
    
    print(f"  ✓ 推理完成")
    
    # 基准模型推理（可选）
    results_dict = {'Ours': results}
    
    if baseline_model:
        baseline_engine = InferenceEngine(baseline_model, device, normalizers)
        baseline_results = baseline_engine.evaluate_dataset(dataloader)
        results_dict['Baseline'] = baseline_results
    
    # 添加Background作为对比
    results_dict['Background'] = {
        'pred': results['bkg'],
        'target': results['target']
    }
    
    # =========================================================================
    # 生成图表
    # =========================================================================
    print("\n生成图表...")
    
    # 定义气压层（示例：37层）
    n_levels = results['pred'].shape[1]
    pressure_levels = np.linspace(1000, 1, n_levels)  # 1000 hPa到1 hPa
    
    # 1. 垂直RMSE廓线图
    print("  生成垂直RMSE廓线图...")
    plot_vertical_rmse_profile(
        results_dict,
        pressure_levels,
        output_dir / 'vertical_rmse_profile.png'
    )
    
    # 2. 偏差空间分布图
    print("  生成偏差空间分布图...")
    plot_spatial_bias_map(
        results['pred'],
        results['target'],
        level_idx=10,  # 选择边界层
        sample_idx=args.case_study_idx,
        output_path=output_dir / 'spatial_bias_map.png'
    )
    
    # 3. 个例可视化
    print("  生成个例可视化...")
    plot_case_study(
        results['obs'],
        results['bkg'],
        results['target'],
        results['pred'],
        level_idx=10,
        sample_idx=args.case_study_idx,
        output_path=output_dir / 'case_study.png'
    )
    
    # 4. 散点图对比
    print("  生成散点图对比...")
    plot_scatter_comparison(
        results['pred'],
        results['target'],
        output_dir / 'scatter_plot.png'
    )
    
    # 5. 相关系数热力图
    print("  生成相关系数热力图...")
    plot_correlation_heatmap(
        results_dict,
        output_dir / 'correlation_heatmap.png'
    )
    
    print("\n" + "=" * 70)
    print("分析完成!")
    print(f"图表保存至: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
