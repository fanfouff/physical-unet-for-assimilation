#!/usr/bin/env python3
"""
推理分析脚本 (Inference Analysis for Paper Figures) - 改进版
===============================================================================

改进内容：
1. Case Study: 添加 Cartopy 地图投影，显示海岸线、国界
2. Correlation: 动态颜色范围 + 廓线图 + 热力图 + 条形图组合
3. Spatial Bias Map: pcolormesh + 等值线 + 统计信息

依赖安装:
    pip install cartopy scipy

用法:
    python inference_analysis_v2.py \
        --checkpoint outputs/experiment/best_model.pth \
        --data_root /path/to/test/data \
        --output_dir figures \
        --lon_min 70 --lon_max 140 --lat_min 15 --lat_max 55
===============================================================================
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from scipy import stats

# =============================================================================
# Cartopy 检测与导入
# =============================================================================
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("⚠️  警告: Cartopy未安装，将使用简化地图绘制")
    print("   安装命令: pip install cartopy")

# =============================================================================
# 项目路径配置
# =============================================================================
target_path = "/home/seu/Fuxi/Unet/satellite_assimilation_v2/"
if target_path not in sys.path:
    sys.path.append(target_path)

sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# Matplotlib 全局配置
# =============================================================================
plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'SimHei', 'Arial'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
})

# =============================================================================
# 导入项目模块
# =============================================================================
from data_pipeline_v2 import (
    LazySatelliteERA5Dataset,
    LevelwiseNormalizer,
    AssimilationMetrics
)

timestamp = time.strftime('%Y%m%d_%H%M%S')


# =============================================================================
# Part 1: 参数解析
# =============================================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='推理分析脚本 (改进版)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='测试数据根目录')
    parser.add_argument('--output_dir', type=str, default=f'figures/{timestamp}',
                        help='图表输出目录')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--case_study_idx', type=int, default=0,
                        help='个例可视化的样本索引')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--stats_file', type=str, default=None)
    parser.add_argument('--baseline_checkpoint', type=str, default=None)
    
    # 地理范围参数 (中国区域默认)
    parser.add_argument('--lon_min', type=float, default=70.0)
    parser.add_argument('--lon_max', type=float, default=140.0)
    parser.add_argument('--lat_min', type=float, default=15.0)
    parser.add_argument('--lat_max', type=float, default=55.0)
    
    # 可视化参数
    parser.add_argument('--level_idx', type=int, default=10,
                        help='可视化的垂直层级索引')

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
        
        obs = obs.to(self.device)
        bkg = bkg.to(self.device)
        mask = mask.to(self.device)
        if aux is not None:
            aux = aux.to(self.device)
        
        if aux is not None:
            pred = self.model(obs, bkg, mask, aux)
        else:
            pred = self.model(obs, bkg, mask)
        
        return pred.cpu()

    @torch.no_grad()
    def evaluate_dataset(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """在整个数据集上评估"""
        self.model.eval()
        
        all_preds, all_targets, all_bkgs, all_obs, all_masks = [], [], [], [], []
        
        for batch in dataloader:
            obs = batch['obs']
            bkg = batch['bkg']
            mask = batch['mask']
            target = batch['target']
            aux = batch.get('aux', None)
            
            pred = self.predict_batch(obs, bkg, mask, aux)
            
            all_preds.append(pred.numpy())
            all_targets.append(target.numpy())
            all_bkgs.append(bkg.numpy())
            all_obs.append(obs.numpy())
            all_masks.append(mask.numpy())
        
        return {
            'pred': np.concatenate(all_preds, axis=0),
            'target': np.concatenate(all_targets, axis=0),
            'bkg': np.concatenate(all_bkgs, axis=0),
            'obs': np.concatenate(all_obs, axis=0),
            'mask': np.concatenate(all_masks, axis=0)
        }


# =============================================================================
# Part 3: 地图绑定工具函数
# =============================================================================
def create_geo_axes(
    fig, 
    subplot_spec, 
    extent: Tuple[float, float, float, float],
    add_features: bool = True
):
    """
    创建带地理特征的坐标轴
    
    Args:
        fig: matplotlib figure
        subplot_spec: subplot specification (e.g., 221)
        extent: (lon_min, lon_max, lat_min, lat_max)
        add_features: 是否添加地理特征
    
    Returns:
        ax: axes object
    """
    if HAS_CARTOPY:
        ax = fig.add_subplot(subplot_spec, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        if add_features:
            # 添加地理特征
            ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle='--', 
                          edgecolor='dimgray')
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
            
            # 添加省界（如果范围在中国）
            try:
                ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
            except:
                pass
            
            # 添加网格线
            gl = ax.gridlines(
                draw_labels=True, 
                linewidth=0.5, 
                color='gray', 
                alpha=0.5, 
                linestyle='--'
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size': 10}
        
        return ax, True
    else:
        ax = fig.add_subplot(subplot_spec)
        return ax, False


def create_coordinate_grids(
    data_shape: Tuple[int, int],
    extent: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建经纬度网格
    
    Args:
        data_shape: (H, W)
        extent: (lon_min, lon_max, lat_min, lat_max)
    
    Returns:
        lon_grid, lat_grid: 2D arrays
    """
    lon_min, lon_max, lat_min, lat_max = extent
    lons = np.linspace(lon_min, lon_max, data_shape[1])
    lats = np.linspace(lat_min, lat_max, data_shape[0])
    return np.meshgrid(lons, lats)


# =============================================================================
# Part 4: 可视化函数 - 垂直RMSE廓线
# =============================================================================
def plot_vertical_rmse_profile(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    pressure_levels: np.ndarray,
    output_path: Path
):
    """
    绘制垂直RMSE廓线图（带改进百分比）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    
    colors_map = {'Background': '#1f77b4', 'Baseline': '#ff7f0e', 'Ours': '#d62728'}
    markers = {'Background': 's', 'Baseline': '^', 'Ours': 'o'}
    
    # =========== (a) RMSE廓线 ===========
    ax1 = axes[0]
    rmse_results = {}
    
    for model_name, results in results_dict.items():
        pred = results['pred']
        target = results['target']
        
        rmse_per_level = []
        for level in range(pred.shape[1]):
            rmse = np.sqrt(np.mean((pred[:, level] - target[:, level]) ** 2))
            rmse_per_level.append(rmse)
        
        rmse_per_level = np.array(rmse_per_level)
        rmse_results[model_name] = rmse_per_level
        
        ax1.plot(
            rmse_per_level, pressure_levels,
            marker=markers.get(model_name, 'o'),
            markersize=5, linewidth=2,
            color=colors_map.get(model_name),
            label=model_name,
            markerfacecolor='white', markeredgewidth=1.5
        )
    
    ax1.set_xlabel('RMSE', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Pressure (hPa)', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Vertical RMSE Profile', fontsize=16, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # =========== (b) RMSE改进百分比 ===========
    ax2 = axes[1]
    
    if 'Background' in rmse_results and 'Ours' in rmse_results:
        improvement = (rmse_results['Background'] - rmse_results['Ours']) / \
                      rmse_results['Background'] * 100
        
        colors_bar = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvement]
        
        bars = ax2.barh(pressure_levels, improvement, color=colors_bar, 
                        alpha=0.7, height=15)
        ax2.axvline(x=0, color='black', linewidth=1.5)
        
        # 添加平均改进标注
        mean_improvement = np.mean(improvement)
        ax2.axvline(x=mean_improvement, color='blue', linewidth=2, 
                    linestyle='--', label=f'Mean: {mean_improvement:.1f}%')
        
        ax2.set_xlabel('RMSE Improvement (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Pressure (hPa)', fontsize=14, fontweight='bold')
        ax2.set_title('(b) RMSE Improvement over Background', 
                     fontsize=16, fontweight='bold')
        ax2.invert_yaxis()
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 保存: {output_path}")


# =============================================================================
# Part 5: 可视化函数 - 偏差空间分布图（改进版）
# =============================================================================
def plot_spatial_bias_map(
    pred: np.ndarray,
    target: np.ndarray,
    level_idx: int,
    sample_idx: int,
    output_path: Path,
    extent: Tuple[float, float, float, float],
    title: str = 'Spatial Bias Map'
):
    """
    绘制偏差空间分布图（带地理坐标、等值线、统计信息）
    """
    # 计算偏差
    bias = pred[sample_idx, level_idx] - target[sample_idx, level_idx]
    
    # 创建经纬度网格
    lon_grid, lat_grid = create_coordinate_grids(bias.shape, extent)
    
    # 使用98百分位数避免极值影响
    vmax = np.percentile(np.abs(bias), 98)
    vmin = -vmax
    
    fig = plt.figure(figsize=(14, 10))
    
    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # 地理特征
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle='--', edgecolor='dimgray')
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.15)
        
        # 填色图 - 使用pcolormesh提高清晰度
        im = ax.pcolormesh(
            lon_grid, lat_grid, bias,
            cmap='RdBu_r', vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading='auto', rasterized=True  # rasterized 提高PDF质量
        )
        
        # 等值线
        levels = np.linspace(vmin, vmax, 15)
        cs = ax.contour(
            lon_grid, lat_grid, bias,
            levels=levels[::2],  # 每隔一条画
            colors='black', linewidths=0.4, alpha=0.6,
            transform=ccrs.PlateCarree()
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.3f')
        
        # 网格线
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, 
                          color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 11}
        gl.ylabel_style = {'size': 11}
        
    else:
        ax = fig.add_subplot(1, 1, 1)
        im = ax.pcolormesh(lon_grid, lat_grid, bias, cmap='RdBu_r',
                           vmin=vmin, vmax=vmax, shading='auto')
        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
    
    ax.set_title(f'{title}\nLevel {level_idx} | Sample {sample_idx}', 
                 fontsize=16, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend='both')
    cbar.set_label('Bias (Prediction − Truth)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # 统计信息
    stats_text = (f'Mean: {bias.mean():.4f}\n'
                  f'Std: {bias.std():.4f}\n'
                  f'RMSE: {np.sqrt(np.mean(bias**2)):.4f}\n'
                  f'Max: {bias.max():.4f}\n'
                  f'Min: {bias.min():.4f}')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 保存: {output_path}")


# =============================================================================
# Part 6: 可视化函数 - 个例研究（带世界地图）
# =============================================================================
def plot_case_study(
    obs: np.ndarray,
    bkg: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    level_idx: int,
    sample_idx: int,
    output_path: Path,
    extent: Tuple[float, float, float, float]
):
    """
    绘制个例可视化（4子图 + 地理坐标）
    """
    # 选择数据
    obs_data = obs[sample_idx, 0]
    bkg_data = bkg[sample_idx, level_idx]
    target_data = target[sample_idx, level_idx]
    pred_data = pred[sample_idx, level_idx]
    
    # 创建经纬度网格
    lon_grid, lat_grid = create_coordinate_grids(obs_data.shape, extent)
    
    # 统一colormap范围
    vmin_atm = min(bkg_data.min(), target_data.min(), pred_data.min())
    vmax_atm = max(bkg_data.max(), target_data.max(), pred_data.max())
    
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(18, 16))
        
        panels = [
            (221, obs_data, 'viridis', None, None, 
             '(a) Satellite Observation (BT)', 'Brightness Temp (K)'),
            (222, bkg_data, 'jet', vmin_atm, vmax_atm, 
             '(b) Background (NWP)', 'Value'),
            (223, target_data, 'jet', vmin_atm, vmax_atm, 
             '(c) Target (3D-Var Analysis)', 'Value'),
            (224, pred_data, 'jet', vmin_atm, vmax_atm, 
             '(d) Prediction (Our Model)', 'Value'),
        ]
        
        for pos, data, cmap, vmin, vmax, title, cbar_label in panels:
            ax = fig.add_subplot(pos, projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            # 地理特征
            ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', edgecolor='gray')
            
            # 绘图
            im = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               transform=ccrs.PlateCarree(), shading='auto')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            # 网格线
            gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 9}
            gl.ylabel_style = {'size': 9}
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label, fontsize=10)
            cbar.ax.tick_params(labelsize=9)
    
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        panels = [
            (axes[0,0], obs_data, 'viridis', None, None, '(a) Observation'),
            (axes[0,1], bkg_data, 'jet', vmin_atm, vmax_atm, '(b) Background'),
            (axes[1,0], target_data, 'jet', vmin_atm, vmax_atm, '(c) Target'),
            (axes[1,1], pred_data, 'jet', vmin_atm, vmax_atm, '(d) Prediction'),
        ]
        
        for ax, data, cmap, vmin, vmax, title in panels:
            im = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap,
                               vmin=vmin, vmax=vmax, shading='auto')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'Case Study: Level {level_idx} | Sample {sample_idx}', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 保存: {output_path}")
    
    # 额外生成差值对比图
    _plot_error_comparison(
        bkg_data, target_data, pred_data, 
        lon_grid, lat_grid, extent,
        output_path.parent / f'{output_path.stem}_error.png',
        level_idx, sample_idx
    )


def _plot_error_comparison(
    bkg: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    extent: Tuple[float, float, float, float],
    output_path: Path,
    level_idx: int,
    sample_idx: int
):
    """绘制误差对比图"""
    bkg_error = bkg - target
    pred_error = pred - target
    improvement = np.abs(bkg_error) - np.abs(pred_error)  # 正值表示改进
    
    vmax_err = max(np.percentile(np.abs(bkg_error), 98), 
                   np.percentile(np.abs(pred_error), 98))
    vmax_imp = np.percentile(np.abs(improvement), 98)
    
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(18, 6))
        
        # (a) Background Error
        ax1 = fig.add_subplot(131, projection=ccrs.PlateCarree())
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.COASTLINE, linewidth=1.0)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
        
        bkg_rmse = np.sqrt(np.mean(bkg_error**2))
        im1 = ax1.pcolormesh(lon_grid, lat_grid, bkg_error, cmap='RdBu_r',
                              vmin=-vmax_err, vmax=vmax_err,
                              transform=ccrs.PlateCarree(), shading='auto')
        ax1.set_title(f'(a) Background Error\nRMSE = {bkg_rmse:.4f}', 
                      fontsize=13, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, extend='both')
        
        # (b) Prediction Error
        ax2 = fig.add_subplot(132, projection=ccrs.PlateCarree())
        ax2.set_extent(extent, crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.COASTLINE, linewidth=1.0)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
        
        pred_rmse = np.sqrt(np.mean(pred_error**2))
        im2 = ax2.pcolormesh(lon_grid, lat_grid, pred_error, cmap='RdBu_r',
                              vmin=-vmax_err, vmax=vmax_err,
                              transform=ccrs.PlateCarree(), shading='auto')
        ax2.set_title(f'(b) Prediction Error\nRMSE = {pred_rmse:.4f}', 
                      fontsize=13, fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, extend='both')
        
        # (c) Improvement Map
        ax3 = fig.add_subplot(133, projection=ccrs.PlateCarree())
        ax3.set_extent(extent, crs=ccrs.PlateCarree())
        ax3.add_feature(cfeature.COASTLINE, linewidth=1.0)
        ax3.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
        
        improvement_pct = (improvement > 0).mean() * 100
        im3 = ax3.pcolormesh(lon_grid, lat_grid, improvement, cmap='RdYlGn',
                              vmin=-vmax_imp, vmax=vmax_imp,
                              transform=ccrs.PlateCarree(), shading='auto')
        ax3.set_title(f'(c) Error Reduction\n{improvement_pct:.1f}% pixels improved', 
                      fontsize=13, fontweight='bold')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, extend='both')
        cbar3.set_label('|Bkg Error| − |Pred Error|', fontsize=10)
        
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # 简化版本...
        pass
    
    plt.suptitle(f'Error Analysis: Level {level_idx} | Sample {sample_idx}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 保存: {output_path}")


# =============================================================================
# Part 7: 可视化函数 - 散点图（改进版）
# =============================================================================
def plot_scatter_comparison(
    pred: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    title: str = 'Scatter Plot: Prediction vs Truth'
):
    """绘制散点图（带密度着色 + 误差分布）"""
    
    # 数据准备
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]
    
    n_samples = min(100000, len(pred_flat))
    indices = np.random.choice(len(pred_flat), n_samples, replace=False)
    pred_sample = pred_flat[indices]
    target_sample = target_flat[indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # =========== (a) 密度散点图 ===========
    ax1 = axes[0]
    
    h = ax1.hist2d(
        target_sample, pred_sample,
        bins=150, cmap='YlOrRd',
        norm=colors.LogNorm(vmin=1, vmax=None)
    )
    
    # 1:1线
    lims = [min(target_sample.min(), pred_sample.min()),
            max(target_sample.max(), pred_sample.max())]
    ax1.plot(lims, lims, 'b--', linewidth=2, alpha=0.8, label='1:1 Line')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    
    # 回归线
    z = np.polyfit(target_sample, pred_sample, 1)
    p = np.poly1d(z)
    ax1.plot(lims, p(lims), 'g-', linewidth=2, alpha=0.8, 
             label=f'Fit: y = {z[0]:.3f}x + {z[1]:.3f}')
    
    # 计算指标
    rmse = np.sqrt(np.mean((pred_sample - target_sample) ** 2))
    corr = np.corrcoef(pred_sample, target_sample)[0, 1]
    bias = np.mean(pred_sample - target_sample)
    mae = np.mean(np.abs(pred_sample - target_sample))
    
    stats_text = (f'N = {n_samples:,}\n'
                  f'RMSE = {rmse:.4f}\n'
                  f'MAE = {mae:.4f}\n'
                  f'r = {corr:.4f}\n'
                  f'Bias = {bias:.4f}')
    
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    ax1.set_xlabel('Truth', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Prediction', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Density Scatter Plot', fontsize=16, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    plt.colorbar(h[3], ax=ax1, label='Density (log)')
    
    # =========== (b) 误差分布 ===========
    ax2 = axes[1]
    errors = pred_sample - target_sample
    
    ax2.hist(errors, bins=100, density=True, alpha=0.7, 
             color='steelblue', edgecolor='black', linewidth=0.3)
    
    # 拟合正态分布
    mu, std = stats.norm.fit(errors)
    x = np.linspace(errors.min(), errors.max(), 200)
    ax2.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2.5,
             label=f'Normal: μ={mu:.4f}, σ={std:.4f}')
    
    ax2.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.8)
    ax2.axvline(mu, color='red', linewidth=1.5, linestyle=':', alpha=0.8)
    
    ax2.set_xlabel('Error (Prediction − Truth)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax2.set_title('(b) Error Distribution', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 保存: {output_path}")


# =============================================================================
# Part 8: 可视化函数 - 相关性分析（改进版）
# =============================================================================
def plot_correlation_analysis(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    pressure_levels: np.ndarray,
    output_path: Path
):
    """
    绘制相关性分析（3合1: 廓线图 + 热力图 + 条形图）
    """
    n_models = len(results_dict)
    n_levels = list(results_dict.values())[0]['pred'].shape[1]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    colors_map = {'Background': '#1f77b4', 'Baseline': '#ff7f0e', 'Ours': '#d62728'}
    markers = {'Background': 's', 'Baseline': '^', 'Ours': 'o'}
    
    # 计算所有模型的相关系数
    all_corrs = {}
    for model_name, results in results_dict.items():
        pred = results['pred']
        target = results['target']
        
        corr_per_level = []
        for level in range(n_levels):
            pred_l = pred[:, level, :, :].flatten()
            target_l = target[:, level, :, :].flatten()
            
            mask = ~(np.isnan(pred_l) | np.isnan(target_l))
            if mask.sum() > 100:
                corr = np.corrcoef(pred_l[mask], target_l[mask])[0, 1]
            else:
                corr = np.nan
            corr_per_level.append(corr)
        
        all_corrs[model_name] = np.array(corr_per_level)
    
    # =========== (a) 相关系数廓线图 ===========
    ax1 = axes[0]
    
    for model_name, corrs in all_corrs.items():
        ax1.plot(
            corrs, pressure_levels,
            marker=markers.get(model_name, 'o'),
            markersize=5, linewidth=2,
            color=colors_map.get(model_name),
            label=model_name,
            markerfacecolor='white', markeredgewidth=1.5
        )
    
    # 动态设置x轴范围
    all_corr_values = np.concatenate([c for c in all_corrs.values() if not np.all(np.isnan(c))])
    corr_min = np.nanmin(all_corr_values)
    x_min = max(0, corr_min - 0.05)
    
    ax1.set_xlabel('Correlation Coefficient', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Pressure (hPa)', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Correlation vs Altitude', fontsize=16, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_yscale('log')
    ax1.set_xlim(x_min, 1.0)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='lower left', fontsize=11)
    
    # =========== (b) 相关系数热力图 ===========
    ax2 = axes[1]
    
    model_names = list(all_corrs.keys())
    corr_matrix = np.array([all_corrs[m] for m in model_names]).T  # [n_levels, n_models]
    
    # 动态颜色范围
    vmin = max(0.0, np.nanmin(corr_matrix) - 0.1)
    vmax = 1.0
    
    im = ax2.imshow(corr_matrix, cmap='RdYlGn', vmin=vmin, vmax=vmax, 
                    aspect='auto', interpolation='nearest')
    
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, fontsize=12, rotation=30, ha='right')
    ax2.set_ylabel('Vertical Level Index', fontsize=12)
    ax2.set_title('(b) Correlation Heatmap', fontsize=16, fontweight='bold')
    
    # 数值标注（每隔几行标注一次避免拥挤）
    step = max(1, n_levels // 15)
    for i in range(0, n_levels, step):
        for j in range(len(model_names)):
            val = corr_matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < (vmin + vmax) / 2 else 'black'
                ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                         fontsize=7, color=text_color, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046)
    cbar.set_label('Correlation', fontsize=11)
    
    # =========== (c) 关键层级RMSE对比 ===========
    ax3 = axes[2]
    
    # 选择关键层级
    key_levels = [0, n_levels//4, n_levels//2, 3*n_levels//4, n_levels-1]
    key_levels = sorted(set([min(l, n_levels-1) for l in key_levels]))
    
    x = np.arange(len(key_levels))
    width = 0.8 / n_models
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        pred = results['pred']
        target = results['target']
        
        rmse_at_levels = []
        for level in key_levels:
            rmse = np.sqrt(np.mean((pred[:, level] - target[:, level]) ** 2))
            rmse_at_levels.append(rmse)
        
        offset = (i - n_models/2 + 0.5) * width
        bars = ax3.bar(x + offset, rmse_at_levels, width * 0.9,
                       label=model_name, color=colors_map.get(model_name), alpha=0.85)
        
        # 添加数值标注
        for bar, val in zip(bars, rmse_at_levels):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax3.set_xlabel('Pressure Level', fontsize=14, fontweight='bold')
    ax3.set_ylabel('RMSE', fontsize=14, fontweight='bold')
    ax3.set_title('(c) RMSE at Key Levels', fontsize=16, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{pressure_levels[l]:.0f}' for l in key_levels], 
                         rotation=45, ha='right')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 保存: {output_path}")


# =============================================================================
# Part 9: 综合指标表格
# =============================================================================
def plot_metrics_table(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    output_path: Path
):
    """绘制综合指标表格"""
    
    metrics_data = []
    bkg_rmse_all = None
    
    # 先计算Background的RMSE作为基准
    if 'Background' in results_dict:
        bkg = results_dict['Background']
        bkg_rmse_all = np.sqrt(np.mean((bkg['pred'] - bkg['target']) ** 2))
    
    for model_name, results in results_dict.items():
        pred = results['pred'].flatten()
        target = results['target'].flatten()
        
        mask = ~(np.isnan(pred) | np.isnan(target))
        pred = pred[mask]
        target = target[mask]
        
        rmse = np.sqrt(np.mean((pred - target) ** 2))
        mae = np.mean(np.abs(pred - target))
        bias = np.mean(pred - target)
        corr = np.corrcoef(pred, target)[0, 1]
        
        # 计算改进率
        if bkg_rmse_all is not None and model_name != 'Background':
            improvement = (bkg_rmse_all - rmse) / bkg_rmse_all * 100
        else:
            improvement = 0.0
        
        metrics_data.append({
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'Bias': bias,
            'Corr': corr,
            'Improv': improvement
        })
    
    # 创建表格
    fig, ax = plt.subplots(figsize=(14, 2 + 0.5 * len(metrics_data)))
    ax.axis('off')
    
    headers = ['Model', 'RMSE', 'MAE', 'Bias', 'Correlation', 'RMSE Improv. (%)']
    
    cell_text = []
    for d in metrics_data:
        row = [
            d['Model'],
            f"{d['RMSE']:.5f}",
            f"{d['MAE']:.5f}",
            f"{d['Bias']:.5f}",
            f"{d['Corr']:.5f}",
            f"{d['Improv']:.1f}" if d['Model'] != 'Background' else '—'
        ]
        cell_text.append(row)
    
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)
    
    # 表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 高亮最优值
    for row_idx, d in enumerate(metrics_data, start=1):
        # 绿色高亮最好的模型（非Background）
        if d['Model'] == 'Ours':
            for col_idx in range(len(headers)):
                table[(row_idx, col_idx)].set_facecolor('#d5f5e3')
    
    plt.title('Comprehensive Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 保存: {output_path}")


# =============================================================================
# Part 10: 模型加载
# =============================================================================
def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """加载模型"""
    from models.backbone import create_model, UNetConfig

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = checkpoint.get('args', {})

    if args.get('model', 'physics_unet') != 'vanilla_unet':
        config = UNetConfig(
            fusion_mode=args.get('fusion_mode', 'gated'),
            use_aux=args.get('use_aux', True),
            mask_aware=args.get('mask_aware', True)
        )
        model = create_model(args.get('model', 'physics_unet'), config=config)
    else:
        model = create_model('vanilla_unet')

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


# =============================================================================
# Part 11: 主函数
# =============================================================================
def main():
    """主函数"""
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("🚀 推理分析脚本 (改进版)")
    print("=" * 70)
    print(f"  Cartopy 地图支持: {'✓ 已启用' if HAS_CARTOPY else '✗ 未安装'}")
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")
    
    # 地理范围
    extent = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    print(f"  地理范围: {extent}")
    print("=" * 70 + "\n")

    # =========================================================================
    # 加载模型
    # =========================================================================
    print("📦 加载模型...")
    model = load_model(args.checkpoint, device)
    print("  ✓ 主模型加载完成")

    baseline_model = None
    if args.baseline_checkpoint:
        baseline_model = load_model(args.baseline_checkpoint, device)
        print("  ✓ 基准模型加载完成")

    # =========================================================================
    # 加载数据
    # =========================================================================
    print("\n📂 加载测试数据...")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise ValueError(f"数据目录不存在: {data_root}")

    file_list = sorted(data_root.glob('**/*.npz'))
    if not file_list:
        raise ValueError(f"未找到.npz文件: {data_root}")

    dataset = LazySatelliteERA5Dataset(
        file_list=[str(f) for f in file_list],
        use_aux=True
    )

    if args.stats_file and Path(args.stats_file).exists():
        stats = np.load(args.stats_file)
        dataset.obs_normalizer = LevelwiseNormalizer(
            stats['obs_mean'], stats['obs_std'], name='obs')
        dataset.bkg_normalizer = LevelwiseNormalizer(
            stats['bkg_mean'], stats['bkg_std'], name='bkg')
        dataset.target_normalizer = LevelwiseNormalizer(
            stats['target_mean'], stats['target_std'], name='target')
    else:
        dataset.compute_statistics(n_samples=min(1000, len(dataset)))

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f"  样本数: {len(dataset)}")

    # =========================================================================
    # 推理
    # =========================================================================
    print("\n🔮 开始推理...")

    normalizers = {
        'obs': dataset.obs_normalizer,
        'bkg': dataset.bkg_normalizer,
        'target': dataset.target_normalizer
    }

    engine = InferenceEngine(model, device, normalizers)
    results = engine.evaluate_dataset(dataloader)
    print(f"  ✓ 推理完成 | Shape: {results['pred'].shape}")

    # 构建对比字典
    results_dict = {'Ours': results}
    
    if baseline_model:
        baseline_engine = InferenceEngine(baseline_model, device, normalizers)
        results_dict['Baseline'] = baseline_engine.evaluate_dataset(dataloader)

    results_dict['Background'] = {
        'pred': results['bkg'],
        'target': results['target']
    }

    # =========================================================================
    # 生成图表
    # =========================================================================
    print("\n📊 生成图表...")

    n_levels = results['pred'].shape[1]
    pressure_levels = np.linspace(1000, 1, n_levels)
    level_idx = min(args.level_idx, n_levels - 1)

    # 1. 垂直RMSE廓线图
    print("  [1/6] 垂直RMSE廓线图...")
    plot_vertical_rmse_profile(results_dict, pressure_levels,
                               output_dir / 'vertical_rmse_profile.png')

    # 2. 偏差空间分布图
    print("  [2/6] 偏差空间分布图...")
    plot_spatial_bias_map(results['pred'], results['target'],
                          level_idx, args.case_study_idx,
                          output_dir / 'spatial_bias_map.png', extent)

    # 3. 个例可视化
    print("  [3/6] 个例可视化（含误差分析）...")
    plot_case_study(results['obs'], results['bkg'], 
                    results['target'], results['pred'],
                    level_idx, args.case_study_idx,
                    output_dir / 'case_study.png', extent)

    # 4. 散点图
    print("  [4/6] 散点图对比...")
    plot_scatter_comparison(results['pred'], results['target'],
                            output_dir / 'scatter_plot.png')

    # 5. 相关性分析
    print("  [5/6] 相关性分析...")
    plot_correlation_analysis(results_dict, pressure_levels,
                              output_dir / 'correlation_analysis.png')

    # 6. 指标表格
    print("  [6/6] 综合指标表格...")
    plot_metrics_table(results_dict, output_dir / 'metrics_table.png')

    print("\n" + "=" * 70)
    print("✅ 分析完成!")
    print(f"📁 图表保存至: {output_dir.absolute()}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()