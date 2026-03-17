#!/usr/bin/env python3
"""
===============================================================================
推理分析脚本 V3 (Inference Analysis for Paper Figures)
===============================================================================

核心修复 (相对于 V2):
  1. 数据反归一化: 所有绘图/指标均在真实物理空间计算 (e.g. 温度 K)
  2. 垂直气压层: 使用真实 ERA5 37层数组, 支持自适应检测 137层
  3. 地理坐标: 支持从 batch 读取真实 lat/lon; 分辨率推导; 无效值 Mask
  4. 视觉专业性: 符合气象期刊审稿标准 (TwoSlopeNorm, 对数反转Y轴等)

依赖:
    pip install torch numpy matplotlib scipy seaborn cartopy

用法:
    python inference_analysis_v3.py \
        --checkpoint outputs/experiment/best_model.pth \
        --data_root /path/to/test/data \
        --output_dir figures \
        --stats_file /path/to/stats.npz \
        --lon_min 70 --lon_max 140 --lat_min 15 --lat_max 55 \
        --resolution 0.25
===============================================================================
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')  # 无头模式, 避免 DISPLAY 问题
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm, LogNorm
import seaborn as sns
from scipy import stats as sp_stats

# =============================================================================
# Cartopy 检测
# =============================================================================
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("⚠️  Cartopy 未安装, 将使用简化地图.  pip install cartopy")

# =============================================================================
# 项目路径 — 按需修改
# =============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

# =============================================================================
# 全局 Matplotlib 配置 (期刊标准)
# =============================================================================
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['DejaVu Sans', 'Arial', 'Helvetica'],
    'mathtext.fontset':   'dejavusans',
    'axes.unicode_minus': False,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'font.size':          12,
    'axes.labelsize':     14,
    'axes.titlesize':     15,
    'legend.fontsize':    11,
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
})

# =============================================================================
# 项目模块导入 (与 data_pipeline_v2 对齐)
# =============================================================================
from data_pipeline_v2 import (
    LazySatelliteERA5Dataset,
    InMemorySatelliteDataset,
    LevelwiseNormalizer,
    AssimilationMetrics,
)

TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')

# =============================================================================
# 真实 ERA5 气压层 (hPa)
# =============================================================================
ERA5_37_LEVELS = np.array([
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
     750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
     250, 225, 200, 175, 150, 125, 100,  70,  50,  30,
      20,  10,   7,   5,   3,   2,   1,
], dtype=np.float64)

ERA5_137_LEVELS_APPROX = np.array([
    # 仅列出代表性值; 实际应从 ECMWF L137 表查取
    # 这里用对数等距近似, 绘图时一律采用 log Y 轴
    *np.round(np.logspace(np.log10(1013.25), np.log10(0.01), 137), 4)
])


def get_pressure_levels(n_levels: int) -> np.ndarray:
    """
    根据模型层数返回对应的气压层数组.
    - 37 层 → ERA5 标准 37 层
    - 其他  → 对数等距近似 (绘图时使用 log Y 轴)
    """
    if n_levels == 37:
        return ERA5_37_LEVELS.copy()
    if n_levels == 137:
        return np.round(np.logspace(np.log10(1013.25), np.log10(0.01), 137), 4)
    # fallback: 对数等距
    warnings.warn(
        f"未知层数 {n_levels}, 使用对数等距气压层近似. "
        "建议在配置文件中指定真实气压层."
    )
    return np.round(np.logspace(np.log10(1013.25), np.log10(0.01), n_levels), 4)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Part 1: 参数解析                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Inference Analysis V3 — 反归一化 + 真实气压层 + 期刊标准可视化',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint',   type=str, default="/home/lrx/Unet/satellite_assimilation_v2/train_ddp/outputs/ours_noaux_full_128/best_model.pth",required=True,  help='模型 checkpoint')
    p.add_argument('--data_root',    type=str, required=True,  help='测试数据根目录')
    p.add_argument('--output_dir',   type=str, default=f'figures_v3/{TIMESTAMP}')
    p.add_argument('--stats_file',   type=str, default=None,   help='标准化统计量 .npz')
    p.add_argument('--baseline_checkpoint', type=str, default=None)

    p.add_argument('--batch_size',   type=int,   default=16)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--device',       type=str,   default='cuda')
    p.add_argument('--case_study_idx', type=int, default=0)
    p.add_argument('--level_idx',    type=int,   default=10,
                   help='可视化的垂直层索引 (0-based)')

    # 地理范围
    p.add_argument('--lon_min', type=float, default=70.0)
    p.add_argument('--lon_max', type=float, default=140.0)
    p.add_argument('--lat_min', type=float, default=15.0)
    p.add_argument('--lat_max', type=float, default=55.0)
    p.add_argument('--resolution', type=float, default=0.25,
                   help='数据空间分辨率 (°), 用于自动推导网格')

    # 物理量
    p.add_argument('--variable_name', type=str, default='Temperature',
                   help='物理量名称, 用于标注')
    p.add_argument('--variable_unit', type=str, default='K',
                   help='物理量单位')
   
    # ... 已有参数保持不变 ...

    # ========== 新增: 空间转置开关 ==========
    p.add_argument('--transpose_spatial', action='store_true',
                   help='如果输出图像出现横纵条纹/马赛克, '
                        '使用此参数转置空间维度 (H,W) → (W,H)')

    # ========== 增量模式 ==========
    p.add_argument('--use_increment', action='store_true',
                   help='增量模型推理: analysis = bkg_phys + pred_inc_phys')
    p.add_argument('--increment_stats', type=str, default='',
                   help='增量统计量文件 (.npz, 含inc_mean/inc_std)')

    return p.parse_args()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Part 2: 坐标网格工具                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def create_coordinate_grids(
    data_shape: Tuple[int, int],
    extent: Tuple[float, float, float, float],
    resolution: Optional[float] = None,
    lat_array: Optional[np.ndarray] = None,
    lon_array: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建经纬度 2-D 网格.

    优先级:
      1. 直接使用提供的 lat_array / lon_array (最准确)
      2. 根据 resolution 从 extent 推导 (避免拉伸)
      3. 回退: linspace (仅当 1,2 均不可用时)

    Returns
    -------
    lon2d, lat2d : ndarray, shape (H, W)
    """
    H, W = data_shape

    if lat_array is not None and lon_array is not None:
        # 情形 1: 真实坐标
        if lat_array.ndim == 1 and lon_array.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon_array, lat_array)
        else:
            lon2d, lat2d = lon_array, lat_array
        return lon2d, lat2d

    lon_min, lon_max, lat_min, lat_max = extent

    if resolution is not None and resolution > 0:
        # 情形 2: 由分辨率推导, 保证像素与物理间距一致
        n_lon = int(round((lon_max - lon_min) / resolution))
        n_lat = int(round((lat_max - lat_min) / resolution))
        # 如果推导出的网格尺寸与数据不匹配, 发出警告后回退
        if n_lon != W or n_lat != H:
            warnings.warn(
                f"分辨率 {resolution}° 推导网格 ({n_lat}×{n_lon}) 与数据 ({H}×{W}) 不一致, "
                "将使用 linspace 回退."
            )
            lons = np.linspace(lon_min, lon_max, W)
            lats = np.linspace(lat_max, lat_min, H)   # 纬度从北到南
        else:
            lons = np.linspace(lon_min, lon_max, W, endpoint=False) + resolution / 2
            lats = np.linspace(lat_max, lat_min, H, endpoint=False) - resolution / 2
    else:
        # 情形 3: 回退
        lons = np.linspace(lon_min, lon_max, W)
        lats = np.linspace(lat_max, lat_min, H)

    lon2d, lat2d = np.meshgrid(lons, lats)
    return lon2d, lat2d


def mask_invalid(data: np.ndarray, fill_values=(-9999, 0)) -> np.ndarray:
    """
    将无效值 (NaN / 指定 fill_values / 极端值) 替换为 np.nan,
    以便绘图时 Mask 掉, 不影响色标.
    """
    out = data.astype(np.float64).copy()
    out[~np.isfinite(out)] = np.nan
    for fv in fill_values:
        out[out == fv] = np.nan
    return out


def robust_vlim(data: np.ndarray, pct: float = 2.0):
    """返回去掉上下 pct% 极值后的 (vmin, vmax)"""
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        return -1, 1
    lo = np.percentile(valid, pct)
    hi = np.percentile(valid, 100 - pct)
    return float(lo), float(hi)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Part 3: 推理引擎 (含反归一化)                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class InferenceEngine:
    """
    推理引擎 — 在返回结果前执行 **反归一化**,
    确保后续绘图/指标均在真实物理空间.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        target_normalizer: LevelwiseNormalizer,
        bkg_normalizer: LevelwiseNormalizer,
        obs_normalizer: Optional[LevelwiseNormalizer] = None,
        inc_normalizer: Optional[LevelwiseNormalizer] = None,
    ):
        self.model = model
        self.device = device
        self.target_norm = target_normalizer
        self.bkg_norm = bkg_normalizer
        self.obs_norm = obs_normalizer
        self.inc_norm = inc_normalizer   # 非None时启用增量推理

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def predict_batch(self, obs, bkg, mask, aux=None):
        self.model.eval()
        obs  = obs.to(self.device)
        bkg  = bkg.to(self.device)
        mask = mask.to(self.device)
        if aux is not None:
            aux = aux.to(self.device)
            pred = self.model(obs, bkg, mask, aux)
        else:
            pred = self.model(obs, bkg, mask)
        return pred.cpu()

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        denormalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        在整个数据集上推理, 并可选地执行反归一化.

        返回字典键:
            pred, target, bkg, obs, mask
            — 均为 numpy, shape (N, C, H, W)
            — 若 denormalize=True, pred/target/bkg 已还原为真实物理量
        """
        self.model.eval()
        all_pred, all_target, all_bkg, all_obs, all_mask = [], [], [], [], []

        for batch in dataloader:
            obs    = batch['obs']
            bkg    = batch['bkg']
            mask   = batch['mask']
            target = batch['target']
            aux    = batch.get('aux', None)

            pred = self.predict_batch(obs, bkg, mask, aux)

            all_pred.append(pred.numpy())
            all_target.append(target.numpy())
            all_bkg.append(bkg.numpy())
            all_obs.append(obs.numpy())
            all_mask.append(mask.numpy())

        result = {
            'pred':   np.concatenate(all_pred,   axis=0),
            'target': np.concatenate(all_target, axis=0),
            'bkg':    np.concatenate(all_bkg,    axis=0),
            'obs':    np.concatenate(all_obs,    axis=0),
            'mask':   np.concatenate(all_mask,   axis=0),
        }

        # ===================== 核心: 反归一化 ===================== #
        if denormalize:
            if self.inc_norm is not None:
                # 增量模式: analysis = bkg_phys + inc_phys
                bkg_phys = self.bkg_norm.inverse_transform(result['bkg'])
                inc_phys = self.inc_norm.inverse_transform(result['pred'])
                result['pred']   = bkg_phys + inc_phys          # analysis
                result['target'] = self.target_norm.inverse_transform(result['target'])
                result['bkg']    = bkg_phys
                print("  ✓ 增量推理: analysis = bkg + pred_increment")
            else:
                result['pred']   = self.target_norm.inverse_transform(result['pred'])
                result['target'] = self.target_norm.inverse_transform(result['target'])
                result['bkg']    = self.bkg_norm.inverse_transform(result['bkg'])
            if self.obs_norm is not None:
                result['obs'] = self.obs_norm.inverse_transform(result['obs'])
            print("  ✓ 反归一化完成 — 数据已还原为真实物理量")
        # =========================================================== #

        return result


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Part 4: Geo-Axes 工厂                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _add_geo_features(ax, extent):
    """给 Cartopy GeoAxes 添加海岸线/国界/网格线"""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE,  linewidth=1.0, edgecolor='k')
    ax.add_feature(cfeature.BORDERS,    linewidth=0.6, linestyle='--', edgecolor='dimgray')
    ax.add_feature(cfeature.LAND,       facecolor='#f0f0f0', alpha=0.25)
    try:
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
    except Exception:
        pass
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    return gl


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Part 5: 垂直 RMSE 廓线 (气象期刊标准)                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def plot_vertical_rmse_profile(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    pressure_levels: np.ndarray,
    output_path: Path,
    unit: str = 'K',
):
    """
    期刊标准垂直 RMSE 廓线图:
      - X 轴在上方 (tick_top)
      - Y 轴对数反转 (1000 → 1 hPa)
      - 带对流层顶标注
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True)

    palette = {'Background': '#4575b4', 'Baseline': '#fc8d59', 'Ours': '#d73027'}
    markers = {'Background': 's', 'Baseline': '^', 'Ours': 'o'}
    rmse_dict: Dict[str, np.ndarray] = {}

    n_levels = len(pressure_levels)

    # ---------- (a) RMSE 廓线 ----------
    ax = axes[0]
    for name, res in results_dict.items():
        pred, tgt = res['pred'], res['target']
        n_lev = min(pred.shape[1], n_levels)
        rmse_per_lev = np.array([
            np.sqrt(np.nanmean((pred[:, l] - tgt[:, l]) ** 2))
            for l in range(n_lev)
        ])
        rmse_dict[name] = rmse_per_lev
        ax.plot(rmse_per_lev, pressure_levels[:n_lev],
                marker=markers.get(name, 'o'), ms=4, lw=2,
                color=palette.get(name, 'gray'), label=name,
                markerfacecolor='white', markeredgewidth=1.5)

    ax.axhline(100, color='purple', ls=':', lw=1.2, label='Tropopause (~100 hPa)')

    # 气象期刊标准: X 轴置顶, Y 对数反转
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(f'RMSE ({unit})', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pressure (hPa)', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylim(pressure_levels[0], pressure_levels[-1])
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_title('(a) Vertical RMSE Profile', fontsize=14, fontweight='bold', pad=40)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    # ---------- (b) 改进百分比 ----------
    ax2 = axes[1]
    if 'Background' in rmse_dict and 'Ours' in rmse_dict:
        bkg_r = rmse_dict['Background']
        our_r = rmse_dict['Ours']
        n_common = min(len(bkg_r), len(our_r))
        improv = (bkg_r[:n_common] - our_r[:n_common]) / (bkg_r[:n_common] + 1e-12) * 100

        bar_colors = ['#27ae60' if v > 0 else '#e74c3c' for v in improv]
        # barh 需要 y 位置
        ax2.barh(pressure_levels[:n_common], improv, color=bar_colors,
                 alpha=0.75, height=0.03 * pressure_levels[:n_common])
        ax2.axvline(0, color='k', lw=1.2)
        mean_imp = np.nanmean(improv)
        ax2.axvline(mean_imp, color='navy', lw=1.8, ls='--',
                    label=f'Mean: {mean_imp:.1f}%')

        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        ax2.set_xlabel('RMSE Improvement (%)', fontsize=13, fontweight='bold')
        ax2.set_title('(b) Improvement over Background', fontsize=14,
                      fontweight='bold', pad=40)
        ax2.set_yscale('log')
        ax2.invert_yaxis()
        ax2.set_ylim(pressure_levels[0], pressure_levels[-1])
        ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax2.grid(True, which='both', alpha=0.25, axis='x')
        ax2.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_path}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  替换内容: plot_spatial_bias_map / plot_case_study / _plot_error_comparison║
# ║  核心改动:                                                                ║
# ║    1. pcolormesh → contourf (levels≥50) + fallback pcolormesh gouraud     ║
# ║    2. min/max → np.nanpercentile 鲁棒色标                                 ║
# ║    3. TwoSlopeNorm(vcenter=0) 严格零值居中                                ║
# ║    4. 海岸线/国界 zorder=10, 网格线 alpha=0.3                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# =============================================================================
# 通用辅助: 平滑绘制引擎
# =============================================================================

def _smooth_plot(
    ax,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    data: np.ndarray,
    cmap: str = 'jet',
    vmin: float = None,
    vmax: float = None,
    norm=None,
    n_contour_levels: int = 60,
    use_cartopy: bool = False,
):
    """
    平滑绘制引擎: 优先 contourf, 失败则 fallback.

    Parameters
    ----------
    ax : matplotlib Axes (或 GeoAxes)
    lon2d, lat2d : 2-D coordinate grids
    data : 2-D field to plot
    cmap : colormap name
    vmin, vmax : color limits (ignored if norm is provided)
    norm : matplotlib Normalize instance (e.g. TwoSlopeNorm)
    n_contour_levels : contourf 等级数, 越大越平滑
    use_cartopy : 是否传 transform=PlateCarree()

    Returns
    -------
    mappable : QuadMesh 或 QuadContourSet, 可用于 colorbar
    """
    transform_kw = dict(transform=ccrs.PlateCarree()) if (use_cartopy and HAS_CARTOPY) else {}

    # ---------- 准备 contourf levels ----------
    if norm is not None:
        # 对于 TwoSlopeNorm, 从 vmin/vmax 生成等距 levels
        _vmin = norm.vmin
        _vmax = norm.vmax
    else:
        _vmin = vmin
        _vmax = vmax

    if _vmin is not None and _vmax is not None and np.isfinite(_vmin) and np.isfinite(_vmax):
        levels = np.linspace(_vmin, _vmax, n_contour_levels)
    else:
        # 自动
        valid = data[np.isfinite(data)]
        if len(valid) == 0:
            levels = np.linspace(-1, 1, n_contour_levels)
        else:
            levels = np.linspace(np.nanpercentile(valid, 1),
                                 np.nanpercentile(valid, 99),
                                 n_contour_levels)

    # ---------- 尝试 contourf (最平滑) ----------
    try:
        # contourf 不接受含 NaN 的数据, 先用 0 填充 (被 levels 裁剪后不影响)
        data_filled = np.where(np.isfinite(data), data, np.nanmean(data[np.isfinite(data)]))

        if norm is not None:
            mappable = ax.contourf(
                lon2d, lat2d, data_filled,
                levels=levels, cmap=cmap, norm=norm,
                extend='both', **transform_kw
            )
        else:
            mappable = ax.contourf(
                lon2d, lat2d, data_filled,
                levels=levels, cmap=cmap,
                vmin=vmin, vmax=vmax,
                extend='both', **transform_kw
            )
        return mappable

    except Exception as e:
        warnings.warn(f"contourf 失败 ({e}), 回退到 pcolormesh(gouraud)")

    # ---------- fallback: pcolormesh + gouraud ----------
    try:
        data_filled = np.where(np.isfinite(data), data, np.nan)
        kw = dict(cmap=cmap, shading='gouraud', rasterized=True, **transform_kw)
        if norm is not None:
            kw['norm'] = norm
        else:
            if vmin is not None:
                kw['vmin'] = vmin
            if vmax is not None:
                kw['vmax'] = vmax

        mappable = ax.pcolormesh(lon2d, lat2d, data_filled, **kw)
        return mappable

    except Exception:
        pass

    # ---------- 最终 fallback: imshow + bilinear ----------
    extent_img = [lon2d.min(), lon2d.max(), lat2d.min(), lat2d.max()]
    kw_im = dict(cmap=cmap, origin='upper', aspect='auto',
                 extent=extent_img, interpolation='bilinear')
    if norm is not None:
        kw_im['norm'] = norm
    else:
        if vmin is not None:
            kw_im['vmin'] = vmin
        if vmax is not None:
            kw_im['vmax'] = vmax

    if use_cartopy and HAS_CARTOPY:
        kw_im['transform'] = ccrs.PlateCarree()

    data_filled = np.where(np.isfinite(data), data, np.nanmean(data[np.isfinite(data)]))
    mappable = ax.imshow(data_filled, **kw_im)
    return mappable


def _add_geo_features(ax, extent, coastline_lw=1.2, border_lw=0.7):
    """
    给 GeoAxes 添加海岸线/国界/网格线.
    海岸线和国界设置高 zorder 确保不被覆盖.
    """
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # zorder=10 确保线条在填色之上
    ax.add_feature(cfeature.COASTLINE, linewidth=coastline_lw,
                   edgecolor='black', zorder=10)
    ax.add_feature(cfeature.BORDERS, linewidth=border_lw,
                   linestyle='--', edgecolor='dimgray', zorder=10)
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', alpha=0.15, zorder=1)

    try:
        ax.add_feature(cfeature.STATES, linewidth=0.3,
                       edgecolor='gray', zorder=10)
    except Exception:
        pass

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.4,
        color='gray', alpha=0.3, linestyle='--', zorder=5
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    return gl


# =============================================================================
# Part 6: 偏差空间分布图 (Spatial Bias Map) — 重构版
# =============================================================================
def plot_spatial_bias_map(
    pred: np.ndarray,
    target: np.ndarray,
    level_idx: int,
    sample_idx: int,
    output_path: Path,
    extent: Tuple[float, float, float, float],
    resolution: float = 0.25,
    unit: str = 'K',
    pressure_levels: Optional[np.ndarray] = None,
    lat_array: Optional[np.ndarray] = None,
    lon_array: Optional[np.ndarray] = None,
    transpose_data: bool = False,           # ← 新增
):
    """
    偏差空间分布图 (期刊标准)
    """
    # ---- 1. 数据准备 ----
    bias_raw = pred[sample_idx, level_idx] - target[sample_idx, level_idx]
    bias = mask_invalid(bias_raw, fill_values=(-9999,))

    # ======== 核心修复: 转置必须在取 H,W 之前 ========
    if transpose_data:
        bias = bias.T

    H, W = bias.shape
    lon2d, lat2d = create_coordinate_grids(
        (H, W), extent, resolution,
        lat_array=lat_array, lon_array=lon_array
    )

    # ---- 2. 鲁棒色标: 98th percentile, 零值居中 ----
    valid_bias = bias[np.isfinite(bias)]
    if len(valid_bias) == 0:
        vabs = 1.0
    else:
        vabs = max(np.nanpercentile(np.abs(valid_bias), 98), 1e-6)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs)

    plev_str = ''
    if pressure_levels is not None and level_idx < len(pressure_levels):
        plev_str = f' | {pressure_levels[level_idx]:.0f} hPa'

    # ---- 3. 绘图 ----
    fig = plt.figure(figsize=(12, 9))

    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        _add_geo_features(ax, extent)

        im = _smooth_plot(
            ax, lon2d, lat2d, bias,
            cmap='RdBu_r', norm=norm,
            n_contour_levels=60, use_cartopy=True
        )

        try:
            bias_filled = np.where(
                np.isfinite(bias), bias,
                np.nanmean(valid_bias) if len(valid_bias) > 0 else 0
            )
            clevels = np.linspace(-vabs, vabs, 11)
            cs = ax.contour(
                lon2d, lat2d, bias_filled,
                levels=clevels, colors='black',
                linewidths=0.4, alpha=0.5,
                transform=ccrs.PlateCarree(), zorder=8
            )
            ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
        except Exception:
            pass
    else:
        ax = fig.add_subplot(1, 1, 1)
        im = _smooth_plot(
            ax, lon2d, lat2d, bias,
            cmap='RdBu_r', norm=norm,
            n_contour_levels=60, use_cartopy=False
        )
        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)

    ax.set_title(
        f'Spatial Bias (Pred − Truth){plev_str}\nSample {sample_idx}',
        fontsize=15, fontweight='bold'
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        extend='both', shrink=0.9)
    cbar.set_label(f'Bias ({unit})', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    if len(valid_bias) > 0:
        txt = (
            f'Mean:  {np.mean(valid_bias):+.4f} {unit}\n'
            f'Std:   {np.std(valid_bias):.4f} {unit}\n'
            f'RMSE:  {np.sqrt(np.mean(valid_bias**2)):.4f} {unit}\n'
            f'Max:   {np.max(valid_bias):+.4f} {unit}\n'
            f'Min:   {np.min(valid_bias):+.4f} {unit}\n'
            f'|Bias|>1σ: {np.mean(np.abs(valid_bias) > np.std(valid_bias))*100:.1f}%'
        )
        ax.text(
            0.02, 0.98, txt, transform=ax.transAxes, fontsize=9.5,
            va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='white',
                      ec='gray', alpha=0.92, lw=0.8)
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_path}")
# =============================================================================
# Part 7: 个例研究 (Case Study) 四面板 — 重构版
# =============================================================================
def plot_case_study(
    obs: np.ndarray,
    bkg: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    level_idx: int,
    sample_idx: int,
    output_path: Path,
    extent: Tuple[float, float, float, float],
    resolution: float = 0.25,
    unit: str = 'K',
    pressure_levels: Optional[np.ndarray] = None,
    lat_array: Optional[np.ndarray] = None,
    lon_array: Optional[np.ndarray] = None,
    transpose_data: bool = False,           # ← 新增
):
    """
    个例四面板 (期刊标准)
    """
    # ---- 提取 2D 切片 ----
    obs_2d  = mask_invalid(obs[sample_idx, 0])
    bkg_2d  = mask_invalid(bkg[sample_idx, level_idx])
    tgt_2d  = mask_invalid(target[sample_idx, level_idx])
    pred_2d = mask_invalid(pred[sample_idx, level_idx])

    # ======== 核心修复: 先转置, 再取 H,W ========
    if transpose_data:
        obs_2d  = obs_2d.T
        bkg_2d  = bkg_2d.T
        tgt_2d  = tgt_2d.T
        pred_2d = pred_2d.T

    H, W = bkg_2d.shape
    lon2d, lat2d = create_coordinate_grids(
        (H, W), extent, resolution,
        lat_array=lat_array, lon_array=lon_array
    )

    # ---- 鲁棒色标 (1st–99th percentile) ----
    obs_valid = obs_2d[np.isfinite(obs_2d)]
    if len(obs_valid) > 0:
        obs_vmin = np.percentile(obs_valid, 1)
        obs_vmax = np.percentile(obs_valid, 99)
    else:
        obs_vmin, obs_vmax = 180, 320

    atm_valid = np.concatenate([
        bkg_2d[np.isfinite(bkg_2d)],
        tgt_2d[np.isfinite(tgt_2d)],
        pred_2d[np.isfinite(pred_2d)],
    ])
    if len(atm_valid) > 0:
        atm_vmin = np.percentile(atm_valid, 1)
        atm_vmax = np.percentile(atm_valid, 99)
    else:
        atm_vmin, atm_vmax = 200, 300

    plev_str = ''
    if pressure_levels is not None and level_idx < len(pressure_levels):
        plev_str = f'{pressure_levels[level_idx]:.0f} hPa'

    panels = [
        (obs_2d,  'viridis',   obs_vmin, obs_vmax,
         f'(a) Satellite Obs (Ch 1)', f'BT ({unit})'),
        (bkg_2d,  'RdYlBu_r',  atm_vmin, atm_vmax,
         f'(b) Background (NWP)',     f'{unit}'),
        (tgt_2d,  'RdYlBu_r',  atm_vmin, atm_vmax,
         f'(c) Analysis (Target)',    f'{unit}'),
        (pred_2d, 'RdYlBu_r',  atm_vmin, atm_vmax,
         f'(d) Prediction (Ours)',    f'{unit}'),
    ]

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(17, 15))
        for i, (data, cmap, vmin, vmax, title, clabel) in enumerate(panels, 1):
            ax = fig.add_subplot(2, 2, i, projection=ccrs.PlateCarree())
            _add_geo_features(ax, extent, coastline_lw=1.0, border_lw=0.5)
            im = _smooth_plot(
                ax, lon2d, lat2d, data,
                cmap=cmap, vmin=vmin, vmax=vmax,
                n_contour_levels=60, use_cartopy=True
            )
            ax.set_title(title, fontsize=13, fontweight='bold')
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                              extend='both', shrink=0.85)
            cb.set_label(clabel, fontsize=10)
            cb.ax.tick_params(labelsize=9)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 13))
        for ax_obj, (data, cmap, vmin, vmax, title, clabel) in zip(axes.flat, panels):
            im = _smooth_plot(
                ax_obj, lon2d, lat2d, data,
                cmap=cmap, vmin=vmin, vmax=vmax,
                n_contour_levels=60, use_cartopy=False
            )
            ax_obj.set_title(title, fontsize=13, fontweight='bold')
            ax_obj.set_xlabel('Lon (°E)')
            ax_obj.set_ylabel('Lat (°N)')
            cb = plt.colorbar(im, ax=ax_obj, fraction=0.046, extend='both')
            cb.set_label(clabel, fontsize=10)

    fig.suptitle(
        f'Case Study — Level {level_idx} ({plev_str})  |  Sample {sample_idx}',
        fontsize=17, fontweight='bold', y=1.0
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_path}")

    # ---- 误差三面板 (传递 transpose_data) ----
    _plot_error_comparison(
        bkg_2d, tgt_2d, pred_2d,        # 已经转置过的数据
        lon2d, lat2d, extent,
        output_path.parent / f'{output_path.stem}_error.png',
        level_idx, sample_idx, unit, plev_str,
        already_transposed=True,         # 告知不要重复转置
    )


def _plot_error_comparison(
    bkg: np.ndarray,
    tgt: np.ndarray,
    pred: np.ndarray,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    extent: Tuple[float, float, float, float],
    output_path: Path,
    level_idx: int,
    sample_idx: int,
    unit: str = 'K',
    plev_str: str = '',
    transpose_data: bool = False,       # ← 新增 (独立调用时使用)
    already_transposed: bool = False,   # ← 新增 (从 case_study 内部调用时为 True)
):
    """
    误差对比三面板:
      (a) Background Error   — RdBu_r, TwoSlopeNorm(0)
      (b) Prediction Error   — RdBu_r, TwoSlopeNorm(0)
      (c) Error Reduction    — RdYlGn, TwoSlopeNorm(0)
    """
    bkg_err  = bkg - tgt
    pred_err = pred - tgt
    improvement = np.abs(bkg_err) - np.abs(pred_err)

    # ======== 转置逻辑 ========
    # 如果从 case_study 传入的数据已经转置过, 跳过
    # 如果独立调用且 transpose_data=True, 则转置
    if transpose_data and not already_transposed:
        bkg_err     = bkg_err.T
        pred_err    = pred_err.T
        improvement = improvement.T
        # 需要重新生成网格
        H, W = bkg_err.shape
        lon2d, lat2d = create_coordinate_grids(
            (H, W), extent,
        )

    # ---- 鲁棒色标 ----
    valid_bkg_err  = bkg_err[np.isfinite(bkg_err)]
    valid_pred_err = pred_err[np.isfinite(pred_err)]
    valid_improv   = improvement[np.isfinite(improvement)]

    vabs_err = 1e-6
    for arr in [valid_bkg_err, valid_pred_err]:
        if len(arr) > 0:
            vabs_err = max(vabs_err, np.nanpercentile(np.abs(arr), 98))
    norm_err = TwoSlopeNorm(vcenter=0, vmin=-vabs_err, vmax=vabs_err)

    vabs_imp = 1e-6
    if len(valid_improv) > 0:
        vabs_imp = max(vabs_imp, np.nanpercentile(np.abs(valid_improv), 98))
    norm_imp = TwoSlopeNorm(vcenter=0, vmin=-vabs_imp, vmax=vabs_imp)

    bkg_rmse  = np.sqrt(np.nanmean(bkg_err**2))  if len(valid_bkg_err)  > 0 else 0
    pred_rmse = np.sqrt(np.nanmean(pred_err**2)) if len(valid_pred_err) > 0 else 0
    imp_pct   = np.nanmean(improvement > 0) * 100 if len(valid_improv) > 0 else 0

    triplets = [
        (bkg_err,     norm_err, 'RdBu_r',
         f'(a) Background Error\nRMSE = {bkg_rmse:.4f} {unit}',
         f'Error ({unit})'),
        (pred_err,    norm_err, 'RdBu_r',
         f'(b) Prediction Error\nRMSE = {pred_rmse:.4f} {unit}',
         f'Error ({unit})'),
        (improvement, norm_imp, 'RdYlGn',
         f'(c) Error Reduction\n{imp_pct:.1f}% pixels improved',
         f'|Bkg Err| − |Pred Err| ({unit})'),
    ]

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(20, 6.5))
        for i, (data, nrm, cmap, title, clabel) in enumerate(triplets, 1):
            ax = fig.add_subplot(1, 3, i, projection=ccrs.PlateCarree())
            _add_geo_features(ax, extent, coastline_lw=1.0, border_lw=0.5)
            im = _smooth_plot(
                ax, lon2d, lat2d, data,
                cmap=cmap, norm=nrm,
                n_contour_levels=60, use_cartopy=True
            )
            ax.set_title(title, fontsize=12, fontweight='bold')
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                              extend='both', shrink=0.85)
            cb.set_label(clabel, fontsize=9)
            cb.ax.tick_params(labelsize=8)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax_obj, (data, nrm, cmap, title, clabel) in zip(axes, triplets):
            im = _smooth_plot(
                ax_obj, lon2d, lat2d, data,
                cmap=cmap, norm=nrm,
                n_contour_levels=60, use_cartopy=False
            )
            ax_obj.set_title(title, fontsize=12, fontweight='bold')
            ax_obj.set_xlabel('Lon (°E)')
            ax_obj.set_ylabel('Lat (°N)')
            cb = plt.colorbar(im, ax=ax_obj, fraction=0.046, extend='both')
            cb.set_label(clabel, fontsize=9)

    fig.suptitle(
        f'Error Analysis — Level {level_idx} ({plev_str})  |  Sample {sample_idx}',
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_path}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Part 8: 散点图 (真实物理单位 + 对称区间)                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def plot_scatter_comparison(
    pred: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    unit: str = 'K',
    variable: str = 'Temperature',
    max_points: int = 100_000,
    physical_min: float = 100.0,    # 大气温度物理下限 (K)
    physical_max: float = 350.0,    # 大气温度物理上限 (K)
):
    """
    密度散点图 + 误差直方图.
    基于物理常识过滤无效值, 坐标轴标注真实单位与对称区间.

    Parameters
    ----------
    physical_min : float
        物理量合理下限. 大气温度不应低于 ~100 K (平流层顶极端低温 ~180 K,
        留足余量设为 100 K). 所有 < physical_min 的值视为无效填充残差.
    physical_max : float
        物理量合理上限. 地表极端高温不超过 ~330 K, 留余量设为 350 K.
    """
    # ================================================================
    # 1. 展平
    # ================================================================
    pf = pred.flatten().astype(np.float64)
    tf = target.flatten().astype(np.float64)

    # ================================================================
    # 2. 基于物理常识的合理阈值过滤 (核心修复)
    #    - 替代失效的 == 0 精确匹配
    #    - 同时滤除 NaN / Inf / 填充残差 / 离群极端值
    # ================================================================
    valid = (
        np.isfinite(pf) & np.isfinite(tf)       # 排除 NaN, Inf
        & (tf >= physical_min)                    # target 物理下限
        & (tf <= physical_max)                    # target 物理上限
        & (pf >= physical_min)                    # pred   物理下限
        & (pf <= physical_max)                    # pred   物理上限
    )

    pf = pf[valid]
    tf = tf[valid]

    n_total    = pred.size
    n_filtered = n_total - len(pf)
    print(f"    数据清洗: 总点数 {n_total:,} → 有效 {len(pf):,} "
          f"(滤除 {n_filtered:,}, {n_filtered/n_total*100:.2f}%)")

    if len(pf) < 100:
        print(f"    ⚠️ 有效点数不足 (<100), 跳过散点图")
        return

    # ================================================================
    # 3. 随机下采样 (保证性能)
    # ================================================================
    if len(pf) > max_points:
        idx = np.random.default_rng(42).choice(len(pf), max_points, replace=False)
        pf_plot, tf_plot = pf[idx], tf[idx]
    else:
        pf_plot, tf_plot = pf, tf

    # ================================================================
    # 4. 指标计算 (使用全部有效点, 非下采样)
    # ================================================================
    rmse = np.sqrt(np.mean((pf - tf) ** 2))
    mae  = np.mean(np.abs(pf - tf))
    bias = np.mean(pf - tf)
    corr = np.corrcoef(pf, tf)[0, 1]

    # ================================================================
    # 5. 绘图
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # -------- (a) 密度散点图 --------
    ax1 = axes[0]
    h = ax1.hist2d(
        tf_plot, pf_plot, bins=200, cmap='YlOrRd',
        norm=LogNorm(vmin=1), rasterized=True,
    )

    # 对称物理区间 (基于有效数据, 而非被 0 拉偏)
    lo = min(tf_plot.min(), pf_plot.min())
    hi = max(tf_plot.max(), pf_plot.max())
    margin = (hi - lo) * 0.02
    lims = [lo - margin, hi + margin]

    ax1.plot(lims, lims, 'b--', lw=1.8, alpha=0.8, label='1:1')

    # 回归线
    z = np.polyfit(tf_plot, pf_plot, 1)
    ax1.plot(lims, np.poly1d(z)(lims), 'g-', lw=1.8, alpha=0.8,
             label=f'Fit: y={z[0]:.4f}x{z[1]:+.2f}')

    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_aspect('equal')

    stats_txt = (
        f'N = {len(pf):,}\n'
        f'RMSE = {rmse:.4f} {unit}\n'
        f'MAE  = {mae:.4f} {unit}\n'
        f'Bias = {bias:+.4f} {unit}\n'
        f'r    = {corr:.5f}'
    )
    ax1.text(
        0.05, 0.95, stats_txt, transform=ax1.transAxes, fontsize=11,
        va='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.92),
    )

    ax1.set_xlabel(f'Truth ({unit})', fontsize=13, fontweight='bold')
    ax1.set_ylabel(f'Prediction ({unit})', fontsize=13, fontweight='bold')
    ax1.set_title(f'(a) {variable}: Prediction vs Truth',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.25)
    plt.colorbar(h[3], ax=ax1, label='Count (log scale)')

    # -------- (b) 误差分布 --------
    ax2 = axes[1]
    errors = pf_plot - tf_plot
    ax2.hist(errors, bins=120, density=True, alpha=0.7,
             color='steelblue', edgecolor='k', linewidth=0.2)
    mu, sigma = sp_stats.norm.fit(errors)
    xr = np.linspace(errors.min(), errors.max(), 300)
    ax2.plot(xr, sp_stats.norm.pdf(xr, mu, sigma), 'r-', lw=2.5,
             label=f'N(μ={mu:.4f}, σ={sigma:.4f})')
    ax2.axvline(0, color='k', lw=1.3, ls='--')
    ax2.axvline(mu, color='r', lw=1.3, ls=':')

    ax2.set_xlabel(f'Error ({unit})', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Error Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Part 9: 相关性分析 (廓线 + 热力图 + 条形图)                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def plot_correlation_analysis(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    pressure_levels: np.ndarray,
    output_path: Path,
    unit: str = 'K',
):
    n_levels = len(pressure_levels)

    palette = {'Background': '#4575b4', 'Baseline': '#fc8d59', 'Ours': '#d73027'}
    markers = {'Background': 's', 'Baseline': '^', 'Ours': 'o'}

    all_corrs: Dict[str, np.ndarray] = {}
    for name, res in results_dict.items():
        p, t = res['pred'], res['target']
        n_lev = min(p.shape[1], n_levels)
        corrs = []
        for l in range(n_lev):
            pf = p[:, l].flatten()
            tf = t[:, l].flatten()
            valid = np.isfinite(pf) & np.isfinite(tf)
            if valid.sum() > 100:
                corrs.append(np.corrcoef(pf[valid], tf[valid])[0, 1])
            else:
                corrs.append(np.nan)
        all_corrs[name] = np.array(corrs)

    fig, axes = plt.subplots(1, 3, figsize=(19, 8))

    # -------- (a) 相关系数廓线 --------
    ax1 = axes[0]
    for name, corrs in all_corrs.items():
        n = len(corrs)
        ax1.plot(corrs, pressure_levels[:n],
                 marker=markers.get(name, 'o'), ms=4, lw=2,
                 color=palette.get(name, 'gray'), label=name,
                 markerfacecolor='white', markeredgewidth=1.4)
    all_vals = np.concatenate([c[np.isfinite(c)] for c in all_corrs.values()])
    x_lo = max(0, np.nanmin(all_vals) - 0.05)
    ax1.set_xlim(x_lo, 1.0)

    ax1.xaxis.tick_top();  ax1.xaxis.set_label_position('top')
    ax1.set_xlabel('Correlation', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Pressure (hPa)', fontsize=13, fontweight='bold')
    ax1.set_yscale('log'); ax1.invert_yaxis()
    ax1.set_ylim(pressure_levels[0], pressure_levels[min(n_levels-1, len(corrs)-1)])
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.grid(True, which='both', alpha=0.25)
    ax1.legend(loc='lower left', fontsize=10)
    ax1.set_title('(a) Correlation Profile', fontsize=14, fontweight='bold', pad=40)

    # -------- (b) 热力图 --------
    ax2 = axes[1]
    model_names = list(all_corrs.keys())
    mat = np.column_stack([all_corrs[m] for m in model_names])   # (n_levels, n_models)
    vmin_h = max(0, np.nanmin(mat) - 0.1)
    im = ax2.imshow(mat, cmap='RdYlGn', vmin=vmin_h, vmax=1.0,
                    aspect='auto', interpolation='nearest')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=25, ha='right')
    ax2.set_ylabel('Level Index')
    ax2.set_title('(b) Correlation Heatmap', fontsize=14, fontweight='bold')
    step = max(1, n_levels // 12)
    for i in range(0, n_levels, step):
        for j in range(len(model_names)):
            v = mat[i, j]
            if np.isfinite(v):
                tc = 'white' if v < (vmin_h + 1.0) / 2 else 'black'
                ax2.text(j, i, f'{v:.2f}', ha='center', va='center',
                         fontsize=7, color=tc, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046, label='Corr')

    # -------- (c) 关键层 RMSE 条形图 --------
    ax3 = axes[2]
    key_idx = sorted(set([0, n_levels // 4, n_levels // 2,
                          3 * n_levels // 4, n_levels - 1]))
    key_idx = [k for k in key_idx if k < n_levels]
    x = np.arange(len(key_idx))
    n_models = len(results_dict)
    w = 0.8 / n_models
    for i, (name, res) in enumerate(results_dict.items()):
        p, t = res['pred'], res['target']
        vals = [np.sqrt(np.nanmean((p[:, l] - t[:, l]) ** 2)) for l in key_idx]
        offset = (i - n_models / 2 + 0.5) * w
        bars = ax3.bar(x + offset, vals, w * 0.9, label=name,
                       color=palette.get(name, 'gray'), alpha=0.85)
        for b, v in zip(bars, vals):
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height(),
                     f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{pressure_levels[l]:.0f} hPa' for l in key_idx],
                         rotation=35, ha='right')
    ax3.set_ylabel(f'RMSE ({unit})', fontsize=13, fontweight='bold')
    ax3.set_title('(c) RMSE at Key Levels', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Part 10: 综合指标表格 (反归一化后的真实值)                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def plot_metrics_table(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
    unit: str = 'K',
):
    bkg_rmse_global = None
    if 'Background' in results_dict:
        b = results_dict['Background']
        pf = b['pred'].flatten()
        tf = b['target'].flatten()
        valid = np.isfinite(pf) & np.isfinite(tf)
        bkg_rmse_global = np.sqrt(np.mean((pf[valid] - tf[valid]) ** 2))

    rows = []
    for name, res in results_dict.items():
        pf = res['pred'].flatten()
        tf = res['target'].flatten()
        valid = np.isfinite(pf) & np.isfinite(tf)
        pf, tf = pf[valid], tf[valid]
        rmse = np.sqrt(np.mean((pf - tf) ** 2))
        mae  = np.mean(np.abs(pf - tf))
        bias = np.mean(pf - tf)
        corr = np.corrcoef(pf, tf)[0, 1]
        improv = ((bkg_rmse_global - rmse) / bkg_rmse_global * 100
                  if bkg_rmse_global and name != 'Background' else 0.0)
        rows.append(dict(Model=name, RMSE=rmse, MAE=mae, Bias=bias,
                         Corr=corr, Improv=improv))

    fig, ax = plt.subplots(figsize=(14, 1.5 + 0.6 * len(rows)))
    ax.axis('off')

    headers = ['Model', f'RMSE ({unit})', f'MAE ({unit})', f'Bias ({unit})',
               'Correlation', 'RMSE Improv. (%)']
    cell_text = []
    for d in rows:
        cell_text.append([
            d['Model'],
            f"{d['RMSE']:.5f}",
            f"{d['MAE']:.5f}",
            f"{d['Bias']:.5f}",
            f"{d['Corr']:.5f}",
            f"{d['Improv']:.1f}" if d['Model'] != 'Background' else '—',
        ])

    tbl = ax.table(cellText=cell_text, colLabels=headers,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 2.0)
    for i in range(len(headers)):
        tbl[(0, i)].set_facecolor('#2c3e50')
        tbl[(0, i)].set_text_props(color='white', fontweight='bold')
    for ri, d in enumerate(rows, start=1):
        if d['Model'] == 'Ours':
            for ci in range(len(headers)):
                tbl[(ri, ci)].set_facecolor('#d5f5e3')

    ax.set_title(f'Comprehensive Metrics (denormalized, unit: {unit})',
                 fontsize=15, fontweight='bold', pad=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: {output_path}")

    # ---- 同时打印文本指标到 stdout ----
    print("\n" + "─" * 72)
    print(f"  {'模型':<18} {'RMSE':>10} {'MAE':>10} {'Bias':>10} {'Corr':>10} {'Improv%':>10}")
    print("─" * 72)
    for d in rows:
        improv_str = f"{d['Improv']:+.2f}%" if d['Model'] != 'Background' else '   —'
        print(f"  {d['Model']:<18} {d['RMSE']:>10.5f} {d['MAE']:>10.5f} {d['Bias']:>+10.5f} {d['Corr']:>10.5f} {improv_str:>10}")
    print("─" * 72 + "\n")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Part 11: 模型加载                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    加载训练好的模型.
    按需适配你自己的 create_model / UNetConfig 接口.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = checkpoint.get('args', {})

    try:
        from models.backbone import create_model, UNetConfig
    except ImportError:
        raise ImportError(
            "无法导入 models.backbone, 请确保项目结构正确或修改此函数."
        )

    if args.get('model', 'physics_unet') != 'vanilla_unet':
        config = UNetConfig(
            fusion_mode=args.get('fusion_mode', 'gated'),
            use_aux=args.get('use_aux', True),
            mask_aware=args.get('mask_aware', True),
        )
        model = create_model(args.get('model', 'physics_unet'), config=config)
    else:
        model = create_model('vanilla_unet')

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Part 12: 主流程                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def main():
    """主函数"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("🚀  Inference Analysis V3 — 反归一化 · 真实气压层 · 期刊级可视化")
    print("=" * 72)
    print(f"  Cartopy:      {'✓' if HAS_CARTOPY else '✗'}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"  Device:       {device}")
    extent = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    print(f"  Extent:       {extent}")
    print(f"  Resolution:   {args.resolution}°")
    print(f"  Variable:     {args.variable_name} ({args.variable_unit})")
    print("=" * 72 + "\n")

    # ================================================================ #
    #  1. 加载模型                                                      #
    # ================================================================ #
    print("📦 加载模型 ...")
    model = load_model(args.checkpoint, device)
    print("  ✓ 主模型就绪")

    baseline_model = None
    if args.baseline_checkpoint:
        baseline_model = load_model(args.baseline_checkpoint, device)
        print("  ✓ 基线模型就绪")

    # ================================================================ #
    #  2. 加载数据 + 标准化器                                            #
    # ================================================================ #
    print("\n📂 加载测试数据 ...")
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")

    _all = sorted(f for f in data_root.glob('**/*.npz')
                 if f.name not in ('stats.npz', 'dataset_split.json', 'increment_stats.npz'))
    if not _all:
        raise FileNotFoundError(f"未在 {data_root} 找到 .npz 文件")
    # 过滤损坏文件
    file_list = []
    for _f in _all:
        try:
            _d = np.load(str(_f))
            if _d['target'].sum() != 0:
                file_list.append(str(_f))
        except Exception:
            pass
    print(f"  有效文件: {len(file_list)} / {len(_all)}")

    dataset = LazySatelliteERA5Dataset(file_list=file_list, use_aux=True)

    # 加载/计算标准化统计量
    if args.stats_file and Path(args.stats_file).exists():
        st = np.load(args.stats_file)
        dataset.obs_normalizer    = LevelwiseNormalizer(st['obs_mean'],    st['obs_std'],    name='obs')
        dataset.bkg_normalizer    = LevelwiseNormalizer(st['bkg_mean'],    st['bkg_std'],    name='bkg')
        dataset.target_normalizer = LevelwiseNormalizer(st['target_mean'], st['target_std'], name='target')
        print(f"  ✓ 统计量已加载: {args.stats_file}")
    else:
        dataset.compute_statistics(n_samples=min(1000, len(dataset)))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"  ✓ 样本数: {len(dataset)}")

    # ================================================================ #
    #  3. 推理 + 反归一化                                                #
    # ================================================================ #
    print("\n🔮 推理中 (含反归一化) ...")
    inc_norm = None
    if getattr(args, 'use_increment', False) and args.increment_stats:
        inc_st = np.load(args.increment_stats)
        inc_norm = LevelwiseNormalizer(inc_st['inc_mean'], inc_st['inc_std'], name='increment')
        print(f"  ✓ 增量统计量已加载: {args.increment_stats}")
    engine = InferenceEngine(
        model, device,
        target_normalizer=dataset.target_normalizer,
        bkg_normalizer=dataset.bkg_normalizer,
        obs_normalizer=dataset.obs_normalizer,
        inc_normalizer=inc_norm,
    )
    results = engine.evaluate_dataset(dataloader, denormalize=True)
    print(f"  pred 形状:   {results['pred'].shape}")
    print(f"  pred 范围:   [{np.nanmin(results['pred']):.2f}, {np.nanmax(results['pred']):.2f}] {args.variable_unit}")
    print(f"  target 范围: [{np.nanmin(results['target']):.2f}, {np.nanmax(results['target']):.2f}] {args.variable_unit}")

    # ================================================================ #
    #  4. 构建对比字典                                                   #
    # ================================================================ #
    results_dict = {'Ours': results}

    if baseline_model:
        bl_engine = InferenceEngine(
            baseline_model, device,
            target_normalizer=dataset.target_normalizer,
            bkg_normalizer=dataset.bkg_normalizer,
            obs_normalizer=dataset.obs_normalizer,
        )
        results_dict['Baseline'] = bl_engine.evaluate_dataset(dataloader, denormalize=True)

    # Background: 反归一化后的 bkg 作为"预测"
    results_dict['Background'] = {
        'pred':   results['bkg'],
        'target': results['target'],
    }

    # ================================================================ #
    #  5. 确定气压层                                                     #
    # ================================================================ #
    n_levels = results['pred'].shape[1]
    pressure_levels = get_pressure_levels(n_levels)
    level_idx = min(args.level_idx, n_levels - 1)
    print(f"\n  垂直层数:   {n_levels}")
    print(f"  气压层范围: {pressure_levels[0]:.1f} – {pressure_levels[-1]:.2f} hPa")
    print(f"  可视化层:   index={level_idx} → {pressure_levels[level_idx]:.1f} hPa")

    # ================================================================ #
    #  6. 尝试从 dataset 读取真实 lat/lon                                #
    # ================================================================ #
    lat_arr, lon_arr = None, None
    try:
        sample0 = dataset[0]
        if 'lat' in sample0:
            lat_arr = sample0['lat'].numpy()
        if 'lon' in sample0:
            lon_arr = sample0['lon'].numpy()
        if lat_arr is not None:
            print(f"  ✓ 从 dataset 读取真实 lat/lon")
    except Exception:
        pass

    # ================================================================ #
    #  7. 常用变量简写                                                   #
    # ================================================================ #
    unit = args.variable_unit
    var  = args.variable_name
    res  = args.resolution
    do_transpose = args.transpose_spatial

    # ================================================================ #
    #  8. 生成图表                                                       #
    # ================================================================ #
    print("\n📊 生成图表 ...")

    if do_transpose:
        print("  ⚠️  已启用 --transpose_spatial, 空间维度将被转置")

    # [1/6] 垂直 RMSE 廓线
    print("  [1/6] 垂直 RMSE 廓线 ...")
    plot_vertical_rmse_profile(
        results_dict, pressure_levels,
        output_dir / 'vertical_rmse_profile.png',
        unit=unit,
    )

    # [2/6] 偏差空间分布
    print("  [2/6] 偏差空间分布 ...")
    plot_spatial_bias_map(
        results['pred'], results['target'],
        level_idx, args.case_study_idx,
        output_dir / 'spatial_bias_map.png',
        extent, resolution=res, unit=unit,
        pressure_levels=pressure_levels,
        lat_array=lat_arr, lon_array=lon_arr,
        transpose_data=do_transpose,
    )

    # [3/6] 个例可视化
    print("  [3/6] 个例可视化 ...")
    plot_case_study(
        results['obs'], results['bkg'],
        results['target'], results['pred'],
        level_idx, args.case_study_idx,
        output_dir / 'case_study.png',
        extent, resolution=res, unit=unit,
        pressure_levels=pressure_levels,
        lat_array=lat_arr, lon_array=lon_arr,
        transpose_data=do_transpose,
    )

    # [4/6] 散点图
    print("  [4/6] 散点图 ...")
    plot_scatter_comparison(
        results['pred'], results['target'],
        output_dir / 'scatter_plot.png',
        unit=unit, variable=var,
    )

    # [5/6] 相关性分析
    print("  [5/6] 相关性分析 ...")
    plot_correlation_analysis(
        results_dict, pressure_levels,
        output_dir / 'correlation_analysis.png',
        unit=unit,
    )

    # [6/6] 综合指标表格
    print("  [6/6] 综合指标表格 ...")
    plot_metrics_table(
        results_dict,
        output_dir / 'metrics_table.png',
        unit=unit,
    )

    print("\n" + "=" * 72)
    print("✅  分析完成!")
    print(f"📁  图表目录: {output_dir.resolve()}")
    print("=" * 72 + "\n")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  入口                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    main()
