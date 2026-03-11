#!/usr/bin/env python3
"""
prepare_v2_data.py  —  数据准备脚本 v3（地理空间插值版）

核心改进:
  ✘  旧方案: obs[:4096].reshape(64,64) → 暴力折叠，空间拓扑被破坏
  ✔  新方案: 基于真实经纬度，用 Delaunay 三角化 + 线性插值将离散观测
             映射到具有物理意义的 H×W 规则经纬度网格

功能:
  1. 基于经纬度将 (N, C) 散点插值到 (H, W) 规则网格
  2. 生成基于 KD-Tree 距离的观测覆盖掩码 (Mask)
  3. 按比例划分 train / val / test
  4. 计算并保存逐通道统计量 (stats.npz)
  5. 生成划分元信息 (dataset_split.json)

用法:
  python prepare_v2_data.py \
      --source_dir /data2/lrx/era_obs \
      --target_dir /data2/lrx/era_obs/npz \
      --coord_source separate_files \
      --resolution 0.25

坐标来源 (--coord_source):
  separate_files : 查找 *_lat.npy / *_lon.npy（与 _X.npy 同目录同前缀）
  from_x_first2  : _X.npy 前两列为 [lat, lon]，观测通道从第3列起（C_obs=15）
  from_y_first2  : _Y.npy 前两列为 [lat, lon]，目标通道从第3列起（C_tgt=35）
"""

import numpy as np
import os
import sys
import json
import argparse
import traceback
from pathlib import Path

# 科学插值 —— 替代暴力 reshape 的核心工具
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree


# ========================== 默认配置 ==========================
DEFAULT_SOURCE_DIR = "/data2/lrx_true/era_obs"
DEFAULT_TARGET_DIR = "/data2/lrx_true/era_obs/npz_v2"
H, W = 64, 64
RESOLUTION = 0.25        # 网格分辨率（度），0.25° ≈ 25 km
COORD_SOURCE = "separate_files"
# ==============================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="数据准备脚本 v3 — 地理空间插值版",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--source_dir", type=str, default=DEFAULT_SOURCE_DIR,
                   help="原始 .npy 数据根目录")
    p.add_argument("--target_dir", type=str, default=DEFAULT_TARGET_DIR,
                   help="输出 .npz 数据根目录")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio",   type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--resolution",  type=float, default=RESOLUTION,
                   help="网格空间分辨率（度），默认 0.25")
    p.add_argument("--grid_h", type=int, default=H, help="网格行数（纬度方向）")
    p.add_argument("--grid_w", type=int, default=W, help="网格列数（经度方向）")
    p.add_argument("--coord_source", type=str, default=COORD_SOURCE,
                   choices=["separate_files", "from_x_first2", "from_y_first2"],
                   help="经纬度坐标来源方式")
    p.add_argument("--interp_method", type=str, default="linear",
                   choices=["linear", "nearest"],
                   help="插值方法：linear（线性，推荐）| nearest（最近邻）")
    p.add_argument("--mask_radius_factor", type=float, default=2.0,
                   help="掩码半径 = factor × resolution（默认 2.0）")
    return p.parse_args()


# ================================================================
#  1. 数据加载：解耦坐标来源
# ================================================================
def load_sample_data(x_path, y_path, coord_source="separate_files"):
    """
    加载单个样本的全部原始数据。

    Returns
    -------
    lat      : (N,)       纬度
    lon      : (N,)       经度
    obs_data : (N, C_obs) 观测通道
    tgt_data : (N, C_tgt) 目标通道
    """
    base = str(x_path).replace("_X.npy", "")

    if coord_source == "separate_files":
        # ---------- 独立坐标文件 ----------
        lat_path = base + "_lat.npy"
        lon_path = base + "_lon.npy"
        if not os.path.exists(lat_path) or not os.path.exists(lon_path):
            raise FileNotFoundError(
                f"未找到坐标文件:\n  {lat_path}\n  {lon_path}\n"
                "请确认 _lat.npy / _lon.npy 存在，或改用 --coord_source from_x_first2"
            )
        lat = np.load(lat_path).astype(np.float64).ravel()
        lon = np.load(lon_path).astype(np.float64).ravel()
        obs_data = np.load(x_path).astype(np.float64)    # (N, 17)
        tgt_data = np.load(y_path).astype(np.float64)    # (N, 37)

    elif coord_source == "from_x_first2":
        # ---------- _X.npy 前两列为 [lat, lon] ----------
        raw_x = np.load(x_path).astype(np.float64)       # (N, 17)
        lat, lon = raw_x[:, 0], raw_x[:, 1]
        obs_data = raw_x[:, 2:]                           # (N, 15)
        tgt_data = np.load(y_path).astype(np.float64)     # (N, 37)

    elif coord_source == "from_y_first2":
        # ---------- _Y.npy 前两列为 [lat, lon] ----------
        raw_y = np.load(y_path).astype(np.float64)        # (N, 37)
        lat, lon = raw_y[:, 0], raw_y[:, 1]
        tgt_data = raw_y[:, 2:]                            # (N, 35)
        obs_data = np.load(x_path).astype(np.float64)      # (N, 17)

    else:
        raise ValueError(f"未知坐标来源: {coord_source}")

    return lat, lon, obs_data, tgt_data


# ================================================================
#  2. 规则网格构建
# ================================================================
def create_regular_grid(lat, lon, h, w, resolution,
                        center_lat=None, center_lon=None):
    """
    以散点集的中位数（或指定中心）为中心，构建 H×W 的规则经纬度网格。

    对于 h=w=64, resolution=0.25°:
        网格覆盖范围 = (64-1) × 0.25 = 15.75°

    Parameters
    ----------
    lat, lon     : (N,)  散点坐标（仅用于自动推断中心）
    h, w         : int   网格尺寸
    resolution   : float 网格分辨率（度）
    center_lat/lon : float 手动指定中心（None=自动取中位数）

    Returns
    -------
    lat_1d  : (h,)    纬度一维坐标
    lon_1d  : (w,)    经度一维坐标
    lat2d   : (h, w)  纬度二维网格
    lon2d   : (h, w)  经度二维网格
    """
    if center_lat is None:
        center_lat = float(np.nanmedian(lat))
    if center_lon is None:
        center_lon = float(np.nanmedian(lon))

    half_lat = (h - 1) * resolution / 2.0
    half_lon = (w - 1) * resolution / 2.0

    lat_1d = np.linspace(center_lat - half_lat,
                         center_lat + half_lat, h)
    lon_1d = np.linspace(center_lon - half_lon,
                         center_lon + half_lon, w)

    # meshgrid 默认 'xy' 索引：
    #   行方向(axis-0) = 纬度，列方向(axis-1) = 经度
    #   lat2d[i, j] = lat_1d[i];  lon2d[i, j] = lon_1d[j]
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    return lat_1d, lon_1d, lat2d, lon2d


# ================================================================
#  3. 高效多通道插值（核心函数）
# ================================================================
def interpolate_channels(scatter_lonlat, values, lon2d, lat2d,
                         method="linear"):
    """
    将 (N, C) 的多通道散点数据高效插值到 (H, W) 规则网格。

    策略:
        1) 用 LinearNDInterpolator 做 Delaunay 三角化
           —— 三角化只算一次，C 个通道共享同一个三角网
        2) 凸包外的 NaN 区域用 NearestNDInterpolator 回退填充
        3) rescale=True 消除 lon/lat 量纲差异

    Parameters
    ----------
    scatter_lonlat : (N, 2)  散点坐标 [lon, lat]
    values         : (N, C)  各通道值
    lon2d, lat2d   : (H, W)  目标网格坐标
    method         : 'linear' | 'nearest'

    Returns
    -------
    grid : (C, H, W)  插值后的网格数据
    """
    N = scatter_lonlat.shape[0]
    if values.ndim == 1:
        values = values[:, np.newaxis]
    C = values.shape[1]
    H, W = lon2d.shape

    # ---------- 退化情况 ----------
    if N == 0:
        return np.zeros((C, H, W), dtype=np.float32)

    # 清洗 NaN（后续由 mask 控制真正的有效区域）
    vals = np.nan_to_num(values, nan=0.0).astype(np.float64)

    try:
        if method == "linear" and N >= 4:
            # ----- Delaunay 三角化 + 重心线性插值 -----
            interp_lin = LinearNDInterpolator(
                scatter_lonlat, vals,
                fill_value=np.nan, rescale=True
            )
            # 返回 (H, W, C)
            result = interp_lin(lon2d, lat2d)

            # 凸包外区域 → 最近邻回退
            nan_mask = np.isnan(result)
            if np.any(nan_mask):
                interp_nn = NearestNDInterpolator(
                    scatter_lonlat, vals, rescale=True
                )
                result[nan_mask] = interp_nn(lon2d, lat2d)[nan_mask]
        else:
            # ----- 点太少或指定 nearest -----
            interp_nn = NearestNDInterpolator(
                scatter_lonlat, vals, rescale=True
            )
            result = interp_nn(lon2d, lat2d)   # (H, W, C)

    except Exception as e:
        print(f"    ⚠️  插值异常 ({e})，返回全零网格")
        return np.zeros((C, H, W), dtype=np.float32)

    # 最终清理
    result = np.nan_to_num(result, nan=0.0)

    # (H, W, C) → (C, H, W)  —— PyTorch / Channel-first 约定
    if result.ndim == 3:
        grid = np.transpose(result, (2, 0, 1))
    else:
        # C==1 时 result 可能是 (H, W)
        grid = result[np.newaxis, :, :]

    return grid.astype(np.float32)


# ================================================================
#  4. 观测覆盖掩码（基于 KD-Tree 距离）
# ================================================================
def generate_obs_mask(obs_lon, obs_lat, lon2d, lat2d, max_dist_deg):
    """
    基于 KD-Tree 计算每个网格点到最近观测点的经纬度距离，
    距离 ≤ max_dist_deg 的网格标为 1（有观测覆盖），否则标为 0。

    NOTE: 这里用的是经纬度平面距离（度），对中低纬度区域足够精确。
    如需高纬度精度，可替换为 BallTree + haversine。

    Parameters
    ----------
    obs_lon, obs_lat : (N,)  观测点经纬度
    lon2d, lat2d     : (H, W) 网格经纬度
    max_dist_deg     : float  掩码截断距离（度）

    Returns
    -------
    mask : (H, W) float32  二值掩码
    """
    H, W = lon2d.shape

    if len(obs_lon) == 0:
        return np.zeros((H, W), dtype=np.float32)

    obs_pts  = np.column_stack([obs_lon, obs_lat])          # (N, 2)
    grid_pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])  # (H*W, 2)

    tree = cKDTree(obs_pts)
    dists, _ = tree.query(grid_pts, k=1)                   # 最近邻距离
    dist_grid = dists.reshape(H, W)

    mask = (dist_grid <= max_dist_deg).astype(np.float32)
    return mask


# ================================================================
#  5. ★ 核心函数：单样本转换 ★
# ================================================================
def convert_single_sample(x_path, y_path,
                          lat, lon, obs_data, tgt_data,
                          h=64, w=64, resolution=0.25,
                          interp_method="linear",
                          mask_radius=None,
                          center_lat=None, center_lon=None):
    """
    将一组 (N, C) 离散观测/目标散点，基于真实经纬度插值到
    具有物理意义的 H×W 规则经纬度网格。

    ┌──────────────────────────────────────────────────────────┐
    │  旧方案（已废弃）:                                       │
    │    obs[:4096].reshape(64, 64)                            │
    │    → 一维序列按 64 列硬换行，空间拓扑完全破坏             │
    │                                                          │
    │  新方案:                                                 │
    │    1. 以散点中位数为中心，构建 64×64 的 0.25° 规则网格    │
    │    2. Delaunay 三角化 → 重心线性插值                      │
    │    3. 凸包外 → NearestNDInterpolator 回退                │
    │    4. KD-Tree 距离 → 二值覆盖掩码                        │
    └──────────────────────────────────────────────────────────┘

    Parameters
    ----------
    x_path, y_path : Path
        原始文件路径（仅用于可重复随机种子和日志）
    lat, lon       : (N,)       散点经纬度
    obs_data       : (N, C_obs) 观测通道值（如 17 通道亮温）
    tgt_data       : (N, C_tgt) 目标通道值（如 37 层 ERA5 温度/湿度）
    h, w           : int        网格尺寸
    resolution     : float      网格分辨率（度）
    interp_method  : str        插值方法
    mask_radius    : float      掩码半径（度），默认 2 × resolution
    center_lat/lon : float      网格中心（None = 自动中位数）

    Returns
    -------
    dict:
        obs    : (C_obs, H, W)  插值后的观测场（无覆盖区域已置零）
        bkg    : (C_tgt, H, W)  模拟背景场
        target : (C_tgt, H, W)  插值后的目标场
        mask   : (1, H, W)      观测覆盖掩码（1=有覆盖, 0=空缺）
        aux    : (4, H, W)      辅助特征 [norm_lat, norm_lon, obs_density, 0]
        lat2d  : (H, W)         网格纬度（可视化/后处理用）
        lon2d  : (H, W)         网格经度
    """
    n_obs_ch = obs_data.shape[1]
    n_tgt_ch = tgt_data.shape[1]

    if mask_radius is None:
        mask_radius = resolution * 2.0

    # =========================================================
    # Step 1 : 构建规则经纬度网格
    # =========================================================
    # 例如 0.25° × 64 → 覆盖 ~16° 的区域
    lat_1d, lon_1d, lat2d, lon2d = create_regular_grid(
        lat, lon, h, w, resolution, center_lat, center_lon
    )

    # =========================================================
    # Step 2 : 筛选落在网格范围内（含边界缓冲）的有效散点
    # =========================================================
    margin = resolution * 3   # 边界外扩 3 个格点，减少边缘效应
    valid_idx = (
        np.isfinite(lat) & np.isfinite(lon)
        & (lat >= lat_1d[0]  - margin)
        & (lat <= lat_1d[-1] + margin)
        & (lon >= lon_1d[0]  - margin)
        & (lon <= lon_1d[-1] + margin)
    )

    lat_v = lat[valid_idx]
    lon_v = lon[valid_idx]
    obs_v = obs_data[valid_idx]
    tgt_v = tgt_data[valid_idx]

    # 散点坐标矩阵：[lon, lat] 与 meshgrid(lon, lat) 一致
    scatter_ll = np.column_stack([lon_v, lat_v])      # (N_valid, 2)

    # =========================================================
    # Step 3 : 多通道插值 —— 观测场
    # =========================================================
    # LinearNDInterpolator 内部只做一次 Delaunay 三角化
    # C_obs 个通道共享同一个三角网，效率远高于逐通道 griddata
    obs_grid = interpolate_channels(
        scatter_ll, obs_v, lon2d, lat2d, interp_method
    )   # (C_obs, H, W)

    # =========================================================
    # Step 4 : 多通道插值 —— 目标场
    # =========================================================
    tgt_grid = interpolate_channels(
        scatter_ll, tgt_v, lon2d, lat2d, interp_method
    )   # (C_tgt, H, W)

    # =========================================================
    # Step 5 : 生成观测覆盖掩码
    # =========================================================
    # 基于 KD-Tree: 网格点到最近观测点距离 ≤ mask_radius → 1
    mask_2d = generate_obs_mask(
        lon_v, lat_v, lon2d, lat2d, mask_radius
    )   # (H, W)
    mask = mask_2d[np.newaxis, :, :]                   # (1, H, W)

    # 对观测场施加掩码：无覆盖区域置零
    # （后续 Partial Conv / Mask-aware 机制可据此区分有效/缺失）
    obs_grid = obs_grid * mask

    # =========================================================
    # Step 6 : 构建模拟背景场
    # =========================================================
    # 实际工作中应使用 NWP 6h 预报场。
    # 这里简单模拟：target + 小幅高斯噪声
    rng = np.random.RandomState(hash(str(x_path)) % (2**31))
    bkg_noise = rng.normal(0, 0.3, tgt_grid.shape).astype(np.float32)
    bkg_grid  = (tgt_grid + bkg_noise).astype(np.float32)

    # =========================================================
    # Step 7 : 辅助特征（Auxiliary Features）
    # =========================================================
    aux_grid = np.zeros((4, h, w), dtype=np.float32)

    # [0] 归一化纬度 ∈ [-1, 1]
    lat_span = max(lat_1d[-1] - lat_1d[0], 1e-6)
    aux_grid[0] = ((lat2d - lat2d.mean()) / (lat_span / 2)).astype(np.float32)

    # [1] 归一化经度 ∈ [-1, 1]
    lon_span = max(lon_1d[-1] - lon_1d[0], 1e-6)
    aux_grid[1] = ((lon2d - lon2d.mean()) / (lon_span / 2)).astype(np.float32)

    # [2] 观测密度（归一化到 [0, 1]）
    #     统计每个网格点 mask_radius 范围内的观测点数
    if len(lat_v) > 0:
        tree = cKDTree(scatter_ll)
        grid_flat = np.column_stack([lon2d.ravel(), lat2d.ravel()])
        neighbor_counts = tree.query_ball_point(grid_flat, mask_radius)
        density = np.array([len(c) for c in neighbor_counts],
                           dtype=np.float32).reshape(h, w)
        dmax = density.max()
        if dmax > 0:
            density /= dmax
        aux_grid[2] = density

    # [3] 预留通道 —— 可扩展为陆海掩膜、太阳天顶角等
    aux_grid[3] = 0.0

    return {
        "obs":    obs_grid.astype(np.float32),    # (C_obs, H, W)
        "bkg":    bkg_grid.astype(np.float32),    # (C_tgt, H, W)
        "target": tgt_grid.astype(np.float32),    # (C_tgt, H, W)
        "mask":   mask.astype(np.float32),         # (1, H, W)
        "aux":    aux_grid,                        # (4, H, W)
        "lat2d":  lat2d.astype(np.float32),        # (H, W)
        "lon2d":  lon2d.astype(np.float32),        # (H, W)
    }


# ================================================================
#  数据集划分
# ================================================================
def split_dataset(file_list, train_ratio, val_ratio, seed):
    """按比例随机划分数据集"""
    rng = np.random.RandomState(seed)
    n = len(file_list)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    return {
        "train": [file_list[i] for i in indices[:n_train]],
        "val":   [file_list[i] for i in indices[n_train:n_train + n_val]],
        "test":  [file_list[i] for i in indices[n_train + n_val:]],
    }


# ================================================================
#  统计量计算（修复版 —— 原版 Welford 算法有 count 跨通道共享的 Bug）
# ================================================================
def compute_statistics(data_dir, split="train", use_mask_for_obs=True):
    """
    逐通道计算均值和标准差。仅使用训练集，避免数据泄露。

    使用单趟 Σx / Σx² 方法（numpy 向量化，远快于原版逐像素 Python 循环）。

    Parameters
    ----------
    data_dir : str/Path
    split    : str   使用哪个子集
    use_mask_for_obs : bool  对 obs 是否只统计 mask=1 的像素

    Returns
    -------
    dict : {'{key}_mean': (C,), '{key}_std': (C,)}  for key in [obs, bkg, target, aux]
    """
    split_dir = Path(data_dir) / split
    npz_files = sorted(split_dir.glob("*.npz"))

    if not npz_files:
        print(f"  ⚠️  {split_dir} 中无 .npz 文件，跳过统计量计算")
        return None

    print(f"  📊 计算统计量（{len(npz_files)} 个 {split} 样本）...")

    keys = ["obs", "bkg", "target", "aux"]

    # 累加器
    ch_sum    = {}
    ch_sq_sum = {}
    ch_count  = {}

    for idx, npz_path in enumerate(npz_files):
        data = np.load(npz_path)
        mask_2d = data["mask"][0] if "mask" in data else None   # (H, W)

        for key in keys:
            if key not in data:
                continue
            arr = data[key]          # (C, H, W)
            C = arr.shape[0]

            if key not in ch_sum:
                ch_sum[key]    = np.zeros(C, dtype=np.float64)
                ch_sq_sum[key] = np.zeros(C, dtype=np.float64)
                ch_count[key]  = np.zeros(C, dtype=np.float64)

            for c in range(C):
                pixels = arr[c]      # (H, W)

                # 决定有效像素区域
                if use_mask_for_obs and key == "obs" and mask_2d is not None:
                    valid = (mask_2d > 0.5) & np.isfinite(pixels)
                else:
                    valid = np.isfinite(pixels)

                px = pixels[valid].astype(np.float64)
                ch_sum[key][c]    += px.sum()
                ch_sq_sum[key][c] += (px ** 2).sum()
                ch_count[key][c]  += px.size

        if (idx + 1) % 200 == 0:
            print(f"    进度: {idx + 1}/{len(npz_files)}")

    # 汇总
    result = {}
    for key in ch_sum:
        cnt  = np.maximum(ch_count[key], 1)
        mean = ch_sum[key] / cnt
        var  = ch_sq_sum[key] / cnt - mean ** 2
        var  = np.maximum(var, 0)            # 数值安全
        std  = np.sqrt(var)
        std[std < 1e-6] = 1.0               # 防除零

        result[f"{key}_mean"] = mean.astype(np.float32)
        result[f"{key}_std"]  = std.astype(np.float32)

    return result


# ================================================================
#  主流程
# ================================================================
def convert_and_split_data(args):
    """主函数：转换、划分、统计"""

    src = Path(args.source_dir)
    tgt = Path(args.target_dir)
    h, w = args.grid_h, args.grid_w
    res  = args.resolution
    mask_radius = args.mask_radius_factor * res

    # 创建输出目录
    for s in ["train", "val", "test"]:
        (tgt / s).mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("  数据准备脚本 v3 — 地理空间插值版")
    print("=" * 66)
    print(f"  源目录       : {src}")
    print(f"  输出目录     : {tgt}")
    print(f"  网格尺寸     : {h} × {w}")
    print(f"  空间分辨率   : {res}° (≈ {res * 111:.0f} km)")
    print(f"  网格覆盖范围 : {(h-1)*res:.2f}° (lat) × {(w-1)*res:.2f}° (lon)")
    print(f"  掩码半径     : {mask_radius:.2f}°")
    print(f"  插值方法     : {args.interp_method}")
    print(f"  坐标来源     : {args.coord_source}")
    print(f"  数据集划分   : train={args.train_ratio} / "
          f"val={args.val_ratio} / "
          f"test={1 - args.train_ratio - args.val_ratio:.2f}")
    print(f"  随机种子     : {args.seed}")
    print("=" * 66)

    # ---- Step 1: 扫描数据文件 ----
    print(f"\n[1/4] 扫描数据文件...")
    x_files = sorted(src.glob("**/*_X.npy"))

    if not x_files:
        print(f"  ❌ 未找到任何 _X.npy，请检查路径: {src}")
        return False

    valid_pairs = []
    for xp in x_files:
        yp = Path(str(xp).replace("_X.npy", "_Y.npy"))
        if yp.exists():
            valid_pairs.append((xp, yp))

    print(f"  找到 {len(valid_pairs)} 个有效 (X, Y) 数据对")
    if not valid_pairs:
        print("  ❌ 无有效数据对！")
        return False

    # ---- Step 2: 划分数据集 ----
    print(f"\n[2/4] 划分数据集...")
    splits = split_dataset(valid_pairs, args.train_ratio,
                           args.val_ratio, args.seed)
    for s in ["train", "val", "test"]:
        print(f"  {s:>5s}: {len(splits[s]):>6d} 样本")

    # ---- Step 3: 转换 ----
    print(f"\n[3/4] 转换数据（地理空间插值）...")
    split_info = {"train": [], "val": [], "test": []}
    n_done, n_fail = 0, 0
    n_total = len(valid_pairs)

    for split_name, pairs in splits.items():
        split_dir = tgt / split_name
        print(f"\n  ── {split_name} ({len(pairs)} 样本) ──")

        for i, (xp, yp) in enumerate(pairs):
            try:
                # 加载坐标与原始数据
                lat, lon, obs_raw, tgt_raw = load_sample_data(
                    xp, yp, args.coord_source
                )

                # 核心：地理空间插值
                sample = convert_single_sample(
                    x_path=xp, y_path=yp,
                    lat=lat, lon=lon,
                    obs_data=obs_raw, tgt_data=tgt_raw,
                    h=h, w=w, resolution=res,
                    interp_method=args.interp_method,
                    mask_radius=mask_radius,
                )

                # 保存
                save_name = xp.stem.replace("_X", "") + ".npz"
                np.savez(split_dir / save_name, **sample)

                split_info[split_name].append(save_name)
                n_done += 1

                # 进度报告（含覆盖率诊断）
                coverage = sample["mask"].mean() * 100
                if (n_done) % 100 == 0 or i == 0:
                    print(f"    [{n_done:>5d}/{n_total}] "
                          f"{save_name:<40s} "
                          f"obs={sample['obs'].shape} "
                          f"tgt={sample['target'].shape} "
                          f"coverage={coverage:.1f}% "
                          f"pts_in_grid={int((lat >= 0).sum())}")

            except FileNotFoundError as e:
                n_fail += 1
                if n_fail <= 5:
                    print(f"    ⚠️  跳过: {e}")
                elif n_fail == 6:
                    print(f"    ⚠️  后续同类错误不再逐条打印...")
            except Exception as e:
                n_fail += 1
                if n_fail <= 5:
                    print(f"    ❌ 处理 {xp.name} 时出错: {e}")
                    traceback.print_exc()

    print(f"\n  转换完成: ✅ {n_done} 成功 / ❌ {n_fail} 失败")

    if n_done == 0:
        print("  ❌ 没有成功转换任何样本，请检查坐标配置！")
        return False

    # ---- Step 4: 统计量 ----
    print(f"\n[4/4] 计算统计量...")
    stats = compute_statistics(tgt, split="train")

    if stats:
        stats_path = tgt / "stats.npz"
        np.savez(stats_path, **stats)
        print(f"  ✅ 统计量已保存: {stats_path}")
        # 打印摘要
        for key in ["obs", "bkg", "target", "aux"]:
            m = stats.get(f"{key}_mean")
            s = stats.get(f"{key}_std")
            if m is not None:
                print(f"     {key:>6s}: mean=[{m.min():.3f} ~ {m.max():.3f}], "
                      f"std=[{s.min():.3f} ~ {s.max():.3f}], "
                      f"channels={len(m)}")

    # ---- 保存划分信息 ----
    split_meta = {
        "config": {
            "resolution": res,
            "grid_size": [h, w],
            "interp_method": args.interp_method,
            "mask_radius": mask_radius,
            "coord_source": args.coord_source,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        },
        "counts": {
            "train": len(split_info["train"]),
            "val":   len(split_info["val"]),
            "test":  len(split_info["test"]),
        },
        "files": split_info,
    }
    split_json = tgt / "dataset_split.json"
    with open(split_json, "w") as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 划分信息已保存: {split_json}")

    # ---- 完成 ----
    print(f"\n{'=' * 66}")
    print(f"  ✅ 数据准备完成！")
    print(f"{'=' * 66}")
    print(f"  输出目录: {tgt}")
    print(f"    ├── train/              ({len(split_info['train'])} 文件)")
    print(f"    ├── val/                ({len(split_info['val'])} 文件)")
    print(f"    ├── test/               ({len(split_info['test'])} 文件)")
    print(f"    ├── stats.npz           (逐通道均值/标准差)")
    print(f"    └── dataset_split.json  (配置 & 文件列表)")
    print(f"{'=' * 66}")

    return True


# ================================================================
if __name__ == "__main__":
    args = parse_args()
    ok = convert_and_split_data(args)
    sys.exit(0 if ok else 1)