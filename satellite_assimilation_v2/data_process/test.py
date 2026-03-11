#!/usr/bin/env python3
"""
diagnose_data.py — 诊断数据结构，找出经纬度坐标位置
"""
import numpy as np
import h5py
import json
from pathlib import Path

# ========== 改成你的实际路径 ==========
SAMPLE_DIR = Path("/data/lrx_true/era_obs/2024/01")
# =====================================

# 找第一个样本
h5_files = sorted(SAMPLE_DIR.glob("collocation_*.h5"))
if not h5_files:
    print("❌ 没找到 .h5 文件，请检查路径")
    exit(1)

h5_path = h5_files[0]
base = h5_path.stem  # e.g. "collocation_20240117_1552"
print(f"样本: {base}\n")

# ──────────────────────────────────────
# 1. 检查 _X.npy
# ──────────────────────────────────────
x_path = SAMPLE_DIR / f"{base}_X.npy"
if x_path.exists():
    X = np.load(x_path)
    print(f"_X.npy: shape={X.shape}, dtype={X.dtype}")
    print(f"  前3行:\n{X[:3, :]}")
    print(f"  各列范围:")
    for c in range(min(X.shape[1], 20)):
        col = X[:, c]
        print(f"    col[{c:>2d}]: min={col.min():>12.4f}  max={col.max():>12.4f}  "
              f"mean={col.mean():>12.4f}  (NaN={np.isnan(col).sum()})")
    print()

# ──────────────────────────────────────
# 2. 检查 _Y.npy
# ──────────────────────────────────────
y_path = SAMPLE_DIR / f"{base}_Y.npy"
if y_path.exists():
    Y = np.load(y_path)
    print(f"_Y.npy: shape={Y.shape}, dtype={Y.dtype}")
    print(f"  前3行(前10列):\n{Y[:3, :10]}")
    print(f"  各列范围(前10列):")
    for c in range(min(Y.shape[1], 10)):
        col = Y[:, c]
        print(f"    col[{c:>2d}]: min={col.min():>12.4f}  max={col.max():>12.4f}  "
              f"mean={col.mean():>12.4f}")
    print()

# ──────────────────────────────────────
# 3. 检查 pressure.npy
# ──────────────────────────────────────
p_path = SAMPLE_DIR / f"{base}_pressure.npy"
if p_path.exists():
    P = np.load(p_path)
    print(f"_pressure.npy: shape={P.shape}, dtype={P.dtype}")
    if P.ndim == 1:
        print(f"  values: {P}")
    else:
        print(f"  前3行:\n{P[:3]}")
    print()

# ──────────────────────────────────────
# 4. ★ 检查 .h5 文件（关键！经纬度大概率在这里）
# ──────────────────────────────────────
print(f"{'='*60}")
print(f".h5 文件结构: {h5_path.name}")
print(f"{'='*60}")

def print_h5_structure(name, obj):
    """递归打印 HDF5 结构"""
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Dataset):
        print(f"  {indent}📊 {name}: shape={obj.shape}, dtype={obj.dtype}")
        # 检查是否像经纬度
        low_name = name.lower()
        if any(kw in low_name for kw in ["lat", "lon", "geo", "coord", "position"]):
            data = obj[:]
            print(f"  {indent}   ★ 可能是坐标！ min={data.min():.4f}, max={data.max():.4f}")
    elif isinstance(obj, h5py.Group):
        print(f"  {indent}📁 {name}/")

with h5py.File(h5_path, "r") as f:
    print(f"  根目录 attrs: {list(f.attrs.keys())}")
    f.visititems(print_h5_structure)
    
    # 尝试直接读取常见的经纬度字段名
    print(f"\n  ── 尝试读取常见坐标字段 ──")
    candidates = [
        "lat", "lon", "latitude", "longitude",
        "Latitude", "Longitude", "LAT", "LON",
        "obs_lat", "obs_lon", "era5_lat", "era5_lon",
        "geo/lat", "geo/lon", "geolocation/latitude", "geolocation/longitude",
    ]
    for key in candidates:
        try:
            data = f[key][:]
            print(f"  ✅ f['{key}']: shape={data.shape}, "
                  f"min={data.min():.4f}, max={data.max():.4f}")
        except KeyError:
            pass

# ──────────────────────────────────────
# 5. 检查 metadata.json
# ──────────────────────────────────────
meta_path = SAMPLE_DIR / f"{base}_metadata.json"
if meta_path.exists():
    print(f"\n{'='*60}")
    print(f"metadata.json 内容:")
    print(f"{'='*60}")
    with open(meta_path) as f:
        meta = json.load(f)
    # 格式化输出，但限制长度
    meta_str = json.dumps(meta, indent=2, ensure_ascii=False)
    if len(meta_str) > 3000:
        print(meta_str[:3000])
        print(f"\n  ... (截断，共 {len(meta_str)} 字符)")
    else:
        print(meta_str)

# ──────────────────────────────────────
# 6. 汇总
# ──────────────────────────────────────
print(f"\n{'='*60}")
print("诊断总结")
print(f"{'='*60}")
print(f"  X shape: {X.shape if x_path.exists() else 'N/A'}")
print(f"  Y shape: {Y.shape if y_path.exists() else 'N/A'}")
print(f"  总样本数 (此目录): {len(list(SAMPLE_DIR.glob('*_X.npy')))}")
print(f"  _lat.npy 存在: {(SAMPLE_DIR / f'{base}_lat.npy').exists()}")
print(f"  _lon.npy 存在: {(SAMPLE_DIR / f'{base}_lon.npy').exists()}")
print(f"\n请把以上输出完整贴给我，我会据此修改 prepare_v3_data.py")
