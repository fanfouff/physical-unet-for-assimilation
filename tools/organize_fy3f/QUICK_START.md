# FY-3F MWTS 完整工具集

## 📦 工具清单（9个文件）

### 🗂️ **文件整理工具** ⭐新增
1. **organize_fy3f_files.py** - 按年月整理文件
2. **ORGANIZE_FILES_README.md** - 文件整理工具说明

### 🌐 **网格化处理工具**
3. **fy3f_to_era5_format.py** - 主网格化工具
4. **batch_process_monthly.sh** - 批处理脚本
5. **visualize_fy3f_gridded.py** - 数据可视化
6. **FY3F_GRIDDING_GUIDE.md** - 网格化详细指南

### 🔗 **数据配准工具**
7. **collocation_fy3d_era5_fixed.py** - FY-3F与ERA5配准（已修复）
8. **README_FIX.md** - 配准工具修复说明

### 📚 **总文档**
9. **QUICK_START.md** - 快速开始指南（本文件）

---

## 🚀 完整工作流程

### 步骤1: 整理文件（新功能！）

```bash
# 将散乱的FY-3F文件按年月整理到不同文件夹
python organize_fy3f_files.py /data2/lrx/obs -o /data2/lrx/fy3f_organized

# 输出结构:
# fy3f_organized/
# ├── 2024/
# │   ├── 01/
# │   ├── 02/
# │   └── ...
# ├── 2025/
# │   ├── 01/
# │   └── ...
```

### 步骤2: 网格化数据

```bash
# 将整理好的文件网格化为ERA5格式
python fy3f_to_era5_format.py /data2/lrx/fy3f_organized/2025/01/ \
    -o /data2/lrx/fy3f_gridded/ --start 20250101 --end 20250131

# 或使用批处理脚本
bash batch_process_monthly.sh
```

### 步骤3: 可视化检查

```bash
# 查看网格化结果
python visualize_fy3f_gridded.py \
    /data2/lrx/fy3f_gridded/2025/01/fy3f_mwts_gridded_202501.nc \
    --stats --plot-channel 10 --plot-coverage
```

### 步骤4: 数据配准（可选，用于机器学习）

```bash
# 与ERA5配准
python collocation_fy3d_era5_fixed.py \
    /data2/lrx/fy3f_organized/2025/01/FY3F_*.HDF \
    /data2/lrx/era5_2/2025/01/era5_*.grib \
    -o training_data/
```

---

## 📖 详细使用说明

### 工具1: 文件整理 ⭐推荐首先使用

**功能**: 将FY-3F文件按年月分类到不同文件夹

```bash
# 基本用法（复制文件，保留原文件）
python organize_fy3f_files.py /data2/lrx/obs -o /data2/lrx/fy3f_organized

# 移动文件（原位置文件会被移走）
python organize_fy3f_files.py /data2/lrx/obs -o /data2/lrx/fy3f_organized --move

# 试运行（只查看不执行）
python organize_fy3f_files.py /data2/lrx/obs -o /data2/lrx/fy3f_organized --dry-run

# 查看结构
python organize_fy3f_files.py /data2/lrx/fy3f_organized --list
```

**输入**: 
```
/data2/lrx/obs/
├── FY3F_MWTS-_ORBA_L1_20250119_0440_033KM_V0.HDF
├── FY3F_MWTS-_ORBA_L1_20250119_0622_033KM_V0.HDF
├── FY3F_MWTS-_ORBA_L1_20250120_0058_033KM_V0.HDF
└── ...（所有文件混在一起）
```

**输出**:
```
/data2/lrx/fy3f_organized/
├── 2025/
│   ├── 01/
│   │   ├── FY3F_MWTS-_ORBA_L1_20250119_0440_033KM_V0.HDF
│   │   ├── FY3F_MWTS-_ORBA_L1_20250119_0622_033KM_V0.HDF
│   │   └── ...
│   ├── 02/
│   │   └── ...
```

📖 **详细文档**: ORGANIZE_FILES_README.md

---

### 工具2: 网格化处理

**功能**: 将卫星轨道数据转换为规则网格（类似ERA5）

```bash
# 基本用法
python fy3f_to_era5_format.py /data2/lrx/fy3f_organized/2025/01/ \
    -o /data2/lrx/fy3f_gridded/

# 指定时间范围
python fy3f_to_era5_format.py /data2/lrx/fy3f_organized/ \
    -o /data2/lrx/fy3f_gridded/ --start 20250101 --end 20250131

# 自定义参数
python fy3f_to_era5_format.py /data2/lrx/fy3f_organized/2025/01/ \
    -o /data2/lrx/fy3f_gridded/ \
    --resolution 0.5 \
    --time-window 6 \
    --grid-method nearest
```

**输出**:
```
/data2/lrx/fy3f_gridded/
├── 2025/
│   ├── 01/
│   │   └── fy3f_mwts_gridded_202501.nc  # NetCDF格式
│   ├── 02/
│   │   └── fy3f_mwts_gridded_202502.nc
```

📖 **详细文档**: FY3F_GRIDDING_GUIDE.md

---

### 工具3: 数据可视化

**功能**: 读取和可视化网格化数据

```bash
# 显示统计信息
python visualize_fy3f_gridded.py \
    /data2/lrx/fy3f_gridded/2025/01/fy3f_mwts_gridded_202501.nc --stats

# 绘制通道10
python visualize_fy3f_gridded.py \
    /data2/lrx/fy3f_gridded/2025/01/fy3f_mwts_gridded_202501.nc \
    --plot-channel 10 --output-dir figures/

# 绘制覆盖图
python visualize_fy3f_gridded.py \
    /data2/lrx/fy3f_gridded/2025/01/fy3f_mwts_gridded_202501.nc \
    --plot-coverage --output-dir figures/

# 绘制时间序列
python visualize_fy3f_gridded.py \
    /data2/lrx/fy3f_gridded/2025/01/fy3f_mwts_gridded_202501.nc \
    --plot-timeseries --lat 30 --lon 120 --output-dir figures/
```

---

### 工具4: 数据配准（用于机器学习）

**功能**: 将FY-3F与ERA5配准，生成训练数据

```bash
# 基本用法
python collocation_fy3d_era5_fixed.py \
    /data2/lrx/fy3f_organized/2025/01/FY3F_MWTS-_ORBA_L1_20250111_1957_033KM_V0.HDF \
    /data2/lrx/era5_2/2025/01/era5_20250111_12.grib \
    -o training_data/

# 批量处理
for hdf in /data2/lrx/fy3f_organized/2025/01/*.HDF; do
    date=$(echo $hdf | grep -oP '\d{8}')
    python collocation_fy3d_era5_fixed.py $hdf \
        /data2/lrx/era5_2/2025/01/era5_${date}_*.grib \
        -o training_data/
done
```

**输出**:
- `training_data_X.npy` - 卫星亮温 (N, 98)
- `training_data_Y.npy` - 温度廓线 (N, 37)
- `training_data_pressure.npy` - 气压层
- `training_data_metadata.json` - 元数据

📖 **详细文档**: README_FIX.md

---

## 🎯 典型使用场景

### 场景1: 数据整理和管理

```bash
# 步骤1: 整理文件
python organize_fy3f_files.py /data2/lrx/obs -o /data2/lrx/fy3f_organized

# 步骤2: 查看结构
python organize_fy3f_files.py /data2/lrx/fy3f_organized --list

# 步骤3: 检查每个月的文件数
for month_dir in /data2/lrx/fy3f_organized/2025/*/; do
    count=$(ls -1 $month_dir/*.HDF 2>/dev/null | wc -l)
    month=$(basename $month_dir)
    echo "2025年${month}月: $count 个文件"
done
```

### 场景2: 批量网格化处理

```bash
# 方法1: 使用批处理脚本（推荐）
bash batch_process_monthly.sh

# 方法2: 手动循环处理
for month in {01..12}; do
    python fy3f_to_era5_format.py \
        /data2/lrx/fy3f_organized/2025/${month}/ \
        -o /data2/lrx/fy3f_gridded/ \
        --start 2025${month}01 --end 2025${month}31
done
```

### 场景3: 数据分析流程

```python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取网格化数据
ds = xr.open_dataset('/data2/lrx/fy3f_gridded/2025/01/fy3f_mwts_gridded_202501.nc')

# 2. 基本统计
print(f"时间点数: {len(ds.time)}")
print(f"通道数: {len(ds.channel)}")
print(f"亮温范围: {ds['bt'].min().values:.2f} - {ds['bt'].max().values:.2f} K")

# 3. 计算月平均
bt_monthly = ds['bt'].mean(dim='time')

# 4. 选择通道10绘图
bt_ch10 = bt_monthly.sel(channel=10)
bt_ch10.plot(figsize=(12, 6))
plt.title('FY-3F MWTS Channel 10 - Monthly Mean')
plt.savefig('monthly_mean_ch10.png')

# 5. 计算覆盖率
coverage = (ds['observation_count'] > 0).mean(dim='time')
print(f"平均覆盖率: {coverage.mean().values * 100:.1f}%")
```

---

## 💡 最佳实践

### 1. 文件整理

✅ **推荐做法**:
```bash
# 先试运行
python organize_fy3f_files.py /data2/lrx/obs -o /data2/lrx/fy3f_organized --dry-run

# 确认无误后，复制文件（保留原文件）
python organize_fy3f_files.py /data2/lrx/obs -o /data2/lrx/fy3f_organized

# 检查结果
python organize_fy3f_files.py /data2/lrx/fy3f_organized --list

# 确认无误后再删除原文件（可选）
```

❌ **不推荐**:
```bash
# 直接移动文件，没有备份
python organize_fy3f_files.py /data2/lrx/obs -o /data2/lrx/fy3f_organized --move
```

### 2. 网格化处理

✅ **推荐做法**:
```bash
# 先用小数据集测试
python fy3f_to_era5_format.py /data2/lrx/fy3f_organized/2025/01/ \
    -o test/ --start 20250101 --end 20250102

# 确认无误后再处理全月
python fy3f_to_era5_format.py /data2/lrx/fy3f_organized/2025/01/ \
    -o /data2/lrx/fy3f_gridded/ --start 20250101 --end 20250131
```

### 3. 参数选择

| 场景 | 推荐参数 |
|------|----------|
| 快速预览 | `--resolution 1.0 --time-window 6 --grid-method nearest` |
| 标准处理 | `--resolution 0.25 --time-window 3 --grid-method nearest` |
| 高质量输出 | `--resolution 0.25 --time-window 3 --grid-method linear` |
| 存储受限 | `--resolution 0.5 --time-window 6` |

---

## ⚙️ 系统要求

### Python版本
- Python >= 3.7

### 依赖包
```bash
# organize_fy3f_files.py - 无需额外依赖（只用标准库）

# fy3f_to_era5_format.py
pip install xarray numpy scipy h5py netcdf4

# visualize_fy3f_gridded.py
pip install xarray numpy matplotlib cartopy

# collocation_fy3d_era5_fixed.py
pip install xarray numpy scipy h5py cfgrib
```

### 磁盘空间
- 原始FY-3F文件: ~50MB/文件
- 网格化NetCDF: ~100-200MB/月（取决于分辨率）
- 建议预留: 原始数据大小的3-5倍

---

## 📊 数据流程图

```
原始FY-3F文件（散乱）
         ↓
    [organize_fy3f_files.py]
         ↓
按年月整理的文件
         ↓
    [fy3f_to_era5_format.py]
         ↓
网格化NetCDF文件
         ↓
    [visualize_fy3f_gridded.py]
         ↓
图表和统计
         
         
网格化NetCDF文件 + ERA5数据
         ↓
    [collocation_fy3d_era5_fixed.py]
         ↓
机器学习训练数据
```

---

## ❓ 快速FAQ

**Q: 应该先整理还是先网格化？**
A: **推荐先整理**！这样可以：
- 更好地管理文件
- 按月批处理更方便
- 容易追踪处理进度

**Q: 文件整理用copy还是move？**
A: **推荐copy**（默认），更安全。确认无误后再手动删除原文件。

**Q: 网格化需要多长时间？**
A: 取决于文件数量和参数：
- 1个月数据（~150个文件）: 10-30分钟
- 使用nearest方法更快
- 更粗的网格（0.5°或1.0°）更快

**Q: 网格覆盖率为什么低？**
A: 卫星轨道数据天然有缺口。解决方法：
- 增加时间窗口（6或12小时）
- 使用粗网格（0.5°或1.0°）
- 合并多天数据

---

## 📞 获取帮助

```bash
# 查看各工具的详细帮助
python organize_fy3f_files.py --help
python fy3f_to_era5_format.py --help
python visualize_fy3f_gridded.py --help
python collocation_fy3d_era5_fixed.py --help
```

## 📚 文档索引

1. **ORGANIZE_FILES_README.md** - 文件整理工具详细说明
2. **FY3F_GRIDDING_GUIDE.md** - 网格化工具完整指南
3. **README_FIX.md** - 配准工具修复说明
4. **本文件 (QUICK_START.md)** - 快速开始和总览

---

祝数据处理顺利！🎉
