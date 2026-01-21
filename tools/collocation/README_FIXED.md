# FY-3D/FY-3F 与 ERA5 数据配准工具 - 修正版使用说明

## 🔧 问题修复

原始代码假设HDF文件使用以下变量名：
- `Brightness_Temperature`
- `Latitude`
- `Longitude`

但实际的FY-3F MWTS文件使用的是：
- `Data/Earth_Obs_BT` (亮温数据)
- `Geolocation/Latitude` (纬度)
- `Geolocation/Longitude` (经度)

**修正版代码已经更新，可以自动识别两种格式！**

## 📋 实际文件结构

根据 view_hdf.py 的运行结果，FY-3F文件结构为：

```
顶层变量/组:
├── Data
│   └── Earth_Obs_BT         # 亮温数据 (扫描线, 扫描角度, 通道)
├── Geolocation
│   ├── Latitude             # 纬度
│   ├── Longitude            # 经度
│   ├── Scnlin_daycnt       # 扫描线日计数（时间）
│   ├── Scnlin_mscnt        # 扫描线毫秒计数（时间）
│   ├── Altitude
│   ├── LandCover
│   ├── LandSeaMask
│   ├── SensorAzimuth
│   ├── SensorZenith
│   ├── SolarAzimuth
│   └── SolarZenith
└── QA
    ├── QA_Flag_Process
    ├── QA_Score
    └── Quality_Flag_Scnlin  # 质量标记
```

## 🚀 快速开始

### 步骤 1: 测试文件读取

首先验证您的FY-3D/FY-3F文件是否可以正确读取：

```bash
python test_fy3d_read.py /data2/lrx/obs/FY3F_MWTS-_ORBD_L1_20250111_1957_033KM_V0.HDF
```

**期望输出：**
```
======================================================================
FY-3D/FY-3F 文件结构测试
======================================================================
文件: /data2/lrx/obs/FY3F_MWTS-_ORBD_L1_20250111_1957_033KM_V0.HDF

✓ 文件成功打开

1️⃣  顶层结构:
   ├── Data
   ├── Geolocation
   ├── QA

2️⃣  亮温数据:
   ✓ 找到: Data/Earth_Obs_BT
   形状: (883, 90, 13)
   数据类型: float32
   样本数据 (第一个像元): [...]
   数据范围: [..., ...] K

3️⃣  地理定位数据:
   ✓ 找到: Geolocation/Latitude
   ✓ 找到: Geolocation/Longitude
   纬度形状: (883, 90)
   纬度范围: [..., ...]°
   经度范围: [..., ...]°

...

======================================================================
测试总结
======================================================================
✅ 所有必需数据都已找到，文件可以正常处理！

建议使用以下命令进行配准:
python collocation_fy3d_era5_fixed.py /data2/lrx/obs/FY3F_MWTS-_ORBD_L1_20250111_1957_033KM_V0.HDF <ERA5_file>
```

### 步骤 2: 执行配准

使用修正版脚本进行数据配准：

```bash
python collocation_fy3d_era5_fixed.py \
    /data2/lrx/obs/FY3F_MWTS-_ORBD_L1_20250111_1957_033KM_V0.HDF \
    /data2/lrx/era5/era5_20250111_12.grib \
    -o output_data
```

**自定义参数：**
```bash
python collocation_fy3d_era5_fixed.py \
    <FY3D_file> <ERA5_file> \
    --bt-min 80.0 \          # 最小亮温（K）
    --bt-max 350.0 \         # 最大亮温（K）
    --max-distance 0.3 \     # 最大匹配距离（度）
    --format both \          # 输出格式: numpy, hdf5, both
    -o my_output
```

### 步骤 3: 验证输出

配准完成后，会生成以下文件：

```
output_data_X.npy              # 亮温数据 (N, 13)
output_data_Y.npy              # 温度廓线 (N, 37)
output_data_pressure.npy       # 气压层 (37,)
output_data_metadata.json      # 元数据
output_data.h5                 # HDF5格式（可选）
```

验证数据：
```python
import numpy as np

X = np.load('output_data_X.npy')
Y = np.load('output_data_Y.npy')
pressure = np.load('output_data_pressure.npy')

print(f"亮温数据: {X.shape}")      # (N, 13)
print(f"温度廓线: {Y.shape}")      # (N, 37)
print(f"气压层: {pressure.shape}") # (37,)
```

## 📊 完整示例

### 示例 1: 单文件处理

```bash
# 1. 测试读取
python test_fy3d_read.py /data2/lrx/obs/FY3F_MWTS-_ORBD_L1_20250111_1957_033KM_V0.HDF

# 2. 执行配准
python collocation_fy3d_era5_fixed.py \
    /data2/lrx/obs/FY3F_MWTS-_ORBD_L1_20250111_1957_033KM_V0.HDF \
    /data2/lrx/era5/era5_20250111_12.grib \
    -o collocation_20250111

# 3. 可视化（可选）
python visualize_collocation.py collocation_20250111 -o ./plots
```

### 示例 2: 批量处理

如果您有多个文件需要处理，可以使用批处理脚本（需要先更新batch_collocation.py以使用修正版的读取器）：

```bash
python batch_collocation.py \
    /data2/lrx/obs \           # FY-3D文件目录
    /data2/lrx/era5 \          # ERA5文件目录
    -o ./batch_output \        # 输出目录
    -n 10 \                    # 限制处理前10个文件
    --merge                    # 合并所有输出
```

## 🔍 常见问题

### Q1: 如何查看HDF文件结构？

```python
import h5py

with h5py.File('your_file.HDF', 'r') as f:
    def print_structure(name, obj):
        print(name)
    f.visititems(print_structure)
```

或使用提供的工具：
```bash
python view_hdf.py /data2/lrx/obs/FY3F_MWTS-_ORBD_L1_20250111_1957_033KM_V0.HDF
```

### Q2: 亮温数据范围异常？

亮温合理范围通常是 **80-350 K**。如果看到异常值：

```bash
# 调整质量控制参数
python collocation_fy3d_era5_fixed.py \
    <files> \
    --bt-min 80.0 \
    --bt-max 350.0
```

### Q3: 匹配率太低？

如果空间匹配成功率低于50%，尝试增加最大匹配距离：

```bash
python collocation_fy3d_era5_fixed.py \
    <files> \
    --max-distance 1.0  # 增加到1度
```

### Q4: 内存不足？

对于大文件，使用批处理模式或减少质量控制范围以过滤更多异常数据。

## 🔄 与原版的区别

### 主要改进：

1. **✅ 自动格式检测**
   - 支持两种HDF格式（原始和FY-3F）
   - 自动选择正确的变量路径

2. **✅ 改进的时间处理**
   - 支持FY-3F的时间格式（Scnlin_daycnt/mscnt）
   - 从文件名提取时间作为后备

3. **✅ 更好的错误处理**
   - 清晰的错误消息
   - 文件结构验证

4. **✅ 调试信息**
   - 显示实际使用的变量路径
   - 打印数据范围以便验证

### 代码更改对照：

```python
# 原始代码
bt = f['Brightness_Temperature'][:]
lat = f['Latitude'][:]
lon = f['Longitude'][:]

# 修正版代码
if 'Data/Earth_Obs_BT' in f:
    bt = f['Data/Earth_Obs_BT'][:]
elif 'Brightness_Temperature' in f:
    bt = f['Brightness_Temperature'][:]

if 'Geolocation/Latitude' in f:
    lat = f['Geolocation/Latitude'][:]
    lon = f['Geolocation/Longitude'][:]
elif 'Latitude' in f:
    lat = f['Latitude'][:]
    lon = f['Longitude'][:]
```

## 📝 Python API 使用

在您的代码中使用修正版读取器：

```python
from collocation_fy3d_era5_fixed import DataCollocation

# 创建配准对象
collocation = DataCollocation()

# 执行配准
X, Y = collocation.process(
    fy3d_file='/data2/lrx/obs/FY3F_MWTS-_ORBD_L1_20250111_1957_033KM_V0.HDF',
    era5_file='/data2/lrx/era5/era5_20250111_12.grib',
    bt_min=80.0,
    bt_max=350.0,
    max_distance=0.5
)

# 保存结果
collocation.save_numpy('my_data')
collocation.save_hdf5('my_data.h5')

print(f"生成了 {len(X)} 个训练样本")
print(f"X shape: {X.shape}")  # (N, 13) - 13通道亮温
print(f"Y shape: {Y.shape}")  # (N, 37) - 37层温度廓线
```

## 🎯 下一步

1. **验证数据质量**：使用 visualize_collocation.py 生成图表
2. **批量处理**：处理整个月/年的数据
3. **训练模型**：使用生成的 (X, Y) 对训练机器学习模型
4. **优化参数**：根据您的应用调整质量控制和匹配参数

## 📧 支持

如果遇到问题：
1. 先运行 `test_fy3d_read.py` 验证文件格式
2. 检查错误消息中的具体变量路径
3. 确认ERA5文件格式正确（GRIB格式，包含温度变量）

---

**更新日期**: 2025-01-21  
**版本**: 2.0.0 (修正版)
