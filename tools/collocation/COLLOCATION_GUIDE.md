# FY-3D与ERA5卫星资料同化数据配准工具

## 📚 项目简介

这是一套完整的卫星资料同化数据处理工具，用于将**FY-3D MWTS L1C5卫星亮温数据**与**ERA5再分析温度廓线**进行时空配准（Pixel-wise Collocation），生成用于机器学习的训练数据对。

### 🎯 核心功能

1. **数据读取**: 读取FY-3D H5格式和ERA5 GRIB格式数据
2. **质量控制**: 自动过滤异常亮温值（<100K或>400K）
3. **时空匹配**: 使用KDTree或pyresample进行空间配准
4. **垂直插值**: 将ERA5的37层温度廓线匹配到每个卫星像元
5. **批量处理**: 支持多文件批处理和数据合并
6. **数据可视化**: 丰富的可视化工具检查数据质量

## 📦 数据格式

### 输入数据

**FY-3D MWTS L1C5** (风云三号微波温度计):
```
文件格式: HDF5 (.L1c)
文件命名: FY3D_MWTSX_ORBT_L2_ATP_MLT_NUL_YYYYMMDD_HHMM_033KM_MS.L1c

数据结构:
├── Brightness_Temperature  # (Scan_Line, Scan_Angle, 13_Channels)
├── Latitude                # (Scan_Line, Scan_Angle)
├── Longitude               # (Scan_Line, Scan_Angle)
├── Obs_Time                # 观测时间
└── Quality_Flag            # 质量标记（可选）

通道信息:
- 13个微波通道
- 探测范围: 50.3 GHz - 57.6 GHz (O2吸收带)
- 垂直分辨率: ~5-10 km
```

**ERA5 Reanalysis**:
```
文件格式: GRIB (.grib)
文件命名: era5_YYYYMMDD_HH.grib

数据结构:
- 变量: 't' (temperature)
- 维度: (time, isobaricInhPa, latitude, longitude)
- 气压层: 37层 (1-1000 hPa)
- 时间分辨率: 1小时
- 空间分辨率: 0.25° × 0.25°
```

### 输出数据

```python
X: (N_samples, 13_Channels)  # 卫星亮温
Y: (N_samples, 37_Levels)    # 温度廓线
pressure_levels: (37,)        # 气压层信息
```

## 🚀 快速开始

### 安装依赖

```bash
# 基础依赖
pip install h5py xarray numpy scipy

# GRIB支持
pip install eccodes cfgrib

# 可选：高级重采样
pip install pyresample

# 可选：可视化
pip install matplotlib cartopy
```

**使用Conda（推荐）**:
```bash
conda create -n satellite_da python=3.9
conda activate satellite_da
conda install -c conda-forge h5py xarray eccodes cfgrib scipy pyresample matplotlib cartopy
```

### 方法1: 单文件处理（基础）

```bash
# 使用KDTree方法（推荐，速度快）
python collocation_fy3d_era5.py \
    FY3D_MWTSX_ORBT_L2_ATP_MLT_NUL_20210701_1455_033KM_MS.L1c \
    era5_20210701_12.grib \
    -o output_data

# 自定义参数
python collocation_fy3d_era5.py \
    <FY3D文件> <ERA5文件> \
    --bt-min 80.0 \          # 最小亮温
    --bt-max 350.0 \         # 最大亮温
    --max-distance 0.3 \     # 最大匹配距离（度）
    --format hdf5            # 输出格式
```

**输出**:
```
output_data_X.npy           # 亮温数据
output_data_Y.npy           # 温度廓线
output_data_pressure.npy    # 气压层
output_data_metadata.json   # 元数据
```

### 方法2: 使用pyresample（高级）

```bash
python collocation_pyresample.py \
    FY3D_MWTSX_ORBT_L2_ATP_MLT_NUL_20210701_1455_033KM_MS.L1c \
    era5_20210701_12.grib
```

**特点**:
- 更精确的重采样算法
- 支持多种插值方法
- 适合高精度应用

### 方法3: 批量处理

```bash
# 处理整个目录
python batch_collocation.py \
    ./fy3d_data \      # FY-3D文件目录
    ./era5_data \      # ERA5文件目录
    -o ./batch_output  # 输出目录

# 限制处理数量
python batch_collocation.py ./fy3d_data ./era5_data -n 10

# 处理并合并
python batch_collocation.py ./fy3d_data ./era5_data --merge
```

**输出结构**:
```
batch_output/
├── pair_0001_X.npy
├── pair_0001_Y.npy
├── pair_0002_X.npy
├── pair_0002_Y.npy
├── ...
├── merged_dataset_X.npy    # 合并后的数据
├── merged_dataset_Y.npy
└── batch_processing_log.json
```

## 📊 数据可视化

```bash
# 生成所有可视化图表
python visualize_collocation.py output_data -o ./plots

# 指定FY-3D文件以绘制覆盖图
python visualize_collocation.py output_data --fy3d <FY3D文件>
```

**生成的图表**:
1. `bt_histogram.png` - 亮温分布直方图
2. `temperature_profiles.png` - 温度廓线
3. `correlation_matrix.png` - 通道相关性矩阵
4. `bt_vs_temp.png` - 亮温-温度散点图
5. `data_coverage.png` - 数据空间覆盖图

## 🔬 Python API 使用

### 基础用法

```python
from collocation_fy3d_era5 import DataCollocation

# 创建配准对象
collocation = DataCollocation()

# 执行配准
X, Y = collocation.process(
    fy3d_file='FY3D_MWTSX_ORBT_L2_ATP_MLT_NUL_20210701_1455_033KM_MS.L1c',
    era5_file='era5_20210701_12.grib',
    bt_min=100.0,
    bt_max=400.0,
    max_distance=0.5
)

# 保存结果
collocation.save_numpy('my_data')
collocation.save_hdf5('my_data.h5')

# 访问中间结果
bt_data = collocation.fy3d_reader.data['bt']
temp_profiles = collocation.era5_reader.matched_profiles['profiles']
```

### 自定义处理流程

```python
from collocation_fy3d_era5 import FY3D_Reader, ERA5_Reader
import numpy as np

# 1. 读取FY-3D
fy3d = FY3D_Reader('fy3d_file.L1c')
fy3d.read()
fy3d.quality_control(bt_min=80.0, bt_max=350.0)
sat_points = fy3d.flatten_to_points()

# 2. 读取ERA5
era5 = ERA5_Reader('era5_file.grib')
era5.read()
era5.build_spatial_tree()

# 3. 空间匹配
profiles = era5.interpolate_to_satellite(sat_points, max_distance=0.3)

# 4. 自定义后处理
X = sat_points['bt']
Y = profiles

# 归一化
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
Y_norm = (Y - Y.mean(axis=0)) / Y.std(axis=0)
```

### 使用pyresample

```python
from collocation_pyresample import PyresampleCollocation

collocator = PyresampleCollocation()

# 读取数据
collocator.read_fy3d('fy3d_file.L1c')
collocator.read_era5('era5_file.grib')

# 重采样（可调整参数）
collocator.resample_era5_to_swath(
    radius_of_influence=50000,  # 50 km
    neighbours=1,               # 最近邻数量
    fill_value=np.nan
)

# 创建训练数据
X, Y = collocator.create_training_data()
```

## 🛠️ 高级功能

### 1. 时间窗口匹配

```python
from datetime import datetime, timedelta

def find_era5_for_time_window(fy3d_time, era5_files, window_hours=3):
    """
    在时间窗口内查找最接近的ERA5文件
    """
    closest_file = None
    min_diff = timedelta(hours=999)
    
    for era5_file in era5_files:
        # 从文件名提取时间
        era5_time = extract_time_from_filename(era5_file)
        time_diff = abs(fy3d_time - era5_time)
        
        if time_diff <= timedelta(hours=window_hours) and time_diff < min_diff:
            min_diff = time_diff
            closest_file = era5_file
    
    return closest_file
```

### 2. 多变量配准

```python
# 修改 ERA5_Reader 以支持多变量
class MultiVariableERA5Reader(ERA5_Reader):
    def read_multiple_variables(self, variables=['t', 'q', 'u', 'v']):
        """读取多个变量"""
        ds = xr.open_dataset(self.filepath, engine='cfgrib')
        
        self.data = {}
        for var in variables:
            if var in ds:
                self.data[var] = ds[var].values
        
        return self.data
```

### 3. 垂直插值到特定层

```python
from scipy.interpolate import interp1d

def interpolate_to_custom_levels(profiles, original_levels, target_levels):
    """
    将温度廓线插值到自定义气压层
    
    Parameters:
    -----------
    profiles : ndarray (N, n_original_levels)
    original_levels : ndarray (n_original_levels,)
    target_levels : ndarray (n_target_levels,)
    """
    n_samples = profiles.shape[0]
    n_target = len(target_levels)
    
    interpolated = np.zeros((n_samples, n_target))
    
    for i in range(n_samples):
        # 创建插值函数（log压力坐标）
        f = interp1d(np.log(original_levels), profiles[i], 
                    kind='linear', fill_value='extrapolate')
        interpolated[i] = f(np.log(target_levels))
    
    return interpolated
```

### 4. 数据增强

```python
def augment_data(X, Y, noise_level=0.01):
    """
    数据增强：添加随机噪声
    """
    X_aug = X + np.random.normal(0, noise_level * X.std(), X.shape)
    Y_aug = Y + np.random.normal(0, noise_level * Y.std(), Y.shape)
    
    return X_aug, Y_aug
```

## 📈 数据质量控制

### 质量标记

```python
def advanced_quality_control(bt, lat, lon):
    """
    高级质量控制
    """
    qc_mask = np.ones(bt.shape, dtype=bool)
    
    # 1. 亮温范围检查
    qc_mask &= (bt >= 100) & (bt <= 400)
    
    # 2. 地理范围检查
    qc_mask &= (lat >= -90) & (lat <= 90)
    qc_mask &= (lon >= -180) & (lon <= 180)
    
    # 3. 通道一致性检查
    # 相邻通道不应有过大差异
    for i in range(bt.shape[-1] - 1):
        diff = np.abs(bt[..., i] - bt[..., i+1])
        qc_mask[..., i] &= diff < 50  # K
    
    # 4. 空间一致性检查
    # 可以添加窗口平均等检查
    
    return qc_mask
```

### 统计诊断

```python
def diagnostic_statistics(X, Y):
    """
    计算数据集统计信息
    """
    stats = {
        'X': {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0),
            'missing': np.sum(np.isnan(X), axis=0)
        },
        'Y': {
            'mean': np.mean(Y, axis=0),
            'std': np.std(Y, axis=0),
            'min': np.min(Y, axis=0),
            'max': np.max(Y, axis=0),
            'missing': np.sum(np.isnan(Y), axis=0)
        }
    }
    
    return stats
```

## 🎓 最佳实践

### 1. 数据预处理

```python
# 标准化处理流程
def preprocess_pipeline(fy3d_file, era5_file):
    """标准化预处理流程"""
    
    # 读取数据
    collocation = DataCollocation()
    X, Y = collocation.process(fy3d_file, era5_file)
    
    # 归一化
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)
    
    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std
    
    # 保存归一化参数
    np.savez('normalization_params.npz',
             X_mean=X_mean, X_std=X_std,
             Y_mean=Y_mean, Y_std=Y_std)
    
    return X_norm, Y_norm
```

### 2. 交叉验证分割

```python
from sklearn.model_selection import train_test_split

def split_train_val_test(X, Y, val_size=0.15, test_size=0.15, random_state=42):
    """
    划分训练/验证/测试集
    """
    # 先分出测试集
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    
    # 再分出验证集
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"训练集: {len(X_train)}")
    print(f"验证集: {len(X_val)}")
    print(f"测试集: {len(X_test)}")
    
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
```

### 3. 性能优化

```python
# 使用内存映射处理大数据
X_mmap = np.load('large_X.npy', mmap_mode='r')
Y_mmap = np.load('large_Y.npy', mmap_mode='r')

# 批量读取
def batch_generator(X, Y, batch_size=1000):
    """数据批量生成器"""
    n_samples = len(X)
    for i in range(0, n_samples, batch_size):
        yield X[i:i+batch_size], Y[i:i+batch_size]
```

## 🐛 常见问题

### Q1: ImportError: No module named 'cfgrib'

**解决方案**:
```bash
conda install -c conda-forge cfgrib eccodes
```

### Q2: 内存不足

**解决方案**:
- 使用批处理模式
- 减小 max_distance 参数
- 使用内存映射文件

### Q3: 时间不匹配

**解决方案**:
- 检查文件命名格式
- 调整 time_tolerance 参数
- 手动指定时间对应关系

### Q4: 空间匹配失败

**解决方案**:
- 增大 max_distance (如1.0度)
- 检查经纬度范围
- 使用pyresample方法

## 📚 参考文献

1. FY-3D MWTS数据手册
2. ERA5 Documentation: https://confluence.ecmwf.int/display/CKB/ERA5
3. PyResample: https://pyresample.readthedocs.io/

## 📧 技术支持

如有问题，请检查：
1. 数据格式是否正确
2. 依赖包是否完整安装
3. 文件路径是否正确

---

**更新日期**: 2025-01-19  
**版本**: 1.0.0
