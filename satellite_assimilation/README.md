# 卫星数据同化深度学习框架 (Physics-Aware Satellite Data Assimilation)

> **版本**: 2.1  
> **状态**: 顶会/顶刊标准 + 配准数据支持  
> **最后更新**: 2026-01

---

## 📋 目录

1. [项目概述](#1-项目概述)
2. [核心创新点](#2-核心创新点)
3. [项目结构](#3-项目结构)
4. [快速开始](#4-快速开始)
5. [配准数据使用](#5-配准数据使用)
6. [模型架构](#6-模型架构)
7. [训练指南](#7-训练指南)
8. [消融实验](#8-消融实验)
9. [FAQ](#9-faq)

---

## 1. 项目概述

### 1.1 任务定义

将**FY-3F卫星亮温观测**同化到**ERA5再分析背景场**，生成高精度**大气温度廓线**。

### 1.2 支持的数据格式

本框架支持两种数据格式：

| 格式 | 文件类型 | 数据形状 | 适用场景 |
|------|----------|----------|----------|
| **配准点数据** | `.npy`, `.h5` | `(N, C)` | 温度廓线反演 (MLP模型) |
| **空间网格数据** | `.npz` | `(B, C, H, W)` | 空间同化 (U-Net模型) |

---

## 2. 核心创新点

| 创新点 | 模块 | 说明 |
|--------|------|------|
| 逐通道标准化 | `PointwiseNormalizer` | 保留各气压层信号特征 |
| 通道注意力 | `PhysicsAwareMLP` | 学习亮温通道重要性 |
| 加权损失 | `WeightedMSELoss` | 增强平流层权重 |
| 残差连接 | 多层残差块 | 稳定训练 |

---

## 3. 项目结构

```
satellite_assimilation/
├── README.md                      # 本文档
├── requirements.txt               # 依赖
│
├── collocated_data_loader.py     # 🆕 配准数据加载器 (支持.npy/.h5)
├── train_profile.py              # 🆕 温度廓线反演训练脚本
├── run_training.sh               # 🆕 交互式训练脚本 (配准数据)
│
├── data_pipeline_v2.py           # 空间数据管道 (支持.npz)
├── train.py                      # U-Net训练脚本
├── run_experiments.sh            # 消融实验脚本
│
└── models/
    ├── __init__.py
    └── backbone.py               # U-Net骨干网络
```

---

## 4. 快速开始

### 4.1 环境安装

```bash
pip install torch numpy h5py matplotlib tensorboard
```

### 4.2 使用交互式脚本

```bash
# 赋予执行权限
chmod +x run_training.sh

# 启动交互式菜单
./run_training.sh
```

交互式菜单提供：
- 📊 查看配准数据
- 🚀 单次训练
- 🔬 消融实验
- ⚡ 快速测试
- 🔄 断点续训
- 📈 查看结果

---

## 5. 配准数据使用

### 5.1 数据目录结构

配准后的数据应组织为：

```
/data2/lrx/era_obs/
├── 2024/
│   ├── 01/
│   │   ├── collocation_20240115_0530_X.npy   # 亮温 (N, 17)
│   │   ├── collocation_20240115_0530_Y.npy   # 温度廓线 (N, 37)
│   │   └── collocation_20240115_0530.h5      # HDF5格式 (可选)
│   ├── 02/
│   │   └── ...
│   └── ...
└── 2025/
    └── ...
```

### 5.2 命令行训练

```bash
# 基础训练
python train_profile.py \
    --data_root /data2/lrx/era_obs \
    --exp_name my_experiment \
    --model physics_mlp \
    --epochs 100

# 指定年月
python train_profile.py \
    --data_root /data2/lrx/era_obs \
    --year 2024 \
    --months 01 02 03 \
    --model physics_mlp \
    --epochs 100

# 使用加权损失
python train_profile.py \
    --data_root /data2/lrx/era_obs \
    --model physics_mlp \
    --loss weighted_mse \
    --epochs 100
```

### 5.3 Python代码使用

```python
from collocated_data_loader import (
    CollocatedDataset,
    create_mlp_model,
    ProfileMetrics
)

# 加载数据
dataset = CollocatedDataset(
    data_root='/data2/lrx/era_obs',
    compute_stats=True,
    year_filter='2024',
    month_filter=['01', '02', '03']
)

# 创建模型
model = create_mlp_model('physics_mlp')

# 前向传播
sample = dataset[0]
pred = model(sample['x'].unsqueeze(0))
print(f"输出形状: {pred.shape}")  # [1, 37]

# 评估
metrics = ProfileMetrics()
results = metrics.levelwise_rmse(pred, sample['y'].unsqueeze(0))
```

---

## 6. 模型架构

### 6.1 MLP模型 (用于配准点数据)

| 模型名称 | 参数量 | 特点 |
|----------|--------|------|
| `simple_mlp` | ~50K | 4层全连接，基线 |
| `res_mlp` | ~3M | 残差MLP，6层 |
| `physics_mlp` | ~350K | 通道注意力 + 残差 |

### 6.2 U-Net模型 (用于空间网格数据)

| 模型名称 | 参数量 | 特点 |
|----------|--------|------|
| `physics_unet_lite` | ~5M | 轻量级 |
| `physics_unet` | ~27M | 标准版 |
| `vanilla_unet` | ~13M | 消融基线 |

---

## 7. 训练指南

### 7.1 推荐配置

| 场景 | 模型 | epochs | batch_size | lr |
|------|------|--------|------------|-----|
| 快速测试 | physics_mlp | 10 | 512 | 1e-3 |
| 标准训练 | physics_mlp | 100 | 256 | 1e-3 |
| 完整训练 | physics_mlp | 200 | 128 | 5e-4 |

### 7.2 损失函数

| 损失函数 | 说明 |
|----------|------|
| `mse` | 均方误差 (默认) |
| `mae` | 平均绝对误差 |
| `huber` | Huber损失 |
| `weighted_mse` | 加权MSE (平流层权重×2) |

---

## 8. 消融实验

### 8.1 使用交互式脚本

```bash
./run_training.sh
# 选择 3) 🔬 消融实验
```

### 8.2 手动运行

```bash
# 基线
python train_profile.py --model simple_mlp --exp_name ablation_baseline

# 残差MLP
python train_profile.py --model res_mlp --exp_name ablation_resmlp

# 物理感知MLP
python train_profile.py --model physics_mlp --exp_name ablation_physics
```

### 8.3 预期结果

| 模型 | 全局RMSE | 平流层RMSE | 对流层RMSE |
|------|----------|------------|------------|
| simple_mlp | ~1.5 K | ~2.0 K | ~1.3 K |
| res_mlp | ~1.2 K | ~1.5 K | ~1.1 K |
| physics_mlp | ~1.0 K | ~1.2 K | ~0.9 K |

---

## 9. FAQ

### Q1: 数据目录中找不到文件？

确保数据目录结构正确：
```
/data2/lrx/era_obs/YYYY/MM/collocation_*_X.npy
```

### Q2: h5py相关错误？

安装h5py：
```bash
pip install h5py
```

### Q3: 如何查看训练曲线？

```bash
tensorboard --logdir outputs/exp_name/logs
```

### Q4: 如何断点续训？

```bash
python train_profile.py --resume outputs/exp_name/best_model.pth
```

---

## 📧 联系方式

如有问题，请提交Issue。

---

**祝实验顺利！** 🚀

---

## 1. 项目概述

### 1.1 任务定义

将**FY-3F卫星亮温观测**同化到**ERA5再分析背景场**，生成高精度**大气温度廓线**。

```
输入:
├── 观测场 (X_obs): FY-3F亮温, [B, 17, H, W], 含NaN轨道空隙
├── 背景场 (X_bkg): ERA5温度, [B, 37, H, W]
├── 辅助场 (X_aux): 纬度/经度/太阳天顶角/地表类型, [B, 4, H, W]
└── 掩码 (Mask): 有效性指示, [B, 1, H, W]

输出:
└── 分析场 (Y_pred): 同化后温度廓线, [B, 37, H, W]
```

### 1.2 物理背景

| 数据 | 来源 | 物理意义 |
|------|------|----------|
| FY-3F MWTS | 微波温度探测仪 | 17通道亮温，各通道对不同气压层敏感 |
| ERA5 | ECMWF再分析 | 37层气压面温度场 (1000hPa - 1hPa) |
| 辅助特征 | 计算得到 | 影响辐射传输的地理/时间因素 |

### 1.3 核心挑战

1. **轨道空隙**: 卫星扫描存在NaN缺测区域
2. **尺度差异**: 平流层温度变化 (~5K) 远小于对流层 (~30K)
3. **非线性映射**: 亮温→温度的逆辐射传输是强非线性问题

---

## 2. 核心创新点

### 2.1 创新点总结

| 创新点 | 模块 | 物理意义 | 预期提升 |
|--------|------|----------|----------|
| **逐通道Z-Score** | `LevelwiseNormalizer` | 保留各层信号特征 | 平流层RMSE ↓40% |
| **光谱适配器** | `SpectralAdapterStemV2` | 学习逆RTM映射 | 全局RMSE ↓15% |
| **辅助特征编码** | `AuxiliaryEncoder` | 融入地理先验 | 边界区域 ↓20% |
| **掩码感知卷积** | `MaskAwareConv2d` | 显式处理缺测 | 缺测鲁棒性 ↑90% |
| **门控融合** | `GatedFusion` | 自适应观测/背景权重 | 训练稳定性 ↑ |

### 2.2 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Physics-Aware U-Net Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  X_obs [B,17,H,W] ──┐                                                       │
│                     │                                                       │
│  X_bkg [B,37,H,W] ──┼──> SpectralAdapterStemV2 ──> [B,64,H,W]              │
│                     │         │                         │                   │
│  X_aux [B,4,H,W]  ──┤         │ SE-Block               │                   │
│                     │         │ AuxEncoder              │                   │
│  Mask  [B,1,H,W]  ──┘         │ GatedFusion             │                   │
│                               ▼                         ▼                   │
│                         ┌─────────────────────────────────────┐             │
│                         │         U-Net Encoder               │             │
│                         │  64 → 128 → 256 → 512              │             │
│                         │  (ResBlocks + SE + Downsample)      │             │
│                         └──────────────┬──────────────────────┘             │
│                                        │                                    │
│                         ┌──────────────▼──────────────────────┐             │
│                         │         Bottleneck (CBAM)           │             │
│                         └──────────────┬──────────────────────┘             │
│                                        │                                    │
│                         ┌──────────────▼──────────────────────┐             │
│                         │         U-Net Decoder               │             │
│                         │  512 → 256 → 128 → 64              │             │
│                         │  (ResBlocks + Skip + Upsample)      │             │
│                         └──────────────┬──────────────────────┘             │
│                                        │                                    │
│                         ┌──────────────▼──────────────────────┐             │
│                         │      Output Head + Residual         │             │
│                         │         Y_pred [B,37,H,W]           │             │
│                         └─────────────────────────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 项目结构

```
satellite_assimilation/
├── README.md                      # 本文档
├── requirements.txt               # 依赖列表
│
├── data_pipeline.py              # V1数据管道 (基础版)
├── data_pipeline_v2.py           # V2数据管道 (顶会标准)
│
├── models/
│   ├── __init__.py
│   └── backbone.py               # U-Net骨干网络
│
├── train.py                      # 训练脚本 (argparse接口)
├── evaluate.py                   # 评估脚本
│
├── run_experiments.sh            # Bash自动化脚本
│
├── outputs/                      # 实验输出
│   ├── {exp_name}/
│   │   ├── config.json           # 实验配置
│   │   ├── best_model.pth        # 最佳模型
│   │   ├── checkpoint_*.pth      # 检查点
│   │   ├── train.log             # 训练日志
│   │   └── logs/                 # TensorBoard日志
│   └── experiment_status.csv     # 实验状态汇总
│
└── docs/
    ├── ARCHITECTURE.md           # 架构详解
    ├── EXPERIMENTS.md            # 实验指南
    └── ABLATION.md               # 消融实验
```

---

## 4. 快速开始

### 4.1 环境安装

```bash
# 创建环境
conda create -n sat_assim python=3.10
conda activate sat_assim

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tensorboard h5py tqdm
```

### 4.2 最小示例

```python
import torch
from data_pipeline_v2 import InMemorySatelliteDataset, create_synthetic_data_v2
from models.backbone import create_model

# 1. 准备数据
obs, bkg, target, aux = create_synthetic_data_v2(n_samples=100)
dataset = InMemorySatelliteDataset(
    obs, bkg, target, aux, compute_stats=True
)

# 2. 创建模型
model = create_model('physics_unet')

# 3. 前向传播
sample = dataset[0]
pred = model(
    sample['obs'].unsqueeze(0),
    sample['bkg'].unsqueeze(0),
    sample['mask'].unsqueeze(0),
    sample['aux'].unsqueeze(0)
)
print(f"输出形状: {pred.shape}")  # [1, 37, 64, 64]
```

### 4.3 命令行训练

```bash
# 基础训练
python train.py \
    --exp_name my_experiment \
    --data_root /path/to/data \
    --model physics_unet \
    --epochs 100

# 完整配置
python train.py \
    --exp_name full_experiment \
    --data_root /path/to/data \
    --model physics_unet \
    --fusion_mode gated \
    --use_aux true \
    --mask_aware true \
    --batch_size 16 \
    --lr 0.0001 \
    --loss combined \
    --tensorboard true
```

---

## 5. 数据准备

### 5.1 数据格式

每个样本保存为`.npz`文件:

```python
np.savez(
    'sample_0001.npz',
    obs=obs_array,      # [17, H, W], float32, 含NaN
    bkg=bkg_array,      # [37, H, W], float32
    target=target_array, # [37, H, W], float32
    aux=aux_array       # [4, H, W], float32, 可选
)
```

### 5.2 辅助特征计算

```python
def compute_aux_features(lat, lon, time, land_mask):
    """
    计算辅助特征
    
    Args:
        lat: 纬度 [H, W], 范围 [-90, 90]
        lon: 经度 [H, W], 范围 [-180, 180]
        time: datetime对象
        land_mask: 地表类型 [H, W], {0: 海洋, 1: 陆地}
    
    Returns:
        aux: [4, H, W]
    """
    # 归一化到 [-1, 1]
    lat_norm = lat / 90.0
    lon_norm = lon / 180.0
    
    # 计算太阳天顶角 (简化版)
    hour = time.hour + time.minute / 60
    sza = np.cos(2 * np.pi * (hour - 12) / 24)  # 简化计算
    sza = np.full_like(lat, sza)
    
    aux = np.stack([lat_norm, lon_norm, sza, land_mask], axis=0)
    return aux.astype(np.float32)
```

### 5.3 统计量预计算

对于大数据集，建议预计算统计量:

```python
from data_pipeline_v2 import LazySatelliteERA5Dataset

# 创建数据集
dataset = LazySatelliteERA5Dataset.from_directory('/path/to/data')

# 计算并保存统计量
dataset.compute_statistics(
    n_samples=10000,  # 使用部分样本
    save_path='/path/to/stats.npz'
)
```

---

## 6. 模型架构

### 6.1 可用模型

| 模型名称 | 参数量 | 特点 | 适用场景 |
|----------|--------|------|----------|
| `physics_unet_lite` | ~2M | 轻量级 | 快速实验/调试 |
| `physics_unet` | ~15M | 标准版 | 正式训练 |
| `physics_unet_large` | ~50M | 大型版 | 最终实验 |
| `vanilla_unet` | ~15M | 无物理感知 | 消融对比 |

### 6.2 配置示例

```python
from models.backbone import UNetConfig, PhysicsAwareUNet

config = UNetConfig(
    # 输入
    obs_channels=17,
    bkg_channels=37,
    aux_channels=4,
    out_channels=37,
    
    # Stem
    stem_channels=64,
    fusion_mode='gated',  # 'concat', 'add', 'gated'
    use_aux=True,
    mask_aware=True,
    
    # Encoder
    encoder_channels=[64, 128, 256, 512],
    encoder_depths=[2, 2, 2, 2],
    
    # Bottleneck
    bottleneck_channels=512,
    use_attention=True,
    
    # Decoder
    decoder_channels=[256, 128, 64, 64],
    
    # 训练
    dropout=0.1,
    deep_supervision=False
)

model = PhysicsAwareUNet(config)
```

---

## 7. 训练指南

### 7.1 命令行参数

```bash
python train.py --help
```

**关键参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | physics_unet | 模型类型 |
| `--fusion_mode` | gated | 融合模式 |
| `--use_aux` | true | 使用辅助特征 |
| `--mask_aware` | true | 掩码感知 |
| `--loss` | mse | 损失函数 (mse/mae/huber/combined) |
| `--grad_loss_weight` | 0.1 | 梯度损失权重 |
| `--amp` | true | 混合精度训练 |

### 7.2 损失函数

**Combined Loss (推荐):**

$$L_{total} = L_{MSE} + \lambda_{grad} \cdot L_{grad} + \lambda_{deep} \cdot L_{deep}$$

其中:
- $L_{grad}$: Sobel梯度损失，保留锋面结构
- $L_{deep}$: 深度监督损失，加速收敛

### 7.3 学习率调度

```python
# Cosine Annealing (推荐)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# OneCycle
scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps)
```

### 7.4 训练技巧

1. **预热**: 前5个epoch使用较小学习率
2. **梯度裁剪**: `grad_clip=1.0` 防止梯度爆炸
3. **混合精度**: 启用AMP加速训练，减少显存
4. **早停**: 监控验证损失，防止过拟合

---

## 8. 消融实验

### 8.1 自动化脚本

```bash
# 运行快速消融实验 (5组核心对比)
./run_experiments.sh quick

# 运行完整消融实验 (24组)
./run_experiments.sh ablation

# 单次实验
./run_experiments.sh single physics_unet gated true true

# 汇总结果
./run_experiments.sh summary
```

### 8.2 实验设计

| 实验 | 模型 | 融合 | 辅助 | 掩码 | 验证假设 |
|------|------|------|------|------|----------|
| Baseline | vanilla_unet | concat | ✗ | ✗ | 基准线 |
| +LevelNorm | physics_unet | add | ✗ | ✗ | 逐通道标准化有效性 |
| +Adapter | physics_unet | gated | ✗ | ✗ | 光谱适配器有效性 |
| +Aux | physics_unet | gated | ✓ | ✗ | 辅助特征有效性 |
| +Mask (Full) | physics_unet | gated | ✓ | ✓ | 掩码感知有效性 |

### 8.3 结果表格模板

| 方法 | 全局RMSE | 平流层RMSE | 对流层RMSE | 梯度相关 | 缺测Drop |
|------|----------|------------|------------|----------|----------|
| Baseline | 1.50 K | 2.10 K | 1.40 K | 0.65 | -50% |
| +LevelNorm | 1.35 K | **1.20 K** | 1.38 K | 0.68 | -48% |
| +Adapter | 1.28 K | 1.15 K | 1.32 K | 0.72 | -45% |
| +Aux | 1.25 K | 1.13 K | 1.30 K | 0.75 | -40% |
| **Full (Ours)** | **1.22 K** | 1.12 K | **1.28 K** | **0.78** | **-5%** |

---

## 9. 评估指标

### 9.1 核心指标

```python
from data_pipeline_v2 import AssimilationMetrics

metrics = AssimilationMetrics()

# 分层RMSE
results = metrics.levelwise_rmse(pred, target)
print(f"全局: {results['global']:.3f} K")
print(f"平流层: {results['stratosphere']:.3f} K")
print(f"对流层: {results['troposphere']:.3f} K")

# 梯度相似性
grad = metrics.gradient_loss(pred, target)
print(f"梯度RMSE: {grad['grad_rmse']:.3f}")
print(f"梯度相关: {grad['grad_correlation']:.3f}")

# 缺测鲁棒性
robust = metrics.gap_robustness_test(model, dataloader)
```

### 9.2 可视化

```python
from data_pipeline_v2 import plot_levelwise_rmse, plot_gap_robustness

# 分层RMSE廓线图
plot_levelwise_rmse(results, save_path='levelwise_rmse.png')

# 缺测鲁棒性曲线
plot_gap_robustness(robust, save_path='gap_robustness.png')
```

---

## 10. 论文对比实验

### 10.1 对比对象

#### 传统方法 (第一层级)

| 方法 | 来源 | 说明 |
|------|------|------|
| 3D-Var | WRF-DA | 业务系统标杆 |
| Kriging插值 | - | 最弱基线 |

#### 深度学习基线 (第二层级)

| 方法 | 说明 |
|------|------|
| Vanilla U-Net | 无物理感知的原始U-Net |
| ResNet-FPN | 特征金字塔网络 |

#### 领域SOTA (第三层级)

| 方法 | 论文 | 说明 |
|------|------|------|
| PAVMT-Unet | [参考文献] | 需复现对比 |
| Swin-UNet | Vision Transformer | Transformer架构 |

### 10.2 重要可视化

1. **偏差分布图 (Bias Map)**
   - 展示 `(Prediction - Truth)` 的空间分布
   - 关注缺测区域的表现

2. **功率谱密度 (PSD)**
   - 证明保留高频细节
   - 对比不同方法的平滑程度

3. **通道注意力权重**
   - 分析SE模块学到的通道重要性
   - 验证物理一致性

---

## 11. API参考

### 11.1 数据模块

```python
# 懒加载数据集
from data_pipeline_v2 import LazySatelliteERA5Dataset

dataset = LazySatelliteERA5Dataset(
    file_list=['sample1.npz', 'sample2.npz'],
    use_aux=True,
    mask_mode='any',
    cache_size=100
)

# 内存数据集
from data_pipeline_v2 import InMemorySatelliteDataset

dataset = InMemorySatelliteDataset(
    obs_data, bkg_data, target_data, aux_data,
    compute_stats=True
)
```

### 11.2 模型模块

```python
from models.backbone import create_model, UNetConfig

# 使用预设配置
model = create_model('physics_unet')

# 自定义配置
config = UNetConfig(fusion_mode='gated', use_aux=True)
model = create_model('physics_unet', config=config)
```

### 11.3 评估模块

```python
from data_pipeline_v2 import AssimilationMetrics

metrics = AssimilationMetrics(
    pressure_levels=pressure_array,
    stratosphere_threshold=100.0
)

# 分层RMSE
results = metrics.levelwise_rmse(pred, target)

# 梯度损失
grad = metrics.gradient_loss(pred, target)

# 缺测鲁棒性
robust = metrics.gap_robustness_test(model, loader)
```

---

## 12. FAQ

### Q1: 内存不足怎么办?

**A**: 使用`LazySatelliteERA5Dataset`替代内存加载:
```python
dataset = LazySatelliteERA5Dataset.from_directory(
    '/path/to/data',
    cache_size=0  # 禁用缓存
)
```

### Q2: 训练不收敛?

**A**: 检查以下几点:
1. 确保数据已正确标准化
2. 降低学习率 (`--lr 1e-5`)
3. 增加预热轮数 (`--warmup_epochs 10`)
4. 使用梯度裁剪 (`--grad_clip 1.0`)

### Q3: 如何断点续训?

**A**: 使用`--resume`参数:
```bash
python train.py --resume outputs/exp_xxx/checkpoint_epoch50.pth
```

### Q4: TensorBoard查看训练曲线?

**A**: 
```bash
tensorboard --logdir outputs/exp_xxx/logs
```

### Q5: 如何添加自定义损失函数?

**A**: 修改`train.py`中的`CombinedLoss`类，或创建新的损失类。

---

## 📚 参考文献

1. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. CVPR.
2. Liu, G., et al. (2018). Image inpainting for irregular holes using partial convolutions. ECCV.
3. [PAVMT-Unet相关论文]
4. [ERA5数据集文档]
5. [FY-3F卫星参数]

---

## 📝 更新日志

### V2.0 (2026-01)
- ✅ 添加懒加载数据集支持TB级数据
- ✅ 添加辅助特征编码器
- ✅ 添加掩码感知卷积
- ✅ 完善消融实验指标
- ✅ 添加Bash自动化脚本
- ✅ 完善文档

### V1.0 (2026-01)
- ✅ 基础数据管道
- ✅ SpectralAdapterStem模块
- ✅ 基础测试

---

## 📧 联系方式

如有问题，请提交Issue或联系作者。

---

**祝实验顺利！** 🚀
