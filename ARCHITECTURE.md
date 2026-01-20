# 改进版 PAVMT-Unet 深度学习架构文档

## 📖 项目概述

本项目基于 PAVMT-Unet 论文进行改进，专门用于**卫星-再分析数据同化任务**。主要针对FY-3F卫星微波亮温数据与ERA5再分析背景场的融合问题，设计了物理感知的深度学习架构。

### 核心改进

1. **SpectralAdapter（光谱适配器）**：解决卫星亮温与背景场物理意义不同的问题
2. **Mask-Aware VCA（掩膜感知垂直通道注意力）**：处理卫星观测的缺失值问题
3. **AtmosphericPhysicLoss（大气物理约束损失）**：引入梯度损失保持锋面结构

---

## 🏗️ 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         改进版 PAVMT-Unet                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   卫星亮温 (B, 13, H, W)        背景场 (B, 37, H, W)                     │
│         │                              │                                │
│         └──────────┬───────────────────┘                                │
│                    ▼                                                    │
│            ┌───────────────┐                                           │
│            │SpectralAdapter│  ← 辐射空间 → 状态空间映射                   │
│            └───────┬───────┘                                           │
│                    ▼                                                    │
│         融合特征 (B, 37, H, W)                                          │
│                    │                                                    │
│                    ▼                                                    │
│            ┌───────────────┐                                           │
│            │   VRS Stem    │  ← 垂直重建主干                             │
│            └───────┬───────┘                                           │
│                    ▼                                                    │
│         ┌──────────┴──────────┐                                        │
│         │                     │                                        │
│    ┌────▼────┐          ┌────▼────┐                                   │
│    │ Encoder │◄─ VCA ──►│ Decoder │                                   │
│    │         │          │         │                                   │
│    │ Stage 1 │──────────│ Stage 4 │  ConvBlock                        │
│    │ Stage 2 │──────────│ Stage 3 │  ConvBlock                        │
│    │ Stage 3 │──────────│ Stage 2 │  PA-Mamba + MSA                   │
│    │ Stage 4 │──────────│ Stage 1 │  PA-Mamba + MSA                   │
│    └────┬────┘          └────▲────┘                                   │
│         │                     │                                        │
│         └──────────┬──────────┘                                        │
│                    ▼                                                    │
│            ┌───────────────┐                                           │
│            │     CVSC      │  ← 跨尺度垂直-空间耦合                       │
│            └───────┬───────┘                                           │
│                    ▼                                                    │
│            ┌───────────────┐                                           │
│            │  Output Head  │                                           │
│            └───────┬───────┘                                           │
│                    ▼                                                    │
│         分析场 (B, 37, H, W)                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 模块详解

### 1. SpectralAdapter（光谱适配器）

**目的**：将卫星辐射空间（亮温）映射到状态空间（温度廓线），并与背景场自适应融合。

**设计理念**：
- 卫星观测的13个微波通道的亮温与ERA5的37层温度在物理意义上不同
- 不能简单拼接，需要学习从辐射空间到状态空间的映射
- 使用可学习的权重实现自适应融合

**架构**：
```python
SpectralAdapter:
    ├── spectral_transform:  # 辐射→状态空间映射
    │   ├── Conv2d(13 → hidden_dim, 1×1)
    │   ├── BatchNorm2d + GELU
    │   ├── Conv2d(hidden_dim → 37, 1×1)
    │   └── BatchNorm2d + GELU
    │
    └── fusion_weight:  # 可学习融合权重 (1, 37, 1, 1)
        
    前向传播:
        sat_transformed = spectral_transform(satellite)
        alpha = sigmoid(fusion_weight)
        fused = alpha * sat_transformed + (1 - alpha) * background
```

**数学表达**：
$$\mathbf{F} = \sigma(\mathbf{W}) \cdot \mathcal{T}(\mathbf{X}_{sat}) + (1 - \sigma(\mathbf{W})) \cdot \mathbf{X}_{bg}$$

其中：
- $\mathcal{T}$: 光谱变换网络
- $\mathbf{W}$: 可学习权重
- $\sigma$: Sigmoid函数

---

### 2. Mask-Aware VCA（掩膜感知垂直通道注意力）

**目的**：根据观测掩膜动态调整注意力权重，有效观测区域增强卫星信息，无效区域依赖背景场。

**设计理念**：
- 卫星数据存在缺失值（云污染、扫描间隙等）
- 有效观测区域应更多依赖卫星信息
- 无效区域应更多依赖背景场先验

**架构**：
```python
MaskAwareVCA:
    ├── encoder_enhance:  # 增强encoder特征
    │   └── DW-Conv2d(C, 3×3) + BN + GELU
    │
    ├── decoder_gate:  # 处理decoder信号
    │   └── Upsample(2×) + Conv2d(1×1) + BN
    │
    ├── base_attention:  # 基础注意力
    │   └── Conv(C → C/4) + GELU + Conv(C/4 → C) + Sigmoid
    │
    └── mask_modulator:  # 掩膜调制
        └── Conv(C+1 → C, 3×3) + GELU + Conv(C → C, 1×1) + Sigmoid
```

**注意力计算**：
```python
# 基础注意力
combined = encoder_enhance(x_encoder) + decoder_gate(g_decoder)
base_alpha = base_attention(combined)

# 掩膜调制
if mask is not None:
    mask_resized = interpolate(mask, size=(H, W))
    mask_effect = sigmoid(mask_scale) * mask_resized
    mask_mod = mask_modulator(concat([combined, mask_resized]))
    alpha = base_alpha * (1 - mask_effect) + mask_mod * mask_effect
else:
    alpha = base_alpha

# 输出
output = x_encoder * alpha + g_decoder * (1 - alpha)
```

**物理意义**：
- `mask=1`（有效观测）：增强encoder（卫星）信息权重
- `mask=0`（无效观测）：抑制卫星信息，依赖decoder（背景场）先验

---

### 3. AtmosphericPhysicLoss（大气物理约束损失）

**目的**：综合多种损失约束，保证预测结果的物理一致性。

**组成**：

#### 3.1 L1损失（鲁棒性）
$$\mathcal{L}_{L1} = \frac{1}{N}\sum_{i}|\hat{y}_i - y_i|$$

比MSE对异常值更鲁棒，适合气象数据中可能存在的极端值。

#### 3.2 梯度损失（保持锋面结构）
使用Sobel算子计算空间梯度：

$$\nabla_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * \mathbf{X}$$

$$\nabla_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * \mathbf{X}$$

梯度损失：
$$\mathcal{L}_{grad} = \text{MSE}(\nabla_x\hat{\mathbf{Y}}, \nabla_x\mathbf{Y}) + \text{MSE}(\nabla_y\hat{\mathbf{Y}}, \nabla_y\mathbf{Y})$$

**物理意义**：保持气象场中的锋面、槽脊等梯度特征。

#### 3.3 物理约束（非负惩罚）
$$\mathcal{L}_{phys} = \lambda_{penalty} \cdot \text{mean}(\text{ReLU}(-\hat{\mathbf{Y}}))$$

确保浓度、温度(K)等物理量非负。

#### 3.4 总损失
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{L1} + \lambda_2 \mathcal{L}_{grad} + \lambda_3 \mathcal{L}_{phys}$$

默认参数：$\lambda_1=1.0$, $\lambda_2=0.5$, $\lambda_3=0.1$

---

### 4. PA-Mamba Mixer（物理感知Mamba混合器）

**目的**：模拟大气平流-扩散过程，捕捉长程空间依赖。

**三路并行架构**：

```
输入特征 (B, C, H, W)
         │
    ┌────┼────┬────────────┐
    ▼    │    ▼            ▼
┌───────┐│┌───────┐  ┌───────────┐
│DW-Conv│││Forward │  │ Backward  │
│(扩散) │││  SSM   │  │   SSM     │
└───┬───┘│└───┬───┘  └─────┬─────┘
    │    │    │            │
    │    │    │ 空间增强   │ 空间增强
    │    │    ▼            ▼
    │    │ ┌─────┐      ┌─────┐
    │    │ │Scan →│      │← Scan│
    │    │ └──┬──┘      └──┬──┘
    │    │    │            │
    └────┴────┴─────┬──────┘
                    ▼
              Concatenate
                    │
                    ▼
              Linear → C
                    │
                    ▼
              输出 + 残差
```

**物理对应**：
- **静态扩散路径**：模拟污染物的各向同性扩散
- **前向平流路径**：模拟"上风"方向的输送
- **后向平流路径**：模拟"下风"回流和局部涡旋

---

### 5. CVSC（跨尺度垂直-空间耦合器）

**目的**：在瓶颈层增强垂直和空间维度的特征交互。

**架构**：
```
输入
  │
  ▼
Channel Shuffle  ← 打破通道独立性
  │
  ├──────────────────────────────────┐
  │                                  │
  ▼                                  │
ConvPath                             │
  │                                  │
  ▼                                  │
PA-Mamba                             │
  │                                  │
  ▼                                  │
Self-Attention                       │
  │                                  │
  ▼                                  │
FFN                                  │
  │                                  │
  ▼                                  │
Fusion ◄─────────────────────────────┘
  │
  ▼
输出
```

**设计理念**：
- Channel Shuffle促进垂直层之间的信息交流
- 多路径设计模拟数据同化中的"增量校正"范式
- 残差连接保持背景场先验

---

## 📊 数据流

### 输入数据
```python
satellite:  (Batch, 13, H, W)   # FY-3F微波亮温，13通道
background: (Batch, 37, H, W)   # ERA5背景场，37层
mask:       (Batch, 1, H, W)    # 观测掩膜，1=有效，0=无效
```

### 输出数据
```python
analysis:   (Batch, 37, H, W)   # 分析场，37层温度廓线
```

### 数据集返回格式
```python
{
    'satellite':  torch.Tensor,  # 卫星亮温
    'background': torch.Tensor,  # ERA5背景场
    'label':      torch.Tensor,  # 分析场（标签）
    'mask':       torch.Tensor   # 观测掩膜
}
```

---

## 🔧 训练配置

### 推荐参数
```python
# 模型
base_channels = 64
num_stages = 4
num_heads = 8

# 损失函数
lambda_l1 = 1.0
lambda_gradient = 0.5
lambda_physics = 0.1

# 优化器
optimizer = AdamW
lr = 3e-4
weight_decay = 0.05

# 学习率调度
warmup_epochs = 5
scheduler = Cosine Annealing
min_lr = 1e-6

# 训练
epochs = 300
batch_size = 8
grad_clip = 1.0
```

### 参数分组策略
```python
# 卷积/线性层使用权重衰减
decay_params: weight_decay = 0.05

# Mamba状态空间矩阵和所有bias不使用权重衰减
no_decay_params: weight_decay = 0.0
```

---

## 📈 评估指标

1. **RMSE（均方根误差）**
   $$RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2}$$

2. **CORR（相关系数）**
   $$CORR = \frac{\sum_i(y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_i(y_i - \bar{y})^2 \sum_i(\hat{y}_i - \bar{\hat{y}})^2}}$$

3. **BIAS（偏差）**
   $$BIAS = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)$$

---

## 🚀 使用示例

### 快速开始
```bash
# 使用合成数据测试
python train.py --use_synthetic --train_samples 1000 --epochs 50

# 使用真实数据
python train.py --data_dir ./data --epochs 300 --batch_size 8
```

### 代码示例
```python
import torch
from models.improved_pavmt_unet import ImprovedPAVMTUnet
from modules.atmospheric_loss import AtmosphericPhysicLoss

# 创建模型
model = ImprovedPAVMTUnet(
    n_sat_channels=13,
    n_background_layers=37,
    base_channels=64,
    use_mask=True
)

# 创建损失函数
criterion = AtmosphericPhysicLoss(
    lambda_l1=1.0,
    lambda_gradient=0.5,
    lambda_physics=0.1
)

# 前向传播
satellite = torch.randn(4, 13, 64, 64)
background = torch.randn(4, 37, 64, 64)
mask = torch.randint(0, 2, (4, 1, 64, 64)).float()
label = torch.randn(4, 37, 64, 64)

output = model(satellite, background, mask)
loss = criterion(output, label, mask)

# 反向传播
loss.backward()
```

---

## 📁 项目结构

```
pavmt_unet_improved/
├── modules/
│   ├── __init__.py
│   ├── spectral_adapter.py      # SpectralAdapter模块
│   ├── mask_aware_vca.py        # Mask-Aware VCA模块
│   └── atmospheric_loss.py      # 大气物理约束损失
├── models/
│   └── improved_pavmt_unet.py   # 完整模型定义
├── data/
│   └── dataset.py               # 数据集类
├── train.py                     # 训练脚本
└── ARCHITECTURE.md              # 本文档
```

---

## 📚 参考文献

1. PAVMT-Unet: A Physics-Aware Hybrid Network for SO₂ Vertical Profile Assimilation
2. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
3. U-Net: Convolutional Networks for Biomedical Image Segmentation

---

## ⚙️ 依赖要求

```
torch >= 2.0.0
numpy >= 1.21.0
h5py >= 3.0.0
xarray >= 0.19.0
tensorboard >= 2.10.0
tqdm >= 4.62.0
mamba_ssm >= 1.0.0 (可选)
```

---

## 📝 更新日志

- **v1.0.0**: 初始版本
  - SpectralAdapter实现
  - Mask-Aware VCA实现
  - AtmosphericPhysicLoss实现
  - 完整训练流程
