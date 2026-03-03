# PAS-Net 消融实验与对比实验指南

## 📋 概述

本目录包含 PAS-Net (Physics-Aware Spectral-Vertical Network) 论文的完整实验代码，包括消融实验和对比实验。

## 📁 文件结构

```
experiments/
├── models/
│   ├── __init__.py
│   ├── backbone.py          # 模型定义 (PAS-Net + 对比方法)
│   └── losses.py             # 损失函数 (混合物理损失)
├── train_experiment.py       # 实验训练脚本
├── evaluate_all.py           # 评估和结果汇总脚本
├── run_pasnet_experiments.sh # 自动化实验脚本
└── README.md                 # 本文件
```

## 🔬 实验设计

### 消融实验 (Ablation Study)

| 实验ID | 配置 | 说明 |
|--------|------|------|
| A1 | Full PAS-Net | 完整模型 (Baseline) |
| A2 | w/o Level-wise Norm | 使用全局标准化替代逐层标准化 |
| A3 | w/o Spectral Adapter | 直接拼接输入，无光谱适配器 |
| A4 | w/o Gradient Loss | 仅使用MSE损失，无Sobel梯度损失 |
| A5 | w/o Auxiliary Features | 不使用辅助地理/时间特征 |
| A6 | w/o Mask-Aware | 不使用掩码感知卷积 |
| A7 | w/o SE Block | 不使用通道注意力 |
| A8 | Fusion: concat | 使用拼接融合替代门控融合 |
| A9 | Fusion: add | 使用加法融合替代门控融合 |

### 对比实验 (Comparison Study)

| 实验ID | 方法 | 说明 |
|--------|------|------|
| C1 | Vanilla U-Net | 标准U-Net基线 |
| C2 | ResUNet | 残差U-Net |
| C3 | Attention U-Net | 注意力U-Net |

## 🚀 快速开始

### 1. 环境配置

```bash
# 确保已安装必要依赖
pip install torch torchvision numpy matplotlib tensorboard
```

### 2. 数据准备

将数据放置在以下结构：
```
/data2/lrx/era_obs/npz/
├── train/
│   ├── sample_0001.npz
│   ├── sample_0002.npz
│   └── ...
└── stats.npz  # 预计算的标准化统计量
```

每个 `.npz` 文件应包含:
- `obs`: 观测数据 [17, H, W]
- `bkg`: 背景场 [37, H, W]
- `target`: 目标分析场 [37, H, W]
- `aux`: 辅助特征 [4, H, W] (可选)

### 3. 运行实验

#### 方式一：使用自动化脚本 (推荐)

```bash
# 修改脚本中的数据路径
vim run_pasnet_experiments.sh

# 运行消融实验
./run_pasnet_experiments.sh ablation

# 运行对比实验
./run_pasnet_experiments.sh comparison

# 运行所有实验
./run_pasnet_experiments.sh all

# 快速测试 (5 epochs)
./run_pasnet_experiments.sh quick
```

#### 方式二：手动运行单个实验

```bash
# 单GPU训练
CUDA_VISIBLE_DEVICES=0 python train_experiment.py \
    --exp_name "A1_PASNet_Full" \
    --data_root "/data2/lrx/era_obs/npz/train" \
    --model pasnet \
    --fusion_mode gated \
    --use_aux true \
    --mask_aware true \
    --use_spectral_adapter true \
    --loss hybrid \
    --epochs 100

# 多GPU训练 (DDP)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_experiment.py \
    --exp_name "A1_PASNet_Full" \
    --data_root "/data2/lrx/era_obs/npz/train" \
    --model pasnet \
    --fusion_mode gated \
    --use_aux true \
    --mask_aware true \
    --use_spectral_adapter true \
    --loss hybrid \
    --epochs 100
```

### 4. 评估与结果汇总

```bash
# 生成结果表格和可视化
python evaluate_all.py \
    --output_dir outputs/experiments \
    --results_dir outputs/results \
    --generate_tables \
    --plot
```

## 📊 实验参数详解

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | pasnet | 模型类型 |
| `--fusion_mode` | str | gated | 融合模式: gated/concat/add |
| `--use_aux` | bool | true | 是否使用辅助特征 |
| `--mask_aware` | bool | true | 是否使用掩码感知 |
| `--use_spectral_adapter` | bool | true | 是否使用光谱适配器 |
| `--norm_mode` | str | levelwise | 标准化模式: levelwise/global |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 100 | 训练轮数 |
| `--batch_size` | int | 16 | 每GPU批大小 |
| `--lr` | float | 1e-4 | 学习率 |
| `--loss` | str | hybrid | 损失函数: mse/mae/hybrid |
| `--grad_loss_weight` | float | 0.1 | 梯度损失权重 |
| `--profile_loss_weight` | float | 0.5 | 高度加权损失权重 |

### 损失函数

混合物理损失 (Hybrid Physics Loss):
```
L = λ_mae * L_mae + λ_grad * L_grad + λ_profile * L_profile
```

- `L_mae`: MAE损失 (数值精度)
- `L_grad`: Sobel梯度损失 (结构保持)
- `L_profile`: 高度加权损失 (垂直一致性)

## 📈 预期结果

完成实验后，你应该能够填充论文中的以下表格：

### Table 1: 消融实验结果

| 配置 | Global RMSE | Trop. RMSE | Strat. RMSE |
|------|-------------|------------|-------------|
| Full PAS-Net | **X.XX** | **X.XX** | **X.XX** |
| w/o Level-wise Norm | X.XX | X.XX | X.XX |
| w/o Spectral Adapter | X.XX | X.XX | X.XX |
| w/o Gradient Loss | X.XX | X.XX | X.XX |
| ... | ... | ... | ... |

### Table 2: 对比实验结果

| Method | Global RMSE | Trop. RMSE | Strat. RMSE | #Params |
|--------|-------------|------------|-------------|---------|
| Vanilla U-Net | X.XX | X.XX | X.XX | X.XXM |
| ResUNet | X.XX | X.XX | X.XX | X.XXM |
| Attention U-Net | X.XX | X.XX | X.XX | X.XXM |
| **PAS-Net (Ours)** | **X.XX** | **X.XX** | **X.XX** | X.XXM |

## ❓ 常见问题

### Q1: 如何处理显存不足？
- 减小 `batch_size`
- 使用 `--amp true` 开启混合精度训练
- 使用 `pasnet_lite` 轻量版模型

### Q2: 如何恢复中断的训练？
```bash
python train_experiment.py --resume outputs/exp_name/checkpoint_latest.pth ...
```

### Q3: 如何只评估某个检查点？
```bash
python evaluate_all.py --checkpoint outputs/exp_name/checkpoint_best.pth
```

## 📝 引用

如果使用本代码，请引用：

```bibtex
@article{pasnet2024,
  title={Physics-Aware Spectral-Vertical Adaptation Network for High-Fidelity 
         Atmospheric Profile Retrieval from FY-3F Satellite Observations},
  author={Your Name},
  journal={Neural Computation},
  year={2024}
}
```

## 📞 联系方式

如有问题，请联系 [your-email@example.com]
