# 卫星数据同化多GPU训练说明

## 文件结构

```
├── train.py              # 原始单卡训练脚本 (不变)
├── train_ddp.py          # 新增: 多卡DDP训练脚本
├── data_pipeline_v2.py   # 数据管道 (不变)
├── run_experiments.sh    # 原始实验脚本 (不变)
├── run_experiments_ddp.sh # 新增: 支持多卡的实验脚本
└── README_DDP.md         # 本说明文件
```

## 快速开始

### 1. 查看GPU状态

```bash
./run_experiments_ddp.sh status
```

输出示例：
```
========================================================================
GPU 状态
========================================================================
  GPU 0: NVIDIA GeForce ... | 显存: 8674/24576 MiB | 利用率: 0%
  GPU 1: NVIDIA GeForce ... | 显存: 8660/24576 MiB | 利用率: 0%
  GPU 2: NVIDIA GeForce ... | 显存: 8942/24576 MiB | 利用率: 18%
  GPU 3: NVIDIA GeForce ... | 显存: 8/24576 MiB | 利用率: 0%

[GPU] 可用GPU (显存<1GB): 3
========================================================================
```

### 2. 单卡训练 (使用原始 train.py)

```bash
# 使用GPU 3运行
./run_experiments_ddp.sh single physics_unet gated true true 3

# 或直接使用原脚本
CUDA_VISIBLE_DEVICES=3 python train.py --exp_name test --data_root /path/to/data
```

### 3. 多卡训练 (使用新的 train_ddp.py)

```bash
# 使用GPU 2,3 (2卡)
./run_experiments_ddp.sh single_ddp physics_unet gated true true 2,3

# 使用GPU 0,1,2,3 (4卡)
./run_experiments_ddp.sh single_ddp physics_unet gated true true 0,1,2,3
```

### 4. 直接使用 torchrun 启动多卡训练

```bash
# 2卡训练
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    train_ddp.py \
    --exp_name my_experiment \
    --data_root /data2/lrx/era_obs/npz/train \
    --model physics_unet \
    --fusion_mode gated \
    --use_aux true \
    --mask_aware true \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001

# 4卡训练
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    train_ddp.py \
    --exp_name my_experiment_4gpu \
    --data_root /data2/lrx/era_obs/npz/train \
    ...
```

### 5. 消融实验 (多卡)

```bash
# 使用GPU 2,3运行快速消融实验
./run_experiments_ddp.sh quick_ddp 2,3

# 使用4卡运行完整消融实验
./run_experiments_ddp.sh ablation_ddp 0,1,2,3
```

### 6. 断点续训 (多卡)

```bash
./run_experiments_ddp.sh resume_ddp outputs/exp_xxx/checkpoint.pth exp_xxx 2,3
```

### 7. 自动模式 (自动检测空闲GPU)

```bash
./run_experiments_ddp.sh auto physics_unet gated true true
```

## 参数说明

### train_ddp.py 新增参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sync_bn` | 是否使用同步BatchNorm | true |
| `--find_unused_parameters` | DDP是否查找未使用参数 | false |

### 学习率自动缩放

多卡训练时，学习率会自动按GPU数量线性缩放：

```
effective_lr = base_lr * num_gpus
```

例如：使用4卡，`--lr 0.0001` → 实际学习率 `0.0004`

### 有效批大小

```
effective_batch_size = batch_size_per_gpu * num_gpus
```

例如：使用4卡，`--batch_size 16` → 有效批大小 `64`

## 常见问题

### Q1: NCCL错误

如果遇到NCCL通信错误，尝试：

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

### Q2: 内存不足

减小 `--batch_size` 或使用更少的GPU。

### Q3: 模型加载问题

DDP训练保存的模型可以直接用单卡加载：

```python
# 加载到单卡
model = create_model('physics_unet')
checkpoint = torch.load('best_model.pth', map_location='cuda:0')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Q4: 如何只使用部分GPU?

通过 `CUDA_VISIBLE_DEVICES` 环境变量指定：

```bash
# 只使用GPU 2和3
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_ddp.py ...
```

## 性能对比

| 配置 | 有效批大小 | 每epoch时间 (估计) |
|------|------------|-------------------|
| 1x GPU | 16 | ~10 min |
| 2x GPU | 32 | ~5.5 min |
| 4x GPU | 64 | ~3 min |

*注：实际时间取决于数据量和GPU型号*

## 联系

如有问题，请检查日志文件 `outputs/*/train.log`。
