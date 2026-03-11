#!/usr/bin/env python3
"""
===============================================================================
卫星数据同化 - 多GPU分布式训练脚本 (DDP)
Satellite Data Assimilation - Multi-GPU Distributed Training Script
===============================================================================

用法:
    # 单机多卡 (推荐使用 torchrun)
    torchrun --nproc_per_node=2 train_ddp.py --exp_name test_ddp --data_root /path/to/data
    
    # 或使用传统方式
    python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py ...

    # 指定GPU
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_ddp.py ...

功能:
    1. 基于 PyTorch DistributedDataParallel (DDP)
    2. 自动数据分片和梯度同步
    3. 支持混合精度训练
    4. 仅在主进程保存检查点和日志
    5. 兼容原 train.py 的所有参数

===============================================================================
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline_v2 import (
    InMemorySatelliteDataset,
    LazySatelliteERA5Dataset,
    LevelwiseNormalizer,
    AssimilationMetrics,
    DataConfig,
    ModelConfig
)


# =============================================================================
# Part 1: 分布式工具函数
# =============================================================================

def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun 方式启动
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM 方式启动
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        # 单卡模式
        print("[DDP] 未检测到分布式环境，使用单卡模式")
        return 0, 1, 0
    
    # 设置当前设备
    torch.cuda.set_device(local_rank)
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30)
    )
    
    # 同步所有进程
    dist.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """判断是否为主进程"""
    return rank == 0


def print_rank0(msg: str, rank: int):
    """仅在主进程打印"""
    if is_main_process(rank):
        print(msg)


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """跨进程平均张量"""
    if world_size == 1:
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


# =============================================================================
# Part 2: 参数解析
# =============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='卫星数据同化多GPU训练脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # === 实验配置 ===
    exp_group = parser.add_argument_group('实验配置')
    exp_group.add_argument('--exp_name', type=str, default='experiment_ddp',
                           help='实验名称')
    exp_group.add_argument('--output_dir', type=str, default='outputs',
                           help='输出目录')
    exp_group.add_argument('--seed', type=int, default=42,
                           help='随机种子')
    exp_group.add_argument('--resume', type=str, default=None,
                           help='恢复训练的checkpoint路径')
    
    # === 数据配置 ===
    data_group = parser.add_argument_group('数据配置')
    data_group.add_argument('--data_root', type=str, required=True,
                            help='数据根目录')
    data_group.add_argument('--stats_file', type=str, default=None,
                            help='预计算统计量文件路径')
    data_group.add_argument('--train_ratio', type=float, default=0.8,
                            help='训练集比例')
    data_group.add_argument('--val_ratio', type=float, default=0.1,
                            help='验证集比例')
    data_group.add_argument('--num_workers', type=int, default=4,
                            help='DataLoader工作进程数')
    
    # === 模型配置 ===
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument('--model', type=str, default='physics_unet',
                             choices=['physics_unet', 'physics_unet_lite', 
                                     'physics_unet_large', 'vanilla_unet', 'fuxi_da'],
                             help='模型类型')
    model_group.add_argument('--fusion_mode', type=str, default='gated',
                             choices=['concat', 'add', 'gated'],
                             help='融合模式')
    model_group.add_argument('--use_aux', type=str, default='true',
                             choices=['true', 'false'],
                             help='是否使用辅助特征')
    model_group.add_argument('--mask_aware', type=str, default='true',
                             choices=['true', 'false'],
                             help='是否使用掩码感知')
    model_group.add_argument('--use_spectral_stem', type=str, default='true',
                             help='是否使用SpectralAdapterStemV2 (false=消融V4: 标准卷积Stem)')
    model_group.add_argument('--deep_supervision', type=str, default='false',
                             choices=['true', 'false'],
                             help='是否使用深度监督')
    
    # === 训练配置 ===
    train_group = parser.add_argument_group('训练配置')
    train_group.add_argument('--epochs', type=int, default=100,
                             help='训练轮数')
    train_group.add_argument('--batch_size', type=int, default=16,
                             help='每个GPU的批大小')
    train_group.add_argument('--lr', type=float, default=1e-4,
                             help='基础学习率 (会根据GPU数量自动缩放)')
    train_group.add_argument('--weight_decay', type=float, default=1e-5,
                             help='权重衰减')
    train_group.add_argument('--scheduler', type=str, default='cosine',
                             choices=['cosine', 'onecycle', 'none'],
                             help='学习率调度器')
    train_group.add_argument('--warmup_epochs', type=int, default=5,
                             help='预热轮数')
    train_group.add_argument('--grad_clip', type=float, default=1.0,
                             help='梯度裁剪阈值')
    
    # === 损失函数配置 ===
    loss_group = parser.add_argument_group('损失函数配置')
    loss_group.add_argument('--loss', type=str, default='mse',
                            choices=['mse', 'mae', 'huber', 'combined'],
                            help='损失函数类型')
    loss_group.add_argument('--grad_loss_weight', type=float, default=0.1,
                            help='梯度损失权重 (仅combined模式)')
    loss_group.add_argument('--deep_loss_weight', type=float, default=0.3,
                            help='深度监督损失权重')
    loss_group.add_argument('--use_increment', default=False, action='store_true',
                            help='训练增量目标 (Δ=target-bkg) 而非绝对温度')
    loss_group.add_argument('--increment_stats', type=str, default='',
                            help='增量统计文件路径 (.npz，含inc_mean/inc_std)')
    
    # === 设备配置 ===
    device_group = parser.add_argument_group('设备配置')
    device_group.add_argument('--amp', type=str, default='true',
                              choices=['true', 'false'],
                              help='是否使用混合精度训练')
    device_group.add_argument('--sync_bn', type=str, default='true',
                              choices=['true', 'false'],
                              help='是否使用同步BatchNorm')
    device_group.add_argument('--find_unused_parameters', type=str, default='false',
                              choices=['true', 'false'],
                              help='DDP是否查找未使用参数')
    
    # === 日志配置 ===
    log_group = parser.add_argument_group('日志配置')
    log_group.add_argument('--log_interval', type=int, default=10,
                           help='日志打印间隔 (步)')
    log_group.add_argument('--val_interval', type=int, default=1,
                           help='验证间隔 (轮)')
    log_group.add_argument('--save_interval', type=int, default=200,
                           help='保存间隔 (轮)')
    log_group.add_argument('--tensorboard', type=str, default='true',
                           choices=['true', 'false'],
                           help='是否使用TensorBoard')
    
    args = parser.parse_args()
    
    # 转换字符串布尔值
    args.use_aux = args.use_aux.lower() == 'true'
    args.mask_aware = args.mask_aware.lower() == 'true'
    args.use_spectral_stem = args.use_spectral_stem.lower() == 'true'
    args.deep_supervision = args.deep_supervision.lower() == 'true'
    args.amp = args.amp.lower() == 'true'
    args.tensorboard = args.tensorboard.lower() == 'true'
    args.sync_bn = args.sync_bn.lower() == 'true'
    args.find_unused_parameters = args.find_unused_parameters.lower() == 'true'
    
    return args


# =============================================================================
# Part 3: 工具函数
# =============================================================================

def set_seed(seed: int, rank: int = 0) -> None:
    """设置随机种子 (每个进程不同)"""
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace
) -> None:
    """保存检查点 (仅保存模型本身，不带DDP包装)"""
    # 如果是DDP模型，获取内部模型
    model_to_save = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'args': vars(args)
    }
    torch.save(checkpoint, path)
    print(f"  ✓ 检查点已保存: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    map_location: str = 'cpu'
) -> Tuple[int, float]:
    """加载检查点"""
    checkpoint = torch.load(path, map_location=map_location)
    
    # 处理DDP模型
    model_to_load = model.module if hasattr(model, 'module') else model
    
    # 处理可能的 'module.' 前缀问题
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 移除 'module.' 前缀
        else:
            new_state_dict[k] = v
    
    model_to_load.load_state_dict(new_state_dict)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"  ✓ 检查点已加载: {path}")
    print(f"    Epoch: {epoch}, Best Val Loss: {best_val_loss:.6f}")
    
    return epoch, best_val_loss


# =============================================================================
# Part 4: 损失函数
# =============================================================================

class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(
        self,
        grad_weight: float = 0.1,
        deep_weight: float = 0.3,
        base_loss: str = 'mse'
    ):
        super().__init__()
        
        self.grad_weight = grad_weight
        self.deep_weight = deep_weight
        
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss()
        elif base_loss == 'mae':
            self.base_loss = nn.L1Loss()
        elif base_loss == 'huber':
            self.base_loss = nn.HuberLoss()
        else:
            self.base_loss = nn.MSELoss()
        
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
    
    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        pred_flat = pred.view(B * C, 1, H, W)
        target_flat = target.view(B * C, 1, H, W)
        
        pred_gx = F.conv2d(pred_flat, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred_flat, self.sobel_y, padding=1)
        target_gx = F.conv2d(target_flat, self.sobel_x, padding=1)
        target_gy = F.conv2d(target_flat, self.sobel_y, padding=1)
        
        return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        deep_preds: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        base = self.base_loss(pred, target)
        grad = self.gradient_loss(pred, target)
        
        deep = torch.tensor(0.0, device=pred.device)
        if deep_preds:
            for dp in deep_preds:
                deep = deep + self.base_loss(dp, target)
            deep = deep / len(deep_preds)
        
        total = base + self.grad_weight * grad + self.deep_weight * deep
        
        return total, {
            'total': total.item(),
            'base': base.item(),
            'grad': grad.item(),
            'deep': deep.item()
        }


# =============================================================================
# Part 5: 分布式训练器
# =============================================================================

class DDPTrainer:
    """分布式数据并行训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_sampler: DistributedSampler,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object],
        criterion: nn.Module,
        device: torch.device,
        args: argparse.Namespace,
        rank: int,
        world_size: int,
        writer: Optional[object] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.writer = writer
        
        # 增量训练统计量 (可选)
        if getattr(args, 'use_increment', False) and args.increment_stats:
            inc_data = np.load(args.increment_stats)
            st_data  = np.load(args.stats_file)
            self._inc_mean = torch.tensor(inc_data['inc_mean'], dtype=torch.float32)
            self._inc_std  = torch.tensor(inc_data['inc_std'],  dtype=torch.float32)
            self._bkg_mean = torch.tensor(st_data['bkg_mean'],    dtype=torch.float32)
            self._bkg_std  = torch.tensor(st_data['bkg_std'],     dtype=torch.float32)
            self._tgt_mean = torch.tensor(st_data['target_mean'], dtype=torch.float32)
            self._tgt_std  = torch.tensor(st_data['target_std'],  dtype=torch.float32)
            print_rank0(f"增量训练模式已启用 (inc_std range: "
                        f"{inc_data['inc_std'].min():.3f}–{inc_data['inc_std'].max():.3f} K)", rank)
        else:
            self._inc_mean = None

        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if args.amp else None
        
        # 评估指标
        self.metrics = AssimilationMetrics()
        
        # 最佳验证损失
        self.best_val_loss = float('inf')
        
        # 输出目录 (仅主进程创建)
        self.exp_dir = Path(args.output_dir) / args.exp_name
        if is_main_process(rank):
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            with open(self.exp_dir / 'config.json', 'w') as f:
                json.dump(vars(args), f, indent=2)
    
    def _to_inc_target(self, bkg: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """将归一化绝对温度目标转换为归一化增量目标.
        bkg, target: (B, C, H, W) 归一化值
        returns: (B, C, H, W) 归一化增量
        """
        def v(t): return t.to(bkg.device).view(1, -1, 1, 1)
        bkg_phys = bkg    * v(self._bkg_std) + v(self._bkg_mean)
        tgt_phys = target * v(self._tgt_std) + v(self._tgt_mean)
        inc_phys = tgt_phys - bkg_phys
        return (inc_phys - v(self._inc_mean)) / v(self._inc_std)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        # 设置epoch以打乱数据
        self.train_sampler.set_epoch(epoch)
        
        total_loss = 0
        loss_components = {'base': 0, 'grad': 0, 'deep': 0}
        n_batches = len(self.train_loader)
        
        for i, batch in enumerate(self.train_loader):
            obs = batch['obs'].to(self.device, non_blocking=True)
            bkg = batch['bkg'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True)
            target = batch['target'].to(self.device, non_blocking=True)
            aux = batch.get('aux')
            if aux is not None:
                aux = aux.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()

            if self._inc_mean is not None:
                target = self._to_inc_target(bkg, target)
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    output = self.model(obs, bkg, mask, aux)
                    if isinstance(output, tuple):
                        pred, deep_preds = output
                    else:
                        pred, deep_preds = output, None
                    loss, loss_dict = self.criterion(pred, target, deep_preds)
                
                self.scaler.scale(loss).backward()
                
                if self.args.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(obs, bkg, mask, aux)
                if isinstance(output, tuple):
                    pred, deep_preds = output
                else:
                    pred, deep_preds = output, None
                
                loss, loss_dict = self.criterion(pred, target, deep_preds)
                loss.backward()
                
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip
                    )
                
                self.optimizer.step()
            
            total_loss += loss_dict['total']
            for k in loss_components:
                loss_components[k] += loss_dict.get(k, 0)
            
            # 日志 (仅主进程)
            if (i + 1) % self.args.log_interval == 0 and is_main_process(self.rank):
                print(f"  [{i+1}/{n_batches}] Loss: {loss_dict['total']:.6f}")
        
        # 平均损失
        avg_loss = total_loss / n_batches
        for k in loss_components:
            loss_components[k] /= n_batches
        
        # 跨进程同步损失 (用于准确的日志记录)
        if self.world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            avg_loss = reduce_tensor(avg_loss_tensor, self.world_size).item()
        
        return {'loss': avg_loss, **loss_components}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证 (所有进程参与，结果同步)"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in self.val_loader:
            obs = batch['obs'].to(self.device, non_blocking=True)
            bkg = batch['bkg'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True)
            target = batch['target'].to(self.device, non_blocking=True)
            aux = batch.get('aux')
            if aux is not None:
                aux = aux.to(self.device, non_blocking=True)

            if self._inc_mean is not None:
                target = self._to_inc_target(bkg, target)
            
            output = self.model(obs, bkg, mask, aux)
            if isinstance(output, tuple):
                pred, _ = output
            else:
                pred = output
            
            loss, _ = self.criterion(pred, target, None)
            total_loss += loss.item()
            
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
        
        # 合并
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        levelwise = self.metrics.levelwise_rmse(all_preds, all_targets)
        grad_metrics = self.metrics.gradient_loss(all_preds, all_targets)
        
        # 跨进程同步
        if self.world_size > 1:
            metrics_tensor = torch.tensor([
                avg_loss,
                levelwise['global'].item(),
                levelwise['stratosphere'].item(),
                levelwise['troposphere'].item(),
                grad_metrics['grad_rmse'].item(),
                grad_metrics['grad_correlation'].item()
            ], device=self.device)
            
            metrics_tensor = reduce_tensor(metrics_tensor, self.world_size)
            
            avg_loss = metrics_tensor[0].item()
            global_rmse = metrics_tensor[1].item()
            strat_rmse = metrics_tensor[2].item()
            trop_rmse = metrics_tensor[3].item()
            grad_rmse = metrics_tensor[4].item()
            grad_corr = metrics_tensor[5].item()
        else:
            global_rmse = levelwise['global'].item()
            strat_rmse = levelwise['stratosphere'].item()
            trop_rmse = levelwise['troposphere'].item()
            grad_rmse = grad_metrics['grad_rmse'].item()
            grad_corr = grad_metrics['grad_correlation'].item()
        
        return {
            'loss': avg_loss,
            'global_rmse': global_rmse,
            'strat_rmse': strat_rmse,
            'trop_rmse': trop_rmse,
            'grad_rmse': grad_rmse,
            'grad_corr': grad_corr
        }
    
    def train(self, start_epoch: int = 0) -> None:
        """完整训练流程"""
        if is_main_process(self.rank):
            print("\n" + "=" * 70)
            print(f"开始分布式训练: {self.args.exp_name}")
            print(f"GPU数量: {self.world_size}")
            print(f"每GPU批大小: {self.args.batch_size}")
            print(f"有效批大小: {self.args.batch_size * self.world_size}")
            print(f"总轮数: {self.args.epochs}")
            print("=" * 70)
        
        for epoch in range(start_epoch, self.args.epochs):
            if is_main_process(self.rank):
                print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
                print("-" * 40)
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            if is_main_process(self.rank):
                print(f"  训练 Loss: {train_metrics['loss']:.6f}")
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
                if is_main_process(self.rank):
                    current_lr = self.scheduler.get_last_lr()[0]
                    print(f"  学习率: {current_lr:.2e}")
            
            # 验证
            if (epoch + 1) % self.args.val_interval == 0:
                val_metrics = self.validate(epoch)
                
                if is_main_process(self.rank):
                    print(f"  验证 Loss: {val_metrics['loss']:.6f}")
                    print(f"  全局RMSE: {val_metrics['global_rmse']:.4f} K")
                    print(f"  平流层RMSE: {val_metrics['strat_rmse']:.4f} K")
                    print(f"  对流层RMSE: {val_metrics['trop_rmse']:.4f} K")
                    
                    # 保存最佳模型
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        save_checkpoint(
                            self.exp_dir / 'best_model.pth',
                            self.model, self.optimizer, self.scheduler,
                            epoch, self.best_val_loss, self.args
                        )
                    
                    # TensorBoard
                    if self.writer:
                        self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                        self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                        self.writer.add_scalar('RMSE/global', val_metrics['global_rmse'], epoch)
                        self.writer.add_scalar('RMSE/stratosphere', val_metrics['strat_rmse'], epoch)
                        self.writer.add_scalar('RMSE/troposphere', val_metrics['trop_rmse'], epoch)
            
            # 定期保存
            if (epoch + 1) % self.args.save_interval == 0 and is_main_process(self.rank):
                save_checkpoint(
                    self.exp_dir / f'checkpoint_epoch{epoch+1}.pth',
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.best_val_loss, self.args
                )
            
            # 同步所有进程
            if self.world_size > 1:
                dist.barrier()
        
        if is_main_process(self.rank):
            print("\n" + "=" * 70)
            print("训练完成!")
            print(f"最佳验证损失: {self.best_val_loss:.6f}")
            print(f"模型保存至: {self.exp_dir}")
            print("=" * 70)


# =============================================================================
# Part 6: 主函数
# =============================================================================

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 初始化分布式环境
    rank, world_size, local_rank = setup_distributed()
    
    # 设置随机种子
    set_seed(args.seed, rank)
    
    # 设备
    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print_rank0(f"使用 {world_size} 个GPU进行训练", rank)
    
    # =========================================================================
    # 数据准备
    # =========================================================================
    print_rank0("\n准备数据...", rank)
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print_rank0(f"警告: 数据目录不存在: {data_root}", rank)
        print_rank0("使用合成数据进行演示...", rank)
        
        from data_pipeline_v2 import create_synthetic_data_v2
        obs, bkg, target, aux = create_synthetic_data_v2(n_samples=100)
        
        dataset = InMemorySatelliteDataset(
            obs_data=obs,
            bkg_data=bkg,
            target_data=target,
            aux_data=aux if args.use_aux else None,
            compute_stats=True
        )
    else:
        _all_files = sorted(f for f in data_root.glob('**/*.npz')
                            if f.name not in ('stats.npz', 'dataset_split.json', 'increment_stats.npz'))
        if not _all_files:
            raise ValueError(f"数据目录中未找到.npz文件: {data_root}")
        
        # 过滤全零目标的损坏文件 (每个进程独立过滤，速度快)
        file_list = []
        n_corrupt = 0
        for _f in _all_files:
            try:
                _d = np.load(str(_f))
                if _d['target'].sum() != 0:
                    file_list.append(str(_f))
                else:
                    n_corrupt += 1
            except Exception:
                n_corrupt += 1
        print_rank0(f"  有效文件: {len(file_list)}, 损坏文件: {n_corrupt}", rank)
        
        dataset = LazySatelliteERA5Dataset(
            file_list=file_list,
            use_aux=args.use_aux
        )
        
        if args.stats_file and Path(args.stats_file).exists():
            stats = np.load(args.stats_file)
            dataset.obs_normalizer = LevelwiseNormalizer(
                stats['obs_mean'], stats['obs_std'], name='obs'
            )
            dataset.bkg_normalizer = LevelwiseNormalizer(
                stats['bkg_mean'], stats['bkg_std'], name='bkg'
            )
            dataset.target_normalizer = LevelwiseNormalizer(
                stats['target_mean'], stats['target_std'], name='target'
            )
        else:
            # 仅在主进程计算统计量
            if is_main_process(rank):
                dataset.compute_statistics(n_samples=min(1000, len(dataset)))
            if world_size > 1:
                dist.barrier()
    
    # 划分数据集
    n_total = len(dataset)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val
    
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print_rank0(f"  训练集: {len(train_set)}", rank)
    print_rank0(f"  验证集: {len(val_set)}", rank)
    print_rank0(f"  测试集: {len(test_set)}", rank)
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # DDP训练时建议丢弃不完整的batch
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # =========================================================================
    # 模型准备
    # =========================================================================
    print_rank0("\n准备模型...", rank)
    
    # 动态导入模型
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(ROOT_DIR)
    sys.path.append(ROOT_DIR)
    
    try:
        from models.backbone import create_model, UNetConfig
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent / 'models'))
        from backbone import create_model, UNetConfig
    
    # 模型配置
    if args.model not in ('vanilla_unet', 'fuxi_da'):
        config = UNetConfig(
            fusion_mode=args.fusion_mode,
            use_aux=args.use_aux,
            mask_aware=args.mask_aware,
            use_spectral_stem=args.use_spectral_stem,
            deep_supervision=args.deep_supervision
        )
        model = create_model(args.model, config=config)
    else:
        model = create_model(args.model)
    
    # 同步BatchNorm
    if args.sync_bn and world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print_rank0("  ✓ 已转换为SyncBatchNorm", rank)
    
    model = model.to(device)
    
    # 包装为DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.find_unused_parameters
        )
        print_rank0("  ✓ 已包装为DistributedDataParallel", rank)
    
    print_rank0(f"  参数量: {count_parameters(model):,}", rank)
    
    # =========================================================================
    # 优化器和调度器
    # =========================================================================
    # 学习率线性缩放
    scaled_lr = args.lr * world_size
    print_rank0(f"  基础学习率: {args.lr}, 缩放后: {scaled_lr}", rank)
    
    optimizer = AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=args.weight_decay
    )
    
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=scaled_lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader)
        )
    else:
        scheduler = None
    
    # =========================================================================
    # 损失函数
    # =========================================================================
    if args.loss == 'combined':
        criterion = CombinedLoss(
            grad_weight=args.grad_loss_weight,
            deep_weight=args.deep_loss_weight
        )
    else:
        base_criterion = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'huber': nn.HuberLoss()
        }[args.loss]
        
        class SimpleLoss(nn.Module):
            def __init__(self, loss_fn):
                super().__init__()
                self.loss_fn = loss_fn
            
            def forward(self, pred, target, deep_preds=None):
                loss = self.loss_fn(pred, target)
                return loss, {'total': loss.item(), 'base': loss.item(), 
                             'grad': 0, 'deep': 0}
        
        criterion = SimpleLoss(base_criterion)
    
    criterion = criterion.to(device)
    
    # =========================================================================
    # TensorBoard (仅主进程)
    # =========================================================================
    writer = None
    if args.tensorboard and is_main_process(rank):
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(args.output_dir) / args.exp_name / 'logs'
            writer = SummaryWriter(log_dir)
            print(f"  TensorBoard日志: {log_dir}")
        except ImportError:
            print("  警告: TensorBoard不可用")
    
    # =========================================================================
    # 恢复训练
    # =========================================================================
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(
            args.resume, model, optimizer, scheduler,
            map_location=f'cuda:{local_rank}' if world_size > 1 else 'cpu'
        )
        start_epoch += 1
    
    # =========================================================================
    # 训练
    # =========================================================================
    trainer = DDPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler if train_sampler else DistributedSampler(train_set, 1, 0),
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        args=args,
        rank=rank,
        world_size=world_size,
        writer=writer
    )
    
    try:
        trainer.train(start_epoch)
    finally:
        # 清理
        if writer:
            writer.close()
        cleanup_distributed()


if __name__ == '__main__':
    main()
