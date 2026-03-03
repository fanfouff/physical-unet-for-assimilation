#!/usr/bin/env python3
"""
===============================================================================
卫星数据同化训练脚本 (支持预划分数据集)
Training Script with Pre-split Dataset Support
===============================================================================

用法:
    # 使用预划分数据集 (默认)
    python train.py --data_root /path/to/npz --use_split true
    
    # 使用原始方式 (random_split)
    python train.py --data_root /path/to/npz --use_split false

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
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# DDP相关导入
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

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
# Part 1: 参数解析
# =============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数 (增强版)"""
    parser = argparse.ArgumentParser(
        description='卫星数据同化训练脚本 (支持预划分数据集)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # === 实验配置 ===
    exp_group = parser.add_argument_group('实验配置')
    exp_group.add_argument('--exp_name', type=str, default='experiment',
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
    data_group.add_argument('--use_split', type=str, default='true',
                            choices=['true', 'false'],
                            help='是否使用预划分的数据集 (train/val/test目录)')
    data_group.add_argument('--train_ratio', type=float, default=0.8,
                            help='训练集比例 (仅当use_split=false时使用)')
    data_group.add_argument('--val_ratio', type=float, default=0.1,
                            help='验证集比例 (仅当use_split=false时使用)')
    data_group.add_argument('--num_workers', type=int, default=4,
                            help='DataLoader工作进程数')
    
    # === 模型配置 ===
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument('--model', type=str, default='physics_unet',
                             choices=['physics_unet', 'physics_unet_lite', 
                                     'physics_unet_large', 'vanilla_unet'],
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
    model_group.add_argument('--deep_supervision', type=str, default='false',
                             choices=['true', 'false'],
                             help='是否使用深度监督')
    
    # === 训练配置 ===
    train_group = parser.add_argument_group('训练配置')
    train_group.add_argument('--epochs', type=int, default=100,
                             help='训练轮数')
    train_group.add_argument('--batch_size', type=int, default=16,
                             help='批大小')
    train_group.add_argument('--lr', type=float, default=1e-4,
                             help='初始学习率')
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
    
    # === 设备配置 ===
    device_group = parser.add_argument_group('设备配置')
    device_group.add_argument('--device', type=str, default='auto',
                              help='设备 (auto/cpu/cuda/cuda:0)')
    device_group.add_argument('--amp', type=str, default='true',
                              choices=['true', 'false'],
                              help='是否使用混合精度训练')
    device_group.add_argument('--multi_gpu', type=str, default='false',
                              choices=['true', 'false'],
                              help='是否使用多GPU训练 (DataParallel)')
    device_group.add_argument('--ddp', type=str, default='false',
                              choices=['true', 'false'],
                              help='是否使用分布式数据并行 (DDP, 需配合torchrun使用)')
    
    # === 日志配置 ===
    log_group = parser.add_argument_group('日志配置')
    log_group.add_argument('--log_interval', type=int, default=10,
                           help='日志打印间隔 (步)')
    log_group.add_argument('--val_interval', type=int, default=1,
                           help='验证间隔 (轮)')
    log_group.add_argument('--save_interval', type=int, default=10,
                           help='保存间隔 (轮)')
    log_group.add_argument('--tensorboard', type=str, default='true',
                           choices=['true', 'false'],
                           help='是否使用TensorBoard')
    
    args = parser.parse_args()
    
    # 转换字符串布尔值
    args.use_aux = args.use_aux.lower() == 'true'
    args.mask_aware = args.mask_aware.lower() == 'true'
    args.deep_supervision = args.deep_supervision.lower() == 'true'
    args.amp = args.amp.lower() == 'true'
    args.tensorboard = args.tensorboard.lower() == 'true'
    args.use_split = args.use_split.lower() == 'true'
    args.multi_gpu = args.multi_gpu.lower() == 'true'
    args.ddp = args.ddp.lower() == 'true'
    
    return args


# =============================================================================
# Part 2: 工具函数
# =============================================================================

def set_seed(seed: int) -> None:
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """获取设备"""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


# =============================================================================
# DDP 辅助函数
# =============================================================================

def setup_ddp():
    """初始化DDP环境"""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    return local_rank


def cleanup_ddp():
    """清理DDP环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """判断是否为主进程"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """获取进程总数"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """获取当前进程rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def print_main(*args, **kwargs):
    """仅在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)


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
    """保存检查点 (支持DDP/DataParallel)"""
    # 处理DDP/DataParallel包装的模型
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'args': vars(args)
    }
    torch.save(checkpoint, path)
    print_main(f"  ✓ 检查点已保存: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None
) -> Tuple[int, float]:
    """加载检查点 (支持DDP/DataParallel)"""
    checkpoint = torch.load(path, map_location='cpu')
    
    # 处理DDP/DataParallel包装的模型
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
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
# Part 3: 损失函数
# =============================================================================

class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    L = L_mse + λ_grad * L_grad + λ_deep * L_deep
    """
    
    def __init__(
        self,
        grad_weight: float = 0.1,
        deep_weight: float = 0.3,
        base_loss: str = 'mse'
    ):
        super().__init__()
        
        self.grad_weight = grad_weight
        self.deep_weight = deep_weight
        
        # 基础损失
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss()
        elif base_loss == 'mae':
            self.base_loss = nn.L1Loss()
        elif base_loss == 'huber':
            self.base_loss = nn.HuberLoss()
        else:
            self.base_loss = nn.MSELoss()
        
        # Sobel算子
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
    
    def gradient_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算梯度损失"""
        B, C, H, W = pred.shape
        
        # 展平通道
        pred_flat = pred.view(B * C, 1, H, W)
        target_flat = target.view(B * C, 1, H, W)
        
        # 计算梯度
        pred_gx = F.conv2d(pred_flat, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred_flat, self.sobel_y, padding=1)
        
        target_gx = F.conv2d(target_flat, self.sobel_x, padding=1)
        target_gy = F.conv2d(target_flat, self.sobel_y, padding=1)
        
        # L1损失
        loss_gx = F.l1_loss(pred_gx, target_gx)
        loss_gy = F.l1_loss(pred_gy, target_gy)
        
        return loss_gx + loss_gy
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        deep_preds: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: 主输出 [B, C, H, W]
            target: 目标 [B, C, H, W]
            deep_preds: 深度监督输出列表 (可选)
        
        Returns:
            total_loss, loss_dict
        """
        # 基础损失
        base = self.base_loss(pred, target)
        
        # 梯度损失
        grad = self.gradient_loss(pred, target)
        
        # 深度监督损失
        deep = torch.tensor(0.0, device=pred.device)
        if deep_preds:
            for dp in deep_preds:
                deep = deep + self.base_loss(dp, target)
            deep = deep / len(deep_preds)
        
        # 总损失
        total = base + self.grad_weight * grad + self.deep_weight * deep
        
        return total, {
            'total': total.item(),
            'base': base.item(),
            'grad': grad.item(),
            'deep': deep.item()
        }


# =============================================================================
# Part 4: 训练器
# =============================================================================

class Trainer:
    """训练器类"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object],
        criterion: nn.Module,
        device: torch.device,
        args: argparse.Namespace,
        writer: Optional[object] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.args = args
        self.writer = writer
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if args.amp else None
        
        # 评估指标
        self.metrics = AssimilationMetrics()
        
        # 最佳验证损失
        self.best_val_loss = float('inf')
        
        # 输出目录 (仅主进程创建)
        self.exp_dir = Path(args.output_dir) / args.exp_name
        if is_main_process():
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存配置
            with open(self.exp_dir / 'config.json', 'w') as f:
                # 过滤掉不可序列化的属性
                config_dict = {k: v for k, v in vars(args).items() 
                              if not k.startswith('train_sampler')}
                json.dump(config_dict, f, indent=2)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        loss_components = {'base': 0, 'grad': 0, 'deep': 0}
        n_batches = len(self.train_loader)
        
        for i, batch in enumerate(self.train_loader):
            # 数据移动到设备
            obs = batch['obs'].to(self.device)
            bkg = batch['bkg'].to(self.device)
            mask = batch['mask'].to(self.device)
            target = batch['target'].to(self.device)
            aux = batch.get('aux')
            if aux is not None:
                aux = aux.to(self.device)
            
            # 前向传播 (混合精度)
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    output = self.model(obs, bkg, mask, aux)
                    if isinstance(output, tuple):
                        pred, deep_preds = output
                    else:
                        pred, deep_preds = output, None
                    
                    loss, loss_dict = self.criterion(pred, target, deep_preds)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
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
            
            # 累计损失
            total_loss += loss_dict['total']
            for k in loss_components:
                loss_components[k] += loss_dict.get(k, 0)
            
            # 日志
            if (i + 1) % self.args.log_interval == 0:
                print_main(f"  [{i+1}/{n_batches}] Loss: {loss_dict['total']:.6f}")
        
        # 平均
        avg_loss = total_loss / n_batches
        for k in loss_components:
            loss_components[k] /= n_batches
        
        return {'loss': avg_loss, **loss_components}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in self.val_loader:
            obs = batch['obs'].to(self.device)
            bkg = batch['bkg'].to(self.device)
            mask = batch['mask'].to(self.device)
            target = batch['target'].to(self.device)
            aux = batch.get('aux')
            if aux is not None:
                aux = aux.to(self.device)
            
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
        
        return {
            'loss': avg_loss,
            'global_rmse': levelwise['global'].item(),
            'strat_rmse': levelwise['stratosphere'].item(),
            'trop_rmse': levelwise['troposphere'].item(),
            'grad_rmse': grad_metrics['grad_rmse'].item(),
            'grad_corr': grad_metrics['grad_correlation'].item()
        }
    
    def train(self, start_epoch: int = 0) -> None:
        """完整训练流程 (支持DDP)"""
        print_main("\n" + "=" * 70)
        print_main(f"开始训练: {self.args.exp_name}")
        print_main(f"设备: {self.device}")
        print_main(f"总轮数: {self.args.epochs}")
        if hasattr(self.args, 'ddp') and self.args.ddp:
            print_main(f"DDP模式: rank={get_rank()}, world_size={get_world_size()}")
        print_main("=" * 70)
        
        for epoch in range(start_epoch, self.args.epochs):
            # DDP: 设置sampler的epoch以确保每个epoch数据打乱方式不同
            if hasattr(self.args, 'train_sampler') and self.args.train_sampler is not None:
                self.args.train_sampler.set_epoch(epoch)
            
            print_main(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            print_main("-" * 40)
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            print_main(f"  训练 Loss: {train_metrics['loss']:.6f}")
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print_main(f"  学习率: {current_lr:.2e}")
            
            # 验证
            if (epoch + 1) % self.args.val_interval == 0:
                val_metrics = self.validate(epoch)
                print_main(f"  验证 Loss: {val_metrics['loss']:.6f}")
                print_main(f"  全局RMSE: {val_metrics['global_rmse']:.4f} K")
                print_main(f"  平流层RMSE: {val_metrics['strat_rmse']:.4f} K")
                print_main(f"  对流层RMSE: {val_metrics['trop_rmse']:.4f} K")
                
                # 保存最佳模型 (仅主进程)
                if val_metrics['loss'] < self.best_val_loss and is_main_process():
                    self.best_val_loss = val_metrics['loss']
                    save_checkpoint(
                        self.exp_dir / 'best_model.pth',
                        self.model, self.optimizer, self.scheduler,
                        epoch, self.best_val_loss, self.args
                    )
                
                # TensorBoard (仅主进程)
                if self.writer and is_main_process():
                    self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                    self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                    self.writer.add_scalar('RMSE/global', val_metrics['global_rmse'], epoch)
                    self.writer.add_scalar('RMSE/stratosphere', val_metrics['strat_rmse'], epoch)
                    self.writer.add_scalar('RMSE/troposphere', val_metrics['trop_rmse'], epoch)
            
            # 定期保存 (仅主进程)
            if (epoch + 1) % self.args.save_interval == 0 and is_main_process():
                save_checkpoint(
                    self.exp_dir / f'checkpoint_epoch{epoch+1}.pth',
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.best_val_loss, self.args
                )
        
        print_main("\n" + "=" * 70)
        print_main("训练完成!")
        print_main(f"最佳验证损失: {self.best_val_loss:.6f}")
        print_main(f"模型保存至: {self.exp_dir}")
        print_main("=" * 70)


# =============================================================================
# Part 5: 数据加载函数
# =============================================================================

def load_presplit_datasets(
    data_root: Path,
    use_aux: bool,
    stats_file: Optional[str] = None
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    加载预划分的数据集
    
    Args:
        data_root: 数据根目录 (包含 train/, val/, test/ 子目录)
        use_aux: 是否使用辅助特征
        stats_file: 统计量文件路径
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    print("\n使用预划分的数据集...")
    
    # 检查目录结构
    train_dir = data_root / 'train'
    val_dir = data_root / 'val'
    test_dir = data_root / 'test'
    
    for d, name in [(train_dir, '训练集'), (val_dir, '验证集'), (test_dir, '测试集')]:
        if not d.exists():
            raise ValueError(f"{name}目录不存在: {d}")
    
    # 加载统计量
    obs_norm = None
    bkg_norm = None
    target_norm = None
    
    if stats_file and Path(stats_file).exists():
        print(f"  加载统计量: {stats_file}")
        stats = np.load(stats_file)
        obs_norm = LevelwiseNormalizer(stats['obs_mean'], stats['obs_std'], name='obs')
        bkg_norm = LevelwiseNormalizer(stats['bkg_mean'], stats['bkg_std'], name='bkg')
        target_norm = LevelwiseNormalizer(stats['target_mean'], stats['target_std'], name='target')
    
    # 创建数据集
    train_files = sorted([str(f) for f in train_dir.glob('*.npz')])
    val_files = sorted([str(f) for f in val_dir.glob('*.npz')])
    test_files = sorted([str(f) for f in test_dir.glob('*.npz')])
    
    if not train_files:
        raise ValueError(f"训练集为空: {train_dir}")
    
    print(f"  训练集: {len(train_files)} 文件")
    print(f"  验证集: {len(val_files)} 文件")
    print(f"  测试集: {len(test_files)} 文件")
    
    # 创建数据集实例
    train_dataset = LazySatelliteERA5Dataset(
        file_list=train_files,
        obs_normalizer=obs_norm,
        bkg_normalizer=bkg_norm,
        target_normalizer=target_norm,
        use_aux=use_aux
    )
    
    val_dataset = LazySatelliteERA5Dataset(
        file_list=val_files,
        obs_normalizer=obs_norm,
        bkg_normalizer=bkg_norm,
        target_normalizer=target_norm,
        use_aux=use_aux
    )
    
    test_dataset = LazySatelliteERA5Dataset(
        file_list=test_files,
        obs_normalizer=obs_norm,
        bkg_normalizer=bkg_norm,
        target_normalizer=target_norm,
        use_aux=use_aux
    )
    
    return train_dataset, val_dataset, test_dataset


# =============================================================================
# Part 6: 主函数
# =============================================================================

def main():
    """主函数 (增强版 - 支持多GPU)"""
    # 解析参数
    args = parse_args()
    
    # =========================================================================
    # 多GPU初始化
    # =========================================================================
    local_rank = 0
    world_size = 1
    
    if args.ddp:
        # DDP模式: 使用torchrun启动
        local_rank = setup_ddp()
        world_size = get_world_size()
        device = torch.device(f'cuda:{local_rank}')
        print_main(f"[DDP] 初始化完成: rank={get_rank()}, world_size={world_size}")
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        # DataParallel模式: 使用所有可见GPU
        device = torch.device('cuda:0')
        print(f"[DataParallel] 检测到 {torch.cuda.device_count()} 块GPU")
    else:
        # 单GPU或CPU
        device = get_device(args.device)
    
    print_main(f"使用设备: {device}")
    if args.ddp:
        print_main(f"  - DDP模式, 进程数: {world_size}")
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"  - DataParallel模式, GPU数: {torch.cuda.device_count()}")
    
    # 设置随机种子 (考虑rank以确保不同进程数据不同)
    set_seed(args.seed + get_rank())
    
    # =========================================================================
    # 数据准备
    # =========================================================================
    print_main("\n准备数据...")
    
    # 检查数据目录
    data_root = Path(args.data_root)
    if not data_root.exists():
        print_main(f"警告: 数据目录不存在: {data_root}")
        print_main("使用合成数据进行演示...")
        
        # 创建合成数据
        from data_pipeline_v2 import create_synthetic_data_v2
        obs, bkg, target, aux = create_synthetic_data_v2(n_samples=100)
        
        dataset = InMemorySatelliteDataset(
            obs_data=obs,
            bkg_data=bkg,
            target_data=target,
            aux_data=aux if args.use_aux else None,
            compute_stats=True
        )
        
        # 使用原始方式划分
        n_total = len(dataset)
        n_train = int(n_total * args.train_ratio)
        n_val = int(n_total * args.val_ratio)
        n_test = n_total - n_train - n_val
        
        train_set, val_set, test_set = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(args.seed)
        )
    else:
        # 使用真实数据
        if args.use_split:
            # 方式1: 使用预划分的数据集
            train_set, val_set, test_set = load_presplit_datasets(
                data_root, args.use_aux, args.stats_file
            )
        else:
            # 方式2: 使用原始的 random_split 方式
            print_main("\n使用 random_split 方式划分数据集...")
            file_list = sorted(data_root.glob('**/*.npz'))
            
            # 排除 train/val/test 子目录中的文件
            file_list = [f for f in file_list 
                        if not any(p in f.parts for p in ['train', 'val', 'test'])]
            
            if not file_list:
                raise ValueError(f"数据目录中未找到.npz文件: {data_root}")
            
            dataset = LazySatelliteERA5Dataset(
                file_list=[str(f) for f in file_list],
                use_aux=args.use_aux
            )
            
            # 计算或加载统计量
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
                dataset.compute_statistics(n_samples=min(1000, len(dataset)))
            
            # 划分数据集
            n_total = len(dataset)
            n_train = int(n_total * args.train_ratio)
            n_val = int(n_total * args.val_ratio)
            n_test = n_total - n_train - n_val
            
            train_set, val_set, test_set = random_split(
                dataset, [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(args.seed)
            )
    
    print_main(f"  训练集: {len(train_set)}")
    print_main(f"  验证集: {len(val_set)}")
    print_main(f"  测试集: {len(test_set)}")
    
    # =========================================================================
    # 创建DataLoader (DDP需要DistributedSampler)
    # =========================================================================
    train_sampler = None
    val_sampler = None
    shuffle_train = True
    
    if args.ddp:
        # DDP模式使用分布式采样器
        train_sampler = DistributedSampler(
            train_set, 
            num_replicas=world_size, 
            rank=get_rank(),
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_set,
            num_replicas=world_size,
            rank=get_rank(),
            shuffle=False
        )
        shuffle_train = False  # 使用sampler时不能shuffle
        print_main(f"  使用DistributedSampler")
    
    # 调整batch_size: DDP模式下每个进程的batch_size
    effective_batch_size = args.batch_size
    if args.ddp:
        print_main(f"  每进程batch_size: {args.batch_size}, 有效batch_size: {args.batch_size * world_size}")
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"  总batch_size: {args.batch_size} (分布到 {torch.cuda.device_count()} 块GPU)")
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=args.ddp  # DDP模式下丢弃最后不完整的batch
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
    print_main("\n准备模型...")
    
    # 动态导入模型
    try:
        from models.backbone import create_model, UNetConfig
    except ImportError:
        # 如果backbone.py不在models目录，尝试直接导入
        sys.path.insert(0, str(Path(__file__).parent / 'models'))
        from backbone import create_model, UNetConfig
    
    # 模型配置
    if args.model != 'vanilla_unet':
        config = UNetConfig(
            fusion_mode=args.fusion_mode,
            use_aux=args.use_aux,
            mask_aware=args.mask_aware,
            deep_supervision=args.deep_supervision
        )
        model = create_model(args.model, config=config)
    else:
        model = create_model(args.model)
    
    model = model.to(device)
    print_main(f"  参数量: {count_parameters(model):,}")
    
    # =========================================================================
    # 多GPU包装
    # =========================================================================
    if args.ddp:
        # DDP包装
        model = DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
        print_main(f"  模型已包装为DDP (rank={local_rank})")
    elif args.multi_gpu and torch.cuda.device_count() > 1:
        # DataParallel包装
        model = DataParallel(model)
        print(f"  模型已包装为DataParallel (GPU: {list(range(torch.cuda.device_count()))})")
    
    # =========================================================================
    # 优化器和调度器
    # =========================================================================
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
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
        # 包装普通损失
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
    if args.tensorboard and is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(args.output_dir) / args.exp_name / 'logs'
            writer = SummaryWriter(log_dir)
            print_main(f"  TensorBoard日志: {log_dir}")
        except ImportError:
            print_main("  警告: TensorBoard不可用")
    
    # =========================================================================
    # 恢复训练
    # =========================================================================
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        start_epoch += 1
    
    # =========================================================================
    # 训练
    # =========================================================================
    # 传递额外参数给Trainer
    args.train_sampler = train_sampler  # DDP需要在每个epoch设置sampler的epoch
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        args=args,
        writer=writer
    )
    
    try:
        trainer.train(start_epoch)
    finally:
        # 清理
        if writer:
            writer.close()
        if args.ddp:
            cleanup_ddp()


if __name__ == '__main__':
    main()