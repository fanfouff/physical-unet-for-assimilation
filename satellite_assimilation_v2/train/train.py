#!/usr/bin/env python3
"""
===============================================================================
卫星数据同化训练脚本 (Training Script)
===============================================================================

用法:
    python train.py --exp_name test_run --data_root /path/to/data --epochs 100

功能:
    1. 支持命令行参数配置
    2. 自动实验ID管理
    3. 断点续训
    4. TensorBoard日志
    5. 分布式训练支持 (DDP)
    6. 自动消融实验配置

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
from torch.utils.data import DataLoader, random_split
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
# Part 1: 参数解析
# =============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='卫星数据同化训练脚本',
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
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
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
    scheduler: Optional[object] = None
) -> Tuple[int, float]:
    """加载检查点"""
    checkpoint = torch.load(path, map_location='cpu')
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
        
        # 输出目录
        self.exp_dir = Path(args.output_dir) / args.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
    
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
                print(f"  [{i+1}/{n_batches}] Loss: {loss_dict['total']:.6f}")
        
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
        """完整训练流程"""
        print("\n" + "=" * 70)
        print(f"开始训练: {self.args.exp_name}")
        print(f"设备: {self.device}")
        print(f"总轮数: {self.args.epochs}")
        print("=" * 70)
        
        for epoch in range(start_epoch, self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            print("-" * 40)
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            print(f"  训练 Loss: {train_metrics['loss']:.6f}")
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"  学习率: {current_lr:.2e}")
            
            # 验证
            if (epoch + 1) % self.args.val_interval == 0:
                val_metrics = self.validate(epoch)
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
            if (epoch + 1) % self.args.save_interval == 0:
                save_checkpoint(
                    self.exp_dir / f'checkpoint_epoch{epoch+1}.pth',
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.best_val_loss, self.args
                )
        
        print("\n" + "=" * 70)
        print("训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print(f"模型保存至: {self.exp_dir}")
        print("=" * 70)


# =============================================================================
# Part 5: 主函数
# =============================================================================

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备
    device = get_device(args.device)
    print(f"使用设备: {device}")
    
    # =========================================================================
    # 数据准备
    # =========================================================================
    print("\n准备数据...")
    
    # 检查数据目录
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"警告: 数据目录不存在: {data_root}")
        print("使用合成数据进行演示...")
        
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
    else:
        # 使用真实数据
        file_list = sorted(data_root.glob('**/*.npz'))
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
    
    print(f"  训练集: {len(train_set)}")
    print(f"  验证集: {len(val_set)}")
    print(f"  测试集: {len(test_set)}")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # =========================================================================
    # 模型准备
    # =========================================================================
    print("\n准备模型...")
    
    # 动态导入模型
    # train.py 开头的代码
    import sys
    import os

    # 获取 train.py 所在目录的上级目录（也就是项目根目录）
    # 项目根目录：/home/seu/Fuxi/Unet/satellite_assimilation_v2
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 先拿到 train 文件夹路径
    ROOT_DIR = os.path.dirname(ROOT_DIR)  # 再往上一级到项目根目录

    # 将项目根目录添加到 Python 搜索路径中
    sys.path.append(ROOT_DIR)

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
    print(f"  参数量: {count_parameters(model):,}")
    
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
    # TensorBoard
    # =========================================================================
    writer = None
    if args.tensorboard:
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
            args.resume, model, optimizer, scheduler
        )
        start_epoch += 1
    
    # =========================================================================
    # 训练
    # =========================================================================
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
    
    trainer.train(start_epoch)
    
    # 关闭TensorBoard
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
