#!/usr/bin/env python3
"""
===============================================================================
温度廓线反演训练脚本 (Training Script for Temperature Profile Retrieval)
===============================================================================

支持配准后的数据格式:
- collocation_YYYYMMDD_HHMM_X.npy (亮温, N×17)
- collocation_YYYYMMDD_HHMM_Y.npy (温度廓线, N×37)

用法:
    python train_profile.py --data_root /data2/lrx/era_obs --epochs 100

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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

# 导入数据加载器和模型
from collocated_data_loader import (
    CollocatedDataset,
    PointwiseNormalizer,
    ProfileMetrics,
    create_mlp_model,
    scan_data_directory
)


# =============================================================================
# Part 1: 参数解析
# =============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='温度廓线反演训练脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # === 实验配置 ===
    exp_group = parser.add_argument_group('实验配置')
    exp_group.add_argument('--exp_name', type=str, default='profile_retrieval',
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
                            help='配准数据根目录 (如: /data2/lrx/era_obs)')
    data_group.add_argument('--stats_file', type=str, default=None,
                            help='预计算统计量文件路径')
    data_group.add_argument('--year', type=str, default=None,
                            help='年份筛选 (如: 2024)')
    data_group.add_argument('--months', nargs='+', default=None,
                            help='月份筛选 (如: 01 02 03)')
    data_group.add_argument('--train_ratio', type=float, default=0.8,
                            help='训练集比例')
    data_group.add_argument('--val_ratio', type=float, default=0.1,
                            help='验证集比例')
    data_group.add_argument('--num_workers', type=int, default=4,
                            help='DataLoader工作进程数')
    data_group.add_argument('--max_samples', type=int, default=None,
                            help='最大样本数 (用于调试)')
    
    # === 模型配置 ===
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument('--model', type=str, default='physics_mlp',
                             choices=['simple_mlp', 'res_mlp', 'physics_mlp'],
                             help='模型类型')
    model_group.add_argument('--hidden_dim', type=int, default=256,
                             help='隐藏层维度')
    model_group.add_argument('--n_layers', type=int, default=6,
                             help='网络层数')
    model_group.add_argument('--dropout', type=float, default=0.1,
                             help='Dropout比例')
    
    # === 训练配置 ===
    train_group = parser.add_argument_group('训练配置')
    train_group.add_argument('--epochs', type=int, default=100,
                             help='训练轮数')
    train_group.add_argument('--batch_size', type=int, default=256,
                             help='批大小')
    train_group.add_argument('--lr', type=float, default=1e-3,
                             help='初始学习率')
    train_group.add_argument('--weight_decay', type=float, default=1e-5,
                             help='权重衰减')
    train_group.add_argument('--scheduler', type=str, default='cosine',
                             choices=['cosine', 'plateau', 'none'],
                             help='学习率调度器')
    train_group.add_argument('--patience', type=int, default=10,
                             help='早停耐心值')
    train_group.add_argument('--grad_clip', type=float, default=1.0,
                             help='梯度裁剪阈值')
    
    # === 损失函数配置 ===
    loss_group = parser.add_argument_group('损失函数配置')
    loss_group.add_argument('--loss', type=str, default='mse',
                            choices=['mse', 'mae', 'huber', 'weighted_mse'],
                            help='损失函数类型')
    
    # === 设备配置 ===
    device_group = parser.add_argument_group('设备配置')
    device_group.add_argument('--device', type=str, default='auto',
                              help='设备 (auto/cpu/cuda/cuda:0)')
    device_group.add_argument('--amp', action='store_true',
                              help='使用混合精度训练')
    
    # === 日志配置 ===
    log_group = parser.add_argument_group('日志配置')
    log_group.add_argument('--log_interval', type=int, default=100,
                           help='日志打印间隔 (步)')
    log_group.add_argument('--val_interval', type=int, default=1,
                           help='验证间隔 (轮)')
    log_group.add_argument('--save_interval', type=int, default=10,
                           help='保存间隔 (轮)')
    log_group.add_argument('--tensorboard', action='store_true',
                           help='使用TensorBoard')
    
    args = parser.parse_args()
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
    args: argparse.Namespace,
    normalizers: Dict = None
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
    
    if normalizers:
        checkpoint['x_mean'] = normalizers['x'].mean
        checkpoint['x_std'] = normalizers['x'].std
        checkpoint['y_mean'] = normalizers['y'].mean
        checkpoint['y_std'] = normalizers['y'].std
    
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

class WeightedMSELoss(nn.Module):
    """
    加权MSE损失
    
    对不同气压层使用不同权重，增加平流层权重
    """
    
    def __init__(self, n_levels: int = 37, stratosphere_weight: float = 2.0):
        super().__init__()
        
        # 默认ERA5气压层
        pressure_levels = np.array([
            1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
            750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
            250, 225, 200, 175, 150, 125, 100, 70, 50, 30,
            20, 10, 7, 5, 3, 2, 1
        ])[:n_levels]
        
        # 平流层（<=100hPa）权重更高
        weights = np.ones(n_levels)
        weights[pressure_levels <= 100] = stratosphere_weight
        
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 37)
            target: (B, 37)
        """
        mse = (pred - target) ** 2  # (B, 37)
        weighted_mse = mse * self.weights.unsqueeze(0)  # (B, 37)
        return weighted_mse.mean()


def get_loss_fn(loss_name: str, n_levels: int = 37) -> nn.Module:
    """获取损失函数"""
    if loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'mae':
        return nn.L1Loss()
    elif loss_name == 'huber':
        return nn.HuberLoss()
    elif loss_name == 'weighted_mse':
        return WeightedMSELoss(n_levels)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# =============================================================================
# Part 4: 训练器
# =============================================================================

class ProfileTrainer:
    """温度廓线反演训练器"""
    
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
        normalizers: Dict,
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
        self.normalizers = normalizers
        self.writer = writer
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if args.amp else None
        
        # 评估指标
        self.metrics = ProfileMetrics()
        
        # 最佳验证损失
        self.best_val_loss = float('inf')
        
        # 早停计数器
        self.patience_counter = 0
        
        # 输出目录
        self.exp_dir = Path(args.output_dir) / args.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # 保存标准化器
        self._save_normalizers()
    
    def _save_normalizers(self) -> None:
        """保存标准化器"""
        np.savez(
            self.exp_dir / 'normalizers.npz',
            x_mean=self.normalizers['x'].mean,
            x_std=self.normalizers['x'].std,
            y_mean=self.normalizers['y'].mean,
            y_std=self.normalizers['y'].std
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        n_batches = len(self.train_loader)
        
        for i, batch in enumerate(self.train_loader):
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                
                self.scaler.scale(loss).backward()
                
                if self.args.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)
                
                loss.backward()
                
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # 日志
            if (i + 1) % self.args.log_interval == 0:
                avg_loss = total_loss / (i + 1)
                print(f"  [{i+1}/{n_batches}] Loss: {loss.item():.6f} (Avg: {avg_loss:.6f})")
        
        return {'loss': total_loss / n_batches}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in self.val_loader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            
            pred = self.model(x)
            loss = self.criterion(pred, y)
            
            total_loss += loss.item()
            
            # 逆标准化
            pred_raw = self.normalizers['y'].inverse_transform(pred.cpu())
            target_raw = batch['y_raw']
            
            all_preds.append(pred_raw)
            all_targets.append(target_raw)
        
        # 合并
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        levelwise = self.metrics.levelwise_rmse(all_preds, all_targets)
        
        return {
            'loss': avg_loss,
            'global_rmse': levelwise['global'].item(),
            'strat_rmse': levelwise['stratosphere'].item(),
            'trop_rmse': levelwise['troposphere'].item(),
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
            
            # 验证
            if (epoch + 1) % self.args.val_interval == 0:
                val_metrics = self.validate(epoch)
                print(f"  验证 Loss: {val_metrics['loss']:.6f}")
                print(f"  全局RMSE: {val_metrics['global_rmse']:.4f} K")
                print(f"  平流层RMSE: {val_metrics['strat_rmse']:.4f} K")
                print(f"  对流层RMSE: {val_metrics['trop_rmse']:.4f} K")
                
                # 学习率调度
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"  学习率: {current_lr:.2e}")
                
                # 保存最佳模型
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    save_checkpoint(
                        self.exp_dir / 'best_model.pth',
                        self.model, self.optimizer, self.scheduler,
                        epoch, self.best_val_loss, self.args, self.normalizers
                    )
                else:
                    self.patience_counter += 1
                
                # 早停
                if self.patience_counter >= self.args.patience:
                    print(f"\n早停: {self.args.patience} 轮未改善")
                    break
                
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
                    epoch, self.best_val_loss, self.args, self.normalizers
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
    
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        print(f"错误: 数据目录不存在: {data_root}")
        sys.exit(1)
    
    # 扫描数据目录
    scan_result = scan_data_directory(args.data_root)
    
    if scan_result.get('total_files', 0) == 0:
        print(f"错误: 数据目录中未找到配准文件")
        print(f"请确保目录结构为: {data_root}/YYYY/MM/collocation_*.npy")
        sys.exit(1)
    
    # 加载数据集
    try:
        dataset = CollocatedDataset(
            data_root=args.data_root,
            compute_stats=True,
            year_filter=args.year,
            month_filter=args.months,
            max_samples_per_file=args.max_samples,
            verbose=True
        )
    except Exception as e:
        print(f"错误: 加载数据失败 - {e}")
        sys.exit(1)
    
    # 划分数据集
    n_total = len(dataset)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val
    
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(train_set):,}")
    print(f"  验证集: {len(val_set):,}")
    print(f"  测试集: {len(test_set):,}")
    
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
    
    model = create_mlp_model(
        model_name=args.model,
        in_channels=17,
        out_channels=37,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    
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
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        scheduler = None
    
    # =========================================================================
    # 损失函数
    # =========================================================================
    criterion = get_loss_fn(args.loss, n_levels=37)
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
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1
    
    # =========================================================================
    # 训练
    # =========================================================================
    normalizers = dataset.get_normalizers()
    
    trainer = ProfileTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        args=args,
        normalizers=normalizers,
        writer=writer
    )
    
    trainer.train(start_epoch)
    
    # 关闭TensorBoard
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
