"""
改进版 PAVMT-Unet 训练脚本
==========================
完整的训练流程,包含:
- 数据加载
- 模型初始化
- 训练循环
- 验证评估
- 检查点保存
- TensorBoard日志

作者: 基于PAVMT-Unet改进
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.improved_pavmt_unet import ImprovedPAVMTUnet
from modules.atmospheric_loss import AtmosphericPhysicLoss, AtmosphericPhysicLossV2
from data.dataset import SyntheticDataset, SatelliteAssimilationDataset


# ============================================
# 评估指标
# ============================================

def compute_rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """计算RMSE"""
    diff = (pred - target) ** 2
    if mask is not None:
        diff = diff * mask
        rmse = torch.sqrt(diff.sum() / (mask.sum() * pred.shape[1] + 1e-8))
    else:
        rmse = torch.sqrt(diff.mean())
    return rmse.item()


def compute_corr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算相关系数"""
    # 展平所有维度
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # 去均值
    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()
    
    # 计算相关系数
    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
    
    corr = numerator / (denominator + 1e-8)
    return corr.item()


def compute_bias(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算偏差"""
    return (pred - target).mean().item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """计算所有评估指标"""
    return {
        'rmse': compute_rmse(pred, target, mask),
        'corr': compute_corr(pred, target),
        'bias': compute_bias(pred, target)
    }


# ============================================
# 训练器类
# ============================================

class Trainer:
    """
    模型训练器
    
    管理训练循环、验证、日志记录和检查点保存
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: argparse.Namespace
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 日志
        self.writer = SummaryWriter(config.log_dir)
        
        # 最佳指标
        self.best_val_rmse = float('inf')
        self.best_epoch = 0
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        
        # 梯度裁剪
        self.grad_clip = config.grad_clip
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        loss_components = {'l1': 0, 'gradient': 0, 'physics': 0}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            satellite = batch['satellite'].to(self.device)
            background = batch['background'].to(self.device)
            label = batch['label'].to(self.device)
            mask = batch['mask'].to(self.device) if self.config.use_mask else None
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(satellite, background, mask)
                    loss, loss_dict = self.criterion(output, label, mask, return_components=True)
                
                # 反向传播 (混合精度)
                self.scaler.scale(loss).backward()
                
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(satellite, background, mask)
                loss, loss_dict = self.criterion(output, label, mask, return_components=True)
                
                # 反向传播
                loss.backward()
                
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
            
            # 累积损失
            total_loss += loss.item()
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key]
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # 学习率调度
        self.scheduler.step()
        
        # 平均损失
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return {'total': avg_loss, **avg_components}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0
        all_metrics = {'rmse': 0, 'corr': 0, 'bias': 0}
        num_batches = len(self.val_loader)
        
        for batch in self.val_loader:
            satellite = batch['satellite'].to(self.device)
            background = batch['background'].to(self.device)
            label = batch['label'].to(self.device)
            mask = batch['mask'].to(self.device) if self.config.use_mask else None
            
            output = self.model(satellite, background, mask)
            loss = self.criterion(output, label, mask)
            
            total_loss += loss.item()
            
            # 计算指标
            metrics = compute_metrics(output, label, mask)
            for key in all_metrics:
                all_metrics[key] += metrics[key]
        
        # 平均
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_rmse': self.best_val_rmse,
            'config': self.config
        }
        
        # 保存最新检查点
        latest_path = Path(self.config.checkpoint_dir) / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best.pth'
            torch.save(checkpoint, best_path)
        
        # 定期保存
        if epoch % self.config.save_every == 0:
            epoch_path = Path(self.config.checkpoint_dir) / f'epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """完整训练流程"""
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        
        for epoch in range(1, self.config.epochs + 1):
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch)
            
            # 记录日志
            self.writer.add_scalar('Train/Loss', train_losses['total'], epoch)
            self.writer.add_scalar('Train/L1', train_losses['l1'], epoch)
            self.writer.add_scalar('Train/Gradient', train_losses['gradient'], epoch)
            self.writer.add_scalar('Train/Physics', train_losses['physics'], epoch)
            
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/RMSE', val_metrics['rmse'], epoch)
            self.writer.add_scalar('Val/CORR', val_metrics['corr'], epoch)
            self.writer.add_scalar('Val/Bias', val_metrics['bias'], epoch)
            
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print(f"  Train Loss: {train_losses['total']:.4f} "
                  f"(L1: {train_losses['l1']:.4f}, Grad: {train_losses['gradient']:.4f}, "
                  f"Phys: {train_losses['physics']:.4f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                  f"RMSE: {val_metrics['rmse']:.4f}, "
                  f"CORR: {val_metrics['corr']:.4f}, "
                  f"Bias: {val_metrics['bias']:.4f}")
            
            # 保存检查点
            is_best = val_metrics['rmse'] < self.best_val_rmse
            if is_best:
                self.best_val_rmse = val_metrics['rmse']
                self.best_epoch = epoch
                print(f"  >>> 新的最佳RMSE: {self.best_val_rmse:.4f}")
            
            self.save_checkpoint(epoch, is_best)
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"最佳验证RMSE: {self.best_val_rmse:.4f} (Epoch {self.best_epoch})")
        print("=" * 60)
        
        self.writer.close()


# ============================================
# 主函数
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description='训练改进版PAVMT-Unet')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--n_sat_channels', type=int, default=13, help='卫星通道数')
    parser.add_argument('--n_levels', type=int, default=37, help='垂直层数')
    parser.add_argument('--height', type=int, default=64, help='图像高度')
    parser.add_argument('--width', type=int, default=64, help='图像宽度')
    parser.add_argument('--use_synthetic', action='store_true', help='使用合成数据')
    parser.add_argument('--train_samples', type=int, default=1000, help='训练样本数')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集比例')
    
    # 模型参数
    parser.add_argument('--base_channels', type=int, default=64, help='基础通道数')
    parser.add_argument('--num_stages', type=int, default=4, help='编解码器阶段数')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--use_mask', action='store_true', default=True, help='使用掩膜')
    
    # 损失函数参数
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='L1损失权重')
    parser.add_argument('--lambda_gradient', type=float, default=0.5, help='梯度损失权重')
    parser.add_argument('--lambda_physics', type=float, default=0.1, help='物理约束权重')
    parser.add_argument('--use_loss_v2', action='store_true', help='使用V2损失函数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='初始学习率')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='最小学习率')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='预热轮数')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='检查点目录')
    parser.add_argument('--save_every', type=int, default=10, help='保存间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.log_dir = Path(args.log_dir) / timestamp
    args.checkpoint_dir = Path(args.checkpoint_dir) / timestamp
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # ============================================
    # 数据集
    # ============================================
    print("\n加载数据集...")
    
    if args.use_synthetic:
        # 使用合成数据 (用于测试)
        full_dataset = SyntheticDataset(
            num_samples=args.train_samples,
            n_sat_channels=args.n_sat_channels,
            n_levels=args.n_levels,
            height=args.height,
            width=args.width
        )
    else:
        # 使用真实数据
        full_dataset = SatelliteAssimilationDataset(
            satellite_dir=Path(args.data_dir) / 'satellite',
            background_dir=Path(args.data_dir) / 'background',
            label_dir=Path(args.data_dir) / 'label',
            n_sat_channels=args.n_sat_channels,
            n_levels=args.n_levels,
            height=args.height,
            width=args.width
        )
    
    # 划分训练集和验证集
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # ============================================
    # 模型
    # ============================================
    print("\n初始化模型...")
    
    model = ImprovedPAVMTUnet(
        n_sat_channels=args.n_sat_channels,
        n_background_layers=args.n_levels,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        num_heads=args.num_heads,
        use_mask=args.use_mask
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # ============================================
    # 损失函数
    # ============================================
    if args.use_loss_v2:
        criterion = AtmosphericPhysicLossV2(
            lambda_l1=args.lambda_l1,
            lambda_gradient=args.lambda_gradient,
            lambda_physics=args.lambda_physics
        )
    else:
        criterion = AtmosphericPhysicLoss(
            lambda_l1=args.lambda_l1,
            lambda_gradient=args.lambda_gradient,
            lambda_physics=args.lambda_physics
        )
    
    # ============================================
    # 优化器
    # ============================================
    # 参数分组: 卷积/线性层使用权重衰减, 其他不使用
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and ('conv' in name.lower() or 'linear' in name.lower() or 'proj' in name.lower()):
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=args.lr)
    
    # ============================================
    # 学习率调度器
    # ============================================
    # Warmup + Cosine Annealing
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=args.warmup_epochs
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.min_lr
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )
    
    # ============================================
    # 恢复训练
    # ============================================
    start_epoch = 0
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"从Epoch {start_epoch} 继续训练")
    
    # ============================================
    # 训练
    # ============================================
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=args
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
