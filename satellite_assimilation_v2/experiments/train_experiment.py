#!/usr/bin/env python3
"""
===============================================================================
PAS-Net 实验训练脚本
Experiment Training Script for PAS-Net
===============================================================================

支持:
1. 消融实验的各种配置变体
2. 对比方法训练
3. 分布式训练 (DDP)
4. 混合精度训练
5. 详细的评估指标

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

# 添加模型路径
sys.path.insert(0, str(Path(__file__).parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent))

from models.backbone import (
    PASNet, VanillaUNet, ResUNet, AttentionUNet,
    UNetConfig, create_model
)
from models.losses import (
    HybridPhysicsLoss, CombinedLoss, AblationLoss,
    MaskedMSELoss, MaskedMAELoss, SobelGradientLoss
)


# =============================================================================
# Part 1: 分布式工具函数
# =============================================================================

def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        print("[DDP] 单卡模式")
        return 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30)
    )
    dist.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def print_rank0(msg: str, rank: int):
    if is_main_process(rank):
        print(msg)


# =============================================================================
# Part 2: 数据加载 (复用原有pipeline)
# =============================================================================

def create_datasets(args, rank, world_size):
    """创建数据集"""
    # 尝试导入原有数据pipeline
    try:
        from data_pipeline_v2 import (
            InMemorySatelliteDataset,
            LazySatelliteERA5Dataset,
            LevelwiseNormalizer,
            create_synthetic_data_v2
        )
    except ImportError:
        # 如果没有原始文件，使用简化版
        print_rank0("[WARNING] 未找到 data_pipeline_v2.py，使用合成数据测试", rank)
        return create_synthetic_datasets(args, rank, world_size)
    
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        print_rank0(f"[WARNING] 数据目录不存在: {data_root}，使用合成数据", rank)
        return create_synthetic_datasets(args, rank, world_size)
    
    # 加载真实数据
    file_list = sorted(data_root.glob('**/*.npz'))
    if not file_list:
        print_rank0("[WARNING] 未找到.npz文件，使用合成数据", rank)
        return create_synthetic_datasets(args, rank, world_size)
    
    # 创建数据集
    dataset = LazySatelliteERA5Dataset(
        file_list=[str(f) for f in file_list],
        use_aux=args.use_aux
    )
    
    # 加载统计量
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
        if is_main_process(rank):
            dataset.compute_statistics(n_samples=min(1000, len(dataset)))
        if world_size > 1:
            dist.barrier()
    
    # 划分数据集
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val
    
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    return train_set, val_set, test_set, dataset


def create_synthetic_datasets(args, rank, world_size):
    """创建合成数据集 (用于测试)"""
    from data_pipeline_v2 import InMemorySatelliteDataset, create_synthetic_data_v2
    
    print_rank0("[INFO] 使用合成数据进行测试...", rank)
    
    obs, bkg, target, aux = create_synthetic_data_v2(n_samples=500)
    
    dataset = InMemorySatelliteDataset(
        obs_data=obs, bkg_data=bkg, target_data=target, aux_data=aux,
        compute_stats=True
    )
    
    n_train = 400
    n_val = 50
    n_test = 50
    
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    return train_set, val_set, test_set, dataset


# =============================================================================
# Part 3: 评估指标
# =============================================================================

class ExperimentMetrics:
    """实验评估指标"""
    
    def __init__(
        self,
        n_levels: int = 37,
        stratosphere_threshold: float = 100.0
    ):
        # ERA5气压层
        self.pressure_levels = np.array([
            1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
            750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
            250, 225, 200, 175, 150, 125, 100, 70, 50, 30,
            20, 10, 7, 5, 3, 2, 1
        ])[:n_levels]
        
        self.strat_mask = self.pressure_levels <= stratosphere_threshold
        self.trop_mask = ~self.strat_mask
    
    @staticmethod
    def rmse(pred, target, dim=None):
        return torch.sqrt(((pred - target) ** 2).mean(dim=dim))
    
    @staticmethod
    def mae(pred, target, dim=None):
        return torch.abs(pred - target).mean(dim=dim)
    
    def compute_all(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, float]:
        """计算所有指标"""
        B, C, H, W = pred.shape
        
        # 全局指标
        global_rmse = self.rmse(pred, target).item()
        global_mae = self.mae(pred, target).item()
        
        # 分层指标
        levelwise_rmse = self.rmse(pred, target, dim=(0, 2, 3))  # [C]
        
        # 对流层/平流层
        strat_rmse = self.rmse(
            pred[:, self.strat_mask], 
            target[:, self.strat_mask]
        ).item()
        trop_rmse = self.rmse(
            pred[:, self.trop_mask], 
            target[:, self.trop_mask]
        ).item()
        
        return {
            'global_rmse': global_rmse,
            'global_mae': global_mae,
            'stratosphere_rmse': strat_rmse,
            'troposphere_rmse': trop_rmse,
            'levelwise_rmse': levelwise_rmse.cpu().numpy()
        }


# =============================================================================
# Part 4: 训练器
# =============================================================================

class ExperimentTrainer:
    """实验训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        device: torch.device,
        args,
        rank: int,
        world_size: int,
        train_sampler=None,
        writer=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.train_sampler = train_sampler
        self.writer = writer
        
        self.scaler = torch.cuda.amp.GradScaler() if args.amp else None
        self.metrics = ExperimentMetrics()
        self.best_val_loss = float('inf')
        
        self.exp_dir = Path(args.output_dir) / args.exp_name
        if is_main_process(rank):
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            with open(self.exp_dir / 'config.json', 'w') as f:
                json.dump(vars(args), f, indent=2)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        if self.train_sampler:
            self.train_sampler.set_epoch(epoch)
        
        total_loss = 0
        loss_components = {}
        
        for i, batch in enumerate(self.train_loader):
            obs = batch['obs'].to(self.device, non_blocking=True)
            bkg = batch['bkg'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True)
            target = batch['target'].to(self.device, non_blocking=True)
            aux = batch.get('aux')
            if aux is not None:
                aux = aux.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
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
            
            total_loss += loss.item()
            
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v
            
            # 打印进度
            if i % self.args.log_interval == 0 and is_main_process(self.rank):
                print(f"  Epoch {epoch} [{i}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(self.train_loader)
        for k in loss_components:
            loss_components[k] /= len(self.train_loader)
        
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
                pred = output[0]
            else:
                pred = output
            
            loss, _ = self.criterion(pred, target)
            total_loss += loss.item()
            
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算详细指标
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        metrics = self.metrics.compute_all(preds, targets)
        
        return {'val_loss': avg_loss, **metrics}
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """保存检查点"""
        if not is_main_process(self.rank):
            return
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'args': vars(self.args)
        }
        
        # 保存最新
        torch.save(checkpoint, self.exp_dir / 'checkpoint_latest.pth')
        
        # 保存最佳
        if is_best:
            torch.save(checkpoint, self.exp_dir / 'checkpoint_best.pth')
            print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.6f})")
    
    def train(self, start_epoch: int = 0):
        """完整训练流程"""
        print_rank0(f"\n{'='*70}", self.rank)
        print_rank0(f"开始训练: {self.args.exp_name}", self.rank)
        print_rank0(f"{'='*70}\n", self.rank)
        
        for epoch in range(start_epoch, self.args.epochs):
            epoch_start = time.time()
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch)
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, OneCycleLR):
                    pass  # OneCycleLR在每步更新
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # 打印结果
            if is_main_process(self.rank):
                print(f"\nEpoch {epoch}/{self.args.epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_metrics['loss']:.6f}")
                print(f"  Val Loss: {val_metrics['val_loss']:.6f}")
                print(f"  Global RMSE: {val_metrics['global_rmse']:.4f} K")
                print(f"  Troposphere RMSE: {val_metrics['troposphere_rmse']:.4f} K")
                print(f"  Stratosphere RMSE: {val_metrics['stratosphere_rmse']:.4f} K")
            
            # TensorBoard
            if self.writer and is_main_process(self.rank):
                self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
                self.writer.add_scalar('RMSE/global', val_metrics['global_rmse'], epoch)
                self.writer.add_scalar('RMSE/troposphere', val_metrics['troposphere_rmse'], epoch)
                self.writer.add_scalar('RMSE/stratosphere', val_metrics['stratosphere_rmse'], epoch)
            
            # 保存检查点
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            if epoch % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics['val_loss'], is_best)
        
        # 保存最终结果
        if is_main_process(self.rank):
            results = {
                'best_val_loss': self.best_val_loss,
                'final_epoch': self.args.epochs,
                'exp_name': self.args.exp_name
            }
            with open(self.exp_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n{'='*70}")
            print(f"训练完成! 最佳验证损失: {self.best_val_loss:.6f}")
            print(f"{'='*70}\n")


# =============================================================================
# Part 5: 参数解析
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='PAS-Net实验训练脚本')
    
    # 实验配置
    parser.add_argument('--exp_name', type=str, default='experiment')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)
    
    # 数据配置
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--stats_file', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 模型配置
    parser.add_argument('--model', type=str, default='pasnet',
                        choices=['pasnet', 'pasnet_lite', 'pasnet_large',
                                'pasnet_no_se', 'pasnet_no_adapter',
                                'vanilla_unet', 'res_unet', 'attention_unet'])
    parser.add_argument('--fusion_mode', type=str, default='gated',
                        choices=['gated', 'concat', 'add', 'none'])
    parser.add_argument('--use_aux', type=str, default='true')
    parser.add_argument('--mask_aware', type=str, default='true')
    parser.add_argument('--use_spectral_adapter', type=str, default='true')
    parser.add_argument('--norm_mode', type=str, default='levelwise',
                        choices=['levelwise', 'global'])
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'onecycle', 'none'])
    
    # 损失函数配置
    parser.add_argument('--loss', type=str, default='hybrid',
                        choices=['mse', 'mae', 'hybrid', 'combined'])
    parser.add_argument('--grad_loss_weight', type=float, default=0.1)
    parser.add_argument('--profile_loss_weight', type=float, default=0.5)
    
    # 设备配置
    parser.add_argument('--amp', type=str, default='true')
    parser.add_argument('--sync_bn', type=str, default='true')
    
    # 日志配置
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--tensorboard', type=str, default='true')
    
    args = parser.parse_args()
    
    # 转换布尔值
    for key in ['use_aux', 'mask_aware', 'use_spectral_adapter', 'amp', 'sync_bn', 'tensorboard']:
        setattr(args, key, getattr(args, key).lower() == 'true')
    
    return args


# =============================================================================
# Part 6: 主函数
# =============================================================================

def main():
    args = parse_args()
    
    # 初始化分布式
    rank, world_size, local_rank = setup_distributed()
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    # 设置随机种子
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    
    print_rank0(f"\n{'='*70}", rank)
    print_rank0(f"PAS-Net 实验: {args.exp_name}", rank)
    print_rank0(f"{'='*70}", rank)
    print_rank0(f"模型: {args.model}", rank)
    print_rank0(f"融合模式: {args.fusion_mode}", rank)
    print_rank0(f"损失函数: {args.loss}", rank)
    print_rank0(f"GPU数量: {world_size}", rank)
    
    # 加载数据
    print_rank0("\n加载数据...", rank)
    train_set, val_set, test_set, dataset = create_datasets(args, rank, world_size)
    
    # 采样器
    train_sampler = DistributedSampler(train_set, world_size, rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, world_size, rank, shuffle=False) if world_size > 1 else None
    
    # DataLoader
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print_rank0(f"  训练集: {len(train_set)}", rank)
    print_rank0(f"  验证集: {len(val_set)}", rank)
    
    # 创建模型
    print_rank0("\n创建模型...", rank)
    
    if args.model.startswith('pasnet'):
        config = UNetConfig(
            fusion_mode=args.fusion_mode,
            use_aux=args.use_aux,
            mask_aware=args.mask_aware,
            use_spectral_adapter=args.use_spectral_adapter
        )
        model = create_model(args.model, config)
    else:
        model = create_model(args.model)
    
    # 同步BN
    if args.sync_bn and world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model = model.to(device)
    
    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"  参数量: {n_params:,}", rank)
    
    # 优化器
    scaled_lr = args.lr * world_size
    optimizer = AdamW(model.parameters(), lr=scaled_lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer, max_lr=scaled_lr,
            epochs=args.epochs, steps_per_epoch=len(train_loader)
        )
    else:
        scheduler = None
    
    # 损失函数
    if args.loss == 'hybrid':
        criterion = HybridPhysicsLoss(
            grad_weight=args.grad_loss_weight,
            profile_weight=args.profile_loss_weight
        )
    elif args.loss == 'combined':
        criterion = CombinedLoss(grad_weight=args.grad_loss_weight)
    elif args.loss == 'mse':
        class SimpleMSE(nn.Module):
            def __init__(self):
                super().__init__()
                self.loss = nn.MSELoss()
            def forward(self, pred, target, deep_preds=None):
                l = self.loss(pred, target)
                return l, {'total': l.item()}
        criterion = SimpleMSE()
    else:
        class SimpleMAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.loss = nn.L1Loss()
            def forward(self, pred, target, deep_preds=None):
                l = self.loss(pred, target)
                return l, {'total': l.item()}
        criterion = SimpleMAE()
    
    criterion = criterion.to(device)
    
    # TensorBoard
    writer = None
    if args.tensorboard and is_main_process(rank):
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(args.output_dir) / args.exp_name / 'logs'
            writer = SummaryWriter(log_dir)
        except ImportError:
            pass
    
    # 训练
    trainer = ExperimentTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        args=args,
        rank=rank,
        world_size=world_size,
        train_sampler=train_sampler,
        writer=writer
    )
    
    try:
        trainer.train()
    finally:
        if writer:
            writer.close()
        cleanup_distributed()


if __name__ == '__main__':
    main()
