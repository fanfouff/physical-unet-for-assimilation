#!/usr/bin/env python3
"""
===============================================================================
损失函数定义 - 混合物理损失
Loss Functions - Hybrid Physics Loss
===============================================================================

包含:
1. 基础损失 (MSE, MAE, Huber)
2. 梯度损失 (Sobel Gradient Loss)
3. 高度加权损失 (Height-Weighted Loss)
4. 混合物理损失 (Hybrid Physics Loss)
5. 分层损失 (Level-wise Loss)

===============================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# =============================================================================
# Part 1: 基础损失函数
# =============================================================================

class MaskedMSELoss(nn.Module):
    """掩码MSE损失 - 仅在有效区域计算"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss = (pred - target) ** 2
        
        if mask is not None:
            # 扩展掩码到所有通道
            if mask.shape[1] == 1:
                mask = mask.expand_as(pred)
            loss = loss * mask
            
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        
        return loss


class MaskedMAELoss(nn.Module):
    """掩码MAE损失"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss = torch.abs(pred - target)
        
        if mask is not None:
            if mask.shape[1] == 1:
                mask = mask.expand_as(pred)
            loss = loss * mask
            
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        
        return loss


# =============================================================================
# Part 2: 梯度损失 (Sobel Gradient Loss)
# =============================================================================

class SobelGradientLoss(nn.Module):
    """
    Sobel梯度损失 - 保持结构完整性
    
    计算预测场和目标场的Sobel梯度差异
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
        # Sobel算子 (不需要梯度)
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _compute_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算Sobel梯度"""
        B, C, H, W = x.shape
        
        # 展平通道维度
        x_flat = x.view(B * C, 1, H, W)
        
        grad_x = F.conv2d(x_flat, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_flat, self.sobel_y, padding=1)
        
        grad_x = grad_x.view(B, C, H, W)
        grad_y = grad_y.view(B, C, H, W)
        
        return grad_x, grad_y
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pred_gx, pred_gy = self._compute_gradient(pred)
        tgt_gx, tgt_gy = self._compute_gradient(target)
        
        loss_x = torch.abs(pred_gx - tgt_gx)
        loss_y = torch.abs(pred_gy - tgt_gy)
        loss = loss_x + loss_y
        
        if mask is not None:
            if mask.shape[1] == 1:
                mask = mask.expand_as(pred)
            loss = loss * mask
            
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class GradientMagnitudeLoss(nn.Module):
    """梯度幅值损失"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
        sobel_x = torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B * C, 1, H, W)
        
        gx = F.conv2d(x_flat, self.sobel_x, padding=1)
        gy = F.conv2d(x_flat, self.sobel_y, padding=1)
        
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return mag.view(B, C, H, W)
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pred_mag = self._gradient_magnitude(pred)
        tgt_mag = self._gradient_magnitude(target)
        
        loss = torch.abs(pred_mag - tgt_mag)
        
        if mask is not None:
            if mask.shape[1] == 1:
                mask = mask.expand_as(pred)
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss


# =============================================================================
# Part 3: 高度加权损失 (Height-Weighted Loss)
# =============================================================================

class HeightWeightedLoss(nn.Module):
    """
    高度加权损失 - 给平流层和边界层更高权重
    
    用于解决垂直尺度差异问题
    """
    
    def __init__(
        self, 
        n_levels: int = 37,
        pressure_levels: Optional[np.ndarray] = None,
        stratosphere_threshold: float = 100.0,  # hPa
        boundary_threshold: float = 850.0,      # hPa
        stratosphere_weight: float = 2.0,
        boundary_weight: float = 1.5,
        base_loss: str = 'mse'
    ):
        super().__init__()
        
        # ERA5默认气压层
        if pressure_levels is None:
            pressure_levels = np.array([
                1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
                750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
                250, 225, 200, 175, 150, 125, 100, 70, 50, 30,
                20, 10, 7, 5, 3, 2, 1
            ])
        
        # 计算层权重
        weights = np.ones(n_levels)
        for i, p in enumerate(pressure_levels[:n_levels]):
            if p <= stratosphere_threshold:
                weights[i] = stratosphere_weight
            elif p >= boundary_threshold:
                weights[i] = boundary_weight
        
        # 归一化
        weights = weights / weights.sum() * n_levels
        
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        
        # 基础损失
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        else:
            self.base_loss = nn.L1Loss(reduction='none')
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W] 预测
            target: [B, C, H, W] 目标
            mask: [B, 1, H, W] 可选掩码
        """
        loss = self.base_loss(pred, target)  # [B, C, H, W]
        
        # 应用层权重
        weights = self.weights.view(1, -1, 1, 1)  # [1, C, 1, 1]
        loss = loss * weights
        
        if mask is not None:
            if mask.shape[1] == 1:
                mask = mask.expand_as(pred)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        
        return loss.mean()


class LevelWiseLoss(nn.Module):
    """分层损失 - 分别计算各层损失"""
    
    def __init__(self, n_levels: int = 37, base_loss: str = 'mse'):
        super().__init__()
        self.n_levels = n_levels
        
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        else:
            self.base_loss = nn.L1Loss(reduction='none')
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        返回各层损失
        """
        loss = self.base_loss(pred, target)  # [B, C, H, W]
        
        # 各层平均损失
        levelwise = loss.mean(dim=(0, 2, 3))  # [C]
        
        # 全局损失
        global_loss = loss.mean()
        
        return {
            'levelwise': levelwise,
            'global': global_loss
        }


# =============================================================================
# Part 4: 混合物理损失 (Hybrid Physics Loss) - 论文核心
# =============================================================================

class HybridPhysicsLoss(nn.Module):
    """
    混合物理损失 - 论文核心损失函数
    
    L = λ_mae * L_mae + λ_grad * L_grad + λ_profile * L_profile
    
    其中:
    - L_mae: MAE损失 (鲁棒数值回归)
    - L_grad: Sobel梯度损失 (结构保持)
    - L_profile: 高度加权损失 (垂直一致性)
    """
    
    def __init__(
        self,
        mae_weight: float = 1.0,
        grad_weight: float = 0.1,
        profile_weight: float = 0.5,
        deep_weight: float = 0.3,
        n_levels: int = 37,
        pressure_levels: Optional[np.ndarray] = None
    ):
        super().__init__()
        
        self.mae_weight = mae_weight
        self.grad_weight = grad_weight
        self.profile_weight = profile_weight
        self.deep_weight = deep_weight
        
        # 子损失函数
        self.mae_loss = MaskedMAELoss()
        self.grad_loss = SobelGradientLoss()
        self.profile_loss = HeightWeightedLoss(
            n_levels=n_levels,
            pressure_levels=pressure_levels,
            base_loss='mae'
        )
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deep_preds: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: [B, 37, H, W] 主输出
            target: [B, 37, H, W] 目标
            mask: [B, 1, H, W] 有效性掩码
            deep_preds: 深度监督输出列表
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失字典
        """
        # MAE损失
        l_mae = self.mae_loss(pred, target, mask)
        
        # 梯度损失
        l_grad = self.grad_loss(pred, target, mask)
        
        # 高度加权损失
        l_profile = self.profile_loss(pred, target, mask)
        
        # 组合
        total = (
            self.mae_weight * l_mae + 
            self.grad_weight * l_grad + 
            self.profile_weight * l_profile
        )
        
        # 深度监督
        l_deep = torch.tensor(0.0, device=pred.device)
        if deep_preds:
            for dp in deep_preds:
                l_deep = l_deep + self.mae_loss(dp, target, mask)
            l_deep = l_deep / len(deep_preds)
            total = total + self.deep_weight * l_deep
        
        loss_dict = {
            'total': total.item(),
            'mae': l_mae.item(),
            'grad': l_grad.item(),
            'profile': l_profile.item(),
            'deep': l_deep.item() if isinstance(l_deep, torch.Tensor) else l_deep
        }
        
        return total, loss_dict


class CombinedLoss(nn.Module):
    """
    简化版组合损失 (兼容原train_ddp.py)
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
        
        if base_loss == 'mse':
            self.base_loss = nn.MSELoss()
        elif base_loss == 'mae':
            self.base_loss = nn.L1Loss()
        else:
            self.base_loss = nn.HuberLoss()
        
        self.grad_loss = SobelGradientLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        deep_preds: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        base = self.base_loss(pred, target)
        grad = self.grad_loss(pred, target)
        
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
            'deep': deep.item() if isinstance(deep, torch.Tensor) else deep
        }


# =============================================================================
# Part 5: 消融实验专用损失
# =============================================================================

class AblationLoss(nn.Module):
    """
    消融实验损失函数
    
    可通过配置开关各损失项
    """
    
    def __init__(
        self,
        use_mae: bool = True,
        use_grad: bool = True,
        use_profile: bool = True,
        mae_weight: float = 1.0,
        grad_weight: float = 0.1,
        profile_weight: float = 0.5,
        n_levels: int = 37
    ):
        super().__init__()
        
        self.use_mae = use_mae
        self.use_grad = use_grad
        self.use_profile = use_profile
        
        self.mae_weight = mae_weight if use_mae else 0.0
        self.grad_weight = grad_weight if use_grad else 0.0
        self.profile_weight = profile_weight if use_profile else 0.0
        
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.grad_loss = SobelGradientLoss() if use_grad else None
        self.profile_loss = HeightWeightedLoss(n_levels=n_levels) if use_profile else None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        deep_preds: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        loss_dict = {}
        total = torch.tensor(0.0, device=pred.device)
        
        # MAE/MSE基础损失
        if self.use_mae:
            l_mae = self.mae_loss(pred, target)
            total = total + self.mae_weight * l_mae
            loss_dict['mae'] = l_mae.item()
        else:
            l_mse = self.mse_loss(pred, target)
            total = total + l_mse
            loss_dict['mse'] = l_mse.item()
        
        # 梯度损失
        if self.use_grad and self.grad_loss is not None:
            l_grad = self.grad_loss(pred, target)
            total = total + self.grad_weight * l_grad
            loss_dict['grad'] = l_grad.item()
        
        # 高度加权损失
        if self.use_profile and self.profile_loss is not None:
            l_profile = self.profile_loss(pred, target)
            total = total + self.profile_weight * l_profile
            loss_dict['profile'] = l_profile.item()
        
        loss_dict['total'] = total.item()
        
        return total, loss_dict


# =============================================================================
# Part 6: 测试
# =============================================================================

def test_losses():
    """测试所有损失函数"""
    print("=" * 70)
    print("损失函数测试")
    print("=" * 70)
    
    B, C, H, W = 4, 37, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)
    mask = (torch.rand(B, 1, H, W) > 0.3).float()
    
    # 测试各损失函数
    losses = [
        ('MaskedMSELoss', MaskedMSELoss()),
        ('MaskedMAELoss', MaskedMAELoss()),
        ('SobelGradientLoss', SobelGradientLoss()),
        ('GradientMagnitudeLoss', GradientMagnitudeLoss()),
        ('HeightWeightedLoss', HeightWeightedLoss()),
    ]
    
    for name, loss_fn in losses:
        try:
            if 'Masked' in name or 'Gradient' in name:
                loss = loss_fn(pred, target, mask)
            else:
                loss = loss_fn(pred, target)
            print(f"  {name:25s}: {loss.item():.6f}")
        except Exception as e:
            print(f"  {name:25s}: ERROR - {e}")
    
    # 测试混合损失
    print("\n组合损失:")
    hybrid = HybridPhysicsLoss()
    total, loss_dict = hybrid(pred, target, mask)
    for k, v in loss_dict.items():
        print(f"  {k:15s}: {v:.6f}")
    
    print("=" * 70)


if __name__ == "__main__":
    test_losses()
