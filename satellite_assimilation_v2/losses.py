"""
===============================================================================
物理感知损失函数库 (Physics-Aware Loss Functions)
===============================================================================

包含多种物理约束损失函数，用于卫星数据同化任务

核心损失函数：
    1. HybridPhysicsLoss - 混合物理损失（基础回归 + 梯度保持 + 垂直廓线加权）
    2. VerticalProfileLoss - 垂直廓线一致性损失
    3. GradientPreservingLoss - 梯度保持损失
    4. PhysicsAwareMSELoss - 物理感知MSE损失（用于PAVMT-Unet论文）

参考：
    - PAVMT-Unet论文中的Physics-Aware MSE
    - 气象同化中的多尺度损失设计

===============================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np


# =============================================================================
# Part 1: 梯度计算工具
# =============================================================================

class SobelGradientOperator(nn.Module):
    """
    Sobel梯度算子
    
    用于计算2D气象场的空间梯度，保持锋面和气旋的锐利边界
    """
    def __init__(self):
        super().__init__()
        
        # Sobel卷积核（不需要训练）
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算梯度幅值
        
        Args:
            x: [B, C, H, W] 输入特征图
            
        Returns:
            grad: [B, C, H, W] 梯度幅值
        """
        B, C, H, W = x.shape
        
        # 对每个通道分别计算梯度
        x_reshaped = x.view(B * C, 1, H, W)
        
        # 计算x和y方向梯度
        gx = F.conv2d(x_reshaped, self.sobel_x, padding=1)
        gy = F.conv2d(x_reshaped, self.sobel_y, padding=1)
        
        # 梯度幅值
        grad = torch.sqrt(gx**2 + gy**2 + 1e-8)
        
        return grad.view(B, C, H, W)


# =============================================================================
# Part 2: 核心损失函数
# =============================================================================

class HybridPhysicsLoss(nn.Module):
    """
    混合物理损失函数
    
    结合三个关键组件：
        1. 基础回归损失（L1 Loss，对异常值更鲁棒）
        2. 梯度损失（保持气象场的锐度/纹理）
        3. 垂直廓线加权损失（给予关键层级更高权重）
    
    数学表达：
        L_total = w_mae * L_mae + w_grad * L_grad + w_profile * L_profile
    
    Args:
        w_mae: 基础MAE损失权重（默认1.0）
        w_grad: 梯度损失权重（默认0.5）
        w_profile: 垂直廓线损失权重（默认2.0）
        reduction: 损失聚合方式 ('mean' 或 'sum')
    
    Example:
        >>> criterion = HybridPhysicsLoss(w_mae=1.0, w_grad=0.5, w_profile=2.0)
        >>> pred = torch.randn(4, 37, 128, 128)
        >>> target = torch.randn(4, 37, 128, 128)
        >>> loss = criterion(pred, target)
    """
    
    def __init__(
        self,
        w_mae: float = 1.0,
        w_grad: float = 0.5,
        w_profile: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.w_mae = w_mae
        self.w_grad = w_grad
        self.w_profile = w_profile
        self.reduction = reduction
        
        # 梯度计算器
        self.sobel = SobelGradientOperator()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pressure_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        前向传播
        
        Args:
            pred: [B, C, H, W] 预测值
            target: [B, C, H, W] 目标值
            pressure_weights: [1, C, 1, 1] 可选的气压层权重
            
        Returns:
            total_loss: 总损失
            loss_dict: 各个损失组件的字典
        """
        # 1. 基础回归损失（L1 Loss对异常值更鲁棒）
        loss_mae = F.l1_loss(pred, target, reduction=self.reduction)
        
        # 2. 梯度损失（保持气象场的锐度/纹理）
        grad_pred = self.sobel(pred)
        grad_target = self.sobel(target)
        loss_grad = F.l1_loss(grad_pred, grad_target, reduction=self.reduction)
        
        # 3. 垂直廓线加权损失
        if pressure_weights is not None:
            # pressure_weights: [1, C, 1, 1] 或 [C]
            if pressure_weights.dim() == 1:
                pressure_weights = pressure_weights.view(1, -1, 1, 1)
            
            # 加权MAE
            weighted_error = torch.abs(pred - target) * pressure_weights
            loss_profile = weighted_error.mean() if self.reduction == 'mean' else weighted_error.sum()
        else:
            loss_profile = torch.tensor(0.0, device=pred.device)
        
        # 总损失
        total_loss = (
            self.w_mae * loss_mae +
            self.w_grad * loss_grad +
            self.w_profile * loss_profile
        )
        
        # 返回损失字典（用于日志记录）
        loss_dict = {
            'total': total_loss.item(),
            'mae': loss_mae.item(),
            'grad': loss_grad.item(),
            'profile': loss_profile.item() if isinstance(loss_profile, torch.Tensor) else 0.0
        }
        
        return total_loss, loss_dict


class VerticalProfileLoss(nn.Module):
    """
    垂直廓线一致性损失
    
    强制模型在垂直方向上保持物理一致性
    
    策略：
        - 低层（边界层）: 高权重 (近地表污染物/温度关键)
        - 中层（对流层）: 标准权重
        - 高层（平流层）: 高权重 (臭氧层、高空急流)
    
    Args:
        n_levels: 垂直层数（默认37层）
        boundary_layer_top: 边界层顶层索引（默认10层）
        stratosphere_bottom: 平流层底层索引（默认30层）
        high_weight: 关键层权重（默认2.0）
        low_weight: 普通层权重（默认1.0）
    """
    
    def __init__(
        self,
        n_levels: int = 37,
        boundary_layer_top: int = 10,
        stratosphere_bottom: int = 30,
        high_weight: float = 2.0,
        low_weight: float = 1.0
    ):
        super().__init__()
        
        # 构建权重张量
        weights = torch.ones(n_levels) * low_weight
        weights[:boundary_layer_top] = high_weight  # 边界层
        weights[stratosphere_bottom:] = high_weight  # 平流层
        
        self.register_buffer('level_weights', weights.view(1, -1, 1, 1))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
            
        Returns:
            loss: 加权损失
        """
        weighted_error = torch.abs(pred - target) * self.level_weights
        return weighted_error.mean()


class GradientPreservingLoss(nn.Module):
    """
    梯度保持损失
    
    专注于保持气象场的空间梯度（锋面、气旋边界）
    
    Args:
        loss_type: 损失类型 ('l1' 或 'l2')
        normalize: 是否归一化梯度幅值
    """
    
    def __init__(self, loss_type: str = 'l1', normalize: bool = False):
        super().__init__()
        self.sobel = SobelGradientOperator()
        self.loss_type = loss_type
        self.normalize = normalize
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
            
        Returns:
            loss: 梯度损失
        """
        grad_pred = self.sobel(pred)
        grad_target = self.sobel(target)
        
        if self.normalize:
            grad_pred = grad_pred / (grad_pred.mean() + 1e-8)
            grad_target = grad_target / (grad_target.mean() + 1e-8)
        
        if self.loss_type == 'l1':
            return F.l1_loss(grad_pred, grad_target)
        else:
            return F.mse_loss(grad_pred, grad_target)


class PhysicsAwareMSELoss(nn.Module):
    """
    物理感知MSE损失（参考PAVMT-Unet论文）
    
    添加了惩罚项以抑制非物理的负浓度值
    
    Args:
        penalty_coef: 负值惩罚系数（论文中使用5.0）
    """
    
    def __init__(self, penalty_coef: float = 5.0):
        super().__init__()
        self.penalty_coef = penalty_coef
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
            
        Returns:
            loss: MSE损失 + 负值惩罚
        """
        # 基础MSE损失
        mse_loss = F.mse_loss(pred, target)
        
        # 负值惩罚（仅对预测的负值施加惩罚）
        negative_mask = (pred < 0).float()
        negative_penalty = (pred * negative_mask).abs().mean()
        
        return mse_loss + self.penalty_coef * negative_penalty


# =============================================================================
# Part 3: 组合损失（用于深度监督）
# =============================================================================

class DeepSupervisionLoss(nn.Module):
    """
    深度监督损失
    
    对U-Net的多个解码器层级应用损失
    
    Args:
        base_loss: 基础损失函数
        aux_weights: 辅助输出的权重列表
    
    Example:
        >>> base_criterion = HybridPhysicsLoss()
        >>> criterion = DeepSupervisionLoss(base_criterion, aux_weights=[0.4, 0.2, 0.1])
        >>> pred_main = torch.randn(4, 37, 128, 128)
        >>> pred_aux = [torch.randn(4, 37, 64, 64), torch.randn(4, 37, 32, 32)]
        >>> target = torch.randn(4, 37, 128, 128)
        >>> loss = criterion(pred_main, target, pred_aux)
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        aux_weights: list = [0.4, 0.2, 0.1]
    ):
        super().__init__()
        self.base_loss = base_loss
        self.aux_weights = aux_weights
    
    def forward(
        self,
        pred_main: torch.Tensor,
        target: torch.Tensor,
        pred_aux: Optional[list] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_main: [B, C, H, W] 主输出
            target: [B, C, H, W] 目标
            pred_aux: 辅助输出列表（不同分辨率）
            **kwargs: 传递给base_loss的其他参数
            
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        # 主损失
        if isinstance(self.base_loss.forward(pred_main, target, **kwargs), tuple):
            main_loss, loss_dict = self.base_loss(pred_main, target, **kwargs)
        else:
            main_loss = self.base_loss(pred_main, target, **kwargs)
            loss_dict = {'total': main_loss.item()}
        
        total_loss = main_loss
        
        # 辅助损失
        if pred_aux is not None:
            for i, (pred_i, weight) in enumerate(zip(pred_aux, self.aux_weights)):
                # 下采样target以匹配辅助输出的分辨率
                target_i = F.interpolate(
                    target, size=pred_i.shape[-2:],
                    mode='bilinear', align_corners=False
                )
                
                if isinstance(self.base_loss.forward(pred_i, target_i, **kwargs), tuple):
                    aux_loss, _ = self.base_loss(pred_i, target_i, **kwargs)
                else:
                    aux_loss = self.base_loss(pred_i, target_i, **kwargs)
                
                total_loss = total_loss + weight * aux_loss
                loss_dict[f'aux_{i}'] = aux_loss.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# =============================================================================
# Part 4: 工厂函数
# =============================================================================

def create_loss_function(
    loss_type: str = 'hybrid',
    **kwargs
) -> nn.Module:
    """
    损失函数工厂
    
    Args:
        loss_type: 损失类型
            - 'hybrid': HybridPhysicsLoss
            - 'profile': VerticalProfileLoss
            - 'gradient': GradientPreservingLoss
            - 'physics_mse': PhysicsAwareMSELoss
            - 'mse': 标准MSE
            - 'mae': 标准MAE
        **kwargs: 传递给损失函数的参数
    
    Returns:
        loss_fn: 损失函数实例
    
    Example:
        >>> criterion = create_loss_function('hybrid', w_mae=1.0, w_grad=0.5)
    """
    loss_registry = {
        'hybrid': HybridPhysicsLoss,
        'profile': VerticalProfileLoss,
        'gradient': GradientPreservingLoss,
        'physics_mse': PhysicsAwareMSELoss,
        'mse': lambda **kw: nn.MSELoss(),
        'mae': lambda **kw: nn.L1Loss(),
        'huber': lambda **kw: nn.HuberLoss()
    }
    
    if loss_type not in loss_registry:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available: {list(loss_registry.keys())}"
        )
    
    return loss_registry[loss_type](**kwargs)


# =============================================================================
# Part 5: 使用示例和测试
# =============================================================================

if __name__ == '__main__':
    """测试损失函数"""
    print("=" * 70)
    print("测试物理感知损失函数")
    print("=" * 70)
    
    # 创建测试数据
    batch_size = 4
    n_channels = 37
    height, width = 128, 128
    
    pred = torch.randn(batch_size, n_channels, height, width)
    target = torch.randn(batch_size, n_channels, height, width)
    
    # 测试1: HybridPhysicsLoss
    print("\n1. HybridPhysicsLoss:")
    criterion = HybridPhysicsLoss(w_mae=1.0, w_grad=0.5, w_profile=2.0)
    
    # 创建气压权重（示例：给边界层和平流层更高权重）
    pressure_weights = torch.ones(1, n_channels, 1, 1)
    pressure_weights[0, :10, 0, 0] = 2.0  # 边界层
    pressure_weights[0, 30:, 0, 0] = 2.0  # 平流层
    
    loss, loss_dict = criterion(pred, target, pressure_weights)
    print(f"   总损失: {loss.item():.6f}")
    for key, value in loss_dict.items():
        print(f"   {key}: {value:.6f}")
    
    # 测试2: VerticalProfileLoss
    print("\n2. VerticalProfileLoss:")
    criterion = VerticalProfileLoss(n_levels=37)
    loss = criterion(pred, target)
    print(f"   损失: {loss.item():.6f}")
    
    # 测试3: GradientPreservingLoss
    print("\n3. GradientPreservingLoss:")
    criterion = GradientPreservingLoss()
    loss = criterion(pred, target)
    print(f"   损失: {loss.item():.6f}")
    
    # 测试4: PhysicsAwareMSELoss
    print("\n4. PhysicsAwareMSELoss:")
    criterion = PhysicsAwareMSELoss(penalty_coef=5.0)
    loss = criterion(pred, target)
    print(f"   损失: {loss.item():.6f}")
    
    # 测试5: DeepSupervisionLoss
    print("\n5. DeepSupervisionLoss:")
    base_criterion = HybridPhysicsLoss()
    criterion = DeepSupervisionLoss(base_criterion, aux_weights=[0.4, 0.2])
    
    pred_aux = [
        torch.randn(batch_size, n_channels, 64, 64),
        torch.randn(batch_size, n_channels, 32, 32)
    ]
    loss, loss_dict = criterion(pred, target, pred_aux, pressure_weights=pressure_weights)
    print(f"   总损失: {loss.item():.6f}")
    for key, value in loss_dict.items():
        print(f"   {key}: {value:.6f}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
