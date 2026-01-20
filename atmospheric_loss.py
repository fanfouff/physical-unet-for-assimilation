"""
大气物理约束损失函数 (AtmosphericPhysicLoss)
============================================
综合损失函数,包含:
1. L1损失 - 对异常值更鲁棒
2. 梯度损失 - 保持气象锋面结构
3. 物理约束 - 非负惩罚等

作者: 基于PAVMT-Unet改进
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class SobelOperator(nn.Module):
    """
    Sobel算子模块
    
    用于计算图像的水平和垂直梯度
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel核 - 水平方向 (检测垂直边缘)
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Sobel核 - 垂直方向 (检测水平边缘)
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 注册为buffer (不参与训练,但会随模型保存/加载)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算输入的水平和垂直梯度
        
        参数:
            x: (B, C, H, W) 输入张量
            
        返回:
            grad_x: (B, C, H, W) 水平梯度
            grad_y: (B, C, H, W) 垂直梯度
        """
        B, C, H, W = x.shape
        
        # 扩展Sobel核到所有通道
        sobel_x = self.sobel_x.expand(C, 1, 3, 3)
        sobel_y = self.sobel_y.expand(C, 1, 3, 3)
        
        # 分组卷积 (每个通道独立计算梯度)
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)
        
        return grad_x, grad_y


class AtmosphericPhysicLoss(nn.Module):
    """
    大气物理约束损失函数
    
    Loss = λ1 * L1_Loss + λ2 * Gradient_Loss + λ3 * Physics_Constraint
    
    组成部分:
    1. L1 Loss: |pred - target|, 对异常值更鲁棒
    2. Gradient Loss: MSE(∇pred, ∇target), 保持锋面结构
    3. Physics Constraint: ReLU(-pred) * penalty, 非负惩罚
    
    物理意义:
    - L1损失确保预测值整体准确
    - 梯度损失保持气象场的空间梯度特征(如温度锋、气压槽)
    - 非负约束确保物理量(如浓度)非负
    """
    
    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_gradient: float = 0.5,
        lambda_physics: float = 0.1,
        negative_penalty: float = 5.0,
        use_smooth_l1: bool = False,
        smooth_l1_beta: float = 1.0
    ):
        """
        参数:
            lambda_l1: L1损失权重
            lambda_gradient: 梯度损失权重
            lambda_physics: 物理约束权重
            negative_penalty: 负值惩罚系数
            use_smooth_l1: 是否使用Smooth L1 (Huber Loss)
            smooth_l1_beta: Smooth L1的beta参数
        """
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_gradient = lambda_gradient
        self.lambda_physics = lambda_physics
        self.negative_penalty = negative_penalty
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1_beta = smooth_l1_beta
        
        # Sobel算子
        self.sobel = SobelOperator()
        
    def compute_l1_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算L1损失 (可选: 掩膜加权)
        
        参数:
            pred: 预测值 (B, C, H, W)
            target: 目标值 (B, C, H, W)
            mask: 可选的有效区域掩膜 (B, 1, H, W)
        """
        if self.use_smooth_l1:
            loss = F.smooth_l1_loss(pred, target, beta=self.smooth_l1_beta, reduction='none')
        else:
            loss = torch.abs(pred - target)
        
        if mask is not None:
            # 调整mask尺寸
            if mask.shape[2:] != pred.shape[2:]:
                mask = F.interpolate(mask.float(), size=pred.shape[2:], mode='nearest')
            # 加权损失
            loss = loss * mask
            # 归一化
            loss = loss.sum() / (mask.sum() * pred.shape[1] + 1e-8)
        else:
            loss = loss.mean()
            
        return loss
    
    def compute_gradient_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算梯度损失
        
        使用Sobel算子计算空间梯度,然后比较预测和目标的梯度
        
        参数:
            pred: 预测值 (B, C, H, W)
            target: 目标值 (B, C, H, W)
            mask: 可选的有效区域掩膜 (B, 1, H, W)
        """
        # 计算预测值的梯度
        pred_grad_x, pred_grad_y = self.sobel(pred)
        
        # 计算目标值的梯度
        target_grad_x, target_grad_y = self.sobel(target)
        
        # 梯度的MSE损失
        grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x, reduction='none')
        grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y, reduction='none')
        
        grad_loss = grad_loss_x + grad_loss_y
        
        if mask is not None:
            if mask.shape[2:] != pred.shape[2:]:
                mask = F.interpolate(mask.float(), size=pred.shape[2:], mode='nearest')
            grad_loss = grad_loss * mask
            grad_loss = grad_loss.sum() / (mask.sum() * pred.shape[1] + 1e-8)
        else:
            grad_loss = grad_loss.mean()
            
        return grad_loss
    
    def compute_physics_constraint(
        self, 
        pred: torch.Tensor,
        constraint_type: str = 'non_negative'
    ) -> torch.Tensor:
        """
        计算物理约束损失
        
        参数:
            pred: 预测值 (B, C, H, W)
            constraint_type: 约束类型
                - 'non_negative': 非负约束 (用于浓度、温度K等)
                - 'bounded': 有界约束
        """
        if constraint_type == 'non_negative':
            # 对负值进行惩罚: ReLU(-pred)
            negative_values = F.relu(-pred)
            physics_loss = negative_values.mean() * self.negative_penalty
            
        elif constraint_type == 'bounded':
            # 有界约束 (假设归一化后应在[0,1])
            below_zero = F.relu(-pred)
            above_one = F.relu(pred - 1)
            physics_loss = (below_zero.mean() + above_one.mean()) * self.negative_penalty
            
        else:
            physics_loss = torch.tensor(0.0, device=pred.device)
            
        return physics_loss
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        计算总损失
        
        参数:
            pred: 预测场 (B, C, H, W)
            target: 目标场 (B, C, H, W)  
            mask: 有效区域掩膜 (B, 1, H, W), 可选
            return_components: 是否返回各损失分量
            
        返回:
            total_loss: 总损失
            (可选) loss_dict: 各分量损失字典
        """
        # 1. L1 损失
        l1_loss = self.compute_l1_loss(pred, target, mask)
        
        # 2. 梯度损失
        gradient_loss = self.compute_gradient_loss(pred, target, mask)
        
        # 3. 物理约束
        physics_loss = self.compute_physics_constraint(pred)
        
        # 总损失
        total_loss = (
            self.lambda_l1 * l1_loss +
            self.lambda_gradient * gradient_loss +
            self.lambda_physics * physics_loss
        )
        
        if return_components:
            loss_dict = {
                'total': total_loss.item(),
                'l1': l1_loss.item(),
                'gradient': gradient_loss.item(),
                'physics': physics_loss.item(),
                'weighted_l1': (self.lambda_l1 * l1_loss).item(),
                'weighted_gradient': (self.lambda_gradient * gradient_loss).item(),
                'weighted_physics': (self.lambda_physics * physics_loss).item()
            }
            return total_loss, loss_dict
        
        return total_loss


class AtmosphericPhysicLossV2(AtmosphericPhysicLoss):
    """
    大气物理约束损失函数 V2 - 增强版
    
    增加了:
    1. 垂直一致性损失 (确保相邻层之间的平滑过渡)
    2. 结构相似性损失 (SSIM)
    3. 多尺度梯度损失
    """
    
    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_gradient: float = 0.5,
        lambda_physics: float = 0.1,
        lambda_vertical: float = 0.2,
        lambda_ssim: float = 0.1,
        negative_penalty: float = 5.0
    ):
        super().__init__(
            lambda_l1=lambda_l1,
            lambda_gradient=lambda_gradient,
            lambda_physics=lambda_physics,
            negative_penalty=negative_penalty
        )
        
        self.lambda_vertical = lambda_vertical
        self.lambda_ssim = lambda_ssim
        
    def compute_vertical_consistency_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算垂直一致性损失
        
        确保预测场在垂直方向(通道维度)有合理的过渡
        
        参数:
            pred: (B, C, H, W) - C是垂直层数
            target: (B, C, H, W)
        """
        # 计算相邻层之间的差异
        pred_diff = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        target_diff = target[:, 1:, :, :] - target[:, :-1, :, :]
        
        # 垂直梯度的MSE
        vertical_loss = F.mse_loss(pred_diff, target_diff)
        
        return vertical_loss
    
    def compute_ssim_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        window_size: int = 11,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        计算结构相似性损失 (1 - SSIM)
        
        SSIM考虑亮度、对比度和结构三个方面
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 使用平均池化近似高斯滤波
        pad = window_size // 2
        
        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=pad)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=pad) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=pad) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_pred_target
        
        ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        if reduction == 'mean':
            return 1 - ssim.mean()
        return 1 - ssim
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """计算增强版总损失"""
        
        # 基础损失
        l1_loss = self.compute_l1_loss(pred, target, mask)
        gradient_loss = self.compute_gradient_loss(pred, target, mask)
        physics_loss = self.compute_physics_constraint(pred)
        
        # 增强损失
        vertical_loss = self.compute_vertical_consistency_loss(pred, target)
        ssim_loss = self.compute_ssim_loss(pred, target)
        
        # 总损失
        total_loss = (
            self.lambda_l1 * l1_loss +
            self.lambda_gradient * gradient_loss +
            self.lambda_physics * physics_loss +
            self.lambda_vertical * vertical_loss +
            self.lambda_ssim * ssim_loss
        )
        
        if return_components:
            loss_dict = {
                'total': total_loss.item(),
                'l1': l1_loss.item(),
                'gradient': gradient_loss.item(),
                'physics': physics_loss.item(),
                'vertical': vertical_loss.item(),
                'ssim': ssim_loss.item()
            }
            return total_loss, loss_dict
        
        return total_loss


# ============================================
# 辅助函数: 用于1D廓线反演的损失
# ============================================
class ProfileInversionLoss(nn.Module):
    """
    廓线反演专用损失函数
    
    用于从亮温反演温度廓线的任务
    结合观测一致性约束 (使用Jacobian矩阵)
    """
    
    def __init__(
        self,
        jacobian: torch.Tensor,
        lambda_profile: float = 1.0,
        lambda_obs: float = 0.1,
        lambda_smooth: float = 0.05
    ):
        """
        参数:
            jacobian: Jacobian矩阵 (n_channels, n_levels), 描述廓线变化对亮温的影响
            lambda_profile: 廓线MSE权重
            lambda_obs: 观测一致性权重
            lambda_smooth: 廓线平滑性权重
        """
        super().__init__()
        
        self.register_buffer('K', jacobian)  # (n_ch, n_lev)
        self.lambda_profile = lambda_profile
        self.lambda_obs = lambda_obs
        self.lambda_smooth = lambda_smooth
        
    def forward(
        self,
        pred_profile: torch.Tensor,
        true_profile: torch.Tensor,
        true_bt: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            pred_profile: 预测廓线 (B, n_levels)
            true_profile: 真实廓线 (B, n_levels)
            true_bt: 真实观测亮温 (B, n_channels)
        """
        # 1. 廓线MSE
        profile_loss = F.mse_loss(pred_profile, true_profile)
        
        # 2. 观测一致性: 模拟正向辐射传输
        # rec_bt = K @ pred_profile
        rec_bt = torch.matmul(pred_profile, self.K.T)  # (B, n_ch)
        obs_loss = F.mse_loss(rec_bt, true_bt)
        
        # 3. 廓线平滑性: 抑制垂直方向的剧烈波动
        profile_diff = pred_profile[:, 1:] - pred_profile[:, :-1]
        smooth_loss = (profile_diff ** 2).mean()
        
        # 总损失
        total_loss = (
            self.lambda_profile * profile_loss +
            self.lambda_obs * obs_loss +
            self.lambda_smooth * smooth_loss
        )
        
        return total_loss


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 AtmosphericPhysicLoss")
    print("=" * 60)
    
    batch_size = 4
    channels = 37  # 垂直层数
    H, W = 64, 64
    
    # 创建测试数据
    pred = torch.randn(batch_size, channels, H, W) * 0.5 + 0.5
    target = torch.randn(batch_size, channels, H, W) * 0.5 + 0.5
    mask = torch.ones(batch_size, 1, H, W)
    mask[:, :, 10:30, 20:40] = 0  # 部分区域无效
    
    # 引入一些负值来测试物理约束
    pred[:, :5, :, :] = pred[:, :5, :, :] - 0.8
    
    print(f"\n输入形状:")
    print(f"  pred: {pred.shape}")
    print(f"  target: {target.shape}")
    print(f"  mask: {mask.shape}")
    print(f"  pred 负值比例: {(pred < 0).float().mean():.2%}")
    
    # 测试基础版本
    print("\n--- AtmosphericPhysicLoss ---")
    criterion = AtmosphericPhysicLoss(
        lambda_l1=1.0,
        lambda_gradient=0.5,
        lambda_physics=0.1
    )
    
    loss, loss_dict = criterion(pred, target, mask, return_components=True)
    print(f"总损失: {loss.item():.4f}")
    print("各分量:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    # 测试V2版本
    print("\n--- AtmosphericPhysicLoss V2 ---")
    criterion_v2 = AtmosphericPhysicLossV2(
        lambda_l1=1.0,
        lambda_gradient=0.5,
        lambda_physics=0.1,
        lambda_vertical=0.2,
        lambda_ssim=0.1
    )
    
    loss_v2, loss_dict_v2 = criterion_v2(pred, target, mask, return_components=True)
    print(f"总损失: {loss_v2.item():.4f}")
    print("各分量:")
    for k, v in loss_dict_v2.items():
        print(f"  {k}: {v:.4f}")
    
    # 测试廓线反演损失
    print("\n--- ProfileInversionLoss ---")
    n_channels = 13
    n_levels = 37
    
    # 模拟Jacobian矩阵
    jacobian = torch.randn(n_channels, n_levels) * 0.1
    
    criterion_profile = ProfileInversionLoss(jacobian)
    
    pred_profile = torch.randn(batch_size, n_levels)
    true_profile = torch.randn(batch_size, n_levels)
    true_bt = torch.randn(batch_size, n_channels)
    
    profile_loss = criterion_profile(pred_profile, true_profile, true_bt)
    print(f"廓线损失: {profile_loss.item():.4f}")
    
    # 梯度检查
    print("\n--- 梯度检查 ---")
    pred.requires_grad = True
    loss = criterion(pred, target)
    loss.backward()
    print(f"pred梯度形状: {pred.grad.shape}")
    print(f"梯度范围: [{pred.grad.min():.4f}, {pred.grad.max():.4f}]")
