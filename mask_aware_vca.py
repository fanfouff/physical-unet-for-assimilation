"""
Mask-Aware Vertical Channel Attention (MA-VCA) 模块
===================================================
在原始PAVMT-Unet的VCA模块基础上,引入观测掩膜机制。

核心逻辑:
- mask=1: 有效观测区域,增强卫星信息权重
- mask=0: 无效/填充区域,抑制卫星信息,依赖背景场先验

作者: 基于PAVMT-Unet改进
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VerticalChannelAttention(nn.Module):
    """
    原始 VCA 模块 (PAVMT-Unet)
    
    用于skip connection中动态校准encoder特征
    """
    
    def __init__(self, in_channels: int, reduction: int = 4):
        """
        参数:
            in_channels: 输入通道数
            reduction: 通道压缩比例
        """
        super().__init__()
        
        # 深度可分离卷积 - 增强encoder特征的垂直局部信息
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=3, padding=1, 
            groups=in_channels, bias=False
        )
        
        # 上采样后的gating信号处理
        self.gate_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        )
        
        # 注意力权重生成
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x_encoder: torch.Tensor, g_decoder: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x_encoder: encoder的skip特征 (B, C, H, W)
            g_decoder: decoder的gating信号 (B, C, H/2, W/2)
            
        返回:
            calibrated: 校准后的特征 (B, C, H, W)
        """
        # 增强encoder的垂直局部特征
        x_enhanced = self.dw_conv(x_encoder)
        
        # 上采样gating信号
        g_upsampled = self.gate_conv(g_decoder)
        
        # 确保尺寸匹配
        if g_upsampled.shape[2:] != x_enhanced.shape[2:]:
            g_upsampled = F.interpolate(
                g_upsampled, size=x_enhanced.shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        # 融合生成注意力图
        combined = x_enhanced + g_upsampled
        alpha = self.attention(combined)
        
        # 加权校准
        calibrated = x_encoder * alpha
        
        return calibrated


class MaskAwareVCA(nn.Module):
    """
    掩膜感知的垂直通道注意力模块 (MA-VCA)
    
    在原始VCA基础上引入观测掩膜机制:
    - 有效观测区域(mask=1): 增强encoder(卫星)信息权重
    - 无效观测区域(mask=0): 抑制卫星信息,依赖decoder(背景场)先验
    
    架构:
        x_encoder (from encoder, contains satellite info)
            │
            ▼
        [DW-Conv] ─────────────────────┐
            │                          │
            ▼                          │
        [+ g_decoder (upsampled)]      │
            │                          │
            ▼                          │
        [Attention Conv]               │
            │                          │
            ▼                          │
        [Mask Modulation] ◄── mask     │
            │                          │
            ▼                          │
        [α * x_encoder] ◄──────────────┘
            │
            ▼
        calibrated output
    """
    
    def __init__(
        self, 
        in_channels: int, 
        mask_influence: float = 0.5,
        min_attention: float = 0.1,
        max_attention: float = 0.9
    ):
        """
        参数:
            in_channels: 输入通道数
            mask_influence: 掩膜对注意力的影响强度 (0-1)
            min_attention: 无效区域的最小注意力值
            max_attention: 有效区域的最大注意力值
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.mask_influence = mask_influence
        self.min_attention = min_attention
        self.max_attention = max_attention
        
        # ============================================
        # Encoder特征增强分支
        # ============================================
        self.encoder_enhance = nn.Sequential(
            # 深度可分离卷积 - 保持垂直层独立性
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        
        # ============================================
        # Decoder gating信号处理
        # ============================================
        self.decoder_gate = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        # ============================================
        # 基础注意力生成
        # ============================================
        self.base_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # ============================================
        # 掩膜调制网络
        # 学习如何根据mask调整注意力
        # ============================================
        self.mask_modulator = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 可学习的掩膜影响强度
        self.mask_scale = nn.Parameter(torch.tensor(mask_influence))
        
    def forward(
        self, 
        x_encoder: torch.Tensor, 
        g_decoder: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x_encoder: encoder的skip特征 (B, C, H, W) - 包含卫星信息
            g_decoder: decoder的gating信号 (B, C, H/2, W/2) - 包含背景场先验
            mask: 观测掩膜 (B, 1, H_orig, W_orig) 或 None
                  1=有效观测, 0=无效/填充
                  
        返回:
            calibrated: 校准后的特征 (B, C, H, W)
        """
        B, C, H, W = x_encoder.shape
        
        # 1. 增强encoder特征
        x_enhanced = self.encoder_enhance(x_encoder)
        
        # 2. 处理decoder gating信号
        g_processed = self.decoder_gate(g_decoder)
        
        # 确保尺寸匹配
        if g_processed.shape[2:] != (H, W):
            g_processed = F.interpolate(
                g_processed, size=(H, W), 
                mode='bilinear', align_corners=False
            )
        
        # 3. 融合encoder和decoder信息生成基础注意力
        combined = x_enhanced + g_processed
        base_alpha = self.base_attention(combined)
        
        # 4. 掩膜调制 (如果提供了掩膜)
        if mask is not None:
            # 下采样mask到当前特征图尺寸
            mask_resized = F.interpolate(
                mask.float(), size=(H, W), 
                mode='nearest'
            )
            
            # 使用掩膜调制网络
            mask_input = torch.cat([combined, mask_resized], dim=1)
            mask_modulation = self.mask_modulator(mask_input)
            
            # 计算最终注意力
            # mask=1时: 增强注意力(更多依赖encoder/卫星)
            # mask=0时: 抑制注意力(更多依赖decoder/背景场)
            mask_effect = torch.sigmoid(self.mask_scale) * mask_resized
            
            # 调制后的注意力
            alpha = base_alpha * (1 - mask_effect) + mask_modulation * mask_effect
            
            # 限制注意力范围
            alpha = alpha * (self.max_attention - self.min_attention) + self.min_attention
        else:
            alpha = base_alpha
        
        # 5. 校准输出
        # 注意: 这里使用残差连接,确保信息流通
        calibrated = x_encoder * alpha + g_processed * (1 - alpha)
        
        return calibrated


class MaskAwareVCAv2(nn.Module):
    """
    MA-VCA V2 - 增强版
    
    增加了:
    1. 多头注意力机制
    2. 空间-通道双重调制
    3. 自适应残差连接
    """
    
    def __init__(
        self, 
        in_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, \
            f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"
        
        # Encoder 特征处理
        self.encoder_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        
        # Decoder gating 处理
        self.decoder_proj = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels * 2, in_channels // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 掩膜融合
        self.mask_fusion = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 自适应残差权重
        self.residual_weight = nn.Parameter(torch.zeros(1))
        
    def forward(
        self, 
        x_encoder: torch.Tensor, 
        g_decoder: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, C, H, W = x_encoder.shape
        
        # 处理encoder和decoder特征
        x_enc = self.encoder_proj(x_encoder)
        g_dec = self.decoder_proj(g_decoder)
        
        if g_dec.shape[2:] != (H, W):
            g_dec = F.interpolate(g_dec, size=(H, W), mode='bilinear', align_corners=False)
        
        # 通道注意力
        combined_feat = torch.cat([
            x_enc.mean(dim=[2, 3]), 
            g_dec.mean(dim=[2, 3])
        ], dim=1)
        ca = self.channel_attention(combined_feat).view(B, C, 1, 1)
        
        # 空间注意力
        combined_spatial = torch.cat([
            x_enc.mean(dim=1, keepdim=True),
            g_dec.mean(dim=1, keepdim=True)
        ], dim=1)
        sa = self.spatial_attention(combined_spatial)
        
        # 基础注意力
        base_attention = ca * sa
        
        # 掩膜调制
        if mask is not None:
            mask_resized = F.interpolate(
                mask.float(), size=(H, W), mode='nearest'
            )
            mask_input = torch.cat([x_enc * base_attention, mask_resized], dim=1)
            mask_weight = self.mask_fusion(mask_input)
            
            # 有效观测区域增强encoder权重
            final_attention = base_attention * (1 + mask_resized * mask_weight)
            final_attention = torch.clamp(final_attention, 0.05, 0.95)
        else:
            final_attention = base_attention
        
        # 输出
        out = x_encoder * final_attention + g_dec * (1 - final_attention)
        
        # 自适应残差
        residual_w = torch.sigmoid(self.residual_weight)
        out = out * (1 - residual_w) + x_encoder * residual_w
        
        return out


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 Mask-Aware VCA 模块")
    print("=" * 60)
    
    batch_size = 4
    channels = 64
    H, W = 32, 32
    H_orig, W_orig = 128, 128  # 原始图像尺寸
    
    # 创建测试数据
    x_encoder = torch.randn(batch_size, channels, H, W)
    g_decoder = torch.randn(batch_size, channels, H // 2, W // 2)
    
    # 创建掩膜 (模拟: 部分区域有观测,部分区域无观测)
    mask = torch.zeros(batch_size, 1, H_orig, W_orig)
    mask[:, :, 20:100, 30:90] = 1.0  # 只有部分区域有观测
    
    print(f"\n输入形状:")
    print(f"  x_encoder: {x_encoder.shape}")
    print(f"  g_decoder: {g_decoder.shape}")
    print(f"  mask: {mask.shape}")
    print(f"  mask有效区域比例: {mask.mean():.2%}")
    
    # 测试原始VCA
    print("\n--- 原始 VCA ---")
    vca_orig = VerticalChannelAttention(channels)
    out_orig = vca_orig(x_encoder, g_decoder)
    print(f"输出形状: {out_orig.shape}")
    
    # 测试 MA-VCA
    print("\n--- Mask-Aware VCA ---")
    ma_vca = MaskAwareVCA(channels)
    
    # 不带mask
    out_no_mask = ma_vca(x_encoder, g_decoder, mask=None)
    print(f"无mask输出形状: {out_no_mask.shape}")
    
    # 带mask
    out_with_mask = ma_vca(x_encoder, g_decoder, mask=mask)
    print(f"有mask输出形状: {out_with_mask.shape}")
    
    # 检验mask的影响
    diff = (out_with_mask - out_no_mask).abs().mean()
    print(f"mask影响 (平均差异): {diff:.6f}")
    
    # 测试 MA-VCA V2
    print("\n--- Mask-Aware VCA V2 ---")
    ma_vca_v2 = MaskAwareVCAv2(channels)
    out_v2 = ma_vca_v2(x_encoder, g_decoder, mask=mask)
    print(f"V2输出形状: {out_v2.shape}")
    
    # 参数量统计
    print("\n参数量统计:")
    for name, model in [
        ("原始VCA", vca_orig),
        ("MA-VCA", ma_vca),
        ("MA-VCA V2", ma_vca_v2)
    ]:
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,}")
