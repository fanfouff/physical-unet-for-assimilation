#!/usr/bin/env python3
"""
===============================================================================
PAS-Net 模型定义及对比方法
Physics-Aware Spectral-Vertical Network with Baseline Methods
===============================================================================

包含模型:
1. PAS-Net (Physics-Aware Spectral-Vertical Network) - 本文方法
2. Vanilla U-Net - 基线方法
3. ResUNet - 残差U-Net
4. Attention U-Net - 注意力U-Net
5. PAVMT-UNet - 气象领域参考方法 (简化版)

消融实验配置:
- w/o Level-wise Norm: 使用全局标准化
- w/o Spectral Adapter: 直接拼接输入
- w/o Gradient Loss: 仅使用MSE/MAE损失
- w/o Auxiliary Features: 不使用地理/时间信息
- w/o Mask-Aware: 不使用掩码感知卷积

===============================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import math


# =============================================================================
# Part 1: 配置类
# =============================================================================

@dataclass
class UNetConfig:
    """U-Net模型配置"""
    # 输入通道
    obs_channels: int = 17          # FY-3F MWTS通道数
    bkg_channels: int = 37          # ERA5气压层数
    aux_channels: int = 4           # 辅助特征数
    out_channels: int = 37          # 输出通道 (与bkg相同)
    
    # 架构参数
    base_channels: int = 64         # 基础通道数
    depth: int = 4                  # U-Net深度
    
    # 融合模式
    fusion_mode: str = 'gated'      # 'concat', 'add', 'gated'
    
    # 功能开关 (用于消融实验)
    use_spectral_adapter: bool = True   # 使用光谱适配器
    use_aux: bool = True                # 使用辅助特征
    mask_aware: bool = True             # 使用掩码感知
    use_se_block: bool = True           # 使用SE注意力
    deep_supervision: bool = False      # 深度监督
    
    # 其他
    dropout: float = 0.1
    se_reduction: int = 8


# =============================================================================
# Part 2: 基础模块
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (通道注意力)"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.GELU(),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid()
        )
        self._attention = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.se(x)
        self._attention = att.detach()
        return x * att
    
    def get_attention(self) -> Optional[torch.Tensor]:
        return self._attention


class ConvBlock(nn.Module):
    """基础卷积块: Conv -> BN -> GELU"""
    
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        kernel_size: int = 3,
        use_bn: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """双卷积块"""
    
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int,
        mid_ch: Optional[int] = None,
        residual: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.residual = residual
        
        self.conv1 = ConvBlock(in_ch, mid_ch, dropout=dropout)
        self.conv2 = ConvBlock(mid_ch, out_ch, dropout=dropout)
        
        if residual and in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity() if residual else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        if self.residual:
            out = out + self.shortcut(x)
        return out


class DownBlock(nn.Module):
    """下采样块"""
    
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        use_se: bool = False,
        residual: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, residual=residual, dropout=dropout)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        x = self.se(x)
        return x


class UpBlock(nn.Module):
    """上采样块"""
    
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        skip_ch: int,
        use_se: bool = False,
        residual: bool = False,
        bilinear: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch + skip_ch, out_ch, residual=residual, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
            self.conv = DoubleConv(in_ch + skip_ch, out_ch, residual=residual, dropout=dropout)
        
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # 处理尺寸不匹配
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.se(x)
        return x


# =============================================================================
# Part 3: 辅助特征编码器
# =============================================================================

class AuxiliaryEncoder(nn.Module):
    """辅助地理/时间特征编码器"""
    
    def __init__(self, n_aux: int = 4, embed_dim: int = 32, periodic: bool = True):
        super().__init__()
        self.periodic = periodic
        in_ch = n_aux * 2 if periodic else n_aux
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
    
    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        if self.periodic:
            aux_scaled = aux * math.pi
            aux = torch.cat([torch.sin(aux_scaled), torch.cos(aux_scaled)], dim=1)
        return self.encoder(aux)


# =============================================================================
# Part 4: 光谱适配器 (Spectral Adapter Stem)
# =============================================================================

class SpectralAdapterStem(nn.Module):
    """
    光谱适配器茎干模块 - 论文核心创新点
    
    功能:
    1. 模拟逆辐射传输模型 (RTM^-1)
    2. 通道注意力权重学习
    3. 掩码感知融合
    """
    
    def __init__(
        self,
        obs_channels: int = 17,
        bkg_channels: int = 37,
        aux_channels: int = 4,
        latent_channels: int = 64,
        fusion_mode: str = 'gated',
        use_aux: bool = True,
        mask_aware: bool = True,
        use_se: bool = True,
        se_reduction: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.fusion_mode = fusion_mode
        self.use_aux = use_aux
        self.mask_aware = mask_aware
        self.latent_channels = latent_channels
        
        # 观测投影 (模拟逆RTM)
        obs_layers = [
            nn.Conv2d(obs_channels, latent_channels, 1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.GELU(),
        ]
        if use_se:
            obs_layers.append(SEBlock(latent_channels, se_reduction))
        obs_layers.extend([
            nn.Conv2d(latent_channels, latent_channels, 1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.GELU(),
            nn.Dropout2d(dropout)
        ])
        self.obs_projection = nn.Sequential(*obs_layers)
        
        # 背景投影
        self.bkg_projection = nn.Sequential(
            nn.Conv2d(bkg_channels, latent_channels, 1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.GELU()
        )
        
        # 辅助特征编码器
        aux_embed_dim = 32 if use_aux else 0
        self.aux_encoder = AuxiliaryEncoder(aux_channels, aux_embed_dim) if use_aux else None
        
        # 融合层
        if fusion_mode == 'concat':
            concat_ch = latent_channels * 2 + aux_embed_dim
            if mask_aware:
                concat_ch += 1
            self.fusion = nn.Sequential(
                nn.Conv2d(concat_ch, latent_channels, 1, bias=False),
                nn.BatchNorm2d(latent_channels),
                nn.GELU(),
                nn.Conv2d(latent_channels, latent_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(latent_channels),
                nn.GELU()
            )
        elif fusion_mode == 'add':
            self.alpha = nn.Parameter(torch.ones(latent_channels, 1, 1))
            self.fusion = nn.Sequential(
                nn.Conv2d(latent_channels, latent_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(latent_channels),
                nn.GELU()
            )
        elif fusion_mode == 'gated':
            gate_in = latent_channels * 2 + aux_embed_dim
            self.gate_net = nn.Sequential(
                nn.Conv2d(gate_in, latent_channels, 1, bias=False),
                nn.BatchNorm2d(latent_channels),
                nn.Sigmoid()
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(latent_channels, latent_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(latent_channels),
                nn.GELU()
            )
        
        # 残差连接
        self.bkg_residual = nn.Conv2d(bkg_channels, latent_channels, 1, bias=False)
    
    def forward(
        self, 
        obs: torch.Tensor, 
        bkg: torch.Tensor, 
        mask: torch.Tensor,
        aux: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 观测投影 + 掩码门控
        obs_feat = self.obs_projection(obs)
        mask_exp = mask if mask.shape[1] == 1 else mask.mean(dim=1, keepdim=True)
        obs_gated = obs_feat * mask_exp
        
        # 背景投影
        bkg_feat = self.bkg_projection(bkg)
        
        # 辅助特征
        aux_feat = self.aux_encoder(aux) if self.use_aux and aux is not None else None
        
        # 融合
        if self.fusion_mode == 'concat':
            feat_list = [obs_gated, bkg_feat]
            if aux_feat is not None:
                feat_list.append(aux_feat)
            if self.mask_aware:
                feat_list.append(mask_exp)
            fused = self.fusion(torch.cat(feat_list, dim=1))
            
        elif self.fusion_mode == 'add':
            fused = bkg_feat + self.alpha * obs_gated
            fused = self.fusion(fused)
            
        elif self.fusion_mode == 'gated':
            gate_input = [obs_feat, bkg_feat]
            if aux_feat is not None:
                gate_input.append(aux_feat)
            gate = self.gate_net(torch.cat(gate_input, dim=1)) * mask_exp
            fused = bkg_feat * (1 - gate) + obs_gated * gate
            fused = self.fusion(fused)
        
        # 残差连接
        return fused + self.bkg_residual(bkg)


class SimpleInputStem(nn.Module):
    """
    简单输入茎干 - 消融实验用 (w/o Spectral Adapter)
    直接拼接所有输入
    """
    
    def __init__(
        self,
        obs_channels: int = 17,
        bkg_channels: int = 37,
        aux_channels: int = 4,
        latent_channels: int = 64,
        use_aux: bool = True,
        mask_aware: bool = True
    ):
        super().__init__()
        
        total_ch = obs_channels + bkg_channels
        if use_aux:
            total_ch += aux_channels
        if mask_aware:
            total_ch += 1
        
        self.use_aux = use_aux
        self.mask_aware = mask_aware
        
        self.projection = nn.Sequential(
            nn.Conv2d(total_ch, latent_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.GELU()
        )
    
    def forward(
        self, 
        obs: torch.Tensor, 
        bkg: torch.Tensor, 
        mask: torch.Tensor,
        aux: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # NaN填充后直接拼接
        feat_list = [obs, bkg]
        if self.use_aux and aux is not None:
            feat_list.append(aux)
        if self.mask_aware:
            mask_exp = mask if mask.shape[1] == 1 else mask.mean(dim=1, keepdim=True)
            feat_list.append(mask_exp)
        
        return self.projection(torch.cat(feat_list, dim=1))


# =============================================================================
# Part 5: PAS-Net (本文方法)
# =============================================================================

class PASNet(nn.Module):
    """
    Physics-Aware Spectral-Vertical Network (PAS-Net)
    
    本文提出的主要架构，包含:
    1. 光谱适配器茎干 (SpectralAdapterStem)
    2. U-Net编解码器
    3. 可选深度监督
    """
    
    def __init__(self, config: Optional[UNetConfig] = None):
        super().__init__()
        
        config = config or UNetConfig()
        self.config = config
        self.deep_supervision = config.deep_supervision
        
        base = config.base_channels
        
        # 输入茎干
        if config.use_spectral_adapter:
            self.stem = SpectralAdapterStem(
                obs_channels=config.obs_channels,
                bkg_channels=config.bkg_channels,
                aux_channels=config.aux_channels,
                latent_channels=base,
                fusion_mode=config.fusion_mode,
                use_aux=config.use_aux,
                mask_aware=config.mask_aware,
                use_se=config.use_se_block,
                dropout=config.dropout
            )
        else:
            self.stem = SimpleInputStem(
                obs_channels=config.obs_channels,
                bkg_channels=config.bkg_channels,
                aux_channels=config.aux_channels,
                latent_channels=base,
                use_aux=config.use_aux,
                mask_aware=config.mask_aware
            )
        
        # Encoder
        self.inc = DoubleConv(base, base, residual=True, dropout=config.dropout)
        
        self.downs = nn.ModuleList()
        channels = [base]
        for i in range(config.depth):
            in_ch = base * (2 ** i)
            out_ch = base * (2 ** (i + 1))
            out_ch = min(out_ch, base * 8)  # 限制最大通道数
            self.downs.append(DownBlock(
                in_ch, out_ch, 
                use_se=config.use_se_block, 
                residual=True,
                dropout=config.dropout
            ))
            channels.append(out_ch)
        
        # Decoder
        self.ups = nn.ModuleList()
        self.deep_heads = nn.ModuleList() if config.deep_supervision else None
        
        for i in range(config.depth):
            in_ch = channels[-(i+1)]
            skip_ch = channels[-(i+2)]
            out_ch = skip_ch
            self.ups.append(UpBlock(
                in_ch, out_ch, skip_ch,
                use_se=config.use_se_block,
                residual=True,
                dropout=config.dropout
            ))
            
            if config.deep_supervision and i < config.depth - 1:
                self.deep_heads.append(nn.Conv2d(out_ch, config.out_channels, 1))
        
        # 输出头
        self.outc = nn.Conv2d(base, config.out_channels, 1)
    
    def forward(
        self, 
        obs: torch.Tensor, 
        bkg: torch.Tensor, 
        mask: torch.Tensor,
        aux: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        
        # Stem
        x = self.stem(obs, bkg, mask, aux)
        x = self.inc(x)
        
        # Encoder
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        # Decoder
        deep_outputs = []
        for i, up in enumerate(self.ups):
            x = up(x, skips[-(i+2)])
            
            if self.deep_supervision and self.deep_heads and i < len(self.deep_heads):
                deep_out = self.deep_heads[i](x)
                # 上采样到原始尺寸
                deep_out = F.interpolate(deep_out, size=obs.shape[2:], mode='bilinear', align_corners=True)
                deep_outputs.append(deep_out)
        
        # Output
        out = self.outc(x)
        
        if self.deep_supervision and deep_outputs:
            return out, deep_outputs
        return out


# =============================================================================
# Part 6: Vanilla U-Net (基线方法)
# =============================================================================

class VanillaUNet(nn.Module):
    """
    标准U-Net - 基线方法
    
    特点: 简单拼接输入，无物理先验
    """
    
    def __init__(
        self, 
        in_channels: int = 55,  # 17 + 37 + 1 (obs + bkg + mask)
        out_channels: int = 37,
        base_channels: int = 64,
        depth: int = 4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        base = base_channels
        
        # 初始卷积
        self.inc = DoubleConv(in_channels, base)
        
        # Encoder
        self.downs = nn.ModuleList()
        channels = [base]
        for i in range(depth):
            in_ch = base * (2 ** i)
            out_ch = min(base * (2 ** (i + 1)), base * 8)
            self.downs.append(DownBlock(in_ch, out_ch))
            channels.append(out_ch)
        
        # Decoder
        self.ups = nn.ModuleList()
        for i in range(depth):
            in_ch = channels[-(i+1)]
            skip_ch = channels[-(i+2)]
            out_ch = skip_ch
            self.ups.append(UpBlock(in_ch, out_ch, skip_ch))
        
        self.outc = nn.Conv2d(base, out_channels, 1)
    
    def forward(
        self, 
        obs: torch.Tensor, 
        bkg: torch.Tensor, 
        mask: torch.Tensor,
        aux: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 简单拼接
        mask_exp = mask if mask.shape[1] == 1 else mask.mean(dim=1, keepdim=True)
        x = torch.cat([obs, bkg, mask_exp], dim=1)
        
        x = self.inc(x)
        
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        for i, up in enumerate(self.ups):
            x = up(x, skips[-(i+2)])
        
        return self.outc(x)


# =============================================================================
# Part 7: ResUNet (残差U-Net对比方法)
# =============================================================================

class ResUNet(nn.Module):
    """
    Residual U-Net - 对比方法
    
    特点: 加入残差连接的U-Net
    """
    
    def __init__(
        self, 
        in_channels: int = 55,
        out_channels: int = 37,
        base_channels: int = 64,
        depth: int = 4
    ):
        super().__init__()
        
        base = base_channels
        
        self.inc = DoubleConv(in_channels, base, residual=True)
        
        self.downs = nn.ModuleList()
        channels = [base]
        for i in range(depth):
            in_ch = base * (2 ** i)
            out_ch = min(base * (2 ** (i + 1)), base * 8)
            self.downs.append(DownBlock(in_ch, out_ch, residual=True))
            channels.append(out_ch)
        
        self.ups = nn.ModuleList()
        for i in range(depth):
            in_ch = channels[-(i+1)]
            skip_ch = channels[-(i+2)]
            out_ch = skip_ch
            self.ups.append(UpBlock(in_ch, out_ch, skip_ch, residual=True))
        
        self.outc = nn.Conv2d(base, out_channels, 1)
    
    def forward(
        self, 
        obs: torch.Tensor, 
        bkg: torch.Tensor, 
        mask: torch.Tensor,
        aux: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        mask_exp = mask if mask.shape[1] == 1 else mask.mean(dim=1, keepdim=True)
        x = torch.cat([obs, bkg, mask_exp], dim=1)
        
        x = self.inc(x)
        
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        for i, up in enumerate(self.ups):
            x = up(x, skips[-(i+2)])
        
        return self.outc(x)


# =============================================================================
# Part 8: Attention U-Net (注意力U-Net对比方法)
# =============================================================================

class AttentionGate(nn.Module):
    """注意力门控模块"""
    
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 尺寸对齐
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.psi(self.relu(g1 + x1))
        return x * psi


class AttentionUpBlock(nn.Module):
    """带注意力门控的上采样块"""
    
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attention = AttentionGate(in_ch, skip_ch, skip_ch // 2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # 尺寸对齐
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        
        # 注意力加权
        skip = self.attention(x, skip)
        
        return self.conv(torch.cat([x, skip], dim=1))


class AttentionUNet(nn.Module):
    """
    Attention U-Net - 对比方法
    
    参考: Attention U-Net: Learning Where to Look for the Pancreas
    """
    
    def __init__(
        self, 
        in_channels: int = 55,
        out_channels: int = 37,
        base_channels: int = 64,
        depth: int = 4
    ):
        super().__init__()
        
        base = base_channels
        
        self.inc = DoubleConv(in_channels, base)
        
        self.downs = nn.ModuleList()
        channels = [base]
        for i in range(depth):
            in_ch = base * (2 ** i)
            out_ch = min(base * (2 ** (i + 1)), base * 8)
            self.downs.append(DownBlock(in_ch, out_ch))
            channels.append(out_ch)
        
        self.ups = nn.ModuleList()
        for i in range(depth):
            in_ch = channels[-(i+1)]
            skip_ch = channels[-(i+2)]
            out_ch = skip_ch
            self.ups.append(AttentionUpBlock(in_ch, out_ch, skip_ch))
        
        self.outc = nn.Conv2d(base, out_channels, 1)
    
    def forward(
        self, 
        obs: torch.Tensor, 
        bkg: torch.Tensor, 
        mask: torch.Tensor,
        aux: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        mask_exp = mask if mask.shape[1] == 1 else mask.mean(dim=1, keepdim=True)
        x = torch.cat([obs, bkg, mask_exp], dim=1)
        
        x = self.inc(x)
        
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        for i, up in enumerate(self.ups):
            x = up(x, skips[-(i+2)])
        
        return self.outc(x)


# =============================================================================
# Part 9: 模型工厂函数
# =============================================================================

def create_model(
    model_type: str,
    config: Optional[UNetConfig] = None,
    **kwargs
) -> nn.Module:
    """
    模型工厂函数
    
    Args:
        model_type: 模型类型
            - 'pasnet' / 'physics_unet': PAS-Net (本文方法)
            - 'pasnet_lite' / 'physics_unet_lite': 轻量版
            - 'pasnet_large' / 'physics_unet_large': 大型版
            - 'vanilla_unet': 标准U-Net基线
            - 'res_unet': 残差U-Net
            - 'attention_unet': 注意力U-Net
        config: 模型配置 (仅对PAS-Net有效)
    
    Returns:
        nn.Module: 模型实例
    """
    model_type = model_type.lower()
    
    # PAS-Net系列
    if model_type in ['pasnet', 'physics_unet']:
        config = config or UNetConfig()
        return PASNet(config)
    
    elif model_type in ['pasnet_lite', 'physics_unet_lite']:
        config = config or UNetConfig(base_channels=32, depth=3)
        return PASNet(config)
    
    elif model_type in ['pasnet_large', 'physics_unet_large']:
        config = config or UNetConfig(base_channels=96, depth=5)
        return PASNet(config)
    
    # 消融实验变体
    elif model_type == 'pasnet_no_adapter':
        config = config or UNetConfig()
        config.use_spectral_adapter = False
        return PASNet(config)
    
    elif model_type == 'pasnet_no_se':
        config = config or UNetConfig()
        config.use_se_block = False
        return PASNet(config)
    
    elif model_type == 'pasnet_no_aux':
        config = config or UNetConfig()
        config.use_aux = False
        return PASNet(config)
    
    elif model_type == 'pasnet_no_mask':
        config = config or UNetConfig()
        config.mask_aware = False
        return PASNet(config)
    
    # 对比方法
    elif model_type == 'vanilla_unet':
        return VanillaUNet(**kwargs)
    
    elif model_type == 'res_unet':
        return ResUNet(**kwargs)
    
    elif model_type == 'attention_unet':
        return AttentionUNet(**kwargs)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Part 10: 测试
# =============================================================================

def test_models():
    """测试所有模型"""
    print("=" * 70)
    print("模型测试")
    print("=" * 70)
    
    B, H, W = 2, 64, 64
    obs = torch.randn(B, 17, H, W)
    bkg = torch.randn(B, 37, H, W)
    mask = (torch.rand(B, 1, H, W) > 0.3).float()
    aux = torch.randn(B, 4, H, W)
    
    models_to_test = [
        ('pasnet', {}),
        ('vanilla_unet', {}),
        ('res_unet', {}),
        ('attention_unet', {}),
        ('pasnet_no_adapter', {}),
        ('pasnet_no_se', {}),
        ('pasnet_no_aux', {}),
    ]
    
    for model_name, kwargs in models_to_test:
        try:
            model = create_model(model_name, **kwargs)
            n_params = sum(p.numel() for p in model.parameters())
            
            out = model(obs, bkg, mask, aux)
            if isinstance(out, tuple):
                out = out[0]
            
            print(f"  {model_name:25s} | Params: {n_params:>10,} | Output: {tuple(out.shape)}")
            
        except Exception as e:
            print(f"  {model_name:25s} | ERROR: {e}")
    
    print("=" * 70)


if __name__ == "__main__":
    test_models()
