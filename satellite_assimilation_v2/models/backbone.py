"""
===============================================================================
物理感知U-Net骨干网络 (Physics-Aware U-Net Backbone)
===============================================================================

完整的端到端卫星数据同化模型架构

结构:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  X_obs [B,17,H,W] ──┐                                                       │
│                     ├──> SpectralAdapterStemV2 ──> F_stem [B,64,H,W]       │
│  X_bkg [B,37,H,W] ──┤                                   │                   │
│                     │                                   ▼                   │
│  X_aux [B,4,H,W]  ──┤                            ┌─────────────┐            │
│                     │                            │   Encoder   │            │
│  Mask  [B,1,H,W]  ──┘                            │  (ResBlocks │            │
│                                                  │  + Downsamp)│            │
│                                                  └──────┬──────┘            │
│                                                         │                   │
│                                                         ▼                   │
│                                                  ┌─────────────┐            │
│                                                  │  Bottleneck │            │
│                                                  │  (Attention)│            │
│                                                  └──────┬──────┘            │
│                                                         │                   │
│                                                         ▼                   │
│                                                  ┌─────────────┐            │
│                                                  │   Decoder   │            │
│                                                  │  (ResBlocks │            │
│                                                  │  + Upsample)│            │
│                                                  └──────┬──────┘            │
│                                                         │                   │
│                                                         ▼                   │
│                                                  ┌─────────────┐            │
│                                                  │  Output Head│──> Y_pred │
│                                                  │  [B,37,H,W] │   [B,37,H,W]
│                                                  └─────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

===============================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import math

# 导入V2模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline_v2 import (
    SpectralAdapterStemV2, 
    ModelConfig,
    DataConfig
)


# =============================================================================
# Part 1: 配置类
# =============================================================================

@dataclass
class UNetConfig:
    """U-Net配置"""
    # 输入配置
    obs_channels: int = 17
    bkg_channels: int = 37
    aux_channels: int = 4
    out_channels: int = 37
    
    # Stem配置
    stem_channels: int = 64
    fusion_mode: str = 'gated'  # 'concat', 'add', 'gated'
    use_aux: bool = True
    mask_aware: bool = True
    use_spectral_stem: bool = True  # False → 消融V4: 标准3×3卷积替代频谱适配茎干
    
    # Encoder配置
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    encoder_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    
    # Bottleneck配置
    bottleneck_channels: int = 512
    use_attention: bool = True
    attention_heads: int = 8
    
    # Decoder配置
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 64])
    
    # 训练配置
    dropout: float = 0.1
    
    # 深度监督
    deep_supervision: bool = False


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
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class ConvBNReLU(nn.Module):
    """卷积 + BatchNorm + 激活"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        activation: str = 'gelu',
        bias: bool = False
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride, padding, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        self.act = {
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True),
            'none': nn.Identity()
        }.get(activation, nn.GELU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    残差块 (Pre-activation ResNet风格)
    
    结构: x -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> + x
    """
    
    def __init__(
        self,
        channels: int,
        expansion: int = 1,
        stride: int = 1,
        dropout: float = 0.0,
        use_se: bool = True,
        se_reduction: int = 16
    ):
        super().__init__()
        
        mid_channels = channels * expansion
        
        self.conv1 = ConvBNReLU(channels, mid_channels, 3, stride, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        self.se = SEBlock(channels, se_reduction) if use_se else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()
        
        # 如果stride>1，需要下采样shortcut
        self.shortcut = nn.Identity()
        if stride > 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride),
                nn.Conv2d(channels, channels, 1, bias=False)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.dropout(out)
        
        return self.act(out + identity)


class DownsampleBlock(nn.Module):
    """下采样块: 特征图尺寸减半，通道数增加"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # 第一个block负责通道变换和下采样
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # 后续blocks保持通道数
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(out_channels, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class UpsampleBlock(nn.Module):
    """上采样块: 特征图尺寸加倍，通道数减少"""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        n_blocks: int = 2,
        dropout: float = 0.0,
        upsample_mode: str = 'bilinear'  # 'bilinear', 'nearest', 'transpose'
    ):
        super().__init__()
        
        # 上采样
        if upsample_mode == 'transpose':
            self.upsample = nn.ConvTranspose2d(
                in_channels, in_channels, 4, stride=2, padding=1
            )
        else:
            self.upsample = nn.Upsample(
                scale_factor=2, mode=upsample_mode, 
                align_corners=True if upsample_mode == 'bilinear' else None
            )
        
        # Skip connection融合
        self.fusion = ConvBNReLU(in_channels + skip_channels, out_channels, 1, 1, 0)
        
        # 残差块
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(out_channels, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # 处理尺寸不匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        x = self.blocks(x)
        return x


# =============================================================================
# Part 3: 注意力模块
# =============================================================================

class SpatialAttention(nn.Module):
    """空间注意力 (用于Bottleneck)"""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        self.query = nn.Conv2d(channels, channels // reduction, 1)
        self.key = nn.Conv2d(channels, channels // reduction, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Q, K, V
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C']
        k = self.key(x).view(B, -1, H * W)                      # [B, C', HW]
        v = self.value(x).view(B, -1, H * W)                    # [B, C, HW]
        
        # Attention
        attn = self.softmax(torch.bmm(q, k))  # [B, HW, HW]
        out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x


class ChannelAttention(nn.Module):
    """通道注意力"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(channels, reduction)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# =============================================================================
# Part 4: 完整的物理感知U-Net
# =============================================================================

class PhysicsAwareUNet(nn.Module):
    """
    物理感知U-Net - 完整的端到端卫星数据同化模型
    
    创新点:
    1. SpectralAdapterStem: 物理感知的多源数据融合入口
    2. Mask-Aware Processing: 显式处理观测缺失
    3. Multi-scale Skip Connections: 保留多尺度特征
    4. Deep Supervision (可选): 加速收敛
    """
    
    def __init__(self, config: Optional[UNetConfig] = None):
        super().__init__()
        
        self.config = config or UNetConfig()
        cfg = self.config
        
        # =====================================================================
        # 1. 物理感知Stem (或消融V4: 标准卷积Stem)
        # =====================================================================
        if cfg.use_spectral_stem:
            self.stem = SpectralAdapterStemV2(
                obs_channels=cfg.obs_channels,
                bkg_channels=cfg.bkg_channels,
                aux_channels=cfg.aux_channels,
                latent_channels=cfg.stem_channels,
                fusion_mode=cfg.fusion_mode,
                use_aux=cfg.use_aux,
                mask_aware=cfg.mask_aware,
                dropout=cfg.dropout
            )
        else:
            # V4消融: 无SpectralStem，用标准3×3卷积拼接输入
            in_ch = cfg.obs_channels + cfg.bkg_channels
            if cfg.use_aux:
                in_ch += cfg.aux_channels
            self.stem = _SimpleStem(in_ch, cfg.stem_channels, cfg.dropout)
        
        # =====================================================================
        # 2. Encoder (下采样路径)
        # =====================================================================
        self.encoder_blocks = nn.ModuleList()
        in_ch = cfg.stem_channels
        
        for i, (out_ch, depth) in enumerate(zip(cfg.encoder_channels, cfg.encoder_depths)):
            self.encoder_blocks.append(
                DownsampleBlock(in_ch, out_ch, n_blocks=depth, dropout=cfg.dropout)
            )
            in_ch = out_ch
        
        # =====================================================================
        # 3. Bottleneck (带注意力)
        # =====================================================================
        self.bottleneck = nn.Sequential(
            ResidualBlock(cfg.bottleneck_channels, dropout=cfg.dropout),
            CBAM(cfg.bottleneck_channels) if cfg.use_attention else nn.Identity(),
            ResidualBlock(cfg.bottleneck_channels, dropout=cfg.dropout)
        )
        
        # =====================================================================
        # 4. Decoder (上采样路径)
        # =====================================================================
        self.decoder_blocks = nn.ModuleList()
        
        # 反转encoder通道以获取skip connection的通道数
        encoder_channels_reversed = list(reversed(cfg.encoder_channels[:-1])) + [cfg.stem_channels]
        
        in_ch = cfg.bottleneck_channels
        for i, (out_ch, skip_ch) in enumerate(zip(cfg.decoder_channels, encoder_channels_reversed)):
            self.decoder_blocks.append(
                UpsampleBlock(in_ch, skip_ch, out_ch, n_blocks=2, dropout=cfg.dropout)
            )
            in_ch = out_ch
        
        # =====================================================================
        # 5. 输出头
        # =====================================================================
        self.output_head = nn.Sequential(
            ConvBNReLU(cfg.decoder_channels[-1], cfg.decoder_channels[-1], 3, 1, 1),
            nn.Conv2d(cfg.decoder_channels[-1], cfg.out_channels, 1)
        )
        
        # =====================================================================
        # 6. 深度监督 (可选)
        # =====================================================================
        if cfg.deep_supervision:
            self.deep_heads = nn.ModuleList([
                nn.Conv2d(ch, cfg.out_channels, 1) 
                for ch in cfg.decoder_channels[:-1]
            ])
        else:
            self.deep_heads = None
        
        # 残差连接 (从背景场直接到输出)
        self.bkg_skip = nn.Conv2d(cfg.bkg_channels, cfg.out_channels, 1)
        
        self._init_weights()
        self._print_info()
    
    def _init_weights(self) -> None:
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _print_info(self) -> None:
        """打印模型信息"""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n{'='*70}")
        print(f"PhysicsAwareUNet 模型信息")
        print(f"{'='*70}")
        print(f"  Stem通道: {self.config.stem_channels}")
        print(f"  Encoder通道: {self.config.encoder_channels}")
        print(f"  Decoder通道: {self.config.decoder_channels}")
        print(f"  融合模式: {self.config.fusion_mode}")
        print(f"  使用注意力: {self.config.use_attention}")
        print(f"  深度监督: {self.config.deep_supervision}")
        print(f"  总参数量: {n_params:,}")
        print(f"  可训练参数: {n_trainable:,}")
        print(f"{'='*70}\n")
    
    def forward(
        self,
        obs: torch.Tensor,
        bkg: torch.Tensor,
        mask: torch.Tensor,
        aux: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        前向传播
        
        Args:
            obs: 观测场 [B, 17, H, W]
            bkg: 背景场 [B, 37, H, W]
            mask: 有效性掩码 [B, 1, H, W]
            aux: 辅助特征 [B, 4, H, W] (可选)
        
        Returns:
            pred: 预测场 [B, 37, H, W]
            deep_outputs: 深度监督输出列表 (如果启用)
        """
        # 1. Stem: 物理感知融合
        x = self.stem(obs, bkg, mask, aux)  # [B, 64, H, W]
        
        # 2. Encoder: 存储skip connections
        skips = [x]  # 第一个skip是stem输出
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skips.append(x)
        
        # 3. Bottleneck
        x = self.bottleneck(x)
        
        # 4. Decoder: 使用skip connections
        skips = skips[:-1]  # 最后一个不需要（就是bottleneck的输入）
        skips = list(reversed(skips))
        
        deep_outputs = []
        for i, (decoder_block, skip) in enumerate(zip(self.decoder_blocks, skips)):
            x = decoder_block(x, skip)
            
            # 深度监督
            if self.deep_heads is not None and i < len(self.deep_heads):
                deep_out = self.deep_heads[i](x)
                deep_out = F.interpolate(deep_out, size=bkg.shape[2:], mode='bilinear', align_corners=True)
                deep_outputs.append(deep_out)
        
        # 5. 输出头
        out = self.output_head(x)
        
        # 6. 残差连接 (从背景场)
        bkg_residual = self.bkg_skip(bkg)
        out = out + bkg_residual
        
        if self.config.deep_supervision and self.training:
            return out, deep_outputs
        return out
    
    def get_stem_attention(self) -> Optional[torch.Tensor]:
        """获取Stem的SE注意力权重"""
        return self.stem.get_se_attention()


# =============================================================================
# Part 5: 轻量级变体
# =============================================================================

# =============================================================================
# Part 5b: 消融辅助模块
# =============================================================================

class _SimpleStem(nn.Module):
    """V4消融: 无频谱感知的标准卷积茎干，直接拼接所有输入后做两层3×3卷积。"""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )

    def forward(
        self,
        obs: torch.Tensor,
        bkg: torch.Tensor,
        mask: torch.Tensor,
        aux: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        parts = [obs * mask, bkg]
        if aux is not None:
            parts.append(aux)
        return self.net(torch.cat(parts, dim=1))


class PhysicsAwareUNetLite(PhysicsAwareUNet):
    """轻量级版本 - 用于快速实验"""
    
    def __init__(self, config: Optional[UNetConfig] = None, **kwargs):
        # 使用传入的config或创建默认配置
        if config is not None:
            # 覆盖一些特定参数
            config.encoder_channels = [32, 64, 128, 256]
            config.encoder_depths = [1, 1, 1, 1]
            config.bottleneck_channels = 256
            config.decoder_channels = [128, 64, 32, 32]
            config.stem_channels = 32
            config.use_attention = False
        else:
            config = UNetConfig(
                encoder_channels=[32, 64, 128, 256],
                encoder_depths=[1, 1, 1, 1],
                bottleneck_channels=256,
                decoder_channels=[128, 64, 32, 32],
                stem_channels=32,
                use_attention=False,
                **kwargs
            )
        super().__init__(config)


class PhysicsAwareUNetLarge(PhysicsAwareUNet):
    """大型版本 - 用于最终实验"""
    
    def __init__(self, config: Optional[UNetConfig] = None, **kwargs):
        if config is not None:
            config.encoder_channels = [64, 128, 256, 512]
            config.encoder_depths = [3, 4, 6, 3]
            config.bottleneck_channels = 512
            config.decoder_channels = [256, 128, 64, 64]
            config.stem_channels = 64
            config.use_attention = True
            config.deep_supervision = True
        else:
            config = UNetConfig(
                encoder_channels=[64, 128, 256, 512],
                encoder_depths=[3, 4, 6, 3],
                bottleneck_channels=512,
                decoder_channels=[256, 128, 64, 64],
                stem_channels=64,
                use_attention=True,
                deep_supervision=True,
                **kwargs
            )
        super().__init__(config)


# =============================================================================
# Part 6: 基线模型 (用于消融实验)
# =============================================================================

class VanillaUNet(nn.Module):
    """
    原始U-Net (无物理感知模块)
    
    用于消融实验对比
    """
    
    def __init__(
        self,
        in_channels: int = 54,  # 17 + 37 = 观测 + 背景
        out_channels: int = 37,
        base_channels: int = 64
    ):
        super().__init__()
        
        # 简单拼接输入
        self.input_conv = ConvBNReLU(in_channels, base_channels, 3, 1, 1)
        
        # Encoder
        self.enc1 = self._make_layer(base_channels, base_channels * 2)
        self.enc2 = self._make_layer(base_channels * 2, base_channels * 4)
        self.enc3 = self._make_layer(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = self._make_layer(base_channels * 8, base_channels * 8)
        
        # Decoder
        self.dec3 = self._make_up_layer(base_channels * 8, base_channels * 4)
        self.dec2 = self._make_up_layer(base_channels * 4, base_channels * 2)
        self.dec1 = self._make_up_layer(base_channels * 2, base_channels)
        
        # Output
        self.output = nn.Conv2d(base_channels, out_channels, 1)
        
        print(f"[VanillaUNet] 参数量: {sum(p.numel() for p in self.parameters()):,}")
    
    def _make_layer(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNReLU(in_ch, out_ch, 3, 1, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1, 1)
        )
    
    def _make_up_layer(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNReLU(in_ch * 2, out_ch, 3, 1, 1),  # *2 for skip connection
            ConvBNReLU(out_ch, out_ch, 3, 1, 1)
        )
    
    def forward(
        self,
        obs: torch.Tensor,
        bkg: torch.Tensor,
        mask: torch.Tensor,
        aux: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """保持与PhysicsAwareUNet相同的接口"""
        # 简单拼接 (忽略mask和aux)
        x = torch.cat([obs * mask, bkg], dim=1)  # [B, 54, H, W]
        
        # Encoder
        x0 = self.input_conv(x)  # [B, 64, H, W]
        x1 = self.enc1(x0)       # [B, 128, H/2, W/2]
        x2 = self.enc2(x1)       # [B, 256, H/4, W/4]
        x3 = self.enc3(x2)       # [B, 512, H/8, W/8]
        
        # Bottleneck (无下采样)
        x = self.bottleneck(x3)  # [B, 512, H/16, W/16]
        
        # Decoder with skip connections
        # 先上采样再拼接
        x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)  # [B, 1024, H/8, W/8]
        x = self.dec3(x)               # [B, 256, H/4, W/4]
        
        x = torch.cat([x, x2], dim=1)  # [B, 512, H/4, W/4]
        x = self.dec2(x)               # [B, 128, H/2, W/2]
        
        x = torch.cat([x, x1], dim=1)  # [B, 256, H/2, W/2]
        x = self.dec1(x)               # [B, 64, H, W]
        
        return self.output(x)


# =============================================================================
# Part 7: 模型工厂
# =============================================================================

# =============================================================================
# Part 6b: FuXi-DA 架构迁移 (B4对比基线)
# =============================================================================
class FuXiDAUNet(nn.Module):
    """
    B4对比基线: FuXi-DA风格的数据同化网络 (层次化CNN版本)。

    参考: FuXi-DA (Xu et al., 2024; https://github.com/xuxiaoze/FuXi-DA)
    原始FuXi-DA使用Swin Transformer + Patch Merging实现层次化编码,
    本实现将其迁移为纯CNN架构(Conv + ResBlock + CBAM), 适配64×64 patch。

    架构对应关系:
    ┌────────────────────┬─────────────────────────────────────┐
    │   FuXi-DA 原始      │   本CNN迁移版本                      │
    ├────────────────────┼─────────────────────────────────────┤
    │ Patch Embedding    │ 1×1 Conv 独立 Embedding + 拼接融合    │
    │ Swin-T Block       │ ResidualBlock + SE Attention         │
    │ Patch Merging      │ Stride-2 ConvBNReLU (空间下采样)      │
    │ Linear Decoder     │ Bilinear Upsample + Skip + Conv      │
    │ Residual Output    │ 增量 + bkg_skip (相同)                │
    └────────────────────┴─────────────────────────────────────┘

    空间分辨率:  H×W  ──▶  H/2×W/2  ──▶  H/4×W/4  ──▶  H/2×W/2  ──▶  H×W
    通道数:       C0   ──▶    C1     ──▶    C2     ──▶    C1     ──▶   C0
                (96)       (192)        (384)        (192)        (96)

    修改要点 (相对原始全分辨率版本):
    1. 编码器: 两次 stride=2 卷积, 将 384通道卷积从 64×64 降至 16×16,
       FLOPs 降低 16×
    2. 解码器: 双线性上采样 + skip connection 恢复空间分辨率
    3. CBAM 注意力: 在 16×16 (256位置) 上计算, 而非 64×64 (4096位置),
       空间注意力矩阵从 4096² 降至 256², 内存降低 256×
    """

    def __init__(
        self,
        obs_channels: int = 17,
        bkg_channels: int = 37,
        aux_channels: int = 4,
        out_channels: int = 37,
        embed_dim: int = 96,
        depths: List[int] = None,
        num_heads: List[int] = None,   # 保留API兼容, CNN版本不使用
        window_size: int = 4,           # 保留API兼容, CNN版本不使用
        dropout: float = 0.1,
    ):
        super().__init__()
        depths = depths or [2, 1, 1]   # 3级编码器各阶段的 ResBlock 数量
        C0, C1, C2 = embed_dim, embed_dim * 2, embed_dim * 4  # 96, 192, 384

        # ==================== 1. 独立 Embedding ====================
        # 对应 FuXi-DA 的输入投影: obs 和 bkg 分别映射到 embed 空间
        self.obs_embed = nn.Sequential(
            nn.Conv2d(obs_channels, embed_dim, 1, bias=False),
            nn.GroupNorm(min(32, embed_dim), embed_dim),
            nn.GELU(),
        )
        self.bkg_embed = nn.Sequential(
            nn.Conv2d(bkg_channels, embed_dim, 1, bias=False),
            nn.GroupNorm(min(32, embed_dim), embed_dim),
            nn.GELU(),
        )
        self.aux_embed = nn.Sequential(
            nn.Conv2d(aux_channels, embed_dim // 4, 1, bias=False),
            nn.GroupNorm(min(8, embed_dim // 4), embed_dim // 4),
            nn.GELU(),
        ) if aux_channels > 0 else None

        # 融合输入通道 = obs_embed + bkg_embed + [aux_embed] + mask
        fusion_in = embed_dim * 2 + (embed_dim // 4 if aux_channels > 0 else 0) + 1

        # ==================== 2. 层次化编码器 ====================
        # 对应 FuXi-DA 的 Swin-T 多阶段 + Patch Merging

        # Stage 0: 全分辨率 H×W, C0 通道
        self.enc0 = nn.Sequential(
            ConvBNReLU(fusion_in, C0, 3, 1, 1),
            *[ResidualBlock(C0, dropout=dropout) for _ in range(depths[0])]
        )
        # Stage 1: H/2 × W/2, C1 通道 (Stride-2 ≈ Patch Merging)
        self.down1 = ConvBNReLU(C0, C1, 3, stride=2, padding=1)
        self.enc1 = nn.Sequential(
            *[ResidualBlock(C1, dropout=dropout) for _ in range(depths[1])]
        )
        # Stage 2 (瓶颈): H/4 × W/4, C2 通道 + 通道-空间注意力
        self.down2 = ConvBNReLU(C1, C2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            *[ResidualBlock(C2, dropout=dropout) for _ in range(depths[2])],
            CBAM(C2),      # 在 16×16 上计算注意力, 而非原来的 64×64
        )

        # ==================== 3. 层次化解码器 ====================
        # 对应 FuXi-DA 的 Linear Decoder, 但增加 Skip Connection

        # Up Stage 1: H/4 → H/2, 与 enc1 的 skip 拼接后融合
        self.dec_fuse1 = ConvBNReLU(C2 + C1, C1, 1, 1, 0)   # 1×1 融合
        self.dec1 = ResidualBlock(C1, dropout=dropout)

        # Up Stage 2: H/2 → H, 与 enc0 的 skip 拼接后融合
        self.dec_fuse2 = ConvBNReLU(C1 + C0, C0, 1, 1, 0)   # 1×1 融合
        self.dec2 = ResidualBlock(C0, dropout=dropout)

        # ==================== 4. 输出头 (增量映射) ====================
        self.output_head = nn.Sequential(
            ConvBNReLU(C0, C0, 3, 1, 1),
            nn.Conv2d(C0, out_channels, 1),
        )

        # ==================== 5. 背景残差连接 ====================
        self.bkg_skip = nn.Conv2d(bkg_channels, out_channels, 1)

        # ==================== 模型信息 ====================
        n_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}")
        print(f"FuXiDAUNet (B4对比基线 - 层次化CNN版本)")
        print(f"  embed_dim: {embed_dim}")
        print(f"  通道级数: {C0} → {C1} → {C2}")
        print(f"  分辨率级数: H×W → H/2×W/2 → H/4×W/4")
        print(f"  编码器深度(ResBlocks/stage): {depths}")
        print(f"  参数量: {n_params:,}")
        print(f"{'='*60}")

    def forward(
        self,
        obs: torch.Tensor,
        bkg: torch.Tensor,
        mask: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs:  观测场  [B, 17, H, W]
            bkg:  背景场  [B, 37, H, W]
            mask: 有效掩码 [B, 1,  H, W]
            aux:  辅助特征 [B, 4,  H, W] (可选)

        Returns:
            分析场 [B, 37, H, W] = 估计增量 + 背景投影
        """
        # ---- 1. 独立 Embedding ----
        obs_f = self.obs_embed(obs * mask)          # [B, C0, H, W]
        bkg_f = self.bkg_embed(bkg)                 # [B, C0, H, W]

        parts = [obs_f, bkg_f, mask]
        if aux is not None and self.aux_embed is not None:
            parts.insert(2, self.aux_embed(aux))    # [B, C0/4, H, W]
        x = torch.cat(parts, dim=1)                 # [B, fusion_in, H, W]

        # ---- 2. 层次化编码 (逐级空间下采样) ----
        s0 = self.enc0(x)                            # [B, C0,  H,    W   ]
        s1 = self.enc1(self.down1(s0))               # [B, C1,  H/2,  W/2 ]
        s2 = self.enc2(self.down2(s1))               # [B, C2,  H/4,  W/4 ]

        # ---- 3. 层次化解码 (逐级上采样 + Skip Connection) ----
        d1 = F.interpolate(s2, size=s1.shape[2:],
                           mode='bilinear', align_corners=True)
        d1 = self.dec1(self.dec_fuse1(
            torch.cat([d1, s1], dim=1)))             # [B, C1,  H/2,  W/2 ]

        d0 = F.interpolate(d1, size=s0.shape[2:],
                           mode='bilinear', align_corners=True)
        d0 = self.dec2(self.dec_fuse2(
            torch.cat([d0, s0], dim=1)))             # [B, C0,  H,    W   ]

        # ---- 4. 增量 + 背景残差 ----
        inc = self.output_head(d0)                   # [B, out_ch, H, W]
        return inc + self.bkg_skip(bkg)

def create_model(
    model_name: str = 'physics_unet',
    **kwargs
) -> nn.Module:
    """
    模型工厂函数
    
    Args:
        model_name: 模型名称
            - 'physics_unet': 标准物理感知U-Net
            - 'physics_unet_lite': 轻量级版本
            - 'physics_unet_large': 大型版本
            - 'vanilla_unet': 原始U-Net (消融对比)
    
    Returns:
        nn.Module
    """
    models = {
        'physics_unet': PhysicsAwareUNet,
        'pasnet': PhysicsAwareUNet,
        'physics_unet_lite': PhysicsAwareUNetLite,
        'physics_unet_large': PhysicsAwareUNetLarge,
        'vanilla_unet': VanillaUNet,
        'fuxi_da': FuXiDAUNet,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


# =============================================================================
# Part 8: 测试
# =============================================================================

def test_models():
    """测试所有模型"""
    print("=" * 70)
    print("测试模型架构")
    print("=" * 70)
    
    B, H, W = 2, 64, 64
    obs = torch.randn(B, 17, H, W)
    bkg = torch.randn(B, 37, H, W)
    mask = (torch.rand(B, 1, H, W) > 0.3).float()
    aux = torch.randn(B, 4, H, W)
    
    # 测试各个模型
    for model_name in ['physics_unet_lite', 'physics_unet', 'vanilla_unet']:
        print(f"\n--- 测试: {model_name} ---")
        
        if model_name == 'vanilla_unet':
            model = create_model(model_name)
        else:
            model = create_model(model_name)
        
        # 前向传播
        out = model(obs, bkg, mask, aux)
        if isinstance(out, tuple):
            out, deep = out
            print(f"  深度监督输出: {len(deep)} 个")
        
        print(f"  输出形状: {out.shape}")
        
        # 反向传播测试
        loss = out.mean()
        loss.backward()
        print(f"  ✓ 梯度反向传播成功")
    
    print("\n" + "=" * 70)
    print("所有模型测试通过!")
    print("=" * 70)


if __name__ == "__main__":
    test_models()


# =============================================================================
# Part 9: 额外对比基线 (B5 AttentionUNet / B6 PixelMLP / B7 ResUNet)
# =============================================================================

class _AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g,  F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_l,  F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1,    1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class AttentionUNet(nn.Module):
    """B5: Attention U-Net (Oktay et al. 2018).
    Encoder/Decoder 与 VanillaUNet 完全相同，skip 处加 Attention Gate。
    接口: forward(obs, bkg, mask, aux=None) → (B, 37, H, W)
    """
    def __init__(self, in_channels: int = 54, out_channels: int = 37, base_channels: int = 64):
        super().__init__()
        C = base_channels
        self.input_conv = ConvBNReLU(in_channels, C, 3, 1, 1)
        def _down(cin, cout):
            return nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(cin, cout, 3, 1, 1), ConvBNReLU(cout, cout, 3, 1, 1))
        self.enc1 = _down(C,   C*2);  self.enc2 = _down(C*2, C*4);  self.enc3 = _down(C*4, C*8)
        self.bottleneck = nn.Sequential(ConvBNReLU(C*8, C*8, 3, 1, 1), ConvBNReLU(C*8, C*8, 3, 1, 1))
        self.ag3 = _AttentionGate(C*8, C*8, C*4)
        self.ag2 = _AttentionGate(C*4, C*4, C*2)
        self.ag1 = _AttentionGate(C*2, C*2, C)
        def _up(cin, cout):
            return nn.Sequential(ConvBNReLU(cin*2, cin, 3, 1, 1), ConvBNReLU(cin, cout, 3, 1, 1))
        self.dec3 = _up(C*8, C*4);  self.dec2 = _up(C*4, C*2);  self.dec1 = _up(C*2, C)
        self.output = nn.Conv2d(C, out_channels, 1)
        print(f"[AttentionUNet] 参数量: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, obs, bkg, mask, aux=None):
        x = torch.cat([obs * mask, bkg], dim=1)
        x0 = self.input_conv(x);  x1 = self.enc1(x0);  x2 = self.enc2(x1);  x3 = self.enc3(x2)
        xb = self.bottleneck(x3)
        def up(feat, skip, ag, dec):
            f = F.interpolate(feat, size=skip.shape[2:], mode='bilinear', align_corners=True)
            return dec(torch.cat([f, ag(f, skip)], dim=1))
        x = up(xb, x3, self.ag3, self.dec3)
        x = up(x,  x2, self.ag2, self.dec2)
        x = up(x,  x1, self.ag1, self.dec1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.output(x)


class PixelMLP(nn.Module):
    """B6: Pixel-wise MLP — 无空间交互的最简基线"""
    def __init__(self, in_channels: int = 17 + 37 + 1, out_channels: int = 37, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),      nn.GELU(),
            nn.Linear(hidden, out_channels),
        )
        print(f"[PixelMLP] 参数量: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, obs, bkg, mask, aux=None):
        x = torch.cat([obs * mask, bkg, mask], dim=1)
        return self.net(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _ResBlock2(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
                                  nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.net(x) + x)


class ResUNet(nn.Module):
    """B7: Residual U-Net"""
    def __init__(self, in_channels: int = 54, out_channels: int = 37, base_channels: int = 64):
        super().__init__()
        C = base_channels
        self.input_conv = ConvBNReLU(in_channels, C, 3, 1, 1)
        def _down(cin, cout):
            return nn.Sequential(nn.MaxPool2d(2), _ResBlock2(cin), ConvBNReLU(cin, cout, 3, 1, 1))
        self.enc1 = _down(C,   C*2);  self.enc2 = _down(C*2, C*4);  self.enc3 = _down(C*4, C*8)
        self.bottleneck = nn.Sequential(_ResBlock2(C*8), _ResBlock2(C*8))
        def _up(cin, cout):
            return nn.Sequential(ConvBNReLU(cin*2, cin, 3, 1, 1), _ResBlock2(cin), ConvBNReLU(cin, cout, 3, 1, 1))
        self.dec3 = _up(C*8, C*4);  self.dec2 = _up(C*4, C*2);  self.dec1 = _up(C*2, C)
        self.output = nn.Conv2d(C, out_channels, 1)
        print(f"[ResUNet] 参数量: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, obs, bkg, mask, aux=None):
        x = torch.cat([obs * mask, bkg], dim=1)
        x0 = self.input_conv(x);  x1 = self.enc1(x0);  x2 = self.enc2(x1);  x3 = self.enc3(x2)
        xb = self.bottleneck(x3)
        def up(feat, skip, dec):
            f = F.interpolate(feat, size=skip.shape[2:], mode='bilinear', align_corners=True)
            return dec(torch.cat([f, skip], dim=1))
        x = up(xb, x3, self.dec3);  x = up(x, x2, self.dec2);  x = up(x, x1, self.dec1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.output(x)


# 注册新模型到 create_model
_EXTRA_MODELS = {'attn_unet': AttentionUNet, 'pixel_mlp': PixelMLP, 'res_unet': ResUNet}

# monkey-patch create_model 加入新模型 (避免修改原函数)
_orig_create_model = create_model
def create_model(model_name: str = 'physics_unet', **kwargs) -> 'nn.Module':
    if model_name in _EXTRA_MODELS:
        kwargs.pop('config', None)  # extra models do not accept UNetConfig
        return _EXTRA_MODELS[model_name](**kwargs)
    return _orig_create_model(model_name, **kwargs)
