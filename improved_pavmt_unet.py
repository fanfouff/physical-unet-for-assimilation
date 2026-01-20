"""
改进版 PAVMT-Unet: 用于卫星数据同化的物理感知混合网络
=======================================================

改进要点:
1. SpectralAdapter: 卫星亮温到状态空间的映射
2. Mask-Aware VCA: 掩膜感知的垂直通道注意力
3. AtmosphericPhysicLoss: 大气物理约束损失

架构:
    卫星亮温 (B, 13, H, W)  +  背景场 (B, 37, H, W)
                    │                    │
                    └──────┬─────────────┘
                           ↓
                    [SpectralAdapter]
                           ↓
                    融合特征 (B, 37, H, W)
                           ↓
                    [VRS Stem] - 垂直重建
                           ↓
              ┌────────────┴────────────┐
              ↓                         ↓
         [Encoder]                 [Decoder]
          ├─ConvBlock──────VCA─────ConvBlock
          ├─ConvBlock──────VCA─────ConvBlock  
          ├─PA-Mamba+MSA───VCA───PA-Mamba+MSA
          └─PA-Mamba+MSA───VCA───PA-Mamba+MSA
                    ↓
                 [CVSC]
                    ↓
              分析场 (B, 37, H, W)

作者: 基于PAVMT-Unet改进
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math

# 尝试导入mamba_ssm (如果可用)
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available, using fallback implementation")


# ============================================
# 基础模块
# ============================================

class ConvBlock(nn.Module):
    """残差卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 残差连接
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()
        
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.residual(x))


class VerticalReconstructionStem(nn.Module):
    """
    垂直重建主干 (VRS)
    
    用于编码输入数据的垂直结构信息
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class MambaBlockFallback(nn.Module):
    """
    Mamba块的后备实现 (当mamba_ssm不可用时)
    
    使用1D卷积模拟序列建模
    """
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        
        d_inner = d_model * expand
        
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        
        # 用1D深度可分离卷积模拟SSM
        self.conv = nn.Conv1d(
            d_inner, d_inner, kernel_size=4, padding=2, 
            groups=d_inner, bias=False
        )
        
        self.out_proj = nn.Linear(d_inner, d_model)
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        x = self.norm(x)
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Conv1D
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv(x)[:, :, :x.shape[2]]
        x = x.transpose(1, 2)  # (B, L, D)
        
        x = self.act(x) * F.silu(z)
        x = self.out_proj(x)
        
        return x + residual


class PAMambaMixer(nn.Module):
    """
    Physics-Aware Mamba Mixer
    
    三路并行架构:
    1. 静态扩散路径 (DW-Conv)
    2. 前向平流路径 (Forward SSM)
    3. 后向平流路径 (Backward SSM)
    """
    
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        
        self.dim = dim
        
        # 输入投影
        self.in_proj = nn.Linear(dim, dim * 3)
        
        # 静态扩散路径 - 深度可分离卷积
        self.diffusion_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # 动态平流路径 - 前向和后向
        if MAMBA_AVAILABLE:
            self.forward_ssm = Mamba(d_model=dim, d_state=d_state)
            self.backward_ssm = Mamba(d_model=dim, d_state=d_state)
        else:
            self.forward_ssm = MambaBlockFallback(dim, d_state)
            self.backward_ssm = MambaBlockFallback(dim, d_state)
        
        # 空间增强
        self.forward_spatial = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.backward_spatial = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        
        # 输出投影
        self.out_proj = nn.Linear(dim * 3, dim)
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 1. 扩散路径
        diffusion_out = self.diffusion_conv(x)  # (B, C, H, W)
        
        # 2. 准备序列输入
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_norm = self.norm(x_flat)
        
        # 3. 前向平流路径
        x_2d = x_norm.transpose(1, 2).view(B, C, H, W)
        forward_spatial = self.forward_spatial(x_2d)
        forward_seq = forward_spatial.flatten(2).transpose(1, 2)
        forward_out = self.forward_ssm(forward_seq)
        forward_out = forward_out.transpose(1, 2).view(B, C, H, W)
        
        # 4. 后向平流路径
        backward_spatial = self.backward_spatial(x_2d)
        backward_seq = backward_spatial.flatten(2).transpose(1, 2)
        backward_seq = torch.flip(backward_seq, dims=[1])  # 反转序列
        backward_out = self.backward_ssm(backward_seq)
        backward_out = torch.flip(backward_out, dims=[1])
        backward_out = backward_out.transpose(1, 2).view(B, C, H, W)
        
        # 5. 融合三路输出
        combined = torch.cat([diffusion_out, forward_out, backward_out], dim=1)
        combined = combined.flatten(2).transpose(1, 2)  # (B, H*W, 3C)
        out = self.out_proj(combined)  # (B, H*W, C)
        out = out.transpose(1, 2).view(B, C, H, W)
        
        return out + x  # 残差连接


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        
        # 展平空间维度
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_norm = self.norm(x_flat)
        
        # 计算QKV
        qkv = self.qkv(x_norm).reshape(B, H*W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 输出
        out = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        out = self.proj(out)
        out = out.transpose(1, 2).view(B, C, H, W)
        
        return out + x


class CVSC(nn.Module):
    """
    Cross-scale Vertical-Spatial Coupler
    
    瓶颈处的跨尺度垂直-空间耦合器
    """
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        
        # Channel shuffle促进垂直交互
        self.channel_shuffle = True
        
        # 卷积路径
        self.conv_path = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        
        # PA-Mamba路径
        self.mamba_path = PAMambaMixer(dim)
        
        # Self-Attention路径
        self.attention_path = MultiHeadSelfAttention(dim, num_heads)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        
        self.norm = nn.LayerNorm(dim)
        self.fusion = nn.Conv2d(dim, dim, 1)
        
    def channel_shuffle_op(self, x: torch.Tensor, groups: int = 4) -> torch.Tensor:
        """通道混洗操作"""
        B, C, H, W = x.shape
        x = x.view(B, groups, C // groups, H, W)
        x = x.transpose(1, 2).contiguous()
        return x.view(B, C, H, W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Channel shuffle
        if self.channel_shuffle:
            x = self.channel_shuffle_op(x)
        
        # 卷积残差
        conv_out = self.conv_path(x) + x
        
        # PA-Mamba
        mamba_out = self.mamba_path(conv_out)
        
        # Self-Attention
        attn_out = self.attention_path(mamba_out)
        
        # FFN
        out_flat = attn_out.flatten(2).transpose(1, 2)
        out_flat = self.norm(out_flat)
        out_flat = self.ffn(out_flat) + out_flat
        out = out_flat.transpose(1, 2).view(B, C, H, W)
        
        # 融合
        out = self.fusion(out) + x
        
        return out


# ============================================
# SpectralAdapter (从modules导入)
# ============================================

class SpectralAdapter(nn.Module):
    """光谱适配器 - 卫星亮温到状态空间映射"""
    
    def __init__(
        self,
        n_sat_channels: int = 13,
        n_background_layers: int = 37,
        hidden_dim: int = 64,
        init_sat_weight: float = 0.3
    ):
        super().__init__()
        
        self.spectral_transform = nn.Sequential(
            nn.Conv2d(n_sat_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, n_background_layers, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_background_layers),
            nn.GELU()
        )
        
        self.fusion_weight = nn.Parameter(
            torch.ones(1, n_background_layers, 1, 1) * init_sat_weight
        )
        
    def forward(self, satellite: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        sat_transformed = self.spectral_transform(satellite)
        alpha = torch.sigmoid(self.fusion_weight)
        fused = alpha * sat_transformed + (1 - alpha) * background
        return fused


# ============================================
# MaskAwareVCA (从modules导入)
# ============================================

class MaskAwareVCA(nn.Module):
    """掩膜感知的垂直通道注意力"""
    
    def __init__(self, in_channels: int, mask_influence: float = 0.5):
        super().__init__()
        
        self.encoder_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        
        self.decoder_gate = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        self.base_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.mask_modulator = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.mask_scale = nn.Parameter(torch.tensor(mask_influence))
        
    def forward(
        self, 
        x_encoder: torch.Tensor, 
        g_decoder: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, C, H, W = x_encoder.shape
        
        x_enhanced = self.encoder_enhance(x_encoder)
        g_processed = self.decoder_gate(g_decoder)
        
        if g_processed.shape[2:] != (H, W):
            g_processed = F.interpolate(g_processed, size=(H, W), mode='bilinear', align_corners=False)
        
        combined = x_enhanced + g_processed
        base_alpha = self.base_attention(combined)
        
        if mask is not None:
            mask_resized = F.interpolate(mask.float(), size=(H, W), mode='nearest')
            mask_input = torch.cat([combined, mask_resized], dim=1)
            mask_modulation = self.mask_modulator(mask_input)
            mask_effect = torch.sigmoid(self.mask_scale) * mask_resized
            alpha = base_alpha * (1 - mask_effect) + mask_modulation * mask_effect
        else:
            alpha = base_alpha
        
        return x_encoder * alpha + g_processed * (1 - alpha)


# ============================================
# 完整的改进版PAVMT-Unet
# ============================================

class ImprovedPAVMTUnet(nn.Module):
    """
    改进版 PAVMT-Unet
    
    用于卫星-再分析数据同化任务
    
    主要改进:
    1. SpectralAdapter: 处理异构输入(卫星亮温 + 背景场)
    2. MaskAwareVCA: 掩膜感知的注意力机制
    3. 支持不同分辨率的输入输出
    
    参数:
        n_sat_channels: 卫星通道数 (默认13, FY-3F MWTS)
        n_background_layers: 背景场垂直层数 (默认37, ERA5)
        base_channels: 基础通道数
        num_stages: 编码器/解码器阶段数
    """
    
    def __init__(
        self,
        n_sat_channels: int = 13,
        n_background_layers: int = 37,
        base_channels: int = 64,
        num_stages: int = 4,
        num_heads: int = 8,
        use_mask: bool = True
    ):
        super().__init__()
        
        self.n_sat_channels = n_sat_channels
        self.n_background_layers = n_background_layers
        self.use_mask = use_mask
        
        # ============================================
        # SpectralAdapter: 输入融合
        # ============================================
        self.spectral_adapter = SpectralAdapter(
            n_sat_channels=n_sat_channels,
            n_background_layers=n_background_layers,
            hidden_dim=base_channels
        )
        
        # ============================================
        # VRS Stem: 垂直重建
        # ============================================
        self.vrs_stem = VerticalReconstructionStem(
            in_channels=n_background_layers,
            out_channels=base_channels
        )
        
        # ============================================
        # Encoder
        # ============================================
        encoder_channels = [base_channels * (2 ** i) for i in range(num_stages)]
        
        self.encoder_stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        for i in range(num_stages):
            in_ch = encoder_channels[i-1] if i > 0 else base_channels
            out_ch = encoder_channels[i]
            
            if i < 2:
                # 前两阶段: ConvBlock
                stage = nn.Sequential(
                    ConvBlock(in_ch if i == 0 else in_ch, out_ch),
                    ConvBlock(out_ch, out_ch)
                )
            else:
                # 后两阶段: PA-Mamba + MSA
                stage = nn.Sequential(
                    PAMambaMixer(in_ch if i == 0 else out_ch),
                    MultiHeadSelfAttention(out_ch, num_heads)
                )
            
            self.encoder_stages.append(stage)
            
            if i < num_stages - 1:
                self.downsample.append(
                    nn.Conv2d(out_ch, encoder_channels[i+1], 3, stride=2, padding=1)
                )
        
        # ============================================
        # CVSC Bottleneck
        # ============================================
        self.cvsc = CVSC(encoder_channels[-1], num_heads)
        
        # ============================================
        # Decoder
        # ============================================
        decoder_channels = encoder_channels[::-1]
        
        self.decoder_stages = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.vca_modules = nn.ModuleList()
        
        for i in range(num_stages):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i+1] if i < num_stages - 1 else decoder_channels[i]
            
            if i < 2:
                # 前两阶段: PA-Mamba + MSA (对应encoder后两阶段)
                stage = nn.Sequential(
                    PAMambaMixer(in_ch),
                    MultiHeadSelfAttention(in_ch, num_heads)
                )
            else:
                # 后两阶段: ConvBlock (对应encoder前两阶段)
                stage = nn.Sequential(
                    ConvBlock(in_ch, out_ch),
                    ConvBlock(out_ch, out_ch)
                )
            
            self.decoder_stages.append(stage)
            
            # VCA模块
            self.vca_modules.append(MaskAwareVCA(in_ch))
            
            if i < num_stages - 1:
                self.upsample.append(
                    nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
                )
        
        # ============================================
        # Output Head
        # ============================================
        self.output_head = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=4),
            nn.BatchNorm2d(base_channels // 2),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, n_background_layers, 3, padding=1),
        )
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        satellite: torch.Tensor,
        background: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            satellite: 卫星亮温 (B, n_sat_channels, H, W)
            background: ERA5背景场 (B, n_background_layers, H, W)
            mask: 观测掩膜 (B, 1, H, W), 可选
            
        返回:
            analysis: 分析场 (B, n_background_layers, H, W)
        """
        # 1. SpectralAdapter: 融合卫星和背景场
        fused = self.spectral_adapter(satellite, background)
        
        # 2. VRS Stem
        x = self.vrs_stem(fused)
        
        # 3. Encoder (保存skip connections)
        encoder_features = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            encoder_features.append(x)
            if i < len(self.downsample):
                x = self.downsample[i](x)
        
        # 4. CVSC Bottleneck
        x = self.cvsc(x)
        
        # 5. Decoder (使用VCA处理skip connections)
        for i, (stage, vca) in enumerate(zip(self.decoder_stages, self.vca_modules)):
            # 获取对应的encoder特征 (逆序)
            skip_idx = len(encoder_features) - 1 - i
            skip_feat = encoder_features[skip_idx]
            
            # VCA融合 (如果使用mask)
            if self.use_mask:
                calibrated = vca(skip_feat, x, mask)
            else:
                calibrated = vca(skip_feat, x, None)
            
            # Decoder stage
            x = stage(calibrated)
            
            # 上采样
            if i < len(self.upsample):
                x = self.upsample[i](x)
        
        # 6. Output Head
        analysis = self.output_head(x)
        
        return analysis


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 Improved PAVMT-Unet")
    print("=" * 60)
    
    # 配置
    batch_size = 2
    n_sat = 13
    n_bg = 37
    H, W = 128, 128
    
    # 创建模型
    model = ImprovedPAVMTUnet(
        n_sat_channels=n_sat,
        n_background_layers=n_bg,
        base_channels=64,
        num_stages=4
    )
    
    # 创建输入
    satellite = torch.randn(batch_size, n_sat, H, W)
    background = torch.randn(batch_size, n_bg, H, W)
    mask = torch.randint(0, 2, (batch_size, 1, H, W)).float()
    
    print(f"\n输入形状:")
    print(f"  satellite: {satellite.shape}")
    print(f"  background: {background.shape}")
    print(f"  mask: {mask.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(satellite, background, mask)
    
    print(f"\n输出形状: {output.shape}")
    
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # 测试训练模式
    print("\n测试训练模式...")
    model.train()
    output = model(satellite, background, mask)
    
    # 模拟反向传播
    loss = output.mean()
    loss.backward()
    print("反向传播成功!")
    
    # 打印各模块参数
    print("\n各模块参数量:")
    module_params = {
        'SpectralAdapter': sum(p.numel() for p in model.spectral_adapter.parameters()),
        'VRS Stem': sum(p.numel() for p in model.vrs_stem.parameters()),
        'Encoder': sum(p.numel() for p in model.encoder_stages.parameters()),
        'CVSC': sum(p.numel() for p in model.cvsc.parameters()),
        'Decoder': sum(p.numel() for p in model.decoder_stages.parameters()),
        'VCA': sum(p.numel() for p in model.vca_modules.parameters()),
        'Output Head': sum(p.numel() for p in model.output_head.parameters())
    }
    
    for name, params in module_params.items():
        print(f"  {name}: {params:,} ({params/total_params*100:.1f}%)")
