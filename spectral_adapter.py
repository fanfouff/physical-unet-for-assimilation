"""
SpectralAdapter 模块
====================
用于将卫星辐射空间(亮温)映射到状态空间(温度廓线),
并与ERA5背景场进行自适应加权融合。

作者: 基于PAVMT-Unet改进
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAdapter(nn.Module):
    """
    光谱适配器模块
    
    将卫星亮温数据从辐射空间映射到状态空间，并与背景场进行加权融合。
    
    输入:
        satellite: (Batch, n_sat_channels, H, W) - 卫星亮温数据
        background: (Batch, n_background_layers, H, W) - ERA5背景场
        
    输出:
        fused: (Batch, n_background_layers, H, W) - 融合后的特征
    """
    
    def __init__(
        self,
        n_sat_channels: int = 13,
        n_background_layers: int = 37,
        hidden_dim: int = 64,
        init_sat_weight: float = 0.3,
        learnable_weight: bool = True
    ):
        """
        参数:
            n_sat_channels: 卫星通道数 (FY-3F MWTS为13通道)
            n_background_layers: 背景场垂直层数 (ERA5为37层)
            hidden_dim: 中间隐藏层维度
            init_sat_weight: 卫星数据的初始融合权重
            learnable_weight: 是否使用可学习的融合权重
        """
        super().__init__()
        
        self.n_sat_channels = n_sat_channels
        self.n_background_layers = n_background_layers
        
        # ============================================
        # 辐射空间 -> 状态空间 映射网络
        # 使用1x1卷积保持空间分辨率，仅做通道变换
        # ============================================
        self.spectral_transform = nn.Sequential(
            # 第一层: 升维到隐藏空间
            nn.Conv2d(n_sat_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            
            # 第二层: 映射到状态空间维度
            nn.Conv2d(hidden_dim, n_background_layers, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_background_layers),
            nn.GELU(),
        )
        
        # ============================================
        # 可学习的融合权重
        # ============================================
        if learnable_weight:
            # 使用逐通道的可学习权重,允许不同高度层有不同的融合比例
            # 初始化: 低层更依赖卫星观测,高层更依赖背景场
            self.fusion_weight = nn.Parameter(
                torch.ones(1, n_background_layers, 1, 1) * init_sat_weight
            )
            # 可选: 空间自适应权重
            self.spatial_weight_net = nn.Sequential(
                nn.Conv2d(n_background_layers * 2, n_background_layers, 
                         kernel_size=3, padding=1, groups=n_background_layers),
                nn.Sigmoid()
            )
            self.use_spatial_weight = True
        else:
            self.register_buffer(
                'fusion_weight', 
                torch.ones(1, n_background_layers, 1, 1) * init_sat_weight
            )
            self.use_spatial_weight = False
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        satellite: torch.Tensor, 
        background: torch.Tensor,
        return_weights: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            satellite: (B, n_sat_channels, H, W) 卫星亮温
            background: (B, n_background_layers, H, W) 背景场
            return_weights: 是否返回融合权重(用于可视化)
            
        返回:
            fused: (B, n_background_layers, H, W) 融合特征
            weights: (可选) 融合权重
        """
        B, C_sat, H, W = satellite.shape
        
        # 1. 将卫星数据从辐射空间映射到状态空间
        sat_transformed = self.spectral_transform(satellite)  # (B, n_bg, H, W)
        
        # 2. 计算融合权重
        if self.use_spatial_weight and self.training:
            # 空间自适应权重: 根据卫星和背景场的差异动态调整
            combined = torch.cat([sat_transformed, background], dim=1)
            spatial_weight = self.spatial_weight_net(combined)  # (B, n_bg, H, W)
            # 结合通道权重和空间权重
            alpha = torch.sigmoid(self.fusion_weight) * spatial_weight
        else:
            # 仅使用通道权重
            alpha = torch.sigmoid(self.fusion_weight)
        
        # 3. 加权融合
        # fused = alpha * sat_transformed + (1 - alpha) * background
        fused = alpha * sat_transformed + (1 - alpha) * background
        
        if return_weights:
            return fused, alpha
        return fused


class SpectralAdapterV2(nn.Module):
    """
    SpectralAdapter V2 - 增强版
    
    增加了:
    1. 残差连接
    2. 通道注意力机制
    3. 多尺度特征提取
    """
    
    def __init__(
        self,
        n_sat_channels: int = 13,
        n_background_layers: int = 37,
        hidden_dim: int = 64,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.n_sat_channels = n_sat_channels
        self.n_background_layers = n_background_layers
        
        # 多尺度卫星特征提取
        self.sat_branch = nn.ModuleList([
            # 1x1 卷积 - 点级特征
            nn.Sequential(
                nn.Conv2d(n_sat_channels, hidden_dim // 2, 1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.GELU()
            ),
            # 3x3 卷积 - 局部特征
            nn.Sequential(
                nn.Conv2d(n_sat_channels, hidden_dim // 2, 3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.GELU()
            )
        ])
        
        # 融合多尺度特征
        self.sat_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, n_background_layers, 1),
            nn.BatchNorm2d(n_background_layers)
        )
        
        # 通道注意力 - 学习哪些卫星通道对哪些大气层更重要
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_background_layers, n_background_layers // 4),
            nn.ReLU(),
            nn.Linear(n_background_layers // 4, n_background_layers),
            nn.Sigmoid()
        )
        
        # 可学习的融合门控
        self.gate = nn.Sequential(
            nn.Conv2d(n_background_layers * 2, n_background_layers, 1),
            nn.Sigmoid()
        )
        
    def forward(self, satellite: torch.Tensor, background: torch.Tensor):
        B = satellite.shape[0]
        
        # 多尺度卫星特征
        sat_features = []
        for branch in self.sat_branch:
            sat_features.append(branch(satellite))
        sat_multi = torch.cat(sat_features, dim=1)
        
        # 映射到状态空间
        sat_state = self.sat_fusion(sat_multi)
        
        # 通道注意力加权
        ca_weight = self.channel_attention(sat_state).view(B, -1, 1, 1)
        sat_state = sat_state * ca_weight
        
        # 自适应门控融合
        combined = torch.cat([sat_state, background], dim=1)
        gate = self.gate(combined)
        
        # 融合输出
        fused = gate * sat_state + (1 - gate) * background
        
        return fused


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    # 测试 SpectralAdapter
    batch_size = 4
    n_sat = 13
    n_bg = 37
    H, W = 64, 64
    
    adapter = SpectralAdapter(n_sat_channels=n_sat, n_background_layers=n_bg)
    
    sat_data = torch.randn(batch_size, n_sat, H, W)
    bg_data = torch.randn(batch_size, n_bg, H, W)
    
    output, weights = adapter(sat_data, bg_data, return_weights=True)
    
    print(f"输入 - 卫星数据: {sat_data.shape}")
    print(f"输入 - 背景场: {bg_data.shape}")
    print(f"输出 - 融合特征: {output.shape}")
    print(f"融合权重: {weights.shape}")
    print(f"权重范围: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # 测试 V2 版本
    adapter_v2 = SpectralAdapterV2(n_sat_channels=n_sat, n_background_layers=n_bg)
    output_v2 = adapter_v2(sat_data, bg_data)
    print(f"\nV2 输出: {output_v2.shape}")
    
    # 参数量统计
    params_v1 = sum(p.numel() for p in adapter.parameters())
    params_v2 = sum(p.numel() for p in adapter_v2.parameters())
    print(f"\nV1 参数量: {params_v1:,}")
    print(f"V2 参数量: {params_v2:,}")
