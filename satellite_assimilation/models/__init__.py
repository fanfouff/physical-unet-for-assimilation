"""
卫星数据同化模型模块
"""

from .backbone import (
    # 配置
    UNetConfig,
    
    # 模型
    PhysicsAwareUNet,
    PhysicsAwareUNetLite,
    PhysicsAwareUNetLarge,
    VanillaUNet,
    
    # 工厂函数
    create_model,
    
    # 基础模块
    ConvBNReLU,
    ResidualBlock,
    DownsampleBlock,
    UpsampleBlock,
    SEBlock,
    CBAM,
    SpatialAttention,
    ChannelAttention,
)

__all__ = [
    'UNetConfig',
    'PhysicsAwareUNet',
    'PhysicsAwareUNetLite', 
    'PhysicsAwareUNetLarge',
    'VanillaUNet',
    'create_model',
    'ConvBNReLU',
    'ResidualBlock',
    'DownsampleBlock',
    'UpsampleBlock',
    'SEBlock',
    'CBAM',
    'SpatialAttention',
    'ChannelAttention',
]
