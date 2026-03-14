"""
===============================================================================
卫星数据同化深度学习框架 V2.0 - 顶会/顶刊标准版
Satellite Data Assimilation Deep Learning Framework
===============================================================================

改进内容 (基于审稿意见):
1. 懒加载机制 (Lazy Loading) - 支持TB级数据
2. 辅助地理/时间信息嵌入 (Auxiliary Inputs) - 纬度/经度/太阳天顶角
3. 掩码感知卷积 (Mask-Aware Convolution) - 显式传递缺测信息
4. 非线性融合策略 (Non-linear Fusion) - Concat+Conv替代线性加法
5. 消融实验评估指标 (Ablation Metrics) - 分层RMSE/通道显著性/缺测鲁棒性

===============================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, Union, List, Callable
import numpy as np
from pathlib import Path
import warnings
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json


# =============================================================================
# Part 1: 配置类 (Configuration)
# =============================================================================

@dataclass
class DataConfig:
    """数据配置"""
    n_obs_channels: int = 17          # FY-3F MWTS通道数
    n_bkg_levels: int = 37            # ERA5气压层数
    n_aux_features: int = 4           # 辅助特征数 (lat, lon, sza, land_mask)
    height: int = 64
    width: int = 64
    dtype: torch.dtype = torch.float32
    
    # 标准化参数文件路径
    stats_file: Optional[str] = None
    

@dataclass
class ModelConfig:
    """模型配置"""
    latent_channels: int = 64         # 融合后潜在维度
    se_reduction: int = 8             # SE模块压缩比
    fusion_mode: str = 'concat'       # 'concat', 'add', 'gated'
    use_aux_features: bool = True     # 是否使用辅助特征
    mask_aware: bool = True           # 是否使用掩码感知卷积
    dropout: float = 0.1


# =============================================================================
# Part 2: 物理感知标准化器 (Physics-Aware Normalizer) - 增强版
# =============================================================================
class LevelwiseNormalizer:
    """
    逐层/逐通道 Z-Score 标准化器 (修复版)
    
    修复: 4D 张量 (N,C,H,W) 的广播问题
    """

    def __init__(
        self,
        mean=None,
        std=None,
        eps: float = 1e-8,
        name: str = "normalizer",
    ):
        self.mean = mean
        self.std = std
        self.eps = eps
        self.name = name
        self._fitted = mean is not None and std is not None
        self._count = None
        self._M2 = None

    def fit(self, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        n_channels = data.shape[-3]
        data_flat = data.transpose(0, 2, 3, 1).reshape(-1, n_channels)
        self.mean = np.nanmean(data_flat, axis=0)
        self.std = np.nanstd(data_flat, axis=0)
        self.std = np.where(self.std < self.eps, self.eps, self.std)
        self._fitted = True
        return self

    def partial_fit(self, data):
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        n_channels = data.shape[-3]
        if self._count is None:
            self._count = np.zeros(n_channels)
            self.mean = np.zeros(n_channels)
            self._M2 = np.zeros(n_channels)
        data_flat = data.transpose(0, 2, 3, 1).reshape(-1, n_channels)
        for i in range(n_channels):
            valid_values = data_flat[:, i][~np.isnan(data_flat[:, i])]
            for x in valid_values:
                self._count[i] += 1
                delta = x - self.mean[i]
                self.mean[i] += delta / self._count[i]
                delta2 = x - self.mean[i]
                self._M2[i] += delta * delta2
        self.std = np.sqrt(self._M2 / np.maximum(self._count - 1, 1))
        self.std = np.where(self.std < self.eps, self.eps, self.std)
        self._fitted = True
        return self

    # ================================================================
    # 核心修复: _reshape_stats
    # ================================================================
    @staticmethod
    def _reshape_stats(stat, x_ndim, is_tensor=False):
        """
        将 (C,) 的统计量 reshape 为可与 x 广播的形状:
          x_ndim=4  (N,C,H,W) → (1, C, 1, 1)
          x_ndim=3  (C,H,W)   → (C, 1, 1)
          x_ndim=2  (N,C)     → (1, C)
          x_ndim=1  (C,)      → (C,)
        """
        if x_ndim == 4:
            shape = (1, -1, 1, 1)
        elif x_ndim == 3:
            shape = (-1, 1, 1)
        elif x_ndim == 2:
            shape = (1, -1)
        else:
            shape = (-1,)

        return stat.reshape(shape)

    # ================================================================
    # 修复后的 transform
    # ================================================================
    def transform(self, x, inplace=False):
        if not self._fitted:
            raise RuntimeError(f"[{self.name}] Normalizer not fitted!")

        is_tensor = isinstance(x, torch.Tensor)
        device = x.device if is_tensor else None

        if is_tensor:
            mean = torch.tensor(self.mean, dtype=x.dtype, device=device)
            std  = torch.tensor(self.std,  dtype=x.dtype, device=device)
        else:
            mean = self.mean.copy()
            std  = self.std.copy()
            if not inplace:
                x = x.copy()

        mean = self._reshape_stats(mean, x.ndim, is_tensor)
        std  = self._reshape_stats(std,  x.ndim, is_tensor)

        return (x - mean) / std

    # ================================================================
    # 修复后的 inverse_transform
    # ================================================================
    def inverse_transform(self, x):
        if not self._fitted:
            raise RuntimeError(f"[{self.name}] Normalizer not fitted!")

        is_tensor = isinstance(x, torch.Tensor)
        device = x.device if is_tensor else None

        if is_tensor:
            mean = torch.tensor(self.mean, dtype=x.dtype, device=device)
            std  = torch.tensor(self.std,  dtype=x.dtype, device=device)
        else:
            mean = self.mean.copy()
            std  = self.std.copy()

        mean = self._reshape_stats(mean, x.ndim, is_tensor)
        std  = self._reshape_stats(std,  x.ndim, is_tensor)

        return x * std + mean

    # ================================================================
    # 以下方法保持不变
    # ================================================================
    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std, eps=self.eps, name=self.name)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(
            mean=data['mean'], std=data['std'],
            eps=float(data['eps']), name=str(data['name']),
        )

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std,
                'eps': self.eps, 'name': self.name}

    def load_state_dict(self, state):
        self.mean = state['mean']
        self.std = state['std']
        self.eps = state.get('eps', 1e-8)
        self.name = state.get('name', 'normalizer')
        self._fitted = True

# =============================================================================
# Part 3: 懒加载数据集 (Lazy Loading Dataset)
# =============================================================================

class LazySatelliteERA5Dataset(Dataset):
    """
    支持TB级数据的懒加载数据集
    
    改进点:
    1. 不在__init__中加载全部数据，而是维护文件索引
    2. 使用内存映射 (mmap) 或 HDF5 延迟读取
    3. 支持分布式训练的数据分片
    
    文件组织结构 (推荐):
    data_root/
    ├── 2024/
    │   ├── 01/
    │   │   ├── sample_0001.npz  # 包含 obs, bkg, target, aux
    │   │   ├── sample_0002.npz
    │   │   └── ...
    │   └── 02/
    │       └── ...
    └── stats.npz  # 预计算的标准化统计量
    """
    
    def __init__(
        self,
        file_list: List[str],
        obs_normalizer: Optional[LevelwiseNormalizer] = None,
        bkg_normalizer: Optional[LevelwiseNormalizer] = None,
        target_normalizer: Optional[LevelwiseNormalizer] = None,
        config: Optional[DataConfig] = None,
        mask_mode: str = 'any',
        fill_value: float = 0.0,
        use_aux: bool = True,
        transform: Optional[Callable] = None,
        cache_size: int = 0  # LRU缓存大小，0表示不缓存
    ):
        """
        Args:
            file_list: 数据文件路径列表 (npz或h5格式)
            obs_normalizer: 观测标准化器
            bkg_normalizer: 背景标准化器  
            target_normalizer: 目标标准化器
            config: 数据配置
            mask_mode: 掩码模式 ('any', 'all', 'channel')
            fill_value: NaN填充值
            use_aux: 是否加载辅助特征
            transform: 数据增强函数
            cache_size: LRU缓存大小
        """
        super().__init__()
        
        self.file_list = sorted(file_list)
        self.n_samples = len(file_list)
        self.config = config or DataConfig()
        self.mask_mode = mask_mode
        self.fill_value = fill_value
        self.use_aux = use_aux
        self.transform = transform
        
        # 标准化器
        self.obs_normalizer = obs_normalizer or LevelwiseNormalizer(name='obs')
        self.bkg_normalizer = bkg_normalizer or LevelwiseNormalizer(name='bkg')
        self.target_normalizer = target_normalizer or LevelwiseNormalizer(name='target')
        
        # 简单的LRU缓存
        self._cache: Dict[int, Dict] = {}
        self._cache_order: List[int] = []
        self._cache_size = cache_size
        
        print(f"[LazySatelliteERA5Dataset] 初始化完成:")
        print(f"  样本数: {self.n_samples}")
        print(f"  使用辅助特征: {self.use_aux}")
        print(f"  缓存大小: {self._cache_size}")
    
    @classmethod
    def from_directory(
        cls,
        data_root: str,
        pattern: str = "**/*.npz",
        stats_file: Optional[str] = None,
        **kwargs
    ) -> 'LazySatelliteERA5Dataset':
        """
        从目录创建数据集
        
        Args:
            data_root: 数据根目录
            pattern: 文件匹配模式
            stats_file: 统计量文件路径
        """
        root = Path(data_root)
        file_list = sorted([str(p) for p in root.glob(pattern)])
        
        if not file_list:
            raise ValueError(f"No files found in {data_root} with pattern {pattern}")
        
        # 加载预计算的统计量
        obs_norm, bkg_norm, tgt_norm = None, None, None
        if stats_file and Path(stats_file).exists():
            stats = np.load(stats_file)
            obs_norm = LevelwiseNormalizer(stats['obs_mean'], stats['obs_std'], name='obs')
            bkg_norm = LevelwiseNormalizer(stats['bkg_mean'], stats['bkg_std'], name='bkg')
            tgt_norm = LevelwiseNormalizer(stats['target_mean'], stats['target_std'], name='target')
            print(f"  ✓ 加载预计算统计量: {stats_file}")
        
        return cls(
            file_list=file_list,
            obs_normalizer=obs_norm,
            bkg_normalizer=bkg_norm,
            target_normalizer=tgt_norm,
            **kwargs
        )
    
    def compute_statistics(
        self, 
        n_samples: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        使用Welford在线算法计算统计量
        
        Args:
            n_samples: 用于计算统计量的样本数 (None=全部)
            save_path: 保存路径
        """
        indices = list(range(self.n_samples))
        if n_samples:
            indices = indices[:n_samples]
        
        print(f"[计算统计量] 处理 {len(indices)} 个样本...")
        
        for i, idx in enumerate(indices):
            data = self._load_raw(idx)
            self.obs_normalizer.partial_fit(data['obs'])
            self.bkg_normalizer.partial_fit(data['bkg'])
            self.target_normalizer.partial_fit(data['target'])
            
            if (i + 1) % 100 == 0:
                print(f"  处理进度: {i+1}/{len(indices)}")
        
        print("  ✓ 统计量计算完成")
        
        if save_path:
            np.savez(
                save_path,
                obs_mean=self.obs_normalizer.mean,
                obs_std=self.obs_normalizer.std,
                bkg_mean=self.bkg_normalizer.mean,
                bkg_std=self.bkg_normalizer.std,
                target_mean=self.target_normalizer.mean,
                target_std=self.target_normalizer.std
            )
            print(f"  ✓ 保存至: {save_path}")
    
    def _load_raw(self, idx: int) -> Dict[str, np.ndarray]:
        """加载原始数据 (未标准化)"""
        file_path = self.file_list[idx]
        
        if file_path.endswith('.npz'):
            data = np.load(file_path)
            return {
                'obs': data['obs'],      # [C_obs, H, W]
                'bkg': data['bkg'],      # [C_bkg, H, W]
                'target': data['target'], # [C_bkg, H, W]
                'aux': data.get('aux', None)  # [C_aux, H, W] 可选
            }
        elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            import h5py
            with h5py.File(file_path, 'r') as f:
                return {
                    'obs': f['obs'][:],
                    'bkg': f['bkg'][:],
                    'target': f['target'][:],
                    'aux': f['aux'][:] if 'aux' in f else None
                }
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def _generate_mask(self, obs: np.ndarray) -> np.ndarray:
        """生成有效性掩码"""
        nan_mask = np.isnan(obs)
        
        if self.mask_mode == 'any':
            invalid = nan_mask.any(axis=0, keepdims=True)
            mask = (~invalid).astype(np.float32)
        elif self.mask_mode == 'all':
            invalid = nan_mask.all(axis=0, keepdims=True)
            mask = (~invalid).astype(np.float32)
        elif self.mask_mode == 'channel':
            mask = (~nan_mask).astype(np.float32)
        else:
            raise ValueError(f"Unknown mask_mode: {self.mask_mode}")
        
        return mask
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        # 检查缓存
        if idx in self._cache:
            return self._cache[idx]
        
        # 加载原始数据
        raw = self._load_raw(idx)
        
        obs_raw = raw['obs'].copy()
        bkg_raw = raw['bkg'].copy()
        target_raw = raw['target'].copy()
        aux_raw = raw['aux']
        
        # Step 1: 生成掩码 (NaN填充前!)
        mask = self._generate_mask(obs_raw)
        
        # Step 2: 标准化
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obs_norm = self.obs_normalizer.transform(obs_raw)
            bkg_norm = self.bkg_normalizer.transform(bkg_raw)
            target_norm = self.target_normalizer.transform(target_raw)
        
        # Step 3: 填充NaN
        obs_norm = np.nan_to_num(obs_norm, nan=self.fill_value)
        
        # Step 4: 构建输出
        sample = {
            'obs': torch.tensor(obs_norm, dtype=self.config.dtype),
            'bkg': torch.tensor(bkg_norm, dtype=self.config.dtype),
            'mask': torch.tensor(mask, dtype=self.config.dtype),
            'target': torch.tensor(target_norm, dtype=self.config.dtype),
        }
        
        # 辅助特征
        if self.use_aux and aux_raw is not None:
            sample['aux'] = torch.tensor(aux_raw, dtype=self.config.dtype)
        
        # 数据增强
        if self.transform:
            sample = self.transform(sample)
        
        # 更新缓存
        if self._cache_size > 0:
            self._cache[idx] = sample
            self._cache_order.append(idx)
            if len(self._cache_order) > self._cache_size:
                oldest = self._cache_order.pop(0)
                self._cache.pop(oldest, None)
        
        return sample
    
    def get_normalizers(self) -> Dict[str, LevelwiseNormalizer]:
        """获取所有标准化器"""
        return {
            'obs': self.obs_normalizer,
            'bkg': self.bkg_normalizer,
            'target': self.target_normalizer
        }


# =============================================================================
# Part 4: 内存数据集 (用于小数据集或测试)
# =============================================================================

class InMemorySatelliteDataset(Dataset):
    """
    内存加载数据集 (适用于小数据集)
    
    与LazySatelliteERA5Dataset接口兼容
    """
    
    def __init__(
        self,
        obs_data: np.ndarray,
        bkg_data: np.ndarray,
        target_data: np.ndarray,
        aux_data: Optional[np.ndarray] = None,
        obs_normalizer: Optional[LevelwiseNormalizer] = None,
        bkg_normalizer: Optional[LevelwiseNormalizer] = None,
        target_normalizer: Optional[LevelwiseNormalizer] = None,
        compute_stats: bool = False,
        mask_mode: str = 'any',
        fill_value: float = 0.0,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.obs_data = obs_data.astype(np.float32)
        self.bkg_data = bkg_data.astype(np.float32)
        self.target_data = target_data.astype(np.float32)
        self.aux_data = aux_data.astype(np.float32) if aux_data is not None else None
        
        self.n_samples = obs_data.shape[0]
        self.mask_mode = mask_mode
        self.fill_value = fill_value
        self.dtype = dtype
        
        # 标准化器
        self.obs_normalizer = obs_normalizer or LevelwiseNormalizer(name='obs')
        self.bkg_normalizer = bkg_normalizer or LevelwiseNormalizer(name='bkg')
        self.target_normalizer = target_normalizer or LevelwiseNormalizer(name='target')
        
        if compute_stats:
            print("[InMemorySatelliteDataset] 计算统计量...")
            self.obs_normalizer.fit(self.obs_data)
            self.bkg_normalizer.fit(self.bkg_data)
            self.target_normalizer.fit(self.target_data)
        
        print(f"[InMemorySatelliteDataset] 样本数: {self.n_samples}")
    
    def _generate_mask(self, obs: np.ndarray) -> np.ndarray:
        nan_mask = np.isnan(obs)
        if self.mask_mode == 'any':
            invalid = nan_mask.any(axis=0, keepdims=True)
            return (~invalid).astype(np.float32)
        elif self.mask_mode == 'all':
            invalid = nan_mask.all(axis=0, keepdims=True)
            return (~invalid).astype(np.float32)
        else:
            return (~nan_mask).astype(np.float32)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obs_raw = self.obs_data[idx].copy()
        bkg_raw = self.bkg_data[idx].copy()
        target_raw = self.target_data[idx].copy()
        
        mask = self._generate_mask(obs_raw)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obs_norm = self.obs_normalizer.transform(obs_raw)
            bkg_norm = self.bkg_normalizer.transform(bkg_raw)
            target_norm = self.target_normalizer.transform(target_raw)
        
        obs_norm = np.nan_to_num(obs_norm, nan=self.fill_value)
        
        sample = {
            'obs': torch.tensor(obs_norm, dtype=self.dtype),
            'bkg': torch.tensor(bkg_norm, dtype=self.dtype),
            'mask': torch.tensor(mask, dtype=self.dtype),
            'target': torch.tensor(target_norm, dtype=self.dtype),
        }
        
        if self.aux_data is not None:
            sample['aux'] = torch.tensor(self.aux_data[idx], dtype=self.dtype)
        
        return sample
    
    def get_normalizers(self) -> Dict[str, LevelwiseNormalizer]:
        return {
            'obs': self.obs_normalizer,
            'bkg': self.bkg_normalizer,
            'target': self.target_normalizer
        }


# =============================================================================
# Part 5: SE模块 (Squeeze-and-Excitation Block)
# =============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (通道注意力)
    
    新增: 支持导出注意力权重用于可解释性分析
    """
    
    def __init__(
        self, 
        channels: int, 
        reduction: int = 16,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        reduced_channels = max(channels // reduction, 4)
        
        act_fn = {
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True)
        }.get(activation, nn.GELU())
        
        self.squeeze_excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1, bias=True),
            act_fn,
            nn.Conv2d(reduced_channels, channels, 1, bias=True),
            nn.Sigmoid()
        )
        
        # 存储最近一次的注意力权重 (用于可解释性)
        self._last_attention: Optional[torch.Tensor] = None
        self._init_weights()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        attention = self.squeeze_excite(x)  # [B, C, 1, 1]
        self._last_attention = attention.detach()
        return x * attention
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取最近一次的注意力权重 (用于通道显著性分析)"""
        return self._last_attention


# =============================================================================
# Part 6: 辅助特征编码器 (Auxiliary Feature Encoder)
# =============================================================================

class AuxiliaryEncoder(nn.Module):
    """
    辅助地理/时间特征编码器
    
    输入特征:
    - 纬度 (Latitude): 影响对流层顶高度
    - 经度 (Longitude): 影响局部气候特征
    - 太阳天顶角 (Solar Zenith Angle): 影响辐射加热
    - 地表类型 (Land Mask): 影响发射率
    
    编码方式:
    - 连续变量: 周期编码 (sin/cos) + 线性投影
    - 离散变量: Embedding
    """
    
    def __init__(
        self,
        n_aux_features: int = 4,
        embed_dim: int = 32,
        use_periodic_encoding: bool = True
    ):
        """
        Args:
            n_aux_features: 辅助特征数
            embed_dim: 嵌入维度
            use_periodic_encoding: 是否使用周期编码
        """
        super().__init__()
        
        self.use_periodic_encoding = use_periodic_encoding
        
        if use_periodic_encoding:
            # 周期编码: 每个特征 -> (sin, cos) -> 2倍维度
            self.encoder = nn.Sequential(
                nn.Conv2d(n_aux_features * 2, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU()
            )
        else:
            # 简单线性投影
            self.encoder = nn.Sequential(
                nn.Conv2d(n_aux_features, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU()
            )
    
    def _periodic_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        周期编码
        
        Args:
            x: [B, C, H, W] 值域 [-1, 1] 或 [0, 1]
        Returns:
            [B, 2C, H, W]
        """
        # 归一化到 [0, 2π]
        x_scaled = x * torch.pi
        return torch.cat([torch.sin(x_scaled), torch.cos(x_scaled)], dim=1)
    
    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        """
        Args:
            aux: [B, C_aux, H, W]
        Returns:
            [B, embed_dim, H, W]
        """
        if self.use_periodic_encoding:
            aux = self._periodic_encode(aux)  # [B, 2*C_aux, H, W]
        return self.encoder(aux)  # [B, embed_dim, H, W]


# =============================================================================
# Part 7: 掩码感知卷积 (Mask-Aware Convolution)
# =============================================================================

class PartialConv2d(nn.Module):
    """
    Partial Convolution (部分卷积)
    
    来源: "Image Inpainting for Irregular Holes Using Partial Convolutions" (Liu et al., ECCV 2018)
    
    核心思想:
    - 卷积只在有效像素上进行
    - 输出会根据有效像素比例进行重新归一化
    - 掩码也会被传播
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        
        # 掩码卷积核 (全1，用于计算有效像素比例)
        self.register_buffer(
            'mask_kernel',
            torch.ones(1, 1, kernel_size, kernel_size)
        )
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 特征 [B, C_in, H, W]
            mask: 有效性掩码 [B, 1, H, W], 1=有效, 0=无效
        
        Returns:
            output: 卷积结果 [B, C_out, H', W']
            updated_mask: 更新后的掩码 [B, 1, H', W']
        """
        # 将无效位置置零
        x_masked = x * mask
        
        # 计算有效像素比例
        with torch.no_grad():
            # 每个输出位置看到的有效像素数
            valid_count = F.conv2d(
                mask, 
                self.mask_kernel, 
                stride=self.stride, 
                padding=self.padding
            )
            # 最大可能的像素数
            max_count = self.kernel_size ** 2
            # 比例因子
            ratio = max_count / (valid_count + 1e-8)
            # 更新掩码: 只要看到一个有效像素，输出就有效
            updated_mask = (valid_count > 0).float()
        
        # 执行卷积
        output = self.conv(x_masked)
        
        # 重新归一化
        output = output * ratio * updated_mask
        
        return output, updated_mask


class MaskAwareConv2d(nn.Module):
    """
    掩码感知卷积 (简化版)
    
    将掩码作为额外通道拼接，让网络学习如何处理缺测
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()
        
        # 输入 = 特征 + 掩码
        self.conv = nn.Conv2d(
            in_channels + 1,  # +1 for mask channel
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
            mask: [B, 1, H, W]
        Returns:
            [B, C_out, H, W]
        """
        # 拼接掩码
        x_with_mask = torch.cat([x, mask], dim=1)  # [B, C+1, H, W]
        return self.act(self.bn(self.conv(x_with_mask)))


# =============================================================================
# Part 8: 光谱适配器茎干模块 V2 (SpectralAdapterStem V2)
# =============================================================================

class SpectralAdapterStemV2(nn.Module):
    """
    光谱适配器茎干模块 V2 - 顶会标准版
    
    改进点:
    1. 辅助特征融合 (Auxiliary Feature Fusion)
    2. 掩码感知处理 (Mask-Aware Processing)
    3. 非线性融合策略 (Non-linear Fusion)
    4. 支持多种融合模式 (concat/add/gated)
    
    数据流:
    
    X_obs [B,17,H,W] --> [1x1 Conv] --> [GELU] --> [SE] --> obs_feat [B,C,H,W]
                                                               |
                                                               v  (× Mask)
    X_bkg [B,37,H,W] --> [1x1 Conv] ---------------------------> FUSION --> [3x3 Conv] --> F_out
                                                               ^
    X_aux [B,4,H,W]  --> [AuxEncoder] -------------------------+
                                                               ^
    Mask  [B,1,H,W]  --> (broadcast) --------------------------+
    """
    
    def __init__(
        self,
        obs_channels: int = 17,
        bkg_channels: int = 37,
        aux_channels: int = 4,
        latent_channels: int = 64,
        se_reduction: int = 8,
        fusion_mode: str = 'concat',  # 'concat', 'add', 'gated'
        use_aux: bool = True,
        mask_aware: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            obs_channels: 观测通道数 (17: FY-3F MWTS)
            bkg_channels: 背景层数 (37: ERA5)
            aux_channels: 辅助特征数 (4: lat, lon, sza, land)
            latent_channels: 潜在特征维度
            se_reduction: SE模块压缩比
            fusion_mode: 融合模式
                - 'concat': 拼接后卷积 (最灵活)
                - 'add': 加权相加 (参数少)
                - 'gated': 门控融合 (自适应)
            use_aux: 是否使用辅助特征
            mask_aware: 是否使用掩码感知卷积
            dropout: Dropout比例
        """
        super().__init__()
        
        self.obs_channels = obs_channels
        self.bkg_channels = bkg_channels
        self.latent_channels = latent_channels
        self.fusion_mode = fusion_mode
        self.use_aux = use_aux
        self.mask_aware = mask_aware
        
        # =====================================================================
        # 观测投影分支: 模拟逆辐射传输模型 (RTM^{-1})
        # X_obs [B, 17, H, W] -> [B, C_lat, H, W]
        # =====================================================================
        self.obs_projection = nn.Sequential(
            nn.Conv2d(obs_channels, latent_channels, 1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.GELU(),
            SEBlock(latent_channels, reduction=se_reduction, activation='gelu'),
            nn.Conv2d(latent_channels, latent_channels, 1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        
        # =====================================================================
        # 背景投影分支
        # X_bkg [B, 37, H, W] -> [B, C_lat, H, W]
        # =====================================================================
        self.bkg_projection = nn.Sequential(
            nn.Conv2d(bkg_channels, latent_channels, 1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.GELU()
        )
        
        # =====================================================================
        # 辅助特征编码器 (可选)
        # X_aux [B, 4, H, W] -> [B, 32, H, W]
        # =====================================================================
        aux_embed_dim = 32 if use_aux else 0
        if use_aux:
            self.aux_encoder = AuxiliaryEncoder(
                n_aux_features=aux_channels,
                embed_dim=aux_embed_dim,
                use_periodic_encoding=True
            )
        else:
            self.aux_encoder = None
        
        # =====================================================================
        # 融合模块
        # =====================================================================
        if fusion_mode == 'concat':
            # 拼接融合: [obs, bkg, aux (optional), mask] -> conv
            concat_channels = latent_channels * 2 + aux_embed_dim
            if mask_aware:
                concat_channels += 1  # +1 for mask
            
            self.fusion = nn.Sequential(
                nn.Conv2d(concat_channels, latent_channels, 1, bias=False),
                nn.BatchNorm2d(latent_channels),
                nn.GELU(),
                nn.Conv2d(latent_channels, latent_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(latent_channels),
                nn.GELU()
            )
            
        elif fusion_mode == 'add':
            # 加权融合: bkg + α * (obs ⊙ mask)
            self.alpha = nn.Parameter(torch.ones(latent_channels, 1, 1))
            self.fusion = nn.Sequential(
                nn.Conv2d(latent_channels, latent_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(latent_channels),
                nn.GELU()
            )
            
        elif fusion_mode == 'gated':
            # 门控融合: 学习动态权重
            # 修复: 去掉BN，用tanh确保初始gate≈0.5（对称融合）
            gate_in_ch = latent_channels * 2 + aux_embed_dim
            self.gate_net = nn.Sequential(
                nn.Conv2d(gate_in_ch, latent_channels, 1, bias=True),
                nn.Tanh()                 # output ∈ (-1,1)  →  gate = 0.5+0.5*tanh ∈ (0,1)
            )
            # 初始化: weight=0, bias=0 → tanh(0)=0 → gate=0.5（初始均匀融合）
            nn.init.zeros_(self.gate_net[0].weight)
            nn.init.zeros_(self.gate_net[0].bias)
            self.fusion = nn.Sequential(
                nn.Conv2d(latent_channels, latent_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(latent_channels),
                nn.GELU()
            )
        
        # 残差连接 (从背景场)
        self.bkg_residual = nn.Conv2d(bkg_channels, latent_channels, 1, bias=False)
        
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
        """打印模块信息"""
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[SpectralAdapterStemV2] 初始化完成:")
        print(f"  观测: {self.obs_channels}ch -> 潜在: {self.latent_channels}ch")
        print(f"  背景: {self.bkg_channels}ch -> 潜在: {self.latent_channels}ch")
        print(f"  融合模式: {self.fusion_mode}")
        print(f"  辅助特征: {self.use_aux}, 掩码感知: {self.mask_aware}")
        print(f"  参数量: {n_params:,}")
    
    def forward(
        self, 
        obs: torch.Tensor, 
        bkg: torch.Tensor, 
        mask: torch.Tensor,
        aux: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            obs: 观测场 [B, 17, H, W]
            bkg: 背景场 [B, 37, H, W]
            mask: 有效性掩码 [B, 1, H, W]
            aux: 辅助特征 [B, 4, H, W] (可选)
        
        Returns:
            fused: 融合特征 [B, C_lat, H, W]
        """
        B, _, H, W = obs.shape
        
        # =================================================================
        # Step 1: 观测投影 (逆RTM)
        # =================================================================
        obs_feat = self.obs_projection(obs)  # [B, C_lat, H, W]
        
        # 掩码门控: 缺测区域置零
        if mask.shape[1] == 1:
            mask_expanded = mask
        else:
            mask_expanded = mask.mean(dim=1, keepdim=True)
        
        obs_gated = obs_feat * mask_expanded  # [B, C_lat, H, W]
        
        # =================================================================
        # Step 2: 背景投影
        # =================================================================
        bkg_feat = self.bkg_projection(bkg)  # [B, C_lat, H, W]
        
        # =================================================================
        # Step 3: 辅助特征编码 (可选)
        # =================================================================
        if self.use_aux and aux is not None:
            aux_feat = self.aux_encoder(aux)  # [B, 32, H, W]
        else:
            aux_feat = None
        
        # =================================================================
        # Step 4: 特征融合
        # =================================================================
        if self.fusion_mode == 'concat':
            # 拼接融合
            feat_list = [obs_gated, bkg_feat]
            if aux_feat is not None:
                feat_list.append(aux_feat)
            if self.mask_aware:
                feat_list.append(mask_expanded)
            
            concat_feat = torch.cat(feat_list, dim=1)
            fused = self.fusion(concat_feat)
            
        elif self.fusion_mode == 'add':
            # 加权相加
            fused = bkg_feat + self.alpha * obs_gated
            fused = self.fusion(fused)
            
        elif self.fusion_mode == 'gated':
            # 门控融合 (修复版: tanh-based gate, 无硬mask截断)
            gate_raw = [obs_feat, bkg_feat]
            if aux_feat is not None:
                gate_raw.append(aux_feat)
            gate_input = torch.cat(gate_raw, dim=1)
            
            # gate ∈ (0,1)，初始≈0.5; obs_gated已在缺测处为0，无需再mask gate
            gate = 0.5 + 0.5 * self.gate_net(gate_input)
            
            fused = bkg_feat * (1 - gate) + obs_gated * gate
            fused = self.fusion(fused)
        
        # =================================================================
        # Step 5: 残差连接 (从原始背景场)
        # =================================================================
        bkg_skip = self.bkg_residual(bkg)  # [B, C_lat, H, W]
        fused = fused + bkg_skip
        
        return fused
    
    def get_se_attention(self) -> Optional[torch.Tensor]:
        """获取SE模块的注意力权重 (用于可解释性)"""
        for module in self.obs_projection:
            if isinstance(module, SEBlock):
                return module.get_attention_weights()
        return None


# =============================================================================
# Part 9: 消融实验评估指标 (Ablation Study Metrics)
# =============================================================================

class AssimilationMetrics:
    """
    数据同化评估指标集合
    
    包含:
    1. 分层RMSE (Level-wise RMSE)
    2. 通道显著性分析 (Channel Saliency)
    3. 缺测鲁棒性曲线 (Gap Robustness)
    4. 梯度相似性 (Gradient Similarity)
    """
    
    def __init__(
        self,
        pressure_levels: Optional[np.ndarray] = None,
        stratosphere_threshold: float = 100.0  # hPa
    ):
        """
        Args:
            pressure_levels: 气压层数组 [37], 单位hPa
            stratosphere_threshold: 平流层界限 (默认100hPa)
        """
        # 默认ERA5气压层 (hPa)
        if pressure_levels is None:
            self.pressure_levels = np.array([
                1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
                750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
                250, 225, 200, 175, 150, 125, 100, 70, 50, 30,
                20, 10, 7, 5, 3, 2, 1
            ])
        else:
            self.pressure_levels = pressure_levels
        
        self.strat_threshold = stratosphere_threshold
        self.n_levels = len(self.pressure_levels)
        
        # 区分对流层和平流层
        self.strat_mask = self.pressure_levels <= stratosphere_threshold
        self.trop_mask = ~self.strat_mask
    
    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """计算RMSE"""
        mse = ((pred - target) ** 2).mean(dim=dim)
        return torch.sqrt(mse)
    
    def levelwise_rmse(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        分层RMSE
        
        Args:
            pred: [B, 37, H, W]
            target: [B, 37, H, W]
        
        Returns:
            {
                'levelwise': [37] 各层RMSE,
                'global': 全局RMSE,
                'stratosphere': 平流层RMSE,
                'troposphere': 对流层RMSE,
                'pressure_levels': [37] 气压层
            }
        """
        B, C, H, W = pred.shape
        
        # 各层RMSE: [37]
        levelwise = self.rmse(pred, target, dim=(0, 2, 3))  # [C]
        
        # 全局RMSE
        global_rmse = self.rmse(pred, target)
        
        # 平流层/对流层分开
        strat_pred = pred[:, self.strat_mask, :, :]
        strat_target = target[:, self.strat_mask, :, :]
        strat_rmse = self.rmse(strat_pred, strat_target)
        
        trop_pred = pred[:, self.trop_mask, :, :]
        trop_target = target[:, self.trop_mask, :, :]
        trop_rmse = self.rmse(trop_pred, trop_target)
        
        return {
            'levelwise': levelwise,
            'global': global_rmse,
            'stratosphere': strat_rmse,
            'troposphere': trop_rmse,
            'pressure_levels': torch.tensor(self.pressure_levels)
        }
    
    @staticmethod
    def gradient_loss(
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        梯度相似性 (Sobel)
        
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
        
        Returns:
            {
                'grad_rmse': 梯度RMSE,
                'grad_correlation': 梯度相关系数
            }
        """
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        B, C, H, W = pred.shape
        
        # 展平通道维度
        pred_flat = pred.view(B * C, 1, H, W)
        target_flat = target.view(B * C, 1, H, W)
        
        # 计算梯度
        pred_gx = F.conv2d(pred_flat, sobel_x, padding=1)
        pred_gy = F.conv2d(pred_flat, sobel_y, padding=1)
        pred_grad = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-8)
        
        target_gx = F.conv2d(target_flat, sobel_x, padding=1)
        target_gy = F.conv2d(target_flat, sobel_y, padding=1)
        target_grad = torch.sqrt(target_gx**2 + target_gy**2 + 1e-8)
        
        # 梯度RMSE
        grad_rmse = torch.sqrt(((pred_grad - target_grad) ** 2).mean())
        
        # 梯度相关系数
        pred_grad_flat = pred_grad.flatten()
        target_grad_flat = target_grad.flatten()
        
        pred_mean = pred_grad_flat.mean()
        target_mean = target_grad_flat.mean()
        
        numerator = ((pred_grad_flat - pred_mean) * (target_grad_flat - target_mean)).sum()
        denominator = torch.sqrt(
            ((pred_grad_flat - pred_mean) ** 2).sum() * 
            ((target_grad_flat - target_mean) ** 2).sum()
        )
        grad_corr = numerator / (denominator + 1e-8)
        
        return {
            'grad_rmse': grad_rmse,
            'grad_correlation': grad_corr
        }
    
    @staticmethod
    def gap_robustness_test(
        model: nn.Module,
        dataloader: DataLoader,
        gap_ratios: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        device: str = 'cpu'
    ) -> Dict[str, List[float]]:
        """
        缺测鲁棒性测试
        
        人为制造不同比例的缺失，测试模型表现
        
        Args:
            model: 模型
            dataloader: 数据加载器
            gap_ratios: 测试的缺失比例列表
            device: 设备
        
        Returns:
            {
                'gap_ratios': 缺失比例,
                'rmse_values': 对应RMSE,
                'baseline_rmse': 100%缺失时的RMSE (背景场误差)
            }
        """
        model.eval()
        results = {ratio: [] for ratio in gap_ratios}
        
        with torch.no_grad():
            for batch in dataloader:
                obs = batch['obs'].to(device)
                bkg = batch['bkg'].to(device)
                target = batch['target'].to(device)
                original_mask = batch['mask'].to(device)
                aux = batch.get('aux')
                if aux is not None:
                    aux = aux.to(device)
                
                for ratio in gap_ratios:
                    if ratio == 0.0:
                        # 使用原始掩码
                        mask = original_mask
                    else:
                        # 人为添加缺失
                        B, _, H, W = obs.shape
                        artificial_gap = (torch.rand(B, 1, H, W, device=device) < ratio)
                        mask = original_mask * (~artificial_gap).float()
                    
                    # 模型前向传播
                    if hasattr(model, 'forward'):
                        # 假设模型接受 (obs, bkg, mask, aux) 并返回预测
                        try:
                            pred = model(obs, bkg, mask, aux)
                        except:
                            pred = model(obs * mask, bkg, mask)
                    
                    # 计算RMSE
                    rmse = torch.sqrt(((pred - target) ** 2).mean()).item()
                    results[ratio].append(rmse)
        
        # 计算平均
        rmse_values = [np.mean(results[r]) for r in gap_ratios]
        
        return {
            'gap_ratios': gap_ratios,
            'rmse_values': rmse_values,
            'baseline_rmse': rmse_values[-1] if gap_ratios[-1] == 1.0 else None
        }
    
    @staticmethod
    def channel_saliency_analysis(
        se_weights: torch.Tensor,
        channel_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        通道显著性分析
        
        从SE模块的注意力权重分析各通道的重要性
        
        Args:
            se_weights: SE注意力权重 [B, C, 1, 1] 或 [C]
            channel_names: 通道名称列表
        
        Returns:
            {
                'mean_weights': 平均权重,
                'std_weights': 权重标准差,
                'ranking': 重要性排序,
                'channel_names': 通道名称
            }
        """
        if se_weights.dim() == 4:
            weights = se_weights.squeeze(-1).squeeze(-1)  # [B, C]
        elif se_weights.dim() == 2:
            weights = se_weights
        else:
            weights = se_weights.unsqueeze(0)
        
        mean_weights = weights.mean(dim=0)  # [C]
        std_weights = weights.std(dim=0)    # [C]
        ranking = torch.argsort(mean_weights, descending=True)
        
        n_channels = mean_weights.shape[0]
        if channel_names is None:
            channel_names = [f"Ch{i+1}" for i in range(n_channels)]
        
        return {
            'mean_weights': mean_weights,
            'std_weights': std_weights,
            'ranking': ranking,
            'channel_names': channel_names
        }


# =============================================================================
# Part 10: 可视化工具 (Visualization)
# =============================================================================

def plot_levelwise_rmse(
    results: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
    title: str = "Level-wise RMSE Profile"
) -> None:
    """
    绘制分层RMSE廓线图
    
    Args:
        results: levelwise_rmse的返回值
        save_path: 保存路径
        title: 图标题
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    pressure = results['pressure_levels'].numpy()
    rmse = results['levelwise'].numpy()
    
    fig, ax = plt.subplots(figsize=(8, 10))
    
    ax.plot(rmse, pressure, 'b-o', linewidth=2, markersize=4)
    ax.axhline(y=100, color='r', linestyle='--', label='Tropopause (~100 hPa)')
    
    ax.set_xlabel('RMSE (K)', fontsize=12)
    ax.set_ylabel('Pressure (hPa)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylim([1000, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 添加统计信息
    textstr = f"Global: {results['global']:.3f} K\n"
    textstr += f"Troposphere: {results['troposphere']:.3f} K\n"
    textstr += f"Stratosphere: {results['stratosphere']:.3f} K"
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 保存至: {save_path}")
    
    plt.close()


def plot_gap_robustness(
    results: Dict,
    save_path: Optional[str] = None,
    title: str = "Gap Robustness Curve"
) -> None:
    """
    绘制缺测鲁棒性曲线
    
    Args:
        results: gap_robustness_test的返回值
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    gap_ratios = results['gap_ratios']
    rmse_values = results['rmse_values']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(gap_ratios, rmse_values, 'b-o', linewidth=2, markersize=8)
    
    if results['baseline_rmse']:
        ax.axhline(y=results['baseline_rmse'], color='r', linestyle='--', 
                   label=f'Background Error: {results["baseline_rmse"]:.3f} K')
    
    ax.set_xlabel('Gap Ratio', fontsize=12)
    ax.set_ylabel('RMSE (K)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 保存至: {save_path}")
    
    plt.close()


# =============================================================================
# Part 11: 测试函数
# =============================================================================

def create_synthetic_data_v2(
    n_samples: int = 100,
    n_obs_channels: int = 17,
    n_bkg_levels: int = 37,
    n_aux_features: int = 4,
    height: int = 64,
    width: int = 64,
    nan_ratio: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    创建合成测试数据 (含辅助特征)
    """
    np.random.seed(42)
    
    # 观测数据
    obs_data = np.random.normal(230, 30, (n_samples, n_obs_channels, height, width))
    for c in range(n_obs_channels):
        obs_data[:, c, :, :] += (c - n_obs_channels // 2) * 5
    
    # 添加NaN
    nan_mask = np.random.random((n_samples, 1, height, width)) < nan_ratio
    nan_mask = np.broadcast_to(nan_mask, obs_data.shape).copy()
    obs_data[nan_mask] = np.nan
    
    # 背景数据
    bkg_data = np.random.normal(260, 25, (n_samples, n_bkg_levels, height, width))
    for l in range(n_bkg_levels):
        bkg_data[:, l, :, :] += (l - n_bkg_levels // 2) * 2
    
    # 目标数据
    target_data = bkg_data + np.random.normal(0, 2, bkg_data.shape)
    
    # 辅助特征: [lat, lon, sza, land_mask]
    aux_data = np.zeros((n_samples, n_aux_features, height, width), dtype=np.float32)
    
    # 纬度: [-1, 1]
    lat_grid = np.linspace(-1, 1, height).reshape(1, 1, height, 1)
    aux_data[:, 0, :, :] = np.broadcast_to(lat_grid, (n_samples, 1, height, width)).squeeze(1)
    
    # 经度: [-1, 1]
    lon_grid = np.linspace(-1, 1, width).reshape(1, 1, 1, width)
    aux_data[:, 1, :, :] = np.broadcast_to(lon_grid, (n_samples, 1, height, width)).squeeze(1)
    
    # 太阳天顶角: [0, 1]
    aux_data[:, 2, :, :] = np.random.uniform(0, 1, (n_samples, height, width))
    
    # 地表类型: {0, 1}
    aux_data[:, 3, :, :] = np.random.randint(0, 2, (n_samples, height, width))
    
    return (
        obs_data.astype(np.float32), 
        bkg_data.astype(np.float32), 
        target_data.astype(np.float32),
        aux_data
    )


def test_lazy_dataset():
    """测试懒加载数据集 (模拟)"""
    print("=" * 70)
    print("测试 InMemorySatelliteDataset (替代懒加载测试)")
    print("=" * 70)
    
    obs, bkg, target, aux = create_synthetic_data_v2(n_samples=20)
    
    dataset = InMemorySatelliteDataset(
        obs_data=obs,
        bkg_data=bkg,
        target_data=target,
        aux_data=aux,
        compute_stats=True
    )
    
    sample = dataset[0]
    print(f"\n样本数据形状:")
    for key, val in sample.items():
        print(f"  {key}: {val.shape}")
    
    return dataset


def test_spectral_adapter_v2():
    """测试SpectralAdapterStemV2"""
    print("\n" + "=" * 70)
    print("测试 SpectralAdapterStemV2")
    print("=" * 70)
    
    # 测试不同融合模式
    for fusion_mode in ['concat', 'add', 'gated']:
        print(f"\n--- 融合模式: {fusion_mode} ---")
        
        adapter = SpectralAdapterStemV2(
            obs_channels=17,
            bkg_channels=37,
            aux_channels=4,
            latent_channels=64,
            fusion_mode=fusion_mode,
            use_aux=True,
            mask_aware=True
        )
        
        B, H, W = 4, 64, 64
        obs = torch.randn(B, 17, H, W)
        bkg = torch.randn(B, 37, H, W)
        mask = (torch.rand(B, 1, H, W) > 0.3).float()
        aux = torch.randn(B, 4, H, W)
        
        output = adapter(obs, bkg, mask, aux)
        print(f"  输出形状: {output.shape}")
        
        # 测试梯度
        loss = output.mean()
        loss.backward()
        print(f"  ✓ 梯度反向传播成功")
        
        # 获取SE注意力权重
        se_weights = adapter.get_se_attention()
        if se_weights is not None:
            print(f"  SE权重形状: {se_weights.shape}")
    
    return adapter


def test_metrics():
    """测试评估指标"""
    print("\n" + "=" * 70)
    print("测试 AssimilationMetrics")
    print("=" * 70)
    
    metrics = AssimilationMetrics()
    
    # 创建模拟预测和目标
    B, C, H, W = 4, 37, 64, 64
    pred = torch.randn(B, C, H, W)
    target = pred + torch.randn(B, C, H, W) * 0.5
    
    # 分层RMSE
    levelwise = metrics.levelwise_rmse(pred, target)
    print(f"\n分层RMSE:")
    print(f"  全局: {levelwise['global']:.4f} K")
    print(f"  对流层: {levelwise['troposphere']:.4f} K")
    print(f"  平流层: {levelwise['stratosphere']:.4f} K")
    
    # 梯度相似性
    grad = metrics.gradient_loss(pred, target)
    print(f"\n梯度指标:")
    print(f"  梯度RMSE: {grad['grad_rmse']:.4f}")
    print(f"  梯度相关: {grad['grad_correlation']:.4f}")
    
    return metrics


def test_full_pipeline_v2():
    """测试完整流水线"""
    print("\n" + "=" * 70)
    print("测试完整流水线 V2")
    print("=" * 70)
    
    # 1. 准备数据
    obs, bkg, target, aux = create_synthetic_data_v2(n_samples=32)
    
    # 2. 创建数据集
    dataset = InMemorySatelliteDataset(
        obs_data=obs, bkg_data=bkg, target_data=target, aux_data=aux,
        compute_stats=True
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 3. 创建模型
    adapter = SpectralAdapterStemV2(
        fusion_mode='gated',
        use_aux=True,
        mask_aware=True
    )
    head = nn.Conv2d(64, 37, 1)
    
    # 4. 训练
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(head.parameters()), 
        lr=1e-4
    )
    
    print("\n训练中...")
    for epoch in range(3):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            fused = adapter(
                batch['obs'], batch['bkg'], 
                batch['mask'], batch.get('aux')
            )
            pred = head(fused)
            
            loss = F.mse_loss(pred, batch['target'])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"  Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.6f}")
    
    # 5. 评估
    metrics = AssimilationMetrics()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        fused = adapter(batch['obs'], batch['bkg'], batch['mask'], batch.get('aux'))
        pred = head(fused)
        
        results = metrics.levelwise_rmse(pred, batch['target'])
        print(f"\n评估结果:")
        print(f"  全局RMSE: {results['global']:.4f} K")
        print(f"  对流层RMSE: {results['troposphere']:.4f} K")
        print(f"  平流层RMSE: {results['stratosphere']:.4f} K")
    
    print("\n✓ 完整流水线测试通过!")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    test_lazy_dataset()
    test_spectral_adapter_v2()
    test_metrics()
    test_full_pipeline_v2()
    
    print("\n" + "=" * 70)
    print("所有测试通过! ✓")
    print("=" * 70)
