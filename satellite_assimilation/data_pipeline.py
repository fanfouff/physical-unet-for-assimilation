"""
===============================================================================
卫星数据同化深度学习框架 - 数据管道与物理感知模型入口模块
Satellite Data Assimilation Deep Learning Framework
===============================================================================

作者: AI Research Scientist
任务: 将FY-3F卫星亮温(BT)同化到ERA5背景场，生成高精度大气温度廓线

数据规格:
- 观测场 (X_obs): FY-3F亮温, Shape: [B, 17, H, W], 含NaN轨道空隙
- 背景场 (X_bkg): ERA5温度场, Shape: [B, 37, H, W]
- 真值场 (Y_true): ERA5分析场, Shape: [B, 37, H, W]

核心创新:
1. 逐通道Z-Score标准化 (Level-wise Standardization)
2. 光谱适配器茎干模块 (SpectralAdapterStem) - 模拟逆辐射传输模型
3. 不确定性门控融合 (Uncertainty-Gated Fusion)
4. 挤压-激励注意力机制 (Squeeze-and-Excitation Block)

===============================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, Union, List
import numpy as np
from pathlib import Path
import warnings


# =============================================================================
# Part 1: 物理感知标准化器 (Physics-Aware Normalizer)
# =============================================================================

class LevelwiseNormalizer:
    """
    逐层/逐通道Z-Score标准化器
    
    物理背景:
    - 平流层温度变化范围小 (~200-250K, σ~5K)
    - 对流层温度变化范围大 (~220-310K, σ~20K)
    - 地表通道变化最剧烈 (~250-320K, σ~30K)
    
    如果使用全局MinMax缩放，平流层的微弱信号将被淹没！
    
    公式: x_normalized = (x - μ_c) / σ_c, 其中c为通道索引
    """
    
    def __init__(
        self, 
        mean: Union[np.ndarray, torch.Tensor, None] = None,
        std: Union[np.ndarray, torch.Tensor, None] = None,
        eps: float = 1e-8
    ):
        """
        Args:
            mean: 各通道均值, shape [C] 或 [C, 1, 1]
            std: 各通道标准差, shape [C] 或 [C, 1, 1]
            eps: 防止除零的小常数
        """
        self.mean = mean
        self.std = std
        self.eps = eps
        self._fitted = mean is not None and std is not None
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'LevelwiseNormalizer':
        """
        从数据计算各通道的均值和标准差
        
        Args:
            data: 输入数据, shape [..., C, H, W] 或合并后的 [N, C, H, W]
        
        Returns:
            self (支持链式调用)
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        # 确保数据至少是4D: [N, C, H, W]
        if data.ndim == 3:
            data = data[np.newaxis, ...]  # [C, H, W] -> [1, C, H, W]
        
        n_channels = data.shape[-3]  # C维度
        
        # 沿着N, H, W维度计算统计量，保留C维度
        # data shape: [N, C, H, W]
        # 展平为 [N*H*W, C] 来计算
        data_flat = data.transpose(0, 2, 3, 1).reshape(-1, n_channels)  # [N*H*W, C]
        
        # 处理NaN值：使用nanmean和nanstd
        self.mean = np.nanmean(data_flat, axis=0)  # [C]
        self.std = np.nanstd(data_flat, axis=0)    # [C]
        
        # 防止标准差为零
        self.std = np.where(self.std < self.eps, self.eps, self.std)
        
        self._fitted = True
        return self
    
    def transform(
        self, 
        x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        应用逐通道标准化
        
        Args:
            x: 输入数据, shape [..., C, H, W]
        
        Returns:
            标准化后的数据, 保持输入类型
        """
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted! Call fit() first or provide mean/std.")
        
        is_tensor = isinstance(x, torch.Tensor)
        device = x.device if is_tensor else None
        
        if is_tensor:
            mean = torch.tensor(self.mean, dtype=x.dtype, device=device)
            std = torch.tensor(self.std, dtype=x.dtype, device=device)
        else:
            mean = self.mean
            std = self.std
        
        # 调整shape以支持广播: [C] -> [C, 1, 1]
        while mean.ndim < x.ndim:
            mean = mean[..., np.newaxis] if not is_tensor else mean.unsqueeze(-1)
            std = std[..., np.newaxis] if not is_tensor else std.unsqueeze(-1)
        
        return (x - mean) / std
    
    def inverse_transform(
        self, 
        x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        逆标准化（用于恢复原始物理单位）
        
        Args:
            x: 标准化后的数据
        
        Returns:
            原始尺度的数据
        """
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted!")
        
        is_tensor = isinstance(x, torch.Tensor)
        device = x.device if is_tensor else None
        
        if is_tensor:
            mean = torch.tensor(self.mean, dtype=x.dtype, device=device)
            std = torch.tensor(self.std, dtype=x.dtype, device=device)
        else:
            mean = self.mean
            std = self.std
        
        while mean.ndim < x.ndim:
            mean = mean[..., np.newaxis] if not is_tensor else mean.unsqueeze(-1)
            std = std[..., np.newaxis] if not is_tensor else std.unsqueeze(-1)
        
        return x * std + mean
    
    def state_dict(self) -> Dict[str, np.ndarray]:
        """导出状态（用于保存）"""
        return {'mean': self.mean, 'std': self.std, 'eps': self.eps}
    
    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        """加载状态"""
        self.mean = state['mean']
        self.std = state['std']
        self.eps = state.get('eps', 1e-8)
        self._fitted = True


# =============================================================================
# Part 2: 卫星-ERA5数据集类 (SatelliteERA5Dataset)
# =============================================================================

class SatelliteERA5Dataset(Dataset):
    """
    卫星数据同化任务的PyTorch Dataset
    
    核心功能:
    1. 动态掩码生成: 从X_obs的NaN位置生成validity_mask
    2. 逐通道Z-Score标准化: 保留各层级的物理信息
    3. NaN安全填充: 标准化后用0填充NaN，但保留掩码信息
    
    输出字典:
    {
        'obs': torch.Tensor,      # 观测场 (标准化+NaN填0), [17, H, W]
        'bkg': torch.Tensor,      # 背景场 (标准化), [37, H, W]  
        'mask': torch.Tensor,     # 有效性掩码, [1, H, W] or [17, H, W]
        'target': torch.Tensor    # 目标场 (标准化), [37, H, W]
    }
    """
    
    def __init__(
        self,
        obs_data: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        bkg_data: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        target_data: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
        bkg_mean: Optional[np.ndarray] = None,
        bkg_std: Optional[np.ndarray] = None,
        target_mean: Optional[np.ndarray] = None,
        target_std: Optional[np.ndarray] = None,
        compute_stats: bool = False,
        mask_mode: str = 'any',  # 'any': 任一通道NaN则masked, 'all': 全部NaN才masked
        fill_value: float = 0.0,
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            obs_data: 观测数据 (卫星亮温), shape [N, 17, H, W]
            bkg_data: 背景数据 (ERA5), shape [N, 37, H, W]
            target_data: 目标数据 (ERA5分析场), shape [N, 37, H, W]
            obs_mean: 观测各通道均值, shape [17]
            obs_std: 观测各通道标准差, shape [17]
            bkg_mean: 背景各层均值, shape [37]
            bkg_std: 背景各层标准差, shape [37]
            target_mean: 目标各层均值 (通常与bkg相同), shape [37]
            target_std: 目标各层标准差, shape [37]
            compute_stats: 是否从数据计算统计量 (若为True且未提供mean/std)
            mask_mode: 掩码生成模式
                - 'any': 任一通道为NaN则该像素被masked
                - 'all': 所有通道均为NaN才被masked  
                - 'channel': 逐通道生成mask [17, H, W]
            fill_value: NaN填充值 (标准化后)
            dtype: 输出张量数据类型
        """
        super().__init__()
        
        # 转换为numpy数组（如果需要）
        self.obs_data = self._to_numpy(obs_data)      # [N, 17, H, W]
        self.bkg_data = self._to_numpy(bkg_data)      # [N, 37, H, W]
        self.target_data = self._to_numpy(target_data)  # [N, 37, H, W]
        
        # 验证数据形状
        self._validate_shapes()
        
        self.n_samples = self.obs_data.shape[0]
        self.n_obs_channels = self.obs_data.shape[1]   # 17
        self.n_bkg_levels = self.bkg_data.shape[1]     # 37
        self.mask_mode = mask_mode
        self.fill_value = fill_value
        self.dtype = dtype
        
        # 初始化标准化器
        self.obs_normalizer = LevelwiseNormalizer(mean=obs_mean, std=obs_std)
        self.bkg_normalizer = LevelwiseNormalizer(mean=bkg_mean, std=bkg_std)
        
        # target通常与bkg使用相同统计量（同为ERA5温度场）
        if target_mean is None and bkg_mean is not None:
            target_mean = bkg_mean
        if target_std is None and bkg_std is not None:
            target_std = bkg_std
        self.target_normalizer = LevelwiseNormalizer(mean=target_mean, std=target_std)
        
        # 如果需要，从数据计算统计量
        if compute_stats:
            self._compute_statistics()
        
        # 验证标准化器已就绪
        self._check_normalizers()
        
        print(f"[SatelliteERA5Dataset] 初始化完成:")
        print(f"  样本数: {self.n_samples}")
        print(f"  观测通道: {self.n_obs_channels}, 背景层数: {self.n_bkg_levels}")
        print(f"  掩码模式: {self.mask_mode}")
        print(f"  观测统计: μ∈[{self.obs_normalizer.mean.min():.2f}, {self.obs_normalizer.mean.max():.2f}], "
              f"σ∈[{self.obs_normalizer.std.min():.2f}, {self.obs_normalizer.std.max():.2f}]")
        print(f"  背景统计: μ∈[{self.bkg_normalizer.mean.min():.2f}, {self.bkg_normalizer.mean.max():.2f}], "
              f"σ∈[{self.bkg_normalizer.std.min():.2f}, {self.bkg_normalizer.std.max():.2f}]")
    
    @staticmethod
    def _to_numpy(data: Union[np.ndarray, List, torch.Tensor]) -> np.ndarray:
        """转换为numpy数组"""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return np.stack(data, axis=0)
        return np.asarray(data)
    
    def _validate_shapes(self) -> None:
        """验证数据形状一致性"""
        assert self.obs_data.ndim == 4, f"obs_data must be 4D, got {self.obs_data.ndim}D"
        assert self.bkg_data.ndim == 4, f"bkg_data must be 4D, got {self.bkg_data.ndim}D"
        assert self.target_data.ndim == 4, f"target_data must be 4D, got {self.target_data.ndim}D"
        
        N_obs, _, H_obs, W_obs = self.obs_data.shape
        N_bkg, _, H_bkg, W_bkg = self.bkg_data.shape
        N_tgt, _, H_tgt, W_tgt = self.target_data.shape
        
        assert N_obs == N_bkg == N_tgt, \
            f"Sample count mismatch: obs={N_obs}, bkg={N_bkg}, target={N_tgt}"
        assert (H_obs, W_obs) == (H_bkg, W_bkg) == (H_tgt, W_tgt), \
            f"Spatial dimension mismatch: obs=({H_obs},{W_obs}), bkg=({H_bkg},{W_bkg}), target=({H_tgt},{W_tgt})"
    
    def _compute_statistics(self) -> None:
        """从数据计算统计量"""
        print("[SatelliteERA5Dataset] 计算逐通道统计量...")
        
        if not self.obs_normalizer._fitted:
            self.obs_normalizer.fit(self.obs_data)
            print(f"  ✓ 观测统计量已计算 (17 channels)")
        
        if not self.bkg_normalizer._fitted:
            self.bkg_normalizer.fit(self.bkg_data)
            print(f"  ✓ 背景统计量已计算 (37 levels)")
        
        if not self.target_normalizer._fitted:
            self.target_normalizer.fit(self.target_data)
            print(f"  ✓ 目标统计量已计算 (37 levels)")
    
    def _check_normalizers(self) -> None:
        """检查所有标准化器是否就绪"""
        if not self.obs_normalizer._fitted:
            raise ValueError("Observation normalizer not fitted! Provide obs_mean/obs_std or set compute_stats=True")
        if not self.bkg_normalizer._fitted:
            raise ValueError("Background normalizer not fitted! Provide bkg_mean/bkg_std or set compute_stats=True")
        if not self.target_normalizer._fitted:
            raise ValueError("Target normalizer not fitted! Provide target_mean/target_std or set compute_stats=True")
    
    def _generate_mask(self, obs: np.ndarray) -> np.ndarray:
        """
        从观测数据生成有效性掩码
        
        Args:
            obs: 观测数据, shape [C, H, W]
        
        Returns:
            mask: 有效性掩码 (1=有效, 0=无效/NaN)
                - mode='any' or 'all': shape [1, H, W]
                - mode='channel': shape [C, H, W]
        """
        # 检测NaN位置: True表示NaN
        nan_mask = np.isnan(obs)  # [C, H, W]
        
        if self.mask_mode == 'any':
            # 任一通道为NaN则该像素无效
            invalid = nan_mask.any(axis=0, keepdims=True)  # [1, H, W]
            mask = (~invalid).astype(np.float32)  # [1, H, W]
        elif self.mask_mode == 'all':
            # 所有通道均为NaN才无效
            invalid = nan_mask.all(axis=0, keepdims=True)  # [1, H, W]
            mask = (~invalid).astype(np.float32)  # [1, H, W]
        elif self.mask_mode == 'channel':
            # 逐通道掩码
            mask = (~nan_mask).astype(np.float32)  # [C, H, W]
        else:
            raise ValueError(f"Unknown mask_mode: {self.mask_mode}")
        
        return mask
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            dict: {
                'obs': [17, H, W] - 标准化后的观测场，NaN填0
                'bkg': [37, H, W] - 标准化后的背景场
                'mask': [1, H, W] 或 [17, H, W] - 有效性掩码
                'target': [37, H, W] - 标准化后的目标场
            }
        """
        # 提取原始数据
        obs_raw = self.obs_data[idx].copy()       # [17, H, W]
        bkg_raw = self.bkg_data[idx].copy()       # [37, H, W]
        target_raw = self.target_data[idx].copy() # [37, H, W]
        
        # === Step 1: 生成掩码（在填充NaN之前！） ===
        mask = self._generate_mask(obs_raw)  # [1, H, W] or [17, H, W]
        
        # === Step 2: 标准化 ===
        # 标准化时暂时忽略NaN警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            obs_norm = self.obs_normalizer.transform(obs_raw)    # [17, H, W]
            bkg_norm = self.bkg_normalizer.transform(bkg_raw)    # [37, H, W]
            target_norm = self.target_normalizer.transform(target_raw)  # [37, H, W]
        
        # === Step 3: 填充NaN（标准化后！） ===
        # 用fill_value (默认0.0) 填充NaN，这在标准化空间中是合理的
        obs_norm = np.nan_to_num(obs_norm, nan=self.fill_value)
        
        # === Step 4: 转换为PyTorch张量 ===
        obs_tensor = torch.tensor(obs_norm, dtype=self.dtype)       # [17, H, W]
        bkg_tensor = torch.tensor(bkg_norm, dtype=self.dtype)       # [37, H, W]
        mask_tensor = torch.tensor(mask, dtype=self.dtype)          # [1, H, W] or [17, H, W]
        target_tensor = torch.tensor(target_norm, dtype=self.dtype) # [37, H, W]
        
        return {
            'obs': obs_tensor,
            'bkg': bkg_tensor,
            'mask': mask_tensor,
            'target': target_tensor
        }
    
    def get_normalizers(self) -> Dict[str, LevelwiseNormalizer]:
        """获取所有标准化器（用于推理时的逆变换）"""
        return {
            'obs': self.obs_normalizer,
            'bkg': self.bkg_normalizer,
            'target': self.target_normalizer
        }
    
    @staticmethod
    def compute_statistics_from_files(
        obs_paths: List[str],
        bkg_paths: List[str],
        n_obs_channels: int = 17,
        n_bkg_levels: int = 37
    ) -> Dict[str, np.ndarray]:
        """
        从文件列表计算统计量（用于大数据集）
        
        使用Welford算法进行在线计算，避免内存溢出
        """
        # 初始化在线统计
        obs_count = np.zeros(n_obs_channels)
        obs_mean = np.zeros(n_obs_channels)
        obs_M2 = np.zeros(n_obs_channels)
        
        bkg_count = np.zeros(n_bkg_levels)
        bkg_mean = np.zeros(n_bkg_levels)
        bkg_M2 = np.zeros(n_bkg_levels)
        
        # Welford在线算法
        def update_stats(count, mean, M2, new_values, channel_idx):
            """更新单通道统计量"""
            valid_values = new_values[~np.isnan(new_values)]
            for x in valid_values:
                count[channel_idx] += 1
                delta = x - mean[channel_idx]
                mean[channel_idx] += delta / count[channel_idx]
                delta2 = x - mean[channel_idx]
                M2[channel_idx] += delta * delta2
        
        print(f"计算统计量: {len(obs_paths)} 个文件...")
        
        for i, (obs_path, bkg_path) in enumerate(zip(obs_paths, bkg_paths)):
            if (i + 1) % 100 == 0:
                print(f"  处理 {i+1}/{len(obs_paths)}...")
            
            obs_data = np.load(obs_path)  # [C, H, W]
            bkg_data = np.load(bkg_path)  # [L, H, W]
            
            for c in range(n_obs_channels):
                update_stats(obs_count, obs_mean, obs_M2, 
                           obs_data[c].flatten(), c)
            
            for l in range(n_bkg_levels):
                update_stats(bkg_count, bkg_mean, bkg_M2,
                           bkg_data[l].flatten(), l)
        
        # 计算最终标准差
        obs_std = np.sqrt(obs_M2 / (obs_count - 1))
        bkg_std = np.sqrt(bkg_M2 / (bkg_count - 1))
        
        return {
            'obs_mean': obs_mean,
            'obs_std': obs_std,
            'bkg_mean': bkg_mean,
            'bkg_std': bkg_std
        }


# =============================================================================
# Part 3: 挤压-激励模块 (Squeeze-and-Excitation Block)
# =============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (通道注意力)
    
    论文: "Squeeze-and-Excitation Networks" (Hu et al., 2018)
    
    物理意义:
    - 不同卫星通道对不同气压层的敏感性不同
    - SE模块学习自适应地重标定通道权重
    - 模拟辐射传输方程中的权重函数
    
    流程:
    1. Squeeze: 全局平均池化 [B, C, H, W] -> [B, C, 1, 1]
    2. Excitation: FC -> ReLU -> FC -> Sigmoid
    3. Scale: 逐通道乘法
    """
    
    def __init__(
        self, 
        channels: int, 
        reduction: int = 16,
        activation: str = 'relu'
    ):
        """
        Args:
            channels: 输入通道数
            reduction: 中间层压缩比
            activation: 激活函数 ('relu', 'gelu', 'silu')
        """
        super().__init__()
        
        reduced_channels = max(channels // reduction, 4)  # 至少4个通道
        
        # 激活函数选择
        act_fn = {
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True)
        }.get(activation, nn.ReLU(inplace=True))
        
        self.squeeze_excite = nn.Sequential(
            # Squeeze: Global Average Pooling
            nn.AdaptiveAvgPool2d(1),  # [B, C, H, W] -> [B, C, 1, 1]
            # Excitation: 两层全连接
            nn.Conv2d(channels, reduced_channels, 1, bias=True),  # [B, C, 1, 1] -> [B, C/r, 1, 1]
            act_fn,
            nn.Conv2d(reduced_channels, channels, 1, bias=True),  # [B, C/r, 1, 1] -> [B, C, 1, 1]
            nn.Sigmoid()  # 输出注意力权重 [0, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            重标定后的特征 [B, C, H, W]
        """
        # [B, C, H, W] -> [B, C, 1, 1] -> [B, C, H, W]
        attention = self.squeeze_excite(x)  # [B, C, 1, 1]
        return x * attention  # 广播乘法


# =============================================================================
# Part 4: 光谱适配器茎干模块 (SpectralAdapterStem)
# =============================================================================

class SpectralAdapterStem(nn.Module):
    """
    光谱适配器茎干模块 - 创新的多源数据融合入口
    
    设计假设:
    - 卫星亮温(BT)与大气温度廓线之间的关系是非线性的
    - 这种非线性由辐射传输方程(RTE)控制
    - 1x1卷积+SE模块可以学习RTE的逆映射
    
    核心创新:
    1. 辐射-状态投影 (Radiance-to-State Projection):
       - 将17通道亮温投影到高维潜在空间
       - 模拟逆辐射传输模型(RTM)
    
    2. 不确定性门控融合 (Uncertainty-Gated Fusion):
       - 使用validity_mask对卫星特征进行门控
       - 缺测区域严格回退到背景场先验
       - 公式: F_fused = Conv(X_bkg) + α · (Project(X_obs) ⊙ Mask)
    
    3. 可学习融合权重:
       - α为可学习标量或向量
       - 允许网络自动调整观测与背景的相对权重
    
    数据流:
    X_obs [B, 17, H, W] ---> 1x1 Conv -> GELU -> SE -> Project [B, C_lat, H, W]
                                                           |
                                                           v (× Mask)
    X_bkg [B, 37, H, W] ---> 1x1 Conv ----------------> + α·(...) -> F_fused [B, C_lat, H, W]
    """
    
    def __init__(
        self,
        obs_channels: int = 17,
        bkg_channels: int = 37,
        latent_channels: int = 64,
        se_reduction: int = 8,
        fusion_mode: str = 'learnable_scalar',  # 'learnable_scalar', 'learnable_channel', 'fixed'
        initial_alpha: float = 1.0,
        use_bkg_projection: bool = True
    ):
        """
        Args:
            obs_channels: 观测通道数 (默认17: FY-3F MWTS)
            bkg_channels: 背景层数 (默认37: ERA5气压层)
            latent_channels: 融合后的潜在特征维度
            se_reduction: SE模块的压缩比
            fusion_mode: 融合模式
                - 'learnable_scalar': α为单一可学习标量
                - 'learnable_channel': α为逐通道可学习向量 [C_lat]
                - 'fixed': α固定为initial_alpha
            initial_alpha: α的初始值
            use_bkg_projection: 是否对背景场进行投影（否则直接pad/crop）
        """
        super().__init__()
        
        self.obs_channels = obs_channels
        self.bkg_channels = bkg_channels
        self.latent_channels = latent_channels
        self.fusion_mode = fusion_mode
        
        # =====================================================================
        # 观测投影分支: X_obs [B, 17, H, W] -> [B, C_lat, H, W]
        # 物理含义: 学习逆辐射传输模型 (RTM^{-1})
        # =====================================================================
        self.obs_projection = nn.Sequential(
            # 1x1卷积: 光谱混合/通道扩展
            # [B, 17, H, W] -> [B, C_lat, H, W]
            nn.Conv2d(obs_channels, latent_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_channels),  # 稳定训练
            nn.GELU(),  # 平滑的非线性激活
            
            # SE模块: 学习通道间的依赖关系
            # 物理意义: 学习不同卫星通道对反演的贡献权重
            SEBlock(latent_channels, reduction=se_reduction, activation='gelu'),
            
            # 额外的非线性变换
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.GELU()
        )
        
        # =====================================================================
        # 背景投影分支: X_bkg [B, 37, H, W] -> [B, C_lat, H, W]
        # =====================================================================
        if use_bkg_projection:
            self.bkg_projection = nn.Sequential(
                nn.Conv2d(bkg_channels, latent_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(latent_channels),
                nn.GELU()
            )
        else:
            # 如果bkg_channels == latent_channels，可以跳过投影
            assert bkg_channels == latent_channels, \
                "bkg_channels must equal latent_channels when use_bkg_projection=False"
            self.bkg_projection = nn.Identity()
        
        # =====================================================================
        # 融合权重 α
        # =====================================================================
        if fusion_mode == 'learnable_scalar':
            # 单一可学习标量
            self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        elif fusion_mode == 'learnable_channel':
            # 逐通道可学习权重 [C_lat, 1, 1]
            self.alpha = nn.Parameter(torch.full((latent_channels, 1, 1), initial_alpha))
        elif fusion_mode == 'fixed':
            # 固定权重
            self.register_buffer('alpha', torch.tensor(initial_alpha))
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        
        # =====================================================================
        # 融合后的优化
        # =====================================================================
        self.fusion_refine = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.GELU()
        )
        
        self._init_weights()
        
        # 打印模块信息
        print(f"[SpectralAdapterStem] 初始化完成:")
        print(f"  观测: {obs_channels}ch -> 投影: {latent_channels}ch")
        print(f"  背景: {bkg_channels}ch -> 投影: {latent_channels}ch")
        print(f"  融合模式: {fusion_mode}, 初始α={initial_alpha}")
    
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
    
    def forward(
        self, 
        obs: torch.Tensor, 
        bkg: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播: 融合观测场和背景场
        
        Args:
            obs: 观测场 (卫星亮温), [B, 17, H, W], 已标准化，NaN已填0
            bkg: 背景场 (ERA5), [B, 37, H, W], 已标准化
            mask: 有效性掩码, [B, 1, H, W] 或 [B, 17, H, W]
                  1=有效观测, 0=缺测/NaN
        
        Returns:
            fused: 融合特征, [B, C_lat, H, W]
        """
        B, _, H, W = obs.shape
        
        # =================================================================
        # Step 1: 观测投影 (逆RTM)
        # [B, 17, H, W] -> [B, C_lat, H, W]
        # =================================================================
        obs_features = self.obs_projection(obs)  # [B, C_lat, H, W]
        
        # =================================================================
        # Step 2: 掩码门控
        # 确保mask能够广播到obs_features的shape
        # =================================================================
        if mask.shape[1] == 1:
            # [B, 1, H, W] -> 直接广播
            mask_expanded = mask  
        elif mask.shape[1] == self.obs_channels:
            # [B, 17, H, W] -> 需要投影到C_lat或取平均
            # 使用平均作为综合有效性指标
            mask_expanded = mask.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
        
        # 门控: 缺测区域的观测特征被置为0
        obs_gated = obs_features * mask_expanded  # [B, C_lat, H, W]
        
        # =================================================================
        # Step 3: 背景投影
        # [B, 37, H, W] -> [B, C_lat, H, W]
        # =================================================================
        bkg_features = self.bkg_projection(bkg)  # [B, C_lat, H, W]
        
        # =================================================================
        # Step 4: 不确定性门控融合
        # F_fused = bkg_features + α · obs_gated
        # 
        # 物理意义:
        # - 在有观测处: F_fused ≈ bkg + α·obs（观测修正背景）
        # - 在无观测处: F_fused = bkg（回退到先验）
        # =================================================================
        fused = bkg_features + self.alpha * obs_gated  # [B, C_lat, H, W]
        
        # =================================================================
        # Step 5: 融合后优化
        # 3x3卷积进行空间特征提取
        # =================================================================
        fused = self.fusion_refine(fused)  # [B, C_lat, H, W]
        
        return fused
    
    def get_alpha(self) -> torch.Tensor:
        """获取当前的融合权重α（用于监控训练）"""
        return self.alpha.detach().clone()


# =============================================================================
# Part 5: 完整的SpectralAdapter模块 (带更多功能)
# =============================================================================

class SpectralAdapterFull(nn.Module):
    """
    完整版光谱适配器 - 包含更多高级功能
    
    额外特性:
    1. 多尺度特征融合
    2. 残差连接
    3. 交叉注意力（可选）
    """
    
    def __init__(
        self,
        obs_channels: int = 17,
        bkg_channels: int = 37,
        latent_channels: int = 64,
        se_reduction: int = 8,
        use_cross_attention: bool = False,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        
        # 基础SpectralAdapterStem
        self.stem = SpectralAdapterStem(
            obs_channels=obs_channels,
            bkg_channels=bkg_channels,
            latent_channels=latent_channels,
            se_reduction=se_reduction,
            fusion_mode='learnable_channel'
        )
        
        # 可选: 交叉注意力机制
        if use_cross_attention:
            self.cross_attn = SpatialCrossAttention(
                dim=latent_channels,
                n_heads=n_heads,
                dropout=dropout
            )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, 1),
            nn.BatchNorm2d(latent_channels),
            nn.GELU()
        )
    
    def forward(
        self, 
        obs: torch.Tensor, 
        bkg: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            obs: [B, 17, H, W]
            bkg: [B, 37, H, W]
            mask: [B, 1, H, W] or [B, 17, H, W]
        
        Returns:
            [B, C_lat, H, W]
        """
        # 基础融合
        fused = self.stem(obs, bkg, mask)  # [B, C_lat, H, W]
        
        # 可选: 交叉注意力增强
        if self.use_cross_attention:
            fused = self.cross_attn(fused, mask)
        
        # 输出
        return self.output_proj(fused)


class SpatialCrossAttention(nn.Module):
    """
    空间交叉注意力模块（简化版）
    
    用于增强观测-背景的空间一致性
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Layer Norm (在通道维度)
        self.norm = nn.GroupNorm(1, dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            mask: [B, 1, H, W]
        """
        B, C, H, W = x.shape
        
        # 残差
        residual = x
        x = self.norm(x)
        
        # QKV投影
        qkv = self.qkv(x)  # [B, 3C, H, W]
        qkv = qkv.reshape(B, 3, self.n_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, HW, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, HW, HW]
        
        # 掩码：无效位置的注意力权重设为极小值
        mask_flat = mask.reshape(B, 1, 1, H * W)  # [B, 1, 1, HW]
        attn = attn.masked_fill(mask_flat == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        out = attn @ v  # [B, heads, HW, head_dim]
        out = out.transpose(2, 3).reshape(B, C, H, W)
        
        out = self.proj(out)
        
        return residual + out


# =============================================================================
# Part 6: 使用示例和测试
# =============================================================================

def create_synthetic_data(
    n_samples: int = 100,
    n_obs_channels: int = 17,
    n_bkg_levels: int = 37,
    height: int = 64,
    width: int = 64,
    nan_ratio: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    创建合成测试数据
    
    Args:
        n_samples: 样本数
        n_obs_channels: 观测通道数
        n_bkg_levels: 背景层数
        height, width: 空间维度
        nan_ratio: NaN占比（模拟轨道空隙）
    
    Returns:
        obs_data, bkg_data, target_data
    """
    np.random.seed(42)
    
    # 观测数据: 亮温范围约 150-300K
    obs_data = np.random.normal(230, 30, (n_samples, n_obs_channels, height, width))
    
    # 添加通道间差异（物理真实性）
    for c in range(n_obs_channels):
        # 高层通道温度低，低层通道温度高
        obs_data[:, c, :, :] += (c - n_obs_channels // 2) * 5
    
    # 添加NaN（模拟轨道空隙）
    nan_mask = np.random.random((n_samples, 1, height, width)) < nan_ratio
    nan_mask = np.broadcast_to(nan_mask, obs_data.shape).copy()
    obs_data[nan_mask] = np.nan
    
    # 背景数据: ERA5温度 约 200-310K
    bkg_data = np.random.normal(260, 25, (n_samples, n_bkg_levels, height, width))
    for l in range(n_bkg_levels):
        # 气压层差异
        bkg_data[:, l, :, :] += (l - n_bkg_levels // 2) * 2
    
    # 目标数据: 背景 + 小扰动
    target_data = bkg_data + np.random.normal(0, 2, bkg_data.shape)
    
    return obs_data.astype(np.float32), bkg_data.astype(np.float32), target_data.astype(np.float32)


def test_dataset():
    """测试数据集类"""
    print("=" * 70)
    print("测试 SatelliteERA5Dataset")
    print("=" * 70)
    
    # 创建合成数据
    obs, bkg, target = create_synthetic_data(n_samples=10)
    
    # 创建数据集（自动计算统计量）
    dataset = SatelliteERA5Dataset(
        obs_data=obs,
        bkg_data=bkg,
        target_data=target,
        compute_stats=True,
        mask_mode='any'
    )
    
    # 测试__getitem__
    sample = dataset[0]
    print(f"\n样本数据形状:")
    for key, val in sample.items():
        print(f"  {key}: {val.shape}, dtype={val.dtype}")
    
    # 检查掩码
    mask = sample['mask']
    valid_ratio = mask.mean().item()
    print(f"\n有效数据比例: {valid_ratio:.2%}")
    
    # 检查NaN填充
    obs_tensor = sample['obs']
    assert not torch.isnan(obs_tensor).any(), "观测数据中仍存在NaN！"
    print("✓ NaN已正确填充")
    
    # 测试DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    print(f"\n批次数据形状:")
    for key, val in batch.items():
        print(f"  {key}: {val.shape}")
    
    return dataset


def test_spectral_adapter():
    """测试SpectralAdapterStem模块"""
    print("\n" + "=" * 70)
    print("测试 SpectralAdapterStem")
    print("=" * 70)
    
    # 创建模块
    adapter = SpectralAdapterStem(
        obs_channels=17,
        bkg_channels=37,
        latent_channels=64,
        fusion_mode='learnable_channel'
    )
    
    # 打印参数量
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"\n模块参数量: {n_params:,}")
    
    # 测试前向传播
    B, H, W = 4, 64, 64
    obs = torch.randn(B, 17, H, W)
    bkg = torch.randn(B, 37, H, W)
    mask = (torch.rand(B, 1, H, W) > 0.3).float()  # 30% 缺测
    
    print(f"\n输入形状:")
    print(f"  obs: {obs.shape}")
    print(f"  bkg: {bkg.shape}")
    print(f"  mask: {mask.shape}")
    
    # 前向传播
    output = adapter(obs, bkg, mask)
    print(f"\n输出形状: {output.shape}")
    
    # 验证掩码效果
    # 在mask=0的位置，观测贡献应该为0
    alpha = adapter.get_alpha()
    print(f"\n融合权重α形状: {alpha.shape}")
    print(f"融合权重α范围: [{alpha.min().item():.4f}, {alpha.max().item():.4f}]")
    
    # 测试梯度
    loss = output.mean()
    loss.backward()
    print("\n✓ 梯度反向传播成功")
    
    return adapter


def test_full_pipeline():
    """测试完整流水线"""
    print("\n" + "=" * 70)
    print("测试完整流水线: Dataset + SpectralAdapterStem")
    print("=" * 70)
    
    # 1. 准备数据
    obs, bkg, target = create_synthetic_data(n_samples=32, height=64, width=64)
    
    # 2. 创建数据集
    dataset = SatelliteERA5Dataset(
        obs_data=obs,
        bkg_data=bkg,
        target_data=target,
        compute_stats=True
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 3. 创建模型
    adapter = SpectralAdapterStem(
        obs_channels=17,
        bkg_channels=37,
        latent_channels=64
    )
    
    # 4. 前向传播测试
    batch = next(iter(dataloader))
    output = adapter(batch['obs'], batch['bkg'], batch['mask'])
    
    print(f"\n完整流水线测试:")
    print(f"  输入观测: {batch['obs'].shape}")
    print(f"  输入背景: {batch['bkg'].shape}")
    print(f"  输入掩码: {batch['mask'].shape}")
    print(f"  融合输出: {output.shape}")
    print(f"  目标形状: {batch['target'].shape}")
    
    # 5. 简单训练步骤
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    
    # 用一个简单的1x1卷积将融合特征映射回目标维度
    head = nn.Conv2d(64, 37, 1)
    
    for epoch in range(3):
        for batch in dataloader:
            optimizer.zero_grad()
            
            fused = adapter(batch['obs'], batch['bkg'], batch['mask'])
            pred = head(fused)
            
            # MSE损失
            loss = F.mse_loss(pred, batch['target'])
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    
    print("\n✓ 完整流水线测试通过!")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    # 运行所有测试
    dataset = test_dataset()
    adapter = test_spectral_adapter()
    test_full_pipeline()
    
    print("\n" + "=" * 70)
    print("所有测试通过! ✓")
    print("=" * 70)
