#!/usr/bin/env python3
"""
===============================================================================
卫星数据同化数据加载器 - 支持配准后的.npy/.h5格式
Data Loader for Collocated Satellite Data
===============================================================================

支持的数据格式:
1. .npy文件 (collocation_YYYYMMDD_HHMM_X.npy, _Y.npy)
2. .h5文件 (collocation_YYYYMMDD_HHMM.h5)

数据结构:
- X: (N, 17) - FY-3F亮温，17通道
- Y: (N, 37) - ERA5温度廓线，37层

===============================================================================
"""

from __future__ import annotations

import os
import sys
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import warnings

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    warnings.warn("h5py not installed, .h5 file support disabled")


# =============================================================================
# Part 1: 配置类
# =============================================================================

@dataclass
class CollocatedDataConfig:
    """配准数据配置"""
    n_obs_channels: int = 17          # FY-3F MWTS通道数
    n_bkg_levels: int = 37            # ERA5气压层数
    dtype: torch.dtype = torch.float32
    

# =============================================================================
# Part 2: 标准化器
# =============================================================================

class PointwiseNormalizer:
    """
    逐通道Z-Score标准化器 (适用于点数据)
    """
    
    def __init__(
        self, 
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        eps: float = 1e-8,
        name: str = "normalizer"
    ):
        self.mean = mean
        self.std = std
        self.eps = eps
        self.name = name
        self._fitted = mean is not None and std is not None
    
    def fit(self, data: np.ndarray) -> 'PointwiseNormalizer':
        """从数据计算统计量 (data: [N, C])"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # 忽略NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean = np.nanmean(data, axis=0)
            self.std = np.nanstd(data, axis=0)
        
        self.std = np.where(self.std < self.eps, self.eps, self.std)
        self._fitted = True
        return self
    
    def partial_fit(self, data: np.ndarray) -> 'PointwiseNormalizer':
        """增量更新 (Welford算法)"""
        if not hasattr(self, '_count') or self._count is None:
            self._count = 0
            self._M1 = None
            self._M2 = None
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        for row in data:
            if np.any(np.isnan(row)):
                continue
            
            self._count += 1
            
            if self._M1 is None:
                self._M1 = row.copy()
                self._M2 = np.zeros_like(row)
            else:
                delta = row - self._M1
                self._M1 += delta / self._count
                delta2 = row - self._M1
                self._M2 += delta * delta2
        
        if self._count > 1:
            self.mean = self._M1
            self.std = np.sqrt(self._M2 / (self._count - 1))
            self.std = np.where(self.std < self.eps, self.eps, self.std)
            self._fitted = True
        
        return self
    
    def transform(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """应用标准化"""
        if not self._fitted:
            raise RuntimeError(f"[{self.name}] Normalizer not fitted!")
        
        is_tensor = isinstance(x, torch.Tensor)
        
        if is_tensor:
            mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.tensor(self.std, dtype=x.dtype, device=x.device)
        else:
            mean, std = self.mean, self.std
        
        return (x - mean) / std
    
    def inverse_transform(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """逆标准化"""
        if not self._fitted:
            raise RuntimeError(f"[{self.name}] Normalizer not fitted!")
        
        is_tensor = isinstance(x, torch.Tensor)
        
        if is_tensor:
            mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.tensor(self.std, dtype=x.dtype, device=x.device)
        else:
            mean, std = self.mean, self.std
        
        return x * std + mean
    
    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std, eps=self.eps, name=self.name)
    
    @classmethod
    def load(cls, path: str) -> 'PointwiseNormalizer':
        data = np.load(path)
        return cls(
            mean=data['mean'],
            std=data['std'],
            eps=float(data['eps']),
            name=str(data['name'])
        )
    
    def state_dict(self) -> Dict:
        return {'mean': self.mean, 'std': self.std, 'eps': self.eps, 'name': self.name}


# =============================================================================
# Part 3: 配准数据集
# =============================================================================

class CollocatedDataset(Dataset):
    """
    配准后数据集 - 支持.npy和.h5格式
    
    数据目录结构:
    data_root/
    ├── YYYY/
    │   ├── MM/
    │   │   ├── collocation_YYYYMMDD_HHMM_X.npy  # 亮温 (N, 17)
    │   │   ├── collocation_YYYYMMDD_HHMM_Y.npy  # 温度廓线 (N, 37)
    │   │   └── collocation_YYYYMMDD_HHMM.h5     # 或HDF5格式
    """
    
    def __init__(
        self,
        data_root: str,
        x_normalizer: Optional[PointwiseNormalizer] = None,
        y_normalizer: Optional[PointwiseNormalizer] = None,
        compute_stats: bool = False,
        year_filter: Optional[str] = None,
        month_filter: Optional[List[str]] = None,
        max_samples_per_file: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        verbose: bool = True
    ):
        """
        Args:
            data_root: 数据根目录
            x_normalizer: 输入标准化器
            y_normalizer: 输出标准化器
            compute_stats: 是否从数据计算统计量
            year_filter: 年份筛选 (如 '2024')
            month_filter: 月份筛选 (如 ['01', '02'])
            max_samples_per_file: 每个文件最大样本数 (用于调试)
            dtype: 数据类型
            verbose: 是否打印详细信息
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.dtype = dtype
        self.verbose = verbose
        self.max_samples_per_file = max_samples_per_file
        
        # 标准化器
        self.x_normalizer = x_normalizer or PointwiseNormalizer(name='X_bt')
        self.y_normalizer = y_normalizer or PointwiseNormalizer(name='Y_temp')
        
        # 查找数据文件
        self.file_pairs = self._find_file_pairs(year_filter, month_filter)
        
        if len(self.file_pairs) == 0:
            raise ValueError(f"未找到数据文件: {data_root}")
        
        if verbose:
            print(f"[CollocatedDataset] 找到 {len(self.file_pairs)} 个数据文件")
        
        # 加载所有数据
        self.X, self.Y = self._load_all_data()
        
        if verbose:
            print(f"[CollocatedDataset] 总样本数: {len(self.X):,}")
            print(f"  X shape: {self.X.shape}")
            print(f"  Y shape: {self.Y.shape}")
        
        # 计算或加载统计量
        if compute_stats:
            if verbose:
                print("[CollocatedDataset] 计算统计量...")
            self.x_normalizer.fit(self.X)
            self.y_normalizer.fit(self.Y)
        
        # 标准化
        if self.x_normalizer._fitted and self.y_normalizer._fitted:
            self.X_norm = self.x_normalizer.transform(self.X)
            self.Y_norm = self.y_normalizer.transform(self.Y)
            if verbose:
                print("[CollocatedDataset] ✓ 数据已标准化")
        else:
            self.X_norm = self.X
            self.Y_norm = self.Y
            if verbose:
                print("[CollocatedDataset] ⚠ 未标准化 (请设置 compute_stats=True)")
    
    def _find_file_pairs(
        self, 
        year_filter: Optional[str],
        month_filter: Optional[List[str]]
    ) -> List[Dict]:
        """查找所有数据文件对"""
        file_pairs = []
        
        # 遍历年份目录
        year_dirs = sorted(self.data_root.glob('*'))
        
        for year_dir in year_dirs:
            if not year_dir.is_dir():
                continue
            
            year = year_dir.name
            
            # 年份筛选
            if year_filter and year != year_filter:
                continue
            
            # 遍历月份目录
            for month_dir in sorted(year_dir.glob('*')):
                if not month_dir.is_dir():
                    continue
                
                month = month_dir.name
                
                # 月份筛选
                if month_filter and month not in month_filter:
                    continue
                
                # 查找.npy文件对
                x_files = sorted(month_dir.glob('*_X.npy'))
                for x_file in x_files:
                    y_file = Path(str(x_file).replace('_X.npy', '_Y.npy'))
                    if y_file.exists():
                        file_pairs.append({
                            'x_file': x_file,
                            'y_file': y_file,
                            'year': year,
                            'month': month
                        })
                
                # 查找.h5文件
                h5_files = sorted(month_dir.glob('*.h5'))
                for h5_file in h5_files:
                    # 检查是否已经添加了对应的npy文件
                    base_name = h5_file.stem
                    npy_already = any(
                        base_name in str(p['x_file']) 
                        for p in file_pairs
                    )
                    if not npy_already:
                        file_pairs.append({
                            'h5_file': h5_file,
                            'year': year,
                            'month': month
                        })
        
        return file_pairs
    
    def _load_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载所有数据"""
        X_list = []
        Y_list = []
        
        for i, fp in enumerate(self.file_pairs):
            try:
                X = None
                Y = None
                
                if 'x_file' in fp:
                    # 加载.npy格式
                    X = np.load(fp['x_file'])
                    Y = np.load(fp['y_file'])
                elif 'h5_file' in fp and HAS_H5PY:
                    # 加载.h5格式
                    with h5py.File(fp['h5_file'], 'r') as f:
                        if 'X_brightness_temperature' in f:
                            X = f['X_brightness_temperature'][:]
                            Y = f['Y_temperature_profile'][:]
                        else:
                            # 尝试其他键名
                            keys = list(f.keys())
                            X = f[keys[0]][:]
                            Y = f[keys[1]][:] if len(keys) > 1 else None
                elif 'h5_file' in fp and not HAS_H5PY:
                    if self.verbose and i == 0:
                        print("  ⚠ 跳过.h5文件 (h5py未安装)")
                    continue
                
                if X is not None and Y is not None:
                    # 限制样本数
                    if self.max_samples_per_file:
                        X = X[:self.max_samples_per_file]
                        Y = Y[:self.max_samples_per_file]
                    
                    X_list.append(X)
                    Y_list.append(Y)
                    
                    if self.verbose and (i + 1) % 10 == 0:
                        print(f"  加载进度: {i+1}/{len(self.file_pairs)}")
            
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ 加载失败: {fp} - {e}")
        
        if not X_list:
            raise ValueError("没有成功加载任何数据文件")
        
        X_all = np.concatenate(X_list, axis=0).astype(np.float32)
        Y_all = np.concatenate(Y_list, axis=0).astype(np.float32)
        
        # 移除含NaN的样本
        valid_mask = ~(np.any(np.isnan(X_all), axis=1) | np.any(np.isnan(Y_all), axis=1))
        X_clean = X_all[valid_mask]
        Y_clean = Y_all[valid_mask]
        
        if self.verbose:
            n_removed = len(X_all) - len(X_clean)
            if n_removed > 0:
                print(f"  移除 {n_removed} 个含NaN的样本")
        
        return X_clean, Y_clean
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'x': torch.tensor(self.X_norm[idx], dtype=self.dtype),
            'y': torch.tensor(self.Y_norm[idx], dtype=self.dtype),
            'x_raw': torch.tensor(self.X[idx], dtype=self.dtype),
            'y_raw': torch.tensor(self.Y[idx], dtype=self.dtype),
        }
    
    def get_normalizers(self) -> Dict[str, PointwiseNormalizer]:
        return {
            'x': self.x_normalizer,
            'y': self.y_normalizer
        }
    
    def save_normalizers(self, path: str) -> None:
        """保存标准化器"""
        np.savez(
            path,
            x_mean=self.x_normalizer.mean,
            x_std=self.x_normalizer.std,
            y_mean=self.y_normalizer.mean,
            y_std=self.y_normalizer.std
        )
        print(f"  ✓ 标准化器已保存: {path}")
    
    @classmethod
    def load_normalizers(cls, path: str) -> Tuple[PointwiseNormalizer, PointwiseNormalizer]:
        """加载标准化器"""
        data = np.load(path)
        x_norm = PointwiseNormalizer(data['x_mean'], data['x_std'], name='X_bt')
        y_norm = PointwiseNormalizer(data['y_mean'], data['y_std'], name='Y_temp')
        return x_norm, y_norm


# =============================================================================
# Part 4: 基于MLP的温度廓线反演模型
# =============================================================================

class ResidualMLPBlock(nn.Module):
    """残差MLP块"""
    
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        
        hidden_dim = dim * expansion
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class TemperatureProfileMLP(nn.Module):
    """
    温度廓线反演MLP模型
    
    输入: FY-3F亮温 (B, 17)
    输出: ERA5温度廓线 (B, 37)
    
    架构:
    - 输入投影
    - 多层残差MLP块
    - 输出投影
    """
    
    def __init__(
        self,
        in_channels: int = 17,
        out_channels: int = 37,
        hidden_dim: int = 256,
        n_layers: int = 6,
        expansion: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 残差MLP块
        self.layers = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, expansion, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels)
        )
        
        self._init_weights()
        self._print_info()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _print_info(self) -> None:
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[TemperatureProfileMLP] 参数量: {n_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 17) 亮温
        Returns:
            (B, 37) 温度廓线
        """
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.output_proj(x)


class PhysicsAwareMLP(nn.Module):
    """
    物理感知MLP - 包含SE注意力和残差连接
    
    创新点:
    1. 通道注意力: 学习各亮温通道的重要性
    2. 残差连接: 从输入亮温直接连接到输出
    3. 多尺度融合: 不同层捕捉不同尺度的特征
    """
    
    def __init__(
        self,
        in_channels: int = 17,
        out_channels: int = 37,
        hidden_dims: List[int] = [128, 256, 512, 256, 128],
        se_reduction: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 通道注意力 (SE-style)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // se_reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // se_reduction, in_channels),
            nn.Sigmoid()
        )
        
        # 编码器
        encoder_layers = []
        prev_dim = in_channels
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.GELU(),
            nn.Linear(hidden_dims[-1], out_channels)
        )
        
        # 残差连接: 从输入亮温直接投影到输出维度
        self.residual = nn.Linear(in_channels, out_channels)
        
        # 可学习的残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
        self._print_info()
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _print_info(self) -> None:
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[PhysicsAwareMLP] 参数量: {n_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 17) 亮温
        Returns:
            (B, 37) 温度廓线
        """
        # 通道注意力
        attn = self.channel_attention(x)
        x_attended = x * attn
        
        # 编码-解码
        features = self.encoder(x_attended)
        output = self.decoder(features)
        
        # 残差连接
        residual = self.residual(x)
        output = output + self.residual_weight * residual
        
        return output
    
    def get_channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        """获取通道注意力权重 (用于可解释性)"""
        return self.channel_attention(x)


# =============================================================================
# Part 5: 评估指标
# =============================================================================

class ProfileMetrics:
    """温度廓线评估指标"""
    
    def __init__(
        self,
        pressure_levels: Optional[np.ndarray] = None,
        stratosphere_threshold: float = 100.0
    ):
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
        self.strat_mask = self.pressure_levels <= stratosphere_threshold
        self.trop_mask = ~self.strat_mask
    
    def rmse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算RMSE"""
        return torch.sqrt(((pred - target) ** 2).mean())
    
    def mae(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算MAE"""
        return (pred - target).abs().mean()
    
    def levelwise_rmse(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        分层RMSE
        
        Args:
            pred: (N, 37)
            target: (N, 37)
        """
        # 各层RMSE
        levelwise = torch.sqrt(((pred - target) ** 2).mean(dim=0))
        
        # 全局
        global_rmse = self.rmse(pred, target)
        
        # 平流层/对流层
        strat_rmse = self.rmse(pred[:, self.strat_mask], target[:, self.strat_mask])
        trop_rmse = self.rmse(pred[:, self.trop_mask], target[:, self.trop_mask])
        
        return {
            'levelwise': levelwise,
            'global': global_rmse,
            'stratosphere': strat_rmse,
            'troposphere': trop_rmse,
            'pressure_levels': torch.tensor(self.pressure_levels)
        }
    
    def bias_profile(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算偏差廓线 (B, 37) -> (37,)"""
        return (pred - target).mean(dim=0)


# =============================================================================
# Part 6: 模型工厂
# =============================================================================

def create_mlp_model(
    model_name: str = 'physics_mlp',
    in_channels: int = 17,
    out_channels: int = 37,
    **kwargs
) -> nn.Module:
    """
    创建MLP模型
    
    Args:
        model_name: 模型名称
            - 'simple_mlp': 简单MLP
            - 'res_mlp': 残差MLP
            - 'physics_mlp': 物理感知MLP
    """
    if model_name == 'simple_mlp':
        return nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )
    
    elif model_name == 'res_mlp':
        return TemperatureProfileMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=kwargs.get('hidden_dim', 256),
            n_layers=kwargs.get('n_layers', 6),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    elif model_name == 'physics_mlp':
        return PhysicsAwareMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dims=kwargs.get('hidden_dims', [128, 256, 512, 256, 128]),
            se_reduction=kwargs.get('se_reduction', 4),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Part 7: 工具函数
# =============================================================================

def scan_data_directory(data_root: str, verbose: bool = True) -> Dict:
    """
    扫描数据目录，返回可用的年月信息
    """
    data_root = Path(data_root)
    
    if not data_root.exists():
        return {'error': f'目录不存在: {data_root}'}
    
    result = {
        'data_root': str(data_root),
        'years': {},
        'total_files': 0,
        'total_samples': 0
    }
    
    for year_dir in sorted(data_root.glob('*')):
        if not year_dir.is_dir():
            continue
        
        year = year_dir.name
        result['years'][year] = {}
        
        for month_dir in sorted(year_dir.glob('*')):
            if not month_dir.is_dir():
                continue
            
            month = month_dir.name
            
            # 统计文件
            npy_files = list(month_dir.glob('*_X.npy'))
            h5_files = list(month_dir.glob('*.h5'))
            
            n_files = len(npy_files) + len(h5_files)
            
            # 估计样本数 (从第一个文件)
            n_samples = 0
            if npy_files:
                try:
                    X = np.load(npy_files[0])
                    n_samples = len(X) * len(npy_files)
                except:
                    pass
            
            result['years'][year][month] = {
                'npy_files': len(npy_files),
                'h5_files': len(h5_files),
                'estimated_samples': n_samples
            }
            
            result['total_files'] += n_files
            result['total_samples'] += n_samples
    
    if verbose:
        print(f"\n📊 数据目录扫描: {data_root}")
        print(f"{'='*50}")
        
        for year, months in result['years'].items():
            print(f"\n📅 {year}年:")
            for month, info in months.items():
                print(f"   {month}月: {info['npy_files']} npy + {info['h5_files']} h5, "
                      f"约 {info['estimated_samples']:,} 样本")
        
        print(f"\n总计: {result['total_files']} 文件, 约 {result['total_samples']:,} 样本")
    
    return result


# =============================================================================
# Part 8: 测试
# =============================================================================

def test_with_synthetic_data():
    """使用合成数据测试"""
    print("=" * 70)
    print("使用合成数据测试")
    print("=" * 70)
    
    # 创建合成数据
    n_samples = 1000
    n_channels = 17
    n_levels = 37
    
    X = np.random.normal(230, 30, (n_samples, n_channels))
    Y = np.random.normal(260, 25, (n_samples, n_levels))
    
    # 标准化
    x_norm = PointwiseNormalizer(name='X').fit(X)
    y_norm = PointwiseNormalizer(name='Y').fit(Y)
    
    X_scaled = x_norm.transform(X)
    Y_scaled = y_norm.transform(Y)
    
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"X mean: {X.mean():.2f}, std: {X.std():.2f}")
    print(f"X_scaled mean: {X_scaled.mean():.4f}, std: {X_scaled.std():.4f}")
    
    # 测试模型
    for model_name in ['simple_mlp', 'res_mlp', 'physics_mlp']:
        print(f"\n--- 测试 {model_name} ---")
        model = create_mlp_model(model_name)
        
        x = torch.tensor(X_scaled[:8], dtype=torch.float32)
        y = model(x)
        print(f"  输出形状: {y.shape}")
        
        # 反向传播测试
        loss = y.mean()
        loss.backward()
        print(f"  ✓ 梯度计算成功")
    
    print("\n" + "=" * 70)
    print("测试通过!")
    print("=" * 70)


if __name__ == "__main__":
    test_with_synthetic_data()
