"""
卫星数据同化数据集
==================
支持:
1. FY-3F卫星亮温数据
2. ERA5背景场数据  
3. 3D-Var分析场作为标签
4. 观测掩膜

作者: 基于PAVMT-Unet改进
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import h5py
import xarray as xr


class SatelliteAssimilationDataset(Dataset):
    """
    卫星数据同化数据集
    
    返回:
        satellite: 卫星亮温 (n_channels, H, W)
        background: ERA5背景场 (n_levels, H, W)
        label: 分析场/真值 (n_levels, H, W)
        mask: 观测掩膜 (1, H, W), 1=有效, 0=无效
    """
    
    def __init__(
        self,
        satellite_dir: Union[str, Path],
        background_dir: Union[str, Path],
        label_dir: Union[str, Path],
        file_list: Optional[List[str]] = None,
        n_sat_channels: int = 13,
        n_levels: int = 37,
        height: int = 400,
        width: int = 400,
        normalize: bool = True,
        bt_min: float = 100.0,
        bt_max: float = 400.0,
        transform=None
    ):
        """
        参数:
            satellite_dir: 卫星数据目录
            background_dir: 背景场目录
            label_dir: 标签(分析场)目录
            file_list: 文件名列表 (不含扩展名)
            n_sat_channels: 卫星通道数
            n_levels: 垂直层数
            height, width: 图像尺寸
            normalize: 是否归一化
            bt_min, bt_max: 亮温有效范围
            transform: 数据增强
        """
        self.satellite_dir = Path(satellite_dir)
        self.background_dir = Path(background_dir)
        self.label_dir = Path(label_dir)
        
        self.n_sat_channels = n_sat_channels
        self.n_levels = n_levels
        self.height = height
        self.width = width
        self.normalize = normalize
        self.bt_min = bt_min
        self.bt_max = bt_max
        self.transform = transform
        
        # 获取文件列表
        if file_list is not None:
            self.file_list = file_list
        else:
            # 自动发现文件
            self.file_list = self._discover_files()
        
        # 预计算归一化参数 (可以从训练集统计)
        self.sat_mean = None
        self.sat_std = None
        self.bg_mean = None
        self.bg_std = None
        
    def _discover_files(self) -> List[str]:
        """自动发现数据文件"""
        files = []
        for ext in ['*.npy', '*.npz', '*.h5', '*.nc']:
            files.extend([f.stem for f in self.satellite_dir.glob(ext)])
        return sorted(list(set(files)))
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def _load_satellite(self, file_stem: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载卫星数据并生成掩膜
        
        返回:
            data: (n_channels, H, W)
            mask: (1, H, W) - 1=有效, 0=无效
        """
        # 尝试不同格式
        for ext in ['.npy', '.npz', '.h5', '.nc']:
            file_path = self.satellite_dir / (file_stem + ext)
            if file_path.exists():
                break
        
        if ext == '.npy':
            data = np.load(file_path)
        elif ext == '.npz':
            npz = np.load(file_path)
            data = npz['brightness_temperature']  # 假设的key
        elif ext == '.h5':
            with h5py.File(file_path, 'r') as f:
                data = f['BT'][:]  # FY-3数据的典型key
        elif ext == '.nc':
            ds = xr.open_dataset(file_path)
            data = ds['bt'].values
            ds.close()
        else:
            raise FileNotFoundError(f"Cannot find satellite data: {file_stem}")
        
        # 确保shape正确: (C, H, W)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim == 3 and data.shape[0] != self.n_sat_channels:
            data = data.transpose(2, 0, 1)
        
        # 生成掩膜: 有效亮温范围内为1
        mask = ((data >= self.bt_min) & (data <= self.bt_max)).all(axis=0, keepdims=True)
        mask = mask.astype(np.float32)
        
        # 将无效值设为NaN (后续填充)
        data = np.where(
            (data >= self.bt_min) & (data <= self.bt_max),
            data,
            np.nan
        )
        
        # 填充NaN (使用均值或插值)
        for c in range(data.shape[0]):
            if np.isnan(data[c]).any():
                mean_val = np.nanmean(data[c])
                data[c] = np.where(np.isnan(data[c]), mean_val, data[c])
        
        return data.astype(np.float32), mask
    
    def _load_background(self, file_stem: str) -> np.ndarray:
        """加载背景场数据"""
        for ext in ['.npy', '.grib', '.nc']:
            file_path = self.background_dir / (file_stem + ext)
            if file_path.exists():
                break
        
        if ext == '.npy':
            data = np.load(file_path)
        elif ext == '.grib':
            # 使用cfgrib加载
            import cfgrib
            ds = cfgrib.open_dataset(file_path)
            data = ds['t'].values  # 温度
            ds.close()
        elif ext == '.nc':
            ds = xr.open_dataset(file_path)
            data = ds['t'].values
            ds.close()
        else:
            raise FileNotFoundError(f"Cannot find background data: {file_stem}")
        
        # 确保shape: (n_levels, H, W)
        if data.ndim == 3 and data.shape[0] != self.n_levels:
            data = data.transpose(2, 0, 1)
        
        return data.astype(np.float32)
    
    def _load_label(self, file_stem: str) -> np.ndarray:
        """加载标签(分析场)"""
        for ext in ['.npy', '.nc']:
            file_path = self.label_dir / (file_stem + ext)
            if file_path.exists():
                break
        
        if ext == '.npy':
            data = np.load(file_path)
        elif ext == '.nc':
            ds = xr.open_dataset(file_path)
            data = ds['analysis'].values
            ds.close()
        else:
            raise FileNotFoundError(f"Cannot find label data: {file_stem}")
        
        return data.astype(np.float32)
    
    def _normalize(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """
        归一化数据
        
        data_type: 'satellite', 'background', 'label'
        """
        if data_type == 'satellite':
            # Min-Max归一化到[0, 1]
            return (data - self.bt_min) / (self.bt_max - self.bt_min)
        elif data_type in ['background', 'label']:
            # Z-score或Min-Max
            # 这里使用逐层min-max
            for i in range(data.shape[0]):
                min_val = data[i].min()
                max_val = data[i].max()
                if max_val - min_val > 1e-6:
                    data[i] = (data[i] - min_val) / (max_val - min_val)
            return data
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据样本
        
        返回字典:
            'satellite': (n_channels, H, W)
            'background': (n_levels, H, W)
            'label': (n_levels, H, W)
            'mask': (1, H, W)
        """
        file_stem = self.file_list[idx]
        
        # 加载数据
        satellite, mask = self._load_satellite(file_stem)
        background = self._load_background(file_stem)
        label = self._load_label(file_stem)
        
        # 归一化
        if self.normalize:
            satellite = self._normalize(satellite, 'satellite')
            background = self._normalize(background, 'background')
            label = self._normalize(label, 'label')
        
        # 数据增强
        if self.transform is not None:
            satellite, background, label, mask = self.transform(
                satellite, background, label, mask
            )
        
        return {
            'satellite': torch.from_numpy(satellite),
            'background': torch.from_numpy(background),
            'label': torch.from_numpy(label),
            'mask': torch.from_numpy(mask)
        }


class SyntheticDataset(Dataset):
    """
    合成数据集 - 用于测试和调试
    
    生成模拟的卫星、背景场和标签数据
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        n_sat_channels: int = 13,
        n_levels: int = 37,
        height: int = 64,
        width: int = 64,
        noise_level: float = 0.1,
        missing_ratio: float = 0.2
    ):
        self.num_samples = num_samples
        self.n_sat_channels = n_sat_channels
        self.n_levels = n_levels
        self.height = height
        self.width = width
        self.noise_level = noise_level
        self.missing_ratio = missing_ratio
        
        # 预生成数据
        self._generate_data()
    
    def _generate_data(self):
        """生成模拟数据"""
        # 生成"真实"的温度场 (标签)
        # 使用简单的物理模型: 温度随高度递减
        
        self.labels = []
        self.backgrounds = []
        self.satellites = []
        self.masks = []
        
        for _ in range(self.num_samples):
            # 基础温度场
            x = np.linspace(0, 4*np.pi, self.width)
            y = np.linspace(0, 4*np.pi, self.height)
            X, Y = np.meshgrid(x, y)
            
            # 水平温度变化 (模拟天气系统)
            base_temp = 280 + 20 * np.sin(X) * np.cos(Y)
            
            # 垂直廓线 (温度随高度递减)
            heights = np.linspace(0, 1, self.n_levels)[:, np.newaxis, np.newaxis]
            lapse_rate = 6.5 / 1000 * 10000  # 约65K/10km
            
            label = base_temp - lapse_rate * heights
            label = label.astype(np.float32)
            
            # 背景场 = 真值 + 系统偏差 + 噪声
            background = label + np.random.randn(*label.shape).astype(np.float32) * 5
            
            # 卫星亮温 = 辐射传输模拟 (简化)
            # 实际上是温度的加权平均 (不同通道对应不同权重函数)
            weighting_functions = self._generate_weighting_functions()
            satellite = np.zeros((self.n_sat_channels, self.height, self.width), dtype=np.float32)
            
            for c in range(self.n_sat_channels):
                wf = weighting_functions[c][:, np.newaxis, np.newaxis]
                satellite[c] = (label * wf).sum(axis=0)
            
            # 添加仪器噪声
            satellite += np.random.randn(*satellite.shape).astype(np.float32) * self.noise_level * 10
            
            # 生成掩膜 (模拟缺失数据)
            mask = np.ones((1, self.height, self.width), dtype=np.float32)
            n_missing = int(self.height * self.width * self.missing_ratio)
            missing_idx = np.random.choice(self.height * self.width, n_missing, replace=False)
            mask_flat = mask.reshape(-1)
            mask_flat[missing_idx] = 0
            mask = mask_flat.reshape(1, self.height, self.width)
            
            self.labels.append(label)
            self.backgrounds.append(background)
            self.satellites.append(satellite)
            self.masks.append(mask)
        
        # 归一化
        self._normalize_all()
    
    def _generate_weighting_functions(self) -> np.ndarray:
        """生成模拟的权重函数 (辐射传输)"""
        wfs = np.zeros((self.n_sat_channels, self.n_levels))
        
        for c in range(self.n_sat_channels):
            # 每个通道的权重函数峰值在不同高度
            peak_level = int((c / self.n_sat_channels) * self.n_levels * 0.8)
            wfs[c] = np.exp(-((np.arange(self.n_levels) - peak_level) ** 2) / (2 * 5 ** 2))
            wfs[c] /= wfs[c].sum()
        
        return wfs.astype(np.float32)
    
    def _normalize_all(self):
        """归一化所有数据"""
        # 计算统计量
        all_sat = np.stack(self.satellites)
        all_bg = np.stack(self.backgrounds)
        all_label = np.stack(self.labels)
        
        # Min-Max归一化
        self.sat_min = all_sat.min()
        self.sat_max = all_sat.max()
        self.bg_min = all_bg.min()
        self.bg_max = all_bg.max()
        
        for i in range(self.num_samples):
            self.satellites[i] = (self.satellites[i] - self.sat_min) / (self.sat_max - self.sat_min)
            self.backgrounds[i] = (self.backgrounds[i] - self.bg_min) / (self.bg_max - self.bg_min)
            self.labels[i] = (self.labels[i] - self.bg_min) / (self.bg_max - self.bg_min)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'satellite': torch.from_numpy(self.satellites[idx]),
            'background': torch.from_numpy(self.backgrounds[idx]),
            'label': torch.from_numpy(self.labels[idx]),
            'mask': torch.from_numpy(self.masks[idx])
        }


# ============================================
# 数据增强
# ============================================

class RandomFlip:
    """随机翻转"""
    
    def __init__(self, p_h: float = 0.5, p_v: float = 0.5):
        self.p_h = p_h
        self.p_v = p_v
    
    def __call__(self, sat, bg, label, mask):
        if np.random.rand() < self.p_h:
            sat = np.flip(sat, axis=2).copy()
            bg = np.flip(bg, axis=2).copy()
            label = np.flip(label, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        
        if np.random.rand() < self.p_v:
            sat = np.flip(sat, axis=1).copy()
            bg = np.flip(bg, axis=1).copy()
            label = np.flip(label, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        
        return sat, bg, label, mask


class RandomRotate90:
    """随机旋转90度"""
    
    def __call__(self, sat, bg, label, mask):
        k = np.random.randint(4)
        sat = np.rot90(sat, k, axes=(1, 2)).copy()
        bg = np.rot90(bg, k, axes=(1, 2)).copy()
        label = np.rot90(label, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(1, 2)).copy()
        return sat, bg, label, mask


class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试数据集类")
    print("=" * 60)
    
    # 创建合成数据集
    print("\n创建合成数据集...")
    dataset = SyntheticDataset(
        num_samples=100,
        n_sat_channels=13,
        n_levels=37,
        height=64,
        width=64,
        missing_ratio=0.2
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取样本
    sample = dataset[0]
    print(f"\n样本形状:")
    for key, value in sample.items():
        print(f"  {key}: {value.shape}, dtype={value.dtype}")
    
    print(f"\n数据范围:")
    print(f"  satellite: [{sample['satellite'].min():.3f}, {sample['satellite'].max():.3f}]")
    print(f"  background: [{sample['background'].min():.3f}, {sample['background'].max():.3f}]")
    print(f"  label: [{sample['label'].min():.3f}, {sample['label'].max():.3f}]")
    print(f"  mask有效比例: {sample['mask'].mean():.2%}")
    
    # 创建DataLoader
    print("\n创建DataLoader...")
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    print(f"批次形状:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    # 测试数据增强
    print("\n测试数据增强...")
    transform = Compose([
        RandomFlip(),
        RandomRotate90()
    ])
    
    sat = sample['satellite'].numpy()
    bg = sample['background'].numpy()
    label = sample['label'].numpy()
    mask = sample['mask'].numpy()
    
    sat_aug, bg_aug, label_aug, mask_aug = transform(sat, bg, label, mask)
    print(f"增强后形状: {sat_aug.shape}")
