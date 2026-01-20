#!/usr/bin/env python3
"""
数据可视化工具
用于检查配准结果和数据质量
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import xarray as xr


class CollocationVisualizer:
    """配准数据可视化工具"""
    
    def __init__(self, X=None, Y=None, pressure_levels=None):
        self.X = X
        self.Y = Y
        self.pressure_levels = pressure_levels
        
    def load_data(self, filepath_prefix):
        """从文件加载数据"""
        self.X = np.load(f'{filepath_prefix}_X.npy')
        self.Y = np.load(f'{filepath_prefix}_Y.npy')
        self.pressure_levels = np.load(f'{filepath_prefix}_pressure.npy')
        
        print(f"✓ 加载数据:")
        print(f"  X: {self.X.shape}")
        print(f"  Y: {self.Y.shape}")
        print(f"  压力层: {len(self.pressure_levels)}")
    
    def plot_brightness_temperature_histogram(self, channel_idx=0, save_path=None):
        """绘制亮温分布直方图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Brightness Temperature Distribution', fontsize=16, fontweight='bold')
        
        n_channels = self.X.shape[1]
        channels_to_plot = [0, n_channels//3, n_channels*2//3, n_channels-1]
        
        for idx, (ax, ch) in enumerate(zip(axes.flat, channels_to_plot)):
            bt_channel = self.X[:, ch]
            
            ax.hist(bt_channel, bins=100, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Brightness Temperature (K)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Channel {ch+1}', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_val = np.mean(bt_channel)
            std_val = np.std(bt_channel)
            ax.axvline(mean_val, color='r', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.2f} K')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5)
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5,
                      label=f'Std: {std_val:.2f} K')
            ax.legend(fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存: {save_path}")
        
        plt.show()
    
    def plot_temperature_profiles(self, n_samples=10, save_path=None):
        """绘制温度廓线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        fig.suptitle('Temperature Profiles', fontsize=16, fontweight='bold')
        
        # 随机选择样本
        indices = np.random.choice(len(self.Y), min(n_samples, len(self.Y)), replace=False)
        
        # 左图：单个廓线
        for idx in indices:
            profile = self.Y[idx]
            ax1.plot(profile, self.pressure_levels, alpha=0.6, linewidth=2)
        
        ax1.set_xlabel('Temperature (K)', fontsize=12)
        ax1.set_ylabel('Pressure (hPa)', fontsize=12)
        ax1.set_title(f'{len(indices)} Random Profiles', fontsize=13)
        ax1.invert_yaxis()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 右图：平均廓线 + 标准差
        mean_profile = np.mean(self.Y, axis=0)
        std_profile = np.std(self.Y, axis=0)
        
        ax2.plot(mean_profile, self.pressure_levels, 'b-', linewidth=3, label='Mean')
        ax2.fill_betweenx(self.pressure_levels, 
                         mean_profile - std_profile,
                         mean_profile + std_profile,
                         alpha=0.3, label='±1 Std')
        
        ax2.set_xlabel('Temperature (K)', fontsize=12)
        ax2.set_ylabel('Pressure (hPa)', fontsize=12)
        ax2.set_title('Mean Profile ± Std', fontsize=13)
        ax2.invert_yaxis()
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存: {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, save_path=None):
        """绘制亮温通道间的相关性矩阵"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 计算相关性
        corr_matrix = np.corrcoef(self.X.T)
        
        # 绘制
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # 添加colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='Correlation Coefficient')
        
        # 设置标签
        n_channels = self.X.shape[1]
        ax.set_xticks(range(n_channels))
        ax.set_yticks(range(n_channels))
        ax.set_xticklabels([f'Ch{i+1}' for i in range(n_channels)], rotation=45)
        ax.set_yticklabels([f'Ch{i+1}' for i in range(n_channels)])
        
        ax.set_title('Channel Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Channel', fontsize=12)
        ax.set_ylabel('Channel', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存: {save_path}")
        
        plt.show()
    
    def plot_bt_vs_temperature(self, channel_idx=0, level_idx=10, save_path=None):
        """绘制亮温与温度的散点图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Brightness Temperature vs Temperature', 
                    fontsize=16, fontweight='bold')
        
        # 选择不同的气压层
        n_levels = len(self.pressure_levels)
        levels_to_plot = [0, n_levels//3, n_levels*2//3, n_levels-1]
        
        for ax, level_idx in zip(axes.flat, levels_to_plot):
            # 随机采样（避免点太多）
            n_samples = min(5000, len(self.X))
            indices = np.random.choice(len(self.X), n_samples, replace=False)
            
            bt = self.X[indices, channel_idx]
            temp = self.Y[indices, level_idx]
            
            # 散点图
            scatter = ax.scatter(bt, temp, alpha=0.5, s=10, c=temp, 
                               cmap='coolwarm', edgecolors='none')
            
            # 拟合直线
            z = np.polyfit(bt, temp, 1)
            p = np.poly1d(z)
            bt_sorted = np.sort(bt)
            ax.plot(bt_sorted, p(bt_sorted), "r-", linewidth=2, 
                   label=f'Fit: y={z[0]:.3f}x+{z[1]:.1f}')
            
            # 计算相关系数
            corr = np.corrcoef(bt, temp)[0, 1]
            
            pressure = self.pressure_levels[level_idx]
            ax.set_xlabel(f'BT Channel {channel_idx+1} (K)', fontsize=11)
            ax.set_ylabel(f'Temperature at {pressure:.0f} hPa (K)', fontsize=11)
            ax.set_title(f'Level {level_idx+1}: {pressure:.0f} hPa, R={corr:.3f}', 
                        fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # 添加colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(scatter, cax=cax, label='Temp (K)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存: {save_path}")
        
        plt.show()
    
    def plot_data_coverage(self, fy3d_file=None, save_path=None):
        """绘制数据空间覆盖图"""
        if fy3d_file is None:
            print("需要提供FY-3D文件以绘制覆盖图")
            return
        
        # 读取经纬度
        with h5py.File(fy3d_file, 'r') as f:
            lat = f['Latitude'][:]
            lon = f['Longitude'][:]
        
        fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={'projection': 'PlateCarree'})
        
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.gridlines(draw_labels=True, alpha=0.3)
            
            # 绘制卫星轨道
            scatter = ax.scatter(lon.flatten(), lat.flatten(), 
                               c='red', s=1, alpha=0.3, 
                               transform=ccrs.PlateCarree())
            
            ax.set_title('FY-3D MWTS Data Coverage', fontsize=14, fontweight='bold')
            
        except ImportError:
            # 如果没有cartopy，用简单的散点图
            ax = plt.axes()
            ax.scatter(lon.flatten(), lat.flatten(), c='red', s=1, alpha=0.3)
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.set_title('FY-3D MWTS Data Coverage', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存: {save_path}")
        
        plt.show()
    
    def generate_all_plots(self, output_dir='./plots', fy3d_file=None):
        """生成所有可视化图"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n📊 生成可视化图...")
        
        self.plot_brightness_temperature_histogram(
            save_path=f'{output_dir}/bt_histogram.png'
        )
        
        self.plot_temperature_profiles(
            save_path=f'{output_dir}/temperature_profiles.png'
        )
        
        self.plot_correlation_matrix(
            save_path=f'{output_dir}/correlation_matrix.png'
        )
        
        self.plot_bt_vs_temperature(
            save_path=f'{output_dir}/bt_vs_temp.png'
        )
        
        if fy3d_file:
            self.plot_data_coverage(
                fy3d_file=fy3d_file,
                save_path=f'{output_dir}/data_coverage.png'
            )
        
        print(f"\n✅ 所有图表已保存到: {output_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化配准数据')
    parser.add_argument('data_prefix', help='数据文件前缀')
    parser.add_argument('-o', '--output', default='./plots',
                       help='输出目录')
    parser.add_argument('--fy3d', help='FY-3D文件（用于绘制覆盖图）')
    
    args = parser.parse_args()
    
    # 创建可视化器
    viz = CollocationVisualizer()
    viz.load_data(args.data_prefix)
    
    # 生成所有图表
    viz.generate_all_plots(args.output, args.fy3d)


if __name__ == '__main__':
    main()
