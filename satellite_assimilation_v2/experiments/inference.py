import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, List

# 导入项目中已有的模块
from models.backbone import create_model, UNetConfig
from data_pipeline_v2 import LazySatelliteERA5Dataset, LevelwiseNormalizer

class PASNetInference:
    def __init__(self, exp_dir: str, data_root: str, stats_file: str, device: str = 'cuda'):
        self.exp_dir = Path(exp_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 1. 加载配置
        with open(self.exp_dir / 'config.json', 'r') as f:
            self.args = json.load(f)
        
        # 2. 初始化数据集与标准化器
        stats = np.load(stats_file)
        self.obs_norm = LevelwiseNormalizer(stats['obs_mean'], stats['obs_std'], name='obs')
        self.bkg_norm = LevelwiseNormalizer(stats['bkg_mean'], stats['bkg_std'], name='bkg')
        self.target_norm = LevelwiseNormalizer(stats['target_mean'], stats['target_std'], name='target')
        
        # 仅加载测试集数据
        all_files = sorted(list(Path(data_root).glob('**/*.npz')))
        test_files = all_files[int(len(all_files)*0.85):] # 参考 0.15 的测试比例 [cite: 173]
        
        self.dataset = LazySatelliteERA5Dataset(
            file_list=[str(f) for f in test_files],
            obs_normalizer=self.obs_norm,
            bkg_normalizer=self.bkg_norm,
            target_normalizer=self.target_norm,
            use_aux=self.args.get('use_aux', True)
        )
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        
        # 3. 创建并加载模型
        config = UNetConfig(
            fusion_mode=self.args.get('fusion_mode', 'gated'),
            use_aux=self.args.get('use_aux', True),
            mask_aware=self.args.get('mask_aware', True),
            use_spectral_adapter=self.args.get('use_spectral_adapter', True)
        )
        self.model = create_model(self.args.get('model', 'pasnet'), config).to(self.device)
        
        checkpoint = torch.load(self.exp_dir / 'checkpoint_best.pth', map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        # 移除 DDP 可能引入的 'module.' 前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def run_and_plot(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        all_rmses, all_corrs = [], []
        
        # 用于绘图的单个样本数据
        plot_sample = None
        
        print(f"开始推理实验: {self.args['exp_name']}...")
        for i, batch in enumerate(self.loader):
            obs, bkg, mask = batch['obs'].to(self.device), batch['bkg'].to(self.device), batch['mask'].to(self.device)
            target = batch['target'].to(self.device)
            aux = batch.get('aux').to(self.device) if 'aux' in batch else None
            
            output = self.model(obs, bkg, mask, aux)
            pred = output[0] if isinstance(output, tuple) else output
            
            # 逆标准化还原物理值 (K)
            pred_phys = self.target_norm.inverse_transform(pred).cpu().numpy()
            target_phys = self.target_norm.inverse_transform(target).cpu().numpy()
            bkg_phys = self.bkg_norm.inverse_transform(bkg).cpu().numpy()
            
            # 计算分层指标 [cite: 177]
            rmse = np.sqrt(np.mean((pred_phys - target_phys)**2, axis=(0, 2, 3)))
            corr = [np.corrcoef(pred_phys[0, l].flatten(), target_phys[0, l].flatten())[0,1] for l in range(pred_phys.shape[1])]
            
            all_rmses.append(rmse)
            all_corrs.append(corr)
            
            if i == 0: plot_sample = (bkg_phys[0], target_phys[0], pred_phys[0])

        # 1. 绘制分层指标曲线 (参考 PAVMT-Unet Fig. 7) 
        self._plot_metrics(np.mean(all_rmses, axis=0), np.mean(all_corrs, axis=0), output_dir)
        
        # 2. 绘制水平分布图 (参考 PAVMT-Unet Fig. 5) 
        self._plot_spatial(plot_sample, output_dir)
        
        # 3. 绘制剖面增量图 (参考 PAVMT-Unet Fig. 6) 
        self._plot_vertical_increment(plot_sample, output_dir)

    def _plot_metrics(self, rmse, corr, out):
        levels = np.arange(len(rmse))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(rmse, levels, 'r-o', label='PAS-Net')
        ax1.set_ylabel('Vertical Layer'); ax1.set_xlabel('RMSE (K)'); ax1.invert_yaxis(); ax1.grid(True); ax1.legend()
        ax2.plot(corr, levels, 'b-o', label='PAS-Net')
        ax2.set_xlabel('Correlation Coefficient'); ax2.invert_yaxis(); ax2.grid(True); ax2.legend()
        plt.tight_layout()
        plt.savefig(out / 'metrics_profile.png', dpi=200)
        plt.close()

    def _plot_spatial(self, data, out):
        bkg, tgt, pred = data
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        titles = ['Background Field', '3D-Var (Ground Truth)', 'PAS-Net (Ours)']
        for i, (img, title) in enumerate(zip([bkg[0], tgt[0], pred[0]], titles)):
            im = axes[i].imshow(img, cmap='RdYlBu_r')
            axes[i].set_title(title); plt.colorbar(im, ax=axes[i])
        plt.savefig(out / 'spatial_distribution.png', dpi=200)
        plt.close()

    def _plot_vertical_increment(self, data, out):
        bkg, tgt, pred = data
        inc_gt = tgt - bkg
        inc_pred = pred - bkg
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        im1 = ax1.imshow(inc_gt[:, 32, :], aspect='auto', cmap='RdBu_r') # 取中间切片
        ax1.set_title('3D-Var Increment Profile'); plt.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(inc_pred[:, 32, :], aspect='auto', cmap='RdBu_r')
        ax2.set_title('PAS-Net Increment Profile'); plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.savefig(out / 'vertical_increment.png', dpi=200)
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--stats_file', type=str, required=True)
    parser.add_argument('--output_root', type=str, default='outputs/inference_results')
    args = parser.parse_args()
    
    inf = PASNetInference(args.exp_dir, args.data_root, args.stats_file)
    inf.run_and_plot(Path(args.output_root) / Path(args.exp_dir).name)