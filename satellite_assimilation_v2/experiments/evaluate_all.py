#!/usr/bin/env python3
"""
===============================================================================
PAS-Net 实验评估脚本
Evaluation Script for PAS-Net Experiments
===============================================================================

功能:
1. 评估所有实验的检查点
2. 生成论文所需的结果表格 (LaTeX格式)
3. 绘制分层RMSE曲线图
4. 生成可视化对比图

===============================================================================
"""

from __future__ import annotations

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent))

from models.backbone import create_model, UNetConfig


# =============================================================================
# Part 1: 评估指标
# =============================================================================

class Evaluator:
    """模型评估器"""
    
    # ERA5气压层 (hPa)
    PRESSURE_LEVELS = np.array([
        1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
        750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
        250, 225, 200, 175, 150, 125, 100, 70, 50, 30,
        20, 10, 7, 5, 3, 2, 1
    ])
    
    def __init__(self, stratosphere_threshold: float = 100.0):
        self.strat_threshold = stratosphere_threshold
        self.strat_mask = self.PRESSURE_LEVELS <= stratosphere_threshold
        self.trop_mask = ~self.strat_mask
    
    @staticmethod
    def rmse(pred: np.ndarray, target: np.ndarray, axis=None) -> np.ndarray:
        return np.sqrt(np.mean((pred - target) ** 2, axis=axis))
    
    @staticmethod
    def mae(pred: np.ndarray, target: np.ndarray, axis=None) -> np.ndarray:
        return np.mean(np.abs(pred - target), axis=axis)
    
    @staticmethod
    def correlation(pred: np.ndarray, target: np.ndarray) -> float:
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        return np.corrcoef(pred_flat, target_flat)[0, 1]
    
    def evaluate(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        device: torch.device
    ) -> Dict[str, float]:
        """评估模型"""
        model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                obs = batch['obs'].to(device)
                bkg = batch['bkg'].to(device)
                mask = batch['mask'].to(device)
                target = batch['target']
                aux = batch.get('aux')
                if aux is not None:
                    aux = aux.to(device)
                
                output = model(obs, bkg, mask, aux)
                if isinstance(output, tuple):
                    pred = output[0]
                else:
                    pred = output
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.numpy())
        
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # 计算指标
        results = {
            'global_rmse': self.rmse(preds, targets),
            'global_mae': self.mae(preds, targets),
            'correlation': self.correlation(preds, targets),
        }
        
        # 分层RMSE
        levelwise_rmse = self.rmse(preds, targets, axis=(0, 2, 3))
        results['levelwise_rmse'] = levelwise_rmse
        
        # 平流层/对流层
        strat_pred = preds[:, self.strat_mask]
        strat_target = targets[:, self.strat_mask]
        results['stratosphere_rmse'] = self.rmse(strat_pred, strat_target)
        
        trop_pred = preds[:, self.trop_mask]
        trop_target = targets[:, self.trop_mask]
        results['troposphere_rmse'] = self.rmse(trop_pred, trop_target)
        
        return results


# =============================================================================
# Part 2: 结果收集
# =============================================================================

def collect_experiment_results(exp_dir: Path) -> Optional[Dict]:
    """收集单个实验的结果"""
    results_file = exp_dir / 'results.json'
    config_file = exp_dir / 'config.json'
    checkpoint_file = exp_dir / 'checkpoint_best.pth'
    
    if not results_file.exists():
        return None
    
    with open(results_file) as f:
        results = json.load(f)
    
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        results['config'] = config
    
    results['exp_dir'] = str(exp_dir)
    results['has_checkpoint'] = checkpoint_file.exists()
    
    return results


def collect_all_results(output_dir: Path) -> List[Dict]:
    """收集所有实验结果"""
    all_results = []
    
    for exp_dir in output_dir.iterdir():
        if exp_dir.is_dir():
            result = collect_experiment_results(exp_dir)
            if result:
                all_results.append(result)
    
    return all_results


# =============================================================================
# Part 3: LaTeX表格生成
# =============================================================================

def generate_ablation_table(results: List[Dict]) -> str:
    """生成消融实验LaTeX表格"""
    
    # 筛选消融实验
    ablation_results = [r for r in results if r.get('config', {}).get('exp_name', '').startswith('A')]
    
    if not ablation_results:
        return "% No ablation results found"
    
    table = r"""
\begin{table}[htbp]
\centering
\caption{Ablation Study Results on FY-3F Dataset}
\label{tab:ablation}
\begin{tabular}{l|ccc}
\toprule
\textbf{Configuration} & \textbf{Global RMSE} & \textbf{Troposphere RMSE} & \textbf{Stratosphere RMSE} \\
\midrule
"""
    
    configs = [
        ('A1', 'Full PAS-Net', 'baseline'),
        ('A2', 'w/o Level-wise Norm', 'norm_mode=global'),
        ('A3', 'w/o Spectral Adapter', 'use_spectral_adapter=false'),
        ('A4', 'w/o Gradient Loss', 'loss=mse'),
        ('A5', 'w/o Auxiliary Features', 'use_aux=false'),
        ('A6', 'w/o Mask-Aware', 'mask_aware=false'),
        ('A7', 'w/o SE Block', 'model=pasnet_no_se'),
        ('A8', 'Fusion: concat', 'fusion_mode=concat'),
        ('A9', 'Fusion: add', 'fusion_mode=add'),
    ]
    
    for exp_id, name, _ in configs:
        # 查找对应结果
        result = next((r for r in ablation_results if exp_id in r.get('exp_name', '')), None)
        
        if result:
            global_rmse = result.get('global_rmse', '-')
            trop_rmse = result.get('troposphere_rmse', '-')
            strat_rmse = result.get('stratosphere_rmse', '-')
            
            if isinstance(global_rmse, (int, float)):
                global_rmse = f"{global_rmse:.4f}"
            if isinstance(trop_rmse, (int, float)):
                trop_rmse = f"{trop_rmse:.4f}"
            if isinstance(strat_rmse, (int, float)):
                strat_rmse = f"{strat_rmse:.4f}"
        else:
            global_rmse = trop_rmse = strat_rmse = '-'
        
        # 加粗最佳行
        if exp_id == 'A1':
            table += f"\\textbf{{{name}}} & \\textbf{{{global_rmse}}} & \\textbf{{{trop_rmse}}} & \\textbf{{{strat_rmse}}} \\\\\n"
        else:
            table += f"{name} & {global_rmse} & {trop_rmse} & {strat_rmse} \\\\\n"
    
    table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return table


def generate_comparison_table(results: List[Dict]) -> str:
    """生成对比实验LaTeX表格"""
    
    table = r"""
\begin{table}[htbp]
\centering
\caption{Comparison with State-of-the-Art Methods}
\label{tab:comparison}
\begin{tabular}{l|cccc}
\toprule
\textbf{Method} & \textbf{Global RMSE (K)} & \textbf{Trop. RMSE} & \textbf{Strat. RMSE} & \textbf{\#Params} \\
\midrule
"""
    
    methods = [
        ('C1', 'Vanilla U-Net', 'vanilla_unet'),
        ('C2', 'ResUNet', 'res_unet'),
        ('C3', 'Attention U-Net', 'attention_unet'),
        ('A1', '\\textbf{PAS-Net (Ours)}', 'pasnet'),
    ]
    
    for exp_id, name, model_type in methods:
        result = next((r for r in results if exp_id in r.get('exp_name', '')), None)
        
        if result:
            global_rmse = result.get('global_rmse', '-')
            trop_rmse = result.get('troposphere_rmse', '-')
            strat_rmse = result.get('stratosphere_rmse', '-')
            n_params = result.get('n_params', '-')
            
            if isinstance(global_rmse, (int, float)):
                global_rmse = f"{global_rmse:.4f}"
            if isinstance(trop_rmse, (int, float)):
                trop_rmse = f"{trop_rmse:.4f}"
            if isinstance(strat_rmse, (int, float)):
                strat_rmse = f"{strat_rmse:.4f}"
            if isinstance(n_params, int):
                n_params = f"{n_params/1e6:.2f}M"
        else:
            global_rmse = trop_rmse = strat_rmse = n_params = '-'
        
        table += f"{name} & {global_rmse} & {trop_rmse} & {strat_rmse} & {n_params} \\\\\n"
    
    table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return table


# =============================================================================
# Part 4: 可视化
# =============================================================================

def plot_levelwise_rmse(results: List[Dict], save_path: Optional[str] = None):
    """绘制分层RMSE曲线"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pressure_levels = Evaluator.PRESSURE_LEVELS
    
    # 定义颜色和标记
    styles = {
        'A1': {'color': 'red', 'linestyle': '-', 'marker': 'o', 'label': 'PAS-Net (Full)'},
        'A3': {'color': 'blue', 'linestyle': '--', 'marker': 's', 'label': 'w/o Spectral Adapter'},
        'A4': {'color': 'green', 'linestyle': '-.', 'marker': '^', 'label': 'w/o Gradient Loss'},
        'C1': {'color': 'gray', 'linestyle': ':', 'marker': 'x', 'label': 'Vanilla U-Net'},
    }
    
    for exp_id, style in styles.items():
        result = next((r for r in results if exp_id in r.get('exp_name', '')), None)
        if result and 'levelwise_rmse' in result:
            rmse = np.array(result['levelwise_rmse'])
            ax.semilogy(rmse, pressure_levels[:len(rmse)], **style)
    
    ax.invert_yaxis()
    ax.set_xlabel('RMSE (K)', fontsize=12)
    ax.set_ylabel('Pressure (hPa)', fontsize=12)
    ax.set_title('Level-wise RMSE Comparison', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 添加平流层分界线
    ax.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Stratosphere boundary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    
    plt.close()


def plot_sample_comparison(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    save_path: Optional[str] = None,
    level_idx: int = 20  # 约300hPa
):
    """绘制样本对比图"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    model.eval()
    
    with torch.no_grad():
        obs = batch['obs'].to(device)
        bkg = batch['bkg'].to(device)
        mask = batch['mask'].to(device)
        target = batch['target']
        aux = batch.get('aux')
        if aux is not None:
            aux = aux.to(device)
        
        output = model(obs, bkg, mask, aux)
        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output
        
        pred = pred.cpu()
    
    # 选择第一个样本
    bkg_sample = batch['bkg'][0, level_idx].numpy()
    pred_sample = pred[0, level_idx].numpy()
    target_sample = target[0, level_idx].numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    vmin = min(bkg_sample.min(), target_sample.min())
    vmax = max(bkg_sample.max(), target_sample.max())
    
    im1 = axes[0].imshow(bkg_sample, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title('Background')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    im2 = axes[1].imshow(target_sample, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title('ERA5 (Target)')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    im3 = axes[2].imshow(pred_sample, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    axes[2].set_title('PAS-Net (Prediction)')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    error = pred_sample - target_sample
    im4 = axes[3].imshow(error, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[3].set_title(f'Error (RMSE={np.sqrt((error**2).mean()):.4f} K)')
    plt.colorbar(im4, ax=axes[3], shrink=0.8)
    
    for ax in axes:
        ax.axis('off')
    
    plt.suptitle(f'Level {level_idx} (~{Evaluator.PRESSURE_LEVELS[level_idx]} hPa)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    
    plt.close()


# =============================================================================
# Part 5: 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='PAS-Net实验评估脚本')
    parser.add_argument('--output_dir', type=str, default='outputs/experiments',
                        help='实验输出目录')
    parser.add_argument('--results_dir', type=str, default='outputs/results',
                        help='结果保存目录')
    parser.add_argument('--generate_tables', action='store_true',
                        help='生成LaTeX表格')
    parser.add_argument('--plot', action='store_true',
                        help='生成可视化图表')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PAS-Net 实验结果评估")
    print("=" * 70)
    
    # 收集结果
    print("\n收集实验结果...")
    all_results = collect_all_results(output_dir)
    print(f"  找到 {len(all_results)} 个实验结果")
    
    # 打印摘要
    print("\n实验摘要:")
    print("-" * 70)
    for result in sorted(all_results, key=lambda x: x.get('exp_name', '')):
        exp_name = result.get('exp_name', 'Unknown')
        val_loss = result.get('best_val_loss', '-')
        if isinstance(val_loss, float):
            val_loss = f"{val_loss:.6f}"
        print(f"  {exp_name}: Val Loss = {val_loss}")
    
    # 生成LaTeX表格
    if args.generate_tables:
        print("\n生成LaTeX表格...")
        
        ablation_table = generate_ablation_table(all_results)
        with open(results_dir / 'ablation_table.tex', 'w') as f:
            f.write(ablation_table)
        print(f"  消融实验表格: {results_dir / 'ablation_table.tex'}")
        
        comparison_table = generate_comparison_table(all_results)
        with open(results_dir / 'comparison_table.tex', 'w') as f:
            f.write(comparison_table)
        print(f"  对比实验表格: {results_dir / 'comparison_table.tex'}")
    
    # 生成可视化
    if args.plot:
        print("\n生成可视化图表...")
        plot_levelwise_rmse(
            all_results, 
            save_path=str(results_dir / 'levelwise_rmse.png')
        )
    
    # 保存完整结果
    with open(results_dir / 'all_results.json', 'w') as f:
        # 转换numpy数组为列表
        serializable_results = []
        for r in all_results:
            sr = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    sr[k] = v.tolist()
                else:
                    sr[k] = v
            serializable_results.append(sr)
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n完整结果已保存: {results_dir / 'all_results.json'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
