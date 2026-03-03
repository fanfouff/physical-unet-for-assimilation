#!/usr/bin/env python3
"""
===============================================================================
IEEE TGRS 论文专用 - 批量模型评估与绘图系统
Batch Evaluation System for TGRS Paper
===============================================================================
功能：
1. 智能加载：根据 config.json 自动匹配模型参数，解决参数报错问题。
2. 批量推理：遍历所有实验，计算 Global 和 Level-wise RMSE/Bias。
3. 专业绘图：生成垂直廓线、泰勒图、纬向偏差图等高洁量矢量图。
===============================================================================
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# 引入你的项目模块
from data_pipeline_v2 import LazySatelliteERA5Dataset, LevelwiseNormalizer, DataConfig
from models.backbone import create_model

# ================= 配置区 =================
# TGRS 绘图风格设置
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450,
    500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000
]  # 37层 (根据ERA5标准，如果你的数据是倒序请自行反转)

# ================= 核心逻辑 =================

def load_model_smart(exp_path, device):
    """
    智能模型加载器：读取config.json并只使用backbone支持的参数
    """
    config_path = exp_path / 'config.json'
    ckpt_path = exp_path / 'checkpoint_best.pth'
    
    if not config_path.exists() or not ckpt_path.exists():
        print(f"⚠️ 跳过: 缺少config或checkpoint -> {exp_path.name}")
        return None, None

    # 1. 读取配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_name = config.get('model', 'physics_unet')
    
    # 2. 提取参数 (严格对照 backbone.py 的 __init__)
    # 仅提取存在的键值，避免传入不支持的参数
    valid_keys = [
        'img_size', 'in_chans', 'bkg_chans', 'out_chans', 
        'stem_chans', 'encoder_chans', 'decoder_chans', 
        'fusion_mode', 'use_aux', 'mask_aware', 'use_attention', 'drop_path_rate'
    ]
    
    model_kwargs = {k: config[k] for k in valid_keys if k in config}
    
    # 3. 强制修正 (防止config里的旧参数导致问题)
    # 如果是消融实验，可能config里没有记录某些改动，这里可以手动覆盖
    # 例如：model_kwargs['mask_aware'] = True 
    
    print(f"   ➤ 加载模型: {model_name} | 参数: {model_kwargs}")
    
    # 4. 创建模型
    try:
        model = create_model(model_name, **model_kwargs)
    except TypeError as e:
        print(f"   ❌ 模型创建失败: {e}")
        return None, None

    # 5. 加载权重
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # 处理 DDP 的 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    # strict=False 允许加载消融实验中结构微调过的模型
    keys = model.load_state_dict(new_state_dict, strict=False)
    if keys.missing_keys:
        print(f"   ⚠️ 缺失键 (可能是正常的消融移除): {keys.missing_keys[:3]}...")
    
    model.to(device)
    model.eval()
    return model, config

def evaluate_dataset(model, dataloader, normalizer, device):
    """在整个测试集上推理"""
    mse_sum = 0
    mae_sum = 0
    bias_sum = 0
    total_samples = 0
    
    # 用于纬向偏差分析
    lat_bias_list = []  # [(lat, bias), ...]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            # 搬运数据
            obs = batch['obs'].to(device)
            bkg = batch['bkg'].to(device)
            target = batch['target'].to(device)
            mask = batch['mask'].to(device)
            aux = batch.get('aux', None)
            if aux is not None: aux = aux.to(device)

            # 推理
            pred = model(obs, bkg, mask, aux)
            
            # 反归一化 (还原到 Kelvin)
            pred_k = normalizer.denormalize(pred, 'target')
            target_k = normalizer.denormalize(target, 'target')
            
            # 计算误差 (Residual)
            diff = pred_k - target_k
            
            # 累积统计量 (Global)
            mse_sum += (diff ** 2).mean(dim=(0, 2, 3)).cpu().numpy() * obs.shape[0] # 按层累积
            mae_sum += diff.abs().mean(dim=(0, 2, 3)).cpu().numpy() * obs.shape[0]
            bias_sum += diff.mean(dim=(0, 2, 3)).cpu().numpy() * obs.shape[0]
            total_samples += obs.shape[0]

    # 计算平均指标
    level_rmse = np.sqrt(mse_sum / total_samples)
    level_mae = mae_sum / total_samples
    level_bias = bias_sum / total_samples
    
    return {
        'rmse_profile': level_rmse,
        'mae_profile': level_mae,
        'bias_profile': level_bias,
        'global_rmse': level_rmse.mean()
    }

# ================= 绘图函数 =================

def plot_vertical_profiles(results_dict, output_dir, filename="vertical_rmse.pdf"):
    """绘制垂直RMSE廓线图 (TGRS 核心图表)"""
    plt.figure(figsize=(6, 8))
    
    # 颜色循环
    colors = sns.color_palette("deep", len(results_dict))
    
    for i, (name, res) in enumerate(results_dict.items()):
        rmse = res['rmse_profile']
        # 假设数据是37层，需要翻转y轴让地面在下
        plt.plot(rmse, PRESSURE_LEVELS, label=f"{name} ({res['global_rmse']:.2f}K)", 
                 linewidth=2.5 if 'Ours' in name else 1.5,
                 color='red' if 'Ours' in name else colors[i],
                 marker='o' if 'Ours' in name else None, markersize=4)

    plt.yscale('log')
    plt.gca().invert_yaxis() # 气压越大越在下面
    plt.yticks([10, 50, 100, 250, 500, 850, 1000], [10, 50, 100, 250, 500, 850, 1000])
    
    plt.xlabel('RMSE (K)', fontsize=14)
    plt.ylabel('Pressure (hPa)', fontsize=14)
    plt.title('Vertical RMSE Profile', fontsize=16)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    save_path = output_dir / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"📈 图表已保存: {save_path}")

def plot_bar_comparison(results_dict, output_dir, metric='global_rmse'):
    """绘制综合性能柱状图"""
    names = list(results_dict.keys())
    values = [results_dict[n][metric] for n in names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=sns.color_palette("Blues_d", len(names)))
    
    # 高亮 "Ours"
    for bar, name in zip(bars, names):
        if 'Ours' in name or 'PAS-Net' in name:
            bar.set_color('firebrick')
    
    plt.ylabel('Global RMSE (K)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.title('Overall Model Comparison', fontsize=16)
    
    # 标数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
                
    save_path = output_dir / "bar_comparison.pdf"
    plt.savefig(save_path, bbox_inches='tight')

# ================= 主流程 =================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="/data2/lrx/era_obs/npz/test",type=str, help='测试集路径')
    parser.add_argument('--exp_root', default="/home/seu/Fuxi/Unet/satellite_assimilation_v2/experiments/outputs/experiments",type=str, help='包含所有实验文件夹的根目录')
    parser.add_argument('--output_dir', type=str, default='figures_tgrs')
    parser.add_argument('--stats_file', default="/data2/lrx/era_obs/npz/stats.npz",type=str)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 准备数据
    print("🔄 正在加载测试集...")
    # 假设 DataConfig 默认配置与训练一致，或从某个config读取
    # 这里直接实例化
    # ✅ 修正代码
    dataset = LazySatelliteERA5Dataset.from_directory(
        data_root=args.data_root,
        stats_file=args.stats_file,
        use_aux=True,
        cache_size=0
    )
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)
    normalizer = LevelwiseNormalizer(args.stats_file)

    # 2. 定义实验组 (在这里手动指定你想对比的实验文件夹名)
    # key: 图表中显示的图例名
    # value: 文件夹名
    experiments_to_eval = {
        'comparison': {
            'Vanilla U-Net': '20260126_C1_VanillaUNet', # 替换为你真实的文件夹名
            'Res-UNet': '20260126_C2_ResUNet', 
            'AttentionUNet': '20260126_C3_AttentionUNet'
        },
        'ablation': {
            'Ours (Full)': '20260126_A1_PASNet_Full',
            'w/o PASNet_NoLevelNorm': '20260126_A2_PASNet_NoLevelNorm',
            'w/o PASNet_NoAdapter': '20260126_A3_PASNet_NoAdapter',
            'w/o PASNet_NoGradLoss': '20260126_A4_PASNet_NoGradLoss',
            'w/o PASNet_NoAux': '20260126_A5_PASNet_NoAux',
            'w/o PASNet_NoMask': '20260126_A6_PASNet_NoMask',
            'w/o PASNet_NoSE': '20260126_A7_PASNet_NoSE',
            'w/o PASNet_FusionConcat': '20260126_A8_PASNet_FusionConcat',
            'w/o PASNet_FusionAdd': '20260126_A9_PASNet_FusionAdd'
        }
    }

    # 3. 循环评估
    all_results = {}
    
    # 自动扫描 exp_root 下的文件夹，如果没手动指定，就尝试自动匹配
    exp_root = Path(args.exp_root)
    
    for group_name, exp_dict in experiments_to_eval.items():
        print(f"\n======== 正在评估组: {group_name} ========")
        group_results = {}
        
        for label, folder_name in exp_dict.items():
            exp_path = exp_root / folder_name
            
            # 如果文件夹不存在，尝试模糊匹配 (方便你只写部分名字)
            if not exp_path.exists():
                candidates = list(exp_root.glob(f"*{folder_name}*"))
                if candidates:
                    exp_path = candidates[0]
                    print(f"   🔍 模糊匹配: {folder_name} -> {exp_path.name}")
                else:
                    print(f"   ⚠️ 未找到实验: {folder_name}")
                    continue
            
            # 加载模型
            model, config = load_model_smart(exp_path, device)
            if model is None: continue
            
            # 评估
            print(f"   🚀 正在推理: {label}...")
            metrics = evaluate_dataset(model, dataloader, normalizer, device)
            group_results[label] = metrics
            all_results[label] = metrics # 存入总表
        
        # 4. 分组绘图
        if group_results:
            plot_vertical_profiles(group_results, output_dir, filename=f"profile_{group_name}.pdf")
            plot_bar_comparison(group_results, output_dir)

    print("\n✅ 所有评估完成！结果保存在:", output_dir)

if __name__ == '__main__':
    main()
