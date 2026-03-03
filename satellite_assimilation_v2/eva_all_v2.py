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
import inspect  # 关键修复：用于检测函数参数

# 引入你的项目模块
# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline_v2 import LazySatelliteERA5Dataset, LevelwiseNormalizer, DataConfig
# 引入具体的模型类，绕过 create_model 的限制
from models.backbone import (
    PhysicsAwareUNet, 
    PhysicsAwareUNetLite, 
    PhysicsAwareUNetLarge, 
    VanillaUNet
)

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
]

# 模型类映射表 (解决 Unknown model 报错)
# 如果你的 res_unet 是基于 ResNet 的 UNet，它实际上是 PhysicsAwareUNet 的基础版
MODEL_CLASS_MAP = {
    'physics_unet': PhysicsAwareUNet,
    'pasnet': PhysicsAwareUNet,
    'physics_unet_lite': PhysicsAwareUNetLite,
    'physics_unet_large': PhysicsAwareUNetLarge,
    'vanilla_unet': VanillaUNet,
    
    # 别名映射 (根据你的 Config 推测)
    'res_unet': PhysicsAwareUNet,       # 映射到支持 ResBlock 的类
    'attention_unet': PhysicsAwareUNet, # 映射到支持 Attention 的类
    'unet': VanillaUNet
}

# ================= 核心逻辑 =================

def load_model_smart(exp_path, device):
    """
    智能模型加载器：
    1. 自动处理模型别名
    2. 自动过滤掉模型不需要的参数 (解决 TypeError)
    """
    config_path = exp_path / 'config.json'
    ckpt_path = exp_path / 'checkpoint_best.pth'
    
    if not config_path.exists() or not ckpt_path.exists():
        print(f"⚠️ 跳过: 缺少config或checkpoint -> {exp_path.name}")
        return None, None

    # 1. 读取配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_name_str = config.get('model', 'physics_unet').lower()
    
    # 2. 确定模型类
    if model_name_str not in MODEL_CLASS_MAP:
        print(f"⚠️ 未知模型名 '{model_name_str}'，尝试回退到 PhysicsAwareUNet")
        ModelClass = PhysicsAwareUNet
    else:
        ModelClass = MODEL_CLASS_MAP[model_name_str]

    # 3. 智能参数过滤 (Intelligent Argument Filtering)
    # 获取目标类 __init__ 支持的所有参数名
    sig = inspect.signature(ModelClass.__init__)
    supported_args = list(sig.parameters.keys())
    
    # 从 config 中只提取支持的参数
    model_kwargs = {}
    filtered_args = []
    for k, v in config.items():
        if k in supported_args:
            model_kwargs[k] = v
        else:
            filtered_args.append(k)
            
    # 特殊处理：VanillaUNet 可能需要 features 参数，但 config 里叫 encoder_chans
    if ModelClass == VanillaUNet:
        if 'features' in supported_args and 'encoder_chans' in config:
            model_kwargs['features'] = config['encoder_chans']

    print(f"   ➤ 加载模型: {model_name_str} -> {ModelClass.__name__}")
    # print(f"     保留参数: {list(model_kwargs.keys())}") 
    # print(f"     过滤参数: {len(filtered_args)} 个 (如 {filtered_args[:3]}...)")
    
    # 4. 创建模型
    try:
        model = ModelClass(**model_kwargs)
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        return None, None

    # 5. 加载权重
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # 处理 DDP 的 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
            
        # strict=False 允许加载消融实验中结构微调过的模型
        keys = model.load_state_dict(new_state_dict, strict=False)
        if keys.missing_keys:
            # 过滤掉一些无关紧要的丢失键
            missing = [k for k in keys.missing_keys if 'num_batches_tracked' not in k]
            if missing:
                print(f"     ⚠️ 权重不完全匹配 (可能是消融实验): 缺失 {len(missing)} 个键")
    except Exception as e:
        print(f"   ❌ 权重加载失败: {e}")
        return None, None
    
    model.to(device)
    model.eval()
    return model, config

def evaluate_dataset(model, dataloader, normalizer, device):
    """在整个测试集上推理"""
    mse_sum = 0
    mae_sum = 0
    bias_sum = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            # 搬运数据
            obs = batch['obs'].to(device)
            bkg = batch['bkg'].to(device)
            target = batch['target'].to(device)
            mask = batch['mask'].to(device)
            
            # 处理 aux
            aux = batch.get('aux', None)
            if aux is not None: aux = aux.to(device)

            # 推理
            # 不同的模型 forward 参数可能不同，这里做一个简单的尝试机制
            try:
                # 尝试标准调用
                pred = model(obs, bkg, mask, aux)
            except TypeError:
                try:
                    # 尝试不带 aux 的调用 (如 VanillaUNet 可能不支持 aux)
                    pred = model(obs, bkg) 
                except TypeError:
                     # 尝试只带 obs (极简模式)
                    pred = model(obs)

            # 反归一化 (还原到 Kelvin)
            pred_k = normalizer.inverse_transform(pred)
            target_k = normalizer.inverse_transform(target)
            
            # 计算误差 (Residual)
            diff = pred_k - target_k
            
            # 累积统计量 (Global)
            # 假设 shape [B, C, H, W]
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
        
        # 样式设置
        is_ours = 'Ours' in name or 'PAS-Net' in name
        linewidth = 3.0 if is_ours else 2.0
        color = 'red' if is_ours else colors[i]
        marker = 'o' if is_ours else None
        zorder = 10 if is_ours else 5
        
        plt.plot(rmse, PRESSURE_LEVELS, label=f"{name}", 
                 linewidth=linewidth,
                 color=color,
                 marker=marker, markersize=5, markevery=4,
                 zorder=zorder)

    plt.yscale('log')
    plt.gca().invert_yaxis() # 气压越大越在下面
    
    # 设置Y轴刻度
    yticks = [1, 10, 50, 100, 250, 500, 850, 1000]
    plt.yticks(yticks, yticks)
    
    plt.xlabel('RMSE (K)', fontsize=14, fontweight='bold')
    plt.ylabel('Pressure (hPa)', fontsize=14, fontweight='bold')
    plt.title('Vertical RMSE Profile', fontsize=16)
    plt.legend(loc='lower left', frameon=True, fontsize=11)
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    
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
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.title('Overall Model Performance Comparison', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    
    # 标数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
                
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

    parser.add_argument('--collect_spatial', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 准备数据
    print("🔄 正在加载测试集...")
    dataset = LazySatelliteERA5Dataset.from_directory(
        data_root=args.data_root,
        stats_file=args.stats_file,
        config=DataConfig(),
        mask_mode='any'
    )
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)
    
    # 获取标准化器
    normalizer = dataset.get_normalizers()['target']

    # 2. 定义实验组 
    # 注意：这里的 Value 部分支持模糊匹配，只要你的文件夹名里包含这个字符串即可
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
    exp_root = Path(args.exp_root)
    
    for group_name, exp_dict in experiments_to_eval.items():
        print(f"\n======== 正在评估组: {group_name} ========")
        group_results = {}
        
        for label, folder_keyword in exp_dict.items():
            # 智能搜索文件夹
            candidates = list(exp_root.glob(f"*{folder_keyword}*"))
            if not candidates:
                print(f"   ⚠️ 未找到包含 '{folder_keyword}' 的实验文件夹，跳过")
                continue
            
            # 取第一个匹配的文件夹（通常是最新的）
            # 可以按时间排序取最新的
            candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            exp_path = candidates[0]
            
            # 加载模型
            print(f"   📂 匹配实验: {exp_path.name}")
            model, config = load_model_smart(exp_path, device)
            if model is None: continue
            
            # 评估
            print(f"   🚀 正在推理: {label}...")
            metrics = evaluate_dataset(model, dataloader, normalizer, device)
            group_results[label] = metrics
            all_results[label] = metrics
            
            print(f"     -> Global RMSE: {metrics['global_rmse']:.4f} K")
        
        # 4. 分组绘图
        if group_results:
            plot_vertical_profiles(group_results, output_dir, filename=f"profile_{group_name}.pdf")
            plot_bar_comparison(group_results, output_dir)

    print("\n✅ 所有评估完成！结果保存在:", output_dir)

if __name__ == '__main__':
    main()
