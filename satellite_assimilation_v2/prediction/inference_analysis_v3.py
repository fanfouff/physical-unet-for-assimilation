#!/usr/bin/env python3
"""
===============================================================================
卫星数据同化 - 评估与 Case Study 脚本
功能:
    1. 加载 best_model.pth 进行推理
    2. 计算各个高度层/通道的 RMSE Improvement over Background (%)
    3. 绘制特定样本的 Case Study 对比图 (Bkg, Obs, Pred, Target, Errors)
===============================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, random_split

# 导入项目自带的模块 (与 train_ddp.py 保持一致)
sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline_v2 import LazySatelliteERA5Dataset, InMemorySatelliteDataset
from models.backbone import create_model, UNetConfig

def parse_args():
    parser = argparse.ArgumentParser(description='评估与 Case Study 脚本')
    parser.add_argument('--checkpoint', type=str, required=True, help='best_model.pth 的路径')
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录')
    parser.add_argument('--stats_file', type=str, required=True, help='统计量文件 (stats.npz) 路径')
    parser.add_argument('--case_idx', type=int, default=0, help='用于 Case Study 的测试集样本索引')
    parser.add_argument('--channel_idx', type=int, default=0, help='用于 Case Study 可视化的高度层/通道索引')
    parser.add_argument('--batch_size', type=int, default=16, help='推理 Batch Size')
    return parser.parse_args()

def load_model_and_args(checkpoint_path, device):
    """加载模型及其训练时的配置"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = argparse.Namespace(**checkpoint['args'])
    
    # 构建模型配置
    if train_args.model not in ('vanilla_unet', 'fuxi_da'):
        config = UNetConfig(
            fusion_mode=train_args.fusion_mode,
            use_aux=train_args.use_aux,
            mask_aware=train_args.mask_aware,
            use_spectral_stem=getattr(train_args, 'use_spectral_stem', True),
            deep_supervision=False # 推理时不需要 deep supervision
        )
        model = create_model(train_args.model, config=config)
    else:
        model = create_model(train_args.model)
    
    # 加载权重 (处理 DDP 留下的 'module.' 前缀)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    
    return model, train_args

def get_test_loader(train_args, data_root, batch_size):
    """复现训练时的数据划分以获取纯净的测试集"""
    data_root = Path(data_root)
    _all_files = sorted(f for f in data_root.glob('**/*.npz') 
                        if f.name not in ('stats.npz', 'dataset_split.json', 'increment_stats.npz'))
    
    # 简单的文件过滤 (同 train_ddp.py)
    file_list = []
    for _f in _all_files:
        try:
            _d = np.load(str(_f))
            if _d['target'].sum() != 0:
                file_list.append(str(_f))
        except Exception:
            pass
            
    dataset = LazySatelliteERA5Dataset(file_list=file_list, use_aux=train_args.use_aux)
    
    n_total = len(dataset)
    n_train = int(n_total * train_args.train_ratio)
    n_val = int(n_total * train_args.val_ratio)
    n_test = n_total - n_train - n_val
    
    # 必须使用相同的 seed 才能保证切分出来的 test_set 和训练时一致
    _, _, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(train_args.seed)
    )
    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader, test_set

def denormalize(tensor_norm, mean, std, device):
    """将归一化的张量还原为物理量"""
    mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std = torch.tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    return tensor_norm * std + mean

def evaluate_metrics(model, test_loader, train_args, stats, device):
    """计算 per pressure level RMSE Improvement"""
    print("正在测试集上计算物理 RMSE 指标...")
    
    # 提取统计量
    bkg_mean, bkg_std = stats['bkg_mean'], stats['bkg_std']
    tgt_mean, tgt_std = stats['target_mean'], stats['target_std']
    
    use_inc = getattr(train_args, 'use_increment', False)
    if use_inc:
        inc_stats = np.load(train_args.increment_stats)
        inc_mean, inc_std = inc_stats['inc_mean'], inc_stats['inc_std']

    mse_pred_sum = 0
    mse_bkg_sum = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            obs = batch['obs'].to(device)
            bkg = batch['bkg'].to(device)
            mask = batch['mask'].to(device)
            target = batch['target'].to(device)
            aux = batch.get('aux').to(device) if batch.get('aux') is not None else None
            
            # 模型推理
            output = model(obs, bkg, mask, aux)
            pred = output[0] if isinstance(output, tuple) else output
            
            # 还原为物理空间 (Physical Space)
            bkg_phys = denormalize(bkg, bkg_mean, bkg_std, device)
            tgt_phys = denormalize(target, tgt_mean, tgt_std, device)
            
            if use_inc:
                # 如果训练的是增量，模型预测的是归一化的增量
                pred_inc_phys = denormalize(pred, inc_mean, inc_std, device)
                pred_phys = bkg_phys + pred_inc_phys
            else:
                # 否则预测的就是目标场
                pred_phys = denormalize(pred, tgt_mean, tgt_std, device)
            
            # 累加每个 channel (Level) 的 Squared Error
            # 维度假定为 (B, C, H, W)
            se_pred = (pred_phys - tgt_phys) ** 2
            se_bkg = (bkg_phys - tgt_phys) ** 2
            
            # 按照 (B, H, W) 维度求和，保留 C 维度
            mse_pred_sum += se_pred.sum(dim=(0, 2, 3))
            mse_bkg_sum += se_bkg.sum(dim=(0, 2, 3))
            total_samples += bkg.shape[0] * bkg.shape[2] * bkg.shape[3]

    # 计算最终 RMSE
    rmse_pred = torch.sqrt(mse_pred_sum / total_samples).cpu().numpy()
    rmse_bkg = torch.sqrt(mse_bkg_sum / total_samples).cpu().numpy()
    
    # 计算提升百分比
    improvement = (rmse_bkg - rmse_pred) / rmse_bkg * 100
    
    print("\n" + "="*50)
    print("高度层/通道评估结果 (RMSE in Physical Units):")
    print(f"{'Level/Ch':<10} | {'Bkg RMSE':<10} | {'Pred RMSE':<10} | {'Improvement (%)':<15}")
    print("-" * 50)
    for c in range(len(rmse_bkg)):
        print(f"Channel {c:<7} | {rmse_bkg[c]:<10.4f} | {rmse_pred[c]:<10.4f} | {improvement[c]:.2f}%")
    print("="*50)
    
    return rmse_pred, rmse_bkg, improvement

def plot_case_study(model, test_set, train_args, stats, device, case_idx, channel_idx):
    """绘制单样本直观对比图"""
    print(f"\n正在生成 Case Study 图像 (Sample Index: {case_idx}, Channel: {channel_idx})...")
    
    data = test_set[case_idx]
    
    # 增加 Batch 维度并移动到 device
    obs = data['obs'].unsqueeze(0).to(device)
    bkg = data['bkg'].unsqueeze(0).to(device)
    mask = data['mask'].unsqueeze(0).to(device)
    target = data['target'].unsqueeze(0).to(device)
    aux = data['aux'].unsqueeze(0).to(device) if 'aux' in data else None
    
    with torch.no_grad():
        output = model(obs, bkg, mask, aux)
        pred = output[0] if isinstance(output, tuple) else output

    # 反归一化
    bkg_phys = denormalize(bkg, stats['bkg_mean'], stats['bkg_std'], device)[0, channel_idx].cpu().numpy()
    tgt_phys = denormalize(target, stats['target_mean'], stats['target_std'], device)[0, channel_idx].cpu().numpy()
    
    use_inc = getattr(train_args, 'use_increment', False)
    if use_inc:
        inc_stats = np.load(train_args.increment_stats)
        pred_inc_phys = denormalize(pred, inc_stats['inc_mean'], inc_stats['inc_std'], device)
        pred_phys = (denormalize(bkg, stats['bkg_mean'], stats['bkg_std'], device) + pred_inc_phys)[0, channel_idx].cpu().numpy()
    else:
        pred_phys = denormalize(pred, stats['target_mean'], stats['target_std'], device)[0, channel_idx].cpu().numpy()
    
    # 提取观测并遮罩
    obs_phys = denormalize(obs, stats['obs_mean'], stats['obs_std'], device)[0, channel_idx].cpu().numpy()
    mask_np = mask[0, 0].cpu().numpy() # 假设 mask 在空间上是共享的或者取第0个channel
    obs_phys_masked = np.where(mask_np > 0.5, obs_phys, np.nan)
    
    # 计算误差
    error_bkg = bkg_phys - tgt_phys
    error_pred = pred_phys - tgt_phys
    
    # 画图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Case Study (Test Index: {case_idx}, Channel/Level: {channel_idx})', fontsize=16)
    
    vmin = np.nanmin(tgt_phys)
    vmax = np.nanmax(tgt_phys)
    err_vmax = max(np.abs(error_bkg).max(), np.abs(error_pred).max())
    
    # 1. Background
    im0 = axes[0, 0].imshow(bkg_phys, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Background (Bkg)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # 2. Observation
    im1 = axes[0, 1].imshow(obs_phys_masked, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Observation (Masked)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 3. Target
    im2 = axes[0, 2].imshow(tgt_phys, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('Target (Truth)')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 4. Prediction
    im3 = axes[1, 0].imshow(pred_phys, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Model Prediction')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 5. Background Error
    im4 = axes[1, 1].imshow(error_bkg, cmap='RdBu_r', vmin=-err_vmax, vmax=err_vmax)
    axes[1, 1].set_title(f'Bkg Error (RMSE: {np.sqrt(np.mean(error_bkg**2)):.3f})')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # 6. Prediction Error
    im5 = axes[1, 2].imshow(error_pred, cmap='RdBu_r', vmin=-err_vmax, vmax=err_vmax)
    axes[1, 2].set_title(f'Pred Error (RMSE: {np.sqrt(np.mean(error_pred**2)):.3f})')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    save_path = f'case_study_idx{case_idx}_ch{channel_idx}.png'
    plt.savefig(save_path, dpi=300)
    print(f"可视化图像已保存至: {save_path}")
    plt.show()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型与配置
    print(f"加载检查点: {args.checkpoint}")
    model, train_args = load_model_and_args(args.checkpoint, device)
    
    # 2. 加载统计量
    print(f"加载统计量: {args.stats_file}")
    stats = np.load(args.stats_file)
    
    # 3. 获取测试集
    print(f"加载测试集...")
    test_loader, test_set = get_test_loader(train_args, args.data_root, args.batch_size)
    print(f"测试集样本数: {len(test_set)}")
    
    # 4. 评估指标 (Per Level RMSE Improvement)
    evaluate_metrics(model, test_loader, train_args, stats, device)
    
    # 5. 绘制 Case Study
    plot_case_study(model, test_set, train_args, stats, device, args.case_idx, args.channel_idx)

if __name__ == '__main__':
    main()