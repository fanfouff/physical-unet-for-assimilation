"""
eval_all_experiments.py — 所有实验汇总评估

运行:
  python3 prediction/eval_all_experiments.py \
    --test_root /data2/lrx/npz_64_real/test \
    --stats_file /data2/lrx/npz_64_real/stats.npz \
    --increment_stats /data2/lrx/npz_64_real/increment_stats.npz \
    --output_dir prediction/figures_ablation_comparison
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
warnings.filterwarnings("ignore")

PRESSURE_LEVELS = [1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,
                   300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000]


class LevelwiseNormalizer:
    def __init__(self, mean, std): self.mean, self.std = mean, std
    def inverse_transform(self, x): return x * self.std[:,None,None] + self.mean[:,None,None]


def load_test_files(root):
    EXCL = {'increment_stats.npz','stats.npz'}
    out = []
    for f in sorted(Path(root).glob("*.npz")):
        if f.name in EXCL: continue
        try:
            d = np.load(f)
            if d['target'].sum() != 0: out.append(f)
        except: pass
    return out


@torch.no_grad()
def evaluate_dl_model(ckpt_path, test_files, stats, inc_stats, device='cuda'):
    # 项目根目录: prediction/../ = satellite_assimilation_v2
    _proj_root = str(Path(__file__).parent.parent)
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    from models.backbone import create_model, UNetConfig

    ckpt = torch.load(ckpt_path, map_location='cpu')
    _raw_args = ckpt.get('args', {})
    if isinstance(_raw_args, dict):
        model_args = argparse.Namespace(**_raw_args)
    elif isinstance(_raw_args, argparse.Namespace):
        model_args = _raw_args
    else:
        model_args = argparse.Namespace()

    model_name = getattr(model_args, 'model', 'physics_unet')

    if model_name in ('vanilla_unet', 'fuxi_da'):
        model = create_model(model_name)
    else:
        cfg = UNetConfig(
            fusion_mode=getattr(model_args, 'fusion_mode', 'gated'),
            use_aux=getattr(model_args, 'use_aux', True),
            mask_aware=getattr(model_args, 'mask_aware', True),
            use_spectral_stem=getattr(model_args, 'use_spectral_stem', True),
        )
        model = create_model(model_name, config=cfg)

    sd = {k.replace('module.',''):v for k,v in ckpt['model_state_dict'].items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    bkg_norm = LevelwiseNormalizer(stats['bkg_mean'], stats['bkg_std'])
    tgt_norm = LevelwiseNormalizer(stats['target_mean'], stats['target_std'])
    inc_norm = LevelwiseNormalizer(inc_stats['inc_mean'], inc_stats['inc_std']) if inc_stats else None

    pl_sq = np.zeros(37); pl_cnt = np.zeros(37)
    rmse_all, rmse_bkg_all, mae_all, bias_all, corr_all = [],[],[],[],[]

    # Build obs_normalizer from stats
    obs_norm_fn = LevelwiseNormalizer(stats['obs_mean'], stats['obs_std']) if 'obs_mean' in stats else None

    for f in tqdm(test_files, desc=f"  {Path(ckpt_path).parent.name[:35]}", leave=False):
        try:
            d = np.load(f)
            # 物理值 → 归一化值（与训练时DataLoader一致）
            obs_phys = d['obs']
            bkg_phys_raw = d['bkg']
            tgt_phys = d['target']
            mask_raw = d['mask']
            # 归一化: (x - mean) / std
            def norm(x, mean, std): return (x - mean[:,None,None]) / std[:,None,None]
            obs_normalized = norm(obs_phys, stats['obs_mean'], stats['obs_std']) if 'obs_mean' in stats else obs_phys
            bkg_normalized = norm(bkg_phys_raw, stats['bkg_mean'], stats['bkg_std'])
            obs_n = torch.tensor(obs_normalized).float().unsqueeze(0).to(device)
            bkg_n = torch.tensor(bkg_normalized).float().unsqueeze(0).to(device)
            mask  = torch.tensor(mask_raw).float().unsqueeze(0).to(device)
            aux   = torch.tensor(d['aux']).float().unsqueeze(0).to(device) if 'aux' in d else None
        except: continue

        pred = model(obs_n, bkg_n, mask, aux).squeeze(0).cpu().numpy()
        # 反归一化: 增量模式 → analysis = bkg_phys + inc_phys
        bkg_phys = bkg_phys_raw
        if inc_norm:
            ana_phys = bkg_phys + inc_norm.inverse_transform(pred)
        else:
            ana_phys = tgt_norm.inverse_transform(pred)

        sq = (ana_phys-tgt_phys)**2; sq_b = (bkg_phys-tgt_phys)**2
        pl_sq += sq.reshape(37,-1).mean(1); pl_cnt += 1
        rmse_all.append(float(np.sqrt(sq.mean())))
        rmse_bkg_all.append(float(np.sqrt(sq_b.mean())))
        mae_all.append(float(np.abs(ana_phys-tgt_phys).mean()))
        bias_all.append(float((ana_phys-tgt_phys).mean()))
        a=ana_phys.ravel(); b=tgt_phys.ravel()
        corr_all.append(float(np.corrcoef(a,b)[0,1]))

    return {
        'rmse': float(np.mean(rmse_all)), 'rmse_bkg': float(np.mean(rmse_bkg_all)),
        'mae': float(np.mean(mae_all)), 'bias': float(np.mean(bias_all)),
        'corr': float(np.mean(corr_all)), 'n_files': len(rmse_all),
        'per_level_rmse': np.sqrt(pl_sq/np.maximum(pl_cnt,1)),
    }


def get_experiments(base_dir):
    outs = Path(base_dir)/'train_ddp'/'outputs'
    return [
        {'id':'full',     'label':'Ours (Full)',           'type':'full',     'ckpt': str(outs/'increment_era5_bkg_64x64'/'best_model.pth')},
        {'id':'v1',       'label':'w/o Aux (V1)',          'type':'ablation', 'ckpt': str(outs/'ablation_v1_no_aux'/'experiment_ddp'/'best_model.pth')},
        {'id':'v2',       'label':'w/o MaskConv (V2)',     'type':'ablation', 'ckpt': str(outs/'ablation_v2_no_mask_aware'/'experiment_ddp'/'best_model.pth')},
        {'id':'v3',       'label':'w/o GatedFusion (V3)',  'type':'ablation', 'ckpt': str(outs/'ablation_v3_fusion_add'/'experiment_ddp'/'best_model.pth')},
        {'id':'v4',       'label':'w/o SpectralStem (V4)', 'type':'ablation', 'ckpt': str(outs/'ablation_v4_no_spectral_stem'/'experiment_ddp'/'best_model.pth')},
        {'id':'b3',       'label':'VanillaUNet (B3)',      'type':'compare',  'ckpt': str(outs/'compare_b3_vanilla_unet'/'experiment_ddp'/'best_model.pth')},
        {'id':'b4',       'label':'FuXi-DA (B4)',          'type':'compare',  'ckpt': str(outs/'compare_b4_fuxi_da'/'experiment_ddp'/'best_model.pth')},
    ]


def print_table(rows):
    print("\n| 方法 | 类型 | RMSE(K) | MAE(K) | Bias(K) | Corr | 改善% |")
    print("|------|------|---------|--------|---------|------|-------|")
    for r in rows:
        rmse = f"{r['rmse']:.4f}" if not (r['rmse'] != r['rmse']) else "—"
        mae  = f"{r.get('mae',0):.4f}" if 'mae' in r else "—"
        bias = f"{r.get('bias',0):+.4f}" if 'bias' in r else "—"
        corr = f"{r.get('corr',0):.5f}" if 'corr' in r else "—"
        imp  = f"{r.get('improve_pct',0):+.2f}%" if 'improve_pct' in r else "—"
        print(f"| {r['label']} | {r.get('type','')} | {rmse} | {mae} | {bias} | {corr} | {imp} |")

    print("\n% LaTeX:")
    print(r"\begin{table}[h]\centering\caption{Ablation \& Comparison Results}")
    print(r"\begin{tabular}{llccccc}\toprule")
    print(r"Method & Type & RMSE(K) & MAE(K) & Bias(K) & Corr & Improv.\\\midrule")
    for r in rows:
        label = r['label']; t = r.get('type','')
        rmse = f"{r['rmse']:.4f}" if r.get('rmse',float('nan'))==r.get('rmse',float('nan')) else "—"
        mae  = f"{r.get('mae',0):.4f}"
        bias = f"{r.get('bias',0):+.4f}"
        corr = f"{r.get('corr',0):.5f}" if 'corr' in r else "—"
        imp  = f"{r.get('improve_pct',0):+.2f}\\%"
        print(f"{label} & {t} & {rmse} & {mae} & {bias} & {corr} & {imp}\\\\")
    print(r"\bottomrule\end{tabular}\end{table}")


def plot_profiles(rows, out_path):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cmap = {'bkg':'grey','oi':'#27ae60','full':'black','ablation':'#e67e22','compare':'#8e44ad'}
        lsty = {'bkg':'--','oi':'--','full':'-','ablation':'-.','compare':':'}
        fig, ax = plt.subplots(figsize=(7,9))
        for r in rows:
            if 'per_level_rmse' not in r: continue
            t = r.get('type',''); c = cmap.get(t,'#95a5a6'); ls = lsty.get(t,'-')
            lw = 2.5 if t=='full' else 1.8 if t=='compare' else 1.5
            ax.plot(r['per_level_rmse'], PRESSURE_LEVELS, color=c, ls=ls, lw=lw, label=r['label'])
        ax.set_xlabel('RMSE (K)',fontsize=12); ax.set_ylabel('Pressure (hPa)',fontsize=12)
        ax.set_title('Vertical RMSE: Ablation & Comparison',fontsize=13)
        ax.invert_yaxis(); ax.set_yscale('log')
        ax.set_yticks([10,50,100,200,500,1000]); ax.set_yticklabels(['10','50','100','200','500','1000'])
        ax.legend(fontsize=9,loc='lower right'); ax.grid(alpha=0.25)
        plt.tight_layout(); plt.savefig(out_path,dpi=200,bbox_inches='tight'); plt.close()
        print(f"  图表: {out_path}")
    except Exception as e:
        print(f"  (绘图失败: {e})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_root',       default='/data2/lrx/npz_64_real/test')
    p.add_argument('--stats_file',      default='/data2/lrx/npz_64_real/stats.npz')
    p.add_argument('--increment_stats', default='/data2/lrx/npz_64_real/increment_stats.npz')
    p.add_argument('--output_dir',      default='prediction/figures_ablation_comparison')
    p.add_argument('--device',          default='cuda')
    p.add_argument('--base_dir',        default=str(Path(__file__).parent.parent))
    p.add_argument('--skip_missing',    action='store_true')
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stats     = dict(np.load(args.stats_file))
    inc_stats = dict(np.load(args.increment_stats)) if Path(args.increment_stats).exists() else None
    test_files = load_test_files(args.test_root)
    print(f"测试文件: {len(test_files)}")

    # 背景场基线
    print("\n[背景场] 计算...")
    pl_bkg=np.zeros(37); pl_cnt=np.zeros(37); rmse_bkg_all=[]
    for f in tqdm(test_files,desc="背景场",leave=False):
        d=np.load(f); bp=d['bkg']; tp=d['target']  # 已是物理值，无需反归一化
        sq=(bp-tp)**2; pl_bkg+=sq.reshape(37,-1).mean(1); pl_cnt+=1
        rmse_bkg_all.append(float(np.sqrt(sq.mean())))
    bkg_rmse = float(np.mean(rmse_bkg_all))
    bkg_per  = np.sqrt(pl_bkg/np.maximum(pl_cnt,1))
    rows = [{'id':'bkg','label':'Background (ERA5)','type':'bkg',
             'rmse':bkg_rmse,'mae':bkg_rmse,'bias':0.,'corr':0.,'improve_pct':0.,
             'n_files':len(rmse_bkg_all),'per_level_rmse':bkg_per}]
    print(f"  背景场 RMSE: {bkg_rmse:.4f} K")

    # OI基线
    oi_dir = Path(args.base_dir)/'prediction'/'oi_results_64'
    if (oi_dir/'metrics.npy').exists():
        om = np.load(oi_dir/'metrics.npy',allow_pickle=True).item()
        pl_oi = np.load(oi_dir/'per_level_rmse_ana.npy') if (oi_dir/'per_level_rmse_ana.npy').exists() else bkg_per
        rows.append({'id':'b2','label':'OI/1DVar (B2)','type':'oi',
                     'rmse':om['rmse_ana'],'mae':om.get('mae_ana',0.),'bias':om.get('bias_ana',0.),
                     'corr':0.,'improve_pct':om['improve_pct'],'n_files':om['n_files'],'per_level_rmse':pl_oi})
        print(f"[OI] RMSE: {om['rmse_ana']:.4f} K  改善: {om['improve_pct']:.2f}%")

    # DL模型
    for exp in get_experiments(args.base_dir):
        ckpt = exp['ckpt']
        if not Path(ckpt).exists():
            if args.skip_missing:
                print(f"\n[跳过] {exp['label']}")
                continue
            rows.append({**exp,'rmse':float('nan'),'mae':float('nan'),'bias':float('nan'),
                         'corr':float('nan'),'improve_pct':float('nan'),'n_files':0})
            continue
        print(f"\n[{exp['id']}] 评估: {exp['label']}")
        try:
            res = evaluate_dl_model(ckpt, test_files, stats, inc_stats, device=args.device)
            imp = (bkg_rmse - res['rmse']) / bkg_rmse * 100
            rows.append({**exp, **res, 'improve_pct': imp})
            print(f"  RMSE={res['rmse']:.4f}K  改善={imp:.2f}%")
        except Exception as e:
            print(f"  ERROR: {e}")

    print_table(rows)

    # 保存
    save = [{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items() if k!='per_level_rmse'} for r in rows]
    with open(out_dir/'results_summary.json','w') as f: json.dump(save,f,indent=2,ensure_ascii=False)
    print(f"\n保存: {out_dir}/results_summary.json")

    valid = [r for r in rows if 'per_level_rmse' in r and r.get('rmse',float('nan'))==r.get('rmse',float('nan'))]
    plot_profiles(valid, str(out_dir/'vertical_rmse_comparison.png'))

    # 柱状图
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        color_map = {'bkg':'#bdc3c7','oi':'#27ae60','full':'#2c3e50','ablation':'#e67e22','compare':'#8e44ad'}
        vrows = [r for r in rows if r.get('rmse',float('nan'))==r.get('rmse',float('nan'))]
        labels=[r['label'] for r in vrows]; rmse=[r['rmse'] for r in vrows]; types=[r.get('type','') for r in vrows]
        colors=[color_map.get(t,'#95a5a6') for t in types]
        fig,ax=plt.subplots(figsize=(10,5.5))
        bars=ax.barh(labels,rmse,color=colors,edgecolor='white',height=0.6)
        for bar,v in zip(bars,rmse): ax.text(v+0.003,bar.get_y()+bar.get_height()/2,f'{v:.4f}',va='center',fontsize=9)
        ax.set_xlabel('RMSE (K)',fontsize=12); ax.set_title('Ablation & Comparison: RMSE Summary',fontsize=13)
        ax.invert_yaxis(); ax.grid(axis='x',alpha=0.3)
        legend=[Patch(facecolor=c,label=l) for c,l in [(color_map['bkg'],'Background'),(color_map['oi'],'OI/1DVar (B2)'),
                (color_map['compare'],'Comparison (B3/B4)'),(color_map['ablation'],'Ablation (V1-V4)'),(color_map['full'],'Ours (Full)')]]
        ax.legend(handles=legend,fontsize=9,loc='lower right')
        plt.tight_layout(); plt.savefig(out_dir/'bar_rmse_comparison.png',dpi=200,bbox_inches='tight'); plt.close()
        print(f"  图表: {out_dir}/bar_rmse_comparison.png")
    except Exception as e: print(f"  (柱图失败: {e})")

    print("\n✓ 汇总评估完成!")


if __name__ == '__main__':
    main()
