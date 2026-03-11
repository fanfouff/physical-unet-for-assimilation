"""
OI/1DVar 基线 (B2) — 最优插值线性同化基线

原理: Δx_a = K (y - H x_b),  K = B H^T (H B H^T + R)^{-1}
  x_b: 背景场 bkg (37 levels)
  y:   卫星观测 obs BT (17 channels)
  H:   从训练集最小二乘估计 (BT ← T, shape 17×37)
  B:   对角，训练集背景误差方差
  R:   对角，观测误差方差 (残差估计)

运行:
  python3 prediction/oi_baseline.py \
    --train_root /data2/lrx/npz_64_real/train \
    --test_root  /data2/lrx/npz_64_real/test  \
    --stats_file /data2/lrx/npz_64_real/stats.npz \
    --output_dir prediction/oi_results_64
"""
import argparse
import warnings
from pathlib import Path
import numpy as np
from tqdm import tqdm
warnings.filterwarnings("ignore")


def load_files(root, exclude=None):
    exclude = exclude or set()
    files = []
    for f in sorted(Path(root).glob("*.npz")):
        if f.name in exclude:
            continue
        try:
            d = np.load(f)
            if d['target'].sum() != 0:
                files.append(f)
        except Exception:
            pass
    return files


class OIBaseline:
    def __init__(self, n_obs=17, n_lev=37):
        self.n_obs, self.n_lev = n_obs, n_lev
        self.H = self.sigma_b = self.sigma_obs = None
        self.bkg_mean = self.obs_mean = None

    def fit(self, train_files, max_pixels=300_000):
        print(f"[OI] 从 {len(train_files)} 个训练文件估计统计量...")
        obs_list, bkg_list, tgt_list = [], [], []
        for f in tqdm(train_files, desc="采样"):
            try:
                d = np.load(f)
                obs = d['obs']; bkg = d['bkg']; tgt = d['target']; mask = d['mask']
            except Exception:
                continue
            mask_flat = mask[0].ravel().astype(bool)
            idx = np.where(mask_flat)[0]
            if len(idx) == 0:
                continue
            obs_list.append(obs.reshape(self.n_obs, -1).T[idx])
            bkg_list.append(bkg.reshape(self.n_lev, -1).T[idx])
            tgt_list.append(tgt.reshape(self.n_lev, -1).T[idx])
            if sum(len(x) for x in obs_list) > max_pixels:
                break

        obs_all = np.concatenate(obs_list, 0)
        bkg_all = np.concatenate(bkg_list, 0)
        tgt_all = np.concatenate(tgt_list, 0)
        print(f"[OI] 样本数: {len(obs_all):,}")

        self.bkg_mean = bkg_all.mean(0)
        self.obs_mean = obs_all.mean(0)
        X = bkg_all - self.bkg_mean
        Y = obs_all - self.obs_mean
        lam = 1e-4 * (X**2).mean()
        XtX = X.T @ X / len(X)
        XtY = X.T @ Y / len(X)
        H_T = np.linalg.solve(XtX + lam * np.eye(self.n_lev), XtY)
        self.H = H_T.T  # (17, 37)

        self.sigma_b = (bkg_all - tgt_all).std(0) + 1e-6  # (37,)
        y_pred = (bkg_all - self.bkg_mean) @ self.H.T + self.obs_mean
        self.sigma_obs = (obs_all - y_pred).std(0) + 1e-6  # (17,)
        print(f"[OI] sigma_b: {self.sigma_b.min():.3f}~{self.sigma_b.max():.3f} K  "
              f"sigma_obs: {self.sigma_obs.min():.3f}~{self.sigma_obs.max():.3f} K")
        print("[OI] 训练完成 ✓")

    def predict(self, obs, bkg, mask):
        H, W = bkg.shape[1], bkg.shape[2]
        analysis = bkg.copy()
        mask_flat = mask[0].ravel().astype(bool)
        if not mask_flat.any():
            return analysis
        obs_flat = obs.reshape(self.n_obs, -1).T
        bkg_flat = bkg.reshape(self.n_lev, -1).T
        obs_v = obs_flat[mask_flat]
        bkg_v = bkg_flat[mask_flat]

        Hxb = (bkg_v - self.bkg_mean) @ self.H.T + self.obs_mean
        innov = obs_v - Hxb

        sigma_b2 = self.sigma_b ** 2
        BHt = sigma_b2[:, None] * self.H.T       # (37, 17)
        HBHt = (self.H * sigma_b2[None, :]) @ self.H.T  # (17, 17)
        S = HBHt + np.diag(self.sigma_obs ** 2)
        K = BHt @ np.linalg.inv(S)  # (37, 17)

        ana_v = bkg_v + innov @ K.T
        ana_flat = bkg_flat.copy()
        ana_flat[mask_flat] = ana_v
        return ana_flat.T.reshape(self.n_lev, H, W)


def evaluate(test_files, model, stats=None):
    # 数据已为物理值(K), stats接口保留供兼容
    rmse_ana_all, rmse_bkg_all, mae_all, bias_all = [], [], [], []
    pl_sq_ana = np.zeros(37); pl_sq_bkg = np.zeros(37); pl_cnt = np.zeros(37)

    for f in tqdm(test_files, desc="评估"):
        try:
            d = np.load(f)
            obs, bkg, tgt, mask = d['obs'], d['bkg'], d['target'], d['mask']
        except Exception:
            continue
        ana = model.predict(obs, bkg, mask)

        sq_ana = (ana - tgt)**2
        sq_bkg = (bkg - tgt)**2
        pl_sq_ana += sq_ana.reshape(37, -1).mean(1)
        pl_sq_bkg += sq_bkg.reshape(37, -1).mean(1)
        pl_cnt += 1

        rmse_ana_all.append(float(np.sqrt(sq_ana.mean())))
        rmse_bkg_all.append(float(np.sqrt(sq_bkg.mean())))
        mae_all.append(float(np.abs(ana - tgt).mean()))
        bias_all.append(float((ana - tgt).mean()))

    rmse_ana = np.mean(rmse_ana_all)
    rmse_bkg = np.mean(rmse_bkg_all)
    improve  = (rmse_bkg - rmse_ana) / rmse_bkg * 100

    return {
        'rmse_ana': float(rmse_ana), 'rmse_bkg': float(rmse_bkg),
        'mae_ana': float(np.mean(mae_all)), 'bias_ana': float(np.mean(bias_all)),
        'improve_pct': float(improve), 'n_files': len(rmse_ana_all),
        'per_level_rmse_ana': np.sqrt(pl_sq_ana / np.maximum(pl_cnt, 1)),
        'per_level_rmse_bkg': np.sqrt(pl_sq_bkg / np.maximum(pl_cnt, 1)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_root', required=True)
    p.add_argument('--test_root',  required=True)
    p.add_argument('--stats_file', required=True)
    p.add_argument('--output_dir', default='oi_results')
    p.add_argument('--max_pixels', type=int, default=300_000)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    EXCL = {'increment_stats.npz', 'stats.npz'}

    print("[OI] 扫描文件...")
    train_files = load_files(args.train_root, EXCL)
    test_files  = load_files(args.test_root,  EXCL)
    print(f"  训练: {len(train_files)}  测试: {len(test_files)}")

    stats = dict(np.load(args.stats_file))
    oi = OIBaseline()
    oi.fit(train_files, max_pixels=args.max_pixels)
    res = evaluate(test_files, oi, stats)

    print("\n" + "="*60)
    print("B2 OI/1DVar — 测试集结果")
    print("="*60)
    print(f"  有效样本: {res['n_files']}")
    print(f"  背景 RMSE:  {res['rmse_bkg']:.4f} K")
    print(f"  OI RMSE:    {res['rmse_ana']:.4f} K")
    print(f"  OI MAE:     {res['mae_ana']:.4f} K")
    print(f"  OI Bias:    {res['bias_ana']:+.4f} K")
    print(f"  改善:       {res['improve_pct']:.2f}%")
    print("="*60)

    # 保存
    save_res = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}
    import json
    with open(out / 'metrics.json', 'w') as f:
        json.dump(save_res, f, indent=2)
    np.save(out / 'metrics.npy', res)
    np.save(out / 'per_level_rmse_ana.npy', res['per_level_rmse_ana'])
    np.save(out / 'per_level_rmse_bkg.npy', res['per_level_rmse_bkg'])

    # 绘图
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plevs = [1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,
                 300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000]
        fig, ax = plt.subplots(figsize=(6,8))
        ax.plot(res['per_level_rmse_bkg'], plevs, 'b--o', ms=3, label='Background')
        ax.plot(res['per_level_rmse_ana'], plevs, 'r-s',  ms=3, label='OI Analysis (B2)')
        ax.set_xlabel('RMSE (K)'); ax.set_ylabel('Pressure (hPa)')
        ax.set_title('OI Baseline: Vertical RMSE')
        ax.invert_yaxis(); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / 'oi_vertical_rmse.png', dpi=150, bbox_inches='tight')
        print(f"  图表: {out}/oi_vertical_rmse.png")
    except Exception as e:
        print(f"  (绘图跳过: {e})")

    print(f"\n结果保存至: {out}/")


if __name__ == '__main__':
    main()
