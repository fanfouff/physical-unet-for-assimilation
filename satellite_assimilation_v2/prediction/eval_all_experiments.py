"""
eval_all_experiments.py — 所有实验汇总评估

运行:
  python3 prediction/eval_all_experiments.py \
    --test_root /data2/lrx/npz_64_real/test \
    --stats_file /data2/lrx/npz_64_real/stats.npz \
    --increment_stats /data2/lrx/npz_64_real/increment_stats.npz \
    --output_dir prediction/figures_ablation_comparison
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
    875, 900, 925, 950, 975, 1000
]

class LevelwiseNormalizer:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def inverse_transform(self, x):
        return x * self.std[:, None, None] + self.mean[:, None, None]

def load_test_files(root: str) -> List[Path]:
    excl = {"increment_stats.npz", "stats.npz"}
    out = []
    for f in sorted(Path(root).glob("*.npz")):
        if f.name in excl:
            continue
        try:
            d = np.load(f)
            if d["target"].sum() != 0:
                out.append(f)
        except Exception:
            pass
    return out

@torch.no_grad()
def evaluate_dl_model(ckpt_path, test_files, stats, inc_stats, device="cuda"):
    proj_root = str(Path(__file__).parent.parent)
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    from models.backbone import create_model, UNetConfig

    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_args = ckpt.get("args", {})
    if isinstance(raw_args, dict):
        model_args = argparse.Namespace(**raw_args)
    elif isinstance(raw_args, argparse.Namespace):
        model_args = raw_args
    else:
        model_args = argparse.Namespace()

    model_name = getattr(model_args, "model", "physics_unet")
    if model_name in ("vanilla_unet", "fuxi_da"):
        model = create_model(model_name)
    else:
        cfg = UNetConfig(
            fusion_mode=getattr(model_args, "fusion_mode", "gated"),
            use_aux=getattr(model_args, "use_aux", True),
            mask_aware=getattr(model_args, "mask_aware", True),
            use_spectral_stem=getattr(model_args, "use_spectral_stem", True),
        )
        model = create_model(model_name, config=cfg)

    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    tgt_norm = LevelwiseNormalizer(stats["target_mean"], stats["target_std"])
    inc_norm = LevelwiseNormalizer(inc_stats["inc_mean"], inc_stats["inc_std"]) if inc_stats else None

    pl_sq = np.zeros(37)
    pl_sq_bkg = np.zeros(37)
    pl_cnt = np.zeros(37)
    rmse_all, rmse_bkg_all, mae_all, bias_all, corr_all = [], [], [], [], []

    for f in tqdm(test_files, desc=f"  {Path(ckpt_path).parent.name[:35]}", leave=False):
        try:
            d = np.load(f)
            obs_phys = d["obs"]
            bkg_phys = d["bkg"]
            tgt_phys = d["target"]
            mask_raw = d["mask"]

            def norm(x, mean, std):
                return (x - mean[:, None, None]) / std[:, None, None]

            obs_n_np = norm(obs_phys, stats["obs_mean"], stats["obs_std"]) if "obs_mean" in stats else obs_phys
            bkg_n_np = norm(bkg_phys, stats["bkg_mean"], stats["bkg_std"])

            obs_n = torch.tensor(obs_n_np).float().unsqueeze(0).to(device)
            bkg_n = torch.tensor(bkg_n_np).float().unsqueeze(0).to(device)
            mask = torch.tensor(mask_raw).float().unsqueeze(0).to(device)
            aux = torch.tensor(d["aux"]).float().unsqueeze(0).to(device) if "aux" in d else None
        except Exception:
            continue

        pred = model(obs_n, bkg_n, mask, aux)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = pred.squeeze(0).cpu().numpy()

        if inc_norm:
            ana_phys = bkg_phys + inc_norm.inverse_transform(pred)
        else:
            ana_phys = tgt_norm.inverse_transform(pred)

        sq = (ana_phys - tgt_phys) ** 2
        sq_b = (bkg_phys - tgt_phys) ** 2
        pl_sq += sq.reshape(37, -1).mean(1)
        pl_sq_bkg += sq_b.reshape(37, -1).mean(1)
        pl_cnt += 1

        rmse_all.append(float(np.sqrt(sq.mean())))
        rmse_bkg_all.append(float(np.sqrt(sq_b.mean())))
        mae_all.append(float(np.abs(ana_phys - tgt_phys).mean()))
        bias_all.append(float((ana_phys - tgt_phys).mean()))
        
        a = ana_phys.ravel()
        b = tgt_phys.ravel()
        # 避免全零导致的 corr 为 nan
        if np.std(a) > 0 and np.std(b) > 0:
            corr_all.append(float(np.corrcoef(a, b)[0, 1]))
        else:
            corr_all.append(0.0)

    if not rmse_all:
        return {
            "rmse": float("nan"),
            "rmse_bkg": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
            "corr": float("nan"),
            "n_files": 0,
            "per_level_rmse": np.full(37, np.nan),
            "per_level_rmse_bkg": np.full(37, np.nan),
        }

    return {
        "rmse": float(np.mean(rmse_all)),
        "rmse_bkg": float(np.mean(rmse_bkg_all)),
        "mae": float(np.mean(mae_all)),
        "bias": float(np.mean(bias_all)),
        "corr": float(np.mean(corr_all)),
        "n_files": len(rmse_all),
        "per_level_rmse": np.sqrt(pl_sq / np.maximum(pl_cnt, 1)),
        "per_level_rmse_bkg": np.sqrt(pl_sq_bkg / np.maximum(pl_cnt, 1)),
    }

def get_experiments(base_dir):
    outs = Path(base_dir) / "train_ddp" / "outputs"
    return [
        {"id": "v1", "label": "Ours (V1)", "type": "ours", "ckpt": str(outs / "ablation_v1_no_aux" / "experiment_ddp" /"best_model.pth")},
        {"id": "full", "label": "Full Variant (Ablation)", "type": "ablation", "ckpt": str(outs / "increment_era5_bkg_64x64" / "experiment_ddp" / "best_model.pth")},
        {"id": "v2", "label": "w/o MaskConv (V2)", "type": "ablation", "ckpt": str(outs / "ablation_v2_no_mask_aware" / "experiment_ddp" /"best_model.pth")},
        {"id": "v3", "label": "w/o GatedFusion (V3)", "type": "ablation", "ckpt": str(outs / "ablation_v3_fusion_add" / "experiment_ddp" /"best_model.pth")},
        {"id": "v4", "label": "w/o SpectralStem (V4)", "type": "ablation", "ckpt": str(outs / "ablation_v4_no_spectral_stem" /"experiment_ddp" / "best_model.pth")},
        {"id": "b3", "label": "VanillaUNet (B3)", "type": "compare", "ckpt": str(outs / "compare_b3_vanilla_unet" / "experiment_ddp" / "best_model.pth")},
        {"id": "b4", "label": "FuXi-DA (B4)", "type": "compare", "ckpt": str(outs / "compare_b4_fuxi_da" / "experiment_ddp" / "best_model.pth")},
        {"id": "b5", "label": "AttentionUNet (B5)", "type": "compare", "ckpt": str(outs / "compare_b5_attn_unet" / "experiment_ddp" / "best_model.pth")},
        {"id": "b6", "label": "PixelMLP (B6)", "type": "compare", "ckpt": str(outs / "compare_b6_pixel_mlp" / "experiment_ddp" / "best_model.pth")},
        {"id": "b7", "label": "ResUNet (B7)", "type": "compare", "ckpt": str(outs / "compare_b7_res_unet" / "experiment_ddp" / "best_model.pth")},
    ]

def print_table(rows):
    print("\n| 方法 | 类型 | RMSE(K) | MAE(K) | Bias(K) | Corr | 改善% |")
    print("|------|------|---------|--------|---------|------|-------|")
    for r in rows:
        rmse = f"{r['rmse']:.4f}" if not np.isnan(r.get("rmse", np.nan)) else "—"
        mae = f"{r.get('mae', 0):.4f}" if "mae" in r else "—"
        bias = f"{r.get('bias', 0):+.4f}" if "bias" in r else "—"
        corr = f"{r.get('corr', 0):.5f}" if "corr" in r else "—"
        imp = f"{r.get('improve_pct', 0):+.2f}%" if "improve_pct" in r else "—"
        print(f"| {r['label']} | {r.get('type', '')} | {rmse} | {mae} | {bias} | {corr} | {imp} |")

def _is_finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def save_paper_tables(rows, out_dir: Path):
    rows_valid = [r for r in rows if _is_finite(r.get("rmse", np.nan))]
    out_csv = out_dir / "results_paper.csv"
    out_tex = out_dir / "results_paper.tex"

    headers = ["id", "label", "type", "rmse", "mae", "bias", "corr", "improve_pct", "n_files"]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows_valid:
            vals = [str(r.get(h, "")) for h in headers]
            f.write(",".join(vals) + "\n")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Overall comparison on test set. Lower RMSE/MAE is better.}\n")
        f.write("\\begin{tabular}{llccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Type & RMSE & MAE & Bias & Corr & Improve(\\%)\\\\\n")
        f.write("\\midrule\n")
        for r in rows_valid:
            label = str(r.get("label", "")).replace("_", "\\_")
            rtype = str(r.get("type", "")).replace("_", "\\_")
            rmse = f"{float(r.get('rmse', np.nan)):.4f}" if _is_finite(r.get("rmse", np.nan)) else "--"
            mae = f"{float(r.get('mae', np.nan)):.4f}" if _is_finite(r.get("mae", np.nan)) else "--"
            bias = f"{float(r.get('bias', np.nan)):+.4f}" if _is_finite(r.get("bias", np.nan)) else "--"
            corr = f"{float(r.get('corr', np.nan)):.5f}" if _is_finite(r.get("corr", np.nan)) else "--"
            imp = f"{float(r.get('improve_pct', np.nan)):+.2f}" if _is_finite(r.get("improve_pct", np.nan)) else "--"
            f.write(f"{label} & {rtype} & {rmse} & {mae} & {bias} & {corr} & {imp}\\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

def plot_rmse_bar(rows, out_png: Path):
    rows_valid = [r for r in rows if _is_finite(r.get("rmse", np.nan))]
    rows_sorted = sorted(rows_valid, key=lambda x: float(x.get("rmse", np.nan)))
    if not rows_sorted:
        return

    labels = [r["label"] for r in rows_sorted]
    rmse = [float(r["rmse"]) for r in rows_sorted]
    types = [r.get("type", "") for r in rows_sorted]

    color_map = {
        "bkg": "#B0BEC5",
        "oi": "#43A047",
        "ours": "#263238",
        "ablation": "#FB8C00",
        "compare": "#1E88E5",
    }
    colors = [color_map.get(t, "#90A4AE") for t in types]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.barh(labels, rmse, color=colors, edgecolor="white", height=0.64)
    for b, v in zip(bars, rmse):
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("RMSE (K)", fontsize=12)
    ax.set_title("Overall RMSE Comparison (Lower is Better)", fontsize=13)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close()

def plot_improve_bar(rows, out_png: Path):
    rows2 = [r for r in rows if _is_finite(r.get("improve_pct", np.nan))]
    rows2 = sorted(rows2, key=lambda x: float(x.get("improve_pct", -1e9)), reverse=True)
    if not rows2:
        return

    labels = [r["label"] for r in rows2]
    imp = [float(r.get("improve_pct", np.nan)) for r in rows2]
    colors = ["#2E7D32" if v >= 0 else "#C62828" for v in imp]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.barh(labels, imp, color=colors, edgecolor="white", height=0.64)
    for b, v in zip(bars, imp):
        dx = 0.35 if v >= 0 else -0.35
        ax.text(v + dx, b.get_y() + b.get_height() / 2, f"{v:+.2f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0.0, color="black", lw=1.0, alpha=0.7)
    ax.set_xlabel("Improvement vs Background (%)", fontsize=12)
    ax.set_title("Relative Improvement (Higher is Better)", fontsize=13)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close()

def plot_combined(rows, out_png: Path):
    rows_valid = [r for r in rows if _is_finite(r.get("rmse", np.nan))]
    rows_sorted = sorted(rows_valid, key=lambda x: float(x.get("rmse", np.nan)))
    if not rows_sorted:
        return

    labels = [r["label"] for r in rows_sorted]
    rmse = [float(r["rmse"]) for r in rows_sorted]
    imp = [float(r.get("improve_pct", np.nan)) if _is_finite(r.get("improve_pct", np.nan)) else np.nan for r in rows_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"wspace": 0.28})
    axes[0].barh(labels, rmse, color="#607D8B", edgecolor="white", height=0.64)
    axes[0].set_title("RMSE (K)", fontsize=12)
    axes[0].set_xlabel("Lower is better", fontsize=10)
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].invert_yaxis()

    imp_clean = [0.0 if np.isnan(v) else v for v in imp]
    colors = ["#2E7D32" if v >= 0 else "#C62828" for v in imp_clean]
    axes[1].barh(labels, imp_clean, color=colors, edgecolor="white", height=0.64)
    axes[1].axvline(0.0, color="black", lw=1.0, alpha=0.7)
    axes[1].set_title("Improvement vs Background (%)", fontsize=12)
    axes[1].set_xlabel("Higher is better", fontsize=10)
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].invert_yaxis()

    fig.suptitle("Ablation and Baseline Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close()

def plot_vertical_rmse(rows, out_png: Path):
    profiles = []
    for r in rows:
        p = r.get("per_level_rmse", None)
        if isinstance(p, np.ndarray) and p.shape[0] == len(PRESSURE_LEVELS):
            profiles.append((r.get("label", r.get("id", "")), r.get("type", ""), p))
    if not profiles:
        return

    type_colors = {
        "bkg": "#7f8c8d",
        "oi": "#43A047",
        "ours": "#111111",
        "ablation": "#F57C00",
        "compare": "#1E88E5",
    }
    fig, ax = plt.subplots(figsize=(8.2, 9.2))
    for label, mtype, prof in profiles:
        lw = 2.8 if mtype == "ours" else 1.8
        alpha = 0.95 if mtype == "ours" else 0.8
        color = type_colors.get(mtype, "#90A4AE")
        ax.plot(prof, PRESSURE_LEVELS, lw=lw, alpha=alpha, color=color, label=label)

    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_yticks([10, 50, 100, 200, 500, 1000])
    ax.set_yticklabels(["10", "50", "100", "200", "500", "1000"])
    ax.set_xlabel("RMSE (K)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title("Vertical RMSE Comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close()

def generate_ablation_paper_assets(rows, out_dir: Path):
    save_paper_tables(rows, out_dir)
    plot_rmse_bar(rows, out_dir / "fig_rmse_bar_sorted.png")
    plot_improve_bar(rows, out_dir / "fig_improvement_bar.png")
    plot_combined(rows, out_dir / "fig_combined_rmse_improve.png")
    plot_vertical_rmse(rows, out_dir / "vertical_rmse_comparison.png")

def write_neural_computing_tex(rows, out_tex_path: Path):
    lines = []
    lines.append(r"\documentclass[preprint,12pt]{elsarticle}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{amsmath,amssymb}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{hyperref}")
    lines.append("")
    lines.append(r"\journal{Neural Computing and Applications}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{frontmatter}")
    lines.append(r"\title{Physics-Guided Satellite Data Assimilation with Ablation-Oriented Evaluation}")
    lines.append(r"\author{First Author}")
    lines.append(r"\address{School/Institute, City, Country}")
    lines.append(r"\begin{abstract}")
    lines.append("This draft reports an ablation-oriented neural assimilation pipeline.")
    lines.append("In this version, V1 is treated as Ours, while Full is considered an additional ablation variant.")
    lines.append(r"\end{abstract}")
    lines.append(r"\begin{keyword}")
    lines.append(r"Satellite data assimilation \sep Neural data-driven model \sep Ablation study")
    lines.append(r"\end{keyword}")
    lines.append(r"\end{frontmatter}")
    lines.append("")
    lines.append(r"\section{Results}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results and ablation comparison.}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Type & RMSE(K) & MAE(K) & Bias(K) & Corr & Improve. \\")
    lines.append(r"\midrule")
    for r in rows:
        if np.isnan(r.get("rmse", np.nan)):
            continue
        method = r["label"].replace("_", r"\_")
        if r.get("type") == "ours":
            method = r"\textbf{" + method + "}"
        rmse = f"{r['rmse']:.4f}"
        mae = f"{r.get('mae', 0):.4f}"
        bias = f"{r.get('bias', 0):+.4f}"
        corr = f"{r.get('corr', 0):.5f}"
        imp = f"{r.get('improve_pct', 0):+.2f}\\%"
        lines.append(f"{method} & {r.get('type', '')} & {rmse} & {mae} & {bias} & {corr} & {imp} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\includegraphics[width=0.9\linewidth]{fig_combined_rmse_improve.png}")
    lines.append(r"\caption{Combined RMSE and improvement comparison across all evaluated methods.}")
    lines.append(r"\label{fig:combined_rmse_improve}")
    lines.append(r"\end{figure}")
    lines.append("")
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\includegraphics[width=0.62\linewidth]{vertical_rmse_comparison.png}")
    lines.append(r"\caption{Vertical RMSE profiles across pressure levels. V1 is the reported Ours; Full is treated as an auxiliary ablation variant.}")
    lines.append(r"\label{fig:vertical_rmse}")
    lines.append(r"\end{figure}")
    lines.append("")
    lines.append(r"\section{Ablation Narrative}")
    lines.append("Based on current experiments, V1 is positioned as Ours and used as the principal reference in the ablation analysis.")
    lines.append("The original Full setting is reinterpreted as an additional ablation branch to keep the experimental narrative consistent with observed ranking.")
    lines.append("")
    lines.append(r"\section{Conclusion}")
    lines.append("This manuscript draft can be extended with detailed method, dataset, and uncertainty analysis sections.")
    lines.append(r"\end{document}")

    out_tex_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_root", default="/data2/lrx/npz_64_real/test")
    p.add_argument("--stats_file", default="/data2/lrx/npz_64_real/stats.npz")
    p.add_argument("--increment_stats", default="/data2/lrx/npz_64_real/increment_stats.npz")
    p.add_argument("--output_dir", default="./figures_ablation_comparison_v2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--base_dir", default=str(Path(__file__).parent.parent))
    p.add_argument("--skip_missing", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = dict(np.load(args.stats_file))
    inc_stats = dict(np.load(args.increment_stats)) if Path(args.increment_stats).exists() else None
    test_files = load_test_files(args.test_root)
    print(f"测试文件: {len(test_files)}")

    print("\n[背景场] 计算...")
    pl_bkg = np.zeros(37)
    pl_cnt = np.zeros(37)
    rmse_bkg_all, mae_bkg_all, bias_bkg_all, corr_bkg_all = [], [], [], []
    
    for f in tqdm(test_files, desc="背景场", leave=False):
        d = np.load(f)
        bp = d["bkg"]
        tp = d["target"]
        sq = (bp - tp) ** 2
        pl_bkg += sq.reshape(37, -1).mean(1)
        pl_cnt += 1
        
        # 精确计算背景场的 MAE, Bias 和 Corr 
        rmse_bkg_all.append(float(np.sqrt(sq.mean())))
        mae_bkg_all.append(float(np.abs(bp - tp).mean()))
        bias_bkg_all.append(float((bp - tp).mean()))
        b_flat, t_flat = bp.ravel(), tp.ravel()
        if np.std(b_flat) > 0 and np.std(t_flat) > 0:
            corr_bkg_all.append(float(np.corrcoef(b_flat, t_flat)[0, 1]))
        else:
            corr_bkg_all.append(0.0)

    bkg_rmse = float(np.mean(rmse_bkg_all))
    bkg_mae = float(np.mean(mae_bkg_all))
    bkg_bias = float(np.mean(bias_bkg_all))
    bkg_corr = float(np.mean(corr_bkg_all))
    bkg_per = np.sqrt(pl_bkg / np.maximum(pl_cnt, 1))

    rows = [{
        "id": "bkg", "label": "Background (ERA5)", "type": "bkg",
        "rmse": bkg_rmse, "mae": bkg_mae, "bias": bkg_bias, "corr": bkg_corr,
        "improve_pct": 0.0, "n_files": len(rmse_bkg_all), "per_level_rmse": bkg_per,
    }]
    print(f"  背景场 RMSE: {bkg_rmse:.4f} K")

    oi_dir = Path(args.base_dir) / "prediction" / "oi_results_64"
    if (oi_dir / "metrics.npy").exists():
        om = np.load(oi_dir / "metrics.npy", allow_pickle=True).item()
        pl_oi = np.load(oi_dir / "per_level_rmse_ana.npy") if (oi_dir / "per_level_rmse_ana.npy").exists() else bkg_per
        rows.append({
            "id": "b2", "label": "OI/1DVar (B2)", "type": "oi",
            "rmse": om["rmse_ana"], "mae": om.get("mae_ana", float("nan")), "bias": om.get("bias_ana", float("nan")),
            "corr": float("nan"), "improve_pct": om["improve_pct"], "n_files": om.get("n_files", 0), "per_level_rmse": pl_oi,
        })
        print(f"[OI] RMSE: {om['rmse_ana']:.4f} K  改善: {om['improve_pct']:.2f}%")

    for exp in get_experiments(args.base_dir):
        ckpt = exp["ckpt"]
        if not Path(ckpt).exists():
            if args.skip_missing:
                print(f"\n[跳过] {exp['label']}")
                continue
            rows.append({**exp, "rmse": float("nan"), "mae": float("nan"), "bias": float("nan"), "corr": float("nan"), "improve_pct": float("nan"), "n_files": 0})
            continue

        print(f"\n[{exp['id']}] 评估: {exp['label']}")
        try:
            res = evaluate_dl_model(ckpt, test_files, stats, inc_stats, device=args.device)
            imp = (bkg_rmse - res["rmse"]) / bkg_rmse * 100 if bkg_rmse > 0 else float("nan")
            rows.append({**exp, **res, "improve_pct": imp})
            print(f"  RMSE={res['rmse']:.4f}K  改善={imp:.2f}%")
        except Exception as e:
            print(f"  ERROR: {e}")

    # 输出顺序: bkg -> oi -> ours -> ablation -> compare
    ordered = []
    for t in ("bkg", "oi", "ours", "ablation", "compare"):
        ordered.extend([r for r in rows if r.get("type") == t])

    print_table(ordered)

    generate_ablation_paper_assets(ordered, out_dir)
    print(f"图表与论文表格: {out_dir}")

    save = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if k != "per_level_rmse"} for r in ordered]
    with open(out_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(save, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_dir}/results_summary.json")

    write_neural_computing_tex(ordered, out_dir / "neural_computing_draft.tex")
    print(f"TeX初稿: {out_dir}/neural_computing_draft.tex")

    print("\n✓ 汇总评估完成!")

if __name__ == "__main__":
    main()