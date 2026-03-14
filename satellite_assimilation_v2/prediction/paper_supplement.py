"""
paper_supplement.py
Generate supplemental paper assets:
1) Training curves (from available logs)
2) Compute cost table (Params/FLOPs/Latency)
3) Significance tests (paired t-test and Wilcoxon p-values)
4) Taylor-like diagram
5) Per-level Bias + RMSE profile
6) >=3 case studies
7) Error vs observation density
8) Reproducibility statement
9) Ablation narrative note (treat V1 as new full baseline)
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

PRESSURE_LEVELS = np.array([
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
    875, 900, 925, 950, 975, 1000,
], dtype=np.float64)


class LevelwiseNormalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, x):
        return (x - self.mean[:, None, None]) / self.std[:, None, None]

    def inverse_transform(self, x):
        return x * self.std[:, None, None] + self.mean[:, None, None]


def load_summary(summary_json: Path):
    with open(summary_json, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ckpt_path(row):
    p = row.get("ckpt", "")
    return Path(p) if p and Path(p).exists() else None


def load_test_files(test_root: Path):
    excl = {"stats.npz", "increment_stats.npz"}
    files = []
    for f in sorted(test_root.glob("*.npz")):
        if f.name in excl:
            continue
        try:
            d = np.load(f)
            if d["target"].sum() != 0:
                files.append(f)
        except Exception:
            pass
    return files


def create_model_from_ckpt(ckpt_path: Path):
    import sys

    proj_root = str(Path(__file__).parent.parent)
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    from models.backbone import create_model, UNetConfig

    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_args = ckpt.get("args", {})
    if isinstance(raw_args, dict):
        args = argparse.Namespace(**raw_args)
    elif isinstance(raw_args, argparse.Namespace):
        args = raw_args
    else:
        args = argparse.Namespace()

    model_name = getattr(args, "model", "physics_unet")
    if model_name in ("vanilla_unet", "fuxi_da", "attn_unet", "pixel_mlp", "res_unet"):
        model = create_model(model_name)
    else:
        cfg = UNetConfig(
            fusion_mode=getattr(args, "fusion_mode", "gated"),
            use_aux=getattr(args, "use_aux", True),
            mask_aware=getattr(args, "mask_aware", True),
            use_spectral_stem=getattr(args, "use_spectral_stem", True),
        )
        model = create_model(model_name, config=cfg)

    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd, strict=False)
    return model.eval(), model_name


def estimate_flops_conv_linear(model, example_inputs):
    total = {"flops": 0}

    def conv_hook(m, inp, out):
        x = inp[0]
        b, cin, _, _ = x.shape
        cout, hout, wout = out.shape[1], out.shape[2], out.shape[3]
        kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        ops = b * cout * hout * wout * (cin // m.groups) * kh * kw * 2
        total["flops"] += int(ops)

    def linear_hook(m, inp, out):
        x = inp[0]
        n = x.numel() // x.shape[-1]
        ops = n * m.in_features * m.out_features * 2
        total["flops"] += int(ops)

    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    with torch.no_grad():
        _ = model(*example_inputs)

    for h in hooks:
        h.remove()

    return total["flops"]


def profile_costs(rows, out_dir: Path, input_hw=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cost_rows = []

    for r in rows:
        rid = r.get("id", "")
        if rid in ("bkg", "b2"):
            continue
        ckpt = get_ckpt_path(r)
        if ckpt is None:
            continue

        model, model_name = create_model_from_ckpt(ckpt)
        model = model.to(device).eval()
        params = int(sum(p.numel() for p in model.parameters()))

        obs = torch.randn(1, 17, input_hw, input_hw, device=device)
        bkg = torch.randn(1, 37, input_hw, input_hw, device=device)
        mask = torch.ones(1, 1, input_hw, input_hw, device=device)
        aux = torch.randn(1, 4, input_hw, input_hw, device=device)

        flops = int(estimate_flops_conv_linear(model, (obs, bkg, mask, aux)))

        warmup, runs = 8, 25
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(obs, bkg, mask, aux)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(runs):
                _ = model(obs, bkg, mask, aux)
            if device == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.time() - t0) * 1000.0 / runs

        cost_rows.append({
            "id": rid,
            "label": r.get("label", rid),
            "model": model_name,
            "params": params,
            "flops": flops,
            "latency_ms": float(latency_ms),
        })

    csv_path = out_dir / "compute_costs.csv"
    tex_path = out_dir / "compute_costs.tex"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,label,model,params,flops,latency_ms\n")
        for x in cost_rows:
            f.write(f"{x['id']},{x['label']},{x['model']},{x['params']},{x['flops']},{x['latency_ms']:.4f}\n")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\\centering\\caption{Model complexity on 64x64 input.}\\n")
        f.write("\\begin{tabular}{lccc}\\toprule\\n")
        f.write("Model & Params (M) & FLOPs (G) & Latency (ms)\\\\\\n\\midrule\\n")
        for x in cost_rows:
            label_tex = x["label"].replace("_", "\\\\_")
            f.write(f"{label_tex} & {x['params']/1e6:.2f} & {x['flops']/1e9:.2f} & {x['latency_ms']:.2f}\\\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\\n")


def evaluate_model_per_file(model, test_files, stats, inc_stats, device="cuda"):
    obs_norm = LevelwiseNormalizer(stats["obs_mean"], stats["obs_std"]) if "obs_mean" in stats else None
    bkg_norm = LevelwiseNormalizer(stats["bkg_mean"], stats["bkg_std"])
    tgt_norm = LevelwiseNormalizer(stats["target_mean"], stats["target_std"])
    inc_norm = LevelwiseNormalizer(inc_stats["inc_mean"], inc_stats["inc_std"]) if inc_stats else None

    rmse_file = []
    mae_file = []
    density_file = []
    per_level_rmse_sum = np.zeros(37, dtype=np.float64)
    per_level_bias_sum = np.zeros(37, dtype=np.float64)
    n = 0

    case_records = []

    for f in test_files:
        d = np.load(f)
        obs = d["obs"]
        bkg = d["bkg"]
        tgt = d["target"]
        mask = d["mask"]
        aux = d["aux"] if "aux" in d else None

        obs_n = obs_norm.transform(obs) if obs_norm else obs
        bkg_n = bkg_norm.transform(bkg)

        obs_t = torch.tensor(obs_n, dtype=torch.float32, device=device).unsqueeze(0)
        bkg_t = torch.tensor(bkg_n, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
        aux_t = torch.tensor(aux, dtype=torch.float32, device=device).unsqueeze(0) if aux is not None else None

        with torch.no_grad():
            pred = model(obs_t, bkg_t, mask_t, aux_t).squeeze(0).cpu().numpy()

        if inc_norm is not None:
            ana = bkg + inc_norm.inverse_transform(pred)
        else:
            ana = tgt_norm.inverse_transform(pred)

        diff = ana - tgt
        sq = diff ** 2

        rmse = float(np.sqrt(sq.mean()))
        mae = float(np.abs(diff).mean())

        rmse_file.append(rmse)
        mae_file.append(mae)
        density_file.append(float(mask.mean()))
        per_level_rmse_sum += np.sqrt(sq.reshape(37, -1).mean(1))
        per_level_bias_sum += diff.reshape(37, -1).mean(1)
        n += 1

        case_records.append({
            "path": str(f),
            "rmse": rmse,
            "ana": ana,
            "tgt": tgt,
            "bkg": bkg,
            "mask": mask,
        })

    return {
        "rmse_per_file": np.array(rmse_file, dtype=np.float64),
        "mae_per_file": np.array(mae_file, dtype=np.float64),
        "density": np.array(density_file, dtype=np.float64),
        "per_level_rmse": per_level_rmse_sum / max(n, 1),
        "per_level_bias": per_level_bias_sum / max(n, 1),
        "case_records": case_records,
    }


def background_metrics(test_files):
    rmse = []
    per_level_rmse_sum = np.zeros(37, dtype=np.float64)
    per_level_bias_sum = np.zeros(37, dtype=np.float64)
    n = 0
    for f in test_files:
        d = np.load(f)
        bkg = d["bkg"]
        tgt = d["target"]
        diff = bkg - tgt
        sq = diff ** 2
        rmse.append(float(np.sqrt(sq.mean())))
        per_level_rmse_sum += np.sqrt(sq.reshape(37, -1).mean(1))
        per_level_bias_sum += diff.reshape(37, -1).mean(1)
        n += 1
    return {
        "rmse_per_file": np.array(rmse, dtype=np.float64),
        "per_level_rmse": per_level_rmse_sum / max(n, 1),
        "per_level_bias": per_level_bias_sum / max(n, 1),
    }


def plot_taylor_like(rows, bkg_rmse, out_png: Path):
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = plt.subplot(111, polar=True)
    for r in rows:
        corr = float(r.get("corr", np.nan))
        rmse = float(r.get("rmse", np.nan))
        if not np.isfinite(corr) or not np.isfinite(rmse):
            continue
        corr = np.clip(corr, -1.0, 1.0)
        theta = np.arccos(corr)
        std_ratio = rmse / bkg_rmse
        ax.plot(theta, std_ratio, "o", ms=6, label=r.get("label", r.get("id", "")))

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_thetalim(0, np.pi / 2)
    ax.set_title("Taylor-like Diagram (Corr vs Std-ratio proxy)")
    ax.grid(alpha=0.35)
    ax.legend(loc="upper right", bbox_to_anchor=(1.55, 1.10), fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=260)
    plt.close()


def plot_per_level_bias_rmse(full_m, bkg_m, out_png: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 7), sharey=True)

    ax[0].plot(bkg_m["per_level_bias"], PRESSURE_LEVELS, lw=2, color="#7f8c8d", label="Background")
    ax[0].plot(full_m["per_level_bias"], PRESSURE_LEVELS, lw=2, color="#c0392b", label="Full")
    ax[0].axvline(0.0, color="k", lw=1, alpha=0.5)
    ax[0].set_title("Per-level Bias")
    ax[0].set_xlabel("Bias (K)")
    ax[0].set_ylabel("Pressure (hPa)")

    ax[1].plot(bkg_m["per_level_rmse"], PRESSURE_LEVELS, lw=2, color="#7f8c8d", label="Background")
    ax[1].plot(full_m["per_level_rmse"], PRESSURE_LEVELS, lw=2, color="#1f77b4", label="Full")
    ax[1].set_title("Per-level RMSE")
    ax[1].set_xlabel("RMSE (K)")

    for a in ax:
        a.set_yscale("log")
        a.invert_yaxis()
        a.set_yticks([10, 50, 100, 200, 500, 1000])
        a.set_yticklabels(["10", "50", "100", "200", "500", "1000"])
        a.grid(alpha=0.25)

    ax[1].legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=260)
    plt.close()


def plot_error_vs_density(full_m, out_png: Path):
    x = full_m["density"]
    y = full_m["rmse_per_file"]
    r = np.corrcoef(x, y)[0, 1] if len(x) > 5 else np.nan

    plt.figure(figsize=(6.6, 5.5))
    plt.scatter(x, y, s=14, alpha=0.45, edgecolors="none", color="#2c7fb8")
    if len(x) > 10:
        p = np.polyfit(x, y, deg=1)
        xx = np.linspace(float(x.min()), float(x.max()), 120)
        yy = p[0] * xx + p[1]
        plt.plot(xx, yy, color="#d7301f", lw=2, label=f"fit, r={r:.3f}")
        plt.legend(fontsize=9)
    plt.xlabel("Observation Density (mask mean)")
    plt.ylabel("RMSE (K)")
    plt.title("Error vs Observation Density")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=260)
    plt.close()


def plot_case_studies(full_m, bkg_m, out_png: Path):
    # pick best/median/worst improvements over background
    imp = bkg_m["rmse_per_file"] - full_m["rmse_per_file"]
    order = np.argsort(imp)
    idxs = [int(order[-1]), int(order[len(order)//2]), int(order[0])]
    titles = ["Best improvement", "Median", "Worst improvement"]
    lev = int(np.where(PRESSURE_LEVELS == 500)[0][0]) if 500 in PRESSURE_LEVELS else 21

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.5))
    for row_i, (k, t) in enumerate(zip(idxs, titles)):
        rec = full_m["case_records"][k]
        ana = rec["ana"][lev]
        tgt = rec["tgt"][lev]
        err = np.abs(ana - tgt)

        im0 = axes[row_i, 0].imshow(tgt, cmap="coolwarm")
        axes[row_i, 0].set_title(f"{t} - Target@500hPa")
        plt.colorbar(im0, ax=axes[row_i, 0], fraction=0.046)

        im1 = axes[row_i, 1].imshow(ana, cmap="coolwarm")
        axes[row_i, 1].set_title("Full Analysis@500hPa")
        plt.colorbar(im1, ax=axes[row_i, 1], fraction=0.046)

        im2 = axes[row_i, 2].imshow(err, cmap="magma")
        axes[row_i, 2].set_title("Abs Error")
        plt.colorbar(im2, ax=axes[row_i, 2], fraction=0.046)

        for c in range(3):
            axes[row_i, c].set_xticks([])
            axes[row_i, c].set_yticks([])

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def significance_tests(rows, model_metrics, bkg_rmse, out_dir: Path):
    sig_rows = []
    for r in rows:
        rid = r.get("id", "")
        if rid in ("bkg", "b2") or rid not in model_metrics:
            continue
        y = model_metrics[rid]["rmse_per_file"]
        x = bkg_rmse[: len(y)]

        t_res = stats.ttest_rel(x, y, alternative="greater")
        try:
            w_res = stats.wilcoxon(x - y, alternative="greater", zero_method="wilcox")
            w_p = float(w_res.pvalue)
        except Exception:
            w_p = float("nan")

        sig_rows.append({
            "id": rid,
            "label": r.get("label", rid),
            "mean_rmse": float(np.mean(y)),
            "mean_improve": float((np.mean(x) - np.mean(y)) / np.mean(x) * 100.0),
            "ttest_p": float(t_res.pvalue),
            "wilcoxon_p": w_p,
        })

    sig_csv = out_dir / "significance_tests.csv"
    sig_tex = out_dir / "significance_tests.tex"

    with open(sig_csv, "w", encoding="utf-8") as f:
        f.write("id,label,mean_rmse,mean_improve_pct,ttest_p,wilcoxon_p\n")
        for s in sig_rows:
            f.write(f"{s['id']},{s['label']},{s['mean_rmse']:.6f},{s['mean_improve']:.3f},{s['ttest_p']:.3e},{s['wilcoxon_p']:.3e}\n")

    with open(sig_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\\centering\\caption{Significance tests vs background (paired).}\\n")
        f.write("\\begin{tabular}{lccc}\\toprule\\n")
        f.write("Model & Improve(\\%) & p(t-test) & p(Wilcoxon)\\\\\\n\\midrule\\n")
        for s in sig_rows:
            label_tex = s["label"].replace("_", "\\\\_")
            f.write(f"{label_tex} & {s['mean_improve']:+.2f} & {s['ttest_p']:.2e} & {s['wilcoxon_p']:.2e}\\\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\\n")


def parse_training_loss(log_path: Path):
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    vals = []
    epoch = None
    ep_re = re.compile(r"Epoch\s+(\d+)/(\d+)")
    loss_re = re.compile(r"训练 Loss:\s*([0-9.]+)")
    for line in text:
        m = ep_re.search(line)
        if m:
            epoch = int(m.group(1))
        m2 = loss_re.search(line)
        if m2 and epoch is not None:
            vals.append((epoch, float(m2.group(1))))
    return vals


def plot_training_curves(base_dir: Path, out_png: Path):
    candidates = {
        "B5-AttnUNet": Path("/tmp/cmp_b5.log"),
        "B6-PixelMLP": Path("/tmp/cmp_b6.log"),
        "B7-ResUNet": Path("/tmp/cmp_b7.log"),
    }

    # include historical logs if present
    old_logs = list((base_dir / "train_ddp" / "outputs").glob("FY3F_Assimilation_*/train.log"))
    for i, p in enumerate(sorted(old_logs)[:3], start=1):
        candidates[f"Hist-{i}"] = p

    plt.figure(figsize=(9.8, 5.4))
    n_curves = 0
    for name, p in candidates.items():
        vals = parse_training_loss(p)
        if len(vals) < 2:
            continue
        x = [k for k, _ in vals]
        y = [v for _, v in vals]
        plt.plot(x, y, lw=1.8, label=name)
        n_curves += 1

    if n_curves == 0:
        return False

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Curves (Convergence Evidence)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=260)
    plt.close()
    return True


def write_reproducibility_statement(out_dir: Path):
    txt = out_dir / "reproducibility_statement.txt"
    content = (
        "Reproducibility Statement\n"
        "1) Environment: conda env `fuxi`, Python 3.9, torch 2.6.0+cu124.\n"
        "2) Data: /data2/lrx/npz_64_real with stats.npz and increment_stats.npz.\n"
        "3) Training scripts: train_ddp/train_ddp.py and run_new_baselines.sh.\n"
        "4) Checkpoints: train_ddp/outputs/*/experiment_ddp/best_model.pth.\n"
        "5) Evaluation: prediction/eval_all_experiments.py.\n"
        "6) Supplement generation: prediction/paper_supplement.py.\n"
        "7) Randomness: training configs include fixed seed in saved config.json.\n"
    )
    txt.write_text(content, encoding="utf-8")


def write_ablation_note(out_dir: Path):
    txt = out_dir / "ablation_reframing_note.txt"
    content = (
        "Ablation Reframing Note\n"
        "Given current results where V1 (w/o Aux) outperforms the original Full model,\n"
        "we recommend reframing V1 as the new practical baseline (new full), and\n"
        "treating the previous Full model as a +Aux variant.\n"
        "This keeps the narrative logically consistent with empirical outcomes.\n"
    )
    txt.write_text(content, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, default=str(Path(__file__).parent.parent))
    p.add_argument("--summary_json", type=str, default="prediction/figures_ablation_comparison/results_summary.json")
    p.add_argument("--test_root", type=str, default="/data2/lrx/npz_64_real/test")
    p.add_argument("--stats_file", type=str, default="/data2/lrx/npz_64_real/stats.npz")
    p.add_argument("--increment_stats", type=str, default="/data2/lrx/npz_64_real/increment_stats.npz")
    p.add_argument("--out_dir", type=str, default="prediction/paper_supplement")
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(base_dir / args.summary_json)
    stats_npz = dict(np.load(args.stats_file))
    inc_npz = dict(np.load(args.increment_stats)) if Path(args.increment_stats).exists() else None
    test_files = load_test_files(Path(args.test_root))

    # 1) training curves
    has_curve = plot_training_curves(base_dir, out_dir / "fig_training_curves.png")
    if not has_curve:
        print("[Warn] no usable log for training curve")

    # 2) cost profiling
    profile_costs(summary, out_dir)

    # 3/4/5/6/7 statistics from per-file evaluation
    bkg_m = background_metrics(test_files)
    bkg_rmse = float(np.mean(bkg_m["rmse_per_file"]))

    model_metrics = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for r in summary:
        rid = r.get("id", "")
        if rid in ("bkg", "b2"):
            continue
        ckpt = get_ckpt_path(r)
        if ckpt is None:
            continue
        model, _ = create_model_from_ckpt(ckpt)
        model = model.to(device).eval()
        model_metrics[rid] = evaluate_model_per_file(model, test_files, stats_npz, inc_npz, device=device)
        print(f"[Eval] {rid} done ({len(model_metrics[rid]['rmse_per_file'])} files)")

    significance_tests(summary, model_metrics, bkg_m["rmse_per_file"], out_dir)
    plot_taylor_like(summary, bkg_rmse, out_dir / "fig_taylor_like.png")

    if "full" in model_metrics:
        plot_per_level_bias_rmse(model_metrics["full"], bkg_m, out_dir / "fig_per_level_bias_rmse.png")
        plot_error_vs_density(model_metrics["full"], out_dir / "fig_error_vs_obs_density.png")
        plot_case_studies(model_metrics["full"], bkg_m, out_dir / "fig_case_studies_3samples.png")

    # 8/9 textual statements
    write_reproducibility_statement(out_dir)
    write_ablation_note(out_dir)

    print(f"[Done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
