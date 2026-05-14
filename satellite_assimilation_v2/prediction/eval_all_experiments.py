"""
eval_all_experiments.py — 所有实验汇总评估 (完整版)

运行:
  python3 prediction/eval_all_experiments.py \
    --test_root /data2/lrx/npz_64_real/test \
    --stats_file /data2/lrx/npz_64_real/stats.npz \
    --increment_stats /data2/lrx/npz_64_real/increment_stats.npz \
    --output_dir prediction/figures_ablation_comparison

新增功能 (v2):
  1. 单样本五面板可视化 (obs/bkg/pred/target/error)
  2. 空间误差热力图 (指定气压层)
  3. 推理延迟/吞吐量基准测试
  4. 分组统计表 (平流层/对流层/近地面)
  5. 统计显著性检验 (paired bootstrap CI)
  6. 误差分布直方图 (概率密度)
  7. 极端案例分析 (best/worst case)
"""
try:
    import yaml
except ImportError:
    yaml = None

import csv
import argparse
import json
import sys
import warnings
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
    875, 900, 925, 950, 975, 1000
]

# 气压层分组索引
STRAT_IDX = [i for i, p in enumerate(PRESSURE_LEVELS) if p <= 100]     # 1-100 hPa
TROP_IDX  = [i for i, p in enumerate(PRESSURE_LEVELS) if 100 < p <= 500]  # 100-500 hPa
LOW_IDX   = [i for i, p in enumerate(PRESSURE_LEVELS) if p > 500]     # 500-1000 hPa

# 代表性气压层 (用于空间误差图)
REPRESENTATIVE_LEVELS = {
    'stratosphere': (10, 5),    # 10 hPa, index 5
    'upper_trop':   (300, 17),  # 300 hPa, index 17
    'mid_trop':     (500, 21),  # 500 hPa, index 21
    'low_trop':     (850, 30),  # 850 hPa, index 30
}

# 图表开关键名
PLOT_OPTIONS = [
    "tables",
    "rmse_bar",
    "improve_bar",
    "combined",
    "vertical",
    "loss",
    "resources",
    "rmse_vs_params",
    "grouped",
    "significance",
    "sample_panels",
    "spatial_maps",
    "extreme_cases",
    "error_distribution",
    "latency",
    "missing_rate",      # 新增
    "se_channels",       # 新增
    "gap_robustness",    # 新增
    "latex_draft",
]


# =============================================================================
# 标签清理映射（将内部实验 ID 映射为论文级名称）
# =============================================================================
LABEL_MAP = {
    # ---- Ours ----
    "Ours-ours_with_full": "PASNet (Ours)",
    "ours_with_full": "PASNet (Ours)",
    "PASNet (Ours)": "PASNet (Ours)",
    # ---- Ablation ----
    "Ablation-v1_noaux_mse": "(v1) w/o L_vert",
    "Ablation-v2_noaux_no_deep_supervision_64": "(v2) w/o deep sup.",
    "Ablation-v3_fusion_add_no_deepsupervision_64": "(v3) Additive fusion",
    "Ablation-v4_noaux_spectral_stem_64": "(v4) w/o Spectral Stem",
    "Ablation-v5_noaux_no_mask_aware_64": "(v5) w/o mask-aware",
    "Ablation-v6_noaux_no_mask_aware_64": "(v6) w/o mask-aware (v6)",
    # ---- Baselines ----
    "Baseline-compare_b3_vanilla_unet_64": "Vanilla U-Net",
    "Baseline-compare_b4_fuxi_da_64": "FuXi-DA",
    "Baseline-compare_b4_fuxi_da_64_tttf": "FuXi-DA (tuned)",
    "Baseline-compare_b5_attn_unet_64": "Attention U-Net",
    "Baseline-compare_b6_pixel_mlp_64": "Pixel-MLP",
    "Baseline-compare_b7_res_unet_64": "ResU-Net",
    "Baseline-compare_b9_fengwu_64": "FengWu",
    "Baseline-compare_b10_with_mamba": "Mamba (stem)",
    "Baseline-compare_b11_background_only_64": "Background-only (B11)",
    "Baseline-compare_b12_obs_only_64": "Obs-only (B12)",
    "Baseline-compare_b11_background_only_128": "Background-only (B11)",
    "Baseline-compare_b12_obs_only_128": "Obs-only (B12)",
    # ---- References ----
    "Background (ERA5)": "Background (ERA5)",
    "OI/1DVar (B2)": "OI/1DVar",
}

# 表格/文中使用的核心方法子集（用于避免图表过于拥挤）
CORE_METHODS_FOR_VERTICAL = {
    "Background (ERA5)", "OI/1DVar", "PASNet (Ours)",
    "Vanilla U-Net", "FuXi-DA (tuned)", "FengWu",
    "(v4) w/o Spectral Stem", "Mamba (stem)",
    "Background-only (B11)", "Obs-only (B12)",
}

CORE_METHODS_FOR_GAP = {
    "PASNet (Ours)", "Mamba (stem)", "(v3) Additive fusion",
    "FengWu", "FuXi-DA (tuned)", "Attention U-Net",
    "ResU-Net", "Vanilla U-Net", "FuXi-DA",
    "Background-only (B11)", "Obs-only (B12)",
}


def clean_label(raw_label: str) -> str:
    """将内部实验 ID 映射为论文级名称。"""
    if raw_label in LABEL_MAP:
        return LABEL_MAP[raw_label]

    for prefix in ("Ours-", "Ablation-", "Baseline-compare_"):
        if raw_label.startswith(prefix):
            stripped = raw_label[len(prefix):]
            if stripped in LABEL_MAP:
                return LABEL_MAP[stripped]

    cleaned = re.sub(r"_(?:64|128)$", "", raw_label)
    if cleaned in LABEL_MAP:
        return LABEL_MAP[cleaned]
    return raw_label


def _parse_csv_set(value: str) -> Set[str]:
    if not value:
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def _id_query_matches_exp_id(query: str, exp_id: str) -> bool:
    """Allow shorthand id matching, e.g. b11 -> compare_b11_background_only_64."""
    q = str(query or "").strip().lower()
    eid = str(exp_id or "").strip().lower()
    if not q or not eid:
        return False

    if q == eid or eid.startswith(q + "_"):
        return True

    for prefix in ("compare_", "ablation_", "ours_"):
        if eid.startswith(prefix):
            no_prefix = eid[len(prefix):]
            if q == no_prefix or no_prefix.startswith(q + "_"):
                return True

    if re.fullmatch(r"b\d+", q):
        return re.search(rf"(^|[_-]){re.escape(q)}([_-]|$)", eid) is not None

    return False


def resolve_plot_selection(raw_plots: str) -> Set[str]:
    """
    将 --plots 解析为集合。
    支持: all 或逗号分隔的键名。
    """
    picks = _parse_csv_set(raw_plots.lower()) if raw_plots else {"all"}
    if not picks or "all" in picks:
        return set(PLOT_OPTIONS)

    unknown = sorted([p for p in picks if p not in PLOT_OPTIONS])
    if unknown:
        raise ValueError(
            f"未知 plots 选项: {unknown}。可选: {PLOT_OPTIONS} 或 all"
        )
    return picks


def filter_experiments(experiments: list, exp_ids_csv: str, exp_types_csv: str) -> list:
    """按 id/type 过滤实验列表。"""
    id_set = _parse_csv_set(exp_ids_csv)
    id_set_lower = {x.lower() for x in id_set}
    type_set = {x.lower() for x in _parse_csv_set(exp_types_csv)}

    id_filter_on = bool(id_set_lower and "all" not in id_set_lower)
    type_filter_on = bool(type_set and "all" not in type_set)

    out = []
    for exp in experiments:
        if id_filter_on and not any(
            _id_query_matches_exp_id(q, exp.get("id")) for q in id_set_lower
        ):
            continue
        if type_filter_on and str(exp.get("type", "")).lower() not in type_set:
            continue
        out.append(exp)
    return out


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


def infer_test_resolution(test_root: str, test_files: List[Path]) -> Optional[int]:
    """从路径名或样本形状推断测试分辨率(如 64/128)。"""
    m = re.search(r"(?:^|[^0-9])(64|128)(?:[^0-9]|$)", str(test_root))
    if m:
        return int(m.group(1))

    if not test_files:
        return None

    try:
        d = np.load(test_files[0])
        t = d["target"]
        if t.ndim == 3:
            return int(t.shape[-1])
    except Exception:
        return None
    return None


def count_parameters(model) -> float:
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def compute_gflops(model, sample_inputs, device="cuda"):
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        flops_analyzer = FlopCountAnalysis(model, sample_inputs)
        flops_analyzer.unsupported_ops_warnings(False)
        flops_analyzer.uncalled_modules_warnings(False)
        return flops_analyzer.total() / 1e9
    except ImportError:
        try:
            from thop import profile
            macs, _ = profile(model, inputs=sample_inputs, verbose=False)
            return macs * 2 / 1e9
        except ImportError:
            params = sum(p.numel() for p in model.parameters())
            return params * 2 / 1e9
    except Exception:
        return float("nan")


def measure_vram(model, sample_inputs, device="cuda"):
    if not torch.cuda.is_available() or "cpu" in str(device):
        return 0.0, 0.0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats(device)
        _ = model(*sample_inputs)
        mem_inf = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    try:
        train_inputs = []
        for inp in sample_inputs:
            if inp is not None and isinstance(inp, torch.Tensor) and inp.is_floating_point():
                train_inputs.append(inp.detach().clone().requires_grad_(False))
            else:
                train_inputs.append(inp)
        out = model(*train_inputs)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()
        mem_train = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    except Exception:
        mem_train = float("nan")
    finally:
        model.eval()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return mem_inf, mem_train


def _is_finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def save_figure(out_path: Path, dpi: int = 300, bbox_inches: str = "tight"):
    """Save both PNG (raster) and PDF (vector) for each figure."""
    out_path = Path(out_path)
    plt.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches=bbox_inches)


def _resolve_baseline_dir(yaml_cfg: Optional[dict], base_dir: str, test_res: Optional[int], yaml_key: str, default_prefix: str) -> Optional[str]:
    """Resolve a baseline results directory from YAML or resolution-based default."""
    cfg = yaml_cfg or {}
    path = cfg.get(yaml_key)
    if path:
        return str(path)
    if test_res in (64, 128):
        candidate = Path(base_dir) / "prediction" / f"{default_prefix}_{test_res}"
        if candidate.exists():
            return str(candidate)
    return None


def _append_external_baseline(
    rows: list,
    yaml_cfg: Optional[dict],
    baseline_dir: Optional[str],
    baseline_name: str,
    baseline_id: str,
    baseline_type: str,
    bkg_rmse: float,
    bkg_per: np.ndarray,
    n_expected: int,
    allow_better_default: bool = True,
    rmse_key: str = "rmse_ana",
    mae_key: str = "mae_ana",
    bias_key: str = "bias_ana",
) -> None:
    """Load baseline metrics from directory and append to rows if consistent."""
    if baseline_dir is None:
        print(f"[INFO] 未提供 {baseline_name} 目录，且未找到匹配分辨率目录，跳过 {baseline_name}")
        return

    bdir = Path(baseline_dir)
    metrics_file = bdir / "metrics.npy"
    if not metrics_file.exists():
        print(f"[INFO] {baseline_name}目录存在但缺少 metrics.npy: {bdir}")
        return

    m = np.load(metrics_file, allow_pickle=True).item()
    allow_key = f"allow_{baseline_id}_better_than_bkg"
    allow_better_than_bkg = bool((yaml_cfg or {}).get(allow_key, allow_better_default))

    n_files = int(m.get("n_files", -1)) if "n_files" in m else -1
    rmse_bkg_in_file = m.get("rmse_bkg", float("nan"))

    mismatch = False
    if n_files > 0 and n_files != n_expected:
        mismatch = True
        print(f"[WARN] {baseline_name} n_files={n_files} 与当前评估样本数={n_expected} 不一致，跳过 {baseline_name}")
    if _is_finite(rmse_bkg_in_file) and abs(float(rmse_bkg_in_file) - bkg_rmse) > 1e-3:
        mismatch = True
        print(
            f"[WARN] {baseline_name} 文件内 rmse_bkg 与当前背景RMSE不一致: "
            f"{float(rmse_bkg_in_file):.4f} vs {bkg_rmse:.4f}，跳过 {baseline_name}"
        )
    if mismatch:
        return

    per_level = np.load(bdir / "per_level_rmse_ana.npy") \
        if (bdir / "per_level_rmse_ana.npy").exists() else bkg_per
    rmse_val = float(m[rmse_key])
    imp = (bkg_rmse - rmse_val) / bkg_rmse * 100 if bkg_rmse > 0 else float("nan")

    if (not allow_better_than_bkg) and _is_finite(rmse_val) and _is_finite(bkg_rmse) and rmse_val < bkg_rmse:
        print(f"[WARN] Detected {baseline_name} RMSE better than ERA5 background; skip by current rule.")
        print(f"[WARN] {baseline_name} RMSE={rmse_val:.4f} < BKG RMSE={bkg_rmse:.4f}; set {allow_key}: true to keep it")
        return

    rows.append({
        "id": baseline_id,
        "label": baseline_name,
        "type": baseline_type,
        "rmse": rmse_val,
        "mae": m.get(mae_key, float("nan")),
        "bias": m.get(bias_key, float("nan")),
        "corr": float("nan"),
        "improve_pct": imp,
        "n_files": n_files if n_files > 0 else n_expected,
        "per_level_rmse": per_level,
    })
    print(f"[INFO] {baseline_name} 基线已加载: {bdir}")


# =============================================================================
# Part A: 模型加载与推理 (核心函数)
# =============================================================================

def _load_model(ckpt_path, device="cuda"):
    """加载模型，返回 (model, model_args, use_aux)"""
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

    def _as_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() == "true"
        return bool(v)

    use_aux = _as_bool(getattr(model_args, "use_aux", True))
    mask_aware = _as_bool(getattr(model_args, "mask_aware", True))
    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}

    # 对 b11/b12 这类基线，从 checkpoint 的 stem 输入通道自动推断 aux 通道数。
    stem_w = sd.get("stem.0.conv.weight", None)
    inferred_aux_channels = None
    if stem_w is not None and hasattr(stem_w, "shape") and len(stem_w.shape) == 4:
        in_ch = int(stem_w.shape[1])
        if model_name == "background_only":
            inferred_aux_channels = max(0, in_ch - 37)
        elif model_name == "obs_only":
            inferred_aux_channels = max(0, in_ch - 18)

    if inferred_aux_channels is not None:
        use_aux = inferred_aux_channels > 0

    stem_w = sd.get("stem.0.conv.weight", None)
    expected_total_in_channels = None
    expected_obs_channels = None
    if stem_w is not None and hasattr(stem_w, "shape") and len(stem_w.shape) == 4:
        expected_total_in_channels = int(stem_w.shape[1])
        aux_ch = 4 if use_aux else 0
        mask_ch = 1 if mask_aware else 0
        if model_name == "background_only":
            expected_obs_channels = 0
        elif model_name == "obs_only":
            expected_obs_channels = expected_total_in_channels - aux_ch - mask_ch
        else:
            expected_obs_channels = expected_total_in_channels - 37 - aux_ch - mask_ch
        if expected_obs_channels is not None and expected_obs_channels < 0:
            expected_obs_channels = None

    if model_name == "fuxi_da":
        aux_ch = inferred_aux_channels if inferred_aux_channels is not None else (4 if use_aux else 0)
        model = create_model(model_name, aux_channels=aux_ch)
    elif model_name in ("background_only", "obs_only"):
        aux_ch = inferred_aux_channels if inferred_aux_channels is not None else (4 if use_aux else 0)
        model = create_model(model_name, aux_channels=aux_ch)
    elif model_name in ("vanilla_unet", "fengwu"):
        model = create_model(model_name)
    else:
        cfg = UNetConfig(
            fusion_mode=getattr(model_args, "fusion_mode", "gated"),
            use_aux=use_aux,
            mask_aware=_as_bool(getattr(model_args, "mask_aware", True)),
            use_spectral_stem=_as_bool(getattr(model_args, "use_spectral_stem", True)),
        )
        model = create_model(model_name, config=cfg)

    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    model._codex_model_name = model_name
    model._codex_use_aux = use_aux
    model._codex_mask_aware = mask_aware
    model._codex_expected_total_in_channels = expected_total_in_channels
    model._codex_expected_obs_channels = expected_obs_channels
    return model, model_args, use_aux


def _actual_input_channels(model, obs_n, bkg_n, mask, aux) -> int:
    model_name = getattr(model, "_codex_model_name", "")
    use_aux = bool(getattr(model, "_codex_use_aux", False))
    mask_aware = bool(getattr(model, "_codex_mask_aware", True))
    aux_ch = int(aux.shape[1]) if (use_aux and aux is not None) else 0
    mask_ch = int(mask.shape[1]) if (mask_aware and mask is not None) else 0

    if model_name == "background_only":
        return int(bkg_n.shape[1]) + aux_ch
    if model_name == "obs_only":
        return int(obs_n.shape[1]) + aux_ch + mask_ch
    return int(obs_n.shape[1]) + int(bkg_n.shape[1]) + aux_ch + mask_ch


def _validate_model_dataset_compatibility(model, npz_path, obs_n, bkg_n, mask, aux):
    expected_obs = getattr(model, "_codex_expected_obs_channels", None)
    expected_total = getattr(model, "_codex_expected_total_in_channels", None)
    model_name = getattr(model, "_codex_model_name", "unknown")

    if expected_obs is None and expected_total is None:
        return

    actual_obs = int(obs_n.shape[1])
    actual_total = _actual_input_channels(model, obs_n, bkg_n, mask, aux)

    if expected_obs is not None and actual_obs != expected_obs:
        hint = ""
        if expected_obs == 17 and actual_obs == 13:
            hint = " Likely FY3D 13-channel data is being evaluated with a FY3F 17-channel checkpoint."
        raise RuntimeError(
            "Dataset/checkpoint channel mismatch: "
            f"{Path(npz_path).name} provides obs={actual_obs}, total_input={actual_total}, "
            f"but checkpoint '{model_name}' expects obs={expected_obs}, total_input={expected_total}.{hint}"
        )


def _prepare_inputs(npz_path, stats, use_aux, device="cuda"):
    """从npz文件准备模型输入，返回 (obs_n, bkg_n, mask, aux, bkg_phys, tgt_phys, obs_phys, lat2d, lon2d)"""
    d = np.load(npz_path)
    obs_phys = d["obs"]
    bkg_phys = d["bkg"]
    tgt_phys = d["target"]
    mask_raw = d["mask"]
    lat2d = d.get("lat2d", None)
    lon2d = d.get("lon2d", None)

    def norm(x, mean, std):
        return (x - mean[:, None, None]) / std[:, None, None]

    obs_n_np = norm(obs_phys, stats["obs_mean"], stats["obs_std"]) \
        if "obs_mean" in stats else obs_phys
    bkg_n_np = norm(bkg_phys, stats["bkg_mean"], stats["bkg_std"])

    obs_n = torch.tensor(obs_n_np).float().unsqueeze(0).to(device)
    bkg_n = torch.tensor(bkg_n_np).float().unsqueeze(0).to(device)
    mask = torch.tensor(mask_raw).float().unsqueeze(0).to(device)
    aux = torch.tensor(d["aux"]).float().unsqueeze(0).to(device) \
        if (use_aux and "aux" in d) else None

    return obs_n, bkg_n, mask, aux, bkg_phys, tgt_phys, obs_phys, lat2d, lon2d


def _run_inference(model, obs_n, bkg_n, mask, aux, bkg_phys, stats, inc_stats):
    """运行推理，返回物理空间分析场 ana_phys (37, H, W)"""
    with torch.no_grad():
        pred = model(obs_n, bkg_n, mask, aux)
    if isinstance(pred, tuple):
        pred = pred[0]
    pred = pred.squeeze(0).cpu().numpy()

    tgt_norm = LevelwiseNormalizer(stats["target_mean"], stats["target_std"])
    if inc_stats is not None:
        inc_norm = LevelwiseNormalizer(inc_stats["inc_mean"], inc_stats["inc_std"])
        ana_phys = bkg_phys + inc_norm.inverse_transform(pred)
    else:
        ana_phys = tgt_norm.inverse_transform(pred)
    return ana_phys


# =============================================================================
# Part B: 原有评估函数 (保持不变)
# =============================================================================

@torch.no_grad()
def evaluate_dl_model(ckpt_path, test_files, stats, inc_stats, device="cuda"):
    model, model_args, use_aux = _load_model(ckpt_path, device)

    params_m = count_parameters(model)
    gflops_val = float("nan")
    mem_inf_mb = 0.0
    mem_train_mb = 0.0

    if test_files:
        try:
            inputs = _prepare_inputs(test_files[0], stats, use_aux, device)
            obs_s, bkg_s, mask_s, aux_s = inputs[:4]
            _validate_model_dataset_compatibility(model, test_files[0], obs_s, bkg_s, mask_s, aux_s)
            sample_inputs = (obs_s, bkg_s, mask_s, aux_s)
            gflops_val = compute_gflops(model, sample_inputs, device)
            mem_inf_mb, mem_train_mb = measure_vram(model, sample_inputs, device)
        except Exception as e:
            print(f"    [WARN] 预检查/资源测量失败: {e}")
            return {
                "rmse": float("nan"), "rmse_bkg": float("nan"),
                "mae": float("nan"), "bias": float("nan"), "corr": float("nan"),
                "n_files": 0,
                "per_level_rmse": np.full(37, np.nan),
                "per_level_rmse_bkg": np.full(37, np.nan),
                "params_m": params_m, "gflops_inf": gflops_val,
                "mem_inf_mb": mem_inf_mb, "mem_train_mb": mem_train_mb,
                "per_sample_rmse": [], "per_sample_rmse_bkg": [],
            }

    tgt_norm = LevelwiseNormalizer(stats["target_mean"], stats["target_std"])
    inc_norm = LevelwiseNormalizer(inc_stats["inc_mean"], inc_stats["inc_std"]) if inc_stats else None

    pl_sq = np.zeros(37)
    pl_sq_bkg = np.zeros(37)
    pl_cnt = np.zeros(37)
    rmse_all, rmse_bkg_all, mae_all, bias_all, corr_all = [], [], [], [], []
    # 收集逐样本逐层误差 (用于后续统计检验)
    per_sample_rmse = []
    per_sample_rmse_bkg = []

    for f in tqdm(test_files, desc=f"  {Path(ckpt_path).parent.name[:35]}", leave=False):
        try:
            inputs = _prepare_inputs(f, stats, use_aux, device)
            obs_n, bkg_n, mask_t, aux_t, bkg_phys, tgt_phys = inputs[:6]
            _validate_model_dataset_compatibility(model, f, obs_n, bkg_n, mask_t, aux_t)
        except Exception:
            continue

        ana_phys = _run_inference(model, obs_n, bkg_n, mask_t, aux_t, bkg_phys, stats, inc_stats)

        sq = (ana_phys - tgt_phys) ** 2
        sq_b = (bkg_phys - tgt_phys) ** 2
        pl_sq += sq.reshape(37, -1).mean(1)
        pl_sq_bkg += sq_b.reshape(37, -1).mean(1)
        pl_cnt += 1

        sample_rmse = float(np.sqrt(sq.mean()))
        sample_rmse_bkg = float(np.sqrt(sq_b.mean()))
        rmse_all.append(sample_rmse)
        rmse_bkg_all.append(sample_rmse_bkg)
        per_sample_rmse.append(sample_rmse)
        per_sample_rmse_bkg.append(sample_rmse_bkg)
        mae_all.append(float(np.abs(ana_phys - tgt_phys).mean()))
        bias_all.append(float((ana_phys - tgt_phys).mean()))

        a = ana_phys.ravel()
        b = tgt_phys.ravel()
        if np.std(a) > 0 and np.std(b) > 0:
            corr_all.append(float(np.corrcoef(a, b)[0, 1]))
        else:
            corr_all.append(0.0)

    if not rmse_all:
        return {
            "rmse": float("nan"), "rmse_bkg": float("nan"),
            "mae": float("nan"), "bias": float("nan"), "corr": float("nan"),
            "n_files": 0,
            "per_level_rmse": np.full(37, np.nan),
            "per_level_rmse_bkg": np.full(37, np.nan),
            "params_m": params_m, "gflops_inf": gflops_val,
            "mem_inf_mb": mem_inf_mb, "mem_train_mb": mem_train_mb,
            "per_sample_rmse": [], "per_sample_rmse_bkg": [],
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
        "params_m": params_m,
        "gflops_inf": gflops_val,
        "mem_inf_mb": mem_inf_mb,
        "mem_train_mb": mem_train_mb,
        "per_sample_rmse": per_sample_rmse,
        "per_sample_rmse_bkg": per_sample_rmse_bkg,
    }

# =============================================================================
# Part C: 新增功能集
# =============================================================================

# [新增] 缺测率分布统计
def plot_missing_rate_distribution(test_files: List[Path], out_dir: Path):
    missing_rates = []
    for f in test_files:
        try:
            m = np.load(f)["mask"]
            missing_rates.append(1.0 - m.mean())
        except Exception: pass

    if not missing_rates: return
    mr = np.array(missing_rates)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mr, bins=50, color='#1976D2', edgecolor='white', alpha=0.85)
    ax.set_xlabel("Missing Rate", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Observation Missing Rate Distribution", fontsize=13)
    ax.axvline(mr.mean(), color='red', ls='--', label=f'Mean={mr.mean():.3f}')
    ax.axvline(np.median(mr), color='orange', ls=':', label=f'Median={np.median(mr):.3f}')
    ax.legend()
    plt.tight_layout()
    out_path = out_dir / "fig_missing_rate_dist.png"
    save_figure(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    [OK] 缺测率分布图: {out_path}")


# [新增] SE 通道注意力解释性分析
def analyze_se_channels(model, test_files: List[Path], stats: dict, use_aux: bool, device: str, out_dir: Path):
    all_weights = []
    for f in tqdm(test_files[:min(500, len(test_files))], desc="  收集SE权重", leave=False):
        try:
            inputs = _prepare_inputs(f, stats, use_aux, device)
            obs_n, bkg_n, mask_t, aux_t = inputs[:4]
            with torch.no_grad(): _ = model(obs_n, bkg_n, mask_t, aux_t)

            se_w = getattr(model, 'get_stem_attention', lambda: None)()
            if se_w is None and hasattr(model, 'stem'):
                se_w = getattr(model.stem, 'get_se_attention', lambda: None)()

            if se_w is not None:
                all_weights.append(se_w.squeeze().cpu().numpy())
        except Exception: continue

    if not all_weights:
        print("    [WARN] 未收集到SE权重，可能是模型不支持")
        return

    weights = np.stack(all_weights)
    mean_w = weights.mean(axis=0)
    std_w = weights.std(axis=0)
    ranking = np.argsort(mean_w)[::-1]
    n_ch = len(mean_w)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(n_ch)
    ax.bar(x, mean_w[ranking], yerr=std_w[ranking],
           color='#1976D2', edgecolor='white', capsize=1.5, alpha=0.85)
    ax.set_xlabel("Latent Channel (sorted by mean SE weight)", fontsize=12)
    ax.set_ylabel("SE Attention Weight", fontsize=12)
    ax.set_title("Squeeze-and-Excitation Channel Attention Weights\n"
                 "(Averaged over test set)", fontsize=13)
    ax.set_xticks(x[::4])
    ax.set_xticklabels([f"Ch{ranking[i]+1}" for i in x[::4]], fontsize=8)

    # SE sigmoid 输出的均匀基线应为 0.5，而不是 1/n_ch。
    uniform_baseline = 0.5
    ax.axhline(uniform_baseline, color='red', ls='--', lw=1.2,
               label=f'Uniform baseline ({uniform_baseline})')
    ax.set_ylim(0, min(1.0, mean_w.max() * 1.3))
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    out_path = out_dir / "fig_se_channel_weights.png"
    save_figure(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    np.savez(out_dir / "se_weights_analysis.npz", mean_weights=mean_w, std_weights=std_w, ranking=ranking)
    print(f"    [OK] SE通道分析图: {out_path} (Top-1 权重: {mean_w[ranking[0]]:.4f})")


# [新增] 缺测率 Gap Robustness 曲线
def evaluate_gap_robustness(experiments: list, test_files: List[Path], stats: dict, inc_stats: dict, device: str, out_dir: Path, n_samples: int = 100):
    gap_ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    all_results = {}
    test_sub = test_files[:n_samples]

    for exp in experiments:
        ckpt = exp["ckpt"]
        if not Path(ckpt).exists(): continue
        try:
            model, _, use_aux = _load_model(ckpt, device)
            if test_sub:
                pre_inputs = _prepare_inputs(test_sub[0], stats, use_aux, device)
                _validate_model_dataset_compatibility(model, test_sub[0], *pre_inputs[:4])
            results_per_ratio = {r: [] for r in gap_ratios}

            for f in tqdm(test_sub, desc=f"  Gap Robustness: {exp['label']}", leave=False):
                try:
                    inputs = _prepare_inputs(f, stats, use_aux, device)
                    obs_n, bkg_n, mask_t, aux_t, bkg_phys, tgt_phys = inputs[:6]
                    B, _, H, W = obs_n.shape

                    for ratio in gap_ratios:
                        if ratio == 0.0:
                            mask_use = mask_t
                        else:
                            artificial_gap = (torch.rand(B, 1, H, W, device=device) < ratio)
                            mask_use = mask_t * (~artificial_gap).float()
                        obs_masked = obs_n * mask_use
                        ana = _run_inference(model, obs_masked, bkg_n, mask_use, aux_t, bkg_phys, stats, inc_stats)
                        results_per_ratio[ratio].append(float(np.sqrt(np.mean((ana - tgt_phys)**2))))
                except Exception: continue

            curve = {r: np.mean(v) for r, v in results_per_ratio.items() if v}
            if curve:
                all_results[exp["id"]] = {"label": exp["label"], "curve": curve, "type": exp["type"]}
            del model; torch.cuda.empty_cache()
        except Exception: continue

    if not all_results:
        return

    # ---- 绘图 (publication quality) ----
    fig, ax = plt.subplots(figsize=(8, 5.5))

    type_styles = {
        "ours": {"color": "#D32F2F", "lw": 2.8, "marker": "o", "ms": 7, "zorder": 10},
        "ablation": {"color": "#F57C00", "lw": 1.8, "marker": "s", "ms": 5, "zorder": 6},
        "compare": {"color": "#1E88E5", "lw": 1.5, "marker": "^", "ms": 5, "zorder": 5},
    }

    for eid, res in all_results.items():
        label = clean_label(res["label"])
        if label not in CORE_METHODS_FOR_GAP:
            continue

        curve = res["curve"]
        ratios = sorted(curve.keys())
        rmses = [curve[r] for r in ratios]
        style = type_styles.get(
            res["type"],
            {"color": "#7E57C2", "lw": 1.2, "marker": "D", "ms": 4, "zorder": 4}
        )
        ax.plot(ratios, rmses, ls='-', marker=style["marker"],
                label=label, color=style["color"],
                lw=style["lw"], markersize=style["ms"],
                zorder=style["zorder"], alpha=0.9)

    ax.set_xlabel("Artificial Gap Ratio", fontsize=13)
    ax.set_ylabel("RMSE (K)", fontsize=13)
    ax.set_title("Gap Robustness: RMSE vs. Observation Missing Rate", fontsize=14)
    ax.set_xlim(-0.03, 1.03)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9, ncol=1)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out_path = out_dir / "fig_gap_robustness_comparison.png"
    save_figure(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    with open(out_dir / "gap_robustness_results.json", "w") as f:
        json.dump({eid: {str(k): v for k, v in res["curve"].items()} for eid, res in all_results.items()}, f, indent=2)
    print(f"    [OK] Gap Robustness图: {out_path}")

def plot_single_sample_panels(
    model, test_file: Path, stats: dict, inc_stats: dict,
    use_aux: bool, device: str, out_dir: Path,
    level_idx: int = 21, tag: str = "sample"
):
    """
    五面板可视化: obs(通道0) / bkg / prediction / target / error
    level_idx: 可视化的气压层索引 (默认21 = 500hPa)
    """
    inputs = _prepare_inputs(test_file, stats, use_aux, device)
    obs_n, bkg_n, mask_t, aux_t, bkg_phys, tgt_phys, obs_phys, lat2d, lon2d = inputs

    ana_phys = _run_inference(model, obs_n, bkg_n, mask_t, aux_t, bkg_phys, stats, inc_stats)

    error = ana_phys - tgt_phys
    plev = PRESSURE_LEVELS[level_idx]

    fig, axes = plt.subplots(1, 5, figsize=(26, 5))

    # Panel 1: Observation (channel 0, 用于展示卫星观测覆盖)
    obs_ch0 = obs_phys[0]  # 第1个通道
    mask_2d = np.load(test_file)["mask"].squeeze()
    obs_masked = np.where(mask_2d > 0.5, obs_ch0, np.nan)
    im0 = axes[0].imshow(obs_masked, cmap="RdYlBu_r", aspect="equal")
    axes[0].set_title(f"Observation (Ch1)\nmask coverage: {mask_2d.mean():.1%}", fontsize=11)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: Background
    im1 = axes[1].imshow(bkg_phys[level_idx], cmap="RdYlBu_r", aspect="equal")
    axes[1].set_title(f"Background\n{plev} hPa", fontsize=11)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Prediction (Analysis)
    im2 = axes[2].imshow(ana_phys[level_idx], cmap="RdYlBu_r", aspect="equal")
    axes[2].set_title(f"Analysis (Pred)\n{plev} hPa", fontsize=11)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Target (ERA5 analysis)
    im3 = axes[3].imshow(tgt_phys[level_idx], cmap="RdYlBu_r", aspect="equal")
    axes[3].set_title(f"Target (ERA5)\n{plev} hPa", fontsize=11)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    # Panel 5: Error
    vmax_err = max(abs(np.nanpercentile(error[level_idx], 2)),
                   abs(np.nanpercentile(error[level_idx], 98)))
    im4 = axes[4].imshow(error[level_idx], cmap="bwr", vmin=-vmax_err, vmax=vmax_err, aspect="equal")
    rmse_this = np.sqrt(np.mean(error[level_idx] ** 2))
    axes[4].set_title(f"Error (Pred−Target)\nRMSE={rmse_this:.3f} K", fontsize=11)
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Single Sample Analysis — {Path(test_file).stem}", fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = out_dir / f"fig_5panel_{tag}_{plev}hPa.png"
    save_figure(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    [OK] 五面板图: {out_path}")


# =============================================================================
# Part D: 新增功能 2 — 空间误差热力图 (多层)
# =============================================================================

def plot_spatial_error_maps(
    model, test_files: List[Path], stats: dict, inc_stats: dict,
    use_aux: bool, device: str, out_dir: Path,
    n_samples: int = 50
):
    """
    对多个样本取平均，绘制4个代表性气压层的空间误差分布
    """
    n_use = min(n_samples, len(test_files))
    error_accum = None
    count = 0

    for f in tqdm(test_files[:n_use], desc="  空间误差累积", leave=False):
        try:
            inputs = _prepare_inputs(f, stats, use_aux, device)
            obs_n, bkg_n, mask_t, aux_t, bkg_phys, tgt_phys = inputs[:6]
            ana_phys = _run_inference(model, obs_n, bkg_n, mask_t, aux_t, bkg_phys, stats, inc_stats)
            err = ana_phys - tgt_phys  # (37, 64, 64)
            if error_accum is None:
                error_accum = np.zeros_like(err)
                error_sq_accum = np.zeros_like(err)
            error_accum += err
            error_sq_accum += err ** 2
            count += 1
        except Exception:
            continue

    if count == 0:
        return

    mean_error = error_accum / count  # 平均偏差 (bias map)
    rmse_map = np.sqrt(error_sq_accum / count)  # RMSE map

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for col, (name, (plev, lidx)) in enumerate(REPRESENTATIVE_LEVELS.items()):
        # Row 0: Mean bias
        vmax_b = max(abs(np.percentile(mean_error[lidx], 2)),
                     abs(np.percentile(mean_error[lidx], 98)), 0.01)
        im0 = axes[0, col].imshow(mean_error[lidx], cmap="bwr",
                                   vmin=-vmax_b, vmax=vmax_b, aspect="equal")
        axes[0, col].set_title(f"Mean Bias\n{plev} hPa ({name})", fontsize=10)
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04, format="%.3f")

        # Row 1: RMSE map
        im1 = axes[1, col].imshow(rmse_map[lidx], cmap="YlOrRd", aspect="equal")
        axes[1, col].set_title(f"RMSE Map\n{plev} hPa ({name})", fontsize=10)
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04, format="%.3f")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Spatial Error Distribution (averaged over {count} samples)", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = out_dir / "fig_spatial_error_maps.png"
    save_figure(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    [OK] 空间误差图: {out_path}")


# =============================================================================
# Part E: 新增功能 3 — 推理延迟/吞吐量基准测试
# =============================================================================

def benchmark_latency(
    experiments: list, test_files: List[Path], stats: dict,
    device: str, out_dir: Path,
    n_warmup: int = 10, n_repeat: int = 50
):
    """
    测量每个模型的单样本推理延迟 (ms) 和吞吐量 (samples/s)
    """
    if not test_files:
        return {}

    results = {}
    sample_file = test_files[0]

    for exp in experiments:
        ckpt = exp["ckpt"]
        if not Path(ckpt).exists():
            continue

        try:
            model, _, use_aux = _load_model(ckpt, device)
            inputs = _prepare_inputs(sample_file, stats, use_aux, device)
            obs_n, bkg_n, mask_t, aux_t = inputs[:4]

            model.eval()
            # Warmup
            with torch.no_grad():
                for _ in range(n_warmup):
                    _ = model(obs_n, bkg_n, mask_t, aux_t)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark
            latencies = []
            with torch.no_grad():
                for _ in range(n_repeat):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                    t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

                    if t0 is not None:
                        t0.record()
                        _ = model(obs_n, bkg_n, mask_t, aux_t)
                        t1.record()
                        torch.cuda.synchronize()
                        latencies.append(t0.elapsed_time(t1))  # ms
                    else:
                        import time
                        s = time.perf_counter()
                        _ = model(obs_n, bkg_n, mask_t, aux_t)
                        latencies.append((time.perf_counter() - s) * 1000)

            lat_mean = np.mean(latencies)
            lat_std = np.std(latencies)
            throughput = 1000.0 / lat_mean  # samples/s

            results[exp["id"]] = {
                "label": exp["label"],
                "latency_ms": lat_mean,
                "latency_std_ms": lat_std,
                "throughput_sps": throughput,
            }
            print(f"    {exp['label']}: {lat_mean:.2f} ± {lat_std:.2f} ms, "
                  f"{throughput:.1f} samples/s")

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    [WARN] {exp['label']}: 延迟测量失败 - {e}")

    # 绘图
    if results:
        _plot_latency_chart(results, out_dir)

    return results


def _plot_latency_chart(results: dict, out_dir: Path):
    """绘制延迟/吞吐量对比图"""
    labels = [v["label"] for v in results.values()]
    latencies = [v["latency_ms"] for v in results.values()]
    lat_stds = [v["latency_std_ms"] for v in results.values()]
    throughputs = [v["throughput_sps"] for v in results.values()]

    # 按延迟排序
    order = np.argsort(latencies)
    labels = [labels[i] for i in order]
    latencies = [latencies[i] for i in order]
    lat_stds = [lat_stds[i] for i in order]
    throughputs = [throughputs[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Latency
    bars1 = ax1.barh(labels, latencies, xerr=lat_stds, color="#5C6BC0",
                     edgecolor="white", capsize=3)
    ax1.set_xlabel("Latency (ms)", fontsize=12)
    ax1.set_title("Inference Latency (Lower is Better)", fontsize=13)
    ax1.grid(axis="x", alpha=0.25)
    ax1.invert_yaxis()
    for b, v in zip(bars1, latencies):
        ax1.text(v + 0.5, b.get_y() + b.get_height() / 2,
                 f"{v:.1f}", va="center", fontsize=9)

    # Throughput
    bars2 = ax2.barh(labels, throughputs, color="#26A69A", edgecolor="white")
    ax2.set_xlabel("Throughput (samples/s)", fontsize=12)
    ax2.set_title("Inference Throughput (Higher is Better)", fontsize=13)
    ax2.grid(axis="x", alpha=0.25)
    ax2.invert_yaxis()
    for b, v in zip(bars2, throughputs):
        ax2.text(v + 1, b.get_y() + b.get_height() / 2,
                 f"{v:.0f}", va="center", fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "fig_latency_throughput.png"
    save_figure(out_path, dpi=260, bbox_inches="tight")
    plt.close()
    print(f"    [OK] 延迟/吞吐量图: {out_path}")


# =============================================================================
# Part F: 新增功能 4 — 分组统计表 (平流层/对流层/近地面)
# =============================================================================

def compute_grouped_metrics(rows, out_dir: Path):
    """
    根据 per_level_rmse 计算三层分组统计
    输出 CSV + LaTeX 表格
    """
    header = ["Method", "Type",
              "Strat RMSE", "Trop RMSE", "LowTrop RMSE",
              "Global RMSE"]

    table_rows = []
    for r in rows:
        plr = r.get("per_level_rmse", None)
        if plr is None or not isinstance(plr, np.ndarray) or plr.shape[0] != 37:
            continue
        strat = float(np.sqrt(np.mean(plr[STRAT_IDX] ** 2)))
        trop = float(np.sqrt(np.mean(plr[TROP_IDX] ** 2)))
        low = float(np.sqrt(np.mean(plr[LOW_IDX] ** 2)))
        glob = float(np.sqrt(np.mean(plr ** 2)))
        table_rows.append({
            "Method": r["label"], "Type": r.get("type", ""),
            "Strat RMSE": strat, "Trop RMSE": trop,
            "LowTrop RMSE": low, "Global RMSE": glob,
        })

    if not table_rows:
        return

    # CSV
    csv_path = out_dir / "grouped_rmse_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for tr in table_rows:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in tr.items()})

    # LaTeX
    tex_path = out_dir / "grouped_rmse_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{RMSE by atmospheric layer group (K). "
                r"Strat: 1--100\,hPa; Trop: 100--500\,hPa; Low: 500--1000\,hPa.}" + "\n")
        f.write(r"\begin{tabular}{llcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Method & Type & Stratosphere & Troposphere & Lower Trop. & Global \\" + "\n")
        f.write(r"\midrule" + "\n")
        for tr in table_rows:
            method = tr["Method"].replace("_", r"\_")
            f.write(f"{method} & {tr['Type']} & "
                    f"{tr['Strat RMSE']:.4f} & {tr['Trop RMSE']:.4f} & "
                    f"{tr['LowTrop RMSE']:.4f} & {tr['Global RMSE']:.4f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    print(f"    [OK] 分组统计表: {csv_path}, {tex_path}")

    # 绘制分组柱状图
    _plot_grouped_bar(table_rows, out_dir)


def _plot_grouped_bar(table_rows, out_dir: Path):
    """分组 RMSE 柱状图"""
    labels = [r["Method"] for r in table_rows]
    strat = [r["Strat RMSE"] for r in table_rows]
    trop = [r["Trop RMSE"] for r in table_rows]
    low = [r["LowTrop RMSE"] for r in table_rows]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - width, strat, width, label="Stratosphere (1-100 hPa)", color="#7E57C2")
    b2 = ax.bar(x, trop, width, label="Troposphere (100-500 hPa)", color="#26A69A")
    b3 = ax.bar(x + width, low, width, label="Lower Trop. (500-1000 hPa)", color="#EF5350")

    ax.set_ylabel("RMSE (K)", fontsize=12)
    ax.set_title("RMSE by Atmospheric Layer Group", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    save_figure(out_dir / "fig_grouped_rmse_bar.png", dpi=240, bbox_inches="tight")
    plt.close()
    print(f"    [OK] 分组柱状图: {out_dir / 'fig_grouped_rmse_bar.png'}")


# =============================================================================
# Part G: 新增功能 5 — 统计显著性检验 (Paired Bootstrap CI)
# =============================================================================

def bootstrap_paired_test(
    rmse_ours: np.ndarray, rmse_other: np.ndarray,
    n_bootstrap: int = 10000, ci_level: float = 0.95
) -> Dict:
    """
    Paired bootstrap 差异检验
    H0: mean(rmse_ours) >= mean(rmse_other)  (ours不优于other)
    返回: delta_mean, CI_low, CI_high, p_value
    """
    n = len(rmse_ours)
    assert n == len(rmse_other), "样本数必须相同"

    diff = rmse_ours - rmse_other  # 负值表示ours更好
    observed_delta = np.mean(diff)

    boot_deltas = np.zeros(n_bootstrap)
    rng = np.random.RandomState(42)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_deltas[i] = np.mean(diff[idx])

    alpha = 1 - ci_level
    ci_low = np.percentile(boot_deltas, 100 * alpha / 2)
    ci_high = np.percentile(boot_deltas, 100 * (1 - alpha / 2))

    # p-value: 零假设下差异>=0的概率
    p_value = np.mean(boot_deltas >= 0)

    return {
        "delta_mean": float(observed_delta),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def run_significance_tests(rows, out_dir: Path):
    """
    以 ours 为基准，对所有其他方法做 paired bootstrap test
    """
    # 找 ours
    ours_row = None
    for r in rows:
        if r.get("type") == "ours" and r.get("per_sample_rmse"):
            ours_row = r
            break
    if ours_row is None:
        print("    [SKIP] 未找到 ours 类型的结果，跳过显著性检验")
        return

    ours_rmse = np.array(ours_row["per_sample_rmse"])
    test_results = []

    for r in rows:
        if r.get("id") == ours_row.get("id"):
            continue
        other_rmse = r.get("per_sample_rmse", [])
        if not other_rmse or len(other_rmse) != len(ours_rmse):
            continue

        res = bootstrap_paired_test(ours_rmse, np.array(other_rmse))
        res["method"] = r["label"]
        res["type"] = r.get("type", "")
        test_results.append(res)

        sig_mark = "***" if res["p_value"] < 0.001 else \
                   "**" if res["p_value"] < 0.01 else \
                   "*" if res["p_value"] < 0.05 else "n.s."
        print(f"    {r['label']}: Δ={res['delta_mean']:+.4f} K, "
              f"95% CI=[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}], "
              f"p={res['p_value']:.4f} {sig_mark}")

    if not test_results:
        return

    # 保存
    with open(out_dir / "significance_tests.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # LaTeX 表格
    with open(out_dir / "significance_tests.tex", "w") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Paired bootstrap test: Ours vs. each method. "
                r"$\Delta$ = RMSE(Ours) $-$ RMSE(Other); negative means Ours is better.}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Method & $\Delta$ RMSE (K) & 95\% CI & $p$-value & Sig. \\" + "\n")
        f.write(r"\midrule" + "\n")
        for tr in test_results:
            method = tr["method"].replace("_", r"\_")
            sig = r"$\checkmark$" if tr["significant"] else "---"
            f.write(f"{method} & {tr['delta_mean']:+.4f} & "
                    f"[{tr['ci_low']:+.4f}, {tr['ci_high']:+.4f}] & "
                    f"{tr['p_value']:.4f} & {sig} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    # 可视化: Forest plot
    _plot_forest(test_results, ours_row["label"], out_dir)
    print(f"    [OK] 显著性检验: {out_dir / 'significance_tests.json'}")


def _plot_forest(test_results, ours_label, out_dir: Path):
    """Forest plot: 展示每个方法相对于 Ours 的 RMSE 差异 + CI"""
    labels = [tr["method"] for tr in test_results]
    deltas = [tr["delta_mean"] for tr in test_results]
    ci_lows = [tr["ci_low"] for tr in test_results]
    ci_highs = [tr["ci_high"] for tr in test_results]
    sigs = [tr["significant"] for tr in test_results]

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5 + 1)))

    for i in range(len(labels)):
        color = "#2E7D32" if deltas[i] < 0 else "#C62828"
        marker = "o" if sigs[i] else "s"
        ax.errorbar(deltas[i], y[i],
                    xerr=[[deltas[i] - ci_lows[i]], [ci_highs[i] - deltas[i]]],
                    fmt=marker, color=color, capsize=4, capthick=1.5,
                    markersize=8, linewidth=1.5)

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(f"ΔRMSE: {ours_label} − Other (K)\n← Ours better | Other better →",
                  fontsize=11)
    ax.set_title("Paired Bootstrap Test (95% CI)", fontsize=13)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()

    plt.tight_layout()
    save_figure(out_dir / "fig_forest_plot_significance.png", dpi=260, bbox_inches="tight")
    plt.close()


# =============================================================================
# Part H: 新增功能 6 — 误差分布直方图
# =============================================================================

def plot_error_distribution(
    experiments: list, test_files: List[Path], stats: dict, inc_stats: dict,
    device: str, out_dir: Path, n_samples: int = 100
):
    """
    对每个模型收集 pred-target 误差像素，绘制概率密度直方图
    """
    n_use = min(n_samples, len(test_files))
    all_errors = {}

    for exp in experiments:
        ckpt = exp["ckpt"]
        if not Path(ckpt).exists():
            continue

        try:
            model, _, use_aux = _load_model(ckpt, device)
        except Exception:
            continue

        errors = []
        for f in tqdm(test_files[:n_use], desc=f"  误差分布 {exp['id']}", leave=False):
            try:
                inputs = _prepare_inputs(f, stats, use_aux, device)
                obs_n, bkg_n, mask_t, aux_t, bkg_phys, tgt_phys = inputs[:6]
                ana_phys = _run_inference(model, obs_n, bkg_n, mask_t, aux_t,
                                          bkg_phys, stats, inc_stats)
                err = (ana_phys - tgt_phys).ravel()
                # 随机采样以限制内存
                if len(err) > 10000:
                    err = np.random.choice(err, 10000, replace=False)
                errors.append(err)
            except Exception:
                continue

        if errors:
            all_errors[exp["label"]] = np.concatenate(errors)

        del model
        torch.cuda.empty_cache()

    # 加入背景场误差
    bkg_errors = []
    for f in test_files[:n_use]:
        try:
            d = np.load(f)
            err = (d["bkg"] - d["target"]).ravel()
            if len(err) > 10000:
                err = np.random.choice(err, 10000, replace=False)
            bkg_errors.append(err)
        except Exception:
            continue
    if bkg_errors:
        all_errors["Background"] = np.concatenate(bkg_errors)

    if not all_errors:
        return

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 7))

    type_colors = {
        "Background": "#7f8c8d",
    }
    default_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    ci = 0

    for label, errs in all_errors.items():
        color = type_colors.get(label, default_colors[ci % len(default_colors)])
        if label != "Background":
            ci += 1
        lw = 2.5 if label == "Background" else 1.8
        ls = "--" if label == "Background" else "-"

        # KDE-like histogram
        ax.hist(errs, bins=120, density=True, alpha=0.0, range=(-3, 3))  # 不画
        from scipy.ndimage import uniform_filter1d
        counts, bin_edges = np.histogram(errs, bins=200, density=True, range=(-4, 4))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        counts_smooth = uniform_filter1d(counts.astype(float), size=5)
        ax.plot(bin_centers, counts_smooth, color=color, lw=lw, ls=ls,
                label=f"{label} (σ={np.std(errs):.3f})", alpha=0.85)

    ax.axvline(0, color="black", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Error: Prediction − Target (K)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title("Error Distribution Comparison", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-3.5, 3.5)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out_path = out_dir / "fig_error_distribution.png"
    save_figure(out_path, dpi=260, bbox_inches="tight")
    plt.close()
    print(f"    [OK] 误差分布图: {out_path}")


# =============================================================================
# Part I: 新增功能 7 — 极端案例分析 (Best/Worst)
# =============================================================================

def analyze_extreme_cases(
    ours_exp: dict, test_files: List[Path], stats: dict, inc_stats: dict,
    device: str, out_dir: Path, n_extreme: int = 3
):
    """
    找出 Ours 模型的 best-N 和 worst-N 样本，逐个绘制五面板图
    """
    ckpt = ours_exp["ckpt"]
    if not Path(ckpt).exists():
        print("    [SKIP] Ours ckpt 不存在")
        return

    model, _, use_aux = _load_model(ckpt, device)

    sample_rmse = []
    for f in tqdm(test_files, desc="  极端案例搜索", leave=False):
        try:
            inputs = _prepare_inputs(f, stats, use_aux, device)
            obs_n, bkg_n, mask_t, aux_t, bkg_phys, tgt_phys = inputs[:6]
            ana_phys = _run_inference(model, obs_n, bkg_n, mask_t, aux_t,
                                      bkg_phys, stats, inc_stats)
            rmse = float(np.sqrt(np.mean((ana_phys - tgt_phys) ** 2)))
            sample_rmse.append((rmse, f))
        except Exception:
            continue

    sample_rmse.sort(key=lambda x: x[0])

    # Best N
    print(f"\n    --- Best {n_extreme} cases ---")
    for i, (rmse, f) in enumerate(sample_rmse[:n_extreme]):
        print(f"    #{i+1} RMSE={rmse:.4f} K  {f.name}")
        for lidx in [5, 17, 21, 30]:  # 10, 300, 500, 850 hPa
            plot_single_sample_panels(
                model, f, stats, inc_stats, use_aux, device, out_dir,
                level_idx=lidx, tag=f"best{i+1}"
            )

    # Worst N
    print(f"\n    --- Worst {n_extreme} cases ---")
    for i, (rmse, f) in enumerate(sample_rmse[-n_extreme:]):
        print(f"    #{i+1} RMSE={rmse:.4f} K  {f.name}")
        for lidx in [5, 17, 21, 30]:
            plot_single_sample_panels(
                model, f, stats, inc_stats, use_aux, device, out_dir,
                level_idx=lidx, tag=f"worst{i+1}"
            )

    # 保存排名
    ranking = [{"rank": i+1, "rmse": r, "file": str(f.name)}
               for i, (r, f) in enumerate(sample_rmse)]
    with open(out_dir / "sample_rmse_ranking.json", "w") as fout:
        json.dump(ranking, fout, indent=2)
    print(f"    [OK] 样本排名: {out_dir / 'sample_rmse_ranking.json'}")

    del model
    torch.cuda.empty_cache()


# =============================================================================
# Part J: 原有绘图函数 (保留)
# =============================================================================

def save_paper_tables(rows, out_dir: Path):
    rows_valid = [r for r in rows if _is_finite(r.get("rmse", np.nan))]
    out_csv = out_dir / "results_paper.csv"
    out_tex = out_dir / "results_paper.tex"

    headers = ["id", "label", "type", "rmse", "mae", "bias", "corr",
               "improve_pct", "n_files", "params_m", "gflops_inf",
               "mem_inf_mb", "mem_train_mb"]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows_valid:
            vals = []
            for h in headers:
                v = r.get(h, "")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Overall comparison on test set.}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{llccccccc}\n\\toprule\n")
        f.write("Method & Type & RMSE & MAE & Bias & Corr & Improve(\\%) & Params(M) & GFLOPs\\\\\n")
        f.write("\\midrule\n")
        for r in rows_valid:
            label = str(r.get("label", "")).replace("_", "\\_")
            rtype = str(r.get("type", "")).replace("_", "\\_")
            rmse = f"{float(r['rmse']):.4f}" if _is_finite(r.get("rmse")) else "--"
            mae = f"{float(r['mae']):.4f}" if _is_finite(r.get("mae")) else "--"
            bias = f"{float(r['bias']):+.4f}" if _is_finite(r.get("bias")) else "--"
            corr = f"{float(r['corr']):.5f}" if _is_finite(r.get("corr")) else "--"
            imp = f"{float(r['improve_pct']):+.2f}" if _is_finite(r.get("improve_pct")) else "--"
            params = f"{float(r.get('params_m', 0)):.2f}" if _is_finite(r.get("params_m")) else "--"
            gflops = f"{float(r.get('gflops_inf', 0)):.2f}" if _is_finite(r.get("gflops_inf")) else "--"
            f.write(f"{label} & {rtype} & {rmse} & {mae} & {bias} & {corr} & "
                    f"{imp} & {params} & {gflops}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")


def plot_rmse_bar(rows, out_png: Path):
    rows_valid = [r for r in rows if _is_finite(r.get("rmse", np.nan))]
    rows_sorted = sorted(rows_valid, key=lambda x: float(x.get("rmse", np.nan)))
    if not rows_sorted:
        return

    labels = [clean_label(r["label"]) for r in rows_sorted]
    rmse = [float(r["rmse"]) for r in rows_sorted]
    types = [r.get("type", "") for r in rows_sorted]
    color_map = {"bkg": "#B0BEC5", "oi": "#43A047", "ours": "#263238",
                 "ablation": "#FB8C00", "compare": "#1E88E5"}
    colors = [color_map.get(t, "#90A4AE") for t in types]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.barh(labels, rmse, color=colors, edgecolor="white", height=0.64)
    for b, v in zip(bars, rmse):
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=9)
    ax.set_xlabel("RMSE (K)", fontsize=12)
    ax.set_title("Overall RMSE Comparison (Lower is Better)", fontsize=13)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    plt.tight_layout()
    save_figure(out_png, dpi=240, bbox_inches="tight")
    plt.close()


def plot_improve_bar(rows, out_png: Path):
    rows2 = [r for r in rows if _is_finite(r.get("improve_pct", np.nan))]
    rows2 = sorted(rows2, key=lambda x: float(x.get("improve_pct", -1e9)),
                   reverse=True)
    if not rows2:
        return

    labels = [clean_label(r["label"]) for r in rows2]
    imp = [float(r.get("improve_pct", np.nan)) for r in rows2]
    colors = ["#2E7D32" if v >= 0 else "#C62828" for v in imp]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.barh(labels, imp, color=colors, edgecolor="white", height=0.64)
    for b, v in zip(bars, imp):
        dx = 0.35 if v >= 0 else -0.35
        ax.text(v + dx, b.get_y() + b.get_height() / 2, f"{v:+.2f}%",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0.0, color="black", lw=1.0, alpha=0.7)
    ax.set_xlabel("Improvement vs Background (%)", fontsize=12)
    ax.set_title("Relative Improvement (Higher is Better)", fontsize=13)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    plt.tight_layout()
    save_figure(out_png, dpi=240, bbox_inches="tight")
    plt.close()


def plot_combined(rows, out_png: Path):
    rows_valid = [r for r in rows if _is_finite(r.get("rmse", np.nan))]
    rows_sorted = sorted(rows_valid, key=lambda x: float(x.get("rmse", np.nan)))
    if not rows_sorted:
        return

    labels = [clean_label(r["label"]) for r in rows_sorted]
    rmse = [float(r["rmse"]) for r in rows_sorted]
    imp = [float(r.get("improve_pct", np.nan))
           if _is_finite(r.get("improve_pct")) else np.nan for r in rows_sorted]

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
    save_figure(out_png, dpi=260, bbox_inches="tight")
    plt.close()

def plot_rmse_vs_params(rows, out_png: Path):
    valid = [r for r in rows
             if _is_finite(r.get("rmse")) and _is_finite(r.get("params_m"))
             and r.get("type") not in ["bkg", "oi"]]
    if not valid:
        return

    type_colors = {"ours": "#D32F2F", "ablation": "#FB8C00", "compare": "#1E88E5"}
    type_markers = {"ours": "*", "ablation": "s", "compare": "o"}

    # 1. 增大画布，给标注和图例留足空间
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=120)
    texts = []
    for r in valid:
        t = r.get("type", "")
        c = type_colors.get(t, "#90A4AE")
        m = type_markers.get(t, "o")
        ms = 220 if t == "ours" else 120
        label = clean_label(r["label"])

        ax.scatter(r["params_m"], r["rmse"], s=ms, c=c, marker=m, edgecolors="white",
                   zorder=5, alpha=0.95, linewidth=1.2)
        
# 2. 初始位置直接放在点上，释放排布自由度
        x, y = r["params_m"], r["rmse"]
        
        # 可选：如果您想顺手处理类似 "(v6) w/o mask-aware (v6)" 的重复问题
        # 可以增强 clean_label 或在这里做简单的替换
        # label = label.replace(" (v6)", "") 
        
        texts.append(ax.text(
            x, y, 
            label,
            fontsize=7.8,
            alpha=0.95,
            zorder=6,
            ha='center', va='center' # 改为 center，让 adjust_text 决定最终方向
        ))

    # 3. 强化 adjustText，彻底解决重叠
    try:
        from adjustText import adjust_text
        adjust_text(
            texts,
            ax=ax,
            # 增大文字之间、文字与点之间的排斥力
            force_text=(0.5, 1.2),    # 增大垂直方向的排斥力 (x, y)
            force_points=(0.5, 0.8),  # 让文字更用力地躲开数据点
            expand_points=(1.5, 1.5), # 将数据点的虚拟边界放大，防止文字贴太近
            expand_text=(1.2, 1.2),   # 将文字的虚拟边界放大
            arrowprops=dict(arrowstyle='-', color='#78909C', lw=0.8, alpha=0.8), # 使用没有箭头的实线，视觉上更清爽
            max_iterations=2000       # 增加迭代次数，确保算法有时间把重叠推开
        )
    except ImportError:
        pass

    # 4. 图例外置，完全不挡数据
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#D32F2F',
               markersize=15, label='Ours', markeredgecolor='white', markeredgewidth=1.2),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#FB8C00',
               markersize=11, label='Ablation', markeredgecolor='white', markeredgewidth=1.2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E88E5',
               markersize=11, label='Baseline', markeredgecolor='white', markeredgewidth=1.2),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=11,
        loc='upper right',
        bbox_to_anchor=(1.28, 1),
        borderaxespad=0,
        frameon=True,
        fancybox=True,
        shadow=False
    )

    # 5. 美化坐标轴和网格
    ax.set_xlabel("Parameters (Millions)", fontsize=14, fontweight='medium', labelpad=10)
    ax.set_ylabel("RMSE (K)", fontsize=14, fontweight='medium', labelpad=10)
    ax.set_title("RMSE vs. Model Size (Lower-Left is Better)", fontsize=15, fontweight='medium', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.tick_params(axis='both', labelsize=12)

    # 6. 调整布局，适配外置图例
    plt.subplots_adjust(right=0.82)
    save_figure(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_vertical_rmse(rows, out_png: Path):
    profiles = []
    for r in rows:
        p = r.get("per_level_rmse", None)
        if isinstance(p, np.ndarray) and p.shape[0] == len(PRESSURE_LEVELS):
            label = clean_label(r.get("label", ""))
            profiles.append((label, r.get("type", ""), p))
    if not profiles:
        return

    profiles_core = [p for p in profiles if p[0] in CORE_METHODS_FOR_VERTICAL]
    if len(profiles_core) < 3:
        profiles_core = profiles

    type_styles = {
        "bkg": {"color": "#9E9E9E", "lw": 2.0, "ls": "--"},
        "oi": {"color": "#43A047", "lw": 1.8, "ls": "-."},
        "ours": {"color": "#000000", "lw": 3.0, "ls": "-"},
        "ablation": {"color": "#F57C00", "lw": 1.6, "ls": "-"},
        "compare": {"color": "#1E88E5", "lw": 1.6, "ls": "-"},
    }
    fig, ax = plt.subplots(figsize=(7, 9))
    for label, mtype, prof in profiles_core:
        style = type_styles.get(mtype, {"color": "#90A4AE", "lw": 1.4, "ls": "-"})
        ax.plot(prof, PRESSURE_LEVELS,
                lw=style["lw"], ls=style["ls"], color=style["color"],
                label=label, alpha=0.9,
                zorder=10 if mtype == "ours" else 5)

    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 700, 1000])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100",
                        "200", "300", "500", "700", "1000"], fontsize=10)
    ax.set_xlabel("RMSE (K)", fontsize=13)
    ax.set_ylabel("Pressure (hPa)", fontsize=13)
    ax.set_title("Vertical RMSE Profile Comparison", fontsize=14)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9, edgecolor='gray')
    plt.tight_layout()
    save_figure(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    [OK] 垂直RMSE图 (核心方法): {out_png}")


def plot_loss_curves(rows, out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    valid_rows = [r for r in rows if r.get("type") not in ["bkg", "oi"]]
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(valid_rows), 1)))
    plotted = False

    for idx, r in enumerate(valid_rows):
        ckpt_path = r.get("ckpt", "")
        if not ckpt_path:
            continue
        # 搜索 train_history.csv
        ckpt_dir = Path(ckpt_path).parent
        csv_path = None
        for d in [ckpt_dir, ckpt_dir.parent, ckpt_dir.parent.parent]:
            candidate = d / "train_history.csv"
            if candidate.exists():
                csv_path = candidate
                break
        if csv_path is None:
            continue

        epochs, train_loss, val_loss = [], [], []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epochs.append(int(row['epoch']))
                    train_loss.append(float(row['train_loss']))
                    val_loss.append(float(row['val_loss']))
                except (ValueError, KeyError):
                    continue
        if not epochs:
            continue

        # 过滤 resume 的重复记录
        last_start_idx = len(epochs) - 1 - epochs[::-1].index(1) if 1 in epochs else 0
        epochs = epochs[last_start_idx:]
        train_loss = train_loss[last_start_idx:]
        val_loss = val_loss[last_start_idx:]

        clean_lbl = clean_label(r['label'])
        axes[0].plot(epochs, train_loss, label=clean_lbl, color=colors[idx], lw=1.5)
        axes[1].plot(epochs, val_loss, label=clean_lbl, color=colors[idx], lw=1.5)
        plotted = True

    if not plotted:
        plt.close()
        return

    axes[0].set_title("Train Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Validation Loss", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    save_figure(out_png, dpi=260, bbox_inches="tight")
    plt.close()


def plot_resources(rows, out_png: Path):
    res_rows = [r for r in rows
                if r.get("type") not in ["bkg", "oi"]
                and _is_finite(r.get("params_m", np.nan))]
    if not res_rows:
        return

    labels = [clean_label(r["label"]) for r in res_rows]
    params = [r["params_m"] for r in res_rows]
    gflops = [r.get("gflops_inf", 0) for r in res_rows]
    mem_train = [r.get("mem_train_mb", 0) for r in res_rows]
    mem_inf = [r.get("mem_inf_mb", 0) for r in res_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), gridspec_kw={"wspace": 0.15})

    bars1 = axes[0].barh(labels, params, color="#5C6BC0", edgecolor="white")
    axes[0].set_title("Parameters (Millions)", fontsize=12)
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].invert_yaxis()
    for b, v in zip(bars1, params):
        axes[0].text(v, b.get_y() + b.get_height() / 2, f" {v:.2f}",
                     va="center", fontsize=9)

    bars2 = axes[1].barh(labels, gflops, color="#26A69A", edgecolor="white")
    axes[1].set_title("Inference GFLOPs (Batch=1)", fontsize=12)
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].invert_yaxis()
    axes[1].set_yticks([])
    for b, v in zip(bars2, gflops):
        if _is_finite(v):
            axes[1].text(v, b.get_y() + b.get_height() / 2, f" {v:.2f}",
                         va="center", fontsize=9)

    x = np.arange(len(labels))
    width = 0.38
    axes[2].barh(x - width / 2, mem_train, width, label='Train VRAM', color="#EF5350")
    axes[2].barh(x + width / 2, mem_inf, width, label='Inference VRAM', color="#42A5F5")
    axes[2].set_yticks(x)
    axes[2].set_yticklabels([])
    axes[2].set_title("Peak VRAM Usage (MB, Batch=1)", fontsize=12)
    axes[2].grid(axis="x", alpha=0.25)
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=9)
    for b, v in zip(axes[2].patches[:len(mem_train)], mem_train):
        if _is_finite(v):
            axes[2].text(v, b.get_y() + b.get_height() / 2, f" {v:.0f}",
                         va="center", fontsize=8)

    fig.suptitle("Model Resources and Computation Overhead", fontsize=14)
    plt.tight_layout()
    save_figure(out_png, dpi=260, bbox_inches="tight")
    plt.close()


# =============================================================================
# Part K: YAML 配置支持
# =============================================================================

def get_experiments_from_yaml(yaml_cfg: dict) -> list:
    experiments = []
    for exp in yaml_cfg.get("experiments", []):
        experiments.append({
            "id": exp["id"],
            "label": exp["label"],
            "type": exp["type"],
            "ckpt": exp["ckpt"],
        })
    return experiments


def get_experiments_from_legacy(base_dir: str) -> list:
    outs = Path(base_dir) / "train_ddp" / "outputs"
    return [
        {"id": "v1", "label": "Ours (V1)", "type": "ours",
         "ckpt": str(outs / "ablation_v1_no_aux" / "experiment_ddp" / "best_model.pth")},
        {"id": "full", "label": "Full Variant", "type": "ablation",
         "ckpt": str(outs / "increment_era5_bkg_64x64" / "experiment_ddp" / "best_model.pth")},
    ]


# =============================================================================
# Part L: 汇总生成管线
# =============================================================================

def generate_ablation_paper_assets(
    rows, out_dir: Path,
    experiments: list = None,
    test_files: list = None,
    stats: dict = None,
    inc_stats: dict = None,
    device: str = "cuda",
    selected_plots: Set[str] = None,
):
    """完整生成管线: 原有 + 7个新增功能"""
    selected_plots = set(PLOT_OPTIONS) if selected_plots is None else selected_plots

    def enabled(key: str) -> bool:
        return key in selected_plots

    print("\n" + "=" * 60)
    print("生成论文图表和统计结果...")
    print("=" * 60)

    # --- 原有功能 ---
    if enabled("tables"):
        print("\n[1/12] 结果表格 (CSV + LaTeX)...")
        save_paper_tables(rows, out_dir)
    else:
        print("\n[1/12] SKIP: tables")

    if enabled("rmse_bar"):
        print("[2/12] RMSE 柱状图...")
        plot_rmse_bar(rows, out_dir / "fig_rmse_bar_sorted.png")
    else:
        print("[2/12] SKIP: rmse_bar")

    if enabled("improve_bar"):
        print("[3/12] 改善率柱状图...")
        plot_improve_bar(rows, out_dir / "fig_improvement_bar.png")
    else:
        print("[3/12] SKIP: improve_bar")

    if enabled("combined"):
        print("[4/12] 组合对比图...")
        plot_combined(rows, out_dir / "fig_combined_rmse_improve.png")
    else:
        print("[4/12] SKIP: combined")

    if enabled("vertical"):
        print("[5/12] 垂直RMSE剖面...")
        plot_vertical_rmse(rows, out_dir / "vertical_rmse_comparison.png")
    else:
        print("[5/12] SKIP: vertical")

    if enabled("loss"):
        print("[6/12] Loss曲线...")
        plot_loss_curves(rows, out_dir / "fig_loss_curves.png")
    else:
        print("[6/12] SKIP: loss")

    if enabled("resources"):
        print("[7/12] 资源消耗图...")
        plot_resources(rows, out_dir / "fig_resources.png")
    else:
        print("[7/12] SKIP: resources")

    if enabled("rmse_vs_params"):
        print("[8/12] RMSE vs 参数量散点图...")
        plot_rmse_vs_params(rows, out_dir / "fig_rmse_vs_params.png")
    else:
        print("[8/12] SKIP: rmse_vs_params")

    # --- 新增功能 (需要模型推理) ---
    if experiments and test_files and stats:
        # [新增] 缺失率统计
        if enabled("missing_rate"):
            print("\n[++] 分析缺测率分布...")
            plot_missing_rate_distribution(test_files, out_dir)

        if enabled("grouped"): compute_grouped_metrics(rows, out_dir)
        if enabled("significance"): run_significance_tests(rows, out_dir)

        ours_exp = next((exp for exp in experiments if exp.get("type") == "ours" and Path(exp["ckpt"]).exists()), None)
        
        need_ours_model = any(enabled(k) for k in ("sample_panels", "spatial_maps", "se_channels"))
        if ours_exp and (need_ours_model or enabled("extreme_cases")):
            print("\n[++] Ours 模型相关分析 (Panels, 空间误差, 极端案例, SE权重)...")
            model_ours, _, use_aux_ours = _load_model(ours_exp["ckpt"], device) if need_ours_model else (None, None, None)

            if enabled("sample_panels") and need_ours_model: pass # ...执行...
            if enabled("spatial_maps") and need_ours_model: pass # ...执行...

            # [新增] SE Channels 注意力权重视觉化
            if enabled("se_channels") and need_ours_model:
                analyze_se_channels(model_ours, test_files, stats, use_aux_ours, device, out_dir)
            
            if model_ours is not None:
                del model_ours; torch.cuda.empty_cache()

            if enabled("extreme_cases"): analyze_extreme_cases(ours_exp, test_files, stats, inc_stats, device, out_dir, n_extreme=3)

        dl_experiments = [e for e in experiments if Path(e["ckpt"]).exists()]
        if enabled("error_distribution"): plot_error_distribution(dl_experiments, test_files, stats, inc_stats, device, out_dir, n_samples=min(100, len(test_files)))
        if enabled("latency"): benchmark_latency(dl_experiments, test_files, stats, device, out_dir)
        
        # [新增] Gap Robustness
        if enabled("gap_robustness"):
            print("\n[++] 评估模型 Gap Robustness...")
            evaluate_gap_robustness(dl_experiments, test_files, stats, inc_stats, device, out_dir, n_samples=min(100, len(test_files)))
        # 找 ours 实验
        ours_exp = None
        for exp in experiments:
            if exp.get("type") == "ours" and Path(exp["ckpt"]).exists():
                ours_exp = exp
                break

        if enabled("grouped"):
            print("\n[9/12] 分组统计表 (平流层/对流层/近地面)...")
            compute_grouped_metrics(rows, out_dir)
        else:
            print("\n[9/12] SKIP: grouped")

        if enabled("significance"):
            print("[10/12] 统计显著性检验 (Paired Bootstrap)...")
            run_significance_tests(rows, out_dir)
        else:
            print("[10/12] SKIP: significance")

        need_ours_model = any(enabled(k) for k in ("sample_panels", "spatial_maps"))

        if ours_exp and (need_ours_model or enabled("extreme_cases")):
            print("[11/12] 单样本可视化 + 空间误差图 + 极端案例...")

            model_ours, _, use_aux_ours = (None, None, None)
            if need_ours_model:
                model_ours, _, use_aux_ours = _load_model(ours_exp["ckpt"], device)

            # 五面板可视化 (中间一个样本)
            if enabled("sample_panels") and need_ours_model:
                mid_idx = len(test_files) // 2
                for lidx in [5, 17, 21, 30]:
                    plot_single_sample_panels(
                        model_ours, test_files[mid_idx], stats, inc_stats,
                        use_aux_ours, device, out_dir,
                        level_idx=lidx, tag="median"
                    )
            else:
                print("    [SKIP] sample_panels")

            # 空间误差热力图
            if enabled("spatial_maps") and need_ours_model:
                plot_spatial_error_maps(
                    model_ours, test_files, stats, inc_stats,
                    use_aux_ours, device, out_dir, n_samples=min(100, len(test_files))
                )
            else:
                print("    [SKIP] spatial_maps")

            if model_ours is not None:
                del model_ours
                torch.cuda.empty_cache()

            # 极端案例
            if enabled("extreme_cases"):
                analyze_extreme_cases(
                    ours_exp, test_files, stats, inc_stats,
                    device, out_dir, n_extreme=3
                )
            else:
                print("    [SKIP] extreme_cases")
        else:
            if not (need_ours_model or enabled("extreme_cases")):
                print("[11/12] SKIP: 未选择 sample_panels/spatial_maps/extreme_cases")
            else:
                print("[11/12] SKIP: 未找到 ours 实验")

        print("[12/12] 误差分布直方图 + 推理延迟...")
        # 只对DL模型做
        dl_experiments = [e for e in experiments if Path(e["ckpt"]).exists()]
        if enabled("error_distribution"):
            plot_error_distribution(
                dl_experiments, test_files, stats, inc_stats,
                device, out_dir, n_samples=min(100, len(test_files))
            )
        else:
            print("    [SKIP] error_distribution")

        if enabled("latency"):
            benchmark_latency(dl_experiments, test_files, stats, device, out_dir)
        else:
            print("    [SKIP] latency")
    else:
        print("\n[9-12] SKIP: 缺少 experiments/test_files/stats，跳过推理相关分析")

    print("\n" + "=" * 60)
    print(f"✓ 所有图表生成完成!  输出目录: {out_dir}")
    print("=" * 60)


# =============================================================================
# Part M: LaTeX 论文草稿生成
# =============================================================================
# =============================================================================
# Part M: LaTeX 论文草稿生成 (Neurocomputing 版本)
# =============================================================================

def write_neurocomputing_tex(rows, out_tex_path: Path):
    lines = []
    lines.append(r"\documentclass[preprint,12pt]{elsarticle}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{amsmath,amssymb}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\usepackage{xcolor}")
    lines.append("")
    # 修改目标期刊为 Neurocomputing
    lines.append(r"\journal{Neurocomputing}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{frontmatter}")
    lines.append(r"\title{Physics-Guided Satellite Data Assimilation with Ablation-Oriented Evaluation}")
    lines.append(r"\author{Liu Ruixin}")
    lines.append(r"\address{School of Automation, Southeast University, Nanjing, China}")
    lines.append(r"\begin{abstract}")
    lines.append("This draft reports an ablation-oriented neural assimilation pipeline. "
                 "We evaluate model accuracy, computational overhead, and statistical significance.")
    lines.append(r"\end{abstract}")
    lines.append(r"\begin{keyword}")
    lines.append(r"Satellite data assimilation \sep Neural network \sep Ablation study \sep Statistical significance")
    lines.append(r"\end{keyword}")
    lines.append(r"\end{frontmatter}")
    lines.append("")

    out_dir = out_tex_path.parent

    # Main table
    if (out_dir / "results_paper.tex").exists():
        lines.append(r"\section{Results}")
        lines.append(r"\input{results_paper.tex}")
        lines.append("")

    # Grouped table
    if (out_dir / "grouped_rmse_table.tex").exists():
        lines.append(r"\subsection{Layer-wise Performance}")
        lines.append(r"\input{grouped_rmse_table.tex}")
        lines.append("")

    # Significance
    if (out_dir / "significance_tests.tex").exists():
        lines.append(r"\subsection{Statistical Significance}")
        lines.append(r"\input{significance_tests.tex}")
        lines.append("")

    # Figures
    for fig_name, caption in [
        ("fig_combined_rmse_improve.png", "Combined RMSE and improvement comparison."),
        ("vertical_rmse_comparison.png", "Vertical RMSE profiles across pressure levels."),
        ("fig_loss_curves.png", "Training and validation loss curves."),
        ("fig_resources.png", "Model parameters, GFLOPs, and VRAM usage."),
        ("fig_rmse_vs_params.png", "RMSE vs model size trade-off."),
        ("fig_latency_throughput.png", "Inference latency and throughput comparison."),
        ("fig_error_distribution.png", "Prediction error probability density."),
        ("fig_spatial_error_maps.png", "Spatial mean bias and RMSE maps at representative levels."),
        ("fig_forest_plot_significance.png", "Forest plot of paired bootstrap significance tests."),
        ("fig_grouped_rmse_bar.png", "RMSE breakdown by atmospheric layer group."),
    ]:
        if not (out_dir / fig_name).exists():
            continue
        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.9\linewidth]{" + fig_name + "}")
        lines.append(r"\caption{" + caption + "}")
        lines.append(r"\end{figure}")
        lines.append("")

    lines.append(r"\section{Conclusion}")
    lines.append("See generated figures and tables for detailed analysis.")
    lines.append(r"\end{document}")

    out_tex_path.write_text("\n".join(lines), encoding="utf-8")

# =============================================================================
# Part N: 控制台输出
# =============================================================================

def print_table(rows):
    print("\n| 方法 | 类型 | RMSE(K) | MAE(K) | Bias(K) | Corr | 改善% | Params(M) |")
    print("|------|------|---------|--------|---------|------|-------|-----------|")
    for r in rows:
        rmse = f"{r['rmse']:.4f}" if _is_finite(r.get("rmse")) else "—"
        mae = f"{r.get('mae', 0):.4f}" if "mae" in r and _is_finite(r.get("mae")) else "—"
        bias = f"{r.get('bias', 0):+.4f}" if "bias" in r and _is_finite(r.get("bias")) else "—"
        corr = f"{r.get('corr', 0):.5f}" if "corr" in r and _is_finite(r.get("corr")) else "—"
        imp = f"{r.get('improve_pct', 0):+.2f}%" if "improve_pct" in r and _is_finite(r.get("improve_pct")) else "—"
        params = f"{r.get('params_m', 0):.2f}" if _is_finite(r.get("params_m")) else "—"
        print(f"| {r['label']} | {r.get('type', '')} | {rmse} | {mae} | "
              f"{bias} | {corr} | {imp} | {params} |")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None,
                   help="YAML 配置文件路径（优先级最高）")
    p.add_argument("--test_root", default="/data2/lrx/npz_64_real/test")
    p.add_argument("--stats_file", default="/data2/lrx/npz_64_real/stats.npz")
    p.add_argument("--increment_stats", default="/data2/lrx/npz_64_real/increment_stats.npz")
    p.add_argument("--output_dir", default="./figures_ablation_comparison_v3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--base_dir", default=str(Path(__file__).parent.parent))
    p.add_argument("--skip_missing", action="store_true")
    p.add_argument(
        "--plots",
        default="all",
        help=(
            "要生成的图表键名，逗号分隔；默认 all。"
            f"可选: {','.join(PLOT_OPTIONS)}"
        ),
    )
    p.add_argument(
        "--exp_ids",
        default="all",
        help="仅评估指定实验ID，逗号分隔（默认 all）",
    )
    p.add_argument(
        "--exp_types",
        default="all",
        help="仅评估指定实验类型，逗号分隔（如 ours,ablation,compare；默认 all）",
    )
    p.add_argument(
        "--list_plot_options",
        action="store_true",
        help="仅打印可用图表键并退出",
    )
    args = p.parse_args()

    if args.list_plot_options:
        print("可选图表键:")
        for k in PLOT_OPTIONS:
            print(f"  - {k}")
        return

    selected_plots = resolve_plot_selection(args.plots)

    # YAML 配置
    yaml_cfg = None
    if args.config and Path(args.config).exists():
        if yaml is None:
            raise ImportError("需要 pyyaml: pip install pyyaml")
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)
        print(f"[INFO] 从 YAML 加载配置: {args.config}")
        args.test_root = yaml_cfg.get("test_root", args.test_root)
        args.stats_file = yaml_cfg.get("stats_file", args.stats_file)
        args.increment_stats = yaml_cfg.get("increment_stats", args.increment_stats)
        args.output_dir = yaml_cfg.get("output_dir", args.output_dir)
        args.device = yaml_cfg.get("device", args.device)
        args.skip_missing = yaml_cfg.get("skip_missing", args.skip_missing)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = dict(np.load(args.stats_file))
    inc_stats = dict(np.load(args.increment_stats)) \
        if Path(args.increment_stats).exists() else None
    test_files = load_test_files(args.test_root)
    test_res = infer_test_resolution(args.test_root, test_files)
    print(f"测试文件: {len(test_files)}")
    if test_res is not None:
        print(f"[INFO] 推断测试分辨率: {test_res}x{test_res}")

    # ---- 背景场 ----
    print("\n[背景场] 计算...")
    pl_bkg = np.zeros(37)
    pl_cnt = np.zeros(37)
    rmse_bkg_all, mae_bkg_all, bias_bkg_all, corr_bkg_all = [], [], [], []
    per_sample_rmse_bkg_global = []

    for f in tqdm(test_files, desc="背景场", leave=False):
        d = np.load(f)
        bp, tp = d["bkg"], d["target"]
        sq = (bp - tp) ** 2
        pl_bkg += sq.reshape(37, -1).mean(1)
        pl_cnt += 1
        s_rmse = float(np.sqrt(sq.mean()))
        rmse_bkg_all.append(s_rmse)
        per_sample_rmse_bkg_global.append(s_rmse)
        mae_bkg_all.append(float(np.abs(bp - tp).mean()))
        bias_bkg_all.append(float((bp - tp).mean()))
        b_flat, t_flat = bp.ravel(), tp.ravel()
        if np.std(b_flat) > 0 and np.std(t_flat) > 0:
            corr_bkg_all.append(float(np.corrcoef(b_flat, t_flat)[0, 1]))
        else:
            corr_bkg_all.append(0.0)

    bkg_rmse = float(np.mean(rmse_bkg_all))
    bkg_per = np.sqrt(pl_bkg / np.maximum(pl_cnt, 1))
    rows = [{
        "id": "bkg", "label": "Background (ERA5)", "type": "bkg",
        "rmse": bkg_rmse,
        "mae": float(np.mean(mae_bkg_all)),
        "bias": float(np.mean(bias_bkg_all)),
        "corr": float(np.mean(corr_bkg_all)),
        "improve_pct": 0.0,
        "n_files": len(rmse_bkg_all),
        "per_level_rmse": bkg_per,
        "per_sample_rmse": per_sample_rmse_bkg_global,
        "per_sample_rmse_bkg": per_sample_rmse_bkg_global,
    }]
    print(f"  背景场 RMSE: {bkg_rmse:.4f} K")

    # ---- 变分基线（OI/3DVar/4DVar） ----
    oi_dir_path = _resolve_baseline_dir(yaml_cfg, args.base_dir, test_res, "oi_results_dir", "oi_results")
    _append_external_baseline(
        rows=rows,
        yaml_cfg=yaml_cfg,
        baseline_dir=oi_dir_path,
        baseline_name="OI/1DVar (B2)",
        baseline_id="oi",
        baseline_type="oi",
        bkg_rmse=bkg_rmse,
        bkg_per=bkg_per,
        n_expected=len(rmse_bkg_all),
        allow_better_default=False,
    )

    var3d_dir_path = _resolve_baseline_dir(yaml_cfg, args.base_dir, test_res, "var3d_results_dir", "var3d_results")
    _append_external_baseline(
        rows=rows,
        yaml_cfg=yaml_cfg,
        baseline_dir=var3d_dir_path,
        baseline_name="3DVar (B8)",
        baseline_id="3dvar",
        baseline_type="var3d",
        bkg_rmse=bkg_rmse,
        bkg_per=bkg_per,
        n_expected=len(rmse_bkg_all),
        allow_better_default=True,
    )

    var4d_dir_path = _resolve_baseline_dir(yaml_cfg, args.base_dir, test_res, "var4d_results_dir", "var4d_results")
    _append_external_baseline(
        rows=rows,
        yaml_cfg=yaml_cfg,
        baseline_dir=var4d_dir_path,
        baseline_name="4DVar (B9)",
        baseline_id="4dvar",
        baseline_type="var4d",
        bkg_rmse=bkg_rmse,
        bkg_per=bkg_per,
        n_expected=len(rmse_bkg_all),
        allow_better_default=True,
    )

    # ---- 深度学习实验 ----
    if yaml_cfg is not None:
        exp_list = get_experiments_from_yaml(yaml_cfg)
    else:
        exp_list = get_experiments_from_legacy(args.base_dir)

    exp_list = filter_experiments(exp_list, args.exp_ids, args.exp_types)
    print(f"[INFO] 参与评估的实验数: {len(exp_list)}")

    for exp in exp_list:
        ckpt = exp["ckpt"]
        if not Path(ckpt).exists():
            if args.skip_missing:
                print(f"\n[跳过] {exp['label']}  (ckpt 不存在)")
                continue
            rows.append({
                **exp, "rmse": float("nan"), "mae": float("nan"),
                "bias": float("nan"), "corr": float("nan"),
                "improve_pct": float("nan"), "n_files": 0,
            })
            continue

        print(f"\n[{exp['id']}] 评估: {exp['label']}")
        try:
            res = evaluate_dl_model(ckpt, test_files, stats, inc_stats,
                                    device=args.device)
            imp = (bkg_rmse - res["rmse"]) / bkg_rmse * 100 \
                if bkg_rmse > 0 else float("nan")
            rows.append({**exp, **res, "improve_pct": imp})
            print(f"  RMSE={res['rmse']:.4f}K  改善={imp:.2f}%")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ---- 排序输出 ----
    ordered = []
    for t in ("bkg", "oi", "var3d", "var4d", "ours", "ablation", "compare"):
        ordered.extend([r for r in rows if r.get("type") == t])
    print_table(ordered)

    # ---- 生成所有图表 (含新增7项) ----
    generate_ablation_paper_assets(
        ordered, out_dir,
        experiments=exp_list,
        test_files=test_files,
        stats=stats,
        inc_stats=inc_stats,
        device=args.device,
        selected_plots=selected_plots,
    )

    # ---- JSON 汇总 ----
    save = []
    for r in ordered:
        entry = {}
        for k, v in r.items():
            if k in ("per_level_rmse", "per_level_rmse_bkg"):
                entry[k] = v.tolist() if isinstance(v, np.ndarray) else v
            elif k in ("per_sample_rmse", "per_sample_rmse_bkg"):
                continue  # 太大，不存JSON
            else:
                entry[k] = v
        save.append(entry)
    with open(out_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(save, f, indent=2, ensure_ascii=False)

    # ---- LaTeX 草稿 ----
    if "latex_draft" in selected_plots:
        write_neurocomputing_tex(ordered, out_dir / "neurocomputing_draft.tex")
    else:
        print("[SKIP] latex_draft")
    print(f"\n✓ 汇总评估完成!  结果: {out_dir}")

if __name__ == "__main__":
    main()
