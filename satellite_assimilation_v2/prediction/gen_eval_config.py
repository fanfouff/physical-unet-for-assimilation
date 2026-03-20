#!/usr/bin/env python3
"""
gen_eval_config.py — 自动扫描实验输出目录，生成评估所需的 YAML 配置文件

用法:
  python3 gen_eval_config.py \
    --exp_root /home/lrx/Unet/satellite_assimilation_v2/train_ddp/outputs/figures_ablation_comparison_noaux128 \
    --test_root /data2/lrx/npz_64_real/test \
    --stats_file /data2/lrx/npz_64_real/stats.npz \
    --increment_stats /data2/lrx/npz_64_real/increment_stats.npz \
    --output_yaml eval_config_neurocomputing.yaml

也可以指定多个 --exp_root 来扫描多个目录。
"""

import argparse
import json
import re
from pathlib import Path
from collections import OrderedDict

# ============================================================
# pip install pyyaml  (如果尚未安装)
# ============================================================
try:
    import yaml
except ImportError:
    raise ImportError("请先安装 pyyaml:  pip install pyyaml")


# ------------------------------------------------------------------
# 目录名 → 实验元数据 的映射规则
# 优先精确匹配；匹配不到再走正则推断
# ------------------------------------------------------------------
KNOWN_EXPERIMENTS = OrderedDict([
    # ── Ours ──
    ("ours_noaux_full_128", {
        "id": "v1",
        "label": "Ours (V1)",
        "type": "ours",
    }),
    ("ours_mse_128", {
        "id": "ours_mse",
        "label": "Ours-MSE",
        "type": "ours",
    }),
    # ── Ablation ──
    ("increment_era5_bkg_128x128", {
        "id": "full",
        "label": "Full Variant (Ablation)",
        "type": "ablation",
    }),
    ("v2_noaux_no_deep_supervision_128", {
        "id": "v2",
        "label": "w/o DeepSupervision (V2)",
        "type": "ablation",
    }),
    ("v3_fusion_add_no_deepsupervision128", {
        "id": "v3",
        "label": "w/o GatedFusion (V3)",
        "type": "ablation",
    }),
    ("v4_noaux_spectral_stem_128", {
        "id": "v4",
        "label": "w/o SpectralStem (V4)",
        "type": "ablation",
    }),
    ("v5_noaux_no_mask_aware_128", {
        "id": "v5",
        "label": "w/o MaskAware-MSE (V5)",
        "type": "ablation",
    }),
    ("v6_noaux_no_mask_aware_128", {
        "id": "v6",
        "label": "w/o MaskAware-Combined (V6)",
        "type": "ablation",
    }),
    # ── Baselines ──
    ("compare_b3_vanilla_unet_128", {
        "id": "b3",
        "label": "VanillaUNet (B3)",
        "type": "compare",
    }),
    ("compare_b4_fuxi_da_128", {
        "id": "b4",
        "label": "FuXi-DA (B4)",
        "type": "compare",
    }),
    ("compare_b5_attn_unet_128", {
        "id": "b5",
        "label": "AttentionUNet (B5)",
        "type": "compare",
    }),
    ("compare_b6_pixel_mlp_128", {
        "id": "b6",
        "label": "PixelMLP (B6)",
        "type": "compare",
    }),
    ("compare_b7_res_unet_128", {
        "id": "b7",
        "label": "ResUNet (B7)",
        "type": "compare",
    }),
])

# 正则推断规则  (pattern, type_guess, label_template)
REGEX_RULES = [
    (r"^ours",      "ours",     "Ours-{name}"),
    (r"^v\d+",      "ablation", "Ablation-{name}"),
    (r"^compare_b", "compare",  "Baseline-{name}"),
    (r"^increment", "ablation", "Increment-{name}"),
]


def infer_experiment(dir_name: str) -> dict:
    """对未知目录名做正则推断"""
    for pat, etype, label_tmpl in REGEX_RULES:
        if re.match(pat, dir_name):
            return {
                "id": dir_name,
                "label": label_tmpl.format(name=dir_name),
                "type": etype,
            }
    return {
        "id": dir_name,
        "label": dir_name,
        "type": "unknown",
    }


def scan_experiments(exp_roots: list, ckpt_name: str = "best_model.pth") -> list:
    """
    扫描一个或多个目录，找到所有含有 ckpt_name 的子目录，
    并映射成实验元数据列表。
    """
    results = []
    seen_ids = set()

    for root in exp_roots:
        root = Path(root)
        if not root.exists():
            print(f"[WARN] 目录不存在，跳过: {root}")
            continue

        for ckpt_path in sorted(root.rglob(ckpt_name)):
            exp_dir = ckpt_path.parent
            dir_name = exp_dir.name

            # 精确匹配 (使用 copy 防止修改原始字典)
            if dir_name in KNOWN_EXPERIMENTS:
                meta = KNOWN_EXPERIMENTS[dir_name].copy()
            else:
                meta = infer_experiment(dir_name)

            # 去重
            if meta["id"] in seen_ids:
                meta["id"] = f"{meta['id']}_{exp_dir.parent.name}"
            seen_ids.add(meta["id"])

            meta["ckpt"] = str(ckpt_path.resolve())

            # 顺便记录 config.json 路径（如果存在）
            config_json = exp_dir / "config.json"
            if config_json.exists():
                meta["config_json"] = str(config_json.resolve())
                # 读取 config 中的关键参数
                try:
                    with open(config_json) as f:
                        cfg = json.load(f)
                    meta["model_name"] = cfg.get("model", "physics_unet")
                    meta["loss_type"] = cfg.get("loss_type", "")
                    meta["use_aux"] = cfg.get("use_aux", None)
                except Exception:
                    pass

            # 记录 train_history.csv 路径
            history_csv = exp_dir / "train_history.csv"
            if history_csv.exists():
                meta["train_history"] = str(history_csv.resolve())

            results.append(meta)

    # 排序: ours → ablation → compare → unknown
    type_order = {"ours": 0, "ablation": 1, "compare": 2, "unknown": 3}
    results.sort(key=lambda x: (type_order.get(x["type"], 99), x["id"]))

    return results


def build_yaml_dict(args, experiments: list) -> dict:
    """组装完整的 YAML 配置字典"""
    return {
        # ── 数据路径 ──
        "test_root": str(args.test_root),
        "stats_file": str(args.stats_file),
        "increment_stats": str(args.increment_stats),

        # ── 输出 ──
        "output_dir": str(args.output_dir),

        # ── 评估参数 ──
        "device": args.device,
        "skip_missing": args.skip_missing,

        # ── OI 基线路径（可选） ──
        "oi_results_dir": str(args.oi_results_dir) if args.oi_results_dir else None,

        # ── 实验列表 ──
        "experiments": experiments,
    }


# ============================================================
# 自定义 YAML Dumper —— 让输出更美观
# ============================================================
class PrettyDumper(yaml.SafeDumper):
    pass

def str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

PrettyDumper.add_representer(str, str_representer)


def main():
    p = argparse.ArgumentParser(description="自动生成评估 YAML 配置文件")

    p.add_argument(
        "--exp_root", nargs="+", required=True,
        help="实验输出目录（可指定多个）"
    )
    p.add_argument(
        "--test_root",
        default="/data2/lrx/npz_64_real/test",
        help="测试集 npz 根目录"
    )
    p.add_argument(
        "--stats_file",
        default="/data2/lrx/npz_64_real/stats.npz",
        help="归一化统计文件"
    )
    p.add_argument(
        "--increment_stats",
        default="/data2/lrx/npz_64_real/increment_stats.npz",
        help="增量统计文件"
    )
    p.add_argument(
        "--output_dir",
        default="./figures_neurocomputing_eval", # [修改处] 与自动化脚本保持一致
        help="评估结果输出目录"
    )
    p.add_argument(
        "--output_yaml",
        default="eval_config_neurocomputing.yaml", # [修改处] 默认输出名更清晰
        help="输出的 YAML 文件路径"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_missing", action="store_true")
    p.add_argument(
        "--oi_results_dir",
        default=None,
        help="OI/1DVar 结果目录（可选）"
    )
    p.add_argument(
        "--ckpt_name",
        default="best_model.pth",
        help="checkpoint 文件名"
    )

    args = p.parse_args()

    # 1. 扫描实验
    print(f"扫描目录: {args.exp_root}")
    experiments = scan_experiments(args.exp_root, ckpt_name=args.ckpt_name)
    print(f"发现 {len(experiments)} 个实验:")
    for e in experiments:
        status = "✓" if Path(e["ckpt"]).exists() else "✗"
        print(f"  [{status}] {e['id']:20s}  {e['type']:10s}  {e['label']}")

    # 2. 组装
    cfg = build_yaml_dict(args, experiments)

    # 3. 写出
    out_path = Path(args.output_yaml)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# ====================================================\n")
        f.write("# eval_all_experiments.py 配置文件\n")
        f.write(f"# 自动生成 by gen_eval_config.py\n")
        f.write(f"# 目标期刊: Neurocomputing\n")
        f.write(f"# 实验数: {len(experiments)}\n")
        f.write("# ====================================================\n\n")
        yaml.dump(cfg, f, Dumper=PrettyDumper,
                  default_flow_style=False,
                  allow_unicode=True,
                  sort_keys=False,
                  width=120)

    print(f"\n✓ 配置已写入: {out_path}")
    print(f"  后续运行:")
    print(f"    python3 eval_all_experiments.py --config {out_path}")


if __name__ == "__main__":
    main()