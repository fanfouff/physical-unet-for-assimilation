#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 可选自动激活 conda 环境（默认 fuxi）
if [[ "${AUTO_ACTIVATE_CONDA:-1}" == "1" && -f "$HOME/miniconda3/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/bin/activate"
  conda activate "${CONDA_ENV:-fuxi}" || {
    echo "[WARN] conda 环境激活失败，继续使用当前 Python: $PYTHON_BIN"
  }
fi

GEN_SCRIPT="$SCRIPT_DIR/gen_eval_config.py"
EVAL_SCRIPT="$SCRIPT_DIR/eval_all_experiments.py"

DEFAULT_ROOT_64="/home/lrx/Unet/satellite_assimilation_v2/train_ddp/outputs/figures_ablation_comparison_noaux64"
DEFAULT_ROOT_128="/home/lrx/Unet/satellite_assimilation_v2/train_ddp/outputs/figures_ablation_comparison_noaux128"

DEFAULT_TEST_ROOT_64="/data3/lrx/era_obs/npz_64_real/test"
DEFAULT_STATS_FILE_64="/data3/lrx/era_obs/npz_64_real/stats.npz"
DEFAULT_INCREMENT_STATS_64="/data3/lrx/era_obs/npz_64_real/increment_stats.npz"

DEFAULT_TEST_ROOT_128="/data3/lrx/npz_128_A202604221458120255/test"
DEFAULT_STATS_FILE_128="/data3/lrx/npz_128_A202604221458120255/stats.npz"
DEFAULT_INCREMENT_STATS_128="/data3/lrx/npz_128_A202604221458120255/increment_stats.npz"

# 先给默认值，后续会根据 ROOT_MODE 自动改写
DEFAULT_TEST_ROOT="$DEFAULT_TEST_ROOT_64"
DEFAULT_STATS_FILE="$DEFAULT_STATS_FILE_64"
DEFAULT_INCREMENT_STATS="$DEFAULT_INCREMENT_STATS_64"

if [[ ! -f "$GEN_SCRIPT" || ! -f "$EVAL_SCRIPT" ]]; then
  echo "[ERROR] 未找到脚本文件:"
  echo "  $GEN_SCRIPT"
  echo "  $EVAL_SCRIPT"
  exit 1
fi

echo "==============================================="
echo "  Eval 自动化菜单 (选实验 + 选图表)"
echo "==============================================="

echo
echo "[1] 仅 noaux64"
echo "[2] 仅 noaux128"
echo "[3] 同时 noaux64 + noaux128"
echo "[4] 自定义 exp_root（可多个，空格分隔）"
read -r -p "请选择实验目录来源 [3]: " ROOT_MODE
ROOT_MODE="${ROOT_MODE:-3}"

EXP_ROOTS=()
case "$ROOT_MODE" in
  1)
    EXP_ROOTS=("$DEFAULT_ROOT_64")
    ;;
  2)
    EXP_ROOTS=("$DEFAULT_ROOT_128")
    ;;
  3)
    EXP_ROOTS=("$DEFAULT_ROOT_64" "$DEFAULT_ROOT_128")
    ;;
  4)
    read -r -p "输入 exp_root 列表（空格分隔）: " CUSTOM_ROOTS
    if [[ -z "${CUSTOM_ROOTS// }" ]]; then
      echo "[ERROR] 你没有输入任何 exp_root"
      exit 1
    fi
    read -r -a EXP_ROOTS <<< "$CUSTOM_ROOTS"
    ;;
  *)
    echo "[ERROR] 无效选项: $ROOT_MODE"
    exit 1
    ;;
esac

for root in "${EXP_ROOTS[@]}"; do
  if [[ ! -d "$root" ]]; then
    echo "[ERROR] 目录不存在: $root"
    exit 1
  fi
done

# 根据实验目录选择默认数据集：选 64 用 64 数据，选 128 用 128 数据。
DATA_SCALE="64"
case "$ROOT_MODE" in
  1)
    DATA_SCALE="64"
    ;;
  2)
    DATA_SCALE="128"
    ;;
  3)
    # 同时评估 64 + 128 时，默认给 64；你仍可在后续输入框手动改为 128。
    DATA_SCALE="64"
    ;;
  4)
    HAS_64=0
    HAS_128=0
    for root in "${EXP_ROOTS[@]}"; do
      [[ "$root" == *"64"* ]] && HAS_64=1
      [[ "$root" == *"128"* ]] && HAS_128=1
    done
    if [[ "$HAS_64" -eq 1 && "$HAS_128" -eq 0 ]]; then
      DATA_SCALE="64"
    elif [[ "$HAS_128" -eq 1 && "$HAS_64" -eq 0 ]]; then
      DATA_SCALE="128"
    else
      DATA_SCALE="64"
    fi
    ;;
esac

if [[ "$DATA_SCALE" == "128" ]]; then
  DEFAULT_TEST_ROOT="$DEFAULT_TEST_ROOT_128"
  DEFAULT_STATS_FILE="$DEFAULT_STATS_FILE_128"
  DEFAULT_INCREMENT_STATS="$DEFAULT_INCREMENT_STATS_128"
else
  DEFAULT_TEST_ROOT="$DEFAULT_TEST_ROOT_64"
  DEFAULT_STATS_FILE="$DEFAULT_STATS_FILE_64"
  DEFAULT_INCREMENT_STATS="$DEFAULT_INCREMENT_STATS_64"
fi

echo "[INFO] 已自动匹配默认数据集: ${DATA_SCALE}x${DATA_SCALE}"
echo "       test_root=$DEFAULT_TEST_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
DEFAULT_YAML="$SCRIPT_DIR/eval_configs/eval_config_auto_${DATA_SCALE}_${TS}.yaml"
DEFAULT_OUT_DIR="$SCRIPT_DIR/figures/figures_eval_menu_${DATA_SCALE}"

read -r -p "test_root [$DEFAULT_TEST_ROOT]: " TEST_ROOT
TEST_ROOT="${TEST_ROOT:-$DEFAULT_TEST_ROOT}"
read -r -p "stats_file [$DEFAULT_STATS_FILE]: " STATS_FILE
STATS_FILE="${STATS_FILE:-$DEFAULT_STATS_FILE}"
read -r -p "increment_stats [$DEFAULT_INCREMENT_STATS]: " INC_STATS
INC_STATS="${INC_STATS:-$DEFAULT_INCREMENT_STATS}"
read -r -p "生成的 YAML 路径 [$DEFAULT_YAML]: " OUTPUT_YAML
OUTPUT_YAML="${OUTPUT_YAML:-$DEFAULT_YAML}"
read -r -p "评估输出目录 [$DEFAULT_OUT_DIR]: " OUTPUT_DIR
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUT_DIR}"

mkdir -p "$(dirname "$OUTPUT_YAML")" "$OUTPUT_DIR"

echo
echo "[INFO] 生成 YAML 配置..."
"$PYTHON_BIN" "$GEN_SCRIPT" \
  --exp_root "${EXP_ROOTS[@]}" \
  --test_root "$TEST_ROOT" \
  --stats_file "$STATS_FILE" \
  --increment_stats "$INC_STATS" \
  --output_dir "$OUTPUT_DIR" \
  --output_yaml "$OUTPUT_YAML" \
  --skip_missing

echo
echo "[INFO] YAML 中可选实验:"
"$PYTHON_BIN" - "$OUTPUT_YAML" <<'PY'
import sys
from pathlib import Path
try:
    import yaml
except ImportError:
    print("[ERROR] 请先安装 pyyaml: pip install pyyaml")
    raise

cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
for i, exp in enumerate(cfg.get("experiments", []), 1):
    print(f"  {i:02d}. id={exp.get('id')} | type={exp.get('type')} | label={exp.get('label')}")
PY

echo
read -r -p "选择实验ID (逗号分隔, 支持 b11/b12 简写, 或 all) [all]: " EXP_IDS
EXP_IDS="${EXP_IDS:-all}"

read -r -p "选择实验类型 (ours,ablation,compare, 或 all) [all]: " EXP_TYPES
EXP_TYPES="${EXP_TYPES:-all}"

echo
echo "图表预设:"
echo "  [1] all (全部图 + 表 + latex)"
echo "  [2] fast_core (tables,rmse_bar,improve_bar,combined,vertical,resources,rmse_vs_params,latex_draft)"
echo "  [3] paper_main (tables,combined,vertical,grouped,significance,resources,rmse_vs_params,latex_draft)"
echo "  [4] visual_debug (sample_panels,spatial_maps,extreme_cases,error_distribution,latency)"
echo "  [5] 自定义 plots 键 (逗号分隔)"
read -r -p "请选择图表预设 [1]: " PLOT_MODE
PLOT_MODE="${PLOT_MODE:-1}"

case "$PLOT_MODE" in
  1)
    PLOTS="all"
    ;;
  2)
    PLOTS="tables,rmse_bar,improve_bar,combined,vertical,resources,rmse_vs_params,latex_draft"
    ;;
  3)
    PLOTS="tables,combined,vertical,grouped,significance,resources,rmse_vs_params,latex_draft"
    ;;
  4)
    PLOTS="sample_panels,spatial_maps,extreme_cases,error_distribution,latency"
    ;;
  5)
    echo "可选 plots 键:"
    echo "tables,rmse_bar,improve_bar,combined,vertical,loss,resources,rmse_vs_params,grouped,significance,sample_panels,spatial_maps,extreme_cases,error_distribution,latency,latex_draft"
    read -r -p "输入 plots 键(逗号分隔): " PLOTS
    if [[ -z "${PLOTS// }" ]]; then
      echo "[ERROR] 自定义 plots 不能为空"
      exit 1
    fi
    ;;
  *)
    echo "[ERROR] 无效图表预设: $PLOT_MODE"
    exit 1
    ;;
esac

read -r -p "device [cuda]: " DEVICE
DEVICE="${DEVICE:-cuda}"

CMD=(
  "$PYTHON_BIN" "$EVAL_SCRIPT"
  --config "$OUTPUT_YAML"
  --output_dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --exp_ids "$EXP_IDS"
  --exp_types "$EXP_TYPES"
  --plots "$PLOTS"
  --skip_missing
)

echo
echo "[RUN] ${CMD[*]}"
"${CMD[@]}"

echo
echo "[DONE] 评估完成"
echo "  输出目录: $OUTPUT_DIR"
echo "  YAML配置 : $OUTPUT_YAML"