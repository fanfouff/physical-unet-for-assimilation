#!/usr/bin/env bash
#===============================================================================
# 数据效率实验 — SwinUNet on data_efficiency_64_real_data3
# 每个实验 2 GPUs, 两两并行跑
#===============================================================================
set -euo pipefail

TRAIN_SCRIPT="$(cd "$(dirname "$0")" && pwd)/train_ddp.py"
OUTPUT_DIR="/home/lrx/Unet/satellite_assimilation_v2/train_ddp/outputs/data_efficiency_64_real_data3"
SPLIT_DIR="/home/lrx/Unet/satellite_assimilation_v2/train_ddp/splits/data_efficiency_64_real_data3"
DATA_ROOT="/data3/lrx/era_obs/npz_64_real"
STATS_FILE="$DATA_ROOT/stats.npz"
INC_STATS="$DATA_ROOT/increment_stats.npz"

# 公共参数 (与已完成 physics_unet 实验一致, model 换成 swin_unet)
COMMON_ARGS=(
  --output_dir "$OUTPUT_DIR"
  --data_root "$DATA_ROOT"
  --stats_file "$STATS_FILE"
  --increment_stats "$INC_STATS"
  --model swin_unet
  --fusion_mode gated
  --use_aux false
  --mask_aware true
  --use_spectral_stem true
  --deep_supervision true
  --epochs 200
  --batch_size 8
  --lr 0.0001
  --weight_decay 1e-05
  --scheduler cosine
  --warmup_epochs 5
  --grad_clip 1.0
  --grad_accum_steps 2
  --loss combined
  --grad_loss_weight 0.1
  --deep_loss_weight 0.3
  --vert_loss_weight 0.05
  --use_increment
  --amp false
  --sync_bn true
  --split_mode file
  --num_workers 4
  --seed 42
  --log_interval 10
  --val_interval 1
  --save_interval 200
)

run_exp() {
  local tag="$1"
  local gpu0="$2"
  local gpu1="$3"
  local port="$4"

  local split_file="$SPLIT_DIR/split_${tag}.json"
  local logfile="$OUTPUT_DIR/swin_unet_split_${tag}_run.log"

  echo "============================================================"
  echo "Launching: swin_unet_split_${tag} on GPUs $gpu0,$gpu1 (port $port)"
  echo "  split:  $split_file"
  echo "  log:    $logfile"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$gpu0,$gpu1" \
    nohup torchrun --nproc_per_node=2 --master_port="$port" \
      "$TRAIN_SCRIPT" \
      --exp_name "swin_unet_split_${tag}" \
      --split_file "$split_file" \
      "${COMMON_ARGS[@]}" \
      > "$logfile" 2>&1 &

  echo "  PID: $!"
}

# ---- Batch 1: 两个实验并行 (各占2张卡) ----
# run_exp tag gpu0 gpu1 port
run_exp "010pct" 2 3 29500
run_exp "025pct" 0 1 29501

echo ""
echo "Waiting for batch 1 (010pct + 025pct) to finish..."
wait

echo ""
echo "Batch 1 done. Starting batch 2..."

run_exp "050pct" 2 3 29500
run_exp "100pct" 0 1 29501

echo ""
echo "Waiting for batch 2 (050pct + 100pct) to finish..."
wait

echo ""
echo "All 4 swin_unet data-efficiency experiments finished."
