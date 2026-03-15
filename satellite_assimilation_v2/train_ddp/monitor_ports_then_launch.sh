#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lrx/Unet/satellite_assimilation_v2/train_ddp"
cd "$ROOT_DIR"

source ~/miniconda3/bin/activate
conda activate fuxi

DATA_ROOT="/data2/lrx/npz_128_real"
STATS_FILE="/data2/lrx/npz_128_real/stats.npz"
INC_STATS="/data2/lrx/npz_128_real/increment_stats.npz"
COMMON_ARGS="--data_root $DATA_ROOT --stats_file $STATS_FILE --increment_stats $INC_STATS --use_increment --batch_size 16 --epochs 200"

WATCH_PORTS=(29512 29813 29516 29619 29417)
NEW_CMDS=(
"CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29814 train_ddp.py --exp_name compare_b4_fuxi_da_128 $COMMON_ARGS --model fuxi_da --use_aux false --deep_supervision false"
"CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29615 train_ddp.py --exp_name compare_b5_attn_unet_128 $COMMON_ARGS --model attn_unet --use_aux false --deep_supervision false"
"CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29616 train_ddp.py --exp_name compare_b6_pixel_mlp_128 $COMMON_ARGS --model pixel_mlp --use_aux false --deep_supervision false"
"CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29617 train_ddp.py --exp_name compare_b7_res_unet_128 $COMMON_ARGS --model res_unet --use_aux false --deep_supervision false"
)

is_port_running() {
  local p="$1"
  pgrep -af "torchrun.*--master_port=${p}.*train_ddp.py" >/dev/null 2>&1
}

any_training_running() {
  pgrep -af "train_ddp.py" >/dev/null 2>&1
}

echo "[$(date '+%F %T')] monitor started"
echo "watch ports: ${WATCH_PORTS[*]}"

for i in "${!WATCH_PORTS[@]}"; do
  wp="${WATCH_PORTS[$i]}"
  cmd="${NEW_CMDS[$i]}"

  echo "[$(date '+%F %T')] waiting old port $wp to finish..."
  while is_port_running "$wp"; do
    sleep 20
  done

  echo "[$(date '+%F %T')] old port $wp finished"

  while any_training_running; do
    echo "[$(date '+%F %T')] other training still running, wait before launch #$((i+1))"
    sleep 30
  done

  echo "[$(date '+%F %T')] launching #$((i+1)): $cmd"
  bash -lc "$cmd"
  echo "[$(date '+%F %T')] launch #$((i+1)) finished"
done

echo "[$(date '+%F %T')] all monitor+launch tasks done"
