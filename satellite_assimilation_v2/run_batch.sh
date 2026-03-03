#!/bin/bash

# ================= 配置区 =================
# 你的测试数据路径 (包含 .npz 文件)
DATA_ROOT="/data2/lrx/era_obs/npz/test"

# 你的实验日志根目录 (里面应该有一堆实验文件夹)
# 假设你的实验都在这里
EXP_ROOT="/home/seu/Fuxi/Unet/satellite_assimilation_v2/experiments/outputs/experiments" 
# 或者绝对路径: "/home/seu/Fuxi/Unet/satellite_assimilation_v2/outputs"

# 统计文件 (必须和训练时用的一样)
STATS_FILE="/data2/lrx/era_obs/npz/stats.npz"

# 输出图片的位置
OUTPUT_DIR="paper_figures"
# ==========================================

echo "开始批量评估..."
echo "数据目录: $DATA_ROOT"
echo "实验目录: $EXP_ROOT"

python batch_evaluation.py \
    --data_root "$DATA_ROOT" \
    --exp_root "$EXP_ROOT" \
    --stats_file "$STATS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --device "cuda"

echo "完成! 请查看 $OUTPUT_DIR 目录下的 PDF 文件。"
