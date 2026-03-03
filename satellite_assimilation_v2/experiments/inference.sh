#!/bin/bash
#===============================================================================
# PAS-Net 一键推理脚本
#===============================================================================

# 配置路径 (请根据实际路径修改)
DATA_ROOT="/data2/lrx/era_obs/npz/train"
STATS_FILE="/data2/lrx/era_obs/npz/stats.npz"
EXP_DIR="outputs/experiments"
OUT_DIR="outputs/inference_plots"

mkdir -p "$OUT_DIR"

echo "------------------------------------------------------------"
echo "开始批量同化推理..."
echo "------------------------------------------------------------"

# 遍历 experiments 目录下所有的子文件夹 (A1-A9, C1-C3)
for dir in "$EXP_DIR"/*; do
    if [ -d "$dir" ] && [ -f "$dir/checkpoint_best.pth" ]; then
        exp_name=$(basename "$dir")
        echo ">> 正在处理: $exp_name"
        
        python inference.py \
            --exp_dir "$dir" \
            --data_root "$DATA_ROOT" \
            --stats_file "$STATS_FILE" \
            --output_root "$OUT_DIR"
            
        if [ $? -eq 0 ]; then
            echo "   [SUCCESS] $exp_name 可视化已生成."
        else
            echo "   [FAILED] $exp_name 处理出错."
        fi
    fi
done

echo "------------------------------------------------------------"
echo "所有推理任务完成！结果保存在: $OUT_DIR"
echo "------------------------------------------------------------"