#!/bin/bash
##############################################################################
# 推理分析启动脚本 - 生成论文图表
# 
# 用法: bash run_inference_analysis.sh
#
# 功能:
#   - 在测试集上运行训练好的模型
#   - 生成论文所需的5类图表
#   - 可选：与基准模型对比
##############################################################################

# =============================================================================
# 配置区域
# =============================================================================

# 模型checkpoint路径
CHECKPOINT="/home/seu/Fuxi/Unet/satellite_assimilation_v2/outputs/FY3F_Assimilation_vanilla_unet_concat_auxfalse_maskfalse_20260125_202947/best_model.pth"

# 基准模型checkpoint（可选，留空则不对比）
BASELINE_CHECKPOINT=""  # 例如: "outputs/baseline/best_model.pth"

# 测试数据路径
DATA_ROOT="/data2/lrx/era_obs/npz/test" # 格式为.npz文件目录

# 统计量文件（可选）
STATS_FILE=""  # 留空则自动计算

# 输出目录
OUTPUT_DIR="figures"

# 个例可视化的样本索引
CASE_STUDY_IDX=0

# 批大小
BATCH_SIZE=16

# DataLoader工作进程数
NUM_WORKERS=4

# 设备
DEVICE="cuda"  # cuda 或 cpu

# =============================================================================
# 检查必需文件
# =============================================================================

if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: 找不到模型checkpoint: $CHECKPOINT"
    exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "错误: 找不到测试数据目录: $DATA_ROOT"
    exit 1
fi

# =============================================================================
# 启动推理分析
# =============================================================================

echo "======================================================================"
echo "启动推理分析"
echo "======================================================================"
echo "模型: $CHECKPOINT"
if [ -n "$BASELINE_CHECKPOINT" ]; then
    echo "基准模型: $BASELINE_CHECKPOINT"
fi
echo "测试数据: $DATA_ROOT"
echo "输出目录: $OUTPUT_DIR"
echo "======================================================================"
echo ""

# 构建命令
CMD="python inference_analysis.py \
    --checkpoint $CHECKPOINT \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --case_study_idx $CASE_STUDY_IDX \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --device $DEVICE"

# 添加基准模型（如果提供）
if [ -n "$BASELINE_CHECKPOINT" ]; then
    CMD="$CMD --baseline_checkpoint $BASELINE_CHECKPOINT"
fi

# 添加统计量文件（如果提供）
if [ -n "$STATS_FILE" ]; then
    CMD="$CMD --stats_file $STATS_FILE"
fi

# 显示完整命令
echo "执行命令:"
echo "$CMD"
echo ""

# 执行推理
eval $CMD

echo ""
echo "======================================================================"
echo "推理分析完成！"
echo "======================================================================"
echo ""
echo "生成的图表:"
echo "  1. $OUTPUT_DIR/vertical_rmse_profile.png - 垂直RMSE廓线图"
echo "  2. $OUTPUT_DIR/spatial_bias_map.png - 偏差空间分布图"
echo "  3. $OUTPUT_DIR/case_study.png - 个例可视化"
echo "  4. $OUTPUT_DIR/scatter_plot.png - 散点图对比"
echo "  5. $OUTPUT_DIR/correlation_heatmap.png - 相关系数热力图"
echo ""
echo "======================================================================"
