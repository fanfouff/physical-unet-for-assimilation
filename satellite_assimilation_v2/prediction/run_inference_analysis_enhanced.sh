#!/bin/bash
##############################################################################
# 增强版推理分析启动脚本 - IEEE TGRS 风格图表
# Enhanced Inference Analysis Script - IEEE TGRS Style Figures
# 
# 用法: bash run_inference_analysis_enhanced.sh
#
# 功能:
#   - 在测试集上运行训练好的模型
#   - 生成 IEEE TGRS 标准的5类图表
#   - 支持消融实验和对比实验
#   - 自动生成LaTeX表格
#   - 同时输出 PDF 和 PNG 格式
##############################################################################

# =============================================================================
# 配置区域 (请根据实际路径修改)
# =============================================================================

# --- 必需参数 ---

# 主模型checkpoint路径 (Ours - 最佳模型)
CHECKPOINT="/home/seu/Fuxi/Unet/satellite_assimilation_v2/train_ddp/outputs/FY3F_Assimilation_physics_unet_gated_auxtrue_maskfalse_2gpu_20260128_201638/best_model.pth"

# 测试数据路径
DATA_ROOT="/data2/lrx/era_obs/npz/test"

# --- 可选参数 ---

# 基准模型checkpoint (用于对比实验)
BASELINE_CHECKPOINT="/home/seu/Fuxi/Unet/satellite_assimilation_v2/experiments/outputs/experiments/20260126_vanilla_unet/checkpoint_best.pth"

# 消融实验checkpoints (可以添加多个)
ABLATION_CHECKPOINTS=(
    # "/home/seu/Fuxi/Unet/satellite_assimilation_v2/experiments/outputs/experiments/20260126_no_aux/checkpoint_best.pth"
    # "/home/seu/Fuxi/Unet/satellite_assimilation_v2/experiments/outputs/experiments/20260126_no_mask/checkpoint_best.pth"
    # "/home/seu/Fuxi/Unet/satellite_assimilation_v2/experiments/outputs/experiments/20260126_no_physics/checkpoint_best.pth"
)

# 统计量文件（可选，留空则自动计算）
STATS_FILE=""

# 输出目录
OUTPUT_DIR="figures_v2/A2"

# 个例可视化的样本索引
CASE_STUDY_IDX=0

# 空间分析使用的层级索引 (对应500hPa和850hPa)
# 注意: 需要根据实际的气压层配置调整
SPATIAL_LEVELS="15 25"

# --- 运行参数 ---

# 批大小
BATCH_SIZE=16

# DataLoader工作进程数
NUM_WORKERS=4

# 设备
DEVICE="cuda"  # cuda 或 cpu

# 是否保存PDF格式 (推荐开启，用于论文投稿)
SAVE_PDF=true

# 是否生成LaTeX表格
SAVE_LATEX_TABLE=true

# =============================================================================
# 检查必需文件
# =============================================================================

echo "======================================================================"
echo "增强版推理分析 - IEEE TGRS 风格"
echo "======================================================================"
echo ""

# 检查主模型
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: 找不到主模型checkpoint: $CHECKPOINT"
    echo ""
    echo "提示: 请修改脚本中的 CHECKPOINT 变量为实际的模型路径"
    echo "示例路径: /home/seu/Fuxi/Unet/satellite_assimilation_v2/experiments/outputs/experiments/20260126_XXX/checkpoint_best.pth"
    exit 1
fi

# 检查测试数据
if [ ! -d "$DATA_ROOT" ]; then
    echo "错误: 找不到测试数据目录: $DATA_ROOT"
    echo ""
    echo "提示: 请修改脚本中的 DATA_ROOT 变量为实际的测试数据路径"
    exit 1
fi

# 检查基准模型（如果提供）
if [ -n "$BASELINE_CHECKPOINT" ] && [ ! -f "$BASELINE_CHECKPOINT" ]; then
    echo "警告: 找不到基准模型: $BASELINE_CHECKPOINT"
    echo "将跳过对比实验..."
    BASELINE_CHECKPOINT=""
fi

# =============================================================================
# 显示配置
# =============================================================================

echo "配置信息:"
echo "  主模型:     $CHECKPOINT"
if [ -n "$BASELINE_CHECKPOINT" ]; then
    echo "  基准模型:   $BASELINE_CHECKPOINT"
else
    echo "  基准模型:   未提供 (将跳过对比实验)"
fi

if [ ${#ABLATION_CHECKPOINTS[@]} -gt 0 ]; then
    echo "  消融模型:   ${#ABLATION_CHECKPOINTS[@]} 个"
    for i in "${!ABLATION_CHECKPOINTS[@]}"; do
        echo "    [$((i+1))] ${ABLATION_CHECKPOINTS[$i]}"
    done
else
    echo "  消融模型:   未提供 (将跳过消融实验)"
fi

echo "  测试数据:   $DATA_ROOT"
echo "  输出目录:   $OUTPUT_DIR"
echo "  批大小:     $BATCH_SIZE"
echo "  设备:       $DEVICE"
echo "  保存PDF:    $SAVE_PDF"
echo "  生成LaTeX:  $SAVE_LATEX_TABLE"
echo ""

# 统计测试文件数量
NUM_TEST_FILES=$(find "$DATA_ROOT" -name "*.npz" | wc -l)
echo "发现测试文件: $NUM_TEST_FILES 个"
echo ""

# =============================================================================
# 构建命令
# =============================================================================

CMD="python inference_analysis_enhanced.py \
    --checkpoint $CHECKPOINT \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --case_study_idx $CASE_STUDY_IDX \
    --spatial_levels $SPATIAL_LEVELS"

# 添加基准模型（如果提供）
if [ -n "$BASELINE_CHECKPOINT" ]; then
    CMD="$CMD --baseline_checkpoint $BASELINE_CHECKPOINT"
fi

# 添加消融模型（如果提供）
if [ ${#ABLATION_CHECKPOINTS[@]} -gt 0 ]; then
    CMD="$CMD --ablation_checkpoints"
    for ckpt in "${ABLATION_CHECKPOINTS[@]}"; do
        CMD="$CMD $ckpt"
    done
fi

# 添加统计量文件（如果提供）
if [ -n "$STATS_FILE" ]; then
    CMD="$CMD --stats_file $STATS_FILE"
fi

# 添加输出格式选项
if [ "$SAVE_PDF" = true ]; then
    CMD="$CMD --save_pdf"
fi

if [ "$SAVE_LATEX_TABLE" = true ]; then
    CMD="$CMD --save_latex_table"
fi

# =============================================================================
# 执行推理分析
# =============================================================================

echo "======================================================================"
echo "执行命令:"
echo "$CMD"
echo "======================================================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 执行
eval $CMD

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "======================================================================"
echo "分析完成!"
echo "======================================================================"
echo ""
echo "耗时: ${DURATION} 秒"
echo ""
echo "生成的图表目录: $OUTPUT_DIR"
echo ""
echo "图表说明:"
echo ""
echo "  [图组 A] 垂直廓线对比 (Vertical Profiles):"
echo "    - vertical_profile_ablation.png     消融实验RMSE/Bias廓线"
echo "    - vertical_profile_comparison.png   对比实验RMSE/Bias廓线"
echo ""
echo "  [图组 B] 综合指标柱状图 (Bar Charts):"
echo "    - bar_chart_ablation.png            消融实验Global RMSE/Corr"
echo "    - bar_chart_comparison.png          对比实验Global RMSE/Corr"
echo ""
echo "  [图组 C] 空间分布分析 (Spatial Analysis):"
echo "    - spatial_analysis_500hPa.png       500hPa偏差分布与改进图"
echo "    - spatial_analysis_850hPa.png       850hPa偏差分布与改进图"
echo ""
echo "  [图组 D] 泰勒图 (Taylor Diagram):"
echo "    - taylor_diagram.png                多模型性能综合对比"
echo ""

if [ "$SAVE_PDF" = true ]; then
    echo "  注: 所有图表同时保存了 PNG 和 PDF 格式"
    echo ""
fi

if [ "$SAVE_LATEX_TABLE" = true ]; then
    echo "  LaTeX 表格:"
    echo "    - ablation_table.tex                消融实验结果表格"
    echo "    - comparison_table.tex              方法对比结果表格"
    echo ""
fi

echo "使用建议:"
echo "  1. 用于论文投稿时，推荐使用 PDF 格式 (矢量图，质量更高)"
echo "  2. 用于预览或演示时，使用 PNG 格式即可"
echo "  3. LaTeX 表格可以直接复制到论文中使用"
echo ""
echo "======================================================================"
echo ""
echo "下一步:"
echo "  1. 检查图表质量: ls -lh $OUTPUT_DIR"
echo "  2. 调整参数后重新运行: bash run_inference_analysis_enhanced.sh"
echo "  3. 如需修改消融实验配置，请编辑脚本中的 ABLATION_CHECKPOINTS 数组"
echo ""
echo "======================================================================"
