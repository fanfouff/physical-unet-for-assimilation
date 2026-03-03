#!/bin/bash
#===============================================================================
# PAS-Net 消融实验 & 对比实验 完整脚本
# Ablation Study & Comparison Experiments for PAS-Net
#===============================================================================
#
# 实验组:
#
# 【消融实验 - Ablation Study】
# A1. Full PAS-Net (完整模型)
# A2. w/o Level-wise Norm (无逐层标准化 -> 使用全局标准化)
# A3. w/o Spectral Adapter (无光谱适配器 -> 直接拼接)
# A4. w/o Gradient Loss (无梯度损失)
# A5. w/o Auxiliary Features (无辅助特征)
# A6. w/o Mask-Aware (无掩码感知)
# A7. w/o SE Block (无SE注意力)
# A8. Fusion: concat (拼接融合)
# A9. Fusion: add (加法融合)
#
# 【对比实验 - Comparison】
# C1. Vanilla U-Net (标准U-Net基线)
# C2. ResUNet (残差U-Net)
# C3. Attention U-Net (注意力U-Net)
#
#===============================================================================

set -e

#===============================================================================
# 配置区 (请根据实际情况修改)
#===============================================================================

# 数据路径
DATA_ROOT="/data2/lrx/era_obs/npz/train"
STATS_FILE="/data2/lrx/era_obs/npz/stats.npz"

# 输出目录
OUTPUT_DIR="outputs/experiments"
RESULTS_DIR="outputs/results"

# GPU配置
GPUS="0,1"  # 多卡: "0,1,2,3", 单卡: "0"
N_GPUS=2

# 训练配置
EPOCHS=100
BATCH_SIZE=16  # 每GPU
LR=0.0001
SEED=42

# Python路径
PYTHON="python"

# 日期戳
TIMESTAMP=$(date +"%Y%m%d")

#===============================================================================
# 彩色输出
#===============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_exp()     { echo -e "${MAGENTA}[EXPERIMENT]${NC} $1"; }

#===============================================================================
# 实验运行函数
#===============================================================================

run_experiment() {
    local exp_id=$1
    local exp_name=$2
    local model_type=$3
    local fusion_mode=$4
    local use_aux=$5
    local mask_aware=$6
    local use_adapter=$7
    local loss_type=$8
    local extra_args=${9:-""}
    
    local full_exp_name="${TIMESTAMP}_${exp_id}_${exp_name}"
    local log_dir="${OUTPUT_DIR}/${full_exp_name}"
    
    mkdir -p "$log_dir"
    
    echo ""
    echo "========================================================================"
    log_exp "实验: $exp_id - $exp_name"
    echo "========================================================================"
    echo "  模型类型: $model_type"
    echo "  融合模式: $fusion_mode"
    echo "  辅助特征: $use_aux"
    echo "  掩码感知: $mask_aware"
    echo "  光谱适配器: $use_adapter"
    echo "  损失函数: $loss_type"
    echo "  日志目录: $log_dir"
    echo "------------------------------------------------------------------------"
    
    # 构建命令
    local cmd
    if [ "$N_GPUS" -gt 1 ]; then
        cmd="CUDA_VISIBLE_DEVICES=$GPUS torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=$N_GPUS \
            train_experiment.py"
    else
        cmd="CUDA_VISIBLE_DEVICES=$GPUS $PYTHON train_experiment.py"
    fi
    
    cmd="$cmd \
        --exp_name \"$full_exp_name\" \
        --output_dir \"$OUTPUT_DIR\" \
        --data_root \"$DATA_ROOT\" \
        --model \"$model_type\" \
        --fusion_mode \"$fusion_mode\" \
        --use_aux \"$use_aux\" \
        --mask_aware \"$mask_aware\" \
        --use_spectral_adapter \"$use_adapter\" \
        --loss \"$loss_type\" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --seed $SEED \
        --tensorboard true"
    
    if [ -f "$STATS_FILE" ]; then
        cmd="$cmd --stats_file \"$STATS_FILE\""
    fi
    
    if [ -n "$extra_args" ]; then
        cmd="$cmd $extra_args"
    fi
    
    echo "执行命令: $cmd"
    echo ""
    
    # 执行
    eval "$cmd" 2>&1 | tee "$log_dir/train.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "实验 $exp_id 完成!"
        echo "$exp_id,$exp_name,success" >> "$RESULTS_DIR/experiment_status.csv"
        return 0
    else
        log_error "实验 $exp_id 失败!"
        echo "$exp_id,$exp_name,failed" >> "$RESULTS_DIR/experiment_status.csv"
        return 1
    fi
}

#===============================================================================
# 消融实验
#===============================================================================

run_ablation_study() {
    log_info "开始消融实验..."
    
    mkdir -p "$RESULTS_DIR"
    echo "exp_id,exp_name,status" > "$RESULTS_DIR/experiment_status.csv"
    
    # A1. Full PAS-Net (完整模型 - Baseline)
    run_experiment "A1" "PASNet_Full" \
        "pasnet" "gated" "true" "true" "true" "hybrid"
    
    # A2. w/o Level-wise Norm (通过修改标准化策略实现)
    run_experiment "A2" "PASNet_NoLevelNorm" \
        "pasnet" "gated" "true" "true" "true" "hybrid" \
        "--norm_mode global"
    
    # A3. w/o Spectral Adapter
    run_experiment "A3" "PASNet_NoAdapter" \
        "pasnet" "gated" "true" "true" "false" "hybrid"
    
    # A4. w/o Gradient Loss
    run_experiment "A4" "PASNet_NoGradLoss" \
        "pasnet" "gated" "true" "true" "true" "mse"
    
    # A5. w/o Auxiliary Features
    run_experiment "A5" "PASNet_NoAux" \
        "pasnet" "gated" "false" "true" "true" "hybrid"
    
    # A6. w/o Mask-Aware
    run_experiment "A6" "PASNet_NoMask" \
        "pasnet" "gated" "true" "false" "true" "hybrid"
    
    # A7. w/o SE Block
    run_experiment "A7" "PASNet_NoSE" \
        "pasnet_no_se" "gated" "true" "true" "true" "hybrid"
    
    # A8. Fusion: concat
    run_experiment "A8" "PASNet_FusionConcat" \
        "pasnet" "concat" "true" "true" "true" "hybrid"
    
    # A9. Fusion: add
    run_experiment "A9" "PASNet_FusionAdd" \
        "pasnet" "add" "true" "true" "true" "hybrid"
    
    log_success "消融实验完成!"
}

#===============================================================================
# 对比实验
#===============================================================================

run_comparison_study() {
    log_info "开始对比实验..."
    
    mkdir -p "$RESULTS_DIR"
    
    # C1. Vanilla U-Net (基线)
    run_experiment "C1" "VanillaUNet" \
        "vanilla_unet" "none" "false" "false" "false" "mse"
    
    # C2. ResUNet
    run_experiment "C2" "ResUNet" \
        "res_unet" "none" "false" "false" "false" "mse"
    
    # C3. Attention U-Net
    run_experiment "C3" "AttentionUNet" \
        "attention_unet" "none" "false" "false" "false" "mse"
    
    log_success "对比实验完成!"
}

#===============================================================================
# 快速测试 (用于调试)
#===============================================================================

run_quick_test() {
    log_info "运行快速测试 (5 epochs)..."
    
    local orig_epochs=$EPOCHS
    EPOCHS=5
    
    run_experiment "TEST" "QuickTest" \
        "pasnet" "gated" "true" "true" "true" "hybrid"
    
    EPOCHS=$orig_epochs
    
    log_success "快速测试完成!"
}

#===============================================================================
# 结果汇总
#===============================================================================

summarize_results() {
    log_info "汇总实验结果..."
    
    local summary_file="$RESULTS_DIR/summary_${TIMESTAMP}.md"
    
    cat > "$summary_file" << EOF
# PAS-Net 实验结果汇总

生成时间: $(date)

## 实验配置

| 参数 | 值 |
|------|-----|
| 数据目录 | $DATA_ROOT |
| 训练轮数 | $EPOCHS |
| 批大小 (每GPU) | $BATCH_SIZE |
| 学习率 | $LR |
| GPU | $GPUS |

## 消融实验结果

| 实验ID | 配置 | 全局RMSE | 对流层RMSE | 平流层RMSE |
|--------|------|----------|------------|------------|
| A1 | Full PAS-Net | - | - | - |
| A2 | w/o Level-wise Norm | - | - | - |
| A3 | w/o Spectral Adapter | - | - | - |
| A4 | w/o Gradient Loss | - | - | - |
| A5 | w/o Auxiliary Features | - | - | - |
| A6 | w/o Mask-Aware | - | - | - |
| A7 | w/o SE Block | - | - | - |
| A8 | Fusion: concat | - | - | - |
| A9 | Fusion: add | - | - | - |

## 对比实验结果

| 方法 | 全局RMSE | 对流层RMSE | 平流层RMSE | 参数量 |
|------|----------|------------|------------|--------|
| Vanilla U-Net | - | - | - | - |
| ResUNet | - | - | - | - |
| Attention U-Net | - | - | - | - |
| **PAS-Net (Ours)** | - | - | - | - |

## 备注

请使用 \`evaluate_all.py\` 脚本填充上述结果表格。

EOF
    
    log_success "结果汇总已保存: $summary_file"
}

#===============================================================================
# 帮助信息
#===============================================================================

show_help() {
    cat << EOF
PAS-Net 实验运行脚本

用法:
    ./run_pasnet_experiments.sh [命令]

命令:
    ablation        运行完整消融实验 (A1-A9)
    comparison      运行对比实验 (C1-C3)
    all             运行所有实验
    quick           快速测试 (5 epochs)
    summary         生成结果汇总
    help            显示帮助

示例:
    # 运行消融实验
    ./run_pasnet_experiments.sh ablation
    
    # 运行所有实验
    ./run_pasnet_experiments.sh all
    
    # 快速测试
    ./run_pasnet_experiments.sh quick

配置:
    请修改脚本顶部的配置区设置数据路径、GPU等参数。

EOF
}

#===============================================================================
# 主程序
#===============================================================================

main() {
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$RESULTS_DIR"
    
    case "${1:-help}" in
        ablation)
            run_ablation_study
            summarize_results
            ;;
        comparison)
            run_comparison_study
            summarize_results
            ;;
        all)
            run_ablation_study
            run_comparison_study
            summarize_results
            ;;
        quick)
            run_quick_test
            ;;
        summary)
            summarize_results
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
