#!/bin/bash
#===============================================================================
# 卫星数据同化实验自动化脚本
# Satellite Data Assimilation Experiment Automation Script
#===============================================================================
#
# 功能:
#   1. 自动生成实验ID
#   2. 消融实验循环
#   3. 断点续训支持
#   4. 实验结果汇总
#
# 用法:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
#
#===============================================================================

set -e  # 遇错退出

#===============================================================================
# 配置区
#===============================================================================

# 实验名称前缀
EXP_PREFIX="FY3F_Assimilation"

# 数据路径 (请修改为实际路径)
DATA_ROOT="/data2/lrx/era_obs"
STATS_FILE="/data2/lrx/era_obs/stats.npz"

# 输出目录
OUTPUT_DIR="outputs"

# GPU配置
export CUDA_VISIBLE_DEVICES=0

# Python路径 (如有必要)
PYTHON="python"

#===============================================================================
# 消融实验参数
#===============================================================================

# 模型类型
MODELS=("physics_unet" "vanilla_unet")

# 融合模式
FUSION_MODES=("gated" "concat" "add")

# 是否使用辅助特征
USE_AUX_FLAGS=("true" "false")

# 是否使用掩码感知
MASK_AWARE_FLAGS=("true" "false")

# 训练配置
EPOCHS=100
BATCH_SIZE=16
LR=0.0001

#===============================================================================
# 辅助函数
#===============================================================================

# 生成时间戳
timestamp() {
    date +"%Y%m%d_%H%M%S"
}

# 彩色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查上一条命令是否成功
check_status() {
    if [ $? -eq 0 ]; then
        log_success "$1 完成"
        return 0
    else
        log_error "$1 失败"
        return 1
    fi
}

#===============================================================================
# 实验函数
#===============================================================================

run_single_experiment() {
    local model=$1
    local fusion=$2
    local use_aux=$3
    local mask_aware=$4
    
    # 生成实验ID
    local exp_id="${EXP_PREFIX}_${model}_${fusion}_aux${use_aux}_mask${mask_aware}_$(timestamp)"
    local log_dir="${OUTPUT_DIR}/${exp_id}"
    
    mkdir -p "$log_dir"
    
    echo ""
    echo "========================================================================"
    echo "实验: $exp_id"
    echo "========================================================================"
    echo "  模型: $model"
    echo "  融合模式: $fusion"
    echo "  辅助特征: $use_aux"
    echo "  掩码感知: $mask_aware"
    echo "  日志目录: $log_dir"
    echo "------------------------------------------------------------------------"
    
    # 构建命令
    local cmd="$PYTHON train.py \
        --exp_name \"$exp_id\" \
        --output_dir \"$OUTPUT_DIR\" \
        --data_root \"$DATA_ROOT\" \
        --model \"$model\" \
        --fusion_mode \"$fusion\" \
        --use_aux \"$use_aux\" \
        --mask_aware \"$mask_aware\" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --loss combined \
        --tensorboard true"
    
    # 如果有统计量文件
    if [ -f "$STATS_FILE" ]; then
        cmd="$cmd --stats_file \"$STATS_FILE\""
    fi
    
    # 执行训练
    echo "执行: $cmd"
    echo ""
    
    # 运行并记录日志
    eval "$cmd" 2>&1 | tee "$log_dir/train.log"
    
    # 检查结果
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "实验 $exp_id 完成"
        echo "$exp_id,success" >> "$OUTPUT_DIR/experiment_status.csv"
        return 0
    else
        log_error "实验 $exp_id 失败"
        echo "$exp_id,failed" >> "$OUTPUT_DIR/experiment_status.csv"
        return 1
    fi
}

#===============================================================================
# 消融实验
#===============================================================================

run_ablation_study() {
    log_info "开始消融实验..."
    
    # 初始化状态文件
    echo "experiment_id,status" > "$OUTPUT_DIR/experiment_status.csv"
    
    local total=0
    local success=0
    local failed=0
    
    # 主循环: 物理感知模型的消融
    for model in "${MODELS[@]}"; do
        if [ "$model" == "vanilla_unet" ]; then
            # Vanilla U-Net只需要运行一次作为基线
            run_single_experiment "$model" "concat" "false" "false" && ((success++)) || ((failed++))
            ((total++))
        else
            # 物理感知模型的消融实验
            for fusion in "${FUSION_MODES[@]}"; do
                for use_aux in "${USE_AUX_FLAGS[@]}"; do
                    for mask_aware in "${MASK_AWARE_FLAGS[@]}"; do
                        run_single_experiment "$model" "$fusion" "$use_aux" "$mask_aware" \
                            && ((success++)) || ((failed++))
                        ((total++))
                    done
                done
            done
        fi
    done
    
    # 汇总
    echo ""
    echo "========================================================================"
    echo "消融实验完成"
    echo "========================================================================"
    echo "  总实验数: $total"
    echo "  成功: $success"
    echo "  失败: $failed"
    echo "  成功率: $(echo "scale=2; $success*100/$total" | bc)%"
    echo "========================================================================"
}

#===============================================================================
# 快速消融实验 (核心对比)
#===============================================================================

run_quick_ablation() {
    log_info "开始快速消融实验 (核心对比)..."
    
    echo "experiment_id,status" > "$OUTPUT_DIR/experiment_status.csv"
    
    local experiments=(
        # 基线: Vanilla U-Net
        "vanilla_unet concat false false"
        # 创新1: + 逐通道标准化 (通过physics_unet实现)
        "physics_unet add false false"
        # 创新2: + SpectralAdapter
        "physics_unet gated false false"
        # 创新3: + 辅助特征
        "physics_unet gated true false"
        # 创新4: + 掩码感知 (完整模型)
        "physics_unet gated true true"
    )
    
    local total=${#experiments[@]}
    local success=0
    local failed=0
    
    for exp in "${experiments[@]}"; do
        read -r model fusion use_aux mask_aware <<< "$exp"
        run_single_experiment "$model" "$fusion" "$use_aux" "$mask_aware" \
            && ((success++)) || ((failed++))
    done
    
    echo ""
    echo "========================================================================"
    echo "快速消融实验完成"
    echo "========================================================================"
    echo "  总实验数: $total"
    echo "  成功: $success"
    echo "  失败: $failed"
    echo "========================================================================"
}

#===============================================================================
# 单次实验
#===============================================================================

run_single() {
    local model=${1:-"physics_unet"}
    local fusion=${2:-"gated"}
    local use_aux=${3:-"true"}
    local mask_aware=${4:-"true"}
    
    run_single_experiment "$model" "$fusion" "$use_aux" "$mask_aware"
}

#===============================================================================
# 断点续训
#===============================================================================

resume_experiment() {
    local checkpoint=$1
    local exp_name=$2
    
    if [ ! -f "$checkpoint" ]; then
        log_error "检查点不存在: $checkpoint"
        return 1
    fi
    
    log_info "恢复训练: $checkpoint"
    
    $PYTHON train.py \
        --exp_name "$exp_name" \
        --output_dir "$OUTPUT_DIR" \
        --data_root "$DATA_ROOT" \
        --resume "$checkpoint" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        2>&1 | tee "${OUTPUT_DIR}/${exp_name}/resume.log"
}

#===============================================================================
# 结果汇总
#===============================================================================

summarize_results() {
    log_info "汇总实验结果..."
    
    local summary_file="$OUTPUT_DIR/results_summary.md"
    
    cat > "$summary_file" << 'EOF'
# 实验结果汇总

生成时间: $(date)

## 实验配置

| 参数 | 值 |
|------|-----|
| 数据目录 | $DATA_ROOT |
| 训练轮数 | $EPOCHS |
| 批大小 | $BATCH_SIZE |
| 学习率 | $LR |

## 实验状态

EOF
    
    if [ -f "$OUTPUT_DIR/experiment_status.csv" ]; then
        echo "| 实验ID | 状态 |" >> "$summary_file"
        echo "|--------|------|" >> "$summary_file"
        tail -n +2 "$OUTPUT_DIR/experiment_status.csv" | while IFS=, read -r exp_id status; do
            echo "| $exp_id | $status |" >> "$summary_file"
        done
    fi
    
    echo "" >> "$summary_file"
    echo "## 最佳模型" >> "$summary_file"
    echo "" >> "$summary_file"
    
    # 查找所有best_model.pth
    find "$OUTPUT_DIR" -name "best_model.pth" -type f | while read -r model_path; do
        exp_dir=$(dirname "$model_path")
        echo "- $exp_dir" >> "$summary_file"
    done
    
    log_success "结果汇总已保存: $summary_file"
}

#===============================================================================
# 帮助信息
#===============================================================================

show_help() {
    cat << EOF
卫星数据同化实验自动化脚本

用法:
    ./run_experiments.sh [命令] [参数...]

命令:
    ablation        运行完整消融实验
    quick           运行快速消融实验 (核心对比)
    single          运行单次实验
    resume          断点续训
    summary         汇总实验结果
    help            显示帮助

示例:
    # 运行快速消融实验
    ./run_experiments.sh quick
    
    # 运行单次实验
    ./run_experiments.sh single physics_unet gated true true
    
    # 断点续训
    ./run_experiments.sh resume outputs/exp_xxx/checkpoint.pth exp_xxx
    
    # 汇总结果
    ./run_experiments.sh summary

配置:
    请修改脚本顶部的配置区，设置正确的数据路径和GPU配置。

EOF
}

#===============================================================================
# 主程序
#===============================================================================

main() {
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    # 解析命令
    case "${1:-help}" in
        ablation)
            run_ablation_study
            ;;
        quick)
            run_quick_ablation
            ;;
        single)
            shift
            run_single "$@"
            ;;
        resume)
            shift
            resume_experiment "$@"
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

# 运行
main "$@"
