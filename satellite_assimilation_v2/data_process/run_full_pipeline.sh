#!/bin/bash
#===============================================================================
# 卫星数据同化 - 完整实验流程脚本 (增强版)
# Complete Experiment Pipeline with Data Preparation
#===============================================================================
#
# 功能:
#   1. 数据准备 (转换 + 划分)
#   2. 训练实验 (支持预划分数据集)
#   3. 消融实验
#   4. 结果汇总
#
# 用法:
#   chmod +x run_full_pipeline.sh
#   
#   # 完整流程 (数据准备 + 训练)
#   ./run_full_pipeline.sh full
#   
#   # 仅数据准备
#   ./run_full_pipeline.sh prepare
#   
#   # 使用预划分数据训练
#   ./run_full_pipeline.sh train
#
#===============================================================================

set -e  # 遇错退出

#===============================================================================
# 全局配置
#===============================================================================

# 实验名称前缀
EXP_PREFIX="FY3F_Assimilation"

# 数据路径配置
RAW_DATA_DIR="/data2/lrx/era_obs"              # 原始 .npy 数据目录
PREPARED_DATA_DIR="/data2/lrx/era_obs/npz"     # 准备后的 .npz 数据目录
STATS_FILE="${PREPARED_DATA_DIR}/stats.npz"     # 统计量文件

# 数据集划分配置
TRAIN_RATIO=0.8
VAL_RATIO=0.1
SEED=42

# 输出目录
OUTPUT_DIR="outputs"

# GPU配置
export CUDA_VISIBLE_DEVICES=0

# Python路径
PYTHON="python"

# 训练配置
EPOCHS=100
BATCH_SIZE=16
LR=0.0001

#===============================================================================
# 颜色输出
#===============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
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

log_step() {
    echo -e "${MAGENTA}[STEP]${NC} $1"
}

#===============================================================================
# 数据准备函数
#===============================================================================

prepare_data() {
    log_step "步骤 1: 数据准备"
    echo "========================================================================"
    echo "  原始数据: $RAW_DATA_DIR"
    echo "  输出目录: $PREPARED_DATA_DIR"
    echo "  训练比例: $TRAIN_RATIO"
    echo "  验证比例: $VAL_RATIO"
    echo "  随机种子: $SEED"
    echo "========================================================================"
    
    # 检查原始数据
    if [ ! -d "$RAW_DATA_DIR" ]; then
        log_error "原始数据目录不存在: $RAW_DATA_DIR"
        return 1
    fi
    
    # 运行数据准备脚本
    $PYTHON prepare_v2_data.py \
        --source_dir "$RAW_DATA_DIR" \
        --target_dir "$PREPARED_DATA_DIR" \
        --train_ratio $TRAIN_RATIO \
        --val_ratio $VAL_RATIO \
        --seed $SEED
    
    if [ $? -eq 0 ]; then
        log_success "数据准备完成!"
        
        # 验证输出
        if [ -d "${PREPARED_DATA_DIR}/train" ] && \
           [ -d "${PREPARED_DATA_DIR}/val" ] && \
           [ -d "${PREPARED_DATA_DIR}/test" ] && \
           [ -f "$STATS_FILE" ]; then
            log_success "数据集划分验证通过"
            
            # 显示统计信息
            train_count=$(find "${PREPARED_DATA_DIR}/train" -name "*.npz" | wc -l)
            val_count=$(find "${PREPARED_DATA_DIR}/val" -name "*.npz" | wc -l)
            test_count=$(find "${PREPARED_DATA_DIR}/test" -name "*.npz" | wc -l)
            
            echo ""
            log_info "数据集统计:"
            echo "  训练集: $train_count 文件"
            echo "  验证集: $val_count 文件"
            echo "  测试集: $test_count 文件"
            echo "  统计量: $STATS_FILE"
            
            return 0
        else
            log_error "数据集划分验证失败!"
            return 1
        fi
    else
        log_error "数据准备失败!"
        return 1
    fi
}

#===============================================================================
# 训练函数 (使用预划分数据)
#===============================================================================

run_single_experiment_with_split() {
    local model=$1
    local fusion=$2
    local use_aux=$3
    local mask_aware=$4
    
    # 生成实验ID
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local exp_id="${EXP_PREFIX}_${model}_${fusion}_aux${use_aux}_mask${mask_aware}_${timestamp}"
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
    echo "  数据目录: $PREPARED_DATA_DIR"
    echo "  日志目录: $log_dir"
    echo "------------------------------------------------------------------------"
    
    # 检查数据集
    if [ ! -d "${PREPARED_DATA_DIR}/train" ]; then
        log_error "预划分的训练集不存在: ${PREPARED_DATA_DIR}/train"
        log_info "请先运行: ./run_full_pipeline.sh prepare"
        return 1
    fi
    
    # 构建训练命令
    local cmd="$PYTHON train_with_split.py \
        --exp_name \"$exp_id\" \
        --output_dir \"$OUTPUT_DIR\" \
        --data_root \"$PREPARED_DATA_DIR\" \
        --model \"$model\" \
        --fusion_mode \"$fusion\" \
        --use_aux \"$use_aux\" \
        --mask_aware \"$mask_aware\" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --loss combined \
        --tensorboard true \
        --seed $SEED"
    
    # 添加统计量文件
    if [ -f "$STATS_FILE" ]; then
        cmd="$cmd --stats_file \"$STATS_FILE\""
    fi
    
    # 执行训练
    log_info "开始训练..."
    echo ""
    
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
# 快速消融实验
#===============================================================================

run_quick_ablation_with_split() {
    log_step "步骤 2: 快速消融实验 (使用预划分数据)"
    
    # 初始化状态文件
    echo "experiment_id,status" > "$OUTPUT_DIR/experiment_status.csv"
    
    local experiments=(
        # 基线: Vanilla U-Net
        "vanilla_unet concat false false"
        # 创新1: + 逐通道标准化
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
        if run_single_experiment_with_split "$model" "$fusion" "$use_aux" "$mask_aware"; then
            ((success++))
        else
            ((failed++))
        fi
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
# 完整流程
#===============================================================================

run_full_pipeline() {
    log_info "开始完整实验流程"
    echo "========================================================================"
    echo "  1. 数据准备"
    echo "  2. 消融实验"
    echo "  3. 结果汇总"
    echo "========================================================================"
    echo ""
    
    # 步骤1: 数据准备
    if prepare_data; then
        log_success "步骤1完成"
    else
        log_error "步骤1失败,中止流程"
        return 1
    fi
    
    echo ""
    sleep 2
    
    # 步骤2: 训练实验
    if run_quick_ablation_with_split; then
        log_success "步骤2完成"
    else
        log_warning "步骤2部分失败"
    fi
    
    echo ""
    sleep 2
    
    # 步骤3: 结果汇总
    log_step "步骤 3: 结果汇总"
    summarize_results
    
    echo ""
    log_success "完整流程执行完成!"
}

#===============================================================================
# 结果汇总
#===============================================================================

summarize_results() {
    local summary_file="$OUTPUT_DIR/results_summary.md"
    
    cat > "$summary_file" << EOF
# 卫星数据同化实验结果汇总

生成时间: $(date)

## 数据集信息

| 项目 | 值 |
|------|-----|
| 数据目录 | $PREPARED_DATA_DIR |
| 训练集 | $(find "${PREPARED_DATA_DIR}/train" -name "*.npz" 2>/dev/null | wc -l) 样本 |
| 验证集 | $(find "${PREPARED_DATA_DIR}/val" -name "*.npz" 2>/dev/null | wc -l) 样本 |
| 测试集 | $(find "${PREPARED_DATA_DIR}/test" -name "*.npz" 2>/dev/null | wc -l) 样本 |
| 随机种子 | $SEED |

## 训练配置

| 参数 | 值 |
|------|-----|
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
    find "$OUTPUT_DIR" -name "best_model.pth" -type f 2>/dev/null | while read -r model_path; do
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
卫星数据同化 - 完整实验流程脚本

用法:
    ./run_full_pipeline.sh [命令]

命令:
    full            运行完整流程 (数据准备 + 训练 + 汇总)
    prepare         仅执行数据准备 (转换 + 划分)
    train           使用预划分数据执行训练
    quick           快速消融实验 (5个核心对比实验)
    summary         汇总实验结果
    help            显示此帮助信息

示例:
    # 第一次运行 - 完整流程
    ./run_full_pipeline.sh full
    
    # 仅准备数据
    ./run_full_pipeline.sh prepare
    
    # 使用已准备的数据训练
    ./run_full_pipeline.sh train
    
    # 汇总结果
    ./run_full_pipeline.sh summary

配置:
    请在脚本顶部的"全局配置"区域修改以下参数:
    - RAW_DATA_DIR: 原始 .npy 数据目录
    - PREPARED_DATA_DIR: 准备后的 .npz 数据目录
    - TRAIN_RATIO, VAL_RATIO: 数据集划分比例
    - SEED: 随机种子 (确保可重复性)

注意事项:
    1. 数据准备和训练使用相同的随机种子,确保数据划分一致
    2. 预划分的数据集结构:
       npz/
       ├── train/       (训练集)
       ├── val/         (验证集)
       ├── test/        (测试集)
       ├── stats.npz    (统计量)
       └── dataset_split.json  (划分信息)
    3. 如果修改了数据划分参数,需要重新运行 prepare 步骤

EOF
}

#===============================================================================
# 主程序
#===============================================================================

main() {
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    # 打印横幅
    echo ""
    echo "========================================================================"
    echo "  卫星数据同化 - 完整实验流程"
    echo "  Satellite Data Assimilation - Complete Pipeline"
    echo "========================================================================"
    echo ""
    
    # 解析命令
    case "${1:-help}" in
        full)
            run_full_pipeline
            ;;
        prepare)
            prepare_data
            ;;
        train)
            run_quick_ablation_with_split
            ;;
        quick)
            # 检查数据是否已准备
            if [ ! -d "${PREPARED_DATA_DIR}/train" ]; then
                log_warning "数据未准备,自动执行数据准备..."
                prepare_data || exit 1
                echo ""
            fi
            run_quick_ablation_with_split
            ;;
        summary)
            summarize_results
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
    
    echo ""
}

# 运行
main "$@"
