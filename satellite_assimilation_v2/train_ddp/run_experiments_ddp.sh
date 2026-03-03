#!/bin/bash
#===============================================================================
# 卫星数据同化实验自动化脚本 - 多GPU版
# Satellite Data Assimilation Experiment Automation Script - Multi-GPU Version
#===============================================================================
#
# 功能:
#   1. 支持单卡/多卡训练自动切换
#   2. 自动检测可用GPU
#   3. 消融实验循环
#   4. 断点续训支持
#   5. 实验结果汇总
#
# 用法:
#   chmod +x run_experiments_ddp.sh
#   ./run_experiments_ddp.sh [命令] [参数...]
#
# 多卡示例:
#   ./run_experiments_ddp.sh single_ddp --gpus 2,3
#   ./run_experiments_ddp.sh quick_ddp --gpus 0,1,2,3
#
#===============================================================================

set -e  # 遇错退出

#===============================================================================
# 配置区
#===============================================================================

# 实验名称前缀
EXP_PREFIX="FY3F_Assimilation"

# 数据路径 (请修改为实际路径)
DATA_ROOT="/data2/lrx/era_obs/npz/train"
STATS_FILE="/data2/lrx/era_obs/npz/stats.npz"

# 输出目录
OUTPUT_DIR="outputs"

# 默认GPU配置
DEFAULT_GPUS="3"  # 根据nvidia-smi，GPU 3空闲

# Python路径
PYTHON="python"

# DDP启动方式: torchrun (推荐) 或 launch
DDP_LAUNCHER="torchrun"

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
BATCH_SIZE=16  # 每GPU的batch_size
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

log_gpu() {
    echo -e "${CYAN}[GPU]${NC} $1"
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

# 获取可用GPU列表
get_available_gpus() {
    # 返回显存使用低于1GB的GPU
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F',' '$2 < 1000 {print $1}' | tr '\n' ',' | sed 's/,$//'
}

# 计算GPU数量
count_gpus() {
    local gpus=$1
    echo "$gpus" | tr ',' '\n' | wc -l
}

# 显示GPU状态
show_gpu_status() {
    echo ""
    echo "========================================================================"
    echo "GPU 状态"
    echo "========================================================================"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader | \
    while IFS=',' read -r idx name mem_used mem_total util; do
        echo "  GPU $idx: $name | 显存: $mem_used/$mem_total MiB | 利用率: $util"
    done
    echo ""
    
    local available=$(get_available_gpus)
    if [ -n "$available" ]; then
        log_gpu "可用GPU (显存<1GB): $available"
    else
        log_warning "没有检测到空闲GPU"
    fi
    echo "========================================================================"
}

#===============================================================================
# 单卡训练函数 (使用原始 train.py)
#===============================================================================

run_single_experiment() {
    local model=$1
    local fusion=$2
    local use_aux=$3
    local mask_aware=$4
    local gpu=${5:-$DEFAULT_GPUS}
    
    # 生成实验ID
    local exp_id="${EXP_PREFIX}_${model}_${fusion}_aux${use_aux}_mask${mask_aware}_$(timestamp)"
    local log_dir="${OUTPUT_DIR}/${exp_id}"
    
    mkdir -p "$log_dir"
    
    echo ""
    echo "========================================================================"
    echo "实验: $exp_id (单卡模式)"
    echo "========================================================================"
    echo "  模型: $model"
    echo "  融合模式: $fusion"
    echo "  辅助特征: $use_aux"
    echo "  掩码感知: $mask_aware"
    echo "  GPU: $gpu"
    echo "  日志目录: $log_dir"
    echo "------------------------------------------------------------------------"
    
    # 构建命令
    local cmd="CUDA_VISIBLE_DEVICES=$gpu $PYTHON train.py \
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
    
    if [ -f "$STATS_FILE" ]; then
        cmd="$cmd --stats_file \"$STATS_FILE\""
    fi
    
    echo "执行: $cmd"
    echo ""
    
    eval "$cmd" 2>&1 | tee "$log_dir/train.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "实验 $exp_id 完成"
        echo "$exp_id,success,single,$gpu" >> "$OUTPUT_DIR/experiment_status.csv"
        return 0
    else
        log_error "实验 $exp_id 失败"
        echo "$exp_id,failed,single,$gpu" >> "$OUTPUT_DIR/experiment_status.csv"
        return 1
    fi
}

#===============================================================================
# 多卡训练函数 (使用 train_ddp.py)
#===============================================================================

run_ddp_experiment() {
    local model=$1
    local fusion=$2
    local use_aux=$3
    local mask_aware=$4
    local gpus=${5:-$DEFAULT_GPUS}
    
    local n_gpus=$(count_gpus "$gpus")
    
    # 生成实验ID
    local exp_id="${EXP_PREFIX}_${model}_${fusion}_aux${use_aux}_mask${mask_aware}_${n_gpus}gpu_$(timestamp)"
    local log_dir="${OUTPUT_DIR}/${exp_id}"
    
    mkdir -p "$log_dir"
    
    echo ""
    echo "========================================================================"
    echo "实验: $exp_id (多卡模式)"
    echo "========================================================================"
    echo "  模型: $model"
    echo "  融合模式: $fusion"
    echo "  辅助特征: $use_aux"
    echo "  掩码感知: $mask_aware"
    echo "  GPU: $gpus (共 $n_gpus 卡)"
    echo "  有效批大小: $((BATCH_SIZE * n_gpus))"
    echo "  日志目录: $log_dir"
    echo "------------------------------------------------------------------------"
    
    # 构建DDP命令
    local cmd
    if [ "$DDP_LAUNCHER" == "torchrun" ]; then
        cmd="CUDA_VISIBLE_DEVICES=$gpus torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=$n_gpus \
            train_ddp.py"
    else
        cmd="CUDA_VISIBLE_DEVICES=$gpus $PYTHON -m torch.distributed.launch \
            --nproc_per_node=$n_gpus \
            --use_env \
            train_ddp.py"
    fi
    
    cmd="$cmd \
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
        --tensorboard true \
        --sync_bn true"
    
    if [ -f "$STATS_FILE" ]; then
        cmd="$cmd --stats_file \"$STATS_FILE\""
    fi
    
    echo "执行: $cmd"
    echo ""
    
    eval "$cmd" 2>&1 | tee "$log_dir/train.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "实验 $exp_id 完成"
        echo "$exp_id,success,ddp,$gpus" >> "$OUTPUT_DIR/experiment_status.csv"
        return 0
    else
        log_error "实验 $exp_id 失败"
        echo "$exp_id,failed,ddp,$gpus" >> "$OUTPUT_DIR/experiment_status.csv"
        return 1
    fi
}

#===============================================================================
# 智能模式选择
#===============================================================================

run_auto_experiment() {
    local model=$1
    local fusion=$2
    local use_aux=$3
    local mask_aware=$4
    local gpus=${5:-"auto"}
    
    # 自动检测GPU
    if [ "$gpus" == "auto" ]; then
        gpus=$(get_available_gpus)
        if [ -z "$gpus" ]; then
            log_warning "没有检测到空闲GPU，使用默认GPU: $DEFAULT_GPUS"
            gpus=$DEFAULT_GPUS
        fi
    fi
    
    local n_gpus=$(count_gpus "$gpus")
    
    log_gpu "使用GPU: $gpus (共 $n_gpus 卡)"
    
    if [ $n_gpus -gt 1 ]; then
        run_ddp_experiment "$model" "$fusion" "$use_aux" "$mask_aware" "$gpus"
    else
        run_single_experiment "$model" "$fusion" "$use_aux" "$mask_aware" "$gpus"
    fi
}

#===============================================================================
# 消融实验 - 单卡版
#===============================================================================

run_ablation_study() {
    local gpu=${1:-$DEFAULT_GPUS}
    
    log_info "开始消融实验 (单卡模式, GPU: $gpu)..."
    
    echo "experiment_id,status,mode,gpus" > "$OUTPUT_DIR/experiment_status.csv"
    
    local total=0
    local success=0
    local failed=0
    
    for model in "${MODELS[@]}"; do
        if [ "$model" == "vanilla_unet" ]; then
            run_single_experiment "$model" "concat" "false" "false" "$gpu" \
                && ((success++)) || ((failed++))
            ((total++))
        else
            for fusion in "${FUSION_MODES[@]}"; do
                for use_aux in "${USE_AUX_FLAGS[@]}"; do
                    for mask_aware in "${MASK_AWARE_FLAGS[@]}"; do
                        run_single_experiment "$model" "$fusion" "$use_aux" "$mask_aware" "$gpu" \
                            && ((success++)) || ((failed++))
                        ((total++))
                    done
                done
            done
        fi
    done
    
    echo ""
    echo "========================================================================"
    echo "消融实验完成 (单卡)"
    echo "========================================================================"
    echo "  总实验数: $total"
    echo "  成功: $success"
    echo "  失败: $failed"
    echo "  成功率: $(echo "scale=2; $success*100/$total" | bc)%"
    echo "========================================================================"
}

#===============================================================================
# 消融实验 - 多卡版
#===============================================================================

run_ablation_study_ddp() {
    local gpus=${1:-$DEFAULT_GPUS}
    
    log_info "开始消融实验 (多卡模式, GPUs: $gpus)..."
    
    echo "experiment_id,status,mode,gpus" > "$OUTPUT_DIR/experiment_status.csv"
    
    local total=0
    local success=0
    local failed=0
    
    for model in "${MODELS[@]}"; do
        if [ "$model" == "vanilla_unet" ]; then
            run_ddp_experiment "$model" "concat" "false" "false" "$gpus" \
                && ((success++)) || ((failed++))
            ((total++))
        else
            for fusion in "${FUSION_MODES[@]}"; do
                for use_aux in "${USE_AUX_FLAGS[@]}"; do
                    for mask_aware in "${MASK_AWARE_FLAGS[@]}"; do
                        run_ddp_experiment "$model" "$fusion" "$use_aux" "$mask_aware" "$gpus" \
                            && ((success++)) || ((failed++))
                        ((total++))
                    done
                done
            done
        fi
    done
    
    echo ""
    echo "========================================================================"
    echo "消融实验完成 (多卡)"
    echo "========================================================================"
    echo "  总实验数: $total"
    echo "  成功: $success"
    echo "  失败: $failed"
    echo "  成功率: $(echo "scale=2; $success*100/$total" | bc)%"
    echo "========================================================================"
}

#===============================================================================
# 快速消融实验 - 多卡版
#===============================================================================

run_quick_ablation_ddp() {
    local gpus=${1:-$DEFAULT_GPUS}
    
    log_info "开始快速消融实验 (多卡模式, GPUs: $gpus)..."
    
    echo "experiment_id,status,mode,gpus" > "$OUTPUT_DIR/experiment_status.csv"
    
    local experiments=(
        "vanilla_unet concat false false"
        "physics_unet add false false"
        "physics_unet gated false false"
        "physics_unet gated true false"
        "physics_unet gated true true"
    )
    
    local total=${#experiments[@]}
    local success=0
    local failed=0
    
    for exp in "${experiments[@]}"; do
        read -r model fusion use_aux mask_aware <<< "$exp"
        run_ddp_experiment "$model" "$fusion" "$use_aux" "$mask_aware" "$gpus" \
            && ((success++)) || ((failed++))
    done
    
    echo ""
    echo "========================================================================"
    echo "快速消融实验完成 (多卡)"
    echo "========================================================================"
    echo "  总实验数: $total"
    echo "  成功: $success"
    echo "  失败: $failed"
    echo "========================================================================"
}

#===============================================================================
# 单次DDP实验
#===============================================================================

run_single_ddp() {
    local model=${1:-"physics_unet"}
    local fusion=${2:-"gated"}
    local use_aux=${3:-"true"}
    local mask_aware=${4:-"true"}
    local gpus=${5:-$DEFAULT_GPUS}
    
    run_ddp_experiment "$model" "$fusion" "$use_aux" "$mask_aware" "$gpus"
}

#===============================================================================
# 断点续训 - DDP版
#===============================================================================

resume_experiment_ddp() {
    local checkpoint=$1
    local exp_name=$2
    local gpus=${3:-$DEFAULT_GPUS}
    
    if [ ! -f "$checkpoint" ]; then
        log_error "检查点不存在: $checkpoint"
        return 1
    fi
    
    local n_gpus=$(count_gpus "$gpus")
    log_info "恢复训练 (DDP): $checkpoint, GPUs: $gpus"
    
    local cmd
    if [ "$DDP_LAUNCHER" == "torchrun" ]; then
        cmd="CUDA_VISIBLE_DEVICES=$gpus torchrun \
            --standalone \
            --nnodes=1 \
            --nproc_per_node=$n_gpus \
            train_ddp.py"
    else
        cmd="CUDA_VISIBLE_DEVICES=$gpus $PYTHON -m torch.distributed.launch \
            --nproc_per_node=$n_gpus \
            --use_env \
            train_ddp.py"
    fi
    
    cmd="$cmd \
        --exp_name \"$exp_name\" \
        --output_dir \"$OUTPUT_DIR\" \
        --data_root \"$DATA_ROOT\" \
        --resume \"$checkpoint\" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE"
    
    eval "$cmd" 2>&1 | tee "${OUTPUT_DIR}/${exp_name}/resume.log"
}

#===============================================================================
# 结果汇总
#===============================================================================

summarize_results() {
    log_info "汇总实验结果..."
    
    local summary_file="$OUTPUT_DIR/results_summary.md"
    
    cat > "$summary_file" << EOF
# 实验结果汇总

生成时间: $(date)

## 实验配置

| 参数 | 值 |
|------|-----|
| 数据目录 | $DATA_ROOT |
| 训练轮数 | $EPOCHS |
| 批大小 (每GPU) | $BATCH_SIZE |
| 学习率 | $LR |

## 实验状态

EOF
    
    if [ -f "$OUTPUT_DIR/experiment_status.csv" ]; then
        echo "| 实验ID | 状态 | 模式 | GPUs |" >> "$summary_file"
        echo "|--------|------|------|------|" >> "$summary_file"
        tail -n +2 "$OUTPUT_DIR/experiment_status.csv" | while IFS=, read -r exp_id status mode gpus; do
            echo "| $exp_id | $status | $mode | $gpus |" >> "$summary_file"
        done
    fi
    
    echo "" >> "$summary_file"
    echo "## 最佳模型" >> "$summary_file"
    echo "" >> "$summary_file"
    
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
卫星数据同化实验自动化脚本 - 多GPU版

用法:
    ./run_experiments_ddp.sh [命令] [参数...]

命令:
    === 单卡模式 (使用 train.py) ===
    ablation [gpu]              运行完整消融实验 (单卡)
    quick [gpu]                 运行快速消融实验 (单卡)
    single [model] [fusion] [use_aux] [mask_aware] [gpu]
                                运行单次实验 (单卡)
    resume [checkpoint] [exp_name]
                                断点续训 (单卡)
    
    === 多卡模式 (使用 train_ddp.py) ===
    ablation_ddp [gpus]         运行完整消融实验 (多卡)
    quick_ddp [gpus]            运行快速消融实验 (多卡)
    single_ddp [model] [fusion] [use_aux] [mask_aware] [gpus]
                                运行单次实验 (多卡)
    resume_ddp [checkpoint] [exp_name] [gpus]
                                断点续训 (多卡)
    
    === 智能模式 ===
    auto [model] [fusion] [use_aux] [mask_aware]
                                自动检测GPU并选择最佳模式
    
    === 工具 ===
    status                      显示GPU状态
    summary                     汇总实验结果
    help                        显示帮助

示例:
    # 显示GPU状态
    ./run_experiments_ddp.sh status
    
    # 使用GPU 3运行单次实验
    ./run_experiments_ddp.sh single physics_unet gated true true 3
    
    # 使用GPU 2,3运行多卡实验
    ./run_experiments_ddp.sh single_ddp physics_unet gated true true 2,3
    
    # 使用GPU 0,1,2,3运行快速消融实验
    ./run_experiments_ddp.sh quick_ddp 0,1,2,3
    
    # 自动检测空闲GPU运行实验
    ./run_experiments_ddp.sh auto physics_unet gated true true
    
    # 断点续训 (多卡)
    ./run_experiments_ddp.sh resume_ddp outputs/exp_xxx/checkpoint.pth exp_xxx 2,3

环境变量:
    CUDA_VISIBLE_DEVICES        可覆盖默认GPU设置

配置:
    请修改脚本顶部的配置区，设置正确的数据路径。
    默认GPU: $DEFAULT_GPUS

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
        # 单卡命令
        ablation)
            shift
            run_ablation_study "$@"
            ;;
        quick)
            shift
            # 复用原始脚本的快速消融逻辑
            log_info "开始快速消融实验 (单卡)..."
            echo "experiment_id,status,mode,gpus" > "$OUTPUT_DIR/experiment_status.csv"
            local gpu=${1:-$DEFAULT_GPUS}
            local experiments=(
                "vanilla_unet concat false false"
                "physics_unet add false false"
                "physics_unet gated false false"
                "physics_unet gated true false"
                "physics_unet gated true true"
            )
            for exp in "${experiments[@]}"; do
                read -r model fusion use_aux mask_aware <<< "$exp"
                run_single_experiment "$model" "$fusion" "$use_aux" "$mask_aware" "$gpu"
            done
            ;;
        single)
            shift
            run_single_experiment "$@"
            ;;
        resume)
            shift
            local checkpoint=$1
            local exp_name=$2
            local gpu=${3:-$DEFAULT_GPUS}
            if [ ! -f "$checkpoint" ]; then
                log_error "检查点不存在: $checkpoint"
                exit 1
            fi
            CUDA_VISIBLE_DEVICES=$gpu $PYTHON train.py \
                --exp_name "$exp_name" \
                --output_dir "$OUTPUT_DIR" \
                --data_root "$DATA_ROOT" \
                --resume "$checkpoint" \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE
            ;;
        
        # 多卡命令
        ablation_ddp)
            shift
            run_ablation_study_ddp "$@"
            ;;
        quick_ddp)
            shift
            run_quick_ablation_ddp "$@"
            ;;
        single_ddp)
            shift
            run_single_ddp "$@"
            ;;
        resume_ddp)
            shift
            resume_experiment_ddp "$@"
            ;;
        
        # 智能模式
        auto)
            shift
            run_auto_experiment "$@"
            ;;
        
        # 工具命令
        status)
            show_gpu_status
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
