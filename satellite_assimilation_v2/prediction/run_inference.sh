#!/bin/bash
# =============================================================================
# run_inference.sh - 自动化推理分析脚本
# =============================================================================
#
# 功能：
#   1. 自动生成带时间戳的输出目录
#   2. 记录运行参数和环境信息
#   3. 支持多种命名模式
#   4. 自动归档和压缩结果
#
# 用法：
#   chmod +x run_inference.sh
#   ./run_inference.sh [选项]
#
# 示例：
#   ./run_inference.sh -c best_model.pth -d /data/test -n "exp01_unet"
#   ./run_inference.sh --quick   # 快速模式，使用默认参数
#
# =============================================================================

set -e  # 遇错即停

# =============================================================================
# 颜色定义
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# 默认参数
# =============================================================================
CHECKPOINT=""
DATA_ROOT=""
EXP_NAME=""
BASE_OUTPUT_DIR="./figures"
BATCH_SIZE=16
NUM_WORKERS=4
CASE_STUDY_IDX=0
LEVEL_IDX=10
DEVICE="cuda"
STATS_FILE=""
BASELINE_CHECKPOINT=""

# 地理范围（默认中国区域）
LON_MIN=70
LON_MAX=140
LAT_MIN=15
LAT_MAX=55

# 其他选项
COMPRESS=false
VERBOSE=false
DRY_RUN=false

# =============================================================================
# 帮助信息
# =============================================================================
show_help() {
    cat << EOF
${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
${GREEN}自动化推理分析脚本${NC}
${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${YELLOW}用法:${NC}
    $0 [选项]

${YELLOW}必需参数:${NC}
    -c, --checkpoint PATH     模型checkpoint路径
    -d, --data PATH           测试数据目录

${YELLOW}可选参数:${NC}
    -n, --name NAME           实验名称（用于输出目录命名）
    -o, --output DIR          输出基础目录 (默认: ./figures)
    -b, --batch-size N        批大小 (默认: 16)
    -w, --workers N           DataLoader工作进程数 (默认: 4)
    --case-idx N              个例可视化样本索引 (默认: 0)
    --level-idx N             可视化层级索引 (默认: 10)
    --device DEVICE           设备 cuda/cpu (默认: cuda)
    --stats-file PATH         统计量文件路径
    --baseline PATH           基准模型checkpoint路径

${YELLOW}地理范围:${NC}
    --lon-min N               最小经度 (默认: 70)
    --lon-max N               最大经度 (默认: 140)
    --lat-min N               最小纬度 (默认: 15)
    --lat-max N               最大纬度 (默认: 55)

${YELLOW}其他选项:${NC}
    --compress                完成后压缩结果目录
    --verbose                 显示详细输出
    --dry-run                 仅显示命令，不执行
    -h, --help                显示帮助信息

${YELLOW}示例:${NC}
    # 基础用法
    $0 -c outputs/exp01/best.pth -d /data/test -n "physics_unet_v1"

    # 完整参数
    $0 -c best.pth -d /data/test -n "exp01" \\
       --baseline baseline.pth \\
       --lon-min 100 --lon-max 130 \\
       --compress --verbose

    # 使用配置文件
    source config.env && $0

${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}
EOF
}

# =============================================================================
# 日志函数
# =============================================================================
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# =============================================================================
# 参数解析
# =============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--checkpoint)
                CHECKPOINT="$2"
                shift 2
                ;;
            -d|--data)
                DATA_ROOT="$2"
                shift 2
                ;;
            -n|--name)
                EXP_NAME="$2"
                shift 2
                ;;
            -o|--output)
                BASE_OUTPUT_DIR="$2"
                shift 2
                ;;
            -b|--batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            -w|--workers)
                NUM_WORKERS="$2"
                shift 2
                ;;
            --case-idx)
                CASE_STUDY_IDX="$2"
                shift 2
                ;;
            --level-idx)
                LEVEL_IDX="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --stats-file)
                STATS_FILE="$2"
                shift 2
                ;;
            --baseline)
                BASELINE_CHECKPOINT="$2"
                shift 2
                ;;
            --lon-min)
                LON_MIN="$2"
                shift 2
                ;;
            --lon-max)
                LON_MAX="$2"
                shift 2
                ;;
            --lat-min)
                LAT_MIN="$2"
                shift 2
                ;;
            --lat-max)
                LAT_MAX="$2"
                shift 2
                ;;
            --compress)
                COMPRESS=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# =============================================================================
# 验证参数
# =============================================================================
validate_args() {
    local has_error=false
    
    if [[ -z "$CHECKPOINT" ]]; then
        log_error "必须指定 --checkpoint"
        has_error=true
    elif [[ ! -f "$CHECKPOINT" ]]; then
        log_error "Checkpoint文件不存在: $CHECKPOINT"
        has_error=true
    fi
    
    if [[ -z "$DATA_ROOT" ]]; then
        log_error "必须指定 --data"
        has_error=true
    elif [[ ! -d "$DATA_ROOT" ]]; then
        log_error "数据目录不存在: $DATA_ROOT"
        has_error=true
    fi
    
    if [[ -n "$BASELINE_CHECKPOINT" && ! -f "$BASELINE_CHECKPOINT" ]]; then
        log_error "Baseline文件不存在: $BASELINE_CHECKPOINT"
        has_error=true
    fi
    
    if [[ -n "$STATS_FILE" && ! -f "$STATS_FILE" ]]; then
        log_warn "统计量文件不存在: $STATS_FILE (将自动计算)"
    fi
    
    if $has_error; then
        echo ""
        show_help
        exit 1
    fi
}

# =============================================================================
# 生成输出目录名
# =============================================================================
generate_output_dir() {
    # 时间戳
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    
    # 从checkpoint路径提取模型名
    local model_name=""
    if [[ -n "$CHECKPOINT" ]]; then
        # 提取父目录名或文件名（不含扩展名）
        local ckpt_basename=$(basename "$CHECKPOINT" .pth)
        local ckpt_dirname=$(basename $(dirname "$CHECKPOINT"))
        
        if [[ "$ckpt_basename" == "best_model" || "$ckpt_basename" == "checkpoint" ]]; then
            model_name="${ckpt_dirname}"
        else
            model_name="${ckpt_basename}"
        fi
    fi
    
    # 组合目录名
    local dir_name=""
    
    if [[ -n "$EXP_NAME" ]]; then
        # 使用用户指定的实验名
        dir_name="${EXP_NAME}_${timestamp}"
    elif [[ -n "$model_name" ]]; then
        # 使用模型名
        dir_name="${model_name}_${timestamp}"
    else
        # 仅使用时间戳
        dir_name="inference_${timestamp}"
    fi
    
    # 清理非法字符
    dir_name=$(echo "$dir_name" | sed 's/[^a-zA-Z0-9_-]/_/g')
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${dir_name}"
}

# =============================================================================
# 显示配置信息
# =============================================================================
show_config() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}📊 推理分析配置${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  ${YELLOW}模型设置:${NC}"
    echo -e "    Checkpoint:     ${CHECKPOINT}"
    [[ -n "$BASELINE_CHECKPOINT" ]] && echo -e "    Baseline:       ${BASELINE_CHECKPOINT}"
    echo -e "    设备:           ${DEVICE}"
    echo ""
    echo -e "  ${YELLOW}数据设置:${NC}"
    echo -e "    数据目录:       ${DATA_ROOT}"
    [[ -n "$STATS_FILE" ]] && echo -e "    统计量文件:     ${STATS_FILE}"
    echo -e "    批大小:         ${BATCH_SIZE}"
    echo -e "    工作进程:       ${NUM_WORKERS}"
    echo ""
    echo -e "  ${YELLOW}可视化设置:${NC}"
    echo -e "    Case索引:       ${CASE_STUDY_IDX}"
    echo -e "    层级索引:       ${LEVEL_IDX}"
    echo -e "    地理范围:       [${LON_MIN}°E - ${LON_MAX}°E, ${LAT_MIN}°N - ${LAT_MAX}°N]"
    echo ""
    echo -e "  ${YELLOW}输出设置:${NC}"
    echo -e "    输出目录:       ${OUTPUT_DIR}"
    echo -e "    压缩结果:       ${COMPRESS}"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# =============================================================================
# 保存运行配置
# =============================================================================
save_config() {
    local config_file="${OUTPUT_DIR}/run_config.json"
    
    cat > "$config_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "user": "$(whoami)",
    "python_version": "$(python3 --version 2>&1)",
    "pytorch_version": "$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')",
    "cuda_available": "$(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')",
    "parameters": {
        "checkpoint": "${CHECKPOINT}",
        "data_root": "${DATA_ROOT}",
        "output_dir": "${OUTPUT_DIR}",
        "batch_size": ${BATCH_SIZE},
        "num_workers": ${NUM_WORKERS},
        "case_study_idx": ${CASE_STUDY_IDX},
        "level_idx": ${LEVEL_IDX},
        "device": "${DEVICE}",
        "stats_file": "${STATS_FILE}",
        "baseline_checkpoint": "${BASELINE_CHECKPOINT}",
        "lon_range": [${LON_MIN}, ${LON_MAX}],
        "lat_range": [${LAT_MIN}, ${LAT_MAX}]
    }
}
EOF
    
    log_info "配置已保存: $config_file"
}

# =============================================================================
# 构建Python命令
# =============================================================================
build_command() {
    CMD="python3 inference_analysis_v2.py"
    CMD+=" --checkpoint \"${CHECKPOINT}\""
    CMD+=" --data_root \"${DATA_ROOT}\""
    CMD+=" --output_dir \"${OUTPUT_DIR}\""
    CMD+=" --batch_size ${BATCH_SIZE}"
    CMD+=" --num_workers ${NUM_WORKERS}"
    CMD+=" --case_study_idx ${CASE_STUDY_IDX}"
    CMD+=" --level_idx ${LEVEL_IDX}"
    CMD+=" --device ${DEVICE}"
    CMD+=" --lon_min ${LON_MIN}"
    CMD+=" --lon_max ${LON_MAX}"
    CMD+=" --lat_min ${LAT_MIN}"
    CMD+=" --lat_max ${LAT_MAX}"
    
    [[ -n "$STATS_FILE" ]] && CMD+=" --stats_file \"${STATS_FILE}\""
    [[ -n "$BASELINE_CHECKPOINT" ]] && CMD+=" --baseline_checkpoint \"${BASELINE_CHECKPOINT}\""
}

# =============================================================================
# 运行推理
# =============================================================================
run_inference() {
    log_step "开始推理分析..."
    echo ""
    
    local start_time=$(date +%s)
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    # 保存配置
    save_config
    
    # 构建命令
    build_command
    
    if $VERBOSE; then
        echo -e "${CYAN}执行命令:${NC}"
        echo "$CMD"
        echo ""
    fi
    
    if $DRY_RUN; then
        log_warn "Dry-run模式，不执行实际命令"
        echo ""
        echo "$CMD"
        return 0
    fi
    
    # 执行命令并记录日志
    local log_file="${OUTPUT_DIR}/inference.log"
    
    echo "# 推理日志" > "$log_file"
    echo "# 开始时间: $(date)" >> "$log_file"
    echo "# 命令: $CMD" >> "$log_file"
    echo "" >> "$log_file"
    
    if $VERBOSE; then
        eval "$CMD" 2>&1 | tee -a "$log_file"
    else
        eval "$CMD" 2>&1 | tee -a "$log_file"
    fi
    
    local exit_code=${PIPESTATUS[0]}
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "" >> "$log_file"
    echo "# 结束时间: $(date)" >> "$log_file"
    echo "# 耗时: ${duration}秒" >> "$log_file"
    echo "# 退出码: $exit_code" >> "$log_file"
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "推理失败，退出码: $exit_code"
        log_error "请查看日志: $log_file"
        exit $exit_code
    fi
    
    log_info "推理完成，耗时: ${duration}秒"
}

# =============================================================================
# 压缩结果
# =============================================================================
compress_results() {
    if ! $COMPRESS; then
        return 0
    fi
    
    log_step "压缩结果目录..."
    
    local archive_name="${OUTPUT_DIR}.tar.gz"
    
    tar -czf "$archive_name" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"
    
    log_info "已创建压缩包: $archive_name"
    
    # 显示压缩包大小
    local size=$(du -h "$archive_name" | cut -f1)
    log_info "压缩包大小: $size"
}

# =============================================================================
# 显示结果摘要
# =============================================================================
show_summary() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ 推理分析完成${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  ${YELLOW}输出目录:${NC} ${OUTPUT_DIR}"
    echo ""
    echo -e "  ${YELLOW}生成的文件:${NC}"
    
    if [[ -d "$OUTPUT_DIR" ]]; then
        ls -lh "$OUTPUT_DIR" | tail -n +2 | while read line; do
            echo "    $line"
        done
    fi
    
    echo ""
    
    if $COMPRESS && [[ -f "${OUTPUT_DIR}.tar.gz" ]]; then
        echo -e "  ${YELLOW}压缩包:${NC} ${OUTPUT_DIR}.tar.gz"
        echo ""
    fi
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# =============================================================================
# 主函数
# =============================================================================
main() {
    # 解析参数
    parse_args "$@"
    
    # 验证参数
    validate_args
    
    # 生成输出目录名
    generate_output_dir
    
    # 显示配置
    show_config
    
    # 确认执行
    if ! $DRY_RUN; then
        read -p "是否继续执行? [Y/n] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
            log_warn "用户取消执行"
            exit 0
        fi
    fi
    
    # 运行推理
    run_inference
    
    # 压缩结果
    compress_results
    
    # 显示摘要
    show_summary
}

# =============================================================================
# 入口
# =============================================================================
main "$@"
