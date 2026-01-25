#!/bin/bash
"""
===============================================================================
FY-3F温度廓线反演训练工具 - 交互式脚本
Temperature Profile Retrieval Training Tool - Interactive Script
===============================================================================

功能:
  1. 查看可用的配准数据
  2. 配置训练参数
  3. 执行消融实验
  4. 断点续训
  5. 结果汇总

用法:
  chmod +x run_training.sh
  ./run_training.sh

===============================================================================
"""

# ============= 配置参数 =============
DATA_ROOT="/data2/lrx/era_obs"
OUTPUT_DIR="/data2/lrx/outputs/sat"
TRAIN_SCRIPT="train_profile.py"

# ============= 颜色定义 =============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# ============= 辅助函数 =============

timestamp() {
    date +"%Y%m%d_%H%M%S"
}

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

print_header() {
    echo ""
    echo -e "${CYAN}=========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}=========================================${NC}"
    echo ""
}

# ============= 数据扫描函数 =============

scan_data() {
    print_header "📊 扫描配准数据目录"
    
    if [ ! -d "$DATA_ROOT" ]; then
        log_error "数据目录不存在: $DATA_ROOT"
        return 1
    fi
    
    echo "数据目录: $DATA_ROOT"
    echo ""
    
    # 遍历年份
    total_files=0
    for year_dir in "$DATA_ROOT"/*/; do
        if [ -d "$year_dir" ]; then
            year=$(basename "$year_dir")
            echo -e "${MAGENTA}📅 ${year}年:${NC}"
            
            # 遍历月份
            for month_dir in "$year_dir"*/; do
                if [ -d "$month_dir" ]; then
                    month=$(basename "$month_dir")
                    
                    # 统计文件数
                    npy_count=$(find "$month_dir" -name "*_X.npy" 2>/dev/null | wc -l)
                    h5_count=$(find "$month_dir" -name "*.h5" 2>/dev/null | wc -l)
                    
                    if [ $npy_count -gt 0 ] || [ $h5_count -gt 0 ]; then
                        echo "   ${month}月: ${npy_count} npy文件, ${h5_count} h5文件"
                        total_files=$((total_files + npy_count + h5_count))
                    fi
                fi
            done
            echo ""
        fi
    done
    
    echo -e "${GREEN}总计: ${total_files} 个数据文件${NC}"
    echo ""
    
    # 使用Python脚本获取更详细的信息
    read -p "是否查看详细样本统计? [y/n]: " show_detail
    if [ "$show_detail" = "y" ] || [ "$show_detail" = "Y" ]; then
        python -c "
from collocated_data_loader import scan_data_directory
scan_data_directory('$DATA_ROOT', verbose=True)
"
    fi
}

# ============= 选择年份 =============

select_year() {
    echo ""
    echo "请选择年份:"
    echo "  1) 所有年份"
    echo "  2) 2024年"
    echo "  3) 2025年"
    echo "  4) 自定义年份"
    echo ""
    read -p "请输入选择 [1-4]: " year_choice
    
    case $year_choice in
        1)
            YEAR_FILTER=""
            echo "✓ 选择: 所有年份"
            ;;
        2)
            YEAR_FILTER="2024"
            echo "✓ 选择: 2024年"
            ;;
        3)
            YEAR_FILTER="2025"
            echo "✓ 选择: 2025年"
            ;;
        4)
            read -p "请输入年份（如 2024）: " custom_year
            YEAR_FILTER="$custom_year"
            echo "✓ 选择: ${custom_year}年"
            ;;
        *)
            echo "无效选择，使用所有年份"
            YEAR_FILTER=""
            ;;
    esac
}

# ============= 选择月份 =============

select_months() {
    echo ""
    echo "请选择月份:"
    echo "  1) 所有月份"
    echo "  2) 第一季度 (1-3月)"
    echo "  3) 第二季度 (4-6月)"
    echo "  4) 第三季度 (7-9月)"
    echo "  5) 第四季度 (10-12月)"
    echo "  6) 上半年 (1-6月)"
    echo "  7) 下半年 (7-12月)"
    echo "  8) 自定义月份"
    echo ""
    read -p "请输入选择 [1-8]: " month_choice
    
    case $month_choice in
        1)
            MONTH_FILTER=""
            echo "✓ 选择: 所有月份"
            ;;
        2)
            MONTH_FILTER="01 02 03"
            echo "✓ 选择: 第一季度 (1-3月)"
            ;;
        3)
            MONTH_FILTER="04 05 06"
            echo "✓ 选择: 第二季度 (4-6月)"
            ;;
        4)
            MONTH_FILTER="07 08 09"
            echo "✓ 选择: 第三季度 (7-9月)"
            ;;
        5)
            MONTH_FILTER="10 11 12"
            echo "✓ 选择: 第四季度 (10-12月)"
            ;;
        6)
            MONTH_FILTER="01 02 03 04 05 06"
            echo "✓ 选择: 上半年 (1-6月)"
            ;;
        7)
            MONTH_FILTER="07 08 09 10 11 12"
            echo "✓ 选择: 下半年 (7-12月)"
            ;;
        8)
            read -p "请输入月份（用空格分隔，如 01 02 05）: " custom_months
            MONTH_FILTER="$custom_months"
            echo "✓ 选择: $custom_months"
            ;;
        *)
            echo "无效选择，使用所有月份"
            MONTH_FILTER=""
            ;;
    esac
}

# ============= 选择模型 =============

select_model() {
    echo ""
    echo "请选择模型:"
    echo "  1) simple_mlp   - 简单MLP (基线)"
    echo "  2) res_mlp      - 残差MLP"
    echo "  3) physics_mlp  - 物理感知MLP (推荐)"
    echo ""
    read -p "请输入选择 [1-3]: " model_choice
    
    case $model_choice in
        1)
            MODEL="simple_mlp"
            echo "✓ 选择: 简单MLP"
            ;;
        2)
            MODEL="res_mlp"
            echo "✓ 选择: 残差MLP"
            ;;
        3)
            MODEL="physics_mlp"
            echo "✓ 选择: 物理感知MLP"
            ;;
        *)
            MODEL="physics_mlp"
            echo "无效选择，使用物理感知MLP"
            ;;
    esac
}

# ============= 选择训练参数 =============

select_training_params() {
    echo ""
    echo "请选择训练配置:"
    echo "  1) 快速测试  (epochs=10, batch=512)"
    echo "  2) 标准训练  (epochs=100, batch=256)"
    echo "  3) 完整训练  (epochs=200, batch=128)"
    echo "  4) 自定义参数"
    echo ""
    read -p "请输入选择 [1-4]: " param_choice
    
    case $param_choice in
        1)
            EPOCHS=10
            BATCH_SIZE=512
            LR=0.001
            echo "✓ 选择: 快速测试"
            ;;
        2)
            EPOCHS=100
            BATCH_SIZE=256
            LR=0.001
            echo "✓ 选择: 标准训练"
            ;;
        3)
            EPOCHS=200
            BATCH_SIZE=128
            LR=0.0005
            echo "✓ 选择: 完整训练"
            ;;
        4)
            read -p "训练轮数 [100]: " custom_epochs
            EPOCHS=${custom_epochs:-100}
            
            read -p "批大小 [256]: " custom_batch
            BATCH_SIZE=${custom_batch:-256}
            
            read -p "学习率 [0.001]: " custom_lr
            LR=${custom_lr:-0.001}
            
            echo "✓ 自定义: epochs=$EPOCHS, batch=$BATCH_SIZE, lr=$LR"
            ;;
        *)
            EPOCHS=100
            BATCH_SIZE=256
            LR=0.001
            echo "无效选择，使用标准配置"
            ;;
    esac
}

# ============= 显示配置摘要 =============

show_summary() {
    print_header "📋 配置摘要"
    
    echo "数据配置:"
    echo "  数据目录: $DATA_ROOT"
    echo "  输出目录: $OUTPUT_DIR"
    
    if [ -n "$YEAR_FILTER" ]; then
        echo "  年份筛选: $YEAR_FILTER"
    else
        echo "  年份筛选: 所有年份"
    fi
    
    if [ -n "$MONTH_FILTER" ]; then
        echo "  月份筛选: $MONTH_FILTER"
    else
        echo "  月份筛选: 所有月份"
    fi
    
    echo ""
    echo "模型配置:"
    echo "  模型类型: $MODEL"
    
    echo ""
    echo "训练配置:"
    echo "  训练轮数: $EPOCHS"
    echo "  批大小: $BATCH_SIZE"
    echo "  学习率: $LR"
    
    echo ""
}

# ============= 构建命令 =============

build_command() {
    EXP_NAME="${EXP_PREFIX}_${MODEL}_$(timestamp)"
    
    CMD="python $TRAIN_SCRIPT \
        --exp_name \"$EXP_NAME\" \
        --output_dir \"$OUTPUT_DIR\" \
        --data_root \"$DATA_ROOT\" \
        --model \"$MODEL\" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR"
    
    if [ -n "$YEAR_FILTER" ]; then
        CMD="$CMD --year $YEAR_FILTER"
    fi
    
    if [ -n "$MONTH_FILTER" ]; then
        CMD="$CMD --months $MONTH_FILTER"
    fi
}

# ============= 执行单次训练 =============

run_single_training() {
    print_header "🚀 开始训练"
    
    # 选择配置
    select_year
    select_months
    select_model
    select_training_params
    
    EXP_PREFIX="profile_retrieval"
    build_command
    show_summary
    
    # 确认
    read -p "确认以上配置并开始训练? [y/n]: " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消"
        return
    fi
    
    # 创建日志目录
    LOG_DIR="$OUTPUT_DIR/$EXP_NAME"
    mkdir -p "$LOG_DIR"
    
    echo ""
    echo "实验ID: $EXP_NAME"
    echo "日志目录: $LOG_DIR"
    echo ""
    
    # 选择运行方式
    read -p "是否使用screen在后台运行? [y/n]: " use_screen
    
    if [ "$use_screen" = "y" ] || [ "$use_screen" = "Y" ]; then
        screen_name="train_$EXP_NAME"
        echo ""
        echo "启动screen会话: $screen_name"
        echo "使用 'screen -r $screen_name' 重新连接"
        echo "使用 'Ctrl+A D' 断开连接"
        screen -dmS "$screen_name" bash -c "$CMD 2>&1 | tee $LOG_DIR/train.log; echo ''; echo '训练完成! 按任意键退出...'; read"
        log_success "Screen会话已启动"
    else
        echo "执行: $CMD"
        echo ""
        eval $CMD 2>&1 | tee "$LOG_DIR/train.log"
    fi
}

# ============= 消融实验 =============

run_ablation() {
    print_header "🔬 消融实验"
    
    echo "消融实验将对比以下模型:"
    echo "  1. simple_mlp  - 简单MLP基线"
    echo "  2. res_mlp     - 残差MLP"
    echo "  3. physics_mlp - 物理感知MLP"
    echo ""
    
    # 选择数据范围
    select_year
    select_months
    
    # 选择训练参数
    select_training_params
    
    echo ""
    echo "将执行 3 组实验..."
    read -p "确认开始消融实验? [y/n]: " confirm
    
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消"
        return
    fi
    
    # 实验状态文件
    mkdir -p "$OUTPUT_DIR"
    echo "experiment_id,model,status,global_rmse" > "$OUTPUT_DIR/ablation_results.csv"
    
    # 运行实验
    MODELS=("simple_mlp" "res_mlp" "physics_mlp")
    
    for model in "${MODELS[@]}"; do
        MODEL="$model"
        EXP_PREFIX="ablation"
        build_command
        
        echo ""
        echo "=========================================="
        echo "实验: $EXP_NAME"
        echo "模型: $MODEL"
        echo "=========================================="
        
        LOG_DIR="$OUTPUT_DIR/$EXP_NAME"
        mkdir -p "$LOG_DIR"
        
        # 执行训练
        eval $CMD 2>&1 | tee "$LOG_DIR/train.log"
        
        if [ $? -eq 0 ]; then
            log_success "实验 $EXP_NAME 完成"
            echo "$EXP_NAME,$MODEL,success," >> "$OUTPUT_DIR/ablation_results.csv"
        else
            log_error "实验 $EXP_NAME 失败"
            echo "$EXP_NAME,$MODEL,failed," >> "$OUTPUT_DIR/ablation_results.csv"
        fi
    done
    
    print_header "📊 消融实验完成"
    echo "结果文件: $OUTPUT_DIR/ablation_results.csv"
}

# ============= 快速测试 =============

run_quick_test() {
    print_header "⚡ 快速测试"
    
    echo "快速测试配置:"
    echo "  模型: physics_mlp"
    echo "  轮数: 5"
    echo "  批大小: 512"
    echo "  最大样本: 10000"
    echo ""
    
    read -p "确认开始快速测试? [y/n]: " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消"
        return
    fi
    
    EXP_NAME="quick_test_$(timestamp)"
    
    CMD="python $TRAIN_SCRIPT \
        --exp_name \"$EXP_NAME\" \
        --output_dir \"$OUTPUT_DIR\" \
        --data_root \"$DATA_ROOT\" \
        --model physics_mlp \
        --epochs 5 \
        --batch_size 512 \
        --max_samples 100"
    
    echo ""
    echo "执行: $CMD"
    echo ""
    
    eval $CMD
}

# ============= 断点续训 =============

resume_training() {
    print_header "🔄 断点续训"
    
    # 列出可用的检查点
    echo "可用的检查点:"
    echo ""
    
    checkpoints=$(find "$OUTPUT_DIR" -name "*.pth" -type f 2>/dev/null | head -20)
    
    if [ -z "$checkpoints" ]; then
        log_warning "未找到检查点文件"
        return
    fi
    
    i=1
    declare -a ckpt_array
    for ckpt in $checkpoints; do
        echo "  $i) $ckpt"
        ckpt_array[$i]="$ckpt"
        i=$((i+1))
    done
    
    echo ""
    read -p "请选择检查点编号: " ckpt_choice
    
    CHECKPOINT="${ckpt_array[$ckpt_choice]}"
    
    if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
        log_error "无效的检查点选择"
        return
    fi
    
    echo "选择: $CHECKPOINT"
    
    # 获取原始实验名
    exp_dir=$(dirname "$CHECKPOINT")
    exp_name=$(basename "$exp_dir")
    
    CMD="python $TRAIN_SCRIPT \
        --exp_name \"${exp_name}_resumed\" \
        --output_dir \"$OUTPUT_DIR\" \
        --data_root \"$DATA_ROOT\" \
        --resume \"$CHECKPOINT\" \
        --epochs 100"
    
    echo ""
    echo "执行: $CMD"
    read -p "确认继续? [y/n]: " confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        eval $CMD
    fi
}

# ============= 结果汇总 =============

show_results() {
    print_header "📊 实验结果汇总"
    
    if [ ! -d "$OUTPUT_DIR" ]; then
        log_warning "输出目录不存在: $OUTPUT_DIR"
        return
    fi
    
    echo "输出目录: $OUTPUT_DIR"
    echo ""
    
    # 列出所有实验
    echo "已完成的实验:"
    echo ""
    
    for exp_dir in "$OUTPUT_DIR"/*/; do
        if [ -d "$exp_dir" ]; then
            exp_name=$(basename "$exp_dir")
            
            # 检查是否有最佳模型
            if [ -f "$exp_dir/best_model.pth" ]; then
                status="✓ 完成"
            elif [ -f "$exp_dir/config.json" ]; then
                status="⚠ 进行中/中断"
            else
                status="? 未知"
            fi
            
            echo "  $exp_name: $status"
        fi
    done
    
    echo ""
    
    # 显示消融结果
    if [ -f "$OUTPUT_DIR/ablation_results.csv" ]; then
        echo "消融实验结果:"
        cat "$OUTPUT_DIR/ablation_results.csv"
        echo ""
    fi
}

# ============= 帮助信息 =============

show_help() {
    print_header "📖 帮助信息"
    
    echo "FY-3F温度廓线反演训练工具"
    echo ""
    echo "功能说明:"
    echo "  1) 查看数据     - 扫描配准数据目录，显示可用的年月和样本数"
    echo "  2) 单次训练     - 配置并执行一次训练"
    echo "  3) 消融实验     - 自动运行多组对比实验"
    echo "  4) 快速测试     - 使用少量数据快速验证"
    echo "  5) 断点续训     - 从检查点恢复训练"
    echo "  6) 查看结果     - 汇总所有实验结果"
    echo ""
    echo "数据目录结构:"
    echo "  $DATA_ROOT/"
    echo "  ├── YYYY/"
    echo "  │   ├── MM/"
    echo "  │   │   ├── collocation_YYYYMMDD_HHMM_X.npy  # 亮温 (N, 17)"
    echo "  │   │   ├── collocation_YYYYMMDD_HHMM_Y.npy  # 温度廓线 (N, 37)"
    echo "  │   │   └── collocation_YYYYMMDD_HHMM.h5"
    echo ""
    echo "模型说明:"
    echo "  simple_mlp  - 4层全连接网络，基线模型"
    echo "  res_mlp     - 带残差连接的MLP，6层"
    echo "  physics_mlp - 物理感知MLP，带通道注意力"
    echo ""
    echo "输出目录: $OUTPUT_DIR"
    echo ""
    
    # 显示Python脚本帮助
    read -p "是否查看Python训练脚本的详细参数? [y/n]: " show_py_help
    if [ "$show_py_help" = "y" ] || [ "$show_py_help" = "Y" ]; then
        python $TRAIN_SCRIPT --help
    fi
}

# ============= 主菜单 =============

main_menu() {
    while true; do
        print_header "🌡️ FY-3F温度廓线反演训练工具"
        
        echo "配置信息:"
        echo "  数据目录: $DATA_ROOT"
        echo "  输出目录: $OUTPUT_DIR"
        echo ""
        
        echo "请选择操作:"
        echo "  1) 📊 查看配准数据"
        echo "  2) 🚀 单次训练"
        echo "  3) 🔬 消融实验"
        echo "  4) ⚡ 快速测试"
        echo "  5) 🔄 断点续训"
        echo "  6) 📈 查看结果"
        echo "  7) 📖 帮助"
        echo "  8) 🚪 退出"
        echo ""
        read -p "请输入选择 [1-8]: " choice
        
        case $choice in
            1)
                scan_data
                ;;
            2)
                run_single_training
                ;;
            3)
                run_ablation
                ;;
            4)
                run_quick_test
                ;;
            5)
                resume_training
                ;;
            6)
                show_results
                ;;
            7)
                show_help
                ;;
            8)
                echo "退出"
                exit 0
                ;;
            *)
                log_warning "无效选择，请重试"
                ;;
        esac
        
        echo ""
        read -p "按回车键继续..." pause
    done
}

# ============= 主程序 =============

# 检查Python环境
if ! command -v python &> /dev/null; then
    log_error "Python未安装"
    exit 1
fi

# 检查训练脚本
if [ ! -f "$TRAIN_SCRIPT" ]; then
    log_error "训练脚本不存在: $TRAIN_SCRIPT"
    log_info "请确保以下文件在当前目录:"
    echo "  - train_profile.py"
    echo "  - collocated_data_loader.py"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 启动主菜单
main_menu
