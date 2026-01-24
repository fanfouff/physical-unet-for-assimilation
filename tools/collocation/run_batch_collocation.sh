#!/bin/bash
"""
FY-3F与ERA5批量配准 - 增强版快速启动脚本
支持月份选择和时间窗口配置
"""

# ============= 配置参数 =============
FY3F_DIR="/data2/lrx/fy3f_organized"
ERA5_DIR="/data2/lrx/era5_2/split_data"
OUTPUT_DIR="/data2/lrx/era_obs"
COLLOCATION_SCRIPT="collocation_fy3d_era5_fixed.py"

# ============= 函数定义 =============

# 显示可用的年月
show_available_months() {
    echo ""
    echo "📅 扫描可用的年月..."
    echo ""
    
    if [ ! -d "$FY3F_DIR" ]; then
        echo "错误: FY-3F目录不存在"
        return 1
    fi
    
    years=$(find "$FY3F_DIR" -maxdepth 1 -type d -name "[0-9][0-9][0-9][0-9]" | sort)
    
    if [ -z "$years" ]; then
        echo "未找到年份目录"
        return 1
    fi
    
    for year_path in $years; do
        year=$(basename "$year_path")
        echo "📅 $year 年:"
        
        months=$(find "$year_path" -maxdepth 1 -type d -name "[0-9][0-9]" | sort)
        
        if [ -z "$months" ]; then
            echo "   (无月份数据)"
            continue
        fi
        
        for month_path in $months; do
            month=$(basename "$month_path")
            file_count=$(find "$month_path" -name "FY3F*.HDF" | wc -l)
            echo "   $month 月: $file_count 个文件"
        done
        echo ""
    done
}

# 选择年份
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

# 选择月份
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

# 选择时间窗口
select_time_window() {
    echo ""
    echo "请选择ERA5时间窗口:"
    echo "  1) 3小时  (精确匹配，文件数最多)"
    echo "  2) 6小时  (标准窗口，推荐)"
    echo "  3) 12小时 (宽松匹配)"
    echo "  4) 24小时 (最宽松)"
    echo ""
    read -p "请输入选择 [1-4]: " window_choice
    
    case $window_choice in
        1)
            TIME_WINDOW=3
            echo "✓ 选择: 3小时窗口"
            ;;
        2)
            TIME_WINDOW=6
            echo "✓ 选择: 6小时窗口（推荐）"
            ;;
        3)
            TIME_WINDOW=12
            echo "✓ 选择: 12小时窗口"
            ;;
        4)
            TIME_WINDOW=24
            echo "✓ 选择: 24小时窗口"
            ;;
        *)
            echo "无效选择，使用默认6小时"
            TIME_WINDOW=6
            ;;
    esac
}

# 构建命令
build_command() {
    CMD="python batch_collocation.py \
        --fy3f \"$FY3F_DIR\" \
        --era5 \"$ERA5_DIR\" \
        --output \"$OUTPUT_DIR\" \
        --collocation-script \"$COLLOCATION_SCRIPT\" \
        --time-window $TIME_WINDOW"
    
    if [ -n "$YEAR_FILTER" ]; then
        CMD="$CMD --year $YEAR_FILTER"
    fi
    
    if [ -n "$MONTH_FILTER" ]; then
        CMD="$CMD --months $MONTH_FILTER"
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        CMD="$CMD --dry-run"
    fi
}

# 显示配置摘要
show_summary() {
    echo ""
    echo "=========================================="
    echo "配置摘要"
    echo "=========================================="
    echo "FY-3F目录: $FY3F_DIR"
    echo "ERA5目录: $ERA5_DIR"
    echo "输出目录: $OUTPUT_DIR"
    echo "时间窗口: ${TIME_WINDOW}小时"
    
    if [ -n "$YEAR_FILTER" ]; then
        echo "年份筛选: $YEAR_FILTER"
    else
        echo "年份筛选: 所有年份"
    fi
    
    if [ -n "$MONTH_FILTER" ]; then
        echo "月份筛选: $MONTH_FILTER"
    else
        echo "月份筛选: 所有月份"
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        echo "模式: 试运行（不实际执行）"
    else
        echo "模式: 实际执行"
    fi
    echo "=========================================="
    echo ""
}

# ============= 主程序 =============

echo "=========================================="
echo "FY-3F与ERA5批量配准工具（增强版）"
echo "=========================================="
echo ""

# 检查目录和脚本
echo "配置检查:"
echo "  FY-3F目录: $FY3F_DIR"
echo "  ERA5目录: $ERA5_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  配准脚本: $COLLOCATION_SCRIPT"
echo ""

if [ ! -d "$FY3F_DIR" ]; then
    echo "❌ 错误: FY-3F目录不存在: $FY3F_DIR"
    exit 1
fi

if [ ! -d "$ERA5_DIR" ]; then
    echo "❌ 错误: ERA5目录不存在: $ERA5_DIR"
    exit 1
fi

if [ ! -f "$COLLOCATION_SCRIPT" ]; then
    echo "❌ 错误: 配准脚本不存在: $COLLOCATION_SCRIPT"
    echo "请确保 collocation_fy3d_era5_fixed.py 在当前目录"
    exit 1
fi

echo "✓ 所有文件和目录检查通过"

# 主菜单
while true; do
    echo ""
    echo "=========================================="
    echo "主菜单"
    echo "=========================================="
    echo "  1) 查看可用的年月数据"
    echo "  2) 配置并执行配准"
    echo "  3) 快速执行（默认配置）"
    echo "  4) 查看帮助"
    echo "  5) 退出"
    echo ""
    read -p "请输入选择 [1-5]: " main_choice
    
    case $main_choice in
        1)
            show_available_months
            ;;
        2)
            # 配置模式
            echo ""
            echo "=========================================="
            echo "配置向导"
            echo "=========================================="
            
            # 选择年份
            select_year
            
            # 选择月份
            select_months
            
            # 选择时间窗口
            select_time_window
            
            # 选择试运行或执行
            echo ""
            read -p "是否先试运行（查看将要做什么）? [y/n]: " dry_run_choice
            if [ "$dry_run_choice" = "y" ] || [ "$dry_run_choice" = "Y" ]; then
                DRY_RUN="true"
            else
                DRY_RUN="false"
            fi
            
            # 显示配置摘要
            show_summary
            
            # 确认执行
            read -p "确认以上配置并开始执行? [y/n]: " confirm
            if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
                echo "已取消"
                continue
            fi
            
            # 构建并执行命令
            build_command
            
            # 是否使用screen
            if [ "$DRY_RUN" = "false" ]; then
                read -p "是否使用screen在后台运行? [y/n]: " use_screen
                
                if [ "$use_screen" = "y" ] || [ "$use_screen" = "Y" ]; then
                    screen_name="fy3f_collocation_$(date +%Y%m%d_%H%M%S)"
                    echo ""
                    echo "启动screen会话: $screen_name"
                    echo "使用 'screen -r $screen_name' 重新连接"
                    screen -dmS "$screen_name" bash -c "$CMD; echo ''; echo '配准完成! 按任意键退出...'; read"
                    echo "✓ Screen会话已启动"
                else
                    eval $CMD
                fi
            else
                eval $CMD
            fi
            ;;
        3)
            # 快速执行（默认配置）
            echo ""
            echo "快速执行模式（使用默认配置）"
            YEAR_FILTER=""
            MONTH_FILTER=""
            TIME_WINDOW=6
            DRY_RUN="false"
            
            show_summary
            
            read -p "确认执行? [y/n]: " confirm
            if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
                echo "已取消"
                continue
            fi
            
            build_command
            
            read -p "是否使用screen在后台运行? [y/n]: " use_screen
            
            if [ "$use_screen" = "y" ] || [ "$use_screen" = "Y" ]; then
                screen_name="fy3f_collocation_$(date +%Y%m%d_%H%M%S)"
                echo ""
                echo "启动screen会话: $screen_name"
                echo "使用 'screen -r $screen_name' 重新连接"
                screen -dmS "$screen_name" bash -c "$CMD; echo ''; echo '配准完成! 按任意键退出...'; read"
                echo "✓ Screen会话已启动"
            else
                eval $CMD
            fi
            ;;
        4)
            # 帮助
            python batch_collocation.py --help
            ;;
        5)
            echo "退出"
            exit 0
            ;;
        *)
            echo "无效选择，请重试"
            ;;
    esac
done