# 批量配准工具 - 增强功能快速参考

## 🆕 新增功能概览

### 1. ✨ 月份选择
**之前**: 只能处理所有月份  
**现在**: 可以选择特定月份处理

### 2. ⏱️ 时间窗口配置
**之前**: 固定6小时窗口  
**现在**: 可选择 3h / 6h / 12h / 24h

---

## 📊 功能对比

| 功能 | 基础版 | 增强版 |
|------|--------|--------|
| 处理所有数据 | ✅ | ✅ |
| 选择年份 | ❌ | ✅ |
| 选择月份 | ❌ | ✅ |
| 时间窗口 | 固定6h | 3h/6h/12h/24h |
| 交互式界面 | 简单 | 丰富 |
| 批处理预设 | ❌ | ✅ (季度/半年) |

---

## 🚀 快速使用

### 方式1: 交互式脚本（推荐）

```bash
# 使用增强版交互式脚本
bash run_batch_collocation_enhanced.sh
```

**新菜单**:
- 查看可用的年月数据
- 配置并执行（年份/月份/时间窗口）
- 快速执行
- 查看帮助

### 方式2: 命令行（快速）

```bash
# 例1: 只处理2024年1-3月，使用3小时窗口
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 \
    --months 01 02 03 \
    --time-window 3

# 例2: 只处理2025年，使用12小时窗口
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2025 \
    --time-window 12
```

---

## 🎯 典型使用场景

### 场景1: 快速测试单月

```bash
# 先测试1月的数据
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 --months 01 --time-window 6
```

### 场景2: 分季度处理

```bash
# 第一季度
python batch_collocation.py ... --months 01 02 03

# 第二季度
python batch_collocation.py ... --months 04 05 06

# 第三季度
python batch_collocation.py ... --months 07 08 09

# 第四季度
python batch_collocation.py ... --months 10 11 12
```

### 场景3: 不同ERA5时间间隔

```bash
# ERA5是3小时间隔 → 用3小时窗口
python batch_collocation.py ... --time-window 3

# ERA5是6小时间隔 → 用6小时窗口
python batch_collocation.py ... --time-window 6

# ERA5是12小时间隔 → 用12小时窗口
python batch_collocation.py ... --time-window 12
```

---

## ⏱️ 时间窗口选择指南

| ERA5间隔 | 推荐窗口 | 说明 |
|----------|----------|------|
| 3小时 (00,03,06,09...) | 3小时 | 精确匹配 |
| 6小时 (00,06,12,18) | 6小时 | **最常用** |
| 12小时 (00,12) | 12小时 | 数据稀疏时 |
| 不确定 | 6小时 | 默认选择 |

**如何检查ERA5间隔**:
```bash
ls /data2/lrx/era5_2/split_data/2024/01/ | head
# 看文件名中的小时，判断间隔
```

---

## 📈 交互式界面预览

### 年份选择
```
请选择年份:
  1) 所有年份
  2) 2024年
  3) 2025年
  4) 自定义年份
```

### 月份选择
```
请选择月份:
  1) 所有月份
  2) 第一季度 (1-3月)
  3) 第二季度 (4-6月)
  4) 第三季度 (7-9月)
  5) 第四季度 (10-12月)
  6) 上半年 (1-6月)
  7) 下半年 (7-12月)
  8) 自定义月份
```

### 时间窗口选择
```
请选择ERA5时间窗口:
  1) 3小时  (精确匹配，文件数最多)
  2) 6小时  (标准窗口，推荐)
  3) 12小时 (宽松匹配)
  4) 24小时 (最宽松)
```

---

## 💡 最佳实践

### 1. 推荐工作流

```bash
# 步骤1: 使用交互式脚本，选择单月试运行
bash run_batch_collocation_enhanced.sh
# 选择: 配置并执行 → 2024年 → 01月 → 6小时 → 试运行

# 步骤2: 确认无误后，处理整季度
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 --months 01 02 03 --time-window 6

# 步骤3: 分批处理全年
# 第一季度、第二季度...
```

### 2. 并行处理

```bash
# 终端1
python batch_collocation.py ... --months 01 02 03

# 终端2
python batch_collocation.py ... --months 04 05 06

# 终端3
python batch_collocation.py ... --months 07 08 09

# 终端4
python batch_collocation.py ... --months 10 11 12
```

---

## 📝 命令行参数快速参考

### 新增参数

```bash
--year <YYYY>                 # 只处理指定年份
--months <MM> [MM ...]        # 只处理指定月份（空格分隔）
--time-window <3|6|12|24>     # ERA5时间窗口（小时）
```

### 完整示例

```bash
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \      # FY-3F目录
    --era5 /data2/lrx/era5_2/split_data \   # ERA5目录
    --output /data2/lrx/era_obs \           # 输出目录
    --year 2024 \                           # ✨新：年份筛选
    --months 01 02 03 \                     # ✨新：月份筛选
    --time-window 6 \                       # ✨新：时间窗口
    --dry-run                               # 试运行
```

---

## 📚 详细文档

1. **ENHANCED_FEATURES_GUIDE.md** - 增强功能完整指南
2. **BATCH_COLLOCATION_GUIDE.md** - 批量配准基础指南
3. **README.md** - 总览

---

## ❓ 快速FAQ

**Q: 时间窗口怎么选？**
A: 
- 不确定 → 用6小时（默认）
- ERA5是3小时间隔 → 用3小时
- ERA5是12小时间隔 → 用12小时

**Q: 如何只处理某几个月？**
A: 使用 `--months` 参数
```bash
--months 01 03 05  # 只处理1、3、5月
```

**Q: 能同时处理多个年份吗？**
A: 不能直接指定，但可以：
- 不加 `--year` 参数 → 处理所有年份
- 或分别运行两次

**Q: 交互式和命令行，哪个更好？**
A:
- 新手 → 用交互式 (`run_batch_collocation_enhanced.sh`)
- 熟练/批处理 → 用命令行 (`batch_collocation.py`)

---

## 🎉 开始使用

```bash
# 最简单的方式 - 交互式脚本
bash run_batch_collocation_enhanced.sh

# 或直接命令行（处理2024年第一季度）
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 --months 01 02 03 --time-window 6
```

享受新功能！🚀
