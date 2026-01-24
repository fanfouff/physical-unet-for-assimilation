# 🎉 批量配准工具更新说明 (v2.0)

## ✨ 新增功能

### 1. 月份选择功能
**之前**: 只能一次性处理所有月份  
**现在**: 可以灵活选择特定月份

```bash
# 只处理1-3月
python batch_collocation.py ... --months 01 02 03

# 只处理夏季
python batch_collocation.py ... --months 06 07 08
```

### 2. 时间窗口配置
**之前**: 固定6小时时间窗口  
**现在**: 可选择 3h / 6h / 12h / 24h

```bash
# 使用3小时窗口（精确匹配）
python batch_collocation.py ... --time-window 3

# 使用12小时窗口（宽松匹配）
python batch_collocation.py ... --time-window 12
```

### 3. 年份筛选
**新增**: 可以只处理特定年份

```bash
# 只处理2024年
python batch_collocation.py ... --year 2024
```

### 4. 增强版交互式脚本
**新增**: 更友好的交互式界面

```bash
bash run_batch_collocation_enhanced.sh
```

**新功能**:
- 查看可用年月数据
- 预设选项（季度、半年）
- 配置向导
- 参数验证

---

## 📦 更新文件

### 核心文件（已更新）

1. **batch_collocation.py** ⭐
   - 添加 `--year` 参数
   - 添加 `--months` 参数  
   - 添加 `--time-window` 参数
   - 改进时间匹配逻辑

2. **run_batch_collocation_enhanced.sh** 🆕
   - 交互式配置向导
   - 预设月份选项（季度/半年）
   - 时间窗口选择
   - 自动参数验证

### 新增文档

3. **ENHANCED_FEATURES_GUIDE.md** 🆕
   - 详细功能说明
   - 使用场景示例
   - 时间窗口选择指南

4. **QUICK_REFERENCE.md** 🆕
   - 快速参考
   - 命令示例
   - 最佳实践

---

## 🚀 快速开始

### 方式1: 交互式（推荐新手）

```bash
bash run_batch_collocation_enhanced.sh
```

选择菜单选项：
- 查看可用数据
- 配置并执行（年份/月份/时间窗口）
- 快速执行

### 方式2: 命令行（推荐熟练用户）

```bash
# 示例1: 处理2024年第一季度，6小时窗口
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 \
    --months 01 02 03 \
    --time-window 6

# 示例2: 只处理2025年1月，3小时窗口
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2025 \
    --months 01 \
    --time-window 3
```

---

## 🎯 使用场景

### 场景1: 快速测试
```bash
# 先测试单月
python batch_collocation.py ... --year 2024 --months 01
```

### 场景2: 分季度处理
```bash
# Q1
python batch_collocation.py ... --months 01 02 03

# Q2
python batch_collocation.py ... --months 04 05 06
```

### 场景3: ERA5不同时间间隔
```bash
# ERA5是3小时间隔
python batch_collocation.py ... --time-window 3

# ERA5是12小时间隔
python batch_collocation.py ... --time-window 12
```

---

## 📊 功能对比表

| 功能 | v1.0 (旧版) | v2.0 (新版) |
|------|------------|------------|
| 批量配准 | ✅ | ✅ |
| 年份筛选 | ❌ | ✅ 新增 |
| 月份筛选 | ❌ | ✅ 新增 |
| 时间窗口配置 | ❌ (固定6h) | ✅ 3h/6h/12h/24h |
| 交互式界面 | 基础 | 增强版 |
| 预设选项 | ❌ | ✅ 季度/半年 |

---

## ⏱️ 时间窗口说明

### 什么是时间窗口？
时间窗口决定FY-3F观测与ERA5数据的最大允许时间差。

### 如何选择？

| ERA5时间间隔 | 推荐窗口 |
|-------------|----------|
| 3小时 (00,03,06...) | 3小时 |
| 6小时 (00,06,12,18) | 6小时 ⭐ |
| 12小时 (00,12) | 12小时 |

**检查ERA5间隔**:
```bash
ls /data2/lrx/era5_2/split_data/2024/01/ | head
# 查看文件名判断间隔
```

### 时间窗口效果示例

ERA5时间: 12:00

| FY-3F时间 | 时间差 | 3h窗口 | 6h窗口 | 12h窗口 |
|-----------|--------|--------|--------|---------|
| 10:00 | 2h | ✅ | ✅ | ✅ |
| 09:00 | 3h | ✅ | ✅ | ✅ |
| 08:00 | 4h | ❌ | ✅ | ✅ |
| 06:00 | 6h | ❌ | ✅ | ✅ |
| 03:00 | 9h | ❌ | ❌ | ✅ |

---

## 💡 最佳实践

### 1. 推荐工作流

```bash
# 步骤1: 查看可用数据
bash run_batch_collocation_enhanced.sh
# 选择: 1) 查看可用的年月数据

# 步骤2: 试运行单月
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 --months 01 --time-window 6 --dry-run

# 步骤3: 确认无误后批量处理
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 --months 01 02 03 --time-window 6
```

### 2. 并行处理策略

```bash
# 终端1: 第一季度
python batch_collocation.py ... --months 01 02 03

# 终端2: 第二季度
python batch_collocation.py ... --months 04 05 06

# 终端3: 第三季度
python batch_collocation.py ... --months 07 08 09

# 终端4: 第四季度
python batch_collocation.py ... --months 10 11 12
```

### 3. 时间窗口选择建议

1. **检查ERA5间隔**:
   ```bash
   ls /data2/lrx/era5_2/split_data/2024/01/ | head
   ```

2. **根据间隔选择**:
   - 3小时间隔 → 用3小时窗口
   - 6小时间隔 → 用6小时窗口
   - 12小时间隔 → 用12小时窗口

3. **不确定时**:
   - 默认使用6小时（最常用）

---

## 📚 文档索引

### 主要文档
1. **README.md** - 总览
2. **QUICK_REFERENCE.md** ⭐ - 快速参考（推荐先看）
3. **ENHANCED_FEATURES_GUIDE.md** - 详细功能指南

### 其他文档
4. **BATCH_COLLOCATION_GUIDE.md** - 批量配准基础指南
5. **QUICK_START.md** - 快速开始
6. **ORGANIZE_FILES_README.md** - 文件整理

---

## ❓ 常见问题

**Q: 旧版命令还能用吗？**
A: 能！向后兼容，不加新参数就是旧版行为。

**Q: 必须指定月份吗？**
A: 不必须。不加 `--months` 就处理所有月份。

**Q: 时间窗口默认是多少？**
A: 6小时（如果不指定）。

**Q: 如何只处理某个月？**
A: 
```bash
python batch_collocation.py ... --year 2024 --months 01
```

**Q: 交互式脚本和命令行哪个好？**
A:
- 新手 → 交互式脚本
- 熟练/自动化 → 命令行

---

## 🔄 迁移指南

### 从v1.0升级到v2.0

**旧版命令**:
```bash
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs
```

**新版命令（添加筛选）**:
```bash
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 \              # 新增
    --months 01 02 03 \        # 新增
    --time-window 6            # 新增
```

**无需修改旧脚本**，新参数都是可选的！

---

## 📞 获取帮助

```bash
# 查看所有参数
python batch_collocation.py --help

# 查看交互式脚本
bash run_batch_collocation_enhanced.sh
```

---

## 🎉 总结

### 核心改进
1. ✅ 月份选择 - 灵活处理特定月份
2. ✅ 时间窗口 - 适应不同ERA5间隔
3. ✅ 年份筛选 - 专注特定年份
4. ✅ 增强界面 - 更友好的交互

### 立即体验

```bash
# 最简单的方式
bash run_batch_collocation_enhanced.sh

# 或者命令行
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 --months 01 --time-window 6
```

享受新功能！🚀
