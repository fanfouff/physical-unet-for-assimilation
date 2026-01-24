# 批量配准增强功能使用指南

## 🆕 新增功能

### 1. 月份选择
可以只处理特定的月份，而不是处理所有数据

### 2. 时间窗口配置
可以选择ERA5数据的时间匹配窗口：3h, 6h, 12h, 24h

---

## 🚀 使用方法

### 方式1: 交互式脚本（推荐新手）

```bash
bash run_batch_collocation_enhanced.sh
```

**交互式菜单**:
```
==========================================
主菜单
==========================================
  1) 查看可用的年月数据
  2) 配置并执行配准
  3) 快速执行（默认配置）
  4) 查看帮助
  5) 退出
```

**配置向导示例**:
```
请选择年份:
  1) 所有年份
  2) 2024年
  3) 2025年
  4) 自定义年份

请选择月份:
  1) 所有月份
  2) 第一季度 (1-3月)
  3) 第二季度 (4-6月)
  4) 第三季度 (7-9月)
  5) 第四季度 (10-12月)
  6) 上半年 (1-6月)
  7) 下半年 (7-12月)
  8) 自定义月份

请选择ERA5时间窗口:
  1) 3小时  (精确匹配，文件数最多)
  2) 6小时  (标准窗口，推荐)
  3) 12小时 (宽松匹配)
  4) 24小时 (最宽松)
```

---

### 方式2: 命令行参数（推荐高级用户）

#### 基本示例

```bash
# 1. 处理所有数据，使用6小时窗口（默认）
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs

# 2. 只处理2024年，使用6小时窗口
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024

# 3. 只处理2024年1-3月，使用3小时窗口
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 \
    --months 01 02 03 \
    --time-window 3

# 4. 试运行（查看将要做什么）
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 \
    --months 01 02 \
    --time-window 6 \
    --dry-run
```

---

## 📊 时间窗口说明

### 时间窗口的含义

时间窗口决定了FY-3F观测与ERA5数据的最大允许时间差。

| 窗口 | 说明 | 优点 | 缺点 | 推荐场景 |
|------|------|------|------|----------|
| 3小时 | 精确匹配 | 时间最接近 | 配准成功率较低 | ERA5数据密集（3小时间隔） |
| 6小时 | 标准窗口 | 平衡精度和成功率 | - | **推荐默认使用** |
| 12小时 | 宽松匹配 | 成功率高 | 时间差较大 | ERA5数据稀疏（12小时间隔） |
| 24小时 | 最宽松 | 成功率最高 | 时间差可能很大 | 仅用于测试或数据极少时 |

### 时间窗口示例

**ERA5文件**: `era5_20240115_12.grib` (12:00)

**FY-3F文件时间 vs 时间窗口**:

| FY-3F时间 | 时间差 | 3h窗口 | 6h窗口 | 12h窗口 | 24h窗口 |
|-----------|--------|--------|--------|---------|---------|
| 10:00 | 2小时 | ✅ | ✅ | ✅ | ✅ |
| 09:00 | 3小时 | ✅ | ✅ | ✅ | ✅ |
| 08:00 | 4小时 | ❌ | ✅ | ✅ | ✅ |
| 06:00 | 6小时 | ❌ | ✅ | ✅ | ✅ |
| 03:00 | 9小时 | ❌ | ❌ | ✅ | ✅ |
| 00:00 | 12小时 | ❌ | ❌ | ✅ | ✅ |

---

## 🎯 使用场景

### 场景1: 快速测试（处理单个月份）

```bash
# 只处理2024年1月，用于快速测试
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 \
    --months 01 \
    --time-window 6
```

### 场景2: 分季度处理

```bash
# 处理第一季度
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 \
    --months 01 02 03 \
    --time-window 6

# 处理第二季度
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 \
    --months 04 05 06 \
    --time-window 6
```

### 场景3: ERA5数据不同时间间隔

```bash
# ERA5数据是3小时间隔（00, 03, 06, 09, 12, 15, 18, 21）
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --time-window 3

# ERA5数据是6小时间隔（00, 06, 12, 18）
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --time-window 6

# ERA5数据是12小时间隔（00, 12）
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --time-window 12
```

### 场景4: 并行处理不同年份

```bash
# 终端1: 处理2024年
screen -S coll_2024
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024

# 终端2: 处理2025年
screen -S coll_2025
python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2025
```

---

## 📈 输出示例

### 使用月份筛选

```bash
$ python batch_collocation.py \
    --fy3f /data2/lrx/fy3f_organized \
    --era5 /data2/lrx/era5_2/split_data \
    --output /data2/lrx/era_obs \
    --year 2024 \
    --months 01 02 03 \
    --time-window 3

======================================================================
FY-3F 与 ERA5 批量配准工具
======================================================================
FY-3F目录: /data2/lrx/fy3f_organized
ERA5目录: /data2/lrx/era5_2/split_data
输出目录: /data2/lrx/era_obs
时间窗口: 3小时              ← 显示时间窗口
年份筛选: 2024              ← 显示年份筛选
月份筛选: 01, 02, 03        ← 显示月份筛选

找到 3 个月份需要处理:      ← 只处理筛选的月份
  2024年01月: 50 个FY-3F文件
  2024年02月: 45 个FY-3F文件
  2024年03月: 48 个FY-3F文件

======================================================================
处理 2024年01月
======================================================================
...
```

---

## 💡 最佳实践

### 1. 时间窗口选择建议

**检查ERA5数据时间间隔**:
```bash
# 查看ERA5文件名，判断时间间隔
ls /data2/lrx/era5_2/split_data/2024/01/ | head -10

# 示例输出：
# era5_20240115_00.grib
# era5_20240115_03.grib  ← 如果有03，说明是3小时间隔
# era5_20240115_06.grib
# era5_20240115_09.grib
# era5_20240115_12.grib
```

**根据间隔选择窗口**:
- ERA5是3小时间隔 → 用3小时窗口
- ERA5是6小时间隔 → 用6小时窗口
- ERA5是12小时间隔 → 用12小时窗口

### 2. 分批处理策略

**推荐**: 先处理单月测试，再批量处理

```bash
# 步骤1: 试运行查看
python batch_collocation.py ... --months 01 --dry-run

# 步骤2: 处理单月测试
python batch_collocation.py ... --months 01

# 步骤3: 确认无误后，批量处理
python batch_collocation.py ... --months 01 02 03 04 05 06
```

### 3. 并行处理建议

如果有多个服务器或GPU，可以并行处理：

**服务器1**: 处理上半年
```bash
python batch_collocation.py ... --months 01 02 03 04 05 06
```

**服务器2**: 处理下半年
```bash
python batch_collocation.py ... --months 07 08 09 10 11 12
```

---

## 🔍 监控和检查

### 检查处理进度

```bash
# 查看已完成的月份
ls -la /data2/lrx/era_obs/2024/

# 统计每月的配准文件数
for dir in /data2/lrx/era_obs/2024/*/; do
    month=$(basename $dir)
    count=$(ls -1 $dir/*_X.npy 2>/dev/null | wc -l)
    echo "2024年${month}月: $count 个配准文件"
done
```

### 检查时间窗口效果

```bash
# 对比不同时间窗口的配准成功率
# 查看日志或统计输出文件数
```

---

## ❓ 常见问题

**Q: 应该选择多大的时间窗口？**
A: 
- 默认6小时适合大多数情况
- 如果配准成功率低，可以增大窗口（12h或24h）
- 如果要求时间精度高，可以减小窗口（3h）

**Q: 如何知道ERA5数据的时间间隔？**
A: 查看ERA5文件名：
```bash
ls /data2/lrx/era5_2/split_data/2024/01/ | grep era5
```
看文件名中的小时数，计算间隔。

**Q: 月份筛选支持跨年吗？**
A: 需要分别处理：
```bash
# 2024年12月
python batch_collocation.py ... --year 2024 --months 12

# 2025年1月
python batch_collocation.py ... --year 2025 --months 01
```

**Q: 如何重新处理某个月份？**
A: 
1. 删除该月的输出文件
2. 重新运行指定月份

```bash
# 删除2024年1月的输出
rm -rf /data2/lrx/era_obs/2024/01/*

# 重新处理
python batch_collocation.py ... --year 2024 --months 01
```

---

## 📝 完整命令参考

```bash
python batch_collocation.py \
    --fy3f <FY3F目录> \
    --era5 <ERA5目录> \
    --output <输出目录> \
    [--year <年份>] \
    [--months <月份1> <月份2> ...] \
    [--time-window <3|6|12|24>] \
    [--collocation-script <脚本路径>] \
    [--mode <single|merge>] \
    [--dry-run]
```

**参数说明**:
- `--year`: 只处理指定年份
- `--months`: 只处理指定月份（空格分隔）
- `--time-window`: ERA5时间窗口（小时）
- `--dry-run`: 试运行模式

---

祝配准顺利！🎉
