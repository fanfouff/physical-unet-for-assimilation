# ERA5 GRIB文件分割工具

将大的ERA5 GRIB文件按照时间（年-月-日-小时）分割成多个小文件。

## 📋 前提条件

### 必需的Python包

```bash
# 方案1: 使用 pygrib (推荐，更快)
pip install pygrib

# 方案2: 使用 xarray + cfgrib
pip install xarray cfgrib eccodes
```

**注意**: 
- `pygrib` 在某些系统上安装可能需要系统库，如遇问题可使用conda:
  ```bash
  conda install -c conda-forge pygrib
  ```

## 🚀 快速开始

### 方法1: 简单脚本 (推荐)

```bash
# 基本用法
python split_era5_simple.py your_data.grib

# 指定输出目录
python split_era5_simple.py your_data.grib ./output_folder
```

**特点**:
- ✅ 最简单，最快速
- ✅ 只需要 pygrib 库
- ✅ 自动创建 YYYY/MM 目录结构
- ✅ 输出文件名: `era5_YYYYMMDD_HH.grib`

### 方法2: 完整脚本 (更多选项)

```bash
# 使用默认设置
python split_era5_grib.py your_data.grib

# 自定义输出目录
python split_era5_grib.py your_data.grib -o ./my_output

# 使用 xarray 方法
python split_era5_grib.py your_data.grib -m xarray

# 使用 pygrib 方法
python split_era5_grib.py your_data.grib -m pygrib

# 输出为 NetCDF 格式 (仅 xarray 方法)
python split_era5_grib.py your_data.grib -f netcdf
```

## 📁 输出结构

```
split_data/
├── 2021/
│   ├── 01/
│   │   ├── era5_20210101_00.grib
│   │   ├── era5_20210101_12.grib
│   │   ├── era5_20210102_00.grib
│   │   └── ...
│   ├── 02/
│   │   └── ...
│   └── ...
└── 2022/
    ├── 01/
    └── ...
```

每个文件包含:
- 单个时间步（00:00 或 12:00）
- 所有37个气压层
- 温度变量数据

## 💡 使用示例

### 示例1: 处理下载的数据

```python
# 假设你下载的文件名为 era5_temperature_2021-2022.grib
python split_era5_simple.py era5_temperature_2021-2022.grib
```

### 示例2: 批量处理多个文件

```bash
for file in *.grib; do
    python split_era5_simple.py "$file" "./split_${file%.grib}"
done
```

### 示例3: 在Python脚本中调用

```python
from split_era5_simple import split_grib_simple

# 分割文件
split_grib_simple('my_data.grib', './output')
```

## 🔍 两种方法对比

| 特性 | split_era5_simple.py | split_era5_grib.py |
|------|---------------------|-------------------|
| 速度 | ⚡⚡⚡ 非常快 | ⚡⚡ 较快 |
| 依赖 | pygrib | xarray/cfgrib 或 pygrib |
| 内存使用 | 低 | 中等 |
| 输出格式 | GRIB | GRIB 或 NetCDF |
| 复杂度 | 简单 | 功能更全 |
| 推荐使用 | 大多数情况 | 需要NetCDF输出时 |

## ⚙️ 性能提示

### 对于100GB的文件:

1. **内存**: 脚本使用流式处理，通常只需2-4GB内存

2. **磁盘空间**: 确保有足够空间（至少100GB + 分割后文件大小）

3. **处理时间**: 
   - pygrib方法: 约30-60分钟
   - xarray方法: 约60-120分钟
   - 取决于磁盘I/O速度

4. **优化建议**:
   ```bash
   # 使用SSD存储可显著加速
   # 如果可能，将输入和输出放在同一磁盘上
   ```

## 📊 验证分割结果

```python
import pygrib
from pathlib import Path

# 检查分割后的文件
output_dir = Path('./split_data')

# 统计文件数量
files = list(output_dir.rglob('*.grib'))
print(f"共分割为 {len(files)} 个文件")

# 检查单个文件
sample_file = files[0]
grbs = pygrib.open(str(sample_file))
print(f"\n{sample_file.name} 包含:")
for grb in grbs:
    print(f"  - {grb.validDate}, {grb.level}hPa")
grbs.close()
```

## 🛠️ 故障排除

### 问题1: ImportError: No module named 'pygrib'

**解决方案**:
```bash
# 使用 conda (推荐)
conda install -c conda-forge pygrib

# 或使用 pip (可能需要编译)
pip install pygrib
```

### 问题2: 内存不足

**解决方案**: 
- 使用 `split_era5_simple.py` (内存占用更小)
- 确保至少有4GB可用内存

### 问题3: 文件读取失败

**解决方案**:
```bash
# 检查文件是否完整
grib_ls your_file.grib | head

# 或使用Python检查
python -c "import pygrib; pygrib.open('your_file.grib')"
```

### 问题4: 处理速度慢

**解决方案**:
- 使用 SSD 存储
- 使用 `split_era5_simple.py` (更快)
- 确保没有其他程序占用磁盘I/O

## 📝 后续处理

分割后的文件可以用于:

```python
# 读取单个时间步
import pygrib
grbs = pygrib.open('split_data/2021/01/era5_20210101_00.grib')

# 或使用 xarray
import xarray as xr
ds = xr.open_dataset('split_data/2021/01/era5_20210101_00.grib', engine='cfgrib')

# 提取特定气压层的数据
temp_500hpa = grbs.select(level=500)[0].values
```

## 📞 常见问题

**Q: 分割后的文件可以重新合并吗?**  
A: 可以，使用 xarray 或 cdo 工具都可以合并。

**Q: 如何只分割特定月份的数据?**  
A: 可以修改脚本添加时间过滤，或者先用其他工具提取特定时间段。

**Q: 支持其他ERA5变量吗?**  
A: 支持! 脚本适用于任何ERA5的GRIB格式数据。

## 📄 许可证

此脚本为示例代码，可自由使用和修改。

---

**作者提示**: 建议先用小文件测试，确认输出格式符合需求后再处理大文件。
