import os
import sys
import argparse
import logging
import pygrib
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 仿照 view_BG.py 的彩色日志系统 ---
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    base_format = "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + base_format + reset,
        logging.INFO: green + base_format + reset,
        logging.WARNING: yellow + base_format + reset,
        logging.ERROR: red + base_format + reset,
        logging.CRITICAL: bold_red + base_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logging():
    logger = logging.getLogger("TempVisualizer")
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)
    return logger

logger = setup_logging()

# --- 参数解析 ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize GRIB Temperature on Global Map")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the GRIB file")
    parser.add_argument('--level', type=int, default=137, help="Vertical level to visualize (e.g., 137 or 500)")
    parser.add_argument('--output', type=str, default="temp_map.png", help="Output image path")
    return parser.parse_args()

def visualize_temperature(file_path, target_level, output_path):
    try:
        with pygrib.open(file_path) as grbs:
            # 尝试查找匹配变量和层级的数据消息
            # 通常变量名为 'Temperature'，层级由参数指定
            try:
                selected_grbs = grbs.select(name='Temperature', level=target_level)
                if not selected_grbs:
                    logger.error(f"No Temperature data found for level {target_level}")
                    return
                msg = selected_grbs[0]
            except Exception:
                logger.warning(f"Could not find exact match for 'Temperature' at level {target_level}. Trying first message.")
                msg = grbs.message(1)

            logger.info(f"Processing: {msg.name} at Level: {msg.level}, Date: {msg.dataDate}")

            # 提取数据、经度和纬度
            data = msg.values
            lats, lons = msg.latlons()

            # --- 绘图逻辑 ---
            fig = plt.figure(figsize=(15, 8))
            ax = plt.axes(projection=ccrs.PlateCarree()) # 使用等距圆柱投影
            
            # 添加地理特征
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

            # 绘制伪彩色图
            # cmap='RdYlBu_r' 常用作气温展示（红暖蓝冷）
            clevs = np.linspace(data.min(), data.max(), 100)
            im = ax.contourf(lons, lats, data, clevs, transform=ccrs.PlateCarree(), cmap='RdYlBu_r')

            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, aspect=50)
            cbar.set_label(f'Temperature ({msg.units})')

            plt.title(f"Global Temperature Field - Level {msg.level}\nDate: {msg.dataDate} {msg.dataTime:04d}", fontsize=15)
            
            # 保存图片
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Successfully saved visualization to {output_path}")
            plt.close()

    except Exception as e:
        logger.error(f"An error occurred: {e}")

def main():
    args = parse_arguments()
    if not os.path.exists(args.file_path):
        logger.critical(f"File not found: {args.file_path}")
        return
    
    visualize_temperature(args.file_path, args.level, args.output)

if __name__ == "__main__":
    main()