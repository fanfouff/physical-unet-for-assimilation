"""查看IC,BG数据集中第一个文件的数据"""
import os
import sys
import time
import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import pygrib
import logging
logger = logging.getLogger()
class CustomFormatter(logging.Formatter):
    """自定义 Formatter，为不同的日志级别添加 ANSI 颜色代码"""

    # ANSI 颜色代码
    grey = "\x1b[38;20m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # 日志格式模板
    # 我们使用 %(levelname)-8s 来确保级别名称（如 "INFO", "ERROR"）对齐
    base_format = "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s"

    # 映射：日志级别 -> 带颜色的格式
    FORMATS = {
        logging.DEBUG: bold_red + base_format + reset,
        logging.INFO: green + base_format + reset,
        logging.WARNING: yellow + base_format + reset,
        logging.ERROR: red + base_format + reset,
        logging.CRITICAL: bold_red + base_format + reset
    }

    def format(self, record):
        # 1. 获取此记录级别的格式
        log_fmt = self.FORMATS.get(record.levelno)
        
        # 2. 创建一个临时的 formatter 实例并使用它
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
def parse_arguments():
    parser = argparse.ArgumentParser(description="View GRIB data")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the GRIB file")
    return parser.parse_args()
# 查看grib文件中的数据(经纬度范围，pressure level，时间范围，温度变量值)
def view_grib_data(file_path):
    with pygrib.open(file_path) as grib_file:
        # 打印文件句柄信息
        print(f"grib_file: {grib_file}")
        
        # .messages 属性返回的是消息总数（int）
        count = grib_file.messages 
        print(f"Total messages in GRIB file: {count}")
        
        # 修正：直接遍历 grib_file 对象来获取每一条消息 (msg)
        for msg in grib_file:
            logger.debug(f"--- GRIB Message Info ---")
            logger.debug(f"Message Number: {msg.messagenumber}")
            logger.debug(f"Name: {msg.name}")
            logger.debug(f"Level: {msg.level}")
            logger.debug(f"Data Date: {msg.dataDate}")
            logger.debug(f"Data Time: {msg.dataTime}")
            logger.debug(f"Values Shape: {msg.values.shape}")
            logger.debug("-" * 40)
def main():
    # 1. 获取 root logger
    logger = logging.getLogger()
    # 关键修改：将全局级别从 INFO 改为 DEBUG（允许 DEBUG 及以上级别日志输出）
    logger.setLevel(logging.DEBUG)

    # 2. 清除所有可能已存在的 handler（防止重复日志）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 3. 创建一个 StreamHandler (输出到 stdout)
    handler = logging.StreamHandler(sys.stdout)
    
    # 可选：如果 handler 有单独的级别设置，也需要改为 DEBUG（默认继承logger级别，此处保险起见）
    handler.setLevel(logging.DEBUG)
    
    # 4. 将 handler 的 formatter 设置为 CustomFormatter 实例
    handler.setFormatter(CustomFormatter())

    # 5. 将 handler 添加到 logger
    logger.addHandler(handler)
    args = parse_arguments()
    file_path = args.file_path
    view_grib_data(file_path)
if __name__ == "__main__":
    main()
    
