# 运行view_BG.py
#! /bin/bash
SCRIPT_PATH=$(cd "$(dirname "$0")"; pwd)
# data_1011
FILE_PATH="/data1/tonghua/BG/GB.2023050300"
# data_0403
FILE_PATH_1="/data1/lrx/data_0403/IC/IC.2021020112"
python $SCRIPT_PATH/tools/view_BG.py --file_path $FILE_PATH_1