# 运行view_BG.py
#! /bin/bash
cd ~/Fuxi/Unet
BASE_PATH=$(pwd)
LEVEL=137
OUTPUT_DIR="$BASE_PATH/outputs/visualizations"
# data_1011
FILE_PATH="/data1/tonghua/BG/GB.2023050300"
# data_0403
FILE_PATH_1="/data1/lrx/data_0403/IC/IC.2021020112"
python $BASE_PATH/tools/visualize_temprature.py --file_path $FILE_PATH --level $LEVEL --output "$OUTPUT_DIR/global_temp_1011_$LEVEL.png"