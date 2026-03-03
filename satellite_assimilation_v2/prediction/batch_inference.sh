#!/bin/bash
# batch_inference.sh - 批量推理多个模型

MODELS=(
    "/home/seu/Fuxi/Unet/satellite_assimilation_v2/train_ddp/outputs/FY3F_Assimilation_physics_unet_gated_auxtrue_masktrue_2gpu_20260128_190438/best_model.pth:physics_full"
    "/home/seu/Fuxi/Unet/satellite_assimilation_v2/train_ddp/outputs/FY3F_Assimilation_physics_unet_gated_auxtrue_maskfalse_2gpu_20260128_201638/best_model.pth:physics_nomask"
    "/home/seu/Fuxi/Unet/satellite_assimilation_v2/train_ddp/outputs/FY3F_Assimilation_physics_unet_gated_auxfalse_masktrue_2gpu_20260128_201819/best_model.pth:physics_noaux"
)

DATA_ROOT="/data2/lrx/era_obs/npz/test"

for model_info in "${MODELS[@]}"; do
    IFS=':' read -r ckpt name <<< "$model_info"
    
    echo "========================================"
    echo "处理模型: $name"
    echo "========================================"
    
    ./run_inference.sh \
        --checkpoint "$ckpt" \
        --data "$DATA_ROOT" \
        --name "$name" \
        --compress \
        <<< "y"  # 自动确认
    
    echo ""
done

echo "所有模型处理完成！"
