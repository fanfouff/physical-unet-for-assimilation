source ~/miniconda3/bin/activate
conda activate fuxi
cd /home/lrx/Unet/satellite_assimilation_v2/train_ddp

DATA_ROOT="/data2/lrx/npz_64_real"
STATS_FILE="/data2/lrx/npz_64_real/stats.npz"
INC_STATS="/data2/lrx/npz_64_real/increment_stats.npz"
COMMON_ARGS="--data_root $DATA_ROOT --stats_file $STATS_FILE --increment_stats $INC_STATS --use_increment --batch_size 16 --epochs 200"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29514 train_ddp.py --output_dir "outputs/figures_ablation_comparison_noaux64" --exp_name v6_noaux_no_mask_aware_64 $COMMON_ARGS --use_aux true --mask_aware false --use_spectral_stem true --deep_supervision false --loss combined
