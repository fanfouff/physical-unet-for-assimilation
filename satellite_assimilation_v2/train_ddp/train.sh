source ~/miniconda3/bin/activate
conda activate fuxi
cd /home/lrx/Unet/satellite_assimilation_v2/train_ddp

DATA_ROOT="/data2/lrx/npz_128_real"
STATS_FILE="/data2/lrx/npz_128_real/stats.npz"
INC_STATS="/data2/lrx/npz_128_real/increment_stats.npz"
COMMON_ARGS="--data_root $DATA_ROOT --stats_file $STATS_FILE --increment_stats $INC_STATS --use_increment --batch_size 16 --epochs 200"

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=35514 train_ddp.py --output_dir "outputs/figures_ablation_comparison_noaux128" --exp_name fengwu_128 $COMMON_ARGS --use_aux true --mask_aware true --use_spectral_stem true --deep_supervision false  --model fengwu
