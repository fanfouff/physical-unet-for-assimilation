source ~/miniconda3/bin/activate
conda activate fuxi
cd ~/Unet/satellite_assimilation_v2/prediction/
python3 gen_eval_config.py \
  --exp_root /home/lrx/Unet/satellite_assimilation_v2/train_ddp/outputs/figures_ablation_comparison_noaux128 \
  --test_root /data2/lrx/npz_128_real/test \
  --stats_file /data2/lrx/npz_128_real/stats.npz \
  --increment_stats /data2/lrx/npz_128_real/increment_stats.npz \
  --output_dir figures/figures_ablation_comparison_v4_128x128 \
  --output_yaml eval_config_128.yaml \
  --skip_missing

# ────── Step 3: 运行评估 ──────
# python3 prediction/eval_all_experiments.py --config eval_config_128.yaml
