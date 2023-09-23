CUDA_VISIBLE_DEVICES=4 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 --master_port 29700 \
python plot_gt_val_bbox.py \
--config_file exps/cross_domain/0801_cyclegan_paired_fine_tune/config.yaml \
--opts \
    PLOT.IMG_IDS []
