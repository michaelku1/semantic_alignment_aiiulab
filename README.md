# BlenDA: Domain Adaptive Object Detection through Diffusion Blending


## Contributions:
- We are the first to consider diffusion-based synthesized images to retain semantics of small objects, thus becoming robust against domain shift.
- We leverage a mix-up training method and introduces a dynamic weighting scheme to stabilize domain transfer.
- Experimental results show that our method surpasses previous Faster-RCNN based and Transformer based methods.


![teaser_a](https://github.com/michaelku1/semantic_alignment_aiiulab/assets/48415065/20c1c4cb-e20c-4b55-aa8e-c4a876800b20)



## Training and Evaluation


start from pretrain:

`CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_d/run_dist_launch.sh 1 --master_port 29116 python main.py --config_file configs/contrastive_from_39_epochs.yaml --opts OUTPUTexps_s_DIR exps_bs4_retrain/debug TRAIN.EPOCHS 102 DATASET.NUM_CLASSES 9 TRAIN.BATCH_SIZE 4 MODEL.STAGE train_AQT DATASET.DA_MODE NUM_FTuda MODEL.NUM_FEATURE_LEVELS 1 RESUME exps_bs4_retrain/AQT_pretrain/checkpoint0040.pth FINETUNE True EMA True CONTRASTIVE True LASS_Fue LOSS.INTER_CLASS_COEF 1. LOSS.INTRA_CLASS_COEF 1.`

evaluation:

`CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 --master_port 29110 python main.py --config_file configs/contrastive.yaml --opts OUTPUT_DIR ./exps/exp1/contrastive TRAIN.EPOCHS 100 DATASET.NUM_CLASSES 9 TRAIN.BATCH_SIZE 2 FINETUNE False RESUME ${MDOEL_CHECKPONIT_PATH.pth} EVAL True`


## Ackowledgement

