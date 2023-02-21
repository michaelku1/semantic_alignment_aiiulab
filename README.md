## training:
1. swtich import statement to desired model 
2. refer to the following commands

## start from pretrain:
CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_d/run_dist_launch.sh 1 --master_port 29116 python main.py --config_file configs/contrastive_from_39_epochs.yaml --opts OUTPUTexps_s_DIR exps_bs4_retrain/debug TRAIN.EPOCHS 102 DATASET.NUM_CLASSES 9 TRAIN.BATCH_SIZE 4 MODEL.STAGE train_AQT DATASET.DA_MODE NUM_FTuda MODEL.NUM_FEATURE_LEVELS 1 RESUME exps_bs4_retrain/AQT_pretrain/checkpoint0040.pth FINETUNE True EMA True CONTRASTIVE TrLASS_Fue LOSS.INTER_CLASS_COEF 1. LOSS.INTRA_CLASS_COEF 1.

## evaluation:
CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 --master_port 29110 python main.py --config_file configs/contrastive.yaml --opts OUTPUT_DIR ./exps/exp1/contrastive TRAIN.EPOCHS 100 DATASET.NUM_CLASSES 9 TRAIN.BATCH_SIZE 2 FINETUNE False RESUME ${MDOEL_CHECKPONIT_PATH.pth} EVAL True

## source warmup
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 --master_port 29114 python main.py --config_file configs/contrastive.yaml  --opts OUTPUT_DIR exps/pretrain_source_subset TRAIN.EPOCHS 300 DATASET.NUM_CLASSES 4 TRAIN.BATCH_SIZE 2  MODEL.STAGE train_AQT DATASET.DA_MODE source_only
