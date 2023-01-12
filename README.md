## training:
1. swtich import statement to desired model 
2. refer to the following commands

## train from scratch:

CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 --master_port 29110 python main.py --config_file configs/contrastive.yaml --opts OUTPUT_DIR ./exps/exp1/contrastive TRAIN.EPOCHS 100 DATASET.NUM_CLASSES 9 TRAIN.BATCH_SIZE 2 FINETUNE False MODEL.STAGE train_AQT

## resume training:

CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 --master_port 29110 python main.py --config_file configs/contrastive.yaml --opts OUTPUT_DIR ./exps/exp1/contrastive TRAIN.EPOCHS 100 DATASET.NUM_CLASSES 9 TRAIN.BATCH_SIZE 2 FINETUNE False RESUME ${MDOEL_CHECKPONIT_PATH.pth}

## finetune, retrain (different from resume training):

CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 --master_port 29110 python main.py --config_file configs/contrastive.yaml --opts OUTPUT_DIR ./exps/exp1/contrastive TRAIN.EPOCHS 100 DATASET.NUM_CLASSES 9 TRAIN.BATCH_SIZE 2 FINETUNE True RESUME ${MDOEL_CHECKPONIT_PATH.pth} MODEL.STAGE ${stage_name}

## evaluation:

CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 --master_port 29110 python main.py --config_file configs/contrastive.yaml --opts OUTPUT_DIR ./exps/exp1/contrastive TRAIN.EPOCHS 100 DATASET.NUM_CLASSES 9 TRAIN.BATCH_SIZE 2 FINETUNE False RESUME ${MDOEL_CHECKPONIT_PATH.pth} EVAL True

