# ------------------------------------------------------------------------
# Modified by Wei-Jie Huang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import os
import argparse
import random
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import datasets
import datasets.DAOD as DAOD
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from models import build_model
from config import get_cfg_defaults
from util.plot_utils import plot_gt_bbox_from_dataset


def setup(args):
    cfg = get_cfg_defaults()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    utils.init_distributed_mode(cfg)
    cfg.freeze()

    if cfg.OUTPUT_DIR:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        os.system(f'cp {args.config_file} {cfg.OUTPUT_DIR}')
        ddetr_src = 'models/deformable_detr.py'
        ddetr_des = Path(cfg.OUTPUT_DIR) / 'deformable_detr.py.backup'
        dtrans_src = 'models/deformable_transformer.py'
        dtrans_des = Path(cfg.OUTPUT_DIR) / 'deformable_transformer.py.backup'
        main_src = 'main.py'
        main_des = Path(cfg.OUTPUT_DIR) / 'main.py.backup'
        os.system(f'cp {ddetr_src} {ddetr_des}')
        os.system(f'cp {dtrans_src} {dtrans_des}')
        os.system(f'cp {main_src} {main_des}')

    return cfg


def main(cfg):
    print("git:\n  {}\n".format(utils.get_sha()))
    print(cfg)

    if cfg.MODEL.FROZEN_WEIGHTS is not None:
        assert cfg.MODEL.MASKS, "Frozen training is meant for segmentation only"

    device = torch.device(cfg.DEVICE)

    # fix the seed for reproducibility
    seed = cfg.SEED + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model, criterion, postprocessors = build_model(cfg)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    # coco transform
    dataset_train = build_dataset(image_set='train', cfg=cfg)
    dataset_val_src = build_dataset(image_set='val_source', cfg=cfg)
    dataset_val_cro = build_dataset(image_set='val_cross', cfg=cfg)
    dataset_val_tgt = build_dataset(image_set='val_target', cfg=cfg)
    plot_gt_bbox_from_dataset(dataset_val_cro, img_ids=cfg.PLOT.IMG_IDS, prefix='gt', output_dir=cfg.OUTPUT_DIR)

    if cfg.DIST.DISTRIBUTED:
        if cfg.CACHE_MODE:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val_src = samplers.NodeDistributedSampler(dataset_val_src, shuffle=False) if dataset_val_src is not None else None
            sampler_val_cro = samplers.NodeDistributedSampler(dataset_val_cro, shuffle=False)
            sampler_val_tgt = samplers.NodeDistributedSampler(dataset_val_tgt, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val_src = samplers.NodeDistributedSampler(dataset_val_src, shuffle=False) if dataset_val_src is not None else None
            sampler_val_cro = samplers.DistributedSampler(dataset_val_cro, shuffle=False)
            sampler_val_tgt = samplers.DistributedSampler(dataset_val_tgt, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val_src = samplers.NodeDistributedSampler(dataset_val_src, shuffle=False) if dataset_val_src is not None else None
        sampler_val_cro = torch.utils.data.SequentialSampler(dataset_val_cro)
        sampler_val_tgt = torch.utils.data.SequentialSampler(dataset_val_tgt)

    if 'uda' in cfg.DATASET.DA_MODE:
        assert cfg.TRAIN.BATCH_SIZE % 2 == 0, f'cfg.TRAIN.BATCH_SIZE {cfg.TRAIN.BATCH_SIZE} should be a multiple of 2'
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, cfg.TRAIN.BATCH_SIZE//2, drop_last=True)
            
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=DAOD.collate_fn, num_workers=cfg.NUM_WORKERS,
                                       pin_memory=True)

    else:
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, cfg.TRAIN.BATCH_SIZE, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                       pin_memory=True)
    data_loader_val_src = DataLoader(dataset_val_src, cfg.TRAIN.BATCH_SIZE, sampler=sampler_val_src,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                     pin_memory=True) if dataset_val_src is not None else None
    data_loader_val_cro = DataLoader(dataset_val_cro, cfg.TRAIN.BATCH_SIZE, sampler=sampler_val_cro,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                     pin_memory=True)
    data_loader_val_tgt = DataLoader(dataset_val_tgt, cfg.TRAIN.BATCH_SIZE, sampler=sampler_val_tgt,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                     pin_memory=True)

    print('')
    print('All parameters:')
    for n, p in model_without_ddp.named_parameters():
        print(f'    {n}')

    # model DDP
    if cfg.DIST.DISTRIBUTED:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.DIST.GPU])  # original implementation
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.DIST.GPU], find_unused_parameters=True)  # for vpt
        model_without_ddp = model.module

    if cfg.DATASET.DATASET_FILE == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", cfg)
        base_ds_tgt = get_coco_api_from_dataset(coco_val)
    else:
        base_ds_src = get_coco_api_from_dataset(dataset_val_src)
        base_ds_tgt = get_coco_api_from_dataset(dataset_val_tgt)

    if cfg.MODEL.FROZEN_WEIGHTS is not None:
        checkpoint = torch.load(cfg.MODEL.FROZEN_WEIGHTS, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(cfg.OUTPUT_DIR)

    if cfg.EVAL and cfg.FINETUNE:
        raise ValueError('Please turn off `FINETUNE` when evaluation')

    if cfg.MODEL.STAGE == 'visual_prompt_tuning':
        if not cfg.EVAL:
            if not cfg.FINETUNE:
                raise ValueError('The config key `FINTUNE` should be set as `True` in visual prompt tuning')

    # find all checkpoints
    ckp_dir = Path(args.ckp_dir)
    ckp_paths = sorted([p for p in ckp_dir.iterdir() if p.suffix == '.pth' and p.name != 'checkpoint.pth'])
    ckp_paths = [ckp_path for i, ckp_path in enumerate(ckp_paths) if i % args.epoch_step == 0]

    for i, ckp_path in enumerate(tqdm(ckp_paths, desc='checkpoints')):
        checkpoint = torch.load(str(ckp_path), map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]

        if i == 0:
            if len(missing_keys) > 0:
                print('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                print('Unexpected Keys: {}'.format(unexpected_keys))

        CURRENT_EPOCH = checkpoint['epoch'] if 'epoch' in checkpoint else -1

        # evaluate(model, criterion, postprocessors,
        #          data_loader_val_src, base_ds_src, device, cfg,
        #          prefix=f'eval_src_epoch={CURRENT_EPOCH}')

        evaluate(model, criterion, postprocessors,
                 data_loader_val_cro, base_ds_tgt, device, cfg,
                 category_ids=cfg.DATASET.CATEGORY_IDS,  # for bdd
                 prefix=f'eval_cro_epoch={CURRENT_EPOCH}')

        # evaluate(model, criterion, postprocessors,
        #          data_loader_val_tgt, base_ds_tgt, device, cfg,
        #          category_ids=cfg.DATASET.CATEGORY_IDS,  # for bdd
        #          prefix=f'eval_tgt_epoch={CURRENT_EPOCH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR Detector')
    parser.add_argument('--config_file', default='', type=str)
    parser.add_argument('--ckp_dir', default='', type=str)
    parser.add_argument('--epoch_step', default=10, type=int)
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = setup(args)
    main(cfg)
