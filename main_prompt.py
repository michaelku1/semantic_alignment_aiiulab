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
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import datasets
import datasets.DAOD as DAOD
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from config import get_cfg_defaults


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
    dataset_val_tgt = build_dataset(image_set='val_target', cfg=cfg)

    if cfg.DIST.DISTRIBUTED:
        if cfg.CACHE_MODE:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val_src = samplers.NodeDistributedSampler(dataset_val_src, shuffle=False) if dataset_val_src is not None else None
            sampler_val_tgt = samplers.NodeDistributedSampler(dataset_val_tgt, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val_src = samplers.NodeDistributedSampler(dataset_val_src, shuffle=False) if dataset_val_src is not None else None
            sampler_val_tgt = samplers.DistributedSampler(dataset_val_tgt, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val_src = samplers.NodeDistributedSampler(dataset_val_src, shuffle=False) if dataset_val_src is not None else None
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
    data_loader_val_tgt = DataLoader(dataset_val_tgt, cfg.TRAIN.BATCH_SIZE, sampler=sampler_val_tgt,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                     pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    print('')
    print('All parameters:')
    for n, p in model_without_ddp.named_parameters():
        print(f'    {n}')

    model_stages = [
        'train_AQT',
        'train_encoder',
        'train_decoder',
        'visual_prompt_tuning'
    ]
    if cfg.MODEL.STAGE not in model_stages:
        raise ValueError(f'Model stage should be one of {", ".join(model_stages)}, but got {cfg.MODEL.STAGE}')

    if cfg.MODEL.STAGE == 'train_AQT':
        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                    if not match_name_keywords(n, cfg.TRAIN.LR_BACKBONE_NAMES) and not match_name_keywords(n, cfg.TRAIN.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
                "lr": cfg.TRAIN.LR,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.LR_BACKBONE_NAMES) and p.requires_grad],
                "lr": cfg.TRAIN.LR_BACKBONE,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.LR_LINEAR_PROJ_MULT,
            }
        ]

    elif cfg.MODEL.STAGE == 'train_encoder':
        trainable_layer = []
        # freeze all model parts except encoder
        for n, p in model_without_ddp.named_parameters():
            if match_name_keywords(n, ['encoder']):
                p.requires_grad_(True)
                trainable_layer.append(n)
            else:
                p.requires_grad_(False)
                
        frozen_layer = []
        # TODO freeze space and channel discriminators from encoder
        for n, p in model_without_ddp.named_parameters():
            if match_name_keywords(n, ['space_attn', 'channel_attn', 'space_D', 'channel_D']):
                p.requires_grad_(False)
                frozen_layer.append(n)
                
        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                    if not match_name_keywords(n, cfg.TRAIN.LR_BACKBONE_NAMES) and not match_name_keywords(n, cfg.TRAIN.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
                "lr": cfg.TRAIN.LR,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.LR_BACKBONE_NAMES) and p.requires_grad],
                "lr": cfg.TRAIN.LR_BACKBONE,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.LR_LINEAR_PROJ_MULT,
            }
        ]

    elif cfg.MODEL.STAGE == 'train_decoder':
        for n, p in model_without_ddp.named_parameters():
            if match_name_keywords(n, ['decoder']):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                    if not match_name_keywords(n, cfg.TRAIN.LR_BACKBONE_NAMES) and not match_name_keywords(n, cfg.TRAIN.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
                "lr": cfg.TRAIN.LR,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.LR_BACKBONE_NAMES) and p.requires_grad],
                "lr": cfg.TRAIN.LR_BACKBONE,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.LR_LINEAR_PROJ_MULT,
            }
        ]

    elif cfg.MODEL.STAGE == 'visual_prompt_tuning':
        print()
        print('Visual prompt tuning parameters:')
        names = [
            cfg.TRAIN.LR_HEAD_NAMES,
            cfg.TRAIN.LR_PROMPT_NAMES,
            ['prompt_proj'],
            ['prompt_dropout']
        ]
        names = sum(names, [])
        for n, p in model_without_ddp.named_parameters():
            if match_name_keywords(n, names):
                p.requires_grad_(True)  # train prompt embedding, prompt projection, prompt dropout
                print(f'    {n}')
            else:
                p.requires_grad_(False)  # freeze encoder & decoder

        param_dicts = [
            # head
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.LR_HEAD_NAMES) and p.requires_grad],
                "lr": cfg.TRAIN.LR_HEAD
            },

            # prompt
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.LR_PROMPT_NAMES) and p.requires_grad],
                "lr": cfg.TRAIN.LR_PROMPT
            }
        ]

    else:
        raise ValueError('Please specify training mode')
    
    if cfg.TRAIN.SGD:
        optimizer = torch.optim.SGD(param_dicts, lr=cfg.TRAIN.LR, momentum=0.9,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
                                      
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP) # (, steps, gamma=0.1)

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

    # load checkpoint and continue training
    if cfg.RESUME and not cfg.FINETUNE: # [BUG] write after freezing cfgs
        if cfg.RESUME.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.RESUME, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(cfg.RESUME, map_location='cpu')
        print('The model is loaded')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        # also load the states of optimizer and lr scheduler
        # while lr, initial_lr, step_size, base_lrs are new
        if not cfg.EVAL and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            # # pg_old here stands for the initialised optimizer above
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']  # replace checkpoint lr with new lr
                pg['initial_lr'] = pg_old['initial_lr']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            override_resumed_lr_drop = True
            if override_resumed_lr_drop:
                print('Warning: (hack) override_resumed_lr_drop is set to True, so cfg.TRAIN.LR_DROP would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = cfg.TRAIN.LR_DROP
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            START_EPOCH = checkpoint['epoch'] + 1
            print('The optimizer is loaded')
        else:
            START_EPOCH = 0

        # check the resumed model
        if not cfg.EVAL:
            print()
            print('Start evaluation before training')
            print('=== Target Domain ===')
            test_tgt_stats, _ = evaluate(
                model, criterion, postprocessors, data_loader_val_tgt, base_ds_tgt, device, cfg,
                category_ids=cfg.DATASET.CATEGORY_IDS,  # for bdd
                prefix='init_eval_tgt'
            )

            log_stats = {
                **{f'test_tgt_{k}': v for k, v in test_tgt_stats.items()},
                'epoch': 'before training',
                'n_parameters': n_parameters
            }
                        
            if cfg.OUTPUT_DIR and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    # load checkpoint and start a new training
    elif cfg.RESUME and cfg.FINETUNE:
        if cfg.RESUME.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.RESUME, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(cfg.RESUME, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        START_EPOCH = 0

        if not cfg.EVAL:
            print()
            print('Start evaluation before fine tuning')
            if dataset_val_src is not None:
                print('=== Source Domain ===')
                test_src_stats, coco_evaluator, src_query_idx_to_class_labels = evaluate(model, criterion, postprocessors,
                                                          data_loader_val_src, base_ds_src, device, cfg, prefix='init_eval_tgt')
            else:
                test_src_stats = {}

            print('=== Target Domain ===')
            test_tgt_stats, coco_evaluator, tgt_query_idx_to_class_labels = evaluate(model, criterion, postprocessors,
                                                      data_loader_val_tgt, base_ds_tgt, device, cfg,
                                                      category_ids=cfg.DATASET.CATEGORY_IDS,  # for bdd
                                                      prefix='init_eval_tgt')
            
            log_stats = {
                **{f'test_src_{k}': v for k, v in test_src_stats.items()},
                **{f'test_tgt_{k}': v for k, v in test_tgt_stats.items()},
                'epoch': 'before fine tuning',
                'n_parameters': n_parameters
            }
                        
            if cfg.OUTPUT_DIR and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_dir = Path(output_dir / 'query_idx_to_class_labels')
                save_dir.mkdir(exist_ok=True, parents=True)
                torch.save(src_query_idx_to_class_labels, str(save_dir / 'src_epoch=-1.pt'))
                torch.save(tgt_query_idx_to_class_labels, str(save_dir / 'tgt_epoch=-1.pt'))

    # start a new training with random initialized weights
    else:
        START_EPOCH = 0

    if cfg.EVAL:
        CURRENT_EPOCH = checkpoint['epoch'] if 'epoch' in checkpoint else -1

        test_src_stats = {}
        if dataset_val_src is not None:
            print('=== Source Domain ===')
            test_src_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                      data_loader_val_src, base_ds_src, device, cfg,
                                                      prefix=f'eval_src_epoch={CURRENT_EPOCH}')

        print('=== Target Domain ===')
        test_tgt_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                  data_loader_val_tgt, base_ds_tgt, device, cfg,
                                                  category_ids=cfg.DATASET.CATEGORY_IDS,  # for bdd
                                                  prefix=f'eval_tgt_epoch={CURRENT_EPOCH}')
        
        log_stats = {
            **{f'test_src_{k}': v for k, v in test_src_stats.items()},
            **{f'test_tgt_{k}': v for k, v in test_tgt_stats.items()},
            'epoch': CURRENT_EPOCH,
            'n_parameters': n_parameters
        }

        if cfg.OUTPUT_DIR and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        return
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    print("Start training")
    start_time = time.time()
    for epoch in range(START_EPOCH, cfg.TRAIN.EPOCHS):
        if cfg.DIST.DISTRIBUTED:
            sampler_train.set_epoch(epoch)
        
        train_stats, probs = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            postprocessors, cfg=cfg, prefix=f'train_epoch={epoch}')

        lr_scheduler.step()
        if cfg.OUTPUT_DIR:
            checkpoint_paths = [checkpoint_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % cfg.TRAIN.LR_DROP == 0 or (epoch + 1) % 1 == 0:
                checkpoint_paths.append(checkpoint_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'cfg': cfg,
                }, checkpoint_path)

        test_src_stats = {}
        if dataset_val_src is not None:
            print('=== Source Domain ===')
            test_src_stats, coco_evaluator, src_query_idx_to_class_labels = evaluate(
                model, criterion, postprocessors, data_loader_val_src, base_ds_src, device, cfg, prefix=f'eval_src_epoch={epoch}'
            )
            
        print('=== Target Domain ===')
        test_tgt_stats, coco_evaluator, tgt_query_idx_to_class_labels = evaluate(
            model, criterion, postprocessors, data_loader_val_tgt, base_ds_tgt, device, cfg,
            category_ids=cfg.DATASET.CATEGORY_IDS,  # for bdd
            prefix=f'eval_tgt_epoch={epoch}'
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_src_{k}': v for k, v in test_src_stats.items()},
                     **{f'test_tgt_{k}': v for k, v in test_tgt_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # log per epoch stats
        if cfg.OUTPUT_DIR and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

            save_dir = Path(output_dir / 'query_idx_to_class_labels')
            save_dir.mkdir(exist_ok=True, parents=True)
            torch.save(src_query_idx_to_class_labels, str(save_dir / f'src_epoch={epoch}.pt'))
            torch.save(tgt_query_idx_to_class_labels, str(save_dir / f'tgt_epoch={epoch}.pt'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR Detector')
    parser.add_argument('--config_file', default='', type=str)
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = setup(args)
    main(cfg)