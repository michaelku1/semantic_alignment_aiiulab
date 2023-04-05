import argparse
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import get_cfg_defaults
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from models import build_model
import util.misc as utils


def get_cfg(exp_dir_path):
    exp_dir = Path(exp_dir_path)
    if not exp_dir.exists():
        raise FileNotFoundError(f'{str(exp_dir)} does not exist')

    cfg_file = list(exp_dir.glob('*.yaml'))[0]
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(cfg_file))

    cfg.PLOT.PLOT_BBOX = False
    cfg.PLOT.PLOT_MAP = False
    print(cfg)

    return cfg


def check_log(log):
    msg = log[0]
    if 'test_coco_eval_bbox_cat-id=0' in msg:
        raise RuntimeError('There are already mAP results for each class. Check it.')


def prepare_data(cfg):
    dataset_val = build_dataset(image_set='val', cfg=cfg)
    base_ds = get_coco_api_from_dataset(dataset_val)

    if cfg.DIST.DISTRIBUTED:
        if cfg.CACHE_MODE:
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, cfg.TRAIN.BATCH_SIZE, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                 pin_memory=True)

    return data_loader_val, base_ds


def eval(exp_dir_path, cfg):
    exp_dir = Path(exp_dir_path)
    if not exp_dir.exists():
        raise FileNotFoundError(f'{str(exp_dir)} does not exist')

    log_file = exp_dir / 'log.txt'
    with log_file.open('r') as f:
        log = f.readlines()
    check_log(log)

    device = torch.device('cuda')
    data_loader_val, base_ds = prepare_data(cfg)

    model, criterion, postprocessors, postprocessors_target = build_model(cfg)
    model.to(device)
    model_without_ddp = model

    new_log = []
    t = tqdm(log)
    for msg in t:
        epoch = msg.split('"epoch": ')[-1].split(', "n_parameters"')[0]
        ckp_file = exp_dir / ('checkpoint{}.pth'.format(epoch.zfill(4)))
        memory_ckp_file = exp_dir / 'memory_prototypes' / 'ema_prototypes_epoch_{}.pt'.format(epoch.zfill(4))
        
        checkpoint = torch.load(str(ckp_file), map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        model_without_ddp.m_items = torch.load(str(memory_ckp_file), map_location='cpu')
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, postprocessors_target,
                                              data_loader_val, base_ds, device, cfg, prefix='eval')
        
        head_msg, tail_msg = msg.split('"epoch":')
        tail_msg = '"epoch":' + tail_msg
        for k, v in test_stats.items():
            if not k.startswith('coco_eval_bbox_cat-id'):
                continue
        
            head_msg += f'"test_{k}": {v}, '
        new_msg = head_msg + tail_msg
        new_log.append(new_msg)

    new_log_file = exp_dir / 'new_log.txt'
    with new_log_file.open('w') as f:
        for msg in new_log:
            f.write(msg)

    print(f'write new log in {str(new_log_file)}')


if __name__ == '__main__':
    """
    CUDA_VISIBLE_DEVICES=5,6 GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 --master_port 29500 \
    python fulfill_eval_per_class.py ...
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir_path', type=str)
    args = parser.parse_args()

    cfg = get_cfg(exp_dir_path=args.exp_dir_path)
    utils.init_distributed_mode(cfg)

    eval(exp_dir_path=args.exp_dir_path, cfg=cfg)