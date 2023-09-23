import argparse
import random

import numpy as np
import torch
import util.misc as utils
from datasets import build_dataset
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

    return cfg


def main(cfg):
    print("git:\n  {}\n".format(utils.get_sha()))
    print(cfg)

    # fix the seed for reproducibility
    seed = cfg.SEED + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # coco transform
    dataset_train = build_dataset(image_set='train', cfg=cfg)
    dataset_val_src = build_dataset(image_set='val_source', cfg=cfg)
    dataset_val_tgt = build_dataset(image_set='val_target', cfg=cfg)
    dataset_val_cross = build_dataset(image_set='val_cross', cfg=cfg)

    with open('./img_ids_with_wrong_labels.txt', 'r') as f:
        lines = f.readlines()
        img_ids = [int(l.split('\n')[0]) for l in lines]

    coco = dataset_val_src.coco
    lines = [f'id={img_id}_{coco.loadImgs(ids=[img_id])[0]["file_name"].split("/")[-1]}\n' for img_id in img_ids]
    with open('./img_ids_file_names.txt', 'w') as f:
        f.writelines(lines)
    exit()

    # plot_gt_bbox_from_dataset(dataset_train, img_ids=cfg.PLOT.IMG_IDS, prefix='train', output_dir='./gt_val_bbox')
    plot_gt_bbox_from_dataset(dataset_val_src, img_ids=cfg.PLOT.IMG_IDS, prefix='val_src', output_dir='./gt_val_bbox')
    plot_gt_bbox_from_dataset(dataset_val_tgt, img_ids=cfg.PLOT.IMG_IDS, prefix='val_tgt', output_dir='./gt_val_bbox')
    plot_gt_bbox_from_dataset(dataset_val_cross, img_ids=cfg.PLOT.IMG_IDS, prefix='val_cross', output_dir='./gt_val_bbox')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR Detector')
    parser.add_argument('--config_file', default='', type=str)
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = setup(args)
    main(cfg)
