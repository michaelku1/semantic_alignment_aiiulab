# ----------------------------------------------
# Created by Wei-Jie Huang
# ----------------------------------------------
import copy
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset

from datasets.coco import CocoDetection, make_coco_transforms
from util.misc import get_local_rank, get_local_size, nested_tensor_from_tensor_list


def get_paths(root):
    root = Path(root)
    return {
        'cityscapes': {
            'train_img': root / 'cityscapes/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'cityscapes/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'cityscapes_caronly': {
            'train_img': root / 'cityscapes/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_caronly_train.json',
            'val_img': root / 'cityscapes/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_caronly_val.json',
        },
        'foggy_cityscapes': {
            'train_img': root / 'cityscapes/leftImg8bit_foggy/train',
            'train_anno': root / 'cityscapes/annotations/foggy_cityscapes_train.json',
            'val_img': root / 'cityscapes/leftImg8bit_foggy/val',
            'val_anno': root / 'cityscapes/annotations/foggy_cityscapes_val.json',
        },
        'city_instruction_cityscapes': {
            'train_img': root / 'adaptive_teacher/city_instruction_cityfoggy/leftImg8bit/train',
            'train_anno': root / 'adaptive_teacher/city_instruction_cityfoggy/annotations/cityscapes_train.json',
            'val_img': root / 'adaptive_teacher/city_instruction_cityfoggy/leftImg8bit/val',
            'val_anno': root / 'adaptive_teacher/city_instruction_cityfoggy/annotations/cityscapes_val.json',
        },
        'cyclegan_paired': {
            'train_img': root / 'city_cycle_cityfoggy/JPEGImages_paired',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'city_cycle_cityfoggy/JPEGImages_paired',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'cyclegan_unpaired': {
            'train_img': root / 'city_cycle_cityfoggy/JPEGImages_unpaired',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'city_cycle_cityfoggy/JPEGImages_unpaired',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'sim10k': {
            'train_img': root / 'sim10k/VOC2012/JPEGImages',
            'train_anno': root / 'sim10k/annotations/sim10k_caronly.json',
        },
        'bdd_daytime': {
            'train_img': root / 'bdd_daytime/train',
            'train_anno': root / 'bdd_daytime/annotations/bdd_daytime_train.json',
            'val_img': root / 'bdd_daytime/val',
            'val_anno': root / 'bdd_daytime/annotations/bdd_daytime_val.json',
        }
    }


class DADataset(Dataset):
    def __init__(self, source_img_folder, source_ann_file, target_img_folder, target_ann_file,
                 transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):

        # for source data
        self.source = CocoDetection(
            img_folder=source_img_folder,
            ann_file=source_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )

        # for target data
        self.target = CocoDetection(
            img_folder=target_img_folder,
            ann_file=target_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )

        self.data_domain_type = 'src+tgt'

    def __len__(self):
        return max(len(self.source), len(self.target))

    # def __getitem__(self, idx):
    #     source_img, source_target = self.source[idx % len(self.source)]
    #     # print(source_target)
    #     target_img, _ = self.target[idx % len(self.target)]
    #     return source_img, target_img, source_target

    def __getitem__(self, idx):
        source_img, source_target = self.source[idx % len(self.source)]
        target_img, target_target = self.target[idx % len(self.target)]

        return source_img, target_img, source_target, target_target


# the collate function combines source and target image samples
# the batch argument is a list of samples
def collate_fn(batch):
    # source_imgs, target_imgs, source_targets = list(zip(*batch)) # making it a list of tuple
    source_imgs, target_imgs, source_targets, target_targets = list(zip(*batch)) # making it a list of tuple
    samples = nested_tensor_from_tensor_list(source_imgs + target_imgs)
    
    targets = source_targets + target_targets

    # return samples, source_targets
    return samples, targets


class CrossDomainDADataset(Dataset):
    def __init__(
        self,
        source_img_folder, source_ann_file,
        target_img_folder, target_ann_file,
        target_like_source_img_folder,
        transforms, return_masks, cache_mode=False, local_rank=0, local_size=1
    ):

        # for source data
        self.source = CocoDetection(
            img_folder=source_img_folder,
            ann_file=source_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )

        # for target data
        self.target = CocoDetection(
            img_folder=target_img_folder,
            ann_file=target_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )

        # for target_like data
        self.target_like_source = CocoDetection(
            img_folder=target_like_source_img_folder,
            ann_file=source_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )

        self.data_domain_type = 'src+tgt-like+tgt'

        assert len(self.source) == len(self.target), (self.source, self.target)
        assert len(self.target) == len(self.target_like_source), (self.target, self.target_like_source)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        source_img, source_target = self.source[idx % len(self.source)]
        target_img, target_target = self.target[idx % len(self.target)]
        target_like_source_img, _ = self.target_like_source[idx % len(self.target_like_source)]

        return source_img, target_img, target_like_source_img, source_target, target_target


def modify_img_paths(dataset):
    if isinstance(dataset, CrossDomainDADataset):
        coco = dataset.target_like_source.coco
    elif isinstance(dataset, CocoDetection):
        coco = dataset.coco
    else:
        raise ValueError('Unknown type of dataset:', type(dataset))

    for i, img_info in enumerate(coco.dataset['images']):
        img_name = Path(img_info['file_name']).name
        coco.dataset['images'][i]['file_name'] = str(img_name)
    
    coco.createIndex()


def collate_fn_cross_domain(batch):
    source_imgs, target_imgs, target_like_source_imgs, source_targets, target_targets = list(zip(*batch))
    samples = nested_tensor_from_tensor_list(source_imgs + target_like_source_imgs + target_imgs)
    targets = source_targets + source_targets + target_targets

    return samples, targets


# wrapper
def build(image_set, cfg):
    paths = get_paths(cfg.DATASET.COCO_PATH)
    source_domain, target_domain = cfg.DATASET.DATASET_FILE.split('_to_')

    if image_set in ['val', 'val_target']:
        return CocoDetection(
            img_folder=paths[target_domain]['val_img'],
            ann_file=paths[target_domain]['val_anno'],
            transforms=make_coco_transforms(image_set),
            return_masks=cfg.MODEL.MASKS,
            cache_mode=cfg.CACHE_MODE,
            local_rank=get_local_rank(),
            local_size=get_local_size()
        )

    elif image_set == 'val_source':
        return CocoDetection(
            img_folder=paths[source_domain]['val_img'],
            ann_file=paths[source_domain]['val_anno'],
            transforms=make_coco_transforms(image_set),
            return_masks=cfg.MODEL.MASKS,
            cache_mode=cfg.CACHE_MODE,
            local_rank=get_local_rank(),
            local_size=get_local_size()
        )

    elif image_set == 'val_target_like_source':
        if cfg.DATASET.DA_MODE == 'cross_domain':
            cross_domain = 'city_instruction_cityscapes'
            return CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        elif cfg.DATASET.DA_MODE == 'cross_domain_cyclegan_paired':
            cross_domain = 'cyclegan_paired'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            modify_img_paths(dataset)
            return dataset
        
        elif cfg.DATASET.DA_MODE == 'cross_domain_cyclegan_unpaired':
            cross_domain = 'cyclegan_unpaired'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            modify_img_paths(dataset)
            return dataset
        else:
            raise ValueError(f'unknown DA_MODE {cfg.DATASET.DA_MODE}')

    elif image_set == 'train':
        if cfg.DATASET.DA_MODE == 'source_only':
            return CocoDetection(
                img_folder=paths[source_domain]['train_img'],
                ann_file=paths[source_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        elif cfg.DATASET.DA_MODE == 'oracle':
            return CocoDetection(
                img_folder=paths[target_domain]['train_img'],
                ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        elif cfg.DATASET.DA_MODE == 'uda':
            return DADataset(
                source_img_folder=paths[source_domain]['train_img'],
                source_ann_file=paths[source_domain]['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        elif cfg.DATASET.DA_MODE == 'cross_domain':
            assert source_domain == 'cityscapes', source_domain
            assert target_domain == 'foggy_cityscapes', target_domain

            return CrossDomainDADataset(
                source_img_folder=paths[source_domain]['train_img'],
                source_ann_file=paths[source_domain]['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                target_like_source_img_folder=paths['city_instruction_cityscapes']['train_img'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        elif cfg.DATASET.DA_MODE == 'cross_domain_cyclegan_paired':
            assert source_domain == 'cityscapes', source_domain
            assert target_domain == 'foggy_cityscapes', target_domain

            dataset = CrossDomainDADataset(
                source_img_folder=paths[source_domain]['train_img'],
                source_ann_file=paths[source_domain]['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                target_like_source_img_folder=paths['cyclegan_paired']['train_img'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            modify_img_paths(dataset)

            return dataset

        elif cfg.DATASET.DA_MODE == 'cross_domain_cyclegan_unpaired':
            assert source_domain == 'cityscapes', source_domain
            assert target_domain == 'foggy_cityscapes', target_domain

            dataset = CrossDomainDADataset(
                source_img_folder=paths[source_domain]['train_img'],
                source_ann_file=paths[source_domain]['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                target_like_source_img_folder=paths['cyclegan_unpaired']['train_img'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            modify_img_paths(dataset)

            return dataset

        else:
            raise ValueError(f'Unknown argument cfg.DATASET.DA_MODE {cfg.DATASET.DA_MODE}')
    raise ValueError(f'unknown image set {image_set}')
