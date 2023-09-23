# ----------------------------------------------
# Created by Wei-Jie Huang
# ----------------------------------------------
import copy
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset

from datasets.coco import CocoDetection, make_coco_transforms, ConvertCocoPolysToMask
from util.misc import get_local_rank, get_local_size, nested_tensor_from_tensor_list


mixup_src_int_ratio = 0.0
mixup_src_tgt_ratio = 0.0


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
        'instruction': {
            'train_img': root / 'city_instruction_cityfoggy/leftImg8bit/train',
            'train_anno': root / 'city_instruction_cityfoggy/annotations/cityscapes_train.json',
            'val_img': root / 'city_instruction_cityfoggy/leftImg8bit/val',
            'val_anno': root / 'city_instruction_cityfoggy/annotations/cityscapes_val.json',
        },
        'heavy_instruction': {
            'train_img': root / 'city_instruction_cityfoggy/leftImg8bit_heavy/train',
            'train_anno': root / 'city_instruction_cityfoggy/annotations/cityscapes_train.json',
            'val_img': root / 'city_instruction_cityfoggy/leftImg8bit_heavy/val',
            'val_anno': root / 'city_instruction_cityfoggy/annotations/cityscapes_val.json',
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
        'mixup0.5_instruction': {
            'train_img': root / 'city_instruction_cityfoggy/mixup0.5_leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'city_instruction_cityfoggy/mixup0.5_leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'mixup0.7_instruction': {
            'train_img': root / 'city_instruction_cityfoggy/mixup0.7_leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'city_instruction_cityfoggy/mixup0.7_leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'mixup0.7_heavy_instruction': {
            'train_img': root / 'city_instruction_cityfoggy/mixup0.7_leftImg8bit_heavy/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'city_instruction_cityfoggy/mixup0.7_leftImg8bit_heavy/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'mixup0.8_heavy_instruction': {
            'train_img': root / 'city_instruction_cityfoggy/mixup0.8_leftImg8bit_heavy/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'city_instruction_cityfoggy/mixup0.8_leftImg8bit_heavy/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'mixup0.9_heavy_instruction': {
            'train_img': root / 'city_instruction_cityfoggy/mixup0.9_leftImg8bit_heavy/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'city_instruction_cityfoggy/mixup0.9_leftImg8bit_heavy/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'c2b_instruction': {
            'train_img': root / 'cityscapes_instruction_bdd/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'cityscapes_instruction_bdd/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'mixup0.7_c2b_instruction': {
            'train_img': root / 'cityscapes_instruction_bdd/mixup0.7_leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'cityscapes_instruction_bdd/mixup0.7_leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'sim10k': {
            'train_img': root / 'sim10k/VOC2012/JPEGImages',
            'train_anno': root / 'sim10k/annotations/sim10k_caronly.json',
        },
        'sim10k_instruction': {
            'train_img': root / 'sim_instruction_cityscapes/VOC2012/JPEGImages',
            'train_anno': root / 'sim10k/annotations/sim10k_caronly.json',
        },
        'bdd_daytime': {
            'train_img': root / 'bdd_daytime/train',
            'train_anno': root / 'bdd_daytime/annotations/bdd_daytime_train.json',
            'val_img': root / 'bdd_daytime/val',
            'val_anno': root / 'bdd_daytime/annotations/bdd_daytime_val.json',
        },
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
        cross_img_folder,
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
        self.cross = CocoDetection(
            img_folder=cross_img_folder,
            ann_file=source_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )

        self.data_domain_type = 'src+cross+tgt'

        assert len(self.source) == len(self.target), (self.source, self.target)
        assert len(self.target) == len(self.cross), (self.target, self.cross)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        source_img, source_target = self.source[idx % len(self.source)]
        target_img, target_target = self.target[idx % len(self.target)]
        cross_img, cross_target = self.cross[idx % len(self.cross)]

        return source_img, target_img, cross_img, source_target, target_target, cross_target


def modify_img_paths(dataset):
    if isinstance(dataset, CrossDomainDADataset):
        coco = dataset.cross.coco
    elif isinstance(dataset, DADataset):
        coco = dataset.source.coco
    elif isinstance(dataset, CocoDetection):
        coco = dataset.coco
    else:
        raise ValueError('Unknown type of dataset:', type(dataset))

    for i, img_info in enumerate(coco.dataset['images']):
        img_name = Path(img_info['file_name']).name
        coco.dataset['images'][i]['file_name'] = str(img_name)
    
    coco.createIndex()


def collate_fn_cross_domain(batch):
    source_imgs, target_imgs, cross_imgs, source_targets, target_targets, cross_targets = list(zip(*batch))
    samples = nested_tensor_from_tensor_list(source_imgs + cross_imgs + target_imgs)
    targets = source_targets + cross_targets + target_targets

    return samples, targets


class MixupDADataset(Dataset):
    def __init__(
        self,
        src_img_folder, src_ann_file,
        tgt_img_folder, tgt_ann_file,
        int_img_folder, int_domain_label,
        transforms, return_masks,
        cache_mode=False, local_rank=0, local_size=1
    ):
        self.src_img_folder = src_img_folder
        self.src_ann_file = src_ann_file
        self.tgt_img_folder = tgt_img_folder
        self.tgt_ann_file = tgt_ann_file
        self.int_img_folder = int_img_folder
        self.int_domain_label = int_domain_label

        self._transforms = transforms
        self.return_masks = return_masks

        self.src = CocoDetection(
            img_folder=src_img_folder,
            ann_file=src_ann_file,
            transforms=None,  # make sure the outputted image is not transformed
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
            to_tensor_first=True
        )

        self.tgt = CocoDetection(
            img_folder=tgt_img_folder,
            ann_file=tgt_ann_file,
            transforms=None,  # make sure the outputted image is not transformed
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
            to_tensor_first=True
        )

        self.int = CocoDetection(
            img_folder=int_img_folder,
            ann_file=src_ann_file,
            transforms=None,  # make sure the outputted image is not transformed
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
            to_tensor_first=True
        )

        assert len(self.src) == len(self.int), f'#src={len(self.src)}, #int={len(self.int)}'
        assert self.src.ids == self.int.ids

    def __len__(self):
        return max(len(self.src), len(self.tgt))

    def __getitem__(self, idx):
        src_img, src_target = self.src[idx % len(self.src)]
        tgt_img, tgt_target = self.tgt[idx % len(self.tgt)]
        int_img, int_target = self.int[idx % len(self.int)]

        # target_x: {
        #     'image_id': 0, 
        #     'annotations': [{
        #         'id': 0,  # annotation id, 與 key 值相同
        #         'image_id': 0,  # image id
        #         'segmentation': [[1064, 449, 1064, ...]],
        #         'category_id': 1,
        #         'iscrowd': 0,
        #         'area': 763,
        #         'bbox': [043, 394, 28, 64]
        #         },
        #     ]
        # }

        return src_img, int_img, tgt_img, src_target, tgt_target

    def transform(self, img, target):
        img, target = self._transforms(img, target)

        return img, target


def collate_fn_mixup(batch):
    src_imgs, tgt_imgs, int_imgs, src_targets, tgt_targets = list(zip(*batch))
    samples = nested_tensor_from_tensor_list(src_imgs + int_imgs + tgt_imgs)
    targets = src_targets + tgt_targets

    return samples, targets


# wrapper
def build(image_set, cfg):
    paths = get_paths(cfg.DATASET.COCO_PATH)
    source_domain, target_domain = cfg.DATASET.DATASET_FILE.split('_to_')

    if 'bdd' in target_domain:
        assert cfg.DATASET.CATEGORY_IDS == [1, 2, 3, 4, 5, 7, 8], 'BDD has no category `train`'
    
    if image_set in ['val', 'val_target']:
        print(f'Val target: {target_domain}')
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
        # sim10k has only training set, no validation set
        if source_domain == 'sim10k':
            return None

        print(f'Val source: {source_domain}')
        return CocoDetection(
            img_folder=paths[source_domain]['val_img'],
            ann_file=paths[source_domain]['val_anno'],
            transforms=make_coco_transforms(image_set),
            return_masks=cfg.MODEL.MASKS,
            cache_mode=cfg.CACHE_MODE,
            local_rank=get_local_rank(),
            local_size=get_local_size()
        )

    elif image_set == 'val_cross':
        if cfg.DATASET.DA_MODE == 'cross_domain_instruction':
            cross_domain = 'instruction'
            print(f'Val cross: {cross_domain}')
            return CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'uda':
            raise ValueError('The cross dataset in uda mode is undefined')

        elif cfg.DATASET.DA_MODE == 'uda_cyclegan_paired_target':
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

        elif cfg.DATASET.DA_MODE == 'uda_cyclegan_unpaired_target':
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

        elif cfg.DATASET.DA_MODE == 'uda_instruction_target':
            cross_domain = 'instruction'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'uda_heavy_instruction_target':
            cross_domain = 'heavy_instruction'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'uda_mixup0.7_instruction_target':
            cross_domain = 'mixup0.7_instruction'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'uda_mixup0.5_instruction_target':
            cross_domain = 'mixup0.5_instruction'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'uda_mixup0.7_heavy_instruction_target':
            cross_domain = 'mixup0.7_heavy_instruction'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'uda_mixup0.8_heavy_instruction_target':
            cross_domain = 'mixup0.8_heavy_instruction'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'uda_mixup0.9_heavy_instruction_target':
            cross_domain = 'mixup0.9_heavy_instruction'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'uda_mixup0.7_c2b_instruction_target':
            cross_domain = 'mixup0.7_c2b_instruction'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'uda_sim10k_instruction_target':
            cross_domain = None  # sim10k has no validation set
            dataset = None

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
        
        elif cfg.DATASET.DA_MODE == 'cross_domain_mixup0.5_instruction':
            cross_domain = 'mixup0.5_instruction'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        
        elif cfg.DATASET.DA_MODE == 'cross_domain_mixup0.7_instruction':
            cross_domain = 'mixup0.7_instruction'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        
        elif cfg.DATASET.DA_MODE == 'cyclegan_paired_only':
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

        elif cfg.DATASET.DA_MODE == 'mixup_c2fc_heavy_instruction':
            cross_domain = 'heavy_instruction'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'mixup_c2b_instruction':
            cross_domain = 'c2b_instruction'
            dataset = CocoDetection(
                img_folder=paths[cross_domain]['val_img'],
                ann_file=paths[cross_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'mixup_s2c_instruction':
            cross_domain = None  # sim10k has no validation set
            dataset = None

        else:
            raise ValueError(f'unknown DA_MODE {cfg.DATASET.DA_MODE}')

        print(f'Val cross: {cross_domain}')
        return dataset

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
            dataset = DADataset(
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
            dataset[0]

        elif cfg.DATASET.DA_MODE == 'uda_cyclegan_paired_target':
            dataset = DADataset(
                source_img_folder=paths['cyclegan_paired']['train_img'],
                source_ann_file=paths['cyclegan_paired']['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            modify_img_paths(dataset)
            return dataset

        elif cfg.DATASET.DA_MODE == 'uda_cyclegan_unpaired_target':
            dataset = DADataset(
                source_img_folder=paths['cyclegan_paired']['train_img'],
                source_ann_file=paths['cyclegan_paired']['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            modify_img_paths(dataset)
            return dataset

        elif cfg.DATASET.DA_MODE == 'uda_instruction_target':
            print('Train source: instruction')
            print(f'Train target: {target_domain}')
            dataset = DADataset(
                source_img_folder=paths['instruction']['train_img'],
                source_ann_file=paths['instruction']['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            return dataset

        elif cfg.DATASET.DA_MODE == 'uda_heavy_instruction_target':
            print('Train source: heavy_instruction')
            print(f'Train target: {target_domain}')
            dataset = DADataset(
                source_img_folder=paths['heavy_instruction']['train_img'],
                source_ann_file=paths['heavy_instruction']['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            return dataset

        elif cfg.DATASET.DA_MODE == 'uda_mixup0.7_instruction_target':
            dataset = DADataset(
                source_img_folder=paths['mixup0.7_instruction']['train_img'],
                source_ann_file=paths['mixup0.7_instruction']['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            return dataset

        elif cfg.DATASET.DA_MODE == 'uda_mixup0.5_instruction_target':
            dataset = DADataset(
                source_img_folder=paths['mixup0.5_instruction']['train_img'],
                source_ann_file=paths['mixup0.5_instruction']['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            return dataset

        elif cfg.DATASET.DA_MODE == 'uda_mixup0.7_heavy_instruction_target':
            dataset = DADataset(
                source_img_folder=paths['mixup0.7_heavy_instruction']['train_img'],
                source_ann_file=paths['mixup0.7_heavy_instruction']['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            return dataset

        elif cfg.DATASET.DA_MODE == 'uda_mixup0.8_heavy_instruction_target':
            dataset = DADataset(
                source_img_folder=paths['mixup0.8_heavy_instruction']['train_img'],
                source_ann_file=paths['mixup0.8_heavy_instruction']['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            return dataset

        elif cfg.DATASET.DA_MODE == 'uda_mixup0.9_heavy_instruction_target':
            dataset = DADataset(
                source_img_folder=paths['mixup0.9_heavy_instruction']['train_img'],
                source_ann_file=paths['mixup0.9_heavy_instruction']['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            return dataset

        elif cfg.DATASET.DA_MODE == 'uda_mixup0.7_c2b_instruction_target':
            dataset = DADataset(
                source_img_folder=paths['mixup0.7_c2b_instruction']['train_img'],
                source_ann_file=paths['mixup0.7_c2b_instruction']['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            return dataset

        elif cfg.DATASET.DA_MODE == 'uda_sim10k_instruction_target':
            dataset = DADataset(
                source_img_folder=paths['sim10k_instruction']['train_img'],
                source_ann_file=paths['sim10k_instruction']['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            return dataset

        elif cfg.DATASET.DA_MODE == 'cross_domain_instruction':
            assert source_domain == 'cityscapes', source_domain
            assert target_domain == 'foggy_cityscapes', target_domain
            cross_domain = 'instruction'

            return CrossDomainDADataset(
                source_img_folder=paths[source_domain]['train_img'],
                source_ann_file=paths[source_domain]['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                cross_img_folder=paths[cross_domain]['train_img'],
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
                cross_img_folder=paths['cyclegan_paired']['train_img'],
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
                cross_img_folder=paths['cyclegan_unpaired']['train_img'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            modify_img_paths(dataset)
            return dataset

        elif cfg.DATASET.DA_MODE == 'cross_domain_mixup0.5_instruction':
            assert source_domain == 'cityscapes', source_domain
            assert target_domain == 'foggy_cityscapes', target_domain

            dataset = CrossDomainDADataset(
                source_img_folder=paths[source_domain]['train_img'],
                source_ann_file=paths[source_domain]['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                cross_img_folder=paths['mixup0.5_instruction']['train_img'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

            return dataset

        elif cfg.DATASET.DA_MODE == 'cross_domain_mixup0.7_instruction':
            assert source_domain == 'cityscapes', source_domain
            assert target_domain == 'foggy_cityscapes', target_domain

            dataset = CrossDomainDADataset(
                source_img_folder=paths[source_domain]['train_img'],
                source_ann_file=paths[source_domain]['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                cross_img_folder=paths['mixup0.7_instruction']['train_img'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

            return dataset

        elif cfg.DATASET.DA_MODE == 'cyclegan_paired_only':
            dataset = CocoDetection(
                img_folder=paths['cyclegan_paired']['train_img'],
                ann_file=paths['cyclegan_paired']['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            modify_img_paths(dataset)
            return dataset

        elif cfg.DATASET.DA_MODE == 'mixup_c2fc_heavy_instruction':
            print('Train source: cityscapes')
            print('Train Interpolation: heavy_instruction')
            print(f'Train target: {target_domain}')
            dataset = MixupDADataset(
                src_img_folder=paths['cityscapes']['train_img'],
                src_ann_file=paths['cityscapes']['train_anno'],
                tgt_img_folder=paths['foggy_cityscapes']['train_img'],
                tgt_ann_file=paths['foggy_cityscapes']['train_anno'],
                int_img_folder=paths['heavy_instruction']['train_img'],
                int_domain_label=1,
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            return dataset

        elif cfg.DATASET.DA_MODE == 'mixup_c2b_instruction':
            dataset = MixupDADataset(
                src_img_folder=paths['cityscapes']['train_img'],
                src_ann_file=paths['cityscapes']['train_anno'],
                tgt_img_folder=paths['bdd_daytime']['train_img'],
                tgt_ann_file=paths['bdd_daytime']['train_anno'],
                int_img_folder=paths['c2b_instruction']['train_img'],
                int_domain_label=1,
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            return dataset

        elif cfg.DATASET.DA_MODE == 'mixup_s2c_instruction':
            dataset = MixupDADataset(
                src_img_folder=paths[source_domain]['train_img'],
                src_ann_file=paths[source_domain]['train_anno'],
                tgt_img_folder=paths[target_domain]['train_img'],
                tgt_ann_file=paths[target_domain]['train_anno'],
                int_img_folder=paths['sim10k_instruction']['train_img'],
                int_domain_label=1,
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
            return dataset

        else:
            raise ValueError(f'Unknown argument cfg.DATASET.DA_MODE {cfg.DATASET.DA_MODE}')
    raise ValueError(f'unknown image set {image_set}')
