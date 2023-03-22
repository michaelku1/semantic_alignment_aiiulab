import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from datasets.coco import CocoDetection, make_coco_transforms
from util.misc import get_local_rank, get_local_size, nested_tensor_from_tensor_list


# def get_paths(root):
#     root = Path(root)
#     return {
#         'cityscapes': {
#             'train_img': root / 'cityscapes/leftImg8bit/train',
#             'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
#             'val_img': root / 'cityscapes/leftImg8bit/val',
#             'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
#         },
#         'cityscapes_caronly': {
#             'train_img': root / 'cityscapes/leftImg8bit/train',
#             'train_anno': root / 'cityscapes/annotations/cityscapes_caronly_train.json',
#             'val_img': root / 'cityscapes/leftImg8bit/val',
#             'val_anno': root / 'cityscapes/annotations/cityscapes_caronly_val.json',
#         },
#         'foggy_cityscapes': {
#             'train_img': root / 'cityscapes/leftImg8bit_foggy/train',
#             'train_anno': root / 'cityscapes/annotations/foggy_cityscapes_train.json',
#             'val_img': root / 'cityscapes/leftImg8bit_foggy/val',
#             'val_anno': root / 'cityscapes/annotations/foggy_cityscapes_val.json',
#         },
#         'sim10k': {
#             'train_img': root / 'sim10k/VOC2012/JPEGImages',
#             'train_anno': root / 'sim10k/annotations/sim10k_caronly.json',
#         },
#         'bdd_daytime': {
#             'train_img': root / 'bdd_daytime/train',
#             'train_anno': root / 'bdd_daytime/annotations/bdd_daytime_train.json',
#             'val_img': root / 'bdd_daytime/val',
#             'val_anno': root / 'bdd_daytime/annotations/bdd_daytime_val.json',
#         }
#     }

# TODO subset of cityscapes
# def get_paths(root):
#     root = Path(root)
#     return {
#         'cityscapes': {
#             'train_img': root / 'cityscapes_subset/leftImg8bit/train',
#             'train_anno': root / 'cityscapes_subset/annotations/train_subset.json',
#             'val_img': root / 'cityscapes_subset/leftImg8bit/val',
#             'val_anno': root / 'cityscapes_subset/annotations/cityscapes_val.json',
#         },
#         'foggy_cityscapes': {
#             'train_img': root / 'cityscapes_subset/leftImg8bit_foggy/train',
#             'train_anno': root / 'cityscapes_subset/annotations/train_subset.json',
#             'val_img': root / 'cityscapes_subset/leftImg8bit_foggy/val',
#             'val_anno': root / 'cityscapes_subset/annotations/val_subset_foggy.json',
#         },

#     }

# TODO single class experiments
def get_paths(root):
    root = Path(root)
    return {
        'cityscapes': {
            'train_img': root / 'cityscapes/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/train_subset_single_cls.json',
            'val_img': root / 'cityscapes/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },

        'foggy_cityscapes': {
            'train_img': root / 'cityscapes/leftImg8bit_foggy/train',
            'train_anno': root / 'cityscapes/annotations/train_subset_foggy_single_cls.json',
            'val_img': root / 'cityscapes/leftImg8bit_foggy/val',
            'val_anno': root / 'cityscapes/annotations/val_subset_foggy_single_cls.json',
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

    def __len__(self):
        return max(len(self.source), len(self.target))

    def __getitem__(self, idx):
        source_img, source_target = self.source[idx % len(self.source)]
        # print(source_target)
        target_img, _ = self.target[idx % len(self.target)]
        return source_img, target_img, source_target


# the collate function combines source and target image samples
# the batch argument is a list of samples
def collate_fn(batch):
    # source_imgs, target_imgs, source_targets = list(zip(*batch)) # making it a list of tuple

    source_imgs, target_imgs, source_targets = list(zip(*batch)) # making it a list of tuple

    # print(source_targets)
    # quit()

    # classes = [3]

    # TODO restrict classes and their boxes
    # new_list = []

    # count = 0
    # for sample in source_targets:
    #     if len(sample['labels']) == 0:
    #         count +=1
    #         continue
    #     labels_list = []
    #     boxes_list = []
    #     area_list = []
    #     iscrowd_list = []
    #     labels = sample['labels']

    #     # if len(torch.where(labels==3))==0:
    #     #     continue

    #     for i,v in enumerate(labels.numpy()):
    #         for cls_idx in classes:
    #             if v == cls_idx:
    #                 labels_list.append(cls_idx)
    #                 boxes_list.append(sample['boxes'][i].numpy())
    #                 area_list.append(sample['area'][i])
    #                 iscrowd_list.append(sample['iscrowd'][i])

    #     sample.update({'labels': torch.as_tensor(labels_list)})
    #     sample.update({'boxes': torch.as_tensor(np.array(boxes_list))})
    #     sample.update({'area': torch.as_tensor(area_list)})
    #     sample.update({'iscrowd': torch.as_tensor(iscrowd_list)})

    #     new_list.append(sample)

    # source_targets_new = tuple(new_list)

    samples = nested_tensor_from_tensor_list(source_imgs + target_imgs)

    # TODO return targets with limited classes
    # return samples, source_targets_new
    
    return samples, source_targets


# wrapper
def build(image_set, cfg):
    paths = get_paths(cfg.DATASET.COCO_PATH)
    source_domain, target_domain = cfg.DATASET.DATASET_FILE.split('_to_')


    if image_set == 'val':
        return CocoDetection(
            img_folder=paths[target_domain]['val_img'],
            ann_file=paths[target_domain]['val_anno'],
            transforms=make_coco_transforms(image_set),
            return_masks=cfg.MODEL.MASKS,
            cache_mode=cfg.CACHE_MODE,
            local_rank=get_local_rank(),
            local_size=get_local_size()
        )
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
        else:
            raise ValueError(f'Unknown argument cfg.DATASET.DA_MODE {cfg.DATASET.DA_MODE}')
    raise ValueError(f'unknown image set {image_set}')