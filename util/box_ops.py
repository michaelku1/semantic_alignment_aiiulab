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

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from PIL import Image
from torchvision.utils import save_image
import os


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def plot_results(info, img_id, img_path):

    '''
    info: [box, class index, predicted probabilities]
    '''

    # TODO: this will change depending on the label set defined in the json file (e.g will be indexed differently if
    # using a subset, or a single class)
    CLASSES = ['person','car','train','rider','truck','motorcycle','bicycle', 'bus'] # whole set
    # CLASSES = ['car', 'rider', 'bus'] # subset
#     CLASSES =  ['bus'] # single class

    img = Image.open(img_path)

    for key in list(info.keys()):
        boxes = info[key][0]
        class_idxs = info[key][1]
        preds = info[key][2]

        plt.figure(figsize=(16,10))
        plt.imshow(img)
        # image = mpimg.imread(p)
        ax = plt.gca()
        # colors = COLORS * 100
        for (xmin, ymin, xmax, ymax), p, class_idx in zip(boxes.tolist(), preds, class_idxs):

            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color='g', linewidth=1))

            text = f'{CLASSES[class_idx-1]}: {p:0.2f}'
            
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        city_name = str(img_path).split('/')[-2]
        plt.axis('off')

        # import pdb; pdb.set_trace()

        if key == 'pred':
            if not os.path.isdir('./box_plot_pred'):
                os.makedirs('./box_plot_pred', exist_ok = True)

            plt.savefig(f'./box_plot_pred/{city_name}_{img_id}.png')

        elif key == 'gt':
            if not os.path.isdir('./box_plot_gt'):
                os.makedirs('./box_plot_gt', exist_ok = True)

            plt.savefig(f'./box_plot_gt/{city_name}_{img_id}.png')

        
        # plt.axis('off')
        # plt.show()


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
