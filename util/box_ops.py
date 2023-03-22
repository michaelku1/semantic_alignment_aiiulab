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

import os
import torch
from torchvision.ops.boxes import box_area
from PIL import Image
from matplotlib import pyplot as plt 


def plot_results_eval(model_name, info, img_id, img_path):

    '''
    This is for plotting boxes for evaluation mode, which read an image from an image path

    img_path: use path to read image
    info: [box, class index, predicted probabilities]
    '''

    CLASSES = ['person','car','train','rider','truck','motorcycle','bicycle', 'bus'] # whole set

    img = Image.open(img_path)

    for key in list(info.keys()):
        boxes = info[key][0]
        class_idxs = info[key][1]
        preds = info[key][2]

        plt.figure(figsize=(16,10))
        plt.imshow(img)
        # image = mpimg.imread(p)
        ax = plt.gca()

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
            if not os.path.isdir(f'./{model_name}/box_plot_pred'):
                os.makedirs(f'./{model_name}/box_plot_pred', exist_ok = True)

            plt.savefig(f'./{model_name}/box_plot_pred/{city_name}_{img_id}.png')

        elif key == 'gt':
            if not os.path.isdir(f'./{model_name}/box_plot_gt'):
                os.makedirs(f'./{model_name}/box_plot_gt', exist_ok = True)

            plt.savefig(f'./{model_name}/box_plot_gt/{city_name}_{img_id}.png')

        
        # plt.axis('off')
        # plt.show()
        plt.close()


def plot_results_train(model_name, info, img_ids, img_paths, samples):

    '''
    This is for plotting boxes for training mode, which uses unmasked samples
    for visualization, note that this done for batch of images

    info: {pred: [box, class index, predicted probabilities],
             gt: [box, class index, predicted probabilities],} 

    where pred and gt contains lists of tensors
    
    img_path: list of image paths
    samples: list of unmasked image tensors
    '''

    CLASSES = ['person','car','train','rider','truck','motorcycle','bicycle', 'bus']
    
    B = len(samples)
    # breakpoint()
    # for predicted results and grountruths
    for key in list(info.keys()):
        boxes = info[key][0]
        class_idxs = info[key][1]
        preds = info[key][2]    

        plt.figure(figsize=(16,10))
        # plt.imshow(samples_i)
        # ax = plt.gca()
        
        # for each sample
        for i in range(B):
            boxes_i = boxes[i]
            preds_i = preds[i]
            class_idxs_i = class_idxs[i]
            img_paths_i = img_paths[i]
            samples_i = samples[i]
            img_ids_i = img_ids[i]

            plt.imshow(samples_i)
            ax = plt.gca()

            # breakpoint()
            # for each box
            for box, p, cls_idx in zip(boxes_i.tolist(), preds_i.tolist(), class_idxs_i):

                xmin, ymin, xmax, ymax = box                
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color='g', linewidth=1))

                text = f'{CLASSES[cls_idx-1]}: {p:0.2f}'
                ax.text(xmin, ymin, text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))


            city_name = str(img_paths_i).split('/')[-2]
            plt.axis('off')
            # breakpoint()
            # import pdb; pdb.set_trace()

            if i//(B//2)==0:
                domain_name = 'src'
            else:
                domain_name = 'tgt'
            
            if key == 'pred':
                if not os.path.isdir(f'./{model_name}_images/box_plot_pred_{domain_name}'):
                    os.makedirs(f'./{model_name}_images/box_plot_pred_{domain_name}', exist_ok = True)

                plt.savefig(f'./{model_name}_images/box_plot_pred_{domain_name}/{city_name}_{img_ids_i}.png')

            elif key == 'gt':
                if not os.path.isdir(f'./{model_name}_images/box_plot_gt_{domain_name}'):
                    os.makedirs(f'./{model_name}_images/box_plot_gt_{domain_name}', exist_ok = True)

                plt.savefig(f'./{model_name}_images/box_plot_gt/_{domain_name}{city_name}_{img_ids_i}.png')

            plt.axis('off')
            plt.close()



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
    # try:
    #     assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # except AssertionError:
    #     import pdb; pdb.set_trace()

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
