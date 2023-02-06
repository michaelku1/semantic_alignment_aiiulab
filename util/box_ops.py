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
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torchvision.ops.boxes import box_area


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


def plot_bbox(
    coco_gt,
    res: Dict,
    root_dir: Path,
    box_save_dir: str,
    score_threshold: int = 0.5,
    img_ids: Optional[List[int]] = None,
    prefix: Optional[str] = None
):
    box_save_dir = Path(box_save_dir)
    if not box_save_dir.exists():
        box_save_dir.mkdir()

    img_ids = img_ids if img_ids is not None else coco_gt.getImgIds()

    # BGR colors for all categories
    colors = [
        (47, 52, 227),
        (63, 153, 246),
        (74, 237, 255),
        (114, 193, 56),
        (181, 192, 77),
        (220, 144, 52),
        (205, 116, 101),
        (226, 97, 149),
        (155, 109, 246),
    ]

    fontFace = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.5
    thickness = 1

    for img_id, output in res.items():
        if img_id not in img_ids:
            continue
        
        info = coco_gt.loadImgs(img_id)[0]
        img_path = root_dir / info['file_name']
        img = cv2.imread(str(img_path))

        label_count = {i: 0 for i in range(1, 9)}
        for score, label, box in zip(output['scores'], output['labels'], output['boxes']):
            if score < score_threshold:
                continue

            label_name = coco_gt.cats[label.item()]['name']
            title = f'{label_name}({score.item():.3f})'
            label_count[label.item()] += 1

            x1, y1, x2, y2 = np.round(box.cpu().numpy()).astype(np.int)

            labelSize = cv2.getTextSize(title, fontFace, fontScale, thickness)
            color = colors[label.item()]

            # plot box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # plot text
            cv2.rectangle(img, (x1, y1), (x1 + labelSize[0][0] - 1, y1 - labelSize[0][1]), color, cv2.FILLED)  # text background
            cv2.putText(img, title, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, fontScale, (255, 255, 255), thickness)

        x, y = 20, 20
        for label, count in label_count.items():
            label_name = coco_gt.cats[label]['name']
            msg = f'# {label_name}: {count}'

            cv2.rectangle(img, (x, y + 5), (x + 200, y - 11), colors[label], cv2.FILLED)  # text background
            cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_COMPLEX, fontScale, (255, 255, 255), thickness)

            y += 11 + 5

        file_name = f'{prefix}_img-id={img_id}.png' if prefix else f'img-id={img_id}.png'
        save_path = Path(box_save_dir) / file_name
        cv2.imwrite(str(save_path), img)
