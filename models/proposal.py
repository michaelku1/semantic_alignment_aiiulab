import cv2
import numpy as np
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

from util.box_ops import box_cxcywh_to_xyxy
from util.plot_utils import tensor_to_cv2


def get_src_class_rois(
    src_feat_map: torch.Tensor,
    src_targets: List[Dict],
    num_classes: int,
    roi_output_shape: Union[int, tuple],
    roi_scale: float,
    debug: bool = False,
    **kwargs
):
    # TODO: only use the encoder outputed feature of the last scale to get source prototypes

    # src_feat_map: (N, H, W, C)
    assert isinstance(src_feat_map, torch.Tensor), '`src_feat_map` is a tensor with shape of (N, C, H, W)'
    
    # only support single scale now
    if src_feat_map.ndim != 4:
        raise NotImplementedError

    assert len(src_feat_map) == len(src_targets)

    if isinstance(roi_output_shape, int):
        roi_output_shape = roi_output_shape, roi_output_shape
    assert isinstance(roi_output_shape, tuple) and len(roi_output_shape) == 2

    C = src_feat_map.shape[1]

    class_rois = {cls_idx: [] for cls_idx in range(1, num_classes + 1)}
    for feat, target in zip(src_feat_map, src_targets):
        labels = target['labels']  # (#box_per_img,)
        boxes = target['boxes']  # (#box_per_img, 4)
        assert len(labels) == len(boxes)

        boxes = box_cxcywh_to_xyxy(boxes)  # (#box_per_img, 4)

        img_h, img_w = target['size'].unbind(0)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)  # (4,)
        boxes = boxes * scale_fct[None, :]

        # already checked the unscaled `boxes` is correct
        if debug:
            assert 'samples' in kwargs, 'Need argument `samples` for debug'
            samples = kwargs['samples']
            img: np.ndarray = tensor_to_cv2(samples.tensors[0], unnormalize=True)  # (H, W, C), input a signle img

            for box in boxes:
                x1, y1, x2, y2 = np.round(box.cpu().numpy()).astype(np.int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  

            cv2.imwrite('./get_source_proposals.png', img)
            print('saved img to ./get_source_proposals.png')
            import pdb; pdb.set_trace()

        class_boxes = {cls_idx: [] for cls_idx in class_rois.keys()}
        for label, box in zip(labels, boxes):
            class_boxes[label.item()].append(box)

        for label, boxes in class_boxes.items():
            if len(boxes) == 0:
                continue  # `box` is empty, pass it
                
            boxes = torch.stack(boxes, dim=0)
            rois = roi_align(feat.unsqueeze(0), [boxes], output_size=roi_output_shape, spatial_scale=roi_scale, aligned=True)  # (#box_per_img, C, roi_h, roi_w)
            class_rois[label].append(rois)
    
    for cls_idx, roi_list in class_rois.items():
        if not roi_list:
            continue  # `roi_list` is empty, pass it
        
        if len(roi_list) == 1:
            # when there is only one source feature map,
            # we don't use `torch.stack(roi_list)` directly
            # because the result shape will be (1, 1, C, roi_h, roi_w)
            rois = roi_list[0]  # (#box_in_batch, C, roi_h, roi_w)
        else:
            rois = torch.stack(roi_list, dim=0)  # (#cls_box_in_batch, C, roi_h, roi_w)

        class_rois[cls_idx] = rois

    assert all([len(rois) == len(class_boxes[cls_idx]) for cls_idx, rois in class_rois.items()])

    return class_rois


def agg_src_rois_to_prototypes(src_class_rois, device):
    class_prototypes = {cls_idx: None for cls_idx in src_class_rois.keys()}
    
    for cls_idx, cls_rois in src_class_rois.items():
        if len(cls_rois) == 0:
            continue  # `cls_rois` is empty, pass it

        assert cls_rois.ndim == 4
        
        cls_rois = cls_rois.mean(dim=(2, 3))  # (#box_in_batch, C)
        cls_rois = F.normalize(cls_rois, dim=-1)  # TODO: normalize along chennel
        cls_rois = cls_rois.mean(0)  # (C,), average across images in a batch
        class_prototypes[cls_idx] = cls_rois
            
    return class_prototypes


def get_tgt_class_rois(
    tgt_feat_map: torch.Tensor,
    tgt_output_logit: torch.Tensor,
    tgt_output_coord: torch.Tensor,
    prob_threshold: float = 0.8,
    roi_output_shape: Union[int, tuple],
    roi_scale: float,
    debug: bool = False,
    **kwargs
):
    # TODO: only use the encoder outputed feature of the last scale to get target prototypes

    # tgt_feat_map: (N, C, H, W)
    assert isinstance(tgt_feat_map, torch.Tensor), '`tgt_feat_map` is a tensor with shape of (N, C, H, W)'
    assert isinstance(tgt_output_logit, torch.Tensor), '`tgt_output_logit` is a tensor with shape of (N, #query, #class)'
    assert isinstance(tgt_output_coord, torch.Tensor), '`tgt_output_coord` is a tensor with shape of (N, #query, 4)'
    
    # only support single scale now
    if tgt_feat_map.ndim != 4:
        raise NotImplementedError

    assert len(tgt_feat_map) == len(tgt_output_logit) == len(tgt_output_coord)

    if isinstance(roi_output_shape, int):
        roi_output_shape = roi_output_shape, roi_output_shape
    assert isinstance(roi_output_shape, tuple) and len(roi_output_shape) == 2

    prob = tgt_output_logit.sigmoid()  # (N, #query, #class)
    img_idx, query_idx, cls_idx = torch.nonzero(porb > prob_threshold, as_tuple=True)
