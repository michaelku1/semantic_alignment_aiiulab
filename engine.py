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
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from util.box_ops import plot_results, plot_results_train
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, total_epoch:int, cur_epoch: int,
                    base_ds, postprocessors, postprocessors_target, visualize_image_ids,
                    max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next() # samples have been transformed
    
    # # ### testing whether samples and targets are matched
    # root = data_loader.dataset.target.root
    # # TODO image from gt
    # img_id = targets[1]['image_id'].item()
    # # put channel to last dim
    # # image from loaded sample
    # target_sample = samples.tensors[1].permute(1,2,0).cpu()
    # img_path = data_loader.dataset.target.coco.loadImgs(img_id)[0]['file_name']
    # full_path = root/img_path
    
    # ### testing whether groundturth boxes have also been transformed
    # # target gts
    # # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0) # original image sizes
    # orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0) # need to use the transformed sizes
    # label_gt = targets[1]['labels']
    # prob_gt = torch.ones(label_gt.size())
    
    # # normalized boxes, duplicate outputs, returns list of dictionaries
    # boxes_gt = postprocessors_target['bbox'](targets[1], orig_target_sizes)[1]['boxes'] # same labels for targets[0] and targets[1] 

    # info_all = {}
    # info_all['gt'] = [boxes_gt, label_gt, prob_gt]

    # # plot results for testing
    # plot_results_train(info_all, img_id, full_path, target_sample)
    # import pdb; pdb.set_trace()

    data_loader_len = len(data_loader)
    # total_iter = data_loader_len*total_epoch
    
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    ### value on the left is current and value on the right is smoothed value
    thresh_record = []
    thresh_tmp_list = []
    missing_source = 0
    for iter in metric_logger.log_every(range(data_loader_len), print_freq, header):
        # if len(targets[0]['labels'])==0:
        #     import pdb; pdb.set_trace()
        #     outputs = model(samples, targets, cur_epoch, total_epoch)

        # cur_iter += iter # update iter

        ### TODO visualize images
        # unpadded_samples = samples.tensors[0][:, :targets[0]['size'][0], :targets[0]['size'][1]]
        # average_image = unpadded_samples.mean(0)
        # plt.figure(figsize=(30, 50))
        # plt.imshow(average_image.detach().cpu())
        # plt.axis('off')
        # plt.savefig(str('./visualization/image/image_{}.png').format(targets[0]['image_id'].cpu().item()), bbox_inches='tight')
        # plt.close()

        # BUG counting number of missing source targets
        for t in targets:
            if t['labels'].nelement() == 0:
                # import pdb; pdb.set_trace()
                missing_source += 1
                continue

        outputs = model(samples, targets, cur_epoch, total_epoch)


        # TODO testing
        if 'thresh' in outputs:
            thresh_record.append(outputs['thresh'])

        if 'thresh_change_occurence' in outputs:
            thresh_tmp_list.append(outputs['thresh_change_occurence'])


        # image_id_list = [t['image_id'] for t in targets]
        # target_image_ids_list = image_id_list[len(samples.tensors)//2:]
        # label_gt_list = [t['labels'] for t in targets]
        # target_label_gt_list = label_gt_list[len(samples.tensors)//2:]

        # import pdb; pdb.set_trace()
        # for image_id in target_image_ids_list:
        #     if image_id in visualize_image_ids:
        #         target_samples = samples.tensors[len(samples.tensors)//2:]
        #         root = data_loader.dataset.target.root # PosixPath
        #         # use sample to check whether the rescaled boxes have been correctly plotted
        #         target_sample_permute = target_samples.permute(1,2,0).cpu()
        #         # get image path
        #         img_path = data_loader.dataset.target.coco.loadImgs(image_id)[0]['file_name']
        #         # pred results
        #         full_path = root/img_path
        #         # target gts
        #         # label_gt = targets[1]['labels']
        #         # prob_gt = torch.ones(label_gt.size())
        #         prob_gt_list = [torch.ones(label_gt.size()) for label_gt in target_label_gt_list]

        #         # normalized boxes, duplicate outputs
        #         boxes_gt = postprocessors_target['bbox'](targets[len(samples.tensors)//2:], orig_target_sizes)[1]['boxes'] # same labels for targets[0] and targets[1] 

        #         keep = torch.arange(boxes_gt.size(0)).numpy() # keep all?
        #         results = postprocessors['bbox'](outputs, orig_target_sizes) # (bs, ...)
        #         boxes_pred = results[1]['boxes']
        #         label_pred = results[1]['labels']
        #         prob_pred = results[1]['scores']

        #         info_all = {}
        #         info_all['pred'] = [boxes_pred, label_pred, prob_pred]
        #         info_all['gt'] = [boxes_gt, label_gt, prob_gt]

        #         # TODO plot both grountruth and predicted boxes
        #         plot_results_train(info_all, image_id, full_path, target_sample_permute)

        # debug mode draws plots
        if type(outputs)==tuple:
            orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            out, features, memory, hs,  = outputs
            # B = samples.tensors.shape[0]
            # cur_image_id = targets[:B//2]['image_id']
            # cur_label_set = list(set(targets[:B//2]['labels'].tolist()))/

            if iter in visualize_image_indices:
                # TODO plot predictions on target data
                root = data_loader.dataset.target.root # PosixPath
                # root = data_loader.dataset.source.root # PosixPath

                # use sample to check whether the rescaled boxes have been correctly plotted
                target_sample = samples.tensors[1].permute(1,2,0).cpu()
                
                # we only want to visualize pseudo gts on target images
                img_id = targets[0]['image_id'].item()

                # import pdb; pdb.set_trace()
                img_path = data_loader.dataset.target.coco.loadImgs(img_id)[0]['file_name']
                # pred results
                full_path = root/img_path

                # target gts
                label_gt = targets[1]['labels']
                prob_gt = torch.ones(label_gt.size())

                # normalized boxes, duplicate outputs
                boxes_gt = postprocessors_target['bbox'](targets[1], orig_target_sizes)[1]['boxes'] # same labels for targets[0] and targets[1] 

                keep = torch.arange(boxes_gt.size(0)).numpy() # keep all?
                # keep = prob_pred

                # import pdb; pdb.set_trace()
                results = postprocessors['bbox'](out, orig_target_sizes) # (bs, ...)
                boxes_pred = results[1]['boxes']
                label_pred = results[1]['labels']
                prob_pred = results[1]['scores']

                info_all = {}
                info_all['pred'] = [boxes_pred, label_pred, prob_pred]
                info_all['gt'] = [boxes_gt, label_gt, prob_gt]

                # TODO plot both grountruth and predicted boxes
                plot_results_train(info_all, img_id, full_path, target_sample)

                loss_dict_n_indices = criterion(out, targets, mode='train')
                loss_dict, returned_indices = loss_dict_n_indices

        # else normal train mode
        else:
            loss_dict = criterion(outputs, targets, mode='train')
        
        weight_dict = criterion.weight_dict

        # import pdb; pdb.set_trace()
        # the loss used for optimization
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # loss_list = [loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict]

        # losses = 0
        # for k in loss_dict.keys():
        #     if k in weight_dict:
        #         loss_k = loss_dict[k] * weight_dict[k]
        #         losses += loss_k

        # import pdb; pdb.set_trace()
        # reduce losses over all GPUs for logging purposes (loss_dict_reduced is scaled)
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        # store unscaled losses
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}

        # store scaled losses
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        # get total loss
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # import pdb; pdb.set_trace()

        # check if total loss is inf
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # BUG test for exploding graident norm (0 total norm)
        # parameters = [p for p in model.parameters() if p.grad is not None]
        # names = [n for n, p in model.named_parameters() if p.grad is not None]
        # for n, p in zip(names, parameters):
        #     if torch.norm(p.grad.detach(), norm_type).to(device) > 1000:
        #         print(n)
        #         quit()

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            # compute gradient norm
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        optimizer.step()
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    
    print(missing_source)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # TODO at the end of each epoch, return the updated cur_iter
    if 'probs' in outputs:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, cur_iter, outputs['probs']
    # if 'thresh' in outputs:
    #     thresh_stats = {}
    #     thresh_stats['max'] = torch.as_tensor(max(thresh_record))
    #     thresh_stats['min'] = torch.as_tensor(min(thresh_record))
    #     thresh_stats['mean'] = torch.as_tensor(sum(thresh_record)/len(thresh_record))
    #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, thresh_stats, outputs
    
    else:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, thresh_tmp_list, outputs


@torch.no_grad()
def evaluate(model, criterion, postprocessors, postprocessors_target, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    
    # import pdb; pdb.set_trace()

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        
        outputs = model(samples, None, None, None)

        # import pdb; pdb.set_trace()
        loss_dict = criterion(outputs, targets, mode='test')
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        # outputs: dict storing 300 predictions
        # list storing scores, labels, and boxes (100)
        # postprocess into format accepted by coco api
        results = postprocessors['bbox'](outputs, orig_target_sizes) # at test time samples are not transformed thus use orig_target_sizes
        # import pdb; pdb.set_trace()
        # TODO plot predictions on target data
        # root = data_loader.dataset.root # PositxPath

        # img_id = targets[1]['image_id'].item() # BUG index error
        # img_path = base_ds.loadImgs(img_id)[0]['file_name']
        # thresh = 0.6
        # # pred results
        # full_path = root/img_path

        # # target gts
        # label_gt = targets[1]['labels']
        # prob_gt = torch.ones(label_gt.size())

        # # normalized boxes, duplicate outputs
        # boxes_gt = postprocessors_target['bbox'](targets[1], orig_target_sizes)[1]['boxes'] # same labels for targets[0] and targets[1] 

        # keep = torch.arange(boxes_gt.size(0)).numpy()
        
        # boxes_pred = results[1]['boxes'][keep]
        # label_pred = results[1]['labels'][keep]
        # prob_pred = results[1]['scores'][keep]

        # info_all = {}
        # info_all['pred'] = [boxes_pred, label_pred, prob_pred]
        # info_all['gt'] = [boxes_gt, label_gt, prob_gt]

        # # TODO plot both grountruth and predicted boxes
        # plot_results(info_all, img_id, full_path)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        
        # dict storing scores, labels and boxes (100 predictions) from bs target images
        # keys store key ids 
        # scores are ranked
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        # import pdb; pdb.set_trace()
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

