# ------------------------------------------------------------------------
# Modified by Chung-Hao
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
from builtins import AssertionError
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
import math
import cv2 as cv
import tracemalloc
import gc

import torchvision

from util import box_ops, plot_utils
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import HungarianMatcher
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer_contrastive import build_deforamble_transformer
from .utils import GradientReversal, FCDiscriminator
import copy
from .memory_ema import Memory
from .utils import compute_CV, weighted_aggregate, weighted_aggregate_tmp, find_thresh  # , attention_module_multi_head

# from .debug_tools import *

# from .deformable_transformer_contrastive import TransformerEncoder, TransformerEncoderLayer

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                 backbone_align=False, space_align=False, channel_align=False, instance_align=False, debug=False, ema=False, feat_aug=False, cam_viz = False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        keep_rate = 0.996
        # TODO: returns updated prototypes
        if ema:
            self.memory = Memory(num_classes, transformer.d_model, keep_rate = keep_rate, num_feature_levels=num_feature_levels)
        
        # normalize_before = False
        # encoder_layer = TransformerEncoderLayer(transformer.d_model, transformer.nhead, transformer.dim_feedforward,
        #                                         0.1, transformer.activation, normalize_before)

        # num_encoder_layers = 1
        # encoder_norm = nn.LayerNorm(transformer.d_model) if normalize_before else None
        # self.encoder_detr = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)


        self.gradient = None # for hooks
        self.cam_viz = cam_viz
        # self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.classifier_prototypes = FCDiscriminator(num_classes, transformer.d_model)
        # self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, transformer.d_model))
        # self.cross_attn_enc = CrossAttention_agg_prototypes(transformer.d_model, transformer.nhead, 0.1)
        # self.cross_attn_dec = CrossAttention_agg_prototypes(transformer.d_model, transformer.nhead, 0.1)
        # self.cross_attn = CrossAttention_agg_encoder(transformer.d_model, transformer.nhead, 0.1)

        self.ema = ema
        self.m_items = nn.Parameter(torch.full((2,num_feature_levels, num_classes-1, transformer.d_model), 1e-6),
                                                                        requires_grad=False).cuda()
        # self.m_items = nn.Parameter(torch.zeros((2, num_feature_levels, num_classes-1, transformer.d_model)),
        #                                                                 requires_grad=False).cuda()

        # torch.nn.init.trunc_normal_(self.m_items, std=0.02)

        # breakpoint()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        # self.matcher = HungarianMatcher(cost_class = 2.0, cost_bbox = 5.0, cost_giou = 2.0)
        # import pdb; pdb.set_trace()
        self.num_classes = num_classes
        # shared linear layer for the classification head
        self.class_embed = nn.Linear(hidden_dim, num_classes) # class embeddings
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) # bbox embeddings
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)

            # initialise projection layer for each scale
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.uda = backbone_align or space_align or channel_align or instance_align
        self.backbone_align = backbone_align
        self.space_align = space_align
        self.channel_align = channel_align
        self.instance_align = instance_align
        self.debug = debug
        self.feat_aug = feat_aug

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # NOTE if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        if backbone_align:
            self.grl = GradientReversal()
            # domain discriminator for backbone alignment
            self.backbone_D = MLP(hidden_dim, hidden_dim, 1, 3)
            for layer in self.backbone_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
        if space_align:
            # domain discriminator for space alignment
            self.space_D = MLP(hidden_dim, hidden_dim, 1, 3)
            for layer in self.space_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
        if channel_align:
            # domain discriminator for channel alignment
            self.channel_D = MLP(hidden_dim, hidden_dim, 1, 3)
            for layer in self.channel_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
        if instance_align:
            # domain discriminator for instance alignment
            self.instance_D = MLP(hidden_dim, hidden_dim, 1, 3)

            # import pdb; pdb.set_trace()
            for layer in self.instance_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)


    def forward(self, samples: NestedTensor, targets, iter_i, cur_epoch, total_epoch):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        # store backbone features and mask tokens
        srcs = []
        masks = []

        # different layer features
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            # num_feature_levels = 4 by defualt
            for l in range(_len_srcs, self.num_feature_levels):

                # one feature level
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask

                # interpolate for downscaled masks
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None

        if not self.two_stage:
            query_embeds = self.query_embed.weight

        
        if self.debug:
            hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, da_output, query_pos, encoder_feat_info = self.transformer(srcs, masks, pos, query_embeds)
        else:
            # tracemalloc.start()
            hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, da_output = self.transformer(srcs, masks, pos, query_embeds)
            # hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, da_output, query_pos, encoder_feat_info = self.transformer(srcs, masks, pos, query_embeds)
            # print(tracemalloc.get_traced_memory())
        
        # breakpoint()

        if self.cam_viz:
            # NOTE apply hook function on the desired feature activation
            h = memory.register_hook(self.get_gradients)   


        outputs_classes = []
        outputs_coords = []

        # multi level outputs
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # final predictions
            outputs_class = self.class_embed[lvl](hs[lvl]) # linear layer

            tmp = self.bbox_embed[lvl](hs[lvl]) # mlp layer

            if reference.shape[-1] == 4:
                # deformable-detr predicts relative coordinates, unlike detr, which predicts absolute coordinates
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # torch.Size([6, 2, 300, 4])
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        
        ### get flattened backbone features
        # if self.training:
        #     # prepare input for encoder
        #     src_flatten = []
        #     mask_flatten = []
        #     lvl_pos_embed_flatten = []
        #     spatial_shapes = []
        #     # srcs: (B, C, H, W)
        #     for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos)):
        #         bs, c, h, w = src.shape
        #         spatial_shape = (h, w)
        #         spatial_shapes.append(spatial_shape)
        #         src = src.flatten(2).transpose(1, 2)
        #         mask = mask.flatten(1)
        #         pos_embed = pos_embed.flatten(2).transpose(1, 2)
        #         lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
        #         lvl_pos_embed_flatten.append(lvl_pos_embed)
        #         src_flatten.append(src) # appends 4 flattened scales
        #         mask_flatten.append(mask)
            
            # mask_flatten = torch.cat(mask_flatten, 1)
            # TODO add lvl postional embedding to the original positional embedding
            # lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        
        if self.training:
            B = outputs_class.shape[1]
            assert B == memory.shape[0]

            last_layer_out = outputs_class[-1]
            # outputs_class_conf = F.softmax(last_layer_out, -1) # (2, 300, 9)
            outputs_class_conf = last_layer_out.sigmoid()

            thresh = 0.8
            thresh_tmp_list = [] # record occuring instances
            try:
                keep = [torch.nonzero(outputs_class_conf[b]>thresh).unsqueeze(0) for b in range(outputs_class_conf.shape[0])] # batch wise
                
                # breakpoint()
                find_empty = [keep_tensor for keep_tensor in keep if keep_tensor.numel() == 0]
                assert len(find_empty) == 0 # assert error if empty
            
            # BUG: only valid when only target class confidences are used 
            except AssertionError:
                # new_keep, new_thresh = find_thresh(outputs_class_conf_tmp, thresh_tmp, keep_tmp)
                for i,v in enumerate(keep):
                    # import pdb; pdb.set_trace()
                    if v.numel()==0:
                        thresh_tmp = torch.max(outputs_class_conf[i], -1)[0].max()-0.05
                        outputs_class_conf_tmp = outputs_class_conf[i] 
                        keep_tmp = torch.nonzero(outputs_class_conf_tmp>thresh_tmp).unsqueeze(0)
                        keep[i] = keep_tmp
                        thresh_tmp_list.append(thresh_tmp)

            ### multi-scale memory reshape
            memory_reshaped = [] # (B,c,h,w)
            sequence_len = []
            for src in srcs:
                w = src.shape[-1]
                h = src.shape[-2]
                c = src.shape[-3]
                flat_length = h*w
                sequence_len.append(flat_length)
            assert sum(sequence_len) == memory.shape[1]

            # breakpoint()
            for src_i in range(len(srcs)):
                if src_i == 0:
                    # index first
                    B,c,h,w = srcs[src_i].shape
                    memory_flat = memory[:,:sequence_len[src_i],:]
                    # then reshape
                    memory_reshaped_tmp = memory_flat.reshape(B,c,h,w)
                    memory_reshaped.append(memory_reshaped_tmp)
                elif src_i == 1:
                    B,c,h,w = srcs[src_i].shape
                    # index first
                    # breakpoint()
                    pre_pos_i = sequence_len[src_i-1]
                    cur_pos_i = sequence_len[src_i-1] + sequence_len[src_i]
                    # breakpoint()
                    memory_flat = memory[:,pre_pos_i:cur_pos_i,:]
                    # then reshape
                    memory_reshaped_tmp = memory_flat.reshape(B,c,h,w)
                    memory_reshaped.append(memory_reshaped_tmp)
                else:
                    B,c,h,w = srcs[src_i].shape
                    # index first
                    # breakpoint()

                    pre_pos_i = sum(sequence_len[:src_i])
                    cur_pos_i = sum(sequence_len[:src_i]) + sequence_len[src_i]
                    # breakpoint()
                    memory_flat = memory[:,pre_pos_i:cur_pos_i,:]
                    # then reshape
                    memory_reshaped_tmp = memory_flat.reshape(B,c,h,w)
                    memory_reshaped.append(memory_reshaped_tmp)

            ### get first layer feature
            # memory_reshaped = memory[:,:feature_w*feature_h].reshape(-1, feature_h, feature_w, feature_c).permute(0,3,1,2)
            
            list_of_labels_enc = [] # list(list())
            list_of_scores_enc = [] # list of tensors
            rescaled_boxes_enc = [] # list of tensors

            ### get source boxes (index from the same list as tgt)
            for batch_idx in range(0, B//2, 1):
                # keep_tmp = keep_src[batch_idx][:,:,0].tolist()[0]
                source_boxes = targets[batch_idx]['boxes']
                source_labels = targets[batch_idx]['labels'].tolist()
                source_scores = torch.ones(source_boxes.shape[0]).cuda()
                # source_scores, _ = torch.max(outputs_class_conf[batch_idx][keep_tmp], dim=1)

                boxes_rescaled = box_ops.box_cxcywh_to_xyxy(source_boxes) # src only, batch size = 1
                # and from relative [0, 1] to absolute [0, height] coordinates
                # img_sizes = torch.stack([t["size"] for t in targets], dim=0)
                img_sizes = targets[batch_idx]["size"]
                img_h, img_w = img_sizes.unbind(0)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0) # different scale factor for different images
                
                # scale all boxes with the corresponding image sizes
                for b in range(boxes_rescaled.shape[0]):
                    boxes_rescaled[b] *= scale_fct

                rescaled_boxes_enc.append(boxes_rescaled) # delist
                list_of_labels_enc.append(source_labels)
                list_of_scores_enc.append(source_scores)


            ### collect target boxes (index from the same list as src)
            for batch_idx in range(B//2, B, 1):
                keep_tmp = keep[batch_idx][:,:,0].tolist()[0] # get list of indices
                keep_label_idx = keep[batch_idx][:,:,1].tolist()[0]
                target_boxes = outputs_coord[-1][batch_idx][keep_tmp] # get last layer predicted boxes
                boxes_rescaled = box_ops.box_cxcywh_to_xyxy(target_boxes)
                # and from relative [0, 1] to absolute [0, height] coordinates
                img_sizes = targets[batch_idx]["size"]
                img_h, img_w = img_sizes.unbind(0)

                # breakpoint()
                # since box tensor is (x,y,x,y)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                
                # 
                for b in range(boxes_rescaled.shape[0]):
                    boxes_rescaled[b] *= scale_fct

                rescaled_boxes_enc.append(boxes_rescaled)
                list_of_labels_enc.append(keep_label_idx)
                scores, _ = torch.max(outputs_class_conf[batch_idx][keep_tmp], dim=1)
                list_of_scores_enc.append(scores)

            # 1. DEBUG: check pseudo label quality
            # check_pseudo_boxes(B, targets, samples, rescaled_boxes_enc, list_of_scores_enc, list_of_labels_enc)

            ### compute src prototypes (NOTE: checked)
            src_boxes = rescaled_boxes_enc[:B//2]
            src_scores = list_of_scores_enc[:B//2]
            src_labels = list_of_labels_enc[:B//2]

            if len(srcs) > 1:
                spatial_scales = [1/8.0, 1/16.0, 1/32.0, 1/64.0] # multi-scale
            else:
                spatial_scales = [1/32.0] # single-scale

            # breakpoint()
            list_of_rois_src = [] # [scale, bs] (num_boxes, 256)
            ### compute src rois first
            for m, scale in zip(memory_reshaped, spatial_scales):
                list_tmp = []
                # for src only
                for batch_idx in range(0, B//2, 1):
                    # input dim: (N, C, H, W)
                    rois = torchvision.ops.roi_align(m[batch_idx].unsqueeze(0), [src_boxes[batch_idx]], output_size=(7, 7), spatial_scale=scale, aligned=True).mean(3).mean(2)
                    list_tmp.append(rois)

                list_of_rois_src.append(list_tmp)

            # NOTE checked
            ### aggregate src prototypes
            list_of_src_prototype = [] # [scale], (num_classes, feat_dim)
            for roi_group in list_of_rois_src:
                # batch dim reduced after aggregation
                # (e.g scale 1, bs 2 --> all reduced)
                # (e.g scale 4, bs 2 --> 4, bs reduced)
                src_prototypes_enc, _ = weighted_aggregate_tmp(B, src_labels, roi_group, src_scores, None, self.num_classes, self.hidden_dim)
                list_of_src_prototype.append(src_prototypes_enc)
            

            src_prototypes_enc = torch.stack(list_of_src_prototype) # (B, cls_num, feat_dim)
            src_prototypes_enc = F.normalize(src_prototypes_enc, dim=-1) # normalize

            # breakpoint()
            # src_prototypes_enc = src_prototypes_enc.mean(0) # average across batches (cls_num, feat_dim)

            # 2. DEBUG: log prototype values

            ### NOTE in case of empty soruce gts, we take src features from memory
            for scale_i in range(src_prototypes_enc.shape[0]):
                for cls_i in range(src_prototypes_enc.shape[1]):
                    # (scale, bs, class, feat_dim)
                    # if empty src targets, the corresponding elements are guaranteed to be zeros
                    if src_prototypes_enc[scale_i][cls_i].sum() == 0:
                        src_prototypes_enc[scale_i][cls_i] = self.m_items[0][scale_i][cls_i,:].detach()
            


            ### NOTE in case of empty soruce gts, we take src features from memory
            # for cls_i in range(src_prototypes_enc.shape[0]):
            #     # (scale, bs, class, feat_dim)
            #     # if empty src targets, the corresponding elements are guaranteed to be zeros
            #     if src_prototypes_enc[cls_i].sum() == 0:
            #         src_prototypes_enc[cls_i] = self.m_items[0,cls_i,:].detach()

            # 3. DEBUG: check filled prototype values  
            tgt_boxes = rescaled_boxes_enc[B//2:] # [bs] (, num_rois, 4)
            tgt_scores = list_of_scores_enc[B//2:] # [bs] (, num_rois)
            tgt_labels = list_of_labels_enc[B//2:] # [bs] [num_rois]

            list_of_rois_tgt = [] # [scale_dim, bs, ] (num_boxes, 256, h, w)
            for m, scale in zip(memory_reshaped, spatial_scales):
                list_tmp = []
                # B//2 since we take features from the original encoder
                for batch_idx in range(0, B//2, 1):
                    # input dim: (N, C, H, W)
                    rois = torchvision.ops.roi_align(m[batch_idx].unsqueeze(0), [tgt_boxes[batch_idx]], output_size=(7,7), spatial_scale=scale, aligned=True)
                    list_tmp.append(rois)
                list_of_rois_tgt.append(list_tmp)


            # 4. DEBUG: check tgt prototype attention
            # check_prototype_attention_map(B, src_prototypes_enc, memory_reshaped, tgt_boxes, targets)
            # breakpoint()

            ### TODO estimate domain bias
            # domain_bias = self.m_items[1] - self.m_items[0]
            # lambda_a = 0.999 * max(0.5, 1-(cur_epoch/total_epoch))
            # biased_src_prototypes = lambda_a*src_prototypes_enc + (1-lambda_a)*domain_bias

            # for detr transformer 
            # src_flatten, level_start_index, valid_ratios, \
            # lvl_pos_embed_flatten, mask_flatten = encoder_feat_info.items()

            list_of_weighted_tgt_rois_final = [] # [scale, bs] (num_rois, 1, feat_dim)
            list_of_weighted_tgt_rois_bg_final = []
            # torch.Size([1, 11, 256, 7, 7])
            for scale_i in range(len(spatial_scales)):
                list_of_weighted_tgt_rois = []
                list_of_weighted_tgt_rois_bg = []
                list_of_rois_tgt_tmp = list_of_rois_tgt[scale_i]

                for bs_i in range(B//2):
                    rois_target = list_of_rois_tgt_tmp[bs_i] # (num_rois, feat_dim, h, w)

                    ### similarity: compute binary mask
                    filters = src_prototypes_enc[scale_i].squeeze(0).unsqueeze(-1).unsqueeze(-1) # (num_class, feat_dim, 1, 1)
                    
                    thresh_mask = 0.8
                                        
                    scores = F.conv2d(rois_target, filters).sigmoid() # (num_rois, num_classes, 7, 7)
                    binary_masks = torch.where(
                        scores > thresh_mask,
                        torch.tensor(1).to(scores.device),
                        torch.tensor(0).to(scores.device)
                    ) # (num_rois, num_classes, 7, 7)
                    # background prototype
                    binary_masks_bg = torch.where(
                        scores < thresh_mask,
                        torch.tensor(1).to(scores.device),
                        torch.tensor(0).to(scores.device)
                    )

                    # torch.save(output_tensor, f'./visualization/output_tensor/output_tensor_{iter_i}.pt')
                    # torch.save(binary_masks, f'./visualization/binary_masks/binary_masks_{iter_i}.pt')

                    # 5. DEBUG: visualize masks
                    # NOTE output contains binary masks for all classes, so we use the predicted 
                    # confidence to get the corresponding mask
                    filtered_rois_target_list = [] # [num_rois] (,256)
                    filtered_rois_bg_target_list = [] # [num_rois] (,256)
                    # filter each roi with the binary mask
                    for roi_index in range(rois_target.shape[0]):
                        tgt_label = tgt_labels[bs_i][roi_index]
                        binary_mask = binary_masks[roi_index, tgt_label-1,:,:] #  (7, 7)
                        binary_mask_bg = binary_masks_bg[roi_index, tgt_label-1,:,:]
                        rois_target_tmp = rois_target.squeeze(0)[roi_index] # (256, 7, 7)７
                        filtered_rois_target = rois_target_tmp * binary_mask # (256, 7, 7)
                        filtered_rois_target_bg = rois_target_tmp * binary_mask_bg
                        mask_pooled = filtered_rois_target.mean(-1).mean(-1) # avg pool
                        mask_pooled_bg = filtered_rois_target_bg.mean(-1).mean(-1) # avg pool

                        
                        filtered_rois_bg_target_list.append(mask_pooled_bg)
                        filtered_rois_target_list.append(mask_pooled)

                    # DEBUG: visualize mask average pooled features
                    
                    try:
                        filtered_rois_target = torch.stack(filtered_rois_target_list).unsqueeze(0)
                        filtered_rois_target_bg = torch.stack(filtered_rois_bg_target_list).unsqueeze(0)
                    except RuntimeError:
                        breakpoint()

                    # NOTE will trigger if wrong computation w/ binary masks
                    if len(filtered_rois_target_list) == 0:
                        breakpoint()

                    # append along batch dim
                    list_of_weighted_tgt_rois.append(filtered_rois_target)
                    list_of_weighted_tgt_rois_bg.append(filtered_rois_target_bg)

                # append along scale
                list_of_weighted_tgt_rois_final.append(list_of_weighted_tgt_rois) # BUG fix this to incorporate batch dim
                list_of_weighted_tgt_rois_bg_final.append(list_of_weighted_tgt_rois_bg)
                
                # breakpoint()
                ### similarity: compute similarity only
                # try:
                    # scores = torch.matmul(rois_target.squeeze(0), src_prototypes_enc_clone.transpose(2,1))
                # except RuntimeError:
                    # import pdb; pdb.set_trace()
                
                # (bs, num_rois, classes)
                # scores_normalized = scores.sigmoid()
                
                # try:
                #     max_scores, _ = torch.max(scores_normalized, 2)
                # except IndexError:
                #     import pdb; pdb.set_trace()

                # reweight, here we only use max scores, but in the case of spatial features, we have to perform aggregation too
                # reweighted_rois_target = torch.mul(max_scores.unsqueeze(-1), rois_target) # (1, num_rois, )
                # list_of_weighted_tgt_rois.append(reweighted_rois_target)
            
            # breakpoint()
            ### perform weighted aggregation across tgt rois
            list_of_tgt_prototypes= [] # [scale] (num_classes, feat_dim)
            list_of_tgt_prototypes_bg = [] # [scale] (, feat_dim)
            # get background confidence scores (probability of roi being a FP)
            # breakpoint()
            bg_tgt_scores = [torch.ones(tgt_scores[bs_i].shape[0]).cuda()-tgt_scores[bs_i] for bs_i in range(len(tgt_scores))]
            
            for roi_scale_group, roi_scale_group_bg in zip(list_of_weighted_tgt_rois_final, list_of_weighted_tgt_rois_bg_final):
                tgt_prototypes_enc, _ = weighted_aggregate_tmp(B, tgt_labels, roi_scale_group, tgt_scores, src_prototypes_enc, self.num_classes, self.hidden_dim)
                tgt_prototypes_enc_bg, _ = weighted_aggregate_tmp(B, tgt_labels, roi_scale_group_bg, bg_tgt_scores, None, self.num_classes, self.hidden_dim) # (class_num, feat_dim)

                tgt_prototypes_enc_bg = tgt_prototypes_enc_bg.mean(0) # (, feat_dim)
                tgt_prototypes_enc = F.normalize(tgt_prototypes_enc, dim=-1)
                list_of_tgt_prototypes.append(tgt_prototypes_enc)
                list_of_tgt_prototypes_bg.append(tgt_prototypes_enc_bg)
            
            tgt_prototypes_enc = torch.stack(list_of_tgt_prototypes) # (scale, num_classes, feat_dim)
            tgt_prototypes_bg_enc = torch.stack(list_of_tgt_prototypes_bg) # (scale, 256)


            # check_prototype_attention_map(tgt_prototypes_enc, memory_reshaped, tgt_boxes)
            prototypes = torch.stack([src_prototypes_enc, tgt_prototypes_enc], dim=0)

            # updated_src_prototypes_enc = torch.repeat_interleave(src_prototypes_enc, 2, dim=0)
            # updated_src_prototypes_enc = updated_src_prototypes_enc.view(memory.shape[0], len(spatial_scales), *updated_src_prototypes_enc.shape[1:])
            # updated_src_prototypes_enc = src_prototypes_enc.view(memory.shape[0], len(spatial_scales), *updated_src_prototypes_enc.shape[1:])
            # updated_src_prototypes_enc = tgt_prototypes_enc.view(memory.shape[0], len(spatial_scales), *updated_src_prototypes_enc.shape[1:])

            ### aggregate encoder features from prototypes
            # updated_src_prototypes_enc = updated_src_prototypes_enc.view(memory.shape[0], len(spatial_scales), *updated_src_prototypes_enc.shape[1:])
            # aggregated_memory, attn_weights = self.cross_attn_enc(updated_src_prototypes_enc, memory, lvl_pos_embed_flatten, None)

            ### TODO multi scale aggregate with cross attention
            # import pdb; pdb.set_trace()
            # class_embeds_enc_list= []
            # memory_h = srcs[0].shape[-2]
            # memory_w = srcs[0].shape[-1]
            # memory_tmp = memory[:, :memory_h*memory_w, :] # e.g (2, 800 ,256)
            # lvl_pos_embed_flatten_tmp = lvl_pos_embed_flatten[:, :memory_h*memory_w, :]
            # mask_flatten_tmp = mask_flatten[:, :memory_h*memory_w]
            # updated_src_prototypes_enc_tmp = updated_src_prototypes_enc[:, 0, :, :]
            # # use cross attention to aggregate encoder semantics for the initialised source prototypes
            # class_embeds_enc, _ = self.cross_attn_enc(updated_src_prototypes_enc_tmp, memory_tmp, lvl_pos_embed_flatten_tmp, mask_flatten_tmp)                
            # class_embeds_enc_list.append(class_embeds_enc)
            
            # class_embeds_enc = torch.stack(class_embeds_enc_list)
            # class_embeds_enc = class_embeds_enc.view(-1, *class_embeds_enc.shape[2:])

            # import pdb; pdb.set_trace()
            # memory_h = memory_reshaped[0].shape[-2]
            # memory_w = memory_reshaped[0].shape[-1]

            # memory_reshaped: (B,c,h,w)
            # fist_layer_feat = memory.view(memory_reshaped[0].shape[0], :memory_reshaped[0].shape[-2]*memory_reshaped[0].shape[-1], memory_reshaped[0].shape[-3])
            # first_layer_lvl_pos_embed_flatten = lvl_pos_embed_flatten[:, :memory_reshaped[0].shape[-2] * memory_reshaped[0].shape[-1], :]
            # first_layer_mask_flatten = mask_flatten[:, :memory_reshaped[0].shape[-2] * memory_reshaped[0].shape[-1]]
            # class_embeds_enc, _ = self.cross_attn_enc(updated_src_prototypes_enc.squeeze(1), fist_layer_feat, first_layer_lvl_pos_embed_flatten, first_layer_mask_flatten)
            # class_embeds_enc = torch.stack(class_embeds_enc_list)
            # class_embeds_enc = class_embeds_enc.view(-1, *class_embeds_enc.shape[2:])

            
            source_alphas = None
            
            prototypes_copy = prototypes.clone()
            if self.ema:
                new_memory = self.memory(self.m_items, prototypes)
                self.m_items = new_memory
                # updated_class_prototypes = new_memory

                ### fill missing class features with memory features (src filled already)
                for B_i in range(prototypes.shape[0]):
                    for cls_i, value in enumerate(prototypes[B_i]):
                        if value.sum().item() == 0.:
                            prototypes_copy[B_i][cls_i] = self.m_items[B_i][cls_i].detach()

                # finally assigned back to prototypes
                prototypes = prototypes_copy # (B, scale, class, feat_dim)


            # breakpoint()
            # TODO reshape to (batch,...) after ema
            # updated_class_embeds_enc = updated_class_embeds_enc.view(len(spatial_scales), memory.shape[0], *updated_class_embeds_enc.shape[1:])

            ### TODO try using first layer prototypes             
            # fist_layer_feat = memory_reshaped[0].view(memory_reshaped[0].shape[0], memory_reshaped[0].shape[-2] * memory_reshaped[0].shape[-1], memory_reshaped[0].shape[-3])
            # first_layer_lvl_pos_embed_flatten = lvl_pos_embed_flatten[:, :memory_reshaped[0].shape[-2] * memory_reshaped[0].shape[-1], :]
            # first_layer_mask_flatten = mask_flatten[:, :memory_reshaped[0].shape[-2] * memory_reshaped[0].shape[-1]]
            # # single scale
            # # import pdb; pdb.set_trace()
            # class_embeds_enc, attn_weights = self.cross_attn_enc(updated_src_prototypes_enc.squeeze(1), fist_layer_feat, first_layer_lvl_pos_embed_flatten, first_layer_mask_flatten)

            ### visualize attention map
            # fist_layer_feat[0] for source 
            # out = self.cos(fist_layer_feat[0], fist_layer_feat[0])
            # attn_mask = torch.mul(fist_layer_feat[0], out.unsqueeze(-1)).reshape(memory_reshaped[0].shape[1], memory_reshaped[0].shape[2], memory_reshaped[0].shape[3])
            # attn_mask_max_activation, _ = torch.max(attn_mask, dim=0)
            # orig_h_scaled = targets[0]['size'][0]//8
            # orig_w_scaled = targets[0]['size'][1]//8

            # attn_mask_max_activation_unpadded = attn_mask_max_activation[:orig_h_scaled.item(), :orig_w_scaled.item()]
            # plt.figure(figsize=(30, 50))
            # plt.imshow(attn_mask_max_activation_unpadded.detach().cpu())
            # plt.axis('off')
            # import pdb; pdb.set_trace()
            # plt.savefig(str('./visualization/encoder_attn_mask_layer1.png'), bbox_inches='tight')

            ### visualize image
            # unpadded_samples = samples.tensors[0][:, :targets[0]['size'][0], :targets[0]['size'][1]]
            # sample = samples.tensors[0]
            # inverted_image_tensor = plot_utils.inverse_transform(sample)
            # inverted_image_tensor = inverted_image_tensor.permute(1,2,0)
            # plt.figure(figsize=(30, 50))
            # plt.imshow(inverted_image_tensor.detach().cpu())
            # plt.axis('off')
            # plt.savefig(str('./visualization/sample_image_padded.png'), bbox_inches='tight')

            # import pdb; pdb.set_trace()

            # last_layer_feature = hs[-1]
            # class_embeds_dec, attn_weights = self.cross_attn_dec(updated_src_prototypes_enc, last_layer_feature, query_pos, None)

            # encoder scores
            # similarity = torch.softmax(torch.matmul(updated_src_prototypes_enc, memory.transpose(2,1)).transpose(1,2), -1)
            # # use these scores to aggregate features
            # class_embeds = torch.matmul(similarity.transpose(2,1), memory) # class embeddings


            ### TODO viz
            # enhanced = torch.matmul(updated_src_prototypes_enc.squeeze(0).t(), similarity)
            # enhanced = torch.matmul(similarity.t(), updated_src_prototypes_enc.squeeze(0))

            # C, H, W for visualization
            # similarity_reshaped = enhanced.reshape(feature_c, feature_h, feature_w)
            # average = similarity_reshaped.sum(0)/similarity_reshaped.shape[0]

            # average_feature_map = features[0].tensors[0].mean(0)
            # average_memory_map = memory_reshaped[0].mean(0)

            # source_memory = memory_reshaped[0]
            # source_memory = source_memory.mean(0)
            # target_memory = memory_reshaped[1]

            # plt.figure(figsize=(30, 50))
            # # plt.imshow(average_feature_map.detach().cpu())
            # # plt.imshow(average_memory_map.detach().cpu())
            # plt.imshow(source_memory.detach().cpu())
            # plt.axis('off')
            # # plt.savefig(str('./visualization/feature/average_feature_map_{}.png').format(cur_iter_num), bbox_inches='tight')
            # plt.savefig(str('./visualization/feature_encoder/source_first_layer_feature_map__encoder{}.png').format(cur_iter_num), bbox_inches='tight')
            # # plt.savefig(str('./visualization/feature_encoder/average_memory_map_{}.png').format(cur_iter_num), bbox_inches='tight')
            # plt.close()

            # import pdb; pdb.set_trace()

        if self.training and self.uda:
            B = outputs_class.shape[1]

            # pseudo predictions
            outputs_class_pseudo = outputs_class[:, B//2:]
            outputs_coord_pseudo = outputs_coord[:, B//2:]

            # replace with new 
            outputs_class = outputs_class[:, :B//2] # only source data has labels, so we index the first one
            outputs_coord = outputs_coord[:, :B//2]

            if self.two_stage:
                enc_outputs_class = enc_outputs_class[:B//2]
                enc_outputs_coord_unact = enc_outputs_coord_unact[:B//2]

            # discriminator outputs
            # backbone here is an MLP for discriminator
            if self.backbone_align:
                da_output['backbone'] = torch.cat([self.backbone_D(self.grl(src.flatten(2).transpose(1, 2))) for src in srcs], dim=1)
            if self.space_align:
                # (2, 1, 256)
                # (2, 1, 6) --> 6 outputs
                da_output['space_query'] = self.space_D(da_output['space_query'])

            if self.channel_align:
                da_output['channel_query'] = self.channel_D(da_output['channel_query'])
            if self.instance_align:
                da_output['instance_query'] = self.instance_D(da_output['instance_query'])

            # for train mode
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_logits_pseudo': outputs_class_pseudo[-1], 'pred_boxes_pseudo': outputs_coord_pseudo[-1]}

        elif not self.training:
            # for eval mode
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        # TODO add prototypes to outputs
        if self.training:
            # src_prototypes_enc = updated_class_prototypes[0, :, :]
            # tgt_prototypes_enc = updated_class_prototypes[1, :, :]
            # src_prototypes_dec = class_embeds_dec[0]
            # tgt_prototypes_dec = class_embeds_dec[1]

            if len(thresh_tmp_list) > 0:
                out['thresh_change_occurence'] = thresh_tmp_list

            # breakpoint()
            
            # src_prototypes = prototypes[0][0].unsqueeze(0)
            # tgt_prototypes = prototypes[1][0].unsqueeze(0)

            src_prototypes = prototypes[0]
            tgt_prototypes = prototypes[1]

            # NOTE: store ema memory items
            memory_prototypes = self.m_items.detach().clone()
            out['prototypes_enc'] = {'src_prototypes_enc': src_prototypes, 'tgt_prototypes_enc': tgt_prototypes, 'tgt_prototypes_bg_enc': tgt_prototypes_bg_enc,
                                    'memory_prototypes': memory_prototypes, 'alpha_values': source_alphas}
            
            # import pdb; pdb.set_trace()
            # out['prototypes_dec'] = {'src_prototypes_dec': src_prototypes_dec, 'tgt_prototypes_dec': tgt_prototypes_dec}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        
        # discriminator outputs
        if self.training and self.uda:
            out['da_output'] = da_output
            # TODO testing
            out['thresh'] = thresh

        if self.debug:
            B = src.shape[0]
            w = src.shape[-1]
            h = src.shape[-2]
            c = src.shape[-3]
            flat_length = h*w

            # in case of multi scale features, we need to index accordinly
            memory_flat = memory[:,:flat_length,:]

            # import pdb; pdb.set_trace()
            # then reshape
            memory_reshaped = memory_flat.reshape(B,c,h,w)

            # move to cpu for plots
            rescaled_boxes = []
            list_of_scores = []
            list_of_labels = []
            for box, score, label in zip(rescaled_boxes_enc, list_of_scores_enc, list_of_labels_enc):
                list_of_scores.append(score.detach().cpu())
                rescaled_boxes.append(box.detach().cpu())
                list_of_labels.append(label)

            # return out, features, memory_reshaped, hs, self.m_items
            return out, rescaled_boxes, list_of_scores, list_of_labels
        else:
            return out
    
    def get_backbone_features(self, x):
        """
        x: samples
        return: features, positional encodings
        """

        return self.backbone[0]
    
    # to store gradients when using hook
    def get_gradients(grad):
        self.gradient = grad

    # get gradients
    def get_activations_gradient(self):
        return self.gradient

    # get feature activations
    def get_activations(memory):
        return memory


    def compute_category_codes(self, source_samples, source_targets):
        num_supp = source_samples.tensors.shape[0]

        # contains scaled boxes

        if self.num_feature_levels == 1:

            ### forward_supp_branch from backbone
            ### support features: torch.Size([25, 2048, 21, 21])

            # the support encoding branch
            features, pos = self.backbone.forward_supp_branch(source_samples, return_interm_layers=False) # features: torch.Size([25, 2048, 21, 21])

            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src)) # torch.Size([25, 256, 21, 21])
                masks.append(mask)
                assert mask is not None
            
            ### srcs: torch.Size([25, 256, 21, 21])
            ### support boxes (scaled, normalised by width, height)
            boxes = [box_ops.box_cxcywh_to_xyxy(t['boxes']) for t in source_targets]
            # and from relative [0, 1] to absolute [0, height] coordinates
            img_sizes = torch.stack([t["size"] for t in source_targets], dim=0)
            img_h, img_w = img_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            # import pdb; pdb.set_trace()

            for b in range(num_supp):
                boxes[b] *= scale_fct[b]
                # import pdb; pdb.set_trace()

            query_embeds = self.query_embed.to(src.device)

            ### torch.Size([25, 5, 256])
            ### task encodings are predefined
            tsp = self.task_positional_encoding(torch.zeros(self.args.episode_size, self.hidden_dim, device=src.device)).unsqueeze(0).expand(num_supp, -1, -1)
            
            
            category_codes_list = list()

            # per category
            for i in range(num_supp // self.args.episode_size):
                # forward support branch computes the category codes as well as aggregates the support and query features
                # takes feature maps, support boxes and query embeds
                # takes 5 different categories at the time
                # transformer_detr --> encoder --> encoder layer
                # srcs are support features
                category_codes_list.append(
                    self.transformer.forward_supp_branch([srcs[0][i*self.args.episode_size: (i+1)*self.args.episode_size]],
                                                         [masks[0][i*self.args.episode_size: (i+1)*self.args.episode_size]],
                                                         [pos[0][i*self.args.episode_size: (i+1)*self.args.episode_size]],
                                                         query_embeds,
                                                         tsp[i*self.args.episode_size: (i+1)*self.args.episode_size],
                                                         boxes[i*self.args.episode_size: (i+1)*self.args.episode_size])
                )


            final_category_codes_list = []

            # 6 encoder layers; hence 6 category codes, each of dimension (5, 6, 5, 256)
            # len(category_codes_list)==5
            # len(category_codes_list[0])==6
            # this is mainly to concat the category codes of the same layer (e.g layer 1, layer2, ...)
            for i in range(self.args.enc_layers):
                final_category_codes_list.append(
                    torch.cat([ccl[i] for ccl in category_codes_list], dim=0)
                )
            # import pdb; pdb.set_trace()

            # (6, 25, 256)
            # (num_layer, num_shot * num_category, embedding_dim)
            return final_category_codes_list

        elif self.num_feature_levels == 4:
            raise NotImplementedError
        else:
            raise NotImplementedError

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, da_gamma=2, return_indices=False, margin = 1, feat_aug=False, Lamda=0.25, eos_coef=0.1):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.da_gamma = da_gamma
        self.return_indices = return_indices
        self.margin = margin
        self.cross_entropy = nn.CrossEntropyLoss()
        self.feat_aug = feat_aug
        self.Lamda = Lamda
        self.eos_coef = eos_coef

        # TODO original detr implementation
        empty_weight = torch.ones(self.num_classes) # original implementation is torch.ones(self.num_classes +1) but not sure why
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def category_token_align_loss(self, outputs):

        import pdb; pdb.set_trace()
        B = outputs.shape[0]
        assert B % 2 == 0

        targets = torch.empty_like(outputs)
        targets[:B//2] = 0
        targets[B//2:] = 1

        # import pdb; pdb.set_trace()
        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')

        if use_focal:
            prob = outputs.sigmoid()
            p_t = prob * targets + (1 - prob) * (1 - targets)
            loss = loss * ((1 - p_t) ** self.da_gamma)

        return loss.mean()

    # deformable detr
    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # (1, 300, 9)

        idx = self._get_src_permutation_idx(indices) # (tensor([0, 0]), tensor([175, 186]))

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # e.g tensor([3, 3])

        # TODO for single class, need to convert target_classes_o to ones since the target_classes_o
        # elements will be used for scattering the one hot vectors later on
        # target_classes_o = torch.ones_like(target_classes_o)


        # TODO set bg label to 0 instead of 9
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) # (1, 300)

        target_classes[idx] = target_classes_o # fill with true labels

        # import pdb; pdb.set_trace()
        # (1, 300, 10)
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)

        
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1) # (dim, index, src tensor)
        
        # (1, 300, 9)
        # import pdb; pdb.set_trace()
        target_classes_onehot = target_classes_onehot[:,:,:-1]

        # import pdb; pdb.set_trace()
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            # import pdb; pdb.set_trace()
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # detr
    def loss_labels_ce(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # set background index to 0
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        
        # idx is used for 
        target_classes[idx] = target_classes_o

        # import pdb; pdb.set_trace()
        loss_ce = F.cross_entropy(src_logits, target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)


        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        # groundtruth masks
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses, 

    # TODO: entropy minimization for target only
    def loss_entropy(self, outputs, targets, indices, num_boxes):

        raise NotImplementedError

    def loss_da(self, outputs, use_focal=False):
        B = outputs.shape[0]
        assert B % 2 == 0
        
        targets = torch.empty_like(outputs)
        targets[:B//2] = 0
        targets[B//2:] = 1

        # import pdb; pdb.set_trace()
        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')

        if use_focal:
            prob = outputs.sigmoid()
            p_t = prob * targets + (1 - prob) * (1 - targets)
            loss = loss * ((1 - p_t) ** self.da_gamma)

        return loss.mean()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # TODO change loss_labels here
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels_bce,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    # L2 loss
    def distance(self, src_feat, tgt_feat):
        eps = 1e-6
        # BUG Function 'PowBackward0' returned nan values in its 0th output
        output = torch.pow((src_feat - tgt_feat), 2.0).mean() + eps

        # print(output)

        return output
    
    def contrastive_loss(self, source, target, bg_proto, alpha_values, margin=1):
        """
        source: (scale, class_num, feat_dim)
        target: (scale, class_num, feat_dim)
        bg_proto: (scale, 256)
        """

        intra_loss = 0.
        inter_loss = 0.
        # bg_loss = 0.

        # target.register_hook(lambda grad: print(torch.isnan(grad).any()))
        # source.register_hook(lambda grad: print(torch.isnan(grad).any()))

        # for scale_i in range(source.shape[0]):

        # TODO try first and second last layer features 
        import pdb; pdb.set_trace()
        source_prototypes = torch.index_select(source, 0, torch.tensor([0,2]).cuda())
        target_prototypes = torch.index_select(target, 0, torch.tensor([0,2]).cuda())


        for scale_i in range(source_prototypes.shape[0]):

            for cls_idx in range(self.num_classes-1):
                tmp_src_feat_1 = source_prototypes[scale_i][cls_idx, :] # per class prototype
                tmp_tgt_feat_1 = target_prototypes[scale_i][cls_idx, :] # per class prototype

                # tmp_tgt_feat_1.register_hook(lambda grad: print(torch.isnan(grad).any()))
                # tmp_tgt_feat_1.register_hook(lambda grad: breakpoint() if torch.isnan(grad).any() == True else print(grad))
                
                # bg loss
                # breakpoint()
                # bg_loss = bg_loss + torch.sqrt(self.distance(bg_proto, tmp_tgt_feat_1))

                # intra
                intra_loss = intra_loss + torch.sqrt(self.distance(tmp_src_feat_1, tmp_tgt_feat_1))
                # intra_loss = intra_loss + self.distance(tmp_src_feat_1, tmp_tgt_feat_1)

                # inter takes into account the current and all other classes
                for cls_idx_next in range(cls_idx+1, self.num_classes-1):
                    tmp_src_feat_2 = source_prototypes[scale_i][cls_idx_next, :]
                    tmp_tgt_feat_2 = target_prototypes[scale_i][cls_idx_next, :]

                    ### original implementation
                    # breakpoint()
                    inter_loss =  inter_loss + ((margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_src_feat_2))) / margin) *torch.max(margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_src_feat_2)),torch.tensor(0).float().cuda())

                    inter_loss =  inter_loss + ((margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_tgt_feat_2))) / margin) *torch.max(margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_tgt_feat_2)),torch.tensor(0).float().cuda())

                    inter_loss =  inter_loss + ((margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_tgt_feat_2))) / margin) *torch.max(margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_tgt_feat_2)),torch.tensor(0).float().cuda())

                    inter_loss =  inter_loss + ((margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_src_feat_2))) / margin) *torch.max(margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_src_feat_2)),torch.tensor(0).float().cuda())

                    # NOTE: 
                    # inter_loss =  inter_loss + torch.pow(
                    # (margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_src_feat_2))) / margin,
                    # 2) * torch.pow(
                    # torch.max(margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_src_feat_2)),
                    #           torch.tensor(0).float().cuda()), 2.0)

                    # inter_loss =  inter_loss + torch.pow(
                    #     (margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_tgt_feat_2))) / margin,
                    #     2) * torch.pow(
                    #     torch.max(margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_tgt_feat_2)),
                    #             torch.tensor(0).float().cuda()), 2.0)

                    # inter_loss =  inter_loss + torch.pow(
                    #     (margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_tgt_feat_2))) / margin,
                    #     2) * torch.pow(
                    #     torch.max(margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_tgt_feat_2)),
                    #             torch.tensor(0).float().cuda()), 2.0)

                    # inter_loss =  inter_loss + torch.pow(
                    #     (margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_src_feat_2))) / margin,
                    #     2) * torch.pow(
                    #     torch.max(margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_src_feat_2)),
                    #             torch.tensor(0).float().cuda()), 2.0)
        
        intra_cross_scale = 0.
        inter_cross_scale = 0.

        # TODO compare the two selected scales first
        cross_scale = [torch.stack([source_prototypes[0], target_prototypes[1]]),
        torch.stack([source_prototypes[1], target_prototypes[0]])] 
        
        # cross-scale loss 
        for scale_i in range(len(cross_scale)):
            source_prototypes = cross_scale[scale_i][0]
            target_prototypes = cross_scale[scale_i][1]

            # breakpoint()
            for cls_idx in range(self.num_classes-1):
                source_cls_scale_i_tmp_1 = source_prototypes[cls_idx,:]
                target_cls_scale_i_tmp_1 = target_prototypes[cls_idx,:]

                intra_cross_scale = intra_cross_scale + torch.sqrt(self.distance(source_cls_scale_i_tmp_1, target_cls_scale_i_tmp_1))
                
                for cls_idx_next in range(cls_idx+1, self.num_classes-1):
                    source_cls_scale_i_tmp_2 = source_prototypes[cls_idx_next,:]
                    target_cls_scale_i_tmp_2 = target_prototypes[cls_idx_next,:]

            
                    inter_cross_scale =  inter_cross_scale + ((margin - torch.sqrt(self.distance(source_cls_scale_i_tmp_1, source_cls_scale_i_tmp_2))) / margin) *torch.max(margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_src_feat_2)),torch.tensor(0).float().cuda())

                    inter_cross_scale =  inter_cross_scale + ((margin - torch.sqrt(self.distance(target_cls_scale_i_tmp_1, target_cls_scale_i_tmp_2))) / margin) *torch.max(margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_tgt_feat_2)),torch.tensor(0).float().cuda())

                    inter_cross_scale =  inter_cross_scale + ((margin - torch.sqrt(self.distance(source_cls_scale_i_tmp_1, target_cls_scale_i_tmp_2))) / margin) *torch.max(margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_tgt_feat_2)),torch.tensor(0).float().cuda())

                    inter_cross_scale =  inter_cross_scale + ((margin - torch.sqrt(self.distance(target_cls_scale_i_tmp_1, source_cls_scale_i_tmp_2))) / margin) *torch.max(margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_src_feat_2)),torch.tensor(0).float().cuda())
        

        intra_cross_scale = intra_cross_scale / source_prototypes.shape[0]
        intra_loss = intra_loss / source.shape[1] # average over class dim

        # bg_loss = bg_loss /source.shape[1]

        # combinations between each class for two domains
        # at the end of iteration the there is one "next class" being left off, thus -1
        inter_cross_scale = inter_cross_scale / (source_prototypes.shape[0] * (source_prototypes.shape[0] - 1) * 2)
        inter_loss = inter_loss / (source.shape[1] * (source.shape[1] - 1) * 2)
        
        # print(intra_loss)
        # print(inter_loss)

        if not torch.is_tensor(intra_loss):
            intra_loss = torch.as_tensor(intra_loss)
        
        if not torch.is_tensor(inter_loss):
            inter_loss = torch.as_tensor(inter_loss)

        # if not torch.is_tensor(bg_loss):
        #     inter_loss = torch.as_tensor(bg_loss)

        # print(type(intra_loss))
        # print(type(inter_loss))
        
        # inter_loss = torch.tensor(0.)
        # intra_loss = torch.tensor(0.)
        # if isinstance(intra_loss, float) or isinstance(inter_loss, float):
        #     import pdb; pdb.set_trace()

        # return intra_loss.cuda(), inter_loss.cuda(), bg_loss.cuda()
        return intra_loss.cuda(), inter_loss.cuda(), intra_cross_scale.cuda(), inter_cross_scale.cuda()

    def forward(self, outputs, targets, mode='train'):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """


        # import pdb; pdb.set_trace()
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        

        # TODO we only want to load source targets
        if mode == 'train':
            targets = targets[:len(targets)//2] # use src only
        elif mode == 'test':
            pass
        else:
            raise NotImplementedError
        

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        if 'enc_outputs' in outputs:
            import pdb; pdb.set_trace()
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'da_output' in outputs:
            for k, v in outputs['da_output'].items():
                losses[f'loss_{k}'] = self.loss_da(v, use_focal='query' in k)

        if 'class_embeds' in outputs:
            class_embeds = outputs['class_embeds']

            # import pdb; pdb.set_trace()
            losses[f'loss_category_token'] = self.category_token_align_loss(class_embeds)

        if 'prototypes_enc' in outputs:
            # source and target are lists of class prototypes
            source_enc = outputs['prototypes_enc']['src_prototypes_enc']
            target_enc = outputs['prototypes_enc']['tgt_prototypes_enc']
            bg_enc = outputs['prototypes_enc']['tgt_prototypes_bg_enc']

            alpha_values = outputs['prototypes_enc']['alpha_values']
            
            # with torch.autograd.set_detect_anomaly(True):
            intra_loss_enc, inter_loss_enc, \
            intra_loss_enc_cross_scale, inter_loss_enc_cross_scale  = self.contrastive_loss(source_enc, target_enc, bg_enc, alpha_values, margin = self.margin)

            ### multi scale 
            # for s_i in range(len(source_enc)):
                # intra_loss_enc, inter_loss_enc = self.contrastive_loss(source_enc[s_i], target_enc[s_i], alpha_values, margin = self.margin)
        
            # import pdb; pdb.set_trace()
            # losses['bg_loss'] = bg_loss
            losses['loss_intra_class_enc'] = intra_loss_enc
            losses['loss_inter_class_enc'] = inter_loss_enc
            losses['loss_intra_class_enc_cross_scale'] = intra_loss_enc_cross_scale
            losses['loss_inter_class_enc_cross_scale'] = inter_loss_enc_cross_scale

        if 'prototypes_dec' in outputs:
            # source and target are lists of class prototypes
            source_dec = outputs['prototypes_dec']['src_prototypes_dec']
            target_dec = outputs['prototypes_dec']['tgt_prototypes_dec']
            # alpha_values = outputs['alpha_values']
            intra_loss_dec, inter_loss_dec = self.contrastive_loss(source_dec, target_dec, None, margin = self.margin)

            losses['loss_intra_class_dec'] = intra_loss_dec
            losses['loss_inter_class_dec'] = inter_loss_dec

        # TODO aug loss here is used in place of ce loss computed here
        if self.feat_aug:
            mean_source = outputs['prototypes']['src_prototypes']
            mean_target = outputs['prototypes']['tgt_prototypes']
            aug_y = self.aug(mean_source, mean_target, outputs['fc'], outputs['features_source'], outputs['y_s'], outputs['source_labels'], outputs['covariance_target'], self.Lamda)
            loss = self.cross_entropy(aug_y, torch.as_tensor(outputs['source_labels']))
            losses[f'aug_loss'] = loss

        # for debugging
        if self.return_indices:
            return losses, indices
        else:
            return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # breakpoint()
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # batch wise
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class PostProcess_for_target(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_bbox = outputs['boxes']
        out_bbox = out_bbox.unsqueeze(0)

        # assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # prob = out_logits.sigmoid()
        # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        # scores = topk_values
        # topk_boxes = topk_indexes // out_logits.shape[2]
        # labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'boxes': b} for b in boxes]

        return results

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)

        # multiple linear layers initialised as an nn.ModuleList
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # use relu except the last layer
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# where the whole deformable transformer and backbone are initialised
def build(cfg):
    device = torch.device(cfg.DEVICE)

    backbone = build_backbone(cfg)

    transformer = build_deforamble_transformer(cfg)
    # the deformable detr performs object detection
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=cfg.DATASET.NUM_CLASSES,
        num_queries=cfg.MODEL.NUM_QUERIES,
        num_feature_levels=cfg.MODEL.NUM_FEATURE_LEVELS,
        aux_loss=cfg.LOSS.AUX_LOSS,
        with_box_refine=cfg.MODEL.WITH_BOX_REFINE,
        two_stage=cfg.MODEL.TWO_STAGE,
        backbone_align=cfg.MODEL.BACKBONE_ALIGN,
        space_align=cfg.MODEL.SPACE_ALIGN,
        channel_align=cfg.MODEL.CHANNEL_ALIGN,
        instance_align=cfg.MODEL.INSTANCE_ALIGN,
        debug = cfg.DEBUG,
        ema = cfg.EMA,
        feat_aug = cfg.FEAT_AUG,
        cam_viz = cfg.CAM_VIZ,
    )
    if cfg.MODEL.MASKS:
        model = DETRsegm(model, freeze_detr=(cfg.MODEL.FROZEN_WEIGHTS is not None))
    
    matcher = build_matcher(cfg)

    weight_dict = {'loss_ce': cfg.LOSS.CLS_LOSS_COEF, 'loss_bbox': cfg.LOSS.BBOX_LOSS_COEF}
    
    # TODO aug loss is equivalent to the cross entropy loss here, but since we compute aug loss at the end,
    # we also want to put the weighting at the end
    # weight_dict = {'loss_bbox': cfg.LOSS.BBOX_LOSS_COEF}
    weight_dict['loss_giou'] = cfg.LOSS.GIOU_LOSS_COEF
    
    if cfg.MODEL.MASKS:
        weight_dict["loss_mask"] = cfg.LOSS.MASK_LOSS_COEF
        weight_dict["loss_dice"] = cfg.LOSS.DICE_LOSS_COEF
    # TODO this is a hack
    if cfg.LOSS.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.MODEL.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    weight_dict['loss_backbone'] = cfg.LOSS.BACKBONE_LOSS_COEF
    weight_dict['loss_space_query'] = cfg.LOSS.SPACE_QUERY_LOSS_COEF
    weight_dict['loss_channel_query'] = cfg.LOSS.CHANNEL_QUERY_LOSS_COEF
    weight_dict['loss_instance_query'] = cfg.LOSS.INSTANCE_QUERY_LOSS_COEF
    weight_dict['loss_inter_class_enc'] = cfg.LOSS.INTER_CLASS_COEF
    weight_dict['loss_intra_class_enc'] = cfg.LOSS.INTRA_CLASS_COEF
    weight_dict['loss_inter_class_enc_cross_scale'] = cfg.LOSS.INTER_CLASS_COEF
    weight_dict['loss_intra_class_enc_cross_scale'] = cfg.LOSS.INTRA_CLASS_COEF
    # weight_dict['bg_loss'] = cfg.LOSS.BG_LOSS_COEF

    # weight_dict['loss_inter_class_dec'] = cfg.LOSS.INTER_CLASS_COEF
    # weight_dict['loss_intra_class_dec'] = cfg.LOSS.INTRA_CLASS_COEF

    # put ce/aug loss at the end due to order of compute
    # weight_dict['loss_aug'] = cfg.LOSS.AUG_LOSS_COEF

    # TODO: remove labels for now since we already got aug loss
    losses = ['labels', 'boxes', 'cardinality']
    # losses = ['boxes', 'cardinality']

    if cfg.MODEL.MASKS:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    # TODO: return indices matching indices for debugging

    if cfg.DEBUG:
        criterion = SetCriterion(cfg.DATASET.NUM_CLASSES, matcher, weight_dict, losses, focal_alpha=cfg.LOSS.FOCAL_ALPHA, da_gamma=cfg.LOSS.DA_GAMMA, return_indices=True, margin = cfg.LOSS.MARGIN)

    else:
        criterion = SetCriterion(cfg.DATASET.NUM_CLASSES, matcher, weight_dict, losses, focal_alpha=cfg.LOSS.FOCAL_ALPHA, da_gamma=cfg.LOSS.DA_GAMMA, return_indices=False, margin = cfg.LOSS.MARGIN, feat_aug=cfg.FEAT_AUG, Lamda=cfg.LOSS.LAMDA, eos_coef=cfg.LOSS.EOS_COEF)
    
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    postprocessors_target = {'bbox': PostProcess_for_target()}
    if cfg.MODEL.MASKS:
        postprocessors['segm'] = PostProcessSegm()
        if cfg.DATASET.DATASET_FILE == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors, postprocessors_target
