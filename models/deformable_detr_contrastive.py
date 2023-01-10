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
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import cv2 as cv
import tracemalloc

import torchvision

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer_contrastive import build_deforamble_transformer
from .utils import GradientReversal
import copy
from .memory_ema import Memory


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                 backbone_align=False, space_align=False, channel_align=False, instance_align=False, debug=False, ema=False):
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
        keep_rate = 0.9
        # TODO: returns updated prototypes
        if ema:
            self.memory = Memory(num_classes, transformer.d_model, keep_rate = keep_rate)

        self.ema = ema
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        
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

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
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
            for layer in self.instance_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)


    def forward(self, samples: NestedTensor, targets):
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

        # if self.debug:
        #     features, pos = self.backbone(samples)

        #     # import pdb; pdb.set_trace()
        #     features = nn.AdaptiveAvgPool2d((1,1))(features[0].tensors)
        #     features = features.mean(3).mean(2)
        
        #     return features

        features, pos = self.backbone(samples)

        # import pdb; pdb.set_trace()

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
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight # weights are the only things you need
        
        # TODO: debug mode, only allow debug mode at test time and targets are returned only at test time
        if self.debug:
            hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, da_output = self.transformer(srcs, masks, pos, query_embeds)
            # import pdb; pdb.set_trace()
        else:
            # tracemalloc.start()
            hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, da_output = self.transformer(srcs, masks, pos, query_embeds)
            # print(tracemalloc.get_traced_memory())
        
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
        # torch.Size([6, 2, 300, 4])
        outputs_coord = torch.stack(outputs_coords)

        # import pdb; pdb.set_trace()

        # if scale == 0:
        #     spatial_scale = 1/8.0
        # elif scale == 1:
        #     spatial_scale = 1/16.0
        # elif scale == 2:
        #     spatial_scale = 1/32.0 # default

        # TODO: inter-, intra- contrastive loss
        if self.training:
            B = outputs_class.shape[1] # get batch size
            assert B == memory.shape[0]

            last_layer_out = outputs_class[-1] # bs is after layer num
            outputs_class_conf = F.softmax(last_layer_out, -1)

            thresh = 0.6

            # size: [torch.Size([31, 2]), torch.Size([11, 2])]
            keep = [torch.nonzero(outputs_class_conf[b]>thresh).unsqueeze(0) for b in range(outputs_class_conf.shape[0])] # batch wise


            # single features
            feature_w = features[0].tensors.shape[-1] # w
            feature_h = features[0].tensors.shape[-2] # h
            feature_c = memory.shape[-1]

            # memory = memory[0] # src
            memory_reshaped = memory.reshape(-1, feature_h, feature_w, feature_c).permute(0,3,1,2)

            # batched rois
            list_of_rois = []

            # unsorted
            list_of_labels = [] # currently one for src, one for tgt
            list_of_scores = []
            rescaled_boxes = [] # boxes rescaled back to fit the image dim

            # collect source and target rois
            for batch_idx in range(B):
                keep_tmp = keep[batch_idx][:,:,0].tolist()[0] # get list of indices
                keep_label_idx = keep[batch_idx][:,:,1].tolist()[0]

                boxes = outputs_coord[-1][batch_idx][keep_tmp] # get last layer predicted boxes

                # TODO rescale boxes (for batch size >1, this is to be modified)
                boxes_rescaled = [box_ops.box_cxcywh_to_xyxy(boxes)] # src only, batch size = 1
                # and from relative [0, 1] to absolute [0, height] coordinates
                img_sizes = torch.stack([t["size"] for t in targets], dim=0)
                img_h, img_w = img_sizes.unbind(1)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

                for b in range(boxes_rescaled[0].shape[0]):
                    boxes_rescaled[0][b] *= scale_fct[0] # batch_size = 1, one image

                # import pdb; pdb.set_trace()
                # rois = torchvision.ops.roi_align(memory_reshaped, boxes_rescaled, output_size=(7, 7), spatial_scale=1/32.0, aligned=True).mean(3).mean(2)

                rescaled_boxes.append(boxes_rescaled[0]) # delist
                list_of_labels.append(keep_label_idx)
                                
                scores = torch.argmax(outputs_class_conf[batch_idx][keep_tmp], dim=1)
                list_of_scores.append(scores)

                # for box_i in range(boxes.shape[0]):
                    # memory --> torch.Size([1, c_dim, spatial_dim, spatial_dim])
                    # boxes --> torch.Size([1, spatial_dim, c_dim])

            # num_points = rescaled_boxes[0].shape[-1]
            # affine_warp = torch.zeros((3,3))
            # diff = max(rescaled_boxes[0].shape[0], rescaled_boxes[1].shape[0]) - min(rescaled_boxes[0].shape[0], rescaled_boxes[1].shape[0])
            

            # TODO pad proposal boxes
            # import pdb; pdb.set_trace()
            
            for batch_idx in range(B):
                # rois: torch.Size([31, 256])
                rois = torchvision.ops.roi_align(memory_reshaped[batch_idx].unsqueeze(0), [rescaled_boxes[batch_idx]], output_size=(7, 7), spatial_scale=1/32.0, aligned=True).mean(3).mean(2)
                list_of_rois.append(rois)

            # labels existing (sorted)
            source_labels = []
            target_labels = []

            weighted_rois_source = []
            weighted_rois_target = []

            for i in range(B):
                label_set = list(set(list_of_labels[i])) # label set for each sample 

                # this is only valid as starting index is 0
                if i//(B//2)==0:
                    source_labels.append(label_set)

                    # import pdb; pdb.set_trace()
                    # confidence guided merging
                    weighted_rois = list_of_rois[i]*list_of_scores[i].unsqueeze(-1) # reweighted rois
                    weighted_rois_source.append(weighted_rois)
                else:
                    target_labels.append(label_set)
                    weighted_rois = list_of_rois[i]*list_of_scores[i].unsqueeze(-1)
                    weighted_rois_target.append(weighted_rois)
            
            # import pdb; pdb.set_trace()

            assert len(weighted_rois_source) == len(source_labels) & len(weighted_rois_target) == len(target_labels),\
            "label and roi lists are not one-to-one"
            
            # import pdb; pdb.set_trace()
            # extract src rois
            source_rois = [] # list of tensors: stores some rois for each class
            target_rois = []
            
            source_labels_all = list_of_labels[:B//2]
            target_labels_all = list_of_labels[B//2:]

            # source_labels is a list of label sets, list_of_labels contain lists of all labels
            assert len(source_labels) == len(source_labels_all) == len(weighted_rois_source), "length should be the same per sample"

            # import pdb; pdb.set_trace()
            # since labels are sorted, the correponding rois are one-to-one with respect to labels
            for i in range(len(source_labels)):
                tmp = []
                # single label
                for label in source_labels[i]:
                    matched_src_idx = torch.nonzero(torch.as_tensor(source_labels_all[i])==label).squeeze(1) # tensor  # e.g matched_idx: torch.Size([11, 1])
                    rois_source = weighted_rois_source[i][matched_src_idx].unsqueeze(1) # tensor
                    # import pdb; pdb.set_trace()
                    tmp.append(rois_source)

                source_rois.append(tmp)

            for i in range(len(target_labels)):
                tmp = []
                for label in target_labels[i]:
                    # e.g matched_idx: torch.Size([11, 1])
                    matched_tgt_idx = torch.nonzero(torch.as_tensor(target_labels_all[i])==label).squeeze(1) # tensor
                    rois_target = weighted_rois_target[i][matched_tgt_idx].unsqueeze(1) # tensor
                    tmp.append(rois_target)
                target_rois.append(tmp)
            
            # aggregate rois for each class
            src_prototypes = torch.zeros((self.num_classes, self.hidden_dim)).cuda()
            tgt_prototypes = torch.zeros((self.num_classes, self.hidden_dim)).cuda()

            epsilon = 1e-6
            # this is to also index labels (one-to-one)
            # batch length
            for i in range(len(source_rois)):
                roi_sample_tmp = source_rois[i] # some rois for a single class
                # some rois
                for j in range(len(roi_sample_tmp)):
                    # src_prototypes.append(torch.sum(source_roi, dim=0)/(torch.sum(list_of_scores[0]) + epsilon))
                    aggregate = torch.sum(roi_sample_tmp[j], dim=0)/(torch.sum(list_of_scores[0]) + epsilon)
                    # import pdb; pdb.set_trace()
                    # store prototype at the corresponding cls index position
                    src_prototypes[source_labels[i][j]] = aggregate
                    
            for i in range(len(target_rois)):
                roi_sample_tmp = target_rois[i]
                for j in range(len(roi_sample_tmp)):
                    # tgt_prototypes.append(torch.sum(target_rois[i], dim=0)/(torch.sum(list_of_scores[1]) + epsilon))
                    aggregate = torch.sum(roi_sample_tmp[j], dim=0)/(torch.sum(list_of_scores[1]) + epsilon)
                    tgt_prototypes[target_labels[i][j]] = aggregate
            
            if self.ema:
                #memory monitoring
                tracemalloc.start()
                updated_src_prototypes, updated_tgt_prototypes = self.memory(src_prototypes, tgt_prototypes)

                # self.source_prototype = updated_src_prototypes
                # self.target_prototype = updated_tgt_prototypes

                print(tracemalloc.get_traced_memory())
                # peak memory
                if tracemalloc.get_traced_memory()[1] > 5000000000:
                    quit()

            else:
                updated_src_prototypes = src_prototypes
                updated_tgt_prototypes = tgt_prototypes

        if self.training and self.uda:
            B = outputs_class.shape[1]
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
        
        # import pdb; pdb.set_trace()
        # store predictions in dictionary
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        # TEST:
        # updated_tgt_prototypes = torch.zeros((B//2, self.num_classes, self.hidden_dim)).cuda()

        # TODO add prototypes to outputs
        if self.training:
            # out['prototypes'] = {'src_prototypes': src_prototypes, 'tgt_prototypes': tgt_prototypes}
            # TODO: ema update
            out['prototypes'] = {'src_prototypes': updated_src_prototypes, 'tgt_prototypes': updated_tgt_prototypes}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        
        # discriminator outputs
        if self.training and self.uda:
            out['da_output'] = da_output

        if self.debug:
            return out, features, memory, hs
        else:
            return out
    

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
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, da_gamma=2, return_indices=False, margin = 1):
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


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # (1, 300, 9)

        idx = self._get_src_permutation_idx(indices) # (tensor([0, 0]), tensor([175, 186]))

        # import pdb; pdb.set_trace()

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # e.g tensor([3, 3])

        # TODO for single class, need to convert target_classes_o to ones since the target_classes_o
        # elements will be used for scattering the one hot vectors later on
        # target_classes_o = torch.ones_like(target_classes_o)

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) # (1, 300)

        target_classes[idx] = target_classes_o # become ones


        # (1, 300, 10)
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)

        
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1) # (dim, index, src tensor)

        # (1, 300, 9)
        target_classes_onehot = target_classes_onehot[:,:,:-1]

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    # for contrastive loss
    def distance(self, src_feat, tgt_feat):
        output = torch.pow(src_feat - tgt_feat, 2.0).mean()
        return output
    
    # TODO in this implementation, the intra class loss between samples is not considered
    def contrastive_loss(self, source, target, margin=1):
        # source and target are tensors

        # intra class: only between source and target
        # inter class: between each class --> three combinations
        # max length is self.num_classes

        source = source.view(-1, source.shape[2])
        target = target.view(-1, target.shape[2])

        intra_loss = 0
        inter_loss = 0

        for cls_idx in range(self.num_classes):
            # i gives a fixed class
            tmp_src_feat_1 = source[cls_idx, :]
            tmp_tgt_feat_1 = target[cls_idx, :]
            
            # import pdb; pdb.set_trace()
            # intra
            intra_loss = intra_loss + self.distance(tmp_src_feat_1, tmp_tgt_feat_1)

            # inter takes into account the current and all other classes
            for cls_idx_next in range(cls_idx+1, self.num_classes):
                tmp_src_feat_2 = source[cls_idx_next, :]
                tmp_tgt_feat_2 = target[cls_idx_next, :]

                # import pdb; pdb.set_trace()
                inter_loss = inter_loss + torch.pow(
                    (margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_src_feat_2))) / margin,
                    2) * torch.pow(
                    torch.max(margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_src_feat_2)),
                              torch.tensor(0).float().cuda()), 2.0)

                inter_loss = inter_loss + torch.pow(
                    (margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_tgt_feat_2))) / margin,
                    2) * torch.pow(
                    torch.max(margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_tgt_feat_2)),
                              torch.tensor(0).float().cuda()), 2.0)

                inter_loss = inter_loss + torch.pow(
                    (margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_tgt_feat_2))) / margin,
                    2) * torch.pow(
                    torch.max(margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_tgt_feat_2)),
                              torch.tensor(0).float().cuda()), 2.0)

                inter_loss = inter_loss + torch.pow(
                    (margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_src_feat_2))) / margin,
                    2) * torch.pow(
                    torch.max(margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_src_feat_2)),
                              torch.tensor(0).float().cuda()), 2.0)
        
        # import pdb; pdb.set_trace()
        # average over all classes*batch_dim 
        intra_loss = intra_loss / source.shape[0] 
        # combinations between each class for two domains
        inter_loss = inter_loss / (source.shape[0] * (source.shape[0] - 1) * 2) # at the of the iteration the there will be one "next class" being left off
    

        # import pdb; pdb.set_trace()

        return intra_loss.cuda(), inter_loss.cuda()

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

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

        if 'prototypes' in outputs:
            # source and target are lists of class prototypes
            source = outputs['prototypes']['src_prototypes']
            target = outputs['prototypes']['tgt_prototypes']
            # import pdb; pdb.set_trace()
            
            intra_loss, inter_loss = self.contrastive_loss(source, target, margin = self.margin)
            # import pdb; pdb.set_trace()
            losses[f'loss_intra'] = intra_loss
            losses[f'inter_loss'] = inter_loss

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
    )
    if cfg.MODEL.MASKS:
        model = DETRsegm(model, freeze_detr=(cfg.MODEL.FROZEN_WEIGHTS is not None))
    
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.LOSS.CLS_LOSS_COEF, 'loss_bbox': cfg.LOSS.BBOX_LOSS_COEF}
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
    weight_dict['loss_inter_class'] = cfg.LOSS.INTER_CLASS_COEF
    weight_dict['loss_intra_class'] = cfg.LOSS.INTRA_CLASS_COEF

    losses = ['labels', 'boxes', 'cardinality']
    if cfg.MODEL.MASKS:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    # TODO: return indices matching indices for debugging

    if cfg.DEBUG:
        criterion = SetCriterion(cfg.DATASET.NUM_CLASSES, matcher, weight_dict, losses, focal_alpha=cfg.LOSS.FOCAL_ALPHA, da_gamma=cfg.LOSS.DA_GAMMA, return_indices=True, margin = cfg.LOSS.MARGIN)

    else:
        criterion = SetCriterion(cfg.DATASET.NUM_CLASSES, matcher, weight_dict, losses, focal_alpha=cfg.LOSS.FOCAL_ALPHA, da_gamma=cfg.LOSS.DA_GAMMA, return_indices=False, margin = cfg.LOSS.MARGIN)
    
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    postprocessors_target = {'bbox': PostProcess_for_target()}
    if cfg.MODEL.MASKS:
        postprocessors['segm'] = PostProcessSegm()
        if cfg.DATASET.DATASET_FILE == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors, postprocessors_target
