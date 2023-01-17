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

import copy
from typing import Optional, List
import math
import torchvision

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
# TODO: use category_feature_attention to get category specific encoder features before learning the query token
from models.utils import DomainAttention, GradientReversal, remove_mask_and_warp, category_feature_attention
# TODO: siamese attention
# from models.attention import SingleHeadSiameseAttention

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 space_align=False, channel_align=False, instance_align=False, debug=True):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        self.space_align = space_align
        self.channel_align = channel_align
        self.instance_align = instance_align

        self.num_feature_levels = num_feature_levels

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points, space_align, channel_align)

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, instance_align)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        # initialise different query tokens
        if space_align:
            self.space_query = nn.Parameter(torch.empty(1, 1, d_model))
        if channel_align:
            # self.channel_query is actually an embedding layer for channel query
            # We keep the name for consistency
            self.channel_query = nn.Linear(d_model, 1)
            self.grl = GradientReversal()
        if instance_align:
            self.instance_query = nn.Parameter(torch.empty(1, 1, d_model))

        self._reset_parameters()

        self.debug = debug

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    # for two-stage
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src) # appends 4 flattened scales
            mask_flatten.append(mask)

        # import pdb; pdb.set_trace()
        src_flatten = torch.cat(src_flatten, 1) # torch.Size([2, 18669, 256])
        mask_flatten = torch.cat(mask_flatten, 1)

        # import pdb; pdb.set_trace()
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        space_query, channel_query, instance_query = None, None, None
        if self.training:
            if self.space_align:
                # initialise space query
                space_query = self.space_query.expand(src_flatten.shape[0], -1, -1)
            if self.channel_align:
                # initialise channel query
                # change view to channel features
                src_warped, pos_warped = remove_mask_and_warp(
                    src_flatten, lvl_pos_embed_flatten, mask_flatten, level_start_index, spatial_shapes
                )

                # gradient reversal layer
                channel_query = self.channel_query(self.grl(src_warped+pos_warped)).flatten(0, 1).transpose(1, 2)
            
        # encoder
        ### encoder inputs use the same set of features
        ### retrieve space, channel query from each layer
        # TODO: pass a class query to encoder without passing it to the self attention layer

        # import pdb; pdb.set_trace()
        memory, space_query, channel_query = self.encoder(
            src_flatten, space_query, channel_query, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten
        )

        # space_query (6, 2, 1, 256)
        da_output = {}
        if self.training:
            if self.space_align:
                # store in da_output
                da_output['space_query'] = torch.cat(space_query, dim=1)

                # import pdb; pdb.set_trace()
            if self.channel_align:
                da_output['channel_query'] = torch.cat(channel_query, dim=1)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            # extract the class embedding of the last decoder layer
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            
            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            # decoder outputs for and object queries for two-stage
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            # decoder outputs for and object queries for one-stage
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        if self.training and self.instance_align:
            # instance queries are initialised as object queries
            instance_query = self.instance_query.expand(tgt.shape[0], -1, -1)
        
        # import pdb; pdb.set_trace()
        # decoder
        hs, inter_references, instance_query = self.decoder(
            tgt, instance_query, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten
        )


        if self.training and self.instance_align:
            da_output['instance_query'] = instance_query

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, da_output

        # return hs, init_reference_out, inter_references_out, None, None, da_output, query_embed

        # TODO return query_embed for cascade arch
        # return hs, init_reference_out, inter_references_out, None, None, da_output, query_embed
        return hs, memory, init_reference_out, inter_references_out, None, None, da_output

    def forward_supp_branch(self, srcs, masks, pos_embeds, query_embed, support_boxes, scale=1):
        assert query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            # 5 is num of levels
            src = src.flatten(2).transpose(1, 2) # (lvl, channel, flattened space)
            mask = mask.flatten(1) # torch.Size([5, 441])
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # torch.Size([5, 441, 256])


            # TODO: remove multi-scale consideration for forward sup branch
            if self.num_feature_levels > 1:
                # broadcast addition
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            # TODO: only consider positional embedding
            else:
                lvl_pos_embed = pos_embed
            
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # tokenized inputs
        # 
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        ### call forward_supp_branch
        inter_category_codes_list = self.encoder.forward_supp_branch(
            src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, support_boxes, scale=scale
        )

        return inter_category_codes_list

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 space_align=False, channel_align=False):
        super().__init__()

        self.space_align = space_align
        self.channel_align = channel_align
        if space_align:
            # forward: multihead --> residual --> linear (return query only)
            ### cross attention
            self.space_attn = DomainAttention(d_model, n_heads, dropout)
        if channel_align:
            self.channel_attn = DomainAttention(d_model, n_heads, dropout)

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src
    
    # src will be updated in the i+1 layer
    def forward(self, src, space_query, channel_query, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention (updated in the direction opposite to the gradient)
        
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if self.training:
            if self.space_align:
                # self attention -> cross attention -> linear layer

                # import pdb; pdb.set_trace()
                space_query = self.space_attn(space_query, src, pos, padding_mask)

            if self.channel_align:
                ### channel features are warped after the first self-attention layer
                src_warped, pos_warped = remove_mask_and_warp(src, pos, padding_mask, level_start_index, spatial_shapes)
                channel_query = self.channel_attn(
                    channel_query, # bsz * num_feature_levels, 1, H*W
                    src_warped.flatten(0, 1).transpose(1, 2), # bsz * num_feature_levels, C, H*W
                    pos_warped.flatten(0, 1).transpose(1, 2)
                )

        # ffn
        # src is not affected by the query tokens
        src = self.forward_ffn(src)

        return src, space_query, channel_query

    def forward_supp_branch(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask, support_boxes, scale=1):
        
        # self attention
        ### support features are flattened (in the wrapper) and then fed to an encoder

        ### the self attention module is the msdeformable attention class
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src) # torch.Size([1, 1176, 256])

        # import pdb; pdb.set_trace()

        support_img_h, support_img_w = spatial_shapes[0, 0], spatial_shapes[0, 1] # image feature spatial shapes

        # spatial scale is used to warp the bounding boxes into scale that
        # accomodates the feature maps, hence a scale is defined
        # reshape src features by the specified support_img_h and support_img_w


        ### 32 since we are only using single scale

        # torch.Size([11, 256, 7, 7]) before mean
        # if multiscales are considered, multiple roi align may be called
        
        ### support boxes may be empty
        # if len(support_boxes[0])==0:
        #     import pdb; pdb.set_trace()

        if scale == 0:
            spatial_scale = 1/8.0
        elif scale == 1:
            spatial_scale = 1/16.0
        elif scale == 2:
            spatial_scale = 1/32.0

        # 
        supp_roi = torchvision.ops.roi_align(src.transpose(1, 2).reshape(src.shape[0], -1, support_img_h, support_img_w), support_boxes, output_size=(7, 7), spatial_scale=spatial_scale, aligned=True).mean(3).mean(2)
        
        # debug
        # support_boxes_cpu = []
        # for box in support_boxes:
        #     box = box.cpu()
        #     support_boxes_cpu.append(box)

        # import pdb; pdb.set_trace()
        # supp_roi = torchvision.ops.roi_align(src.transpose(1, 2).reshape(src.shape[0], -1, support_img_h, support_img_w).cpu(), support_boxes_cpu, output_size=(7, 7), spatial_scale=spatial_scale, aligned=True).mean(3).mean(2)

        ### actual category code
        category_code = supp_roi.sigmoid() # (5, 256)
        
        # category code: torch.Size([5, 441, 256])
        # import pdb; pdb.set_trace()
        if self.QSAttn:
            # siamese attention (single-head)
            # tsp is the task_positional_encoding
            # q, k, v, tsp
            src = self.siamese_attn(src, inverse_sigmoid(category_code).unsqueeze(0).expand(src.shape[0], -1, -1),
                                         category_code.unsqueeze(0).expand(src.shape[0], -1, -1))

            # ffn
            src = self.forward_ffn(src)
        else:
            src = self.forward_ffn(src)

        # import pdb; pdb.set_trace()
        
        ### need to output src in order to get category codes from different encoder layers
        return src, category_code


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, space_query, channel_query, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        space_querys = []
        channel_querys = []

        # get space and channel queries from each layer
        for _, layer in enumerate(self.layers):
            # output, space_query, channel_query will be pass to the next layer
            output, space_query, channel_query = layer(
                output, space_query, channel_query, pos, reference_points, spatial_shapes, level_start_index, padding_mask
            )

            # attach queries from every layer
            space_querys.append(space_query)
            channel_querys.append(channel_query)

        return output, space_querys, channel_querys

    def forward_supp_branch(self, src, spatial_shapes, level_start_index, valid_ratios, pos, padding_mask, support_boxes, scale=1):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        category_codes = []
        for i, layer in enumerate(self.layers):
            # output is updated for each increased layer index
            # output is the encoder outputs, category_code is the sigmoided roi pooled features
            output, category_code = layer.forward_supp_branch(
                output, pos, reference_points, spatial_shapes, level_start_index, padding_mask, support_boxes, scale=scale
            )
            category_codes.append(category_code)
        return category_codes
class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 instance_align=False):
        super().__init__()

        self.instance_align = instance_align
        if instance_align:
            self.instance_attn = DomainAttention(d_model, n_heads, dropout)

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        # import pdb; pdb.set_trace()
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, instance_query, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        # target sequence is object query
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)

        # import pdb; pdb.set_trace()
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # tgt is the object query
        if self.training and self.instance_align:
            instance_query = self.instance_attn(instance_query, tgt, query_pos)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, instance_query


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, instance_query, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output, instance_query = layer(
                output, instance_query, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask
            )

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), instance_query

        # https://github.com/fundamentalvision/Deformable-DETR/issues/43
        return [output], [reference_points], instance_query


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(cfg):
    return DeformableTransformer(
        d_model=cfg.MODEL.HIDDEN_DIM,
        nhead=cfg.MODEL.NHEADS,
        num_encoder_layers=cfg.MODEL.ENC_LAYERS,
        num_decoder_layers=cfg.MODEL.DEC_LAYERS,
        dim_feedforward=cfg.MODEL.DIM_FEEDFORWARD,
        dropout=cfg.MODEL.DROPOUT,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=cfg.MODEL.NUM_FEATURE_LEVELS,
        dec_n_points=cfg.MODEL.DEC_N_POINTS,
        enc_n_points=cfg.MODEL.ENC_N_POINTS,
        two_stage=cfg.MODEL.TWO_STAGE,
        two_stage_num_proposals=cfg.MODEL.NUM_QUERIES,
        space_align=cfg.MODEL.SPACE_ALIGN,
        channel_align=cfg.MODEL.CHANNEL_ALIGN,
        instance_align=cfg.MODEL.INSTANCE_ALIGN,
        debug=cfg.DEBUG)
