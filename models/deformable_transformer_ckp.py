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

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.utils.checkpoint import checkpoint

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
from models.utils import DomainAttention, GradientReversal, remove_mask_and_warp

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 space_align=False, channel_align=False, instance_align=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        self.space_align = space_align
        self.channel_align = channel_align
        self.instance_align = instance_align

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points, space_align, channel_align)

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        # num_encoder_layers = 6

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, instance_align)

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        # num_decoder_layers = 6
        # return_intermediate_dec = True, see L599

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * N, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        if space_align:
            self.space_query = nn.Parameter(torch.empty(1, 1, d_model))  # (1, 1, d_model)
        if channel_align:
            # self.channel_query is actually an embedding layer for channel query
            # We keep the name for consistency
            self.channel_query = nn.Linear(d_model, 1)
            self.grl = GradientReversal()
        if instance_align:
            self.instance_query = nn.Parameter(torch.empty(1, 1, d_model))

        self._reset_parameters()

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

    # We don't need this(Two stage)
    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    # We don't need this(Two stage)
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float, device=memory.device))
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
        # srcs: [(N, C, H_0, W_0),
        #        (N, C, H_1, W_1),
        #        (N, C, H_2, W_2),
        #        (N, C, H_3,  W_3)]
        # masks: [(N, H_0, W_0),
        #         (N, H_1, W_1),
        #         (N, H_2, W_2),
        #         (N, H_3, W_3)]
        # pos_embeds: [(N, C, H_0, W_),
        #              (N, C, H_1, W_1),
        #              (N, C, H_2, W_2),
        #              (N, C, H_3, W_3)]
        # num_queries: (#query, d_model * 2)
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []  # flatten feature along spatial axes for concatenating along spatial axes
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):  # lvl means level or scale
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # (N, HW, C), flatten along dim=2~3
            mask = mask.flatten(1)  # (N, HW), flatten along dim=1~2
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # (N, HW, C)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)  # (N, HW, C), level_embed = multi_scale level(concate multi scale feature)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # spatial_shapes: [(H_0, W_0), (H_1, W_1), (H_2, W_2), (H_3, W_3)]
 
        src_flatten = torch.cat(src_flatten, 1)  # (N, H_0W_0+...H_3W_3, C), cat so that attention can be computed across multi-scale features
        mask_flatten = torch.cat(mask_flatten, 1)  # (N, H_0W_0+...H_3W_3)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # (N, H_0W_0+...H_3W_3, C)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  # (#lvl, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (1,), tensor([0])
            spatial_shapes.prod(1).cumsum(0)[:-1]
            # spatial_shapes.prod(1) = tensor([H_0W_0=14028, H_1W_1=3528, H_2W_2=882, H_3W_3=231])
            # spatial_shapes.prod(1).cumsum(0) = tensor([H_0W_0=14028,
            #                                            H_0W_0+H_1W_1=17556,
            #                                            H_0W_0+H_1W_1+H_2W_2=18438,
            #                                            H_0W_0+H_1W_1+H_2W_2+H_3W_3=18669])
        ))  # tensor([0, H_0W_0=14028, H_0W_0+H_1W_1=17556, H_0W_0+H_1W_1+H_2W_2=18438])
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # (2, 4, 2)

        space_query, channel_query, instance_query = None, None, None

        if self.training:
            if self.space_align:
                space_query = self.space_query.expand(src_flatten.shape[0], -1, -1)  # (1, 1, d_model) → (N, 1, d_model)
            if self.channel_align:
                src_warped, pos_warped = remove_mask_and_warp(
                    src_flatten, lvl_pos_embed_flatten, mask_flatten, level_start_index, spatial_shapes
                )
                channel_query = self.channel_query(self.grl(src_warped+pos_warped)).flatten(0, 1).transpose(1, 2)

        # encoder
        memory, space_query, channel_query = self.encoder(
            src_flatten, space_query, channel_query, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten
        )
        # memory: (N, H_0W_0+...H_3W_3=18669, C), transformed feature map outputed from encoder
        # space_query: list of space queries collected from different layers
        # channel_query: list of channel queries collected from different layers

        da_output = {}
        if self.training:
            if self.space_align:
                da_output['space_query'] = torch.cat(space_query, dim=1)
            if self.channel_align:
                da_output['channel_query'] = torch.cat(channel_query, dim=1)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            # query_embed: (#query, d_model * 2)
            query_embed, tgt = torch.split(query_embed, c, dim=1)  # tgt: object queries, query_embed: positional embedding of object queries
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)  # (N, #query, d_model)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)  # (N, #query, d_model)

            reference_points = self.reference_points(query_embed).sigmoid()  # (N, #query, 2)

            init_reference_out = reference_points

        if self.training and self.instance_align:
            instance_query = self.instance_query.expand(tgt.shape[0], -1, -1)  # (N, 1, d_model)

        # decoder
        import pdb; pdb.set_trace()
        hs, inter_references, instance_query = self.decoder(
            tgt, instance_query, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten
        )
        # hg: (#decoder_layer, N, #query, d_model), object query tensor outputed from decoder
        # inter_references: (#decoder_layer, N, #query, 4), predicted box coordinates from decoder
        # instance_query: (N, 1, d_model)

        if self.training and self.instance_align:
            da_output['instance_query'] = instance_query

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, da_output
        return hs, init_reference_out, inter_references_out, None, None, da_output


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
        
    def ffn_e1(self,src):
        def inner_forward_e1(src):
            src2 = self.activation(self.linear1(src))
            return src2
        if src.requires_grad:
            src2 = checkpoint(inner_forward_e1, src)
        else:
            src2 = self.activation(self.linear1(src))
        return src2

    def ffn_e2(self,src2):
        def inner_forward_e2(src2):
            src2 = self.linear2(src2)
            return src2
        if src2.requires_grad:
            src2 = checkpoint(inner_forward_e2, src2)
        else:
            src2 = self.linear2(src2)
        return src2
        
    def forward(self, src, space_query, channel_query, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # src: (N, H_0W_0+...H_3W_3, C)
        # space_query: (N, 1, d_model)
        # channel_query: ?
        # pos: (N, H_0W_0+...H_3W_3, C)
        # reference_points: ?
        # spatial_shapes: (#lvl, 2), the feature shape in each level
        # level_start_index: tensor([0, H_0W_0=14028, H_0W_0+H_1W_1=17556, H_0W_0+H_1W_1+H_2W_2=18438])
        # padding_mask: (N, H_0W_0+...H_3W_3)

        # self attention
        # src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        def sa(src, pos, reference_points, spatial_shapes, level_start_index, padding_mask):
            return self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src2 = checkpoint(sa, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        
        self_attn: q: src, k: src, v: src
        self_attn: q: src, k: [src, source_domain_token, target_domain_token], v: [src, source_domain_token, target_domain_token]
                attn_w * src + attn_w * source_domain_token + attn_w * target_domain_token
                
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if self.training:
            if self.space_align:
                space_query = self.space_attn(space_query, src, pos, padding_mask)  # (N, 1, d_model), space_query is updated by residual connection while src & pos remains
            if self.channel_align:
                src_warped, pos_warped = remove_mask_and_warp(src, pos, padding_mask, level_start_index, spatial_shapes)   
                
                channel_query = self.channel_attn(
                    channel_query, # bsz * num_feature_levels, 1, H*W
                    src_warped.flatten(0, 1).transpose(1, 2), # bsz * num_feature_levels, C, H*W
                    pos_warped.flatten(0, 1).transpose(1, 2)
                )

        # ffn
        src2 = self.ffn_e1(src)
        src2 = self.dropout2(src2)
        src2 = self.ffn_e2(src2)
        src = src + self.dropout3(src2)
        src = self.norm2(src) 

        return src, space_query, channel_query


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float, device=device),  # (H_,)
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float, device=device)   # (W_,)
            )
            # ref_y: (H_, W_), [[0.5, 0.5, ..., 0.5],
            #                   [1.5, 1.5, ..., 1.5],
            #                   ...,
            #                   [83.5, 83.5, ..., 83.5]]
            # ref_x: (H_, W_), [[0.5, 1.5, ..., 166.5],
            #                   [0.5, 1.5, ..., 166.5],
            #                   ...,
            #                   [0.5, 1.5, ..., 166.5]]
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)  # (1, H_W_) / ?
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)  # (1, H_W_) / ?
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, space_query, channel_query, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        # src: (N, H_0W_0+...H_3W_3, C)
        # space_query: (N, 1, d_model)
        # channel_query: ?
        # spatial_shapes: (#lvl, 2), the feature shape in each level
        # level_start_index: tensor([0, H_0W_0=14028, H_0W_0+H_1W_1=17556, H_0W_0+H_1W_1+H_2W_2=18438])
        # valid_ratios: ?
        # pos: (N, H_0W_0+...H_3W_3, C)
        # padding_mask: (N, H_0W_0+...H_3W_3)

        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        space_querys = []
        channel_querys = []
        for _, layer in enumerate(self.layers):
            output, space_query, channel_query = layer(
                output, space_query, channel_query, pos, reference_points, spatial_shapes, level_start_index, padding_mask
            )
            space_querys.append(space_query)  # collect all space queries from different layers
            channel_querys.append(channel_query)  # collect all channel queries from different layers

        return output, space_querys, channel_querys


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
        
    def ffn_d1(self,tgt):
        def inner_forward_d1(tgt):
            tgt2 = self.activation(self.linear1(tgt))
            return tgt2
        if tgt.requires_grad:
            tgt2 = checkpoint(inner_forward_d1, tgt)
        else:
            tgt2 = self.activation(self.linear1(tgt))
        return tgt2

    def ffn_d2(self,tgt2):
        def inner_forward_d2(tgt2):
            tgt2 = self.linear2(tgt2)
            return tgt2
        if tgt2.requires_grad:
            tgt2 = checkpoint(inner_forward_d2, tgt2)
        else:
            tgt2 = self.linear2(tgt2)
        return tgt2
        
    def forward(self, tgt, instance_query, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # tgt: (N, #query, d_model), object queries
        # instance_query: (N, 1, d_model)
        # query_pos: (N, #query, d_model)
        # reference_points: (2, #query, 4, 2), ?
        # src: (N, H_0W_0+...H_3W_3, C), `memory` from encoder, transformed feature map
        # src_spatial_shapes: (#lvl, 2), the feature shape in each level
        # level_start_index: tensor([0, H_0W_0=14028, H_0W_0+H_1W_1=17556, H_0W_0+H_1W_1+H_2W_2=18438])
        # src_padding_mask: 

        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)  # (N, #query, d_model)
        tgt = tgt + self.dropout2(tgt2)  # (N, #query, d_model)
        tgt = self.norm2(tgt)  # (N, #query, d_model)
        
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.training and self.instance_align:
            instance_query = self.instance_attn(instance_query, tgt, query_pos)  # (N, 1, d_model)

        # ffn
        tgt2 = self.ffn_d1(tgt)
        tgt2 = self.dropout3(tgt2)
        tgt2 = self.ffn_d2(tgt2)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt, instance_query


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate  # always True, see L599
        self.bbox_embed = None  # will be set as `ModuleList[Linear] * 6` in `DeformableDETR`
        self.class_embed = None  # `None` if 1-stage, set as `ModuleList`  if 2-stage

    def forward(self, tgt, instance_query, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        # tgt: (N, #query, d_model), object queries
        # instance_query: (N, 1, d_model)
        # reference_points: (N, #query, 2), initially predicted reference point coordinates by the small network `DeformableTransformer.reference_points`
        # src: (N, H_0W_0+...H_3W_3, C), `memory` from encoder, transformed feature map
        # src_spatial_shapes: (#lvl, 2), the feature shape in each level
        # src_level_start_index: tensor([0, H_0W_0=14028, H_0W_0+H_1W_1=17556, H_0W_0+H_1W_1+H_2W_2=18438])
        # src_valid_ratios: (2, 4, 2), ?
        # query_pos: (N, #query, d_model)
        # src_padding_mask: (N, H_0W_0+...H_3W_3), `mask_flatten`

        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                # 2-stage
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                # 1-stage
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]  # (2, #query, 4, 2)
                # reference_points[:, :, None]: (N, #query, 1, 2)
                # src_valid_ratios[:, None]: (2, 1, 4, 2)
            output, instance_query = layer(
                output, instance_query, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask
            )
            # output: (N, #query, d_model)
            # instance_query: (N, 1, d_model)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output) # (N, #query, d_model) → (N, #query, 4)

                if reference_points.shape[-1] == 4:
                    # (N, 300, 4)
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()   
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()  # updated `reference_points`

            if self.return_intermediate:  # always True, see L599
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:  # always True, see L599
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), instance_query
        
        # never called
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

#important
def build_deforamble_transformer(cfg):
    return DeformableTransformer(
        d_model=cfg.MODEL.HIDDEN_DIM, #256
        nhead=cfg.MODEL.NHEADS, #8
        num_encoder_layers=cfg.MODEL.ENC_LAYERS, #6
        num_decoder_layers=cfg.MODEL.DEC_LAYERS, #6
        dim_feedforward=cfg.MODEL.DIM_FEEDFORWARD, #1024
        dropout=cfg.MODEL.DROPOUT, 
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=cfg.MODEL.NUM_FEATURE_LEVELS, #4
        dec_n_points=cfg.MODEL.DEC_N_POINTS,
        enc_n_points=cfg.MODEL.ENC_N_POINTS,
        two_stage=cfg.MODEL.TWO_STAGE,
        two_stage_num_proposals=cfg.MODEL.NUM_QUERIES,
        space_align=cfg.MODEL.SPACE_ALIGN,
        channel_align=cfg.MODEL.CHANNEL_ALIGN,
        instance_align=cfg.MODEL.INSTANCE_ALIGN)


