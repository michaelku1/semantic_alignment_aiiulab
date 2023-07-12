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

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
from models.utils import (
    DomainAttention,
    GradientReversal,
    remove_mask_and_warp,
    get_valid_ratio,
    get_reference_points,
    add_prompt_embed_to_src,
    prepend_prompt_to_tgt
)

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 space_align=False, channel_align=False, instance_align=False,
                 deep_prompt=True, deep_shared=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        self.space_align = space_align
        self.channel_align = channel_align
        self.instance_align = instance_align

        self.deep_prompt = deep_prompt
        self.deep_shared = deep_shared

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points, space_align, channel_align,
                                                          deep_prompt, deep_shared)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, deep_prompt, deep_shared)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, instance_align)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec, deep_prompt, deep_shared)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

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
        # mask: (N, h, w)
        #   True for ignoring the corresponding keys in attention
        #   False for paying attention to the corresponding keys

        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)  # (bz,), real heights of images
        valid_W = torch.sum(~mask[:, 0, :], 1)  # (bz,), ral widths of images
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)  # (bz, 2)
        return valid_ratio

    def forward(
        self, srcs, masks, pos_embeds, query_embed=None,
        src_encoder_prompt_embeds=None, tgt_encoder_prompt_embeds=None,
        src_decoder_prompt_embeds=None, tgt_decoder_prompt_embeds=None,
        data_domain_type='src+tgt', prompt_domain_type='same'
    ):
        assert self.two_stage or query_embed is not None

        # xxx_prompt_embed:
        #   (num_prompt_tokens, hidden_dim * 2)
        #   (num_layer, num_prompt_tokens, hidden_dim * 2)
        #   None

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
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # (bz, #lvl, 2=(w, h))

        space_query, channel_query, instance_query = None, None, None
        # if self.training:  # original implementation
        if self.training or not self.training:  # compute uda loss even when model is in evaluation
            if self.space_align:
                space_query = self.space_query.expand(src_flatten.shape[0], -1, -1)
            if self.channel_align:
                src_warped, pos_warped = remove_mask_and_warp(
                    src_flatten, lvl_pos_embed_flatten, mask_flatten, level_start_index, spatial_shapes
                )
                channel_query = self.channel_query(self.grl(src_warped+pos_warped)).flatten(0, 1).transpose(1, 2)

        # encoder
        memory, space_query, channel_query = self.encoder(
            src_flatten, space_query, channel_query, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten,
            src_encoder_prompt_embeds, tgt_encoder_prompt_embeds
        )

        da_output = {}
        # if self.training:  # original implementation
        if self.training or not self.training:  # compute uda loss even when model is in evaluation
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
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # if self.training and self.instance_align:  # original implementation
        if self.instance_align:  # compute uda loss even when model is in evaluation
            instance_query = self.instance_query.expand(tgt.shape[0], -1, -1)

        # decoder
        hs, inter_references, instance_query = self.decoder(
            tgt, instance_query, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten,
            src_decoder_prompt_embeds, tgt_decoder_prompt_embeds,
            data_domain_type, prompt_domain_type
        )

        # if self.training and self.instance_align:  # original implementation
        if self.instance_align:  # compute uda loss even when model is in evaluation
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
                 space_align=False, channel_align=False,
                deep_prompt=True, deep_shared=False):
        super().__init__()

        self.space_align = space_align
        self.channel_align = channel_align
        self.deep_prompt = deep_prompt
        self.deep_shared = deep_shared

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

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self, src, space_query, channel_query, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None,
        src_prompt_embed=None, tgt_prompt_embed=None
    ):
        # src: (N, H_0W_0+...H_3W_3, C)
        # space_query: (N, 1, d_model)
        # channel_query: ?
        # pos: (N, H_0W_0+...H_3W_3, C)
        # reference_points: ?
        # spatial_shapes: (#lvl, 2), the feature shape in each level
        # level_start_index: tensor([0, H_0W_0=14028, H_0W_0+H_1W_1=17556, H_0W_0+H_1W_1+H_2W_2=18438])
        # xxx_prompt_embed:
        #   [num_feature_levels](h, w, d_model)
        #   None

        if src_prompt_embed is not None or tgt_prompt_embed is not None:
            src = add_prompt_embed_to_src(src, spatial_shapes, src_prompt_embed, tgt_prompt_embed)

        # import pdb; pdb.set_trace()

        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if self.training:
            if self.space_align:
                space_query = self.space_attn(space_query, src, pos, padding_mask)
            if self.channel_align:
                src_warped, pos_warped = remove_mask_and_warp(src, pos, padding_mask, level_start_index, spatial_shapes)
                channel_query = self.channel_attn(
                    channel_query, # bsz * num_feature_levels, 1, H*W
                    src_warped.flatten(0, 1).transpose(1, 2), # bsz * num_feature_levels, C, H*W
                    pos_warped.flatten(0, 1).transpose(1, 2)
                )

        # ffn
        src = self.forward_ffn(src)

        return src, space_query, channel_query


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, deep_prompt=True, deep_shared=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.deep_prompt = deep_prompt
        self.deep_shared = deep_shared

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

    def forward(
        self, src, space_query, channel_query, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None,
        src_prompt_embeddings=None, tgt_prompt_embeddings=None
    ):
        # src: (N, H_0W_0+...H_3W_3, C)
        # space_query: (N, 1, d_model)
        # channel_query: ?
        # spatial_shapes: (#lvl, 2), the feature shape in each level
        # level_start_index: tensor([0, H_0W_0=14028, H_0W_0+H_1W_1=17556, H_0W_0+H_1W_1+H_2W_2=18438])
        # valid_ratios: ?
        # pos: (N, H_0W_0+...H_3W_3, C)
        # padding_mask: (N, H_0W_0+...H_3W_3)
        # xxx_prompt_embeddings:
        #   shallow: [num_feature_levels](h, w, hidden_dim)
        #   deep shared: [num_feature_levels](h, w, hidden_dim)
        #   deep: [num_layers][num_feature_levels](h, w, hidden_dim)

        prompt_embeddings_dict = {'src': src_prompt_embeddings, 'tgt': tgt_prompt_embeddings}
        for domain, prompt_embeddings in prompt_embeddings_dict.items():
            if prompt_embeddings is not None:
                assert isinstance(prompt_embeddings, (list, nn.ParameterList))

                if isinstance(prompt_embeddings[0], (list, nn.ParameterList)):
                    # deep
                    assert self.deep_prompt and not self.deep_shared
                    assert len(prompt_embeddings) == self.num_layers

                elif isinstance(prompt_embeddings[0], nn.Parameter):
                    # shallow or deep_shared
                    if not self.deep_shared:
                        # shallow
                        prompt_embeddings_dict[domain] = [prompt_embeddings]  # [1][num_feature_levels](h, w, hidden_dim)
                    else:
                        # deep shared
                        prompt_embeddings_dict[domain] = [prompt_embeddings for _ in range(self.num_layers)]  # [num_layers][num_feature_levels](h, w, hidden_dim)
                else:
                    raise ValueError('Known shape of prompt embedding:', prompt_embeddings.shape)
        src_prompt_embeddings = prompt_embeddings_dict['src']
        tgt_prompt_embeddings = prompt_embeddings_dict['tgt']
        # xxx_prompt_embeddings:
        #   [num_layers][num_feature_levels](h, w, d_model)
        #   None

        # import pdb; pdb.set_trace()

        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        space_querys = []
        channel_querys = []
        for layer_idx, layer in enumerate(self.layers):
            src_prompt_embed = None
            if src_prompt_embeddings is not None:
                if layer_idx < len(src_prompt_embeddings):
                    src_prompt_embed = src_prompt_embeddings[layer_idx]
            tgt_prompt_embed = None
            if tgt_prompt_embeddings is not None:
                if layer_idx < len(tgt_prompt_embeddings):
                    tgt_prompt_embed = tgt_prompt_embeddings[layer_idx]
            # xxx_prompt_embed:
            #   [num_feature_levels](h, w, d_model)
            #   None

            # import pdb; pdb.set_trace()

            output, space_query, channel_query = layer(
                output, space_query, channel_query, pos, reference_points, spatial_shapes, level_start_index, padding_mask,
                src_prompt_embed, tgt_prompt_embed
            )
            space_querys.append(space_query)
            channel_querys.append(channel_query)

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

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self, tgt, instance_query, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None,
        src_prompt_embed=None, tgt_prompt_embed=None, data_domain_type='src+tgt', prompt_domain_type='same'
    ):
        # tgt: (N, #query, d_model), object queries
        # instance_query: (N, 1, d_model)
        # query_pos: (N, #query, d_model)
        # reference_points: (2, #query, 4, 2), ?
        # src: (N, H_0W_0+...H_3W_3, C), `memory` from encoder, transformed feature map
        # src_spatial_shapes: (#lvl, 2), the feature shape in each level
        # level_start_index: tensor([0, H_0W_0=14028, H_0W_0+H_1W_1=17556, H_0W_0+H_1W_1+H_2W_2=18438])
        # src_padding_mask:
        #
        # xxx_prompt_embed:
        #   (num_prompt_tokens, hidden_dim * 2)
        #   None

        # TODO:

        N, num_query, C = tgt.shape
        src_prompt_pos = tgt_prompt_pos = None
        if src_prompt_embed is not None:
            src_prompt_embed, src_prompt_pos = torch.split(src_prompt_embed, C, dim=-1)
        if tgt_prompt_embed is not None:
            tgt_prompt_embed, tgt_prompt_pos = torch.split(tgt_prompt_embed, C, dim=-1)
        # xxx_prompt_embed:
        #   (num_prompt_tokens, hidden_dim)
        #   None
        # xxx_prompt_pos:
        #   (num_prompt_tokens, hidden_dim)
        #   None

        # self attention
        # ↓↓↓ the original implementation ↓↓↓
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # import pdb; pdb.set_trace()
        
        # ↓↓↓ prepend prompts ↓↓↓
        if prompt_domain_type == 'tgt_only':
            assert src_prompt_embed is None

            if data_domain_type == 'src+tgt':
                # only apply prompt to the images from the target domain
                assert N % 2 == 0
                src_tgt, tgt_tgt = torch.split(tgt, N // 2, dim=0)  # (N//2, #query, d_model)
                src_pos, tgt_pos = torch.split(query_pos, N // 2, dim=0)  # (N//2, #query, d_model)

                src_q = src_k = self.with_pos_embed(src_tgt, src_pos)  # (N//2, #query, d_model)
                src_tgt2 = self.self_attn(src_q.transpose(0, 1), src_k.transpose(0, 1), src_tgt.transpose(0, 1))[0].transpose(0, 1)  # (N//2, #query, d_model)

                tgt_q = tgt_k = self.with_pos_embed(tgt_tgt, tgt_pos)  # (N//2, #query, d_model)
                tgt_k = prepend_prompt_to_tgt(tgt_k, None, self.with_pos_embed(tgt_prompt_embed, tgt_prompt_pos))  # (N//2, #query + num_prompt_tokens, d_model)
                tgt_v = prepend_prompt_to_tgt(tgt_tgt, None, self.with_pos_embed(tgt_prompt_embed, None))  # (N//2, #query + num_prompt_tokens, d_model)
                tgt_tgt2 = self.self_attn(tgt_q.transpose(0, 1), tgt_k.transpose(0, 1), tgt_v.transpose(0, 1))[0].transpose(0, 1)
                # tgt_q: (#query, N//2, d_model)
                # tgt_k: (#query + num_prompt_tokens, N//2, d_model)
                # tgt_v: (#query + num_prompt_tokens, N//2, d_model)

                tgt2 = torch.cat([src_tgt2, tgt_tgt2], dim=0)
                # import pdb; pdb.set_trace()  # checked!

            elif data_domain_type == 'src_only':
                q = k = self.with_pos_embed(tgt, query_pos)
                tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
                # import pdb; pdb.set_trace()  # checked!

            elif data_domain_type == 'tgt_only':
                q = k = self.with_pos_embed(tgt, query_pos)  # (N, #query, d_model)
                k = prepend_prompt_to_tgt(k, None, self.with_pos_embed(tgt_prompt_embed, tgt_prompt_pos))  # (N, #query + num_prompt_tokens, d_model)
                v = prepend_prompt_to_tgt(tgt, None, self.with_pos_embed(tgt_prompt_embed, None))  # (N, #query + num_prompt_tokens, d_model)
                tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[0].transpose(0, 1)  # (N, #query, d_model)
                # import pdb; pdb.set_trace()  # checked!
            
        else:
            # `prompt_domain_type` is one of 'same', 'separate', or 'inverse'
            assert src_prompt_embed is not None

            if data_domain_type == 'src+tgt':
                assert N % 2 == 0
                src_tgt, tgt_tgt = torch.split(tgt, N // 2, dim=0)  # (N//2, #query, d_model)
                src_pos, tgt_pos = torch.split(query_pos, N // 2, dim=0)  # (N//2, #query, d_model)

                src_q = src_k = self.with_pos_embed(src_tgt, src_pos)  # (N//2, #query, d_model)
                src_k = prepend_prompt_to_tgt(src_k, self.with_pos_embed(src_prompt_embed, src_prompt_pos), None)  # (N//2, #query + num_prompt_tokens, d_model)
                src_v = prepend_prompt_to_tgt(src_tgt, src_prompt_embed, None)  # (N//2, #query + num_prompt_tokens, d_model)
                src_tgt2 = self.self_attn(src_q.transpose(0, 1), src_k.transpose(0, 1), src_v.transpose(0, 1))[0].transpose(0, 1)  # (N//2, #query, d_model)

                tgt_q = tgt_k = self.with_pos_embed(tgt_tgt, tgt_pos)  # (N//2, #query, d_model)
                tgt_k = prepend_prompt_to_tgt(tgt_k, None, self.with_pos_embed(tgt_prompt_embed, tgt_prompt_pos))  # (N//2, #query + num_prompt_tokens, d_model)
                tgt_v = prepend_prompt_to_tgt(tgt_tgt, None, self.with_pos_embed(tgt_prompt_embed, None))  # (N//2, #query + num_prompt_tokens, d_model)
                tgt_tgt2 = self.self_attn(tgt_q.transpose(0, 1), tgt_k.transpose(0, 1), tgt_v.transpose(0, 1))[0].transpose(0, 1)  # (N//2, #query, d_model)

                tgt2 = torch.cat([src_tgt2, tgt_tgt2], dim=0)
                # import pdb; pdb.set_trace()  # checked!

            elif data_domain_type == 'src_only':
                q = k = self.with_pos_embed(tgt, query_pos)  # (N, #query, d_model)
                k = prepend_prompt_to_tgt(k, self.with_pos_embed(src_prompt_embed, src_prompt_pos), None)  # (N, #query + num_prompt_tokens, d_model)
                v = prepend_prompt_to_tgt(tgt, self.with_pos_embed(src_prompt_embed, None), None)  # (N, #query + num_prompt_tokens, d_model)
                tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[0].transpose(0, 1)  # (N, #query, d_model)
                # import pdb; pdb.set_trace()  # checked!

            elif data_domain_type == 'tgt_only':
                q = k = self.with_pos_embed(tgt, query_pos)  # (N, #query, d_model)
                k = prepend_prompt_to_tgt(k, None, self.with_pos_embed(tgt_prompt_embed, tgt_prompt_pos))  # (N, #query + num_prompt_tokens, d_model)
                v = prepend_prompt_to_tgt(tgt, None, self.with_pos_embed(tgt_prompt_embed, None))  # (N, #query + num_prompt_tokens, d_model)
                tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[0].transpose(0, 1)  # (N, #query, d_model)
                # import pdb; pdb.set_trace()  # checked!

        assert tgt2.shape[0] == N
        assert tgt2.shape[1] == num_query
        assert tgt2.shape[2] == C
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.training and self.instance_align:
            instance_query = self.instance_attn(instance_query, tgt, query_pos)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, instance_query


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, deep_prompt=True, deep_shared=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.deep_prompt = deep_prompt
        self.deep_shared = deep_shared
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self, tgt, instance_query, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
        query_pos=None, src_padding_mask=None,
        src_decoder_prompt_embeds=None, tgt_decoder_prompt_embeds=None,
        data_domain_type='src+tgt', prompt_domain_type='same'
    ):
        # tgt: (N, #query, d_model), object queries
        # instance_query: (N, 1, d_model)
        # reference_points: (N, #query, 2), initially predicted reference point coordinates by the small network `DeformableTransformer.reference_points`
        # src: (N, H_0W_0+...H_3W_3, C), `memory` from encoder, transformed feature map
        # src_spatial_shapes: (#lvl, 2), the feature shape in each level
        # src_level_start_index: tensor([0, H_0W_0=14028, H_0W_0+H_1W_1=17556, H_0W_0+H_1W_1+H_2W_2=18438])
        # src_valid_ratios: (2, 4, 2), ?
        # query_pos: (N, #query, d_model)
        # src_padding_mask: (N, H_0W_0+...H_3W_3), `mask_flatten`
        # xxx_prompt_embeds:

        # xxx_prompt_embeds:
        #   shallow: (num_prompt_tokens, d_model * 2)
        #   deep shared: (num_prompt_tokens, d_model * 2)
        #   deep:
        #       (num_layers, num_prompt_tokens, d_model * 2)
        
        prompt_embeds_dict = {'src': src_decoder_prompt_embeds, 'tgt': tgt_decoder_prompt_embeds}
        for domain, prompt_embeds in prompt_embeds_dict.items():
            if prompt_embeds is not None:
                if prompt_embeds.ndim == 2:
                    if not self.deep_shared:
                        # shallow
                        # prompt_embeds: (num_queries, d_model)
                        prompt_embeds_dict[domain] = [prompt_embeds]  # for the 1st layer only
                    else:
                        # deep shared
                        # prompt_embeds: (num_queries, d_model)
                        assert self.deep_prompt and self.deep_shared
                        prompt_embeds_dict[domain] = [prompt_embeds for _ in range(self.num_layers)]
                elif prompt_embeds.ndim == 3:
                    # deep
                    # prompt_embeds: (num_layers, num_queries, d_model)
                    assert self.deep_prompt and not self.deep_shared
                    assert len(prompt_embeds) == self.num_layers
                else:
                    raise ValueError('Known shape of prompt embedding:', prompt_embeds.shape)
        src_decoder_prompt_embeds = prompt_embeds_dict['src']
        tgt_decoder_prompt_embeds = prompt_embeds_dict['tgt']
        # xxx_prompt_embeds:
        #   [num_layers](num_queries, d_model * 2)
        #   (num_layers, num_queries, d_model * 2)
        #   None

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
            
            src_prompt_embed = None
            if src_decoder_prompt_embeds is not None:
                if lid < len(src_decoder_prompt_embeds):
                    src_prompt_embed = src_decoder_prompt_embeds[lid]   
            tgt_prompt_embed = None
            if tgt_decoder_prompt_embeds is not None:
                if lid < len(tgt_decoder_prompt_embeds):
                    tgt_prompt_embed = tgt_decoder_prompt_embeds[lid]     
            # xxx_prompt_embed:
            #   (num_queries, hidden_dim * 2)
            #   None

            # import pdb; pdb.set_trace()

            output, instance_query = layer(
                output, instance_query, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask,
                src_prompt_embed, tgt_prompt_embed,
                data_domain_type, prompt_domain_type
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
        instance_align=cfg.MODEL.INSTANCE_ALIGN)


