# ----------------------------------------------
# Created by Wei-Jie Huang
# ----------------------------------------------


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import gc

def weighted_aggregate(batch_d, list_of_labels, list_of_rois, list_of_scores, num_classes, hidden_dim):
    ### weighted aggregate
    # labels existing (sorted)
    B = batch_d
    source_labels = []
    target_labels = []

    weighted_rois_source = []
    weighted_rois_target = []

    # import pdb; pdb.set_trace()

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
    src_prototypes = torch.zeros((num_classes, hidden_dim)).cuda()
    tgt_prototypes = torch.zeros((num_classes, hidden_dim)).cuda()

    epsilon = 1e-6
    # this is to also index labels (one-to-one)
    # batch length
    for i in range(len(source_rois)):
        roi_sample_tmp = source_rois[i] # some rois for a single class
        # some rois
        for j in range(len(roi_sample_tmp)):
            # src_prototypes.append(torch.sum(source_roi, dim=0)/(torch.sum(list_of_scores[0]) + epsilon))
            aggregate = torch.sum(roi_sample_tmp[j], dim=0)/(torch.sum(list_of_scores[0]) + epsilon)
            # store prototype at the corresponding cls index position
            src_prototypes[source_labels[i][j]] = aggregate
            
    for i in range(len(target_rois)):
        roi_sample_tmp = target_rois[i]
        for j in range(len(roi_sample_tmp)):
            # tgt_prototypes.append(torch.sum(target_rois[i], dim=0)/(torch.sum(list_of_scores[1]) + epsilon))
            aggregate = torch.sum(roi_sample_tmp[j], dim=0)/(torch.sum(list_of_scores[1]) + epsilon)
            tgt_prototypes[target_labels[i][j]] = aggregate

    return src_prototypes, tgt_prototypes

def compute_CV(features, labels, ave_CxA, class_num):
    """
    features: target features
    labels: stored target labels
    ave_CxA: computed inter domain mean
    """
    
    features =torch.cat(features, 0) # (num_of_rois, feat_dim)
    labels = torch.as_tensor(labels).cuda() # (b, num_of_labels)
    # import pdb; pdb.set_trace()
    N = features.size(0) # number of features
    C = class_num
    A = features.size(1) # feat dim

    var_temp = torch.zeros(C, A, A).cuda() # store target intra class variance
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).cuda() # (N, C)
    onehot.scatter_(1, labels.view(-1, 1), 1) # labels --> (b*num_of_labels, 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1
    Amount_CxAxA = Amount_CxA.view(C, A, 1).expand(C, A, A)
    del Amount_CxA, onehot
    gc.collect()

    avg_NxCxA = ave_CxA.expand(N, C, A)
    for c in range(C):
        features_by_sort_c = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        avg_by_sort_c = avg_NxCxA[:, c, :].mul(NxCxA_onehot[:, c, :])
        var_temp_c = features_by_sort_c - avg_by_sort_c
        var_temp[c] = torch.mm(var_temp_c.permute(1,0), var_temp_c).div(Amount_CxAxA[c])
    
    return var_temp.detach() # (C, A, A)


def remove_mask_and_warp(src, pos, padding_mask, level_start_index, spatial_shapes):
    """ Removes padding mask in sequence and warps each level of tokens into fixed-sized sequences.

    Args:
        src, pos (batch_size, sequence_length, d_model): patch tokens and position encodings
        padding_mask (batch_size, sequence_length): key padding mask
        level_start_index (num_feature_levels): start index of each feature level
        spatial_shapes (num_feature_levels, 2): spatial shape (H, W) of each feature level

    Returns:
        src_warped, pos_warped (batch_size, num_feature_levels, C, C): warped patch tokens and
        position encodings. The last two dimensions indicate sequence length (i.e., H*W) and model
        dimension, respectively.
    """

    B, _, C = src.shape
    sqrt_C = int(C ** 0.5)
    src_warped = []
    pos_warped = []
    for start, shape in zip(level_start_index, spatial_shapes):
        H, W = shape
        s = src[:, start:start+H*W].view(B, H, W, C).permute(0, 3, 1, 2)
        p = pos[:, start:start+H*W].view(B, H, W, C).permute(0, 3, 1, 2)
        m = padding_mask[:, start:start+H*W].view(B, H, W)

        not_m = ~m
        real_H = not_m.sum(1).max(1).values
        real_W = not_m.sum(2).max(1).values

        src_warped.append(torch.stack([F.adaptive_avg_pool2d(s_i[:, :real_H[i], :real_W[i]], sqrt_C) for i, s_i in enumerate(s)]))
        pos_warped.append(torch.stack([F.adaptive_avg_pool2d(p_i[:, :real_H[i], :real_W[i]], sqrt_C) for i, p_i in enumerate(p)]))

    # import pdb; pdb.set_trace()
    src_warped = torch.stack(src_warped, dim=1).flatten(-2).transpose(-2, -1)
    pos_warped = torch.stack(pos_warped, dim=1).flatten(-2).transpose(-2, -1)
    return src_warped, pos_warped


class DomainAttention(nn.Module):
    """ Wraps domain-adapting cross attention and MLP into a module.
        The operations are similar to those in Transformer, including normalization
        layers and dropout layers, while MLP is simplified as a linear layer.

    Args:
        d_model: total dimension of the model.
        n_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights.
    """

    def __init__(self, d_model, n_heads, dropout):
        super(DomainAttention, self).__init__()
        self.grl = GradientReversal()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, src, pos=None, padding_mask=None):
        """ Args:
            query (batch_size, num_queries, d_model): discriminator query
            src, pos (batch_size, sequence_length, d_model): patch tokens and position encodings
            padding_mask (batch_size, sequence_length): key padding mask
        """

        # keys are srcs + positional embeddings
        # query input is the space/channel query
        # layers before domain attention will be trained adversarially
        # query --> target sequence, key --> source sequence
        # adversarial training the self attention layer
        
        r_query, _ = self.cross_attn(
            query=query.transpose(0, 1),
            key=self.grl(self.with_pos_embed(src, pos)).transpose(0, 1),
            value=self.grl(src).transpose(0, 1),
            key_padding_mask=padding_mask,
        )
        # residual
        query = query + self.dropout1(r_query.transpose(0, 1))
        query = self.norm1(query)
        # linear layer + dropout layer + layer norm
        query = query + self.dropout2(self.linear(query))
        query = self.norm2(query)
        return query


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(CrossAttention, self).__init__()
        self.grl = GradientReversal()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, src, pos=None, padding_mask=None):
        """ Args:
            query (batch_size, num_queries, d_model): discriminator query
            src, pos (batch_size, sequence_length, d_model): patch tokens and position encodings
            padding_mask (batch_size, sequence_length): key padding mask
        """

        # keys are srcs + positional embeddings
        # query input is the space/channel query
        # layers before domain attention will be trained adversarially
        # query --> target sequence, key --> source sequence
        # adversarial training the self attention layer
        # import pdb; pdb.set_trace()
        # TODO: set grl before src to align encoder/encoder features
        r_query, attn_weights = self.cross_attn(
            query=query.transpose(0, 1),
            key=self.grl(self.with_pos_embed(src, pos)).transpose(0, 1),
            value=self.grl(src.transpose(0, 1)), key_padding_mask=padding_mask,
        )
        # residual
        query = query + self.dropout1(r_query.transpose(0, 1))
        query = self.norm1(query)
        # linear layer + dropout layer + layer norm
        query = query + self.dropout2(self.linear(query))
        query = self.norm2(query)
        return query, attn_weights

class CrossAttentionMemory(nn.Module):

    def __init__(self, d_model, n_heads, dropout):
        super(CrossAttentionMemory, self).__init__()
        self.grl = GradientReversal()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, src, pos=None, padding_mask=None):
        """ Args:
            query (batch_size, num_queries, d_model): discriminator query
            src, pos (batch_size, sequence_length, d_model): patch tokens and position encodings
            padding_mask (batch_size, sequence_length): key padding mask
        """

        # keys are srcs + positional embeddings
        # query input is the space/channel query
        # layers before domain attention will be trained adversarially
        # query --> target sequence, key --> source sequence
        # adversarial training the self attention layer
        # import pdb; pdb.set_trace()
        # TODO: set grl before src to align encoder/encoder features
        # import pdb; pdb.set_trace()
        r_query, attn_weights = self.cross_attn(
            query=self.with_pos_embed(query, pos),
            key=src,
            value=src, key_padding_mask=padding_mask,
        )
        # residual
        query = query + self.dropout1(r_query)
        query = self.norm1(query)
        # linear layer + dropout layer + layer norm
        query = query + self.dropout2(self.linear(query))
        query = self.norm2(query)
        return query, attn_weights

#TODO: cross attention between category codes and encoder features (after self attention layer)
class category_feature_attention(nn.Module):
    """ Wraps domain-adapting cross attention and MLP into a module.
        The operations are similar to those in Transformer, including normalization
        layers and dropout layers, while MLP is simplified as a linear layer.

    Args:
        d_model: total dimension of the model.
        n_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights.
    """

    def __init__(self, d_model, n_heads, dropout):
        super(category_feature_attention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    # query is the target sequence, src is the source sequence
    def forward(self, query, src, pos=None, padding_mask=None):
        """ Args:
            query (batch_size, num_queries, d_model): discriminator query
            src, pos (batch_size, sequence_length, d_model): patch tokens and position encodings
            padding_mask (batch_size, sequence_length): key padding mask
        """

        # keys are srcs + positional embeddings
        # query input is the query token
        # position embedding is needed
        ### as long as src features are flattened (tokenized), position embedding is needed
        r_query, attention_weights = self.cross_attn(
            query=query.transpose(0, 1),
            key=self.with_pos_embed(src, pos).transpose(0, 1),
            value=src.transpose(0, 1),
            key_padding_mask=padding_mask,
        )
        # residual
        query = query + self.dropout1(r_query.transpose(0, 1))
        query = self.norm1(query)
        # linear layer + dropout layer + layer norm
        query = query + self.dropout2(self.linear(query))
        query = self.norm2(query)

        
        return query, attention_weights


# ------------------------------------------------------------------------------------------------------------------------------
# Copy-paste from https://github.com/jvanvugt/pytorch-domain-adaptation/blob/35ac3a5a04b5e1cf5b2145b6c442c2d678362eef/utils.py
# ------------------------------------------------------------------------------------------------------------------------------


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    # identity function
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
