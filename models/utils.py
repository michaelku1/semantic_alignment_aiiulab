# ----------------------------------------------
# Created by Wei-Jie Huang
# ----------------------------------------------


import torch
from torch import nn
import torch.nn.functional as F


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

    # src: (N, H_0W_0+...H_3W_3, C)
    # pos: (N, H_0W_0+...H_3W_3, C)
    # level_start_index = tensor([0, H_0W_0=14028, H_0W_0+H_1W_1=17556, H_0W_0+H_1W_1+H_2W_2=18438])
    # spatial_shapes: (#lvl, 2)

    B, _, C = src.shape
    sqrt_C = int(C ** 0.5)
    src_warped = []
    pos_warped = []
    for start, shape in zip(level_start_index, spatial_shapes):
        H, W = shape
        s = src[:, start:start+H*W].view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        p = pos[:, start:start+H*W].view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        m = padding_mask[:, start:start+H*W].view(B, H, W)  # (B, H, W)

        not_m = ~m｀
        real_H = not_m.sum(1).max(1).values
        real_W = not_m.sum(2).max(1).values

        src_warped.append(torch.stack([F.adaptive_avg_pool2d(s_i[:, :real_H[i], :real_W[i]], sqrt_C) for i, s_i in enumerate(s)]))
        pos_warped.append(torch.stack([F.adaptive_avg_pool2d(p_i[:, :real_H[i], :real_W[i]], sqrt_C) for i, p_i in enumerate(p)]))

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
        # query: (N, 1, d_model)
        # src: (N, H_0W_0+...H_3W_3, C)
        # pos: (N, H_0W_0+...H_3W_3, C)
        # padding_mask: (N, H_0W_0+...H_3W_3)

        r_query, _ = self.cross_attn(
            query=query.transpose(0, 1),  # (1, N, d_model)
            key=self.grl(self.with_pos_embed(src, pos)).transpose(0, 1),  # (H_0W_0+...H_3W_3, N, C)
            value=self.grl(src).transpose(0, 1),  # (H_0W_0+...H_3W_3, N, C)
            key_padding_mask=padding_mask,  # do not pay attention on the keys that the corresponding mask is `True`
        )
        query = query + self.dropout1(r_query.transpose(0, 1))
        query = self.norm1(query)
        query = query + self.dropout2(self.linear(query))
        query = self.norm2(query)
        return query  # (N, 1, d_model)


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
