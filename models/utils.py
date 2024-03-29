# ----------------------------------------------
# Created by Wei-Jie Huang
# ----------------------------------------------


import torch
from torch import nn
# print(nn.__file__)
# quit()

import torch.nn.functional as F
import numpy as np
import gc
import math

def cal_position_embedding(self, rois1, rois2):
    # [num_rois, num_nongt_rois, 4]
    position_matrix = self.extract_position_matrix(rois1, rois2)
    # [num_rois, num_nongt_rois, 64]
    position_embedding = self.extract_position_embedding(position_matrix, feat_dim=256)
    # [64, num_rois, num_nongt_rois]
    position_embedding = position_embedding.permute(2, 0, 1)
    # [1, 64, num_rois, num_nongt_rois]
    position_embedding = position_embedding.unsqueeze(0)

    return position_embedding


def extract_position_embedding(self,position_mat, feat_dim, wave_length=1000.0):
    device = position_mat.device
    # position_mat, [num_rois, num_nongt_rois, 4]
    feat_range = torch.arange(0, feat_dim / 8, device=device)

    dim_mat = torch.full((len(feat_range),), wave_length, device=device).pow(8.0 / feat_dim * feat_range)
    dim_mat = dim_mat.view(1, 1, 1, -1).expand(*position_mat.shape, -1)

    position_mat = position_mat.unsqueeze(3).expand(-1, -1, -1, dim_mat.shape[3])
    position_mat = position_mat * 100.0

    div_mat = position_mat / dim_mat
    sin_mat, cos_mat = div_mat.sin(), div_mat.cos()

    # [num_rois, num_nongt_rois, 4, feat_dim / 4]
    embedding = torch.cat([sin_mat, cos_mat], dim=3)
    # [num_rois, num_nongt_rois, feat_dim]
    embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], embedding.shape[2] * embedding.shape[3])

    return embedding

@staticmethod
def extract_position_matrix(bbox, ref_bbox):
    # (x,y,x,y)
    xmin, ymin, xmax, ymax = torch.chunk(ref_bbox, 4, dim=1)
    bbox_width_ref = xmax - xmin + 1
    bbox_height_ref = ymax - ymin + 1
    center_x_ref = 0.5 * (xmin + xmax)
    center_y_ref = 0.5 * (ymin + ymax)

    xmin, ymin, xmax, ymax = torch.chunk(bbox, 4, dim=1)
    bbox_width = xmax - xmin + 1
    bbox_height = ymax - ymin + 1
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)

    delta_x = center_x - center_x_ref.transpose(0, 1)
    delta_x = delta_x / bbox_width
    delta_x = (delta_x.abs() + 1e-3).log()

    delta_y = center_y - center_y_ref.transpose(0, 1)
    delta_y = delta_y / bbox_height
    delta_y = (delta_y.abs() + 1e-3).log()

    delta_width = bbox_width / bbox_width_ref.transpose(0, 1)
    delta_width = delta_width.log()

    delta_height = bbox_height / bbox_height_ref.transpose(0, 1)
    delta_height = delta_height.log()

    # (x, y w, h)
    position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=2)
    # import pdb
    # pdb.set_trace()
    return position_matrix


# borrowed from dual branch perceiver
def attention_module_multi_head(self, roi_feat, ref_feat, position_embedding,
                                feat_dim=256, dim=(256, 256, 256), group=49,
                                index=0):
    """

    :param roi_feat: [num_rois, feat_dim]
    :param ref_feat: [num_nongt_rois, feat_dim]
    :param position_embedding: [1, emb_dim, num_rois, num_nongt_rois]
    :param feat_dim: should be same as dim[2]
    :param dim: a 3-tuple of (query, key, output)
    :param group:
    :return:
    """

    # normalise q, k, v dimensions by flattened proposal size (token length) 
    dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

    # position_embedding, [1, emb_dim, num_rois, num_nongt_rois]
    # -> position_feat_1, [1, group, num_rois, num_nongt_rois]
    position_feat_1 = F.relu(self.Wgs[index](position_embedding)) # learnable position embedding
    # aff_weight, [num_rois, group, num_nongt_rois, 1]
    aff_weight = position_feat_1.permute(2, 1, 3, 0)
    # aff_weight, [num_rois, group, num_nongt_rois]
    aff_weight = aff_weight.squeeze(3)

    # multi head
    assert dim[0] == dim[1]

    q_data = self.Wqs[index](roi_feat) # source
    q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
    # q_data_batch, [group, num_rois, dim_group[0]]
    q_data_batch = q_data_batch.permute(1, 0, 2)

    k_data = self.Wks[index](ref_feat) # target-like
    k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
    # k_data_batch, [group, num_nongt_rois, dim_group[1]]
    k_data_batch = k_data_batch.permute(1, 0, 2)

    # v_data, [num_nongt_rois, feat_dim]
    v_data = ref_feat # target-like

    # aff, [group, num_rois, num_nongt_rois]
    aff = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))
    aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff # attention normalization
    # aff_scale, [num_rois, group, num_nongt_rois]
    aff_scale = aff_scale.permute(1, 0, 2)

    # weighted_aff, [num_rois, group, num_nongt_rois]
    weighted_aff = (aff_weight + 1e-6).log() + aff_scale
    aff_softmax = F.softmax(weighted_aff, dim=2)

    aff_softmax_reshape = aff_softmax.reshape(aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2])

    # output_t, [num_rois * group, feat_dim]
    output_t = torch.matmul(aff_softmax_reshape, v_data)
    # output_t, [num_rois, group * feat_dim, 1, 1]
    output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
    # linear_out, [num_rois, dim[2], 1, 1]
    linear_out = self.Wvs[index](output_t)

    output = linear_out.squeeze(3).squeeze(2)

    return output

def cosinematrix(A):
    prod = torch.mm(A, A.t())#分子
    norm = torch.norm(A,p=2,dim=1).unsqueeze(0)#分母
    cos = prod.div(torch.mm(norm.t(),norm))
    return cos

def cosine_distance(matrix1,matrix2):

    # breakpoint()
    matrix1_matrix2 = torch.mm(matrix1, matrix2.t())
    norm_1 = torch.norm(matrix1,p=2,dim=1).unsqueeze(0)#分母
    norm_2 = torch.norm(matrix2,p=2,dim=1).unsqueeze(0)#分母
    cos = matrix1_matrix2.div(torch.mm(norm_1.t(),norm_2))
    return cos
    
# a temporary recursive solution to avoid empty keep 
def find_thresh(outputs_class_conf, thresh, keep):
    # import pdb; pdb.set_trace()
    if keep[0].numel() == 0:
        thresh = thresh - 0.05 # initial
        new_keep = [torch.nonzero(outputs_class_conf[b]>thresh).unsqueeze(0) for b in range(outputs_class_conf.shape[0])]
        return find_thresh(outputs_class_conf, thresh, new_keep)
    else:
        # import pdb; pdb.set_trace()
        # new_keep = [torch.nonzero(outputs_class_conf[b]>thresh).unsqueeze(0) for b in range(outputs_class_conf.shape[0])]
        return keep, thresh

# NOTE checked
def weighted_aggregate(batch_d, list_of_labels, list_of_rois, list_of_scores,
                            num_classes, src_prototypes, hidden_dim):
    
    """
    Implementation for weighted aggregation of centroids, the idea is similar to
    a clustering algorithm that returns the clustered centroids/prototypes 

    weighted aggregation for rois: weight --> group rois --> aggregate

    prototypes (can be source or target): (num_classes, m_prototypes, feat_dim)
    list_of_rois: rois pooled from boxes
    list_of_scores: predicted confidence scores 

    """

    B = batch_d//2

    source_labels = [] # store label set
    weighted_rois_source = []

    ### weight rois with the prediction scores
    for i in range(B):
        label_set = list(set(list_of_labels[i])) # label set for each sample 
        source_labels.append(label_set)

        weighted_rois = list_of_rois[i].squeeze(0)*list_of_scores[i].unsqueeze(-1) # reweighted rois
        weighted_rois_source.append(weighted_rois)


    assert len(weighted_rois_source) == len(source_labels)
    "label and roi lists are not one-to-one"
    
    # extract src rois
    source_rois = [] # [bs, num_labels] (num_rois, 1, feat_dim)
    source_scores = []
    source_labels_all = list_of_labels

    assert len(source_labels) == len(source_labels_all) == len(weighted_rois_source), "length should be the same per sample"

    ### use labels to group rois for each class per batch
    for i in range(len(source_labels)):
        tmp = []
        tmp_ = []
        # label in label set
        for label in source_labels[i]:
            matched_src_idx = torch.nonzero(torch.as_tensor(source_labels_all[i])==label).squeeze(1) # matched_idx: (11, 1)
            rois_source = weighted_rois_source[i][matched_src_idx].unsqueeze(1) # (num_rois, 1, feat_dim)
            sum_of_scores = sum(list_of_scores[i][matched_src_idx])

            # breakpoint()
            tmp.append(rois_source)
            tmp_.append(sum_of_scores)
        # per batch
        source_rois.append(tmp)
        source_scores.append(tmp_)

    assert source_rois[0].__len__() == source_scores[0].__len__()

    prototypes = torch.zeros((num_classes-1, hidden_dim)).cuda()
    epsilon = 1e-6    
    
    ### aggregate prototypes
    # for each batch
    for i in range(len(source_rois)):
        roi_gorups = source_rois[i] # some rois for a single class
        scores = source_scores[i]
        # for each roi group
        for j in range(len(roi_gorups)):
            # sum over each class / sum over scores for that class
            # breakpoint()
            aggregate = torch.sum(roi_gorups[j], dim=0)/(scores[j] + epsilon)
            # store prototype at the corresponding cls index position
            prototypes[source_labels[i][j]-1] = aggregate
    
    # class reweighting factor
    alpha_values_src = torch.ones((num_classes))

    # import pdb; pdb.set_trace()
    for cls_i in range(1,num_classes,1):
        if cls_i in list_of_labels[0]:
            indices_list = torch.nonzero(torch.as_tensor(list_of_labels[0])==cls_i).tolist()
            flatten = [index[0] for index in indices_list]
            p_max = max(list_of_scores[0][flatten])
            alpha = 1-p_max
            alpha_values_src[cls_i] = alpha
        else:
            continue

    alphas = alpha_values_src

    return prototypes, alphas

# NOTE we might want to try to pass the prototypes initialised from outside directly instead of updating
# prototypes from outside
def weighted_aggregate_multi_modal(batch_d, list_of_labels, list_of_rois, list_of_scores, num_classes,
                                        prototypes, momentum_update):
    
    """
    Implementation for weighted aggregation of centroids, this is different from weighted_aggregate
    as this considers updating memory from similarity scores, which is different from the weighted
    sum using pred/gt scores; therefore, we consider directly updating global prototypes here

    weighted aggregation for rois: weight --> group rois --> aggregate

    prototypes (can be source or target): (num_classes, m_prototypes, feat_dim)
    list_of_rois: rois pooled from boxes
    list_of_scores: predicted confidence scores 

    """

    # B = batch_d//2

    # source_labels = []
    # weighted_rois_source = []

        ### weight rois with the prediction scores

    # for i in range(B):
    #     label_set = list(set(list_of_labels[i])) # label set for each sample 
    #     source_labels.append(label_set)

    #     # confidence guided merging
    #     weighted_rois = list_of_rois[i].squeeze(0)*list_of_scores[i].unsqueeze(-1) # reweighted rois
    #     weighted_rois_source.append(weighted_rois)

    # assert len(weighted_rois_source) == len(source_labels)
    # "label and roi lists are not one-to-one"
    
    # source_rois = [] # [bs, num_labels] (num_rois, 1, feat_dim)
    # source_scores = []
    # source_labels_all = list_of_labels

    # check length
    # assert len(source_labels) == len(source_labels_all) == len(weighted_rois_source), "length should be the same per sample"

    ### group rois with respect to cls labels
    # for i in range(len(source_labels)):
    #     tmp = []
    #     tmp_ = []
    #     # label in label set
    #     for label in source_labels[i]:
    #         matched_src_idx = torch.nonzero(torch.as_tensor(source_labels_all[i])==label).squeeze(1) # matched_idx: (11, 1)
    #         rois_source = weighted_rois_source[i][matched_src_idx].unsqueeze(1) # (num_rois, 1, feat_dim)
    #         sum_of_scores = sum(list_of_scores[i][matched_src_idx])

    #         # breakpoint()
    #         tmp.append(rois_source)
    #         tmp_.append(sum_of_scores)
    #     # per batch
    #     source_rois.append(tmp)
    #     source_scores.append(tmp_)

    # assert source_rois[0].__len__() == source_scores[0].__len__()
    
    ### refer to https://arxiv.org/abs/2103.04224
    _, num_prototypes, hidden_dim = prototypes.shape

    # cache aggregated prototypes 
    prototypes_local = torch.zeros((num_classes-1, num_prototypes, hidden_dim)).cuda()

    rois_reweighted = []
    # NOTE implementation considers multi-class, multi-moda prototypes
    for i in range(len(list_of_rois)):
        roi_reweighted = []
        # roi_sample_tmp = source_rois[i]
        roi_sample_tmp = list_of_rois[i]
        labels_tmp = list_of_labels[i]
        # for each roi_group
        for j in range(len(roi_sample_tmp)):
            # label_idx = source_labels[i][j]-1
            label_idx = list_of_labels[i][j]-1
            
            # get prototypes given the same class idx
            m_prototypes = prototypes[label_idx]
            label = labels_tmp[j]
            roi = roi_sample_tmp[j].unsqueeze(0)

            # roi = roi_sample_tmp[j].squeeze(1)

            # compute simialrity
            similarity_scores = cosine_distance(roi, m_prototypes)
            # similarity_scores = torch.matmul(roi, m_prototypes.transpose(1,0))
            # similarity_scores = F.softmax(similarity_scores, dim=1)

            # recompute rois given scores of prototypes
            roi_ag = torch.matmul(similarity_scores, m_prototypes) # (num_rois, feat_dim)
            proto_ag = torch.matmul(similarity_scores.transpose(1,0), roi) # (num_prototypes, feat_dim)
            
            # normalize
            roi_ag = F.normalize(roi_ag, p=2, dim=-1).squeeze(0)
            proto_ag = F.normalize(proto_ag, p=2, dim=-1)

            prototypes_local[label_idx] = roi_ag
            roi_reweighted.append(roi_ag)

        roi_reweighted = torch.stack(roi_reweighted)
        rois_reweighted.append(roi_reweighted)
    

    # rois_reweighted = torch.stack(rois_reweighted, dim=0)
    ### aggregate rois for each class
    # for i in range(len(source_rois)):
    #     roi_sample_tmp = source_rois[i] # some rois for a single class
    #     # j label
    #     for j in range(len(roi_sample_tmp)):
    #         aggregate = torch.sum(roi_sample_tmp[j], dim=0)/(torch.sum(list_of_scores[0]) + epsilon)
    #         # store prototype at the corresponding cls index position
    #         prototypes[source_labels[i][j]-1] = aggregate

    # class reweighting factor
    alpha_values_src = torch.ones((num_classes))

    # import pdb; pdb.set_trace()
    for cls_i in range(1,num_classes,1):
        if cls_i in list_of_labels[0]:
            indices_list = torch.nonzero(torch.as_tensor(list_of_labels[0])==cls_i).tolist()
            flatten = [index[0] for index in indices_list]
            p_max = max(list_of_scores[0][flatten])
            alpha = 1-p_max
            alpha_values_src[cls_i] = alpha
        else:
            continue

    alphas = alpha_values_src

    return prototypes_local, rois_reweighted, alphas


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

    var_temp = torch.zeros((C, A, A), device='cuda') # store target intra class variance
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros((N, C), device= 'cuda') # (N, C)
    onehot.scatter_(1, labels.view(-1, 1), 1) # labels --> (b*num_of_labels, 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1
    Amount_CxAxA = Amount_CxA.view(C, A, 1).expand(C, A, A)
    del Amount_CxA, onehot
    gc.collect()

    avg_NxCxA = ave_CxA.expand(N, C, A)

    # import pdb; pdb.set_trace() 
    for c in range(C):
        features_by_sort_c = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        avg_by_sort_c = avg_NxCxA[:, c, :].mul(NxCxA_onehot[:, c, :])
        var_temp_c = features_by_sort_c - avg_by_sort_c
        var_temp[c] = torch.mm(var_temp_c.permute(1,0), var_temp_c).div(Amount_CxAxA[c])
    
    return var_temp.detach() # (C, A, A) checked, values seem ok

def compute_sim(matrix_1, matrix_2):
    # score = torch.matmul(matrix_1, torch.t(matrix_2))
    # # score = score.view(bs*layer*num_box, m)
    # score_normalised = F.softmax(score, dim=0)
    # score_normalised*
    raise NotImplementedError

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


class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()
        # self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        # self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.classifier = nn.Conv2d(ndf*4, 1, kernel_size=4, stride=2, padding=1) # change clasifier dim
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        # x = self.conv3(x)
        # x = self.leaky_relu(x)
        # x = self.conv4(x)
        # x = self.leaky_relu(x)
        x = self.classifier(x)
        #x = self.up_sample(x)
        # x = self.sigmoid(x)

        # 
        return x

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

        # query: torch.Size([1, 2, 256])
        # key: torch.Size([588, 2, 256])
        # value: torch.Size([588, 2, 256])

        r_query, _ = self.cross_attn(
            query=query.transpose(0, 1),
            key=self.grl(self.with_pos_embed(src, pos)).transpose(0, 1),
            value=self.grl(src).transpose(0, 1),
            key_padding_mask=padding_mask,
        )

        # breakpoint()

        # residual
        query = query + self.dropout1(r_query.transpose(0, 1))
        query = self.norm1(query)
        # linear layer + dropout layer + layer norm
        query = query + self.dropout2(self.linear(query))
        query = self.norm2(query)
        return query

class Proto_Proposal_attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(Proto_Proposal_attention, self).__init__()

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, src, pos=None, padding_mask=None, attn_mask=None):
        """ Args:
            query (batch_size * num_rois_padded, flatten_hw, feat_dim): target proposals
            src (batch_size_num_rois_padded, batch_size_num_rois_padded, feat_dim): prototypes
            padding_mask (batch_size_num_rois_padded, batch_size_num_rois_padded, feat_dim): key padding mask
        """
        
        # breakpoint()
        # NOTE: not sure if we should include postional embedding, will try later if necessary
        # NOTE: attn_mask should suffice since target proposal and prototypes have the same padding,
        # which means wherever padding is applied for proposals is also applied to prototypes
        r_query, attn_weights = self.cross_attn(query=query.transpose(0, 1), key=src.transpose(0, 1), value=src.transpose(0, 1), key_padding_mask=None, attn_mask=attn_mask)

        # r_query, attn_weights = self.cross_attn(query=query, key=src, value=src,
        #                                         key_padding_mask=padding_mask, attn_mask= attn_mask)


        # NOTE: avoid nan outputs, assuming that inputs and masks are correct
        # r_query_new = r_query.masked_fill_(torch.isnan(r_query).byte(), 0)
        # attn_weights_new = attn_weights.masked_fill_(torch.isnan(attn_weights).byte(), 0)

        query = src + self.dropout1(r_query.transpose(0, 1))
        # query = src + self.dropout1(r_query.transpose(0, 1))
        query = self.norm1(query)
        # linear layer + dropout layer + layer norm
        query = query + self.dropout2(self.linear(query))
        query = self.norm2(query)

        # query: (batch_size * num_rois_padded, 7x7, feat_dim)
        # attn_weights_new: (batch_size * num_rois_padded, 7x7, 1)
        return query, attn_weights

class CrossAttention_agg_prototypes(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(CrossAttention_agg_prototypes, self).__init__()
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

        # TODO: local query token aggregation
        # r_query, attn_weights = self.cross_attn(
        #     query=query.transpose(0, 1),
        #     key=self.with_pos_embed(src, pos).transpose(0, 1),
        #     value=src.transpose(0, 1), key_padding_mask=padding_mask,
        # )

        # TODO 
        # import pdb; pdb.set_trace
        r_query, attn_weights = self.cross_attn(query=query.transpose(0, 1), key=self.with_pos_embed(src, pos).transpose(0, 1),value=src.transpose(0, 1), key_padding_mask=padding_mask,)

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
