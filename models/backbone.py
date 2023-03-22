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
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    
    ### backbone freezing here and omni-detr is a little different
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()

        self.backbone = backbone

        # freeze layer2, 3, and 4 if not training the backbone
        # determine whether the layers of backbone are to be trained by the names in the list
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        # initialises the layer's variables if return interim layers
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]

        ### TODO change here if you want to return different layers, but here it only returns single layer feature only
        # elif return_first_and_last_layers:
        #     return_layers = {"layer1": "0", "layer4":"2"}
        #     self.strides = [8, 32]
        #     self.num_channels = []

        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        # import pdb; pdb.set_trace()
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

    ## for the sake of convenience
    def support_encoding_net(self, x, return_interm_layers=False):
        out: Dict[str, NestedTensor] = {}
        m = x.mask
        # x = self.meta_conv(x.tensors)
        x = self.backbone.conv1(x.tensors)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)

        # for support feature maps at different scales
        if return_interm_layers:
            # interpolate the mask values according to the desired shape
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['0'] = NestedTensor(x, mask) # layer 2

        x = self.backbone.layer3(x)
        if return_interm_layers:
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['1'] = NestedTensor(x, mask) # layer 3

        x = self.backbone.layer4(x)
        if return_interm_layers:
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['2'] = NestedTensor(x, mask) # layer 4

        # import pdb; pdb.set_trace()

        if return_interm_layers:
            return out

        #
        else:
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['0'] = NestedTensor(x, mask) # output of backbone's last layer

            return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)

        # import pdb; pdb.set_trace()
        
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

# TODO: add backbone debug
# class Backbone_debug(BackboneBase):
#     """ResNet backbone with frozen BatchNorm."""
#     def __init__(self, name: str,
#                  train_backbone: bool,
#                  return_interm_layers: bool,
#                  dilation: bool):
#         norm_layer = FrozenBatchNorm2d
#         backbone = getattr(torchvision.models, name)(
#             replace_stride_with_dilation=[False, False, dilation],
#             pretrained=is_main_process(), norm_layer=norm_layer)

#         # import pdb; pdb.set_trace()
        
#         assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
#         super().__init__(backbone, train_backbone, return_interm_layers)
#         if dilation:
#             self.strides[-1] = self.strides[-1] // 2

#         self.backbone = backbone

    # def forward(self, x):
    #     if cfg.BACKBONE_DEBUG:
    #         x = self.backbone(x)
    #         x = self.avgpool(x)
    #         x = torch.flatten(x,1)

# joins backbone with position embedding layer
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list) # self is a network (including both backbone and positional embedding)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))
        
        return out, pos

    ### TODO: for the sake of implementation, use support encoding network for source images for now
    def forward_supp_branch(self, tensor_list: NestedTensor, return_interm_layers=False):

        # self[0] is the backbone network function
        # self[1] is the positionembedding function
        # xs is a nested tensor object containing both image tensor and mask tensor
        xs = self[0].support_encoding_net(tensor_list, return_interm_layers=return_interm_layers) # torch.Size([25, 2048, 21, 21])
        out: List[NestedTensor] = []
        pos = []
        
        # import pdb; pdb.set_trace()
        for name, x in sorted(xs.items()):
            out.append(x)

        # self[1] os the position encoding function
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        # import pdb; pdb.set_trace()

        # torch.Size([25, 2048, 21, 21])
        # torch.Size([25, 256, 21, 21])
        return out, pos


def build_backbone(cfg):
    position_embedding = build_position_encoding(cfg)
    train_backbone = cfg.TRAIN.LR_BACKBONE > 0
    return_interm_layers = cfg.MODEL.MASKS or (cfg.MODEL.NUM_FEATURE_LEVELS > 1)
    # if cfg.DEBUG:
    #     backbone = Backbone_debug(cfg.MODEL.BACKBONE, train_backbone, return_interm_layers, cfg.MODEL.DILATION)
    # else:
    backbone = Backbone(cfg.MODEL.BACKBONE, train_backbone, return_interm_layers, cfg.MODEL.DILATION)
    model = Joiner(backbone, position_embedding)
    return model
