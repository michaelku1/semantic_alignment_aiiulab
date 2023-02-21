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

# from .deformable_detr import build

# from .deformable_detr_contrastive_bak import build

# from .deformable_detr_contrastive import build

# from .deformable_detr_contrastive_cross_scale import build

from .deformable_detr_contrastive_tgt_proposal_reweight import build

# from .deformable_detr_contrastive_tgt_proposal_reweight_debug import build

def build_model(cfg):
    return build(cfg)

