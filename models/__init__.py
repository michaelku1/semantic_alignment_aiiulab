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

from .deformable_detr import build
from .deformable_detr_prompt_add_1_feat import build as build_vpt

def build_model(cfg):
    if cfg.MODEL.VISUAL_PROMPT.SWITCH:
        print('`build_model` comes from `deformable_detr_prompt_add_1_feat.py`')
        return build_vpt(cfg)
    else:
        print('`build_model` comes from `deformable_detr.py`')
        return build(cfg)

