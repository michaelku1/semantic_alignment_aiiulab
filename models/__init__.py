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
from .deformable_detr_contrastive import build as build_contrastive


def build_model(cfg):
    if cfg.CONTRASTIVE:
        print()
        print('*' * 10)
        print('* Build model for contrastive!!!')
        print('*' * 10)
        return build_contrastive(cfg)

    return build(cfg)
