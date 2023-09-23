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
from .deformable_detr_prompt_add import build as build_vpt_add
from .deformable_detr_prompt_prepend import build as build_vpt_prepend
from .deformable_detr_prompt_prepend_shallow import build as build_vpt_add_prepend_shallow
from .deformable_detr_prompt_add_prepend import build as build_vpt_add_prepend
from .deformable_detr_prompt_add_1_feat import build as build_vpt_add_1_feat
from .deformable_detr_cross_domain import build as build_cross_domain
from .deformable_detr_cross_domain_mixup import build as build_cross_domainn_mixup

def build_model(cfg):
    if cfg.MODEL.VISUAL_PROMPT.SWITCH:
        if cfg.MODEL.VISUAL_PROMPT.LOCATION == 'prepend':
            if not cfg.MODEL.VISUAL_PROMPT.DEEP:
                print('`build_model` comes from `deformable_detr_prompt_prepend_shallow.py`')
                return build_vpt_add_prepend_shallow(cfg)
            else:
                print('`build_model` comes from `deformable_detr_prompt_prepend.py`')
                return build_vpt_prepend(cfg)
        elif cfg.MODEL.VISUAL_PROMPT.LOCATION == 'add':
            print('`build_model` comes from `deformable_detr_prompt_add.py`')
            return build_vpt_add(cfg)
        elif cfg.MODEL.VISUAL_PROMPT.LOCATION == 'add+prepend':
            print('`build_model` comes from `deformable_detr_prompt_add_prepend.py`')
            return build_vpt_add_prepend(cfg)
        elif cfg.MODEL.VISUAL_PROMPT.LOCATION == 'add-1-feat':
            print('`build_model` comes from `deformable_detr_prompt_add_1_feat.py`')
            return build_vpt_add_1_feat(cfg)
        else:
            raise ValueError(f'Wrong key value! `MODEL.VISUAL_PROMPT.LOCATION` should be one of `prepend`, or `add-1-feat`, but got {cfg.MODEL.VISUAL_PROMPT.LOCATION}')
    
    elif 'cross_domain' in cfg.DATASET.DA_MODE:
        assert cfg.MODEL.CROSS_DOMAIN.SWITCH
        print('`build_model` comes from `deformable_detr_cross_domain.py`')
        return build_cross_domain(cfg)

    elif 'mixup' in cfg.DATASET.DA_MODE:
        assert cfg.DATASET.MIXUP.SWITCH
        print('`build_model` comes from `deformable_detr_cross_domain_mixup.py`')
        return build_cross_domainn_mixup(cfg)

    else:
        print('`build_model` comes from `deformable_detr.py`')
        return build(cfg)

