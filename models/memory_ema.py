import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random

# class Memory(nn.Module):
#     def __init__(self, num_classes, key_dim, keep_rate=0.996, num_feature_levels=1):
#         super(Memory, self).__init__()
#         # self.memory_size = memory_size
#         self.key_dim = key_dim
#         self.keep_rate = keep_rate
#         batch_size = 2
#         # F.normalize(torch.rand((m_size, m_dim), dtype=torch.float), dim=1).cuda()
#         self.source_prototype = torch.zeros(batch_size*num_feature_levels, num_classes, key_dim).cuda()
#         self.target_prototype = torch.zeros(batch_size*num_feature_levels, num_classes, key_dim).cuda()  

#         self.num_feature_levels = num_feature_levels
#         # self.class_counts = torch.ones((num_classes)).cuda() # fix category temporarily
    
#     def forward(self, prototype_source, prototype_target):
#         if len(prototype_source) == 1:
#             prototype_source = prototype_source.unsqueeze(0)
#             prototype_target = prototype_target.unsqueeze(0)

#         self.source_prototype = (1-self.keep_rate)*self.source_prototype.detach() + self.keep_rate* prototype_source
#         self.target_prototype = (1-self.keep_rate)*self.target_prototype.detach() + self.keep_rate* prototype_target

#         # self.source_prototype = F.normalize(((1-self.keep_rate)*self.source_prototype.detach() + self.keep_rate* prototype_source), dim=1)
#         # self.target_prototype = F.normalize(((1-self.keep_rate)*self.source_prototype.detach() + self.keep_rate* prototype_source), dim=1)

#         return self.source_prototype, self.target_prototype


class Memory(nn.Module):
    def __init__(self, num_classes, key_dim, keep_rate=0.996, num_feature_levels=1):
        super(Memory, self).__init__()
        batch_size = 2
        self.key_dim = key_dim
        self.keep_rate = keep_rate
        # self.prototypes = torch.zeros(batch_size*num_feature_levels, num_classes, key_dim).cuda()
        # self.prototypes = F.normalize(torch.rand((batch_size*num_feature_levels, num_classes, key_dim), dtype=torch.float), dim=1).cuda()
        self.num_feature_levels = num_feature_levels
        # self.class_counts = torch.ones((num_classes)).cuda() # fix category temporarily
    
    def forward(self, memory, new_prototypes):
        # if more than one scale
        # if len(new_prototypes) > 1:
        #     new_prototypes = torch.mean(new_prototypes, dim=0) # average across scales

        memory_updated = (1-self.keep_rate)*memory.detach() + self.keep_rate* new_prototypes
        # self.source_prototype = F.normalize(((1-self.keep_rate)*self.source_prototype.detach() + self.keep_rate* prototype_source), dim=1)

        return memory_updated
