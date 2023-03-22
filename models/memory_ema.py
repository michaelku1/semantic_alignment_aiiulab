import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random

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
        # breakpoint()
        memory_updated = self.keep_rate*memory.detach() + (1-self.keep_rate)* new_prototypes

        return memory_updated