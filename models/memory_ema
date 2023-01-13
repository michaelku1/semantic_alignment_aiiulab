import torch
import torch.nn as nn


class Memory(nn.Module):
    def __init__(self, num_classes, key_dim, keep_rate=0.9):
        super(Memory, self).__init__()
        # self.memory_size = memory_size
        self.key_dim = key_dim
        self.keep_rate = keep_rate
        self.source_prototype = torch.zeros(num_classes, key_dim).cuda()  # 1 for each class
        self.target_prototype = torch.zeros(num_classes, key_dim).cuda()
        # self.class_counts = torch.ones((num_classes)).cuda() # fix category temporarily

    def forward(self, prototype_source, prototype_target):
        # update memory
        self.source_prototype = (1 - self.keep_rate) * self.source_prototype.detach() + self.keep_rate * prototype_source
        self.target_prototype = (1 - self.keep_rate) * self.target_prototype.detach() + self.keep_rate * prototype_target

        return self.source_prototype, self.target_prototype
