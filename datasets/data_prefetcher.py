# ------------------------------------------------------------------------
# Modified by Wei-Jie Huang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

def to_cuda(samples, targets, device):
    # [{'boxes': tensor([[0.7190, 0.3672, 0.2085, 0.2852], [0.4966, 0.4048, 0.1982, 0.0576]]), 'labels': tensor([3, 3]), 'image_id': tensor([1481]), 'area': tensor([40792.3516,  3441.1697]), 'iscrowd': tensor([0, 0]), 'orig_size': tensor([1024, 2048]), 'size': tensor([ 666, 1332])}]
    # [{'boxes': [tensor([0.7190, 0.3672, 0.2085, 0.2852]), tensor([0.4966, 0.4048, 0.1982, 0.0576])], 'labels': tensor([3, 3]), 'image_id': tensor([1481]), 'area': tensor([40792.3516,  3441.1697]), 'iscrowd': tensor([0, 0]), 'orig_size': tensor([1024, 2048]), 'size': tensor([ 666, 1332])}]
    # import pdb; pdb.set_trace()
    samples = samples.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets

### collated dataloader
class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            ###
            self.next_samples, self.next_targets = next(self.loader) ### collated dataloader
            # import pdb; pdb.set_trace()

        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets

            # prefetch allows record_stream
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                # import pdb; pdb.set_trace()
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets
