import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random
from torch.nn import functional as F

class Memory(nn.Module):
    def __init__(self, memory_size, key_dim, num_classes):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.class_counts = torch.ones((num_classes)).cuda() # fix category temporarily

    
    # update memory featuers
    def get_update_query(self, mem, max_indices, update_indices, score, query, labels, num_classes, train):
        # query: multi-scale category codes
        # max_indices: max score query indices
        # update_indices: 
        # import pdb; pdb.set_trace()
        m, d = mem.size()
        # assert m%num_classes==0 # check
        if train:
            query_update = torch.zeros((m,d)).cuda()
            # # import pdb; pdb.set_trace()

            # update by partition
            for class_id in labels:
                st = int(class_id*(m/num_classes))
                end = int((class_id*(m/num_classes)+(m/num_classes)))

                for j in range(st, end):
                    # if j == 40:
                        # import pdb; pdb.set_trace()
                    idx = torch.nonzero(max_indices.squeeze(1)==j)
                    a, _ = idx.size()

                    if a != 0:
                        # e.g score[:,i] there are 62 query features that are close to the memory 0
                        query_update[j] = torch.sum(((score[idx,j] / torch.max(score[:,j])) *query[idx].squeeze(1)), dim=0)
                        
                    else:
                        # case where there are no retrieved indices of query features for updating the ith memory
                        query_update[j] = 0

                # #TODO: normalize within the category
                # memory_part = query_update[st:end]

                # check memory partition size
                # assert len(memory_part) == len(query_update)/num_classes

                # memory_part = F.normalize(memory_part, dim=0)
                # query_update[st:end] = memory_part

            return query_update
        
        # TODO: no memory update at test time
        else:
            return
    
    # TODO: write two separate functions for reading and writing since the query dimensions are different for read and write
    def get_score_write(self, mem, query):
        # bs, layer, num_box, d = query.size()
        m, d = mem.size()
        # import pdb; pdb.set_trace()
        if len(query.size())==4:
            bs, layer, num_box, d = query.size()
            score = torch.matmul(query, torch.t(mem))# b X l X box X m (memory dim)
            score = score.view(bs*layer*num_box, m)# (b X l X box) X m
            
        else:
            num_box, dims = query.size()
            try:
                score = torch.matmul(query, torch.t(mem))# box X m (memory dim)
            except RuntimeError:
                import pdb; pdb.set_trace()
        
        # score for each query against all memory items
        score_query = F.softmax(score, dim=0)

        # in case score map is zero
        if score_query.shape[0] ==0:
            import pdb; pdb.set_trace()
        # score for each memory item against all queries
        score_memory = F.softmax(score,dim=1)
        
        return score_query, score_memory

    
    # TODO: per memory block reweighted features
    def block_compare(self, memory_block, query):
        # memory_block: (5x256)
        # query: (HxWx256)

        # initialise reweighted feature block 
        # category_block_feat = torch.zeros((query.shape[0], memory_block.shape[1]))  # (HxW, C)
        # compute correlation map given the memory block
        try:
            score = torch.matmul(memory_block, torch.t(query)) # (partitionxHxw)
        except RuntimeError:
            import pdb; pdb.set_trace()
        
        # import pdb; pdb.set_trace()
        ws = torch.matmul(torch.t(score.detach()), memory_block)

        # (hXw, c)
        return ws

    # feat_map is encoder features
    # def forward(self, query, source_feat_map, target_feat_map, labels, keys, num_classes, train=True):
    def forward(self, query, embeddings, query_embed, labels, keys, num_classes, train=True):

        """
        source prototype and target prototype are for memory retrieval; query is the category code/roi-pooled class features
        at train time and normal features at test time

        """

        # source_features: torch.Size([2048, 28, 42])
        # target_features: torch.Size([2048, 28, 42]) --> need 2D average pooling
        # queries: torch.Size([1, 6, 11, 256])
        # keys: torch.Size([9, 256])
        
        #train
        if train:
            # in case of empty boxes
            if len(query)==0:
                # import pdb; pdb.set_trace()

                # read for both source and target features
                # updated_source, softmax_score_query_source, softmax_score_memory = self.read(source_feat_map, keys, num_classes)
                # updated_target, softmax_score_query_target, softmax_score_memory = self.read(target_feat_map, keys, num_classes)

                # spatial_dims = [source_feat_map.shape[2], source_feat_map.shape[3]]
                keys = torch.stack(keys).squeeze(1)
                
                # TODO update memory item list's elements in the corresponding positions
                category_feat_src = self.read(embeddings[0], keys, num_classes)
                category_feat_tgt = self.read(embeddings[1], keys, num_classes)

                updated_features = torch.stack([category_feat_src, category_feat_tgt])

                if updated_features == None:
                    import pdb; pdb.set_trace()

                # import pdb; pdb.set_trace()
                # if empty boxes found, return old memory
                # return updated_features, keys, None, spatial_dims

                return updated_features, keys, None

            else:
                # store spatial dims
                # spatial_dims = [source_feat_map.shape[2], source_feat_map.shape[3]]
                
                ### query is meant to be of three dimensions, thus we can safely index it from the list
                # query = torch.stack(query[0]).unsqueeze(0)
                labels = labels[0] # tensor

                # b X d X h X w in original implementation
                # batch_size, layers, num_box, feat_dim = query.size() # batch X layers X num_box X feat_dim
                query = F.normalize(query, dim=1) # normalize along feature dimension


                # TODO update only the prototype from which the label exists in labels
                for label in labels:
                # query = query.permute(0,2,3,1) # b X h X w X d
                #update (ok)
                    try:
                        updated_memory, labels_padded = self.update(query, keys[label-1], labels, num_classes, train)
                        keys[label-1] = updated_memory
                    except IndexError:
                        import pdb; pdb.set_trace()

                updated_memory = torch.stack(keys).squeeze(1)
                # import pdb; pdb.set_trace()
                # read for both source and target features with the updated memory

                # import pdb; pdb.set_trace()
                category_feat_src = self.read(embeddings[0], updated_memory, num_classes)

                # (9, 700, 256)
                category_feat_tgt = self.read(embeddings[1], updated_memory, num_classes)

                ### eta debug testing
                # category_feat_src = torch.randn((9,spatial_dims[0]*spatial_dims[1],256))
                # category_feat_tgt = torch.randn((9,spatial_dims[0]*spatial_dims[1],256))

                updated_features = torch.stack([category_feat_src, category_feat_tgt])

                # import pdb; pdb.set_trace()

                if updated_features == None:
                    import pdb; pdb.set_trace()

                # return updated_features, updated_memory, labels_padded, spatial_dims
                return updated_features, updated_memory, labels_padded

        # test
        else:
            # here query is normal features
            updated_source, softmax_score_query_source, softmax_score_memory = self.read(query, keys)
            
            # no update at test time
            updated_memory = keys
            updated_features = updated_source

            return updated_features, updated_memory
        
    # TODO: pass labels to enforce memory update at certain partition
    def update(self, query, keys, labels, num_classes, train):
        
        # batch_size, l, num_box, dims = query.size() # b X h X w X d

        # import pdb; pdb.set_trace() 
        if len(query.size()) == 4:
            batch_size, l, num_box, dims = query.size() # b X h X w X d
            query = query.contiguous().view(batch_size*l*num_box, dims) # collapse all dimensions except channel dims
        else:
            num_box, dims = query.size()
            l = 1

        softmax_score_query, softmax_score_memory = self.get_score_write(keys, query)
        
        # (126, 256)
        # query_reshape = query.contiguous().view(batch_size*l*num_box, dims) # collapse all dimensions except channel dims

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)

        # softmax_score_query gives zero dimension
        # print(softmax_score_query.shape)

        # 66 queries in total; therefore one score per query for all 9 memory items
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)
        
        if train:
            # top-1 queries (of each memory) update (weighted sum) & random pick
            # both have the same dimensions torch.Size([9, 256]) where for some memory item the entries are zero, which means no update
            # import pdb; pdb.set_trace()
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query, query, labels, num_classes, train)
            
            # TODO: only part of keys will be updated
            # (memory_size, feature_dim)
            
            # TODO: keep a running mean
            partition = int(len(query_update)/num_classes)
            labels, counts = torch.unique(labels, return_counts=True)

            # keep a running mean on all memory items belongong to the categories in the label list
            for label, count in zip(labels,counts):
                class_counts_temp = []
                label = label.item()
                st = partition*label
                end = partition*label+partition

                # interval
                interval = torch.arange(st,end)

                
                # query updated
                # ((n-1)*old_average + new_value)/n
                
                # old counts for the interval, 1 everywhere else
                # set condition with st and end
                class_counts_temp = [self.class_counts[label] if idx in interval else 1 for idx in range(len(keys))]

                class_counts_temp = torch.as_tensor(class_counts_temp).unsqueeze(1).cuda()

                updated_keys = class_counts_temp * keys + query_update

                self.class_counts[label] = self.class_counts[label] + count
                class_counts_temp[st:end] = class_counts_temp[st:end] + count

                updated_keys_final = updated_keys / class_counts_temp

                # key_updated = ((self.class_counts[label]*keys[st:end].clone())+query_update[st:end]) # (paritition_size, n_dim)
                
                # update count
                # self.class_counts[label] = self.class_counts[label] + count

                # updated memory
                # import pdb; pdb.set_trace()
                # keys[st:end] = keys[st:end].clone()/self.class_counts[label]

            # query_updated = query_update + keys
            # TODO: this normalization normalizes the entire memory
            # updated_memory = F.normalize(query_updated, dim=1) # normalise along channel

            # import pdb; pdb.set_trace()
            updated_memory = F.normalize(updated_keys_final, dim=1)

        # TODO: pad labels to required length (layer)
        labels = self.pad_labels(labels, l)

        # return query_updated.detach(), labels
        return updated_memory.detach(), labels

    # no need to pass labels
    def read(self, query, updated_memory, num_classes):
        # updated_memory is keys
        # query is the source/target prototype

        # import pdb; pdb.set_trace()
        # batch_size, l, num_box, dims = query.size() # b X l X box X d
        
        # import pdb; pdb.set_trace()
        # output 1: normalise over the query dimension: comparing one memory against all queries
        # output 2: normalise over the memory dimension: comparing one query against all memory items (therefore use this score map)
        
        partition = int(updated_memory.shape[0]/(num_classes-1))

        query = query.reshape(updated_memory.shape[1],-1)
        query = torch.t(query) # (HWx256)

        # import pdb; pdb.set_trace()
        block_feat_all = []
        for i in range(0,updated_memory.shape[0],partition):
            # TODO: return category-specific block features, one for each memory block (e.g 9 blocks in total)
            block_feat = self.block_compare(updated_memory[i:i+partition], query) # updated memory since read is only performed after memory update
            block_feat_all.append(block_feat)

        block_feat_all = torch.stack(block_feat_all)
        # import pdb; pdb.set_trace()
        return block_feat_all


    # TODO: used to pad labels to the required length
    def pad_labels(self, labels, l):
        new_list = []
        for i in labels:
            index = torch.where(labels==i)
            temp = [labels[index]]*l
            new_list.append(temp)

        initial = new_list[0] # store first one
        for i in range(1, len(new_list)):
            initial = initial + new_list[i]
        return initial