from util.plot_utils import inverse_transform
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import math

import os

def check_pseudo_boxes(B, targets, samples, rescaled_boxes_enc, list_of_scores_enc, list_of_labels_enc):
    """
    plot pseudo boxes on unpadded samples, save results for visualization

    rescaled_boxes_enc: list of tensors
    """

    dir_name = './visualization_debug'
    if not os.isdir(dir_name/check_pseudo_boxes.__name__):
        os.makedirs(dir_name, exist_ok=True)    

    CLASSES = ['person','car','train','rider','truck','motorcycle','bicycle', 'bus']
    target_bs_num = B//2
    targets_target = targets[:target_bs_num]
    target_rescaled_boxes_enc = rescaled_boxes_enc[:target_bs_num]
    target_scores_enc = list_of_scores_enc[:target_bs_num]
    target_labels_enc = list_of_labels_enc[:target_bs_num]


    for B_i in range(target_bs_num):
        # city_name = str(img_path).split('/')[-2]
        # plot samples
        unmasked_samples = samples.tensors[B_i][:, :targets[B_i]['size'][0], :targets[B_i]['size'][1]]
        # sample = samples.tensors[:B//2] # all target samples
        inverted_image_tensors = inverse_transform(unmasked_samples)
        inverted_image_tensors = inverted_image_tensors.permute(1,2,0)
        plt.figure(figsize=(30, 50))
        plt.imshow(inverted_image_tensors.detach())

        boxes_tmp = target_rescaled_boxes_enc[B_i].detach()
        scores_tmp = target_scores_enc[B_i].detach()
        labels_tmp = target_labels_enc[B_i] # list

        ax = plt.gca() # initialise gca

        for (xmin, ymin, xmax, ymax), p, class_idx in zip(boxes_tmp, scores_tmp, labels_tmp):
            # add patch
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color='g', linewidth=1))
            # add text
            text = f'{CLASSES[class_idx-1]}: {str(p.item())[:4]}'
            # add font style
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        plt.axis('off')
        image_id_tmp = targets[B_i]['image_id'].detach().cpu()
        plt.savefig(f'./visualization_debug/check_pseudo_boxes/sample_{image_id_tmp.item()}.png', bbox_inches='tight')

    return 

def check_prototypes(prototypes, cur_iter_num, save=False):
    """
    save prototypes for further visualization/value checking
    prototypes: (2, C, feat_dim)
    """
    if save:
        torch.save(prototypes, f'prototypes_epoch_cur_iter{cur_iter_num}')


    return NotImplementedError


def check_mask_features():
    return NotImplementedError

# check attention between prototype and proposal 
def check_prototype_attention_map(B, prototype, memory_features, proposals, targets,
                                visualize_class='car', multi_scale=False, *args):
    """
    visualize attention map, reweighted feature map and pseudo boxes for one batch of examples


    prototype: can be either source or target prototype (cls_num, feat_dim)
    memory_features: reshaped memory features, including both src and tgt (scale, bs,)
    proposals: list of tgt boxes (of different sizes)
    """

    dir_name = './visualization_debug'
    full_dir_name = dir_name + '/' + check_prototype_attention_map.__name__
    if not os.path.isdir(full_dir_name):
        os.makedirs(full_dir_name, exist_ok=True)
    
    scale = 1/32.
    CLASSES = ['person','car','train','rider','truck','motorcycle','bicycle', 'bus']
    target_bs_num = B//2
    targets_target = targets[:target_bs_num]


    image_ids = [target['image_id'] for target in targets_target]

    # NOTE: even in the case of multi-scale features, only one feature will be used for visualization
    if not multi_scale:
        memory_features_tgt = memory_features[0]
    else:
        return NotImplementedError

    # breakpoint()

    pooled_feature_batch = [] # [B, ...]
    for B_i in range(target_bs_num):
        pooled_feature_list = []
        tgt_boxes_tmp = proposals[B_i] # list of box tensors
        for tgt_box in tgt_boxes_tmp:
            tgt_box_tmp = tgt_box.detach().clone()
            tgt_box_tmp[0] = tgt_box_tmp[0] * scale
            tgt_box_tmp[1] = tgt_box_tmp[1]* scale
            tgt_box_tmp[2] = tgt_box_tmp[2] * scale
            tgt_box_tmp[3] = tgt_box_tmp[3]* scale

            # down scale box height and width
            height = math.ceil((tgt_box[3] - tgt_box[1]) * scale)
            width = math.ceil((tgt_box[2] - tgt_box[0]) * scale)

            # take one box tensor at a time
            # here we are trying to pooled the entire feature map instead of downscaling it to 7x7
            pooled_feature = torchvision.ops.roi_align(memory_features_tgt[B_i].unsqueeze(0), [tgt_box_tmp.unsqueeze(0)],output_size=(height,width), spatial_scale=1.0, aligned=True)
            pooled_feature_list.append(pooled_feature)

        pooled_feature_batch.append(pooled_feature_list)

    for B_i in range(target_bs_num):
        tgt_boxes = proposals[B_i] # tensor

        tgt_features = pooled_feature_batch[B_i] # [num_rois,...]
        class_idx = CLASSES.index(f'{visualize_class}')
        cls_filter = prototype[class_idx]
        image_id = image_ids[B_i].item()
        cls_filter = cls_filter.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # along number of roi pooled features
        for i in range(len(tgt_features)):
            tgt_features_tmp = tgt_features[i] # (1, feat_dim, h, w)
            tgt_boxes_tmp = tgt_boxes[i]

            # tgt_features_tmp: (1,256,h,w)
            # cls_filter : (1, feat_dim, 1, 1)
            out_similarity = F.conv2d(tgt_features_tmp, cls_filter).sigmoid() # similarity scores
            reweighted_tgt_feature = tgt_features_tmp * out_similarity
            
            out_similarity = out_similarity.cpu()
            reweighted_tgt_feature = reweighted_tgt_feature.cpu()
            # torch.save(tgt_boxes_tmp, f'pseudo_boxes_{image_id}_num_{i}')
            
            torch.save(out_similarity, f'{full_dir_name}/similarity_score_{image_id}_{visualize_class}.pt')
            torch.save(reweighted_tgt_feature, f'{full_dir_name}/reweighted_tgt_feat_{image_id}_{visualize_class}.pt')

    return