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

"""
Plotting utilities to visualize training logs.
"""
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path, PurePath
from typing import Dict, List, Optional
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torchvision.utils import make_grid


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt', mode='test'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # verify valid dir(s) and that every item in list is Path object
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if dir.exists():
            continue
        raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                # import pdb; pdb.set_trace()
                coco_eval = pd.DataFrame(pd.np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                # df.interpolate().ewm(com=ewm_col).mean().plot(
                #                     y=[f'test_{field}', f'train_{field}'],
                #                     ax=axs[j],
                #                     color=[color] * 2,
                #                     style=['-', '--']
                #                 )
                if mode == 'test':
                    df.interpolate().ewm(com=ewm_col).mean().plot(
                                        y=[f'test_{field}'],
                                        ax=axs[j],
                                        color=[color] * 2,
                                        style=['-', '--']
                                    )
                elif mode == 'train':
                    df.interpolate().ewm(com=ewm_col).mean().plot(
                                        y=[f'train_{field}'],
                                        ax=axs[j],
                                        color=[color] * 2,
                                        style=['-', '--']
                                    )


    for ax, field in zip(axs, fields):
        if mode == 'test':
            ax.legend(['test']) # TODO show test and train legends
        elif mode == 'train':
            ax.legend(['train']) # TODO show test and train legends

        ax.set_title(field)

def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs


def plot_bbox(
    img_tensors: Dict,
    res: Dict,
    coco,
    save_dir: str,
    score_threshold: int = 0.5,
    img_ids: Optional[List[int]] = None,
    prefix: Optional[str] = None
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img_ids = img_ids if img_ids is not None else coco.getImgIds()

    # BGR colors for all categories
    colors = [
        (47, 52, 227),
        (63, 153, 246),
        (74, 237, 255),
        (114, 193, 56),
        (181, 192, 77),
        (220, 144, 52),
        (205, 116, 101),
        (226, 97, 149),
        (155, 109, 246),
    ]

    fontFace = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.5
    thickness = 1

    for img_id, img_tensor in img_tensors.items():
        if img_id not in img_ids:
            continue
        
        img = img_tensor_to_cv2(img_tensor)

        output = res[img_id]
        label_count = {i: 0 for i in range(1, 9)}
        import pdb; pdb.set_trace()
        for score, label, box in zip(output['scores'], output['labels'], output['boxes']):
            if score < score_threshold:
                continue

            label_name = coco.cats[label.item()]['name']
            title = f'{label_name}({score.item():.3f})'
            label_count[label.item()] += 1

            x1, y1, x2, y2 = np.round(box.cpu().numpy()).astype(np.int64)

            labelSize = cv2.getTextSize(title, fontFace, fontScale, thickness)
            color = colors[label.item()]
            text_color = (0, 0, 0) if label_name == 'car' else (255, 255, 255)

            # plot box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # plot text
            cv2.rectangle(img, (x1, y1), (x1 + labelSize[0][0] - 1, y1 - labelSize[0][1]), color, cv2.FILLED)  # text background
            cv2.putText(img, title, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, fontScale, text_color, thickness)

        x, y = 20, 20
        for label, count in label_count.items():
            label_name = coco.cats[label]['name']
            msg = f'# {label_name}: {count}'
            color = (0, 0, 0) if label_name == 'car' else (255, 255, 255)
            cv2.rectangle(img, (x, y + 5), (x + 200, y - 11), colors[label], cv2.FILLED)  # text background
            cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_COMPLEX, fontScale, color, thickness)

            y += 11 + 5

        file_name = f'{prefix}_img-id={img_id}.png' if prefix else f'img-id={img_id}.png'
        save_path = Path(save_dir) / file_name
        cv2.imwrite(str(save_path), img)


def plot_tgt_map(
    tgt_tensors,
    tgt_res: List[Dict],
    coco,
    map_save_dir: Path,
    img_ids: Optional[List[int]] = None,
    prefix: Optional[str] = None
):
    assert len(tgt_tensors) == len(tgt_res)

    map_save_dir = map_save_dir / prefix
    map_save_dir.mkdir(parents=True, exist_ok=True)

    img_ids = img_ids if img_ids is not None else coco.getImgIds()

    for img_id, img_tensor in tgt_tensors.items():
        tgt_map_out = tgt_res[img_id]
        # tgt_map_out: {
        #     target['image_id']: {
        #        'box_ids': None,
        #        'box_labels': None,
        #        'box_xyxy': None,
        #        'score_map': None,
        #        'binary_mask': None
        #    }
        # }

        if img_id in img_ids:
            img = img_tensor_to_cv2(img_tensor, normalized=True)
            img_copy = img.copy()

            labels = tgt_map_out['box_labels']  # (#filtered_box,)
            boxes = tgt_map_out['box_xyxy']  # (#filtered_box, 4)
            score_map = tgt_map_out['score_map']  # (#filtered_box, #class, roi_h, roi_w)
            binary_mask = tgt_map_out['binary_mask']  # (#filtered_box, #class, roi_h, roi_w)

            for box in boxes:
                x1, y1, x2, y2 = box.int().numpy()
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            save_path = map_save_dir / f'img-id={img_id}_bbox.png'
            cv2.imwrite(str(save_path), img_copy)

            if 0 < len(boxes) < 10:
                fig, axes = plt.subplots(len(boxes), 3)
                axes = axes[None, :] if axes.ndim == 1 else axes
                for row_axes, box, label, map, mask in zip(axes, boxes, labels, score_map, binary_mask):
                    x1, y1, x2, y2 = box.int().numpy()
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    crop = img[y1:y2, x1:x2, ::-1]  # (h, w, C)
                    
                    try:
                        row_axes[0].imshow(crop)
                        row_axes[0].set_title(coco.cats[label]['name'])
                        row_axes[0].set_axis_off()

                        sns.heatmap(map[label - 1], ax=row_axes[1], square=True)
                        row_axes[1].set_axis_off()

                        row_axes[2].imshow(mask[label - 1])
                        row_axes[2].set_axis_off()
                    except:
                        print('axes:')
                        print(axes)
                        print(axes.shape)
                        print('row_axes:')
                        print(row_axes)
                        print(row_axes.shape)
                        continue
            
                file_name = f'img-id={img_id}_map.png'
                save_path = map_save_dir / file_name
                plt.savefig(str(save_path))


def img_tensor_to_cv2(img_tensor: torch.Tensor, normalized: bool = True) -> np.ndarray:
    """
    img_tensor: (C, H, W)
    """
    assert img_tensor.ndim == 3
    assert img_tensor.shape[0] == 3

    img = img_tensor.cpu().detach().permute(1, 2, 0).numpy()  # (H, W, C)

    if normalized:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std[None, None, -1] + mean[None, None, -1]

    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.copy()  # use `copy()` as https://github.com/opencv/opencv/issues/18120

    return img
