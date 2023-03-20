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
from pathlib import Path
from typing import Dict, List, Optional

import seaborn as sns
import matplotlib.pyplot as plt

import torch


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
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
        if isinstance(logs, Path):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # verify valid dir(s) and that every item in list is Path object
    for i, dir in enumerate(logs):
        if not isinstance(dir, Path):
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
                coco_eval = pd.DataFrame(pd.np.stack(df.test_coco_eval.dropna().values)[:, 1]).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
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
    coco,
    res: Dict,
    root_dir: Path,
    box_save_dir: str,
    score_threshold: int = 0.5,
    img_ids: Optional[List[int]] = None,
    prefix: Optional[str] = None
):
    box_save_dir = Path(box_save_dir)
    if not box_save_dir.exists():
        box_save_dir.mkdir(parents=True, exist_ok=True)

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

    for img_id, output in res.items():
        if img_id not in img_ids:
            continue
        
        info = coco.loadImgs(img_id)[0]
        img_path = root_dir / info['file_name']
        img = cv2.imread(str(img_path))

        label_count = {i: 0 for i in range(1, 9)}
        for score, label, box in zip(output['scores'], output['labels'], output['boxes']):
            if score < score_threshold:
                continue

            label_name = coco.cats[label.item()]['name']
            title = f'{label_name}({score.item():.3f})'
            label_count[label.item()] += 1

            x1, y1, x2, y2 = np.round(box.cpu().numpy()).astype(np.int)

            labelSize = cv2.getTextSize(title, fontFace, fontScale, thickness)
            color = colors[label.item()]

            # plot box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # plot text
            cv2.rectangle(img, (x1, y1), (x1 + labelSize[0][0] - 1, y1 - labelSize[0][1]), color, cv2.FILLED)  # text background
            cv2.putText(img, title, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, fontScale, (255, 255, 255), thickness)

        x, y = 20, 20
        for label, count in label_count.items():
            label_name = coco.cats[label]['name']
            msg = f'# {label_name}: {count}'

            cv2.rectangle(img, (x, y + 5), (x + 200, y - 11), colors[label], cv2.FILLED)  # text background
            cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_COMPLEX, fontScale, (255, 255, 255), thickness)

            y += 11 + 5

        file_name = f'{prefix}_img-id={img_id}.png' if prefix else f'img-id={img_id}.png'
        save_path = Path(box_save_dir) / file_name
        cv2.imwrite(str(save_path), img)

