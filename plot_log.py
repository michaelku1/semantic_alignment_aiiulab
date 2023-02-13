import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', type=str)
    args = parser.parse_args()

    return args


def plot_loss(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f'The file `{str(log_path)}` does not exist')
    dir_path = log_path.parents[0]
    save_path = dir_path / 'loss.png'

    df_log = pd.read_json(log_path, lines=True)
    
    loss_names = [
        'loss',
        'loss_bbox',
        # 'loss_bbox_0',
        # 'loss_bbox_1',
        # 'loss_bbox_2',
        # 'loss_bbox_3',
        # 'loss_bbox_4',
        'loss_ce',
        # 'loss_ce_0',
        # 'loss_ce_1',
        # 'loss_ce_2',
        # 'loss_ce_3',
        # 'loss_ce_4',
        'loss_giou',
        # 'loss_giou_0',
        # 'loss_giou_1',
        # 'loss_giou_2',
        # 'loss_giou_3',
        # 'loss_giou_4',
        'loss_space_query',
        'loss_channel_query',
        'loss_instance_query',
    ]

    plt.figure(figsize=(12, 6))
    for prefix in ['train', 'test']:
        if prefix == 'train':
            line_style = '--'
            alpha = 0.5
        else:
            line_style = '-'
            alpha = 1.0

        for loss_name in loss_names:
            loss_name = f'{prefix}_{loss_name}'
            if loss_name in df_log.columns:
                losses = df_log[loss_name].values
                plt.plot(losses, line_style, label=loss_name, alpha=alpha)
                plt.text(len(losses), losses[-1], loss_name)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig(str(save_path))
    print(f'Saved to {str(save_path)}!')


def plot_map(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f'The file `{str(log_path)}` does not exist')
    dir_path = log_path.parents[0]
    save_path = dir_path / 'map.png'

    df_log = pd.read_json(log_path, lines=True)    
    coco_eval = np.stack(df_log['test_coco_eval_bbox'].dropna().values)[:, 1]
    epoch = np.argmax(coco_eval)

    plt.figure(figsize=(12, 6))
    for cat_id in range(9):
        if cat_id == 0:
            max_mAP = coco_eval[epoch]

            col_name = 'test_coco_eval_bbox'
            label = f'overall ({max_mAP:.4f})'
            line_stype = '-'
            alpha = 1.0

            plt.plot(coco_eval, line_stype, label=label, alpha=alpha)
        else:
            col_name = [col for col in df_log.columns if col.startswith(f'test_coco_eval_bbox_cat_id={cat_id}')][0]
            cat_name = col_name.split('name=')[-1]
            label = cat_name
            line_stype = '--'
            alpha = 0.5

            cat_mAP = np.stack(df_log[col_name].dropna().values)[:, 1]
            label += f' ({cat_mAP[epoch]:.4f})'
            plt.plot(cat_mAP, line_stype, label=label, alpha=alpha)

    plt.xlabel('epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.savefig(str(save_path))
    print(f'Saved to {save_path}!')


if __name__ == '__main__':
    args = get_args()
    plot_loss(log_path=args.log_path)
    plot_map(log_path=args.log_path)
