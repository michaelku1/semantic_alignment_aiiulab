import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def plot_map(log_path, plot_lr, plot_class_map=False, plot_max_prompt_norm=False, pretrained_log_path=None, start_epoch=0):
    exp_name = log_path.split('/')[-2]
    log_path = Path(log_path)
    log_dir = log_path.parents[0]
    save_path = log_dir / 'mAP.png'

    df_log = pd.read_json(str(log_path), lines=True)
    df_log = get_rid_of_str_epoch(df_log)
    coco_eval, max_mAP, max_epoch, class_coco_evals, class_max_mAPs = get_coco_eval_results(df_log)

    fig, ax = plt.subplots(figsize=(15, 8))
    if pretrained_log_path:
        pretrained_log_path = Path(pretrained_log_path)
        df_pretrained_log = pd.read_json(str(pretrained_log_path), lines=True)
        df_pretrained_log = get_rid_of_str_epoch(df_pretrained_log)
        pretrained_coco_eval, pretrained_max_mAP, _, _, _ = get_coco_eval_results(df_pretrained_log)

        plt.plot(pretrained_coco_eval, label='pretrained')

        if start_epoch > 0:
            df_pretrained_log = df_pretrained_log.iloc[:start_epoch]
            ax.axvline(x=start_epoch, ls='--', c='r')

        df_log = pd.concat([df_pretrained_log, df_log], ignore_index=True)
        coco_eval, max_mAP, max_epoch, class_coco_evals, class_max_mAPs = get_coco_eval_results(df_log)
    
    title = f'Max mAP = {max_mAP:.4f} ({max_epoch}-th epoch)'
    if pretrained_log_path:
         title += f' (baseline = {pretrained_max_mAP:.4f})'

    ax.plot(coco_eval, label=exp_name)
    if plot_class_map:
        for (cat_id, cat_name), coco_eval in class_coco_evals.items():
            class_max_mAP = class_max_mAPs[(cat_id, cat_name)]
            ax.plot(coco_eval, label=f'({cat_id}) {cat_name} ({class_max_mAP:.4f})', alpha=0.3)
    if plot_lr:
        ax2 = ax.twinx()
        for col in df_log.columns:
            if 'train_lr' in col:
                ax2.plot(df_log[col], '--', label=col)
        ax2.set_ylabel(col)
    if plot_max_prompt_norm:
        ax2 = ax.twinx()
        for col in df_log.columns:
            if 'max_prompt_norm' in col:
                ax2.plot(df_log[col], '.', label=col)
                ax2.set_ylabel(col)
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.set_ylabel('mAP')
    plt.legend(loc='lower right')
    plt.savefig(str(save_path))
    print(f'Saved to {str(save_path)}!')


def get_rid_of_str_epoch(df_log):
    # the column `epoch` may contain a string `before fine tuning`
    mask = ~df_log['epoch'].apply(lambda e: isinstance(e, str))
    return df_log[mask]


def get_coco_eval_results(df_log):
    coco_eval = np.stack(df_log['test_coco_eval_bbox'].values)
    coco_eval = coco_eval[:, 1]  # IoU = 0.5
    max_epoch = np.argmax(coco_eval)
    max_mAP = coco_eval[max_epoch]

    class_coco_evals = {}
    class_max_mAPs = {}
    for col in df_log.columns:
        if not col.startswith('test_coco_eval_bbox_cat-id'):
            continue

        cat_id_cat_name = col.split('test_coco_eval_bbox_')[-1]  # cat-id=7_cat-name=bicycle
        cat_id = int(cat_id_cat_name.split('_')[0].split('=')[-1])
        cat_name = cat_id_cat_name.split('_')[-1].split('=')[-1]
        
        class_coco_eval = []
        for x in df_log[col]:
            if isinstance(x, list):
                class_coco_eval.append(x[1])  # IoU = 0.5
            else:  # None, because the pretrained log has no column named as `test_coco_eval_bbox_cat-id=x_cat-name=yyy`
                class_coco_eval.append(None)
        class_max_mAP = class_coco_eval[max_epoch]

        class_coco_evals[(cat_id, cat_name)] = class_coco_eval
        class_max_mAPs[(cat_id, cat_name)] = class_max_mAP

    return coco_eval, max_mAP, max_epoch, class_coco_evals, class_max_mAPs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', type=str)
    parser.add_argument('--plot_lr', action='store_true')
    parser.add_argument('--plot_class_map', action='store_true')
    parser.add_argument('--plot_max_prompt_norm', action='store_true')
    parser.add_argument('--pretrained_log_path', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)
    args = parser.parse_args()

    plot_map(
        log_path=args.log_path,
        plot_lr=args.plot_lr,
        plot_class_map=args.plot_class_map,
        plot_max_prompt_norm=args.plot_max_prompt_norm,
        pretrained_log_path=args.pretrained_log_path,
        start_epoch=args.start_epoch
    )
