import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def plot_map(log_path, class_map=False, pretrained_log_path=None, start_epoch=0):
    exp_name = log_path.split('/')[-2]
    log_path = Path(log_path)
    log_dir = log_path.parents[0]
    save_path = log_dir / 'mAP.png'

    df_log = pd.read_json(str(log_path), lines=True)
    coco_eval, max_mAP, max_epoch, class_coco_evals, class_max_mAPs = get_coco_eval_results(df_log)

    plt.figure(figsize=(15, 8))
    if pretrained_log_path:
        pretrained_log_path = Path(pretrained_log_path)
        df_pretrained_log = pd.read_json(str(pretrained_log_path), lines=True)
        pretrained_coco_eval, pretrained_max_mAP, _, _ = get_coco_eval_results(df_pretrained_log)

        plt.plot(pretrained_coco_eval, label='pretrained')

        if start_epoch > 0:
            df_pretrained_log = df_pretrained_log.iloc[:start_epoch]
            plt.axvline(x=start_epoch, ls='--', c='r')

        df_log = pd.concat([df_pretrained_log, df_log], ignore_index=True)
        coco_eval, max_mAP, max_epoch, class_coco_evals, class_max_mAPs = get_coco_eval_results(df_log)
    
    title = f'Max mAP = {max_mAP:.4f} ({max_epoch}-th epoch)'
    if pretrained_log_path:
         title += f' (baseline = {pretrained_max_mAP:.4f})'

    plt.plot(coco_eval, label=exp_name)
    if class_map:
        for (cat_id, cat_name), coco_eval in class_coco_evals.items():
            class_max_mAP = class_max_mAPs[(cat_id, cat_name)]
            plt.plot(coco_eval, label=f'({cat_id}) {cat_name} ({class_max_mAP:.4f})', alpha=0.3)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('mAP')
    plt.legend(loc='lower right')
    plt.savefig(str(save_path))
    print(f'Saved to {str(save_path)}!')


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
    parser.add_argument('class_map', action='store_true')
    parser.add_argument('--pretrained_log_path', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)
    args = parser.parse_args()

    plot_map(
        log_path=args.log_path,
        class_map=args.class_map,
        pretrained_log_path=args.pretrained_log_path,
        start_epoch=args.start_epoch
    )