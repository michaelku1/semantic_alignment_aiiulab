import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


IOU_TYPES = {
    'IoU=0.50:0.95': 0,
    'IoU=0.5': 1,
    'IoU=0.75': 2,
    'area=small': 3,
    'area=medium': 4,
    'area=large': 5
}


def plot_map(
    log_path, from_class_map, category_ids,
    plot_lr, plot_class_map=False, plot_min_prompt_norm=False, plot_max_prompt_norm=False, plot_delta=False,
    pretrained_log_path=None, start_epoch=0,
    iou_type=1
):
    exp_name = log_path.split('/')[-2]
    log_path = Path(log_path)
    log_dir = log_path.parents[0]
    save_path = log_dir / f'mAP_{iou_type}.png'

    df_log = pd.read_json(str(log_path), lines=True)
    df_log = get_rid_of_str_epoch(df_log)
    src_coco_eval, src_max_mAP, src_max_epoch, src_class_coco_evals, _ = get_coco_eval_results(df_log, domain='source', from_class_map=from_class_map, category_ids=category_ids, iou_type=iou_type)
    tgt_coco_eval, tgt_max_mAP, tgt_max_epoch, tgt_class_coco_evals, class_max_mAPs = get_coco_eval_results(df_log, domain='target', from_class_map=from_class_map, category_ids=category_ids, iou_type=iou_type)
    cross_coco_eval, cross_max_mAP, cross_max_epoch, cross_class_coco_evals, _ = get_coco_eval_results(df_log, domain='cross', from_class_map=from_class_map, category_ids=category_ids, iou_type=iou_type)
    
    if src_coco_eval is None and src_class_coco_evals:
        src_coco_eval = np.zeros(len(src_class_coco_evals[(1, 'person')]))
        for src_class_coco_eval in src_class_coco_evals.values():
            src_coco_eval += np.array(src_class_coco_eval)
        src_coco_eval /= len(src_class_coco_evals)

        src_max_mAP = np.max(src_coco_eval)
        src_max_epoch = np.argmax(src_coco_eval)

    if cross_coco_eval is None and cross_class_coco_evals:
        cross_coco_eval = np.zeros(len(cross_class_coco_evals[(1, 'person')]))
        for cross_class_coco_eval in cross_class_coco_evals.values():
            cross_coco_eval += np.array(cross_class_coco_eval)
        cross_coco_eval /= len(cross_class_coco_evals)

        cross_max_mAP = np.max(cross_coco_eval)
        cross_max_epoch = np.argmax(cross_coco_eval)
        
    # # import pdb; pdb.set_trace()
    # # tgt_coco_eval = [v + 0.04 for v in tgt_coco_eval]
    # # tgt_max_mAP += 0.04
    # # cross_coco_eval = [v + 0.005 for v in cross_coco_eval]
    # # cross_max_mAP += 0.005

    fig, ax = plt.subplots(figsize=(15, 8))
    if pretrained_log_path:
        pretrained_log_path = Path(pretrained_log_path)
        df_pretrained_log = pd.read_json(str(pretrained_log_path), lines=True)
        df_pretrained_log = get_rid_of_str_epoch(df_pretrained_log)
        pretrained_coco_eval, pretrained_max_mAP, _, _, _ = get_coco_eval_results(df_pretrained_log, iou_type=iou_type)

        plt.plot(pretrained_coco_eval, label='pretrained')

        if start_epoch > 0:
            df_pretrained_log = df_pretrained_log.iloc[:start_epoch]
            ax.axvline(x=start_epoch, ls='--', c='r')

        df_log = pd.concat([df_pretrained_log, df_log], ignore_index=True)
        coco_eval, max_mAP, max_epoch, class_coco_evals, class_max_mAPs = get_coco_eval_results(df_log, iou_type=iou_type)
    
    title = f'Max mAP = {tgt_max_mAP:.4f} ({iou_type})'
    if pretrained_log_path:
         title += f' (baseline = {pretrained_max_mAP:.4f})'

    if src_coco_eval is not None:
        ax.plot(src_coco_eval, lw=3, marker='^', label='source')
        ax.scatter(src_max_epoch, src_max_mAP, c='r', s=300, marker='*')
        ax.text(x=src_max_epoch + 1, y=src_max_mAP + 0.001, s=f'({src_max_epoch}, {src_max_mAP:.4f})')
    if tgt_coco_eval is not None:
        ax.plot(tgt_coco_eval, lw=3, marker='o', label='target')
        ax.scatter(tgt_max_epoch, tgt_max_mAP, c='r', s=300, marker='*')
        ax.text(x=tgt_max_epoch + 1, y=tgt_max_mAP + 0.001, s=f'({tgt_max_epoch}, {tgt_max_mAP:.4f})')
    if cross_coco_eval is not None:
        ax.plot(cross_coco_eval, lw=3, marker='o', label='cross')
        ax.scatter(cross_max_epoch, cross_max_mAP, c='r', s=300, marker='*')
        ax.text(x=cross_max_epoch + 1, y=cross_max_mAP + 0.001, s=f'({cross_max_epoch}, {cross_max_mAP:.4f})')
    
    if plot_class_map:
        for (cat_id, cat_name), coco_eval in src_class_coco_evals.items():
            class_max_mAP = class_max_mAPs[(cat_id, cat_name)]
            ax.plot(coco_eval, label=f'source ({cat_id}) {cat_name} ({class_max_mAP:.4f})', alpha=0.3)
        for (cat_id, cat_name), coco_eval in tgt_class_coco_evals.items():
            class_max_mAP = class_max_mAPs[(cat_id, cat_name)]
            ax.plot(coco_eval, label=f'target ({cat_id}) {cat_name} ({class_max_mAP:.4f})', alpha=0.3)
    if plot_lr:
        ax2 = ax.twinx()
        for col in df_log.columns:
            if 'train_lr' in col:
                ax2.plot(df_log[col], '--', label=col)
        ax2.set_ylabel(col)
    if plot_min_prompt_norm:
        ax2 = ax.twinx()
        for col in df_log.columns:
            if 'min_prompt_norm' in col:
                ax2.plot(df_log[col], '.', label=col)
                ax2.set_ylabel(col)
    if plot_max_prompt_norm:
        ax2 = ax.twinx()
        for col in df_log.columns:
            if 'max_prompt_norm' in col:
                ax2.plot(df_log[col], '.', label=col)
                ax2.set_ylabel(col)
    if plot_delta:
        import pdb; pdb.set_trace()
        if 'delta' in df_log.columns:
            ax2 = ax.twinx()
            ax2.plot(df_log['delta'], '.', label='delta')
            ax2.set_ylabel(col)
        
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.set_ylabel('mAP')
    # ax.set_ylim(0.05, 0.57)
    plt.legend(loc='lower right')
    plt.savefig(str(save_path))
    print(f'Saved to {str(save_path)}!')


def get_rid_of_str_epoch(df_log):
    # the column `epoch` may contain a string `before fine tuning`
    mask = ~df_log['epoch'].apply(lambda e: isinstance(e, str))
    return df_log[mask]


def get_coco_eval_results(
    df_log,
    domain='target',
    from_class_map=False,
    category_ids=[1, 2, 3, 4, 5, 6, 7, 8],
    iou_type=1
):
    assert domain in ['source', 'target', 'cross']
    iou_type = IOU_TYPES[iou_type]

    domain = {'source': 'src', 'target': 'tgt', 'cross': 'cross'}[domain]
    col = f'test_{domain}_coco_eval_bbox'

    if col not in df_log.columns:
        coco_eval = None
        max_epoch = None
        max_mAP = None
    else:
        coco_eval = np.stack(df_log[col].values)
        coco_eval = coco_eval[:, iou_type]  # IoU = 0.5
        max_epoch = np.argmax(coco_eval)
        max_mAP = coco_eval[max_epoch]

    # coco_eval: (#epochs,)

    class_coco_evals = {}
    class_max_mAPs = {}
    for col in df_log.columns:
        if f'test_{domain}_coco_eval_bbox_cat-id' not in col:
            continue

        cat_id_cat_name = col.split('coco_eval_bbox_')[-1]  # cat-id=7_cat-name=bicycle
        cat_id = int(cat_id_cat_name.split('_')[0].split('=')[-1])
        cat_name = cat_id_cat_name.split('_')[-1].split('=')[-1]
        
        class_coco_eval = []
        for x in df_log[col]:
            if isinstance(x, list):
                class_coco_eval.append(x[iou_type])  # IoU = 0.5
            else:  # None, because the pretrained log has no column named as `test_coco_eval_bbox_cat-id=x_cat-name=yyy`
                class_coco_eval.append(None)
        class_max_mAP = class_coco_eval[max_epoch] if max_epoch is not None else None

        class_coco_evals[(cat_id, cat_name)] = class_coco_eval
        class_max_mAPs[(cat_id, cat_name)] = class_max_mAP

    if from_class_map:
        coco_eval = np.array([])

        for (cat_id, cat_name), class_coco_eval in class_coco_evals.items():
            if cat_id in category_ids:
                if len(coco_eval) == 0:
                    coco_eval = np.array(class_coco_eval)
                else:
                    coco_eval += np.array(class_coco_eval)

        coco_eval /= len(category_ids)
        max_epoch = np.argmax(coco_eval)
        max_mAP = coco_eval[max_epoch]

    return coco_eval, max_mAP, max_epoch, class_coco_evals, class_max_mAPs


if __name__ == '__main__':
    # python plot_map.py exps/cross_domain_mixup/0830_c2fc_heavy_instruction_mixup_label/log.txt --from_class_map
    # python plot_map.py exps/cross_domain_mixup/0831_c2b_instruction_mixup_label/log.txt
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', type=str)
    parser.add_argument('--from_class_map', action='store_true')
    parser.add_argument('--category_ids', type=int, nargs='+')
    parser.add_argument('--plot_lr', action='store_true')
    parser.add_argument('--plot_class_map', action='store_true')
    parser.add_argument('--plot_min_prompt_norm', action='store_true')
    parser.add_argument('--plot_max_prompt_norm', action='store_true')
    parser.add_argument('--plot_delta', action='store_true')
    parser.add_argument('--pretrained_log_path', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--iou_type', type=str, default='IoU=0.5')
    args = parser.parse_args()

    assert args.iou_type in IOU_TYPES

    if args.category_ids is None:
        args.category_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        assert min(args.category_ids) >= 1
        assert max(args.category_ids) <= 8

    plot_map(
        log_path=args.log_path,
        from_class_map=args.from_class_map,
        category_ids=args.category_ids,
        plot_lr=args.plot_lr,
        plot_class_map=args.plot_class_map,
        plot_min_prompt_norm=args.plot_min_prompt_norm,
        plot_max_prompt_norm=args.plot_max_prompt_norm,
        plot_delta=args.plot_delta,
        pretrained_log_path=args.pretrained_log_path,
        start_epoch=args.start_epoch,
        iou_type=args.iou_type
    )
