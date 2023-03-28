import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def plot_map(log_path):
    log_path = Path(log_path)
    log_dir = log_path.parents[0]
    save_path = log_dir / 'mAP.png'

    df_log = pd.read_json(str(log_path), lines=True)
    coco_eval = pd.DataFrame(np.stack(df_log['test_coco_eval_bbox'].dropna().values)[:, 1]).ewm(com=0).mean()
    max_mAP = np.stack(df_log['test_coco_eval_bbox'].dropna().values)[:, 1].max()

    plt.plot(coco_eval)
    plt.title(f'Max mAP = {max_mAP:.4f}')
    plt.xlabel('epoch')
    plt.ylabel('mAP')
    plt.savefig(str(save_path))
    print(f'Saved to {str(save_path)}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', type=str)
    args = parser.parse_args()
    plot_map(log_path=args.log_path)
