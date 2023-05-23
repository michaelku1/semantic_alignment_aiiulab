import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def plot_loss(log_path, loss_names=[], pretrained_log_path=None, start_epoch=0):
    exp_name = log_path.split('/')[-2]
    log_path = Path(log_path)
    log_dir = log_path.parents[0]
    save_path = log_dir / 'loss.png'

    df_log = pd.read_json(str(log_path), lines=True)
    df_log = get_rid_of_str_epoch(df_log)

    train_loss_names = []
    test_loss_names = []
    if len(loss_names) == 0:
        for col in df_log.columns:
            if 'loss' not in col:
                continue
            
            if 'train' in col:
                train_loss_names.append(col)
            elif 'test' in col:
                test_loss_names.append(col)
    else:
        for loss_name in loss_names:
            if loss_name.startswith('train'):
                train_loss_names.append(loss_name)
            elif loss_name.startswith('test'):
                test_loss_names.append(loss_name)
            else:
                train_loss_names.append(f'train_{loss_name}')
                test_loss_names.append(f'test_{loss_name}')
    check_loss_names(df_log, train_loss_names)
    check_loss_names(df_log, test_loss_names)

    fig, ax = plt.subplots(figsize=(15, 8))
    
    if pretrained_log_path:
        pretrained_log_path = Path(pretrained_log_path)
        df_pretrained_log = pd.read_json(str(pretrained_log_path), lines=True)
        df_pretrained_log = get_rid_of_str_epoch(df_pretrained_log)

        for train_loss_name in train_loss_names:
            ax.plot(df_pretrained_log[train_loss_name], label=f'pretrained_{train_loss_name}')
        for test_loss_name in test_loss_names:
            ax.plot(df_pretrained_log[test_loss_name], label=f'pretrained_{test_loss_name}')

        if start_epoch > 0:
            df_pretrained_log = df_pretrained_log.iloc[:start_epoch]
            ax.axvline(x=start_epoch, ls='--', c='r')

        df_log = pd.concat([df_pretrained_log, df_log], ignore_index=True)

    for train_loss_name in train_loss_names:
        ax.plot(df_log[train_loss_name], label=train_loss_name)
    for test_loss_name in test_loss_names:
        ax.plot(df_log[test_loss_name], label=test_loss_name)
    
    ax.set_xlabel('epoch')
    ax.set_ylabel('Loss')
    plt.legend(loc='lower right')
    plt.savefig(str(save_path))
    print(f'Saved to {str(save_path)}!')


def get_rid_of_str_epoch(df_log):
    # the column `epoch` may contain a string `before fine tuning`
    mask = ~df_log['epoch'].apply(lambda e: isinstance(e, str))
    return df_log[mask]


def check_loss_names(df_log, loss_names=[]):
    if len(loss_names) == 0:
        return

    for loss_name in loss_names:
        if loss_name not in df_log.columns:
            raise KeyError(f'The loss name `{loss_name}` can be found in the log')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', type=str)
    parser.add_argument('--loss_names', nargs='+', default=[])
    parser.add_argument('--pretrained_log_path', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)
    args = parser.parse_args()

    plot_loss(
        log_path=args.log_path,
        loss_names=args.loss_names,
        pretrained_log_path=args.pretrained_log_path,
        start_epoch=args.start_epoch
    )
