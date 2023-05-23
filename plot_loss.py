import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def plot_loss(log_path, loss_name=None, pretrained_log_path=None, start_epoch=0):
    exp_name = log_path.split('/')[-2]
    log_path = Path(log_path)
    log_dir = log_path.parents[0]
    save_path = log_dir / 'loss.png'

    df_log = pd.read_json(str(log_path), lines=True)
    df_log = get_rid_of_str_epoch(df_log)

    train_loss_name = '_'.join(['train', loss_name])
    test_loss_name = '_'.join(['test', loss_name])
    check_loss_name(df_log, train_loss_name)
    check_loss_name(df_log, test_loss_name)

    fig, ax = plt.subplots(figsize=(15, 8))
    
    if pretrained_log_path:
        pretrained_log_path = Path(pretrained_log_path)
        df_pretrained_log = pd.read_json(str(pretrained_log_path), lines=True)
        df_pretrained_log = get_rid_of_str_epoch(df_pretrained_log)

        ax.plot(df_pretrained_log[train_loss_name], label=f'pretrained_{train_loss_name}')
        ax.plot(df_pretrained_log[test_loss_name], label=f'pretrained_{test_loss_name}')

        if start_epoch > 0:
            df_pretrained_log = df_pretrained_log.iloc[:start_epoch]
            ax.axvline(x=start_epoch, ls='--', c='r')

        df_log = pd.concat([df_pretrained_log, df_log], ignore_index=True)

    ax.plot(df_log[train_loss_name], label=train_loss_name)
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


def check_loss_name(df_log, loss_name=None):
    if loss_name is None:
        return

    if loss_name not in df_log.columns:
        raise KeyError(f'The loss name `{loss_name}` can be found in the log')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', type=str)
    parser.add_argument('--loss_name', type=str, default=None)
    parser.add_argument('--pretrained_log_path', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)
    args = parser.parse_args()

    plot_loss(
        log_path=args.log_path,
        loss_name=args.loss_name,
        pretrained_log_path=args.pretrained_log_path,
        start_epoch=args.start_epoch
    )
