import argparse
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torch


def main(dir_path):
    dir_path = Path(dir_path)

    for domain in tqdm(['src', 'tgt'], desc='domain'):
        ckps = sorted([p for p in dir_path.iterdir() if domain in str(p) and p.suffix == '.pt'])
        
        fig = plt.figure()
        for ckp in tqdm(ckps, desc='ckp'):
            epoch = int(ckp.name.split('=')[-1].split('.')[0])
            query_idx_to_class_labels = torch.load(str(ckp), map_location='cpu')

            matched_query_count = {k: len(v) for k, v in query_idx_to_class_labels.items()}

            for idx, cls_labels in query_idx_to_class_labels.items():
                count = len(cls_labels)
                plt.scatter(epoch, idx, s=count, c='grey')

        plt.xlabel('Epoch')
        plt.ylabel('Object Query Index')
        save_path = str(dir_path / f'{domain}.png')
        plt.savefig(save_path)
        plt.close()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('query_idx_to_class_label_dir', type=str)
    args = parser.parse_args()

    main(args.query_idx_to_class_label_dir)
