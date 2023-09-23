import re
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path


def main(src_img_dir, src_gt_img_dir, tgt_img_dir, tgt_gt_img_dir, cross_img_dir, save_img_dir):
    src_img_dir = Path(src_img_dir)
    src_gt_img_dir = Path(src_gt_img_dir)
    tgt_img_dir = Path(tgt_img_dir)
    tgt_gt_img_dir = Path(tgt_gt_img_dir)
    cross_img_dir = Path(cross_img_dir)
    save_img_dir = Path(save_img_dir)

    assert src_img_dir.exists()
    assert src_gt_img_dir.exists()
    assert tgt_img_dir.exists()
    assert tgt_gt_img_dir.exists()
    assert cross_img_dir.exists()
    save_img_dir.mkdir(parents=True, exist_ok=True)

    src_img_paths = [p for p in src_img_dir.iterdir() if '.png' in p.suffix]
    src_gt_img_paths = [p for p in src_gt_img_dir.iterdir() if '.png' in p.suffix]
    tgt_img_paths = [p for p in tgt_img_dir.iterdir() if '.png' in p.suffix]
    tgt_gt_img_paths = [p for p in tgt_gt_img_dir.iterdir() if '.png' in p.suffix]
    cross_img_paths = [p for p in cross_img_dir.iterdir() if '.png' in p.suffix]

    assert len(src_img_paths) == len(src_gt_img_paths) == len(tgt_img_paths) == len(tgt_gt_img_paths) == len(cross_img_paths), f'#src={len(src_img_paths)}, #src_gt={len(src_gt_img_paths)}, #tgt={len(tgt_img_paths)}, #tgt_gt={len(tgt_gt_img_paths)}, #cross={len(cross_img_paths)}'

    for src_img_path in tqdm(src_img_paths, desc='combining...'):
        img_id_str = re.search('id=\d+', str(src_img_path)).group()  # 'id=xxx'
        try:
            img_id = img_id_str.split('id=')[-1]
            img_id = int(img_id)
        except:
            raise ValueError(f'Wrong img path {str(src_img_path)}')

        def match_id(p):
            _img_id_str = re.search('id=\d+', str(p)).group()
            _img_id = _img_id_str.split('id=')[-1]
            _img_id = int(_img_id)

            return img_id == _img_id

        src_gt_img_path = list(filter(match_id, src_gt_img_paths))
        tgt_img_path = list(filter(match_id, tgt_img_paths))
        tgt_gt_img_path = list(filter(match_id, tgt_gt_img_paths))
        cross_img_path = list(filter(match_id, cross_img_paths))
        
        assert len(src_gt_img_path) == 1, src_gt_img_path
        assert len(tgt_img_path) == 1, tgt_img_path
        assert len(tgt_gt_img_path) == 1, tgt_gt_img_path
        assert len(cross_img_path) == 1, cross_img_path

        src_gt_img_path = src_gt_img_path[0]
        tgt_img_path = tgt_img_path[0]
        tgt_gt_img_path = tgt_gt_img_path[0]
        cross_img_path = cross_img_path[0]
        
        src_img = cv2.imread(str(src_img_path))
        src_gt_img = cv2.imread(str(src_gt_img_path))
        tgt_img = cv2.imread(str(tgt_img_path))
        tgt_gt_img = cv2.imread(str(tgt_gt_img_path))
        cross_img = cv2.imread(str(cross_img_path))

        h, w, c = src_img.shape
        combined_img = np.zeros((h * 5, w, c))

        imgs = [src_gt_img, src_img, tgt_gt_img, tgt_img, cross_img]
        titles = ['SOURCE GT', 'SOURCE', 'TARGET GT', 'TARGET', 'CROSS']
        for i, (img, title) in enumerate(zip(imgs, titles)):
            start = i * h
            end = (i+1) * h
            combined_img[start:end, ...] = img

            cv2.putText(combined_img, title, (20, end-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        save_path = save_img_dir / f'id={img_id}.png'
        cv2.imwrite(str(save_path), combined_img)


if __name__ == '__main__':
    """
    python combine_gt_and_predicted_val_bbox.py
        --src_img_dir exps/cross_domain/0808_cyclegan_unpaired_fine_tune/plot_bbox/eval_src_epoch=19
        --tgt_img_dir exps/cross_domain/0808_cyclegan_unpaired_fine_tune/plot_bbox/eval_tgt_epoch=19
        --cross_img_dir exps/cross_domain/0808_cyclegan_unpaired_fine_tune/plot_bbox/eval_cross_epoch=19
        --save_img_dir exps/cross_domain/0808_cyclegan_unpaired_fine_tune/plot_bbox/combined
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_img_dir', type=str, default=None)
    parser.add_argument('--src_gt_img_dir', type=str, default='gt_val_bbox/val_src')
    parser.add_argument('--tgt_img_dir', type=str, default=None)
    parser.add_argument('--tgt_gt_img_dir', type=str, default='gt_val_bbox/val_tgt')
    parser.add_argument('--cross_img_dir', type=str, default=None)
    parser.add_argument('--save_img_dir', type=str, default=None)
    args = parser.parse_args()

    assert args.src_img_dir is not None
    assert args.src_gt_img_dir is not None
    assert args.tgt_img_dir is not None
    assert args.tgt_gt_img_dir is not None
    assert args.cross_img_dir is not None
    assert args.save_img_dir is not None

    main(
        src_img_dir=args.src_img_dir,
        src_gt_img_dir=args.src_gt_img_dir,
        tgt_img_dir=args.tgt_img_dir,
        tgt_gt_img_dir=args.tgt_gt_img_dir,
        cross_img_dir=args.cross_img_dir,
        save_img_dir=args.save_img_dir
    )