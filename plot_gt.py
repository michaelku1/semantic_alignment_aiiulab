import argparse
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from pycocotools.coco import COCO


def plot_gt(cityscapes_dir, img_ids, save_dir):
    cityscapes_dir = Path(cityscapes_dir)
    assert cityscapes_dir.exists()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ann_files=[
        cityscapes_dir / 'annotations/cityscapes_train.json',
        cityscapes_dir / 'annotations/cityscapes_val.json',
    ]
    foggy_ann_files=[
        cityscapes_dir / 'annotations/foggy_cityscapes_train.json',
        cityscapes_dir / 'annotations/foggy_cityscapes_val.json',
    ]

    cocos = [COCO(ann_file) for ann_file in ann_files]
    foggy_cocos = [COCO(ann_file) for ann_file in foggy_ann_files]

    check_img_ids(img_ids, cocos=cocos+foggy_cocos)
    
    if (cityscapes_dir / 'leftImg8bit').exists():
        for img_id in tqdm(img_ids, desc='plot source images...'):
            for coco, ann_file in zip(cocos, ann_files):
                if img_id not in coco.getImgIds():
                    continue

                img_info = coco.loadImgs(ids=[img_id])[0]
                if 'train' in str(ann_file):
                    img_path = cityscapes_dir / f'leftImg8bit/train/{img_info["file_name"]}'
                elif 'val' in str(ann_file):
                    img_path = cityscapes_dir / f'leftImg8bit/val/{img_info["file_name"]}'
                else:
                    raise RuntimeError()

                plt.cla()
                img = Image.open(str(img_path))
                plt.imshow(img)
                plt.show()

                ann_ids = coco.getAnnIds(imgIds=[img_id])
                anns = coco.loadAnns(ann_ids)
                coco.showAnns(anns, draw_bbox=True)
                plt.axis('off')
                
                save_path = save_dir / f'src_img_id={img_id}'
                plt.savefig(str(save_path), bbox_inches='tight')

    if (cityscapes_dir / 'leftImg8bit_foggy').exists():
        for img_id in tqdm(img_ids, desc='plot target images...'):
            for coco, ann_file in zip(foggy_cocos, foggy_ann_files):
                if img_id not in coco.getImgIds():
                    continue

                img_info = coco.loadImgs(ids=[img_id])[0]
                if 'train' in str(ann_file):
                    img_path = cityscapes_dir / f'leftImg8bit_foggy/train/{img_info["file_name"]}'
                elif 'val' in str(ann_file):
                    img_path = cityscapes_dir / f'leftImg8bit_foggy/val/{img_info["file_name"]}'
                else:
                    raise RuntimeError()

                plt.cla()
                img = Image.open(str(img_path))
                plt.imshow(img)
                plt.show()

                ann_ids = coco.getAnnIds(imgIds=[img_id])
                anns = coco.loadAnns(ann_ids)
                coco.showAnns(anns, draw_bbox=True)
                plt.axis('off')
                
                save_path = save_dir / f'tgt_img_id={img_id}'
                plt.savefig(str(save_path), bbox_inches='tight')


def check_img_ids(img_ids, cocos):
    all_img_ids = []
    for coco in cocos:
        all_img_ids += coco.getImgIds()

    for img_id in img_ids:
        if img_id not in all_img_ids:
            raise ValueError(f'There is no image with image id={img_id}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cityscapes_dir', type=str, default='/scratch2/users/william/datasets/cityscapes')
    parser.add_argument('save_dir', type=str, default='./gt_images')
    args = parser.parse_args()

    plot_gt(
        cityscapes_dir=args.cityscapes_dir,
        img_ids=[0, 1, 2, 3, 4, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 566, 2656],
        save_dir=args.save_dir
    )