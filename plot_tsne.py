import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
from torch.utils.data import DataLoader

from config import get_cfg_defaults
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
import util.misc as utils


def setup(args):
    cfg = get_cfg_defaults()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    utils.init_distributed_mode(cfg)
    cfg.freeze()

    return cfg


def main(cfg, emb_dir, feat_from):
    print(cfg)

    if cfg.RESUME is not None:
        if emb_dir is not None:
            raise ValueError('Either ckp_path or emb_dir is given')
    else:
        if emb_dir is None:
            raise ValueError('Either ckp_path or emb_dir is given')

    if cfg.RESUME is not None and emb_dir is None:
        print()
        print('Compute embedding')
        
        get_embeddings_from_model(cfg)

        print('Done. Please rerun to reload embeddings')
        exit()
    
    elif emb_dir is not None:
        print()
        print('Load embeddings')
        domain_embeddings = load_embeddings(emb_dir, feat_from)
        print('Loaded')

        assert isinstance(domain_embeddings['src'], torch.Tensor), type(domain_embeddings['src'])
        assert isinstance(domain_embeddings['tgt'], torch.Tensor), type(domain_embeddings['tgt'])
        assert isinstance(domain_embeddings['int'], torch.Tensor), type(domain_embeddings['int'])

        print('Start TSNE')
        plot_tsne(domain_embeddings, emb_dir, feat_from)
        print('Done')


def get_embeddings_from_model(cfg):
    ckp_path = Path(cfg.RESUME)
    if not ckp_path.exists():
        raise FileNotFoundError()

    device = torch.device(cfg.DEVICE)

    # fix the seed for reproducibility
    seed = cfg.SEED + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    dataset_val_src = build_dataset(image_set='val_source', cfg=cfg)
    dataset_val_tgt = build_dataset(image_set='val_target', cfg=cfg)
    dataset_val_int = build_dataset(image_set='val_cross', cfg=cfg)

    if cfg.DIST.DISTRIBUTED:
        if cfg.CACHE_MODE:
            sampler_val_src = samplers.NodeDistributedSampler(dataset_val_src, shuffle=False)
            sampler_val_tgt = samplers.NodeDistributedSampler(dataset_val_tgt, shuffle=False)
            sampler_val_tgt_like_src = samplers.NodeDistributedSampler(dataset_val_int, shuffle=False)
        else:
            sampler_val_src = samplers.NodeDistributedSampler(dataset_val_src, shuffle=False)
            sampler_val_tgt = samplers.DistributedSampler(dataset_val_tgt, shuffle=False)
            sampler_val_tgt_like_src = samplers.DistributedSampler(dataset_val_int, shuffle=False)
    else:
        sampler_val_src = torch.utils.data.SequentialSampler(dataset_val_src, shuffle=False)
        sampler_val_tgt = torch.utils.data.SequentialSampler(dataset_val_tgt, shuffle=False)
        sampler_val_tgt_like_src = torch.utils.data.SequentialSampler(dataset_val_int, shuffle=False)

    data_loader_val_src = DataLoader(dataset_val_src, cfg.TRAIN.BATCH_SIZE, sampler=sampler_val_src,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                     pin_memory=True)
    data_loader_val_tgt = DataLoader(dataset_val_tgt, cfg.TRAIN.BATCH_SIZE, sampler=sampler_val_tgt,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                     pin_memory=True)
    data_loader_val_int = DataLoader(dataset_val_int, cfg.TRAIN.BATCH_SIZE, sampler=sampler_val_tgt_like_src,
                                       drop_last=False, collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                       pin_memory=True)

    model, epoch = load_model(cfg, ckp_path, device)
    
    src_backbone_embeddings, src_memory_embeddings, src_hs_embeddings = collect_embeddings(model, data_loader_val_src, device)
    save_embeddings(src_backbone_embeddings, src_memory_embeddings, src_hs_embeddings, domain='src', ckp_path=cfg.RESUME, epoch=epoch)
    del src_backbone_embeddings, src_memory_embeddings, src_hs_embeddings

    tgt_backbone_embeddings, tgt_memory_embeddings, tgt_hs_embeddings = collect_embeddings(model, data_loader_val_tgt, device)
    save_embeddings(tgt_backbone_embeddings, tgt_memory_embeddings, tgt_hs_embeddings, domain='tgt', ckp_path=cfg.RESUME, epoch=epoch)
    del tgt_backbone_embeddings, tgt_memory_embeddings, tgt_hs_embeddings

    int_backbone_embeddings, int_memory_embeddings, int_hs_embeddings = collect_embeddings(model, data_loader_val_int, device)
    save_embeddings(int_backbone_embeddings, int_memory_embeddings, int_hs_embeddings, domain='int', ckp_path=cfg.RESUME, epoch=epoch)
    del int_backbone_embeddings, int_memory_embeddings, int_hs_embeddings


def load_model(cfg, ckp_path, device):
    model, _, _ = build_model(cfg)
    model.to(device)
    model.eval()
    model_without_ddp = model

    checkpoint = torch.load(str(ckp_path), map_location='cpu')
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    print('The model is loaded')

    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 'released'

    return model, epoch


def collect_embeddings(model, data_loader, device):
    backbone_embeddings = {}
    memory_embeddings = []
    hs_embeddings = {}

    with torch.no_grad():
        for samples, _ in tqdm(data_loader, desc='collect embeddings'):
            samples = samples.to(device)
            _, features, memory, hs = model(samples, output_all=True)

            # If cfg.MODEL.NUM_FEATURE_LEVELS == 4
            #     features: [nested_tensor_0, nested_tensor_1, nested_tensor_2]
            #         features[0].tensors: (2,  512, 84, 167)
            #         features[1].tensors: (2, 1024, 42,  84)
            #         features[2].tensors: (2, 2048, 21,  42)
            # memory: (N, H_0W_0+...H_3W_3=18669, C), transformed feature map outputed from encoder
            # hs: (#decoder_layer, N, #query, d_model), object query tensor outputed from decoder

            for lvl, feat in enumerate(features):
                if lvl not in backbone_embeddings:
                    backbone_embeddings[lvl] = []
                backbone_embeddings[lvl].append(feat.tensors.detach().cpu())

            memory_embeddings.append(memory.detach().cpu())

            for lvl, feat in enumerate(hs):
                if lvl not in hs_embeddings:
                    hs_embeddings[lvl] = []
                hs_embeddings[lvl].append(feat.detach().cpu())

        for lvl, feat_list in backbone_embeddings.items():
            backbone_embeddings[lvl] = torch.cat(feat_list, dim=0)

        memory_embeddings = torch.cat(memory_embeddings, dim=0)

        for lvl, feat_list in hs_embeddings.items():
            hs_embeddings[lvl] = torch.cat(feat_list, dim=0)

    return backbone_embeddings, memory_embeddings, hs_embeddings


def save_embeddings(backbone_embeddings, encoder_embeddings, target_embeddings, domain, ckp_path, epoch):
    # backbone_embeddings,  # {lvl: (500, C, W, H)}
    # memory_embeddings,  # (500, 18669, 256)
    # hs_embeddings  # {lvl: (500, 300, 256)}

    exp_root = Path(ckp_path).parent
    if exp_root.name == 'checkpoints':
        exp_root = Path(exp_root).parent
    print(f'Exp root: {str(exp_root)}')

    embeddings_dir = exp_root / 'embeddings' / f'epoch={epoch}'
    if not embeddings_dir.exists():
        embeddings_dir.mkdir(exist_ok=True, parents=True)

    for lvl, embeddings in backbone_embeddings.items():
        torch.save(embeddings, str(embeddings_dir / f'{domain}_backbone_lvl={lvl}.pth'))

    torch.save(encoder_embeddings, str(embeddings_dir / f'{domain}_encoder.pth'))

    for lvl, embeddings in target_embeddings.items():
        torch.save(embeddings, str(embeddings_dir / f'{domain}_decoder_lvl={lvl}.pth'))

    print(f'Saved {domain} embeddings')


def load_embeddings(emb_dir, feat_from):
    emb_dir = Path(emb_dir)
    if not emb_dir.exists():
        raise FileNotFoundError('Can not find the embeddings')

    domain_embeddings = {
        'src': None,
        'tgt': None,
        'int': None
    }

    if 'backbone' in feat_from:
        target_lvl = int(feat_from.split('/')[-1])

        for p in emb_dir.iterdir():
            if p.suffix != '.pth':
                continue

            if 'backbone' not in str(p):
                continue

            file_name = p.name
            domain = str(file_name).split('_')[0]
            lvl = int(str(file_name).split('_')[-1].split('=')[-1].split('.')[0])

            if lvl != target_lvl:
                continue

            domain_embeddings[domain] = torch.load(p)
            print(f'    loaded {str(p)}')

    elif 'encoder' in feat_from:
        for p in emb_dir.iterdir():
            if p.suffix != '.pth':
                continue

            if 'encoder' not in str(p):
                continue

            file_name = p.name
            domain = str(file_name).split('_')[0]
            domain_embeddings[domain] = torch.load(p)
            print(f'    loaded {str(p)}')

    elif 'decoder' in feat_from:
        target_lvl = int(feat_from.split('/')[-1])

        for p in emb_dir.iterdir():
            if p.suffix != '.pth':
                continue

            if 'decoder' not in str(p):
                continue

            file_name = p.name
            domain = str(file_name).split('_')[0]
            lvl = int(str(file_name).split('_')[-1].split('=')[-1].split('.')[0])

            if lvl != target_lvl:
                continue

            domain_embeddings[domain] = torch.load(p).flatten(start_dim=0, end_dim=1)
            print(f'    loaded {str(p)}')

    else:
        raise ValueError(f'Unknown feat_from={feat_from}')

    return domain_embeddings


def plot_tsne(domain_embeddings, emb_dir, feat_from):
    domain_tsne_embeddings = {}

    num_data = len(domain_embeddings['src'])
    assert num_data == len(domain_embeddings['tgt']) == len(domain_embeddings['int']), f"#src={num_data}, #tgt={len(domain_embeddings['tgt'])}, #int={len(domain_embeddings['int'])}"
    
    all_embeddings = [domain_embeddings['src'], domain_embeddings['tgt'], domain_embeddings['int']]
    import pdb; pdb.set_trace()
    all_embeddings = torch.cat(all_embeddings, dim=0).flatten(start_dim=1).numpy()  # (1500, x)

    transformed_all_embeddings = TSNE(n_components=2, perplexity=5, init='random', n_iter=1000, random_state=42, verbose=1).fit_transform(all_embeddings)
    transformed_src_embeddings = transformed_all_embeddings[:num_data, :]
    transformed_tgt_embeddings = transformed_all_embeddings[num_data:num_data * 2, :]
    transformed_int_embeddings = transformed_all_embeddings[num_data * 2:, :]

    plt.scatter(x=transformed_src_embeddings[:, 0], y=transformed_src_embeddings[:, 1], label='Source Domain')
    plt.scatter(x=transformed_tgt_embeddings[:, 0], y=transformed_tgt_embeddings[:, 1], label='Target Domain')
    plt.scatter(x=transformed_int_embeddings[:, 0], y=transformed_int_embeddings[:, 1], label='Interpolated Domain')
    plt.axis('off')
    plt.legend()
    
    if '/' in feat_from:  # backbone/2 or decoder[5]
        feat_from = feat_from.replace('/', '_lvl=')
    
    save_path = Path(emb_dir) / feat_from
    plt.savefig(str(save_path))
    print(f'Plotted. Saved image to {str(save_path)}')


def check_feat_from(feat_from):
    if 'backbone' in feat_from:
        module, lvl = feat_from.split('/')

        assert module == 'backbone'
        try:
            int(lvl)
        except:
            raise RunTimeError(f'Wrong feat_from={feat_from}')

    elif 'encoder' in feat_from:
        assert feat_from == 'encoder', feat_from
    
    elif 'decoder' in feat_from:
        module, lvl = feat_from.split('/')

        assert module == 'decoder'
        try:
            int(lvl)
        except:
            raise RunTimeError(f'Wrong feat_from={feat_from}')
    

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 --master_port 29100
    # python plot_tsne.py
    # --config_file released_weights/c2fc/r50_uda_c2fc.yaml
    # --opts
    #     DATASET.DA_MODE uda_mixup0.7_heavy_instruction_target
    #     DATASET.MIXUP.SWITCH True
    #     RESUME released_weights/c2fc/cityscapes_to_foggy_cityscapes.pth

    # CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 --master_port 29100 python plot_tsne.py --config_file exps/source_only/0614/config.yaml --feat_from decoder/5 --opts DATASET.DA_MODE uda_mixup0.7_heavy_instruction_target DATASET.MIXUP.SWITCH True RESUME exps/source_only/0614/checkpoints/checkpoint0043.pth

    # CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 --master_port 29100
    # python plot_tsne.py
    # --config_file released_weights/c2fc/r50_uda_c2fc.yaml
    # --emb_dir released_weights/c2fc/embeddings/epoch=released
    # --feat_from decoder/5

    # CUDA_VISIBLE_DEVICES=0 GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 --master_port 29100 python plot_tsne.py --config_file exps/source_only/0614/config.yaml --emb_dir exps/source_only/0614/embeddings/epoch=43 --feat_from decoder/5

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='')
    parser.add_argument('--emb_dir', type=str, default=None)
    parser.add_argument('--feat_from', type=str, default='backbone')
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    check_feat_from(args.feat_from)

    cfg = setup(args)
    main(cfg, args.emb_dir, args.feat_from)