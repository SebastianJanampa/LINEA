# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch
from torch.utils.data import DataLoader

from util.slconfig import SLConfig

import datasets
from datasets import build_dataset, BatchImageCollateFunction


def create(args, classname):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    class_module = getattr(args, classname)
    assert class_module in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(class_module)
    return build_func(args)

def main(args):
    cfg = SLConfig.fromfile(args.config)
    device = args.device

    setattr(cfg, 'coco_path', args.data_path)
    setattr(cfg, 'batch_size_train', 1)
    setattr(cfg, 'batch_size_val', 1)

    if 'HGNetv2' in cfg.backbone:
        cfg.pretrained = False

    # build model
    model, _ = create(cfg, 'modelname')
    model.to(device)
    
    criterion = create(cfg, 'criterionname')

    dataset_val = build_dataset(image_set='val', args=cfg)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False, collate_fn=BatchImageCollateFunction(), num_workers=4)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        model.load_state_dict(state)
        
    # change to device
    model.to(device)

    # transformer parameters
    len_q = cfg.num_queries
    nheads = cfg.nheads
    num_sampling_points = cfg.dec_n_points
    num_points_scale = torch.tensor([1/n for n in num_sampling_points for _ in range(n)], dtype=torch.float32).reshape(-1, 1)

    # folder path
    main_folder = cfg.output_dir
    if 'data/wireframe_processed' in args.data_path:
        append_path = f'{main_folder}/visualization/line_attention_wireframe'

    elif 'data/york_processed' in args.data_path:
        append_path = f'{main_folder}/visualization/line_attention_york'
    os.makedirs(append_path , exist_ok=True)

    with torch.no_grad():

        for i, (samples, targets) in enumerate(data_loader_val):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            sampling_ratios = []
            reference_points = []
            attention_weights = []
            hooks = [
                model.decoder.decoder.layers[-1].cross_attn.sampling_ratios.register_forward_hook(
                    lambda self, input, output: sampling_ratios.append(output[0])
                ),
                model.decoder.decoder.layers[-1].cross_attn.attention_weights.register_forward_hook(
                    lambda self, input, output: attention_weights.append(output[0])
                ),
                model.decoder.decoder.register_forward_hook(
                    lambda self, input, output: reference_points.append(output[0])
                ),
            ]

            output = model(samples, None)

            [(src_idx, tgt_idx)] = criterion(output, targets, return_indices=True)
            
            for hook in hooks:
                hook.remove()    
 
            sampling_ratios = sampling_ratios[0].cpu().view(1, len_q, nheads, sum(num_sampling_points), 1)
            attention_weights = attention_weights[0].cpu().view(1, len_q, nheads, sum(num_sampling_points))
            attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)

            reference_points = reference_points[0][-2:-1].cpu().transpose(1, 2)

            vector = reference_points[:, :, None, :, :2] - reference_points[:, :, None, :, 2:]
            center = 0.5 * (reference_points[:, :, None, :, :2] + reference_points[:, :, None, :, 2:])

            sampling_locations = center + sampling_ratios * num_points_scale * vector * 0.5

            # Plot image
            img = samples[0].permute(1, 2, 0).cpu()
            img = (img - img.min()) / (img.max() - img.min())
            fig, ax = plt.subplots()
            ax.imshow(img, extent=[0, 1, 1, 0])

            reference_points = reference_points.transpose(1, 2)[0, 0]
            sampling_locations = sampling_locations[0]
            attention_weights = attention_weights[0]

            # choose the query idx
            line_idx = src_idx[tgt_idx == 0][0]
            reference_points = reference_points[line_idx]
            sampling_locations = sampling_locations[line_idx]
            attention_weights = attention_weights[line_idx]

            # sampling points
            for j in range(nheads):
                x1, y1 = sampling_locations[j].split(1, dim=-1)
                pos = ax.scatter(x1, y1, marker='*', c=attention_weights[j], cmap='jet', zorder=2)
            cbar = fig.colorbar(pos, ax=ax)
            cbar.ax.tick_params(size=0)
            cbar.set_ticks([])

            # reference lines
            x1, y1, x2, y2 = reference_points.split(1, dim=-1)
            ax.plot((x1[0], x2[0]), (y1[0], y2[0]), c='k', marker='o', zorder=3)

            plt.axis([0, 1, 1, 0])
            plt.axis(False)


            curr_img_id = targets[0]['image_id'].tolist()[0]
            plt.savefig(f'{append_path}/{curr_img_id}.png', bbox_inches="tight", pad_inches=0.0, dpi=100)
            plt.close()

            # check condition to stop program
            if args.num_images is not None and i + 1 >= args.num_images:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualization of Deformable Line Attention')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', default='', help='resume from checkpoint')
    parser.add_argument('-p', '--data-path', type=str, default='data/wireframe_processed', help='data path')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-n', '--num_images', type=int, help='total number of images to plot')
    args = parser.parse_args()
    main(args)
