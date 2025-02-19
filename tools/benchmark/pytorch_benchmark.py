"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import pycuda.driver as cuda
from utils import TimeProfiler
import numpy as np
import os
import time
import torch
from torch import nn

from collections import namedtuple, OrderedDict
import glob
import argparse
from dataset import Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from util.slconfig import SLConfig
from models.registry import MODULE_BUILD_FUNCS


def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser Example')
    parser.add_argument('--config', '-c', default='configs/linea/linea_l.py', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--infer_dir',
                        type=str,
                        default='./data/wireframe_processed/val2017',
                        help="Directory for images to perform inference on.")
    args = parser.parse_args()
    return args

def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


@torch.no_grad()
def warmup(model, data, img_size, n):
    for _ in range(n):
        _ = model(data, img_size)

@torch.no_grad()
def speed(model, data, n):
    times = []
    for i in tqdm(range(n), desc="Running Inference", unit="iteration"):
        blob = data[i]
        samples, target_sizes = blob['images'].unsqueeze(0), blob['orig_target_sizes'].unsqueeze(0)
        torch.cuda.synchronize()
        t_ = time.perf_counter()
        _ = model(samples, target_sizes)
        torch.cuda.synchronize()
        t = time.perf_counter() - t_
        times.append(t)

    # end-to-end model only
    times = sorted(times)
    if len(times) > 100:
        times = times[:100]
    return sum(times) / len(times)

def main():
    FLAGS = parse_args()
    dataset = Dataset(FLAGS.infer_dir)
    blob = torch.ones(1, 3, 640, 640).cuda()

    img_size = torch.tensor([640, 640], device='cuda')

    cfg = SLConfig.fromfile(FLAGS.config)

    if 'HGNetv2' in cfg.backbone:
        cfg.pretrained = False

    linea, _, postprocessor = build_model_main(cfg)

    if FLAGS.resume:
        checkpoint = torch.load(FLAGS.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        linea.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = linea.deploy()
            self.postprocessor = postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().cuda()
    
    warmup(model, blob, img_size, 400)
    t = []
    for _ in range(1):
        t.append(speed(model, dataset, 400))
    avg_latency = 1000 * torch.tensor(t).mean()
    print(f"model: {FLAGS.config}, Latency: {avg_latency:.2f} ms")

    del model
    torch.cuda.empty_cache()
    time.sleep(1)


if __name__ == '__main__':
    main()
