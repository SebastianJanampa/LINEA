"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from util.slconfig import SLConfig
from models.registry import MODULE_BUILD_FUNCS

import argparse
from calflops import calculate_flops

import torch
import torch.nn as nn

def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

def main(args, ):
    """main
    """
    cfg = SLConfig.fromfile(args.config)
    
    if 'HGNetv2' in cfg.backbone:
        cfg.pretrained = False

    model, _, _= build_model_main(cfg)
    
    model = model.deploy()
    model.eval()

    flops, macs, _ = calculate_flops(model=model,
                                     input_shape=(1, 3, 640, 640),
                                     output_as_string=True,
                                     output_precision=4)
    params = sum(p.numel() for p in model.parameters())
    print("Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default= "configs/linea/linea_hgnetv2_lpy", type=str)
    args = parser.parse_args()

    main(args)
