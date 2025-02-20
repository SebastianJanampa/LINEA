# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from collections import Counter
import os
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_optim_params
from util.slconfig import DictAction, SLConfig
from util.profiler import stats
import util.misc as utils

from datasets import build_dataset, LineEvaluator, BatchImageCollateFunction
from engine import train_one_epoch, evaluate, test

from tensorboardX import SummaryWriter
from warmup import LinearWarmup

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--coco_path', type=str, default='data/wireframe_processed')
    # training parameters
    # parser.add_argument('--output_dir', default='',
    #                     help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--find_unused_params', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")

    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)

    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))
    # setup tensorboar writer
    if not args.eval:
        writer = SummaryWriter(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

    if args.eval:
        if 'HGNetv2' in args.backbone:
            args.pretrained = False

    # setup eval_spatial_size
    if isinstance(args.eval_spatial_size, int):
        size = args.eval_spatial_size 
        args.eval_spatial_size = [size, size]

    assert args.eval_spatial_size[0] == args.eval_spatial_size[1], 'We only support square shapes'
    device = torch.device(args.device)

    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_dicts = get_optim_params(args.model_parameters, model_without_ddp)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)

    if args.eval:
        dataset_val = build_dataset(image_set='val', args=args)
        if args.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = DataLoader(dataset_val, 64, sampler=sampler_val, drop_last=False, collate_fn=BatchImageCollateFunction(), num_workers=args.num_workers)
    else:
        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = build_dataset(image_set='val', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train, shuffle=True)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_train = DataLoader(dataset_train, 
                                        args.batch_size_train, 
                                        sampler=sampler_train, 
                                        drop_last=True,
                                        collate_fn=BatchImageCollateFunction(base_size=args.eval_spatial_size[0], base_size_repeat=3), 
                                        # pin_memory=dataset_train.pin_memory,
                                        num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 
                                        args.batch_size_val, 
                                        sampler=sampler_val, 
                                        drop_last=False,
                                        collate_fn=BatchImageCollateFunction(base_size=args.eval_spatial_size[0]), 
                                        # pin_memory=dataset_val.pin_memory,
                                        num_workers=args.num_workers)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list, gamma=0.1)
    warmup_scheduler = LinearWarmup(lr_scheduler, args.warmup_iters) if args.use_warmup else None

    output_dir = Path(args.output_dir)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')   
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.milestones = Counter(args.lr_drop_list)
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        evaluator = LineEvaluator()
        test_stats = test(model, criterion, postprocessors, evaluator,
                        data_loader_val, device, args.output_dir, args=args)
        return

    print(stats(model_without_ddp, args))

    print("-"*41 + " Start training " + "-"*42)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, lr_scheduler=lr_scheduler, warmup_scheduler=warmup_scheduler, args=args)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if warmup_scheduler is None:
            lr_scheduler.step()
        elif warmup_scheduler.finished():
            lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'warmup_scheduler': warmup_scheduler.state_dict() if warmup_scheduler is not None else None,
                    'epoch': epoch,
                    'args': args,
                }
                utils.save_on_master(weights, checkpoint_path)
                
        # eval
        test_stats = evaluate(
            model, criterion, postprocessors, data_loader_val, device, args.output_dir, args=args
        )
        
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        if utils.is_main_process():
            for key in log_stats:
                if key not in ["epoch", "now_time"]:
                    writer.add_scalar(key, log_stats[key], epoch)
            
        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
    writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LINEA training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
