# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
from sys import stdout, path
__root_dir__ = os.path.realpath(os.path.join(__file__, '../../..'))
path.append(__root_dir__)
import time
import warnings

import cv2
import mmcv
import torch
from mmcls import __version__
from mmcls.models import build_classifier
from mmcls.utils import collect_env
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed

from mmfewshot.classification.apis import train_model
from mmfewshot.classification.datasets import build_dataset
from mmfewshot.utils import get_root_logger

# disable multithreading to avoid system being overloaded
cv2.setNumThreads(0)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device', help='device used for training')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    ################################################################################################################
    #### number of few-shot learning config
    parser.add_argument('--num_ways', type=int, default=5, help='number of categories in support set')
    parser.add_argument('--num_shots', type=int, default=1, help='number of samples in each category')
    parser.add_argument('--num_queries', type=int, default=15, help='number of queries')
    parser.add_argument('--num_val_episodes', type=int, default=100, help='number of val episodes')
    parser.add_argument('--num_test_episodes', type=int, default=2000, help='number of test episodes')
    #### dataset config
    parser.add_argument('--dataset_prefix', type=str, default='/dataset', help='prefix of dataset')
    parser.add_argument('--img_size', type=int, default=84, help='image size')
    #### meta test config
    parser.add_argument('--episodes_seed', type=int, default=0, help='seed for generating meta test episodes')
    #### model
    parser.add_argument('--backbone', type=str, default='Conv4', help='which backbone to be used, choose between Conv4 and ResNet12')
    #### runner
    parser.add_argument('--max_iters', type=int, default=100000, help='max iterations of runner')
    ################################################################################################################
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    ################################################################################################################
    #### predefined path
    predefined_path = args.config.replace('config.py', 'predefined.py')
    predefined_cfg = Config.fromfile(predefined_path)
    #### number of few-shot learning config
    predefined_cfg.num_ways = args.num_ways
    predefined_cfg.num_shots = args.num_shots
    predefined_cfg.num_queries = args.num_queries
    predefined_cfg.num_train_queries = args.num_queries + 1
    predefined_cfg.num_val_episodes = args.num_val_episodes
    predefined_cfg.num_test_episodes = args.num_test_episodes
    predefined_cfg.query_batch_size = predefined_cfg.num_ways * predefined_cfg.num_queries
    predefined_cfg.support_batch_size = predefined_cfg.num_ways * predefined_cfg.num_shots
    #### dataset config
    dataset_path = args.dataset_prefix
    data_config = Config.fromfile(osp.join(dataset_path, 'data_config.py'))
    predefined_cfg.dataset_type = data_config.dataset_type
    predefined_cfg.data_prefix = dataset_path
    predefined_cfg.img_size = args.img_size
    predefined_cfg.img_resize_size = int(predefined_cfg.img_size * 1.15)
    if predefined_cfg.dataset_type == 'CUBDataset':
        predefined_cfg.num_classes = (len(data_config.ALL_CLASSES) + 1) // 2
    else:
        # 'MiniImageNetDataset', 'TieRedImageNetDataset'
        predefined_cfg.num_classes = len(data_config.TRAIN_CLASSES)
    #### meta test config
    predefined_cfg.episodes_seed = args.episodes_seed
    #### model
    predefined_cfg.backbone = args.backbone
    if predefined_cfg.backbone == 'Conv4':
        # in Conv4 model, out_channels is decided by input image size
        backbone_out_channels = (predefined_cfg.img_size // 2 // 2 // 2 // 2) ** 2
        predefined_cfg.in_channels = backbone_out_channels * 64
    elif predefined_cfg.backbone == 'ResNet12':
        # in ResNet12 model, out_channels was fitted to 640
        predefined_cfg.in_channels = 640
    #### runner
    predefined_cfg.max_iters = args.max_iters

    #### generate new predefined config file
    predefined_generate_path = args.config.replace('config.py', 'predefined_generate.py')
    with open(predefined_generate_path, 'w') as f:
        f.write(predefined_cfg.pretty_text)
    ################################################################################################################
    
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('/export',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]
        

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # TODO: log to /log/stdout.txt
    # progress bar should stream to other stream
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # log_file = '/log/stdout.txt'
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        seed = args.seed
    elif cfg.get('seed', None) is not None:
        seed = cfg.seed
    else:
        seed = None
    # set random seeds
    if seed is not None:
        logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    model = build_classifier(cfg.model)
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmcls version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        device='cpu' if args.device == 'cpu' else 'cuda',
        meta=meta)


if __name__ == '__main__':
    main()
