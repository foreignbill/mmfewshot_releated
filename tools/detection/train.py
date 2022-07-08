# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import cv2
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash
from mmdet.utils import collect_env

from sys import path
__root_dir__ = os.path.realpath(os.path.join(__file__, '../../..'))
path.append(__root_dir__)

import mmfewshot  # noqa: F401, F403
from mmfewshot import __version__
from mmfewshot.detection.apis import train_detector
from mmfewshot.detection.datasets import build_dataset
from mmfewshot.detection.models import build_detector
from mmfewshot.utils import get_root_logger

# disable multithreading to avoid system being overloaded
cv2.setNumThreads(0)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a FewShot model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work-dir', help='the directory to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
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
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    
    ########################################################################
    #### number of few-shot learning config
    parser.add_argument('--num_support_ways', type=int, default=15, help='number of support ways')
    parser.add_argument('--num_support_shots', type=int, default=10, help='number of support shots')
    parser.add_argument('--num_novel_shots', type=int, default=1, help='number of novel shots')
    #### dataset config
    parser.add_argument('--dataset_prefix', type=str, default='/dataset', help='prefix of dataset')
    #### runner
    parser.add_argument('--max_iters', type=int, default=120000, help='max iterations of runner')
    #### seed
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    ########################################################################

    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value '
        'to be overwritten is a list, it should be like key="[a,b]" or '
        'key=a,b It also allows nested list/tuple values, e.g. '
        'key="[(a,b),(c,d)]" Note that the quotation marks are necessary '
        'and that no white space is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    ################################################################################################################
    #### predefined path
    predefined_path = args.config.replace('config.py', 'predefined.py')
    predefined_cfg = Config.fromfile(predefined_path)
    #### number of few-shot learning config
    predefined_cfg.num_support_ways = args.num_support_ways
    predefined_cfg.num_support_shots = args.num_support_shots
    predefined_cfg.num_novel_shots = args.num_novel_shots
    predefined_cfg.fine_tuning_setting = f'{args.num_novel_shots}SHOT'
    #### dataset config
    dataset_path = args.dataset_prefix
    data_config = Config.fromfile(osp.join(dataset_path, 'data_config.py'))
    predefined_cfg.dataset_type = data_config.dataset_type
    predefined_cfg.data_root = args.dataset_prefix
    predefined_cfg.img_prefix = osp.join(dataset_path, data_config.img_prefix)

    predefined_cfg.train_ann_cfg = data_config.train_ann_cfg
    for i in range(len(predefined_cfg.train_ann_cfg)):
        predefined_cfg.train_ann_cfg[i].update({'ann_file': osp.join(predefined_cfg.img_prefix, predefined_cfg.train_ann_cfg[i]['ann_file'])})
    predefined_cfg.val_ann_cfg = data_config.val_ann_cfg
    for i in range(len(predefined_cfg.val_ann_cfg)):
        predefined_cfg.val_ann_cfg[i].update({'ann_file': osp.join(predefined_cfg.img_prefix, predefined_cfg.val_ann_cfg[i]['ann_file'])})

    #### runner
    predefined_cfg.max_iters = args.max_iters

    if data_config.dataset_type == 'FewShotCocoDataset':
        predefined_cfg.evaluation = dict(interval=min(20000, predefined_cfg.max_iters), metric='bbox', classwise=True)
        predefined_cfg.fine_tuning_dataset_type = 'FewShotCocoDefaultDataset'
        predefined_cfg.num_classes = len(data_config.COCO_SPLIT['ALL_CLASSES'])
    elif data_config.dataset_type == 'FewShotVOCDataset':
        predefined_cfg.evaluation = dict(interval=min(6000, predefined_cfg.max_iters), metric='mAP')
        predefined_cfg.fine_tuning_dataset_type = 'FewShotVOCDefaultDataset'
        predefined_cfg.num_classes = len(data_config.VOC_SPLIT['ALL_CLASSES'])

    #### generate new predefined config file
    predefined_generate_path = args.config.replace('config.py', 'predefined_generate.py')
    with open(predefined_generate_path, 'w') as f:
        f.write(predefined_cfg.pretty_text)
    # return None
    ################################################################################################################
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
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
        rank = 0
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
        # re-set gpu_ids with distributed training mode
        cfg.gpu_ids = range(world_size)
    # if fine tuning, use different data
    if [key for key in cfg].count('data') == 0:
        if data_config.dataset_type == 'FewShotCocoDataset':
            cfg.data = cfg.coco_data
        elif data_config.dataset_type == 'FewShotVOCDataset':
            cfg.data = cfg.voc_data
    #
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
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
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        seed = args.seed
    elif cfg.seed is not None:
        seed = cfg.seed
    elif distributed:
        seed = 0
        Warning(f'When using DistributedDataParallel, each rank will '
                f'initialize different random seed. It will cause different'
                f'random action for each rank. In few shot setting, novel '
                f'shots may be generated by random sampling. If all rank do '
                f'not use same seed, each rank will sample different data.'
                f'It will cause UNFAIR data usage. Therefore, seed is set '
                f'to {seed} for default.')
    else:
        seed = None

    if seed is not None:
        logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(seed, deterministic=args.deterministic)
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # build_detector will do three things, including building model,
    # initializing weights and freezing parameters (optional).
    model = build_detector(cfg.model, logger=logger)
    # build_dataset will do two things, including building dataset
    # and saving dataset into json file (optional).
    datasets = [
        build_dataset(
            cfg.data.train,
            rank=rank,
            work_dir=cfg.work_dir,
            timestamp=timestamp)
    ]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmfewshot version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmfewshot_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
