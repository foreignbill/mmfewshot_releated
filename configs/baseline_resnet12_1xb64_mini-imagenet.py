img_size = 84
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(96, -1)),
    dict(type='CenterCrop', crop_size=84),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
num_ways = 5
num_shots = 1
num_queries = 15
num_val_episodes = 100
num_test_episodes = 2000
data = dict(
    val=dict(
        type='MetaTestDataset',
        num_episodes=100,
        num_ways=5,
        num_shots=1,
        num_queries=15,
        dataset=dict(
            type='MiniImageNetDataset',
            subset='val',
            data_prefix='/home/ghk/workerspace/datasets/mini-imagenet',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', size=(96, -1)),
                dict(type='CenterCrop', crop_size=84),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        meta_test_cfg=dict(
            num_episodes=100,
            num_ways=5,
            fast_test=True,
            test_set=dict(batch_size=16, num_workers=2),
            support=dict(
                batch_size=4,
                num_workers=0,
                drop_last=True,
                train=dict(
                    num_steps=150,
                    optimizer=dict(
                        type='SGD',
                        lr=0.01,
                        momentum=0.9,
                        dampening=0.9,
                        weight_decay=0.001))),
            query=dict(batch_size=75, num_workers=0))),
    test=dict(
        type='MetaTestDataset',
        num_episodes=2000,
        num_ways=5,
        num_shots=1,
        num_queries=15,
        episodes_seed=0,
        dataset=dict(
            type='MiniImageNetDataset',
            subset='test',
            data_prefix='/home/ghk/workerspace/datasets/mini-imagenet',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', size=(96, -1)),
                dict(type='CenterCrop', crop_size=84),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        meta_test_cfg=dict(
            num_episodes=2000,
            num_ways=5,
            fast_test=True,
            test_set=dict(batch_size=16, num_workers=2),
            support=dict(
                batch_size=4,
                num_workers=0,
                drop_last=True,
                train=dict(
                    num_steps=150,
                    optimizer=dict(
                        type='SGD',
                        lr=0.01,
                        momentum=0.9,
                        dampening=0.9,
                        weight_decay=0.001))),
            query=dict(batch_size=75, num_workers=0))),
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type='MiniImageNetDataset',
        data_prefix='/home/ghk/workerspace/datasets/mini-imagenet',
        subset='train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=84),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]))
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(interval=50)
evaluation = dict(by_epoch=True, metric='accuracy', interval=5)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
pin_memory = True
use_infinite_sampler = True
seed = 0
runner = dict(type='EpochBasedRunner', max_epochs=200)
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=0.25,
    step=[60, 120])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=84),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
meta_finetune_cfg = dict(
    num_steps=150,
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001))
model = dict(
    type='Baseline',
    backbone=dict(type='ResNet12'),
    head=dict(type='LinearHead', num_classes=64, in_channels=640),
    meta_test_head=dict(type='LinearHead', num_classes=5, in_channels=640))
work_dir = 'baseline_mini_image_net_work_dir'
tensor_dir = '/tensorboard_log'
gpu_ids = [0]
