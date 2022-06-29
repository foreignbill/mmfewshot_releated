_base_ = ['./predefined_generate.py']

#### image config
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

#### pipeline settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size={{_base_.img_size}}),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=({{_base_.img_resize_size}}, -1)),
    dict(type='CenterCrop', crop_size={{_base_.img_size}}),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

#### dataset settings
data = dict(
    val=dict(
        type={{_base_.test_type}},
        num_episodes={{_base_.num_val_episodes}},
        num_ways={{_base_.num_ways}},
        num_shots={{_base_.num_shots}},
        num_queries={{_base_.num_queries}},
        dataset=dict(
            type={{_base_.dataset_type}},
            subset='val',
            data_prefix={{_base_.data_prefix}},
            pipeline=test_pipeline),
        meta_test_cfg=dict(
            num_episodes={{_base_.num_val_episodes}},
            num_ways={{_base_.num_ways}},
            # whether to cache features in fixed-backbone methods for
            # testing acceleration.
            fast_test=True,
            test_set=dict(batch_size=16, num_workers=2),
            support=dict(batch_size=4, num_workers=0),
            query=dict(batch_size={{_base_.query_batch_size}}, num_workers=0))),
    test=dict(
        type={{_base_.test_type}},
        num_episodes={{_base_.num_test_episodes}},
        num_ways={{_base_.num_ways}},
        num_shots={{_base_.num_shots}},
        num_queries={{_base_.num_queries}},
        # seed for generating meta test episodes
        episodes_seed={{_base_.episodes_seed}},
        dataset=dict(
            type={{_base_.dataset_type}},
            subset='test',
            data_prefix={{_base_.data_prefix}},
            pipeline=test_pipeline),
        meta_test_cfg=dict(
            num_episodes={{_base_.num_test_episodes}},
            num_ways={{_base_.num_ways}},
            # whether to cache features in fixed-backbone methods for
            # testing acceleration.
            fast_test=False,
            test_set=dict(batch_size=16, num_workers=2),
            support=dict(batch_size=4, num_workers=0),
            query=dict(batch_size={{_base_.query_batch_size}}, num_workers=0))),
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='EpisodicDataset',
        num_episodes={{_base_.max_iters}},
        num_ways=10,
        num_shots=5,
        num_queries=5,
        dataset=dict(
            type={{_base_.dataset_type}},
            data_prefix={{_base_.data_prefix}},
            subset='train',
            pipeline=train_pipeline)))
    # train=dict(
    #     type={{_base_.dataset_type}},
    #     data_prefix={{_base_.data_prefix}},
    #     subset='train',
    #     pipeline=train_pipeline))

#### log releated
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(interval=4000)

#### learning config
evaluation = dict(by_epoch=False, metric='accuracy', interval=2000)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/export/baseline/best_accuracy_mean.pth'
resume_from = None
workflow = [('train', 1)]
pin_memory = True
use_infinite_sampler = True
seed = 0
runner = dict(type='IterBasedRunner', max_iters={{_base_.max_iters}})

#### learning config
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=0.25,
    step=[20000, 40000])

#### model
model = dict(
    type='MetaBaseline',
    backbone=dict(type={{_base_.backbone}}),
    head=dict(type='MetaBaselineHead'))

# work config
work_dir = '/export/meta_baseline'
tensor_dir = '/tensorboard_log'
gpu_ids = [0]