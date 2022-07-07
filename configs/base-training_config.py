_base_ = [
    './base-training_predefined_generate.py'
]
# part 1
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
multi_scales = (32, 64, 128, 256, 512, 800)
train_multi_pipelines = dict(
    main=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Resize',
            img_scale=[(1333, 800)],
            keep_ratio=True,
            multiscale_mode='value'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[102.9801, 115.9465, 122.7717],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    auxiliary=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='CropInstance', context_ratio=0.14285714285714285),
        dict(
            type='ResizeToMultiScale',
            multi_scales=[(36.57142857142857, 36.57142857142857),
                          (73.14285714285714, 73.14285714285714),
                          (146.28571428571428, 146.28571428571428),
                          (292.57142857142856, 292.57142857142856),
                          (585.1428571428571, 585.1428571428571),
                          (914.2857142857143, 914.2857142857143)]),
        dict(
            type='MultiImageRandomCrop',
            multi_crop_sizes=[(32, 32), (64, 64), (128, 128), (256, 256),
                              (512, 512), (800, 800)]),
        dict(
            type='MultiImageNormalize',
            mean=[102.9801, 115.9465, 122.7717],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='MultiImageRandomFlip', flip_ratio=0.5),
        dict(type='MultiImagePad', size_divisor=32),
        dict(type='MultiImageFormatBundle'),
        dict(type='MultiImageCollect', keys=['img', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[102.9801, 115.9465, 122.7717],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# part 2
# voc
voc_data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    auxiliary_samples_per_gpu=2,
    auxiliary_workers_per_gpu=2,
    train=dict(
        type='TwoBranchDataset',
        save_dataset=False,
        reweight_dataset=False,
        dataset=dict(
            type={{_base_.dataset_type}},
            ann_cfg={{_base_.train_ann_cfg}},
            data_root={{_base_.data_root}},
            img_prefix={{_base_.img_prefix}},
            multi_pipelines=train_multi_pipelines,
            classes='BASE_CLASSES',
            use_difficult=False,
            instance_wise=False,
            coordinate_offset=[-1, -1, -1, -1],
            dataset_name='main_dataset'),
        auxiliary_dataset=dict(
            copy_from_main_dataset=True,
            instance_wise=True,
            dataset_name='auxiliary_dataset')),
    val=dict(
        type={{_base_.dataset_type}},
        ann_cfg={{_base_.val_ann_cfg}},
        data_root={{_base_.data_root}},
        img_prefix={{_base_.img_prefix}},
        pipeline=test_pipeline,
        coordinate_offset=[-1, -1, -1, -1],
        classes='BASE_CLASSES'),
    test=dict(
        type={{_base_.dataset_type}},
        ann_cfg={{_base_.val_ann_cfg}},
        data_root={{_base_.data_root}},
        img_prefix={{_base_.img_prefix}},
        pipeline=test_pipeline,
        coordinate_offset=[-1, -1, -1, -1],
        test_mode=True,
        classes='BASE_CLASSES'))
# coco
coco_data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    auxiliary_samples_per_gpu=2,
    auxiliary_workers_per_gpu=2,
    train=dict(
        type='TwoBranchDataset',
        save_dataset=False,
        reweight_dataset=False,
        dataset=dict(
            type={{_base_.dataset_type}},
            ann_cfg={_base_.train_ann_cfg},
            data_root={{_base_.data_root}},
            img_prefix={{_base_.img_prefix}},
            multi_pipelines=train_multi_pipelines,
            classes='BASE_CLASSES',
            instance_wise=False,
            dataset_name='main_dataset'),
        auxiliary_dataset=dict(
            copy_from_main_dataset=True,
            instance_wise=True,
            min_bbox_size=8,
            dataset_name='auxiliary_dataset')),
    val=dict(
        type={{_base_.dataset_type}},
        ann_cfg={{_base_.val_ann_cfg}},
        data_root={{_base_.data_root}},
        img_prefix={{_base_.img_prefix}},
        pipeline=test_pipeline,
        classes='BASE_CLASSES'),
    test=dict(
        type={{_base_.dataset_type}},
        ann_cfg={{_base_.val_ann_cfg}},
        data_root={{_base_.data_root}},
        img_prefix={{_base_.img_prefix}},
        pipeline=test_,
        test_mode=True,
        classes='BASE_CLASSES'))
evaluation = {{_base_.evaluation}}
# dict(interval=20000, metric='bbox', classwise=True)
# evaluation = dict(interval=6000, metric='mAP')
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[24000, 32000])
runner = dict(type='IterBasedRunner', max_iters={{_base_.max_iters}})
norm_cfg = dict(type='BN', requires_grad=False)
pretrained = 'pretrained/detectron2_resnet101_caffe.pth'
model = dict(
    type='MPSR',
    pretrained=pretrained,
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        init_cfg=[
            dict(
                type='Caffe2Xavier',
                override=dict(type='Caffe2Xavier', name='lateral_convs')),
            dict(
                type='Caffe2Xavier',
                override=dict(type='Caffe2Xavier', name='fpn_convs'))
        ]),
    rpn_head=dict(
        type='TwoBranchRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', loss_weight=1.0, beta=0.1111111111111111),
        mid_channels=64),
    roi_head=dict(
        type='TwoBranchRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=8, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='TwoBranchBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=8,
            num_classes=60,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0, beta=1.0),
            init_cfg=[
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='shared_fcs')),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.01)),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_reg', std=0.001))
            ],
            num_cls_fcs=2,
            num_reg_fcs=2,
            auxiliary_loss_weight=0.1)),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=None,
            nms=dict(type='nms', iou_threshold=0.7, offset=1),
            min_bbox_size=0,
            max_per_batch=2000),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7, offset=1),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5, offset=1),
            max_per_img=100)),
    rpn_select_levels=[0, 1, 2, 3, 4, 4],
    roi_select_levels=[0, 0, 0, 1, 2, 3])
checkpoint_config = dict(interval=5000)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
use_infinite_sampler = False
seed = 42
work_dir = '/export/base-training'
gpu_ids = [0]
