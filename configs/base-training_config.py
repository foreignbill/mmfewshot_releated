_base_ = [
    './base-training_predefined_generate.py'
]
# part 1
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Resize',
            img_scale=[(1000, 440), (1000, 472), (1000, 504), (1000, 536),
                       (1000, 568), (1000, 600)],
            keep_ratio=True,
            multiscale_mode='value'),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    support=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='CropResizeInstance',
            num_context_pixels=16,
            target_size=(320, 320)),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize',**img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# part 2
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='QueryAwareDataset',
        num_support_ways={{_base_.num_support_ways}},
        num_support_shots={{_base_.num_support_shots}},
        save_dataset=False,
        dataset=dict(
            # dataset type
            # FewShotVOCDataset, FewShotCocoDataset
            type={{_base_.dataset_type}},
            # where ann_file storage
            ann_cfg={{_base_.train_ann_cfg}},
            data_root={{_base_.data_root}},
            img_prefix={{_base_.img_prefix}},
            multi_pipelines=train_multi_pipelines,
            classes='BASE_CLASSES',
            instance_wise=False,
            min_bbox_area=32 * 32,
            dataset_name='query_support_dataset')),
    val=dict(
        # dataset type
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
        pipeline=test_pipeline,
        test_mode=True,
        classes='BASE_CLASSES'),
    # random sample 10 shot base instance to evaluate training
    model_init=dict(
        copy_from_train_dataset=False,
        samples_per_gpu=16,
        workers_per_gpu=1,
        # dataset type
        type={{_base_.dataset_type}},
        ann_cfg={{_base_.train_ann_cfg}},
        data_root={{_base_.data_root}},
        img_prefix={{_base_.img_prefix}},
        pipeline=train_multi_pipelines['support'],
        classes='BASE_CLASSES',
        # random sample 10 shot base instance to evaluate training
        num_base_shots=10,
        instance_wise=True,
        min_bbox_area=32 * 32,
        dataset_name='model_init_dataset'))
evaluation = {{_base_.evaluation}}
# dict(interval=20000, metric='bbox', classwise=True)
# evaluation = dict(interval=6000, metric='mAP')
optimizer = dict(
    type='SGD',
    lr=0.004,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict({'roi_head.bbox_head': dict(lr_mult=2.0)})))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    # differs
    step=[{{_base_.max_iters}}])
runner = dict(type='IterBasedRunner', max_iters={{_base_.max_iters}})
norm_cfg = dict(type='BN', requires_grad=False)
pretrained = 'pretrained/detectron2_resnet50_caffe.pth'
model = dict(
    type='AttentionRPNDetector',
    pretrained=pretrained,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),
        frozen_stages=2,
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='caffe'),
    rpn_head=dict(
        type='AttentionRPNHead',
        in_channels=1024,
        feat_channels=1024,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            scale_major=False,
            strides=[16]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        num_support_ways={{_base_.num_support_ways}},
        num_support_shots={{_base_.num_support_shots}},
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        aggregation_layer=dict(
            type='AggregationLayer',
            aggregator_cfgs=[
                dict(
                    type='DepthWiseCorrelationAggregator',
                    in_channels=1024,
                    with_fc=False)
            ])),
    roi_head=dict(
        type='MultiRelationRoIHead',
        shared_head=dict(
            type='ResLayer',
            pretrained=pretrained,
            depth=50,
            stage=3,
            stride=2,
            dilation=1,
            style='caffe',
            norm_cfg=norm_cfg,
            norm_eval=True),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        bbox_head=dict(
            type='MultiRelationBBoxHead',
            with_avg_pool=True,
            roi_feat_size=14,
            in_channels=2048,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            patch_relation=True,
            local_correlation=True,
            global_relation=True,
            init_cfg=[
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal', name='patch_relation_branch',
                        std=0.01)),
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal', name='patch_relation_fc_cls',
                        std=0.01)),
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal', name='patch_relation_fc_reg',
                        std=0.001)),
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal',
                        name='local_correlation_branch',
                        std=0.01)),
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal',
                        name='local_correlation_fc_cls',
                        std=0.01)),
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal', name='global_relation_branch',
                        std=0.01)),
                dict(
                    type='Normal',
                    override=dict(
                        type='Normal', name='global_relation_fc_cls',
                        std=0.01))
            ]),
        num_support_ways={{_base_.num_support_ways}},
        num_support_shots={{_base_.num_support_shots}}),
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
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=12000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
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
                num=128,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
checkpoint_config = dict(interval=20000)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
use_infinite_sampler = True
seed = 42
work_dir = '/export/base-training'
gpu_ids = [0]
