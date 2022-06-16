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
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
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
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
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
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = '/workspace/datasets/detection/coco/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='QueryAwareDataset',
        num_support_ways=2,
        num_support_shots=9,
        save_dataset=True,
        dataset=dict(
            type='FewShotCocoDefaultDataset',
            ann_cfg=[dict(method='Attention_RPN', setting='10SHOT')],
            img_prefix='/workspace/datasets/detection/coco/coco/',
            num_novel_shots=10,
            num_base_shots=None,
            multi_pipelines=dict(
                query=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='Resize',
                        img_scale=[(1000, 440), (1000, 472), (1000, 504),
                                   (1000, 536), (1000, 568), (1000, 600)],
                        keep_ratio=True,
                        multiscale_mode='value'),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ],
                support=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='CropResizeInstance',
                        num_context_pixels=16,
                        target_size=(320, 320)),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ]),
            classes='NOVEL_CLASSES',
            instance_wise=False,
            min_bbox_area=0,
            dataset_name='query_support_dataset'),
        repeat_times=50),
    val=dict(
        type='FewShotCocoDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=
                '/workspace/datasets/detection/coco/few_shot_ann/coco/annotations/val.json'
            )
        ],
        img_prefix='/workspace/datasets/detection/coco/coco/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1000, 600),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes='NOVEL_CLASSES'),
    test=dict(
        type='FewShotCocoDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=
                '/workspace/datasets/detection/coco/few_shot_ann/coco/annotations/val.json'
            )
        ],
        img_prefix='/workspace/datasets/detection/coco/coco/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1000, 600),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        test_mode=True,
        classes='NOVEL_CLASSES'),
    model_init=dict(
        copy_from_train_dataset=True,
        samples_per_gpu=16,
        workers_per_gpu=1,
        type='FewShotCocoDataset',
        ann_cfg=None,
        img_prefix='/workspace/datasets/detection/coco/coco/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='CropResizeInstance',
                num_context_pixels=16,
                target_size=(320, 320)),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes='NOVEL_CLASSES',
        num_novel_shots=None,
        num_base_shots=None,
        instance_wise=True,
        min_bbox_area=1024,
        dataset_name='model_init_dataset'))
evaluation = dict(
    interval=3000, metric='bbox', classwise=True, class_splits=None)
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict({'roi_head.bbox_head': dict(lr_mult=2.0)})))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.1,
    step=[2000, 3000])
runner = dict(type='IterBasedRunner', max_iters=3000)
norm_cfg = dict(type='BN', requires_grad=False)
pretrained = 'open-mmlab://detectron2/resnet50_caffe'
model = dict(
    type='AttentionRPNDetector',
    pretrained='open-mmlab://detectron2/resnet50_caffe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),
        frozen_stages=2,
        norm_cfg=dict(type='BN', requires_grad=False),
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
        num_support_ways=2,
        num_support_shots=9,
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
            pretrained='open-mmlab://detectron2/resnet50_caffe',
            depth=50,
            stage=3,
            stride=2,
            dilation=1,
            style='caffe',
            norm_cfg=dict(type='BN', requires_grad=False),
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
        num_support_ways=2,
        num_support_shots=9),
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
            max_per_img=100)),
    frozen_parameters=['backbone'])
num_support_ways = 2
num_support_shots = 9
checkpoint_config = dict(interval=3000)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/attention-rpn_r50_c4_4xb2_coco_base-training/attention-rpn_r50_c4_4xb2_coco_10shot-fine-tuning_20211103_003801-94ec8ada.pth'
resume_from = None
workflow = [('train', 1)]
use_infinite_sampler = True
seed = 42
work_dir = './work_dirs/attention-rpn_r50_c4_4xb2_coco_10shot-fine-tuning'
gpu_ids = [0]
