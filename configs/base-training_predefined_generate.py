num_support_ways = 15
num_support_shots = 1
dataset_type = 'FewShotVOCDataset'
data_root = '/workspace/datasets/detection/voc/'
img_prefix = '/workspace/datasets/detection/voc/VOCdevkit/'
split_prefix = 'few_shot_ann/'
train_ann_cfg = [
    dict(
        type='ann_file',
        ann_file=
        '/workspace/datasets/detection/voc/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
    ),
    dict(
        type='ann_file',
        ann_file=
        '/workspace/datasets/detection/voc/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
    )
]
val_ann_cfg = [
    dict(
        type='ann_file',
        ann_file=
        '/workspace/datasets/detection/voc/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    )
]
evaluation = dict(interval=400, metric='mAP')
max_iters = 400
num_novel_shots = 1
fine_tuning_setting = '1SHOT'
fine_tuning_dataset_type = 'FewShotVOCDefaultDataset'
num_classes = 20
num_base_classes = 15
