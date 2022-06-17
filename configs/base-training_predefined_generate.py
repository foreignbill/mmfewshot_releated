num_support_ways = 2
num_support_shots = 10
dataset_type = 'FewShotVOCDataset'
data_root = '/workspace/datasets/detection/voc'
image_prefix = '/workspace/datasets/detection/voc/VOCdevkit/'
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
evaluation = dict(interval=6000, metric='mAP')
max_iters = 120000
