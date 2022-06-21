num_support_ways = 2
num_support_shots = 5
num_novel_shots = 10
dataset_type = 'FewShotVOCDataset'
data_root = '/workspace/datasets/detection/voc/'
img_prefix = '/workspace/datasets/detection/voc/VOCdevkit/'
split_prefix = 'few_shot_ann/'
ann_root = ''
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
fine_tuning_setting = '10SHOT'
evaluation = dict(interval=6000, metric='mAP')
fine_tuning_dataset_type = 'FewShotVOCDefaultDataset'
max_iters = 120000
