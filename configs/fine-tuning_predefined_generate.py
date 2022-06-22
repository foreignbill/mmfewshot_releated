num_support_ways = 2
num_support_shots = 10
num_novel_shots = 10
dataset_type = 'FewShotCocoDataset'
data_root = '/workspace/datasets/detection/coco/'
img_prefix = '/workspace/datasets/detection/coco/coco/'
split_prefix = 'few_shot_ann/'
ann_root = ''
train_ann_cfg = [
    dict(
        type='ann_file',
        ann_file=
        '/workspace/datasets/detection/coco/coco/annotations/train.json')
]
val_ann_cfg = [
    dict(
        type='ann_file',
        ann_file='/workspace/datasets/detection/coco/coco/annotations/val.json'
    )
]
fine_tuning_setting = '10SHOT'
evaluation = dict(interval=20000, metric='bbox', classwise=True)
fine_tuning_dataset_type = 'FewShotCocoDefaultDataset'
max_iters = 100
