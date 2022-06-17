#### 
num_support_ways = 2
num_support_shots = 10
num_novel_shots = 10
#### dataset
dataset_type='FewShotCocoDataset'
data_root='/workspace/datasets/detection/coco/'
image_prefix='coco/'
split_prefix='few_shot_ann/'
ann_root=''
train_ann_cfg=[dict(type='ann_file',ann_file= data_root + image_prefix + 'annotations/train.json')]
val_ann_cfg=[dict(type='ann_file',ann_file= data_root + image_prefix + 'annotations/val.json')]
#### fine-tuning
fine_tuning_setting='SPLIT_1SHOT'