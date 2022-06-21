#### 
num_support_ways = 2
num_support_shots = 10
#### dataset
dataset_type='FewShotCocoDataset'
data_root='/workspace/datasets/detection/coco/'
img_prefix='coco/'
split_prefix='few_shot_ann/'
train_ann_cfg=[dict(type='ann_file',ann_file= data_root + img_prefix + 'annotations/train.json')]
val_ann_cfg=[dict(type='ann_file',ann_file= data_root + img_prefix + 'annotations/val.json')]
# coco
evaluation = dict(interval=20000, metric='bbox', classwise=True)
# voc
# evaluation = dict(interval=6000, metric='mAP')
####
max_iters=120000