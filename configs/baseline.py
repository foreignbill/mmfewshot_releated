#### model
model = dict(
    type='Baseline',
    backbone=dict(type='Conv4'),
    head=dict(type='LinearHead', num_classes=100, in_channels=1600),
    meta_test_head=dict(type='LinearHead', num_classes=5, in_channels=1600))