num_ways = 5
num_shots = 1
num_queries = 15
num_train_queries = 16
num_val_episodes = 100
num_test_episodes = 2000
dataset_type = 'CUBDataset'
data_prefix = '/home/ghk/workerspace/datasets/classification/CUB_200_2011/'
num_classes = 100
img_size = 84
img_resize_size = 96
test_type = 'MetaTestDataset'
episodes_seed = 0
model_name = 'Baseline'
backbone = 'Conv4'
in_channels = 1600
query_batch_size = 75
support_batch_size = 5
max_iters = 100000
gpu_resources = 2048
