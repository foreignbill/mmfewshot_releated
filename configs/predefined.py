#### number of few-shot learning config
num_ways = 5
num_shots = 1
num_queries = 15
num_train_queries = 16
num_val_episodes = 100
num_test_episodes = 2000

#### dataset
dataset_type = 'CUBDataset'
data_prefix = '/home/ghk/workerspace/datasets/classification/omniglot'
num_classes = 1623
img_size = 84
img_resize_size = int(img_size * 1.15)

#### test
test_type = 'MetaTestDataset'

#### meta test
# seed for generating meta test episodes
episodes_seed = 0

#### model
model_name = 'Baseline'
backbone = 'Conv4'
in_channels = 1600

#### batch size
query_batch_size = num_ways * num_queries
support_batch_size = num_ways * num_shots

#### runner
max_iters = 100000

#### gpu resources
gpu_resources = 2048