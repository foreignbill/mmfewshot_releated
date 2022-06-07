#### number of few-shot learning config
num_ways = 5
num_shots = 1
num_queries = 15
num_val_episodes = 100
num_test_episodes = 2000

#### dataset
dataset_type = 'CUBDataset'
data_prefix = '/home/ghk/workerspace/datasets/classification/CUB_200_2011'

#### test
test_type = 'MetaTestDataset'

#### meta test
# seed for generating meta test episodes
episodes_seed = 0

#### model
model_name = 'baseline'
backbone = 'Conv4'

#### batch size
query_batch_size = num_ways * num_queries

#### runner
max_epoch = 200

#### gpu resources
gpu_resources = 2048