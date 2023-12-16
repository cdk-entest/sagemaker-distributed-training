import json
import argparse
import os
import torch
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# 
import torch.distributed as dist

# Parameters and DataLoaders
input_size = 5
output_size = 2
batch_size = 30
data_size = 100

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(), "output size", output.size())
        return output


def _get_train_data_loader(is_distributed:bool):
    # generate data 
    train_set = RandomDataset(input_size, data_size) 
    # 
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_set) if is_distributed else None
    )
    # 
    return torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=train_sampler is None, 
        sampler=train_sampler
    )

def _get_test_data_loader():
    return None 

def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="custom",
    )
    parser.add_argument(
        "--hosts", 
        type=list,
        default=json.loads(os.environ["SM_HOSTS"])
    )
    parser.add_argument(
        "--current-host", 
        type=str,
        default=os.environ["SM_CURRENT_HOST"]
    )
    return parser.parse_args()


def train(args):
    # device 
    device = torch.device("cpu")
    # 
    world_size = len(args.hosts)
    # 
    host_rank = args.hosts.index(args.current_host)
    # 
    print(f'world size {world_size} and host rank {host_rank}')

    dist.init_process_group(
        backend=args.backend,
        rank=host_rank,
        world_size=world_size)

    # model 
    model = Model(input_size, output_size).to(device)
    # model = torch.nn.DataParallel(model)
    model = torch.nn.parallel.DistributedDataParallel(model)

    # data
    train_loader = _get_train_data_loader(is_distributed=True)
    print(f'train_loader_len {len(train_loader.sampler)} data_set_len {len(train_loader.dataset)}')

    # train 
    for data in train_loader:
        input = data.to(device)
        output = model(input)
        print("Outside: input size", input.size(), "output_size", output.size())
    

if __name__=="__main__":
    args = parse_args()
    train(args)