import argparse
import json
import os 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# pytorch ddp 
import torch.distributed as dist

# Parameters and DataLoaders
input_size = 5
output_size = 2
batch_size = 30
data_size = 100

#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        try:
            id = int(torch.cuda.current_device())
            print(torch.cuda.device(id))
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
        except:
            print("not able to print device")

        return output

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


if __name__=="__main__":
    args = parse_args()
    # 
    world_size = len(args.hosts)
    host_rank = args.hosts.index(args.current_host)
    print(f'host rank is {host_rank}')
    dist.init_process_group(
        backend=args.backend,
        rank=host_rank,
        world_size=world_size
    )
    # device 
    device = "cuda"
    # model
    model = Model(input_size, output_size)
    # for single machine gpus
    model = torch.nn.DataParallel(model)
    # multiple machine gpus
    # model = torch.nn.parallel.DistributedDataParallel(model) 
    model.to(device)
    # gen data 
    rand_loader = DataLoader(
        dataset=RandomDataset(input_size, data_size), 
        batch_size=batch_size, 
        shuffle=True)
    # train 
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
        print("Outside: input size", input.size(), "output_size", output.size())
