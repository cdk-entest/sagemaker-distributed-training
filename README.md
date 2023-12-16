---
title: getting started with distributed training on sagemaker
author: haimtran
date: 15 DEC 2023
---

## Basic MPI

- Run on 3 nodes x 2 process per node
- Broadcass node, and receive node
- Agregrate data on nodes

Let create a simple MPI (Message Passing Interface)

```py
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print("Number of MPI processes that will talk to each other:", size)


def point_to_point():
    """Point to point communication
    Send a numpy array (buffer like object) from rank 0 to rank 1
    """
    if rank == 0:
        print("point to point")
        data = np.array([0, 1, 2], dtype=np.intc)  # int in C

        # remember the difference between
        # Upper case API and lower case API
        # Basically uppper case API directly calls C API
        # so it is fast
        # checkout https://mpi4py.readthedocs.io/en/stable/

        comm.Send([data, MPI.INT], dest=1)
    elif rank == 1:
        print(f"Hello I am rank {rank}")
        data = np.empty(3, dtype=np.intc)
        comm.Recv([data, MPI.INT], source=0)
        print("I received some data:", data)

    if rank == 0:
        time.sleep(1)  # give some buffer time for execution to complete
        print("=" * 50)
    return


def broadcast():
    """Broadcast a numpy array from rank 0 to others"""

    if rank == 0:
        print(f"Broadcasting from rank {rank}")
        data = np.arange(10, dtype=np.intc)
    else:
        data = np.empty(10, dtype=np.intc)

    comm.Bcast([data, MPI.INT], root=0)
    print(f"Data at rank {rank}", data)

    if rank == 0:
        time.sleep(1)
        print("=" * 50)
    return


def gather_reduce_broadcast():
    """Gather numpy arrays from all ranks to rank 0
    then take average and broadcast result to other ranks

    It is a useful operation in distributed training:
    train a model in a few MPI workers with different
    input data, then take average weights on rank 0 and
    synchroinze weights on other ranks
    """

    # stuff to gather at each rank
    sendbuf = np.zeros(10, dtype=np.intc) + rank
    recvbuf = None

    if rank == 0:
        print("Gather and reduce")
        recvbuf = np.empty([size, 10], dtype=np.intc)
    comm.Gather(sendbuf, recvbuf, root=0)

    if rank == 0:
        print(f"I am rank {rank}, data I gathered is: {recvbuf}")

        # take average
        # think of it as a prototype of
        # average weights, average gradients etc
        avg = np.mean(recvbuf, axis=0, dtype=np.float)

    else:
        # get averaged array from rank 0
        # think of it as a prototype of
        # synchronizing weights across different MPI procs
        avg = np.empty(10, dtype=np.float)

    # Note that the data type is float here
    # because we took average
    comm.Bcast([avg, MPI.FLOAT], root=0)

    print(f"I am rank {rank}, my avg is: {avg}")
    return


if __name__ == "__main__":
    point_to_point()
    broadcast()
    gather_reduce_broadcast()

```

Then create a SageMaker job to run it

```py

import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow

role = get_execution_role()

# Running 2 processes per host
# if we use 3 instances,
# then we should see 6 MPI processes

distribution = {"mpi": {"enabled": True, "processes_per_host": 2}}

tfest = TensorFlow(
    entry_point="mpi_demo.py",
    role=role,
    framework_version="2.3.0",
    distribution=distribution,
    py_version="py37",
    instance_count=3,
    instance_type="ml.c5.2xlarge",  # 8 cores
    output_path="s3://" + sagemaker.Session().default_bucket() + "/" + "mpi",
)
```

## Pytorch Multiple CPU

- Average gradient and DistributedSampler
- Setup dist.init_process_group
- Wrap model in DataParallel or DistributedDataParallel

Follow [this sm example](https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-python-sdk/pytorch_mnist/mnist.py) and [distributed training workshop example](https://shashankprasanna.com/workshops/pytorch-distributed-sagemaker/2_distributed_training.html)

[!NOTE]

>

```py
if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)
```

Let create model and train script

```py
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
    model = torch.nn.DataParallel(model)
    # model = torch.nn.parallel.DistributedDataParallel(model)

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
```

Let create a training job in SageMaker

```py
estimator_1 = PyTorch(
    role="arn:aws:iam::392194582387:role/RoleForDataScientistUserProfile",
    entry_point="train-multiple-instance-cpu.py",
    framework_version="1.8.0",
    py_version="py3",
    instance_count=2,
    instance_type="ml.c5.2xlarge",
    hyperparameters={
      'backend': 'gloo',
      'model-type': 'custom'
    }
    # distribution={
        # "smdistributed": {"dataparallel": {"enabled": True}}
        # mpirun backend
        # "pytorchddp": {"enable": True}
    # },
)
```

Check the output on host algo-1

```py
#011In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#011In Model: input size torch.Size([20, 5]) output size torch.Size([20, 2])
Outside: input size torch.Size([20, 5]) output_size torch.Size([20, 2])
```

and check output on host algo-2

```py
#011In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#011In Model: input size torch.Size([20, 5]) output size torch.Size([20, 2])
Outside: input size torch.Size([20, 5]) output_size torch.Size([20, 2])
```

## Pytorch DataParallel

- Pin the model to multiple GPUs
- DataParallel automatically split the batch into smaller batches running on GPUs

Follow [this tutorial](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html), first create a simple model

```py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2
batch_size = 30
data_size = 100

#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


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


rand_loader = DataLoader(
    dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True
)


model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)


model.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(), "output_size", output.size())
```

Then create SM training job

```py
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    role="arn:aws:iam::$ACCOUNT_ID:role/RoleForDataScientistUserProfile",
    entry_point="train-torch-data-parallel.py",
    framework_version="2.0.1",
    py_version="py310",
    instance_count=1,
    instance_type="ml.g5.12xlarge",
    distribution={
        # mpirun backend
        "pytorchddp": {"enable": True}
    },
)

estimator.fit()
```

In case of 4 GPUs, the output look like this

```txt
Let's use 4 GPUs!
NCCL version 2.17.1+cuda11.8
algo-1:46:58 [0] configure_nvls_option:287 NCCL WARN NET/OFI Could not find ncclGetVersion symbol
In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
<torch.cuda.device object at 0x7efe8dc55cf0>
NVIDIA A10G
In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
<torch.cuda.device object at 0x7efe8dc54e50>
NVIDIA A10G
In Model: input size torch.Size([8, 5]) output size torch.Size([8, 2])
<torch.cuda.device object at 0x7efe8dc55990>
NVIDIA A10G
In Model: input size torch.Size([6, 5]) output size torch.Size([6, 2])
<torch.cuda.device object at 0x7efe8dc55ab0>
NVIDIA A10G
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
In Model: input size
torch.Size([8, 5])
output size torch.Size([8, 2])
In Model: input size
In Model: input size torch.Size([8, 5])
torch.Size([8, 5])
output size torch.Size([8, 2])
<torch.cuda.device object at 0x7efe8dc55d80>
In Model: input size torch.Size([6, 5])
output size torch.Size([8, 2])
<torch.cuda.device object at 0x7efe8dc55de0>
NVIDIA A10G
<torch.cuda.device object at 0x7efe8dc55d80>
NVIDIA A10G
NVIDIA A10G
output size
torch.Size([6, 2])
<torch.cuda.device object at 0x7efe8dc55a20>
NVIDIA A10G
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
In Model: input size
In Model: input size torch.Size([8, 5])
output size torch.Size([8, 2])
<torch.cuda.device object at 0x7efe8dc558d0>
NVIDIA A10G
torch.Size([8, 5])
output size
torch.Size([8, 2])
In Model: input size#011In Model: input size
torch.Size([6, 5])torch.Size([8, 5])
output size
output size <torch.cuda.device object at 0x7efe8dc55de0>
torch.Size([8, 2])
torch.Size([6, 2])
<torch.cuda.device object at 0x7efe8dc55c60>
NVIDIA A10G
<torch.cuda.device object at 0x7efe8dc55d80>NVIDIA A10G
NVIDIA A10G
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
In Model: input size
torch.Size([3, 5])
output size#011In Model: input size
torch.Size([3, 5]) torch.Size([3, 2])
output size
torch.Size([3, 2])
<torch.cuda.device object at 0x7efe8dc55cc0>
NVIDIA A10G
<torch.cuda.device object at 0x7efe8dc55750>
NVIDIA A10G
In Model: input size
torch.Size([3, 5]) output size torch.Size([3, 2])
<torch.cuda.device object at 0x7efe8dc559f0>
NVIDIA A10G
In Model: input size torch.Size([1, 5]) output size torch.Size([1, 2])
<torch.cuda.device object at 0x7efe8dc54e50>
NVIDIA A10G
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

If there is no GPU or single GPU, output should look like

```txt
In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
<torch.cuda.device object at 0x7f0707748f10>
NVIDIA A10G
Outside: input size torch.Size([30, 5])
output_size torch.Size([30, 2])
In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
<torch.cuda.device object at 0x7f0707748f10>
NVIDIA A10G
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
In Model: input size torch.Size([30, 5]) output size torch.Size([30, 2])
<torch.cuda.device object at 0x7f0707748f10>
NVIDIA A10G
Outside: input size torch.Size([30, 5]) output_size
torch.Size([30, 2])
In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
<torch.cuda.device object at 0x7f0707748f10>
NVIDIA A10G
Outside: input size torch.Size([10, 5]) output_size
torch.Size([10, 2])
2023-12-15 01:51:15,880 sagemaker-training-toolkit INFO     Waiting for the process to finish and give a return code.
2023-12-15 01:51:15,881 sagemaker-training-toolkit INFO     Done waiting for a return code. Received 0 from exiting process.
2023-12-15 01:51:15,881 sagemaker-training-toolkit INFO     Reporting training SUCCESS
```

[DataParallel docs](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)

> **Implements data parallelism at the module level**. This container parallelizes the application of the given module by splitting the input across the specified devices by chunking in the **batch dimension** (other objects will be copied once per device). In the forward pass, the module is replicated on each device, and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.

> [!WARNING]
> It is recommended to use DistributedDataParallel, instead of this class, to do multi-GPU training, even if there is only a single node. See: Use nn.parallel.DistributedDataParallel instead of multiprocessing or nn.DataParallel and Distributed Data Parallel.

## Pytorch Distributed Data Parallel

- [Introduction](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#basic-use-case) explains basics of DDP and compare it with previous Data Parallel.
- [DDP Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)

> Compared to DataParallel, DistributedDataParallel requires one more step to set up, i.e., calling init_process_group. DDP uses multi-process parallelism, and hence there is no GIL contention across model replicas. Moreover, the model is broadcast at DDP construction time instead of in every forward pass, which also helps to speed up training. DDP is shipped with several performance optimization technologies. For a more in-depth explanation, refer to this paper (VLDB’20).

Let create a similar model as before. Wrap model in DistributedDataParallel

```py
device = "cuda"
model = Model(input_size, output_size)
model = torch.nn.parallel.DistributedDataParallel(model)
model.to(device)
```

Init the processing group

```py
world_size = len(args.hosts)
host_rank = args.hosts.index(args.current_host)
print(f'host rank is {host_rank}')
dist.init_process_group(
    backend=args.backend,
    rank=host_rank,
    world_size=world_size)
```

> First wrap model in Data

```py
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
    # model = torch.nn.DataParallel(model)
    model = torch.nn.parallel.DistributedDataParallel(model)
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
```

Then create a SM training job

```py
estimator_3 = PyTorch(
    role="arn:aws:iam::392194582387:role/RoleForDataScientistUserProfile",
    entry_point="train-torch-ddp.py",
    framework_version="2.0.1",
    py_version="py310",
    instance_count=1,
    instance_type="ml.g5.12xlarge",
    hyperparameters={
      'backend': 'gloo',
      'model-type': 'custom'
    },
    distribution={
        # mpirun backend
        "pytorchddp": {"enable": True}
    },
)
```

## SageMaker SDP and SMP Library

- Apply to p4 instance only [HERE](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-data-parallel-support.html)
- DataParallel and

![IMPORTANT]

> Only support some large instances like ml.p4d.24xlarge and ml.p4de.24xlarge. Stoped supporte for P3 instances already at this moment.

First we need to modify code

```py
import torch.distributed as dist
import smdistributed.dataparallel.torch.torch_smddp

dist.init_process_group(backend="smddp")
```

Then create a SM training job

```py
from sagemaker.pytorch import PyTorch

pt_estimator = PyTorch(
    base_job_name="training_job_name_prefix",
    source_dir="subdirectory-to-your-code",
    entry_point="adapted-training-script.py",
    role="SageMakerRole",
    py_version="py310",
    framework_version="2.0.1",

    # For running a multi-node distributed training job, specify a value greater than 1
    # Example: 2,3,4,..8
    instance_count=2,

    # Instance types supported by the SageMaker data parallel library:
    # ml.p4d.24xlarge, ml.p4de.24xlarge
    instance_type="ml.p4d.24xlarge",

    # Activate distributed training with SMDDP
    distribution={ "pytorchddp": { "enabled": True } }  # mpirun, activates SMDDP AllReduce OR AllGather
    # distribution={ "torch_distributed": { "enabled": True } }  # torchrun, activates SMDDP AllGather
    # distribution={ "smdistributed": { "dataparallel": { "enabled": True } } }  # mpirun, activates SMDDP AllReduce OR AllGather
)

pt_estimator.fit("s3://bucket/path/to/training/data")
```

## Reference

- [Distributed Training Workshop](https://shashankprasanna.com/workshops/pytorch-distributed-sagemaker/2_distributed_training.html)

- [Distributed Training Workshop GitHub](https://github.com/shashankprasanna/pytorch-sagemaker-distributed-workshop.git)

- [Data Prallelism Library in SageMaker](https://aws.amazon.com/blogs/aws/managed-data-parallelism-in-amazon-sagemaker-simplifies-training-on-large-datasets/)

- [The Science Behind Amazon SM Distributed Training Engine](https://www.amazon.science/latest-news/the-science-of-amazon-sagemakers-distributed-training-engines)

- [SMDDP Distributed Data Parallel Supported Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-data-parallel-support.html)
