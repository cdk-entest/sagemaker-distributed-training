import argparse
import os
import torch.distributed as dist


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--num-gpus",
    type=int,
    default=int(os.environ['SM_NUM_GPUS'])
  )

  parser.add_argument(
    "--num-nodes",
    type=int,
    default=len(os.environ["SM_HOSTS"])
  )

  world_size = int(os.environ["SM_NUM_GPUS"]) * len(os.environ["SM_HOSTS"])

  parser.add_argument(
    "--word-size",
    type=int, 
    default=int(world_size)
  )

  args = parser.parse_args()
  return args


def train():
  args = parse_args()
  print(f"num of gpus {args.num_gpus}")
  try: 
    local_rank = os.environ["LOCAL_RANK"]
    print(f"local rank {local_rank}")
  except:
    print("there is no local rank")


if __name__=="__main__":
  train()