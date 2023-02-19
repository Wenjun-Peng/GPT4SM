import torch

from tqdm.auto import tqdm
from pathlib import Path

import torch.multiprocessing as mp

import os
import numpy as np
import utils
import random
from parameters import parse_args

from trainer.GPT import GPTTrainer
from trainer.PLMGPT import PLMGPTTrainer
from trainer.CopyGPT import CopyGPTTrainer


def train(rank, args, world_size):
    utils.setup(rank, world_size, args)
    if rank == 0:
        utils.setuplogger()

    trainer = eval(args.trainer)(rank, args)
    trainer.train()


def test(rank, args, world_size):
    utils.setup(rank, world_size, args)
    if rank == 0:
        utils.setuplogger()
        
    trainer = eval(args.trainer)(rank, args)
    trainer.test()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    args = parse_args()
    seed_everything(args.seed)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    if args.enable_ddp:
        assert (
            n_gpus >= args.world_size
        ), f"Requires at least {args.world_size} GPUs to run, but got {n_gpus}"

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if "train" in args.mode:
        if args.enable_ddp:
            mp.spawn(
                train, args=(args, args.world_size), nprocs=args.world_size, join=True
            )
        else:
            train(0, args, 1)
    if "test" in args.mode:
        if args.enable_ddp:
            mp.spawn(
                test, args=(args, args.world_size), nprocs=args.world_size, join=True
            )
        else:
            test(0, args, 1)