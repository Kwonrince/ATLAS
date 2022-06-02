import os
import sys
import random
import argparse

import torch
import numpy as np
from trainer import Trainer
import torch.multiprocessing as mp
from transformers import BartTokenizer


def run(args):
    args.devices = [int(gpu) for gpu in args.devices.split("_")]
    ngpus_per_node = len(args.devices)

    assert ngpus_per_node <= torch.cuda.device_count(
    ), "The number of GPU exceeds max capacity"
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    
def worker(gpu, ngpus_per_node, args):
    trainer = Trainer(args)
    trainer.make_model_env(gpu, ngpus_per_node)
    trainer.train()
    
def main(arguments):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str,
                        default="./cnndm",
                        help="dataset folder location(train, val)")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--batch_size", help="total batch size",
                        type=int, default=4)
    parser.add_argument("--lr", help="initial learning rate",
                        type=float, default=3e-5)
    parser.add_argument("--max_length", help="max length for input document",
                        default=1024, type=int)
    parser.add_argument("--max_decode_step", type=int,
                        default=128, help="maximum decode step")
    parser.add_argument('--num_epochs',
                        help='Number of epochs to train',
                        type=int, default=20)
    parser.add_argument('--num_warmup_steps',
                        help='Number of warmup steps ratio to train',
                        type=float, default=0.1)
    parser.add_argument('--triplet',
                        help='Use triplet network',
                        type=bool, default=True)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--model_dir", type=str, default="./save")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--devices", default='0_1_2_3', type=str,
                        help="gpu device ids to use, concat with '_', ex) '0_1_2_3'")
    parser.add_argument("--workers", type=int,
                        default=4, help="Number of processes(workers) per node."
                        "It should be equal to the number of gpu devices to use in one node")
    parser.add_argument("--world_size", default=1,
                        help="Number of total workers. Initial value should be set to the number of nodes."
                             "Final value will be Num.nodes * Num.devices")
    parser.add_argument("--rank", default=0,
                        help="The priority rank of current node.")
    parser.add_argument("--dist_backend", default="nccl",
                        help="Backend communication method. "
                             "NCCL is used for DistributedDataParallel")
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:29152",
                        help="DistributedDataParallel server")
    parser.add_argument("--gpu", default=None, type=int,
                        help="Manual setting of gpu device. If it is not None, all parallel processes are disabled")
    parser.add_argument("--distributed", action="store_true",
                        help="Use multiprocess distribution or not")
    parser.add_argument("--random_seed", default=1004,type=int,
                        help="Random state(seed)")
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--ablation",action="store_true")
    args = parser.parse_args(arguments)
    
    
    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    args.tokenizer = tokenizer
    args.vocab_size = len(tokenizer.get_vocab())

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    run(args)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))