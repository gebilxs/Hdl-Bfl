import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import numpy as np
import wandb
from tqdm import tqdm

from server import Hdl
from serverbaseline import Hbase


def main(args):
    print("main start")
    if args.algorithm == 'hdl':
        server = Hdl(args,args.times)
    else:
        server = Hbase(args,args.times)
    server.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='hdl')
    parser.add_argument('--dataset', type=str, default='fmnist-100')
    parser.add_argument('--models', type=str,
                        default='resnet,shufflenet,googlenet,alexnet')
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_parties', type=int, default=20)
    parser.add_argument('--c', type=float, default=1.)
    parser.add_argument('--data_partition', type=str, default='noniid-twoclass')
    parser.add_argument('--runfile', type=str, default='cifar10_KTpFL-rsga_noniid-labeldir_20clients_C1_E20')
    # KT-pFL training params
    parser.add_argument('--public_datasize', type=int, default=3000)
    parser.add_argument('--public_dataset', type=str, default='mnist')
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--num_distill', type=int, default=1)
    parser.add_argument('--max_rounds', type=int, default=200)
    parser.add_argument('--num_classes',type=int,default=10)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    # parser.add_argument('--distill_lr', type=float, default=0.01, help='mu3 to update c')
    parser.add_argument('--output_path', type=str, default='results/debug')
    parser.add_argument('-lam', "--lamda", type=float, default=1,
                        help="Regularization weight")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    # OFA parameters
    parser.add_argument('--ofa-eps', default=[1,1,1,1], nargs='+', type=float)
    parser.add_argument('--ofa-stage', default=[1, 2, 3, 4], nargs='+', type=int)
    parser.add_argument('--ofa-loss-weight', default=1, type=float)
    parser.add_argument('--ofa-temperature', default=5, type=float)
    parser.add_argument('--loss_kd_weight', default=1, type=float)
    parser.add_argument('--loss_gt_weight', default=1, type=float)
    parser.add_argument('--temperature', default=5, type=float)
    args = parser.parse_args()
    main(args)