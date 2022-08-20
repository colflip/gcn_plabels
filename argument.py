import argparse
import random

import numpy as np
import torch


def args_parse():
    parser = argparse.ArgumentParser(description='Graph arguments.')
    parser.add_argument('--dataset', type=str, default='Cora', help='Cora/CiteSeer/Photo/cs/Computers/CoraFull')
    parser.add_argument('--spt_num', type=int, default=5, help='support set number')
    parser.add_argument('--qry_num', type=int, default=9, help='query set number')
    parser.add_argument('--self_train_num', type=int, default=30, help='Pseudo-label node number')
    parser.add_argument('--hid_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--mid_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--l_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--w_decay', type=float, default=0.000005, help='learning rate')
    parser.add_argument('--cuda', type=int, default=1, help='cuda number')
    parser.add_argument('--n_way', type=int, default=2, help='default test_classes')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--sample_times', type=int, default=100)
    parser.add_argument('--d_rate', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    params = {
        'dataset': args.dataset,
        'spt_num': args.spt_num,
        'qry_num': args.qry_num,
        'self_train_num': args.self_train_num,
        'epoch': args.epoch,
        'l_rate': args.l_rate,
        'w_decay': args.w_decay,
        'cuda': args.cuda,
        'n_way': args.n_way,
        'device': device,
        'hid_dim': args.hid_dim,
        'mid_dim': args.mid_dim,
        'k': args.k,
        'sample_times': args.sample_times,
        'd_rate': args.d_rate,
        'patience': args.patience
    }
    return params


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
