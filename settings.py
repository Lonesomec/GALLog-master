import argparse
import os
import random
import sys
import time

from func_timeout import func_timeout
from func_timeout import FunctionTimedOut
import numpy as np
import torch
from termcolor import colored, cprint


def init_dataset(dataset_name: str, parser):
    if dataset_name == 'HDFS':
        args = HDFS(parser)
        path = r'G:\delete\Log_dataset\HDFS\HDFS_session'
    elif dataset_name == 'BGL':
        args = BGL(parser)
        path = 'dataset/BGL'
    elif dataset_name == 'Agavue':
        args = Agavue(parser)
        path = r'G:\delete\Log_dataset\Agavue\Agavue(session)'
    elif dataset_name == 'Spirit':
        args = Spirit(parser)
        path = r'G:\delete\Log_dataset\Spirit\Spirit(time)50'
    elif dataset_name == 'ThunderBird':
        args = Tb(parser)
        path = r'G:\delete\Log_dataset\ThunderBird\Tb(time)50'
    elif dataset_name == 'TrainTicket':
        args = Ticket(parser)
        path = r'G:\delete\Log_dataset\TrainTicket\ticket.jsons'
    else:
        print('unsupported dataset!')
        sys.exit()
    return args, path


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def timeout_func():
    strs = input(
        colored('Enter any character to disable the data augmentation, otherwise it will automatically run after 5s!\n',
                'red'))


def enable_time_split():
    select = input('Choose the split strategy: 1.time. others. random\n')
    if select == '1':
        cprint('Selection strategy : chronological', 'red')
        return True
    else:
        cprint('Selection strategy : random', 'red')
        return False


def enable_enhancement():
    try:
        func_timeout(5, timeout_func, args=())
        print(colored('Data augmentation disabled!', 'red'))
        return False
    except FunctionTimedOut as e:
        print(colored('Data augmentation enabled!', 'red'))
        return True


def HDFS(parser):
    parser.add_argument('--dataset', type=str, default="HDFS")
    parser.add_argument('--seed', type=int, default=4, help='seed')
    parser.add_argument('--batch_size', default=128, type=int, help='')
    parser.add_argument('--nb_epochs', default=100, type=int, help='Training epoch')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--threshold', default=0.125, type=float, help='abs threshold value coefficient')
    parser.add_argument('--budget', default=0.05, type=float, help='budget ratio')
    parser.add_argument('--imbalance_weight', default=5, type=int, help='')
    parser.add_argument('--representation_epoch', default=5, type=int,
                        help='dividing line for representation strategy')
    parser.add_argument('--uncertainty_epoch', default=10, type=int, help='dividing line for uncertainty strategy')
    parser.add_argument('--enhancement_epoch', default=5, type=int, help='dividing line for data enhancement')
    parser.add_argument('--uncertainty_interval', default=5, type=int, help='')
    parser.add_argument('--readout', default='max')
    args = parser.parse_args()
    return args


def BGL(parser):
    parser.add_argument('--dataset', type=str, default="BGL",
                        help='Choice:[hdfs, bgl, ticket, spirit, tb, openstack, hadoop, hades, agavue]')
    parser.add_argument('--seed', type=int, default=3, help='seed') # 5for random
    parser.add_argument('--batch_size', default=128, type=int, help='')
    parser.add_argument('--nb_epochs', default=100, type=int, help='Training epoch')
    parser.add_argument('--lr', default=0.0007, type=float, help='Learning rate')
    parser.add_argument('--threshold', default=0.08, type=float, help='abs threshold value coefficient')
    parser.add_argument('--budget', default=0.05, type=float, help='budget ratio')
    parser.add_argument('--imbalance_weight', default=5, type=int, help='')
    parser.add_argument('--representation_epoch', default=5, type=int,
                        help='dividing line for representation strategy')
    parser.add_argument('--uncertainty_epoch', default=10, type=int, help='dividing line for uncertainty strategy')
    parser.add_argument('--enhancement_epoch', default=5, type=int, help='dividing line for data enhancement')
    parser.add_argument('--uncertainty_interval', default=5, type=int, help='')
    parser.add_argument('--readout', default='max')
    args = parser.parse_args()
    return args


def Agavue(parser):
    parser.add_argument('--dataset', type=str, default="Agavue",
                        help='Choice:[hdfs, bgl, ticket, spirit, tb, openstack, hadoop, hades, agavue]')
    parser.add_argument('--seed', type=int, default=4, help='seed')
    parser.add_argument('--batch_size', default=128, type=int, help='')
    parser.add_argument('--nb_epochs', default=150, type=int, help='Training epoch')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--threshold', default=0.125, type=float, help='abs threshold value coefficient')
    parser.add_argument('--budget', default=0.05, type=float, help='budget ratio')
    parser.add_argument('--imbalance_weight', default=5, type=int, help='')
    parser.add_argument('--representation_epoch', default=5, type=int,
                        help='dividing line for representation strategy')
    parser.add_argument('--uncertainty_epoch', default=10, type=int, help='dividing line for uncertainty strategy')
    parser.add_argument('--enhancement_epoch', default=5, type=int, help='dividing line for data enhancement')
    parser.add_argument('--uncertainty_interval', default=5, type=int, help='')
    parser.add_argument('--readout', default='max')
    args = parser.parse_args()
    return args

def Spirit(parser):
    parser.add_argument('--dataset', type=str, default="Spirit",
                        help='Choice:[hdfs, bgl, ticket, spirit, tb, openstack, hadoop, hades, agavue]')
    parser.add_argument('--seed', type=int, default=5, help='seed')
    parser.add_argument('--batch_size', default=32, type=int, help='') # raw=32
    parser.add_argument('--nb_epochs', default=200, type=int, help='Training epoch')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--threshold', default=0.08, type=float, help='abs threshold value coefficient')
    parser.add_argument('--budget', default=0.05, type=float, help='budget ratio')
    parser.add_argument('--imbalance_weight', default=1, type=int, help='')
    parser.add_argument('--representation_epoch', default=5, type=int,
                        help='dividing line for representation strategy')
    parser.add_argument('--uncertainty_epoch', default=10, type=int, help='dividing line for uncertainty strategy')
    parser.add_argument('--enhancement_epoch', default=5, type=int, help='dividing line for data enhancement')
    parser.add_argument('--uncertainty_interval', default=5, type=int, help='')
    parser.add_argument('--readout', default='max')
    args = parser.parse_args()
    return args


def Tb(parser):
    parser.add_argument('--dataset', type=str, default="ThunderBird",
                        help='Choice:[hdfs, bgl, ticket, spirit, tb, openstack, hadoop, hades, agavue]')
    parser.add_argument('--seed', type=int, default=13, help='seed')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--nb_epochs', default=200, type=int, help='Training epoch')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--threshold', default=0.15, type=float, help='abs threshold value coefficient')
    parser.add_argument('--budget', default=0.05, type=float, help='budget ratio')
    parser.add_argument('--imbalance_weight', default=1, type=int, help='')
    parser.add_argument('--representation_epoch', default=5, type=int,
                        help='dividing line for representation strategy')
    parser.add_argument('--uncertainty_epoch', default=10, type=int, help='dividing line for uncertainty strategy')
    parser.add_argument('--enhancement_epoch', default=5, type=int, help='dividing line for data enhancement')
    parser.add_argument('--uncertainty_interval', default=5, type=int, help='')
    parser.add_argument('--readout', default='max')
    args = parser.parse_args()
    return args


def Ticket(parser):
    parser.add_argument('--dataset', type=str, default="TrainTicket",
                        help='Choice:[hdfs, bgl, ticket, spirit, tb, openstack, hadoop, hades, agavue]')
    parser.add_argument('--seed', type=int, default=6, help='seed')
    parser.add_argument('--batch_size', default=16, type=int, help='')
    parser.add_argument('--nb_epochs', default=200, type=int, help='Training epoch')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')# raw=0.0001
    parser.add_argument('--threshold', default=0.05, type=float, help='abs threshold value coefficient')
    parser.add_argument('--budget', default=0.05, type=float, help='budget ratio')
    parser.add_argument('--imbalance_weight', default=1, type=int, help='')
    parser.add_argument('--representation_epoch', default=5, type=int,
                        help='dividing line for representation strategy')
    parser.add_argument('--uncertainty_epoch', default=10, type=int, help='dividing line for uncertainty strategy')
    parser.add_argument('--enhancement_epoch', default=5, type=int, help='dividing line for data enhancement')
    parser.add_argument('--uncertainty_interval', default=5, type=int, help='')
    parser.add_argument('--readout', default='max')
    args = parser.parse_args()
    return args
