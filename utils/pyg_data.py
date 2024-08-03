import json
import os
import random
import sys

import numpy as np
import torch
from sklearn.utils import shuffle
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected

from dataset.Log_dataset import LogDataset, JsonLogDataset


def get_file_path(root_path, file_list, dir_list):
    """
        Gets all files under this file address
    """
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path, dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)


def load_dataset(dataset, divide, is_time_split, path):
    if dataset == 'TrainTicket':
        with open(path, 'r') as f:
            data = json.load(f)
        list_file = shuffle(data)
        list_train = list_file[:int(len(list_file) * divide)]
        list_test = list_file[int(len(list_file) * divide):]
        print(len(list_train))
        print(len(list_test))
        dataset_train = JsonLogDataset(dataset, list_train)
        dataset_test = JsonLogDataset(dataset, list_test)
    else:
        list_file = []
        get_file_path(path, list_file, [])
        if is_time_split:
            list_file.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))  # Sort the file list by file name
        else:
            list_file = shuffle(list_file)
        list_train = list_file[:int(len(list_file) * divide)]
        list_test = list_file[int(len(list_file) * divide):]
        print(len(list_train))
        print(len(list_test))
        # Load data
        dataset_train = LogDataset(dataset, list_train)
        dataset_test = LogDataset(dataset, list_test)
    return dataset_train, dataset_test, len(list_file)
