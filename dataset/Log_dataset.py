import os
import sys

import torch
import json
# from process_data import dataset
from collections import defaultdict
from tqdm import tqdm, trange
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from torch_geometric.data import Data


class LogDataset(Dataset):
    def __init__(self, dataset_name, list_data):
        self.len = len(list_data)
        self.list_data = list_data
        self.data = None
        if dataset_name not in ['TrainTicket']:
            s_vector_file = open('vectors/' + dataset_name + '_Sentence_vector_300d.txt', 'r')
            s_vector_lines = s_vector_file.readlines()
            self.s_features = []
            for l in s_vector_lines:
                self.s_features.append(l.split(': [')[1].strip(']\n').split(', '))  # 300 features
            self.event_num = len(s_vector_lines)

    # Two necessary methods for dataloader to use
    def __getitem__(self, index):
        if isinstance(index, slice):
            # If a slice object is passed in, process the slice operation
            start, stop, step = index.indices(len(self.list_data))
            sliced_file_paths = [self.list_data[i] for i in range(start, stop, step)]
            # Process the sliced file path list one by one and return the processed data
            processed_data = [self.process(file_path) for file_path in tqdm(sliced_file_paths, desc='Loading data: ')]
            return processed_data
        else:
            # If a single index is passed in, a single file path is processed and the processed data is returned
            file_path = self.list_data[index]
            processed_data = self.process(file_path)
            return processed_data

    def __len__(self):
        return self.len

    def process(self, file_path):
        file = open(file_path, 'r', encoding='utf-8')
        line = file.readlines()
        for i in range(len(line)):
            # print(line[i])
            'mark'
            if line[i] == 'network[son<-parent]=\n':
                start = i
            if line[i].find('node') != -1:
                end = i - 1
        # print(start)
        # print(end)
        edge_index = []
        edge_dict = [0] * self.event_num
        for i in range(start + 1, end):
            edge_data = [int(line[i].split('<-')[1].split(':')[0])]
            edge_dict[int(line[i].split('<-')[1].split(':')[0])] += 1
            edge_data.append(int(line[i].split('<-')[0]))
            edge_dict[int(line[i].split('<-')[0])] += 1
            edge_index.append(edge_data)
        for i in range(len(line)):
            # print(line[i])
            'mark'
            if line[i].find('node') != -1:
                start = i
            if line[i].find('Sequence') != -1:
                end = i - 2
        node_info_list = [0] * self.event_num
        event_odr = [0] * self.event_num
        # print(filename)
        for i in range(start + 1, end):
            # print(line[i])
            features_str = line[i].split(':')[1].strip('\n').strip()  # 26,6
            node_num = int(line[i].split(':')[0])
            features = [float(x) for x in self.s_features[node_num]]
            # p_emb = float(features_str.split(',')[1])
            # features.append(p_emb)
            event_odr[node_num] = float(features_str.split(',')[1])
            # features.append(float(edge_dict[int(line[i].split(':')[0].split('E')[1])]))
            node_info_list[node_num] = features
            # for l in range(0, len(node_info_list)):
            #     if node_info_list[l] == 0:
            #         node_info_list[l] = [0.0] * 300
        s = set()
        for p in range(len(edge_index)):
            s.add(edge_index[p][0])
            s.add(edge_index[p][1])
        s = list(sorted(s))
        for p in range(len(edge_index)):
            for q in range(len(s)):
                if edge_index[p][0] == s[q]:
                    edge_index[p][0] = q
                if edge_index[p][1] == s[q]:
                    edge_index[p][1] = q
        q = 0
        while q != len(node_info_list):
            if node_info_list[q] == 0:
                del node_info_list[q]
                continue
            q += 1
        # label
        if line[0] == 'Label=Normal\n' or line[1] == 'Label=Normal\n':
            label = torch.LongTensor([0])
        else:
            label = torch.LongTensor([1])
        node_features = torch.Tensor(node_info_list)
        src = []
        targ = []
        B = []
        for triple in edge_index:
            src.append(triple[0])
            targ.append(triple[1])
            B.append(triple)
        temp = [src, targ]
        temp = torch.LongTensor(temp)
        return Data(x=node_features, edge_index=temp, y=label)


class JsonLogDataset(LogDataset):
    def __init__(self, d, list_data):
        super().__init__(d, list_data)

    def process(self, d):
        node_features = torch.Tensor(d['node_info'])
        src = []  # source
        targ = []  # target
        B = []
        for triple in d['edge_index']:
            src.append(triple[0])
            targ.append(triple[1])
            B.append(triple)
        temp = [src, targ]
        temp = torch.LongTensor(temp)
        if d['trace_bool']:
            label = torch.LongTensor([0])
        else:
            label = torch.LongTensor([1])
        return Data(x=node_features, edge_index=temp, y=label)
