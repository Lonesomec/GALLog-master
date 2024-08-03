import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import shuffle
from torch import optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool
import torch.nn.functional as F
from model import GCN, GAT_NET
from tqdm import tqdm
import warnings
from settings import *
from utils import pyg_data

warnings.simplefilter("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Ablation Task for GALLog")
    # [HDFS, BGL, TrainTicket, Spirit, ThunderBird, Agavue]
    args, dataset_path = init_dataset('BGL', parser)  #
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_ratio = 0.05
    is_time_split = enable_time_split()
    enhancement_enable = enable_enhancement()
    print(args)

    # load the pooling
    start_handle_time = time.perf_counter()
    raw_dataset_train, dataset_test, data_len = pyg_data.load_dataset(args.dataset, divide=0.7,
                                                                      is_time_split=is_time_split, path=dataset_path)
    dataset_train = shuffle(raw_dataset_train)[:int(label_ratio * data_len)]
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=0, pin_memory=False, shuffle=True)
    del dataset_test
    end_handle_time = time.perf_counter()
    print(f"Dataset loading time: {end_handle_time - start_handle_time:0.2f} s")

    ft_size = raw_dataset_train[0].x.shape[1]
    model = GAT_NET(ft_size, 512, 2, 2).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimiser, [50, 100, 120, 150], gamma=0.5)
    c_xent = nn.CrossEntropyLoss(weight=torch.tensor([1., args.imbalance_weight]).to(device))
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=0, pin_memory=False,
                              shuffle=True)
    del dataset_train
    F1s = []
    Epochs = []
    for epoch in tqdm(range(1, args.nb_epochs + 1), total=args.nb_epochs, desc="Epoch: ", position=0, leave=True):
        loss_e = 0
        model.train()
        for data in train_loader:
            optimiser.zero_grad()
            data = data.to(device)
            outputs, _ = model(data)
            loss = c_xent(outputs, data.y)
            loss.backward()
            optimiser.step()
            loss_e += loss
        # 验证
        if epoch == args.nb_epochs:
            model.eval()
            with torch.no_grad():
                # 测试模型
                TP = FP = FN = TN = 0
                for data in test_loader:
                    data = data.to(device)
                    outputs, _ = model(data)
                    _, preds = torch.max(outputs, 1)
                    for k in range(len(preds)):
                        if preds[k] == 1 and data.y[k] == 1:
                            TP += 1
                        if preds[k] == 0 and data.y[k] == 1:
                            FN += 1
                        if preds[k] == 1 and data.y[k] == 0:
                            FP += 1
                        if preds[k] == 0 and data.y[k] == 0:
                            TN += 1
            print(TP)
            print(FP)
            print(TN)
            print(FN)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1 = 2 * precision * recall / (precision + recall)
            F1s.append(F1)
            Epochs.append(epoch)
            ACC = (TP + TN) / (TP + FP + TN + FN)
            print('Test set P: {:.2f}%'.format(100. * precision))
            print('Test set R: {:.2f}%'.format(100. * recall))
            print('Test set F1: {:.2f}%'.format(100. * F1))
            print('Test set ACC: {:.2f}%'.format(100. * ACC))
            print('Finished testing.')
        print('loss : ' + str(loss_e / (epoch + 1)))
        scheduler.step()
    # plt.figure()
    # plt.plot(Epochs, F1s, label='Test F1')
    # plt.xlabel('Epoch')
    # plt.ylabel('F1-score')
    # plt.legend()
    # plt.show()
