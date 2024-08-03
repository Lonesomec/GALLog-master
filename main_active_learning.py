import copy
import logging
import random
import sys
import time
from math import log

from termcolor import cprint
from torch_geometric.data import Data
import numpy as np
import torch
from matplotlib import pyplot as plt
from networkx import vf2pp_is_isomorphic, is_isomorphic
from sklearn.utils import shuffle
from torch import optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import torch.nn as nn
from VGAE_train import vgae_train, vgae_generate
from model import GCN, GAT_NET, VGAEModel, VariationalGraphEncoder
import argparse
from tqdm import tqdm
import warnings
from settings import *

from utils import pyg_data
from utils.utils import *

warnings.simplefilter("ignore")

if __name__ == '__main__':
    # log config
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = 'log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # initial the arguments
    parser = argparse.ArgumentParser("Active Learning Task for Log Anomaly Detection(/)")
    # [HDFS, BGL, TrainTicket, Spirit, ThunderBird, Agavue]
    args, dataset_path = init_dataset('BGL', parser)  #
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_time_split = enable_time_split()
    enhancement_enable = enable_enhancement()
    logger.info(args)

    # load the pooling
    start_handle_time = time.perf_counter()
    raw_dataset_train, dataset_test, data_len = pyg_data.load_dataset(args.dataset, divide=0.7,
                                                                      is_time_split=is_time_split, path=dataset_path)
    labeled_dataset_pooling: list[Data] = raw_dataset_train[:int(args.budget * 1/3 * data_len)]
    unlabeled_dataset_pooling = raw_dataset_train[int(args.budget * 1/3 * data_len):]
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=0, pin_memory=False, shuffle=True)
    end_handle_time = time.perf_counter()
    print(f"Dataset loading time: {end_handle_time - start_handle_time:0.2f} s")

    budget = int(args.budget * 2/3 * data_len)
    repr_budget = budget * 0.5
    ft_size = labeled_dataset_pooling[0].x.shape[1]  # embedding dim
    encoder = VariationalGraphEncoder(ft_size, 512, 256).to(device)
    VGAE_model = VGAEModel(encoder, ft_size).to(device)
    VGAE_optimizer = optim.Adam(VGAE_model.parameters(), lr=0.0001)
    model = GAT_NET(ft_size, 512, 2, 2).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimiser, [50, 100, 120, 150], gamma=0.5)
    c_xent = nn.CrossEntropyLoss(weight=torch.tensor([1., args.imbalance_weight]).to(device))

    F1s = []
    Ps = []
    Rs = []
    Epochs = []
    g_data = []
    for epoch in tqdm(range(1, args.nb_epochs + 1), desc="Epoch: "):
        # Output the information of labeled sample pool and unlabeled sample pool
        logger.info('The number of budget: ' + str(budget))
        logger.info('The number of labeled_dataset_pooling: ' + str(len(labeled_dataset_pooling)))
        logger.info('The number of unlabeled_dataset_pooling: ' + str(len(unlabeled_dataset_pooling)))

        labeled_dataset_pooling = shuffle(labeled_dataset_pooling)
        unlabeled_dataset_pooling = shuffle(unlabeled_dataset_pooling)
        # Create an index mapping
        index_map = list(range(len(unlabeled_dataset_pooling)))
        train_loader = DataLoader(labeled_dataset_pooling, batch_size=args.batch_size, num_workers=0, pin_memory=False,
                                  shuffle=False)
        val_loader = DataLoader(unlabeled_dataset_pooling, batch_size=args.batch_size, num_workers=0, pin_memory=False,
                                shuffle=False)
        loss_e = 0
        increase_samples = 0
        to_remove = []
        to_remove_outs = []

        f = args.threshold * log(3 * epoch + 10, 10)  #
        model.train()
        for data in train_loader:
            optimiser.zero_grad()
            data = data.to(device)
            outputs, _ = model(data)
            loss = c_xent(outputs, data.y)
            loss.backward()
            optimiser.step()
            loss_e += loss
        # Select samples for labeling
        if budget < 0:
            cprint('(Fatel Error : budget < 0 !)', 'red')
            sys.exit()
        model.eval()
        del train_loader
        # Uncertainty Strategy
        if epoch >= args.uncertainty_epoch:  #
            temp_budget = budget
            for batch_idx, data in enumerate(tqdm(val_loader, desc='Selecting samples...', position=0, leave=True)):
                if temp_budget == 0:
                    break
                data = data.to(device)
                outputs, embs = model(data)
                # Select the sample with the smallest absolute difference in predicted probability
                p_outputs = F.softmax(outputs)  # [bs,2]
                # Calculate the difference between the model's predicted class probabilities
                abs_diff = torch.abs(p_outputs[:, 0] - p_outputs[:, 1])
                sorted_values, sorted_indices = torch.sort(abs_diff, descending=False)
                # Samples smaller than the threshold are fuzzy samples. Get all subscripts
                indices_less_than_threshold = sorted_indices[sorted_values < f]
                # The budget is not enough
                if temp_budget < len(indices_less_than_threshold):
                    indices_less_than_threshold = indices_less_than_threshold[:temp_budget]
                for idx in indices_less_than_threshold:
                    data[idx] = data[idx].to(device)
                    # Find the index in the DataLoader and map back to the original dataset
                    original_idx = index_map[batch_idx * val_loader.batch_size + idx]
                    to_remove.append(original_idx)
                temp_budget -= len(indices_less_than_threshold)
        # Representative strategy
        elif epoch <= args.representation_epoch:
            badge_n = int(repr_budget / args.representation_epoch)  # 0.1 in total
            if budget >= badge_n and budget != 0:
                strategy = BadgeSampling(unlabeled_dataset_pooling, model, device)
                to_remove = strategy.query(badge_n)
                if enhancement_enable:
                    # Training VGAE
                    for t_idx in to_remove:
                        g_data.append(unlabeled_dataset_pooling[t_idx])
                    # A deep copy is required
                    if epoch == args.representation_epoch:
                        VGAE_model = vgae_train(copy.deepcopy(g_data), VGAE_model, VGAE_optimizer, device, 128)
                        # torch.save(VGAE_model, 'GAE_model.pth')
                        del g_data
        del val_loader
        # Data enhancement
        if enhancement_enable and len(to_remove) != 0 and epoch > args.enhancement_epoch:
            #  Delete elements (from the back to the front to avoid index changes)
            keep_to_remove = []  #
            count_re = 0
            for p in tqdm(range(len(to_remove)), position=0, leave=True, desc='Data enhancement...'):
                keep_to_remove.append(to_remove[p])  # Add the current element to the list that needs to be retained
                for q in range(p + 1, len(to_remove)):
                    if is_isomorphic(to_networkx(unlabeled_dataset_pooling[to_remove[p]]),
                                     to_networkx(unlabeled_dataset_pooling[to_remove[q]])):  # 4425 4737
                        count_re += 1
                        keep_to_remove.pop()  # Delete the duplicate elements added in the previous step
                        break
            to_remove = keep_to_remove
            # Expand this part of low-confidence samples and use VGAE to get new data
            temp_g_data = []
            for tt_idx in to_remove:
                temp_g_data.append(unlabeled_dataset_pooling[tt_idx])
            generated_gdata = vgae_generate(copy.deepcopy(temp_g_data), VGAE_model, device)
            labeled_dataset_pooling = labeled_dataset_pooling + generated_gdata
        budget -= len(to_remove)
        increase_samples += len(to_remove)
        for idx in sorted(to_remove, reverse=True):
            labeled_dataset_pooling.append(unlabeled_dataset_pooling[idx])
            del unlabeled_dataset_pooling[idx]
        torch.cuda.empty_cache()
        # testing
        if epoch == args.nb_epochs:
            TP = FP = FN = TN = 0
            for data in tqdm(test_loader, desc='Testing...', position=0, leave=True):
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
            print(f'TP:{TP} FP:{FP} TN: {TN} FN:{FN}')
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1 = 2 * precision * recall / (precision + recall)
            ACC = (TP + TN) / (TP + FP + TN + FN)
            Epochs.append(epoch)
            F1s.append(F1)
            Ps.append(precision)
            Rs.append(recall)
            cprint('Test set P: {:.2f}%'.format(100. * precision), 'red')
            cprint('Test set R: {:.2f}%'.format(100. * recall), 'red')
            cprint('Test set F1: {:.2f}%'.format(100. * F1), 'red')
            cprint('Test set ACC: {:.2f}%'.format(100. * ACC), 'red')
            print('Finished testing.')
        logger.info('Increase samples:' + str(increase_samples))
        logger.info('loss : ' + str(float(loss_e) / (epoch + 1)))
        scheduler.step()
        torch.cuda.empty_cache()
    # sys.exit()
    logger.info('All labeled instances ratio: ' + str(len(labeled_dataset_pooling) / data_len))
    max_index = F1s.index(max(F1s))
    logger.info('Best P: {:.2f}%'.format(100. * Ps[max_index]))
    logger.info('Best R: {:.2f}%'.format(100. * Rs[max_index]))
    logger.info('Best F1: {:.2f}%'.format(100. * F1s[max_index]))
    logger.info('Finished testing.\n')
    # plt.figure()
    # plt.plot(Epochs, F1s, label='Test F1')
    # plt.xlabel('Epoch')
    # plt.ylabel('F1-score')
    # plt.legend()
    # plt.show()
    # path = 'model' + args.dataset + '.pth'
    #
    # torch.save(model, path)
    # model = torch.load(path)
