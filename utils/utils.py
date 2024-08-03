import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import entropy
from torch.nn.functional import cosine_similarity

from torch_geometric.loader import DataLoader
import torch
import pdb
from tqdm import tqdm
from scipy import stats
import numpy as np
import torch.nn.functional as F


def init_centers(X, K):
    embs = torch.Tensor(X)
    ind = torch.argmax(torch.norm(embs, 2, 1)).item()
    embs = embs.cuda()
    mu = [embs[ind]]
    indsAll = [ind]
    centInds = [0.] * len(embs)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = torch.cdist(mu[-1].view(1, -1), embs, 2)[0].cpu().numpy()
        else:
            newD = torch.cdist(mu[-1].view(1, -1), embs, 2)[0].cpu().numpy()
            for i in range(len(embs)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(embs[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

def graph_information_entropy(graphs):
    # Computing the information entropy of a graph
    entropies = []
    for graph in graphs:
        degree_sequence = [d for n, d in graph.degree()]
        degree_counts = np.bincount(degree_sequence)
        degree_probs = degree_counts / np.sum(degree_counts)
        ent = entropy(degree_probs)
        entropies.append(ent)
    return np.array(entropies)

def distance(X1, X2, mu):
    Y1, Y2 = mu
    X1_vec, X1_norm_square = X1
    X2_vec, X2_norm_square = X2
    Y1_vec, Y1_norm_square = Y1
    Y2_vec, Y2_norm_square = Y2
    dist = X1_norm_square * X2_norm_square + Y1_norm_square * Y2_norm_square - 2 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec)
    # Numerical errors may cause the distance squared to be negative.
    assert np.min(dist) / np.max(dist) > -1e-4
    dist = np.sqrt(np.clip(dist, a_min=0, a_max=None))
    return dist

def remove_similar_samples(graphs, threshold=0.9):
    features = [np.mean([data['feature'] for _, data in graph.nodes(data=True)], axis=0) for graph in graphs]
    similarity_matrix = cosine_similarity(features)
    to_remove = set()
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            if similarity_matrix[i, j] > threshold:
                to_remove.add(j)
    return [graph for i, graph in enumerate(graphs) if i not in to_remove]

def init_centers(X1, X2, chosen, chosen_list, mu, D2):
    if len(chosen) == 0:
        ind = np.argmax(X1[1] * X2[1])
        mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
        D2 = distance(X1, X2, mu[0]).ravel().astype(float)
        D2[ind] = 0
    else:
        newD = distance(X1, X2, mu[-1]).ravel().astype(float)
        D2 = np.minimum(D2, newD)
        D2[chosen_list] = 0
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in chosen: ind = customDist.rvs(size=1)[0]
        mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
    chosen.add(ind)
    chosen_list.append(ind)
    # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
    return chosen, chosen_list, mu, D2


class BadgeSampling:
    def __init__(self, unlabeled_pool, net, device):
        super(BadgeSampling, self).__init__()
        self.unlabeled_pool = unlabeled_pool
        self.net = net
        self.n_pool = len(unlabeled_pool)
        self.device = device

    def query(self, n=100):
        # n is the label budget
        embs, probs = self.get_embedding(self.unlabeled_pool, return_probs=True)
        embs = embs.numpy()  # Embedding after model
        probs = probs.numpy()  # Probability distribution after softmax

        # the logic below reflects a speedup proposed by Zhang et al.
        # see Appendix D of https://arxiv.org/abs/2306.09910 for more details
        m = self.n_pool
        mu = None
        D2 = None
        chosen = set()
        chosen_list = []
        emb_norms_square = np.sum(embs ** 2, axis=-1)
        max_inds = np.argmax(probs, axis=-1)

        probs = -1 * probs
        probs[np.arange(m), max_inds] += 1
        prob_norms_square = np.sum(probs ** 2, axis=-1)
        for _ in range(n):
            chosen, chosen_list, mu, D2 = init_centers((probs, prob_norms_square), (embs, emb_norms_square), chosen,
                                                       chosen_list, mu, D2)
        return chosen_list

    def get_embedding(self, data_list, return_probs=False):
        loader_te = DataLoader(data_list, batch_size=1000, num_workers=0, pin_memory=False,
                               shuffle=False)
        self.net.eval()
        embedding = torch.zeros([self.n_pool, 256])  #
        probs = torch.zeros(self.n_pool, 2)
        with torch.no_grad():
            for idx, data in enumerate(tqdm(loader_te, desc='Badge selecting', position=0, leave=True)):
                data = data.to(self.device)
                out, e1 = self.net(data)
                idxs = torch.LongTensor(np.arange(1000 * idx, 1000 * idx + len(data)))
                embedding[idxs] = e1.data.cpu()
                if return_probs:
                    pr = F.softmax(out, 1)
                    probs[idxs] = pr.data.cpu()
        if return_probs: return embedding, probs
        return embedding
