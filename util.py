import random

import numpy as np
import torch
import torch.nn as nn
import copy

def one_hot(l, classnum=1):  #classnum fix some special case
    one_hot_l = np.zeros((len(l), max(l.max() + 1, classnum)))
    for i in range(len(l)):
        one_hot_l[i][l[i]] = 1
    return one_hot_l


def setup_seed(seed):
    """
    CRITICAL FIX: Enhanced seed setup for strict reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: torch.use_deterministic_algorithms(True) may cause errors with some ops
    # Uncomment only if all operations support deterministic mode:
    # torch.use_deterministic_algorithms(True)


def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.Linear:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Linear:
            l2_loss.append((module.weight**2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


def aggregate_nighod(adj, feature, num_nig=4):
    '''
    adj: T,N,N
    feature:T,N,N
    '''
    T = adj.shape[0]
    for t in range(T):
        addr = np.where(adj[0] > 0)[0]
        aa = addr
        for t in range(T):
            addr1 = np.random.choice(addr[0], int(nd_avg))

    return

def remove_null_node(Graphs):
    Graphs = np.swapaxes(Graphs, 0, 1)
    now_adj = Graphs[:, 0, :].copy()
    for i in range(1, Graphs.shape[1]):  #time_steps
        now_adj += Graphs[:, i, :].copy()
    d = np.sum(now_adj, axis=1)
    non_zero_index = np.nonzero(d)[0]
    Graphs = Graphs[non_zero_index, :, :]
    Graphs = Graphs[:, :, non_zero_index]
    Graphs = np.swapaxes(Graphs, 0, 1)    
    return Graphs, non_zero_index