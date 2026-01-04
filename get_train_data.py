
import numpy as np
import sys


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def splt_data(adj_rwr, attri, Graphs_adj, look_back, size=3):
    
    nodes = adj_rwr.shape[1]

    adj_rwr_splt = np.zeros((nodes * size, int(look_back), adj_rwr.shape[2]))
    attri_splt = np.zeros((nodes * size, look_back, attri.shape[2]))
    label = np.zeros((nodes * size, nodes))

    for i in range(size):
        adj_rwr_splt[i*nodes : (i+1)*nodes, :, :] = \
            np.swapaxes(adj_rwr[i*look_back : (i+1)*look_back, :, :], 1, 0)

        attri_splt[i*nodes : (i+1)*nodes, :, :] = \
            np.swapaxes(attri[i*look_back : (i+1)*look_back, :, :], 1, 0)

        label[i * nodes:(i + 1) * nodes, :] = Graphs_adj[i + look_back, :, :]

    return adj_rwr_splt, attri_splt, label
