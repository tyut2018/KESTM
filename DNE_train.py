import os
import random
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from algorithms_functio import merge
from evaluate import evaluateDynamicLinkPrediction
from model import AutoEnmodel, Weighted_mse_x
from RWR import get_topo, meanstd_normalization_tensor
from util import *

writer = SummaryWriter()
torch.cuda.empty_cache()


def train_one_epoch(epoch, model, loader, val_loader, optimizer, device):
    t = time.time()
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    # all_pred = []
    # all_lables = []
    for step, (data1, data2, labels) in enumerate(loader):
        #all_lables.append(labels)
        input1 = data1.to(device).to(torch.float32)  
        input2 = data2.to(device).to(torch.float32)  
        labels = labels.to(device).to(torch.long)

        optimizer.zero_grad()
        _, output = model(input1, input2)
        #_, output = model(input1)
        loss_train = loss_func(output, labels)

        l1_, l2_ = torch.tensor([0],
                                dtype=torch.float32).to(device), torch.tensor(
                                    [0], dtype=torch.float32).to(device)
        l1_ = l1_regularization(model, 1e-6)
        l2_ = l2_regularization(model, 1e-6)

        loss = loss_train + l1_ + l2_
        #loss = loss_train  + l2_
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += float(loss.item())

        pred_labels = torch.argmax(output, axis=1)
        #all_pred.append(pred_labels)
        correct += (pred_labels == labels).sum().float()
        total += len(labels)

    writer.add_scalar("Loss/train", train_loss / len(loader), epoch + 1)
    #float(correct*100)/float(BATCH_SIZE)*(batch_idx+1)
    acc_train = (correct / total).cpu().detach().data.numpy()
   
    model.eval()
    for data1_val, data2_val, labels_val in val_loader:
        #all_lables.append(labels)
        input1_val = data1_val.to(device).to(torch.float32) 
        input2_val = data2_val.to(device).to(torch.float32)  
        labels_val = labels_val.to(device).to(torch.long)
        _, output_val = model(input1_val, input2_val)
        #_, output_val = model(input1_val)
        loss_val = loss_func(output_val, labels_val)

    acc_val = metrics.accuracy_score(
        labels_val.cpu().detach().numpy(),
        torch.argmax(output_val, axis=1).cpu().detach().numpy())

    del output
    
    print('Epoch :%4d' % (epoch + 1), '|', 'Loss_train:%.4f' % loss_train.data,
          '|', 'Acc_train:%.4f' % acc_train, '|',
          'Loss_val:%.4f' % loss_val.data, 'Acc_val:%.4f' % acc_val, '|',
          'Time:%.4fs' % (time.time() - t), '|')
    writer.add_scalars("Metrics/train", {
        'acc_train': acc_train,
        'acc_val': acc_val
    }, epoch + 1)
    return acc_val


def test(epoch, model, test_loader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0

        for data1_test, data2_test, labels_test in test_loader:
            input1_test = data1_test.to(device).to(torch.float32)  
            input2_test = data2_test.to(device).to(torch.float32) 
            labels_test = labels_test.to(device).to(torch.long)
            _, output_test = model(input1_test, input2_test)
            #_, output_test = model(input1_test)
            loss_test = loss_func(output_test, labels_test)
            pred_labels = torch.argmax(output_test, axis=1)

        acc_test = metrics.accuracy_score(labels_test.cpu().detach().numpy(),
                                          pred_labels.cpu().detach().numpy())
        f1_test = metrics.f1_score(labels_test.cpu().detach().numpy(),
                                   pred_labels.cpu().detach().numpy(),
                                   average='weighted')
        auc_test = metrics.roc_auc_score(one_hot(
            labels_test.cpu().detach().numpy()),
                                         output_test.cpu().detach().numpy(),
                                         multi_class='ovr',
                                         average='weighted')
        writer.add_scalar("test_loss/epoch", loss_test, epoch + 1)
        writer.add_scalars("Metrics/test", {
            'acc_test': acc_test,
            'f1_test': f1_test,
            'auc_test': auc_test
        }, epoch + 1)
        #print('\n')
        print('Epoch :%4d' % (epoch + 1), '|',
              'Loss_test_:%.4f' % loss_test.data, '|', 'ACC:', acc_test, 'F1:',
              f1_test, 'AUC:', auc_test)
        #print('\n')
        return test_loss, acc_test, f1_test, auc_test


if __name__ == '__main__':
   
    parser = ArgumentParser(
        description='Learns node embeddings for a sequence of graph snapshots')
    parser.add_argument('-t',
                        '--testDataType',
                        default='DBLP',
                        type=str,
                        help='Type of data to test the code')
    parser.add_argument('-c',
                        '--criteria',
                        default='degree',
                        type=str,
                        help='Node Migration criteria')
    parser.add_argument(
        '-rc',
        '--criteria_r',
        default=True,
        type=bool,
        help='Take highest centrality measure to perform node migration')
    parser.add_argument('-l',
                        '--timelength',
                        default=10,
                        type=int,
                        help='Number of time series graph to generate')
    parser.add_argument('-lb',
                        '--lookback',
                        default=10,
                        type=int,
                        help='number of lookbacks')
    parser.add_argument('-beta',
                        '--beta',
                        default=5,
                        type=int,
                        help='number of Loss function argument')
    parser.add_argument('-iter',
                        '--epochs',
                        default=1000,
                        type=int,
                        help='number of epochs')
    parser.add_argument('-emb',
                        '--emb_d',
                        default=16,
                        type=int,
                        help='embedding dimension')
    parser.add_argument('-rd',
                        '--resultdir',
                        type=str,
                        default='./results_link_all',
                        help="result directory name")
    parser.add_argument('-nb',
                        '--node_numb',
                        default=6606,
                        type=int,
                        help='node for test data')
    parser.add_argument('-eta',
                        '--learningrate',
                        default=1e-3,
                        type=float,
                        help='learning rate')
    parser.add_argument('-bs',
                        '--batch',
                        default=6606,
                        type=int,
                        help='batch size')
    parser.add_argument('-ht',
                        '--hypertest',
                        default=0,
                        type=int,
                        help='hyper test')
    parser.add_argument('-fs',
                        '--show',
                        default=0,
                        type=int,
                        help='show figure ')
    parser.add_argument('-exp',
                        '--exp',
                        default='lp',
                        type=str,
                        help='experiments (lp, emb)')
    parser.add_argument('-n_aeunits',
                        '--n_aeunits',
                        default=[500, 300],
                        type=list,
                        help='AEdense_arg')
    parser.add_argument('-ts',
                        '--trainsize',
                        default=3,
                        type=int,
                        help='train_size')
    parser.add_argument('--data',
                        type=str,
                        default='./data/DBLP5.npz',
                        help='path to dataset npz file')
    args = parser.parse_args()

    
    setup_seed(1234)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")  
 
    # Load dataset from relative path
    filename = args.data
    file = np.load(filename)
    Features = file['attmats']  #(n_node, n_time, att_dim)
    Labels = file['labels']  #(n_node, num_classes)
    Graphs_adj = file['adjs']  #(n_time, n_node, n_node)
    Features = meanstd_normalization_tensor(Features)  
    
    # Load pre-computed RWR features
    rwr_path = filename.replace('.npz', '_RWR.npy')
    if not os.path.exists(rwr_path):
        raise FileNotFoundError(f"RWR file not found: {rwr_path}. Please generate it using RWR.py first.")
    Graphs_RWR = np.load(rwr_path)

    graphs_rwr = np.swapaxes(Graphs_RWR, 1, 0)  
    attribute = np.swapaxes(Features, 1, 0)  
   
    class MYDataset(Dataset):
        def __init__(self, look_back, train_size):

            self.n = graphs_rwr.shape[1]
            self.x_data, self.y_data, self.z_data = \
                np.swapaxes(graphs_rwr, 1, 0), np.swapaxes(attribute, 1,0), np.argmax(Labels, axis=1)

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index], self.z_data[index]

        def __len__(self):
            return self.n

    
    dataset = MYDataset(look_back=args.lookback, train_size=args.trainsize)
    tra_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - tra_size - val_size

    train_dataset, val_dataset, test_dataset = Data.random_split(
        dataset, [tra_size, val_size, test_size])

    tra_loader = DataLoader(dataset=train_dataset,
                            batch_size=256,
                            shuffle=False,
                            num_workers=0)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_size,
                            shuffle=False,
                            num_workers=0)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=test_size,
                             shuffle=False,
                             num_workers=0)

    model = AutoEnmodel(node_numb=args.node_numb,
                        emb_d=args.emb_d,
                        t=args.timelength,
                        use_attribute=True)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
 
    model.to(device)
   
    params = list(model.parameters())
    print(len(params))
    param_dic = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_dic[name] = param
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learningrate,
                                 betas=[0.9, 0.999],
                                 eps=1e-08)
   
    loss_func = nn.CrossEntropyLoss()  #
    best_val = 0
    for epoch in range(args.epochs):
        #model.train()
        loss_L1 = 0
        alpha = 0.5
        #scheduler.step()
        acc_val = train_one_epoch(epoch, model, tra_loader, val_loader,
                                  optimizer, device)

        if acc_val > best_val:
            best_val = acc_val
            loss, acc, auc, f1 = test(
                epoch,
                model,
                test_loader,
            )
            test_best_val = [loss, acc, auc, f1]
            if acc > 0.75:
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'Saved best model with acc={acc:.4f}')

    print('________________________________________')
   
    print('finish training')

    # Load and evaluate best model if it exists
    if os.path.exists('best_model.pth'):
        print('Loading best model for final evaluation...')
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        loss, acc, auc, f1 = test(
            0,
            model,
            test_loader,
        )
        print(f'Best model - Loss: {loss:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}')
    else:
        print('No best model saved during training.')

