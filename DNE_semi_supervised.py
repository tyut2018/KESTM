# KESTM - Knowledge-Enhanced Spatio-Temporal Module for Dynamic Graph Node Classification

import random
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

try:
    from model import AutoEnmodel, Weighted_mse_x
    from RWR import get_topo, meanstd_normalization_tensor
    from util import *
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import AutoEnmodel, Weighted_mse_x
    from RWR import get_topo, meanstd_normalization_tensor
    from util import *

def data_augmentation(features, method='none', noise_level=0.01):
    """
    Data augmentation for node features
    Args:
        features: numpy array of shape (n_nodes, n_time, n_features)
        method: 'none', 'noise', 'dropout', 'smooth'
        noise_level: noise intensity or dropout rate
    """
    if method == 'none':
        return features
    
    features_aug = features.copy()
    
    if method == 'noise':
        noise = np.random.randn(*features.shape) * noise_level
        features_aug = features_aug + noise
    
    elif method == 'dropout':
        mask = np.random.rand(*features.shape) > noise_level
        features_aug = features_aug * mask
    
    elif method == 'smooth':
        from scipy.ndimage import gaussian_filter1d
        for i in range(features.shape[0]):
            for j in range(features.shape[2]):
                features_aug[i, :, j] = gaussian_filter1d(features[i, :, j], sigma=noise_level*10)
    
    return features_aug

class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
       
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
       
        pt = torch.exp(-ce_loss)
        
      
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SupervisedContrastiveLoss(nn.Module):
   
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features, labels):
       
        device = features.device
        batch_size = features.shape[0]
        
      
        features = F.normalize(features, dim=1)
        
      
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
      
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device) 
       
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
       
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        
      
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
       
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        
      
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss

writer = None 
torch.cuda.empty_cache()

def train_one_epoch(epoch, model, train_loader, optimizer, device, loss_func, scl_func=None, idx_train=None, idx_val=None, scl_weight=0.0, args=None, verbose=False):
    t = time.time()
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    input1 = train_loader[0]
    input2 = train_loader[1]
    labels = train_loader[2]

    #
    if hasattr(model, 'use_adaptive_kem') and model.use_adaptive_kem:
        if hasattr(model, 'set_adjacency_matrix'):
            model.set_adjacency_matrix(input1)
    
    features, output = model(input1, input2)
    

    if args and args.save_attention and epoch % 10 == 0:  
        attention_weights = features 
        os.makedirs('attention_analysis', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'attention_weights': attention_weights.cpu(),
            'labels': labels.cpu(),
            'train_idx': idx_train.cpu() if hasattr(idx_train, 'cpu') else idx_train,
            'val_acc': 0.0  
        }, f'attention_analysis/attention_epoch_{epoch}.pt')
    
    loss_train = loss_func(output[idx_train], labels[idx_train])
    
    
    if scl_func is not None:
        scl_loss = scl_func(features[idx_train], labels[idx_train])
        loss_train = loss_train + scl_weight * scl_loss
    
   
    if hasattr(model, 'contrastive_loss') and model.contrastive_loss is not None:
        tcl_loss = model.contrastive_loss
        loss_train = loss_train + tcl_loss

    # ===== COMMENTED OUT FOR PAPER ALIGNMENT =====
    # Paper doesn't use L1/L2 regularization in the main loss
    # l1_, l2_ = torch.tensor([0], dtype=torch.float32).to(device), torch.tensor([0], dtype=torch.float32).to(device)
    # l1_ = l1_regularization(model, 1e-6)
    # l2_ = l2_regularization(model, 1e-6)
    # loss = loss_train + l1_ + l2_
    
    loss = loss_train  # Only main loss (Focal + TCL per paper)
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    train_loss = float(loss.item())  

    pred_labels = torch.argmax(output[idx_train], axis=1)

    correct += (pred_labels == labels[idx_train]).sum().float()
    total += len(labels[idx_train])

    if writer is not None:
        writer.add_scalar("Loss/train", train_loss, epoch + 1)  
    acc_train_test = metrics.accuracy_score(labels[idx_train].cpu().detach().numpy(),
                                            torch.argmax(output[idx_train], axis=1).cpu().detach().numpy())
    acc_train = (correct / total).cpu().detach().data.numpy()
    assert np.round(acc_train_test.max()) == np.round(acc_train)

    model.eval()

    with torch.no_grad():
        _, output_val = model(input1, input2)

    loss_val = loss_func(output_val[idx_val], labels[idx_val])

    acc_val = metrics.accuracy_score(labels[idx_val].cpu().detach().numpy(),
                                     torch.argmax(output_val[idx_val], axis=1).cpu().detach().numpy())
    del output
    
    # Print progress every epoch
    print('Epoch :%4d' % (epoch + 1), '|', 'Loss_train:%.4f' % loss_train.data, '|', 'Acc_train:%.4f' % acc_train, '|',
          'Loss_val:%.4f' % loss_val.data, 'Acc_val:%.4f' % acc_val, '|', 'Time:%.4fs' % (time.time() - t), '|')
    
    if writer is not None:
        writer.add_scalars("Metrics/train", {'acc_train': acc_train, 'acc_val': acc_val}, epoch + 1)

    return acc_val

def test(epoch, model, test_loader, loss_func, idx_test, args=None):
   
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0

        input1_test = test_loader[0]
        input2_test = test_loader[1]
        labels_test = test_loader[2]
        _, output_test = model(input1_test, input2_test)
        loss_test = loss_func(output_test[idx_test], labels_test[idx_test])
      
        temperature = 1.0
        
        probs = F.softmax(output_test[idx_test] / temperature, dim=-1).cpu().detach().numpy()
        y_true = labels_test[idx_test].cpu().detach().numpy()
        pred_labels = np.argmax(probs, axis=1)
       
        f1_weighted = metrics.f1_score(y_true, pred_labels, average='weighted')
        f1_macro = metrics.f1_score(y_true, pred_labels, average='macro')
        f1_micro = metrics.f1_score(y_true, pred_labels, average='micro')
        
        acc_test = metrics.accuracy_score(y_true, pred_labels)
        f1_test = f1_weighted
        
        test_class_counts = np.bincount(y_true)
        present_classes = np.where(test_class_counts > 0)[0]
        
        if len(present_classes) < 2:
            auc_test = 0.0
        else:
            try:
                y_one_hot = one_hot(y_true)
                auc_test = metrics.roc_auc_score(y_one_hot[:, present_classes], 
                                                 probs[:, present_classes], 
                                         multi_class='ovr',
                                         average='weighted')
            except ValueError as e:
                try:
                    if len(present_classes) == 2:
                        auc_test = metrics.roc_auc_score(y_one_hot[:, present_classes[0]], 
                                                        probs[:, present_classes[0]])
                    else:
                        auc_test = 0.5
                except:
                    auc_test = 0.5
        
        if writer is not None:
            writer.add_scalar("test_loss/epoch", loss_test, epoch + 1)
            writer.add_scalars("Metrics/test", {'acc_test': acc_test, 'f1_test': f1_test, 'auc_test': auc_test}, epoch + 1)
        
        # Print test results
        print('\nTest Results:')
        print('Loss: %.4f' % loss_test.data)
        print('ACC=%.4f, F1_weighted=%.4f, F1_macro=%.4f, F1_micro=%.4f, AUC=%.4f' % 
              (acc_test, f1_weighted, f1_macro, f1_micro, auc_test))
        
        # Print per-class F1 scores
        f1_per_class = metrics.f1_score(y_true, pred_labels, average=None)
        print("\nPer-Class F1 Scores:")
        for i, f1 in enumerate(f1_per_class):
            print(f"  Class {i}: {f1:.4f}")
        print()
        
        return test_loss, acc_test, auc_test, f1_test

# ============================================================================
# ============================================================================
def compute_structural_features(adj_matrices, use_pagerank=True, use_clustering=True, use_turnover=False):
    import networkx as nx
    
    T = len(adj_matrices)
    n = adj_matrices[0].shape[0]
    
    A_agg = np.zeros((n, n), dtype=np.float32)
    for A in adj_matrices:
        A_agg += (A > 0).astype(np.float32)
    
    pagerank_scores = None
    clustering_coeff = None
    turnover_rates = None
    
    if use_pagerank:
        G = nx.from_numpy_array((A_agg > 0).astype(np.int32), create_using=nx.Graph)
        pr_dict = nx.pagerank(G, alpha=0.85, max_iter=100)
        pagerank_scores = np.array([pr_dict[i] for i in range(n)], dtype=np.float32)
    
    if use_clustering:
        G = nx.from_numpy_array((A_agg > 0).astype(np.int32), create_using=nx.Graph)
        cc_dict = nx.clustering(G)
        clustering_coeff = np.array([cc_dict[i] for i in range(n)], dtype=np.float32)
    
    if use_turnover:
        turnover = np.zeros(n, dtype=np.float32)
        for t in range(T - 1):
            A_t = adj_matrices[t]
            A_t1 = adj_matrices[t + 1]
            
            N_t = (A_t > 0).astype(np.int32)
            N_t1 = (A_t1 > 0).astype(np.int32)
            
            diff = np.abs(N_t - N_t1).sum(axis=1)
            union = ((N_t + N_t1) > 0).sum(axis=1) + 1e-8
            turnover += (diff / union)
        
        turnover_rates = turnover / max(1, T - 1)
    
    return pagerank_scores, clustering_coeff, turnover_rates

USE_FOCAL_LOSS = True
USE_ENHANCED_KEM = True

def enhance_features_adaptive(Features, target_dim=60, train_indices=None):
    from sklearn.decomposition import PCA, FastICA
    import numpy as np
    
    n_nodes, n_time, n_features = Features.shape
    
    Features_flat = Features.reshape(-1, n_features)  # (n_nodes * n_time, 20)
    
    remaining_dim = target_dim - n_features
    if remaining_dim <= 0:
        return Features
    
    import sys
    
    if train_indices is not None:
        train_indices_flat = []
        for t in range(n_time):
            for node_idx in train_indices:
                train_indices_flat.append(node_idx * n_time + t)
        train_features_flat = Features_flat[train_indices_flat]
    else:
        train_features_flat = Features_flat
    
    max_pca_components = min(n_features, train_features_flat.shape[0] - 1)
    pca1 = PCA(n_components=max_pca_components, random_state=42)
    pca1.fit(train_features_flat)
    Features_pca1 = pca1.transform(Features_flat)
    
    feature_list = [Features_flat]
    feature_list.append(Features_pca1)
    
    current_total = Features_flat.shape[1] + Features_pca1.shape[1]
    
    if current_total < target_dim:
        needed = target_dim - current_total
        if needed > 0:
            np.random.seed(42)
            n_combinations = needed
            n_pca_features = Features_pca1.shape[1]
            
            weight_matrix = np.random.randn(n_pca_features, n_combinations) * 0.5
            weight_matrix = weight_matrix / (np.linalg.norm(weight_matrix, axis=0, keepdims=True) + 1e-8)
            
            Features_combined = Features_pca1 @ weight_matrix
            feature_list.append(Features_combined)
            current_total += Features_combined.shape[1]
    
    
    Features_enhanced = np.concatenate(feature_list, axis=1)
    
    current_dim = Features_enhanced.shape[1]
    
    if current_dim == target_dim:
        Features_enhanced = Features_enhanced.reshape(n_nodes, n_time, -1)
        return Features_enhanced
    
    if current_dim > target_dim:
        final_pca = PCA(n_components=target_dim, random_state=42)
        if train_indices is not None:
            final_pca.fit(Features_enhanced[train_indices_flat])
        else:
            final_pca.fit(Features_enhanced)
        Features_enhanced = final_pca.transform(Features_enhanced)
    elif current_dim < target_dim:
        padding = np.zeros((Features_enhanced.shape[0], target_dim - current_dim))
        Features_enhanced = np.concatenate([Features_enhanced, padding], axis=1)
    
    Features_enhanced = Features_enhanced.reshape(n_nodes, n_time, -1)
    
    return Features_enhanced

def get_unified_optimized_config():
    return {
        'dropout': 0.32,
        'tcl_weight': 0.05,
        'transformer_layers': 6,
        'attention_heads': 8,
        'warmup_epochs': 15,
        'use_spatiotemporal_attn': True,
    }

def analyze_dataset_characteristics(Features, Graphs_adj, dataset_name=""):
    n_nodes, n_time, n_features = Features.shape
    
    is_low_dimensional = n_features < 50
    low_dim_threshold = 50
    
    isolated_mask = np.zeros(n_nodes, dtype=bool)
    for t in range(n_time):
        adj_t = Graphs_adj[t]
        np.fill_diagonal(adj_t, 0)
        degrees = adj_t.sum(axis=1)
        isolated_mask = isolated_mask | (degrees == 0)
    isolated_ratio = isolated_mask.sum() / n_nodes
    
    is_extremely_sparse = isolated_ratio > 0.5
    
    adaptive_config = {
        'needs_feature_enhancement': is_low_dimensional,
        'needs_graph_reconstruction': is_extremely_sparse,
        'original_feature_dim': n_features,
        'isolated_node_ratio': isolated_ratio,
    }
    
    if is_low_dimensional:
        if n_features <= 20:
            target_dim = 80
        elif n_features <= 30:
            target_dim = 64
        else:
            target_dim = min(n_features * 2, 64)
        adaptive_config['target_emb_d'] = target_dim
        adaptive_config['patience'] = 40
        adaptive_config['use_feature_dropout'] = 0.12
    else:
        adaptive_config['target_emb_d'] = None
        adaptive_config['patience'] = None
        adaptive_config['use_feature_dropout'] = None
    
    return adaptive_config

def get_adaptive_optimization_config(adaptive_config):
    config = {}
    
    if adaptive_config.get('needs_feature_enhancement', False):
        config['emb_d'] = adaptive_config['target_emb_d']
        config['patience'] = adaptive_config.get('patience', None)
        config['use_feature_dropout'] = adaptive_config.get('use_feature_dropout', None)
        config['enable_feature_enhancement'] = True
    
    return config

def main(emb_d=16):
    """
    Main training function for KESTM model.
    """
    parser = ArgumentParser(description='Learns node embeddings for a sequence of graph snapshots')
    parser.add_argument('-t', '--testDataType', default='DBLP', type=str, help='Type of data to test the code')
    parser.add_argument('-c', '--criteria', default='degree', type=str, help='Node Migration criteria')
    parser.add_argument('-rc',
                        '--criteria_r',
                        default=True,
                        type=bool,
                        help='Take highest centrality measure to perform node migration')
    parser.add_argument('-l', '--timelength', default=10, type=int, help='Number of time series graph to generate')
    parser.add_argument('-lb', '--lookback', default=10, type=int, help='number of lookbacks')
    parser.add_argument('-beta', '--beta', default=5, type=int, help='number of Loss function argument')
    parser.add_argument('-iter', '--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('-emb', '--emb_d', default=64, type=int, help='embedding dimension')
    parser.add_argument('-rd', '--resultdir', type=str, default='./results_link_all', help="result directory name")
    parser.add_argument('-nb', '--node_numb', default=6606, type=int, help='node for test data')
    parser.add_argument('-eta', '--learningrate', default=1e-3, type=float, help='learning rate')

    parser.add_argument('-bs', '--batch', default=6606, type=int, help='batch size')
    parser.add_argument('-dropout', '--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--no-attr',
                        dest='useattr',
                        action="store_false",
                        help='disable attribute data')
    parser.set_defaults(useattr=True)
    
    parser.add_argument('--userwrNN',
                        dest='userwrNN',
                        action="store_true",
                        default=False,
                        help='enable RWR (Random Walk with Restart)')
    
    parser.add_argument('--no-transformer',
                        dest='usetransf',
                        action="store_false",
                        help='disable transformer')
    parser.set_defaults(usetransf=True)
    parser.add_argument('-exp', '--exp', default='lp', type=str, help='experiments (lp, emb)')
    parser.add_argument('-n_aeunits', '--n_aeunits', default=[500, 300], type=list, help='AEdense_arg')
    parser.add_argument('-ts', '--trainsize', default=3, type=str, help='train_size')
    parser.add_argument('-ta', '--tag', default='No_null_node', type=str, help='No_null_node')
    parser.add_argument('--data', type=str, default='./data/DBLP3.npz', help='path to dataset npz')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--norm_method', type=str, default='standard', 
                       choices=['standard', 'minmax', 'robust', 'quantile'],
                       help='normalization method for features')
    parser.add_argument('--data_aug', type=str, default='none',
                       choices=['none', 'noise', 'dropout', 'smooth'],
                       help='data augmentation method')
    parser.add_argument('--aug_level', type=float, default=0.01,
                       help='augmentation level (noise std or dropout rate)')
    parser.add_argument('--save_attention', action='store_true', default=False,
                       help='Save attention weights for visualization')
    parser.add_argument('--add_noise', type=float, default=0.0,
                       help='Add noise to features (0.0-1.0, default: 0.0)')
    parser.add_argument('--missing_rate', type=float, default=0.0,
                       help='Missing data rate (0.0-1.0, default: 0.0)')
    parser.add_argument('--time_steps', type=int, default=8,
                       help='Number of time steps to use (default: 8, optimal for DBLP per paper)')
    
    parser.add_argument('--disable_knowledge', action='store_true', default=False,
                       help='Disable knowledge enhancement module (KEM)')
    parser.add_argument('--disable_temporal_attention', action='store_true', default=False,
                       help='Disable spatiotemporal attention (ASTE)')
    
    parser.add_argument('--disable_temporal_stream', action='store_true', default=False,
                       help='Disable temporal attention stream in ASTE')
    parser.add_argument('--disable_spatial_stream', action='store_true', default=False,
                       help='Disable spatial attention stream in ASTE')
    
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal Loss gamma parameter (default: 2.0)')
    parser.add_argument('--tcl_weight', type=float, default=0.05,
                       help='Temporal Contrastive Loss weight (default: 0.05)')
    parser.add_argument('--scl_weight', type=float, default=0.0,
                       help='Supervised Contrastive Loss weight (default: 0.0)')
    
    parser.add_argument('--use_adaptive_kem', action='store_true', default=False,
                       help='Enable adaptive KEM')
    parser.add_argument('--use_sparsity_aware_attention', action='store_true', default=False,
                       help='Enable sparsity-aware attention')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Print detailed training progress')

    args = parser.parse_args()
    
    # Disable auto-optimization to strictly follow configuration
    use_optimized_config = False
    
    unified_config = get_unified_optimized_config()
    
    setup_seed(args.seed)
    
    import sys
    
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    filename = args.data

    file = np.load(filename)
    
    Features = file['attmats']  #(n_node, n_time, att_dim)
    Labels = file['labels']  #(n_node, num_classes)
    Graphs_adj = file['adjs'].astype(float).copy()
    
    original_time_steps = Features.shape[1]
    time_steps = getattr(args, 'time_steps', 10)
    if time_steps != original_time_steps:
        if time_steps > original_time_steps:
            time_steps = original_time_steps
        else:
            # Keep the latest time_steps
            Features = Features[:, -time_steps:, :]
            Graphs_adj = Graphs_adj[-time_steps:, :, :]
    else:
        time_steps = original_time_steps
    
    args.timelength = time_steps
    
    dataset_characteristics = analyze_dataset_characteristics(Features, Graphs_adj, filename)
    adaptive_config = get_adaptive_optimization_config(dataset_characteristics)
    
    
    needs_feature_enhancement = dataset_characteristics['needs_feature_enhancement'] and use_optimized_config
    needs_graph_reconstruction = dataset_characteristics['needs_graph_reconstruction']
    
    if needs_feature_enhancement:
        target_dim = dataset_characteristics['target_emb_d']
        args.emb_d = target_dim
    
    isolated_mask = None
    if needs_graph_reconstruction:
        original_nodes = Features.shape[0]
        now_adj = Graphs_adj.sum(axis=0)
        np.fill_diagonal(now_adj, 0)
        node_degrees = now_adj.sum(axis=1)
        isolated_mask = (node_degrees == 0)
        isolated_count = isolated_mask.sum()
        connected_count = (~isolated_mask).sum()
        
        
    use_normalization = True
    use_data_aug = (args.data_aug != 'none') if hasattr(args, 'data_aug') else False
    
    # Backup raw adjacency matrix for KEM module before loading RWR
    Graphs_adj_raw = Graphs_adj.copy()
    
    if args.tag == 'No_null_node':
        row, col = np.diag_indices_from(Graphs_adj[0, :, :])
        Graphs_adj[:, row, col] = 1
        
        # Apply same self-loop to raw backup for consistency
        row_raw, col_raw = np.diag_indices_from(Graphs_adj_raw[0, :, :])
        Graphs_adj_raw[:, row_raw, col_raw] = 1

        rwr_candidate = filename.replace('.npz', '_RWR.npy')
        if os.path.exists(rwr_candidate):
            # Load pre-computed RWR features
            rwr_data = np.load(rwr_candidate)
            
            # Slice RWR to match time_steps if necessary
            if rwr_data.shape[0] != time_steps:
                rwr_data = rwr_data[-time_steps:, :, :]
            
            Graphs_adj = rwr_data
            # Graphs_adj_raw keeps the original topology for KEM
        else:
            # If no RWR file, both are the same
            Graphs_adj_raw = Graphs_adj.copy()

    attribute = np.swapaxes(Features, 1, 0)

    class MYDataset(Dataset):
        def __init__(self, look_back, train_size):
            # Graphs_adj shape: (T, N, N)
            self.n = Graphs_adj.shape[1]  # Number of nodes
            self.x_data, self.y_data, self.z_data = \
                np.swapaxes(Graphs_adj, 1, 0),\
                    np.swapaxes(attribute, 1,0),\
                         np.argmax(Labels, axis=1)

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index], self.z_data[index]

        def __len__(self):
            return self.n

    dataset = MYDataset(look_back=args.lookback, train_size=args.trainsize)
    tra_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - tra_size - val_size

    # 70/20/10 split: First split 70% train vs 30% (val+test), then split 30% into 20% val and 10% test
    train_dataset, temp_x, train_y, temp_y = train_test_split([x for x in range(len(dataset))], np.argmax(Labels, axis=1), test_size=0.3, stratify=np.argmax(Labels, axis=1), random_state=args.seed)
    val_dataset, test_dataset, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.3333, stratify=temp_y, random_state=args.seed)
    train_labels = np.argmax(Labels[train_dataset], axis=1)
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    
    safe_class_counts = class_counts.copy()
    safe_class_counts[safe_class_counts == 0] = 1
    
    class_weights = total_samples / (len(class_counts) * safe_class_counts)
    
    if np.any(class_counts == 0):
        max_weight = np.max(class_weights[class_counts > 0]) if np.any(class_counts > 0) else total_samples
        class_weights[class_counts == 0] = max_weight
    
    class_weights = torch.FloatTensor(class_weights)
    
    if USE_FOCAL_LOSS:
        gamma = args.focal_gamma
        loss_func = FocalLoss(alpha=class_weights.to(device), gamma=gamma, reduction='mean')
    else:
        loss_func = nn.CrossEntropyLoss()
    
    train_size = len(train_dataset)
    max_scl_size = 3000
    
    if train_size > max_scl_size:
        scl_func = None
        scl_weight = 0.0
    else:
        scl_func = SupervisedContrastiveLoss(temperature=0.07).to(device)
        scl_weight = args.scl_weight
    
    del temp_x,train_y,val_y,test_y

    del (dataset)

    idx_train = torch.LongTensor(train_dataset)
    idx_val = torch.LongTensor(val_dataset)
    idx_test = torch.LongTensor(test_dataset)
    
    if needs_feature_enhancement:
        Features = enhance_features_adaptive(Features, target_dim=target_dim, train_indices=train_dataset)
        enhanced_feature_dim = Features.shape[2]
        
        attribute = np.swapaxes(Features, 1, 0)  # T*N*D
    
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        features_avg = Features.mean(axis=1)  # (n_nodes, att_dim)
        
        train_features_avg = features_avg[train_dataset]  # (n_train, att_dim)
        
        
        train_features_norm = train_features_avg / (np.linalg.norm(train_features_avg, axis=1, keepdims=True) + 1e-8)
        features_norm = features_avg / (np.linalg.norm(features_avg, axis=1, keepdims=True) + 1e-8)
        
        similarity_matrix = cosine_similarity(features_norm, train_features_norm)  # (n_nodes, n_train)
        
        k_neighbors = min(8, connected_count)
        reconstructed_edges = 0
        similarity_stats = []
        
        train_connected_mask = ~isolated_mask[train_dataset]
        train_connected_indices = np.array(train_dataset)[train_connected_mask]
        
        for i in np.where(isolated_mask)[0]:
            similarities_to_train_connected = similarity_matrix[i, train_connected_mask]
            
            top_k_positions = np.argsort(similarities_to_train_connected)[-k_neighbors:]
            top_k_indices = train_connected_indices[top_k_positions]
            top_k_sims = similarities_to_train_connected[top_k_positions]
            
            similarity_stats.extend(top_k_sims)
            
            for t in range(Graphs_adj.shape[0]):
                for j, weight in zip(top_k_indices, top_k_sims):
                    edge_weight = max(0.3, weight * 0.8)
                    Graphs_adj[t, i, j] = edge_weight
                    Graphs_adj[t, j, i] = edge_weight
                    reconstructed_edges += 1
        
        adj_any_time = (Graphs_adj.sum(axis=0) > 0).astype(float)
        np.fill_diagonal(adj_any_time, 0)
        node_degrees_reconstructed = adj_any_time.sum(axis=1)
        remaining_isolated = (node_degrees_reconstructed == 0).sum()
        
        
        dataset = MYDataset(look_back=args.lookback, train_size=args.trainsize)
    
    if use_normalization and hasattr(args, 'norm_method'):
        if args.norm_method == 'standard':
            Features = meanstd_normalization_tensor(Features, method=args.norm_method)
        else:
            Features = meanstd_normalization_tensor(Features, method=args.norm_method)
        
        attribute = np.swapaxes(Features, 1, 0)  # T*N*D
        dataset = MYDataset(look_back=args.lookback, train_size=args.trainsize)
    
        
        Features_train = Features[train_dataset].copy()
        Features_train = data_augmentation(Features_train, method=args.data_aug, noise_level=args.aug_level)
        Features[train_dataset] = Features_train
        
        attribute = np.swapaxes(Features, 1, 0)  # T*N*D
        dataset = MYDataset(look_back=args.lookback, train_size=args.trainsize)
    

    data1 = torch.FloatTensor(np.swapaxes(Graphs_adj, 1, 0)).to(device).to(torch.float32)

    features = torch.FloatTensor(np.swapaxes(attribute, 1, 0)).to(device).to(torch.float32)
    
    if args.add_noise > 0.0:
        noise = torch.randn_like(features) * args.add_noise
        features_original = features.clone()
        features = features + noise
        noise_norm = torch.norm(noise).item()
        feature_norm = torch.norm(features_original).item()
        actual_noise_ratio = noise_norm / feature_norm
    
    if args.missing_rate > 0.0:
        missing_mask = torch.rand_like(features) < args.missing_rate
        features_before_missing = features.clone()
        features[missing_mask] = 0.0
        total_elements = features.numel()
        missing_elements = missing_mask.sum().item()
        actual_missing_rate = missing_elements / total_elements
    
    labels = torch.LongTensor(np.argmax(Labels, axis=1)).to(device).to(torch.long)
    tra_loader = [data1, features, labels]

    dropout_value = args.dropout  # Use command line argument instead of hardcoded 0.32
    tcl_weight_value = args.tcl_weight
    transformer_layers_value = unified_config['transformer_layers']
    attention_heads_value = unified_config['attention_heads']
    use_spatiotemporal_attn = unified_config['use_spatiotemporal_attn']
    
    use_knowledge = not args.disable_knowledge
    use_spatiotemporal_attn = not args.disable_temporal_attention
    
    use_temporal_stream = not args.disable_temporal_stream
    use_spatial_stream = not args.disable_spatial_stream
    
    use_tcl_flag = True
    use_multiscale_flag = True
    
    knowledge_dim_value = 64

    model = AutoEnmodel(node_numb=Labels.shape[0],
                        emb_d=args.emb_d,
                        t=args.timelength,
                        class_num=Labels.shape[1],
                        attri_emd=features.shape[2],
                        dropout=dropout_value,
                        use_attribute=args.useattr,
                        use_rwr=args.userwrNN,
                        use_Transformer=args.usetransf,
                        use_knowledge=use_knowledge,
                        knowledge_dim=knowledge_dim_value,
                        use_tcl=use_tcl_flag,
                        tcl_weight=tcl_weight_value,
                        contrastive_tau=0.35,
                        use_multiscale=use_multiscale_flag,
                        multiscale_windows=[2, 4, 8],  # Multi-scale temporal windows as in paper
                        transformer_layers=transformer_layers_value,
                        attention_heads=attention_heads_value,
                        multiscale_transformer_layers=4,
                        multiscale_attention_heads=4,
                        use_spatiotemporal_attention=use_spatiotemporal_attn,
                        use_temporal_stream=use_temporal_stream,
                        use_spatial_stream=use_spatial_stream,
                        use_adaptive_kem=args.use_adaptive_kem,
                        use_sparsity_aware_attention=args.use_sparsity_aware_attention)

    if model.use_knowledge and hasattr(model, 'init_knowledge_from_graph'):
        # Initialize Knowledge Enhancement Module with graph topology
        model.init_knowledge_from_graph(Graphs_adj_raw)
    

    model.to(device)
    
    params = list(model.parameters())
    param_dic = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_dic[name] = param
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningrate, betas=[0.9, 0.999], eps=1e-08,
                                weight_decay=1e-4)
    
    best_val = 0
    if adaptive_config and 'patience' in adaptive_config and adaptive_config['patience'] is not None:
        patience = adaptive_config['patience']
    else:
        patience = 60
    patience_counter = 0
    
    
    best_model_state = None
    
    for epoch in range(args.epochs):
        #model.train()
        loss_L1 = 0
        alpha = 0.5
        acc_val = train_one_epoch(epoch, model, tra_loader, optimizer, device, loss_func, scl_func, idx_train, idx_val, scl_weight, args)
        # scheduler.step()  # COMMENTED OUT: No LR scheduler per paper
        
        if train_size > 4000 and epoch % 5 == 0:
            torch.cuda.empty_cache()

        if acc_val > best_val:
            best_val = acc_val
            patience_counter = 0
            print('Epoch :%4d | best_val: %.4f' % (epoch + 1, best_val))
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
                
    
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        model.eval()
        loss, acc, auc, f1 = test(0, model, tra_loader, loss_func, idx_test, args)
        test_best_val = [loss, acc, auc, f1]
    else:
        loss, acc, auc, f1 = test(0, model, tra_loader, loss_func, idx_test, args)
        test_best_val = [loss, acc, auc, f1]


    if hasattr(model, 'use_adaptive_kem') and model.use_adaptive_kem:
        if args.save_attention and hasattr(model, 'get_adaptive_gate_statistics'):
            gate_stats = model.get_adaptive_gate_statistics()
            if gate_stats is not None:
                os.makedirs('adaptive_kem_analysis', exist_ok=True)
                np.save('adaptive_kem_analysis/gate_history.npy', gate_stats['history'])
    
    print('\nTraining Complete')
    print('ACC=%.4f, AUC=%.4f, F1=%.4f' % (test_best_val[1], test_best_val[2], test_best_val[3]))
    return test_best_val
    # End of training
if __name__ == '__main__':
    temp = main()