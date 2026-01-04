import numpy as np
from sklearn import preprocessing
import os


def generate_topo(Graphs, prob_c, step_K):

    n_time, n_node = Graphs.shape[1], Graphs.shape[0]
    P_T = np.zeros((n_node, n_time, n_node))

    for t in range(n_time):
        print(t)
        Gt = Graphs[:, t, :]
        Gt = Gt + np.eye(n_node)

        Dt_inv = np.diag(1 / np.sum(Gt, axis=1))  
        Gt_norm = np.dot(Dt_inv, Gt)  

        Pt0 = np.eye(n_node) 
        Pt_p = Pt0
        for i in range(step_K):
            Pt_c = prob_c * np.dot(Pt_p, Gt_norm) + (
                1 - prob_c) * Pt0 
            P_T[:, t, :] += Pt_c

            Pt_p = Pt_c

    return P_T


def meanstd_normalization_tensor(tensor, method='standard'):
    """
    Normalize features
    Args:
        tensor: (n_node, n_steps, n_dim)
        method: 'standard', 'minmax', 'robust', 'quantile'
    """
    n_node, n_steps, n_dim = tensor.shape
    
    if method == 'standard':
        tensor_reshape = preprocessing.scale(
            np.reshape(tensor, [n_node, n_steps * n_dim]),
            axis=1)
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        tensor_reshape = scaler.fit_transform(
            np.reshape(tensor, [n_node, n_steps * n_dim]))
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        tensor_reshape = scaler.fit_transform(
            np.reshape(tensor, [n_node, n_steps * n_dim]))
    elif method == 'quantile':
        from sklearn.preprocessing import QuantileTransformer
        scaler = QuantileTransformer()
        tensor_reshape = scaler.fit_transform(
            np.reshape(tensor, [n_node, n_steps * n_dim]))
    else:
        tensor_reshape = np.reshape(tensor, [n_node, n_steps * n_dim])
    
    tensor_norm = np.reshape(tensor_reshape, [n_node, n_steps, n_dim])
    return tensor_norm


def get_topo(Graphs):
    prob_c = 0.98
    step_k = 5
    Topology = generate_topo(Graphs, prob_c, step_k)  
    return Topology


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate RWR topology features for dynamic graphs')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to .npz dataset file (e.g., ./data/DBLP3.npz)')
    args = parser.parse_args()

    filename = args.data
    print(f"Loading dataset from {filename}...")
    
    try:
        file = np.load(filename)
        Graphs_adj = file['adjs']  # Shape: (T, N, N)
        print(f"Dataset shape: {Graphs_adj.shape}")
        
        # Swap axes to (N, T, N) as expected by get_topo
        Graphs_adj_swap = np.swapaxes(Graphs_adj, 1, 0)
        
        print("Generating RWR topology features...")
        Graphs_RWR = get_topo(Graphs_adj_swap)
        
        # Save with same naming convention
        save_path = filename.replace('.npz', '_RWR.npy')
        np.save(save_path, Graphs_RWR)
        print(f"✓ Successfully saved RWR topology to {save_path}")
        print(f"  Output shape: {Graphs_RWR.shape}")
        
    except FileNotFoundError:
        print(f"Error: Dataset file not found: {filename}")
    except KeyError as e:
        print(f"Error: Missing key in dataset: {e}")
        print("Expected keys: 'adjs', 'attmats', 'labels'")
    except Exception as e:
        print(f"Error occurred: {e}")
