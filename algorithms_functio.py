import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(0)


def computer_CS(content):
    content_graph = cosine_similarity(content)
    content_graph = content_graph - np.diag(np.diag(content_graph))
    return content_graph


def create_adjMat_content(input_mat, k, weight=1):
    num_node = input_mat.shape[0]
    adjMat_content = np.zeros((num_node, num_node))
    for i in range(num_node):
        SN_node = input_mat[i, :].argsort()[::-1]  
        N_node = SN_node[:k]
        for j in range(k):
            adjMat_content[i, N_node[j]] = weight
    return adjMat_content


def create_adjMat_combined(adjmat_toplgy, adjMat_content):
    adjMat_add = adjmat_toplgy + adjMat_content
    adjMat_combined = np.where(adjMat_add >= 1, 1, 0) 
    return adjMat_combined


def create_degree_time(adjMat_combined_time):
    time_len = adjMat_combined_time.shape[0]
    node_num = adjMat_combined_time.shape[1]
    degree_time = np.zeros((node_num, time_len))

    for t in range(time_len):
        for i in range(node_num):
            degree_time[i, t] = np.count_nonzero(adjMat_combined_time[t, i, :])

    out = cosine_similarity(degree_time)
    if np.diagonal(out).sum() == out.shape[0]:
        out = out - np.diag(np.diag(out))  
    else:
        print("error")
    out = out / out.max()
    return out


def computer_drgree_M(adj):
    return


def merge(topolgy_mat, content_vec):
    adjMat_combined = np.zeros(topolgy_mat.shape)

    for t in range(topolgy_mat.shape[0]):
        content_graph = computer_CS(content_vec[t])
        adjMat_content = create_adjMat_content(content_graph, 10)
        adjMat_combined[t] = create_adjMat_combined(topolgy_mat[t],
                                                    adjMat_content)

    outMat = create_degree_time(adjMat_combined)
    return outMat


