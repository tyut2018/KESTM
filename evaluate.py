import numpy as np


def getEdgeListFromAdjMtx(adj,
                          threshold=0.0,
                          is_undirected=True,
                          edge_pairs=None):
    result = []
    node_num = adj.shape[0]
    indx = np.where(adj > threshold)  
    for num in range(indx[0].size):
        i = indx[0][num]
        j = indx[1][num]
        if (i < j):
            result.append((i, j, adj[i, j]))

    # if edge_pairs:
    #     for (st, ed) in edge_pairs:
    #         if adj[st, ed] >= threshold:
    #             result.append((st, ed, adj[st, ed]))
    # else:
    #     for i in range(node_num):
    #         for j in range(node_num):
    #             if (j == i):
    #                 continue
    #             if (is_undirected and i >= j):
    #                 continue
    #             if adj[i, j] > threshold:
    #                 result.append((i, j, adj[i, j]))  #概率矩阵
    return result


def computePrecisionCurve(predicted_edge_list, true_digraph, max_k=-1):
    if max_k == -1:
        max_k = len(predicted_edge_list)  
    else:
        max_k = min(max_k, len(predicted_edge_list))

    sorted_edges = sorted(predicted_edge_list,
                          key=lambda x: x[2],
                          reverse=True)  
   
    precision_scores = [] 
    delta_factors = [] 
    correct_edge = 0
    for i in range(max_k): 
        if true_digraph[sorted_edges[i][0], sorted_edges[i][1]] > 0:
            correct_edge += 1 
            delta_factors.append(1.0)
        else:
            delta_factors.append(0.0)
        precision_scores.append(1.0 * correct_edge / (i + 1))
    return precision_scores, delta_factors


def computeMAP(predicted_edge_list, true_digraph, max_k=-1):
    node_num = true_digraph.shape[0] 
    node_edges = []
    for i in range(node_num):
        node_edges.append([])
    for (st, ed, w) in predicted_edge_list:
        node_edges[st].append(
            (st, ed, w)) 
    node_AP = [0.0] * node_num
    count = 0
    for i in range(node_num):
        if max(true_digraph[i, :]) == 0:
            continue
        count += 1 
        precision_scores, delta_factors = computePrecisionCurve(
            node_edges[i], true_digraph, max_k) 
        precision_rectified = [
            p * d for p, d in zip(precision_scores, delta_factors)
        ] 
        if (sum(delta_factors) == 0):
            node_AP[i] = 0
        else:
            node_AP[i] = float(sum(precision_rectified) / sum(delta_factors))
    
    if count == 0:
        return sum(node_AP)
    else:
        return sum(node_AP) / count


def evaluateDynamicLinkPrediction(
        graph,  
        estimated_adj,  
        threshold=0,
        n_sample_nodes=None,
        no_python=False,
        is_undirected=True,
        sampling_scheme="u_rand"):
    node_l = None
    test_digraph = graph
    if n_sample_nodes:
        if sampling_scheme == "u_rand":
           
            test_digraph = graph
            node_l = list(range(n_sample_nodes))
        else:
           
            test_digraph = graph

   

    predicted_edge_list = getEdgeListFromAdjMtx(estimated_adj,
                                                threshold,
                                                is_undirected=is_undirected,
                                                edge_pairs=None)

    MAP = computeMAP(predicted_edge_list, test_digraph)
    prec_curv, _ = computePrecisionCurve(predicted_edge_list, test_digraph)
    return (MAP, prec_curv)