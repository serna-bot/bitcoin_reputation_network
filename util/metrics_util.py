import numpy as np
def get_anomaly_metrics(clusters, A, centrality_funcs):
    """
    Get anomaly metrics of clusters

    Args:
        clusters (array): [{node_index, ...}]
        A (np.array): adjacency matrix (i to j)
        centrality_funcs (array): array of functions that takes in the A matrix and a cluster centrality_func(A, cluster)

    Returns: 
        array:
            1. Most negative node in cluster
            2. Fraction of negative edges out of total
            3. Centralities (not going to be implemented)
            4. Sum of edges
    """
    metrics = []
    for cluster in clusters:
        min_node, neg_edge_frac, centralities, sum_edges = None, None, None, None
        number_edges, neg_edges = 0, 0
        for i in cluster:
            #1
            sum_in_edges = np.sum(A[:, i])
            min_node = min(sum_in_edges, min_node) if min_node else sum_in_edges
            for j in cluster:
              #2
                number_edges += 1
                neg_edges += 1 if A[i, j] < 0 else 0
                #4
                sum_edges += A[i, j]
            neg_edge_frac = neg_edges / number_edges
        #3
        # centralities = {}
        # for centrality_func in centrality_funcs:
        #     centralities[centrality_func.__name__] = centrality_func(A, cluster)
        # metrics.append({'min_node': min_node, 'neg_edge_frac': neg_edge_frac, 'centrality': centralities, 'sum_edges': sum_edges})
        metrics.append({'min_node': min_node, 'neg_edge_frac': neg_edge_frac, 'sum_edges': sum_edges})
    return metrics

# def closeness_centrality(A, cluster):
#     pass

# def katz_centrality(A, cluster):
#     pass

# def degree_centrality(A, cluster):
#     pass

def get_lowest_rated_nodes(A, num=-1):
    """
    entry i,j corresponds to an edge from i to j
    Args:
        A (np.array): adjacency matrix
        num (int): number of items to return
    Returns:
        np.array
    """
    ascending = np.sort(np.sum(A, axis=0))
    if num > 0:
        return ascending[:num]
    return ascending
