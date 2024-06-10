import numpy as np
import networkx as nx
from scipy.stats import pearsonr

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
            4. Average rating
    """
    metrics = []
    for cluster in clusters:
        min_node, centralities = None, None
        number_edges, neg_edges, sum_edges, neg_edge_frac, average_rating = 0, 0, 0, 0.0, 0.0
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
        average_rating = sum_edges / len(cluster)
        #3
        centralities = {}
        for centrality_func in centrality_funcs:
            centralities[centrality_func.__name__] = centrality_func(A, cluster)
        metrics.append({'min_node': min_node, 'neg_edge_frac': neg_edge_frac, 'centrality': centralities, 'average_rating': average_rating})
    return metrics

def closeness_centrality(A, cluster):
    """
    Args: 
      cluster (set)
    """
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    centrality = nx.closeness_centrality(G)
    if cluster == None:
        return centrality
    return np.mean([centrality[node] for node in cluster]) 

def katz_centrality(A, cluster):
    alpha = 1 / (max(np.linalg.eigvals(A)).real + 1) - 0.01
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    centrality = nx.katz_centrality_numpy(G, alpha=alpha)
    return np.mean([centrality[node] for node in cluster])

def degree_centrality(A, cluster):
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    centrality = nx.degree_centrality(G)
    if cluster == None:
        return centrality
    return np.mean([centrality[node] for node in cluster])

def rich_club_coefficient(A):
    """
    Calculate the rich-club coefficient for a directed and signed network.

    Args:
        A (np.array): Adjacency matrix of the network.

    Returns:
        dict: Rich-club coefficient for each degree threshold.
    """
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    degrees = np.array([d for _, d in G.degree()])
    sorted_degrees = np.sort(np.unique(degrees))
    
    rich_club_coeffs = {}

    for k in sorted_degrees:
        nodes_greater_k = [n for n, d in G.degree() if d > k]
        subgraph = G.subgraph(nodes_greater_k)
        E_greater_k = subgraph.number_of_edges()
        N_greater_k = len(nodes_greater_k)
        
        if N_greater_k > 1:
            phi_k = 2 * E_greater_k / (N_greater_k * (N_greater_k - 1))
        else:
            phi_k = 0

        rich_club_coeffs[k] = phi_k

    return rich_club_coeffs

def calculate_disassortativity(A):
    """
    Calculate the disassortativity coefficient for a directed and signed network using Pearson correlation.

    Args:
        A (np.array): Adjacency matrix of the network.

    Returns:
        float: Disassortativity coefficient.
    """
    # Create a directed graph from the adjacency matrix
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    
    # Get the in-degree and out-degree of each node
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # Initialize lists to store the degrees of source and target nodes for each edge
    out_degrees_source = []
    in_degrees_target = []

    # Iterate over each edge in the graph
    for u, v in G.edges():
        out_degrees_source.append(out_degrees[u])
        in_degrees_target.append(in_degrees[v])

    # Convert lists to numpy arrays for easier calculation
    out_degrees_source = np.array(out_degrees_source)
    in_degrees_target = np.array(in_degrees_target)

    # Calculate the Pearson correlation coefficient
    if len(out_degrees_source) == 0:
        return 0  # Handle case with no edges

    correlation, _ = pearsonr(out_degrees_source, in_degrees_target)
    
    # Disassortativity is the negative of assortativity
    disassortativity = -correlation
    return disassortativity

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
