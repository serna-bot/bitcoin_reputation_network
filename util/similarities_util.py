from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import igraph as ig
import numpy as np
import networkx as nx


def cosine_sim(A):
    cosine_similarity_matrix = cosine_similarity(A)
    return cosine_similarity_matrix

def pearson_sim(A, col= False):
    """
    Args:
        A (np.array): adjacency matrix
        col (Boolean): whether to return column pearson sim or row pearson sim

    Returns:
        (np.array)
    """
    N = A.shape[0]
    S = np.zeros_like(A)
    for i in range(N):
        for j in range(N):
            if i == j:
                S[i, j] = 1.0
            else:
                if not col:
                    S[i, j] = pearsonr(A[i,], A[j,]).statistic
                else: 
                    S[i, j] = pearsonr(A[:,i], A[:,j]).statistic
    return S

def jaccard_sim(G):
    """
    Args:
        G (ig.Graph)
    """
    if type(G) != ig.Graph:
        raise TypeError("Arg must be of type ig.Graph") 
    return np.array(G.similarity_jaccard())

def adamic_adar_sim(G):
    """
    Args:
        G (ig.Graph)
    """
    if type(G) != ig.Graph:
        raise TypeError("Arg must be of type ig.Graph") 
    return np.array(G.similarity_inverse_log_weighted())

def dice_sim(G):
    """
    Args:
        G (ig.Graph)
    """
    if type(G) != ig.Graph:
        raise TypeError("Arg must be of type ig.Graph") 
    return np.array(G.similarity_dice())

def resource_allocation_sim(A):
    """
    """
    N = A.shape[0]
    similarity_matrix = np.zeros((N, N))

    for node1 in range(N):
        for node2 in range(node1 + 1, N):
            # Find common neighbors
            out_neighbors1 = set(np.where(A[node1, :] != 0)[0])
            in_neighbors1 = set(np.where(A[:, node1] != 0)[0])
            out_neighbors2 = set(np.where(A[node2, :] != 0)[0])
            in_neighbors2 = set(np.where(A[:, node2] != 0)[0])

            common_out_neighbors = out_neighbors1.intersection(out_neighbors2)
            common_in_neighbors = in_neighbors1.intersection(in_neighbors2)

            # Resource allocation similarity
            ra_similarity = 0

            for neighbor in common_out_neighbors:
                degree = np.sum(np.abs(A[neighbor, :])) + np.sum(np.abs(A[:, neighbor]))
                if degree > 0:
                    ra_similarity += A[node1, neighbor] * A[node2, neighbor] / degree

            for neighbor in common_in_neighbors:
                degree = np.sum(np.abs(A[neighbor, :])) + np.sum(np.abs(A[:, neighbor]))
                if degree > 0:
                    ra_similarity += A[neighbor, node1] * A[neighbor, node2] / degree

            similarity_matrix[node1, node2] = ra_similarity
            similarity_matrix[node2, node1] = ra_similarity  # Symmetric matrix

    return similarity_matrix