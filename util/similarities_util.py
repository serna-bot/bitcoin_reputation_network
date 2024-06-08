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
    pass