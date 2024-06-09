from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import igraph as ig
import numpy as np
import networkx as nx


def cosine_sim(A, col):
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
            if not col:
                S[i, j] = np.dot(A[i,:], A[j,:]) / (np.sqrt(np.sum(np.square(A[i,:]))) * np.sqrt(np.sum(np.square(A[j,:]))))
            else: 
                 S[i, j] = np.dot(A[:,i], A[:,j]) / (np.sqrt(np.sum(np.square(A[:,i]))) * np.sqrt(np.sum(np.square(A[:,j]))))
    return S

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
                    S[i, j] = pearsonr(A[i,:], A[j,:]).statistic
                else: 
                    S[i, j] = pearsonr(A[:,i], A[:,j]).statistic
    return S

# def jaccard_similarity(G):
#   nodes = list(G.nodes())
#   jaccard_mat = np.zeros((len(nodes), len(nodes)))
#   for i in range(len(nodes)):
#     for j in range(len(nodes)):
#       neighbors_i = set(G[nodes[i]])
#       neighbors_j = set(G[nodes[j]])

#       set_intersection_card = len(neighbors_i & neighbors_j)
#       set_union_card = len(neighbors_i.union(neighbors_j))
#       if (set_union_card == 0):
#         set_union_card = 1
#       jaccard_mat[i][j] = set_intersection_card/set_union_card

#   return jaccard_mat

def jaccard_sim(A, col):
    """
    Args:
        A (np.array)
        col (boolean)
    """
    N = A.shape[0]
    S = np.zeros_like(A)
    for i in range(N):
        for j in range(N):
            if i == j:
                S[i, j] = 1
            else:
                if not col:
                    N_i = set(np.nonzero(A[i, :])[0])
                    N_j = set(np.nonzero(A[j, :])[0])
                else:
                    N_i = set(np.nonzero(A[:, i])[0])
                    N_j = set(np.nonzero(A[:, j])[0])
                S[i, j] = len(N_i.intersection(N_j)) / len(N_i.union(N_j))

def adamic_adar_sim(A, col):
    """
    Args:
        A (np.array)
        col (boolean)
    """
    N = A.shape[0]
    S = np.zeros_like(A)
    for i in range(N):
        for j in range(N):
            if i == j:
                S[i, j] = 1
            else:
                if not col:
                    N_i = set(np.nonzero(A[i, :])[0])
                    N_j = set(np.nonzero(A[j, :])[0])
                    U = N_i.intersection(N_j)
                    for u in U:
                        S[i, j] += 1/np.log(len(np.nonzero(A[u, :])[0]))
                else:
                    N_i = set(np.nonzero(A[:, i])[0])
                    N_j = set(np.nonzero(A[:, j])[0])
                    U = N_i.intersection(N_j)
                    for u in U:
                        S[i, j] += 1/np.log(len(np.nonzero(A[:, u])[0]))