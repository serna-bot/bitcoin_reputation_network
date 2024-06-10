from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import igraph as ig
import numpy as np
import networkx as nx
from tqdm import tqdm
import json

def save_file(S, sim_name, col):
    f = open('paths.json')
    data = json.load(f)
    path  = data['result_path']
    file_name = path + "/" + sim_name
    if col:
        file_name += "_col"
    else:
        file_name += "_row"
    file_name += ".out"
    np.savetxt(file_name, S, delimiter=' ')
    print("Saved File.")

def cosine_sim(A, col, savefile=False):
    """
    Args:
        A (np.array): adjacency matrix
        col (Boolean): whether to return column pearson sim or row pearson sim
    
    Returns:
        (np.array)
    """
    N = A.shape[0]
    S = np.zeros_like(A)  # Initialize similarity matrix with zeros

    for i in tqdm(range(N)):
        for j in range(N):
            if not col:
                norm_i = np.sqrt(np.sum(np.square(A[i, :])))
                norm_j = np.sqrt(np.sum(np.square(A[j, :])))
                if norm_i == 0 or norm_j == 0:
                    S[i, j] = 0
                else:
                    S[i, j] = np.dot(A[i, :], A[j, :]) / (norm_i * norm_j)
            else:
                norm_i = np.sqrt(np.sum(np.square(A[:, i])))
                norm_j = np.sqrt(np.sum(np.square(A[:, j])))
                if norm_i == 0 or norm_j == 0:
                    S[i, j] = 0 
                else:
                    S[i, j] = np.dot(A[:, i], A[:, j]) / (norm_i * norm_j)
    if savefile:
        save_file(S, "cosine_sim", col)
    return S

def pearson_sim(A, col, savefile):
    """
    Args:
        A (np.array): adjacency matrix
        col (Boolean): whether to return column pearson sim or row pearson sim

    Returns:
        (np.array)
    """
    axis = 0 if col else 1
    row_means = np.mean(A, axis=axis)
    centered_arr = A - row_means[:, np.newaxis]
    covariance_matrix = np.dot(centered_arr, centered_arr.T)
    stds = np.std(A, axis=axis)
    std_outer_product = np.outer(stds, stds)
    std_outer_product[std_outer_product == 0] = 1
    S = covariance_matrix / std_outer_product
    if savefile:
        save_file(S, "pearson_sim", col)
    return S

def jaccard_sim(A, col, savefile):
    """
    Args:
        A (np.array)
        col (boolean)
    """
    G = ig.Graph.Weighted_Adjacency(A)
    g_out = 'in' if col else 'out'
    S = np.array(G.similarity_jaccard(mode=g_out))
    if savefile:
        save_file(S, "jaccard_sim", col)
    return S

def adamic_adar_sim(A, col, savefile):
    """
    Args:
        A (np.array)
        col (boolean)
    """
    G = ig.Graph.Weighted_Adjacency(A)
    g_out = 'in' if col else 'out'
    S = np.array(G.similarity_inverse_log_weighted(mode=g_out))
    if savefile:
        save_file(S, "adamic_adar_sim", col)
    return S