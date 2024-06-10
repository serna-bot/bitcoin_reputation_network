import numpy as np
from sklearn.cluster import SpectralClustering
import igraph as ig

def calculate_modularity(A, cluster, total_positive_weight, total_negative_weight):
    """
    Calculate the modularity Q_s for a given cluster in an undirected graph.

    Args:
        A (np.array): Adjacency matrix
        cluster (set): Set of node indices representing a cluster
        total_positive_weight (float): Total sum of positive weights in the graph
        total_negative_weight (float): Total sum of negative weights in the graph

    Returns:
        float: Modularity Q_s for the given cluster
    """
    if total_positive_weight + total_negative_weight == 0:
        return 0  # Avoid division by zero

    modularity = 0
    cluster_list = list(cluster)
    cluster_size = len(cluster_list)

    for i in range(cluster_size):
        for j in range(cluster_size):
            node_i = cluster_list[i]
            node_j = cluster_list[j]
            w_ij = A[node_i, node_j]
            k_i_plus = np.sum(A[node_i, A[node_i] > 0])
            k_j_plus = np.sum(A[node_j, A[node_j] > 0])
            k_i_minus = np.sum(A[node_i, A[node_i] < 0])
            k_j_minus = np.sum(A[node_j, A[node_j] < 0])

            if total_positive_weight > 0:
                expected_positive = (k_i_plus * k_j_plus) / (2 * total_positive_weight)
            else:
                expected_positive = 0

            if total_negative_weight > 0:
                expected_negative = (k_i_minus * k_j_minus) / (2 * total_negative_weight)
            else:
                expected_negative = 0

            modularity += w_ij - (expected_positive - expected_negative)

    modularity /= (2 * total_positive_weight + 2 * total_negative_weight)
    return modularity

class Louvain:
    def __init__(self, A):
        self.A = A
        self.G = ig.Graph.Weighted_Adjacency(A, mode="undirected", attr="weight", loops=False)

    def run(self):
        louvain = self.G.community_multilevel(weights=self.G.es['weight'], return_levels=False)
        modularity = self.G.modularity(louvain, weights=self.G.es['weight'])
        return louvain, modularity
    
        
class Spectral:
    def __init__(self, A):
        self.A = A
        self.total_positive_weight = np.sum(A[A > 0]) / 2  # Each edge is counted twice in an undirected graph
        self.total_negative_weight = np.sum(A[A < 0]) / 2
        self.calculate_modularity = lambda cluster: calculate_modularity(
            self.A, cluster, self.total_positive_weight, self.total_negative_weight)
    
    def run(self, max_clusters=2, tolerance=1e-5):
        best_modularity = -np.inf
        best_clusters = None
        previous_modularity = None

        for num_clusters in range(2, max_clusters+1):
            clusters = {}
            sc = SpectralClustering(num_clusters, affinity='precomputed', n_init=100)
            sc.fit(self.A)
            for idx, cluster in enumerate(sc.labels_): #labels of each point
                if cluster in clusters:
                    clusters[cluster].add(idx)
                else:
                    clusters[cluster] = {idx}
            modularities = [calculate_modularity(cluster) for cluster in clusters.values()]
            total_modularity = sum(modularities)
            
            if total_modularity > best_modularity:
                best_modularity = total_modularity
                best_clusters = clusters
            
            if previous_modularity is not None and abs(total_modularity - previous_modularity) < tolerance:
                break

            previous_modularity = total_modularity
        
        return best_clusters, best_modularity

