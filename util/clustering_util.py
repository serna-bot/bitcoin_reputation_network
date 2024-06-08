import numpy as np

class Louvain:
    def __init__(self, A):
        self.A = A

        N = self.A.shape[0]
        self.clusters = [{'nodes': {n}, 'modularity': self.calculate_modularity({n}), 'neighbors': None} for n in range(N)]
        self.node_to_clusters = { n:n for n in range(N)} # to get the index of the cluster
        #initiating neighbors
        for node, cluster in enumerate(self.clusters):
            out_nodes, in_nodes = self.find_node_neighbors(self.A, node)
            cluster['neighbors'] = {neighbor: {node} for neighbor in out_nodes.union(in_nodes)}

    def louvain_clustering(self):
        """
        """
        clusters = self.clusters
        node_to_clusters = self.node_to_clusters
        N = self.A.shape[0]

        while True:
            nodes_moved = 0
            for node in range(N):
                curr_cluster_idx = node_to_clusters[node]
                cluster = clusters[curr_cluster_idx]
                node_stats = {'node': node, 'modularity': cluster['modularity'], 'index': curr_cluster_idx}
                for neighbor in cluster['neighbors']:
                    #move node into neighboring clusters
                    new_cluster = set(clusters[neighbor]['nodes'])
                    new_cluster.add(node)
                    new_modularity = self.calculate_modularity(new_cluster)
                    if new_modularity > node_stats['modularity']:
                        node_stats['index'] = neighbor
                        node_stats['modularity'] = new_modularity
                if node_stats['index'] != node_to_clusters[node]:
                    nodes_moved += 1
                    self.move_node(clusters[node_stats['index']], clusters[curr_cluster_idx], node_stats)
                    node_to_clusters[node] = node_stats['index']
            if nodes_moved < 0.1 * N:
                break
        return clusters

    def move_node(self, new_cluster, old_cluster, node_stats):
        """
        """
        node = node_stats['node']

        old_cluster['nodes'].remove(node)
        old_cluster['modularity'] = self.calculate_modularity(old_cluster['nodes'])
        for neighboring_cluster in list(old_cluster['neighbors'].keys()):
            if node in old_cluster['neighbors'][neighboring_cluster]:
                old_cluster['neighbors'][neighboring_cluster].remove(node)
            if not old_cluster['neighbors'][neighboring_cluster]:
                old_cluster['neighbors'].pop(neighboring_cluster)

        new_cluster['nodes'].add(node)
        new_cluster['modularity'] = node_stats['modularity']
        out_nodes, in_nodes = self.find_node_neighbors(node)
        for neighbor in out_nodes.union(in_nodes):
            neighboring_cluster = self.node_to_clusters[neighbor]
            if neighboring_cluster not in new_cluster['neighbors']:
                new_cluster['neighbors'][neighboring_cluster] = {node}
            else:
                new_cluster['neighbors'][neighboring_cluster].add(node)
        

    def find_node_neighbors(self, node):
        col = np.nonzero(self.A[node, :])[0]
        row = np.nonzero(self.A[:, node])[0]
        out_nodes = set(col)
        in_nodes = set(row)
        return out_nodes, in_nodes
        
    def calculate_modularity(self, cluster):
        """
        Args:
            A (np.array): adjacency matrix
            cluster (set): set of node indices
        Returns:
            float
        """
        A = self.A
        N = A.shape[0]

        w_plus = np.sum(A[A > 0])
        w_minus = np.sum(A[A < 0])

        if w_plus + w_minus == 0:
            return 0  # Avoid division by zero

        modularity = 0
        for i in range(N):
            for j in range(N):
                if i in cluster and j in cluster:
                    w_ij = A[i, j]
                    k_i_out_plus = np.sum(A[i, A[i] > 0])
                    k_j_in_plus = np.sum(A[A[:, j] > 0, j])
                    k_i_out_minus = np.sum(A[i, A[i] < 0])
                    k_j_in_minus = np.sum(A[A[:, j] < 0, j])
                    
                    if w_plus > 0:
                        expected_positive = (k_i_out_plus * k_j_in_plus) / (2 * w_plus)
                    else:
                        expected_positive = 0
                    
                    if w_minus > 0:
                        expected_negative = (k_i_out_minus * k_j_in_minus) / (2 * w_minus)
                    else:
                        expected_negative = 0
                    
                    modularity += w_ij - (expected_positive - expected_negative)

        modularity /= (2 * w_plus + 2 * w_minus)
        return modularity


