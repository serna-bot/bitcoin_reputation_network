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
            if old_cluster['neighbors'][neighboring_cluster].has(node):
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
        for node in cluster:
            for neighbors in cluster:
                if neighbors != node:
                    w_out = self.A[node]
        pass


