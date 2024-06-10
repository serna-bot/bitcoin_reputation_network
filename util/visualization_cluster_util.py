import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# Example usage:
#clusters = [set(range(i * 100, (i + 1) * 100)) for i in range(58)]
#clusters.append(set(range(5800, 5881)))
#cluster_graph = create_cluster_graph(adjacency_matrix, clusters, use_modularity=True)

# Visualize the cluster graph
#visualize_cluster_graph(cluster_graph)

def calculate_modularity_between_clusters(A, C1, C2):
    """
    Calculate the modularity between two clusters.

    Args:
        A (np.array): adjacency matrix
        C1 (set): set of node indices in the first cluster
        C2 (set): set of node indices in the second cluster

    Returns:
        float: Modularity between the two clusters
    """
    N = A.shape[0]

    w_plus = np.sum(A[A > 0])
    w_minus = np.sum(A[A < 0])

    if w_plus + w_minus == 0:
        return 0  # Avoid division by zero

    modularity = 0
    for i in C1:
        for j in C2:
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

#This is the first visualization function that we draw the graph by grouping one cluster as one node
def create_cluster_graph(A, clusters, use_modularity=False):
    """
    Create a cluster graph based on the given adjacency matrix and clusters.

    Args:
        A (np.array): adjacency matrix
        clusters (list of sets): list of sets, each set contains node indices of a cluster
        use_modularity (bool): whether to use modularity or number of connections as edge weights

    Returns:
        nx.Graph: Cluster graph
    """
    incoming_weights = np.sum(A, axis=0)
    G = nx.Graph()
    for i, cluster in enumerate(clusters):
        num_nodes = len(cluster)
        cluster_weight = np.sum([incoming_weights[node] for node in cluster])
        G.add_node(i, size=num_nodes, weight=cluster_weight, color=abs(cluster_weight),
                   shape='o' if cluster_weight > 0 else 's')
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if use_modularity:
                edge_weight = calculate_modularity_between_clusters(A, clusters[i], clusters[j])
                if edge_weight > 0:  # Only add edge if modularity is positive
                    G.add_edge(i, j, weight=edge_weight)
            else:
                edge_weight = sum(1 for u in clusters[i] for v in clusters[j] if A[u, v] != 0 or A[v, u] != 0)
                if edge_weight != 0:
                    G.add_edge(i, j, weight=edge_weight)

    return G

def visualize_cluster_graph(G, similarity_method, use_modularity = False):
    """
    Visualize the cluster graph.

    Args:
        G (nx.Graph): Cluster graph
    """
    pos = nx.spring_layout(G)
    
    sizes = [G.nodes[node]['size'] * 10 for node in G.nodes()]
    colors = [G.nodes[node]['color'] for node in G.nodes()]
    shapes = ['o' if G.nodes[node]['weight'] > 0 else 's' for node in G.nodes()]
    unique_shapes = set(shapes)

    fig, ax = plt.subplots(figsize=(18, 15))

    for shape in unique_shapes:
        node_list = [node for node in G.nodes() if G.nodes[node]['shape'] == shape]
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_size=[sizes[node] for node in node_list],
                               node_color=[colors[node] for node in node_list], node_shape=shape, ax=ax)
    
    edges = G.edges()
    if use_modularity:
        weights = [G[u][v]['weight']*2000 for u, v in edges]
    else:
        weights = [G[u][v]['weight']*0.02 for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, ax=ax)
    
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, ax=ax)

    plt.title(similarity_method + " Louvain Cluster Graph")
    plt.show()

