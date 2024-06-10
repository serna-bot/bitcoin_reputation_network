import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#Example of using it
#top_positive_nodes, top_negative_nodes = extract_top_weight_nodes(adjacency_matrix, top_percentage=0.05)
#subgraph = create_top_weight_subgraph(adjacency_matrix, top_positive_nodes, top_negative_nodes)
#visualize_top_weight_subgraph(subgraph)
def extract_top_weight_nodes(A, top_percentage=0.1):
    """
    Extract top positive and negative weight nodes based on incoming edge weights.

    Args:
        A (np.array): adjacency matrix
        top_percentage (float): percentage of top nodes to extract

    Returns:
        tuple: sets of top positive and negative weight nodes
    """
    incoming_weights = np.sum(A, axis=0)
    sorted_nodes = np.argsort(incoming_weights)

    top_count = int(len(sorted_nodes) * top_percentage)
    top_positive_nodes = set(sorted_nodes[-top_count:])
    top_negative_nodes = set(sorted_nodes[:top_count])

    return top_positive_nodes, top_negative_nodes

def create_top_weight_subgraph(A, top_positive_nodes, top_negative_nodes):
    """
    Create a subgraph containing the top positive and negative weight nodes.

    Args:
        A (np.array): adjacency matrix
        top_positive_nodes (set): set of top positive weight nodes
        top_negative_nodes (set): set of top negative weight nodes

    Returns:
        nx.DiGraph: subgraph with top weight nodes
    """
    G = nx.DiGraph()
    for node in top_positive_nodes:
        G.add_node(node, weight='positive')
    for node in top_negative_nodes:
        G.add_node(node, weight='negative')
    for u in top_positive_nodes:
        for v in top_positive_nodes.union(top_negative_nodes):
            if A[u, v] != 0:
                G.add_edge(u, v, weight=A[u, v])

    for u in top_negative_nodes:
        for v in top_positive_nodes.union(top_negative_nodes):
            if A[u, v] != 0:
                G.add_edge(u, v, weight=A[u, v])

    return G

def visualize_top_weight_subgraph(G):
    print(1)
    """
    Visualize the subgraph with top weight nodes.

    Args:
        G (nx.DiGraph): subgraph with top weight nodes
    """
    fig, ax = plt.subplots(figsize=(18, 15))
    pos = nx.spring_layout(G)
    print(2)
    positive_nodes = [node for node in G.nodes() if G.nodes[node]['weight'] == 'positive']
    negative_nodes = [node for node in G.nodes() if G.nodes[node]['weight'] == 'negative']
    print(3)
    nx.draw_networkx_nodes(G, pos, nodelist=positive_nodes, node_size=30, node_color='blue', node_shape='o', alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=negative_nodes, node_size=30, node_color='red', node_shape='s', alpha=0.9)
    print(4)
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0]
    print(5)
    positive_weights = [G[u][v]['weight'] * 0.01 for u, v in positive_edges]
    if positive_weights:
        nx.draw_networkx_edges(G, pos, edgelist=positive_edges, edge_color='blue', width=positive_weights)
    print(6)
    negative_weights = [abs(G[u][v]['weight']) * 0.01 for u, v in negative_edges]
    if negative_weights:
        nx.draw_networkx_edges(G, pos, edgelist=negative_edges, edge_color='red', width=negative_weights, style='dashed')
    print(7)

    plt.title("Top Weight Nodes Subgraph")
    plt.show()




