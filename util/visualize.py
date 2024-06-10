import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import networkx as nx

def reputation_graph(avg_rep, num_nodes, cul_G, fig, title):
    """
    Args:
        avg_rep (list): list of floats for each community
        num_nodes (list): lost of num of nodes in each community
        cul_G: cluster graph (using igraph), each cluster is a node (undirected)
        fig (int): figure number, for allowing the use displaying multiple figures in one cell
        title (String): title of the graph
    """
    node_sizes = [n * 20 for n in num_nodes]
    node_sizes = [abs(r) * 500 for r in avg_rep]  # Scale up for better visibility

    cmap_neg = mcolors.LinearSegmentedColormap.from_list('neg_cmap', ['#e86161', 'white'])
    cmap_pos = mcolors.LinearSegmentedColormap.from_list('pos_cmap', ['white', '#34c9d1'])
    norm_neg = mcolors.Normalize(vmin=min(avg_rep), vmax=0)
    norm_pos = mcolors.Normalize(vmin=0, vmax=max(avg_rep))

    # Define node colors based on the sign of avg_rep
    node_colors = [cmap_neg(norm_neg(r)) if r < 0 else cmap_pos(norm_pos(r)) for r in avg_rep]

    node_labels = {i: f'{avg_rep[i]:.2f}' for i in range(len(avg_rep))}

    # Use a different layout to spread out nodes more
    pos = nx.kamada_kawai_layout(cul_G)

    # Draw the graph
    plt.figure(fig, figsize=(14, 10))
    nx.draw_networkx(cul_G, 
                    pos=pos,
                    node_size=node_sizes, 
                    node_color=node_colors, 
                    labels=node_labels,  # Add labels
                    with_labels=True,
                    cmap=plt.get_cmap('coolwarm'),
                    edgecolors='lightgray',  # Add light gray border around nodes
                    linewidths=1.5)

    # Show plot
    plt.title(title)
    plt.show()