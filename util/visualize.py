import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import networkx as nx

def reputation_graph(avg_rep, num_nodes, cul_G, fig, title, colorbar_label):
    """
    Args:
        avg_rep (list): list of floats for each community
        num_nodes (list): lost of num of nodes in each community
        cul_G: cluster graph (using igraph), each cluster is a node (undirected)
        fig (int): figure number, for allowing the use displaying multiple figures in one cell
        title (String): title of the graph
    """
    node_sizes = [8 * n for n in num_nodes]
    # node_sizes = [abs(r) * 500 for r in avg_rep]  # Scale up for better visibility

    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['#e86161', 'white', '#34c9d1'])

    # Normalize the reputations for color mapping to a fixed range -10 to 10
    min_val, max_val = -10, 10
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    # Define node colors based on normalized values
    node_colors = [cmap(norm(r)) for r in avg_rep]

    # Create labels with truncated reputations
    node_labels = {i: f'{avg_rep[i]:.2f}' for i in range(len(avg_rep))}

    # Define label colors based on node colors (white for dark nodes, black for light nodes)
    label_colors = ['white' if (c[0]*0.299 + c[1]*0.587 + c[2]*0.114) < 0.5 else 'black' for c in node_colors]

    # Use a different layout to spread out nodes more
    pos = nx.kamada_kawai_layout(cul_G)

    # Draw the graph
    plt.figure(fig, figsize=(14, 10))
    nx.draw_networkx_edges(cul_G, pos, alpha=0.5)

    # Draw nodes with sizes and colors
    nodes = nx.draw_networkx_nodes(cul_G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='lightgray', linewidths=1.5)

    # Add labels to nodes
    for i, (x, y) in pos.items():
        plt.text(x, y, node_labels[i], fontsize=8, ha='center', va='center', color=label_colors[i])

    # Add a color bar for the node colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(avg_rep)
    cbar = plt.colorbar(sm, ax=plt.gca(), label=colorbar_label)
    cbar.set_ticks([-10, 0, 10])  # Set colorbar ticks for better understanding
    cbar.set_ticklabels(['-10', '0', '10'])

    # Show plot
    plt.title(title)
    plt.show()