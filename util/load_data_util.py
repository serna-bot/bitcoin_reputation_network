import pandas as pd
import networkx as nx
import igraph as ig

def loadData_iGraph(src, preprocessing=None):
    """
    Load pandas df to create iGraph graph

    Args:
        src (string): src to .csv file
        preprocessing (function): preprocessing function for edge weights

    Returns: 
        iGraph.Graph
    """
    df = pd.read_csv(src)
    if preprocessing:
        df["weight"] = df['weight'].apply(preprocessing)
    G = ig.Graph.DataFrame(df, directed=True)
    print('iGraph Graph loaded.')
    return G
    
def loadData_networkX(src, preprocessing=None):
    """
    Load pandas df to create networkX graph

    Args:
        src (string): src to .csv file
        preprocessing (function): preprocessing function for edge weights

    Returns: 
        networkX.Graph
    """
    df = pd.read_csv(src)
    if preprocessing:
        df["weight"] = df['weight'].apply(preprocessing)
    df.columns = [None] * len(df.columns)

    edgelist_df = df.iloc[:, 0:2].to_numpy()
    weights = df.iloc[:, 0:3].to_numpy()
    G = nx.from_edgelist(edgelist_df, create_using = nx.DiGraph)
    G.add_weighted_edges_from(weights)
    print('networkX Graph loaded.')
    return G

def shiftEdgeWeights(weight):
    """
    Shift all edge weights by a number

    Args:
        weight (callback): for pandas, singular cells

    Returns: 
        int
    """
    SHIFT = 10
    return weight + SHIFT

def transformEdgeWeights(weight):
    """
    Transforms all edge weights by a number

    Args:
        weight (callback): for pandas, singular cells

    Returns: 
        int
    """
    if weight < 0:
        return -1 * 1/ weight
    elif weight > 0:
        return weight
    return weight

def load_subnetwork_iGraph(src, negative_edges = True, preprocessing=None):
    """
    Get subnetwork of only negative or positive edges

    Args:
        src (string): src to .csv file
        negative_edges (Boolean): whether only negative 
        edges or positive edges
    
    Returns:
        iGraph
    """
    df = pd.read_csv(src)
    if negative_edges:
        df = df[df['weight'] <= 0]
    else:
        df = df[df['weight'] >= 0]

    if preprocessing:
        df["weight"] = df['weight'].apply(preprocessing)
    G = ig.Graph.DataFrame(df, directed=True)
    print('iGraph Sub-Graph loaded.')
    return G

def load_subnetwork_networkX(src, negative_edges = True, preprocessing=None):
    df = pd.read_csv(src)
    if negative_edges:
        df = df[df['weight'] <= 0]
    else:
        df = df[df['weight'] >= 0]

    if preprocessing:
        df["weight"] = df['weight'].apply(preprocessing)
    df.columns = [None] * len(df.columns)

    edgelist_df = df.iloc[:, 0:2].to_numpy()
    weights = df.iloc[:, 0:3].to_numpy()
    G = nx.from_edgelist(edgelist_df, create_using = nx.DiGraph)
    G.add_weighted_edges_from(weights)
    print('networkX Sub-Graph loaded.')
    return G