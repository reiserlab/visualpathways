""" some functions for graph analysis """

import numpy as np
import pandas as pd
import networkx as nx
from neuprint import fetch_shortest_paths, fetch_adjacencies, fetch_neurons
from neuprint import NeuronCriteria as NC

def collect_shortest_paths(source_id, target_id, edgelist=None, min_weight=10, timeout=5.0):
    """
    Collect all shortest paths from sources to targets and store path lengths.
    Expand it to work with edgelists (2024-11). If edgelist is None, use neuprint's fetch_shortest_paths.

    Parameters:
    source_id (list): List of source neuron IDs.
    target_id (list): List of target neuron IDs.
    min_weight (int): Minimum weight for the paths. Default is 10.
    timeout (int): Timeout for fetching paths. Default is 5 seconds.
    edgelist (None or pd.DataFrame): Default is None, using neuprint's fetch_shortest_paths.

    Returns:
    pd.DataFrame: DataFrame containing source, target, and path length.
    list: List of all paths.
    """
    if edgelist is not None and not isinstance(edgelist, pd.DataFrame):
        raise ValueError("edgelist must be either None or a pandas DataFrame")
    # check edgelist DataFrame containing edges with columns ['source', 'target', 'weight']
    if edgelist is not None:
        if 'source' not in edgelist.columns or 'target' not in edgelist.columns or 'weight' not in edgelist.columns:
            raise ValueError("edgelist must contain columns ['source', 'target', 'weight']")

    # convert source_id and target_id to 1d array if they are not
    source_id = np.array(source_id).flatten()
    target_id = np.array(target_id).flatten()
    # for neuprint
    if edgelist is None:
        path_len = np.zeros((0, 3))
        paths_all = []
        for i in source_id:
            for j in target_id:
                paths = fetch_shortest_paths(i, j, min_weight=min_weight, timeout=timeout)
                # if path is not empty or timeout
                if len(paths) > 0:
                    path_len = np.vstack((path_len, np.array([i, j, sum(paths['path'] == 0)]))) # pick one path
                    paths['layer'] = -1
                    # add levels to each path
                    for k in paths['path'].unique():
                        n_layers = sum(paths['path'] == k)
                        paths.loc[paths['path'] == k, 'layer'] = np.arange(1, n_layers+1)
                    paths_all.append(paths)
    # for flywire
    else:
        path_len = pd.DataFrame(columns=['source', 'target', 'path_len'])
        path_len = path_len.astype({'source': 'int64', 'target': 'int64', 'path_len': 'int'})        
        paths_all = []
        # filter edgelist by min_weight
        edgelist = edgelist[edgelist['weight'] >= min_weight]
        # create a graph from edgelist
        g = nx.from_pandas_edgelist(edgelist, create_using=nx.DiGraph, edge_attr=True)
        for i in source_id:
            for j in target_id:
                paths = [p for p in nx.all_shortest_paths(g, source=i, target=j)]
                # paths = fetch_shortest_paths_from_edgelist(i, j, edgelist, min_weight=10, max_hops=3)
                # if path is not empty or timeout
                if len(paths) > 0:
                    path_len = pd.concat([path_len, pd.DataFrame([{'source': i, 'target': j, 'path_len': len(paths[0])}])], ignore_index=True)
                    paths_all += paths

                
    path_len = pd.DataFrame(path_len, columns=['source', 'target', 'path_len'])
    path_len = path_len.astype({'source': 'int', 'target': 'int', 'path_len': 'int'})

    return path_len, paths_all



def adj_trans_matrix(ids, min_total_weight=1):
    """
    Generate adjacency and transition matrices for given neuron body IDs from neuprint.
    Normalizes the adjacency matrix by columns
    (fetch_adjacencies() only keeps typed cells, instead, use fetch_neurons()'s upstream).
    Create a transition matrix by transposing the adjacency matrix (now normalizing by rows).
    
    Parameters:
    ids (list or array-like): List or array of neuron body IDs.
    min_total_weight (int): Minimum weight. Default is 1.
    Returns:
    tuple: A tuple containing:
        - m_adj (DataFrame): Adjacency matrix normalized by columns.
        - m_trans (DataFrame): Backward-transition matrix normalized by rows.
        - neuron_df (DataFrame): DataFrame containing neuron information.
    """

    neu, _ = fetch_neurons(NC(bodyId=ids), )
    # query adj matrix from bodyids
    neuron_df, connection_df = fetch_adjacencies(sources=ids, targets=ids, min_total_weight=min_total_weight)
    
    # sum up rois
    conn_df_sum = connection_df.groupby(['bodyId_pre', 'bodyId_post'])['weight'].sum().reset_index()
    mat = conn_df_sum.pivot(index='bodyId_pre', columns='bodyId_post', values='weight')

    # make it square
    mat = mat.reindex(index=ids, columns=ids, fill_value=0)
    # adj matrix, normalized by "upstream"
    mat = mat.div(neu.set_index('bodyId').loc[ids, 'upstream'], axis=1)

    mat = mat.fillna(0)
    
    # adj matrix
    m_adj = mat.copy()
    
    # backward-transition matrix, normalized by row 
    m_trans = mat.copy().T
    
    return m_adj, m_trans, neuron_df
