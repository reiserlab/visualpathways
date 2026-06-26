
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
from pathlib import Path
import sys
from dotenv import find_dotenv
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

import pandas as pd
import numpy as np
import re
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

# from utils import olc_client
# c = olc_client.connect(verbose=True)

# from neuprint import fetch_neurons, fetch_synapses, fetch_adjacencies
# from neuprint import NeuronCriteria as NC, SynapseCriteria as SC
# from neuprint.queries import fetch_all_rois, fetch_roi_hierarchy


def augmented_dendrogram(*args, **kwargs):
    """
    Create a dendrogram plot with additional annotations.
    This function generates a dendrogram using scipy's hierarchy.dendrogram function
    and adds annotations for the heights of the clusters and their inconsistency values.
    Parameters:
    *args : tuple
        Positional arguments passed to scipy.cluster.hierarchy.dendrogram.
    **kwargs : dict
        Keyword arguments passed to scipy.cluster.hierarchy.dendrogram.
        If 'no_plot' is set to True, the dendrogram will not be plotted.
    Returns:
    dict
        A dictionary of data structures computed to render the dendrogram.
        Refer to scipy.cluster.hierarchy.dendrogram documentation for details.
    Notes:
    - The function uses scipy's hierarchy.inconsistent to compute inconsistency values.
    - Red circles are plotted at the heights of the clusters.
    - Inconsistency values are annotated for clusters with an inconsistency coefficient of 3.
    """
    ddata = hierarchy.dendrogram(*args, **kwargs)
    R = hierarchy.inconsistent(*args, d=2)

    if not kwargs.get('no_plot', False):
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            plt.plot(x, y, 'ro')
            plt.annotate("%.3g" % y, (x, y), xytext=(+13, 10),
                         textcoords='offset points',
                         va='top', ha='center')
            
            j = np.where(args[0][:,2] == y)[0][0]
            
            # add inconsistency
            if R[j,2] == 3:
                plt.annotate("%.3g" % R[j,3], (x, y), xytext=(-13, 10),
                         textcoords='offset points',
                         va='top', ha='center')
            # if nb == 3:
            #     plt.annotate("%.2g" % incon, (x, y), xytext=(-11, 10),
            #              textcoords='offset points',
            #              va='top', ha='center')

    return ddata


def cluster_to_singleton(Z, cluster_ids):
    """
    # Given a cluster node, find the singleton node ids and
    # the corresponding row indices in the linkage matrix.

    Parameters:
    Z : ndarray
        The linkage matrix obtained from hierarchical clustering. It is a (n-1) x 4 matrix.
    cluster_ids : list or ndarray
        List or array of cluster ids to be converted to singletons.

    Returns:
    tuple
        singleton : ndarray
            Array of singleton node ids.
        Z_ind : ndarray
            Array of row indices in the linkage matrix corresponding to the cluster ids.
    """  
    n = int(Z.shape[0] + 1)
    singleton = np.empty([0], dtype=int)  # node id
    Z_ind = np.array(cluster_ids)  # row indices

    ii_todo = np.ndarray.flatten(np.int32(Z[cluster_ids, 0:2]))  # starting from here
    while len(ii_todo > 0):
        ii_next = np.empty([0], dtype=int)
        for i in ii_todo:
            if i < n:
                singleton = np.append(singleton, i)  # singleton, nothing more to do
            else:
                # i is a cluster id, need to search again
                Z_ind = np.append(Z_ind, i-n)  # cluster id -> row index
                ii_next = np.append(ii_next, np.int32(Z[i-n, 0:2]))
        ii_todo = ii_next

    # R = {'singleton': singleton, 'Z_ind': Z_ind}
    return singleton, Z_ind


def split_one_cluster(Z, T, ind):
    """
    Splits a single cluster from a hierarchical clustering result.
    Parameters:
    Z (ndarray): The linkage matrix resulting from hierarchical clustering.
    T (ndarray): The flat cluster assignment array.
    ind (int): The index of the cluster to split.
    Returns:
    ndarray: The updated flat cluster assignment array with the specified cluster split.
    """  
    n = int(Z.shape[0]+1)
    ii = np.int32(Z[ind, 0:2]-n)
    singleton, Z_ind = cluster_to_singleton(Z, ii[0])
    T[singleton] = max(T)+1

    return T


def split_cluster(Z, T):
    """
    Search for clusters to split.
    Parameters:
    Z (ndarray): The hierarchical clustering encoded as a linkage matrix.
    T (ndarray): The flat cluster assignment for each observation.
    Returns:
    tuple: A tuple containing:
        - T (ndarray): Updated flat cluster assignment for each observation.
        - id_splited (ndarray): Array of cluster ids that were split.
    """
    n = int(Z.shape[0]+1)
    id_splited = np.empty(shape=0, dtype=np.int32)
    L, M = hierarchy.leaders(Z, T) #get current cluster ids
    for i0 in L:
        # r,c = np.where(Z[:,0:2] == i1)
        # i2 = int(Z[r, 1-c])  # nnb cluster

        i0 = i0 - n
        if Z[i0, 2] > np.percentile(Z[:,2], 90): # height is large enough
            i12 = np.int32(Z[i0, 0:2]) - n
            Z[np.append(i0, i12), :]
            r_pc = Z[i0, 2] / max(Z[i12, 2])
            # r_cc = max(Z[i12, 2]) / min(Z[i12, 2])
            # if the 2 clusters height are vastly different
            if r_pc > 3:
                T = split_one_cluster(Z, T, i0)
                id_splited = np.append(id_splited, i0)
    
    return T, id_splited