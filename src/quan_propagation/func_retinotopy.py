import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Tuple, Optional
import neuprint
from neuprint import fetch_neurons, fetch_adjacencies, NeuronCriteria as NC, SynapseCriteria as SC

def topographic_product(map_coords: np.ndarray, 
                        feature_coords: np.ndarray,
                        max_nb: int = None) -> float:
    """
    Compute the Topographic Product (TP) measure for a neural map.
    
    The TP quantifies how well neighborhood relationships are preserved between
    map space (anatomical locations) and feature space (stimulus preferences).
    
    Parameters
    ----------
    map_coords : np.ndarray
        Array of shape (n_neurons, n_map_dims) containing the positions of 
        neurons in map space (e.g., 2D cortical coordinates).
    feature_coords : np.ndarray
        Array of shape (n_neurons, n_feature_dims) containing the positions 
        of neurons in feature space (e.g., preferred stimulus values).
    
    Returns
    -------
    tp : float
        The topographic product. Lower values indicate better topography.
        TP = 0 indicates perfect neighborhood preservation.
    
    References
    ----------
    Bauer, H. U., & Pawelzik, K. R. (1992). Quantifying the neighborhood 
    preservation of self-organizing feature maps. IEEE Transactions on 
    Neural Networks, 3(4), 570-579.
    
    Examples
    --------
    >>> # Simple 1D feature space mapped to 2D
    >>> map_coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    >>> feature_coords = np.array([[0], [1], [2], [3]])
    >>> tp = topographic_product(map_coords, feature_coords)
    >>> print(f"TP = {tp:.4f}")
    """
    n = len(map_coords)
    
    if n < 2:
        raise ValueError("Need at least 2 neurons to compute topographic product")
    
    if len(feature_coords) != n:
        raise ValueError("map_coords and feature_coords must have same length")
    
    if max_nb is None:
        max_nb = n
    
    # Compute pairwise distances in map space and feature space
    D_map = cdist(map_coords, map_coords, metric='euclidean')
    D_feat = cdist(feature_coords, feature_coords, metric='euclidean')
    
    log_products = []
    
    # For each neuron i
    for i in range(n):
        # Find k-th nearest neighbor in map space
        # Add small random noise to break ties
        map_dists = D_map[i, :].copy()
        map_dists[i] = np.inf  # Exclude self
        map_dists += np.random.uniform(0, 1e-10, n)  # Break ties randomly
        idx_map = np.argsort(map_dists)

        # Find k-th nearest neighbor in feature space
        feat_dists = D_feat[i, :].copy()
        feat_dists[i] = np.inf  # Exclude self
        feat_dists += np.random.uniform(0, 1e-10, n)  # Break ties randomly
        idx_feat = np.argsort(feat_dists)

        # For each neighborhood size k
        for k in range(1, max_nb):
            # indices of k-th nearest neighbors
            n_k_map = idx_map[k-1]
            n_k_feat = idx_feat[k-1]

            # Compute QF_ik and QG_ik ratios
            # QF_ik: ratio of map-space distance to k-th map-neighbor vs to k-th feature-neighbor
            QF_ik = D_map[i, n_k_map] / D_map[i, n_k_feat]

            # QG_ik: ratio of feature-space distance to k-th map-neighbor vs to k-th feature-neighbor
            QG_ik = D_feat[i, n_k_map] / D_feat[i, n_k_feat]
            
            # Compute geometric mean (via log)
            log_product = 0.5 * np.log(QF_ik * QG_ik)
            log_products.append(log_product)

    return np.mean(np.abs(log_products))
    # return log_products


def topographic_product_pvalue(map_coords: np.ndarray,
                               feature_coords: np.ndarray,
                               n_permutations: int = 1000,
                               ) -> Tuple[float, float]:
    """
    Compute topographic product and its statistical significance via permutation test.
    
    Parameters
    ----------
    map_coords : np.ndarray
        Array of shape (n_neurons, n_map_dims) containing neuron positions in map space.
    feature_coords : np.ndarray
        Array of shape (n_neurons, n_feature_dims) containing positions in feature space.
    n_permutations : int, optional
        Number of permutations for the statistical test. Default is 1000.
    
    Returns
    -------
    tp_observed : float
        The observed topographic product value.
    p_value : float
        The p-value from the permutation test. Lower p-values indicate
        statistically significant topography.
    
    Examples
    --------
    >>> # Test a linear map
    >>> map_coords = np.random.randn(20, 2)
    >>> feature_coords = map_coords[:, :1] + np.random.randn(20, 1) * 0.1
    >>> tp, p = topographic_product_pvalue(map_coords, feature_coords)
    >>> print(f"TP = {tp:.4f}, p = {p:.4f}")
    """
    # Compute observed TP
    tp_observed = topographic_product(map_coords, feature_coords)
    
    # Permutation test
    n = len(feature_coords)
    more_ordered_count = 0
    
    for _ in range(n_permutations):
        # Shuffle feature coordinates
        perm_indices = np.random.permutation(n)
        feature_shuffled = feature_coords[perm_indices]
        
        # Compute TP for shuffled data
        tp_shuffled = topographic_product(map_coords, feature_shuffled)

        # Count how many shuffled samples are more ordered (lower TP)
        if tp_shuffled <= tp_observed:
            more_ordered_count += 1
    
    # Compute p-value
    p_value = (more_ordered_count + 1) / (n_permutations + 1)
    
    return tp_observed, p_value


# # Example usage and demonstration
# if __name__ == "__main__":
#     print("="*60)
#     print("Topographic Product (TP) Implementation")
#     print("="*60)
    
#     # Example 1: Perfect linear map
#     print("\nExample 1: Perfect 1D linear map")
#     n_neurons = 20
#     map_coords_1d = np.column_stack([np.linspace(0, 10, n_neurons), 
#                                       np.zeros(n_neurons)])
#     feature_coords_1d = np.linspace(0, 10, n_neurons).reshape(-1, 1)
    
#     tp1 = topographic_product(map_coords_1d, feature_coords_1d, mc_samples=100)
#     print(f"TP = {tp1:.4f} (should be close to 0 for perfect map)")
    
#     # Example 2: Noisy linear map
#     print("\nExample 2: Noisy 1D linear map")
#     np.random.seed(42)
#     map_coords_noisy = np.column_stack([np.linspace(0, 10, n_neurons) + 
#                                          np.random.randn(n_neurons) * 0.5,
#                                          np.random.randn(n_neurons) * 0.5])
#     feature_coords_noisy = (np.linspace(0, 10, n_neurons) + 
#                             np.random.randn(n_neurons) * 0.3).reshape(-1, 1)
    
#     tp2, p2 = topographic_product_pvalue(map_coords_noisy, feature_coords_noisy,
#                                          n_permutations=1000, mc_samples=100)
#     print(f"TP = {tp2:.4f}, p-value = {p2:.4f}")
    
#     # Example 3: Random (no topography)
#     print("\nExample 3: Random arrangement (no topography)")
#     map_coords_random = np.random.randn(n_neurons, 2) * 5
#     feature_coords_random = np.random.randn(n_neurons, 1) * 5
    
#     tp3, p3 = topographic_product_pvalue(map_coords_random, feature_coords_random,
#                                          n_permutations=1000, mc_samples=100)
#     print(f"TP = {tp3:.4f}, p-value = {p3:.4f}")
    
#     print("\n" + "="*60)
#     print("Interpretation:")
#     print("- TP ≈ 0: Perfect neighborhood preservation")
#     print("- Lower TP values indicate better topographic organization")
#     print("- p-value < 0.05: Statistically significant topography")
#     print("="*60)


# define a function to count the number of swaps in bubble sort
def bubble_sort_count_swaps(arr):
    n = len(arr)
    swap_count = 0
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swap_count += 1
    return swap_count

# alt, works if values permutations of range(0, n)
def bubble_sort_swapcount(arr):
    n = len(arr)
    swap_count = 0
    for i in range(n):
        # find the position of the value i in arr
        pos = np.where(arr == i)[0][0]
        # remove this value from arr
        arr = np.delete(arr, pos)
        swap_count += pos
    return swap_count

# retinotopy index
def RI(S, A):
    """
    S: swap count
    A: avg swap count
    """
    return 1 - S / A

# avg swap count
def avg_swap_count(x):
    """
    Calculate the average number of swaps required by bubble sort

    Parameters
    ----------
    x : array-like
        Input array or list (only the length is used).

    Returns
    -------
    float
        The expected average number of swaps for bubble sort on a random
        permutation of length n = len(arr), given by (n-1)*(n-2)/4.
    """
    n = len(x)
    return (n-1) * (n - 2) / 4

# RI for 2 matched point sets, Euclidean distance
def RI_sets(P, Q):
    """
    Calculate the Retinotopy Index (RI) for two matched point sets.

    Parameters
    ----------
    P, Q : DataFrame or array-like
        arrays of points in d-dim, assume same dim and order-matched
    
    Returns
    -------
    float
        a list of Retinotopy Index (RI).
    """
    # convert P, Q to np.ndarray if they are DataFrames
    if isinstance(P, pd.DataFrame):
        P = P.values
    if isinstance(Q, pd.DataFrame):
        Q = Q.values

    # avg swap count
    A = avg_swap_count(P)
    
    # retinotopy index for each point in P
    ri = []

    # iterate over each pair of points in P and Q
    for i in range(len(P)):
        # compute the Euclidean distance between the point in P and all other points in P
        dists_P = np.linalg.norm(P - P[i], axis=1)
        # compute the order of distances, this is the P-order
        P_order = np.argsort(dists_P)
        
        # paired point in Q
        Q_paired = Q[i]

        # rearrange Q based on the P-order
        Q_ordered = Q[P_order]
        
        # compute distances of all points in Q to the point in Q that corresponds to the point in P
        dists_Q = np.linalg.norm(Q_ordered - Q_paired, axis=1)
        
        # compute the order of distances, this is the Q-order
        Q_order = np.argsort(dists_Q)
        
        # swap count for bubble sort on Q-order
        S = bubble_sort_swapcount(Q_order)
        
        # Calculate the Retinotopy Index (RI)
        ri.append(RI(S, A))

    return ri

def RI_rois(ids, roi1, roi2):
    """
    Calculate the Retinotopy Index (RI) for two rois.

    Parameters
    ----------
    ids : list
        List of bodyIds to fetch synapses for.
    roi1 : str
        Name of the first ROI.
    roi2 : str
        Name of the second ROI.

    Returns
    -------
    float
        The Retinotopy Index (RI) value.
    """
    syn_pre = neuprint.fetch_mean_synapses(
        NC(bodyId=ids), SC(type='post', rois=roi1)
    )
    syn_post = neuprint.fetch_mean_synapses(
        NC(bodyId=ids), SC(type='pre', rois=roi2)
    )
    
    ri = RI_sets(syn_post[['x', 'y', 'z']].values, syn_pre[['x', 'y', 'z']].values)
    
    return np.mean(ri)