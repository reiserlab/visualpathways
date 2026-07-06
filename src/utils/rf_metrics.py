"""
Receptive-field spatial-extent metrics over hex-grid weight distributions.

Utilities that quantify how many columns a weighted signal spans:
``count_col`` (columns needed to reach a fraction of the total weight),
``count_col_ahull`` (alpha-hull area of the retained columns, in hex-cell
units), and ``participation_ratio`` (effective number of columns). These
operate on weights indexed by ``'h,v'`` coordinate strings (the effwt_vis*
column convention) and are shared by the RF-size processing and the
figure/gallery scripts.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from utils.alpha_shape import alpha_shape_2d


def count_col(values, factor_thr):
    """
    Count the number of largest entries needed to reach a fraction (factor_thr) of
    the total sum, ignoring non-positive values.

    Accepts:
      - pandas.Series: values with an index (e.g., coordinate strings "x,y").
      - pandas.DataFrame with 1 column: treated as a Series.
      - numpy.ndarray / list-like 1D array: numeric values.

    Returns:
      - (count, kept_indices)
        * If input is a pandas.Series: kept_indices is a pandas Index of the
          original labels for the kept entries.
        * If input is array-like: kept_indices is a numpy array of integer
          positions (indices into the original array) for the kept entries.
    """
    if not (0 <= float(factor_thr) <= 1):
        raise ValueError("factor_thr must be in [0, 1]")

    # Normalize pandas DataFrame (single column) to Series
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("values DataFrame must have exactly one column")
        values = values.squeeze(axis=1)

    # Pandas Series path (preserve index labels)
    if isinstance(values, pd.Series):
        ser = pd.to_numeric(values, errors='coerce')
        ser = ser[ser > 0]
        if ser.empty:
            return 0, pd.Index([], dtype=values.index.dtype)

        sorted_values = ser.sort_values(ascending=False)
        cumsum = sorted_values.cumsum()
        total = cumsum.iloc[-1]
        threshold = float(factor_thr) * total
        mask_thr = cumsum <= threshold
        kept_indices = sorted_values[mask_thr].index
        return int(mask_thr.sum()), kept_indices

    # Numpy / list-like path (return integer positions)
    # arr = np.asarray(values)
    # if arr.ndim != 1:
    #     arr = arr.ravel()
    arr = np.asarray(values).ravel()
    # Filter positives, keep track of original positions
    pos_mask = arr > 0
    if not np.any(pos_mask):
        return 0, np.array([], dtype=int)
    pos_idx = np.nonzero(pos_mask)[0]
    arr_pos = arr[pos_mask] # positive values
    # Sort descending and compute cumulative sum
    order = np.argsort(arr_pos)[::-1] # indices that would sort arr_pos in descending order
    arr_sorted = arr_pos[order]
    cumsum = np.cumsum(arr_sorted)
    total = cumsum[-1]
    threshold = float(factor_thr) * total
    kept_mask = cumsum <= threshold
    kept_pos_in_pos = order[kept_mask]
    kept_indices = pos_idx[kept_pos_in_pos]
    return int(kept_indices.size), kept_indices


def count_col_ahull(df, thr_mode ="max", remove_frac = 0.1, min_remove_frac = 0.0, alpha=0.5, min_alpha=0.0, alpha_reduction=0.1):
    """

    """
    if not (0 <= float(remove_frac) <= 1):
        raise ValueError("remove_frac must be in [0, 1]")
    if not (0 <= float(min_remove_frac) <= float(remove_frac)):
        raise ValueError("min_remove_frac must be in [0, remove_frac]")

    # check df is a single column DataFrame or Series
    if not isinstance(df, (pd.DataFrame, pd.Series)) or (isinstance(df, pd.DataFrame) and df.shape[1] != 1):
        raise ValueError("df must be a pandas DataFrame with a single column or a pandas Series")

    # check df is not empty
    if df.values.sum() == 0:
        warnings.warn("weights sum to 0 Returning NaN results.")
        return [], set(), [np.nan], (np.nan, np.nan) , None, float(remove_frac), alpha

    # convert df to series if it's a single column DataFrame
    if isinstance(df, pd.DataFrame):
        df = df.squeeze(axis=1)

    # get xy coord from index
    xy_coord = df.index.to_series().str.split(',', expand=True).astype(float)
    xy_coord.columns = ['x', 'y']

    # # check df has at least 3 positive entries for alpha shape
    # if (df.values > 0).sum() < 3:
    #     warnings.warn("Not enough positive entries for alpha shape. Returning column count directly.")
    #     return [], set(), [(df.values > 0).sum()], (np.nan, np.nan) , None

    # thresholding effective weight points for alpha shape, "max" or "cumsum".
    # remove_frac starts at the user-supplied value, but if the kept points are
    # insufficient for an alpha shape (empty, fewer than 3 points, or collinear /
    # near-zero-area convex hull), we lower remove_frac by 0.1 and retry. If
    # remove_frac reaches min_remove_frac and the checks still fail, we give up
    # and return.
    if thr_mode not in ("max", "cumsum"):
        raise ValueError("Invalid thr_mode. Choose 'max' or 'cumsum'.")

    cur_frac = float(remove_frac)
    while True:
        if thr_mode == "max":
            # remove points with effective weight smaller than x% of max
            thr_val = cur_frac * df.values.max()
        else:  # cumsum
            # remove smallest x% of total weight
            weights_sorted = df.sort_values()
            cutoff = cur_frac * weights_sorted.sum()
            thr_val = weights_sorted[weights_sorted.cumsum() >= cutoff].iloc[0]

        # thresholding
        mask_pos = df.values > thr_val
        xy_coord_kept = xy_coord[mask_pos]
        df_kept = df[mask_pos]

        # Check if we have enough non-collinear points
        check_failed = False
        if len(xy_coord_kept) == 0:
            check_failed = True
        elif len(xy_coord_kept) < 3:
            check_failed = True
        else:
            try:
                hull = ConvexHull(xy_coord_kept.values)
                if hull.volume < 1e-10:  # Nearly zero area
                    check_failed = True
            except Exception:
                check_failed = True

        if not check_failed:
            break  # enough points, proceed with alpha shape

        if cur_frac <= min_remove_frac:
            # give up: return the (failing) result at the current threshold
            if len(xy_coord_kept) == 0:
                return None, None, [0], (np.nan, np.nan), xy_coord_kept, cur_frac, alpha
            com_x = np.average(xy_coord_kept['x'], weights=df_kept.values)
            com_y = np.average(xy_coord_kept['y'], weights=df_kept.values)
            return None, None, [xy_coord_kept.shape[0]], (com_x, com_y), xy_coord_kept, cur_frac, alpha

        # lower the threshold fraction and retry, not going below min_remove_frac
        cur_frac = round(cur_frac - 0.1, 10)
        if cur_frac < min_remove_frac:
            cur_frac = float(min_remove_frac)

    df = df_kept

    # compute com weighted by values
    com_x = np.average(xy_coord_kept['x'], weights=df.values)
    com_y = np.average(xy_coord_kept['y'], weights=df.values)

    # df.index is a square grid, with range x ~ [-15 19], range y ~ [7 72]
    # nb points in square grid is [+1 +1], in hex grid of side = 1 is [1+1/2, 3/2/sqrt(3)]
    # to morph into a regular hex grid with side length = 1
    # multiply x by 3/2 and y by sqrt(3)/2
    xy_coord_kept.loc[:, 'x'] = xy_coord_kept['x'] * 3/2
    xy_coord_kept.loc[:, 'y'] = xy_coord_kept['y'] * np.sqrt(3)/2

    # correction for edge points
    # for each xy_coord_kept, add six points around it with distance 1 in the corrected hex grid
    directions = np.array([[1, 0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2], [-1, 0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]])
    aux_points = []
    for _, row in xy_coord_kept.iterrows():
        for d in directions:
            aux_points.append([row['x'] + d[0], row['y'] + d[1]])
    # combine edge points with xy_coord_kept for alpha shape
    aux_points = np.array(aux_points)
    aux_points = np.vstack([xy_coord_kept.values, aux_points])

    edge_points, edges, area, used_alpha = alpha_shape_2d(aux_points, alpha=alpha, min_alpha=min_alpha, alpha_reduction=alpha_reduction)

    if len(area) != 1:
        area = [np.nan]

    # in units of hex cell area, which is 3*sqrt(3)/2 * (side length)^2.
    # For side length = 1 a in regular hex grid:
    area = [a / (3 * np.sqrt(3) / 2) for a in area]

    # go back to the original coord
    xy_coord_kept.loc[:, 'x'] = xy_coord_kept['x'] / (3/2)
    xy_coord_kept.loc[:, 'y'] = xy_coord_kept['y'] / (np.sqrt(3)/2)
    if len(edge_points) == 1:
        edge_points[0][:, 0] = edge_points[0][:, 0] / (3/2)
        edge_points[0][:, 1] = edge_points[0][:, 1] / (np.sqrt(3)/2)
    else:
        warnings.warn("Multiple ahulls.")

    return edge_points, edges, area, (com_x, com_y), xy_coord_kept, cur_frac, used_alpha


def participation_ratio(df):
    """
    Compute the participation ratio (1/Σpᵢ² , effective number of columns) from a
    weight distribution over hex grid positions.

    PR = (sum w_i)^2 / sum(w_i^2)

    Only positive weights are included. The result can be converted to a
    spatial area by multiplying by the area of one hex cell.

    Parameters
    ----------
    df : pd.Series or single-column pd.DataFrame
        Weights indexed by 'h,v' coordinate strings (same format as
        effwt_visr columns).

    Returns
    -------
    pr : float
        Participation ratio (effective number of columns).
        Returns np.nan if all weights are zero or negative.
    """
    w = np.asarray(df.values, dtype=float).ravel()
    w = w[w > 0]

    if len(w) == 0:
        return np.nan

    sum_w = w.sum()
    sum_w2 = (w ** 2).sum()

    return sum_w ** 2 / sum_w2
