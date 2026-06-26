"""
Backup of unused functions extracted from partially-used utility files.

These functions were identified as unused throughout the codebase and are preserved
here before removal from their source files.

Additionally, the following ENTIRE files are unused and were left in place:
  - helper_jh.py (duplicate of helper.py's slugify and num_expand)
  - module_analysis.py (all 6 functions unused)
  - ROI_plots.py (all 3 functions unused)
  - ROI_voxels.py (all 3 functions unused)
  - neuron_bag_jh.py (NeuronBag class unused)
  - summary_plot_preprocessor.py (SummaryPlotPreprocessor class unused)
"""


# =============================================================================
# From ROI_calculus.py
# =============================================================================

def syn_per_col_count(
    syn_df,
    roi_str='ME(R)',
    samp=2,
):
    """
    Count number of synapses in each column. Synapses are trimmed.

    Parameters
    ----------
    syn_df : pd.DataFrame
        dataframe with 'bodyId', 'x', 'y', 'z' columns
    roi_str : str
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    samp : int
        sub-sampling factor for depth bins

    Returns
    -------
    syn_col_df : pd.DataFrame
        'hex1_id' : int
            defines column
        'hex2_id' : int
            defines column
        'col_count' : int
            number of synapses in that column (across all depth bins)
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"

    _, _, syn_split_df = trim_by_col_count(
        syn_df
      , roi_str=roi_str
      , samp=samp)
    syn_col_df = syn_split_df[['bodyId','hex1_id','hex2_id','col_count']]\
        .drop_duplicates()\
        .reset_index(drop=True)
    syn_col_df = syn_col_df.groupby(['hex1_id','hex2_id'])['col_count']\
        .sum()\
        .to_frame()\
        .reset_index()
    return syn_col_df


def trim_by_col_count(
    syn_df,
    roi_str='ME(R)',
    samp=2,
    cumsum_min=0.775,
    cumsum_fix=0.999,
):
    """
    Trimming off the "outlier" synapses.

    For each neuron, find a lower threshold on the number synapses in a column in order to retain
        those synapses. This threshold equals the number of synapses in the `rank_thre`'th largest
        column, where `rank_thre` is computed from cell-type information. Namely, we use an elbow
        method on the median synapse count per column vs. rank of column (the column with the
        largest number of synapses has rank 1, the column with the second largest number of
        synapses has rank 2 etc.).

    Parameters
    ----------
    syn_df : pd.DataFrame
        DataFrame with 'bodyId', 'x', 'y', 'z' columns
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    samp : int, default=2
        sub-sampling factor for depth bins
    cumsum_min : float
        minimum fraction of cumulative sum of synapses in columns after trimming
        if knee finder gives a lower fraction then we find the rank with fraction cumsum_fix.
            the value 0.775 was obtained as a dip in the bimodal distribution of the cumsum fraction
            from the knee finder for cell-types in ME with at least 1000 synapses.
    cumsum_fix : float
        a fixed fraction of cumulative sum of synapses in columns that is used if
        knee finder gives a lower fraction than cumsum_min.
            the value 0.999 is such that almost no synapses get trimmed off for large cells except
            for those in columns with very few synapses.

    Returns
    -------
    rank_thre : int
        max. rank of columns that will retain their synapses
    cumsum_thre : float
        cumulative fraction of retained synapses for the median synapse count
    syn_split_df : pd.DataFrame
        'bodyId' : int
            bodyId
        'hex1_id' : int
            defines column
        'hex2_id' : int
            defines column
        'col_count' : int
            number of synapses in that column (across all depth bins)
        'count_thre' : float
            'count_thre' is the lower threshold on 'col_count' that was used to remove synapses
        'bin' : float
            depth bin of synapse
        'count' : int
            number of synapses in that column and depth bin
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"

    if ('hex1_id' in syn_df.columns)\
      & ('hex2_id' in syn_df.columns)\
      & ('bin' in syn_df.columns):
        syn_size_df = syn_df.copy()
    else:
        hex_df = find_hex_ids(syn_df, roi_str=roi_str)
        depth_df = find_depth(syn_df, roi_str=roi_str, samp=samp)
        syn_size_df = pd.concat([syn_df.reset_index(drop=True), hex_df, depth_df], axis=1)

    #count synapses in each column and each depth bin
    syn_split_df = syn_size_df\
        .groupby(['bodyId','hex1_id','hex2_id','bin'])['x']\
        .count()\
        .to_frame()\
        .rename(columns={'x': 'count'})\
        .reset_index()

    #count synapses in each column; only keep columns
    syn_col_df = syn_split_df\
        .groupby(['bodyId','hex1_id','hex2_id'])['count']\
        .apply('sum')\
        .to_frame()\
        .reset_index()
    #count all synapses of a neuron
    syn_col_df['count_all'] = syn_col_df\
        .groupby('bodyId')['count']\
        .transform('sum')\
        .values
    syn_col_df['frac'] = syn_col_df['count'].values / syn_col_df['count_all'].values
    syn_col_df.sort_values(by=['bodyId','frac'], ascending=False, inplace=True)
    #ordered cumulative sum of synapses in each column
    syn_col_df['cum_sum'] = syn_col_df.groupby('bodyId')['frac'].transform('cumsum').values
    #rank columns
    syn_col_df['rank'] = syn_col_df.groupby('bodyId')['count'].cumcount()+1
    cumsum_per_rank_df = syn_col_df.groupby(['bodyId','rank'])['cum_sum'].first().unstack(-1, 1)

    #find elbow in cumsum
    n_pts = int(syn_col_df['rank'].max())
    x_val = np.zeros(n_pts + 1)
    y_val = np.zeros(n_pts + 1)
    x_val[1:] = np.linspace(1, n_pts, n_pts)
    y_val[1:] = cumsum_per_rank_df.median(0)
    kneedle = kneed.KneeLocator(x_val, y_val, S=1.0, curve="concave", direction="increasing")
    rank_thre = kneedle.knee
    if (rank_thre is None) or (y_val[int(rank_thre)]<cumsum_min):
        find_rank = np.where(y_val>=cumsum_fix)[0]
        if find_rank.shape[0]==0:
            rank_thre = y_val.size - 1
        else:
            rank_thre = find_rank[0]
    rank_thre =  int(rank_thre)
    cumsum_thre = y_val[rank_thre]
    #find threshold on count
    syn_col_df = syn_col_df[syn_col_df['rank']<=int(rank_thre)]
    count_thre_df = syn_col_df\
        .groupby('bodyId')['count']\
        .last()\
        .to_frame()\
        .rename(columns={'count': 'count_thre'})\
        .reset_index()
    syn_split_df = syn_split_df.merge(count_thre_df, on='bodyId')

    #count synapses in each column, same format as syn; only keep synapses above count_thre
    syn_split_df['col_count'] = syn_split_df\
        .groupby(['bodyId','hex1_id','hex2_id'])['count']\
        .transform('sum')\
        .values
    syn_split_df = syn_split_df[syn_split_df['col_count'] >= syn_split_df['count_thre']]

    return rank_thre, cumsum_thre, syn_split_df


def find_per_columnbin_coverage(
    syn_df,
    roi_str='ME(R)',
    samp=2,
):
    """
    Count number of neurons that have synapses in each column; the synapses were trimmed off.

    Parameters
    ----------
    syn_df : pd.DataFrame
        DataFrame with 'bodyId', 'x', 'y', 'z' columns
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    samp : int, default=2
        sub-sampling factor for depth bins

    Returns
    -------
    col_count_df : pd.DataFrame
        'hex1_id' : int
            defines column
        'hex2_id' : int
            defines column
        'bodyId_col_count' : int
            number of neurons in that column (across all depth bins)
        'bodyId_count' : int
            number of neurons in that column and depth bin
        'bin' : float
            depth bin of synapse
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"

    _, _, syn_split_df = trim_by_col_count(syn_df, roi_str=roi_str, samp=samp)

    #count how many neurons in each column and depth
    count_df = syn_split_df\
        .groupby(['hex1_id','hex2_id','bin'])['bodyId']\
        .count()\
        .to_frame()\
        .rename(columns={'bodyId': 'count'})\
        .reset_index()
    col_count_df = syn_split_df\
        .groupby(['hex1_id','hex2_id'])['bodyId']\
        .nunique()\
        .to_frame()\
        .rename(columns={'bodyId': 'col_count'})\
        .reset_index()

    count_df = count_df.merge( col_count_df, 'left', on=['hex1_id','hex2_id'] )

    return count_df


def find_per_columnbin_spanned_no_cols(
    syn_df,
    roi_str='ME(R)',
    samp=2,
    trim=True,
):
    """
    For each depth and neuron, count number of columns that synapses lie in.
        Option to trim synapses.

    Parameters
    ----------
    syn_df : pd.DataFrame
        dataframe with 'bodyId', 'x', 'y', 'z' columns
    roi_str : str
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    samp : int
        sub-sampling factor for depth bins
    trim : boolean
        if "outlier" synapses should be dropped, see `trim_by_col_count` for explanation.

    Returns
    -------
    size_df : pd.DataFrame
        'bodyId' : int
            body ID of neuron
        'bin' : int
            depth bin of synapse after trimming
        'size' : int
            number of columns for remaining synapses
    rank_thre : int
        max. rank of columns to keep for each neuron, set to -1 if trim=False
    cumsum_thre : float
        cumulative fraction of synapses that is reached at that rank for the median,
            set to -1 if trim=False
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"

    if trim:
        rank_thre, cumsum_thre, syn_hex_df =\
            trim_by_col_count(syn_df, roi_str=roi_str, samp=samp)
    else:
        hex_df = find_hex_ids(syn_df, roi_str=roi_str)
        depth_df = find_depth(syn_df, roi_str=roi_str, samp=samp)
        syn_hex_df = pd.concat([
                syn_df.reset_index(drop=True)
              , hex_df[['hex1_id','hex2_id']]
              , depth_df]
          , axis=1)
        #dummies
        rank_thre = -1
        cumsum_thre = -1
    syn_hex_df['hex1_id'] = syn_hex_df['hex1_id'].astype(int)
    syn_hex_df['hex2_id'] = syn_hex_df['hex2_id'].astype(int)
    #size per bin is the number of columns for remaining synapses
    size_df = syn_hex_df\
        .groupby(['bodyId','bin'])\
        .apply(count_hex_loc)\
        .reset_index()

    return size_df, rank_thre, cumsum_thre


def count_hex_loc(group):
    """
    Helper function used in DataFrameGroupBy.apply to count columns

    Parameters
    ----------
    group : pd.DataFrame
        'hex1_id' : int
            defines column
        'hex2_id' : int
            defines column

    Returns
    -------
    output: pd.Series
        'size': int
            count how many different tuples (hex1_id, hex2_id) exist
    """
    _, idcs = np.unique(group[['hex1_id','hex2_id']].values, axis=0, return_index=True)
    count_dict = {}
    count_dict['size'] = idcs.shape[0]
    return pd.Series(count_dict)


def find_layers(
    xyz_df,
    roi_str='ME(R)',
    samp=1,
):
    """
    For a dataframe of 3D points, find which layer the points lie in

    Parameters
    ----------
    xyz_df : pd.DataFrame
        DataFrame with 'x', 'y', 'z' columns
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    samp : int, default=1
        sub-sampling factor for depth bins

    Returns
    -------
    layer_df : pd.DataFrame
        'layer' : int
            layer numbers (starting from 1 at the top) that the corresponding points xyz_df lie in
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"

    #load layer tresholds
    depth_bdry = load_layer_thre(roi_str=roi_str)

    #find fine depth
    depth_df = find_depth(xyz_df, roi_str=roi_str, samp=samp)

    layer_ass = np.empty(xyz_df.shape[0])
    layer_ass[:] = np.nan
    for i in range(len(depth_bdry)-1):
        layer_ass[
            (depth_df['depth'] >  depth_bdry[i])\
          & (depth_df['depth'] <= depth_bdry[i + 1])
        ] = i + 1
    layer_df = pd.DataFrame(layer_ass, columns=['layer'])

    return layer_df


# =============================================================================
# From ROI_columns.py
# =============================================================================

def find_holes(pts_all, pts):
    """
    Find the indices of pts_all with holes. A "hole" is a point in pts_all that is not in pts.

    We only consider 3 types of holes:

        1. Holes (x,y) in which the 4 points (x+-1,y), (x,y+-1) are not holes,
        2. Holes (x,y) in which the 2 points (x+-1,y) are not holes but
            at least one of (x,y+-1) is,
        3. Holes (x,y) in which the 2 points (x,y+-1) are not holes but
            at least one of (x+-1,y) is.

    Parameters
    ----------
    pts_all : np.ndarray
        list of 2D integer lattice points
    pts : np.ndarray
        array of 2D integer lattice points that is a sublattice of pts_all

    Returns
    -------
    ids_vh : list[int]
        list of integers that contains the indices of type 1 holes
    ids_h : list[int]
        list of integers that contains the indices of type 2 holes
    ids_v : list[int]
        list of integers that contains the indices of type 3 holes
    """

    #find edge points
    hull = alphashape.alphashape(pts_all, 1)
    hull_pts = np.asarray(hull.exterior.coords)
    edge_ids = np.zeros(hull_pts.shape[0],dtype=int)
    for i in range(hull_pts.shape[0]):
        edge_ids[i] = np.argmin( np.linalg.norm(hull_pts[i][np.newaxis,:]-pts_all, axis=1) )

    #find holes that are not on boundary. dinstinguish if 4 neighbors, 2 horizontal or 2 vertical
    pts_all_df = pd.DataFrame(pts_all, columns=['x','y'])
    pts_df = pd.DataFrame(pts, columns=['x','y'])
    pts_df['sub'] = 1
    pts_all_df = pts_all_df.merge(pts_df, 'left', on=['x','y'])
    sub_ids = pts_all_df[pts_all_df['sub']==1].index.values

    missing_ids = list(set(pts_all_df.index.values)-set(sub_ids))
    pts_all_df['h_neighbors'] = pts_all_df.apply(
        lambda r: list(set(np.where(
            (np.abs(pts_all_df['x'] - r.x) == 1)
          & (np.abs(pts_all_df['y'] - r.y) == 0))[0]) - set(missing_ids))
      , axis=1)
    pts_all_df['v_neighbors'] = pts_all_df.apply(
        lambda r: list(set(np.where(
            (np.abs(pts_all_df['x'] - r.x) == 0)
          & (np.abs(pts_all_df['y'] - r.y) == 1))[0]) - set(missing_ids))
      , axis=1)
    pts_all_df['no_v_neighbors'] = pts_all_df.apply(
        lambda r: len(r['v_neighbors'])
      , axis=1
    )

    pts_all_df['no_h_neighbors'] = pts_all_df.apply(
        lambda r: len(r['h_neighbors'])
      , axis=1
    )

    ids_vh = pts_all_df.iloc[missing_ids][
            (pts_all_df.iloc[missing_ids]['no_v_neighbors'] == 2) &\
            (pts_all_df.iloc[missing_ids]['no_h_neighbors'] == 2)]\
        .index\
        .values
    ids_h = pts_all_df.iloc[missing_ids][
            (pts_all_df.iloc[missing_ids]['no_v_neighbors'] <  2) &\
            (pts_all_df.iloc[missing_ids]['no_h_neighbors'] == 2)]\
        .index\
        .values
    ids_v = pts_all_df.iloc[missing_ids][
            (pts_all_df.iloc[missing_ids]['no_v_neighbors'] == 2) &\
            (pts_all_df.iloc[missing_ids]['no_h_neighbors'] <  2)]\
        .index\
        .values

    return ids_vh, ids_h, ids_v


def smooth_center_columns_w_median(
    roi_str='ME(R)',
    r_neighb=2,
):
    """
    Override column pins by a median column (computed over a local neighborhood of radius r_neighb)
    Missing columns, which form a hole of types 1-3, are filled in.

    This function changes the pickle file roi_str[:-3]+'_col_center_pins.pickle'

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        specifying roi, can only be ME(R), LO(R), LOP(R))
    r_neighb : int, default=2
        number of neighbors in either hex1_id or hex2_id direction
    """
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)', 'ME(L)', 'LO(L)', 'LOP(L)'],\
        f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', 'ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"

    data_path = _get_data_path(reason='data')

    #load manually assigned hex ids
    column_df = load_hexed_body_ids(roi_str=roi_str)
    hex_df = column_df[['hex1_id','hex2_id']]\
        .drop_duplicates()\
        .sort_values(['hex1_id','hex2_id'])\
        .astype('float')\
        .reset_index(drop=True)
    n_col = hex_df.shape[0]

    #load created column centers
    if roi_str[-2]=='R':
        col_all_df = pd.read_pickle(data_path / f'{roi_str[:-3]}_col_center_pins.pickle')
    else:
        col_all_df = pd.read_pickle(data_path / f'{roi_str[:-3]}_L_col_center_pins.pickle')

    col_ids0 = col_all_df[pd.isna(col_all_df.iloc[:,3])==0].index.values
    n_pins = int((col_all_df.shape[1]-3)/3)

    ids_vh, ids_h, ids_v = find_holes(hex_df.values, hex_df.values[col_ids0])
    col_ids = np.concatenate([col_ids0,ids_v,ids_h,ids_vh])
    col_ids = np.sort(col_ids)
    n_col = col_ids.shape[0]

    #define local neighborhood coordinates in hex ids for standard columns and for holes
    x_v, y_v = np.meshgrid(range(-r_neighb,r_neighb+1),range(-r_neighb,r_neighb+1))
    xyv_std = np.concatenate((x_v.reshape((-1,1)), y_v.reshape((-1,1))),axis=1)
    xyv_vh = np.array([[-1,0],[0,0],[1,0],[0,-1],[0,1]])
    xyv_v = np.array([[0,-1],[0,1]])
    xyv_h = np.array([[-1,0],[1,0]])

    col_names = np.array( 100 * hex_df['hex1_id'].values[col_ids]\
        + hex_df['hex2_id'].values[col_ids] )
    col_neighb_names_std = col_names[:, np.newaxis]\
        + 100 * xyv_std[:, 0][np.newaxis, :]\
        + xyv_std[:, 1][np.newaxis, :]
    col_neighb_names_vh = col_names[:, np.newaxis]\
        + 100 * xyv_vh[:,0][np.newaxis, :]\
        + xyv_vh[:, 1][np.newaxis, :]
    col_neighb_names_v = col_names[:, np.newaxis]\
        + 100 * xyv_v[:,0][np.newaxis, :]\
        + xyv_v[:, 1][np.newaxis, :]
    col_neighb_names_h = col_names[:, np.newaxis]\
        + 100 * xyv_h[:,0][np.newaxis, :]\
        + xyv_h[:, 1][np.newaxis, :]

    #go through all columns and find the median over appropriate neighbors
    pins_depth = col_all_df.iloc[col_ids].iloc[:,3:].values.reshape((n_col,-1,3))
    median_local = np.empty((n_col,n_pins,3))
    median_local[:] = np.nan
    for i in range(n_col):
        #collect all indices of existing neighboring columns
        idx = []
        col_neighb_names = np.array([])
        if pd.isna(pins_depth[i,0,0])==0:
            col_neighb_names = col_neighb_names_std[i]
        else:
            if np.isin(col_ids[i],ids_v):
                col_neighb_names = col_neighb_names_v[i]
            elif np.isin(col_ids[i],ids_h):
                col_neighb_names = col_neighb_names_h[i]
            elif np.isin(col_ids[i],ids_vh):
                col_neighb_names = col_neighb_names_vh[i]
        for j in range(col_neighb_names.shape[0]):
            id1 = np.where( col_neighb_names[j]==col_names )[0]
            if id1.shape[0]>0:
                if pd.isna(pins_depth[id1[0],0,0])==0:
                    idx.append( id1[0] )
        if len(idx)>0:
            # for standard columns, compute median relative to bottom
            #     then shift by bottom of standard column
            if pd.isna(pins_depth[i,0,0])==0:
                median_local[i] = np.median(
                        pins_depth[idx] - pins_depth[idx,-1][:,np.newaxis,:]
                      , axis=0)\
                  + pins_depth[i,-1][np.newaxis,np.newaxis,:]
            # for missing column, compute median across neighbors
            else:
                median_local[i] = np.median(pins_depth[idx], axis=0)

    #Define dataframes to store pins
    col_df3 = pd.DataFrame(
        np.concatenate(
            (col_ids[:, np.newaxis], median_local.reshape((n_col,-1)))
          , axis=1)
    )
    col_df3 = col_df3.rename(columns={0: 'col_id'})
    col_df3['col_id'] = col_df3['col_id'].astype(int)
    col_df3 = hex_df.reset_index(names='col_id').merge(col_df3, 'left', on='col_id')

    #ordering of columns should be hex1_id, hex2_id, n_syn
    col_list = list(col_df3)
    col_list[0], col_list[1], col_list[2] = col_list[1], col_list[2], col_list[0]
    col_df3 = col_df3.loc[:,col_list]
    col_df3.iloc[:,2] = col_all_df.iloc[:,2].astype('Int64')
    col_df3.columns = col_all_df.columns
    col_df3['hex1_id'] = col_df3['hex1_id'].astype('Int64')
    col_df3['hex2_id'] = col_df3['hex2_id'].astype('Int64')

    if roi_str[-2]=='R':
        col_df3.to_pickle(data_path / f"{roi_str[:-3]}_col_center_pins.pickle")
    else:
        col_df3.to_pickle(data_path / f"{roi_str[:-3]}_L_col_center_pins.pickle")


def create_center_column_pins(
    anchor_method,
    n_anchor_bottom,
    n_anchor_top,
    roi_str='ME(R)',
    verbose=False,
):
    """
    A pickle file is created (named `roi_str[:-3]+_col_center_pins.pickle` or
    `ME_col_center_pins_old.pickle`) of a Dataframe with columns `hex1_id`, `hex2_id` (not
    duplicated), `n_syn` and then the flattened xyz positions of the corresponding pin.

    Parameters
    ----------
    anchor_method : str
       can only be 'combined' or 'separate'
           'combined' uses PC 1 from all points specified by pc_from
           'separate' uses PC 1 separately for points (specified by pc_from) at the bottom and the
           top
    n_anchor_bottom : int
        if 0 then the bottom anchor is the intersection of PC 1 (as defined per anchor_method)
        with the neuropil ROI
        if >0 & anchor_method='combined', n_anchor_bottom specifies the number of bottom synapses,
        the median of which we use to place the bottom anchor point (along PC 1)
                anchor_method='separate', n_anchor_bottom specifies the number of bottom synapses
                that are used to define a separate PC 1
    n_anchor_top : int
        analogue of n_anchor_bottom but for top instead of bottom
    roi_str : str
        specifying roi, can only be ME(R), LO(R), LOP(R))
    verbose : bool
        print pin creation information

    Returns
    -------
    None
    """
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)', 'ME(L)', 'LO(L)', 'LOP(L)'],\
        f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', 'ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"
    assert anchor_method in ['combined', 'separate'],\
        f"pc_from must be one of 'combined', 'separate', but is actually '{anchor_method}'"
    if anchor_method=='separate':
        assert (n_anchor_bottom>0)&(n_anchor_top>0),\
            f"n_anchor_bottom and n_anchor_top should both be bigger than 0, but are actually '{n_anchor_bottom}' and '{n_anchor_top}'"

    data_path = _get_data_path(reason='data')

    n_neighbors, n_segments, n_pins\
        , thre_std, valid_cols, pc_coord, pc_sign, pc_from\
        = load_roi_pin_params(roi_str=roi_str)

    #find all manually assigned hex ids
    column_df = load_hexed_body_ids(roi_str=roi_str)
    all_hex_ids = (
        column_df[['hex1_id','hex2_id']]\
            .drop_duplicates()\
            .sort_values(['hex1_id','hex2_id']))\
        .values
    n_col = all_hex_ids.shape[0]

    #Initialize array to store number of synapses per column
    n_syn = np.zeros((n_col,1))
    #Initialize array to store xyz positions of depth bins, using spline interpolation
    xyz_pins = np.empty((n_col,3*n_pins))
    xyz_pins[:] = np.nan

    ctr_id = 0
    ctr_syn = 0
    ctr_top = 0
    ctr_bottom = 0
    ctr_straight = 0
    #loop through columns
    for idx in range(n_col):
        # find all body_ids of valid_cols in column idx
        body_ids=[]
        column_hex_df = column_df[
            (column_df['hex1_id']==all_hex_ids[idx,0])\
          & (column_df['hex2_id']==all_hex_ids[idx,1])
        ]
        for j in range(column_hex_df.shape[0]):
            for i in valid_cols:
                if pd.isna(column_hex_df.iloc[j].loc[i])==0:
                    body_ids.append(column_hex_df.iloc[j].loc[i])
        body_ids = list(np.unique(body_ids).astype(int))
        if len(body_ids)==0:
            if verbose:
                print(f"Pin {idx} not created: not enough assigned neurons")
            ctr_id += 1
            continue

        syn_df = fetch_synapses(NC(bodyId=body_ids), SC(rois=roi_str))

        syn_trim_df = trim_syn_by_pc(
            syn_df
          , pc_coord
          , pc_sign
          , pc_from
          , thre_std=thre_std
        )
        if syn_trim_df.shape[0]<n_neighbors:
            if verbose:
                print(f"Pin {idx} not created: not enough trimmed synapses")
            ctr_syn += 1
            continue

        n_syn[idx] = syn_trim_df.shape[0]

        #find anchor points of pins (first and last points), and check if not too far from neighboring synapses
        anchor_bottom, anchor_top = find_anchors(
            syn_trim_df
          , pc_coord
          , pc_sign
          , pc_from
          , anchor_method
          , n_anchor_bottom
          , n_anchor_top
          , roi_str=roi_str
        )
        proj_rad = np.linalg.norm( anchor_top-anchor_bottom )/n_segments
        ordered_pts = syn_trim_df[['x','y','z']].values
        n_bdry = int(ordered_pts.shape[0]*0.05)
        if np.linalg.norm(anchor_top - ordered_pts[-n_bdry:].mean(0)) > proj_rad:
            if verbose:
                print(f"Pin {idx} not created: top anchor point too far from top synapses")
            ctr_top += 1
            continue
        elif np.linalg.norm(anchor_bottom - ordered_pts[:n_bdry].mean(0)) > proj_rad:
            if verbose:
                print(f"Pin {idx} not created: bottom anchor point too far from bottom synapses")
            ctr_bottom += 1
            continue

        #attach anchor points to ordered points, smoothen, and reattach anchor points
        ordered_pts = np.concatenate((anchor_bottom[np.newaxis,:], ordered_pts, anchor_top[np.newaxis,:]),axis=0)
        pts_smooth = find_neighbor_avg(ordered_pts, n_neighbors=n_neighbors)
        pts_smooth = np.concatenate((anchor_bottom[np.newaxis,:], pts_smooth, anchor_top[np.newaxis,:]),axis=0)

        #piecewise linear approximation of pts_smooth at length scale proj_rad (b_straight=1 if straight line; b_straight=0 otherwise)
        shortest_path_line, b_straight = find_shortest_path(pts_smooth, proj_rad)
        ctr_straight += b_straight

        pchip_uniform = find_uniform_interpolation(shortest_path_line, n_pins, mode='PCHIP')

        xyz_pins[idx] = np.squeeze(pchip_uniform.reshape((-1,1)))

    ctr_all = (~np.isnan(xyz_pins[:,0])).sum()
    if verbose:
        print(f"Created {ctr_all} pins in {roi_str}, of which {ctr_straight} are straight.")
        print(f"Missing pins because not enough neurons ({ctr_id}), not enough trimmed synapses ({ctr_syn}), or because the top ({ctr_top}) or bottom ({ctr_bottom}) anchor points are too far from neighboring synapses.")

    col_df = pd.DataFrame( np.concatenate((all_hex_ids, n_syn, xyz_pins), axis=1) )
    col_df = col_df.rename(columns={0: 'hex1_id', 1: 'hex2_id', 2: 'n_syn'})

    if roi_str[-2]=='R':
        if roi_str=='ME(R)':
            if anchor_method=='separate':
                col_df.to_pickle(data_path / f'ME_col_center_pins.pickle')
            else:
                col_df.to_pickle(data_path / f'ME_col_center_pins_old.pickle')
        else:
            col_df.to_pickle(data_path / f'{roi_str[:-3]}_col_center_pins.pickle')
    else:
        if roi_str=='ME(L)':
            if anchor_method=='separate':
                col_df.to_pickle(data_path / f'ME_L_col_center_pins.pickle')
            else:
                col_df.to_pickle(data_path / f'ME_L_col_center_pins_old.pickle')
        else:
            col_df.to_pickle(data_path / f'{roi_str[:-3]}_L_col_center_pins.pickle')


def load_roi_pin_params(roi_str='ME(R)'):
    """
    Load parameters for pin creation.

    All parameters were chosen heuristically.

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        specifying ROI, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    n_neighbors : int
        Number of neighbors to average over
    n_segments : int
        Number of segments for shortest path; should be bigger than 1
    n_pins : int
        Number of depth bins
    thre_std : float
        factor of lateral standard deviation to trim points (if too far)
    valid_cols : list[str]
        cell-types to include in column creation
    pc_coord : int
        specifies which coordinate, i.e., x=0, y=1, z=2, points along depth
    pc_sign : int
        specifies if a vector pointing along pc_coord goes from top to bottom (+1) or bottom to top (-1)
    pc_from : str
        Specifies if PCA is taken from synapses (pc_from='syn') or from the mean synapse position, for each bodyId (pc_from='COM')
    """
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"

    data_path = _get_data_path(reason='params')
    params_fn = data_path / "pin_creation_parameters.xlsx"
    params_df = pd.read_excel(params_fn).convert_dtypes()
    params_df = params_df[ params_df['neuropil']==roi_str[:-3] ]

    n_neighbors = int( params_df[ params_df['parameter']=='n_neighbors' ]['value'].values[0] )
    n_segments = int( params_df[ params_df['parameter']=='n_segments' ]['value'].values[0] )
    n_pins = int( params_df[ params_df['parameter']=='n_pins' ]['value'].values[0] )
    thre_std = float( params_df[ params_df['parameter']=='thre_std' ]['value'].values[0] )
    valid_cols = params_df[ params_df['parameter']=='valid_cols' ]['value'].values[0].split(", ")
    pc_coord = int( params_df[ params_df['parameter']=='pc_coord' ]['value'].values[0] )
    pc_sign = int( params_df[ params_df['parameter']=='pc_sign' ]['value'].values[0] )
    pc_from = str( params_df[ params_df['parameter']=='pc_from' ]['value'].values[0] )

    return n_neighbors\
        , n_segments\
        , n_pins\
        , thre_std\
        , valid_cols\
        , pc_coord\
        , pc_sign\
        , pc_from


def trim_syn_by_pc(
    syn_df,
    pc_coord,
    pc_sign,
    pc_from='syn',
    thre_std=1.0,
):
    """
    Trim synapses to lie close to their PC1

    Parameters
    ----------
    syn_df : pd.DataFrame
        with 'x', 'y', 'z' columns
    pc_coord : int
        specifies which coordinate, i.e., x=0, y=1, z=2, points along depth
    pc_sign : int
        specifies if a vector pointing along pc_coord goes from bottom to top (+1) or top to bottom (-1)
    pc_from : str
        Specifies if PCA is taken from synapses (pc_from='syn') or from the mean synapse position, for each bodyId (pc_from='COM')
    thre_std : float, default=1.0
        max threshold of lateral std

    Returns
    -------
    syn_trim_df : pd.DataFrame
        with 'x', 'y', 'z' columns, after trimming, sorted bottom to top
    """
    assert pc_from in ['syn', 'COM'],\
        f"pc_from must be one of 'syn', 'COM', but is actually '{pc_from}'"

    #points to take PCA from
    if pc_from=='COM':
        pc_pts = syn_df.groupby('bodyId')[['x','y','z']].mean().values
    else:
        pc_pts = syn_df[['x','y','z']].values
    unitary, _, _ = np.linalg.svd(pc_pts.T-pc_pts.T.mean(1)[:,np.newaxis])

    #fix sign of PC1 along one direction (differs between neuropils): positive should be bottom to top
    if np.sign(unitary[pc_coord,0])==pc_sign:
        unitary[:,0] = -unitary[:,0]

    #compute distance in PC2-PC3 plane
    lateral = np.sqrt(
        ((syn_df[['x','y','z']].values-pc_pts.mean(0)[np.newaxis,:])@unitary[:,1])**2
      + ((syn_df[['x','y','z']].values-pc_pts.mean(0)[np.newaxis,:])@unitary[:,2])**2
    )
    #only take points within thre_std std in PC 2-3 plane
    lateral_max = thre_std*np.sqrt((lateral**2).mean())
    syn_trim_df = syn_df[lateral<lateral_max]

    #project onto PC1, and then sort to get bottom to top order
    proj = (syn_trim_df[['x','y','z']].values-pc_pts.mean(0)[np.newaxis,:])@unitary[:,0]
    isort = np.argsort(proj)
    syn_trim_df = syn_trim_df.iloc[isort]

    #redo PCA on trimmed points and resort
    if pc_from=='COM':
        pc_pts = syn_trim_df.groupby('bodyId')[['x','y','z']].mean().values
    else:
        pc_pts = syn_trim_df[['x','y','z']].values
    unitary, _, _ = np.linalg.svd(pc_pts.T-pc_pts.T.mean(1)[:,np.newaxis])
    if np.sign(unitary[pc_coord,0])==pc_sign:
        unitary[:,0] = -unitary[:,0]
    proj = (syn_trim_df[['x','y','z']].values-pc_pts.mean(0)[np.newaxis,:])@unitary[:,0]
    isort = np.argsort(proj)

    return syn_trim_df.iloc[isort]


def find_anchors(
    syn_df,
    pc_coord,
    pc_sign,
    pc_from,
    anchor_method,
    n_anchor_bottom,
    n_anchor_top,
    roi_str='ME(R)',
):
    """
    Find anchor points, i.e., first and last pin points

    Parameters
    ----------
    syn_df : pd.DataFrame
        with 'x', 'y', 'z' columns
    pc_coord : int
        specifies which coordinate, i.e., x=0, y=1, z=2, points along depth
    pc_sign : int
        specifies if a vector pointing along pc_coord goes from bottom to top (+1) or top to bottom (-1)
    pc_from : str
        specifies is PCA is taken from synapses (pc_from='syn') or from the mean synapse position, for each bodyId (pc_from='COM')
    anchor_method : str
       can only be 'combined' or 'separate'
           'combined' uses PC 1 from all points specified by pc_from
           'separate' uses PC 1 separately for points (specified by pc_from) at the bottom and the top
    n_anchor_bottom : int
        if 0 then the bottom anchor is the intersection of PC 1 (as defined per anchor_method) with the neuropil ROI
        if >0 & anchor_method='combined', n_anchor_bottom specifies the number of bottom synapses, the median of which we use
                    to place the bottom anchor point (along PC 1)
                anchor_method='separate', n_anchor_bottom specifies the number of bottom synapses that are used to define a separate PC 1
    n_anchor_top : int
        analogue of n_anchor_bottom but for top instead of bottom
    roi_str : str, default='ME(R)'
        specifying ROI, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    pc1_bottom : np.ndarray (float) of length 3
        3D point where PC1 intersects neuprint ROI at the bottom
    pc1_top : np.ndarray (float) of length 3
        3D point where PC1 intersects neuprint ROI at the top
    """
    assert pc_from in ['syn', 'COM'],\
        f"pc_from must be one of 'syn', 'COM', but is actually '{pc_from}'"
    assert anchor_method in ['combined', 'separate'],\
        f"pc_from must be one of 'combined', 'separate', but is actually '{anchor_method}'"
    if anchor_method=='separate':
        assert (n_anchor_bottom>0)&(n_anchor_top>0),\
            f"n_anchor_bottom and n_anchor_top should both be bigger than 0, but are actually '{n_anchor_bottom}' and '{n_anchor_top}'"
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"

    #PCA on points specified by pc_from
    if pc_from=='COM':
        pc_pts = syn_df.groupby('bodyId')[['x','y','z']].mean().values
    else:
        pc_pts = syn_df[['x','y','z']].values
    unitary, _, _ = np.linalg.svd(pc_pts.T-pc_pts.T.mean(1)[:,np.newaxis])
    if np.sign(unitary[pc_coord,0])==pc_sign:
        unitary[:,0] = -unitary[:,0]

    #sort to get bottom to top order
    proj = (syn_df[['x','y','z']].values-pc_pts.mean(0)[np.newaxis,:])@unitary[:,0]
    isort = np.argsort(proj)
    ordered_pts = syn_df[['x','y','z']].values[isort]

    #load neuropil ROI, used to determine anchor points
    roi_vol = neu.fetch_roi(roi_str)

    if anchor_method=='combined':
        #find intersection of PC1 with neuprint roi
        pc1_line = 10 * (proj.max()-proj.min())\
            * np.linspace(-1,1,1000)[:,np.newaxis]\
            * unitary[:,0]\
            + pc_pts.T.mean(1)
        signed_dists = trimesh.proximity.signed_distance(roi_vol,pc1_line)
        pc1_line = pc1_line[signed_dists>0]
        anchor_bottom = pc1_line[0]
        anchor_top = pc1_line[-1]

        #override pc1_bottom with median bottom synapses projected onto PC 1
        if n_anchor_bottom>0:
            proj_bottom = (np.median(ordered_pts[:n_anchor_bottom],axis=0)-pc_pts.mean(0))@unitary[:,0]
            anchor_bottom = proj_bottom*unitary[:,0]+pc_pts.mean(0).T

        #override pc1_top with median top synapses projected onto PC 1
        if n_anchor_top>0:
            proj_top = (np.median(ordered_pts[-n_anchor_top:],axis=0)-pc_pts.mean(0))@unitary[:,0]
            anchor_top = proj_top*unitary[:,0]+pc_pts.mean(0).T

    if anchor_method=='separate':
        #normalize proj to -1 and 1
        proj = (proj - (proj.max()+proj.min())/2)/(proj.max()-proj.min())*2

        #points to use for bottom PCA
         #the upper threshold 0.1 was chosen manually
         #it is >0 because in ME and LO there are gaps in the synapses for the more bottom layers
        pc_pts_bottom = pc_pts[proj<.1]
        if pc_pts_bottom.shape[0]<n_anchor_bottom:
            pc_pts_bottom = pc_pts[:n_anchor_bottom]

        #points to use for top PCA
         #the lower threshold 0.2 was chosen manually
         #it was chosen to be bigger than the upper threshold used for bottom
        pc_pts_top = pc_pts[proj>.2]
        if pc_pts_top.shape[0]<n_anchor_top:
            pc_pts_top = pc_pts[-n_anchor_top:]

        #compute bottom PCA
        unitary_bottom, _, _ = np.linalg.svd(pc_pts_bottom.T-pc_pts_bottom.T.mean(1)[:,np.newaxis])
        if np.sign(unitary_bottom[pc_coord,0])==pc_sign:
            unitary_bottom[:,0] = -unitary_bottom[:,0]
        proj_bottom = (ordered_pts-pc_pts_bottom.mean(0)[np.newaxis,:])@unitary_bottom[:,0]

        #compute top PCA
        unitary_top, _, _ = np.linalg.svd(pc_pts_top.T-pc_pts_top.T.mean(1)[:,np.newaxis])
        if np.sign(unitary_top[pc_coord,0])==pc_sign:
            unitary_top[:,0] = -unitary_top[:,0]
        proj_top = (ordered_pts-pc_pts_top.mean(0)[np.newaxis,:])@unitary_top[:,0]

        #find intersection of bottom PC1 with neuprint roi
        pc1_bottom_line = 10*(-proj_bottom.min())*np.linspace(-1,0,1000)[:,np.newaxis]\
            * unitary_bottom[:,0] + pc_pts_bottom.T.mean(1)
        signed_dists_bottom = trimesh.proximity.signed_distance(roi_vol,pc1_bottom_line)
        anchor_bottom = pc1_bottom_line[signed_dists_bottom>0][0]

        #find intersection of top PC1 with neuprint roi
        pc1_top_line = 10*proj_top.max()*np.linspace(0,1,1000)[:,np.newaxis]\
            * unitary_top[:,0] + pc_pts_top.T.mean(1)
        signed_dists_top = trimesh.proximity.signed_distance(roi_vol,pc1_top_line)
        anchor_top = pc1_top_line[signed_dists_top>0][-1]

    return anchor_bottom, anchor_top


def find_neighbor_avg(
    pts_sorted,
    n_neighbors=100,
):
    """
    average over n_neighbors (neighbors are defined by the ordering of pts_sorted)

    Parameter
    ---------
    pts_sorted : np.ndarray (float)
        array with `N` number of 3D points ordered by their projection onto PC1
    n_neighbors : int, default=100
        number of neighbors to smooth over

    Returns
    -------
    pts_smooth : np.ndarray (float)
        array with `N` number of 3d points: `pts_sorted` smoothened over <= `n_neighbors`
    """

    pts_smooth = np.zeros((pts_sorted.shape[0]-n_neighbors+1,3))
    for j in range(3):
        pts_smooth[:,j] = np.convolve(
            pts_sorted[:,j]
          , np.ones(n_neighbors)/n_neighbors
          , mode='valid')
    return pts_smooth


def find_shortest_path(pts_smooth, proj_rad):
    """
    compute shortest path from pts_smooth[0] to pts_smooth[-1]

    Parameters
    ----------
    pts_smooth : list
        (array of size Nx3)
    proj_rad : float
        max distance to "hop"

    Returns
    -------
    shortest_path_line : list
        Kx3 list of K points that are the edge points for the shortest path
    b_straight : bool
        0 if column created is bent; 1 if it is straight
    """

    b_straight = True
    if proj_rad>=np.linalg.norm(pts_smooth[0]-pts_smooth[-1]):
        shortest_path_line = np.zeros((2,3))
        shortest_path_line[0] = pts_smooth[0]
        shortest_path_line[1] = pts_smooth[-1]
    else:
        try:
            b_straight = False
            distances = euclidean_distances(pts_smooth)
            distances[distances>proj_rad]=0
            graph = nx.from_numpy_array(distances, create_using=nx.DiGraph)
            path = nx.shortest_path(
                graph
              , source=0
              , target=distances.shape[0]-1
              , weight='weight'
            )
            shortest_path_line = pts_smooth[path]
        except nx.NetworkXNoPath:
            b_straight = True
            shortest_path_line = np.zeros((2,3))
            shortest_path_line[0] = pts_smooth[0]
            shortest_path_line[1] = pts_smooth[-1]

    return shortest_path_line, b_straight


def find_shortest_path_dist(shortest_path_line):

    """
    compute distances of neighboring line segments in path

    Parameters
    ----------
    shortest_path_line : np.ndarray (float)
        array with `K` number of 3D points that lie on a curve.

    Returns
    -------
    shortest_dists : np.ndarray (float)
        distance between neighboring points in `shortest_path_line` (first entry is 0)
    """

    shortest_dists = np.zeros(shortest_path_line.shape[0])
    for i in range(shortest_dists.shape[0]-1):
        shortest_dists[i+1] = np.linalg.norm(shortest_path_line[i+1]-shortest_path_line[i])

    return shortest_dists


def find_uniform_interpolation(
    shortest_path_line,
    n_pts,
    mode='PCHIP',
):
    """
    interpolate shortest_path_line by n_pts uniform points (uniform in the sense of path length)

    Parameters
    ----------
    shortest_path_line : np.ndarray (float)
        array with `K` number of 3D points that lie on a curve
    n_pts : float
        number of interpolation points
    mode : str
        can only be 'cubic', 'linear', or 'PCHIP'

    Returns
    -------
    interp_uniform : np.ndarray[list[float]]
        interpolation of shortest_path_line, array with `n_pts` 3D points
    """

    assert mode in ['cubic', 'linear', 'PCHIP'],\
            f"Mode must be one of 'cubic', 'linear', 'PCHIP', but is actually '{mode}'"

    #find indivudal path lengths and their cumulative sum
    shortest_dists = find_shortest_path_dist(shortest_path_line)
    shortest_dists_cumsum = np.cumsum(shortest_dists)
    #uniformly sample path length using n_pts number of points
    path_uniform = np.linspace(0,shortest_dists_cumsum[-1],n_pts)
    #fit interpolation for each coordinate
    if mode=='cubic':
        f_x = interpolate.interp1d(shortest_dists_cumsum, shortest_path_line[:,0], kind='cubic')
        f_y = interpolate.interp1d(shortest_dists_cumsum, shortest_path_line[:,1], kind='cubic')
        f_z = interpolate.interp1d(shortest_dists_cumsum, shortest_path_line[:,2], kind='cubic')
    elif mode=='linear':
        f_x = interpolate.interp1d(shortest_dists_cumsum, shortest_path_line[:,0], kind='linear')
        f_y = interpolate.interp1d(shortest_dists_cumsum, shortest_path_line[:,1], kind='linear')
        f_z = interpolate.interp1d(shortest_dists_cumsum, shortest_path_line[:,2], kind='linear')
    elif mode=='PCHIP':
        f_x = interpolate.PchipInterpolator(shortest_dists_cumsum, shortest_path_line[:,0])
        f_y = interpolate.PchipInterpolator(shortest_dists_cumsum, shortest_path_line[:,1])
        f_z = interpolate.PchipInterpolator(shortest_dists_cumsum, shortest_path_line[:,2])
    #do interpolation for each coordinate
    x_pred = f_x(path_uniform)
    y_pred = f_y(path_uniform)
    z_pred = f_z(path_uniform)
    interp_uniform = np.vstack([x_pred, y_pred, z_pred]).T

    return interp_uniform


# =============================================================================
# From ROI_layers.py
# =============================================================================

def create_edge_ids(
    roi_str = 'ME(R)',
):
    """
    Find edge coordinates of pins and store in, e.g. ME_hex_ids_edge.csv

    Parameters
    ----------
    syn_df : pd.DataFrame
        dataframe with 'bodyId', 'x', 'y', 'z' columns
    roi_str : str
        neuprint ROI, can only be ME(R), LO(R), LOP(R)
    samp : int
        sub-sampling factor for depth bins

    Returns
    -------
    syn_col_df : pd.DataFrame
        'hex1_id' : int
            defines column
        'hex2_id' : int
            defines column
        'col_count' : int
            number of synapses in that column (across all depth bins)
    """

    data_path = _get_data_path(reason='data')
    cache_path = _get_data_path(reason='data')

    if roi_str[-2]=='R':
        pincushion_f = os.path.join(data_path,roi_str[:-3]+'_col_center_pins.pickle')
    else:
        pincushion_f = os.path.join(data_path,roi_str[:-3]+'_L_col_center_pins.pickle')
    with open(pincushion_f, mode='rb') as pc_fh:
        pin_cushion_df = pd.read_pickle(pc_fh)
    pin_cushion_df = pin_cushion_df.dropna()

    hex_edge_df = fl_get_edge_ids(pin_cushion_df)

    if roi_str[-2]=='R':
        hex_edge_f = cache_path / f"{roi_str[:-3]}_hex_ids_edge.csv"
    else:
        hex_edge_f = cache_path / f"{roi_str[:-3]}_L_hex_ids_edge.csv"
    hex_edge_df.to_csv(hex_edge_f, index=True, header=False)


def load_pins_for_mesh(roi_str):
    """
    Load data for layer mesh creation

    Parameters
    ----------
    roi_str : str
        name of the neuropil, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    pins : np.ndarray
        xyz positions of pin nodes (excl pins on the edge without 2 neighbors)
    n_bins : int
        number of depth bins (same for all pins)
    hex1_valid : np.ndarray
        hex1_ids of included pins
    hex2_valid : np.ndarray
        hex2_ids of included pins
    """
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"

    data_path = _get_data_path(reason='cache')
    col_ids, n_bins, pins = load_pins(roi_str=roi_str)
    pins = pins.astype(float)

    #JH change
    # if (roi_str[:-3]=='LO')|(roi_str=='LOP'):
    #     #manual exclusion of boundary pins that don't have 2 neighbors
    #     if roi_str[:-3]=='LOP':
    #         col_exclude = [0,2,36,37,156,853,883,888,889]
    #     elif roi_str[:-3]=='LO':
    #         col_exclude = [0]
    #     id_exclude = np.where( np.isin( col_ids, col_exclude) )[0]
    #     id_include = np.array([i for i in range(col_ids.shape[0]) if i not in id_exclude])
    #     col_ids = col_ids[id_include]
    #     pins_sorted = pins.reshape((-1,n_bins,3))
    #     pins_sorted = pins_sorted[id_include]
    #     pins = pins_sorted.reshape((-1,3))

    hex_df = get_hex_df(neuropil=f'ME{roi_str[-3:]}')
    hex1_valid = hex_df['hex1_id'].astype('int').values
    hex2_valid = hex_df['hex2_id'].astype('int').values
    #the (1,10) column is effectively a (1,9) column (which doesn't exist)
    hex2_valid[2] = 9
    hex1_valid = hex1_valid[col_ids]
    hex2_valid = hex2_valid[col_ids]

    #find which pins are on the edge
    if roi_str[-2]=='R':
        col_ids_edge_f = data_path / f"{roi_str[:-3]}_hex_ids_edge.csv"
    else:
        col_ids_edge_f = data_path / f"{roi_str[:-3]}_L_hex_ids_edge.csv"
    if not col_ids_edge_f.is_file():
        create_edge_ids(roi_str=roi_str)
    col_ids_edge_df = pd.read_csv(col_ids_edge_f, header=None)
    col_ids_edge = col_ids_edge_df.values[:,0]
    # if (roi_str[:-3]=='LO')|(roi_str[:-3]=='LOP'):
    #     id_exclude = np.where( np.isin( col_ids_edge, col_exclude) )[0]
    #     id_include = np.array([i for i in range(col_ids_edge.shape[0]) if i not in id_exclude])
    #     col_ids_edge = col_ids_edge[id_include]
    idx_edge = np.where(np.isin(col_ids, col_ids_edge))[0]

    return pins, n_bins, hex1_valid, hex2_valid, idx_edge


def load_roi_layer_mesh_params(roi_str='ME(R)'):
    """
    Load parameters for layer mesh creation.

    All parameters were chosen heuristically.

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    alpha : float
        used in alphashape; smaller numbers correspond to more smoothing
    fac_ext : float
        extension parameter of edge pin points towards the boundary
    """
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"

    data_path = _get_data_path(reason='params')
    params_fn = data_path / "layer_mesh_creation_parameters.xlsx"
    params_df = pd.read_excel(params_fn).convert_dtypes()
    params_df = params_df[ params_df['neuropil']==roi_str[:-3] ]

    #parameter alpha is used in alphashape; smaller numbers correspond to more smoothing
    #parameter fac_ext determines how much edge column points are extended towards the boundary,
    alpha = float( params_df[ params_df['parameter']=='alpha' ]['value'].values[0] )
    fac_ext = float( params_df[ params_df['parameter']=='fac_ext' ]['value'].values[0] )

    return alpha, fac_ext


def find_layer_bdrys(
    roi_str='ME(R)',
):
    """
    Find the depth thresholds between layers (very top and very bottom are set manually).

    It generates files in the data path named `ME_layer_bdry.csv`, `LO_layer_bdry.csv`,
        or `LOP_layer_bdry.csv`.

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        neuprint neuropil name, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    None
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"

    data_path = _get_data_path(reason='data')

    n_layers, frac_peaks, canonical_list, syn_type, peak_ids, layer_bdry_pos =\
        load_roi_layer_params(roi_str=roi_str)
    bin_edges, bin_centers = load_depth_bins(roi_str=roi_str, samp=1)

    layers_bdry = np.zeros(n_layers+1)
    layers_bdry[0] = -0.01
    layers_bdry[-1] = 1.01
    for idx, target_type in enumerate(canonical_list):
        syn_df = fetch_synapses(
            NC(type=target_type)
          , SC(rois=roi_str, confidence=.9))

        syn_df = syn_df[syn_df['type']==syn_type[idx]]
        depth_df = find_depth(syn_df, roi_str=roi_str, samp=1)

        count, _ = np.histogram(
            depth_df['depth'].values
          , bins=bin_edges
          , density=True)
        thre_peaks = find_pdf_treshold(count, frac_peaks)
        depth_bottom, depth_top = find_peak_thresholds(
            count
          , bin_centers
          , thre_peaks
          , min_bin_change=5)

        depth_bdrys = np.vstack([depth_bottom, depth_top]).T.reshape((-1,1))
        layers_bdry[layer_bdry_pos[idx]] = np.squeeze(depth_bdrys[peak_ids[idx]])

    #in the LOP, there are gaps between the T4 layer patterns
    #therefore, we define the layer boundaries as means of two neighboring estimates
    if roi_str[:-3]=='LOP':
        only_four_layers_bdry = np.zeros(5)
        only_four_layers_bdry[[0,-1]] = layers_bdry[[0,-1]]
        only_four_layers_bdry[1] = layers_bdry[1:3].mean()
        only_four_layers_bdry[2] = layers_bdry[3:5].mean()
        only_four_layers_bdry[3] = layers_bdry[5:7].mean()
        layers_bdry = only_four_layers_bdry

    if roi_str[-2]=='R':
        pd.DataFrame(layers_bdry).to_csv(
            data_path / f"{roi_str[:-3]}_layer_bdry.csv"
        , float_format='%.2f'
        , index=False
        , header=False)
    else:
        pd.DataFrame(layers_bdry).to_csv(
            data_path / f"{roi_str[:-3]}_L_layer_bdry.csv"
        , float_format='%.2f'
        , index=False
        , header=False)


def load_roi_layer_params(roi_str='ME(R)'):
    """
    Load parameters for layer creation.

    All parameters were chosen heuristically.

    Parameters
    ----------
    roi_str : str, default='ME(R)'
        neuprint ROI, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    n_layers : int
        number of layers
    frac_peaks : float
        fraction of synapses in all peaks
    canonical_list : list[str]
        cell-types to define layer boundaries
    syn_type : list[str]
        synapse type to define layer boundaries
    peak_ids : list[list[int]]
        position of peaks that are used to define layer boundaries.
        length of the outer list equals the length of `canonical_list`.
        length of the inner list equals the number of flanks of peaks that are used to define
            layer boundaries. The numbers mean: `0` is the left flank of the first peak, `1` the
            right flank of the first peak, `2` the left flank of the third peak, etc...
    layer_bdry_pos : list[list[int]]
        position of flanks in layer boundary list.
        length of the outer list equals the length of `canonical_list`.
        length of the inner list equals the number of flanks of peaks that are used to define
            layer boundaries. The numbers mean: `1` is the layer threshold between layers 1 and 2,
            `2` is the layer threshold between layers 2 and 3, etc...
    """
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"

    data_path = _get_data_path(reason='params')
    params_fn = data_path / "layer_creation_parameters.xlsx"
    params_df = pd.read_excel(params_fn).convert_dtypes()
    params_df = params_df[ params_df['neuropil']==roi_str[:-3] ]

    n_layers = int( params_df[ params_df['parameter']=='n_layers' ]['value'].values[0] )
    frac_peaks = float( params_df[ params_df['parameter']=='frac_peaks' ]['value'].values[0] )
    canonical_list = params_df[ params_df['parameter']=='canonical_list' ]['value'].values[0].split(", ")
    syn_type = params_df[ params_df['parameter']=='syn_type' ]['value'].values[0].split(", ")
    peak_ids = ( params_df[ params_df['parameter']=='peak_ids' ]['value'].apply(convert_str_to_list_of_lists_of_ints) ).values[0]
    layer_bdry_pos = ( params_df[ params_df['parameter']=='layer_bdry_pos' ]['value'].apply(convert_str_to_list_of_lists_of_ints) ).values[0]

    return n_layers, frac_peaks, canonical_list, syn_type, peak_ids, layer_bdry_pos


def convert_str_to_list_of_lists_of_ints(s):
    """
    Helper function.
    Converts a string that is a list of lists of integers to that object.

    Parameters
    ----------
    s : str
        string containing the list of lists

    Returns
    -------
    list_all : list[list[int]]
        list of lists of integers contained in s
    """
    s_split = s.split(',')
    list_all = []
    idx = 0
    while idx < len(s.split(',')):
        list = []
        while s_split[idx][-1]!=']':
            list.append( int(s_split[idx].strip('[]')) )
            idx += 1
        list.append( int(s_split[idx].strip('[]')) )
        list_all.append(list)
        idx += 1

    return list_all


def find_pdf_treshold(
    count,
    frac_peaks,
    thre_peaks_step=0.1,
):
    """
    Find threshold on synapse density to reach fraction `frac_peaks` of total,
        e.g. `frac_peaks=0.8` means that we are looking for the threshold on the synapse density
        to reach 80% of all synapses

    Parameters
    ----------
    count : np.ndarray
        synapse distribution
    frac_peaks : float
        fraction of synapses in all peaks
    thre_peaks_step : float
        step size in count to reach fraction `frac_peaks`

    Returns
    -------
    thre_peaks : float
        threshold on count to find peaks
    """
    thre_peaks = count.max()
    count_all = count.sum()
    count_peaks = count[ count>=thre_peaks ].sum()
    while count_peaks/count_all < frac_peaks:
        thre_peaks = thre_peaks-thre_peaks_step
        count_peaks = count[ count>=thre_peaks ].sum()

    return thre_peaks


def find_peak_thresholds(
    count,
    bin_centers,
    thre_peaks,
    min_bin_change=2,
):
    """
    Find the flanks of peaks in the synapse distribution

    Parameters
    ----------
    count : np.ndarray
        synapse distribution
    bin_centers : np.ndarray
        bin centers for depth
    thre_peaks : float
        threshold on synapse distribution to find peaks
    min_bin_change : int
        minimum number of bins that top and bottom of neighboring peaks have to be separated by

    Returns
    -------
    depth_bottom : np.ndarray
        sub-array of bin_centers with the bottom flanks of peaks
    depth_top : np.ndarray
        sub-array of bin_centers with the top flanks of peaks
    """

    #Find peaks
    ind_peak = np.where(count >= thre_peaks)[0]
    diff_ind = np.diff(ind_peak, 1)
    ind_const = np.where(diff_ind <= min_bin_change)[0]
    ind_change = np.where(diff_ind > min_bin_change)[0]
    #how many different peaks
    k = ind_change.shape[0] + 1
    #find inidices that define bottom and top of peak
    ind_bottom = np.zeros(k, dtype=int)
    ind_top = np.zeros(k, dtype=int)
    #find corresponding depths
    depth_bottom = np.zeros(k)
    depth_top = np.zeros(k)
    ind_bottom[0] = ind_peak[ind_const[0]]
    depth_bottom[0] = bin_centers[ind_bottom[0]]
    #loop through peaks
    for i in range(k-1):
        if (ind_peak[ind_change] > ind_bottom[i]).sum() > 0:
            ind_top[i] = ind_peak[ind_change][np.where(ind_peak[ind_change]>ind_bottom[i])[0][0]]
        else:
            ind_top[i] = bin_centers.shape[0]-1
        depth_top[i] = bin_centers[ind_top[i]]
        if (ind_peak[ind_const]>ind_top[i]).sum()>0:
            ind_bottom[i+1] = ind_peak[ind_const][np.where(ind_peak[ind_const]>ind_top[i])[0][0]]
        else:
            ind_bottom[i+1] = 0
        depth_bottom[i+1] = bin_centers[ind_bottom[i+1]]
    ind_top[k-1] = ind_peak[-1]
    depth_top[k-1] = bin_centers[ind_top[k-1]]

    return depth_bottom, depth_top


# =============================================================================
# From graph_utils.py
# =============================================================================

def fetch_shortest_paths_from_edgelist(source, target, edgelist, min_weight=10, max_hops=3):
    """
    Retrieve shortest paths from an edgelist using BFS. Used for FAFB flywire data.

    Parameters:
    source (int): source neuron ID.
    target (int): target neuron ID.
    edgelist (pd.DataFrame): DataFrame containing edges with columns ['source', 'target', 'weight'].
    min_weight (int): Minimum weight for the paths. Default is 10.
    max_hops (int): Maximum number of hops to search. Default is 3.

    Returns:
    pd.DataFrame: DataFrame containing the shortest path.
    """

    # filter edgelist by min_weight
    edgelist = edgelist[edgelist['weight'] >= min_weight]

    # make an adj list from edgelist
    graph = defaultdict(list)
    for u, v, w in edgelist.to_numpy():
        graph[u].append(v)  # Directed edge from u to v

    # Queue for BFS: stores paths
    queue = deque([[source]])
    visited = set()  # Track visited nodes to avoid cycles
    all_paths = []  # To store all shortest paths
    curr_max_len = 1  # Track the maximum path length

    # Perform BFS to find all shortest paths
    found_level = None
    while queue and curr_max_len <= max_hops:
        path = queue.popleft()
        node = path[-1]

        # Stop searching deeper if we've reached the first shortest paths level
        if found_level is not None and len(path) > found_level:
            break

        # If the target is reached, record the path and mark the level
        if node == target:
            if found_level is None:
                found_level = len(path)  # Mark the depth level of shortest path
            all_paths.append(path)
            continue

        # Continue exploring neighbors
        if node not in visited or len(path) < curr_max_len:
            visited.add(node)
            for neighbor in graph[node]:
                new_path = path + [neighbor]
                queue.append(new_path)
                if len(new_path) > curr_max_len:
                    curr_max_len = len(new_path)

    # Convert paths to DataFrame
    # if path exists
    if len(all_paths) > 0:
        paths = [pd.DataFrame({'path': i, 'node': path}) for i, path in enumerate(all_paths)]
        return pd.concat(paths)
    else:
        print(f"No path less than {max_hops} hops found between {source} and {target}")
        return pd.DataFrame()

def collect_fixedlength_paths(source_id, target_id, edgelist=None, num_hops=None, max_num_hops=None, min_weight=10, timeout=5.0):
    """
    Collect all paths of given length (number of hops) from sources to targets and store path lengths.
    Expand it to work with edgelists (2024-11). If edgelist is None, use neuprint's fetch_paths from AZ

    Todo, not working with flywire/edgelist yet.

    Parameters:
    source_id (list): List of source neuron IDs.
    target_id (list): List of target neuron IDs.
    num_hops (int): Number of hops to search for. e.g., 2 hops contains 3 nodes and 2 edges.
    max_num_hops (int): Maximum number of hops to search for. Either num_hops or max_num_hops must be set, but not both.
    min_weight (int): Minimum weight for the paths. Default is 10.
    timeout (int): Timeout for fetching paths. Default is 5 seconds.
    edgelist (None or pd.DataFrame): Default is None, using neuprint's fetch_paths.

    Returns:
    pd.DataFrame: DataFrame containing source, target, and path length.
    list: List of all paths.
    """
    # Ensure that exactly one of path_length or max_path_length is specified
    if num_hops is None and max_num_hops is None:
        raise ValueError("Either num_hops or max_num_hops must be specified")
    if num_hops is not None and max_num_hops is not None:
        raise ValueError("Please specify either num_hops or max_num_hops, but not both.")

    # Determine which mode we're in
    exact_hops_mode = num_hops is not None

    # check num_hops is an integer greater than 0, otherwise exit with a message
    if exact_hops_mode:
        if not isinstance(num_hops, int) or num_hops < 1:
            raise ValueError("num_hops must be an integer greater than 0")
    else:
        if not isinstance(max_num_hops, int) or max_num_hops < 1:
            raise ValueError("max_num_hops must be an integer greater than 0")

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
        # path_len = np.zeros((0, 3))
        paths_all = []
        for i in source_id:
            for j in target_id:
                if exact_hops_mode:
                    paths = fetch_paths(i, j, path_length=num_hops, min_weight=min_weight, timeout=timeout)
                else:
                    paths = fetch_paths(i, j, max_path_length=max_num_hops, min_weight=min_weight, timeout=timeout)
                # if path is not empty or timeout
                if len(paths) > 0:
                    paths_all.append(paths)
                    # # if the dataframe paths_all doesn't exist, create it, else concatenate paths_all with paths
                    # if 'paths_all' not in locals():
                    #     paths_all = paths
                    # else:
                    #     paths_all = pd.concat([paths_all, paths], ignore_index=True)

    return paths_all


# =============================================================================
# From helper.py
# =============================================================================

def add_color_group(df, main_groups, colors):
# function to add the color column to the dataframe

    for index, row in df.iterrows():
        group = row['main_groups']
        if group in main_groups[0]:
            grp = 1
            col = colors[0]
        elif group in main_groups[1]:
            grp = 2
            col = colors[1]
        elif group in main_groups[2]:
            grp = 3
            col = colors[2]
        elif group in main_groups[3]:
            grp = 4
            col = colors[3]
        elif group in main_groups[4]:
            grp = 5
            col = colors[4]
        else:
            grp = 0
            col = '#808080' #colors[5]

        row['color'] = col
        df.loc[index, 'color']= col
        row['group'] = grp
        df.loc[index, 'group']= grp

    df['color'].astype(dtype='object')
    df['group'].astype(dtype='object')

    return df


# =============================================================================
# From input_distr_functions.py
# =============================================================================

def store_1d_input_distrs(instance_name, neuropil, limit, colors_all=None):

    #colors for plotting
    if colors_all==None:
        colors_all = [generate_random_color() for _ in range(limit)]
        colors_all.insert(0, '#808080')
    #upper threshold on number of hex coordinates in 1d
    hex0 = 45

    #get home columns
    syn_df = fetch_synapses(NC(instance=instance_name), SC(rois=[neuropil], type='post'))
    hex_df = find_neuron_hex_ids(syn_df, roi_str=neuropil, method='COM')
    hex_red_df = hex_df[['bodyId','hex1_id','hex2_id']]\
        .rename(columns={'bodyId': 'bodyId_post'})

    #control: use all post synapses
    conn_control_df = fetch_synapse_connections(None, \
                                        NC(instance=instance_name) ,\
                                        SC(rois=[neuropil]))
    syn_control_df = conn_control_df[['bodyId_pre','x_post','y_post','z_post','bodyId_post']]\
        .rename(columns={'x_post': 'x', 'y_post': 'y', 'z_post': 'z'})
    hex_control_df = find_rel_hex(hex_red_df, syn_control_df, neuropil)

    #count fraction of synapses in columns
    hex_control_df = hex_control_df.groupby(['bodyId_post','hex1_id','hex2_id'])['bodyId_pre'].count().to_frame(name='count').reset_index()
    hex_control_df['frac'] = hex_control_df.groupby('bodyId_post')['count'].apply(lambda x: x/x.sum()).values
    hex_control_df['hex_names'] = 100*(hex_control_df['hex1_id']+hex0) + hex_control_df['hex2_id']+hex0
    mean_control_df = hex_control_df.groupby(['bodyId_post','hex_names'])['frac'].first().unstack(-1, 0).mean(0).to_frame(name='frac').reset_index()
    mean_control_df['hex1_id'] = (mean_control_df['hex_names'].values/100).astype(int)-hex0
    mean_control_df['hex2_id'] = mean_control_df['hex_names']-100*(mean_control_df['hex1_id']+hex0)-hex0

    rad_dist_control_df, angle_dist_control_df = find_1d_input_distrs(mean_control_df) #hex_control_df)

    #make a figure with all 1D radial distributions
    rad_dist_control_df['frac'] = rad_dist_control_df['frac']/rad_dist_control_df['frac'].sum()
    fig_rad = plot_1d_distr(rad_dist_control_df, 'radius', 'frac')
    fig_rad.data[0].marker.color = colors_all[0]
    fig_rad.data[0].line.color = colors_all[0]
    fig_rad.data[0].name = 'control'
    y_max = max(fig_rad.data[0].y)

    #make a figure with all 1D angular distributions
    if angle_dist_control_df.shape[0]>0:
        plot_data_df = angle_dist_control_df
        plot_data_df['frac'] = plot_data_df['frac']/plot_data_df['frac'].sum()
    else:
        plot_data_df = pd.DataFrame({'angle': [0], 'frac': [0]})
    fig_angle = plot_1d_polar(plot_data_df, 'angle', 'frac')
    fig_angle.data[0].marker.color = colors_all[0]
    fig_angle.data[0].line.color = colors_all[0]
    fig_angle.data[0].name = 'control'
    y_max2 = max(fig_angle.data[0].r)

    #identify strongest input cell types
    input_instances = fetch_top_input_instances(instance_name, neuropil, limit)

    data_row = []
    fig_ctr = 1
    for input_instance in input_instances['instance'].values:

        #fraction of synapses per column
        hex_instance_df = fetch_n_find_rel_hex(hex_red_df, input_instance, instance_name, neuropil)
        if hex_instance_df.empty:
            continue
        data_row.append(input_instance)

        hex_instance_df = hex_instance_df.groupby(['bodyId_post','hex1_id','hex2_id'])['bodyId_pre'].count().to_frame(name='count').reset_index()
        hex_instance_df['frac'] = hex_instance_df.groupby('bodyId_post')['count'].apply(lambda x: x/x.sum()).values
        hex_instance_df['hex_names'] = 100*(hex_instance_df['hex1_id']+hex0) + hex_instance_df['hex2_id']+hex0
        mean_df = hex_instance_df.groupby(['bodyId_post','hex_names'])['frac'].first().unstack(-1, 0).mean(0).to_frame(name='frac').reset_index()
        mean_df['hex1_id'] = (mean_df['hex_names'].values/100).astype(int)-hex0
        mean_df['hex2_id'] = mean_df['hex_names']-100*(mean_df['hex1_id']+hex0)-hex0

        rad_dist_df, angle_dist_df = find_1d_input_distrs(mean_df) #hex_instance_df)

        #significance: KL divergence to control (normalization into prob distr's is taken care of in entropy function)
        rad_dist_df = rad_dist_df.merge(rad_dist_control_df, on='radius', suffixes=('','_control'))
        rad_pos_df = rad_dist_df[rad_dist_df['frac_control']>0]
        kl_div = entropy(rad_pos_df['frac'].values, rad_pos_df['frac_control'].values)
        data_row.append('%.4f'%kl_div)
        if angle_dist_control_df.shape[0]*angle_dist_df.shape[0]>0:
            angle_dist_df = angle_dist_df.merge(angle_dist_control_df, on='angle', suffixes=('','_control'))
            angle_pos_df = angle_dist_df[angle_dist_df['frac_control']>0]
            kl_div = entropy(angle_pos_df['frac'].values, angle_pos_df['frac_control'].values)
            data_row.append('%.4f'%kl_div)

        #add to radial distribution
        rad_dist_df['frac'] = rad_dist_df['frac']/rad_dist_df['frac'].sum()
        fig = plot_1d_distr(rad_dist_df, 'radius', 'frac')
        fig.data[0].marker.color = colors_all[fig_ctr]
        fig.data[0].line.color = colors_all[fig_ctr]
        fig.data[0].name = input_instance
        y_max = max([max(fig.data[0].y), y_max])
        fig_rad.add_traces(fig.data)
        fig_rad.update_layout(annotations=fig_rad.layout.annotations + fig.layout.annotations)

        #add to angular distribution
        if angle_dist_df.shape[0]>0:
            angle_dist_df['frac'] = angle_dist_df['frac']/angle_dist_df['frac'].sum()
            fig2 = plot_1d_polar(angle_dist_df, 'angle', 'frac')
            fig2.data[0].marker.color = colors_all[fig_ctr]
            fig2.data[0].line.color = colors_all[fig_ctr]
            fig2.data[0].name = input_instance
            y_max2 = max([max(fig2.data[0].r), y_max2])
            fig_angle.add_traces(fig2.data)
            fig_angle.update_layout(annotations=fig_angle.layout.annotations + fig2.layout.annotations)

        fig_ctr += 1

    #save plots
    save_fig_path = Path(Path(find_dotenv()).parent, 'cache', 'input_distribution', 'special cell types')
    save_fig_path.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")

    fig_rad.update_layout(yaxis_range=[0, 1.2*y_max])
    fig_rad.update_layout(title=dict(text=f'Mean radial pre-synapse distribution into {instance_name[:-2]}', x=0.5, y=0.9))
    fig_rad.write_image(os.path.join(save_fig_path, f'{instance_name}_{neuropil[:-3]}_top{limit}_{date}_radial.pdf'))

    fig_angle.update_layout(polar=dict(radialaxis=dict(range=[0, 1.2*y_max2])))
    fig_angle.update_layout(title=dict(text=f'Mean angular pre-synapse distribution into {instance_name[:-2]}', x=0.5, y=0.9))
    fig_angle.write_image(os.path.join(save_fig_path, f'{instance_name}_{neuropil[:-3]}_top{limit}_{date}_angular.pdf'))

    return data_row


# =============================================================================
# From plotting_functions.py
# =============================================================================

def plot_heatmap(
    heatmap,
    anno=None,
    binned=False,
    bins=None,
    bins_text=None,
    pal=None,
    show_colorbar=True,
    show_grid=False,
    show_bounding_box=True,
    anno_text_size=6,
    gap=True,
    equal_aspect_ratio=True,
    manual_margin=True,
):
    """
    Plots heatmap, binned or continuous
    Parameters
    ----------
    heatmap : pandas.DataFrame
        dataframe to be plotted
    anno : pd.DataFrame, default=None
        annotation dataframe
    binned : bool, default=False
        if True, plot binned heatmap
    bins : np.ndarray or list, default=None
        7 boundary values for 6 bins. For cont. color scale, bins[-1] sets the max value.
    bins_text : list, default=None
        list of strings to be used as ticktext for colorbar
    pal : list, default=[]
        palette for heatmap
    show_colorbar : bool, default=True
        show colorbar
    show_grid : bool, default=False
        show all gray gridlines, behind heatmap
    show_bounding_box : bool, default=True
        show bounding box around heatmap
    anno_text_size : int, default=6
        annotation text size
    gap : bool, default=True
        adds a small gap between squares in the heatmap
    equal_aspect_ratio : bool, default=True
        sets the aspect ratio (of initial plot and for interactive scaling) to 1
    manual_margin : bool, default=True
        manually sets the margin of the layout variables 'l', 'r', and 'pad' to 0
    Returns
    -------
    fig : go.Figure
        Plotly figure object containing a heatmap
    """

    # plotting params, cf. Laura
    fig_format = {
        'fig_width': 3
      , 'fig_height': 3
      , 'fig_margin': 0.01
      , 'font_type': 'arial'
      , 'fsize_ticks_pt': 6
      , 'fsize_title_pt': 6
      , 'markersize': 10
      , 'markerlinewidth': 1
      , 'markerlinecolor': 'black'
      , 'ticklen': 3.5
      , 'tickwidth': 1
      , 'axislinewidth': 1.2
    }

    fig_w = (fig_format['fig_width'] - fig_format['fig_margin'])*96
    fig_h = (fig_format['fig_height'] - fig_format['fig_margin'])*96
    fsize_ticks_px = fig_format['fsize_ticks_pt']*(1/72)*96
    fsize_title_px = fig_format['fsize_title_pt']*(1/72)*96

    layout_heatmap = go.Layout(
        paper_bgcolor='rgba(255,255,255,1)'
      , plot_bgcolor='rgba(255,255,255,1)'
      , autosize = False
      , width = fig_w
      , height = fig_h
      , showlegend = False
    )

    layout_xaxis_heatmap = go.layout.XAxis(
        title_font={
            'size': fsize_title_px
          , 'family': fig_format['font_type']
          , 'color' : 'black'
        }
      , ticks=""
      , tickfont={
            'family': fig_format['font_type']
          , 'size': fsize_ticks_px
          , 'color': 'black'
        }
      , tickangle=45
      , side="top"
      , showgrid = False
      , showline= False
    )

    layout_yaxis_heatmap = go.layout.YAxis(
        title_font={
            'size': fsize_title_px
          , 'family': fig_format['font_type']
          , 'color': 'black'
        }
      , ticks=""
      , tickfont={
            'family': fig_format['font_type']
          , 'size': fsize_ticks_px
          , 'color': 'black'
        }
      , showgrid = False
      , showline= False
      , autorange="reversed"
    )

    # if anno is not None, it must have the same shape as heatmap
    if anno is not None:
        assert heatmap.shape == anno.shape, 'heatmap and annotation should have the same shape'

    # if pal provide
    if pal is not None:
        assert len(pal) > 1, 'palette should have at least 2 colors'
    else:
        # default color
        pal = OL_COLOR.HEATMAP.hex

    # initialize figure
    fig = go.Figure(layout = layout_heatmap)
    fig.update_xaxes(layout_xaxis_heatmap)
    fig.update_yaxes(layout_yaxis_heatmap)

    if show_grid:
        xval = heatmap.columns.values
        if (isinstance(xval[0], np.int32)) | (isinstance(xval[0], np.int64)):
            dx = np.diff(xval)
            for j in range(xval.shape[0]-1):
                fig.add_vline(x=xval[j]+dx[j]/2, line_width=.25, line_color="gray")
        yval = heatmap.index.values
        if (isinstance(yval[0], np.int32)) | (isinstance(yval[0], np.int64)):
            dy = np.diff(yval)
            for i in range(yval.shape[0]-1):
                fig.add_hline(y=yval[i]+dy[i]/2, line_width=.25, line_color="gray")

    # plot either binned/discrete or continuous heatmap
    if binned:
        if bins is not None:
            assert len(bins) > 2, 'bins should have at least 3 values, ie. 2 bins'
            if bins_text is None:
                bins_text = bins
            else:
                assert len(bins) == len(bins_text), 'bins and bins_text should have the same length'
        else:
            bins = heatmap.values.max()*np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
            bins_text = [f'{b:.1f}' for b in bins]

        heatmap_bin = heatmap.copy()
        heatmap_bin[:] = 1
        for i in range(len(bins)-1):
            heatmap_bin[(heatmap > bins[i]) & (heatmap <= bins[i+1])] = i+1

        # these 2 var are basically constants, keep them here for possible extension
        bvals = np.arange(len(bins))
        nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values

        # tick positions at boundaries
        bpos = bvals

        # create discrete colorscale
        dcolorscale = []
        for idx, val in enumerate(pal):
            dcolorscale.extend([[nvals[idx], val], [nvals[idx+1], val]])

        # add heatmap
        fig.add_trace(
            go.Heatmap(
                z=heatmap_bin
              , x=heatmap_bin.columns.values
              , y=heatmap_bin.index.values
              , colorscale=dcolorscale
              , zmax=len(bins)-1
              , zmin=1
              , colorbar={
                    'tickvals': bpos
                  , 'ticktext': bins_text
                }
            )
        )
    # continuous heatmap
    else:
        if bins is not None:
            assert len(bins) >=2, 'at least 2 value'
            bins_text = bins
        else:
            # 6 bins linearly spaced
            bins = np.linspace(0, heatmap.values.max(), 2)
            bins_text = [f'{b:.1f}' for b in bins]

        # tick positions at boundaries
        bpos = bins

        fig.add_trace(
            go.Heatmap(
                z=heatmap
              , x=heatmap.columns.values
              , y=heatmap.index.values
              , colorscale= OL_COLOR.HEATMAP.rgb
              , zmin=0
              , zmax=bins[-1]
              , colorbar={
                    'tickvals': bpos
                  , 'ticktext': bins_text
                }
            )
        )

    # bounding box around the plot, works better with asp=1
    if show_bounding_box:
        n_y, n_x = heatmap.shape
        fig.add_shape(
            type='rect', xref='x', yref='y'
          , x0=-0.5, y0=-0.5, x1=n_x-0.5, y1=n_y-0.5
          , line={'color': 'black', 'width': 0.5}
        )

    # with colorbar
    fig.update_traces(showscale=show_colorbar)

    # add space between cells
    if gap:
        fig.update_traces(xgap=1, ygap=1)

    # fix aspect ratio
    if equal_aspect_ratio:
        fig.update_layout(yaxis_scaleanchor="x")

    #set margin
    if manual_margin:
        fig.update_layout(margin={'l':0, 'r':0, 'pad':0})

    # with colorbar
    fig.update_traces(showscale=show_colorbar)

    # add space between cells
    if gap:
        fig.update_traces(xgap=1, ygap=1)

    # fix aspect ratio
    if equal_aspect_ratio:
        fig.update_layout(yaxis_scaleanchor="x")

    #set margin
    if manual_margin:
        fig.update_layout(margin={'l':0, 'r':0, 'pad':0})

    # annotations
    fig.update_traces(
        text=anno
      , texttemplate="%{text}"
      , textfont_size=anno_text_size
      , hovertemplate=None
      , textfont_family=fig_format['font_type']
    )

    return fig


def plot_flip_syn_hist(
    fig_obj,
    hist_celltype_df,
    roi_to_plot,
    row_num, col_num,
    layer_bound_dict,
):
    """
    Modified from https://github.com/reiserlab/optic-lobe-connectome/blob/main/src/utils/plotting_functions.py

    Parameters
    ----------
    fig_obj : go.Figure
        plotly go.Figure to be modified
    hist_df : pd.DataFrame
        hist_df for an individual target celltype. must contain columns 'roi', 'count', and 'hist_cen'
    roi_to_plot : str
        name of ROI for which to extract the relevant target celltype data
    row_num : int
        row position in the subplot
    col_num : int
        column position in the subplot
    layer_bound_dict : dict
        dictionary for the layer boundaries (value) of each ROI (keys)

    Returns
    -------
    go.Figure :
        modified plotly go.Figure `fig_obj`
    """
    y_range = [0, 1]
    gridline_col = 'gainsboro'
    input_color = 'rgb(253, 198, 43, 0.8)'
    tmp_bnd = layer_bound_dict[roi_to_plot]

    for boundary in tmp_bnd:
        fig_obj.add_trace(
            go.Scatter(
                x=y_range
              , y=[boundary, boundary]
              , mode='lines'
              , line={'color': gridline_col, 'width': .75}
              , showlegend=False)
          , row=row_num, col=col_num)

    rel_hist_df = hist_celltype_df[hist_celltype_df['roi'] == roi_to_plot]
    fig_obj.add_trace(
        go.Scatter(
            x=rel_hist_df['count']
          , y=rel_hist_df['hist_cen']
          , fill='tozerox'
          , fillcolor=input_color
          , mode='lines'
          , line={'color': 'black', 'width': .5}
          , showlegend=False
        #   , title=roi_to_plot
        )
      , row=row_num
      , col=col_num
    )

    fig_obj.update_xaxes(range=[-0.01,1.01])
    fig_obj.update_yaxes(title_text=roi_to_plot, showticklabels = False, autorange="reversed", row=row_num, col=col_num)

    return fig_obj


# =============================================================================
# From prop_by_adj.py
# =============================================================================

def solve_column_chunk(args):
    """
    Solve a chunk of columns for matrix inversion
    Returns sparse matrix components (row_indices, col_indices, data_values)
    """
    # Unpack arguments
    A_data, A_indices, A_indptr, A_shape, start_col, end_col, threshold = args

    # Reconstruct sparse matrix from shared components
    A_sparse = scipy.sparse.csc_matrix((A_data, A_indices, A_indptr), shape=A_shape)

    n = A_shape[0]
    chunk_size = end_col - start_col

    # Storage for this chunk's results
    row_list = []
    col_list = []
    data_list = []

    print(f"Process solving columns {start_col}-{end_col-1}")

    try:
        # Create identity matrix for this chunk
        I_chunk = np.zeros((n, chunk_size), dtype=np.float32)
        for j, col_idx in enumerate(range(start_col, end_col)):
            I_chunk[col_idx, j] = 1.0

        # Solve the entire chunk at once
        solutions = scipy.sparse.linalg.spsolve(A_sparse, I_chunk)

        # Handle 1D case (single column)
        if solutions.ndim == 1:
            solutions = solutions.reshape(-1, 1)

        # Process each column in the chunk
        for j, col_idx in enumerate(range(start_col, end_col)):
            col_solution = solutions[:, j]

            # Apply threshold to eliminate numerical zeros
            significant_mask = np.abs(col_solution) > threshold

            if np.any(significant_mask):
                significant_rows = np.where(significant_mask)[0]
                significant_values = col_solution[significant_rows]

                # Store sparse components
                row_list.extend(significant_rows.tolist())
                col_list.extend([col_idx] * len(significant_rows))
                data_list.extend(significant_values.tolist())

        print(f"Process completed columns {start_col}-{end_col-1}, found {len(data_list)} non-zeros")
        return row_list, col_list, data_list

    except Exception as e:
        print(f"Error in process handling columns {start_col}-{end_col-1}: {e}")
        return [], [], []

def parallel_sparse_inverse(A_sparse, n_processes=None, chunk_size=None, threshold=1e-12):
    """
    Compute sparse matrix inverse using multiprocessing
    """
    n = A_sparse.shape[0]

    # Set default parameters
    if n_processes is None:
        n_processes = min(psutil.cpu_count(), 8)  # Don't use too many cores

    if chunk_size is None:
        chunk_size = max(100, n // (n_processes * 4))  # Aim for 4x more chunks than processes

    print(f"Matrix size: {n}x{n}")
    print(f"Using {n_processes} processes")
    print(f"Chunk size: {chunk_size} columns")
    print(f"Threshold: {threshold}")

    # Convert to CSC format for efficient column access
    if not isinstance(A_sparse, scipy.sparse.csc_matrix):
        A_sparse = A_sparse.tocsc()

    # Prepare arguments for multiprocessing
    # We pass the sparse matrix components to avoid pickling issues
    args_list = []
    for start_col in range(0, n, chunk_size):
        end_col = min(start_col + chunk_size, n)
        args = (
            A_sparse.data,
            A_sparse.indices,
            A_sparse.indptr,
            A_sparse.shape,
            start_col,
            end_col,
            threshold
        )
        args_list.append(args)

    print(f"Created {len(args_list)} chunks")

    # Process chunks in parallel
    start_time = time.time()

    with Pool(processes=n_processes) as pool:
        results = pool.map(solve_column_chunk, args_list)

    processing_time = time.time() - start_time
    print(f"Parallel processing completed in {processing_time:.1f} seconds")

    # Combine results from all processes
    print("Combining results...")
    all_row_indices = []
    all_col_indices = []
    all_data_values = []

    for row_list, col_list, data_list in results:
        all_row_indices.extend(row_list)
        all_col_indices.extend(col_list)
        all_data_values.extend(data_list)

    print(f"Total non-zeros found: {len(all_data_values):,}")

    # Create the sparse inverse matrix
    print("Building sparse inverse matrix...")
    A_inv_sparse = scipy.sparse.coo_matrix(
        (all_data_values, (all_row_indices, all_col_indices)),
        shape=(n, n),
        dtype=np.float32
    )

    # Convert to CSC for efficient operations
    A_inv_sparse = A_inv_sparse.tocsc()

    # Calculate statistics
    density = A_inv_sparse.nnz / (n * n)
    memory_gb = (A_inv_sparse.data.nbytes + A_inv_sparse.indices.nbytes + A_inv_sparse.indptr.nbytes) / 1e9

    print(f"\nInverse matrix statistics:")
    print(f"  Shape: {A_inv_sparse.shape}")
    print(f"  Non-zeros: {A_inv_sparse.nnz:,}")
    print(f"  Density: {density:.2e}")
    print(f"  Memory usage: {memory_gb:.2f} GB")
    print(f"  Data type: {A_inv_sparse.dtype}")
    print(f"  Total time: {time.time() - start_time:.1f} seconds")

    return A_inv_sparse

# Memory monitoring function
def monitor_memory():
    """Monitor system memory usage"""
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent:.1f}% ({memory.used/1e9:.1f}/{memory.total/1e9:.1f} GB)")
    if memory.percent > 85:
        print("Warning: High memory usage!")
    return memory.percent


def parallel_sparse_inverse_safe(A_sparse, **kwargs):
    """Wrapper with error handling and cleanup"""
    try:
        print("Starting parallel matrix inversion...")
        monitor_memory()

        # Call your function
        result = parallel_sparse_inverse(A_sparse, **kwargs)

        print("Success! Final memory state:")
        monitor_memory()
        return result

    except MemoryError:
        print("Memory error! Try reducing chunk_size or n_processes")
        gc.collect()
        return None

    except Exception as e:
        print(f"Error during inversion: {e}")
        traceback.print_exc()
        gc.collect()
        return None


# =============================================================================
# From ol_rf.py
# =============================================================================

def pqw_LO(
    bodyId,
    thr_qt=0.0,
    syntype='post',
):
    """
    get RF for LO neuorns using column rois

    Args:

    Returns:

    df : pd.DataFrame
        data frame with columns ['bodyId', 'type', 'instance']
    """

    # bodyId = 33381
    # query synapse data
    neuron_df, roi_counts_df = fetch_neurons(NC(bodyId=bodyId))
    df = roi_counts_df[roi_counts_df['roi'].str.contains('LO_R_col_')] # keep LO col rois
    # thresholding synapse count
    thr_post = df[syntype].quantile(thr_qt)
    df = df[(df[syntype] >= thr_post)]

    # return the input col numbers
    hex = np.array([int(s) for t in df['roi'] for s in re.findall(r'\d+', t)]).reshape(-1,2)
    # add weights
    pqw = pd.DataFrame(
        np.concatenate((hex, df[syntype].to_numpy().reshape(-1,1)), axis=1),
        columns=['q', 'p', 'wt']
        )
    pqw.sort_values(by='wt', ascending=False, inplace=True)
    pqw.reset_index(drop=True, inplace=True)

    return pqw
