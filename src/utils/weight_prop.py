from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import os
import pickle as pkl
from pathlib import Path

from utils.config import CACHE_DIR, PARAMS_DIR
from utils.hex_hex import get_hex_df, all_hex_df
from utils.hex_geometry import find_neuron_hex_ids, load_hexed_body_ids
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses


def compute_bodyId_column_weight(step_all, instance_df, target_bodyId, seed_instance, roi_str='ME(R)'):
    """
    Compute effective weight from seed instance to target instance in each column.

    Parameters
    ----------
    step_all : csr_matrix
        sparse weight matrix
    instance_df : pd.DataFrame
        'instance' : str
            instance name
        'bodyId' : int
            bodyId of instance
    target_bodyId : int
        target bodyId
    seed_instances : str
        seed instance
    roi_str : str
        main target_bodyId neuropil, can only be ME(R,L), LO(R,L), LOP(R,L)

    Returns
    -------
    col_df : pd.DataFrame
        'target bodyId' : int
            target bodyId
        'col_id' : int
            column id
        'effective weight' : float
            effective weight of column on target bodyId
        'norm' : float
            total weight of column on target bodyId
    """
    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)','ME(L)', 'LO(L)', 'LOP(L)', but is actually '{roi_str}'"
    
    bodyIds_to_ints = {instance_df['bodyId'].values[i]: i for i in range(instance_df.shape[0])}

    #associate seed neurons with columns in roi_str
    seed_bodyIds = instance_df[instance_df['instance']==seed_instance]['bodyId'].values
    hex_df = all_hex_df()
    hex_df = hex_df.reset_index(names='col_id')
    me_hex_df = load_hexed_body_ids(roi_str='ME'+roi_str[-3:])
    lop_hex_df = load_hexed_body_ids(roi_str='LOP'+roi_str[-3:])
    if np.isin(seed_instance[:-2], me_hex_df.columns).item():
        me_hex_df = me_hex_df[['hex1_id','hex2_id',seed_instance[:-2]]].reset_index(drop=True).rename(columns={seed_instance[:-2]:'bodyId'}).dropna()
        me_hex_df = me_hex_df.merge(hex_df, on=['hex1_id','hex2_id'], how='left').drop_duplicates()
        seed_cols_df = me_hex_df[np.isin(me_hex_df['bodyId'],seed_bodyIds)]
    elif np.isin(seed_instance[:-2], lop_hex_df.columns).item():
        lop_hex_df = lop_hex_df[['hex1_id','hex2_id',seed_instance[:-2]]].reset_index(drop=True).rename(columns={seed_instance[:-2]:'bodyId'}).dropna()
        lop_hex_df = lop_hex_df.merge(hex_df, on=['hex1_id','hex2_id'], how='left').drop_duplicates()
        seed_cols_df = lop_hex_df[np.isin(lop_hex_df['bodyId'],seed_bodyIds)]
    elif np.isin(seed_instance[:-2], ['R7','R8']).item():
        seed_col_f = CACHE_DIR / f'{seed_instance[:-2]}_col_df.pkl'
        if os.path.exists(seed_col_f):
            seed_all_cols_df = pkl.load( open( seed_col_f, "rb" ) )
        else:
            color_df = pd.read_csv(PARAMS_DIR / 'edge_list_R7R8_filled_in_121724_v1.csv.gz', compression='gzip')
            color_bodyIds = color_df[color_df['instance_pre']==seed_instance]['bodyId_pre'].unique()
            #for new R7R8, just take last four numbers of bodyId
            new_color_bodyIds = np.array([s for s in color_bodyIds if str(s)[:6]==seed_instance[1]*6]) 
            hex1_ids = np.array([int(str(s)[-4:-2]) for s in new_color_bodyIds])
            hex2_ids = np.array([int(str(s)[-2:]) for s in new_color_bodyIds])
            new_color_df = pd.DataFrame({'bodyId':new_color_bodyIds,'hex1_id':hex1_ids,'hex2_id':hex2_ids})
            new_color_df = new_color_df.merge(hex_df, on=['hex1_id','hex2_id'], how='left')
            #for old R7R8, find column assignment through synapses
            neu_color_bodyIds = np.array([s for s in color_bodyIds if str(s)[:6]!=seed_instance[1]*6])
            syn_df = fetch_synapses(NC(bodyId=neu_color_bodyIds), SC(rois=[roi_str]))
            neu_color_df = find_neuron_hex_ids(syn_df, roi_str=roi_str, method='majority')
            #concatenate and store
            seed_all_cols_df = pd.concat([new_color_df,neu_color_df])
            seed_col_f = CACHE_DIR / f'{seed_instance[:-2]}_col_df.pkl'
            pkl.dump(seed_all_cols_df, open(seed_col_f,'wb'))
        seed_cols_df = seed_all_cols_df[np.isin(seed_all_cols_df['bodyId'],seed_bodyIds)]
    else:
        syn_df = fetch_synapses(NC(bodyId=seed_bodyIds), \
                                SC(rois=[roi_str], type='pre'))
        seed_cols_df = find_neuron_hex_ids(syn_df, roi_str=roi_str, method='majority')
        
    #reorder, just in case the ordering of bodyIds has changed
    seed_bodyIds = seed_cols_df['bodyId'].values

    #index of seed instances
    seed_type_ints = np.array([bodyIds_to_ints[item] for item in seed_bodyIds])
    seed_col_ids = seed_cols_df['col_id'].unique()
    n_cols = seed_col_ids.shape[0]

    #index of target bodyId
    target_int = bodyIds_to_ints[target_bodyId]

    #weight in each column
    col = np.zeros(n_cols)
    for j in range(n_cols):
        j0 = np.where( seed_cols_df['col_id'].values==seed_col_ids[j] )[0]
        col[j] = step_all[seed_type_ints[j0],target_int].mean(0).item()

    #normalize by total weight of columns
    norm_fac = np.sum(np.abs(col))
    #store in dataframe
    col_df = pd.DataFrame.from_dict( {\
            'target bodyId': target_bodyId*np.ones(n_cols,dtype='object'),\
            'col_id': seed_col_ids,\
            'effective weight': col,\
            'norm': norm_fac*np.ones(n_cols,dtype='object')} )
    col_df = col_df.merge(hex_df, on='col_id', how='left')

    return col_df


def compute_effective_weight(step_all, instance_df, target_instance, seed_instances):
    """
    Compute effective weight from each neuron of seed_instances to each neuron of target_instance.
    Effective weight is mean over all seed neurons of the same seed instance.

    Parameters
    ----------
    step_all : csr_matrix
        sparse weight matrix
    instance_df : pd.DataFrame
        'instance' : str
            instance name
        'bodyId' : int
            bodyId of instance
    target_instance : str
        target instance
    seed_instances : list of str
        seed instances

    Returns
    -------
    tot_seed_per_neuron_df : pd.DataFrame
        'target bodyId' : int
            target bodyId
        'seed instance' : str
            seed instance name
        'effective weight' : float
            effective weight of seed instance on target bodyId
        'norm' : float
            total weight of seed instance on target bodyId
    """
    bodyIds_to_ints = {instance_df['bodyId'].values[i]: i for i in range(instance_df.shape[0])}

    #index of seed instances
    n_seed = len(seed_instances)
    seed_type_ints = np.zeros(n_seed, dtype='object')
    for i in range(n_seed):
        seed_type_ints[i] = np.array([bodyIds_to_ints[item] for item in instance_df[instance_df['instance']==seed_instances[i]]['bodyId'].values])

    #index of target instances
    target_bodyId = instance_df[instance_df['instance']==target_instance]['bodyId'].values 
    target_ints = np.array([bodyIds_to_ints[item] for item in target_bodyId])
    n_target = target_ints.shape[0]

    #effective weight is mean over all neurons of the same seed instance
    tot_seed_per_neuron = np.zeros((n_target,n_seed))
    for j in range(n_seed):
        if seed_type_ints[j].shape[0]>0:
            for i in range(n_target):
                tot_seed_per_neuron[i,j] = step_all[seed_type_ints[j],target_ints[i]].mean(0).item()  

    #normalize by total weight over seed instances
    norm_fac = np.sum(np.abs(tot_seed_per_neuron),1)
    tot_seed_per_neuron_df = pd.DataFrame.from_dict( {\
            'target bodyId': np.squeeze( ( target_bodyId[:,np.newaxis]*np.ones(tot_seed_per_neuron.shape,dtype='object') ).reshape((-1,1)) ),\
            'seed instance': np.squeeze( ( np.array(seed_instances)[np.newaxis,:]*np.ones(tot_seed_per_neuron.shape,dtype='object') ).reshape((-1,1)) ),\
            'effective weight': np.squeeze( tot_seed_per_neuron.reshape((-1,1)) ),\
            'norm': np.squeeze( ( np.array(norm_fac)[:,np.newaxis]*np.ones(tot_seed_per_neuron.shape,dtype='object') ).reshape((-1,1)) )} )

    return tot_seed_per_neuron_df


def compute_matrix_power(path_to_prop_f, conn_df, n_steps=5, thre_weights=1e-5):
    """
    Compute sum of matrix powers of connectivity matrix; stores the result (~1GB!) in path_to_prop_f.

    Parameters
    ----------
    path_to_prop_f : str
        path to save the sparse propagated weight matrix
    conn_df : pd.DataFrame
        'int_pre' : int
            integer index of pre neuron    
        'int_post' : int
            integer index of post neuron    
        'rel_in_weight' : float
            relative input weight
    n_steps : int, default=10
        number of matrix powers to compute  
    thre_weights : float, default=0.00001
        threshold for setting propagated weights to zero

    Returns
    -------
    step_all : csr_matrix
        sparse propagated weight matrix
    """
    # create sparse matrices
    N_tot = max(conn_df['int_pre'].max(), conn_df['int_post'].max()) + 1
    step1 = csr_matrix((conn_df['rel_in_weight'].values, 
                        (conn_df['int_pre'].values, conn_df['int_post'].values)), 
                            shape = (N_tot, N_tot))

    step = set_small_elements_to_zero(step1, thre_weights)
    step_all = step1
    for i in range(n_steps):
        step = step1@step
        step = set_small_elements_to_zero(step, thre_weights)
        step_all = step_all+step

    if not os.path.exists(path_to_prop_f):
        pkl.dump(step_all, open(path_to_prop_f,'wb'))

    return step_all


def set_small_elements_to_zero(matrix, threshold):
    """
    Set small elements of a matrix to zero.

    Parameters
    ----------
    matrix : csr_matrix
        sparse matrix
    threshold : float
        threshold for setting elements to zero

    Returns
    -------
    new_matrix : csr_matrix
        sparse matrix with small elements set to zero
    """
    # Create a copy of the matrix
    new_matrix = matrix
    
    #threshold nonzero entries
    A = matrix.data
    new_matrix.data = np.where( np.abs(A)>threshold, A, 0 )
    
    #remove diagonal
    new_matrix[range(matrix.shape[0]),range(matrix.shape[0])] = 0    

    #remove 0s
    new_matrix.eliminate_zeros()
    
    return new_matrix