import os
import numpy as np
import pickle as pkl
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv
from queries.completeness import fetch_ol_types_and_instances
from neuprint import NeuronCriteria as NC, fetch_neurons, fetch_adjacencies, merge_neuron_properties


def filter_instance_connectivity(filter, seed_instances, sink_instances, \
                                    date0='2024-08-11', min_weight=5, min_tot_weight=5, r7r8_completion=False, \
                                    nt_included=None, nt_sign=False, excl_pairs=None, incl_instances=None):
    """
    Find instance connectivity based on several choices. 
    It is computed from the neuron-neuron connectivity data 
    by summing the relative input weights of all pre-neurons 
    and averaging over post-neurons.

    Parameters
    ----------
    filter : str
        'all' : use all connections
        'seed': removes inputs to seeds
        'sink': removes outputs of sinks
        'seed_sink': removes inputs to seeds and outputs of sinks
    seed_instances : list
        list of seed instances to be used for filter (can be empty)
    sink_instances : list
        list of sink instances to be used for filter (can be empty)      
    min_weight : int, default=5
        minimum neuron-neuron weight
    min_tot_weight : int, default=5
        minimum total input weight
    r7r8_completion : bool, default=False
        whether to use precomputed completed R7R8 connectivity
    nt_included : list, default=None
        list of neurotransmitters to include
    nt_sign : bool, default=False
        whether to include sign of neurotransmitter
    excl_pairs : array, default=None
        array of size Nx2 of N (instance_pre, instance_post) pairs to exclude
    incl_instances : list, default=None
        list of instances to include

    Returns
    -------
    conn_df : pd.DataFrame
        'instance_pre' : str
            pre-instance  
        'instance_post' : str
            post-instance    
        'rel_in_weight' : int
            averaged summed relative input weight per pre-post instance pair   
    """
    #load connectivity data
    conn_df, instance_df = filter_connectivity(filter, seed_instances, sink_instances, None, \
                                        date0=date0, min_weight=min_weight, min_tot_weight=min_tot_weight, \
                                        r7r8_completion=r7r8_completion, \
                                        nt_included=nt_included, nt_sign=nt_sign, \
                                        excl_pairs=excl_pairs, incl_instances=incl_instances)

    #compute instance connectivity
    conn_instance_df = conn_df.groupby(['instance_pre','bodyId_post']).agg({'instance_post':'first','rel_in_weight':'sum'}).reset_index()
    conn_instance_df = conn_instance_df.groupby(['instance_pre','instance_post'])['rel_in_weight'].sum().to_frame().reset_index()

    #normalize summed rel_in_weight by total number of post-neurons
    instance_df['count'] = instance_df.groupby('instance')['bodyId'].transform('count')
    conn_instance_df = conn_instance_df.merge(instance_df[['instance','count']].drop_duplicates(), left_on='instance_post', right_on='instance')
    conn_instance_df['rel_in_weight'] = conn_instance_df['rel_in_weight'] / conn_instance_df['count']

    return conn_instance_df


def find_instance_connectivity_to_matrix(conn_df):
    """
    From pairs of instance connectivities, compute a square matrix.

    Parameters
    ----------
    conn_df : pd.DataFrame
        'instance_pre' : str
            instance of pre neuron    
        'instance_post' : str
            instance of post neuron    
        'rel_in_weight' : int
            relative input weight between pre and post neuron    
        'int_pre' : int
            integer index of pre neuron
        'int_post' : int
            integer index of post neuron

    Returns
    -------
    conn_matrix : pd.DataFrame
        square dataframe with index 'instance_pre', columns 'instance_post', and entries equal to 'rel_in_weight'
    """
    #Connectivity matrix
    agg_weights_df = conn_df.groupby(['instance_pre', 'instance_post'])['rel_in_weight'].sum().reset_index()
    conn_matrix = agg_weights_df.pivot(index='instance_pre', columns='instance_post', values='rel_in_weight')
    conn_matrix = conn_matrix.fillna(0).astype('float')
    #Make sure matrix is square
    conn_matrix, _ = conn_matrix.align(conn_matrix.T, fill_value=0)

    return conn_matrix


def filter_connectivity(filter, seed_instances, sink_instances, path_to_instance_f, \
                           date0='2024-08-11', min_weight=5, min_tot_weight=5, r7r8_completion=False, \
                           nt_included=None, nt_sign=False, excl_pairs=None, incl_instances=None):
    """
    Find connectivity based on several choices. Store instance_df in path_to_instance_f.

    Parameters
    ----------
    filter : str
        'all' : use all connections
        'seed': removes inputs to seeds
        'sink': removes outputs of sinks
        'seed_sink': removes inputs to seeds and outputs of sinks
    seed_instances : list
        list of seed instances to be used for filter (can be empty)
    sink_instances : list
        list of sink instances to be used for filter (can be empty)     
    path_to_instance_f : str
        path to instance_df file (if None then don't save)
    date0 : str, default='2024-08-11'
        date of connectivity data
    min_weight : int, default=5
        minimum neuron-neuron weight
    min_tot_weight : int, default=5
        minimum total input weight
    r7r8_completion : bool, default=False
        whether to use precomputed completed R7R8 connectivity
    nt_included : list, default=None
        list of neurotransmitters to include (only works with optic-lobe dataset)
    nt_sign : bool, default=False
        whether to include sign of neurotransmitter (only works with optic-lobe dataset)
    excl_pairs : array, default=None
        array of size Nx2 of N (instance_pre, instance_post) pairs to exclude
    incl_instances : list, default=None
        list of instances to include

    Returns
    -------
    conn_df : pd.DataFrame
        'bodyId_pre' : int
            bodyId of pre neuron
        'bodyId_post' : int
            bodyId of post neuron    
        'instance_pre' : str
            instance of pre neuron    
        'instance_post' : str
            instance of post neuron    
        'rel_in_weight' : int
            relative input weight between pre and post neuron    
        'weight' : int
            weight between pre and post neuron    
        'tot_weight' : int
            total input weight of post neuron
        'int_pre' : int
            integer index of pre neuron
        'int_post' : int
            integer index of post neuron
    instance_df : pd.DataFrame
        'bodyId' : int
            bodyId of neuron
        'instance' : str
            instance of neuron
        if (nt_included is not None)|nt_sign then also contains
        'consensusNt' : str
            consensus neurotransmitter of neuron
    """
    assert filter in ['all','seed','sink','seed_sink'],\
            f"filter must be one of 'all','seed','sink','seed_sink' but is actually '{filter}'"

    #load connectivity data
    conn_df = query_connectivity(date0, min_weight, min_tot_weight, r7r8_completion)

    #rename R78 instances
    for hemisphere in ['L','R']:
        for side in ['pre','post']:
            id0 = conn_df[conn_df[f'instance_{side}']==f'R7R8_unclear_{hemisphere}'].index
            if id0.shape[0]>0:
                conn_df.loc[id0,f'instance_{side}'] = f'r78_{hemisphere}'
            id0 = conn_df[conn_df[f'instance_{side}'].str.contains(f'R7[py]_{hemisphere}')].index
            if id0.shape[0]>0:
                conn_df.loc[id0,f'instance_{side}'] = f'R7_{hemisphere}'
            id0 = conn_df[conn_df[f'instance_{side}'].str.contains(f'R8[py]_{hemisphere}')].index
            if id0.shape[0]>0:
                conn_df.loc[id0,f'instance_{side}'] = f'R8_{hemisphere}'

    #get all instances in conn_df
    instance_pre_df = conn_df[['bodyId_pre','instance_pre']].drop_duplicates().rename(columns={'bodyId_pre':'bodyId','instance_pre':'instance'})
    instance_post_df = conn_df[['bodyId_post','instance_post']].drop_duplicates().rename(columns={'bodyId_post':'bodyId','instance_post':'instance'})
    instance_df = pd.concat([instance_pre_df,instance_post_df]).drop_duplicates().sort_values('bodyId')

    #remove input to sources
    if (filter=='seed')|(filter=='seed_sink'):
        id0 = conn_df[np.isin(conn_df['instance_post'],seed_instances)].index
        conn_df = conn_df.drop(id0)

    #remove feedback from outputs
    if (filter=='sink')|(filter=='seed_sink'):
        id0 = conn_df[np.isin(conn_df['instance_pre'],sink_instances)].index
        conn_df = conn_df.drop(id0)

    #only keep certain neurotransmitters and certain exceptions
    instance_exc = ['L1_R','R7_R','R8_R','HBeyelet_R']
    if (nt_included is not None)|nt_sign:
        if os.environ['NEUPRINT_DATASET_NAME']=='flywire':
            neuron_df = conn_df.groupby('bodyId_pre')['consensusNt'].first().reset_index().rename(columns={'bodyId_pre':'bodyId'})
        else:
            neuron_df, _ = fetch_neurons(NC(bodyId=instance_df['bodyId'].unique()))
        instance_df = instance_df.merge(neuron_df[['bodyId','consensusNt']], on='bodyId', how='left')     
        if nt_included is not None:
            instance_df = instance_df[(instance_df['consensusNt'].isin(nt_included))|(instance_df['instance'].isin(instance_exc))]   
            conn_df = conn_df[(conn_df['bodyId_pre'].isin(instance_df['bodyId']))&(conn_df['bodyId_post'].isin(instance_df['bodyId']))]

    #only keep certain instances
    if incl_instances is not None:
        instance_df = instance_df[instance_df['instance'].isin(incl_instances)]
        conn_df = conn_df[(conn_df['bodyId_pre'].isin(instance_df['bodyId']))&(conn_df['bodyId_post'].isin(instance_df['bodyId']))]

    #remove certain pairs of instances
    if excl_pairs is not None:
        conn_instance_tuples = np.array([tuple(row) for row in conn_df[['instance_pre','instance_post']].values])
        excl_pairs_tuples = np.array([tuple(row) for row in excl_pairs]) 
        id1 = np.in1d(conn_instance_tuples, excl_pairs_tuples)
        id0 = np.where( id1[::2]*id1[1::2]==0 )[0]
        conn_df = conn_df.iloc[id0]

    #recompute instances from filtered conn_df
    conn_df = conn_df.reset_index(drop=True)
    instance_pre_df = conn_df[['bodyId_pre','instance_pre']].drop_duplicates().rename(columns={'bodyId_pre':'bodyId','instance_pre':'instance'})
    instance_post_df = conn_df[['bodyId_post','instance_post']].drop_duplicates().rename(columns={'bodyId_post':'bodyId','instance_post':'instance'})
    instance_df = pd.concat([instance_pre_df,instance_post_df]).drop_duplicates().sort_values('bodyId').reset_index(drop=True)
    if (nt_included is not None)|nt_sign:
        instance_df = instance_df.merge(neuron_df[['bodyId','consensusNt']], on='bodyId', how='left') 

    if path_to_instance_f is not None:
        if not os.path.exists(path_to_instance_f):
            pkl.dump(instance_df, open(path_to_instance_f,'wb'))
        

    #flip sign
    nt_inh = ['gaba','glutamate']
    if nt_sign:
        instance_inh = instance_df[instance_df['consensusNt'].isin(nt_inh)]['instance'].values
        id_neg = conn_df[conn_df['instance_pre'].isin(instance_inh)].index
        conn_df.loc[id_neg,'rel_in_weight'] = -conn_df.loc[id_neg,'rel_in_weight']

    #convert bodyIds to ints
    bodyIds_to_ints = {instance_df['bodyId'].values[i]: i for i in range(instance_df.shape[0])}
    ints_pre = np.array([bodyIds_to_ints[item] for item in conn_df['bodyId_pre'].values])
    ints_post = np.array([bodyIds_to_ints[item] for item in conn_df['bodyId_post'].values])

    #store into dataframe
    conn_df['int_pre'] = ints_pre
    conn_df['int_post'] = ints_post

    return conn_df, instance_df


def query_connectivity(date0='2024-08-11', min_weight=5, min_tot_weight=5, \
                       r7r8_completion=False, dataset_name=os.environ['NEUPRINT_DATASET_NAME']):
    """
    Load connectivity data (temporary solution: fixed data that needs to be downloaded from dropbox). 
    Then threshold by minimum weight and minimum total weight.

    Parameters
    ----------
    min_weight : int, default=5
        minimum neuron-neuron weight
    min_tot_weight : int, default=5
        minimum total input weight
    r7r8_completion : bool, default=False
        whether to use precomputed completed R7R8 connectivity

    Returns
    -------
    conn_df : pd.DataFrame
        'bodyId_pre' : int
            bodyId of pre neuron
        'bodyId_post' : int
            bodyId of post neuron    
        'instance_pre' : str
            instance of pre neuron    
        'instance_post' : str
            instance of post neuron    
        'weight' : int
            weight between pre and post neuron    
        'tot_weight' : int
            total input weight of post neuron
    """
    data_path = Path(find_dotenv()).parent / 'cache' / 'connectivity'
    if 'optic-lobe' in dataset_name:
        save_f = data_path / f"{date0}_conn_df_bodyId.p"
    elif 'cns' in dataset_name:
        save_f = data_path / f"{date0}_cns_conn_df_bodyId.p"
    elif 'flywire' in dataset_name:
        save_f = data_path / f"{date0}_flywire_conn_df_bodyId.p"

    if os.path.exists(save_f):
        conn_df = pkl.load( open( save_f, "rb" ) )
    elif 'optic-lobe' in dataset_name:
        print(f'The file cache/connectivity/{date0}_conn_df_bodyId.p does not exist. Get todays connectivity...')
        date0 = pd.Timestamp.now().strftime('%Y-%m-%d')
        save_f = data_path / f"{date0}_conn_df_bodyId.p"
        if os.path.exists(save_f):
            conn_df = pkl.load( open( save_f, "rb" ) )
        else:
            ol_df = fetch_ol_types_and_instances()
            neu_df, conn_df = fetch_adjacencies(
                sources=NC(instance=ol_df['instance']),
                targets=NC(instance=ol_df['instance']), 
                rois=['OL(R)'], include_nonprimary=True,
                batch_size=1000)  
            conn_df['tot_weight'] = conn_df.groupby('bodyId_post')['weight'].transform('sum')
            conn_df = merge_neuron_properties(neu_df, conn_df, 'instance')
            pkl.dump(conn_df, open(save_f, 'wb'))
    else:
        raise ValueError(f"The file {save_f} does not exist. \nPlease download the data for {dataset_name} and store it in the appropriate format there.")

    if r7r8_completion & (('optic-lobe' in dataset_name)|('cns' in dataset_name)):
        color_df = pd.read_csv(Path(find_dotenv()).parent / 'params' / 'edge_list_R7R8_filled_in_121724_v1.csv.gz', compression='gzip')
        #remove corresponding R7, R8 rows from conn_df
        color_df.set_index(['bodyId_pre','bodyId_post'], inplace=True)
        conn_df.set_index(['bodyId_pre','bodyId_post'], inplace=True)
        idx_color = conn_df.index.intersection(color_df.index) 
        conn_df.drop(idx_color, inplace=True)
        #add full color_df to conn_df
        conn_df = pd.concat([conn_df, color_df], axis=0).reset_index()
        conn_df['tot_weight'] = conn_df.groupby('bodyId_post')['weight'].transform('sum')

    #threshold minimum weight and minimum total weight
    conn_df = conn_df[conn_df['weight']>min_weight]
    conn_df = conn_df[conn_df['tot_weight']>min_tot_weight]
    conn_df['rel_in_weight'] = conn_df['weight']/conn_df['tot_weight']
    
    return conn_df
