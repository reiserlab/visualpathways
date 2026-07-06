"""
Assign visual-input hopping groups (T0/T1/T2) to central-brain neurons.

Processing helper that starts from visually-driven VPN instances (T0) and walks
downstream connectivity to label the first two hops (T1, T2) that receive a
sufficient fraction of their input from the visual stream. Kept here as a
processing module; it queries neuprint directly and is currently not wired into
the main pipeline.
"""

import numpy as np
import pandas as pd
import neuprint
from neuprint import fetch_adjacencies


def assign_vpn_hop(meta_cb_vpn,  rois_cb, oltypes, rf_fit, mode='vic', thr_vic=0, thr_inwt=10):
    """
    Compute hopping groups (ghop) for visual input connectivity starting from VPNs

    Parameters:
    -----------
    meta_cb_vpn : pd.DataFrame
        The meta DataFrame containing ['bodyId', 'cell_type_side', 'instance', 'vision', 'ht', 'main_groups']
    rois_cb : list
        List of central brain ROIs
    oltypes : pd.DataFrame
        DataFrame containing all optic lobe cell types
    rf_fit : pd.DataFrame
        DataFrame containing RF fit data with bodyId and r2 columns
    mode : str, optional
        Either 'vic' (visual input contribution) or 'res' (resolution) (default: 'vic')
    thr_vic : float, optional
        Visual input contribution threshold (default: 0)
    thr_inwt : float, optional
        Input weight threshold percentage (default: 10)

    Returns:
    --------
    meta_cb_vpn : pd.DataFrame
        Updated meta DataFrame with 'ghop' column for each bodyId
    meta_type_cb_vpn : pd.DataFrame
        Updated meta DataFrame with 'ghop' column for each cell type and instance
    """

    # Meta by type - different aggregations based on mode
    if mode == 'vic':
        meta_type_cb_vpn = meta_cb_vpn.groupby(['cell_type_side', 'instance']).agg(
            bodyId_count=('bodyId', 'count'),
            vision=('vision', 'median'),
            ht=('ht', 'median'),
            main_groups=('main_groups', 'first'),
        ).reset_index()

        # Filter T0 instances (VIC mode)
        inst_t0 = meta_type_cb_vpn[
            (meta_type_cb_vpn['vision'] > thr_vic/100) &
            (meta_type_cb_vpn['main_groups'] == 'VPN')
        ]['instance'].values

    elif mode == 'res':
        meta_type_cb_vpn = meta_cb_vpn.groupby(['cell_type_side', 'instance']).agg(
            bodyId_count=('bodyId', 'count'),
            area_fit=('area_fit', 'median'),
            area_fit_input=('area_fit_input', 'median'),
            r2=('r2', 'median'),
            col_count=('col_count', 'median'),
            col_count_input=('col_count_input', 'median'),
            vision=('vision', 'median'),
            vision_cv=('vision', lambda x: round(np.std(x) / np.mean(x), 2) if np.mean(x) != 0 else np.nan),
            ht=('ht', 'median'),
            main_groups=('main_groups', 'first'),
        ).reset_index()

        # Filter T0 instances (resolution mode)
        inst_t0 = meta_type_cb_vpn[
            (meta_type_cb_vpn['vision'] > thr_vic/100) &
            (meta_type_cb_vpn['r2'] >= 0) &
            (meta_type_cb_vpn['main_groups'] == 'VPN')
        ]['instance'].values
    else:
        raise ValueError(f"mode must be 'vic' or 'res', got '{mode}'")

    # Get T0 -> T1 connections
    neuron_df, connection_df = fetch_adjacencies(
        sources=inst_t0,
        targets=None,
        rois=rois_cb,
        min_total_weight=1
    )
    conn_t0t1 = neuprint.merge_neuron_properties(neuron_df, connection_df, ['type', 'instance'])\
        .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
        .agg({'weight': 'sum'})\
        .reset_index()\
        .sort_values(by='weight', ascending=False)

    # Filter T1 candidates
    conn_t0t1 = conn_t0t1[
        ~conn_t0t1['instance_post'].str.contains("\\'") &
        ~conn_t0t1['instance_post'].str.contains("\\?") &
        ~conn_t0t1['instance_post'].str.contains("\\+") &
        ~conn_t0t1['instance_post'].str.contains("unclear") &
        ~conn_t0t1['instance_post'].isin(oltypes['instance'])
    ]
    inst = conn_t0t1['instance_post'].unique()

    # Calculate input weights for T1
    neuron_df, connection_df = fetch_adjacencies(sources=None, targets=inst, min_total_weight=1)
    idwt_t1 = neuprint.merge_neuron_properties(
        neuron_df, connection_df, ['type', 'instance']
    ).groupby(['bodyId_post', 'instance_post']).agg({'weight': 'sum'}).reset_index()

    df = pd.merge(
        conn_t0t1.groupby(['bodyId_post', 'instance_post']).agg(conn_wt=('weight', 'sum')).reset_index(),
        idwt_t1[['bodyId_post', 'weight']], on='bodyId_post', how='left'
    )
    df['inwt'] = df['conn_wt'] / df['weight']
    df = pd.merge(df, rf_fit[['bodyId', 'r2']], left_on='bodyId_post', right_on='bodyId', how='left')

    df_inst = df.groupby(['instance_post']).agg(
        bodyId_count=('bodyId_post', 'count'),
        inwt_median=('inwt', 'median'),
        r2=('r2', 'median')
    ).reset_index()

    # Filter T2 instances - different criteria for VIC vs resolution
    if mode == 'vic':
        inst_t1 = df_inst.loc[
            df_inst['inwt_median'] >= thr_inwt/100,
            'instance_post'
        ].unique()
    elif mode == 'res':
        inst_t1 = df_inst.loc[
            (df_inst['inwt_median'] >= thr_inwt/100) &
            (df_inst['r2'] >= 0),
            'instance_post'
        ].unique()

    # Get T0/T1 -> T2 connections
    neuron_df, connection_df = fetch_adjacencies(
        sources=np.concatenate([inst_t0, inst_t1]),
        targets=None,
        rois=rois_cb,
        min_total_weight=1
    )
    conn_t0t1t2 = neuprint.merge_neuron_properties(neuron_df, connection_df, ['type', 'instance'])\
        .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
        .agg({'weight': 'sum'})\
        .reset_index()\
        .sort_values(by='weight', ascending=False)

    # Filter T2 candidates
    conn_t0t1t2 = conn_t0t1t2[
        ~conn_t0t1t2['instance_post'].str.contains("\\'") &
        ~conn_t0t1t2['instance_post'].str.contains("\\?") &
        ~conn_t0t1t2['instance_post'].str.contains("\\+") &
        ~conn_t0t1t2['instance_post'].str.contains("unclear") &
        ~conn_t0t1t2['instance_post'].isin(oltypes['instance']) &
        ~conn_t0t1t2['instance_post'].isin(inst_t1)
    ]
    inst = conn_t0t1t2['instance_post'].unique()

    # Calculate input weights for T2
    neuron_df, connection_df = fetch_adjacencies(sources=None, targets=inst, min_total_weight=1)
    idwt = neuprint.merge_neuron_properties(
        neuron_df, connection_df, ['type', 'instance']
    ).groupby(['bodyId_post', 'instance_post']).agg({'weight': 'sum'}).reset_index()

    df = pd.merge(
        conn_t0t1t2.groupby(['bodyId_post', 'instance_post']).agg(conn_wt=('weight', 'sum')).reset_index(),
        idwt[['bodyId_post', 'weight']], on='bodyId_post', how='left'
    )
    df['inwt'] = df['conn_wt'] / df['weight']
    df = pd.merge(df, rf_fit[['bodyId', 'r2']], left_on='bodyId_post', right_on='bodyId', how='left')

    df_inst = df.groupby(['instance_post']).agg(
        bodyId_count=('bodyId_post', 'count'),
        inwt_median=('inwt', 'median'),
        r2=('r2', 'median')
    ).reset_index()

    # Filter T2 instances - different criteria for VIC vs resolution
    if mode == 'vic':
        inst_t2 = df_inst.loc[
            df_inst['inwt_median'] >= thr_inwt/100,
            'instance_post'
        ].unique()
    elif mode == 'res':
        inst_t2 = df_inst.loc[
            (df_inst['inwt_median'] >= thr_inwt/100) &
            (df_inst['r2'] >= 0),
            'instance_post'
        ].unique()

    # Assign hopping groups
    meta_cb_vpn['ghop'] = meta_cb_vpn['main_groups'].map({'BVNC': 'later', 'VPN': 'VPN'})
    meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t0), 'ghop'] = 'T0'
    meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t1), 'ghop'] = 'T1'
    meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t2), 'ghop'] = 'T2'

    meta_type_cb_vpn['ghop'] = meta_type_cb_vpn['main_groups'].map({'BVNC': 'later', 'VPN': 'VPN'})
    meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t0), 'ghop'] = 'T0'
    meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t1), 'ghop'] = 'T1'
    meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t2), 'ghop'] = 'T2'

    return meta_cb_vpn, meta_type_cb_vpn
