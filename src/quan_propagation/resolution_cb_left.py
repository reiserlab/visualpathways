# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: ol-analysis
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### for left OL, cf. resolution_cb.ipynb

# %%
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
from pathlib import Path
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")


# %%
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
import pandas as pd
import numpy as np
import re
import scipy
import json
import pickle

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import matplotlib as mpl
import matplotlib.pyplot as plt

import neuprint
from neuprint import fetch_neurons, fetch_adjacencies, NeuronCriteria as NC, SynapseCriteria as SC

import navis
import navis.interfaces.neuprint as neu

import connectome_interpreter as ci

# from fafbseg import flywire
# import flybrains

# %%
from quan_propagation.func import count_col

# # Test if count_col is imported
# print(count_col)

# %%
from utils.config import (
    CACHE_DIR, DATA_DIR, DATASET, FIG_DIR, HIT_THRE, N_FLOW, PARAMS_DIR,
)
DATA_DIR.mkdir(parents=True, exist_ok=True)

result_dir = FIG_DIR / 'quan_propagation'
result_dir.mkdir(parents=True, exist_ok=True)

cache_dir = CACHE_DIR / 'quan_propagation'
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Some setups

# %% [markdown]
# ## vpn types, fake left side

# %%
# cell type table xlsx
oltypes = pd.read_excel(PARAMS_DIR / 'Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx')
# rename columns
oltypes = oltypes.rename(columns={'cell type': 'cell_type', 'main groups': 'main_groups'})

# %%
# fake left-side, replace _R with _L, and _L with _R at the same time
oltypes['instance'] = oltypes['instance'].str.replace('_R$', '_TEMP', regex=True)
oltypes['instance'] = oltypes['instance'].str.replace('_L$', '_R', regex=True)
oltypes['instance'] = oltypes['instance'].str.replace('_TEMP$', '_L', regex=True)


# %%
print(oltypes.shape)

oltypes_nonvpn = oltypes[~oltypes['main_groups'].str.contains('VPN')]
oltypes_vpn = oltypes[oltypes['main_groups'].str.contains('VPN')]
oltypes_vcn = oltypes[oltypes['main_groups'].str.contains('VCN')]
oltypes_ol = oltypes[oltypes['main_groups'].str.contains('^ON')]


# %% [markdown]
# ## rois

# %%
# neuprint.fetch_primary_rois()
rois_dict = neuprint.queries.fetch_roi_hierarchy(include_subprimary=False)
# rois_dict['CNS'].keys()

# %%
# combine keys 
rois = set(rois_dict['CNS']['CentralBrain'].keys()) #| set(rois_dict['CNS']['Optic(R)'].keys())

rois = sorted(list(rois))
print(len(rois))

# keep primary rois by removing strings that doesn't have * at the end
rois = [r for r in rois if '*' in r]
# rois = [r for r in rois if '*' not in r]
print(len(rois))

# remove * at the end of the string
rois = [re.sub(r'\*$', '', roi) for roi in rois]
# remove strings that contain 'unspecified' 
rois = [roi for roi in rois if 'unspecified' not in roi]
print(len(rois))

rois_cb = rois.copy()


# %%
# right ol rois
rois = set(rois_dict['CNS']['Optic(R)'].keys())

rois = [re.sub(r'\*$', '', roi) for roi in rois]
rois = [roi for roi in rois if 'unspecified' not in roi]

rois_olr = rois.copy()

# # central brain mesh
# cb_mesh = neu.fetch_roi('CentralBrain')

# %%


# %% [markdown]
# # load and sum up prop matrices

# %%
stepsn = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_l_lat_flow_sum.npz')
stepsn.shape

# %%
prop = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_prop.npz')

# normalize prop by column sums
total_post = prop.sum(axis=0).A1.astype(float)
col_sums_with_inversion = np.reciprocal(
    total_post, where=total_post != 0
)
inprop = prop.multiply(col_sums_with_inversion)
inprop = inprop.astype(np.float32)
# convert to csc
inprop = inprop.tocsc()

inprop.shape

# %%


# %% [markdown]
# ## Get meta

# %%
# this has AN's R7/8
meta_judith = pd.read_csv(DATA_DIR / f'{DATASET}_meta.csv')

# replace na in cell_type with 'bodyId_{bodyId}'
meta_judith['cell_type'] = meta_judith.apply(lambda row: f"bodyId_{int(row['bodyId'])}" if pd.isna(row['cell_type']) else row['cell_type'], axis=1)

# change right to R, left to L in side column
meta_judith['side'] = meta_judith['side'].replace({'right': 'R', 'left': 'L'})

meta_judith['cell_type_side'] = meta_judith.cell_type + '_' + meta_judith.side
        
#  add superclass and class
meta_judith.loc[meta_judith['cell_type'].str.contains('^R7|^R8'), 'superclass'] = 'ol_sensory'
meta_judith.loc[meta_judith['cell_type'].str.contains('^R7|^R8'), 'class'] = 'visual'

print(meta_judith.shape)  

# %%
# LOAD hitting time
ht = pd.read_csv(DATA_DIR / f'{DATASET}_l_flow_{N_FLOW}step_{HIT_THRE}thre_hit.csv')

ht = ht.rename(columns={'hitting_time': 'ht'})
ht = pd.merge(ht, meta_judith[['bodyId', 'idx']], on='idx', how='left') # add bodyId

ht.shape

# %%
# combine
meta = pd.merge(meta_judith, ht[['bodyId', 'ht']], on='bodyId', how='left')


# %%
# make L1 excitatory so it's an ON cell 
meta.loc[meta.cell_type == 'L1', 'sign'] = 1

# %%
# make dictionaries that map indices to meta info
idx_to_idx = dict(zip(meta.idx, meta.idx))
idx_to_bodyId = dict(zip(meta.idx, meta.bodyId))
# combine bodyId and cell_type_side
idx_to_bodyId_cellTypeSide = dict(zip(meta.idx, meta.bodyId.astype(str) + '_' + meta.cell_type_side))
idx_to_bodyId_cellType = dict(zip(meta.idx, meta.bodyId.astype(str) + '_' + meta.cell_type))
idx_to_type_side = dict(zip(meta.idx, meta.cell_type_side))
idx_to_type = dict(zip(meta.idx, meta.cell_type))
idx_to_sign = dict(zip(meta.idx, meta.sign))
idx_to_side = dict(zip(meta.idx, meta.side))
# idx_to_side = dict(zip(meta.idx, meta.soma_side))
idx_to_coords = dict(zip(meta.idx, meta.coords))

type_to_nt = dict(zip(meta.cell_type, meta.nt))
type_side_to_side = dict(zip(meta.cell_type_side, meta.side))
# type_side_to_side = dict(zip(meta.cell_type_side, meta.soma_side))
root_to_type = dict(zip(meta.bodyId, meta.cell_type))
idx_to_root = dict(zip(meta.idx, meta.bodyId))
type_to_sign = {atype:idx_to_sign[idx] for idx, atype in idx_to_type.items()}

bodyId_to_idx = dict(zip(meta.bodyId, meta.idx))

idx_to_modality = dict(zip(meta.idx, meta.superclass))

sign_to_color = {1: '#EE672D', -1: '#1F4695', 0: '#979DA5'}

# %%
# # simplify neurotransmitter types and define colors
# meta['nt'] = meta['nt'].replace(['histamine', 'dopamine', 'octopamine', 'serotonin'], 'other')               
# nt_to_color = {'acetylcholine': '#EE672D', 'glutamate': '#09A64D', 'gaba': '#1F4695', 'other': '#979DA5', 'unclear': '#979DA5'}

# %%
# the cell types one can get hex coordinates for
meta[~meta.coords.isna()].cell_type_side.sort_values().unique()

# %%
np.sqrt(3)/2*3 *2.5**2

# %% [markdown]
# # eff wt

# %% [markdown]
# ## rf by wt prop

# %%
# inidx = meta.idx[meta.cell_type_side.isin(['L1_L', 'L2_L', 'L3_L', 'R7_L', 'R7d_L', 'R8_L', 'R8d_L'])] 
# outidx = meta.idx

# effwt_visl = ci.result_summary(stepsn, inidx, outidx,
#                     inidx_map = idx_to_coords, 
#                     outidx_map = idx_to_bodyId,
#                     display_threshold = 0,
#                     display_output= False
#                     )

# # save effwt_visl to pickle
# effwt_visl.to_pickle(Path(cache_dir, 'effwt_visl.pkl'))

# load effwt_visl from pickle
effwt_visl = pd.read_pickle(Path(cache_dir, 'effwt_visl.pkl'))
print(effwt_visl.shape)

# %%
# del stepsn
# import gc
# gc.collect()

# %% [markdown]
# # CB, visual input contribution, VIC, rf size and avg input rf size

# %% [markdown]
# ## compute 

# %%
# keep left side vpn and BVNC
meta_cb_vpn = meta[(meta['instance'].isin(oltypes_vpn['instance'])) |
                   (~meta['cell_type'].isin(oltypes['cell_type']))
                   ].copy()
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['class'] != 'visual']
meta_cb_vpn.reset_index(drop=True, inplace=True)
meta_cb_vpn.shape, meta_cb_vpn['cell_type_side'].nunique()

# %% [markdown]
# ## elliptical fit

# %%
# SAVE df_fit
# df_fit.to_csv(Path(cache_dir, 'rf_fit_cbvpn_HB_left.csv'), index=False)

# load df_fit
df_fit = pd.read_csv(Path(cache_dir, 'rf_fit_cbvpn_left.csv'))
# df_fit.rename(columns={'size': 'area'}, inplace=True)

# %% [markdown]
# ## main loop

# %%
meta_cb_vpn['col_count'] = np.nan
meta_cb_vpn['col_count_input'] = np.nan
meta_cb_vpn['vision'] = np.nan
meta_cb_vpn['area_fit_input'] = np.nan

# thr_cumsum = 0.4
thr_cumsum = 0.3
factor_area = 1

for i, bodyid in enumerate(meta_cb_vpn['bodyId']):
    values = effwt_visl[str(bodyid)].values
    if values.sum() > 0 and (bodyid in df_fit['bodyId'].values):
        # vision score
        meta_cb_vpn.at[i, 'vision'] = values.sum()

        # col count for target neuron
        col_count, _ = count_col(values, factor_thr=thr_cumsum)
        meta_cb_vpn.at[i, 'col_count'] = col_count / factor_area

        # idx of direct upstream neurons
        idx_prop = inprop[:, bodyId_to_idx[bodyid]].nonzero()[0]

        # input area avg
        idx_keep = pd.DataFrame({'idx': idx_prop})
        idx_keep['bodyId'] = idx_keep['idx'].map(idx_to_bodyId)
        idx_keep['area_fit'] = idx_keep['bodyId'].map(lambda x: df_fit.loc[df_fit['bodyId'] == x, 'area'].values[0] if x in df_fit['bodyId'].values else 0)
        # Flatten sparse column vector to 1D numpy array
        idx_keep['wt'] = inprop[idx_keep['idx'], bodyId_to_idx[bodyid]].toarray().ravel()
        # avoid divide-by-zero if empty
        if idx_keep['wt'].sum() > 0:
            idx_keep['wt_norm'] = idx_keep['wt'] / idx_keep['wt'].sum()
            # handle NaN 
            masked_data = np.ma.masked_array(idx_keep['area_fit'], np.isnan(idx_keep['area_fit']))
            meta_cb_vpn.at[i, 'area_fit_input'] = np.ma.average(masked_data, weights=idx_keep['wt_norm'])

        # inputs' col count, filter by hitting time
        idx_ht = ht.loc[ht['ht'] < ht.loc[ht['bodyId'] == bodyid, 'ht'].values[0], ['idx']]
        # new df with idx, normalized weight, and col_count
        idx_keep = pd.DataFrame({'idx': np.intersect1d(idx_ht['idx'], idx_prop)})
        idx_keep['bodyId'] = idx_keep['idx'].map(idx_to_bodyId)
        idx_keep['col_count'] = idx_keep['bodyId'].map(lambda x: count_col(effwt_visl[str(x)].values, factor_thr=thr_cumsum)[0])
        # Flatten sparse column vector to 1D numpy array
        idx_keep['wt'] = inprop[idx_keep['idx'], bodyId_to_idx[bodyid]].toarray().ravel()
        # avoid divide-by-zero if empty
        if idx_keep['wt'].sum() > 0:
            idx_keep['wt_norm'] = idx_keep['wt'] / idx_keep['wt'].sum()
            # handle NaN in col_count
            masked_data = np.ma.masked_array(idx_keep['col_count'], np.isnan(idx_keep['col_count']))
            meta_cb_vpn.at[i, 'col_count_input'] = np.ma.average(masked_data, weights=idx_keep['wt_norm']) / factor_area


# %%
# add vpn group
meta_cb_vpn = pd.merge(meta_cb_vpn, oltypes_vpn[['cell_type', 'main_groups']], on='cell_type', how='left')
meta_cb_vpn.loc[meta_cb_vpn['main_groups'] != 'VPN', 'main_groups'] = 'BVNC'

# add area and r2
meta_cb_vpn = pd.merge(meta_cb_vpn, df_fit[['bodyId', 'area', 'r2']], on='bodyId', how='left')
meta_cb_vpn.rename(columns={'area': 'area_fit'}, inplace=True)

meta_cb_vpn.shape


# %%
# SAVE
meta_cb_vpn.to_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum30_left.csv'), index=False)

# load
# meta_cb_vpn = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum90_left.csv'), dtype={'bodyId': int, 'col_count': float, 'col_count_input': float})
meta_cb_vpn = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum30_left.csv'), dtype={'bodyId': int})

meta_cb_vpn.shape

# %%
# meta by type
meta_type_cb_vpn = meta_cb_vpn.groupby(['cell_type_side', 'instance']).agg(
    bodyId_count = ('bodyId', 'count'),
    area_fit = ('area_fit', 'median'),
    area_fit_input = ('area_fit_input', 'median'),
    r2 = ('r2', 'median'),
    col_count = ('col_count', 'median'),
    col_count_input = ('col_count_input', 'median'),
    vision = ('vision', 'median'),
    vision_cv = ('vision', lambda x: round(np.std(x) / np.mean(x),2) if np.mean(x) != 0 else np.nan),
    ht = ('ht', 'median'),
    main_groups = ('main_groups', 'first'),
    # ghop = ('ghop', 'first'),
).reset_index()

# %% [markdown]
# ## vision, excl vpn and assign ghop

# %%
thr_inwt = 10 #/1==
thr_vic = 0 #/100

# %%
# load
meta_cb_vpn = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum40_left.csv'), dtype={'bodyId': int})


# %%
# meta by type
meta_type_cb_vpn = meta_cb_vpn.groupby(['cell_type_side', 'instance']).agg(
    bodyId_count = ('bodyId', 'count'),
    vision = ('vision', 'median'),
    vision_cv = ('vision', lambda x: round(np.std(x) / np.mean(x),2) if np.mean(x) != 0 else np.nan),
    ht = ('ht', 'median'),
    main_groups = ('main_groups', 'first')
).reset_index()

# %% [markdown]
# ### exclu vpns by vision < 0.04

# %%
# ax = meta_type_cb_vpn[
#     (meta_type_cb_vpn['vision'] < 0.8) &
#     (meta_type_cb_vpn['main_groups'] == 'VPN')
# ]['vision'].hist(bins=200, grid=False, figsize=(6,4))
# ax.set_xlim(0, 0.1)
# # SAVE
# plt.tight_layout()
# plt.savefig(Path(result_dir, 'meta_cb_vpn_vision_hist_left.png'), dpi=300)
# plt.show()

# %%
# vision > 0.04
inst_t0 = meta_type_cb_vpn[
    (meta_type_cb_vpn['vision'] > thr_vic/100) & # vision > 0.04
    (meta_type_cb_vpn['main_groups'] == 'VPN')]['instance'].values
print(len(inst_t0))

# %%
# SAVE csv
# pd.Series(inst_t0).to_csv(Path(result_dir, 'inst_vpn_thr004.csv'), index=False, header=False)

# %% [markdown]
# ### add VPN hopping groups using neuprint, filter by vpn-vision

# %%
neuron_df, connection_df = fetch_adjacencies(
    sources= inst_t0,
    targets= None,
    rois= rois_cb,
    min_total_weight=1
)
conn_t0t1 = neuprint.merge_neuron_properties(neuron_df, connection_df, ['type', 'instance'])\
    .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
    .agg({'weight': 'sum'})\
    .reset_index()\
    .sort_values(by='weight', ascending=False)

# %%
# get instance_post but remove those with apostophe or ? or +, and exclude vpn
conn_t0t1 = conn_t0t1[~conn_t0t1['instance_post'].str.contains("\\'") & 
                   ~conn_t0t1['instance_post'].str.contains("\\?") &
                   ~conn_t0t1['instance_post'].str.contains("\\+") &
                   ~conn_t0t1['instance_post'].str.contains("unclear") &
                    ~conn_t0t1['instance_post'].isin(oltypes_vpn['instance'])]
inst = conn_t0t1['instance_post'].unique()
len(inst)

# %%
# sum up inputs
neuron_df, connection_df = fetch_adjacencies(sources= None, targets= inst, min_total_weight=1)

idwt_t1 = neuprint.merge_neuron_properties(
    neuron_df, connection_df, ['type', 'instance']
    ).groupby(['bodyId_post', 'instance_post']).agg({'weight': 'sum'}).reset_index()

# %%
df = pd.merge(
    conn_t0t1.groupby(['bodyId_post','instance_post']).agg(conn_wt=('weight', 'sum')).reset_index(),
    idwt_t1[['bodyId_post','weight']], on='bodyId_post', how='left'
    )
df['inwt'] = df['conn_wt'] / df['weight']
df = pd.merge(df, df_fit[['bodyId', 'r2']], left_on='bodyId_post', right_on='bodyId', how='left')

instwt_t1 = df.groupby(['instance_post']).agg(
    bodyId_count = ('bodyId_post', 'count'),
    inwt_median = ('inwt', 'median'),
    r2_median = ('r2', 'median')
).reset_index()

# %%
# SAVE pickle
instwt_t1.to_pickle(Path(cache_dir, f'instwt_t1_{thr_vic}_left.pkl'))

# load pickle
instwt_t1 = pd.read_pickle(Path(cache_dir, f'instwt_t1_{thr_vic}_left.pkl'))

# %%
ax = df['inwt'].hist(bins=100, figsize=(4,2))
ax.set_xlim(0, 0.1)

# %%
# filter
inst_t1 = instwt_t1.loc[instwt_t1['inwt_median'] >= thr_inwt/100, 'instance_post'].unique()
len(inst_t1)

# %%
T1_info, _ = fetch_neurons(NC(instance= inst_t1))
T1_info = T1_info[['bodyId', 'type', 'instance', 'consensusNt', 'flywireType', 'somaSide', 'downstream', 'upstream']]
print(T1_info.shape, T1_info['instance'].nunique())


# %%
neuron_df, connection_df = fetch_adjacencies(
    # sources= T1_info['bodyId'].values,
    sources= np.concatenate([inst_t0, inst_t1]),
    targets= None,
    rois= rois_cb,
    min_total_weight=1
)
conn_t0t1t2 = neuprint.merge_neuron_properties(neuron_df, connection_df, ['type', 'instance'])\
    .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
    .agg({'weight': 'sum'})\
    .reset_index()\
    .sort_values(by='weight', ascending=False)

# %%
# get instance_post but remove those with apostophe or ? or +, and exclude vpn and T1
conn_t0t1t2 = conn_t0t1t2[~conn_t0t1t2['instance_post'].str.contains("\\'") & 
                    ~conn_t0t1t2['instance_post'].str.contains("\\?") &
                    ~conn_t0t1t2['instance_post'].str.contains("\\+") &
                    ~conn_t0t1t2['instance_post'].str.contains("unclear") &
                    ~conn_t0t1t2['instance_post'].isin(oltypes_vpn['instance']) &
                    ~conn_t0t1t2['instance_post'].isin(inst_t1)]
inst = conn_t0t1t2['instance_post'].unique()
len(inst)

# %%
# sum up inputs
neuron_df, connection_df = fetch_adjacencies(sources= None, targets= inst, min_total_weight=1)

idwt = neuprint.merge_neuron_properties(
    neuron_df, connection_df, ['type', 'instance']
    ).groupby(['bodyId_post', 'instance_post']).agg({'weight': 'sum'}).reset_index()

# %%
df = pd.merge(
    conn_t0t1t2.groupby(['bodyId_post','instance_post']).agg(conn_wt=('weight', 'sum')).reset_index(),
    idwt[['bodyId_post','weight']], on='bodyId_post', how='left'
    )
df['inwt'] = df['conn_wt'] / df['weight']
df_inst = df.groupby(['instance_post']).agg(
    bodyId_count = ('bodyId_post', 'count'),
    inwt_median = ('inwt', 'median')
).reset_index()

# %%
ax = df['inwt'].hist(bins=100, figsize=(4,2))
ax.set_xlim(0, 0.1)

# %%
# filter, 10% or 1%
inst_t2 = df_inst.loc[df_inst['inwt_median'] >= thr_inwt/100, 'instance_post'].unique()
len(inst_t2)

# %%
T2_info, _ = fetch_neurons(NC(instance= inst_t2))
T2_info = T2_info[['bodyId', 'type', 'instance', 'consensusNt', 'flywireType', 'somaSide', 'downstream', 'upstream']]
print(T2_info.shape, T2_info['instance'].nunique())

# %%
# meta_cb_vpn.rename(columns={'main groups': 'main_groups'}, inplace=True)
# meta_type_cb_vpn.rename(columns={'main groups': 'main_groups'}, inplace=True)

# %% [markdown]
# ### define ghop

# %%
# add hopping groups
meta_cb_vpn['ghop'] = meta_cb_vpn['main_groups'].map({'BVNC': 'later', 'VPN': 'VPN'})
meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t0), 'ghop'] = 'T0'
meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t1), 'ghop'] = 'T1'
meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t2), 'ghop'] = 'T2'

meta_type_cb_vpn['ghop'] = meta_type_cb_vpn['main_groups'].map({'BVNC': 'later', 'VPN': 'VPN'})
meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t0), 'ghop'] = 'T0'
meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t1), 'ghop'] = 'T1'
meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t2), 'ghop'] = 'T2'

# %%
# SAVE
meta_cb_vpn.to_pickle(Path(cache_dir, f'meta_cb_vpn_cumsum40_ghop_vic_{thr_vic}{thr_inwt}{thr_inwt}_left.pkl'))
meta_type_cb_vpn.to_pickle(Path(cache_dir, f'meta_type_cb_vpn_cumsum40_ghop_vic_{thr_vic}{thr_inwt}{thr_inwt}_left.pkl'))

# load
# meta_cb_vpn = pd.read_pickle(Path(cache_dir, f'meta_cb_vpn_cumsum40_ghop_vic_{thr_vic}{thr_inwt}{thr_inwt}_left.pkl'))
# meta_type_cb_vpn = pd.read_pickle(Path(cache_dir, f'meta_type_cb_vpn_cumsum40_ghop_vic_{thr_vic}{thr_inwt}{thr_inwt}_left.pkl'))

# %% [markdown]
# ## resolution, excl vpn and assign ghop

# %%
# load
meta_cb_vpn = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum40_left.csv'), dtype={'bodyId': int})

# %%
# meta by type
meta_type_cb_vpn = meta_cb_vpn.groupby(['cell_type_side', 'instance']).agg(
    bodyId_count = ('bodyId', 'count'),
    area_fit = ('area_fit', 'median'),
    area_fit_input = ('area_fit_input', 'median'),
    r2 = ('r2', 'median'),
    col_count = ('col_count', 'median'),
    col_count_input = ('col_count_input', 'median'),
    vision = ('vision', 'median'),
    vision_cv = ('vision', lambda x: round(np.std(x) / np.mean(x),2) if np.mean(x) != 0 else np.nan),
    ht = ('ht', 'median'),
    main_groups = ('main_groups', 'first'),
    # ghop = ('ghop', 'first'),
).reset_index()

# %% [markdown]
# ### exclu vpns by vision < 0.04, r2 > 0, bodyId_count > 1

# %%
# vision > 0.04
inst_t0 = meta_type_cb_vpn[
    (meta_type_cb_vpn['vision'] > thr_vic/100) &
    (meta_type_cb_vpn['main_groups'] == 'VPN') &
    # (meta_type_cb_vpn['bodyId_count'] >= 2) &
    (meta_type_cb_vpn['r2'] >= 0)]['instance'].values
print(len(inst_t0))


# %% [markdown]
# ### add VPN hopping groups using neuprint, filter by vpn-vision

# %%
neuron_df, connection_df = fetch_adjacencies(
    sources= inst_t0,
    targets= None,
    rois= rois_cb,
    min_total_weight=1
)
conn_t0t1 = neuprint.merge_neuron_properties(neuron_df, connection_df, ['type', 'instance'])\
    .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
    .agg({'weight': 'sum'})\
    .reset_index()\
    .sort_values(by='weight', ascending=False)

# %%
# get instance_post but remove those with apostophe or ? or +, and exclude vpn
conn_t0t1 = conn_t0t1[~conn_t0t1['instance_post'].str.contains("\\'") & 
                   ~conn_t0t1['instance_post'].str.contains("\\?") &
                   ~conn_t0t1['instance_post'].str.contains("\\+") &
                   ~conn_t0t1['instance_post'].str.contains("unclear") &
                    ~conn_t0t1['instance_post'].isin(oltypes_vpn['instance'])]
inst = conn_t0t1['instance_post'].unique()
len(inst)

# %%
# sum up inputs
neuron_df, connection_df = fetch_adjacencies(sources= None, targets= inst, min_total_weight=1)

idwt = neuprint.merge_neuron_properties(
    neuron_df, connection_df, ['type', 'instance']
    ).groupby(['bodyId_post', 'instance_post']).agg({'weight': 'sum'}).reset_index()

# %%
df = pd.merge(
    conn_t0t1.groupby(['bodyId_post','instance_post']).agg(conn_wt=('weight', 'sum')).reset_index(),
    idwt[['bodyId_post','weight']], on='bodyId_post', how='left'
    )

df = pd.merge(df, df_fit[['bodyId', 'r2']], left_on='bodyId_post', right_on='bodyId', how='left')
df['inwt'] = df['conn_wt'] / df['weight']
df_inst = df.groupby(['instance_post']).agg(
    bodyId_count = ('bodyId_post', 'count'),
    inwt_median = ('inwt', 'median'),
    r2 = ('r2', 'median')
).reset_index()

# %%
ax = df['inwt'].hist(bins=100, figsize=(4,2))
ax.set_xlim(0, 0.1)

# %%
# filter
inst_t1 = df_inst.loc[(df_inst['inwt_median'] >= thr_inwt/100) 
                      & (df_inst['r2'] >= 0)
                      , 'instance_post'].unique()
len(inst_t1)

# %%
T1_info, _ = fetch_neurons(NC(instance= inst_t1))
T1_info = T1_info[['bodyId', 'type', 'instance', 'consensusNt', 'flywireType', 'somaSide', 'downstream', 'upstream']]
print(T1_info.shape, T1_info['instance'].nunique())


# %%
neuron_df, connection_df = fetch_adjacencies(
    sources= np.concatenate([inst_t0, inst_t1]),
    targets= None,
    rois= rois_cb,
    min_total_weight=1
)
conn_t0t1t2 = neuprint.merge_neuron_properties(neuron_df, connection_df, ['type', 'instance'])\
    .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
    .agg({'weight': 'sum'})\
    .reset_index()\
    .sort_values(by='weight', ascending=False)

# %%
# get instance_post but remove those with apostophe or ? or +, and exclude vpn and T1
conn_t0t1t2 = conn_t0t1t2[~conn_t0t1t2['instance_post'].str.contains("\\'") & 
                    ~conn_t0t1t2['instance_post'].str.contains("\\?") &
                    ~conn_t0t1t2['instance_post'].str.contains("\\+") &
                    ~conn_t0t1t2['instance_post'].str.contains("unclear") &
                    ~conn_t0t1t2['instance_post'].isin(oltypes_vpn['instance']) &
                    ~conn_t0t1t2['instance_post'].isin(inst_t1)]
inst = conn_t0t1t2['instance_post'].unique()
len(inst)

# %%
# sum up inputs
neuron_df, connection_df = fetch_adjacencies(sources= None, targets= inst, min_total_weight=1)

idwt = neuprint.merge_neuron_properties(
    neuron_df, connection_df, ['type', 'instance']
    ).groupby(['bodyId_post', 'instance_post']).agg({'weight': 'sum'}).reset_index()

# %%
df = pd.merge(
    conn_t0t1t2.groupby(['bodyId_post','instance_post']).agg(conn_wt=('weight', 'sum')).reset_index(),
    idwt[['bodyId_post','weight']], on='bodyId_post', how='left'
    )
df = pd.merge(df, df_fit[['bodyId', 'r2']], left_on='bodyId_post', right_on='bodyId', how='left')
df['inwt'] = df['conn_wt'] / df['weight']
df_inst = df.groupby(['instance_post']).agg(
    bodyId_count = ('bodyId_post', 'count'),
    inwt_median = ('inwt', 'median'),
    r2 = ('r2', 'median')
).reset_index()

# %%
ax = df['inwt'].hist(bins=100, figsize=(4,2))
ax.set_xlim(0, 0.1)

# %%
# filter
inst_t2 = df_inst.loc[(df_inst['inwt_median'] >= thr_inwt/100) 
                      & (df_inst['r2'] >= 0)
                      , 'instance_post'].unique()
len(inst_t2)

# %%
T2_info, _ = fetch_neurons(NC(instance= inst_t2))
T2_info = T2_info[['bodyId', 'type', 'instance', 'consensusNt', 'flywireType', 'somaSide', 'downstream', 'upstream']]
print(T2_info.shape, T2_info['instance'].nunique())

# %% [markdown]
# ### define ghop

# %%
# add hopping groups
meta_cb_vpn['ghop'] = meta_cb_vpn['main_groups'].map({'BVNC': 'later', 'VPN': 'VPN'})
meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t0), 'ghop'] = 'T0'
meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t1), 'ghop'] = 'T1'
meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t2), 'ghop'] = 'T2'

meta_type_cb_vpn['ghop'] = meta_type_cb_vpn['main_groups'].map({'BVNC': 'later', 'VPN': 'VPN'})
meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t0), 'ghop'] = 'T0'
meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t1), 'ghop'] = 'T1'
meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t2), 'ghop'] = 'T2'

# %%
# SAVE
meta_cb_vpn.to_pickle(Path(cache_dir, f'meta_cb_vpn_cumsum40_ghop_res_{thr_vic}{thr_inwt}{thr_inwt}_left.pkl'))
meta_type_cb_vpn.to_pickle(Path(cache_dir, f'meta_type_cb_vpn_cumsum40_ghop_res_{thr_vic}{thr_inwt}{thr_inwt}_left.pkl'))

# # load
# meta_cb_vpn = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_cumsum40_ghop_res_000101_left.pkl'))
# meta_type_cb_vpn = pd.read_pickle(Path(cache_dir, 'meta_type_cb_vpn_cumsum40_ghop_res_000101_left.pkl'))

# %% [markdown]
# ### query psd

# %%
# syn = neuprint.fetch_synapses(NC(instance = np.concatenate((inst_t0, inst_t1, inst_t2))), 
#                               SC(rois=rois_cb, primary_only=True, type='post'))

# # SAVE pickle
# syn.to_pickle(Path(cache_dir, 'meta_cb_vpn_psd_for_ratio.pkl'))

# load
# syn = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_psd_for_ratio.pkl'))

# %% [markdown]
# # Get pre-syn

# %%
# syn = neuprint.fetch_synapses(NC(bodyId = meta_cb_vpn['bodyId']), SC(rois=rois_cb, primary_only=True, type='pre'))

# # SAVE
# syn.to_pickle(Path(cache_dir, 'meta_cb_vpn_tbar_left.pkl'))

# load
# syn = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_tbar_left.pkl'))

# %% [markdown]
# # rf

# %% [markdown]
# ## conn prop

# %%
outidx

# %%
# DEBUG
inidx = meta.idx[meta.cell_type.isin(['L1', 'L2', 'L3', 'R7', 'R7d', 'R8', 'R8d'])] 

outidx = meta.idx[meta.instance.str.contains('^TuBu04_IBU_R', regex=True)] 

df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

# %%
# ci.plot_mollweide_projection(df)
fig = ci.hex_heatmap(df)
fig
# save
# fig.write_image(Path(result_dir, 'hex_map.pdf'))

# %%
inidx = meta.idx[meta.cell_type.isin(['L1', 'L2', 'L3', 'R7', 'R7d', 'R8', 'R8d'])] 
# outidx = meta.idx[meta.cell_type_side.str.contains('^LC6.*_right', regex=True)] 
# outidx = meta.idx[meta.cell_type_side.str.contains('^PVLP008_a.*_right', regex=True)] 
outidx = meta.idx[meta.cell_type_side.str.contains('^AMMC007.*_R', regex=True)] 
# outidx = meta.idx[meta.cell_type_side.str.contains('^ER4d_right', regex=True)] 

# %%
paths = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 3)
paths = ci.group_paths(paths, idx_to_type_side, idx_to_type)

# %%
idx_inter = meta.idx[meta.cell_type_side.str.contains('^LC6_right')].values

# %%
# paths_filtered = ci.filter_paths(paths, threshold=0.012, necessary_intermediate={3: ['LC6_right']})
paths_filtered = ci.filter_paths(paths, threshold=0.002)
ci.plot_layered_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(11,6))

# %% [markdown]
# # End

# %%



