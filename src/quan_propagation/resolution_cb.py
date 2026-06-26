# %%
%load_ext autoreload
%autoreload 2

# %% [markdown]
# ### ?obs. spatial info in CB
# 
# Using VPNs to access the retinotopy at the input side in OL and at the output side in CB. Assign Retinotopy Index to cell type's outgoing synapses as a way to characterize the amount of retinotopy in regions of CB. Why some regions have higher RIs?
#  
# Slightly weaker version of this is spatial resolution in terms of RF's size or column counts. High vs low res neurons/regions. 
# 
# compute rf size for all CB types, and average rf size of inputs (only earlier hitting time).

# %%
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
import torch

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
import json
import pickle
import scipy

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import matplotlib as mpl
import matplotlib.pyplot as plt

import neuprint
from neuprint import fetch_neurons, fetch_adjacencies, NeuronCriteria as NC, SynapseCriteria as SC

# import navis
# import navis.interfaces.neuprint as neu

# import connectome_interpreter as ci

# # from fafbseg import flywire
# # import flybrains

# %%
from quan_propagation.func import count_col, count_col_ahull, alpha_shape_2d

# # Test if count_col is imported
# print(count_col)

# %%
from utils.config import (
    CACHE_DIR, DATA_DIR, DATASET, FIG_DIR, HIT_THRE, N_FLOW, PARAMS_DIR, SIDE_CHAR,
)
DATA_DIR.mkdir(parents=True, exist_ok=True)

result_dir = FIG_DIR / 'quan_propagation'
result_dir.mkdir(parents=True, exist_ok=True)

cache_dir = CACHE_DIR / 'quan_propagation'
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Some setups

# %% [markdown]
# ## vpn types, rois

# %%
# cell type table xlsx
oltypes = pd.read_excel(PARAMS_DIR / 'Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx')
oltypes.rename(columns={'cell type':'cell_type', 'main groups':'main_groups'}, inplace=True)
print(oltypes.shape)

oltypes_nonvpn = oltypes[~oltypes['main_groups'].str.contains('VPN')]
oltypes_vpn = oltypes[oltypes['main_groups'].str.contains('VPN')]
oltypes_vcn = oltypes[oltypes['main_groups'].str.contains('VCN')]
oltypes_ol = oltypes[oltypes['main_groups'].str.contains('^ON')]

# %% [markdown]
# ## rois

# %%
# neuprint.fetch_primary_rois()
# rois_dict = neuprint.queries.fetch_roi_hierarchy(include_subprimary=False)
# rois_dict['CNS'].keys()

# %%
from utils.query_roi import get_primary_rois
rois_cb = get_primary_rois('CentralBrain')
len(rois_cb)

# %% [markdown]
# ## vpn sub-group

# %%
# # ^MeVP, ^aMe, ^MeTu
# # ^LoVP, ^LT, ^LoVP, ^LC
# # ^LPC, ^LPT, VS\VST\VSm\HS\Nod\dCal\vCal
# # ^LLPC, ^LPLC
# #  s-LNv, l-LNv, etc

# # choose starting vpn
# # vpn_seed = oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^MeTu')]
# # vpn_seed = oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LC')]

# # 
# # vpn_seed = oltypes_vpn[~oltypes_vpn['cell type'].str.contains('MeTu|LC')]

# print(
#     'LC', oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LC')].shape,
#     'LPC', oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LPC')].shape,
#     'MeVP', oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^MeVP')].shape,
#     'LoVP', oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LoVP')].shape,
#     'MeTu', oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^MeTu')].shape,
#     'aMe', oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^aMe')].shape,
#     'LT', oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LT')].shape,
#     'LPT', oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LPT')].shape,
#     'LPLC', oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LPLC')].shape,
#     'LLPC', oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LLPC')].shape,
# )

# # oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LPT')].sort_values(by='no. of cells', ascending=False)
# # oltypes_vpn[~oltypes_vpn['cell type'].str.contains(r'^(LC|LPC|MeVP|LoVP|MeTu|aMe|LT|LPT|LPLC|LLPC)')].sort_values(by='no. of cells', ascending=False)
# # oltypes_vpn[oltypes_vpn['no. of cells'] >= 50]
# # LC, LLPC1/2/3, LPC1/2, LPLC1/2, MeTu1, MeTu3c, MeVP1 

# ax = oltypes_vpn['no. of cells'].plot(kind='hist', range=(0, 50), bins=50)
# ax.set_ylim(0, 50)
# fig = ax.get_figure()
# fig.set_size_inches(6, 2)

# # oltypes_vpn['no. of cells'].argmax()

# %% [markdown]
# ## load and sum up prop matrices

# %%
# # temperary, for testing
# DATA_DIR =  Path(PROJECT_ROOT, 'cache', 'data', 'fromJudith')
# DATASET = 'malecns_v0.9'

# %%
# prop = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_prop.npz') # whole conn matrix, abs wt

# # normalize prop by column sums
# total_post = prop.sum(axis=0).A1.astype(float)
# col_sums_with_inversion = np.reciprocal(
#     total_post, where=total_post != 0
# )
# inprop = prop.multiply(col_sums_with_inversion)
# inprop = inprop.astype(np.float32)
# # convert to csc
# inprop = inprop.tocsc()

# # inprop = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_{SIDE_CHAR}_lat_flow_0.npz') # selected side, lateral, normalized

# inprop.shape

# %%
# inprop_other = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_{"l" if SIDE_CHAR == "r" else "r"}_lat_flow_0.npz')

# %%
# stepsn = scipy.sparse.load_npz(Path(DATA_DIR, 'malecns_v09_lat', f'{DATASET}_{SIDE_CHAR}_lat_flow_sum.npz'))
stepsn = scipy.sparse.load_npz(Path(DATA_DIR, f'{DATASET}_{SIDE_CHAR}_lat_flow_sum.npz'))
stepsn.shape

# %%
# # what's the range of non-zero values in stepsn
# stepsn[stepsn != 0].min()

# %%
# # find the rows and columns that contain the max value
# max_val = stepsn.max()

# # Convert sparse matrix to COO format for efficient row/col extraction
# # coo = steps_cpu[j].tocoo()
# coo = stepsn.tocoo()
# # Find indices where data equals max_val
# max_indices = np.where(coo.data == max_val)[0]
# row_indices = coo.row[max_indices]
# col_indices = coo.col[max_indices]
# max_val, np.column_stack([row_indices, col_indices])

# %% [markdown]
# ## Get meta

# %%
# LOAD Judith meta, this has AN's R7/8, 
meta = pd.read_csv(DATA_DIR / f'{DATASET}_meta.csv')

# replace na in cell_type with 'bodyId_{bodyId}'
meta['cell_type'] = meta.apply(lambda row: f"bodyId_{int(row['bodyId'])}" if pd.isna(row['cell_type']) else row['cell_type'], axis=1)

# # change right to R, left to L in side column
# meta['side'] = meta['side'].replace({'right': 'R', 'left': 'L'})

meta['cell_type_side'] = meta.cell_type + '_' + meta.side
        
#  add superclass and class
meta.loc[meta['cell_type'].str.contains('^R7|^R8'), 'superclass'] = 'ol_sensory'
meta.loc[meta['cell_type'].str.contains('^R7|^R8'), 'class'] = 'visual'

print(meta.shape)  

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
# cell type table xlsx
oltypes = pd.read_excel(PARAMS_DIR / 'Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx')
oltypes.rename(columns={'cell type':'cell_type', 'main groups':'main_groups'}, inplace=True)
print(oltypes.shape)

# %%
# DEBUG
aa = meta.loc[meta['cell_type'].str.contains('^R7|^R8')]
print(aa[~aa.superclass.isin(['ol_sensory'])])
print(aa[~aa['class'].isin(['visual'])])

# %% [markdown]
# ### right and left meta with ht and vic

# %%
# load
vp_cb_r = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_r.p'))
vp_cb_l = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_l.p'))

# %% [markdown]
# ### ? obs. bodyids

# %%
# LOAD Judith meta, this has AN's R7/8, 
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
# LOAD, hitting time
ht = pd.read_csv(DATA_DIR / f'{DATASET}_{SIDE_CHAR}_flow_{N_FLOW}step_{HIT_THRE}thre_hit.csv')

ht = ht.rename(columns={'cell_group': 'instance', 'hitting_time': 'ht'}) 
# ht = pd.merge(ht, meta_judith[['bodyId', 'idx']], on='idx', how='left') # add bodyId

ht.shape

# %%
# # DEBUG, Dm9 is a major input to aMe12
# ht[ht['instance'].isin(['aMe12_R', 'Dm9_R']) ].groupby(['instance']).agg({'hitting_time': 'median'})
# # ht[ht['instance'].isin(['MeVP15_R', 'Mi15_R']) ]
# # ht[ht['instance'].isin(['MeTu3c_R', 'Mi15_R']) ].groupby(['instance']).agg({'hitting_time': 'median'})

# %%
# combine
# meta = pd.merge(meta_judith, ht[['bodyId', 'ht']], on='bodyId', how='left')
meta = pd.merge(meta_judith, ht[['idx', 'ht']], on='idx', how='left')

# %%
# # simplify neurotransmitter types and define colors
# meta['nt'] = meta['nt'].replace(['histamine', 'dopamine', 'octopamine', 'serotonin'], 'other')               
# nt_to_color = {'acetylcholine': '#EE672D', 'glutamate': '#09A64D', 'gaba': '#1F4695', 'other': '#979DA5', 'unclear': '#979DA5'}

# %%
# # DEBUG
# # filter for rows whose cell_type contain R7, R7p, R7y, or R7d and side =='right' and then check for duplicated coords among these rows
# r7_cells = meta[(meta.cell_type.str.contains('^L1')) & (~meta.cell_type.str.contains('unclear')) & (meta.side == 'L')][['bodyId', 'coords', 'instance']]
# aa = r7_cells[r7_cells.duplicated('coords', keep=False)].sort_values(by='coords')

# %% [markdown]
# ### ? obs. type

# %%
# vpn+vcbn, from Judith, 
isleft = ''
# isleft = '_left'

cache_data_dir = Path(PROJECT_ROOT, 'cache', 'data')
# vp_cb_vic = pd.read_pickle(Path(cache_data_dir, 'fromJudith', 'malecns_v09_lat', f'vp_cb{isleft}_vic_w_hit_df.p'))
print(vp_cb_vic.shape)

# filter by VIC
thr_vic = 5e-4
inst = vp_cb_vic.groupby('instance').agg({'VIC':'median'}).reset_index()
inst = inst[inst.VIC > thr_vic]['instance'].unique()
vp_cb_vic = vp_cb_vic[vp_cb_vic.instance.isin(inst)]
print(vp_cb_vic.shape)

oltypes0 = pd.read_excel(Path(PROJECT_ROOT, 'params', 'Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx'))
print(oltypes0.shape)


meta_nonol = vp_cb_vic[~vp_cb_vic['instance'].isin(oltypes0['instance'])]
print(meta_nonol.shape)

meta_ol = meta[meta['instance'].isin(oltypes0['instance'])]
print(meta_ol.shape)
meta_ol = meta_ol[~meta_ol['instance'].isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R7d_R', 'R8_R', 'R8d_R', 'HBeyelet_R'])]
print(meta_ol.shape)

# fit_rf = pd.read_pickle(Path(cache_data_dir, 'fromJudith', 'malecns_v09_lat', f'malecns_v0.9_all_fit_rf.p'))  # only olr ??
# fit_rf.shape

# %%


# %% [markdown]
# # eff wt

# %% [markdown]
# ## eff wt from VPNs

# %%
# inidx = meta.idx[meta.instance.isin(oltypes_vpn['instance'])] 
# outidx = meta.idx[~meta.cell_type.isin(oltypes_nonvpn['cell type'])] 

# effwt_vpnr = ci.result_summary(stepsn, inidx, outidx,
#                     inidx_map = idx_to_bodyId, 
#                     outidx_map = idx_to_bodyId,
#                     display_threshold = 0,
#                     display_output= False
#                     )

# # save effwt_vpnr to pickle
# effwt_vpnr.to_pickle(Path(cache_dir, 'effwt_vpnr.pkl'))

# load effwt_vpnr from pickle
# effwt_vpnr = pd.read_pickle(Path(cache_dir, 'effwt_vpnr.pkl'))
print(effwt_vpnr.shape)

# %% [markdown]
# ## eff by from visual inputs

# %%
# meta[meta.cell_type.isin(['L1', 'L2', 'L3', 'R7', 'R7d', 'R8', 'R8d'])]['cell_type_side'].value_counts()
# meta[meta.cell_type.isin(['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d'])]['cell_type_side'].value_counts()

# %%
# inidx = meta.idx[meta.cell_type_side.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R7d_R', 'R8_R', 'R8d_R'])] 
# # inidx = meta.idx[meta.cell_type_side.isin(['T4a_R', 'T4b_R', 'T4c_R', 'T4d_R', 'T5a_R', 'T5b_R', 'T5c_R', 'T5d_R'])] 
# # outidx = meta.idx[~meta.cell_type.isin(['L1', 'L2', 'L3', 'R7', 'R7d', 'R8', 'R8d'])] 
# outidx = meta.idx

# effwt = ci.result_summary(stepsn, inidx, outidx,
#                     inidx_map = idx_to_coords, 
#                     outidx_map = idx_to_bodyId,
#                     display_threshold = 0,
#                     display_output= False
#                     )
# # effwt_visr.columns = effwt_visr.columns.astype(float).astype(int)
# effwt = effwt[effwt.index != 'nan']

# save effwt_visr to pickle
# effwt_visr.to_pickle(Path(cache_dir, 'effwt_visr.pkl'))
# effwt.to_pickle(Path(cache_dir, 'effwt_t4t5.pkl'))

# load effwt_visr from pickle
effwt_visr = pd.read_pickle(Path(cache_dir, 'effwt_visr.pkl'))
# effwt_t4t5 = pd.read_pickle(Path(cache_dir, 'effwt_t4t5.pkl'))
print(effwt_visr.shape)

# %%
vi_ls = ['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R7d_R', 'R8_R', 'R8d_R', 'HBeyelet_R']
# vi_ls = ['HBeyelet_R']
outidx = meta.idx
for vi in vi_ls:
    inidx = meta.idx[meta.cell_type_side.isin([vi])] 
    effwt = ci.result_summary(stepsn, inidx, outidx,
                        inidx_map = idx_to_bodyId, 
                        outidx_map = idx_to_bodyId,
                        display_threshold = 0,
                        display_output= False
                        )
    effwt = effwt[effwt.index != 'nan'] # for HBeyelet_R which doesn't have coords

    # save 
    effwt.to_pickle(Path(cache_dir, f'effwt_visr_{vi}.pkl'))

# %%
del stepsn
import gc
gc.collect()

# %% [markdown]
# # CB, VIC, rf size and avg input rf size

# %% [markdown]
# ## meta

# %%
# keep right side vpn and BVNC
meta_cb_vpn = meta[(meta['instance'].isin(oltypes_vpn['instance'])) |
                   (~meta['cell_type'].isin(oltypes['cell_type']))
                   ].copy()
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['class'] != 'visual']
meta_cb_vpn.reset_index(drop=True, inplace=True)
meta_cb_vpn.shape, meta_cb_vpn['cell_type_side'].nunique()

# %%
# # which rows missing instance?
# meta_cb_vpn[meta_cb_vpn['instance'].isnull()]

# %%
# # NB, superclass vs ol table
# meta_cb_vpn[meta_cb_vpn['cell_type'].isin(oltypes['cell_type'])]['superclass'].value_counts()

# %% [markdown]
# ## Elliptical fit

# %%
from utils.ol_rf import _compute_rf_params, _clean_rf_params

# %%
all_instances = pd.unique(
    np.concatenate([
        meta_ol['instance'].unique(),
        meta_nonol['instance'].unique()
    ])
)

len(all_instances)

# %% [markdown]
# ### all cells with VIC > thr

# %%
rf_fit = _compute_rf_params(
    meta, stepsn, 
    # in_instances=['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R7d_R', 'R8_R', 'R8d_R', 'HBeyelet_R'], 
    in_instances=['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'], 
    out_instances = all_instances.tolist(),
    cumsum_thre = None,
)
rf_fit = _clean_rf_params(rf_fit)

# df_fit.rename(columns={'size': 'area'}, inplace=True)

print(rf_fit.shape)

# save
rf_fit.to_pickle(Path(cache_dir, 'rf_fit_thr10.pkl'))

# load
# rf_fit = pd.read_pickle(Path(cache_dir, 'rf_fit_thr10.pkl'))

# %%
rf_fit = _compute_rf_params(
    meta, stepsn, 
    # in_instances=['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R7d_R', 'R8_R', 'R8d_R', 'HBeyelet_R'], 
    in_instances=['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'], 
    out_instances = all_instances.tolist(),
    cumsum_thre = 0.7,
)
rf_fit = _clean_rf_params(rf_fit)

# df_fit.rename(columns={'size': 'area'}, inplace=True)

print(rf_fit.shape)

# save
rf_fit.to_pickle(Path(cache_dir, 'rf_fit_thr07.pkl'))

# load
# rf_fit = pd.read_pickle(Path(cache_dir, 'rf_fit_thr10.pkl'))

# %%
rf_fit = pd.read_pickle(Path(cache_dir, 'rf_fit_thr10.pkl'))
rf_fit_2 = pd.read_pickle(Path(cache_dir, 'rf_fit_thr07.pkl'))

# %%
# compare cumsum_thre
df = pd.merge(rf_fit[['instance','bodyId', 'size']], rf_fit_2[['bodyId','size']], on='bodyId', how='left', suffixes=('_thr10', '_thr07'))
# groupby instance and compute the median of ahull and size
df = df.groupby('instance').agg({'size_thr10':'median', 'size_thr07':'median'}).reset_index()
plt.figure(figsize=(6,6))
plt.scatter(df['size_thr10'], df['size_thr07'])
plt.plot([0, df['size_thr10'].max()], [0, df['size_thr10'].max()], 'r--')  # add y=x line for reference
plt.xlim(0, df['size_thr10'].max()*1.1)
plt.ylim(0, df['size_thr07'].max()*1.1)
plt.gca().set_aspect('equal')
plt.show()

# %% [markdown]
# ### obs. OL cells

# %%
rf_fit = _compute_rf_params(
    meta, stepsn, 
    # in_instances=['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R7d_R', 'R8_R', 'R8d_R', 'HBeyelet_R'], 
    in_instances=['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'], 
    out_instances = meta_cb_vpn['instance'].unique().tolist(),
)
rf_fit = _clean_rf_params(rf_fit)

# df_fit.rename(columns={'size': 'area'}, inplace=True)

rf_fit.shape

# %%
# elliptical fit
# rf_fit = pd.read_pickle(CACHE_DIR / f'{DATASET}_all_fit_rf.p')

rf_fit['area'] = rf_fit['a'] * rf_fit['b'] * np.pi / (2/np.sqrt(3))  # area of ellipse in units of hex area
rf_fit.shape

# %%
# save
# rf_fit.to_pickle(Path(cache_dir, f'{DATASET}_rf_ellipse.pkl'))

# load
# rf_fit = pd.read_pickle(Path(cache_dir, f'{DATASET}_rf_ellipse.pkl'))

# %%
# rf_fit.head()
cache_dir, DATASET

# %% [markdown]
# ### obs. CB cells

# %%
# from computing_functions import compute_rf_params, compute_clean_rf_params

# df_fit = compute_rf_params(
#     meta_judith, stepsn, 
#     in_instances=['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R7d_R', 'R8_R', 'R8d_R', 'HBeyelet_R'], 
#     out_instances = meta_cb_vpn['instance'].unique().tolist(),
# )
# df_fit = compute_clean_rf_params(df_fit)

# df_fit.rename(columns={'size': 'area'}, inplace=True)

# %%
# SAVE df_fit
# df_fit.to_csv(Path(cache_dir, 'rf_fit_cbvpn_HB_right.csv'), index=False)

# load df_fit
df_fit = pd.read_csv(Path(cache_dir, 'rf_fit_cbvpn_right.csv'))
# df_fit.rename(columns={'size': 'area'}, inplace=True)

# %%
# df_fit['area'] = df_fit['a'] * df_fit['b'] * np.pi / (2/np.sqrt(3)) # area of ellipse in units of hex area

# %%
# combine all rows in rf_fit and df_fit, remove duplicated rows
df_fit = pd.concat([rf_fit, df_fit], axis=0).drop_duplicates('bodyId').reset_index(drop=True)
df_fit.shape

# %% [markdown]
# ## obs. choose cumsum thr 
# 
# to match ellliptical fit
# 
# case T4, LPC1

# %%
inst_list =[
    'LC18_R',
    'LC21_R',
    'LC11_R',
    'LC25_R',
    'LC15_R',
    'LPLC2_R',
    'LC4_R',
    'LPLC1_R',
    'LC17_R',
    'LC12_R',
    # 'Mi1_R',
    # 'Tm3_R',
    # 'Mi4_R',
    # 'Mi9_R',
    # 'Tm1_R',
    # 'Tm2_R',
    # 'Tm4_R',
    # 'Tm9_R',
    # 'L4_R',
    # 'L5_R',
    'T4a_R',
    'LPC1_R',
    # 'ER2_c_R',
    # 'ER4d_R'
    ]

# %%
# naturla log of 2
np.log(10)
# base-10 log of 2
np.log10(2)

# area ratio between using std bs FWHM
8*np.log(2) 

# %%
ratio = pd.DataFrame(columns=['thr_2dGaussian', 'factor_area', 'ratio', 'ratio_cv'])
for thr_2dGaussian in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
    for factor_area in [1, 8*np.log(2)]:
        meta3 = meta.loc[meta.instance.isin(inst_list)].copy()
        # for each instance, randomly sample 10% bodyIds or 10 cells, whichever is smaller
        meta3 = meta3.groupby('instance').apply(lambda x: x.sample(n=min(20, int(len(x) * 0.1)), random_state=42)).reset_index(drop=True)

        # meta3 = meta.loc[meta.cell_type_side.str.contains('^LPC1_R', regex=True)].copy()
        meta3['col_count'] = np.nan
        meta3['col_count_input'] = np.nan
        meta3['vision'] = np.nan

        for i, bodyid in zip(meta3.index, meta3['bodyId']):
            values = effwt_visr[str(bodyid)].values
            if values.sum() > 0:
                # vision score
                meta3.at[i, 'vision'] = values.sum()

                # col count for target neuron
                col_count, _ = count_col(values, factor_thr=thr_2dGaussian)
                meta3.at[i, 'col_count'] = col_count / factor_area
            # inputs' col count
            idx_ht = ht.loc[ht['ht'] < ht.loc[ht['bodyId'] == bodyid, 'ht'].values[0], ['idx']]
            # idx of direct upstream neurons
            idx_prop = inprop[:, bodyId_to_idx[bodyid]].nonzero()[0]
            # new df with idx, normalized weight, and col_count
            idx_keep = pd.DataFrame({'idx': np.intersect1d(idx_ht['idx'], idx_prop)})
            idx_keep['bodyId'] = idx_keep['idx'].map(idx_to_bodyId)
            idx_keep['col_count'] = idx_keep['bodyId'].map(lambda x: count_col(effwt_visr[str(x)].values, factor_thr=thr_2dGaussian)[0])
            # Flatten sparse column vector to 1D numpy array
            idx_keep['wt'] = inprop[idx_keep['idx'], bodyId_to_idx[bodyid]].toarray().ravel()
            # avoid divide-by-zero if empty
            if idx_keep['wt'].sum() > 0:
                idx_keep['wt_norm'] = idx_keep['wt'] / idx_keep['wt'].sum()
                # handle NaN in col_count
                masked_data = np.ma.masked_array(idx_keep['col_count'], np.isnan(idx_keep['col_count']))
                meta3.at[i, 'col_count_input'] = np.ma.average(masked_data, weights=idx_keep['wt_norm']) / factor_area
        # by type
        meta3_type = meta3.groupby('instance').agg(
            bodyId_count = ('bodyId', 'count'),
            col_count = ('col_count', 'mean'),
            col_count_input = ('col_count_input', 'mean'),
            vision = ('vision', 'mean'),
            vision_cv = ('vision', lambda x: round(np.std(x) / np.mean(x),2) if np.mean(x) != 0 else np.nan),
            ht = ('ht', 'mean'),
        ).reset_index()
        df = pd.merge(meta3_type,
            rf_fit[rf_fit['instance'].isin(inst_list)].groupby('instance').agg({'area':'mean'}).sort_values(by='instance', ascending=True),
            on='instance',
            how='left'
        )
        ratio_value = (df['area'] / df['col_count']).mean()
        ratio_cv = (df['area'] / df['col_count']).std() / ratio_value if ratio_value != 0 else np.nan
        ratio = pd.concat([ratio, pd.DataFrame({'thr_2dGaussian': [thr_2dGaussian], 'factor_area': [factor_area], 'ratio': [ratio_value], 'ratio_cv': [ratio_cv]})], ignore_index=True)

# %% [markdown]
# volume under a 2D Gaussian within a radius of 1 std is about 0.39

# %%
ratio

# %% [markdown]
# ## ahull area

# %%
from utils.ol_rf import compute_rf

# %%
# inidx = meta.idx[meta.cell_type_side.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])]
# # outidx = meta.idx[meta.instance.isin(all_instances)]
# outidx = meta.idx[meta.instance.isin(['LC6_R'])]
# rf = compute_rf(meta, stepsn, inidx=inidx, outidx=outidx)
# df = rf.set_index('coords')[['effective weight']]
# # change col name
# # df.rename(columns={'effective weight': 'value'}, inplace=True)

# %%
# thr_mode = 'max'
# remove_frac = 0.3
# # edge_points, edges, area, com, kept_points = count_col_ahull(df, thr_mode=thr_mode, remove_frac = remove_frac)

# %%
# iterate
meta_ahull = meta[meta.bodyId.isin(rf_fit.bodyId)].copy()

# thr_mode = 'max'
# remove_frac = 0.3
# meta_ahull['ahull'] = np.nan
 
remove_frac_ls = [0.3, 0.4, 0.6]
thr_mode_ls = ['max', 'cumsum']
# initialize ahull columns defined by different thr_mode and remove_frac
for thr_mode in thr_mode_ls:
    for remove_frac in remove_frac_ls:
        col_name = f'ahull_{thr_mode}_{remove_frac}'
        meta_ahull[col_name] = np.nan

inidx = meta.idx[meta.cell_type_side.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])]

for i, row in meta_ahull.iterrows():
    rf = compute_rf(meta, stepsn, inidx=inidx, outidx=bodyId_to_idx[row['bodyId']])
    df = rf.set_index('coords')[['effective weight']]
    for thr_mode in thr_mode_ls:
        for remove_frac in remove_frac_ls: 
            edge_points, edges, area, com, kept_points = count_col_ahull(df, thr_mode=thr_mode, remove_frac = remove_frac)
            col_name = f'ahull_{thr_mode}_{remove_frac}'
            meta_ahull.at[i, col_name] = area

# save
# meta_ahull.to_pickle(Path(cache_dir, 'meta_ahull_scan.pkl'))

# %%
meta_ahull = pd.read_pickle(Path(cache_dir, 'meta_ahull_scan.pkl'))

# %%


# %% [markdown]
# ### plot

# %%
from utils.plotting_functions import plot_gaussian_params
df = effwt_visr[str(bodyid)].copy()
df = pd.DataFrame(list(df.items()), columns=['coords', 'effective weight'])

xy_coord = df['coords'].str.split(',', expand=True).astype(float).values
df['x'] = xy_coord[:,0]
df['y'] = xy_coord[:,1] #*np.sqrt(3)
df['coords'] = df['x'].astype(str) + ',' + df['y'].astype(str)

plot_df = df.set_index('coords')[['effective weight']]
fig_rf = ci.hex_heatmap(plot_df,
                custom_colorscale=[[0, "rgb(255, 255, 255)"], [1, "rgb(200, 20, 0)"]],
                global_min=0
                )
params_single_df = rf_fit.loc[rf_fit.bodyId== bodyid]
fig1 = plot_gaussian_params(params_single_df, example_bid=bodyid, fac=np.sqrt(3))

for trace in fig1.data[1:]:
            fig_rf.add_trace(trace)
# ahull 
com_h = np.average(df['x'], weights=df['effective weight'])
com_v = np.average(df['y'], weights=df['effective weight'])
thr_val = 0.1 * df['effective weight'].max()
mask_pos = df['effective weight'] > thr_val
xy_coord_pos = df[['x', 'y']][mask_pos]
edge_points, edges, area = alpha_shape_2d(xy_coord_pos.values, alpha=0.5)

for edge in edges:
    point1 = xy_coord_pos.values[edge[0]]
    point2 = xy_coord_pos.values[edge[1]]
    fig_rf.add_trace(go.Scatter(
        x=[point1[0], point2[0]],
        y=[point1[1], point2[1]],
        mode='lines',
        line=dict(color='blue', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))
# Plot xy_coord_pos points
fig_rf.add_trace(go.Scatter(
    x=xy_coord_pos.values[:, 0],
    y=xy_coord_pos.values[:, 1],
    mode='markers',
    marker=dict(color='black', size=6),
    name='positions',
    hoverinfo='skip'
))

# Plot center of mass
fig_rf.add_trace(go.Scatter(
    x=[com_h],
    y=[com_v],
    mode='markers',
    marker=dict(color='blue', size=10, symbol='cross'),
    name='center of mass',
    hoverinfo='skip'
))

# %%
from utils.hex_hex import all_hex_df
all_hex = all_hex_df()
# hex_offset = [18, 19]
hex_offset = [0, 0]

all_hex['p'] = all_hex['hex2_id'] - hex_offset[1]
all_hex['q'] = all_hex['hex1_id'] - hex_offset[0]

all_hex['v'] = all_hex['p'] + all_hex['q']
all_hex['h'] = all_hex['q'] - all_hex['p']
all_hex['h'] = - all_hex['h'] # chiasm

# %%

# plot syn_post_mean[['h','v']], color by rgb
fig, ax = plt.subplots(figsize=(4, 4))

# # for each edge in edges, plot a line between the two points
# for edge in edges:
#     point1 = xy_coord_pos.values[edge[0]]
#     point2 = xy_coord_pos.values[edge[1]]
#     ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r-', linewidth=1)

ax.scatter(all_hex['h'], all_hex['v'], marker='.', c='gray',s=5)
ax.scatter(xy_coord_pos.values[:, 0], xy_coord_pos.values[:, 1], marker='o', c='k',s=6)
ax.scatter(com_h, com_v, marker='+', c='b', s=50)
ax.set_aspect('equal')
plt.show()

# %% [markdown]
# ###  plotly version

# %%
# plotly version
fig = go.Figure()

# Plot edges
for edge in edges:
    point1 = xy_coord_pos.values[edge[0]]
    point2 = xy_coord_pos.values[edge[1]]
    fig.add_trace(go.Scatter(
        x=[point1[0], point2[0]],
        y=[point1[1], point2[1]],
        mode='lines',
        line=dict(color='red', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))

# Plot all_hex points
fig.add_trace(go.Scatter(
    x=all_hex['h'],
    y=all_hex['v'],
    mode='markers',
    marker=dict(color='gray', size=5),
    name='all hex',
    hoverinfo='skip'
))

# Plot xy_coord_pos points
fig.add_trace(go.Scatter(
    x=xy_coord_pos.values[:, 0],
    y=xy_coord_pos.values[:, 1],
    mode='markers',
    marker=dict(color='black', size=6),
    name='positions',
    hoverinfo='skip'
))

# Plot center of mass
fig.add_trace(go.Scatter(
    x=[com_h],
    y=[com_v],
    mode='markers',
    marker=dict(color='blue', size=10, symbol='cross'),
    name='center of mass',
    hoverinfo='skip'
))

# Update layout
fig.update_layout(
    width=500,
    height=500,
    template='plotly_white',
    xaxis=dict(scaleanchor="y", scaleratio=1),
    yaxis=dict(scaleanchor="x", scaleratio=1),
    showlegend=True
)

fig.show()

# %%


# %% [markdown]
# ## compare size calculation

# %%
# merge rf_fit[['superclass','instance','bodyId', 'size']], rf_fit_2[['bodyId', 'size']], and meta_ahull[['bodyId', 'ahull_max_0.3', 'ahull_max_0.4', 'ahull_max_0.6', 'ahull_cumsum_0.3', 'ahull_cumsum_0.4', 'ahull_cumsum_0.6']] on bodyId
df_merged = pd.merge(
    rf_fit[['instance','bodyId', 'size']],
    rf_fit_2[['bodyId', 'size']],
    on='bodyId',
    how='left',
    suffixes=('_thr10', '_thr07')
)

df_merged = pd.merge(
    df_merged,
    meta_ahull[['bodyId', 'ahull_max_0.3', 'ahull_max_0.4', 'ahull_max_0.6', 'ahull_cumsum_0.3', 'ahull_cumsum_0.4', 'ahull_cumsum_0.6']],
    on='bodyId',
    how='left'
)


# %%
df_merged[df_merged['instance'].str.contains('^LC|^Mi|^Tm|^T4|^LPLC|^LPC')]['instance'].unique()

# %%
# group df_merged by instance and compute the median of size_thr10, size_thr07, and all ahull columns
df_grouped = df_merged.groupby('instance').agg({
    'size_thr10': 'median',
    'size_thr07': 'median',
    'ahull_max_0.3': 'median',
    'ahull_max_0.4': 'median',
    'ahull_max_0.6': 'median',
    'ahull_cumsum_0.3': 'median',
    'ahull_cumsum_0.4': 'median',
    'ahull_cumsum_0.6': 'median'
}).reset_index()

# sort by 'size_thr10', 
df_grouped = df_grouped.sort_values(by='size_thr10').reset_index(drop=True)

# df_grouped = df_grouped[df_grouped['instance'].str.contains('^LC|^Mi|^Tm|^T4|^LPLC|^LPC') &
#                         ~df_grouped['instance'].str.contains('^Mi19')].reset_index(drop=True)

df_grouped = pd.merge(df_grouped,
                      rf_fit.groupby('instance').agg({'r2':'median'}).reset_index(),
                      on='instance', how='left')

df_grouped.shape

# %%
# scatter plot 'size_thr10', 'size_thr07', 'ahull_max_0.3', 'ahull_max_0.4', 'ahull_max_0.6', 'ahull_cumsum_0.3', 'ahull_cumsum_0.4', 'ahull_cumsum_0.6' vs index
# cols = ['size_thr10', 'size_thr07', 'ahull_max_0.3', 'ahull_max_0.4', 'ahull_max_0.6', 'ahull_cumsum_0.3', 'ahull_cumsum_0.4', 'ahull_cumsum_0.6']
cols = ['size_thr10', 'size_thr07',  'ahull_max_0.4', 'ahull_cumsum_0.6']

symbols = np.where(df_grouped['r2'] <= 0, 'circle-open', 'circle')

fig = go.Figure()
for col in cols:
    fig.add_trace(go.Scatter(
        x=df_grouped.index, y=df_grouped[col], mode='markers',
        marker=dict(size=6, symbol=symbols), name=col,
        customdata=df_grouped['instance'],
        hovertemplate='%{customdata}<br>%{y:.2f}<extra></extra>'))
fig.update_layout(xaxis_title='index (sorted by size_thr10)', yaxis_title='value')
fig.show()

# save html
fig.write_html(Path(cache_dir, 'rf_size_comparison.html'))

# %%
cols = ['size_thr10', 'size_thr07', 'ahull_max_0.4', 'ahull_cumsum_0.6',
        'ahull_max_0.3', 'ahull_max_0.6', 'ahull_cumsum_0.3', 'ahull_cumsum_0.4', 
        ]

# Spearman (rank) correlation: invariant to the scale offsets between
# ellipse-fit vs alpha-hull and across remove_frac, robust to skew/outliers.
# Note: pandas .corr() uses pairwise-complete deletion, so NaN-heavy ahull
# columns mean each cell may use a different subset of neurons.
corr = df_grouped[cols].corr(method='spearman')

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap='viridis', vmin=0, vmax=1)

ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha='right')
ax.set_yticklabels(cols)

# annotate each cell with the correlation value
for i in range(len(cols)):
    for j in range(len(cols)):
        val = corr.iloc[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                color='white' if val < 0.6 else 'black', fontsize=9)

ax.set_title('Spearman correlation of RF size methods')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman's ρ")
fig.tight_layout()
plt.show()

# save figure
fig.savefig(Path(cache_dir, 'rf_size_correlation.png'), dpi=300)

# %%
# scatter plot of meta_ahull.ahull vs rf_fit.size, color by superclass
df = pd.merge(meta_ahull[['bodyId', 'instance', 'superclass', 'ahull']], rf_fit[['bodyId', 'size']], on='bodyId', how='left')
# groupby instance and compute the median of ahull and size
df = df.groupby('instance').agg({'ahull':'median', 'size':'median', 'superclass':'first'}).reset_index()
plt.figure(figsize=(6,6))
plt.scatter(
    df[df.superclass.isin([ 'visual_projection'])]['size'], 
    df[df.superclass.isin([ 'visual_projection'])]['ahull'])
plt.xlabel('RF size (ellipse fit)')
plt.ylabel('RF area (alpha hull)')
plt.title(f'RF area comparison, thr_mode={thr_mode}, remove_frac={remove_frac}')
plt.plot([0, df['size'].max()], [0, df['size'].max()], 'r--')  # add y=x line for reference
plt.xlim(0, df['size'].max()*1.1)
plt.ylim(0, df['ahull'].max()*1.1)
plt.gca().set_aspect('equal')
plt.show()

# %%
import connectome_interpreter as ci
ci.hex_heatmap(df,
                custom_colorscale=[[0, "rgb(255, 255, 255)"], [1, "rgb(200, 20, 0)"]],
                global_min=0
                )

# %% [markdown]
# ## obs. main loop

# %%
# effwt_visr[effwt_visr.index == 'nan']

# %%
# # meta_cb_vpn['col_count'] = np.nan
# # meta_cb_vpn['col_count_input'] = np.nan
# meta_cb_vpn['vision'] = np.nan
# # meta_cb_vpn['area_fit'] = np.nan
# meta_cb_vpn['area_fit_input'] = np.nan
# meta_cb_vpn['area_ahull'] = np.nan
# # meta_cb_vpn['area_ahull_input'] = np.nan

# thr_2dGaussian = 0.4 # vol under 2D Gaussian within 1 std
# factor_area = 1

# for i, bodyid in enumerate(meta_cb_vpn['bodyId']):
#     df = effwt_visr[str(bodyid)]
#     # df = df[df.index != 'nan']
#     if (df.values.sum() > 0) and (bodyid in df_fit['bodyId'].values):
#         # - vision score
#         meta_cb_vpn.at[i, 'vision'] = df.values.sum()

#         # idx of direct upstream neurons
#         idx_prop = inprop[:, bodyId_to_idx[bodyid]].nonzero()[0]

#         # - area_fit and area_fit_input
#         idx_keep = pd.DataFrame({'idx': idx_prop})
#         idx_keep['bodyId'] = idx_keep['idx'].map(idx_to_bodyId)
#         idx_keep['area_fit'] = idx_keep['bodyId'].map(lambda x: df_fit.loc[df_fit['bodyId'] == x, 'area'].values[0] if x in df_fit['bodyId'].values else 0)
#         # Flatten sparse column vector to 1D numpy array
#         idx_keep['wt'] = inprop[idx_keep['idx'], bodyId_to_idx[bodyid]].toarray().ravel()
#         # avoid divide-by-zero if empty
#         if idx_keep['wt'].sum() > 0:
#             idx_keep['wt_norm'] = idx_keep['wt'] / idx_keep['wt'].sum()
#             # avg input col count
#             # meta_cb_vpn.at[i, 'area_fit_input'] = np.average(idx_keep['area_fit'], weights=idx_keep['wt_norm'])
#             # handle NaN 
#             masked_data = np.ma.masked_array(idx_keep['area_fit'], np.isnan(idx_keep['area_fit']))
#             meta_cb_vpn.at[i, 'area_fit_input'] = np.ma.average(masked_data, weights=idx_keep['wt_norm'])

#         # - area_ahull
#         edge_points, edges, area, com = count_col_ahull(df)

#         if len(area)  == 1:    
#             meta_cb_vpn.at[i, 'area_ahull'] = area[0] * thr_2dGaussian

#             # -> todo  DEBUG bodyid = 90589
#             # idx_keep['area_ahull'] = idx_keep['bodyId'].map(lambda x: count_col_ahull(effwt_visr[str(x)])[2][0] * thr_2dGaussian)
#             # # avoid divide-by-zero if empty 
#             # if idx_keep['wt'].sum() > 0:
#             #     # avg input
#             #     masked_data = np.ma.masked_array(idx_keep['area_ahull'], np.isnan(idx_keep['area_ahull']))
#             #     meta_cb_vpn.at[i, 'area_ahull_input'] = np.ma.average(masked_data, weights=idx_keep['wt_norm'])

#         # - col count for target neuron
#         # values = effwt_visr[str(bodyid)].values
#         # col_count, _ = count_col(values, factor_thr=thr_2dGaussian)
#         # meta_cb_vpn.at[i, 'col_count'] = col_count / factor_area

#         # - col_count_input, filter by hitting time
#         # idx_ht = ht.loc[ht['ht'] < ht.loc[ht['bodyId'] == bodyid, 'ht'].values[0], ['idx']]
#         # # new df with idx, normalized weight, and col_count
#         # idx_keep = pd.DataFrame({'idx': np.intersect1d(idx_ht['idx'], idx_prop)})
#         # idx_keep['bodyId'] = idx_keep['idx'].map(idx_to_bodyId)
#         # idx_keep['col_count'] = idx_keep['bodyId'].map(lambda x: count_col(effwt_visr[str(x)].values, factor_thr=thr_2dGaussian)[0])
#         # # Flatten sparse column vector to 1D numpy array
#         # idx_keep['wt'] = inprop[idx_keep['idx'], bodyId_to_idx[bodyid]].toarray().ravel()
#         # # avoid divide-by-zero if empty
#         # if idx_keep['wt'].sum() > 0:
#         #     idx_keep['wt_norm'] = idx_keep['wt'] / idx_keep['wt'].sum()
#         #     # avg input col count
#         #     # meta_cb_vpn.at[i, 'col_count_input'] = np.average(idx_keep['col_count'], weights=idx_keep['wt_norm'])   
#         #     # handle NaN in col_count
#         #     masked_data = np.ma.masked_array(idx_keep['col_count'], np.isnan(idx_keep['col_count']))
#         #     meta_cb_vpn.at[i, 'col_count_input'] = np.ma.average(masked_data, weights=idx_keep['wt_norm']) / factor_area

# %%
# # add vpn group
# meta_cb_vpn = pd.merge(meta_cb_vpn, oltypes_vpn[['instance', 'main_groups']], on='instance', how='left')
# meta_cb_vpn.loc[meta_cb_vpn['main_groups'] != 'VPN', 'main_groups'] = 'BVNC'
# # add area and r2
# meta_cb_vpn = pd.merge(meta_cb_vpn, df_fit[['bodyId', 'area', 'r2']], on='bodyId', how='left')
# meta_cb_vpn.rename(columns={'area': 'area_fit'}, inplace=True)

# meta_cb_vpn.shape

# %%
# # SAVE
# # meta_cb_vpn.to_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum40.csv'), index=False)

# # load
# meta_cb_vpn = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum40.csv'), dtype={'bodyId': int})

# meta_cb_vpn.shape

# %%
# df = meta_cb_vpn[(meta_cb_vpn.area_ahull.notna()) & (meta_cb_vpn.area_fit.notna())]

# %%
# # plot area_fit vs area_ahull
# fig, ax = plt.subplots(figsize=(4, 4))
# ax.scatter(df['area_fit'], df['area_ahull'], c='gray', s=10)
# ax.set_xlabel('Fitted RF area (std 2D Gaussian)')
# ax.set_ylabel('Alpha hull area')
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# ax.plot([1, 400], [1, 400], 'r--')
# # fit a line
# m, b = np.polyfit(df['area_fit'], df['area_ahull'], 1)
# ax.plot([1, 400], [m*1 + b, m*400 + b], 'b-')
# # add text for r value and m, b
# r_value = df['area_fit'].corr(df['area_ahull'])
# ax.text(300,500, f'r={r_value:.2f}\ny={m:.2f}x+{b:.2f}', fontsize=10)
# plt.show()

# %% [markdown]
# ## vic and res, assign ht-group. Only vic for revision

# %%
meta_cb_vpn

# %%
# load vic and hit
vp_cb_r = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_r.p'))
vp_cb_r.shape

# %%
vp_cb_r.head()

# %%
meta_cb_vpn = pd.merge(
    meta_cb_vpn, 
    vp_cb_r
                       

# %%
# load and filter
meta_cb_vpn = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum40.csv'), dtype={'bodyId': int})

thr_vic = 5e-4

meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['vision'] > thr_vic]

# %%
# meta by type
meta_type_cb_vpn = meta_cb_vpn.groupby(['cell_type_side', 'instance']).agg(
    bodyId_count = ('bodyId', 'count'),
    vision = ('vision', 'median'),
    area_fit = ('area_fit', 'median'),
    # area_fit_input = ('area_fit_input', 'median'),
    r2 = ('r2', 'median'),
    # col_count = ('col_count', 'median'),
    # col_count_input = ('col_count_input', 'median'),
    # vision_cv = ('vision', lambda x: round(np.std(x) / np.mean(x),2) if np.mean(x) != 0 else np.nan),
    ht = ('ht', 'median'),
    main_groups = ('main_groups', 'first'),
    # ghop = ('ghop', 'first'),
).reset_index()

# %%
meta_cb_vpn[meta_cb_vpn['ht'].isna()]

# %%
# SAVE
meta_cb_vpn.to_pickle(Path(cache_dir, f'meta_cb_vpn_cumsum40_vic1em4.pkl'))
meta_type_cb_vpn.to_pickle(Path(cache_dir, f'meta_type_cb_vpn_cumsum40_vic1em4.pkl'))

# # load
# meta_cb_vpn = pd.read_pickle(Path(cache_dir, f'meta_cb_vpn_cumsum40_vic1em4.pkl'))
# meta_type_cb_vpn = pd.read_pickle(Path(cache_dir, f'meta_type_cb_vpn_cumsum40_vic1em4.pkl'))

# %% [markdown]
# ## Assign ghop

# %%
# load vic and hit
vp_cb_r = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_r.p'))
vp_cb_r.shape

# %%
vp_cb_r['main_groups'].value_counts(dropna=False)

# %%
thr_inwt = 10 # 10% normalized input is vision
thr_vic = 5e-4

# %%
inst_t0 = vp_cb_r[
    (vp_cb_r['VIC'] > thr_vic) & 
    (vp_cb_r['main_groups'] == 'OL output')]['instance'].values
print(len(inst_t0))

# %% [markdown]
# ### ? obs.

# %%
# load
meta_cb_vpn = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum40.csv'), dtype={'bodyId': int})

# %%
# # DEBUG
# meta_cb_vpn[meta_cb_vpn.instance.str.contains('^KCg-d.*')]['vision'].median()

# %%
# meta by type
meta_type_cb_vpn = meta_cb_vpn.groupby(['cell_type_side', 'instance']).agg(
    bodyId_count = ('bodyId', 'count'),
    # area_fit = ('area_fit', 'median'),
    # area_fit_input = ('area_fit_input', 'median'),
    # r2 = ('r2', 'median'),
    # col_count = ('col_count', 'median'),
    # col_count_input = ('col_count_input', 'median'),
    vision = ('vision', 'median'),
    vision_cv = ('vision', lambda x: round(np.std(x) / np.mean(x),2) if np.mean(x) != 0 else np.nan),
    ht = ('ht', 'median'),
    main_groups = ('main_groups', 'first'),
    # ghop = ('ghop', 'first'),
).reset_index()

# %% [markdown]
# ### obs. exclu vpns by vision < 0.04 or none

# %%
# ax = meta_type_cb_vpn[
#     (meta_type_cb_vpn['vision'] < 0.8) &
#     (meta_type_cb_vpn['main_groups'] == 'VPN')
# ]['vision'].hist(bins=200, grid=False, figsize=(6,4))
# ax.set_xlim(0, 0.1)
# # SAVE
# plt.tight_layout()
# plt.savefig(Path(result_dir, 'meta_cb_vpn_vision_hist.png'), dpi=300)
# plt.show()

# %%
inst_t0 = meta_type_cb_vpn[
    (meta_type_cb_vpn['vision'] > thr_vic/100) & # vision > 0.04
    (meta_type_cb_vpn['main_groups'] == 'VPN')]['instance'].values
print(len(inst_t0))

# # add 'LC9_R' to inst_t0
# inst_t0 = np.append(inst_t0, 'LC9_R')
# print(len(inst_t0))

# %%
# SAVE csv
# pd.Series(inst_t0).to_csv(Path(result_dir, 'inst_vpn_thr004.csv'), index=False, header=False)

# %%
# meta_type_cb_vpn[
#     (meta_type_cb_vpn['vision'] > 0.04) &
#     (meta_type_cb_vpn['vision'] < 0.05) &
#     (meta_type_cb_vpn['main_groups'] == 'VPN')]['instance']

# %% [markdown]
# ### obs. excl vpn by cumsum

# %%
# df = meta_cb_vpn[meta_cb_vpn['main_groups'] == 'VPN'].\
#     groupby('instance').agg({'vision': 'mean'}).sort_values(by='vision', ascending=False)
# df.shape

# %%
# # Sort values in descending order and compute cumsum
# thr_vic = 0.95
# cumsum = df['vision'].cumsum()
# total = cumsum.iloc[-1]

# # Find indices where cumsum reaches $thr_vic of total
# threshold = thr_vic * total
# mask_thr = cumsum <= threshold

# df = df[mask_thr]
# df.shape

# %% [markdown]
# ### obs. add VPN hopping groups using inprop, filter by VIC

# %%
# meta_t0 = meta_cb_vpn[meta_cb_vpn['instance'].isin(inst_t0)]
# idx_t0 = meta_t0.idx
# len(idx_t0)

# %%
# # find T1 idx
# inprop_t0 = inprop[list(idx_t0), :]
# # Get all non-zero column indices at once
# _, col_indices = inprop_t0.nonzero()
# # Convert to set for unique values
# idx_keep = list(set(col_indices))
# print(f"Total unique non-zero indices: {len(idx_keep)}")
# # find entries in idx_keep with column sum >= 0.1
# idx_keep = [idx for idx, val in zip(idx_keep, inprop[list(idx_t0), :][:, idx_keep].sum(axis=0).A1) if val >= 0.1]
# print(len(idx_keep))

# %%
# inst_t1 = meta_cb_vpn.loc[(meta_cb_vpn['idx'].isin(idx_keep)) & (meta_cb_vpn['main_groups'] != 'VPN'), 'instance'].unique()
# meta_t1 = meta_cb_vpn[meta_cb_vpn['instance'].isin(inst_t1)]
# idx_t1 = meta_t1['idx']
# meta_t1.shape

# %%
# # find T2 idx
# inprop_t1 = inprop[list(idx_t1), :]
# # Get all non-zero column indices at once
# _, col_indices = inprop_t1.nonzero()
# # Convert to set for unique values
# idx_keep = list(set(col_indices))
# print(f"Total unique non-zero indices: {len(idx_keep)}")
# # find entries in idx_keep with column sum >= 0.1
# idx_keep = [idx for idx, val in zip(idx_keep, inprop[list(idx_t0), :][:, idx_keep].sum(axis=0).A1) if val >= 0.1]
# # keep those not in idx_t1
# idx_keep = [idx for idx in idx_keep if idx not in idx_t1]
# print(len(idx_keep))

# %%
# inst_t2 = meta_cb_vpn.loc[(meta_cb_vpn['idx'].isin(idx_keep)) & (meta_cb_vpn['main_groups'] != 'VPN'), 'instance'].unique()
# # remove instances already in inst_t1
# inst_t2 = [inst for inst in inst_t2 if inst not in inst_t1]
# meta_t2 = meta_cb_vpn[meta_cb_vpn['instance'].isin(inst_t2)]
# idx_t2 = meta_t2['idx']
# meta_t2.shape

# %%


# %% [markdown]
# ### add VPN hopping groups using neuprint, filter by vpn-vision

# %%
# DEBUG
# oltypes[(oltypes['main groups'] == 'VPN') & (oltypes['instance'].str.contains('.*_L$'))].shape
# vpn_inst_keep = meta_type_cb_vpn.loc[(meta_type_cb_vpn['main_groups'] == 'VPN') & (meta_type_cb_vpn['vision'] > 0.04)
#                                 , 'instance'].values

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
# remove those with apostophe or ? or +, and exclude ol
conn_t0t1 = conn_t0t1[~conn_t0t1['instance_post'].str.contains("\\'") & 
                   ~conn_t0t1['instance_post'].str.contains("\\?") &
                   ~conn_t0t1['instance_post'].str.contains("\\+") &
                   ~conn_t0t1['instance_post'].str.contains("unclear") &
                    # ~conn_t0t1['instance_post'].isin(oltypes_vpn['instance'])
                    ~conn_t0t1['instance_post'].isin(oltypes['instance'])
                    ]
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
# df = pd.merge(df, df_fit[['bodyId', 'r2']], left_on='bodyId_post', right_on='bodyId', how='left')

instwt_t1 = df.groupby(['instance_post']).agg(
    bodyId_count = ('bodyId_post', 'count'),
    inwt_median = ('inwt', 'median'),
    # r2_median = ('r2', 'median')
).reset_index()

# %%
# SAVE pickle
# instwt_t1.to_pickle(Path(cache_dir, f'instwt_t1_{thr_vic:02d}.pkl'))
instwt_t1.to_pickle(Path(cache_dir, f'instwt_t1.pkl'))

# load pickle
instwt_t1 = pd.read_pickle(Path(cache_dir, f'instwt_t1.pkl'))

# %%
# ax = df['inwt'].hist(bins=500, figsize=(4,2))
ax = instwt_t1['inwt_median'].hist(bins=np.logspace(np.log10(instwt_t1['inwt_median'].min()), np.log10(instwt_t1['inwt_median'].max()), 200), figsize=(4,2))
ax.set_xscale('log')
ax.set_xlim(0, 0.5)
# ax.set_ylim(0, 2000)

# %%
thr_inwt/100

# %%
# filter, 10% or 1%
inst_t1 = instwt_t1.loc[instwt_t1['inwt_median'] >= thr_inwt/100, 'instance_post'].unique()
print(instwt_t1.shape[0], len(inst_t1))

# %%
# # DEBUG
# [inst for inst in inst_t1 if re.match(r'^TuBu.*', inst)]

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
                          # ~conn_t0t1t2['instance_post'].isin(oltypes_vpn['instance']) &
                          ~conn_t0t1t2['instance_post'].isin(oltypes['instance']) &
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
# ax = df['inwt'].hist(bins=500, figsize=(4,2))
ax = df['inwt'].hist(bins=np.logspace(np.log10(df['inwt'].min()), np.log10(df['inwt'].max()), 200), figsize=(4,2))
ax.set_xscale('log')
ax.set_xlim(0, 0.5)
# ax.set_xlim(0, 0.1)

# %%
# filter, 10% or 1%
inst_t2 = df_inst.loc[df_inst['inwt_median'] >= thr_inwt/100, 'instance_post'].unique()
len(inst_t2)

# %%
# meta_cb_vpn.rename(columns={'main groups': 'main_groups'}, inplace=True)
# meta_type_cb_vpn.rename(columns={'main groups': 'main_groups'}, inplace=True)

# %% [markdown]
# ### define ghop

# %%
# add hopping groups
vp_cb_ghop_r = vp_cb_r.copy()
# rename main_groups, 'OL output' to 'VPN', 'nonOL' and 'NaN' to 'later'
vp_cb_ghop_r['main_groups'] = vp_cb_ghop_r['main_groups'].replace({
    'OL output': 'VPN',
    'nonOL': 'later',
    np.nan: 'later'
})

vp_cb_ghop_r.loc[vp_cb_ghop_r['instance'].isin(inst_t0), 'ghop'] = 'T0'
vp_cb_ghop_r.loc[vp_cb_ghop_r['instance'].isin(inst_t1), 'ghop'] = 'T1'
vp_cb_ghop_r.loc[vp_cb_ghop_r['instance'].isin(inst_t2), 'ghop'] = 'T2'

# %%
# # add hopping groups

# meta_cb_vpn['ghop'] = meta_cb_vpn['main_groups'].map({'BVNC': 'later', 'VPN': 'VPN'})
# meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t0), 'ghop'] = 'T0'
# meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t1), 'ghop'] = 'T1'
# meta_cb_vpn.loc[meta_cb_vpn['instance'].isin(inst_t2), 'ghop'] = 'T2'

# meta_type_cb_vpn['ghop'] = meta_type_cb_vpn['main_groups'].map({'BVNC': 'later', 'VPN': 'VPN'})
# meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t0), 'ghop'] = 'T0'
# meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t1), 'ghop'] = 'T1'
# meta_type_cb_vpn.loc[meta_type_cb_vpn['instance'].isin(inst_t2), 'ghop'] = 'T2'

# %%
# SAVE
# meta_cb_vpn.to_pickle(Path(cache_dir, f'meta_cb_vpn_cumsum40_ghop_vic_{thr_vic:02d}{thr_inwt:02d}{thr_inwt:02d}.pkl'))
# meta_type_cb_vpn.to_pickle(Path(cache_dir, f'meta_type_cb_vpn_cumsum40_ghop_vic_{thr_vic:02d}{thr_inwt:02d}{thr_inwt:02d}.pkl'))
vp_cb_ghop_r.to_pickle(Path(cache_dir, f'vp_cb_ghop_r.pkl'))

# load
# meta_cb_vpn = pd.read_pickle(Path(cache_dir, f'meta_cb_vpn_cumsum40_ghop_vic_41010.pkl'))
# meta_type_cb_vpn = pd.read_pickle(Path(cache_dir, f'meta_type_cb_vpn_cumsum40_ghop_vic_41010.pkl'))

# %% [markdown]
# ## resolution, excl vpn and assign ghop

# %%
# load
meta_cb_vpn = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum40.csv'), dtype={'bodyId': int})

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
# ### exclu vpns by, r2 > 0, bodyId_count

# %%
# vision > 0.04
inst_t0 = meta_type_cb_vpn[
    (meta_type_cb_vpn['vision'] > thr_vic/100) &
    # (meta_type_cb_vpn['bodyId_count'] >= 2) &
    (meta_type_cb_vpn['r2'] >= 0) &
    (meta_type_cb_vpn['main_groups'] == 'VPN')
    ]['instance'].values
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
                    # ~conn_t0t1['instance_post'].isin(oltypes_vpn['instance'])
                    ~conn_t0t1['instance_post'].isin(oltypes['instance'])
                    ]
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
# # load pickle
# instwt_t1 = pd.read_pickle(Path(cache_dir, 'instwt_t1.pkl'))
# # filter
# inst_t1 = instwt_t1.loc[(instwt_t1['inwt_median'] >= 0.1) 
#                         # & (df_inst['r2'] >= 0)
#                         , 'instance_post'].unique()
# len(inst_t1)

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
                    # ~conn_t0t1t2['instance_post'].isin(oltypes_vpn['instance']) &
                    ~conn_t0t1t2['instance_post'].isin(oltypes['instance']) &
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
meta_cb_vpn.to_pickle(Path(cache_dir, f'meta_cb_vpn_cumsum40_ghop_res_{thr_vic:02d}{thr_inwt:02d}{thr_inwt:02d}_r2.pkl'))
meta_type_cb_vpn.to_pickle(Path(cache_dir, f'meta_type_cb_vpn_cumsum40_ghop_res_{thr_vic:02d}{thr_inwt:02d}{thr_inwt:02d}_r2.pkl'))

# load
# meta_cb_vpn = pd.read_pickle(Path(cache_dir, f'meta_cb_vpn_cumsum40_ghop_res_{thr_vic:02d}{thr_inwt:02d}{thr_inwt:02d}_r2.pkl'))
# meta_type_cb_vpn = pd.read_pickle(Path(cache_dir, f'meta_type_cb_vpn_cumsum40_ghop_res_{thr_vic:02d}{thr_inwt:02d}{thr_inwt:02d}_r2.pkl'))

# %%
meta_type_cb_vpn.ghop.value_counts()

# %% [markdown]
# ## intra-type connection vs resolution

# %%
meta_type = meta_type_cb_vpn[(meta_type_cb_vpn['ghop'] == 'T0') 
                 & (meta_type_cb_vpn['bodyId_count'] >= 50)].sort_values(by='bodyId_count', ascending=True)

# %%
# scatter plot meta_type_cb_vpn['area_fit'] vs meta_type_cb_vpn['bodyId_count']
# and meta_type_cb_vpn['col_count'] vs meta_type_cb_vpn['bodyId_count'], 
fig = px.scatter(meta_type_cb_vpn[(meta_type_cb_vpn.ghop == 'T0') & (meta_type_cb_vpn.bodyId_count >= 55)], 
                 x='area_fit', y='bodyId_count', color='ghop', hover_data=['instance'])
fig.update_layout(width=600, height=400)    
fig.show()

# %%
for i, row in meta_type.iterrows():
    ids = meta_cb_vpn[meta_cb_vpn['instance'] == row['instance']]['bodyId'].values
    df_fit = rf_fit[rf_fit['bodyId'].isin(ids)]
    neuron_df, connection_df = fetch_adjacencies(
    sources= ids,
    targets= ids,
    rois= rois_cb,
    min_total_weight=1
    )
    # conn_df = neuprint.merge_neuron_properties(neuron_df, connection_df, ['type', 'instance'])\
    #     .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
    #     .agg({'weight': 'sum'})\
    #     .reset_index()\
    #     .sort_values(by='weight', ascending=False)
    meta_type.at[i, 'intra_conn'] = connection_df.weight.sum()
    
    

# %%
meta_type['intra_conn_norm'] = meta_type['intra_conn'] / meta_type['bodyId_count'] / (meta_type['bodyId_count'] - 1)

# %%
ids = meta_cb_vpn[meta_cb_vpn['instance'] == 'LC4_R']['bodyId'].values
df_fit = rf_fit[rf_fit['bodyId'].isin(ids)]
neuron_df, connection_df = fetch_adjacencies(
sources= ids,
targets= ids,
rois= rois_cb,
min_total_weight=1
)

# %% [markdown]
# # plots

# %% [markdown]
# ### scale factor and col_vs_area

# %%
# load
meta_type_cb_vpn = pd.read_pickle(Path(cache_dir, 'meta_type_cb_vpn_cumsum40_ghop_res_001010_r2.pkl'))
# meta_type_cb_vpn = meta_type_cb_vpn[~meta_type_cb_vpn['area_fit'].isna()]

# %%
groups = ['T0', 'T1', 'T2']
df_plot = meta_type_cb_vpn[meta_type_cb_vpn['ghop'].isin(['T0','T1','T2'])].copy()
# define color column in df_plot based on ghop column and groups list
df_plot['color'] = df_plot['ghop'].apply(lambda g: px.colors.qualitative.Safe[groups.index(g)] if g in groups else 'lightgray') 

## ##
# xtitle = 'col_count'
# ytitle = 'area_fit'
xtitle = 'area_fit_input'
ytitle = 'area_fit'
# xtitle = 'col_count_input'
# ytitle = 'col_count'

fig_scatter = go.Figure()

for i in range(len(groups)):
    masked = df_plot[df_plot['ghop'] == groups[i]]
    fig_scatter.add_trace(go.Scatter(
        x = masked[xtitle],
        y = masked[ytitle],
        mode='markers',
        name=groups[i],
        marker=dict(size=6, opacity=0.75, line=dict(width=0, color='black'), color=masked['color']),
        text=masked['cell_type_side'] + "_count_" + masked['bodyId_count'].astype(str),
        hovertemplate=f'{xtitle}=%{{x}}<br>{ytitle}=%{{y}}<br>%{{text}}<extra></extra>'
    ))
# add line y=x
fig_scatter.add_shape(
    type='line',
    x0=0, y0=0, x1=300, y1=300,
    line=dict(color='pink', dash='dash'),
    layer='below'
)
# Fix axis ranges and 1:1 aspect
a = dict(scaleanchor='x', scaleratio=1, title=ytitle)
fig_scatter.update_yaxes(**a)
fig_scatter.update_xaxes(title=xtitle)
fig_scatter.update_layout(
    template='plotly_white',
    margin=dict(l=60, r=20, t=60, b=60),
    width=500, height=500
)
# # same ticks for x and y axes
# fig_scatter.update_xaxes(tickmode='array', tickvals=np.arange(0, 101, 50))
# fig_scatter.update_yaxes(tickmode='array', tickvals=np.arange(0, 101, 50))

fig_scatter.show()

# save html
# fig_scatter.write_html(Path(result_dir, 'area_vs_col_csum40_001010.html'))
fig_scatter.write_html(Path(result_dir, 'scafac_area_csum40_001010_r2.html'))
# fig_scatter.write_html(Path(result_dir, 'scafac_col_csum40_001010.html'))

# %% [markdown]
# ### histo of vision score

# %%
# # histogram of meta_cb_vpn['vision'] split by VPN vs Other, by bodyId
# bins = 500
# plt.figure(figsize=(6,3.2))

# _col_main = "main groups"
# vpn_mask = meta_cb_vpn[_col_main] == 'VPN'
# vals_vpn = meta_cb_vpn.loc[vpn_mask, 'vision'].dropna()
# vals_other = meta_cb_vpn.loc[~vpn_mask, 'vision'].dropna()

# plt.hist([vals_vpn, vals_other], bins=bins, stacked=True, label=['VPN','Other'],
#              color=['#1f77b4','#ff7f0e'], alpha=0.5)
# plt.xlabel('vision')
# plt.ylabel('count')
# plt.title("vision distribution ")
# # plt.legend(frameon=False)
# plt.yscale('log')
# plt.grid(True, which='both', linestyle='-', linewidth=0.5)
# plt.tight_layout()
# plt.xlim(0, 0.1)

# print({'VPN_count': len(vals_vpn), 'Other_count': len(vals_other)})

# # SAVE
# # plt.savefig(Path(result_dir, 'meta_cb_vpn_vision_hist_csum90.png'), dpi=300)

# %%
# NB, stacked with log axis doesn't make sense
bins = 500
# df_plot = meta_cb_vpn.copy()
df_plot = meta_type_cb_vpn.copy()
groups = df_plot['ghop'].unique().tolist()
df_plot['color'] = df_plot['ghop'].apply(lambda g: px.colors.qualitative.Safe[groups.index(g)] if g in groups else 'lightgray') 
data_grouped = [df_plot[df_plot['ghop'] == g] for g in groups]
# Create plotly figure
fig = go.Figure()
# Add histogram traces for each group
for i, group in enumerate(groups):
    fig.add_trace(go.Histogram(
        x=data_grouped[i]['vision'].dropna(),
        name=group,
        nbinsx=bins,
        # marker_color=colors_map[group],
        marker_color=data_grouped[i]['color'],
        opacity=0.7,
        histnorm='',  # Use count
    ))
# Update layout
fig.update_layout(
    title="Distribution of Vision Values by Cell Type",
    xaxis_title="vision",
    yaxis_title="count",
    yaxis_type="log",
    barmode='stack',  # Stack the histograms
    width=800,    height=400,
    showlegend=True,
    xaxis_range=[0, 0.5],
    template="plotly_white",
    font=dict(size=12)
)
# Add grid with proper log scale grid lines
fig.update_xaxes(showgrid=True, gridwidth=0.3, gridcolor='lightgray')
fig.update_yaxes(
    minor=dict(
        showgrid=True,
        gridwidth=0.3,
        gridcolor='lightgray',
    )
)
fig.show()
# SAVE fig
# fig.write_html(Path(result_dir, 'meta_cb_vpn_vision_hist_ghop.html'))

# %% [markdown]
# ### ht vs size

# %%
# load
meta_type_cb_vpn = pd.read_pickle(Path(cache_dir, 'meta_type_cb_vpn_cumsum30_ghop_res_041010.pkl'))

# %%
groups = ['T0', 'T1', 'T2', 'VPN']
df_plot = meta_type_cb_vpn.copy()
df_plot['color'] = df_plot['ghop'].apply(lambda g: px.colors.qualitative.Safe[groups.index(g)] if g in groups else 'lightgray') 


fig_scatter = px.scatter(
    df_plot,
    x='vision',
    y='ht',
    opacity=0.5,
    hover_data=['cell_type_side'],
    )
# use per-point colors from df_plot['color']
fig_scatter.update_traces(marker=dict(color=df_plot['color']))

fig_scatter.update_layout(
    title='Vision vs Hitting Time',
    template='plotly_white',
    showlegend=True,
    margin=dict(l=60, r=20, t=60, b=60),
)
# add legend

fig_scatter.update_yaxes(range=[0, 10])
fig_scatter.update_layout(width=500, height=500)
fig_scatter.show()

# save html
# fig_scatter.write_html(Path(result_dir, 'meta_cb_vpn_vision_vs_ht.html'))

# %%
# high vision in later group
meta_type_cb_vpn[(meta_type_cb_vpn['ghop'] == 'later') & (meta_type_cb_vpn['vision'] > 0.05)].sort_values(by='vision', ascending=False)

# %% [markdown]
# ### col_count/area_fit vs r2

# %%
# load
meta_cb_vpn = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum70_convert.csv'), dtype={'bodyId': int})

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

# %%
# meta_type_cb_vpn['r2'].isna().sum()
meta_type_cb_vpn = meta_type_cb_vpn[~meta_type_cb_vpn['r2'].isna()]
meta_type_cb_vpn['ratio'] = meta_type_cb_vpn['col_count'] / meta_type_cb_vpn['area_fit']

# %%
# scatter plot meta_type_cb_vpn['ratio'] vs meta_type_cb_vpn['r2']
df_plot = meta_type_cb_vpn.copy()
df_plot['color'] = df_plot['main_groups'].apply(lambda g: 'blue' if g == 'VPN' else 'orange')
fig_scatter = px.scatter(
    df_plot,
    x='ratio',
    y='r2',
    color='color',
    opacity=0.5,
    hover_data=['cell_type_side'],
)
fig_scatter.update_traces(marker=dict(color=df_plot['color']))
# add horizontal line at y=0.8
fig_scatter.update_layout(
    # xaxis_range=[0, df_plot['ratio'].max()],
    yaxis_range=[0, 1],
)   
fig_scatter.update_layout(width=500, height=500)
fig_scatter.show()

# %% [markdown]
# ## tbar resolution distr in rois

# %%
# load
meta_cb_vpn = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_ghop_res_040505.pkl'))
meta_type_cb_vpn = pd.read_pickle(Path(cache_dir, 'meta_type_cb_vpn_ghop_res_040505.pkl'))

# load
syn = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_tbar.pkl'))
syn = syn[syn['type'] == 'pre']

# %%
# 
syn = pd.merge(syn, meta_cb_vpn[['bodyId', 'area_fit', 'vision', 'col_count','ghop']], on='bodyId', how='left')

# %% [markdown]
# ###  PLOT aligned histograms for individual ROIs

# %%
syn[syn.roi == 'AOTU(R)']

# %%

rois = ['AOTU(R)', 'PLP(R)', 'PVLP(R)', 'SPS(R)', 'others']
fig, axes = plt.subplots(len(rois), 1, figsize=(4, 1.5*len(rois)), sharex=True, sharey=True)
for i, roi in enumerate(rois):
    ax = axes[i]
    if roi != 'others':
        df_roi = syn[(syn['roi'] == roi) & (syn['ghop'].isin(['T0']))]
        # remove rows with NaN in 'col_count'
        df_roi = df_roi[df_roi['col_count'].notna()]
    else:
        df_roi = syn[(~syn['roi'].isin(rois[:-1])) & (syn['ghop'].isin(['T0']))]
        # remove rows with NaN in 'col_count'
        df_roi = df_roi[df_roi['col_count'].notna()]

    ## ##
    ax.hist(df_roi['area_fit'], bins=400, color='skyblue', edgecolor='black', alpha=0.7)
    # ax.hist(df_roi['col_count'], bins=400, color='skyblue', edgecolor='black', alpha=0.7)
    # ax.set_title(f'{roi}')
    ax.text(0.80, 0.95, f'{roi}', transform=ax.transAxes, ha='left', va='top',
        fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    # ax.set_ylabel('tbar count')
    ax.grid(axis='y', alpha=0.75)
    # ax.set_yscale('log')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 12000)
ax.set_xlabel('hex size')
plt.tight_layout()
## ## SAVE
plt.savefig(Path(result_dir, 'meta_cb_vpn_roi_tbar_hist_areaFit.png'), dpi=300)
# plt.savefig(Path(result_dir, 'meta_cb_vpn_roi_tbar_hist_colCount.png'), dpi=300)

plt.show()

# %% [markdown]
# # ol+vpn+cb

# %%
# meta_ol.rename(columns={'hitting_time': 'ht'}, inplace=True)

# %%
# concatenate meta_cb_vpn and meta_ol
meta_type_ol['ghop'] = 'ol'
meta_all = pd.concat([
    meta_type_cb_vpn[['ht','vision', 'ghop','instance']], 
    meta_type_ol[['ht','vision', 'ghop','instance']],
    ], ignore_index=True)

# %%
# scatter plot vision vs ht
df_plot = meta_all.copy()
df_plot['color'] = df_plot['ghop'].apply(lambda g: px.colors.qualitative.Safe[groups.index(g)] if g in groups else 'lightgray') 

fig_scatter = px.scatter(
    df_plot,
    x='vision',
    y='ht',
    opacity=0.5,
    hover_data=['instance'],
    )
# use per-point colors from df_plot['color']
fig_scatter.update_traces(marker=dict(color=df_plot['color']))

fig_scatter.update_layout(
    title='Vision vs Hitting Time',
    template='plotly_white',
    showlegend=True,
    margin=dict(l=60, r=20, t=60, b=60),
)
# plot y = 1/x line
x = np.linspace(0.001, 0.5, 100)
y = 1 / x /10
fig_scatter.add_trace(go.Scatter(x=x, y=y, mode='lines', name='y=1/x', line=dict(color='black', dash='dash')))

fig_scatter.update_yaxes(range=[0, 10])
fig_scatter.update_layout(width=500, height=500)
fig_scatter.show()

# save html
# fig_scatter.write_html(Path(result_dir, 'meta_cb_vpn_vision_vs_ht.html'))

# %%
# find all point to the left of the line y = 1/ (10*x)
df_left = df_plot[df_plot['ht'] < 1 / (10 * df_plot['vision'])]

# %% [markdown]
# # vision contrib from VPNs

# %% [markdown]
# ### compute 

# %%
meta_vpn2cb = meta[~meta['cell_type'].isin(oltypes_nonvpn['cell type'])].copy()
meta_vpn2cb = meta_vpn2cb[meta_vpn2cb['class'] != 'visual']

meta_vpn2cb.reset_index(drop=True, inplace=True)
meta_vpn2cb.shape, meta_vpn2cb['cell_type_side'].nunique()

# %%
meta_vpn2cb['vision'] = np.nan

for i, bodyid in enumerate(meta_vpn2cb['bodyId']):
    meta_vpn2cb.at[i, 'vision'] = effwt_vpnr[str(bodyid)].values.sum()

# %%
# add vpn group
meta_vpn2cb = pd.merge(meta_vpn2cb, oltypes_vpn[['instance', 'main_groups']], on='instance', how='left')
meta_vpn2cb.loc[meta_vpn2cb['main_groups'] != 'VPN', 'main_groups'] = 'BVNC'
meta_vpn2cb.shape

# %%
# set vision = 1 for VPN
meta_vpn2cb.loc[meta_vpn2cb['main_groups'] == 'VPN', 'vision'] = 1

# %%
# SAVE
# meta_vpn2cb.to_csv(Path(cache_dir, 'meta_vpn2cb.csv'), index=False)

# load
meta_vpn2cb = pd.read_csv(Path(cache_dir, 'meta_vpn2cb.csv'), dtype={'bodyId': int})

meta_vpn2cb.shape

# %%
# meta by type
meta_type_vpn2cb = meta_vpn2cb.groupby(['cell_type_side', 'instance']).agg(
    bodyId_count = ('bodyId', 'count'),
    vision = ('vision', 'mean'),
    vision_cv = ('vision', lambda x: round(np.std(x) / np.mean(x),2) if np.mean(x) != 0 else np.nan),
    ht = ('ht', 'mean'),
    main_groups = ('main_groups', 'first')
).reset_index()

# %%
T1_info[T1_info['cell_type_side'].str.contains('TuBu.*', na=False)]

# %%
meta_type_vpn2cb[meta_type_vpn2cb['cell_type_side'].str.contains('TuBu')]

# %% [markdown]
# ### add hoppping groups

# %%
# add hopping groups
meta_vpn2cb['ghop'] = meta_vpn2cb['main_groups'].map({'BVNC': 'later', 'VPN': 'VPN'})
meta_vpn2cb.loc[meta_vpn2cb['instance'].isin(inst_t1), 'ghop'] = 'T1'
meta_vpn2cb.loc[meta_vpn2cb['instance'].isin(inst_t2), 'ghop'] = 'T2'

meta_type_vpn2cb['ghop'] = meta_type_vpn2cb['main_groups'].map({'BVNC': 'later', 'VPN': 'VPN'})
meta_type_vpn2cb.loc[meta_type_vpn2cb['instance'].isin(inst_t1), 'ghop'] = 'T1'
meta_type_vpn2cb.loc[meta_type_vpn2cb['instance'].isin(inst_t2), 'ghop'] = 'T2'

# %% [markdown]
# ### histo of vision score

# %%
# histogram of meta_vpn2cb['vision'] split by VPN vs CB, by bodyId
_col_main = "main_groups"
vpn_mask = meta_vpn2cb[_col_main] == 'VPN'
vals_vpn = meta_vpn2cb.loc[vpn_mask, 'vision'].dropna()
vals_cb = meta_vpn2cb.loc[~vpn_mask, 'vision'].dropna()

data_all = np.concatenate([vals_vpn.values, vals_cb.values])
bins = 500

plt.figure(figsize=(6,3.2))
plt.hist([vals_vpn, vals_cb], bins=bins, stacked=True, label=['VPN','CB'],
             color=['#1f77b4','#ff7f0e'], alpha=0.5)
# plt.hist([vals_other], bins=bins, color=['#ff7f0e'], alpha=0.5)
plt.xlabel('vision')
plt.ylabel('count')
plt.title("vision distribution ")
# plt.legend(frameon=False)
plt.yscale('log')
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.tight_layout()
# plt.xlim(0, 0.5)

print({'VPN_count': len(vals_vpn), 'CB_count': len(vals_cb)})

# SAVE
# plt.savefig(Path(result_dir, 'meta_vpn2cb_vision_hist.png'), dpi=300)

# %%
# scatter plot meta_vpn2cb['vision'] vs meta_vpn2cb['hitting_time'], color by main groups
fig_scatter = px.scatter(meta_type_vpn2cb, x='vision', y='ht', color='ghop', opacity=0.5, \
                         hover_data=['cell_type_side'])
fig_scatter.update_traces(marker=dict(size=2))
fig_scatter.update_layout(
    title='Vision vs Hitting Time',
    template='plotly_white',
    margin=dict(l=60, r=20, t=60, b=60),
)
fig_scatter.update_yaxes(range=[0, 10])
fig_scatter.update_layout(width=500, height=500)
fig_scatter.show()
# save html
# fig_scatter.write_html(Path(result_dir, 'meta_vpn2cb_vision_vs_ht.html'))

# %%
# histogram of meta_type_vpn2cb['vision'] split by VPN vs Other, by cell_type_side
vpn_mask = meta_type_vpn2cb["main_groups"] == 'VPN'
vals_vpn = meta_type_vpn2cb.loc[vpn_mask, 'vision'].dropna()
vals_other = meta_type_vpn2cb.loc[~vpn_mask, 'vision'].dropna()

data_all = np.concatenate([vals_vpn.values, vals_other.values])
bins =500

plt.figure(figsize=(6,3.2))
plt.hist([vals_vpn, vals_other], bins=bins, stacked=True, label=['VPN','Other'],
             color=['#1f77b4','#ff7f0e'], alpha=0.5)
plt.xlabel('vision')
plt.ylabel('count')
# plt.title("vision distribution ")
plt.yscale('log')
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.tight_layout()
# plt.xlim(0, 0.2)

print({'VPN_count': len(vals_vpn), 'Other_count': len(vals_other)})

# SAVE
# plt.savefig(Path(result_dir, 'meta_type_vpn2cb_vision_hist.png'), dpi=300)

# %%
# NB, stacked with log axis doesn't make sense

groups = ['VPN', 'T1', 'T2', 'later']
data_grouped = [meta_type_vpn2cb.loc[meta_type_vpn2cb['ghop'] == g, 'vision'].dropna().values
             for g in groups]
colors_map = {'VPN': '#1f77b4', 'T1': '#2ca02c', 'T2': '#d62728', 'later': '#9467bd'}
bins = 500

# Create plotly figure
fig = go.Figure()

# Add histogram traces for each group
for i, group in enumerate(groups):
    fig.add_trace(go.Histogram(
        x=data_grouped[i],
        name=group,
        nbinsx=bins,
        marker_color=colors_map[group],
        opacity=0.7,
        histnorm='',  # Use count
    ))

# Update layout
fig.update_layout(
    title="Distribution of Vision Values by Cell Type",
    xaxis_title="vision",
    yaxis_title="count",
    yaxis_type="log",
    barmode='stack',  # Stack the histograms
    width=800,
    height=400,
    showlegend=True,
    # xaxis_range=[0, 0.5],
    template="plotly_white",
    font=dict(size=12)
)

# Add grid with proper log scale grid lines
fig.update_xaxes(showgrid=True, gridwidth=0.3, gridcolor='lightgray')
fig.update_yaxes(
    minor=dict(
        showgrid=True,
        gridwidth=0.3,
        gridcolor='lightgray',
    )
)

# Show the plot
fig.show()

# %% [markdown]
# ## threshold: vision > 0.02 (lose LC9) or 0.05, ht ~5, etc

# %%
thr_vision = 0.015
thr_ht = 5

cell_type_side_kept = meta_type_cb_vpn.loc[(meta_type_cb_vpn['vision'] > thr_vision) &
                                           (meta_type_cb_vpn['ht'] <= thr_ht),
                                           'cell_type_side'].unique()
meta_cb_vpn_kept = meta_cb_vpn[meta_cb_vpn['cell_type_side'].isin(cell_type_side_kept)]

len(cell_type_side_kept), meta_cb_vpn_kept.shape

# %%
meta_type_t0t1t2 = meta_type_cb_vpn[meta_type_cb_vpn['ghop'].isin(['VPN','T1', 'T2'])].copy()
meta_type_t0t1t2 = meta_type_t0t1t2[meta_type_t0t1t2['col_count'] > 1]   

# # SAVE
# meta_type_t0t1t2.to_csv(Path(result_dir, 'meta_type_t0t1t2.csv'), index=False)

# # load
# meta_type_t0t1t2 = pd.read_csv(Path(result_dir, 'meta_type_t0t1t2.csv'), dtype={'bodyId': int})

# %%
# determine threshold for vision, by VPN or CB
vals = meta_type_vpn2cb.loc[meta_type_vpn2cb["main_groups"] == 'BVNC', 'vision']
# vals = meta_type_vpn2cb.loc[meta_type_vpn2cb["main_groups"] == 'VPN', 'vision']

# order vals by vision in descending order, compute cumsum and normalize by total sum
# and filter for those with cumsum < 0.95
vals_sorted = vals.sort_values(ascending=False)
vals_cumsum = vals_sorted.cumsum() / vals_sorted.sum()
# find the value corresponding to the 95th percentile
thr_vision = vals_sorted[vals_cumsum > 0.95].max()
# vals_sorted[vals < thr_vision]
thr_vision

# %%
# types with low vision score
cell_type_side_keep = meta_type_vpn2cb.loc[(meta_type_vpn2cb['vision'] > thr_vision), 'cell_type_side'].unique()
len(cell_type_side_keep)

# %% [markdown]
# ## DEBUG, LPC1
# 

# %%
meta_cb_vpn.loc[meta_cb_vpn['instance'].str.contains('^Mi9*_R$'), ['bodyId','instance','col_count','col_count_input','vision']]

# %%
search_str = '^LPC1.*_R$'
# search_str = '^.*_R$'

df1 = meta_cb_vpn.loc[meta_cb_vpn['instance'].str.contains(search_str), ['bodyId','instance','col_count','col_count_input','vision']]
df2 = rf_fit[rf_fit['instance'].str.contains(search_str)]

df_merged = pd.merge(df1, df2, on=['bodyId'], how='inner')
df_merged.head()

# %%
# scatter plot df_merged['area'] vs df_merged['col_count'], using plotly
fig_scatter = go.Figure(
    data=go.Scatter(
        x=df_merged['col_count'],
        y=df_merged['area'],
        mode='markers',
        marker=dict(size=10, color='blue', opacity=0.5),
    )
)
fig_scatter.update_layout(
    title='Column Count vs RF Area for LPC1_R',
    xaxis_title='Column Count',
    yaxis_title='RF Area (in hex units)',
    template='plotly_white',
    margin=dict(l=60, r=20, t=60, b=60),
)
# Fix axis ranges and 1:1 aspect
a = dict(scaleanchor='x', scaleratio=1, title='RF Area')
fig_scatter.update_yaxes(**a)
fig_scatter.update_layout(width=500, height=500)
fig_scatter.show()

# %%
meta[meta.cell_type_side.str.contains('^LPC1.*_right', regex=True)].cell_type_side.unique()

# %%
# DEBUG
inidx = meta.idx[meta.cell_type.isin(['L1', 'L2', 'L3', 'R7', 'R7d', 'R8', 'R8d'])] 
# outidx = meta.idx[meta.cell_type_side.isin(['LPC1_right'])] 
outidx = meta.idx[meta.cell_type_side.str.contains('^LPC1.*_right', regex=True)] 
# outidx = meta.idx[meta.bodyId == 45241] 
# outidx = meta.idx

df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

# %%
rf_fit[rf_fit['bodyId'] == 42330]['area']

# %%
count_col(effwt_visr['42330'].values, factor_thr=0.9)

# %%
np.sum(df['24162'] > 0)
# count_col(df['24162'].values, factor_thr=thr_2dGaussian)
# id = ids_to_bodyId[outidx]
# np.sum(effwt_visr[str(42330)].values - df['42330'].values)

# %%
1+6+12+18

# %%
ci.hex_heatmap(df)

# %% [markdown]
# ## vision score, vpn vs visual inputs

# %%
cell_type_nonOL = meta[~meta['cell_type'].isin(oltypes['cell_type'])].copy()
cell_type_nonOL = cell_type_nonOL[cell_type_nonOL['class'] != 'visual']

inidx = meta.idx[meta.cell_type.isin(oltypes_vpn['cell_type'])] 
outidx = meta.idx[meta.cell_type.isin(cell_type_nonOL['cell_type'])]

df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )
df.shape

# %%
cell_type_nonOL['vision'] = np.nan

thr_2dGaussian = 0.9
for i, bodyid in enumerate(cell_type_nonOL['bodyId']):
    values = df[str(bodyid)].values
    if values.sum() > 0:
        # vision score
        cell_type_nonOL.at[i, 'vision'] = values.sum()

# %%


# %% [markdown]
# # vision + resolution by roi
# 
# each roi collect cell_type with major input therein, and plot distr of col_count or col_count_input/col_count

# %%
# find rois_cb that don't contain '(L)' or '(R)'
[x for x in rois_cb if '(L)' not in x and '(R)' not in x]df['cell_type_side'] = df['cell_type_side'].str.replace('right', 'R', regex=False)df['cell_type_side'] = df['cell_type_side'].str.replace('left', 'L', regex=False)
df['cell_type_side'].head()

# %% [markdown]
# ## pre-syn, mean per roi

# %%
# meansyn = neuprint.fetch_mean_synapses(NC(bodyId = meta_cb_vpn['bodyId']), SC(rois=rois_cb, type='pre', primary_only=True), by_roi=True)

# SAVE 
# meansyn.to_csv(Path(result_dir, 'meta_cb_vpn_mean_preSyn.csv'), index=False)

# load
meansyn = pd.read_csv(Path(result_dir, 'meta_cb_vpn_mean_preSyn.csv'))
meansyn.shape, meansyn['bodyId'].nunique()

# %%
roi_col_count = pd.merge(meansyn[['bodyId', 'roi', 'count']], meta_cb_vpn_kept[['bodyId', 'cell_type_side', 'col_count', 'col_count_input','vision', 'main groups']], 
         on='bodyId', how='left')
print(roi_col_count.shape)
roi_col_count = roi_col_count[roi_col_count['vision'] > thr_vision]
print(roi_col_count.shape)

roi_col_count_tbar = roi_col_count.copy()

# %% [markdown]
# ###  PLOT aligned histograms for individual ROIs

# %%

rois = ['AOTU(R)', 'PLP(R)', 'PVLP(R)', 'SPS(R)', 'others']
fig, axes = plt.subplots(len(rois), 1, figsize=(4, 1.5*len(rois)), sharex=True, sharey=True)
for i, roi in enumerate(rois):
    ax = axes[i]
    if roi != 'others':
        df_roi = roi_col_count_tbar[roi_col_count_tbar['roi'] == roi]
        # remove rows with NaN in 'col_count'
        df_roi = df_roi[df_roi['col_count'].notna()]
        if len(df_roi) != 0:
            # expand by count
            expanded = np.repeat(df_roi['col_count'].to_numpy(), df_roi['count'].to_numpy())
    else:
        df_roi = roi_col_count_tbar[~roi_col_count_tbar['roi'].isin(rois[:-1])]
        # remove rows with NaN in 'col_count'
        df_roi = df_roi[df_roi['col_count'].notna()]
        if len(df_roi) != 0:
            # expand by count
            expanded = np.repeat(df_roi['col_count'].to_numpy(), df_roi['count'].to_numpy())

    ax.hist(expanded, bins=400, color='skyblue', edgecolor='black', alpha=0.7)
    # ax.set_title(f'{roi}')
    ax.text(0.02, 0.95, f'{roi}', transform=ax.transAxes, ha='left', va='top',
        fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    # ax.set_ylabel('tbar count')
    ax.grid(axis='y', alpha=0.75)
    # ax.set_yscale('log')
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 10000)
ax.set_xlabel('hex size')
plt.tight_layout()
plt.show()

# SAVE
plt.savefig(Path(result_dir, 'meta_cb_vpn_roi_tbar_hist.png'), dpi=300)



# %% [markdown]
# ## post-syn, mean per roi

# %%
# # meansyn = neuprint.fetch_mean_synapses(NC(bodyId = meta_cb['bodyId']), SC(rois=rois_cb, type='post', primary_only=True), by_roi=True)
# meansyn = neuprint.fetch_mean_synapses(NC(bodyId = meta_cb_vpn['bodyId']), SC(rois=rois_cb, type='post', primary_only=True), by_roi=True)

# SAVE 
# meansyn.to_csv(Path(result_dir, 'meta_cb_meansyn.csv'), index=False)
# meansyn.to_csv(Path(result_dir, 'meta_cb_vpn_mean_postSyn.csv'), index=False)

# load
# meansyn = pd.read_csv(Path(result_dir, 'meta_cb_meansyn.csv'))
meansyn = pd.read_csv(Path(result_dir, 'meta_cb_vpn_mean_postSyn.csv'))

meansyn.shape, meansyn['bodyId'].nunique()

# %%
roi_col_count = pd.merge(meansyn[['bodyId', 'roi', 'count']], meta_cb_vpn_kept[['bodyId', 'cell_type_side','col_count', 'col_count_input', 'vision', 'main groups']], 
         on='bodyId', how='left')
print(roi_col_count.shape)
roi_col_count = roi_col_count[roi_col_count['vision'] > thr_vision]
print(roi_col_count.shape)

roi_col_count_psd = roi_col_count.copy()

# %% [markdown]
# ## violin, combine tbar and psd

# %%
roi_col_count_tbar.shape, roi_col_count_psd.shape

# %%
print(roi_col_count_tbar['roi'].nunique(), roi_col_count_psd['roi'].nunique())
print(len(set(set(roi_col_count_tbar['roi']) | set(roi_col_count_psd['roi']))))

# %%
fig = go.Figure()
for roi in sorted(roi_col_count_psd['roi'].unique()):
    label = roi
    # - psd
    df_roi = roi_col_count_psd[roi_col_count_psd['roi'] == roi]
    # remove rows with NaN in 'col_count'
    df_roi = df_roi[df_roi['col_count'].notna()]
    color = 'black' if roi.endswith('(R)') else 'red' if roi.endswith('(L)') else 'blue'
    if len(df_roi) != 0:
        # expand by count
        expanded = np.repeat(df_roi['col_count'].to_numpy(), df_roi['count'].to_numpy())
        fig.add_trace(go.Violin(x=[label]*len(expanded), 
                                y=expanded,
                                side='positive',
                                spanmode='hard',
                                scalegroup='all',
                                # scalemode='width',
                                width = 0.6,
                                box_visible=False,
                                meanline_visible=False,
                                points = False,
                                line_color=color,
                                fillcolor=color,
                                # opacity=0.6,
                                name= f'{roi} T-bar',
                                showlegend=False
                            ))
    # - tbar
    df_roi = roi_col_count_tbar[roi_col_count_tbar['roi'] == roi]
    # remove rows with NaN in 'col_count'
    df_roi = df_roi[df_roi['col_count'].notna()]
    color = 'grey' if roi.endswith('(R)') else 'lightcoral' if roi.endswith('(L)') else 'lightblue'
    if len(df_roi) != 0:
        # expand by count
        expanded = np.repeat(df_roi['col_count'].to_numpy(), df_roi['count'].to_numpy())
        fig.add_trace(go.Violin(x=[label]*len(expanded), 
                                y=expanded,
                                side='negative',
                                spanmode='hard',
                                # scalegroup='all',
                                # scalemode='width',
                                width = 0.6,
                                box_visible=False,
                                meanline_visible=False,
                                points = False,
                                line_color=color,
                                fillcolor=color,
                                # opacity=0.6,
                                name= f'{roi} T-bar',
                                showlegend=False
                            ))
    
fig.update_layout(
    title="Column Count, tbar+psd, by ROI",
    xaxis_title="ROI",
    yaxis_title="Column Count",
    template='plotly_white',
    margin=dict(l=60, r=20, t=60, b=60),
)
# fig.update_layout(
#     # violinmode="group",     # side-by-side instead of overlay (optional)
#     violingap=0,            # gap between categories
#     violingroupgap=0        # gap within a category
# )

fig.update_layout(width=1800, height=500)
fig.show()

# SAVE
# fig.write_html(Path(result_dir, 'colCount_twosided_roi_vis5_csum90.html'))

# %%
fig = go.Figure()
for roi in ['PVLP(R)','PLP(L)']:
    label = roi
    # - psd
    df_roi = roi_col_count_psd[roi_col_count_psd['roi'] == roi]
    # remove rows with NaN in 'col_count'
    df_roi = df_roi[df_roi['col_count'].notna()]
    color = 'black' if roi.endswith('(R)') else 'red' if roi.endswith('(L)') else 'blue'
    if len(df_roi) != 0:
        # expand by count
        expanded = np.repeat(df_roi['col_count'].to_numpy(), df_roi['count'].to_numpy())
        fig.add_trace(go.Violin(x=[label]*len(expanded), 
                                y=expanded,
                                side='positive',
                                spanmode='hard',
                                scalegroup='all',
                                scalemode='count',
                                width = 0.6,
                                box_visible=False,
                                meanline_visible=False,
                                points = False,
                                line_color=color,
                                fillcolor=color,
                                # opacity=0.6,
                                name= f'{roi} T-bar',
                                showlegend=False
                            ))
    # - tbar
    df_roi = roi_col_count_tbar[roi_col_count_tbar['roi'] == roi]
    # remove rows with NaN in 'col_count'
    df_roi = df_roi[df_roi['col_count'].notna()]
    color = 'grey' if roi.endswith('(R)') else 'lightcoral' if roi.endswith('(L)') else 'lightblue'
    if len(df_roi) != 0:
        # expand by count
        expanded = np.repeat(df_roi['col_count'].to_numpy(), df_roi['count'].to_numpy())
        fig.add_trace(go.Violin(x=[label]*len(expanded), 
                                y=expanded,
                                side='negative',
                                spanmode='hard',
                                scalegroup='all',
                                scalemode='count',
                                width = 0.6,
                                box_visible=False,
                                meanline_visible=False,
                                points = False,
                                line_color=color,
                                fillcolor=color,
                                # opacity=0.6,
                                name= f'{roi} T-bar',
                                showlegend=False
                            ))
    
fig.update_layout(
    title="Column Count, tbar+psd, by ROI",
    xaxis_title="ROI",
    yaxis_title="Column Count",
    template='plotly_white',
    margin=dict(l=60, r=20, t=60, b=60),
)
# fig.update_layout(
#     # violinmode="group",     # side-by-side instead of overlay (optional)
#     violingap=0,            # gap between categories
#     violingroupgap=0        # gap within a category
# )

fig.update_layout(width=1800, height=500)
fig.show()

# %%
roi = 'PVLP(R)'
roiname = roi
bins = np.linspace(0, 800, 401)

# PSD
df_psd = roi_col_count_psd[(roi_col_count_psd['roi'] == roiname) & (roi_col_count_psd['col_count'].notna())]
vals_psd = df_psd['col_count'].to_numpy()
w_psd = df_psd['count'].to_numpy()

# T-bar
df_tbar = roi_col_count_tbar[(roi_col_count_tbar['roi'] == roiname) & (roi_col_count_tbar['col_count'].notna())]
vals_tbar = df_tbar['col_count'].to_numpy()
w_tbar = df_tbar['count'].to_numpy()

plt.figure(figsize=(6,3.2))
plt.hist([vals_psd, vals_tbar],
         bins=bins,
         weights=[w_psd, w_tbar],
         stacked=True,
         label=['PSD', 'T-bar'],
         color=['#1f77b4', '#ff7f0e'],
         alpha=0.7)
plt.title(f'Column Count Histogram for {roiname}')
plt.xlabel('Column Count')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()



# %%
# df = pd.merge(roi_col_count_tbar[roi_col_count_tbar['roi'] == 'PVLP(R)'], meta_cb_vpn[['bodyId','instance']], on='bodyId', how='left')
df = roi_col_count_psd[roi_col_count_psd['roi'] == 'PVLP(L)'].copy()
df.groupby('cell_type_side').agg(
    bodyId_count=('bodyId','count'),
    count =('count', 'sum'),
    col_count=('col_count', 'mean'),
    col_count_input=('col_count_input', 'mean'),
    main_groups = ('main groups', 'first')
).reset_index().sort_values(by='count', ascending=False).head(20)

# %% [markdown]
# ## avg count per roi

# %%
# a funciton to compute area of an ellipse given a and b
def ellipse_area(a, b):
    return np.pi * a * b
# compute area in units of hex area
def ellipse_area_in_hex(a, b):
    hex_area = 1 / np.sqrt(3)  # area of hexagon with side length 1
    return ellipse_area(a, b) / hex_area

# %%
roi_col_avg = roi_col_count.copy()
cols = ['col_count', 'col_count_input', 'col_count_ratio']
roi_col_avg[cols] = roi_col_avg[cols].fillna(0)

# %%
# normalized count within each ROI
roi_col_avg['count_norm'] = roi_col_avg.groupby('roi')['count'].transform(lambda s: s / s.sum())
roi_col_avg = roi_col_avg.groupby('roi').agg(
        col_count=('col_count', lambda s: np.average(s, weights=roi_col_avg.loc[s.index, 'count_norm'])),
        col_count_input=('col_count_input', lambda s: np.average(s, weights=roi_col_avg.loc[s.index, 'count_norm'])),
        col_count_ratio=('col_count_ratio', lambda s: np.average(s, weights=roi_col_avg.loc[s.index, 'count_norm']))
    ).reset_index()
roi_col_avg.shape

# %%
# plot roi_col_avg['roi'] vs roi_col_avg['col_count'] 
fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(roi_col_avg['roi'], roi_col_avg['col_count'])
ax.set_xlabel('ROI')
ax.set_ylabel('Column Count')
ax.set_title('Column Count by ROI')

# rotate x labels
plt.xticks(rotation=60)
plt.tight_layout()
# tick font size
ax.tick_params(axis='both', which='major', labelsize=8)
plt.show()

# %% [markdown]
# # rf

# %%
_other_side = 'l' if SIDE_CHAR == 'r' else 'r'
stepsn_left = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_{_other_side}_lat_flow_sum.npz')
stepsn_left.shape

# %% [markdown]
# ## prop rf

# %%
# DEBUG
# inidx = meta.idx[meta.cell_type.isin(['L1', 'L2', 'L3', 'R7', 'R7d', 'R8', 'R8d'])] 
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
# inidx = meta.idx[meta.cell_type.isin(['L1', 'L2', 'L3'])] 
outidx = meta.idx[meta.cell_type_side.isin(['AMMC002_L'])] 
# outidx = meta.idx[meta.cell_type_side.str.contains('^MeVP43_R', regex=True)] 
# outidx = meta.idx[meta.bodyId.isin([512514, 519140])] 
# outidx = meta.idx[meta.bodyId == 28862]

df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

# %%
# ci.plot_mollweide_projection(df)
fig = ci.hex_heatmap(df)
# ci.hex_heatmap(effwt_visr[str(meta.bodyId[meta['cell_type_side'] == 'Tm5b_right'].values[6])])
fig
# save
# fig.write_image(Path(result_dir, 'hex_map.pdf'))

# %%
1+6+12+18

# %% [markdown]
# ## path

# %%
# inidx = meta.idx[meta.cell_type.isin(['L1', 'L2', 'L3', 'R7', 'R7d', 'R8', 'R8d'])] 
# inidx = meta.idx[meta.instance.isin(['L1_L', 'L2_L', 'L3_L', 'R7_L', 'R8_L'])] 
# inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R', 'R7d_R', 'R8d_R'])] 
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])]

# outidx = meta.idx[meta.cell_type_side.str.contains('^TuBu07_SBU_R', regex=True)] 
# outidx = meta.idx[meta.cell_type_side.str.contains('^PVLP008_a.*_right', regex=True)] 
# outidx = meta.idx[meta.cell_type_side.str.contains('^TuBu06_SBU_R', regex=True)] 
outidx = meta.idx[meta.bodyId.isin([148507])] 
# outidx = meta.idx[meta.bodyId.isin([27582])] 

# %%
paths = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 3)
paths = ci.group_paths(paths, idx_to_type_side, idx_to_type_side)
# paths = ci.group_paths(paths, pre_group= idx_to_bodyId, post_group= idx_to_type)
len(paths)

# %%
# idx_inter = meta.idx[meta.cell_type_side.str.contains('^LC6_right')].values

# paths_filtered = ci.filter_paths(paths, threshold=0.012, necessary_intermediate={3: ['LC6_right']})
paths_filtered = ci.filter_paths(paths, threshold=0.002)
ci.plot_layered_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %%
# find paths of fixed length with all direct weights above weight_thre, x-axis is number of hops
nr_hops = [1, 2, 3]
weight_thre = [0.001, 0.01, 0.01]

# find paths iteratively and collect them
paths_collection = pd.DataFrame()
for nr_hop, thre_layer in zip(nr_hops, weight_thre):
    paths = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= nr_hop)
    if (paths is None) or (paths.empty):
        print(f'No paths with {nr_hop} hops')
        continue
    paths = ci.group_paths(paths, idx_to_bodyId_cellType, idx_to_bodyId_cellType, combining_method='mean')
    paths = ci.filter_paths(paths, threshold = thre_layer)
    if (paths is None) or (paths.empty):
        print(f'No paths with {nr_hop} hops and threshold {thre_layer}')
        continue
    paths_collection = pd.concat([paths_collection, paths], axis=0)

    ci.plot_layered_paths(paths, 
                       neuron_to_sign = type_to_nt,  
                       sign_color_map = sign_to_color,
                       node_size = 1000, 
                       edge_text = True, 
                       weight_decimals = 2,
                       figsize=(4+nr_hop*3,10))

# %%
# ord by ht
# plots the same paths but across all hops specified above where the x-axis is the flow layer

# attach flow layer information to collected paths
conn_paths_df = paths_collection[['pre','post','weight']].drop_duplicates()
conn_paths_df = conn_paths_df.merge(
    meta[['cell_type_side','hitting_time']].drop_duplicates(), how='left', left_on='pre', right_on='cell_type_side').rename(
    columns={'hitting_time':'pre_layer'}).drop(columns=['cell_type_side']).merge(
    meta[['cell_type_side','hitting_time']].drop_duplicates(), how='left', left_on='post', right_on='cell_type_side').rename(
    columns={'hitting_time':'post_layer'}).drop(columns=['cell_type_side'])

ci.plot_flow_layered_paths(conn_paths_df,
                        neuron_to_sign = type_to_nt,  
                        # sign_color_map = nt_to_color,
                        interactive = False,
                        node_size = 1000,
                        # weight_decimals = 2,
                        edge_text = False,
                        # highlight_nodes = output_types,
                        # file_name = results_folder / f'{dataset}_inputs_right_layered_paths'
                        )

# %% [markdown]
# ## elliptical fit

# %%
inidx = meta_OL[np.isin(meta_OL.instance, in_instances)].idx.values   
outidx = meta_OL[meta_OL.bodyId==example_bid].idx.values   
df = compute_rf(meta_OL, stepsn_OL, inidx, outidx)

# %%
inidx = meta.idx[meta.cell_type.isin(['L1', 'L2', 'L3', 'R7', 'R7d', 'R8', 'R8d'])] 
# outidx = meta.idx[meta.instance.str.contains('^TuBu04_IBU_R', regex=True)] 
outidx = meta.idx[meta.instance.str.contains('^aMe12_R', regex=True)] 

df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

ci.hex_heatmap(df, custom_colorscale=[[0, "rgb(255, 255, 255)"], [1, "rgb(200, 20, 0)"]], global_min=0)

# %%
from utils.external_rf import twoD_Gaussian
from utils.computing_functions import compute_rf
from utils.plotting_functions import plot_gaussian_params

# %%
# todo, not working


df = compute_rf(meta, stepsn, inidx, outidx)
xy_coord = df['coords'].str.split(',', expand=True).astype(float).values
df['x'] = xy_coord[:,0]
df['y'] = np.sqrt(3) * xy_coord[:,1]
df['coords'] = df['x'].astype(str) + ',' + df['y'].astype(str)
plot_df = df.set_index('coords')[['effective weight']]
tot_max = np.abs(plot_df['effective weight']).max()

fig = ci.hex_heatmap(plot_df,
                  custom_colorscale='reds',
                  global_min=0, 
                #   global_max=tot_max,
                #   show_background=True, norm=True
                )
# fig.show()



# %%

bid = 25290
params_single_df = rf_fit.loc[rf_fit.bodyId== bid]
# params_single_df = df_fit.loc[df_fit[df_fit.bodyId==bid].index.values[0]]

# rf_fitted = twoD_Gaussian(df['x'].values, df['y'].values, *(rf_fit.loc[['x0','y0','a','b','phi']]))
# fit_df = pd.DataFrame(rf_fitted*tot_max, index=df['coords'].unique(), columns=['fit'])

# fig = ci.hex_heatmap(fit_df,
#                   custom_colorscale='Rd_r',
#                 #   global_min=-tot_max, 
#                 #   global_max=tot_max,
#                   show_background=True, norm=True)
# fig1 = plot_gaussian_params(params_single_df, example_bid= id)
fig1 = plot_gaussian_params(params_single_df, example_bid=bid, fac=np.sqrt(3))
# fig.add_trace(fig1.data[0])
for trace in fig1.data[1:]:
    fig.add_trace(trace)
fig

# %%
fig1.data

# %%
# rf_fit
# df_fit.loc[df_fit[df_fit.bodyId==bid].index.values[0]]
params_single_df.T

# %% [markdown]
# ## level contour

# %%
# level contour
# df = df['19917']
percentile = 70

coords = []
values = []

for idx in df.index:
    if isinstance(idx, str) and ',' in idx:
        x, y = map(float, idx.split(','))
        coords.append([x, y])
        values.append(df.loc[idx])

coords = np.array(coords)
values = np.array(values)

# Remove zero/nan values
mask = (values > 0) & ~np.isnan(values)
coords = coords[mask,: ]
values = values[mask]

# if len(coords) == 0:
#     return None, None

# Calculate the threshold value
threshold = np.percentile(values, percentile)

# Create a regular grid for interpolation
x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

xi = np.linspace(x_min, x_max, 100)
yi = np.linspace(y_min, y_max, 100)
xi_grid, yi_grid = np.meshgrid(xi, yi)

# Interpolate values onto the grid
zi = griddata(coords, values, (xi_grid, yi_grid), method='cubic', fill_value=0)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the filled contour
contour_filled = ax.contourf(xi_grid, yi_grid, zi, levels=20, cmap='viridis', alpha=0.7)

# Plot the specific level line
level_line = ax.contour(xi_grid, yi_grid, zi, levels=[threshold], colors='red', linewidths=2)
# ax.clabel(level_line, inline=True, fontsize=10, fmt=f'{percentile}%')

# Plot original data points
scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=20, cmap='viridis', edgecolors='black', linewidth=0.5)

# Add colorbar
cbar = plt.colorbar(contour_filled, ax=ax)
cbar.set_label('Weight Values')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')

# asp ratio 1:1
ax.set_box_aspect(1)
ax.set_title(f'Level Plot with {percentile}% Contour Line')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# save
# plt.savefig(Path(result_dir, 'level_plot_PVLP008_a1_right_70604.pdf'))
# plt.savefig(Path(result_dir, 'level_plot_ER2_right_19917.pdf'))

# %% [markdown]
# ## by input syn

# %%
# filter for cell types with a negative area_fit_lmr production
meta_cb_t1.groupby('cell_type').filter(lambda x: x['area_fit_lmr'].prod() < 0)

# %%


# %%


# %%
# # load data
# ucl_hex = pd.read_pickle(EYEMAP_DIR / 'mcns_20240701' / 'ucl_hex_right.pkl')

# load xlsx
ucl_hex = pd.read_excel(EYEMAP_DIR / 'mcns_20240701' / 'pqxyztp_right.xlsx') # right
# ucl_hex = pd.read_excel(EYEMAP_DIR / 'mcns_20240701' / 'pqxyztp_left.xlsx') # left
ucl_hex['hex1'] = ucl_hex['q'] + 18
ucl_hex['hex2'] = ucl_hex['p'] + 19

# %%


# %%
from utils.ol_rf import hexw_columnar
from utils.geometry import sph2Mollweide, sph2Mercator
from utils.geometry import cart2sph, sph2cart
from utils.plotting_functions import plt_mollweide, plt_mercator

# %%


# %%
# Molleweide projection guidelines

# convert to spherical coordinates
rtp2 = cart2sph(ucl_hex[['x','y','z']].values) 

xy = sph2Mollweide(rtp2[:,1:3])
xy[:,0] = -xy[:,0] # flip x axis

xyhex_moll = np.concatenate((xy, ucl_hex[['hex1','hex2']].values), axis=1)
# convert to df and change type of the last 2 columns to int
xyhex_moll = pd.DataFrame(xyhex_moll, columns=['x','y','hex1','hex2'])
xyhex_moll[['hex1','hex2']] = xyhex_moll[['hex1','hex2']].astype(int)

# %%


# %%
# Mercator projection guidelines

# convert to spherical coordinates
rtp2 = cart2sph(ucl_hex[['x','y','z']].values) 

xy = sph2Mercator(rtp2[:,1:3])
xy[:,0] = -xy[:,0] # flip x axis

xyhex_merc = np.concatenate((xy, ucl_hex[['hex1','hex2']].values), axis=1)
# convert to df and change type of the last 2 columns to int
xyhex_merc = pd.DataFrame(xyhex_merc, columns=['x','y','hex1','hex2'])
xyhex_merc[['hex1','hex2']] = xyhex_merc[['hex1','hex2']].astype(int)

# %%
# syn = neuprint.fetch_synapses(NC(bodyId=45241), SC(type='post', primary_only=False))
# syn = syn[syn['roi'].str.contains('LOP_R_col', na=False)]
# # find the 2 numbers after LOP_R_col in syn['roi'], name the col hex1 and hex2
# syn[['hex1', 'hex2']] = syn['roi'].str.extract(r'LOP_R_col_(\d+)_(\d+)')
# # syn = syn[syn['hex1'].notna() & syn['hex2'].notna()]

# %%
ids = [45241]
rf_lst = []
for i in range(len(ids)):
    hexw = hexw_columnar(ids[i], roi_str="LOP(R)") # use the appriroate roi_str
    rf_lst.append(hexw)

# %%
# one cell
df = pd.merge(xyhex_moll, rf_lst[0], on=['hex1', 'hex2'], how='left')
df['wt'] = df['wt'].fillna(0)
df['wt'] = df['wt'] / df['wt'].max()

# Mollweide
rf_moll = df.copy()
# Mercator
rf_merc = rf_moll.copy()
rf_merc[['x','y']] = xyhex_merc[['x','y']]

# %%


# %%
# Molleweide version
fig, ax = plt_mollweide()

ax.scatter(rf_moll['x'].values, rf_moll['y'].values, c=rf_moll['wt'].values, 
            cmap="Reds",vmin=0,vmax=1, s=30)
# add colorbar
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="Reds"), ax=ax)
cbar.set_label('normalized synapse count')

# change size
fig.set_size_inches(12,6)

# # title
# # ax.set_title(f"RF of {target_inst}_{target_id[i]}")
# fig.savefig(CACHE_DIR / 'rf' / f"RF_sum_{inst}_Moll.png")

# %%


# %% [markdown]
# # End


