# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: ol-analysis
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### collaboration between VPNs

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
# ## vpn types

# %%
# cell type table xlsx
oltypes = pd.read_excel(PARAMS_DIR / 'Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx')
# rename columns
oltypes = oltypes.rename(columns={'cell type': 'cell_type', 'main groups': 'main_groups'})
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
# ## Get meta

# %%
# this has AN's R7/8, but R7d/R8d have no coords
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
ht = pd.read_csv(DATA_DIR / f'{DATASET}_{SIDE_CHAR}_flow_{N_FLOW}step_{HIT_THRE}thre_hit.csv')

ht = ht.rename(columns={'hitting_time': 'ht'})
ht = pd.merge(ht, meta_judith[['bodyId', 'idx']], on='idx', how='left') # add bodyId

ht.shape

# %%
# use Judith's
# meta = pd.merge(meta_judith, meta1[['bodyId', 'superclass', 'class']], left_on='bodyId', right_on='bodyId', how='left')
meta = pd.merge(meta_judith, ht[['bodyId', 'ht']], on='bodyId', how='left')
# meta = pd.merge(meta_judith, ht[['idx', 'hitting_time']], on='idx', how='left')

# %%


# %% [markdown]
# # query vpns and connections in CB

# %%
n_vpn, _ = fetch_neurons(NC(type= oltypes_vpn['cell_type']))
n_vpn = n_vpn[['bodyId', 'type', 'instance', 'consensusNt', 'flywireType', 'somaSide', 'downstream', 'upstream']]
n_vpn.shape, n_vpn.instance.nunique()


# %%
neuron_df, connection_df = fetch_adjacencies(
    sources= n_vpn['bodyId'],
    targets= None,
    rois= rois_cb,
    min_total_weight=1
)
conn_t0t1 = neuprint.merge_neuron_properties(neuron_df, connection_df, ['type', 'instance'])\
    .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
    .agg({'weight': 'sum'})\
    .reset_index()

# %%
# get instance_post but remove those with apostophe or ? or +, and exclude vpn and T1
conn_t0t1 = conn_t0t1[~conn_t0t1['instance_post'].str.contains("\\'") & 
                    ~conn_t0t1['instance_post'].str.contains("\\?") &
                    ~conn_t0t1['instance_post'].str.contains("\\+") &
                    ~conn_t0t1['instance_post'].isin(oltypes['instance']) ]

# %%
inst_t0 = conn_t0t1['instance_pre'].unique()
inst_t0 = sorted(inst_t0)

# remove t0-t0 conn, 
conn_t0t1 = conn_t0t1[~conn_t0t1['instance_post'].isin(inst_t0)]
inst_t1 = conn_t0t1['instance_post'].unique()
inst_t1 = sorted(inst_t1)

# then groupby instance
conn_inst = conn_t0t1.groupby(['instance_pre','instance_post']).agg({'weight':'sum'}).reset_index()

len(inst_t0), len(inst_t1), conn_inst.shape

# %%
set(n_vpn.instance).difference(set(inst_t0)), set(inst_t0).difference(set(n_vpn.instance))


# %%
# set(inst_t0).intersection(set(inst_t1)), set(inst_t1).intersection(set(inst_t0))

# %% [markdown]
# # instance-collaboration

# %% [markdown]
# ## build matrix

# %%
# collaboration matrix between inst_t0, rows label input connections

# Initialize the matrix
collab_matrix = np.zeros((len(inst_t0), len(inst_t0)))

# For each pair of inst_t0 nodes, calculate their collaboration score
for j, inst_j in enumerate(inst_t0):
    inst2 = conn_inst.loc[conn_inst['instance_pre'] == inst_j, 'instance_post'].unique()
    for i, inst_i in enumerate(inst_t0):
        el = conn_inst[(conn_inst['instance_pre'] != inst_j)
                       & (conn_inst['instance_pre'] == inst_i)
                       & (conn_inst['instance_post'].isin(inst2))]
        
        collab_matrix[i, j] = el['weight'].sum()

# Convert to DataFrame for easier handling
collab_df = pd.DataFrame(
    collab_matrix,     index=inst_t0,   columns=inst_t0
)

print(f"Collaboration matrix shape: {collab_df.shape}")
print(f"Total collaboration score: {collab_matrix.sum()}")
print(f"Non-zero entries: {np.count_nonzero(collab_matrix)}")

# %%
# save pickle
# collab_df.to_pickle(Path(cache_dir, 'vpn_collab_matrix.pkl'))

# load pickle
collab_df = pd.read_pickle(Path(cache_dir, 'vpn_collab_matrix_noVIC.pkl'))
collab_df.shape

# %% [markdown]
# ## groupby clusters

# %%
# VPN cluster
vpn_clu = pd.read_csv(CACHE_DIR / f'{DATASET}_in_out_clusters.csv')

# %%
inst_keep = [inst for inst in collab_df.index if inst in vpn_clu['instance'].values]
collab_df = collab_df.loc[inst_keep, inst_keep]
collab_df.shape

# %%
# make a new matrix  based on cluster assignement of the rows/columns

nclusters = vpn_clu['cluster'].nunique()

# Initialize the matrix
collab_clu = np.zeros((nclusters, nclusters))

# sum up rows whose indices are in the same cluster
for clu in sorted(vpn_clu['cluster'].unique()):
    inst_row = vpn_clu[vpn_clu['cluster'] == clu]['instance'].values
    if len(inst_row) > 0:
        # find rows whose indices are in inst_row, and sum them
        rows_summed = collab_df.loc[collab_df.index.isin(inst_row)].sum(axis=0)
        for clu2 in sorted(vpn_clu['cluster'].unique()):
            inst_col = vpn_clu[vpn_clu['cluster'] == clu2]['instance'].values
            if len(inst_col) > 0:
                collab_clu[clu-1, clu2-1] = rows_summed.loc[rows_summed.index.isin(inst_col)].values.sum()

# Convert to DataFrame for easier handling
collab_clu = pd.DataFrame(
    collab_clu,     index=sorted(vpn_clu['cluster'].unique()),   columns=sorted(vpn_clu['cluster'].unique())
)

# %%
# plot a heatmap of collab_clu, log-scale 
fig = px.imshow(np.log10(collab_clu), text_auto=True, color_continuous_scale='Viridis')
fig.show()


# %% [markdown]
# ### normalize by vpn count

# %%
meta2 = pd.merge(meta[['bodyId', 'instance', ]], vpn_clu, on='instance',how='left')
meta2 = meta2[meta2['cluster'].notna()]
meta2 = meta2.groupby(['cluster']).size().reset_index(name='count')

# %%
collab_clu_norm = collab_clu.copy()
for clu in collab_clu_norm.index:
    count = meta2.loc[meta2['cluster'] == clu, 'count'].values[0]
    if count > 0:
        collab_clu_norm.loc[clu, :] = collab_clu_norm.loc[clu, :] / count

# %%
# plot a heatmap of collab_clu, log-scale 
fig = px.imshow(np.log10(collab_clu_norm), text_auto=True, color_continuous_scale='Viridis')
fig.show()


# %% [markdown]
# ### spectral clustering

# %%
# symmetrize
A = (collab_clu + collab_clu.T) / 2
# normalize
A = A / A.values.max()
# degree matrix
D = np.diag(A.sum(axis=1))
# Laplacian matrix, normalized and symmetric
L = np.linalg.inv(D)**0.5 @ A.values @ np.linalg.inv(D)**0.
# eigen decomposition
eigenvalues, eigenvectors = np.linalg.eigh(L)

# %%
# plot eigenvalues
plt.figure(figsize=(6, 3))
plt.plot(eigenvalues, marker='o')
# plt.xlim(200, 370)
# plt.ylim(0.0, 1.1)
plt.show()

# %%
# spectral clustering
from sklearn.cluster import KMeans
n_clu = 2
kmeans = KMeans(n_clusters= n_clu, n_init=500)
kmeans.fit(eigenvectors[:, -n_clu:])  # use the last k eigenvectors  
labels = kmeans.labels_

len(labels)

# %%
# combine inst_t0 and labels into a dataframe
clu_clu = pd.DataFrame({'clu': sorted(vpn_clu['cluster'].unique()), 'label': labels})
clu_clu
# clu_clu['cell_type'] = clu_clu['clu'].astype(str).str.replace(r'_[LR]$', '', regex=True)

# %% [markdown]
# ## right side VPN

# %%
# filtering
# keep only instances present in the matrix to avoid KeyError
inst_keep = [inst for inst in collab_df.index if inst in oltypes['instance'].values]
collab_df = collab_df.loc[inst_keep, inst_keep]
collab_df.shape

# %% [markdown]
# ### spectral clustering
#
# https://ai.stanford.edu/~ang/papers/nips01-spectral.pdf

# %%
# symmetrize
A = (collab_df + collab_df.T) / 2

# normalize
A = A / A.values.max()

# degree matrix
D = np.diag(A.sum(axis=1))

# Laplacian matrix, normalized and symmetric
L = np.linalg.inv(D)**0.5 @ A.values @ np.linalg.inv(D)**0.5

# eigen decomposition
eigenvalues, eigenvectors = np.linalg.eigh(L)

# %%
# plot eigenvalues
plt.figure(figsize=(6, 3))
plt.plot(eigenvalues, marker='o')
# plt.xlim(330, 370)
plt.ylim(0.1, 1.1)
plt.show()

# %%
# spectral clustering with k=4
from sklearn.cluster import KMeans
n_clu = 8
kmeans = KMeans(n_clusters= n_clu, n_init=500)
kmeans.fit(eigenvectors[:, -n_clu:])  # use the last k eigenvectors  
labels_8 = kmeans.labels_

n_clu = 4
kmeans = KMeans(n_clusters= n_clu, n_init=500)
kmeans.fit(eigenvectors[:, -n_clu:])  # use the last k eigenvectors  
labels_4 = kmeans.labels_

len(labels_4)

# %%
# combine inst_t0 and labels into a dataframe
inst_clu = pd.DataFrame({'instance': inst_keep, 'label_4': labels_4, 'label_8': labels_8})

inst_clu['cell_type'] = inst_clu['instance'].str.replace(r'_[LR]$', '', regex=True)

# %% [markdown]
# ## compare to VIC-based clustering

# %%
vic_clu = pd.read_csv(CACHE_DIR / f'{DATASET}_in_out_clusters.csv')
sorted(vic_clu.cluster.unique())

# %%
clu_comp = pd.merge(inst_clu, vic_clu, on='instance', suffixes=('_inst', '_vic'))
# sort
# clu_comp = clu_comp.sort_values(by=['cluster', 'instance']).reset_index(drop=True)
clu_comp = clu_comp.sort_values(by=['label_8', 'label_4', 'instance']).reset_index(drop=True)
# reorder columns
clu_comp = clu_comp[[ 'cell_type', 'instance', 'label_8', 'label_4', 'cluster']]
# clu_comp.label_8.value_counts()

# rename label_8 based on value_counts() in descending order
label_mapping = {old_label: new_label for new_label, old_label in enumerate(clu_comp.label_8.value_counts().index)}
clu_comp['label_8'] = clu_comp['label_8'].map(label_mapping)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(clu_comp.label_4.value_counts().index)}
clu_comp['label_4'] = clu_comp['label_4'].map(label_mapping)


# %%
# save
# clu_comp.to_csv(Path(result_dir, 'vpn_cluster_comparison.csv'), index=False)

# # load
clu_comp = pd.read_csv(Path(result_dir, 'vpn_cluster_comparison.csv'))

# %%
# clu_comp[['label','cluster']].value_counts().sort_index()
clu_comp[['label_8','label_4', 'cluster']].value_counts().sort_index()

# %%
# clu_comp[['label','cluster']].value_counts().sort_index()
clu_comp[['label_8','label_4']].value_counts().sort_index()

# %% [markdown]
# ## plot matrix as heatmap

# %%
df_plt = collab_df.copy()

# # reorder rows and columns by clu_comp['instance']
# df_plt = df_plt.loc[clu_comp['instance'], clu_comp['instance']]


# %%
# plot df_plt as heatmap with log color scale
_z = df_plt.values.astype(float)
_z_log = np.log10(_z + 1.0)  # log10(1 + z) to handle zeros
_zmax_log = float(_z_log.max())

# colorbar ticks in linear space, mapped to log scale
if _z.size and np.nanmax(_z) > 0:
    _zmax = float(np.nanmax(_z))
    _max_pow = int(np.floor(np.log10(_zmax)))
    _tick_vals_lin = [0] + [10**p for p in range(0, _max_pow + 1)]
else:
    _tick_vals_lin = [0, 1]
_tick_vals_log = np.log10(np.array(_tick_vals_lin) + 1.0)
_tick_text = [f"{v:,}" for v in _tick_vals_lin]

fig = go.Figure(
    data=go.Heatmap(
        z=_z_log,
        x=df_plt.columns,
        y=df_plt.index,
        colorscale="Viridis",
        colorbar=dict(
            title="Collaboration Score (log10)",
            tickvals=_tick_vals_log,
            ticktext=_tick_text,
        ),
        zmin=0.0,
        zmax=_zmax_log,
        customdata=_z,  # show original values on hover
        hovertemplate="pre=%{y}<br>post=%{x}<br>score=%{customdata}<extra></extra>",
    )
)
fig.update_layout(
    title="Collaboration matrix",
    xaxis_title="instance_post",
    yaxis_title="instance_pre",
    width=1010,
    height=800,
    margin=dict(l=0, r=0, t=0, b=0),  # Smaller, more balanced margins
    # plot_bgcolor='white',  # Set background color to white
    paper_bgcolor='white'  # Set paper background to white
)
# start from upper-left and put x-axis on top
fig.update_yaxes(
    autorange="reversed", 
    categoryorder="array", 
    categoryarray=df_plt.index,
    scaleanchor="x",  # Lock aspect ratio to x-axis
    scaleratio=1      # 1:1 ratio
)
fig.update_xaxes(
    side="top", 
    categoryorder="array", 
    categoryarray=df_plt.columns
)

fig.show()

# %% [markdown]
# ## right + left

# %% [markdown]
# # opponancy matirx
#
# simple motif: (+t0)x(-t0), (+t0)x(+t0-t1)

# %%
n_vpn.head()

# %%
# n_vpn['consensusNt]'].value_counts(dropna=False)

# %%
# def n_vpn['polarity'] to be -1 if GABA or Glutamine, else +1
n_vpn['polarity'] = n_vpn['consensusNt'].apply(lambda x: -1 if x in ['gaba', 'glutamate'] else 1)

# %%
inst_exci = n_vpn[n_vpn['polarity'] == 1]['instance'].unique()
inst_inhi = n_vpn[n_vpn['polarity'] == -1]['instance'].unique()
len(inst_exci), len(inst_inhi)

# %%
# does all inst_inhi contain '_R' or '_L' at the end?
all(re.search(r'_(R|L)$', inst) for inst in inst_inhi)

# %%
# separate inst_inhi into those containing '_R' or '_L' or neither at the end
inst_inhi_R = [inst for inst in inst_inhi if re.search(r'_R$', inst)]
inst_inhi_L = [inst for inst in inst_inhi if re.search(r'_L$', inst)]
inst_inhi_none = [inst for inst in inst_inhi if not re.search(r'_(R|L)$', inst)]

inst_exci_R = [inst for inst in inst_exci if re.search(r'_R$', inst)]
inst_exci_L = [inst for inst in inst_exci if re.search(r'_L$', inst)]
inst_exci_none = [inst for inst in inst_exci if not re.search(r'_(R|L)$', inst)]

# %%
len(inst_inhi_R), len(inst_inhi_L), len(inst_inhi_none), len(inst_exci_R), len(inst_exci_L), len(inst_exci_none)

# %%
# what's in inst_inhi_R but not in inst_inhi_L after removing _R and _L
set([re.sub(r'_R$', '', inst) for inst in inst_inhi_R]) - set([re.sub(r'_L$', '', inst) for inst in inst_inhi_L])

# %%


# %% [markdown]
# ## load meta_type_cb_vpn with  ghop

# %%
meta_type_cb_vpn = pd.read_pickle(Path(cache_dir, 'meta_type_cb_vpn_ghop.pkl'))

# %% [markdown]
# ## (+t0)x(+t0)

# %%
# find  instance_post in conn_inst that have instance_pre in both inst_exci and inst_inhi
inst_post_both = set(conn_inst[conn_inst['instance_pre'].isin(inst_exci)]['instance_post']).intersection(
    set(conn_inst[conn_inst['instance_pre'].isin(inst_inhi)]['instance_post'])
)

conn_pt0pt0 = conn_inst[conn_inst['instance_post'].isin(inst_post_both)]

# %%
conn_pt0pt0.shape

# %%
conn_inst.shape

# %%


# %% [markdown]
# # End

# %%



