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

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### spatial info in CB
#
#  Using VPNs to access the retinotopy at the input side in OL and at the output side in CB. Assign Retinotopy Index to cell type's outgoing synapses as a way to characterize the amount of retinotopy in regions of CB. Why some regions have higher RIs?
#  
# Slightly weaker version of this is spatial resolution in terms of RF's size or column counts. High vs low res neurons/regions. 

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
import json
import pickle
import alphashape
from shapely.geometry import LineString

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
from quan_propagation.func import disk_to_lab, lab_to_rgb, ray_intersection_with_boundary, count_col, hex2cielab
from quan_propagation.func_retinotopy import topographic_product, topographic_product_pvalue, RI_sets, RI_rois

# %%
from utils.config import CACHE_DIR, DATA_DIR, DATASET, FIG_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

result_dir = FIG_DIR / 'quan_propagation'
result_dir.mkdir(parents=True, exist_ok=True)

cache_dir = CACHE_DIR / 'quan_propagation'
cache_dir.mkdir(parents=True, exist_ok=True)

# %%

# %% [markdown]
# # test

# %%
map_coords = np.array([[0, 0],
              [3., 0],
              [2., 0]])
feature_coords = np.array([[0 ,0],
              [2.1, 0],
              [3.3, 0]])

# %%
from scipy.spatial.distance import cdist
n = len(map_coords)
D_map = cdist(map_coords, map_coords, metric='euclidean')
D_feat = cdist(feature_coords, feature_coords, metric='euclidean')

# %%
D_map

# %%
D_feat

# %%
i=0
k=1

# %%
map_dists = D_map[i, :].copy()
map_dists[i] = np.inf  # Exclude self
map_dists += np.random.uniform(0, 1e-10, n)  # Break ties randomly
idx_map = np.argsort(map_dists)

# Find k-th nearest neighbor in feature space
feat_dists = D_feat[i, :].copy()
feat_dists[i] = np.inf  # Exclude self
feat_dists += np.random.uniform(0, 1e-10, n)  # Break ties randomly
idx_feat = np.argsort(feat_dists)

print(idx_map, idx_feat)


# %%
n_k_map = idx_map[k-1]
n_k_feat = idx_feat[k-1]

QF_ik = D_map[i, n_k_map] / D_map[i, n_k_feat]

# QG_ik: ratio of feature-space distance to k-th map-neighbor vs to k-th feature-neighbor
QG_ik = D_feat[i, n_k_map] / D_feat[i, n_k_feat]


# %%
D_map[i, n_k_map], D_map[i, n_k_feat], D_feat[i, n_k_map], D_feat[i, n_k_feat], QF_ik, QG_ik

# %%
(np.log(QF_ik * QG_ik))

# %%
topographic_product(map_coords, feature_coords)

# %%

# %%
# np.random.seed(42)
n_neurons = 100
map_coords = np.column_stack([np.linspace(0, 100, n_neurons) + np.random.randn(n_neurons) * 0.0,
                            np.random.randn(n_neurons) * 0.0])
feature_coords = (np.linspace(0, 100, n_neurons) +  np.random.randn(n_neurons) * 0.0).reshape(-1, 1)

# reverse map coords
feature_coords = feature_coords[ ::-1]

perm_indices = np.random.permutation(n_neurons)
feature_coords_shuffled = feature_coords[perm_indices]

topographic_product(map_coords, feature_coords)

# %%
np.mean(RI_sets(map_coords, feature_coords))

# %% [markdown]
# # test with data

# %%
syn0 = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_tbar_cb.pkl'))
syn0 = syn0[syn0['type'] == 'pre']

# %%
fit_rf = pd.read_pickle(CACHE_DIR / f'{DATASET}_fit_rf.p')

# %%
# inst = 'LC10a_R'
# inst = 'LC12_R'
inst = 'T4b_R'
n_info, _ = fetch_neurons(NC(instance=inst))
len(n_info)

# %%
xy_post = fit_rf.loc[fit_rf.bodyId.isin(n_info['bodyId']), ['bodyId','x0','y0']].set_index('bodyId')

xyz_pre = syn0.loc[(syn0.bodyId.isin(n_info['bodyId']))
                    & (syn0.roi.isin(['AOTU(R)']))
                    , ['bodyId','x','y','z']].groupby('bodyId').mean()

xy_post.shape, xyz_pre.shape

# %%
hex_offset = [18, 19]

# syn in ME
syn = neuprint.queries.fetch_synapses(
    NC(bodyId= n_info['bodyId'].values),
    SC(type='post', rois='ME(R)', primary_only=False)
)
syn = syn[syn['roi'].str.contains(r'^ME_R_col')]

# extract the 2 integers from roi column, each number is proceeded by '_'
# assign them to 2 new columns 'hex1' and 'hex2'
syn['hex1'] = syn['roi'].str.extract(r'_(\d+)_')[0].astype(int)
syn['hex2'] = syn['roi'].str.extract(r'_(\d+)$')[0].astype(int)
# offset
syn['p'] = syn['hex2'] - hex_offset[1]
syn['q'] = syn['hex1'] - hex_offset[0]

# syn_pre = syn.copy()

# group by bodyId, compute the mean and variance of [x y z], and mean of hex1 and hex2
syn = syn.groupby('bodyId').agg({'x':'mean', 'y':'mean', 'z':'mean', 'p':'mean', 'q':'mean'}).reset_index()

# h and v
syn['v'] = syn['p'] + syn['q']
syn['h'] = syn['q'] - syn['p']
syn['h'] = -syn['h'] # chiasm

syn_post_mean = syn.copy()
syn.shape

# %%
# syn in LOP
syn = neuprint.queries.fetch_synapses(
    NC(bodyId= n_info['bodyId'].values),
    SC(type='pre', rois='LOP(R)', primary_only=False)
)
syn = syn[syn['roi'].str.contains(r'^LOP_R_col')]

# extract the 2 integers from roi column, each number is proceeded by '_'
# assign them to 2 new columns 'hex1' and 'hex2'
syn['hex1'] = syn['roi'].str.extract(r'_(\d+)_')[0].astype(int)
syn['hex2'] = syn['roi'].str.extract(r'_(\d+)$')[0].astype(int)
# offset
syn['p'] = syn['hex2'] - hex_offset[1]
syn['q'] = syn['hex1'] - hex_offset[0]


# group by bodyId, compute the mean and variance of [x y z], and mean of hex1 and hex2
syn = syn.groupby('bodyId').agg({'x':'mean', 'y':'mean', 'z':'mean', 'p':'mean', 'q':'mean'}).reset_index()

# h and v
syn['v'] = syn['p'] + syn['q']
syn['h'] = syn['q'] - syn['p']
syn['h'] = -syn['h'] # chiasm

syn_pre_mean = syn.copy()
syn.shape

# %%
# plot syn_post_mean[['h','v']]
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(syn_post_mean['h'], syn_post_mean['v'], alpha=0.5)
plt.scatter(syn_pre_mean['h'], syn_pre_mean['v'], alpha=0.5, c='red')
plt.grid()
plt.show()


# %%
# map_coords = syn_post_mean[['x', 'y', 'z']]
map_coords = syn_post_mean[['p', 'q']]

feature_coords = syn_pre_mean[['x', 'y', 'z']]

# %%
print(f'{inst}', np.mean(RI_sets(map_coords, feature_coords)), topographic_product(map_coords, feature_coords))

# %%
print(f'{inst}', np.mean(RI_sets(map_coords, feature_coords)), topographic_product(map_coords, feature_coords))

# %%
print(f'{inst}', np.mean(RI_sets(map_coords, feature_coords)), topographic_product(map_coords, feature_coords))

# %%
print(f'{inst}', np.mean(RI_sets(map_coords, feature_coords)), topographic_product(map_coords, feature_coords))

# %%
print(f'{inst}', np.mean(RI_sets(map_coords, feature_coords)), topographic_product(map_coords, feature_coords))

# %%
n = len(xy_post)
perm_indices = np.random.permutation(n)
xyz_pre_shuffled = xyz_pre.iloc[perm_indices]
topographic_product(xy_post, xyz_pre_shuffled)

# %%
# syn_post = neuprint.queries.fetch_synapses(
#     NC(bodyId= n_info['bodyId'].values),
#     SC(type='post', rois='LO(R)', primary_only=False)
# )

# %% [markdown]
# # end
