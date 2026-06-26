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

# %% [markdown]
# ### exerted from 'resolution_cb.ipynb', check for eg found in 'contrib_plot_mip.ipynb'

# %%
# %load_ext autoreload
# %autoreload 2

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
import scipy

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

from quan_propagation.func import plot_neuron_with_outlines

# from fafbseg import flywire
# import flybrains

# %%
from utils.config import CACHE_DIR, DATA_DIR, DATASET, FIG_DIR, SIDE_CHAR
DATA_DIR.mkdir(parents=True, exist_ok=True)

result_dir = FIG_DIR / 'quan_propagation'
result_dir.mkdir(parents=True, exist_ok=True)

cache_dir = CACHE_DIR / 'quan_propagation'
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load prop mat

# %%
prop = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_prop.npz') # whole conn matrix, abs wt

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
stepsn = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_{SIDE_CHAR}_lat_flow_sum.npz')
stepsn.shape

# %%
_other_side = 'l' if SIDE_CHAR == 'r' else 'r'
stepsn_left = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_{_other_side}_lat_flow_sum.npz')
stepsn_left.shape

# %% [markdown]
# # Get meta

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
# make L1 excitatory so it's an ON cell 
meta = meta_judith.copy()
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

# %% [markdown]
# # Plot rf

# %%
plot_eg_dir = FIG_DIR / 'quan_propagation' / 'eg_neuron_1101'
plot_eg_dir.mkdir(parents=True, exist_ok=True)

# %%
# rf
ids = [28227, 80342, 17594]
for bid in ids:
    inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
    outidx = meta.idx[meta.bodyId == bid] 
    df = ci.result_summary(stepsn, inidx, outidx,
                        inidx_map = idx_to_coords, 
                        outidx_map = idx_to_bodyId,
                        display_threshold = 0,
                        display_output= False
                        )
    fig = ci.hex_heatmap(df,custom_colorscale='Reds')
    fig.show()
    # save plotly
    fig.write_image(Path(plot_eg_dir, f'rf_{bid}.pdf'))

# %% [markdown]
# # eg

# %% [markdown]
# ## AMMC002_L
#
# inconsistent among cells, 148507 has fewer incoming conn

# %% [markdown]
# ### right

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
outidx = meta.idx[meta.bodyId.isin([148507])] 
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
paths2 = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 4)
paths2 = ci.group_paths(paths2, idx_to_type_side, idx_to_type_side)
len(paths2)

# %%
paths_filtered = ci.filter_paths(paths2, threshold=0.005)
ci.plot_layered_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %% [markdown]
# ### left

# %%
# DEBUG
inidx = meta.idx[meta.instance.isin(['L1_L', 'L2_L', 'L3_L', 'R7_L', 'R8_L'])] 
outidx = meta.idx[meta.cell_type_side.isin(['AMMC002_R'])] 
df = ci.result_summary(stepsn_left, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

# %%
fig = ci.hex_heatmap(df)
fig

# %% [markdown]
# ## LAL304m_R

# %%
# DEBUG
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
outidx = meta.idx[meta.cell_type_side.isin(['LAL304m_R'])] 
df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

# %%
fig = ci.hex_heatmap(df)
fig

# %%
outidx = meta.idx[meta.bodyId.isin([11924])] 
paths = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 3)
paths = ci.group_paths(paths, idx_to_type_side, idx_to_type_side)
len(paths)

# %%
paths_filtered = ci.filter_paths(paths, threshold=0.005)
ci.plot_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %%
paths2 = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 4)
paths2 = ci.group_paths(paths2, idx_to_type_side, idx_to_type_side)
len(paths2)

# %%
paths_filtered = ci.filter_paths(paths2, threshold=0.01)
ci.plot_layered_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %% [markdown]
# ## AVLP714m_R

# %%
# DEBUG
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
outidx = meta.idx[meta.cell_type_side.isin(['AVLP714m_R'])] 
df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

# %%
fig = ci.hex_heatmap(df)
fig

# %%
outidx = meta.idx[meta.bodyId.isin([12282])] 
paths = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 3)
paths = ci.group_paths(paths, idx_to_type_side, idx_to_type_side)
len(paths)

# %%
paths_filtered = ci.filter_paths(paths, threshold=0.003)
ci.plot_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %%
outidx = meta.idx[meta.bodyId.isin([12282])] 
paths2 = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 4)
paths2 = ci.group_paths(paths2, idx_to_type_side, idx_to_type_side)
len(paths2)

# %%
paths_filtered = ci.filter_paths(paths2, threshold=0.005)
ci.plot_layered_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %% [markdown]
# ## LHPD3a2_a_R

# %%
# DEBUG
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
# outidx = meta.idx[meta.cell_type_side.isin(['LHPD3a2_a_R'])] 
outidx = meta.idx[meta.bodyId.isin([87239])] 
df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )
ci.hex_heatmap(df)

# %%
outidx = meta.idx[meta.bodyId.isin([87239])] 
paths = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 3)
paths = ci.group_paths(paths, idx_to_type_side, idx_to_type_side)
len(paths)

# %%
paths_filtered = ci.filter_paths(paths, threshold=0.002)
ci.plot_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %%
paths2 = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 4)
paths2 = ci.group_paths(paths2, idx_to_type_side, idx_to_type_side)
len(paths2)

# %%
paths_filtered = ci.filter_paths(paths2, threshold=0.005)
ci.plot_layered_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %% [markdown]
# ## test bodyId

# %%
# DEBUG
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
# outidx = meta.idx[meta.cell_type_side.isin(['LoVP18_R'])] 
outidx = meta.idx[meta.bodyId.isin([19233])] 
df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )
ci.hex_heatmap(df,custom_colorscale='Reds')

# %%
outidx = meta.idx[meta.bodyId.isin([25289])] 
paths = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 3)
paths = ci.group_paths(paths, idx_to_type_side, idx_to_type_side)
len(paths)

# %%
paths_filtered = ci.filter_paths(paths, threshold=0.002)
ci.plot_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %%
paths2 = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 4)
paths2 = ci.group_paths(paths2, idx_to_type_side, idx_to_type_side)
len(paths2)

# %%
paths_filtered = ci.filter_paths(paths2, threshold=0.005)
ci.plot_layered_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %% [markdown]
# ## ER2_a_R

# %%
# DEBUG
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
outidx = meta.idx[meta.cell_type_side.isin(['ER2_a_R'])] 
df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

# %%
fig = ci.hex_heatmap(df)
fig

# %%
outidx = meta.idx[meta.bodyId.isin([20664])] 
paths = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 3)
paths = ci.group_paths(paths, idx_to_type_side, idx_to_type_side)
len(paths)

# %%
paths_filtered = ci.filter_paths(paths, threshold=0.005)
ci.plot_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %%
paths2 = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 4)
paths2 = ci.group_paths(paths2, idx_to_type_side, idx_to_type_side)
len(paths2)

# %%
paths_filtered = ci.filter_paths(paths2, threshold=0.005)
ci.plot_layered_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %% [markdown]
# ## PLP156_L

# %%
# DEBUG
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
outidx = meta.idx[meta.cell_type_side.isin(['PLP156_L'])] 
df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

# %%
fig = ci.hex_heatmap(df)
fig

# %%
outidx = meta.idx[meta.bodyId.isin([53437])] 
paths = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 3)
paths = ci.group_paths(paths, idx_to_type_side, idx_to_type_side)
len(paths)

# %%
paths_filtered = ci.filter_paths(paths, threshold=0.005)
ci.plot_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %%
paths2 = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 4)
paths2 = ci.group_paths(paths2, idx_to_type_side, idx_to_type_side)
len(paths2)

# %%
paths_filtered = ci.filter_paths(paths2, threshold=0.005)
ci.plot_layered_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %% [markdown]
# ## AOTU002_a_R

# %%
# DEBUG
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
outidx = meta.idx[meta.cell_type_side.isin(['AOTU002_a_R'])] 
df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

# %%
fig = ci.hex_heatmap(df)
fig

# %%
outidx = meta.idx[meta.bodyId.isin([53437])] 
paths = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 3)
paths = ci.group_paths(paths, idx_to_type_side, idx_to_type_side)
len(paths)

# %%
paths_filtered = ci.filter_paths(paths, threshold=0.005)
ci.plot_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %%
paths2 = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 4)
paths2 = ci.group_paths(paths2, idx_to_type_side, idx_to_type_side)
len(paths2)

# %%
paths_filtered = ci.filter_paths(paths2, threshold=0.005)
ci.plot_layered_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %% [markdown]
# ## PLP158_R

# %%
# DEBUG
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
outidx = meta.idx[meta.cell_type_side.isin(['PLP158_R'])] 
df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

# %%
fig = ci.hex_heatmap(df)
fig

# %%
outidx = meta.idx[meta.bodyId.isin([43903])] 
paths = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 3)
paths = ci.group_paths(paths, idx_to_type_side, idx_to_type_side)
len(paths)

# %%
paths_filtered = ci.filter_paths(paths, threshold=0.005)
ci.plot_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %%
