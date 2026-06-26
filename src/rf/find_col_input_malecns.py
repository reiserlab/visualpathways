# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
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
# # General procedure
#
# Given a target (usually in the central brain), we'd like to estimate its RF. We do so by finding its columnar inputs in ME/LO/LOP and combining the RFs of these inputs. So the first step is to find a "good" set of columnar input neurons. This notebook is about finding these neurons by propogating through the network (using pagerank) and visual inspection.
#
# We want to estimate the RF of a target neuron. We first propogate upstream via the connectivity matrix, determine which columnar cells to use, and optionally visualize the results. Then we compute the RF based on this cell type, using a separate script. 

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
# load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
print(f"Project root directory: {PROJECT_ROOT}")

# %%
import plotly.io as pio
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import networkx as nx
import webbrowser
import pickle 

# %%
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
from neuprint import fetch_neurons, fetch_synapses, fetch_adjacencies
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC

from neuprint import fetch_shortest_paths, fetch_paths

import navis
import navis.interfaces.neuprint as neu

from utils.plotting_functions import plot_pyvis_Angel

# %%
# from neuprint.queries import fetch_all_rois, fetch_roi_hierarchy

# # Show the ROI hierarchy, with primary ROIs marked with '*'
# print(fetch_roi_hierarchy(include_subprimary=True, mark_primary=True, format='text'))

# show primary ROIs
# print(fetch_all_rois())

# %%
from utils.config import CACHE_DIR, DATA_DIR, FIG_DIR, HTML_FIG_DIR

# save results if needed
results_dir = FIG_DIR / 'rf'
results_dir.mkdir(parents=True, exist_ok=True)

cache_dir = CACHE_DIR / 'rf'
cache_dir.mkdir(parents=True, exist_ok=True)

# %%
# load OL neurons
neuron_info_olall = pd.read_pickle(DATA_DIR / 'neuron_info_ol.pkl')

# %% [markdown]
# # Example, propagate N=2 hops
#
# ## Start with a target neuron / instance

# %%
min_wt = 3

# %%
# target = sink
# target_inst = ['LC18_R'] # as a list
# target_inst = ['ER4d_R'] # as a list
# target_inst = ['P1_1b_R'] # as a list
target_inst = ['pC1_2a_R'] # as a list
# 'AOTU008_R'
# 'AOTU014'

n_target, _ = fetch_neurons(NC(instance= target_inst))
# n_target, _ = fetch_neurons(NC(type= target_inst))
# pick one cell here
n_target = n_target.iloc[[0]]

id_target = n_target['bodyId'].values

# %%
# id_target = [49738] #LC18
# id_target = [11134] # VSm
# id_target = [20036] # LPLC1
# id_target = [50974] # MeTu4a
# id_target = [15319] # AOTU045
# n_target, _ = fetch_neurons(NC(bodyId= id_target))

# %% [markdown]
# ## Find source nodes and pick reasonable cell types/instances for furture investigation. 
#
# Helpful to refer to the OL paper summary plots
#
# https://reiserlab.github.io/male-drosophila-visual-system-connectome/index.html
#
# ME, as an earlier layer, is preferrable, but need to check possible parallel pathways which might have different RFs.

# %%
# run for N_hops, collect all bodyId_pre. 
# Note: N_hops =2 is already big
N_hops = 1
id_post = id_target
id_upstream = list(id_target)
id_level = []
for i in range(N_hops):
    # n_pre = fetch_simple_connections(None, id_post, min_weight=5)
    # id_post = n_pre['bodyId_pre'].tolist()
    _, adj = fetch_adjacencies(None, id_post, min_total_weight= min_wt)
    id_post = adj['bodyId_pre'].unique().tolist() # post for the next iteration
    # append id_post to the list id_level
    id_level.append(id_post)
    id_upstream += id_post
    id_upstream = list(set(id_upstream))

len(id_upstream)

# %%
# get info on these neurons
n_info, _ = fetch_neurons(NC(bodyId=id_upstream))

# check against a list of OL neurons
# filter out neurons that are not in the OL
n_info_ol = n_info[n_info['bodyId'].isin(neuron_info_olall['bodyId'])]

# for each level
n_level = []
for i in range(len(id_level)):
    n, _ = fetch_neurons(NC(bodyId=id_level[i]))
    n_ol = n[n['bodyId'].isin(neuron_info_olall['bodyId'])]
    n_level.append(n_ol)

# %%
n_info_ol['type'].value_counts().head(5)

# %%
n_level[0]['type'].value_counts().head(10)

# %%
n_level[1]['type'].value_counts().head(10)

# %%
# choose a list of source(s)
source_inst = ["LC16_R"] # as a list
# source_inst = ["T2_R"] # as a list
# source_inst = ["MeTu1_R"] # as a list
# source_inst = ["LC16_R"] # as a list

id_source = n_info_ol.loc[n_info_ol['instance'].isin(source_inst)]['bodyId'].values
len(id_source)

# %% [markdown]
# ## Collect nodes in shortest paths, and their inputs

# %%
from utils.graph_utils import collect_shortest_paths

path_len, paths_all = collect_shortest_paths(id_source, id_target, min_weight=min_wt, timeout=5)
len(paths_all)

# %%
path_len['path_len'].value_counts().sort_index()

# %% [markdown]
# Filter by path length

# %%
# # # Filter by path length
# # NB the path_len values contain end poitns (= hops + 1)
# layer_N = 2

# ind_short = path_len[(path_len['path_len'] > 0) & (path_len['path_len'] <= layer_N)].index.values
# # keep short paths_all
# paths_short = [paths_all[i] for i in ind_short]
# print("number of short paths: ", len(paths_short))
    
# # all ids in paths
# id_path = pd.concat(paths_short, ignore_index=True, axis=0)['bodyId'].unique()
# id_path = [int(i) for i in id_path]

# # # remove na
# # paths_short = paths_short.dropna(subset=['type'])
# # print(paths_short.shape)

# %% [markdown]
# Filter by nt type for a specific hop, TODO

# %%
# # for each data frame in paths_all, 'path' indexes the path, and bodyId are the nodes along the path, from source to sink. Extract the second last node's type
# for pi in range(len(paths_all)):
#     df = paths_all[pi]
#     N = df['path'].max()
#     for pj in range(N):
#         df_pj = df[df['path'] == pj]
#         type = df_pj.iloc[-2]['type']

# %% [markdown]
# Without filtering

# %%
# Without filtering
id_path = pd.concat(paths_all, ignore_index=True, axis=0)['bodyId'].unique()
id_path = [int(i) for i in id_path]
print("number of short paths: ", len(paths_all))

# %%
# fetch neuron info
n_path, _ = fetch_neurons(NC(bodyId=id_path), )
# # # reorder by id_path
# # n_path = n_path.set_index('bodyId').reindex(id_path).reset_index()

# # remove na 
# n_path = n_path.dropna(subset=['instance'])
# id_path = n_path['bodyId'].values

# %%
n_path['type'].value_counts()

# %% [markdown]
# ## Compute contributions via propagation
#
# Next we will construct a backward-transition matrix, propogating from the target upstream to assign significance (pagerank). This will help us determine which types are important input types.

# %%
# query adj matrix from bodyids, including those in paths and connected neighbors
from utils.graph_utils import adj_trans_matrix
m_adj, m_trans, neuron_df = adj_trans_matrix(id_path, min_total_weight=1)

# %%
# propagation
from utils.prop_by_adj import prop_series, prop_series_inf

# trans_inf = prop_series_inf(m_trans)
trans_long = prop_series(m_trans, N_hops)

trans_sum = trans_long.copy()
# Optional: normalize by to max contrib within chosen type/group
trans_sum = trans_sum / np.max(trans_sum)
# keep only the ids in id_path
trans_sum = trans_sum.reindex(index=id_path).reindex(columns=id_path)
print(trans_sum.shape)

# %% [markdown]
# We use the transition matrix series to evaluate how much each cell comtributions to the target neuron ?

# %%
# initial state with the target as the active node
v0 = [1 if np.isin(trans_sum.index[i], n_target['bodyId'].values) else 0 for i in range(len(n_path))]
# equliibrium state
contrib = v0 @ trans_sum

# merge neuron_info on the index of contrib
contrib.name = 'contribution'
contrib = pd.merge(contrib, neuron_df, left_index=True, right_on='bodyId', how='left')

# %%
contrib['contribution'].sum()

# %%
# does all cells contribute by the same amount?
contrib[contrib['instance'].isin(source_inst)].hist('contribution', bins=50)

# %% [markdown]
# Depending on the distribution, one could make a decision to use a subset of this type, or not at all.

# %% [markdown]
# ### Save the relevant ids to compute RFs using "make_RF.ipynb"

# %%
# since we added filters, need to re-fetch the source neurons
g_source, _ = fetch_neurons(NC(bodyId= contrib[contrib['instance'].isin(source_inst)]['bodyId'].values))
g_target = n_target.copy()


# %%
g_source.to_pickle(Path(cache_dir, 'g_source.pkl'))
g_target.to_pickle(Path(cache_dir, 'g_target.pkl'))


# %% [markdown]
# # Visualize network, interactive plot

# %%
# paths_all
# df = pd.concat(paths_short, ignore_index=False, axis=0) # with filter
df = pd.concat(paths_all, ignore_index=False, axis=0) # without filter

# change index to column, call it layer
# df['layer'] = df.index
# use mean index as layer
ids_layer = df.groupby('bodyId').agg({'layer': 'mean'}).reset_index()

# %%
# draw interactive graph
HTML_FIG_DIR.mkdir(parents=True, exist_ok=True)
save_path = str(HTML_FIG_DIR / "prop_tmp.html")

# cell level, not working for larger graphs
plot_pyvis_Angel(m_adj, trans_sum, g_source, g_target, n_path, ids_layer, save_path=save_path, include_nonprimary_links=False)

# type/instant level, TODO

# for notebook
# webbrowser.open_new_tab(save_path)

# %% [markdown]
# # Example, N=3 hops
#
# It might make sense to check what happens with more hops if the target is in a deep layer
#
# The same procedure can be used to compare contributions among different cell types

# %%
# run for N hops, collect all bodyId_pre. 
N_hops = 3
id_post = id_target
id_upstream = list(id_target)
for i in range(N_hops):
    _, adj = fetch_adjacencies(None, id_post, min_total_weight=min_wt)
    id_post = adj['bodyId_pre'].unique().tolist() # post for the next iteration
    id_upstream += id_post
    id_upstream = list(set(id_upstream))

len(id_upstream)


# %%
# get info on these neurons
n_info, _ = fetch_neurons(NC(bodyId=id_upstream))

# filter out neurons that are not in the OL
n_info_ol = n_info[n_info['bodyId'].isin(neuron_info_olall['bodyId'])]

# %%
n_info_ol['type'].value_counts().head(10)

# %% [markdown]
# There are many Dm2 after 3 hops. Dm2 or Mi15 is a better (more columnar) than MeTu. Cm3 is GABAergic. However, there is possibly "over counting" given the larger number. This could be due to "lateral connections" which are not necessarily what we want (direct pathway). Let's compare their contributions.

# %%
source_inst = ["Dm2_R", "Mi15_R"]

# find which n_info_ol has the source_inst
n_source = n_info_ol.loc[n_info_ol['instance'].isin(source_inst)]
id_source = n_source['bodyId'].values
# convert id_path to integer
id_source = [int(i) for i in id_source]
len(id_source)

# %%
# initialize array to store path length
path_len = np.zeros((0, 3))
# collect all shortest paths from sources to target(s) as a list
paths_all = []
for i in id_source:
    for j in id_target:
        paths = fetch_shortest_paths(i, j, min_weight=min_wt, timeout=3)
        # paths = fetch_paths(i, j, path_length=2, min_weight=10, timeout=3)
        # if len(paths_all) == 0:
        #     paths_all = paths
        # else:
        #     paths['path'] = paths['path'] + paths_all['path'].values[-1] + 1
        #     paths_all = pd.concat((paths_all, paths), ignore_index=True, axis=0)
        paths_all.append(paths)
        # pair up id_source, id_target, and path length, append to path_len
        path_len = np.vstack((path_len, np.array([i, j, paths['path'].shape[0]])))

# convert path_len to dataframe
path_len = pd.DataFrame(path_len, columns=['source', 'target', 'path_len'])
# change to integer
path_len = path_len.astype({'source': 'int', 'target': 'int', 'path_len': 'int'})


# %%
from utils.graph_utils import collect_shortest_paths

path_len, paths_all = collect_shortest_paths(id_source, id_target, min_weight=min_wt, timeout=5)


# %%
# Without filtering
id_path = pd.concat(paths_all, ignore_index=True, axis=0)['bodyId'].unique()
id_path = [int(i) for i in id_path]

# %% [markdown]
# Next we will construct a backward-transition matrix, propogating from the target upstream to assign significance (pagerank). This will help us determine which types are important input types.

# %%
# query adj matrix from bodyids, including those in paths and connected neighbors
from utils.graph_utils import adj_trans_matrix
m_adj, m_trans, neuron_df = adj_trans_matrix(id_path, min_total_weight=1)

# %%
# propagation
from prop.prop_by_adj import prop_series, prop_series_inf

# trans_inf = prop_series_inf(m_trans)
trans_long = prop_series(m_trans, N_hops)

trans_sum = trans_long.copy()
# Optional: normalize by to max contrib within chosen type/group
trans_sum = trans_sum / np.max(trans_sum)
# keep only the ids in id_path
trans_sum = trans_sum.reindex(index=id_path).reindex(columns=id_path)
print(trans_sum.shape)

# %% [markdown]
# We use the transition matrix series to evaluate how much each cell comtributions to the target neuron ?

# %%
# initial state with the target as the active node
v0 = [1 if np.isin(trans_sum.index[i], n_target['bodyId'].values) else 0 for i in range(len(id_path))]
# equliibrium state
contrib = v0 @ trans_sum

# merge neuron_info on the index of contrib
contrib.name = 'contribution'
contrib = pd.merge(contrib, neuron_df, left_index=True, right_on='bodyId', how='left')

# %%
contrib['contribution'].sum()

# %%
# overlay the two histograms, plot density, one for each source_inst, specify color
plt.figure()
contrib[contrib['instance'] == source_inst[0]]['contribution'].plot(kind='hist', bins=150, alpha=0.5, color='red', density=True)
contrib[contrib['instance'] == source_inst[1]]['contribution'].plot(kind='hist', bins=150, alpha=0.5, color='blue', density=True)
plt.show()

# %% [markdown]
# Blue has lower but more consistent contributions -- arguably a more reliable choice. 
#
# Though ideally we want a cell type with narrow distribution centered at a large value


