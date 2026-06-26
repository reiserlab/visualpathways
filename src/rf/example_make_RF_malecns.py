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
# %%
# %load_ext autoreload
# %autoreload 2  


# %%
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
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# %%
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
# %%
from neuprint import fetch_neurons, fetch_adjacencies
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC
import neuprint

# %%
# %%
from utils.prop_by_adj import prop_series, prop_series_inf
from utils.graph_utils import adj_trans_matrix, collect_shortest_paths
from utils.plotting_functions import plot_col_heatmap

# %%
# %%
from utils.config import CACHE_DIR, DATA_DIR, EYEMAP_DIR, FIG_DIR, SIDE

# save results if needed
results_dir = FIG_DIR / 'rf'
results_dir.mkdir(parents=True, exist_ok=True)

cache_dir = CACHE_DIR / 'rf'
cache_dir.mkdir(parents=True, exist_ok=True)


# %% [markdown]
#  # Set up projections

# %%
# %%
# # load data
# ucl_hex = pd.read_pickle(EYEMAP_DIR / 'mcns_20240701' / f'ucl_hex_{SIDE}.pkl')

# load xlsx
ucl_hex = pd.read_excel(EYEMAP_DIR / 'mcns_20240701' / f'pqxyztp_{SIDE}.xlsx')
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


# %% [markdown]
# # Compute RF directly from input synapses' hex coord, no hops

# %%
# inst = 'LC10a_R'
inst = 'LC16_R'
n1, _ = fetch_neurons(NC(instance= inst))

# %%
ids = n1['bodyId'].values
# ids = [100368, 21065]
ids

# %%
# compute column weights / RF for each input cell
rf_lst = []
for i in range(len(ids)):
    hexw = hexw_columnar(ids[i], roi_str="LO(R)") # use the appriroate roi_str
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
# all cells
for i in range(len(rf_lst)):
# for i in range(2):
    if i == 0:
        df = pd.merge(xyhex_moll, rf_lst[0], on=['hex1', 'hex2'], how='left')
        df['wt'] = df['wt'].fillna(0)
        df['wt'] = df['wt'] 
    else:
        df2 = pd.merge(xyhex_moll, rf_lst[i], on=['hex1', 'hex2'], how='left')
        df2['wt'] = df2['wt'].fillna(0)
        df2['wt'] = df2['wt'] 
        df['wt'] = df['wt'] + df2['wt']
    
df['wt'] = df['wt'] / df['wt'].max()

# Mollweide
rf_moll = df.copy()
# Mercator
rf_merc = rf_moll.copy()
rf_merc[['x','y']] = xyhex_merc[['x','y']]

# %% [markdown]
#  ## Plot

# %% [markdown]
# ### Mollweide

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
fig.set_size_inches(16,8)

# # title
# # ax.set_title(f"RF of {target_inst}_{target_id[i]}")
# fig.savefig(CACHE_DIR / 'rf' / f"RF_sum_{inst}_Moll.png")


# %% [markdown]
# ### Mercator

# %%
# %%
# Mercator version
fig, ax = plt.subplots(nrows=1, ncols=1)

ax.scatter(rf_merc['x'].values, rf_merc['y'].values, c=rf_merc['wt'].values, 
            cmap="Reds",vmin=0,vmax=1)
# add colorbar
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="Reds"), ax=ax)
cbar.set_label('normalized synapse count')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# title
# ax.set_title(f"RF of {target_inst}_{target_id[i]}")

fig.show()


# %% [markdown]
#  # Compute RF for one cell if:
#
#
#
#  knowing the exact input cell bodyIds. Still need to compute the contribution weights from the graph generated by the shortest path + neighbors from these source and target bodyIds.
#
#
#
#  We can compute RF for each input instance separately, or combine all input bodyIds if centain they're at the same flow level.
#
#

# %%
N_hops = 2 # should not propagate more than the max hops in path finding

# %%
# %%
# load ids from "find_col_input_malecns.ipynb"
g_source = pd.read_pickle(Path(cache_dir, 'g_source.pkl'))
g_target = pd.read_pickle(Path(cache_dir, 'g_target.pkl'))

# Or load from ids
# g_source, _ = fetch_neurons(NC(bodyId= ??))
# g_target, _ = fetch_neurons(NC(bodyId= ??))


# %%
g_target[['bodyId', 'instance', 'type']]

# %%
# %%
print(g_source.instance.value_counts())
print(g_target.instance.value_counts())


# %% [markdown]
#  ### Choose 1 input instances if there are more

# %%
# %%
source_id = g_source['bodyId'].values
# if more than one input instances, choose one
# source_id = g_source[g_source['instance'] == insts.index[0]]['bodyId'].values

target_id = g_target['bodyId'].values

# minimum weight for path
min_wt = 3 

print(len(source_id), len(target_id))


# %%
# %%
# compute column weights / RF for each input cell
rf_lst = []
for i in range(len(source_id)):
    hexw = hexw_columnar(source_id[i], roi_str="LO(R)") # use the appriroate roi_str
    rf_lst.append(hexw)



# %% [markdown]
#  ## Collect nodes in shortest paths, filter by path length, and get their inputs

# %%
# %%
# collect nodes in shortest paths and their incoming neighbors

path_len, paths_all = collect_shortest_paths(source_id, target_id, min_weight=min_wt, timeout=5)

# all ids in paths
id_path = pd.concat(paths_all, ignore_index=True, axis=0)['bodyId'].unique()
id_path = [int(i) for i in id_path]

# # fetch neuron info
n_path, _ = fetch_neurons(NC(bodyId=id_path), )
# # remove na 
n_path = n_path.dropna(subset=['instance'])
id_path = n_path['bodyId'].values

print("nodes in path:", len(id_path))


# %% [markdown]
#  ## Compute contributions via propagation
#
#
#
#  Next we will construct a backward-transition matrix, propogating from the target upstream to assign significance (pagerank). This will help us determine which types are important input types.

# %%
# %%
# get adj matrix for all the nodes in paths and their incoming neighbors, normalized by column
m_adj, m_trans, neuron_df = adj_trans_matrix(id_path, min_total_weight=1)

# trans_inf = prop_series_inf(m_trans)
trans_long = prop_series(m_trans, N_hops)

trans_sum = trans_long.copy()
# Optional: normalize by to max contrib within chosen type/group
trans_sum = trans_sum / np.max(trans_sum)
# keep only the ids in id_path
trans_sum = trans_sum.reindex(index=id_path).reindex(columns=id_path)
print(trans_sum.shape)

# initial state with the target as the active node
v0 = [1 if np.isin(trans_sum.index[i], g_target['bodyId'].values) else 0 for i in range(len(n_path))]
# equliibrium state
contrib = v0 @ trans_sum

# merge neuron_info on the index of contrib
contrib.name = 'contribution'
contrib = pd.merge(contrib, neuron_df, left_index=True, right_on='bodyId', how='left')


# %% [markdown]
#  ## Combine all RFs based on contribution weights, set weights to 0 if <0

# %%
# %%
for i in range(len(rf_lst)):
# for i in range(2):
    if i == 0:
        df = pd.merge(xyhex_moll, rf_lst[0], on=['hex1', 'hex2'], how='left')
        df['wt'] = df['wt'].fillna(0)
        cw = contrib[contrib['bodyId'] == source_id[i]]['contribution']
        cw = cw.clip(lower=0).values[0]
        df['wt'] = df['wt'] * cw
    else:
        df2 = pd.merge(xyhex_moll, rf_lst[i], on=['hex1', 'hex2'], how='left')
        df2['wt'] = df2['wt'].fillna(0)
        cw = contrib[contrib['bodyId'] == source_id[i]]['contribution']
        cw = cw.clip(lower=0).values[0]
        df2['wt'] = df2['wt'] * cw
        df['wt'] = df['wt'] + df2['wt']
    
df['wt'] = df['wt'] / df['wt'].max()

# Mollweide
rf_moll = df.copy()
# Mercator
rf_merc = rf_moll.copy()
rf_merc[['x','y']] = xyhex_merc[['x','y']]




# %% [markdown]
#  ## Plot

# %% [markdown]
# ### Mollweide

# %%
# %%
# Molleweide version
fig, ax = plt_mollweide()

ax.scatter(rf_moll['x'].values, rf_moll['y'].values, c=rf_moll['wt'].values, 
            cmap="Reds",vmin=0,vmax=1, )
# add colorbar
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="Reds"), ax=ax)
cbar.set_label('normalized synapse count')

# change size
fig.set_size_inches(16,8)

# title
# ax.set_title(f"RF of {target_inst}_{target_id[i]}")


# %% [markdown]
# ### Mercator

# %%
# %%
# Mercator version
fig, ax = plt.subplots(nrows=1, ncols=1)

ax.scatter(rf_merc['x'].values, rf_merc['y'].values, c=rf_merc['wt'].values, 
            cmap="Reds",vmin=0,vmax=1)
# add colorbar
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="Reds"), ax=ax)
cbar.set_label('normalized synapse count')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# title
# ax.set_title(f"RF of {target_inst}_{target_id[i]}")

fig.show()


# %%
# %%
# save
# fig.savefig(cache_dir / cellname_RF.png")


# %% [markdown]
# ### hex coord

# %%
rf_hex = df.copy()
rf_hex = rf_hex.rename(columns={'hex1':'hex1_id', 'hex2':'hex2_id'})

# %%
fig = plot_col_heatmap(rf_hex, 'wt', cmax=1, cmin=-1)
fig.show()
# fig.write_image(results_dir / f'rf_hex_{target_id[0]}_{g_source['type'][0]}.png')

# %%
df = rf_hex[rf_hex['wt'] > 0]
print(np.sum(df['hex1_id'] * df['wt'])/df.wt.sum(),
      np.sum(df['hex2_id'] * df['wt'])/df.wt.sum())

# %%

# %% [markdown]
#  ## If more than one instance is desired, suggest do it separately and compare

# %% [markdown]
#  # Compute RFs for many cells if:
#
#
#
#  knowing exactly which input cell type/instance to use and how many hops to the target cell
#
#
#
#  By default should save in cache -- not sync with git

# %% [markdown]
#  ## Example, all "ER4d_R" and construct receptive fields based on all MeTu types 2 hops away as inputs.
#
#
#
#  Slow!

# %%
# %%
# set up parameters
min_wt = 3 # thrshold edge wt >=
# target_inst = 'ER4d_R'
# source_inst = r'^MeTu.*R$' # assuming one source instance for each target instance
target_inst = 'pC1_2a_R'
source_inst = r'^LC16_R$' # assuming one source instance for each target instance
N_hops = 1 # max path length to consider
roi_source = "LO(R)"

save_dir = PROJECT_ROOT / 'cache' / 'rf' / target_inst
save_dir.mkdir(parents=True, exist_ok=True)

neurons_df, _ = fetch_neurons(NC(instance= target_inst))
target_ids = neurons_df['bodyId'].values # bodyId of the neurons

# For each cell, 
# 1/ propagate upstream by N_hops, filter for desired input instances
# 2/ collect all nodes in shortest paths and nbhd
# 3/ get adj and transition matrix, compute contribution
# 4/ combine RFs based on contribution weights
# 5/ save plots

for target_id in target_ids: 
    print("target cell:", target_id)
    
    # 1/ propogate upstream by N_hops, filter for desired input cells
    id_post = target_id
    id_upstream = [target_id]
    for i in range(N_hops):
        # n_pre = fetch_simple_connections(None, id_post, min_weight=min_wt)
        # id_post = n_pre['bodyId_pre'].tolist()
        _, adj = fetch_adjacencies(None, id_post, min_total_weight= min_wt)
        id_post = adj['bodyId_pre'].unique().tolist() # post for the next iteration
        id_upstream += id_post
        id_upstream = list(set(id_upstream))

    # filter for MeTu instances
    n_info, _ = fetch_neurons(NC(bodyId=id_upstream))
    n_info = n_info[n_info['instance'].notna()]
    n_info = n_info[n_info['instance'].str.contains(source_inst)]
    source_id = n_info['bodyId'].values
    
    print("cell count for chosen instance:", len(source_id))

    # if no source cells, skip
    if len(source_id) == 0:
        continue

    # 2/ collect all nodes in shortest paths and nbhd
    path_len, paths_all = collect_shortest_paths(source_id, target_id, min_weight= min_wt, timeout=5)

    # backup
    # paths_all = [] 
    # for i in source_id:
    #     paths = fetch_shortest_paths(i, target_id, min_weight=min_wt, timeout=10)
    #     if len(paths_all) == 0:
    #         paths_all = paths
    #     else:
    #         paths['path'] = paths['path'] + paths_all['path'].values[-1] + 1
    #         paths_all = pd.concat((paths_all, paths), ignore_index=True, axis=0)
    # id_path = np.unique(paths_all['bodyId'])
    # id_path = id_path.astype(int)

    # all ids in paths
    id_path = pd.concat(paths_all, ignore_index=True, axis=0)['bodyId'].unique()
    id_path = [int(i) for i in id_path]
    
    print("nodes in shortest path:", len(id_path))    
    
    # 3/ get adj matrix for all the nodes in paths and their incoming neighbors, normalized by column
    m_adj, m_trans, neuron_df = adj_trans_matrix(id_path, min_total_weight= 1)

    # trans_inf = prop_series_inf(m_trans)
    trans_long = prop_series(m_trans, N_hops)

    trans_sum = trans_long.copy()
    # Optional: normalize by to max contrib within chosen type/group
    trans_sum = trans_sum / np.max(trans_sum)
    # keep only the ids in id_path
    trans_sum = trans_sum.reindex(index=id_path).reindex(columns=id_path)
    print(trans_sum.shape)

    # initial state with the target as the active node
    v0 = [1 if np.isin(trans_sum.index[i], target_id) else 0 for i in range(len(id_path))]
    # equliibrium state
    contrib = v0 @ trans_sum

    # merge neuron_info on the index of contrib
    contrib.name = 'contribution'
    contrib = pd.merge(contrib, neuron_df, left_index=True, right_on='bodyId', how='left')

    # 4/ combine all RFs based on contribution weights, set weights to 0 if <0

    # Molleweide
    rf_moll = xyhex_moll.copy()
    rf_moll['wt'] = 0
    for i in range(len(source_id)):
        cw = contrib[contrib['bodyId'] == source_id[i]]['contribution']
        if len(cw) > 0:
            cw = cw.clip(lower=0).values[0]
            hexw = hexw_columnar(source_id[i], roi_str=roi_source) # use the appriroate roi_str
            df2 = pd.merge(xyhex_moll, hexw, on=['hex1', 'hex2'], how='left')
            df2['wt'] = df2['wt'].fillna(0)
            df2['wt'] = df2['wt'] * cw
            rf_moll['wt'] = rf_moll['wt'] + df2['wt']
    rf_moll['wt'] = rf_moll['wt'] / rf_moll['wt'].max()

    #     cw = contrib[contrib['bodyId'] == source_id[i]]['contribution']
    #     # if cw is not empty
    #     if len(cw) > 0:
    #         cw = cw.clip(lower=0).values[0]
    #         hexw = hexw_columnar(source_id[i], roi_str=roi_source) # use the appriroate roi_str
    #         hexw['wt'] = hexw['wt'] * cw
    #         # if hexw_sum is not defined
    #         if 'hexw_sum' not in locals():
    #             hexw_sum = hexw
    #         else:
    #             hexw_sum = pd.concat([hexw_sum, hexw]).groupby(['hex1', 'hex2']).sum().reset_index()
        
    # df = hexw_sum.copy()
    # # remove hexw_sum
    # del hexw_sum
    # df['wt'] = df['wt'] / df['wt'].max()

    # Mercator
    rf_merc = rf_moll.copy()
    rf_merc[['x','y']] = xyhex_merc[['x','y']]

    # 5/ save plots

    # Molleweide version
    fig, ax = plt_mollweide()
    ax.scatter(rf_moll['x'].values, rf_moll['y'].values, c=rf_moll['wt'].values, 
                cmap="Reds", vmin=0, vmax=1, s=30)
    # add colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="Reds"), ax=ax)
    cbar.set_label('normalized synapse count')
    # change size
    fig.set_size_inches(16,8)
    # title
    ax.set_title(f"RF {target_inst}_{target_id} Molleweide")

    # save fig
    fig.savefig(Path(save_dir, f"RF_{target_inst}_{target_id}_Moll.png"))
    plt.close(fig) # close fig to avoid memory leak

    # # Mercator version
    # fig, ax = plt_mercator()
    # ax.scatter(rf_merc['x'].values, rf_merc['y'].values, c=rf_merc['wt'].values, 
    #             cmap="Reds",vmin=0,vmax=1, )
    # # add colorbar
    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="Reds"), ax=ax)
    # cbar.set_label('normalized synapse count')
    # # # change size
    # # fig.set_size_inches(16,8)
    # # title
    # ax.set_title(f"RF {target_inst}_{target_id} Mercator")

    # # save fig
    # fig.savefig(Path(save_dir, f"RF_{target_inst}_{target_id}_Merc.png"))
    # plt.close(fig) # close fig to avoid memory leak




# %% [markdown]
# # Compute RF for combined sources and/or targets

# %% [markdown]
# ## All VPNs to some P1s, direct connections

# %%
# load OL neurons
neuron_info_olall = pd.read_pickle(DATA_DIR / 'neuron_info_ol.pkl')

# %%
source_inst = 'VPN' # assuming one source instance for each target instance
source_ids = neuron_info_olall.loc[neuron_info_olall.main_groups == source_inst, "bodyId"].values

# %%
# target_inst = 'ER4d_R'
# source_inst = r'^MeTu.*R$' # assuming one source instance for each target instance
# target_inst = 'pC1_1a_R+pC1_1b_R'
# target_inst = ["pC1_2a_R", "pC1_2a/2b_R", "pC1_2b_R"]
# target_inst = ["pC1_9a_R", "pC1_9b_R"]
target_inst = ["pC1_13c_R"]

target_info, _ = fetch_neurons(NC(instance= target_inst))

target_inst_str = '+'.join(target_inst) if isinstance(target_inst, list) else target_inst
target_inst_str = target_inst_str.replace('/', '=')  # replace '/' with '='

# %%
# n1, c1 = fetch_adjacencies(sources=None, targets=NC(instance='P1_1a_R'), min_total_weight=1)
# c1 = c1.groupby(['bodyId_pre', 'bodyId_post']).agg({'weight': 'sum'}).reset_index()
# # conn_df = neuprint.merge_neuron_properties(n1, c1, ['type', 'instance'])


# %%
target_info[['bodyId', 'instance', 'type']]


# %%
# %%
# set up parameters
min_wt = 3 # thrshold edge wt >=
N_hops = 1 # max path length to consider
# roi_source = "LO(R)"

save_dir = PROJECT_ROOT / 'cache' / 'rf' / target_inst_str
save_dir.mkdir(parents=True, exist_ok=True)

# For each target cell, build one hexw from combining 3 hexw from 3 neuropils
# For each source cell, compute contribution (=input-normalized wt) and rf

# 1/ propagate upstream by N_hops, filter for desired input instances
# 2/ collect all nodes in shortest paths and nbhd
# 3/ get adj and transition matrix, compute contribution
# 4/ combine RFs based on contribution weights
# 5/ save plots
# 6/ save the summed plot

rf_lst = []
for i, target_id in enumerate(target_info['bodyId'].values):
    print("target cell:", target_id)

    n1, c1 = fetch_adjacencies(sources=None, targets=NC(bodyId=target_id), min_total_weight=1)
    c1 = c1.groupby(['bodyId_pre', 'bodyId_post']).agg({'weight': 'sum'}).reset_index() #combine rois
    c1 = c1[c1['bodyId_pre'].isin(source_ids)] # keep only source cells
    c1['weight'] = c1['weight'] / target_info.at[i, 'upstream'] # normalize by upstream
    
    # iterate over source cells
    # Molleweide
    rf_moll = xyhex_moll.copy()
    rf_moll['wt'] = 0
    for source_id in c1['bodyId_pre'].values:
        cw = c1[c1['bodyId_pre'] == source_id]['weight'].values[0]
        for roi_str in ['ME(R)', 'LO(R)', 'LOP(R)']:
            hexw = hexw_columnar(source_id, roi_str= roi_str)
            df2 = pd.merge(xyhex_moll, hexw, on=['hex1', 'hex2'], how='left')
            df2['wt'] = df2['wt'].fillna(0)
            df2['wt'] = df2['wt'] * cw
            rf_moll['wt'] = rf_moll['wt'] + df2['wt']
    # optional normalization
    rf_moll['wt'] = rf_moll['wt'] / rf_moll['wt'].max()

    rf_lst.append(rf_moll)

    # # Mercator
    # rf_merc = rf_moll.copy()
    # rf_merc[['x','y']] = xyhex_merc[['x','y']]

    # save plots

    # Molleweide version
    fig, ax = plt_mollweide()
    ax.scatter(rf_moll['x'].values, rf_moll['y'].values, c=rf_moll['wt'].values, 
                cmap="Reds", vmin=0, vmax=1, s=30)
    # add colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="Reds"), ax=ax)
    cbar.set_label('normalized synapse count')
    # change size
    fig.set_size_inches(16,8)
    # title
    ax.set_title(f"RF {target_inst_str}_VPN Molleweide")

    # save fig
    fig.savefig(Path(save_dir, f"RF_{target_inst_str}_{target_id}_VPN_Moll.png"))
    plt.close(fig) # close fig to avoid memory leak

# sum up target ids

# Molleweide version
df = rf_lst[0].copy()
for i in range(1, len(rf_lst)):
    df['wt'] = df['wt'] + df2['wt']
    
df['wt'] = df['wt'] / df['wt'].max()

# Mollweide
rf_moll = df.copy()

# Molleweide version
fig, ax = plt_mollweide()

ax.scatter(rf_moll['x'].values, rf_moll['y'].values, c=rf_moll['wt'].values, 
            cmap="Reds",vmin=0,vmax=1, s=30)
# add colorbar
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="Reds"), ax=ax)
cbar.set_label('normalized synapse count')

# change size
fig.set_size_inches(16,8)

fig.savefig(Path(save_dir, f"RF_{target_inst_str}_VPN_Moll.png"))

# %% [markdown]
# # RF size

# %% [markdown]
# ## propagated weights

# %%
import scipy

cache_data_dir = Path(PROJECT_ROOT, 'cache', 'data')
cache_data_dir.mkdir(parents=True, exist_ok=True)

stepsn = scipy.sparse.load_npz(Path(cache_data_dir, 'fromJudith', 'malecns_v09_lat', 'malecns_v0.9_R_lat_flow_sum.npz'))
stepsn.shape

# %%
from utils.ol_rf import _clean_rf_params, _compute_rf_params

df_fit = _compute_rf_params(
    meta_judith, stepsn,
    # in_instances=['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R', 'R7d_R', 'R8d_R', 'HBeyelet_R'],
    in_instances=['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'],
    out_instances = meta_cb_vpn['instance'].unique().tolist(),
)
df_fit = _clean_rf_params(df_fit)

df_fit.rename(columns={'size': 'area'}, inplace=True)

# %%
stepsn.shape
