# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

import cmap

import pandas as pd
import numpy as np
import re
import os, datetime
# from openpyxl.styles import Font

import neuprint

# %%
from utils.config import CACHE_DIR, DATA_DIR, FIG_DIR, HTML_FIG_DIR

result_dir = FIG_DIR / 'network'
result_dir.mkdir(parents=True, exist_ok=True)

oltypes = pd.read_pickle(DATA_DIR / 'oltypes.pkl')
neuron_info = pd.read_pickle(DATA_DIR / 'neuron_info_ol.pkl')
edgelist = pd.read_pickle(DATA_DIR / 'edgelist_ol.pkl')



# %%
# # flywire
# # read csv
# edgelist = pd.read_csv(Path(DATA_DIR, 'connectome_graph.csv'))

# # make a lookup table to convert ID to short integer
# # this is to save memory
# lookup = pd.DataFrame(pd.concat([edgelist['Source Node  ID'], edgelist['Target Node ID']]).unique(), columns=['ID'])
# lookup['node'] = range(len(lookup))

# # map the lookup table to the edgelist
# edgelist['bodyId_pre'] = edgelist['Source Node  ID'].map(lookup.set_index('ID')['node'])
# edgelist['bodyId_post'] = edgelist['Target Node ID'].map(lookup.set_index('ID')['node'])
# # remove the original columns
# edgelist.drop(columns=['Source Node  ID', 'Target Node ID'], inplace=True)

# edgelist.rename(columns={'Edge Weight':'weight'}, inplace=True)

# convert values to small uint32 types to save memory
# edgelist['bodyId_pre'] = edgelist['bodyId_pre'].astype(np.uint32)
# edgelist['bodyId_post'] = edgelist['bodyId_post'].astype(np.uint32)
# edgelist['weight'] = edgelist['weight'].astype(np.uint32)


# %% [markdown]
# # algorithm
# Define the energy function
#
# $$E(z) = \frac{1}{2} \sum_{i, j = 1}^n w_{ij} (z_i - z_j - sgn(A_{ij} - A_{ji}))^2$$
#
# $$\frac{dE(z)}{dz} = 0 $$ 
#
# where
#
# $$W_{ij} = \frac{A_{ij} + A_{ji}}{2}$$
#
# yields
#
# $$L z = b$$
#
# where
#
# $$L = D - W$$
#
# $$D_{ij} = \delta_{ij} \sum_{k=1}^{n} W_{ik} $$
#
# $$b = \sum_{j = 1}^n w_{ij} sgn(A_{ij} - A_{ji})$$
#
# solve with Moore-Penrose inverse
#
# $$z^* = L^\dagger b$$

# %% [markdown]
# ### With a regularizor to fix source node {s}
#
# $$E(z) = \frac{1}{2} \sum_{i, j = 1}^n w_{ij} (z_i - z_j - sgn(A_{ij} - A_{ji}))^2 + \delta_{ij} \delta_{is} \lambda (z_i - C)^2$$ 
#
# where C is the desired z values for the source nodes and $\lambda$ is its weight
#
# $$\frac{dE(z)}{dz} = 0 $$ 
#
# yields
#
# $$L z = b$$
#
# where
#
# $$L = D - W$$
#
# $$D_{ij} = \delta_{ij} \sum_{k=1}^{n} W_{ik} + \lambda \delta_{is} $$
#
# $$b = \sum_{j = 1}^n w_{ij} sgn(A_{ij} - A_{ji}) + \delta_{is} \lambda C$$
#
# solve with Moore-Penrose inverse
#
# $$z^* = L^\dagger b$$
#
#

# %%
def calc_signal_flow(A):
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    
    W = (A + A.T) / 2

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    b = np.sum(W * np.sign(A - A.T), axis=1)

    L_dagger = np.linalg.pinv(L) # this is the Moore-Penrose inverse
    
    y = L_dagger @ b
    
    return y

def calc_signal_flow_reg(A, source_idx=[], m = 1, y0 = 2):
    if not isinstance(A, np.ndarray):
        A = np.array(A)

    W = (A + A.T) / 2

    D = np.diag(np.sum(W, axis=1))
    D[source_idx, source_idx] = D[source_idx, source_idx] + m

    L = D - W

    b = np.sum(W * np.sign(A - A.T), axis=1)
    b[source_idx] = b[source_idx] + m * y0

    L_dagger = np.linalg.pinv(L) # this is the Moore-Penrose inverse
    
    y = L_dagger @ b

    return y

# %% [markdown]
# # edge list and filtering

# %%
neuron_info.main_groups.value_counts()


# %%
# filter edgelist, keeping only nodes that have a type starting with 'OL' in neuron_info
onn = neuron_info[neuron_info.main_groups.str.startswith('OL')]

el_onn = edgelist[edgelist.bodyId_pre.isin(onn.bodyId) & edgelist.bodyId_post.isin(onn.bodyId)]
el_onn.shape

# %%
# cell graph
el0 = neuprint.merge_neuron_properties(neuron_info, edgelist, ['type','instance'])
# el0 = neuprint.merge_neuron_properties(onn, el_onn, ['type','instance'])

# filter
el = el0[el0['weight'] >= 5]
print(el.shape)

# inst graph
el_inst = el.groupby(['instance_pre', 'instance_post']).agg(
    weight = ("weight", "sum")
).reset_index()

# %%
# # normalize weight by the bodyId_post upstream value in neuron_info
# el = el.merge(neuron_info[['bodyId', 'upstream']], left_on='bodyId_post', right_on='bodyId', how='left')
# el['weight_norm'] = el['weight'] / el['upstream']

# %% [markdown]
# # make conn matrix

# %%
# %%time

# make connectivity matrix
# A = neuprint.connection_table_to_matrix(el, 'instance', sort_by='instance') #cell type
A = neuprint.connection_table_to_matrix(el, 'bodyId', sort_by='instance') # cell, on cluster

# add missing columns/rows with 0
# combine A.columns + A.index
sort_all =  sorted(set(A.columns) | set(A.index))

missing_columns = set(sort_all) - set(A.columns)
for col in missing_columns:
    A[col] = 0
# reorder columns to match the order
A = A[sort_all]

missing_rows = set(sort_all) - set(A.index)
for row in missing_rows:
    A.loc[row] = 0
# reorder rows to match the order
A = A.reindex(sort_all)

A.shape

# %%
# save cell-lvl A
# A.to_pickle(CACHE_DIR / 'network' / 'conn_matrix_cell.pkl')
# A.to_pickle(CACHE_DIR / 'network' / 'conn_matrix_cell_onn.pkl')

# save A.index
# pd.Series(A.index).to_pickle(Path(result_dir, 'conn_matrix_index_cell.pkl'))
# pd.Series(A.index).to_pickle(Path(result_dir, 'conn_matrix_index_cell_onn.pkl'))

# %%
# load, 
A = pd.read_pickle(CACHE_DIR / 'network' / 'conn_matrix_cell_onn.pkl')

# %% [markdown]
# # compute depth

# %%
# source idx
source_ids = neuron_info[neuron_info['type'].str.contains(r'(^L[1-3]{1})|(^R[78]{1})|(HBeyelet)')]['bodyId']
# which A.index is in source_ids
source_ind = np.where(A.index.isin(source_ids))[0]

# %%
# DEBUG
W = (A + A.T) / 2

D = np.diag(np.sum(W, axis=1))
D[source_ind, source_ind] = D[source_ind, source_ind] + m


L = D - W

b = np.sum(W * np.sign(A - A.T), axis=1)


b[source_ind] = b[source_ind] - m * y0

# L_dagger = np.linalg.pinv(L) # this is the Moore-Penrose inverse

# y = L_dagger @ b

# %%
# normalize A by column sum
A_norm = A.div(A.sum(axis=0), axis=1)
# remove na
A_norm = A_norm.fillna(0)

# %%
# %%time
# z = calc_signal_flow(A)
z = calc_signal_flow_reg(A, source_ind, m = 1000, y0= -10)

# save z
# np.save(Path(result_dir, 'graph_height_instance.npy'), z)
# np.save(Path(result_dir, 'graph_height_cell.npy'), z)
# np.save(Path(result_dir, 'graph_height_cell_onn.npy'), z)
# np.save(Path(result_dir, 'graph_height_cell_onn_reg_1000_10.npy'), z)

# %% [markdown]
# ### load z

# %%
# load 
# z = np.load(Path(result_dir, 'graph_height_cell_reg_10_2.npy'))
z = np.load(Path(result_dir, 'graph_height_cell.npy'))

# z = np.load(Path(result_dir, 'graph_height_cell_norm.npy'))
# z = np.load(Path(result_dir, 'graph_height_cell_norm_reg_2_2.npy'))

# z = np.load(Path(result_dir, 'graph_height_cell_onn.npy'))

A_index = pd.read_pickle(Path(result_dir, 'conn_matrix_index_cell.pkl'))
# A_index = pd.read_pickle(Path(result_dir, 'conn_matrix_index_cell_onn.pkl'))

# %%
# pair up with instance and type
nz = pd.DataFrame({
    'bodyId':A_index,
    'z':z})

nz = pd.merge(nz, neuron_info[['bodyId','instance','type','main_groups']], how='left', on='bodyId')

# %%
# forwardness, forward edge wt / total
df = pd.merge(edgelist, nz[['bodyId','z']], left_on='bodyId_pre', right_on='bodyId', how='left')
df = pd.merge(df, nz[['bodyId','z']], left_on='bodyId_post', right_on='bodyId', how='left')
# rename z_x to z_pre, z_y to z_post
df.rename(columns={'z_x':'z_pre', 'z_y':'z_post'}, inplace=True)
# remove bodyId_x, bodyId_y
df.drop(columns=['bodyId_x', 'bodyId_y'], inplace=True)

df['dz'] = df['z_post'] - df['z_pre']

# %%
df[df['dz'] < 0]['weight'].sum() / df['weight'].sum()

# %%
# groupby instance and find the mean z
nz_inst = nz.groupby(['instance']).agg(
    z = ('z', 'mean'),
    std = ('z', 'std'),
    inst_count = ('z', 'count')
    ).reset_index().sort_values(by='z',ascending=True)

nz_inst['cv'] = nz_inst['std'] / nz_inst['z']

# %%
# replace nan in cv with 0
nz_inst['cv'] = nz_inst['cv'].fillna(0)
nz_inst['std'] = nz_inst['std'].fillna(0)
# take the abs
nz_inst['cv'] = nz_inst['cv'].abs()

# %%
# plot z vs variation, size by count
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=nz_inst['z'],
    # y=nz_inst['cv'],
    y=nz_inst['std'],
    mode='markers',
    marker=dict(
        size= nz_inst['inst_count']**0.4 + 3,
        showscale=True
    ),
    text=nz_inst['instance']
))

fig.update_layout(
    title="CV vs Z",
    xaxis_title="Z",
    yaxis_title="std",
    showlegend=False
)

# ylim
# fig.update_yaxes(range=[0, 1])

fig.show()

# save
# pio.write_html(fig, str(result_dir.joinpath('z_variation.html')))


# %%
# nz_inst_100_3 = nz_inst.copy()
# nz_inst_10_3 = nz_inst.copy()
# nz_inst_1000_10 = nz_inst.copy()
# nz_inst_10_2 = nz_inst.copy()

# nz_inst_norm = nz_inst.copy()
nz_inst_norm_reg_2_2 = nz_inst.copy()

# %%
np.std(nz[nz.instance == 'Li30_R']['z'], ddof=1)

# %%
nz_inst

# %%
# hist
plt.hist(nz_inst['cv'], bins=100)
# plt.hist(z, bins=1000)
plt.show()

# %%
# find range of z
print(np.max(z))
print(np.min(z))

# %%
# hist
plt.hist(nz_inst['z'], bins=100)
# plt.hist(z, bins=1000)
plt.show()

# %%
# hist
plt.hist(z, bins=1000)
plt.show()

# %%
nz_inst_gp = nz_inst.merge(neuron_info[['instance','type','main_groups']], how='left', on='instance')


# %%
nz_inst_ol = nz_inst_gp[nz_inst_gp['main_groups'].str.contains('OL')]
nz_inst_ol['main_groups'].unique()

# %% [markdown]
# # Positions in x-axix

# %%
# # cell graph
# el = el0[el0['weight'] >= 5]

# # type graph
# el = el.groupby(['instance_pre', 'instance_post']).agg(
#     weight = ("weight", "sum")
# ).reset_index()

# %%
el_inst.head()

# %%
# make connectivity matrix
A = neuprint.connection_table_to_matrix(el_inst, 'instance', sort_by='instance') #cell type

# add missing columns/rows with 0
# combine A.columns + A.index
sort_all =  sorted(set(A.columns) | set(A.index))

missing_columns = set(sort_all) - set(A.columns)
for col in missing_columns:
    A[col] = 0
# reorder columns to match the order
A = A[sort_all]

missing_rows = set(sort_all) - set(A.index)
for row in missing_rows:
    A.loc[row] = 0
# reorder rows to match the order
A = A.reindex(sort_all)

A.shape

# %%
# normalize A by column sum
A_norm = A.div(A.sum(axis=0), axis=1)
# remove na
A_norm = A_norm.fillna(0)

# adjacency matrix W and laplacian matrix
W = (A_norm + A_norm.T) / 2

D = np.diag(np.sum(W, axis=1))
D_sqrt = np.diag(np.diag(D)**(-0.5))


# %%
# Ng, Jordan, & Weiss Laplacian, max ev
# set diagonal of W to 0
W_NJW = W - np.diag(np.diag(W))
L_NJW = D_sqrt @ W_NJW @ D_sqrt
# solve the eigenvalue problem
eigvals_NJW, eigvecs_NJW = np.linalg.eigh(L_NJW)

# %%
# Normalized Laplacian, L = D^(-1/2)LD^(-1/2), min ev
L = D - W 
L_norm = D_sqrt @ L.values @ D_sqrt 
# solve the eigenvalue problem
eigvals, eigvecs = np.linalg.eigh(L_norm)

# %%
x = eigvecs[:,1]
# x = eigvecs_NJW[:,1]

# %%
# check for nan in A
np.isnan(A).sum().sum()

# replace nan with 0
A = A.fillna(0)


# %% [markdown]
# # plot a graph

# %%
import networkx as nx

# %%
# node info
# combine oltypes with x
df = pd.DataFrame({
    'instance':A.index.values,
    'x':x
})

df = pd.merge(df, oltypes[['instance','main_groups']], how='left', on='instance')

# merge with nz_inst
df = pd.merge(df, nz_inst, how='left', on='instance')
df = df.rename(columns={'z':'y'})

# nt by instance
nt = neuron_info.groupby(['instance'])\
    .agg(
        nt = ('consensusNt', lambda x: x.iloc[0]) # picking the fist value of the list suffices
    ).reset_index()

# merge with neuron_info
df = pd.merge(df, nt, how='left', on='instance')
# convert to +/-1
df.loc[:,'nt'] = df['nt'].apply(lambda x: -1 if x in ['glutamate', 'gaba'] else 1)
df.set_index('instance', inplace=True)

n_info = df.copy()
# remove na
# n_info = n_info.dropna()


# %%
n_info.head()

# %%
# make weighted digraph
G = nx.from_pandas_edgelist(el, 'instance_pre', 'instance_post', edge_attr='weight', create_using=nx.DiGraph())

# # make multiDG from edge list
# G = nx.from_pandas_edgelist(conn_df, 'bodyId_pre', 'bodyId_post', edge_attr='weight', create_using=nx.MultiDiGraph(), edge_key='roi')

# add node attr
G.add_nodes_from((i, dict(d)) for i,d in n_info.iterrows())

# remove self loop
G.remove_edges_from(nx.selfloop_edges(G))

# %%
# draw graph with defined [x y] layout

# construct dict of node positions
pos = {n: (n_info.loc[n, 'x'], n_info.loc[n, 'y']) for n in G.nodes}

plt.figure(figsize=(10,10))
nx.draw(G, pos)


# %% [markdown]
# ## interactive

# %%
from pyvis.network import Network

# %%
# el.head()
G.edges.data()

# %%
# net2.nodes[0]
# net2.edges[0]
len(net2.edges)
# net2.edges

# %%
net2 = Network(directed=True, layout=False)
net2.height = 800 #"900px" "75%"
net2.width = 1600
net2.from_nx(G)

# net2.edges = [e for e in net2.edges if e['width'] > 500]
net2.edges = []
# for i in range(len(net2.edges)):
#     net2.edges[i]['width'] = 2

for i in range(len(net2.nodes)):
    net2.nodes[i]['size'] = 10
    net2.nodes[i]['x'] = net2.nodes[i]['x']*20000
    net2.nodes[i]['y'] = net2.nodes[i]['y']*500

# %%
net2.height = 800 #"900px" "75%"
net2.width = 1600

net2.toggle_physics(False)
net2.show_buttons(filter_=['node', 'edge', 'physics'])

HTML_FIG_DIR.mkdir(parents=True, exist_ok=True)
net2.show(str(HTML_FIG_DIR / "drawDG_cell.html"), notebook=False)
# net2.show()

# %% [markdown]
# # flow direction wrt to layers

# %%
# DEBUG
fig_n = navis.plot3d(
    ss,
    soma=True,
    connectors=True,
    # color='black', 
    linewidth=2,
    inline=False, backend='plotly')

fig_pt = go.Figure(
    data=[go.Scatter3d(
        x=xyz_flow[:,0],
        y=xyz_flow[:,1],
        z=xyz_flow[:,2],
        mode='markers',
        marker=dict(
            size=10,
            color='cyan',                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )
])

# fig_col = px.scatter_3d(xyzpq,
#     x='x', y='y', z='z',
#     title=('med col'),
#     hover_name='bodyId',
#     hover_data=['p', 'q'])

# fig_col.update_traces(marker_size = 6, marker={"color":"gray"}, opacity=0.2)

# fig_mesh = navis.plot3d(
#     [ME_R, LO_R, LOP_R]
#     , color=['yellow','yellow','grey']
#     , alpha=0.2
#     , inline=False
#     , backend='plotly')

fig = go.Figure(data= fig_n.data + fig_pt.data )

fig.update_layout(autosize=False, width=600, height=600)
# fig.update_layout(margin={"l":0, "r":0, "b":0, "t":0})

fig.show()

# %%

df = np.array([[222333, 111222, 4],
          [222333, 444555, 4]])


# %%
df = pd.DataFrame(df, columns=['bodyId_pre', 'bodyId_post', 'weight'])

# %%
# rename bodyIds to small integers starting from 0, and make a lookup table between the two
lookup = pd.DataFrame(pd.concat([df['bodyId_pre'], df['bodyId_post']]).unique(), columns=['bodyId'])
lookup['node'] = range(len(lookup))
# map to the lookup table
df['source'] = df['bodyId_pre'].map(lookup.set_index('bodyId')['node'])
df['target'] = df['bodyId_post'].map(lookup.set_index('bodyId')['node'])
# drop the original columns
df = df.drop(columns=['bodyId_pre', 'bodyId_post'])


# %%
lookup

# %%
# rename columns
df = df.rename(columns={'source':'source', 'target':'target', 'weight':'value'})


