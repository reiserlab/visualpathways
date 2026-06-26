# cluster VPNs based their incoming connections in OL

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

import scipy
# from scipy.cluster.hierarchy import dendrogram, linkage, inconsistent, fcluster
from scipy.cluster import hierarchy

from utils.clustering import augmented_dendrogram, cluster_to_singleton, split_cluster

# import neuprint

# %%
result_dir = PROJECT_ROOT / 'results' / 'pathways'
result_dir.mkdir(parents=True, exist_ok=True)

from utils.config import DATA_DIR

oltypes = pd.read_pickle(Path(DATA_DIR, 'oltypes.pkl'))
# conn_roi = pd.read_pickle(Path(DATA_DIR, 'conn_roi.pkl'))
neuron_info = pd.read_pickle(Path(DATA_DIR, 'neuron_info_ol.pkl'))
edgelist = pd.read_pickle(Path(DATA_DIR, 'edgelist_ol.pkl'))
# syn = pd.read_pickle(Path(DATA_DIR, 'syn.pkl'))

# %% [markdown]
# # edge list and filtering

# %%
neuron_info[neuron_info.main_groups == 'VPN']['type'].nunique()

# %%
# load feature vectors cvs data
fvec = pd.read_csv(Path(DATA_DIR, 'photorecff_vpn_feature_vec.csv'), index_col=0) 

# %%
# hierarchical clustering
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage

# Z = hierarchy.linkage(fvec, method="single", metric='cityblock')
# Z = hierarchy.linkage(fvec, method="average", metric='cityblock')
# Z = hierarchy.linkage(fvec, method="centroid", metric='euclidean')
Z = hierarchy.linkage(fvec, method="ward", metric='euclidean', optimal_ordering=False)

R = hierarchy.inconsistent(Z, d=2)
N = int(Z.shape[0]+1)

# %% [markdown]
# ### define cluster criterion to generate flat clusters

# %%
print(Z.shape)
# print(np.quantile(np.diff(Z[:,2]), 0.99))
xlim0 = 340

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.plot(Z[:,2], '-+')
plt.xlim(xlim0, Z.shape[0]+1)
# plt.ylim(0, 2000)
plt.title('cluster height')

plt.subplot(132)
plt.plot(np.diff(Z[:,2]), '-+')
plt.title('cluster height difference')
plt.xlim(xlim0, Z.shape[0]+1)

plt.subplot(133)
plt.plot(np.diff(np.diff(Z[:,2])), '-+')
plt.title('double difference')
plt.xlim(xlim0, Z.shape[0]+1)

# %%
# T= hierarchy.fcluster(Z, t= 10, criterion= 'distance')
# T= hierarchy.fcluster(Z, t= 1.13, criterion= 'hierarchy.inconsistent')
# T= hierarchy.fcluster(Z, t= 1, criterion= 'monocrit', monocrit= hierarchy.maxRstat(Z, R, 3))
# T= hierarchy.fcluster(Z, t=10, criterion='maxclust_monocrit', monocrit= hierarchy.maxRstat(Z, R, 3))

nclu = 2 + 8
# T= hierarchy.fcluster(Z, t= nclu, criterion='maxclust') #max No. of clusters

# T= hierarchy.fcluster(Z, t= 500, criterion= 'distance')

# T= hierarchy.fcluster(Z, t= 50, criterion='monocrit', monocrit=Z[:,2]) #max No. of clusters
# print(T)
# print(len(T))
# np.unique(T)


T= hierarchy.fcluster(Z, t=4, criterion='maxclust')
T

# %%
import cmap

# hierarchy.set_link_color_palette(['m', 'c', 'y', 'g', 'r', 'b', 'k'])
# hierarchy.set_link_color_palette(None)  # reset to default after use

# set leaf colors and link colors
pal=[i.hex for i in cmap.Colormap('seaborn:tab20').iter_colors()] # pallette
dflt_col = "#808080"   # Unclustered gray

leaf_colors = [pal[i-1] for i in T] # leaf colors
link_cols = {} # link colors
for i, i12 in enumerate(Z[:,:2].astype(int)):
  c1, c2 = (link_cols[x] if x > len(Z) else leaf_colors[x]
    for x in i12)
  link_cols[i+1+len(Z)] = c1 if c1 == c2 else dflt_col


# Plot
fig = plt.figure(figsize=(20, 6))

dn = augmented_dendrogram(Z, p= 8, distance_sort= 'ascending', truncate_mode='level', orientation='top', show_leaf_counts=True, get_leaves= True, link_color_func=lambda x: link_cols[x])
# dn = augmented_dendrogram(Z, orientation='top', show_leaf_counts=True, get_leaves= True)

plt.show()

# %%

n= int(Z.shape[0]+1)
ii = np.int32(Z[int(max(Z[-1, 0:2]))+1-n, 0:2]) - n
print(ii)
print(Z[ii,:])
singleton, Z_ind = cluster_to_singleton(Z, ii[0])

# %%
T3, _ = split_cluster(Z,T)
