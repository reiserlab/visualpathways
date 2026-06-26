# %%
%load_ext autoreload
%autoreload 2

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
import alphashape
from shapely.geometry import LineString
from scipy.spatial.distance import cdist

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
from utils.config import (
    CACHE_DIR, DATA_DIR, DATASET, FIG_DIR, HIT_THRE, N_FLOW, PARAMS_DIR, SIDE, SIDE_CHAR,
)
# DATA_DIR.mkdir(parents=True, exist_ok=True)

result_dir = FIG_DIR / 'quan_propagation'
result_dir.mkdir(parents=True, exist_ok=True)

cache_dir = CACHE_DIR / 'quan_propagation'
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Some setups

# %% [markdown]
# ## vpn types, rois

# %%
# # cell type table xlsx
# oltypes = pd.read_excel(PARAMS_DIR / 'Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx')
# print(oltypes.shape)

# oltypes_nonvpn = oltypes[~oltypes['main groups'].str.contains('VPN')]
# oltypes_vpn = oltypes[oltypes['main groups'].str.contains('VPN')]
# oltypes_vcn = oltypes[oltypes['main groups'].str.contains('VCN')]
# oltypes_ol = oltypes[oltypes['main groups'].str.contains('^ON')]

# %%
from utils.query_roi import get_primary_rois
rois_cb = get_primary_rois('CentralBrain')
len(rois_cb)

# %% [markdown]
# ## 2D hex grid coord

# %%
# edge column
from utils.hex_geometry import fl_get_edge_ids
from utils.hex_hex import all_hex_df

all_hex = all_hex_df()
edge_ids =  fl_get_edge_ids(all_hex)

# center = [hex1, hex2] = [18, 19]
hex_offset = [18, 19]

# rotate
all_hex['p'] = all_hex['hex2_id'] - hex_offset[1]
all_hex['q'] = all_hex['hex1_id'] - hex_offset[0]
all_hex['v'] = all_hex['p'] + all_hex['q']
all_hex['h'] = all_hex['q'] - all_hex['p']
all_hex['h'] = - all_hex['h'] # chiasm

print(all_hex['hex1_id'].max(), all_hex['hex1_id'].min(), all_hex['hex2_id'].max(), all_hex['hex2_id'].min())
print(all_hex['p'].max(), all_hex['p'].min(), all_hex['q'].max(), all_hex['q'].min())
print(all_hex['h'].max(), all_hex['h'].min(), all_hex['v'].max(), all_hex['v'].min())

# rotate
edge_ids['p'] = edge_ids['hex2_id'] - hex_offset[1]
edge_ids['q'] = edge_ids['hex1_id'] - hex_offset[0]
edge_ids['v'] = edge_ids['p'] + edge_ids['q']
edge_ids['h'] = edge_ids['q'] - edge_ids['p']
edge_ids['h'] = - edge_ids['h'] # chiasm

# %%
# DEBUG, guess coords from meta
all_hex['xj'] =all_hex['hex1_id'] + all_hex['hex2_id']
all_hex['yj'] =all_hex['hex1_id'] - all_hex['hex2_id']
print(all_hex['xj'].max(), all_hex['xj'].min(), all_hex['yj'].max(), all_hex['yj'].min())

# %%
# plot the edge ids
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(edge_ids['h'], edge_ids['v'], 'o', markersize=1)
ax.set_aspect('equal')
plt.show()

# %%
# Create alpha shape from edge_ids[['h', 'v']]
points = edge_ids[['h', 'v']].values
alpha = 0.05  # You may need to tune this parameter
boundary_alpha = alphashape.alphashape(points, alpha)

# %%
# eg. intersection with a ray
pt = (20, 40)
ray = LineString([(0, 0), pt])
boundary = boundary_alpha.boundary

# plot the ray and boundary
fig, ax = plt.subplots(figsize=(2, 2))
# Plot the alpha shape boundary
if hasattr(boundary_alpha, 'exterior'):
    x, y = boundary_alpha.exterior.xy
    ax.plot(x, y, color='black', alpha=0.5)
else:
    # If boundary_alpha is a MultiPolygon, plot each
    for geom in boundary_alpha.geoms:
        x, y = geom.exterior.xy
        ax.plot(x, y, color='black', alpha=0.5)
# Overlay the points
ax.plot(edge_ids['h'], edge_ids['v'], 'o', markersize=1, color='gray')
ax.plot(pt[0], pt[1], 'o', markersize=3, color='black')
# Plot the ray
ax.plot([ray.xy[0][0], ray.xy[0][1]], [ray.xy[1][0], ray.xy[1][1]], 
        color='red', alpha=0.5)
ax.set_aspect('equal')

# %%
# Example
intersection = ray_intersection_with_boundary(pt, boundary_alpha)
print(intersection)

# %% [markdown]
# ## retinotopy color

# %%
all_hex = all_hex_df()
all_hex['p'] = all_hex['hex2_id'] - hex_offset[1]
all_hex['q'] = all_hex['hex1_id'] - hex_offset[0]

all_hex['v'] = all_hex['p'] + all_hex['q']
all_hex['h'] = all_hex['q'] - all_hex['p']
all_hex['h'] = - all_hex['h'] # chiasm
all_hex['x'] = all_hex['h'] + 1
all_hex['y'] = all_hex['v'] + 37

all_hex = all_hex[~ ((all_hex['v'] == 0) & (all_hex['h'] == 0))]

# %%
# make color map based on mapping to CIELAB
#  to disk
for i, row in all_hex.iterrows():
    pt = (row['h'], row['v'])
    r = np.sqrt(pt[0]**2 + pt[1]**2)
    mul = 40 / r # 35~max boundary distance
    pt = (pt[0] * mul, pt[1] * mul)
    # angle
    angle = np.arctan2(row['v'], row['h'])
    # radius
    intersection = ray_intersection_with_boundary(pt, boundary_alpha)
    radius = r / np.sqrt(intersection[0]**2 + intersection[1]**2) *0.99
    # save r and angle
    all_hex.at[i, 'r'] = radius
    all_hex.at[i, 'theta'] = angle

# disk to lab
# Use apprehension for vectorized mapping from disk to LAB
r = all_hex['r'].values
theta = all_hex['theta'].values
L = np.full_like(r, 75)
lab = np.array([disk_to_lab(ri, ti, li) for ri, ti, li in zip(r, theta, L)])
all_hex[['L', 'a', 'b']] = lab

# lab to rgb
# Use apprehension for vectorized mapping from LAB to RGB
lab = all_hex[['L', 'a', 'b']].values
_rgb = np.array([lab_to_rgb(Li, ai, bi) for Li, ai, bi in zip(lab[:, 0], lab[:, 1], lab[:, 2])])

# Convert _rgb to hex color strings for Plotly
_hex = [mpl.colors.to_hex(rgb) for rgb in _rgb]

all_hex['hex'] = _hex

# %%
# plot syn_post_mean[['h','v']], color by rgb
fig, ax = plt.subplots(figsize=(4, 4))
x, y = boundary_alpha.exterior.xy
ax.plot(np.array(x)+1, (np.array(y)+37)/np.sqrt(3), color='black', alpha=0.5)
# ax.plot(edge_ids['h'], edge_ids['v'], 'o', markersize=1)
ax.scatter(all_hex['x'], all_hex['y']/np.sqrt(3), marker='o', s=10, c=all_hex['hex'])
ax.set_aspect('equal')
plt.show()

# save plot
# fig.savefig(Path(result_dir, 'retinotopy_color_L75.pdf'), bbox_inches='tight')

# %%
# Use RdYlBu divergent colormap (Red-Yellow-Blue)
# Other options: 'RdBu', 'RdYlGn', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'Spectral'
cmap = plt.cm.RdYlBu  # Updated syntax for newer matplotlib versions

# normalize to 0-1 range and get hex color

h_norm = (all_hex['h'] - all_hex['h'].min()) / (all_hex['h'].max() - all_hex['h'].min())
all_hex['color_h'] = [mpl.colors.to_hex(color) for color in cmap(h_norm)]

v_norm = (all_hex['v'] - all_hex['v'].min()) / (all_hex['v'].max() - all_hex['v'].min())
all_hex['color_v'] = [mpl.colors.to_hex(color) for color in cmap(v_norm)]

# %%
# plot syn_post_mean[['h','v']], color by rgb
fig, ax = plt.subplots(figsize=(4, 4))
x, y = boundary_alpha.exterior.xy
ax.plot(x, y, color='black', alpha=0.5)
# ax.plot(edge_ids['h'], edge_ids['v'], 'o', markersize=1)
ax.scatter(all_hex['h'], all_hex['v'], marker='+', s=10, c=all_hex['color_h'])
ax.set_aspect('equal')
plt.show()

# save plot
# fig.savefig(Path(result_dir, 'retinotopy_color_L65.png'), dpi=300, bbox_inches='tight')

# %%
# plot syn_post_mean[['h','v']], color by rgb
fig, ax = plt.subplots(figsize=(4, 4))
x, y = boundary_alpha.exterior.xy
ax.plot(x, y, color='black', alpha=0.5)
# ax.plot(edge_ids['h'], edge_ids['v'], 'o', markersize=1)
ax.scatter(all_hex['h'], all_hex['v'], marker='+', s=10, c=all_hex['color_v'])
ax.set_aspect('equal')
plt.show()

# save plot
# fig.savefig(Path(result_dir, 'retinotopy_color_L65.png'), dpi=300, bbox_inches='tight')

# %% [markdown]
# ## color retinotopy in ME

# %%
# inst = 'Mi1_R'
# n_info, _ = fetch_neurons(NC(instance=inst))
# # # skeleton
# # sk = neu.fetch_skeletons(n_info['bodyId'].values)

# n_info.shape

# %%
# # syn in ME
# syn = neuprint.queries.fetch_synapses(
#     NC(bodyId= n_info['bodyId'].values),
#     SC(type='pre', rois='ME(R)', primary_only=False)
# )
# syn = syn[syn['roi'].str.contains(r'^ME_R_col')]

# # extract the 2 integers from roi column, each number is proceeded by '_'
# # assign them to 2 new columns 'hex1' and 'hex2'
# syn['hex1'] = syn['roi'].str.extract(r'_(\d+)_')[0].astype(int)
# syn['hex2'] = syn['roi'].str.extract(r'_(\d+)$')[0].astype(int)
# # offset
# syn['hex1'] = syn['hex1'] - hex_offset[0]
# syn['hex2'] = syn['hex2'] - hex_offset[1]

# # syn_pre = syn.copy()

# # group by bodyId, compute the mean and variance of [x y z], and mean of hex1 and hex2
# syn = syn.groupby('bodyId').agg(
#     {'x': ['mean', 'std'], 'y': ['mean', 'std'], 'z': ['mean', 'std'], 'hex1': ['mean'], 'hex2': ['mean']}
# ).reset_index()
# # flatten MultiIndex columns
# syn.columns = ['bodyId'] + [f"{col[0]}_{col[1]}" for col in syn.columns[1:]]

# # h and v
# syn['v'] = syn['hex1_mean'] + syn['hex2_mean']
# syn['h'] = syn['hex1_mean'] - syn['hex2_mean']
# syn['h'] = -syn['h']

# syn_pre_mean = syn.copy()
# syn.shape

# %%
# # make color map based on mapping to CIELAB
# #  to disk
# for i, row in syn_pre_mean.iterrows():
#     pt = (row['h'], row['v'])
#     r = np.sqrt(pt[0]**2 + pt[1]**2)
#     mul = 35 / r # 35~max boundary distance
#     pt = (pt[0] * mul, pt[1] * mul)
#     # angle
#     angle = np.arctan2(row['v'], row['h'])
#     # radius
#     intersection = ray_intersection_with_boundary(pt, boundary_alpha)
#     radius = r / np.sqrt(intersection[0]**2 + intersection[1]**2)
#     # save r and angle
#     syn_pre_mean.at[i, 'r'] = radius
#     syn_pre_mean.at[i, 'theta'] = angle

# # disk to lab
# # Use apprehension for vectorized mapping from disk to LAB
# r = syn_pre_mean['r'].values
# theta = syn_pre_mean['theta'].values
# L = np.full_like(r, 75)
# lab = np.array([disk_to_lab(ri, ti, li) for ri, ti, li in zip(r, theta, L)])
# syn_pre_mean[['L', 'a', 'b']] = lab

# # lab to rgb
# # Use apprehension for vectorized mapping from LAB to RGB
# lab = syn_pre_mean[['L', 'a', 'b']].values
# syn_post_rgb = np.array([lab_to_rgb(Li, ai, bi) for Li, ai, bi in zip(lab[:, 0], lab[:, 1], lab[:, 2])])

# # Convert syn_post_rgb to hex color strings for Plotly
# syn_post_hex = [mpl.colors.to_hex(rgb) for rgb in syn_post_rgb]

# %%
# # plot syn_pre_mean[['h','v']], color by rgb
# fig, ax = plt.subplots(figsize=(4, 4))
# x, y = boundary_alpha.exterior.xy
# ax.plot(x, y, color='black', alpha=0.5)
# # ax.plot(edge_ids['h'], edge_ids['v'], 'o', markersize=1)
# ax.scatter(syn_pre_mean['h'], syn_pre_mean['v'], marker='+', s=10, c=syn_post_rgb)
# ax.set_aspect('equal')
# plt.show()

# # # save plot
# # fig.savefig(result_dir / 'vpn_retinotopy' / f"syn_pre_{inst}.png", dpi=300, bbox_inches='tight')

# %% [markdown]
# # color neuron by hex, then export tbars (to NG)

# %% [markdown]
# ## load data

# %%
syn0 = pd.read_pickle(Path(cache_dir, 'tbar_vp_cb.pkl'))

# %%
# lr = 'r'

# meta_cb_vpn = pd.read_pickle(Path(cache_dir, f'vp_cb_hit_vic_{lr}.p'))
# print(len(meta_cb_vpn))

# thr_vic = 5e-4
# inst = meta_cb_vpn.groupby('instance').agg({'VIC':'median'}).reset_index()
# inst = inst[inst.VIC > thr_vic]['instance'].unique()
# meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.VIC > thr_vic]
# meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.instance.isin(inst)]
# print(len(meta_cb_vpn))

# # change col names
# meta_cb_vpn.rename(columns={'VIC':'vision', 'hitting_time':'ht'}, inplace=True)

# %%
# # meta = pd.read_csv(DATA_DIR / f'{DATASET}_meta.csv')
# # vic_r = pd.read_parquet(Path(DATA_DIR, 'proc', 'malecns_v1.0_r_ol_cb_vic_raw.parquet'))
# ht_r = pd.read_csv(DATA_DIR / f'{DATASET}_r_flow_{N_FLOW}step_{HIT_THRE}thre_hit.csv')

# %%
meta_ahull = pd.read_pickle(Path(DATA_DIR, 'meta_VP_CB_ahull_cumsum60.pkl'))
print(meta_ahull.shape)

# %%
meta_ahull = pd.read_pickle(Path(DATA_DIR, 'meta_VP_CB_ahull_cumsum60.pkl'))
print(meta_ahull.shape)

# remove nan in ahull_size
meta_cb_vpn = meta_ahull.loc[meta_ahull['ahull_size'].notna()].copy()
# add ratio of area to count
meta_cb_vpn['area_to_count'] = meta_cb_vpn['ahull_size'] / meta_cb_vpn['ahull_kept_points']

meta_cb_vpn.rename(columns={'hitting_time':'ht', 'ahull_size':'area_fit', 'ahull_com_x':'x0', 'ahull_com_y':'y0'}, inplace=True)
print(meta_cb_vpn.shape)

# %%
# lr = 'r'

# meta_cb_vpn = pd.read_pickle(Path(cache_dir, f'vp_cb_hit_vic_{lr}.p'))
# meta_cb_vpn.head()

# %%
# # from Judith, 
# isleft = '' if SIDE == 'right' else '_left'

# vp_cb_vic = pd.read_pickle(CACHE_DIR / f'vp_cb{isleft}_vic_w_hit_df.p')
# fit_rf = pd.read_pickle(CACHE_DIR / f'{DATASET}{isleft}_w_hit_fit_rf.p')

# meta_cb_vpn = pd.merge(
#     fit_rf[['bodyId','instance','size','r2', 'hitting_time', 'main_groups', 'x0', 'y0']],
#     vp_cb_vic[['bodyId','VIC']],
#     on='bodyId', how='inner'
# )
# meta_cb_vpn.rename(columns={'size':'area_fit', 'hitting_time':'ht'}, inplace=True)
# print(meta_cb_vpn.shape)
# # thr_vic = 5e-4
# # meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['VIC'] > thr_vic]
# # thr_r2 = 0.05
# # meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['r2'] > thr_r2]
# meta_cb_vpn.shape

# %%
# change coord
# meta_cb_vpn['hex1'] = (meta_cb_vpn['y0']* np.sqrt(3) - meta_cb_vpn['x0'])/ 2
# meta_cb_vpn['hex2'] = (meta_cb_vpn['y0']* np.sqrt(3) + meta_cb_vpn['x0'])/ 2
meta_cb_vpn['hex1'] = (meta_cb_vpn['y0'] - meta_cb_vpn['x0'])/ 2
meta_cb_vpn['hex2'] = (meta_cb_vpn['y0'] + meta_cb_vpn['x0'])/ 2

meta_cb_vpn['p'] = meta_cb_vpn['hex2'] - hex_offset[1]
meta_cb_vpn['q'] = meta_cb_vpn['hex1'] - hex_offset[0]

# rotate
meta_cb_vpn['v'] = meta_cb_vpn['p'] + meta_cb_vpn['q']
meta_cb_vpn['h'] = meta_cb_vpn['q'] - meta_cb_vpn['p']
meta_cb_vpn['h'] = - meta_cb_vpn['h'] # chiasm

# %%
# check alignment of [h v] with the boundary
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(x, y, color='black', alpha=0.5)
ax.scatter(meta_cb_vpn['h'], meta_cb_vpn['v'], marker='o', s=1, c='red')
ax.set_aspect('equal')
plt.show()

# %%
# # save all_hex and edge_ids
# all_hex.to_pickle(Path(cache_dir, 'all_hex_df.pickle'))
# edge_ids.to_pickle(Path(cache_dir, 'edge_ids.pickle'))

# %%
# map to CIELAB color, add r, theta
meta_cb_vpn = hex2cielab(meta_cb_vpn, boundary_alpha, L_val=65, mul_factor=50, radius_scale=1.2)

# %%
# color along h or v, 
# https://matplotlib.org/stable/users/explain/colors/colormaps.html
cmap = plt.cm.managua
# cmap = plt.cm.spring

# normalize to 0-1 range and get hex color

h_norm = (meta_cb_vpn['h'] - meta_cb_vpn['h'].min()) / (meta_cb_vpn['h'].max() - meta_cb_vpn['h'].min())
meta_cb_vpn['color_h'] = [mpl.colors.to_hex(color) for color in cmap(h_norm)]

v_norm = (meta_cb_vpn['v'] - meta_cb_vpn['v'].min()) / (meta_cb_vpn['v'].max() - meta_cb_vpn['v'].min())
meta_cb_vpn['color_v'] = [mpl.colors.to_hex(color) for color in cmap(v_norm)]

# %%
# syn = pd.merge(syn0[['bodyId','roi','x','y','z']],
#             meta_cb_vpn[['bodyId', 'instance', 'VIC', 'ht', 'area_fit', 'area_to_count', 'hex1', 'hex2', 'h','v', 'r','theta', 'color_h', 'color_v', 'color_lab']], 
#             # meta_cb_vpn[['bodyId', 'instance', 'VIC', 'ht', 'area_fit', 'r2', 'x0','y0', 'r', 'theta', 'color_lab']], 
#                on='bodyId', how='inner')
# print(syn.shape)

# %% [markdown]
# ## coverage

# %%
#  r and theta
for i, row in df_fit.iterrows():
    pt = (row['h'], row['v'])
    r = np.sqrt(pt[0]**2 + pt[1]**2)
    mul = 50 / r # make sure it intersects
    pt = (pt[0] * mul, pt[1] * mul)
    # angle
    angle = np.arctan2(row['v'], row['h'])
    # radius
    intersection = ray_intersection_with_boundary(pt, boundary_alpha)
    radius = r / np.sqrt(intersection[0]**2 + intersection[1]**2) * 1.1
    # save r and angle
    df_fit.at[i, 'r'] = np.clip(0, 1, radius)
    df_fit.at[i, 'theta'] = angle

# %%
# plot df_fit r vs theta
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(df_fit['theta'], df_fit['r'], 'o', markersize=2)
ax.set_xlabel('Theta (radians)')
ax.set_ylabel('Radius (normalized)')
plt.show()

# %%
# do a density plot of r vs theta
fig, ax = plt.subplots(figsize=(4, 4))
ax.hexbin(df_fit.loc[df_fit.r2>0, 'h'], df_fit.loc[df_fit.r2>0, 'v'], gridsize=50, cmap='Blues')
ax.set_xlabel('Theta (radians)')
ax.set_ylabel('Radius (normalized)')
# color scale
plt.colorbar(ax.collections[0], ax=ax, label='Density')
# set color scale limits


plt.show()

# %% [markdown]
# ## for Sandro

# %%
roi = 'SMP(R)'

# %%
syn.head()

# %%
# df = syn.loc[syn['roi'] == roi, ['bodyId','roi','x','y','z','x0','y0']].copy()
df = syn[syn['roi'] == roi].copy()
df.to_csv(Path(result_dir, f'tbar_res_{roi}.csv'), index=False)

# %%
df.bodyId.nunique()

# %% [markdown]
# ## for Stuart

# %%
np.nanpercentile(all_hex.h, [5,95]), np.nanpercentile(all_hex.v, [5,95])
# # syn.h.describe(), syn.v.describe()
# np.nanpercentile(syn.h, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]), np.nanpercentile(syn.v, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])

# np.nanpercentile(syn.h, [5,95]), np.nanpercentile(syn.v, [5,95])
np.nanpercentile(syn.h, [1,99]), np.nanpercentile(syn.v, [1,99])

# %%
syn = pd.merge(syn0[['bodyId','roi','x','y','z']],
               meta_cb_vpn[['bodyId', 'instance', 'VIC', 'ht', 'area_fit', 'area_to_count', 'hex1', 'hex2', 'h','v', 'r','theta']], 
               on='bodyId', how='inner')
print(syn.shape)

tbar_ng = syn[['bodyId', 'roi', 'x', 'y', 'z', 'VIC', 'ht', 'area_fit', 'area_to_count', 'hex1', 'hex2', 'r', 'theta', 'v', 'h',]].copy()

# save feather
tbar_ng.to_feather(Path(cache_dir, 'retinotopy_tbar_olr.feather'))

# # load feather
# tbar_ng = pd.read_feather(Path(cache_dir, 'retinotopy_tbar_olr.feather'))

# %%
tbar_ng.area_to_count.describe()

# %%
all_hex = hex2cielab(all_hex, boundary_alpha, L_val=65, mul_factor=50, radius_scale=1)
edge_ids = hex2cielab(edge_ids, boundary_alpha, L_val=65, mul_factor=50, radius_scale=1)
# save all_hex and edge_ids
all_hex.to_pickle(Path(cache_dir, 'all_hex_df.pickle'))
edge_ids.to_pickle(Path(cache_dir, 'edge_ids_df.pickle'))

# %%
# # DEBUG

# load feather
tbar_ng = pd.read_feather(Path(result_dir, 'retinotopy_tbar_olr.feather'))
# syn[~syn.bodyId.isin(tbar_ng['bodyId'].unique())].shape

# df = pd.merge(tbar_ng.drop_duplicates(subset=['bodyId'])[['bodyId','roi','r2','hex1','hex2','h','v']], 
#               syn.drop_duplicates(subset=['bodyId'])[['bodyId','roi','r2','hex1','hex2','h','v']], 
#               on='bodyId', how='inner', suffixes=('_1', '_2'))
# df.shape, syn.bodyId.nunique(), tbar_ng.bodyId.nunique()
# df

# %% [markdown]
# ## 3D plot

# %%
syn = pd.merge(syn0[['bodyId','roi','x','y','z']],
               meta_cb_vpn[['bodyId', 'instance', 'VIC', 'ht', 'area_fit', 'area_to_count', 'hex1', 'hex2', 'h','v', 'r','theta', 'color_lab']], 
               on='bodyId', how='inner')
print(syn.shape)


# %%
syn[syn['roi'] == 'PB'].shape, syn[syn['roi'] == 'AOTU(R)'].shape

# %%
xyz_sample = syn[(syn['roi'].isin(['BU(R)'])) & (syn['area_to_count'] < 5)]
# xyz_sample = syn[syn['roi'] == 'AOTU(R)']

# # Create a random subset of the data
# sample_size = min(10000, len(xyz_sample) * 0.1)  
# # sample_size = len(xyz_sample) * 0.1  # Sample up to 10% of the data
# xyz_sample = xyz_sample.sample(n=int(sample_size), random_state=42)

# Create 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=xyz_sample['x'],
    y=xyz_sample['y'],
    z=xyz_sample['z'],
    mode='markers',
    marker=dict(
        size=2,
        color=xyz_sample['color_lab'],
        opacity=0.8
    ),
    text=xyz_sample['bodyId'],  # Show bodyId on hover
    hovertemplate='<b>Body ID:</b> %{text}<br>' +
                  '<b>X:</b> %{x}<br>' +
                  '<b>Y:</b> %{y}<br>' +
                  '<b>Z:</b> %{z}<br>' +
                  '<extra></extra>'
)])
fig.update_layout(
    # title=f'3D Scatter Plot of Synapses (n={sample_size})',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data',
        xaxis=dict(            showbackground=False,            zeroline=False        ),
        yaxis=dict(            showbackground=False,            zeroline=False        ),
        zaxis=dict(            showbackground=False,            zeroline=False        ),
        bgcolor="rgba(0, 0, 0, 0)"
    ),
    paper_bgcolor="rgba(0, 0, 0, 0)",
    plot_bgcolor="rgba(0, 0, 0, 0)",
    width=900,
    height=700
)

fig.show()

# %%


# %% [markdown]
# ## 3D plot checking neurons

# %%
syn[syn['roi'] == 'BU(R)'].groupby('instance').agg(
    tbar_count = ('bodyId', 'count'),
    id_count = ('bodyId', 'nunique')
).reset_index().sort_values('id_count', ascending=False).head(10)

# %%
syn.head()

# %%
syn2 = pd.merge(syn[syn['roi'] == 'BU(R)'].groupby(['bodyId']).size().reset_index(name='tbar_count'),
        syn[syn['roi'] == 'BU(R)'].drop_duplicates(subset=['bodyId']),
        on='bodyId', how='left').sort_values('tbar_count', ascending=False)

# %%
syn2

# %%
xyz_sample = syn[(syn['roi'].isin(['PB'])) & (syn['r2'] > -1)]
# xyz_sample = syn[syn['roi'] == 'AOTU(R)']

# # Create a random subset of the data
# sample_size = min(10000, len(xyz_sample) * 0.1)  
# # sample_size = len(xyz_sample) * 0.1  # Sample up to 10% of the data
# xyz_sample = xyz_sample.sample(n=int(sample_size), random_state=42)

# Create 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=xyz_sample['x'],
    y=xyz_sample['y'],
    z=xyz_sample['z'],
    mode='markers',
    marker=dict(
        size=2,
        color=xyz_sample['color_h'],
        opacity=0.8
    ),
    text=xyz_sample['bodyId'],  # Show bodyId on hover
    hovertemplate='<b>Body ID:</b> %{text}<br>' +
                  '<b>X:</b> %{x}<br>' +
                  '<b>Y:</b> %{y}<br>' +
                  '<b>Z:</b> %{z}<br>' +
                  '<extra></extra>'
)])
fig.update_layout(
    # title=f'3D Scatter Plot of Synapses (n={sample_size})',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data',
        xaxis=dict(            showbackground=False,            zeroline=False        ),
        yaxis=dict(            showbackground=False,            zeroline=False        ),
        zaxis=dict(            showbackground=False,            zeroline=False        ),
        bgcolor="rgba(0, 0, 0, 0)"
    ),
    paper_bgcolor="rgba(0, 0, 0, 0)",
    plot_bgcolor="rgba(0, 0, 0, 0)",
    width=900,
    height=700
)

fig.show()

# %% [markdown]
# # Retinotopy for VPN numerous types
# 

# %%
syn0 = pd.read_pickle(Path(cache_dir, 'tbar_vp_cb.pkl'))

# %%
# from Judith, 
isleft = '' if SIDE == 'right' else '_left'

vp_cb_vic = pd.read_pickle(CACHE_DIR / f'vp_cb{isleft}_vic_w_hit_df.p')
fit_rf = pd.read_pickle(CACHE_DIR / f'{DATASET}{isleft}_w_hit_fit_rf.p')

meta_cb_vpn = pd.merge(
    fit_rf[['bodyId','instance','size','r2', 'hitting_time', 'main_groups']],
    vp_cb_vic[['bodyId','VIC']],
    on='bodyId', how='left'
)
meta_cb_vpn.rename(columns={'VIC':'vision', 'size':'area_fit', 'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape)
# thr_vic = 5e-4
# thr_r2 = 0.05
# meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['vision'] > thr_vic]
# meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['r2'] > thr_r2]
meta_cb_vpn.shape

# %%
inst_mul = meta_cb_vpn[meta_cb_vpn['main_groups'] == 'OL output']['instance'].value_counts()
inst_mul = inst_mul[inst_mul > 10].index.tolist()
len(inst_mul)

# %%
inst_overlap = []
# loop over all inst_mul
retino_metric = pd.DataFrame(columns=['instance', 'roi', 'TP', 'RI'])
for inst in inst_mul:
    x0y0 = fit_rf.loc[fit_rf['instance'] == inst, ['bodyId','x0','y0', 'b']]
    # check distance > size b
    # Calculate pairwise distances between all points
    distances = cdist(x0y0[['x0', 'y0']], x0y0[['x0', 'y0']], metric='euclidean')
    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)
    # Find the minimum distance for each point (nearest neighbor)
    nearest_distances = distances.min(axis=1)
    # Calculate average nearest distance
    avg_nearest_distance = nearest_distances.mean()
    if avg_nearest_distance < x0y0.b.mean():
        inst_overlap.append(inst)
        continue

    syn = pd.merge(syn0, x0y0, on='bodyId', how='inner')
    # top roi
    top_rois = syn['roi'].value_counts().index.tolist()
    if len(top_rois) == 0:
        continue
    roi = top_rois[0]
    syn = syn[syn['roi'] == roi]
    pts = syn.groupby(['bodyId']).agg(
        {'x': 'mean', 'y': 'mean', 'z': 'mean', 'roi': 'first', 'x0': 'first', 'y0': 'first'}
        ).reset_index()
    df = pd.DataFrame({
        'instance': [inst],
        'roi': [roi],
        'TP': [topographic_product(pts[['x0','y0']], pts[['x','y','z']])],
        'RI': [np.mean(RI_sets(pts[['x0','y0']], pts[['x','y','z']]))]
    })
    retino_metric = pd.concat([retino_metric, df], ignore_index=True)

# save retino_metric
retino_metric.to_csv(Path(result_dir, 'retino_metric_vpn.csv'), index=False)

len(retino_metric)

# %%


# %% [markdown]
# # obs. cont. Retinotopy for CB numerous type 

# %%
inst_mul = meta_cb_vpn[meta_cb_vpn['main_groups'] == 'nonOL']['instance'].value_counts().reset_index()
# initialize new column 'count_all' with 0
inst_mul['count_all'] = 0
# n_info, _ = fetch_neurons(NC(instance=inst_mul.index.values)) -> not working ??
for inst in inst_mul.instance.values:
    n_info, _ = fetch_neurons(NC(instance=inst))
    inst_mul.loc[inst_mul.instance == inst, 'count_all'] = len(n_info)

# inst_mul = pd.merge(inst_mul, n_info['instance'].value_counts(), left_index=True, right_index=True, how='left')
inst_mul['diff'] = inst_mul['count_all'] - inst_mul['count']
inst_mul['diff_mul'] = inst_mul['count_all'] / inst_mul['count']

inst_mul0 = inst_mul.copy()
len(inst_mul)

# %%
inst_mul = inst_mul0.copy()

# inst_mul = inst_mul[~((inst_mul['diff_mul'] > 2) & (inst_mul['count'] <= 3))]
# inst_mul = inst_mul[inst_mul['count_all'] > 3]

inst_mul = inst_mul[inst_mul['count'] > 3]
# inst_mul = inst_mul[inst_mul['diff_mul'] < 2]
# inst_mul = inst_mul[inst_mul['diff'] >= 0]

inst_mul = inst_mul['instance'].tolist()
len(inst_mul)

# %%
# DEBUG
meta_cb_vpn[meta_cb_vpn.instance == 'DNpe008_L']
# is 'AOTU008_R' in inst_mul
'AOTU008_R' in inst_mul

# %%
# loop over all inst_mul
# filter for dist between [x0 y0] > b
inst_overlap = []
retino_metric = pd.DataFrame(columns=['instance', 'count', 'roi', 'TP', 'RI'])
for inst in inst_mul:
    ids = meta_cb_vpn[meta_cb_vpn['instance'] == inst]['bodyId'].values
    x0y0 = fit_rf.loc[fit_rf['bodyId'].isin(ids), ['bodyId','x0','y0', 'b']]
    # check distance > size b
    # Calculate pairwise distances between all points
    distances = cdist(x0y0[['x0', 'y0']], x0y0[['x0', 'y0']], metric='euclidean')
    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)
    # Find the minimum distance for each point (nearest neighbor)
    nearest_distances = distances.min(axis=1)
    # Calculate average nearest distance
    avg_nearest_distance = nearest_distances.mean()
    if avg_nearest_distance < x0y0.b.mean():
        inst_overlap.append(inst)
        continue

    syn = pd.merge(syn0, x0y0, on='bodyId', how='inner')
    # top roi
    top_rois = syn['roi'].value_counts().index.tolist()
    if len(top_rois) == 0:
        continue
    roi = top_rois[0]
    syn = syn[syn['roi'] == roi]
    pts = syn.groupby(['bodyId']).agg(
        {'x': 'mean', 'y': 'mean', 'z': 'mean', 'roi': 'first', 'x0': 'first', 'y0': 'first'}
        ).reset_index()
    df = pd.DataFrame({
        'instance': [inst],
        'count': len(x0y0),
        'roi': [roi],
        'TP': [topographic_product(pts[['x0','y0']], pts[['x','y','z']])],
        'RI': [np.mean(RI_sets(pts[['x0','y0']], pts[['x','y','z']]))]
    })
    retino_metric = pd.concat([retino_metric, df], ignore_index=True)

# SAVE
retino_metric.to_csv(Path(result_dir, 'retino_metric_cb.csv'), index=False)
len(retino_metric)

# %%
retino_metric[retino_metric.instance == 'AOTU008_R']

# %% [markdown]
# ## load prop and meta for plotting rf

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

# inprop = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_{SIDE_CHAR}_lat_flow_0.npz') # selected side, lateral, normalized

inprop.shape

# %%
stepsn = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_{SIDE_CHAR}_lat_flow_sum.npz')
stepsn.shape

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
idx_to_root = dict(zip(meta.idx, meta.bodyId))
idx_to_coords = dict(zip(meta.idx, meta.coords))

bodyId_to_idx = dict(zip(meta.bodyId, meta.idx))
idx_to_sign = dict(zip(meta.idx, meta.sign))
type_to_sign = {atype:idx_to_sign[idx] for idx, atype in idx_to_type.items()}

# %% [markdown]
# ## check instance

# %%
# retino_metric = pd.read_csv(Path(result_dir, 'retino_metric_vpn.csv'))
retino_metric = pd.read_csv(Path(result_dir, 'retino_metric_cb.csv'))

# %%
retino_metric['RI'] = retino_metric['RI'].astype(float).round(2)
retino_metric['TP'] = retino_metric['TP'].astype(float).round(2)
retino_metric['RI_abs'] = retino_metric['RI'].abs()

aa = retino_metric.sort_values('RI_abs', ascending=False)

# %%
# rf
# inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R', 'R7d_R', 'R8d_R'])] 
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
outidx = meta.idx[meta.cell_type_side.isin(['DNg06_L'])] 
# outidx = meta.idx[meta.bodyId.isin([24746])] 
df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords,  outidx_map = idx_to_bodyId,
                    display_threshold = 0,  display_output= False)
ci.hex_heatmap(df,custom_colorscale='Reds')

# %%
retino_metric

# %%
# rf
# inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
outidx = meta.idx[meta.bodyId.isin([135172])] 
df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords,  outidx_map = idx_to_bodyId,
                    display_threshold = 0,  display_output= False)
ci.hex_heatmap(df,custom_colorscale='Reds')

# %%
df[df > 0].dropna()

# %%
# CB3268_R x3
# ER2_d_R x3
# CB0973_R x4, weak
# CB1836_R x4

# %%
# outidx = meta.idx[meta.bodyId.isin([11924])] 
paths = ci.find_paths_of_length(inprop, inidx, outidx, target_layer_number= 4)
paths = ci.group_paths(paths, idx_to_type_side, idx_to_type_side)
# paths = ci.group_paths(paths, idx_to_coords, idx_to_type_side)
len(paths)

# %%
paths.head()

# %%
paths_filtered = ci.filter_paths(paths, threshold=0.01)
ci.plot_paths(paths_filtered, neuron_to_sign=type_to_sign, figsize=(14,4))

# %%


# %% [markdown]
# ### gallery

# %%
retino_metric

# %%
import io
from PIL import Image
import fitz  # PyMuPDF
from matplotlib.backends.backend_pdf import PdfPages
        
inidx = meta.idx[meta.cell_type.isin(['L1', 'L2', 'L3', 'R7', 'R7d', 'R8', 'R8d'])] 
# cb_ids = meta_cb_vpn.loc[meta_cb_vpn['main_groups'] == 'nonOL', 'bodyId'].unique()

# Create a PDF file to save all plots
pdf_path = Path(result_dir, 'gallery_RI_cb.pdf')

with PdfPages(pdf_path) as pdf:
    for i, row in retino_metric.iterrows():
        inst = row['instance']
        ids = meta_cb_vpn[meta_cb_vpn['instance'] == inst]['bodyId'].values
        RI = row['RI_abs']

        # - Create a new figure 
        fig_combined, axs = plt.subplots(4, 4, figsize=(8, 8))
        for j in range(len(ids)):
            outidx = meta.idx[meta.bodyId.isin([ids[j]])] 
            df = ci.result_summary(stepsn, inidx, outidx,
                                inidx_map = idx_to_coords,  outidx_map = idx_to_bodyId,
                                display_threshold = 0,  display_output= False)
            fig_rf = ci.hex_heatmap(df,custom_colorscale='Reds')
            # RF heatmap (convert Plotly to image)
            img_bytes = fig_rf.to_image(format="png", width=800, height=800, scale=2)
            img = Image.open(io.BytesIO(img_bytes))
            # tile 4x4, left to right, top to bottom
            row_idx = j // 4
            col_idx = j % 4
            axs[row_idx, col_idx].imshow(img)
            axs[row_idx, col_idx].axis('off')
            if j == 0:
                axs[row_idx, col_idx].set_title(
                    f'{inst} - bodyId: {ids[j]} - RI: {RI}',
                    fontsize=8
                )
            else:
                axs[row_idx, col_idx].set_title(
                    f'bodyId: {ids[j]}',
                    fontsize=8
                )
            # axs[1, 2].text(0.1, 0.8, textstr, transform=axs[1, 2].transAxes, fontsize=10, va='top', ha='left')
        
        plt.tight_layout()
        # Save the combined figure to the PDF
        pdf.savefig(fig_combined, dpi=300, bbox_inches='tight')
        plt.close(fig_combined)
        
print(f"Saved all plots to {pdf_path}")

# %% [markdown]
# # obs. retinotopy, per roi, neuron-level matching
# 
# for each roi, collect all visual neurons, compute tbar com and retinal coord, compute RI

# %%
isleft = '' if SIDE == 'right' else '_left'

vp_cb_vic = pd.read_pickle(CACHE_DIR / f'vp_cb{isleft}_vic_w_hit_df.p')
fit_rf = pd.read_pickle(CACHE_DIR / f'{DATASET}{isleft}_w_hit_fit_rf.p')

meta_cb_vpn = pd.merge(
    fit_rf[['bodyId','instance', 'x0', 'y0', 'size','r2', 'hitting_time', 'main_groups']],
    vp_cb_vic[['bodyId','VIC']],
    on='bodyId', how='left'
)
meta_cb_vpn.rename(columns={'VIC':'vision', 'size':'area_fit', 'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape)
thr_vic = 5e-4
thr_r2 = 0.05
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['vision'] > thr_vic]
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['r2'] > thr_r2]
meta_cb_vpn.shape

# %%
syn0 = pd.read_pickle(Path(cache_dir, 'tbar_vp_cb.pkl'))

# %%
syn = pd.merge(syn0[['bodyId', 'roi', 'x', 'y', 'z']],
               meta_cb_vpn[['bodyId', 'instance', 'ht', 'area_fit', 'x0','y0','r2', 'main_groups']], 
               on='bodyId', how='inner')
print(syn.shape)

# %%
syn.head()

# %%
# DEBUG
print(syn.bodyId.nunique())

ids = syn0.bodyId.unique()
aa = meta_cb_vpn[~meta_cb_vpn.bodyId.isin(ids)]

# %%
roi = 'AOTU(R)'

ss = syn[syn['roi'] == roi].groupby(['instance','bodyId','main_groups']).agg(
    tbar_count = ('bodyId', 'count'),
    com_x = ('x', 'mean'),
    com_y = ('y', 'mean'),
    com_z = ('z', 'mean'),
    x0 = ('x0', 'first'),
    y0 = ('y0', 'first'),
).reset_index()

# %%
ss.shape

# %%
f'{roi}', np.mean(RI_sets(ss[['x0','y0']], ss[['com_x','com_y', 'com_z']])), topographic_product(ss[['x0','y0']], ss[['com_x','com_y', 'com_z']])

# %%
topographic_product(ss[['com_x','com_y', 'com_z']], ss[['x0','y0']])

# %% [markdown]
# ## loop rois

# %%
ss = syn[syn['roi'] == 'AOTU(R)'].groupby(['bodyId']).agg(
    tbar_count = ('bodyId', 'count'),
    com_x = ('x', 'mean'),
    com_y = ('y', 'mean'),
    com_z = ('z', 'mean'),
    x0 = ('x0', 'first'),
    y0 = ('y0', 'first'),
).reset_index()

# %%
# plot ss['x0'] and ss['y0']
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(ss['x0'], ss['y0'], s=20)
# asp = 1
ax.set_aspect('equal')
plt.show()

# %%
# Initialize empty list to collect results
results_list = []

# In your loop for each roi:
for roi in rois_cb:
    ss = syn[syn['roi'] == roi].groupby(['bodyId']).agg(
        tbar_count = ('bodyId', 'count'),
        com_x = ('x', 'mean'),
        com_y = ('y', 'mean'),
        com_z = ('z', 'mean'),
        x0 = ('x0', 'first'),
        y0 = ('y0', 'first'),
    ).reset_index()

    max_nb = np.min([10, len(ss) // 10])    
    if max_nb > 2:
        # max_nb = len(ss) // 10
        # Calculate metrics
        ri = np.mean(RI_sets(ss[['x0','y0']], ss[['com_x','com_y', 'com_z']]))
        ri_x = np.mean(RI_sets(ss[['x0']], ss[['com_x','com_y', 'com_z']]))
        ri_y = np.mean(RI_sets(ss[['y0']], ss[['com_x','com_y', 'com_z']]))
        tp = topographic_product(ss[['x0','y0']], ss[['com_x','com_y', 'com_z']], max_nb=max_nb)
        tp_x = topographic_product(ss[['x0']], ss[['com_x','com_y', 'com_z']], max_nb=max_nb)
        tp_y = topographic_product(ss[['y0']], ss[['com_x','com_y', 'com_z']], max_nb=max_nb)

        # Append results
        results_list.append({
            'roi': roi,
            'RI': ri,
            'RI_x': ri_x,
            'RI_y': ri_y,
            'tp': tp,
            'tp_x': tp_x,
            'tp_y': tp_y,
            'n_n': len(ss)
        })

# Create DataFrame from list
roi_ri = pd.DataFrame(results_list)

# compute RI_max and tp_max for each row
roi_ri['RI_max'] = roi_ri[['RI','RI_x', 'RI_y']].max(axis=1)
roi_ri['tp_min'] = roi_ri[['tp','tp_x', 'tp_y']].min(axis=1)

# %%
roi_ri.tp.hist(bins=30)

# %%
# plot RI vs tp
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(roi_ri.loc[roi_ri.n_n > 10, 'RI_max'].abs(), roi_ri.loc[roi_ri.n_n > 10,'tp_min'], s=20)
plt.show()

# %%
roi_ri_50 = roi_ri.copy()

# %%
roi_ri_10 = roi_ri.copy()

# %%


# %% [markdown]
# # pokemon plot

# %% [markdown]
# ## load

# %%
lr = 'r'
vp_cb_vic = pd.read_pickle(Path(cache_dir, f'vp_cb_hit_vic_{lr}.p'))

meta_cb_vpn = vp_cb_vic.copy()
meta_cb_vpn.rename(columns={'VIC':'vision',  'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape)

# filter for visual neurons
thr_vic = 5e-4
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['vision'] > thr_vic]
print(meta_cb_vpn.shape)

# %%
# tbars
syn0 = pd.read_pickle(Path(cache_dir, 'tbar_vp_cb.pkl'))

# %%
stepsn = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_{SIDE_CHAR}_lat_flow_sum.npz')
stepsn.shape

# %%
# LOAD Judith meta, this has AN's R7/8, 
meta = pd.read_csv(DATA_DIR / f'{DATASET}_meta.csv')

# # replace na in cell_type with 'bodyId_{bodyId}'
# meta_judith['cell_type'] = meta_judith.apply(lambda row: f"bodyId_{int(row['bodyId'])}" if pd.isna(row['cell_type']) else row['cell_type'], axis=1)

# # change right to R, left to L in side column
# meta_judith['side'] = meta_judith['side'].replace({'right': 'R', 'left': 'L'})

# meta['cell_type_side'] = meta.cell_type + '_' + meta.side
        
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

# idx_to_bodyId_cellTypeSide = dict(zip(meta.idx, meta.bodyId.astype(str) + '_' + meta.cell_type_side))
# idx_to_bodyId_cellType = dict(zip(meta.idx, meta.bodyId.astype(str) + '_' + meta.cell_type))

# idx_to_type = dict(zip(meta.idx, meta.cell_type))
# idx_to_sign = dict(zip(meta.idx, meta.sign))
# idx_to_side = dict(zip(meta.idx, meta.side))
# # idx_to_side = dict(zip(meta.idx, meta.soma_side))
idx_to_coords = dict(zip(meta.idx, meta.coords))

# type_to_nt = dict(zip(meta.cell_type, meta.nt))
# root_to_type = dict(zip(meta.bodyId, meta.cell_type))
# idx_to_root = dict(zip(meta.idx, meta.bodyId))
# type_to_sign = {atype:idx_to_sign[idx] for idx, atype in idx_to_type.items()}

bodyId_to_idx = dict(zip(meta.bodyId, meta.idx))

# idx_to_modality = dict(zip(meta.idx, meta.superclass))

sign_to_color = {1: '#EE672D', -1: '#1F4695', 0: '#979DA5'}

# %%
# edge column
from utils.hex_hex import all_hex_df
all_hex = all_hex_df()
hex_offset = [18, 19]

# rotate
all_hex['p'] = all_hex['hex2_id'] - hex_offset[1]
all_hex['q'] = all_hex['hex1_id'] - hex_offset[0]
all_hex['v'] = all_hex['p'] + all_hex['q']
all_hex['h'] = all_hex['q'] - all_hex['p']
all_hex['h'] = - all_hex['h'] # chiasm
all_hex['x'] = all_hex['h'] + 1
all_hex['y'] = all_hex['v'] + 37

# %%
from utils.core_data import get_sector_map
coords_sector = get_sector_map()

# load sector mapping
# coords_sector = pd.read_pickle(CACHE_DIR / 'coords_to_sectors.pkl')

# %% [markdown]
# ## count tbar in pokemon ball

# %%
plot_dir = Path(result_dir, 'coverage_per_roi')
plot_dir.mkdir(parents=True, exist_ok=True)

# %%
syn = pd.merge(syn0, meta_cb_vpn,  on='bodyId', how='inner')
print(syn.shape)

# %%
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R', 'R7d_R', 'R8d_R'])] 
outidx = meta.idx[meta.bodyId.isin(meta_cb_vpn['bodyId'])] 
df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

# %%
# roi = 'EB'
# tbar_count = syn[syn['roi'] == roi].groupby('bodyId').size().to_dict()
# # weight hex coord by tbar count
# df_weighted = df.copy()
# weights = pd.Series([tbar_count.get(int(col), 0) for col in df_weighted.columns.values], index=df_weighted.columns)
# df_weighted = df_weighted * weights
# df_sum = pd.DataFrame(df_weighted.sum(axis=1), columns=['wt'])
# ci.hex_heatmap(df_sum,custom_colorscale='RdBu_r', global_max=df_sum.wt.max(), global_min=-df_sum.wt.max())

# %%
# all rois
   
tbar_count = syn.groupby('bodyId').size().to_dict()
# weight hex coord by tbar count
df_weighted = df.copy()
weights = pd.Series([tbar_count.get(int(col), 0) for col in df_weighted.columns.values], 
                    index=df_weighted.columns)
df_weighted = df_weighted * weights
df_sum = pd.DataFrame(df_weighted.sum(axis=1), columns=['wt'])
# 1: A, 2: D, 3: P, 4: V, 5: C
df_sum = pd.merge(coords_sector, df_sum, left_on='coords', right_index=True, how='left')
df_plot = df_sum.groupby('sector').agg({'wt': 'sum'}).reset_index()
df_plot['wt'] = df_plot['wt'] / df_plot['wt'].sum() # normalize 

# Draw a diagram with a circle in the center and 4 sectors around it forming an annulus
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'})
# Set limits
ax.set_ylim(0, 1)
ax.set_theta_zero_location('SW')  # Set 45 degrees to top
ax.set_theta_direction(-1)  # Clockwise
r0 = 1 / np.sqrt(5)  # Inner radius

# Draw sectors
for i, row in df_plot.iterrows():
    sector = row['sector']
    wt = row['wt']
    
    if sector == 5:
        # Central disk (sector 0) - draw as a filled wedge covering all angles
        theta = np.linspace(0, 2*np.pi, 100)
        r_fill = np.full_like(theta, r0)
        ax.fill(theta, r_fill, 
                color=plt.cm.RdBu_r(wt / 0.4), 
                edgecolor='k', linewidth=0)
    else:
        # Outer 4 sectors (sectors 1-4), each spanning 90 degrees
        angle_start = np.deg2rad((sector - 1) * 90)
        angle_end = np.deg2rad(sector * 90)

        # Draw sector as a bar from inner radius to outer radius
        ax.bar(
            x=(angle_start + angle_end) / 2,  # Center angle
            height= 1 - r0,  # Height of annulus (outer - inner radius)
            width=(angle_end - angle_start),  # Width in radians (90 degrees)
            bottom= r0,  # Start at inner radius (creates the center circle)
            color=plt.cm.RdBu_r(wt / 0.4),  # Color based on weight
            edgecolor='k',
            linewidth=0,
            align='center'
        )
# remove axis labels and ticks
ax.set_xticks([])
ax.set_yticks([])
# # label sectors
# sector_labels = {1: 'Sector 1', 2: 'Sector 2', 3: 'Sector 3', 4: 'Sector 4', 5: 'Center'}
# for sector, label in sector_labels.items():
#     if sector == 5:
#         ax.text(0, 0, label, ha='center', va='center', fontsize=10)
#     else:
#         angle = np.deg2rad((sector - 1) * 90 + 45)  # Middle of the sector
#         radius = (1 + r0) / 2  # Midway in the annulus
#         ax.text(angle, radius, label, ha='center', va='center', fontsize=10)
plt.tight_layout()
# plt.show()
fig.savefig(plot_dir / f'coverage_tbar_all.pdf', bbox_inches='tight', transparent=True)
plt.close(fig)

# %%
# loop rois
c_max = []
roi_sector_rows = []

for roi in rois_cb:
    # collect tbar
    tbar_count = syn[syn['roi'] == roi].groupby('bodyId').size().to_dict()

    # weight hex coord by tbar count
    df_weighted = df.copy()
    weights = pd.Series(
        [tbar_count.get(int(col), 0) for col in df_weighted.columns.values],
        index=df_weighted.columns
    )
    df_weighted = df_weighted * weights
    df_sum = pd.DataFrame(df_weighted.sum(axis=1), columns=['wt'])

    # 1: A, 2: D, 3: P, 4: V, 5: C
    df_sum = pd.merge(coords_sector, df_sum, left_on='coords', right_index=True, how='left')
    df_plot = df_sum.groupby('sector', as_index=False).agg({'wt': 'sum'})

    total_wt = df_plot['wt'].sum()
    if total_wt > 0:
        df_plot['wt'] = df_plot['wt'] / total_wt
    else:
        df_plot['wt'] = 0.0

    # keep max for optional global color scaling
    c_max.append(df_plot['wt'].max() if len(df_plot) else 0.0)

    # record per-roi sector values, round up to 2 decimal places in wide format
    sector_map = df_plot.set_index('sector')['wt'].to_dict()
    roi_sector_rows.append({
        'roi': roi,
        'A': round(sector_map.get(1, 0.0), 2),
        'D': round(sector_map.get(2, 0.0), 2),
        'P': round(sector_map.get(3, 0.0), 2),
        'V': round(sector_map.get(4, 0.0), 2),
        'C': round(sector_map.get(5, 0.0), 2),
    })

    # Draw a diagram with a circle in the center and 4 sectors around it forming an annulus
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'})
    # Set limits
    ax.set_ylim(0, 1)
    ax.set_theta_zero_location('SW')  # Set 45 degrees to top
    ax.set_theta_direction(-1)  # Clockwise
    r0 = 1 / np.sqrt(5)  # Inner radius

    # Draw sectors
    for i, row in df_plot.iterrows():
        sector = row['sector']
        wt = row['wt']
        
        if sector == 5:
            # Central disk (sector 0) - draw as a filled wedge covering all angles
            theta = np.linspace(0, 2*np.pi, 100)
            r_fill = np.full_like(theta, r0)
            ax.fill(theta, r_fill, 
                    color=plt.cm.RdBu_r(wt / 0.4), 
                    edgecolor='k', linewidth=0)
        else:
            # Outer 4 sectors (sectors 1-4), each spanning 90 degrees
            angle_start = np.deg2rad((sector - 1) * 90)
            angle_end = np.deg2rad(sector * 90)

            # Draw sector as a bar from inner radius to outer radius
            ax.bar(
                x=(angle_start + angle_end) / 2,  # Center angle
                height= 1 - r0,  # Height of annulus (outer - inner radius)
                width=(angle_end - angle_start),  # Width in radians (90 degrees)
                bottom= r0,  # Start at inner radius (creates the center circle)
                color=plt.cm.RdBu_r(wt / 0.4),  # Color based on weight
                edgecolor='k',
                linewidth=0,
                align='center'
            )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    # plt.show()
    fig.savefig(plot_dir / f'cover_{roi}.pdf', bbox_inches='tight', transparent=True)
    plt.close(fig)


# this will be available after the loop finishes
roi_sector_df = pd.DataFrame(roi_sector_rows)
roi_sector_df['max'] = roi_sector_df[['A', 'D', 'P', 'V', 'C']].max(axis=1)
# save to csv
roi_sector_df.to_csv(Path(plot_dir, 'coverage_per_roi.csv'), index=False)

# %%


# %% [markdown]
# # obs. 3x3 or 4x3 retinotopy

# %%
isleft = '' if SIDE == 'right' else '_left'

vp_cb_vic = pd.read_pickle(CACHE_DIR / f'vp_cb{isleft}_vic_w_hit_df.p')
fit_rf = pd.read_pickle(CACHE_DIR / f'{DATASET}{isleft}_w_hit_fit_rf.p')

meta_cb_vpn = pd.merge(
    fit_rf[['bodyId','instance', 'x0', 'y0', 'size','r2', 'hitting_time', 'main_groups']],
    vp_cb_vic[['bodyId','VIC']],
    on='bodyId', how='left'
)
meta_cb_vpn.rename(columns={'VIC':'vision', 'size':'area_fit', 'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape)
thr_vic = 5e-4
thr_r2 = 0.05
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['vision'] > thr_vic]
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['r2'] > thr_r2]
meta_cb_vpn.shape

# %%
# edge column
from utils.hex_hex import all_hex_df
all_hex = all_hex_df()
hex_offset = [18, 19]

# rotate
all_hex['p'] = all_hex['hex2_id'] - hex_offset[1]
all_hex['q'] = all_hex['hex1_id'] - hex_offset[0]
all_hex['v'] = all_hex['p'] + all_hex['q']
all_hex['h'] = all_hex['q'] - all_hex['p']
all_hex['h'] = - all_hex['h'] # chiasm

all_hex['x'] = all_hex['h'] + 1
all_hex['y'] = all_hex['v'] + 37

# %%
# tbars
syn0 = pd.read_pickle(Path(cache_dir, 'tbar_vp_cb.pkl'))

# %%
syn = pd.merge(syn0, meta_cb_vpn,  on='bodyId', how='inner')
print(syn.shape)

# %% [markdown]
# ### DEBUG

# %%
# DEBUG
df = effwt_visr[str(27449)].copy()
xy_coord = df.index.to_series().str.split(',', expand=True).astype(float)
xy_coord.columns = ['h', 'v']
com_h = np.average(xy_coord['h'], weights=df.values)
com_v = np.average(xy_coord['v'], weights=df.values) / np.sqrt(3)
    
com_h, com_v

# %% [markdown]
# ## 3x3 

# %%
import scipy
stepsn = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_{SIDE_CHAR}_lat_flow_sum.npz')
stepsn.shape

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
# ### division

# %%
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
outidx = 0 
df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

# %%
xy_coord = df.index.to_series().str.split(',', expand=True).astype(float).dropna()
xy_coord.columns = ['x', 'y']

# %%
all_hex['x'] = all_hex['h'] + 1
all_hex['y'] = all_hex['v'] + 37

# %%
# # DEBUG
# # plot xy_coord
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.scatter(xy_coord['x'], xy_coord['y'], s=5)
# all_hex(xy_coord['x'], all_hex['y'], s=5)
# ax.set_aspect('equal')
# plt.show()

# %%
# # DEBUG
xy_coord.x.describe(), xy_coord.y.describe()
# all_hex.x.describe(), all_hex.y.describe()

# %%
# np.nanpercentile(fit_rf.x0, [0, 33, 67, 100]), np.nanpercentile(fit_rf.y0, [0, 33, 67, 100])

# %%
# # for [x0, y0] from fit_rf
# # divide the coordinate system into 1-9 divisions defined in a dict
# x1 = -3
# x2 = 3
# y1 = 17
# y2 = 25
# divisions = {
#     1: {'x_min': -np.inf, 'x_max': x1, 'y_min': y2, 'y_max': np.inf},   # Top-Left
#     2: {'x_min': x1, 'x_max': x2, 'y_min': y2, 'y_max': np.inf},        # Top-Center
#     3: {'x_min': x2, 'x_max': np.inf, 'y_min': y2, 'y_max': np.inf},   # Top-Right
#     4: {'x_min': -np.inf, 'x_max': x1, 'y_min': y1, 'y_max': y2},      # Middle-Left
#     5: {'x_min': x1, 'x_max': x2, 'y_min': y1, 'y_max': y2},         # Middle-Center
#     6: {'x_min': x2, 'x_max': np.inf, 'y_min': y1, 'y_max': y2},      # Middle-Right
#     7: {'x_min': -np.inf, 'x_max': x1, 'y_min': -np.inf, 'y_max': y1},  # Bottom-Left
#     8: {'x_min': x1, 'x_max': x2, 'y_min': -np.inf, 'y_max': y1},       # Bottom-Center
#     9: {'x_min': x2, 'x_max': np.inf, 'y_min': -np.inf, 'y_max': y1},      # Bottom-Right
# }

# # map [x0 y0] to division
# def map_to_division(x, y, divisions):
#     for division, bounds in divisions.items():
#         if bounds['x_min'] < x <= bounds['x_max'] and bounds['y_min'] < y <= bounds['y_max']:
#             return division
#     return None
# meta_cb_vpn['division'] = meta_cb_vpn.apply(lambda row: map_to_division(row['x0'], row['y0'], divisions), axis=1)

# %%
# for [x, y] from ci.result_summary
# divide the coordinate system into 1-9 divisions defined in a dict
x1 = -3.5
x2 = 5.5
y1 = 28.5
y2 = 49.5
divisions = {
    1: {'x_min': -np.inf, 'x_max': x1, 'y_min': y2, 'y_max': np.inf},   # Top-Left
    2: {'x_min': x1, 'x_max': x2, 'y_min': y2, 'y_max': np.inf},        # Top-Center
    3: {'x_min': x2, 'x_max': np.inf, 'y_min': y2, 'y_max': np.inf},   # Top-Right
    4: {'x_min': -np.inf, 'x_max': x1, 'y_min': y1, 'y_max': y2},      # Middle-Left
    5: {'x_min': x1, 'x_max': x2, 'y_min': y1, 'y_max': y2},         # Middle-Center
    6: {'x_min': x2, 'x_max': np.inf, 'y_min': y1, 'y_max': y2},      # Middle-Right
    7: {'x_min': -np.inf, 'x_max': x1, 'y_min': -np.inf, 'y_max': y1},  # Bottom-Left
    8: {'x_min': x1, 'x_max': x2, 'y_min': -np.inf, 'y_max': y1},       # Bottom-Center
    9: {'x_min': x2, 'x_max': np.inf, 'y_min': -np.inf, 'y_max': y1},      # Bottom-Right
}

# map [x0 y0] to division
def map_to_division(x, y, divisions):
    for division, bounds in divisions.items():
        if bounds['x_min'] < x <= bounds['x_max'] and bounds['y_min'] < y <= bounds['y_max']:
            return division
    return None

xy_coord['division'] = xy_coord.apply(lambda row: map_to_division(row['x'], row['y'], divisions), axis=1)

# %%
xy_coord.division.value_counts().sort_index()

# %%
# meta_cb_vpn[['division']].value_counts(dropna=False)

# %%
# # plot tbar in 3x3
# fig, ax = plt.subplots(figsize=(6, 6))
# colors = mpl.colormaps.get_cmap('tab10')  # Get a colormap with 10 distinct colors
# for division, group in meta_cb_vpn.groupby('division'):
#     ax.scatter(group['x0'], group['y0'], label=f'Division {division}', color=colors(division))
# ax.set_aspect('equal')
# ax.set_title('RF Center Locations by Division')
# ax.axvline(x=x1, color='gray', linestyle='--')
# ax.axvline(x=x2, color='gray', linestyle='--')
# ax.axhline(y=y1, color='gray', linestyle='--')
# ax.axhline(y=y2, color='gray', linestyle='--')
# ax.legend()
# # save
# # fig.savefig(Path(result_dir, 'division_3x3.png'), dpi=300, bbox_inches='tight')
# plt.show()

# %%
# plot hex and 3x3
fig = go.Figure()
fig.update_layout(
    autosize=False,
    height=260*1.75,
    width=220*2,
    margin={"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
    paper_bgcolor="rgba(255,255,255,255)",
    plot_bgcolor="rgba(255,255,255,255)",
    xaxis=dict(scaleanchor="y", scaleratio=1),
)
goscatter = go.Scatter(
    x=all_hex["x"],
    y=all_hex["y"]/np.sqrt(3), 
    mode="markers",
    marker_symbol= 15,
    marker={
        "size": 14,
        "color": "white",
        "line": {
            "width": 0.5,
            "color": "grey",
        },
    },
    showlegend=False,
)
fig.add_trace(goscatter)
# Add division lines
fig.add_vline(x=x1, line_width=1, line_color="black")
fig.add_vline(x=x2, line_width=1, line_color="black")
fig.add_hline(y=y1/np.sqrt(3), line_width=1, line_color="black")
fig.add_hline(y=y2/np.sqrt(3), line_width=1, line_color="black")
# add division texts
fig.add_annotation(x=( -5 + x1), y=(y2 + 5)/np.sqrt(3), text="1", showarrow=False, font=dict(size=16))
fig.add_annotation(x=( x2 + 5), y=(y2 + 5)/np.sqrt(3), text="3", showarrow=False, font=dict(size=16))  
fig.add_annotation(x=( -5 + x1), y=(y1 + y2)/2/np.sqrt(3), text="4", showarrow=False, font=dict(size=16))

fig.show()
# save
fig.write_image(Path(result_dir, 'division_3x3_hex.pdf'))

# %% [markdown]
# ### 3x3 coverage for rois via tbar count, using prop

# %%
plot_dir = FIG_DIR / 'quan_propagation' / 'coverage_per_roi'
plot_dir.mkdir(parents=True, exist_ok=True)

# %%
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
outidx = meta.idx[meta.bodyId.isin(meta_cb_vpn['bodyId'])] 
# outidx = meta.idx[meta.bodyId.isin(syn[syn['roi'] == 'EB']['bodyId'].unique())] 
# outidx = meta.idx[meta.cell_type_side.isin(['LC6_R'])] 
df = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )

# %%
ci.hex_heatmap(df.sum(axis=1))

# %%
ss_div = pd.DataFrame({'division': list(divisions.keys())})
ss_div.loc[ss_div['division'] == 1, ['x', 'y']] = [1, 3]
ss_div.loc[ss_div['division'] == 2, ['x', 'y']] = [2, 3]
ss_div.loc[ss_div['division'] == 3, ['x', 'y']] = [3, 3]
ss_div.loc[ss_div['division'] == 4, ['x', 'y']] = [1, 2]
ss_div.loc[ss_div['division'] == 5, ['x', 'y']] = [2, 2]
ss_div.loc[ss_div['division'] == 6, ['x', 'y']] = [3, 2]
ss_div.loc[ss_div['division'] == 7, ['x', 'y']] = [1, 1]
ss_div.loc[ss_div['division'] == 8, ['x', 'y']] = [2, 1]
ss_div.loc[ss_div['division'] == 9, ['x', 'y']] = [3, 1]

# %%
# collect tbar
# tbar_count = syn[syn['roi'] == 'EB'].groupby('bodyId').size().to_dict()
# tbar_count = syn0[syn0['bodyId'].isin(meta_cb_vpn[meta_cb_vpn.instance =='LC6_R']['bodyId'])].groupby('bodyId').size().to_dict()
tbar_count = syn[syn['bodyId'].isin(meta_cb_vpn['bodyId'])].groupby('bodyId').size().to_dict()
# weight hex coord by tbar count
df_weighted = df.copy()
weights = pd.Series([tbar_count.get(int(col), 0) for col in df_weighted.columns.values], 
                    index=df_weighted.columns)
df_weighted = df_weighted * weights

df_sum = pd.DataFrame(df_weighted.sum(axis=1), columns=['wt'])
df_sum = pd.merge(xy_coord, df_sum, left_index=True, right_index=True, how='left')

div_wt = pd.merge(ss_div, df_sum.groupby('division').agg({'wt': 'sum'}).reset_index(), on='division', how='left')

fig, ax = plt.subplots(figsize=(5, 3))
# Create a pivot table for the heatmap
heatmap_data = div_wt.pivot(index='y', columns='x', values='wt')
heatmap = ax.imshow(heatmap_data, cmap='Reds', aspect='equal', origin='lower')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.axis('off')
fig.colorbar(heatmap, ax=ax, label='eff wt x tbar count',shrink=0.75)
fig.show()

# %%
c_max

# %%
# loop
c_max = []
for roi in rois_cb:
    # collect tbar
    tbar_count = syn[syn['roi'] == roi].groupby('bodyId').size().to_dict()
    # weight hex coord by tbar count
    df_weighted = df.copy()
    weights = pd.Series([tbar_count.get(int(col), 0) for col in df_weighted.columns.values], 
                        index=df_weighted.columns)
    df_weighted = df_weighted * weights

    df_sum = pd.DataFrame(df_weighted.sum(axis=1), columns=['wt'])
    df_sum = pd.merge(xy_coord, df_sum, left_index=True, right_index=True, how='left')

    div_wt = pd.merge(ss_div, df_sum.groupby('division').agg({'wt': 'sum'}).reset_index(), on='division', how='left')

    c_max.append([roi, div_wt['wt'].max()])

    fig, ax = plt.subplots(figsize=(5, 3))
    # Create a pivot table for the heatmap
    heatmap_data = div_wt.pivot(index='y', columns='x', values='wt')
    heatmap = ax.imshow(heatmap_data, cmap='Reds', aspect='equal', origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Coverage {roi}', fontsize=14)
    fig.colorbar(heatmap, ax=ax, label='eff wt x tbar count',shrink=0.75)
    # save
    fig.savefig(plot_dir / f'coverage_tbar_{roi}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    if roi == 'AOTU(R)':
        fig, ax = plt.subplots(figsize=(5, 3))
        # Create a pivot table for the heatmap
        heatmap_data = div_wt.pivot(index='y', columns='x', values='wt')
        heatmap = ax.imshow(heatmap_data, cmap='Reds', aspect='equal', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.axis('off')
        fig.colorbar(heatmap, ax=ax, label='eff wt x tbar count',shrink=0.75)
        # save
        fig.savefig(plot_dir / f'coverage_tbar_{roi}_cbar.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

# %%
pd.DataFrame(c_max, columns=['roi', 'max_tbar']).sort_values('max_tbar', ascending=False).head(10)

# %% [markdown]
# ### collect tbars in each division and compute tbar com

# %%
ss_div = syn[syn['roi'] == roi].groupby('division').agg(
    bodyId_count = ('bodyId', 'nunique'),
    tbar_count = ('bodyId', 'count'),
    com_x = ('x', 'mean'),
    com_y = ('y', 'mean'),
    com_z = ('z', 'mean'),
).reset_index()

# %%
# fill in divisions with zero counts
all_divisions = pd.DataFrame({'division': list(divisions.keys())})
pd.merge(all_divisions, ss_div, on='division', how='left')
ss_div['bodyId_count'] = ss_div['bodyId_count'].fillna(0).astype(int)
ss_div['tbar_count'] = ss_div['tbar_count'].fillna(0).astype(int)
# ss_div['com_x'] = ss_div['com_x'].fillna(0).astype(int)
# ss_div['com_y'] = ss_div['com_y'].fillna(0).astype(int)
# ss_div['com_z'] = ss_div['com_z'].fillna(0).astype(int)

# %%
# add [x y] to ss_div, division= 1 is [1,3], 2 is [2, 3], 3 is [3, 3], 4 is [1,2], etc
ss_div['x'] = 0
ss_div['y'] = 0
ss_div.loc[ss_div['division'] == 1, ['x', 'y']] = [1, 3]
ss_div.loc[ss_div['division'] == 2, ['x', 'y']] = [2, 3]
ss_div.loc[ss_div['division'] == 3, ['x', 'y']] = [3, 3]
ss_div.loc[ss_div['division'] == 4, ['x', 'y']] = [1, 2]
ss_div.loc[ss_div['division'] == 5, ['x', 'y']] = [2, 2]
ss_div.loc[ss_div['division'] == 6, ['x', 'y']] = [3, 2]
ss_div.loc[ss_div['division'] == 7, ['x', 'y']] = [1, 1]
ss_div.loc[ss_div['division'] == 8, ['x', 'y']] = [2, 1]
ss_div.loc[ss_div['division'] == 9, ['x', 'y']] = [3, 1]

# %% [markdown]
# ### 3x3 coverage for rois via tbar count, using [x0 y0]

# %%
# 3x3 tbar coverage heatmap, test

fig, ax = plt.subplots(figsize=(3, 3))
# Create a pivot table for the heatmap
heatmap_data = ss_div.pivot(index='y', columns='x', values='tbar_count')
heatmap = ax.imshow(heatmap_data, cmap='plasma', aspect='equal', origin='lower')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
fig.colorbar(heatmap, ax=ax, label='Number of tbar',shrink=0.75)
# ax.set_aspect('equal')
plt.show()

# %%
plot_dir = FIG_DIR / 'quan_propagation' / 'coverage_per_roi'
plot_dir.mkdir(parents=True, exist_ok=True)

# %%
rois_dict = neuprint.queries.fetch_roi_hierarchy(include_subprimary=False)

rois = set(rois_dict['CNS']['CentralBrain'].keys()) 
rois = list(rois)
rois.sort()

# remove strings that doesn't have * at the end
rois = [r for r in rois if '*' in r]

# remove * at the end of the string
rois = [re.sub(r'\*$', '', roi) for roi in rois]
# remove strings that contain 'unspecified' 
rois = [roi for roi in rois if 'unspecified' not in roi]
rois_cb = rois.copy()

# %%


# %%
# loop
all_divisions = pd.DataFrame({'division': list(divisions.keys())})
for roi in rois_cb:
    ss_div = syn[syn['roi'] == roi].groupby('division').agg(
    bodyId_count = ('bodyId', 'nunique'),
    tbar_count = ('bodyId', 'count'),
    ).reset_index()
    # fill in divisions with zero counts
    ss_div = pd.merge(all_divisions, ss_div, on='division', how='left')
    ss_div['bodyId_count'] = ss_div['bodyId_count'].fillna(0).astype(int)
    ss_div['tbar_count'] = ss_div['tbar_count'].fillna(0).astype(int)

    ss_div['x'] = 0
    ss_div['y'] = 0
    ss_div.loc[ss_div['division'] == 1, ['x', 'y']] = [1, 3]
    ss_div.loc[ss_div['division'] == 2, ['x', 'y']] = [2, 3]
    ss_div.loc[ss_div['division'] == 3, ['x', 'y']] = [3, 3]
    ss_div.loc[ss_div['division'] == 4, ['x', 'y']] = [1, 2]
    ss_div.loc[ss_div['division'] == 5, ['x', 'y']] = [2, 2]
    ss_div.loc[ss_div['division'] == 6, ['x', 'y']] = [3, 2]
    ss_div.loc[ss_div['division'] == 7, ['x', 'y']] = [1, 1]
    ss_div.loc[ss_div['division'] == 8, ['x', 'y']] = [2, 1]
    ss_div.loc[ss_div['division'] == 9, ['x', 'y']] = [3, 1]

    fig, ax = plt.subplots(figsize=(3, 3))
    # Create a pivot table for the heatmap
    heatmap_data = ss_div.pivot(index='y', columns='x', values='tbar_count')
    heatmap = ax.imshow(heatmap_data, cmap='viridis', aspect='equal', origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    fig.colorbar(heatmap, ax=ax, label='Number of tbar',shrink=0.75)
    # save
    fig.savefig(plot_dir / f'coverage_tbar_{roi}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# %% [markdown]
# ### exploding brain

# %%
# xrange = [20000, 80000]
# yrange = [0, 50000]
# zrange = [10000, 21000, 42000] # central section

# based on 'NO' 
xcenter = 48500
# ycenter = 24500
ycenter = 25000
# zcenter = 24000
# zmax = 42000
zmax = 32000
center = np.array([xcenter, ycenter, 0])

# %%
# assigne each roi in rois_cb a color from the tab20 colormap
import matplotlib as mpl

# Get the tab20 colormap
cmap = mpl.colormaps.get_cmap('tab10')

# Create a dictionary to store ROI colors
roi_colors = {}

# Process each ROI
for roi_str in rois_cb:
    # Check if the ROI ends with '(R)' or '(L)'
    if roi_str.endswith('(L)') or roi_str.endswith('(R)'):
        # Get the base name without the (L) or (R) suffix
        base_name = roi_str[:-3]
        # Assign color based on the base name hash
        color_idx = hash(base_name) % 10
        roi_colors[roi_str] = cmap.colors[color_idx]
    else:
        # For ROIs without (L) or (R) suffix, use the ROI name itself
        color_idx = hash(roi_str) % 10
        roi_colors[roi_str] = cmap.colors[color_idx]

# %%
# load top rois from res plot
top_rois = pd.read_csv(result_dir / 'res_top_rois.csv')
top_rois = list(set(top_rois.values.flatten()))

top_rois_lr = []
for roi in top_rois:
    top_rois_lr.append(roi)
    if roi.endswith('(R)'):
        top_rois_lr.append(roi.replace('R)', 'L)'))
    if roi.endswith('(L)'):
        top_rois_lr.append(roi.replace('L)', 'R)'))

top_rois_lr = sorted(list(set(top_rois_lr)))

# %%
# get shifted rois
roi_ls = []
# for roi_str in rois_cb:
for roi_str in top_rois_lr:
    roi = neu.fetch_roi(roi_str)
    com = roi.vertices.mean(axis=0)
    exp_factor = (zmax - com[2]) / zmax * 5
    xy_shift = (com - center)[:2] * exp_factor
    xy_shift = np.array([xy_shift[0], xy_shift[1], 0])
    roi.vertices += xy_shift
    roi_ls.append(roi)

# %%
# plot original and shifted roi
navis.plot3d(roi_ls, alpha=0.3, color=roi_colors)

# %%
fig_mesh = navis.plot3d(
    roi_ls,
    color=roi_colors,
    alpha=0.3,
    inline=False,
    backend='plotly'
)
fig = go.Figure(fig_mesh.data)
# Add orthographic projection and remove background
fig.update_layout(
    scene=dict(
        camera=dict(
            projection=dict(type='orthographic'),
            eye=dict(x=0, y=0, z=-2),  # Look down from above
            up=dict(x=0, y=-1, z=0)     # Y-axis points up
        ),
        xaxis=dict(
            # backgroundcolor="rgba(0, 0, 0, 0)",
            # gridcolor="rgba(0, 0, 0, 0)",
            # showbackground=False,
            visible=False 
        ),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="rgba(0, 0, 0, 0)"
    ),
    paper_bgcolor="rgba(0, 0, 0, 0)",
    plot_bgcolor="rgba(0, 0, 0, 0)",
    autosize=False,
    width=400,
    height=400,
    margin={"l": 0, "r": 0, "b": 0, "t": 0}
)
fig.show()

# save
# fig.write_image(result_dir / 'top_res_rois_ortho.png', scale=2)

# %% [markdown]
# ### retinotopy index

# %%
# 3d plot of ss_div[['com_x', 'com_y', 'com_z']], label by division
fig = go.Figure()
for div in ss_div['division'].unique():
    div_data = ss_div[ss_div['division'] == div]
    fig.add_trace(go.Scatter3d(
        x=div_data['com_x'],
        y=div_data['com_y'],
        z=div_data['com_z'],
        mode='markers+text',
        marker=dict(size=5),
        text=[str(div)],
        textposition='top center',
        name=str(div)
    ))
fig.show()

# %%
# project com_x and com_y and com_z to 2D via PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
ss_div[['pca_x', 'pca_y']] = pca.fit_transform(ss_div[['com_x', 'com_y', 'com_z']])
ax = ss_div.plot.scatter(x='pca_x', y='pca_y', s=10, c='division', colormap='tab10', figsize=(3,3))

# Add labels to each point
for idx, row in ss_div.iterrows():
    ax.annotate(str(int(row['division'])), (row['pca_x'], row['pca_y']), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.show()

# %%
ri = RI_sets(ss_div[['x','y']].values, ss_div[['pca_x','pca_y']].values)

# %%
np.mean(ri)

# %% [markdown]
# ## 5x1 

# %% [markdown]
# ### division

# %%
np.nanpercentile(fit_rf.y0, [0, 20, 40, 60, 80, 100])

# %%
# divide the coordinate system into 5 divisions
y1 = 15
y2 = 20
y3 = 24
y4 = 28

divisions = {
    1: {'y_min': y4, 'y_max': np.inf},   # top
    2: {'y_min': y3, 'y_max': y4},      
    3: {'y_min': y2, 'y_max': y3},
    4: {'y_min': y1, 'y_max': y2},
    5: {'y_min': -np.inf, 'y_max': y1},      # bottom
}

# map [x0 y0] to division
def map_to_division(x, y, divisions):
    for division, bounds in divisions.items():
        if bounds['y_min'] < y <= bounds['y_max']:
            return division
    return None
meta_cb_vpn['division'] = meta_cb_vpn.apply(lambda row: map_to_division(row['x0'], row['y0'], divisions), axis=1)

# %%
meta_cb_vpn[['division']].value_counts(dropna=False)

# %%
# plot
fig, ax = plt.subplots(figsize=(3, 3))
colors = mpl.colormaps.get_cmap('tab10')  # Get a colormap with 10 distinct colors
for division, group in meta_cb_vpn.groupby('division'):
    ax.scatter(group['x0'], group['y0'], label=f'Division {division}', color=colors(division))
ax.set_xlabel('x0')
ax.set_ylabel('y0')
ax.set_aspect('equal')
ax.set_title('RF Center Locations by Division')
ax.axvline(x=x1, color='gray', linestyle='--')
ax.axvline(x=x2, color='gray', linestyle='--')
ax.axhline(y=y1, color='gray', linestyle='--')
ax.axhline(y=y2, color='gray', linestyle='--')
ax.legend()
plt.show()

# %% [markdown]
# ### collect tbars in each division and compute tbar com

# %%
syn = pd.merge(syn0, meta_cb_vpn,  on='bodyId', how='inner')
print(syn.shape)

# %%
roi = 'SLP(R)'

# %%
ss_div = syn[syn['roi'] == roi].groupby('division').agg(
    bodyId_count = ('bodyId', 'nunique'),
    tbar_count = ('bodyId', 'count'),
    com_x = ('x', 'mean'),
    com_y = ('y', 'mean'),
    com_z = ('z', 'mean'),
).reset_index()

# %%
# add [x y] to ss_div,
ss_div['x'] = 0
ss_div['y'] = 0
ss_div.loc[ss_div['division'] == 1, ['x', 'y']] = [1, 5]
ss_div.loc[ss_div['division'] == 2, ['x', 'y']] = [1, 4]
ss_div.loc[ss_div['division'] == 3, ['x', 'y']] = [1, 3]
ss_div.loc[ss_div['division'] == 4, ['x', 'y']] = [1, 2]
ss_div.loc[ss_div['division'] == 5, ['x', 'y']] = [1, 1]



# %%
# 3d plot of ss_div[['com_x', 'com_y', 'com_z']], label by division
fig = go.Figure()
for div in ss_div['division'].unique():
    div_data = ss_div[ss_div['division'] == div]
    fig.add_trace(go.Scatter3d(
        x=div_data['com_x'],
        y=div_data['com_y'],
        z=div_data['com_z'],
        mode='markers+text',
        marker=dict(size=5),
        text=[str(div)],
        textposition='top center',
        name=str(div)
    ))
fig.show()

# %%
# project com_x and com_y and com_z to 2D via PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
ss_div[['pca_x', 'pca_y']] = pca.fit_transform(ss_div[['com_x', 'com_y', 'com_z']])
ax = ss_div.plot.scatter(x='pca_x', y='pca_y', s=10, c='division', colormap='tab10', figsize=(3,3))

# Add labels to each point
for idx, row in ss_div.iterrows():
    ax.annotate(str(int(row['division'])), (row['pca_x'], row['pca_y']), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.show()

# %% [markdown]
# ### retinotopy index

# %%
ri = RI_sets(ss_div[['x','y']].values, ss_div[['pca_x','pca_y']].values)

# %%
np.mean(ri)

# %% [markdown]
# # obs. LC4

# %% [markdown]
# ### query

# %%
# LC, post-syn in ol
syn = neuprint.queries.fetch_synapses(
    NC(instance = 'LC4_R'),
    SC(type='post', rois=rois_olr, primary_only=False)
)

syn = syn[syn['roi'].str.contains(r'^LO_R_col')]

# extract the 2 integers from roi column, each number is proceeded by '_'
# assign them to 2 new columns 'hex1' and 'hex2'
syn['hex1'] = syn['roi'].str.extract(r'_(\d+)_')[0].astype(int)
syn['hex2'] = syn['roi'].str.extract(r'_(\d+)$')[0].astype(int)
# offset
syn['hex1'] = syn['hex1'] - hex_offset[0]
syn['hex2'] = syn['hex2'] - hex_offset[1]

# group by bodyId, compute the mean and variance of [x y z], and mean of hex1 and hex2
syn = syn.groupby('bodyId').agg(
    {'x': ['mean', 'std'], 'y': ['mean', 'std'], 'z': ['mean', 'std'], 'hex1': ['mean'], 'hex2': ['mean']}
).reset_index()
# flatten MultiIndex columns
syn.columns = ['bodyId'] + [f"{col[0]}_{col[1]}" for col in syn.columns[1:]]

# h and v
syn['v'] = syn['hex1_mean'] + syn['hex2_mean']
syn['h'] = syn['hex1_mean'] - syn['hex2_mean']
syn['h'] = -syn['h']

syn_post_mean = syn.copy()

# %%
# pre-syn in cb
syn = neuprint.queries.fetch_synapses(
    NC(instance = 'LC4_R'),
    SC(type='pre', rois=rois_cb, primary_only=True)
)
syn_pre = syn.copy()

# group by bodyId, compute the mean and variance of [x y z]
syn = syn.groupby('bodyId').agg(
    {'x': ['mean', 'std'], 'y': ['mean', 'std'], 'z': ['mean', 'std']}
).reset_index()
# flatten MultiIndex columns
syn.columns = ['bodyId'] + [f"{col[0]}_{col[1]}" for col in syn.columns[1:]]
syn_pre_mean = syn.copy()



# %%
# skeleton
n_info, _ = fetch_neurons(NC(instance='LC4_R'))
ske = neu.fetch_skeletons(n_info['bodyId'].values)

# %% [markdown]
# ### check DNp02 and DNp11

# %%
# note, DNp11 has a branch in the proximal part of the LO
neuron_df, connection_df = neuprint.queries.fetch_adjacencies(
    sources=None,
    targets=NC(instance = ['DNp11_R']),
    rois=rois_olr,
    min_total_weight=1
)
df = pd.merge(connection_df, neuron_df, left_on='bodyId_pre', right_on='bodyId', how='left')
df.sort_values(by='weight', ascending=False)



# %%
n_info, _ = fetch_neurons(NC(instance="MBON01(y5B'2a)_R"))
n_info, _ = safe_fetch_neurons_by_instance("MBON01(y5B'2a)_R")
# n_info, _ = fetch_neurons(NC(instance= aa.values))

n_info


n_info# n_info, _ = fetch_neurons(NC(instance= aa.values))n_info, _ = fetch_neurons(NC(instance=instance_name))
neuron_df, rois_df = neuprint.fetch_adjacencies(
    sources=NC(instance = ['LC4_R']),
    targets= NC(instance = ['DNp02_R', 'DNp11_R']),
    min_total_weight=1
)

# group by bodyId_pre and bodyId_post, sum up weight
el = rois_df.\
    groupby(['bodyId_pre', 'bodyId_post']).\
    agg({'weight': 'sum'}).\
    reset_index()

# DNp11_R
el_dnp11r = el[el['bodyId_post'] ==  10106]
el_dnp02r = el[el['bodyId_post'] ==  10117]


# combine syn xyz with el
df = pd.merge(syn_post_mean[['bodyId', 'x_mean', 'y_mean', 'z_mean', 'h']], 
            #   el_dnp11r[['bodyId_pre','weight']],
            el_dnp02r[['bodyId_pre','weight']],
            left_on='bodyId', right_on='bodyId_pre', how='left')

# %% [markdown]
# # obs. AOTU roi

# %%
n_me, c_me = fetch_neurons(NC(inputRois= ['ME(R)'], min_roi_inputs = 3, outputRois=['AOTU(R)'], min_roi_outputs = 3))
n_lo, c_lo = fetch_neurons(NC(inputRois= ['LO(R)'], min_roi_inputs = 3, outputRois=['AOTU(R)'], min_roi_outputs = 3))
n_lop, c_lop = fetch_neurons(NC(inputRois= ['LOP(R)'], min_roi_inputs = 3, outputRois=['AOTU(R)'], min_roi_outputs = 3))
print(n_me.shape, n_lo.shape, n_lop.shape)

# %%
c_me = c_me[c_me['roi'].str.contains(r'AOTU\(R')]
# add instance
c_me = c_me.merge(n_me[['bodyId', 'instance']], on='bodyId', how='left')
c_me.instance.value_counts()

# %%
c_lop = c_lop[c_lop['roi'].str.contains(r'AOTU\(R')]
# add instance
c_lop = c_lop.merge(n_lop[['bodyId', 'instance']], on='bodyId', how='left')
c_lop.instance.value_counts()

# %%
# # check if the roi contains AOTU or LO(R) or LO(L)
# c_lo = c_lo[c_lo['roi'].str.contains(r'AOTU|LO\(')]

c_lo = c_lo[c_lo['roi'].str.contains(r'AOTU\(R')]
# add instance
c_lo = c_lo.merge(n_lo[['bodyId', 'instance']], on='bodyId', how='left')

c_lo.instance.value_counts()

# %% [markdown]
# ## all AOTU cells

# %%
# all aotu neurons
n0, _ = fetch_neurons(NC(instance = '^AOTU.*R$'))
n0.instance.nunique()

# %%
# sorted(n0['instance'].unique())

# %% [markdown]
# ## cells downstream of LC10s in AOTU

# %%
# integration of LC10 neurons by AOTUs 
neuron_df, conn_df = neuprint.fetch_adjacencies(
    # sources=['^LC10.*R$'],
    sources=['^LC10.*R$'],
    targets=None,
    rois=['AOTU(R)'],  #in AOTU
    min_total_weight=3
)

conn_df = neuprint.merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])

# matout_lc10 = neuprint.connection_table_to_matrix(conn_df, 'instance', sort_by='instance')

# %%
inst_post = conn_df.groupby(['bodyId_post', 'instance_post']).agg({'weight': 'sum'}).reset_index()['instance_post'].value_counts(dropna=False)
# remove LC10s
inst_post = inst_post[~inst_post.index.str.contains('LC10')]

# %% [markdown]
# ### are all cells of these instances included here? 

# %%
n_info, _ = fetch_neurons(NC(instance=inst_post.index))
n_info = n_info.groupby(['instance']).agg(
    count=("instance", "count"),
).reset_index().sort_values(by='count', ascending=False)

# %%
inst_post = pd.merge(inst_post, n_info, left_index=True, right_on='instance', how='left')

# %%
# inst_post[inst_post['count_x'] != inst_post['count_y']]

# %% [markdown]
# ## visual coord via LC10s

# %%
n_info, _ = fetch_neurons(NC(instance='^LC10.*R$'))
# remove unclear instances
n_info = n_info.loc[~n_info['instance'].str.contains('unclear'), ['bodyId', 'instance', 'type', 'downstream', 'upstream']]
n_info.instance.value_counts(dropna=False)

# %%
# LC10 post-syn in ol
syn = neuprint.queries.fetch_synapses(
    NC(bodyId= n_info['bodyId'].values),
    SC(type='post', rois='LO(R)', primary_only=False)
)

syn = syn[syn['roi'].str.contains(r'^LO_R_col')]

# extract the 2 integers from roi column, each number is proceeded by '_'
# assign them to 2 new columns 'hex1' and 'hex2'
syn['hex1'] = syn['roi'].str.extract(r'_(\d+)_')[0].astype(int)
syn['hex2'] = syn['roi'].str.extract(r'_(\d+)$')[0].astype(int)
# offset
syn['hex1'] = syn['hex1'] - hex_offset[0]
syn['hex2'] = syn['hex2'] - hex_offset[1]

syn_post = syn.copy()

# group by bodyId, compute the mean and variance of [x y z], and mean of hex1 and hex2
syn = syn.groupby('bodyId').agg(
    {'x': ['mean', 'std'], 'y': ['mean', 'std'], 'z': ['mean', 'std'], 'hex1': ['mean'], 'hex2': ['mean']}
).reset_index()
# flatten MultiIndex columns
syn.columns = ['bodyId'] + [f"{col[0]}_{col[1]}" for col in syn.columns[1:]]

# h and v
syn['v'] = syn['hex1_mean'] + syn['hex2_mean']
syn['h'] = syn['hex1_mean'] - syn['hex2_mean']
syn['h'] = -syn['h']

syn_post_mean = syn.copy()

syn.shape

# %%
lc10_coord = pd.merge(n_info, syn_post_mean[['bodyId', 'hex1_mean',	'hex2_mean', 'h', 'v']], how='left', on='bodyId')

# %% [markdown]
# ### pick a type, compute input-com in LO

# %%
inst = ["AOTU008_R"]
# inst = ["AOTU050_R"]
# inst = ["AOTU059_R"]
# inst = ["AOTU038_R"]

neuron_df, conn_df = neuprint.fetch_adjacencies(
    sources=['^LC10.*R$'],
    targets= inst,
    # rois=['AOTU(R)'],  #in AOTU
    min_total_weight=3
)

conn_df = neuprint.merge_neuron_properties(neuron_df, conn_df, ['type', 'instance'])

# bodyId and [p q h v]
id_pqhv = pd.DataFrame({'bodyId': conn_df.bodyId_post.unique(), 'p': np.nan, 'q': np.nan, 'h': np.nan, 'v': np.nan})

# %%
# for each bodyId in id_pqhv, find the [p q] weighted by conn wt
for i, row in id_pqhv.iterrows():
    bodyId = row['bodyId']
    conn = conn_df[conn_df['bodyId_post'] == bodyId]
    conn = conn.merge(lc10_coord[['bodyId', 'hex1_mean', 'hex2_mean', 'h', 'v']], left_on='bodyId_pre', right_on='bodyId', how='left')
    id_pqhv.at[i, 'p'] = (conn['hex2_mean'] * conn['weight']).sum() / conn['weight'].sum()
    id_pqhv.at[i, 'q'] = (conn['hex1_mean'] * conn['weight']).sum() / conn['weight'].sum()
    id_pqhv.at[i, 'h'] = (conn['h'] * conn['weight']).sum() / conn['weight'].sum()
    id_pqhv.at[i, 'v'] = (conn['v'] * conn['weight']).sum() / conn['weight'].sum()

# %%
# pre-syn in cb
syn = neuprint.queries.fetch_synapses(
    NC(instance = inst),
    SC(type='post', rois= 'AOTU(R)', primary_only=True)
)
syn_pre = syn.copy()

# group by bodyId, compute the mean and variance of [x y z]
syn = syn.groupby('bodyId').agg(
    {'x': ['mean', 'std'], 'y': ['mean', 'std'], 'z': ['mean', 'std']}
).reset_index()
# flatten MultiIndex columns
syn.columns = ['bodyId'] + [f"{col[0]}_{col[1]}" for col in syn.columns[1:]]
syn_pre_mean = syn.copy()

syn.shape

# %% [markdown]
# ### PLOT

# %%
roi_mesh = []
roi_name = ['AOTU(R)'] #, 'LO(R)',  'PVLP(R)']
for roi in roi_name:
    roi_mesh.append(neu.fetch_roi(roi))

# %%
# make color map based on mapping to CIELAB
#  to disk
for i, row in id_pqhv.iterrows():
    pt = (row['h'], row['v'])
    r = np.sqrt(pt[0]**2 + pt[1]**2)
    mul = 35 / r # 35~max boundary distance
    pt = (pt[0] * mul, pt[1] * mul)
    # angle
    angle = np.arctan2(row['v'], row['h'])
    # radius
    intersection = ray_intersection_with_boundary(pt, boundary_alpha)
    radius = r / np.sqrt(intersection[0]**2 + intersection[1]**2)
    # save r and angle
    id_pqhv.at[i, 'r'] = radius
    id_pqhv.at[i, 'theta'] = angle

# disk to lab
# Use apprehension for vectorized mapping from disk to LAB
r = id_pqhv['r'].values
theta = id_pqhv['theta'].values
L = np.full_like(r, 75)
lab = np.array([disk_to_lab(ri, ti, li) for ri, ti, li in zip(r, theta, L)])
id_pqhv[['L', 'a', 'b']] = lab

# lab to rgb
# Use apprehension for vectorized mapping from LAB to RGB
lab = id_pqhv[['L', 'a', 'b']].values
id_pqhv_rgb = np.array([lab_to_rgb(Li, ai, bi) for Li, ai, bi in zip(lab[:, 0], lab[:, 1], lab[:, 2])])

# Convert id_pqhv_rgb to hex color strings for Plotly
id_pqhv_hex = [mpl.colors.to_hex(rgb) for rgb in id_pqhv_rgb]

# %%
# import mplcursors
import mplcursors
# plot [['h','v']], color by rgb
fig, ax = plt.subplots(figsize=(4, 4))
x, y = boundary_alpha.exterior.xy
ax.plot(x, y, color='black', alpha=0.5)
# ax.scatter(id_pqhv['h'], id_pqhv['v'], marker='+', s=10, c=id_pqhv_rgb)
# add cursor on hover, show bodyId
# Use mplcursors for interactive hover annotations
scatter = ax.scatter(id_pqhv['h'], id_pqhv['v'], marker='+', s=10, c=id_pqhv_rgb)
mplcursors.cursor(scatter, hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(f"bodyId: {int(id_pqhv.iloc[sel.index]['bodyId'])}")
)
ax.set_aspect('equal')
plt.show()

# save plot
fig.savefig(result_dir / 'vpn_retinotopy' / f"LO_{inst}.png", dpi=300, bbox_inches='tight')

# %% [markdown]
# ### axon-com colored by dendrite position

# %%
fig_mesh = navis.plot3d(
    roi_mesh
    , color = 'grey'    , alpha=0.1    , inline=False    , backend='plotly')

# fig_ske = navis.plot3d(sk_axon[order], palette='viridis', connectors=False, soma=True, linewidth=1, inline=False)

fig_pt = go.Scatter3d(
    x=syn_pre_mean['x_mean'],    y=syn_pre_mean['y_mean'],    z=syn_pre_mean['z_mean'],
    mode='markers',
    marker=dict(
        size=3,
        # color=pts['v'],
        color=id_pqhv_hex,
        # colorscale='Viridis',
        # colorbar=dict(title='v'),
        opacity=1
    ),
    name='Points'
)

fig = go.Figure(data= (fig_pt,) + fig_mesh.data)
# fig = go.Figure(data= fig_ske.data + (fig_pt,) + fig_pt2.data + fig_mesh.data)
fig.update_layout(autosize=False, width=400, height=400)
fig.update_layout(margin={"l":0, "r":0, "b":0, "t":0})
fig.show()

# %%
# # save figure
# fig.write_html(result_dir / 'vpn_retinotopy' / f"synLAB_{inst}.html")

# %% [markdown]
# ## else

# %%
inst = 'AOTU041_R'

n0, _ = fetch_neurons(NC(instance=inst))
n1, c1 = fetch_adjacencies(
    sources=None,
    targets=NC(instance=inst),
    min_total_weight=1
)

# %%
c2 = neuprint.merge_neuron_properties(n1, c1, ['type', 'instance'])\
    .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
    .agg({'weight': 'sum'})\
    .reset_index()

# %%
c2[c2['bodyId_post'] == 10031].sort_values(by='weight', ascending=False).head(10)

# %%
# sk and mesh
sk_cns = neu.fetch_skeletons(n0['bodyId'].values)
mn_cns = neu.fetch_mesh_neuron(n0['bodyId'].values)

# %%

fig_mesh = navis.plot3d(
    roi_mesh
    , color = 'grey'
    , alpha=0.1
    , inline=False
    , backend='plotly')

fig_n_1 = navis.plot3d(
    sk_cns[:1]
    , color = 'blue'
    , alpha=1
    , inline=False
    , backend='plotly')

fig_n_2 = navis.plot3d(
    mn_flywire_cns[1:2]
    , color = 'red'
    , alpha=1
    , inline=False
    , backend='plotly')

fig = go.Figure(data= fig_n_1.data + fig_n_2.data)

# align the camera 
fig.update_layout(scene_camera=dict(
    eye=dict(x=0, y=0, z=-2),
    up=dict(x=0, y=-1, z=0)
))
fig.update_layout(autosize=False, width=900, height=600)
fig.update_layout(margin={"l":0, "r":0, "b":0, "t":0})

# title
# fig.update_layout(title_text="up to 5-hop spread", title_x=0.5, title_y=0.95)

fig.show()

# %%
# flybrains.download_jrc_transforms()

# %% [markdown]
# # obs. AOTU

# %% [markdown]
# ### query cns

# %%
inst = 'AOTU041_R'

n0, _ = fetch_neurons(NC(instance=inst))
n1, c1 = fetch_adjacencies(
    sources=None,
    targets=NC(instance=inst),
    min_total_weight=1
)

# %%
c2 = neuprint.merge_neuron_properties(n1, c1, ['type', 'instance'])\
    .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
    .agg({'weight': 'sum'})\
    .reset_index()

# %%
c2[c2['bodyId_post'] == 10031].sort_values(by='weight', ascending=False).head(10)

# %%
# sk and mesh
sk_cns = neu.fetch_skeletons(n0['bodyId'].values)
mn_cns = neu.fetch_mesh_neuron(n0['bodyId'].values)

# %% [markdown]
# ### query flywire

# %%
# load flywire_783
# https://codex.flywire.ai/api/download
classification = pd.read_csv(DATA_DIR / 'flywire_783' / 'classification.csv')
el_idtype = pd.read_csv(DATA_DIR / 'flywire_783' / 'connections_id_type.csv')

# %%
classification[classification['hemibrain_type'] == 'AOTU041']

# %%
rootids = [720575940637464718, 
           720575940638668659,
           720575940622143245,
           720575940629532777]
sk_flywire = flywire.get_skeletons(rootids)
mn_flywire = flywire.get_mesh_neuron(rootids)

# %%
# # adj
# adj = flywire.get_adjacency(sources= rootids, targets= None, square=False)

# conn
conn = flywire.get_connectivity(x=rootids, upstream=False, downstream=True)

# %%
df1 = conn[conn['pre'] == 720575940637464718].sort_values(by='weight', ascending=False)
df1 = pd.merge(df1, classification[['root_id', 'cell_type', 'hemibrain_type']], left_on='post', right_on='root_id', how='left')

df2 = conn[conn['pre'] == 720575940629532777].sort_values(by='weight', ascending=False)
df2 = pd.merge(df2, classification[['root_id', 'cell_type', 'hemibrain_type']], left_on='post', right_on='root_id', how='left')

# %%
df1.weight.sum(), df2.weight.sum()

# %%
pd.concat([df1.head(10), df2.head(10)], axis=1)

# %%
df1 = conn[conn['pre'] == 720575940638668659].sort_values(by='weight', ascending=False)
df1 = pd.merge(df1, classification[['root_id', 'cell_type', 'hemibrain_type']], left_on='post', right_on='root_id', how='left')

df2 = conn[conn['pre'] == 720575940622143245].sort_values(by='weight', ascending=False)
df2 = pd.merge(df2, classification[['root_id', 'cell_type', 'hemibrain_type']], left_on='post', right_on='root_id', how='left')

# %%
pd.concat([df1.head(10), df2.head(10)], axis=1)

# %% [markdown]
# ### transform and plot

# %%
n0['bodyId'].values

# %%
#  https://github.com/navis-org/navis-flybrains

# sk_flywire_cns = navis.xform_brain(sk_flywire, source='FLYWIRE', target = 'JRCFIB2022M')
mn_flywire_cns = navis.xform_brain(mn_flywire, source='FLYWIRE', target = 'JRCFIB2022Mraw')

# %%
mn_flywire_cns

# %%
roi_mesh = []
roi_name = ['AOTU(R)', 'LO(R)'] #, 'PVLP(R)']

for roi in roi_name:
    roi_mesh.append(neu.fetch_roi(roi))

# %%

fig_mesh = navis.plot3d(
    roi_mesh
    , color = 'grey'
    , alpha=0.1
    , inline=False
    , backend='plotly')

fig_n_1 = navis.plot3d(
    sk_cns[:1]
    , color = 'blue'
    , alpha=1
    , inline=False
    , backend='plotly')

fig_n_2 = navis.plot3d(
    mn_flywire_cns[1:2]
    , color = 'red'
    , alpha=1
    , inline=False
    , backend='plotly')

fig = go.Figure(data= fig_n_1.data + fig_n_2.data)

# align the camera 
fig.update_layout(scene_camera=dict(
    eye=dict(x=0, y=0, z=-2),
    up=dict(x=0, y=-1, z=0)
))
fig.update_layout(autosize=False, width=900, height=600)
fig.update_layout(margin={"l":0, "r":0, "b":0, "t":0})

# title
# fig.update_layout(title_text="up to 5-hop spread", title_x=0.5, title_y=0.95)

fig.show()

# %%
# flybrains.download_jrc_transforms()

# %% [markdown]
# # obs. VPN, mean pre vs post syn

# %% [markdown]
# ### query

# %%
# oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LC10')]
# oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LC')].sort_values(by='no. of cells', ascending=False)
# oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LPC')].sort_values(by='no. of cells', ascending=False)
# oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^MeVP')].sort_values(by='no. of cells', ascending=False)
# oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^MeTu')].sort_values(by='no. of cells', ascending=False)
# oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LPLC')].sort_values(by='no. of cells', ascending=False)
oltypes_vpn[oltypes_vpn['cell type'].str.contains(r'^LLPC')].sort_values(by='no. of cells', ascending=False)

# %%
inst = 'LLPC3_R'

# LC10b_R bodyId= 70262,70519 is incomplete

# %%
n_info, _ = fetch_neurons(NC(instance=inst))

# # LC10b_R only
# # remove row where bodyId is 70262 or 70519
# n_info = n_info[~n_info['bodyId'].isin([70262, 70519])]

# skeleton
sk = neu.fetch_skeletons(n_info['bodyId'].values)

n_info.shape

# %%
# LC, post-syn in ol
syn = neuprint.queries.fetch_synapses(
    # NC(instance = inst),
    NC(bodyId= n_info['bodyId'].values),
    SC(type='post', rois=rois_olr, primary_only=False)
)

syn = syn[syn['roi'].str.contains(r'^LO_R_col')]
# syn = syn[syn['roi'].str.contains(r'^LOP_R_col')]
# syn = syn[syn['roi'].str.contains(r'^ME_R_col')]

# extract the 2 integers from roi column, each number is proceeded by '_'
# assign them to 2 new columns 'hex1' and 'hex2'
syn['hex1'] = syn['roi'].str.extract(r'_(\d+)_')[0].astype(int)
syn['hex2'] = syn['roi'].str.extract(r'_(\d+)$')[0].astype(int)
# offset
syn['hex1'] = syn['hex1'] - hex_offset[0]
syn['hex2'] = syn['hex2'] - hex_offset[1]

syn_post = syn.copy()

# group by bodyId, compute the mean and variance of [x y z], and mean of hex1 and hex2
syn = syn.groupby('bodyId').agg(
    {'x': ['mean', 'std'], 'y': ['mean', 'std'], 'z': ['mean', 'std'], 'hex1': ['mean'], 'hex2': ['mean']}
).reset_index()
# flatten MultiIndex columns
syn.columns = ['bodyId'] + [f"{col[0]}_{col[1]}" for col in syn.columns[1:]]

# h and v
syn['v'] = syn['hex1_mean'] + syn['hex2_mean']
syn['h'] = syn['hex1_mean'] - syn['hex2_mean']
syn['h'] = -syn['h']

syn_post_mean = syn.copy()

syn.shape

# %%
# pre-syn in cb
syn = neuprint.queries.fetch_synapses(
    # NC(instance = inst),
    NC(bodyId= n_info['bodyId'].values),
    SC(type='pre', rois=rois_cb, primary_only=True)
)
syn_pre = syn.copy()

# group by bodyId, compute the mean and variance of [x y z]
syn = syn.groupby('bodyId').agg(
    {'x': ['mean', 'std'], 'y': ['mean', 'std'], 'z': ['mean', 'std']}
).reset_index()
# flatten MultiIndex columns
syn.columns = ['bodyId'] + [f"{col[0]}_{col[1]}" for col in syn.columns[1:]]
syn_pre_mean = syn.copy()

syn.shape

# %%
# # only for LPC2_R
# # syn_post_mean[syn_post_mean.bodyId.isin(syn_pre_mean.bodyId)]
# syn_post_mean = syn_post_mean[syn_post_mean.bodyId.isin(syn_pre_mean.bodyId)]
# sk = sk[np.isin(sk._id, syn_post_mean.bodyId.values)]

# %%
# # only for MeVP6
# syn_post_mean = syn_post_mean[syn_post_mean.bodyId.isin(syn_pre_mean.bodyId)]
# sk = sk[np.isin(sk._id, syn_post_mean.bodyId.values)]

# %% [markdown]
# # obs. PLOT

# %% [markdown]
# ## plot dendrite positions

# %%
# # plot the edge ids
# fig, ax = plt.subplots(figsize=(2, 2))
# ax.plot(edge_ids['h'], edge_ids['v'], 'o', markersize=1)
# ax.plot(syn_post_mean['h'], syn_post_mean['v'], '+', markersize=1)
# ax.set_aspect('equal')
# plt.show()

# %%
# make color map based on mapping to CIELAB
#  to disk
for i, row in syn_post_mean.iterrows():
    pt = (row['h'], row['v'])
    r = np.sqrt(pt[0]**2 + pt[1]**2)
    mul = 35 / r # 35~max boundary distance
    pt = (pt[0] * mul, pt[1] * mul)
    # angle
    angle = np.arctan2(row['v'], row['h'])
    # radius
    intersection = ray_intersection_with_boundary(pt, boundary_alpha)
    radius = r / np.sqrt(intersection[0]**2 + intersection[1]**2)
    # save r and angle
    syn_post_mean.at[i, 'r'] = radius
    syn_post_mean.at[i, 'theta'] = angle

# disk to lab
# Use apprehension for vectorized mapping from disk to LAB
r = syn_post_mean['r'].values
theta = syn_post_mean['theta'].values
L = np.full_like(r, 75)
lab = np.array([disk_to_lab(ri, ti, li) for ri, ti, li in zip(r, theta, L)])
syn_post_mean[['L', 'a', 'b']] = lab

# lab to rgb
# Use apprehension for vectorized mapping from LAB to RGB
lab = syn_post_mean[['L', 'a', 'b']].values
syn_post_rgb = np.array([lab_to_rgb(Li, ai, bi) for Li, ai, bi in zip(lab[:, 0], lab[:, 1], lab[:, 2])])

# Convert syn_post_rgb to hex color strings for Plotly
syn_post_hex = [mpl.colors.to_hex(rgb) for rgb in syn_post_rgb]

# %%
# plot syn_post_mean[['h','v']], color by rgb
fig, ax = plt.subplots(figsize=(4, 4))
x, y = boundary_alpha.exterior.xy
ax.plot(x, y, color='black', alpha=0.5)
# ax.plot(edge_ids['h'], edge_ids['v'], 'o', markersize=1)
ax.scatter(syn_post_mean['h'], syn_post_mean['v'], marker='+', s=10, c=syn_post_rgb)
ax.set_aspect('equal')
plt.show()

# save plot
fig.savefig(result_dir / 'vpn_retinotopy' / f"syn_post_{inst}.png", dpi=300, bbox_inches='tight')

# %%
roi_mesh = []
roi_name = ['LO(R)', 'AOTU(R)'] #, 'PVLP(R)']
for roi in roi_name:
    roi_mesh.append(neu.fetch_roi(roi))

# %%
# fig_mesh = navis.plot3d(
#     roi_mesh
#     # , color=cm(df['sum'])
#     # , color = list(mpl.colors.TABLEAU_COLORS)
#     # , palette= 'Set1'
#     , color = 'grey'
#     , alpha=0.1    , inline=False    , backend='plotly')

# # # Remove rows with NaN in 'weight' before plotting
# # df_plot = df.dropna(subset=['weight'])

# # fig_pt = px.scatter_3d(
# #     df_plot,
# #     x='x_mean', y='y_mean', z='z_mean',
# #     color='weight',  # color points by weight
# #     color_continuous_scale='Viridis'
# # )
# # fig_pt.update_traces(marker=dict(size=5))

# fig_pt2 = px.scatter_3d(
#     syn_post_mean,
#     x='x_mean',    y='y_mean',     z='z_mean',
# )
# fig_pt2.update_traces(marker=dict(size=3, color=syn_post_rgb))  # Change point 

# fig = go.Figure(data= fig_mesh.data + fig_pt2.data)
# # align the camera 
# fig.update_layout(scene_camera=dict(
#     eye=dict(x=0, y=0, z=-2),
#     up=dict(x=0, y=-1, z=0)
# ))
# fig.update_layout(autosize=False, width=900, height=600)
# fig.update_layout(margin={"l":0, "r":0, "b":0, "t":0})
# # title
# # fig.update_layout(title_text="up to 5-hop spread", title_x=0.5, title_y=0.95)
# fig.show()

# %% [markdown]
# ## axon order at midpoint

# %%
# # get central brain parts of the skeleton --> has some problems, e.g. LC10b [0, 8]
# sk_axon = navis.in_volume(sk, cb_mesh, inplace=False)

# %%
# # main cable --> root path can be longest
# sk_main = []
# for i, neuron in enumerate(sk):
#     sk_main.append(navis.longest_neurite(neuron, n=1, reroot_soma=True, from_root=False))

# # sk_main = navis.longest_neurite(sk, n=1, from_root=False)

# %%
# perform PCA on the points in syn_post_mean and syn_pre_mean, find the first principal axis and center
from sklearn.decomposition import PCA

# Combine all points
pts = np.vstack([syn_post_mean[['x_mean', 'y_mean', 'z_mean']].values,
                 syn_pre_mean[['x_mean', 'y_mean', 'z_mean']].values])

# Perform PCA
pca = PCA(n_components=3)
pca.fit(pts)

# Get the first principal component (direction vector)
pc1 = pca.components_[0]

# Get the center (mean) of all points
com = np.mean(pts, axis=0)

# %%
# find cutting plane

from utils.geometry import plane_square

fig_ske = navis.plot3d(sk, inline=False, color = syn_post_hex,  backend='plotly')

fig_mesh = navis.plot3d(
    cb_mesh
    , color = 'grey'    , alpha=0.1    , inline=False    , backend='plotly'
    )

vn = np.array([-1, 0.3, 0.6]) # normal vector
vt = np.array([2.6e4, 2.3e4, 2.3e4]) # translation vector
mf = 3e4 # multiplication factor for the plane size
pl_rot = plane_square(vn, vt, mf)
data1 = {
    'type': 'mesh3d',
    'x': pl_rot['x'],    'y': pl_rot['y'],    'z': pl_rot['z'],
    'delaunayaxis':'x',
    'color': 'gray',
    'opacity': 1,
}

vn = np.array([-1, 0.3, 0.6]) # normal vector
# vt = np.array([2.4e4, 2.3e4, 2.5e4]) # translation vector
vt = np.array([2.45e4, 2.3e4, 2.45e4]) # translation vector
# vt = np.array([3.1e4, 3e4, 3.5e4]) # translation vector
mf = 3e4 # multiplication factor for the plane size
pl_rot = plane_square(vn, vt, mf)
data2 = {
    'type': 'mesh3d',
    'x': pl_rot['x'],    'y': pl_rot['y'],    'z': pl_rot['z'],
    'delaunayaxis':'x',
    'color': 'pink',
    'opacity': 1,
}

fig = go.Figure(data= [data1, data2] + list(fig_mesh.data + fig_ske.data))
# fig = go.Figure(data= list(fig_mesh.data + fig_ske.data))
# fig = go.Figure(data= fig_ske.data + (fig_pt,) + fig_pt2.data + fig_mesh.data)
fig.update_layout(autosize=False, width=900, height=700)
fig.update_layout(margin={"l":0, "r":0, "b":0, "t":0})
fig.show()

# %%
# # todo, 'distal' is not well defined here

# # cut axon using the pc1 plane
# vn = np.array([-1, 0.3, 0.6]) # normal vector
# vn = vn / np.linalg.norm(vn)  # normalize the normal vector
# vt = np.array([2.6e4, 2.3e4, 2.3e4]) # translation vector
# sk_axon = []
# for i, neuron in enumerate(sk):
#     coords = neuron.nodes[['x', 'y', 'z']].values
#     # Calculate the distance from each point to the plane
#     dd = np.abs(np.dot(coords - vt, vn))
#     # Get the index of the point that is closest to the plane
#     closest_idx = np.argmin(dd)
#     # cut
#     subtree = navis.cut_skeleton(neuron, closest_idx, ret='distal')
#     sk_axon.append(subtree)

# %%
# # Find the mask for nodes where dot product > 0
# mask = np.dot(sk[0].nodes[['x', 'y', 'z']].values - com, pc1) > 0
# node_ids = sk[0].nodes.index[mask]

# # Subset the neuron to only those nodes
# sk_axon[0] = navis.subset_neuron(sk[0], node_ids=node_ids)

# %%
# find points closest to the cutting plane

# Loop through all neurons in sk and find the point closest to the plane for each
# Initialize an empty DataFrame for results
pts_cs = pd.DataFrame(columns=['bodyId', 'x', 'y', 'z'])

# cutting plane parameters
vn = np.array([-1, 0.3, 0.6]) # normal vector
vn = vn / np.linalg.norm(vn)  # normalize the normal vector

# vt = np.array([2.6e4, 2.3e4, 2.3e4]) # translation vector
vt = np.array([2.45e4, 2.3e4, 2.45e4]) # translation vector, LC18, LC21, LC11, LPC1, LLPC1, LLPC2, LLPC3
# vt = np.array([3.1e4, 3e4, 3.5e4]) # translation vector, LPC2

for i, neuron in enumerate(sk):
    coords = neuron.nodes[['x', 'y', 'z']].values
    # Calculate the distance from each point to the plane
    dd = np.abs(np.dot(coords - vt, vn))
    # Get the index of the point that is closest to the plane
    closest_idx = np.argmin(dd)
    # Get the closest point
    closest_point = coords[closest_idx]
    # Append as a new row to the DataFrame
    pts_cs.loc[len(pts_cs)] = {
        'bodyId': neuron.id,
        'x': closest_point[0],
        'y': closest_point[1],
        'z': closest_point[2]
    }

# %% [markdown]
# ## axon colored by dendrite position

# %%
# color by syn_post_mean positions
# all
# pts = pd.merge(syn_pre, syn_post_mean[['bodyId','h','v']], on='bodyId', how='left')
# # mean
# pts = pd.merge(syn_pre_mean, syn_post_mean[['bodyId','h','v']], on='bodyId', how='left')

# # sort syn_post_mean['h'] and find the order
# order = syn_post_mean['h'].sort_values(ascending=False).index.values

# %%
fig_mesh = navis.plot3d(
    roi_mesh
    , color = 'grey'    , alpha=0.1    , inline=False    , backend='plotly')

# fig_ske = navis.plot3d(sk_axon[order], palette='viridis', connectors=False, soma=True, linewidth=1, inline=False)

fig_pt = go.Scatter3d(
    x=syn_pre_mean['x_mean'],    y=syn_pre_mean['y_mean'],    z=syn_pre_mean['z_mean'],
    mode='markers',
    marker=dict(
        size=3,
        # color=pts['v'],
        color=syn_post_hex,
        # colorscale='Viridis',
        # colorbar=dict(title='v'),
        opacity=1
    ),
    name='Points'
)

fig_pt2 = px.scatter_3d(
    syn_post_mean,    x='x_mean',    y='y_mean',     z='z_mean',
)
fig_pt2.update_traces(marker=dict(size=3, color=syn_post_rgb))  # Change point 

fig_pt3 = px.scatter_3d(
    pts_cs,    x='x',    y='y',     z='z',
)
fig_pt3.update_traces(marker=dict(size=3, color=syn_post_rgb))  # Change point 

fig = go.Figure(data= (fig_pt,) + fig_pt2.data + fig_pt3.data )
# fig = go.Figure(data= fig_ske.data + (fig_pt,) + fig_pt2.data + fig_mesh.data)
fig.update_layout(autosize=False, width=800, height=600)
fig.update_layout(margin={"l":0, "r":0, "b":0, "t":0})
fig.show()

# %%
# save figure
fig.write_html(result_dir / 'vpn_retinotopy' / f"synLAB_{inst}.html")

# %%
# navis.plot3d(ske[order[::-1]], palette='viridis', connectors=False, soma=True, linewidth=1)

# %%
# reverse order
# order = order[::-1]

# %% [markdown]
# # Retinotopy index

# %% [markdown]
# ## some func

# %%
# define a function to count the number of swaps in bubble sort
def bubble_sort_count_swaps(arr):
    n = len(arr)
    swap_count = 0
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swap_count += 1
    return swap_count

# alt, works if values permutations of range(0, n)
def bubble_sort_swapcount(arr):
    n = len(arr)
    swap_count = 0
    for i in range(n):
        # find the position of the value i in arr
        pos = np.where(arr == i)[0][0]
        # remove this value from arr
        arr = np.delete(arr, pos)
        swap_count += pos
    return swap_count

# retinotopy index
def RI(S, A):
    """
    S: swap count
    A: avg swap count
    """
    return 1 - S / A

# avg swap count
def avg_swap_count(x):
    """
    Calculate the average number of swaps required by bubble sort

    Parameters
    ----------
    x : array-like
        Input array or list (only the length is used).

    Returns
    -------
    float
        The expected average number of swaps for bubble sort on a random
        permutation of length n = len(arr), given by (n-1)*(n-2)/4.
    """
    n = len(x)
    return (n-1) * (n - 2) / 4

# RI for 2 matched point sets, Euclidean distance
def RI_sets(P, Q):
    """
    Calculate the Retinotopy Index (RI) for two matched point sets.

    Parameters
    ----------
    P, Q : DataFrame or array-like
        arrays of points in d-dim, assume same dim and order-matched
    
    Returns
    -------
    float
        a list of Retinotopy Index (RI).
    """
    # convert P, Q to np.ndarray if they are DataFrames
    if isinstance(P, pd.DataFrame):
        P = P.values
    if isinstance(Q, pd.DataFrame):
        Q = Q.values

    # avg swap count
    A = avg_swap_count(P)
    
    # retinotopy index for each point in P
    ri = []

    # iterate over each pair of points in P and Q
    for i in range(len(P)):
        # compute the Euclidean distance between the point in P and all other points in P
        dists_P = np.linalg.norm(P - P[i], axis=1)
        # compute the order of distances, this is the P-order
        P_order = np.argsort(dists_P)
        
        # paired point in Q
        Q_paired = Q[i]

        # rearrange Q based on the P-order
        Q_ordered = Q[P_order]
        
        # compute distances of all points in Q to the point in Q that corresponds to the point in P
        dists_Q = np.linalg.norm(Q_ordered - Q_paired, axis=1)
        
        # compute the order of distances, this is the Q-order
        Q_order = np.argsort(dists_Q)
        
        # swap count for bubble sort on Q-order
        S = bubble_sort_swapcount(Q_order)
        
        # Calculate the Retinotopy Index (RI)
        ri.append(RI(S, A))

    return ri

# %%
# DEBUG
P = np.array([[0, 0],
              [1.3, 0],
              [2.1, 0]])
Q = np.array([[0 ,0],
              [2.1, 0],
              [1.3, 0]])
print(RI_sets(P, Q))

P = np.array([[0, 0],
              [1.1, 0],
              [2.3, 0]])
Q = np.array([[0 ,0],
              [2.3, 0],
              [1.1, 0]])
print(RI_sets(P, Q))

# %% [markdown]
# ## compute retinotopy index

# %%
syn_post_mean.shape, syn_pre_mean.shape

# %%
ri_list = []

# ri = RI_sets(syn_post_mean[['h', 'v']].values, syn_post_mean[['hex1_mean', 'hex2_mean']].values)
# ri_list.append(np.mean(ri)), print('[h v] - [hex]', np.mean(ri))
# ri = RI_sets(syn_post_mean[['h', 'v']].values, syn_post_mean[['x_mean', 'y_mean', 'z_mean']].values)
# ri_list.append(np.mean(ri)), print('[h v] - [x y z]', np.mean(ri))

ri = RI_sets(syn_post_mean[['h', 'v']].values, syn_pre_mean[['x_mean', 'y_mean', 'z_mean']].values)
ri_list.append(np.mean(ri)), print('[h v]', np.mean(ri))
ri = RI_sets(syn_post_mean[['h']].values, syn_pre_mean[['x_mean', 'y_mean', 'z_mean']].values)
ri_list.append(np.mean(ri)), print('[h]', np.mean(ri))
ri = RI_sets(syn_post_mean[['v']].values, syn_pre_mean[['x_mean', 'y_mean', 'z_mean']].values)
ri_list.append(np.mean(ri)), print('[v]', np.mean(ri))
ri = RI_sets(syn_post_mean[['hex1_mean']].values, syn_pre_mean[['x_mean', 'y_mean', 'z_mean']].values)
ri_list.append(np.mean(ri)), print('[hex1]', np.mean(ri))
ri = RI_sets(syn_post_mean[['hex2_mean']].values, syn_pre_mean[['x_mean', 'y_mean', 'z_mean']].values)
ri_list.append(np.mean(ri)), print('[hex2]', np.mean(ri))

ri = RI_sets(syn_post_mean[['h','v']].values, pts_cs[['x', 'y', 'z']].values)
ri_list.append(np.mean(ri)), print('[h v] - cs', np.mean(ri))
ri = RI_sets(syn_post_mean[['h']].values, pts_cs[['x', 'y', 'z']].values)
ri_list.append(np.mean(ri)), print('[h] - cs', np.mean(ri))
ri = RI_sets(syn_post_mean[['v']].values, pts_cs[['x', 'y', 'z']].values)
ri_list.append(np.mean(ri)), print('[v] - cs', np.mean(ri))
ri = RI_sets(syn_post_mean[['hex1_mean']].values, pts_cs[['x', 'y', 'z']].values)
ri_list.append(np.mean(ri)), print('[hex1] - cs', np.mean(ri))
ri = RI_sets(syn_post_mean[['hex2_mean']].values, pts_cs[['x', 'y', 'z']].values)
ri_list.append(np.mean(ri)), print('[hex2] - cs', np.mean(ri))

print(np.max(np.abs(ri_list[0:5])), np.argmax(np.abs(ri_list[0:5])))
print(np.max(np.abs(ri_list[5:10])), np.argmax(np.abs(ri_list[5:10])))

# %% [markdown]
# # VPN + 1 hop in CB

# %%
def RI_rois(ids, roi1, roi2):
    """
    Calculate the Retinotopy Index (RI) for two rois.

    Parameters
    ----------
    ids : list
        List of bodyIds to fetch synapses for.
    roi1 : str
        Name of the first ROI.
    roi2 : str
        Name of the second ROI.

    Returns
    -------
    float
        The Retinotopy Index (RI) value.
    """
    syn_pre = neuprint.fetch_mean_synapses(
        NC(bodyId=ids), SC(type='post', rois=roi1)
    )
    syn_post = neuprint.fetch_mean_synapses(
        NC(bodyId=ids), SC(type='pre', rois=roi2)
    )
    
    ri = RI_sets(syn_post[['x', 'y', 'z']].values, syn_pre[['x', 'y', 'z']].values)
    
    return np.mean(ri)

# %% [markdown]
# ## columnar types

# %%
n_info, _ = fetch_neurons(NC(instance= ['Tm1_R']))
RI_rois(n_info['bodyId'].values, 'ME(R)', 'LO(R)')

# %%
n_info, _ = fetch_neurons(NC(instance= ['Tm2_R']))
RI_rois(n_info['bodyId'].values, 'ME(R)', 'LO(R)')

# %%
n_info, _ = fetch_neurons(NC(instance= ['Tm9_R']))
RI_rois(n_info['bodyId'].values, 'ME(R)', 'LO(R)')

# %%
n_info, _ = fetch_neurons(NC(instance= ['Tm20_R']))
RI_rois(n_info['bodyId'].values, 'ME(R)', 'LO(R)')

# %%
n_info, _ = fetch_neurons(NC(instance= ['T5a_R']))
RI_rois(n_info['bodyId'].values, 'LO(R)', 'LOP(R)')

# %%
n_info, _ = fetch_neurons(NC(instance= ['T4a_R']))
RI_rois(n_info['bodyId'].values, 'ME(R)', 'LOP(R)')

# %% [markdown]
# ## look for multiplet downstream types

# %% [markdown]
# ### LC10s

# %%
n0, _ = fetch_neurons(NC(instance= ['LC10a_R', 'LC10c-1_R', 'LC10c-2_R', 'LC10d_R', 'LC10e_R']))

# %%
n1, c1 = neuprint.queries.fetch_adjacencies(
    sources= n0['bodyId'].values,
    targets= None,
    rois= rois_cb,
    min_total_weight=3
)
c2 = neuprint.merge_neuron_properties(n1, c1, ['type', 'instance'])\
    .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
    .agg({'weight': 'sum'})\
    .reset_index()\
    .sort_values(by='weight', ascending=False)

# %%
c2.groupby(['bodyId_post', 'instance_post']).agg({'weight': 'sum'})\
    .reset_index()\
    .groupby('instance_post').agg({'bodyId_post': 'count'})\
    .reset_index()\
    .sort_values(by='bodyId_post', ascending=False)\
    .head(20)

# %%
n_info, _ = fetch_neurons(NC(instance= ['AOTU008_R', 'AOTU050_R', 'AOTU038_R', 'SMP018_R', 'TuBu06_SBU_R']))

# %%
sk = neu.fetch_skeletons(n_info['bodyId'].values)

# %% [markdown]
# ### LC9

# %%
n0, _ = fetch_neurons(NC(instance= ['LC9_R']))

# %%
n1, c1 = neuprint.queries.fetch_adjacencies(
    sources= n0['bodyId'].values,
    targets= None,
    rois= rois_cb,
    min_total_weight=3
)
c2 = neuprint.merge_neuron_properties(n1, c1, ['type', 'instance'])\
    .groupby(['bodyId_pre', 'bodyId_post', 'instance_pre', 'instance_post'])\
    .agg({'weight': 'sum'})\
    .reset_index()\
    .sort_values(by='weight', ascending=False)

# %%
# types with multiple cells
c3 = c2.groupby(['instance_post']).agg({'weight': 'sum', 'bodyId_post':'nunique'})\
    .reset_index()\
    .sort_values(by='bodyId_post', ascending=False)

# %%
# todo check input percentage

# %%
n_info, _ = fetch_neurons(NC(instance= ['AOTU008_R', 'AOTU050_R', 'AOTU038_R', 'SMP018_R', 'TuBu06_SBU_R']))

# %%
sk = neu.fetch_skeletons(n_info['bodyId'].values)

# %% [markdown]
# ### plot

# %%
navis.plot3d([sk[sk.name == 'AOTU008_R'], cb_mesh],  palette='Set1', backend='plotly') #color_by='name',

# %% [markdown]
# ## RI roi->roi via a cell type

# %%
# n_info, _ = fetch_neurons(NC(instance= ['AOTU008_R']))
n_info, _ = fetch_neurons(NC(instance= ['PVLP004_R']))

# sk = neu.fetch_skeletons(n_info['bodyId'].values)
n_info.shape

# %%
syn = neuprint.queries.fetch_synapses(
    NC(bodyId= n_info['bodyId'].values),
    SC(
        type='post', 
        rois=rois_cb, primary_only=False)
)

# %%
# count the number of synapses in each roi
syn.groupby('roi').agg({'bodyId': 'count'}).\
    reset_index().\
    sort_values(by='bodyId', ascending=False).\
    head(10)

# %%
RI_rois(n_info['bodyId'].values, 'AOTU(R)', 'SIP(R)')

# %% [markdown]
# # end


