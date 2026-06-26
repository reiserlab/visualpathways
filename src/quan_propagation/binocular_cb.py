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
# ### binocular vision in CB
#

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
from utils.config import CACHE_DIR, DATA_DIR, FIG_DIR, HTML_FIG_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

result_dir = FIG_DIR / 'quan_propagation'
result_dir.mkdir(parents=True, exist_ok=True)

cache_dir = CACHE_DIR / 'quan_propagation'
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # load data

# %%
# # load res
# meta_cb_vpn_r = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_cumsum40_ghop_res_041010_r2.pkl'))
# meta_type_cb_vpn_r = pd.read_pickle(Path(cache_dir, 'meta_type_cb_vpn_cumsum40_ghop_res_041010_r2.pkl'))
# meta_cb_vpn_l = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_cumsum40_ghop_res_041010_r2_left.pkl'))
# meta_type_cb_vpn_l = pd.read_pickle(Path(cache_dir, 'meta_type_cb_vpn_cumsum40_ghop_res_041010_r2_left.pkl'))

# %%
# load vic
meta_cb_vpn_r = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_cumsum40_ghop_vic_041010.pkl'))
meta_type_cb_vpn_r = pd.read_pickle(Path(cache_dir, 'meta_type_cb_vpn_cumsum40_ghop_vic_041010.pkl'))
meta_cb_vpn_l = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_cumsum40_ghop_vic_041010_left.pkl'))
meta_type_cb_vpn_l = pd.read_pickle(Path(cache_dir, 'meta_type_cb_vpn_cumsum40_ghop_vic_041010_left.pkl'))

# %% [markdown]
# # binocular index

# %% [markdown]
# ## cb cells

# %%
# find bodyId common in meta_cb_vpn_r and meta_cb_vpn_l
common_bodyId = set(meta_cb_vpn_r['bodyId']).intersection(set(meta_cb_vpn_l['bodyId']))
common_bodyId = list(common_bodyId)
len(common_bodyId)  

# %%
meta_cb = meta_cb_vpn_l[(meta_cb_vpn_l['bodyId'].isin(common_bodyId)) & (meta_cb_vpn_l['ghop'].isin(['T1','T2']))].copy()
meta_cb = pd.merge(meta_cb, 
                   meta_cb_vpn_r.loc[
                       (meta_cb_vpn_r['bodyId'].isin(common_bodyId)) & (meta_cb_vpn_r['ghop'].isin(['T1','T2'])),
                       ['bodyId', 'ht', 'vision', 'col_count', 'area_fit']],
                   on='bodyId', how='left', suffixes=('_l', '_r'))
meta_cb.shape, meta_cb['instance'].nunique(), meta_cb['cell_type'].nunique()

# %%
# filter out these having NaN in vision_l or vision_r
meta_cb = meta_cb[~meta_cb['vision_l'].isna() & ~meta_cb['vision_r'].isna()].copy()
meta_cb.shape, meta_cb['instance'].nunique(), meta_cb['cell_type'].nunique()

# %%
# groupby instance
meta_cb = meta_cb.groupby(['instance','cell_type']).agg(
    cell_count = ('bodyId', 'count'),
    ghop = ('ghop', 'first'),
    ht_r = ('ht_r', 'median'),
    ht_l = ('ht_l', 'median'),
    vision_r = ('vision_r', 'median'),
    # vision_cv_r = ('vision_r', lambda x: round(np.std(x) / np.mean(x),2) if np.mean(x) != 0 else np.nan),
    vision_l = ('vision_l', 'median'),
    # vision_cv_l = ('vision_l', lambda x: round(np.std(x) / np.mean(x),2) if np.mean(x) != 0 else np.nan),
    col_count_r = ('col_count_r', 'median'),
    # area_fit_cv_r = ('area_fit_r', lambda x: round(np.std(x) / np.mean(x),2) if np.mean(x) != 0 else np.nan),
    col_count_l = ('col_count_l', 'median'),
    # area_fit_cv_l = ('area_fit_l', lambda x: round(np.std(x) / np.mean(x),2) if np.mean(x) != 0 else np.nan),
    area_fit_r = ('area_fit_r', 'median'),
    # area_fit_cv_r = ('area_fit_r', lambda x: round(np.std(x) / np.mean(x),2) if np.mean(x) != 0 else np.nan),
    area_fit_l = ('area_fit_l', 'median'),
    # area_fit_cv_l = ('area_fit_l', lambda x: round(np.std(x) / np.mean(x),2) if np.mean(x) != 0 else np.nan),
).reset_index()
meta_cb.shape

# %%
# find all instance that contains '_R' or '_L'
inst_r = meta_cb.loc[(meta_cb['instance'].str.contains('_R')) & (meta_cb.ghop=='T1'), ['instance','cell_type']]
inst_l = meta_cb.loc[(meta_cb['instance'].str.contains('_L')) & (meta_cb.ghop=='T1'), ['instance','cell_type']]
inst = pd.merge(inst_r, inst_l, on='cell_type', how='inner')
cell_type = inst['cell_type'].unique()

# find instance that does not contain '_R' or '_L'
type_m = meta_cb[~meta_cb['instance'].str.contains('_R') 
                 & ~meta_cb['instance'].str.contains('_L')
                 & (meta_cb.ghop=='T1')
                 ]['cell_type'].values

celltypes_t1 = list(cell_type) + list(type_m)

inst_r = meta_cb.loc[(meta_cb['instance'].str.contains('_R')) & (meta_cb.ghop=='T2'), ['instance','cell_type']]
inst_l = meta_cb.loc[(meta_cb['instance'].str.contains('_L')) & (meta_cb.ghop=='T2'), ['instance','cell_type']]
inst = pd.merge(inst_r, inst_l, on='cell_type', how='inner')
cell_type = inst['cell_type'].unique()

# find instance that does not contain '_R' or '_L'
type_m = meta_cb[~meta_cb['instance'].str.contains('_R') 
                 & ~meta_cb['instance'].str.contains('_L')
                 & (meta_cb.ghop=='T2')
                 ]['cell_type'].values

celltypes_t2 = list(cell_type) + list(type_m)

len(celltypes_t1), len(celltypes_t2)


# %%
# dominance
meta_cb['vision_dom'] = (meta_cb['vision_l'] - meta_cb['vision_r']) / (meta_cb['vision_l'] + meta_cb['vision_r'])
# smallest of the two over the sum of the two, ranges from 0 to 0.5
meta_cb['vision_lpr'] = (meta_cb['vision_l'] + meta_cb['vision_r'])
# meta_cb['bi'] = meta_cb[['vision_l', 'vision_r']].max(axis=1) / meta_cb[['vision_l', 'vision_r']].sum(axis=1)

# difference in 
meta_cb['area_fit_lmr'] = meta_cb['area_fit_l'] - meta_cb['area_fit_r']
meta_cb['col_count_lmr'] = meta_cb['col_count_l'] - meta_cb['col_count_r']
meta_cb['ht_lmr'] = (meta_cb['ht_l'] - meta_cb['ht_r']) / (meta_cb['ht_l'] + meta_cb['ht_r'])  / 2

# %%
# keep symmetric cell types, every cell_type should appear twice, remove singletons
meta_cb = meta_cb.groupby('cell_type').filter(lambda x: len(x) == 2)

#  remove cell_type has different ghop
meta_cb = meta_cb.groupby('cell_type').filter(lambda x: x['ghop'].nunique() == 1)
meta_cb.shape,  meta_cb['cell_type'].nunique()

# %%
# filter for cell types with a negative area_fit_lmr production
meta_cb = meta_cb.groupby('cell_type').filter(lambda x: x['area_fit_lmr'].prod() < 0)
meta_cb.shape,  meta_cb['cell_type'].nunique()

# %%
# separate t1 and t2
meta_cb_t1 = meta_cb[meta_cb['cell_type'].isin(celltypes_t1)].copy()
meta_cb_t2 = meta_cb[meta_cb['cell_type'].isin(celltypes_t2)].copy()
meta_cb_t1.shape, meta_cb_t1['cell_type'].nunique(),meta_cb_t2.shape, meta_cb_t2['cell_type'].nunique(),

# %%
# meta_cb.columns
meta_cb_t1.sort_values(by=['instance', 'area_fit_lmr'], ascending=False)

# %%
# DEBUG, something wrong with col_count_l ??

# %% [markdown]
# ## plot area_lmr

# %%
# plotly.go scatter plot, x-axis is cell_type, y-axis is area_fit_lmr, if the instance ends with '_R', color it black, if ends with '_L', color it blue, else gray
meta_cb_t1['color'] = meta_cb_t1['instance'].apply(lambda x: 'black' if x.endswith('_R') else ('blue' if x.endswith('_L') else 'gray'))
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=meta_cb_t1['cell_type'],
    y=meta_cb_t1['area_fit_lmr'],
    mode='markers',
    text=meta_cb_t1['instance'],
    marker=dict(color=meta_cb_t1['color'], size=10),
))
fig.update_layout(
    title='Area Fit L-R Ratio by Cell Type (T1)',
    xaxis_title='Cell Type',
    yaxis_title='Area Fit L-R Ratio',
    showlegend=False,
    template='plotly_white'
)
fig.show()

# save
# fig.write_html(Path(result_dir, 'cumsum40_001010', 'area_fit_lmr_t1.html'))
# save png
fig.write_image(Path(result_dir, 'cumsum40_001010', 'area_fit_lmr_t1.png'), width=1200, height=600, scale=2)


# %%
# t2, plotly.go scatter plot, x-axis is cell_type, y-axis is area_fit_lmr, if the instance ends with '_R', color it black, if ends with '_L', color it blue, else gray
meta_cb_t2['color'] = meta_cb_t2['instance'].apply(lambda x: 'black' if x.endswith('_R') else ('blue' if x.endswith('_L') else 'gray'))
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=meta_cb_t2['cell_type'],
    y=meta_cb_t2['area_fit_lmr'],
    mode='markers',
    text=meta_cb_t2['instance'],
    marker=dict(color=meta_cb_t2['color'], size=5),
))
fig.update_layout(
    title='Area Fit L-R Ratio by Cell Type (T2)',
    xaxis_title='Cell Type',
    yaxis_title='Area Fit L-R Ratio',
    showlegend=False,
    template='plotly_white'
)
fig.show()

# save
# fig.write_html(Path(result_dir, 'cumsum40_001010', 'area_fit_lmr_t2.html'))
# save png
fig.write_image(Path(result_dir, 'cumsum40_001010', 'area_fit_lmr_t2.png'), width=1200, height=600, scale=2)


# %% [markdown]
# ## Plot

# %%
meta_cb

# %%
# scatter plot meta_cb[bi_dom vs bi] with point color mapped to meta_cb['bi']
bi_min = meta_cb['bi'].min()
bi_max = meta_cb['bi'].max()

fig = go.Figure()
for ct, df_ct in meta_cb.groupby('cell_type'):
   fig.add_trace(
      go.Scatter(
         x=df_ct['bi_dom'],
         y=df_ct['ht_lmr'],
         mode='markers',
         name=str(ct),
         marker=dict(
            size=6,
            opacity=0.8,
            color=df_ct['bi'],
            coloraxis='coloraxis'
         ),
         customdata=np.stack([
            df_ct['instance'],
            df_ct['ht_l'],
            df_ct['ht_r'],
            df_ct['vision_l'],
            df_ct['vision_r'],
            df_ct['area_fit_l'],
            df_ct['area_fit_r'],
            df_ct['area_fit_lmr']
         ], axis=-1),
         hovertemplate=(
            "cell_type=%{fullData.name}"
            "<br>instance=%{customdata[0]}"
            "<br>ht_l=%{customdata[1]:.4f}"
            "<br>ht_r=%{customdata[2]:.4f}"
            "<br>vision_l=%{customdata[3]:.4f}"
            "<br>vision_r=%{customdata[4]:.4f}"
            "<br>area_fit_l=%{customdata[5]:.3f}"
            "<br>area_fit_r=%{customdata[6]:.3f}"
            "<br>area_fit_lmr=%{customdata[7]:.3f}"
            "<br>bi_dom=%{x:.3f}"
            "<br>ht_lmr=%{y:.3f}<extra></extra>"
         ),
      )
   )

fig.update_layout(
   width=700,
   height=500,
   template='plotly_white',
   xaxis_title='bi_dom',
   yaxis_title='ht_lmr',
   showlegend=False,
   coloraxis=dict(
      colorscale='Viridis',
      cmin=bi_min,
      cmax=bi_max,
      colorbar=dict(title='bi')
   )
)
fig.show()
# save
# fig.write_image(Path(result_dir, 'bi_dom_vs_bi.png'), scale=2)
# save html
HTML_FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.write_html(HTML_FIG_DIR / 'bi_dom_vs_ht_lmr.html')

# %%



