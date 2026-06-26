# %%
%load_ext autoreload
%autoreload 2

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
import os
os.environ.pop('MPLBACKEND', None)  # Remove the problematic backend
import matplotlib

# %%
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
matplotlib.rcParams['pdf.fonttype'] = 42

from utils.ol_color import OL_COLOR
import cmap

import pandas as pd
import numpy as np
import re
import pickle
import alphashape
from scipy.stats import gaussian_kde
import os, datetime
import scipy
# from openpyxl.styles import Font

import neuprint
from neuprint import fetch_neurons, NeuronCriteria as NC, SynapseCriteria as SC

import navis
import navis.interfaces.neuprint as neu

import connectome_interpreter as ci

from quan_propagation.func import plot_extreme_projection_xy, plot_extreme_projection_xz, get_roi_outline
from quan_propagation.func import plot_neuron_with_outlines

# %%
from utils.config import (
    CACHE_DIR, DATA_DIR, DATASET, FIG_DIR, SIDE, SIDE_CHAR, N_FLOW, HIT_THRE, PARAMS_DIR
)
result_dir = FIG_DIR / 'quan_propagation'
cache_dir = CACHE_DIR / 'quan_propagation'

# %% [markdown]
# # load rois and bbox

# %%
from utils.query_roi import get_primary_rois
rois_cb = get_primary_rois('CentralBrain')
len(rois_cb)

# %%
# roi color
from utils.palettes import load_roi_colors
roi_col = load_roi_colors()

# %% [markdown]
# ### get brain mesh to set plotting ranges

# %%
# cb
xrange = [20000, 80000]
yrange = [0, 50000]

# cb+ol, x-y plane
# xrange = [0, 102000] # 2x, adjust for kde
# yrange = [4000, 55000]
# xrange = [0, 104000] # 2x, adjust for kde, /400
# yrange = [4000, 56000]

# zrange = [10000, 21000, 42000] # central section

# # vnc, x-z plane
# xrange_vnc = [26000, 70000] # adjusted for kde
# zrange_vnc = [50000, 138000]

# yrange_vnc = [35000,  75000]

# # zdivide cb and vnc
# zdivide = 51900

# %%


# %% [markdown]
# # Meta

# %% [markdown]
# ## left and right vpn_cb, from 'assemble_vpn_cb.ipynb'

# %%
# load
vp_cb_r = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_r.p'))
vp_cb_l = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_l.p'))

# %% [markdown]
# # Query tbar

# %%
# vp_cb_vic = pd.read_pickle(cache_dir / 'vp_cb_vic_w_hit_df.p')
# vp_cb_vic_left = pd.read_pickle(cache_dir / 'vp_cb_left_vic_w_hit_df.p')
# ids_lr = list(set(vp_cb_vic['bodyId']).union(set(vp_cb_vic_left['bodyId'])))
ids_lr = list(set(vp_cb_r['bodyId']).union(set(vp_cb_l['bodyId'])))
len(ids_lr)

# %% [markdown]
# ## query tbar in cb + vpn, run once, ~ 100min

# %%
# syn = neuprint.fetch_synapses(NC(bodyId = ids_lr), SC(rois=rois_cb, primary_only=True, type='pre'))
# # syn = neuprint.fetch_synapses(NC(bodyId = meta_cb_vpn['bodyId']), SC(rois=rois_vnc, primary_only=True, type='pre'))

# # SAVE pickle
# syn.to_pickle(Path(cache_dir, 'tbar_vp_cb.pkl'))

# # load
# # syn = pd.read_pickle(Path(cache_dir, 'tbar_vp_cb.pkl'))

# %% [markdown]
# ## cb rois tbar counts, run once, ~ 9 hours

# %%
# # get syn count for each roi and save into a df
# tbar_count_roi_cb = pd.DataFrame(columns=['roi', 'tbar_count'])
# for roi in rois_cb:
#     ss = neuprint.fetch_synapses(NC(), SC(rois=[roi], primary_only=True, type='pre'))
#     tbar_count_roi_cb = pd.concat([tbar_count_roi_cb, pd.DataFrame({'roi':[roi], 'tbar_count':[len(ss)]})], ignore_index=True)

# # save tbar_count_roi_cb
# tbar_count_roi_cb.to_csv(Path(result_dir, 'tbar_count_roi_cb.csv'), index=False)

# %% [markdown]
# # Individual ROI outlines

# %%
# plot_dir = FIG_DIR / 'quan_propagation' / 'roi_outlines'
# plot_dir.mkdir(parents=True, exist_ok=True)

# # load outline
# outline_1 = pd.read_csv(Path(result_dir, f'outline_cb_1.csv'), index_col=False)
# outline_2 = pd.read_csv(Path(result_dir, f'outline_cb_2.csv'), index_col=False)

# def _rotate_outlines(outlines, rot_mat, center):
#     rotated = []
#     for poly in outlines:
#         pts = np.asarray(poly, dtype=float)
#         if pts.size == 0:
#             rotated.append([])
#             continue
#         rotated.append(((rot_mat @ (pts - center).T).T + center).tolist())
#     return rotated

# %%
# rotation_deg = 3


# for roi in rois_cb:
#     outlines_bkgd = [outline_1,  outline_2] + get_roi_outline([roi], 0.0002)

#     center = np.array([(xrange[0] + xrange[1]) / 2, (yrange[0] + yrange[1]) / 2])
#     theta = np.radians(rotation_deg)
#     rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#     outlines_bkgd = _rotate_outlines(outlines_bkgd, rot_mat, center)

#     fig, ax = plt.subplots(figsize=(8, 4))
#     for poly in outlines_bkgd:
#         pts = np.asarray(poly, dtype=float)
#         if pts.size >= 4:
#             ax.plot(pts[:, 0], pts[:, 1], color='black', linewidth=1)

#     ax.set_xlim(20000, 80000)
#     ax.set_ylim(0, 50000)
#     ax.invert_yaxis()
#     # asp ratio = 1
#     ax.set_aspect('equal', adjustable='box')

#     plt.savefig(Path(plot_dir, f'roi_{roi}.pdf'), bbox_inches='tight')
#     plt.close(fig)

# %% [markdown]
# # 2D extre value proj

# %% [markdown]
# ## plot dir

# %%
plot_dir = FIG_DIR / 'quan_propagation' / 'v1'
plot_dir.mkdir(parents=True, exist_ok=True)

plot_eg_dir = FIG_DIR / 'quan_propagation' / 'v1'
plot_eg_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### outlines

# %%
# # sample syn for points
# xy = syn[['x','y']].sample(n=50000, random_state=42)

# # save to csv -> make ashape in R
# xy.to_csv(Path(cache_dir, 'tbar_cb_50k.csv'), index=False)

# load outline
outline_1 = pd.read_csv(Path(result_dir, f'outline_cb_1.csv'), index_col=False)
outline_2 = pd.read_csv(Path(result_dir, f'outline_cb_2.csv'), index_col=False)

outlines_bkgd = [outline_1,  outline_2]

# %% [markdown]
# ## load tbar

# %%
syn0 = pd.read_pickle(Path(cache_dir, 'tbar_vp_cb.pkl'))

# psd for ratio
# syn0 = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_psd_for_ratio.pkl'))

# %%


# %% [markdown]
# ## top rois

# %%
lr = 'r'

meta_cb_vpn = pd.read_pickle(Path(DATA_DIR, f'vp_cb_hit_vic_{lr}.p'))
print(len(meta_cb_vpn))
thr_vic = 5e-4
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.VIC > thr_vic]
print(len(meta_cb_vpn))
# change col names
meta_cb_vpn.rename(columns={'VIC':'vision', 'hitting_time':'ht'}, inplace=True)

syn = pd.merge(syn0, 
               meta_cb_vpn[['bodyId', 'vision', 'ht', 'main_groups']], 
               on='bodyId', how='inner')
print(syn.shape)

# %%
rois = syn.groupby(['roi']).agg(count=('type', 'count')).reset_index().sort_values(by='count', ascending=False)
# roi_base, removing '(R)' or '(L)' at the end
rois['roi_base'] = rois['roi'].str.replace(r'\([RL]\)$', '', regex=True)

# %%
syn.groupby(['roi']).size().sort_values(ascending=False).plot(kind='bar', figsize=(15, 4))
plt.show()

# %% [markdown]
# ## ht bins

# %%
# vic
lr = '' if SIDE == 'right' else '_left'
meta_cb_vpn = pd.read_pickle(CACHE_DIR / f'vp_cb{lr}_vic_w_hit_df.p')

print(len(meta_cb_vpn))
thr_vic = 5e-4
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.VIC > thr_vic]
print(len(meta_cb_vpn))
# change col names
meta_cb_vpn.rename(columns={'VIC':'vision', 'hitting_time':'ht'}, inplace=True)

# %%
# res 
lr = '' if SIDE == 'right' else '_left'
vp_cb_vic = pd.read_pickle(CACHE_DIR / f'vp_cb{lr}_vic_w_hit_df.p')
fit_rf = pd.read_pickle(CACHE_DIR / f'{DATASET}{lr}_w_hit_fit_rf.p')

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
np.nanpercentile(meta_cb_vpn.ht, [0,20,40,60,80, 100])

# %%
# plot two overlapping (with transparency) histograms for 'OL output' and 'nonOL' groups' hitting time, show counts
plt.figure(figsize=(6, 3))
meta_cb_vpn[meta_cb_vpn['main_groups'] == 'OL output'].ht.hist(bins=np.arange(0, 5.5, 0.1), alpha=0.7, label='OL output', color='#64a0d1')
meta_cb_vpn[meta_cb_vpn['main_groups'] != 'OL output'].ht.hist(bins=np.arange(0, 5.5, 0.1), alpha=0.5, label='nonOL', color='black')
plt.xlabel('Hitting time')
plt.xticks(np.arange(0, 5.5, 1))
# plt.yticks(np.arange(0, 2000, 500))
plt.yticks(np.arange(0, 2000, 250))
plt.xlim(1, 5)
# plt.ylim(0, 1500)
plt.ylim(0, 600)
# remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.legend()
# plt.savefig(Path(plot_dir, f'ht_hist_vic.pdf'), bbox_inches='tight')
plt.savefig(Path(plot_dir, f'ht_hist_res.pdf'), bbox_inches='tight')
plt.show()

# %% [markdown]
# ## VIC

# %% [markdown]
# ### individual channels

# %%
# load vp_cb

lr = 'r'
meta_cb_vpn = pd.read_pickle(Path(cache_dir, f'vp_cb_hit_vic_{lr}.p'))
print(len(meta_cb_vpn))

thr_vic = 5e-4
# median
inst = meta_cb_vpn.groupby('instance').agg({'VIC':'median'}).reset_index()
inst = inst[inst.VIC > thr_vic]['instance'].unique()
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.instance.isin(inst)]
print(len(meta_cb_vpn))
# # indiv
# meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.VIC > thr_vic]
# print(len(meta_cb_vpn))

# change col names
meta_cb_vpn.rename(columns={'VIC':'vision', 'hitting_time':'ht'}, inplace=True)

# %%
vi_ls = ['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R7d_R', 'R8_R', 'R8d_R', 'HBeyelet_R']
# vi_ls = ['HBeyelet_R']

for vi in vi_ls:
    # load
    effwt = pd.read_pickle(Path(cache_dir, f'effwt_visr_{vi}.pkl'))

    # col sums, convert to df with column name as bodyId and VIC
    effwt_sum = effwt.sum(axis=0)
    df = pd.DataFrame({'bodyId': effwt_sum.index.astype(int), 'VIC': effwt_sum.values})
    df = pd.merge(df, meta_cb_vpn[['instance', 'bodyId', 'ht', 'main_groups']], on='bodyId', how='inner')
    syn = pd.merge(syn0, df, on='bodyId', how='inner')
    print(syn.shape)

    # by ht, separate vp and cb
    gps = ['OL output', 'nonOL']
    top_rois_ls = []
    tbar_counts = []

    for gp in gps:
        if gp == 'OL output':
            ht_div = [1, 5]
            # vmax_ls = [0.26]
            for i in range(len(ht_div)-1):
                df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
                df_plot['val'] = df_plot['VIC']
                roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
                top_rois = roi_counts.iloc[:6]
                top_rois_ls.append(top_rois)
                tbar_counts.append(len(df_plot))
                percentiles = np.nanpercentile(df_plot['val'], [5, 99])
                if np.isscalar(percentiles):
                    vmin = vmax = percentiles
                else:
                    vmin, vmax = percentiles
                print(vmin, vmax)
                fig, _, _ = plot_extreme_projection_xy(
                    df_plot,
                    outlines_bkgd,
                    xrange = [20000, 80000],
                    yrange = [000, 50000],
                    xnbins = 120 *2, 
                    ynbins = 100 *2, #50000 / 100 *8
                    agg = 'largest',
                    im_norm = 'linear',
                    vmin = 0,
                    # vmax = vmax_ls[i],
                    vmax = vmax,
                    agg_frac = 0.1,
                )
                plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_div[i]}_{ht_div[i+1]}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
                plt.savefig(Path(plot_dir, f'vic_{vi}_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}.pdf'), bbox_inches='tight')
                plt.close(fig)
        elif gp == 'nonOL':
            ht_div = [2, 3.5, 4, 5]
            # vmax_ls = [0.11, 0.06, 0.02]
            for i in range(len(ht_div)-1):
                df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
                df_plot['val'] = df_plot['VIC']
                roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
                top_rois = roi_counts.iloc[:6]
                top_rois_ls.append(top_rois)
                tbar_counts.append(len(df_plot))
                percentiles = np.nanpercentile(df_plot['val'], [5, 99])
                if np.isscalar(percentiles):
                    vmin = vmax = percentiles
                else:
                    vmin, vmax = percentiles
                print(vmin, vmax)
                fig, _, _ = plot_extreme_projection_xy(
                    df_plot,
                    outlines_bkgd,
                    xrange = [20000, 80000],
                    yrange = [000, 50000],
                    xnbins = 120 *2, 
                    ynbins = 100 *2, #50000 / 100 *8
                    agg = 'largest',
                    im_norm = 'linear',
                    vmin = 0,
                    # vmax = vmax_ls[i],
                    vmax = vmax,
                    agg_frac = 0.1,
                )
                plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_div[i]}_{ht_div[i+1]}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
                plt.savefig(Path(plot_dir, f'vic_{vi}_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}.pdf'), bbox_inches='tight')
                plt.close(fig)



# %%


# %% [markdown]
# ### all 5 channels, right side

# %%
# load vp_cb
lr = 'r'
meta_cb_vpn = pd.read_pickle(Path(DATA_DIR, f'vp_cb_hit_vic_{lr}.p'))
print(len(meta_cb_vpn))

thr_vic = 5e-4
# median
inst = meta_cb_vpn.groupby('instance').agg({'VIC':'median'}).reset_index()
inst = inst[inst.VIC > thr_vic]['instance'].unique()
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.instance.isin(inst)]
print(len(meta_cb_vpn))
# # indiv
# meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.VIC > thr_vic]
# print(len(meta_cb_vpn))

# change col names
meta_cb_vpn.rename(columns={'VIC':'vision', 'hitting_time':'ht'}, inplace=True)

# %%
syn = pd.merge(syn0, 
               meta_cb_vpn[['instance', 'bodyId', 'vision', 'ht', 'main_groups']], 
               on='bodyId', how='inner')
print(syn.shape)

# %%
syn[(syn['main_groups'] == 'OL output')].instance.nunique()

# %%
# # save to pickle
# syn[['bodyId','roi','vision']].to_pickle(Path(result_dir, 'tbar_VIC_forJudith.pkl'))

# %%
# by ht, separate vp and cb

gps = ['OL output', 'nonOL']
top_rois_ls = []
tbar_counts = []

for gp in gps:
    if gp == 'OL output':
        ht_div = [1, 5]
        vmax_ls = [0.26]
        for i in range(len(ht_div)-1):
            df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
            df_plot['val'] = df_plot['vision']
            roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
            top_rois = roi_counts.iloc[:6]
            top_rois_ls.append(top_rois)
            tbar_counts.append(len(df_plot))
            percentiles = np.nanpercentile(df_plot['val'], [5, 99])
            if np.isscalar(percentiles):
                vmin = vmax = percentiles
            else:
                vmin, vmax = percentiles
            print(vmin, vmax)
            fig, _, _ = plot_extreme_projection_xy(
                df_plot,
                outlines_bkgd,
                xrange = [20000, 80000],
                yrange = [000, 50000],
                xnbins = 120 *2, 
                ynbins = 100 *2, #50000 / 100 *8
                agg = 'largest',
                im_norm = 'linear',
                vmin = 0,
                vmax = vmax_ls[i],
                # vmax = vmax,
                agg_frac = 0.1,
            )
            plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_div[i]}_{ht_div[i+1]}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
            plt.savefig(Path(plot_dir, f'vic_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}_{lr}.pdf'), bbox_inches='tight')
            plt.close(fig)
    elif gp == 'nonOL':
        ht_div = [2, 3.5, 4, 5]
        vmax_ls = [0.11, 0.06, 0.02]
        for i in range(len(ht_div)-1):
            df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
            df_plot['val'] = df_plot['vision']
            roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
            top_rois = roi_counts.iloc[:6]
            top_rois_ls.append(top_rois)
            tbar_counts.append(len(df_plot))
            percentiles = np.nanpercentile(df_plot['val'], [5, 99])
            if np.isscalar(percentiles):
                vmin = vmax = percentiles
            else:
                vmin, vmax = percentiles
            print(vmin, vmax)
            fig, _, _ = plot_extreme_projection_xy(
                df_plot,
                outlines_bkgd,
                xrange = [20000, 80000],
                yrange = [000, 50000],
                xnbins = 120 *2, 
                ynbins = 100 *2, #50000 / 100 *8
                agg = 'largest',
                im_norm = 'linear',
                vmin = 0,
                vmax = vmax_ls[i],
                # vmax = vmax,
                agg_frac = 0.1,
            )
            plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_div[i]}_{ht_div[i+1]}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
            plt.savefig(Path(plot_dir, f'vic_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}_{lr}.pdf'), bbox_inches='tight')
            plt.close(fig)

# %%
# # get all the roi in top_rois_ls
# all_top_rois = pd.concat(top_rois_ls).index.unique().tolist()

# # get syn count for each roi and save into a df
# tbar_count_roi_cb = pd.DataFrame(columns=['roi', 'tbar_count'])
# for roi in all_top_rois:
#     ss = neuprint.fetch_synapses(NC(), SC(rois=[roi], primary_only=True, type='pre'))
#     tbar_count_roi_cb = pd.concat([tbar_count_roi_cb, pd.DataFrame({'roi':[roi], 'tbar_count':[len(ss)]})], ignore_index=True)

# # save tbar_count_roi_cb
# tbar_count_roi_cb.to_csv(Path(result_dir, 'tbar_count_roi_cb.csv'), index=False)

# %%
set(pd.DataFrame(top_rois_ls).columns) - set(tbar_count_roi_cb.roi)

# %%
# # add more
# tbar_count_roi_cb = pd.read_csv(Path(result_dir, 'tbar_count_roi_cb.csv'))

# roi = 'EB'
# ss = neuprint.fetch_synapses(NC(), SC(rois=[roi], primary_only=True, type='pre'))
# tbar_count_roi_cb = pd.concat([tbar_count_roi_cb, pd.DataFrame({'roi':[roi], 'tbar_count':[len(ss)]})], ignore_index=True)

# # save tbar_count_roi_cb
# tbar_count_roi_cb.to_csv(Path(result_dir, 'tbar_count_roi_cb.csv'), index=False)

# %%
# save roi tbar counts

# load tbar_count_roi_cb
tbar_count_roi_cb = pd.read_csv(Path(result_dir, 'tbar_count_roi_cb.csv'))

# save top_rois_ls as csv
top_rois_df = pd.DataFrame(top_rois_ls).fillna(0)
top_rois_perc_df = top_rois_df.copy()

# for each column in top_rois_df, find the corresponding tbar count from tbar_count_roi_cb
for roi in top_rois_df.columns:
    tbar_count = tbar_count_roi_cb[tbar_count_roi_cb['roi'] == roi]['tbar_count'].values
    top_rois_perc_df[roi] = top_rois_df[roi] / tbar_count[0]

top_rois_df.to_csv(Path(plot_dir, f'top_rois{lr}.csv'), index=False)
top_rois_perc_df.round(3).to_csv(Path(plot_dir, f'top_rois_perc{lr}.csv'), index=False)

# %%
# top-N accounting for perc of tbars in plot
top_rois_df.sum(axis=1) / tbar_counts

# %%
# sum columns in top_rois_perc_df
top_rois_perc_df.sum()

# %%
# # vic histo per roi
# ht_div = [1, 2, 3, 4, 5]
# top_rois_ls = []
# # Find top ROIs that account for at least 90% of total rows
# # threshold = 90

# for ht_bd in ht_div:
#     df_plot = syn.loc[(syn['ht'] >= ht_bd - 0.5) & (syn['ht'] < ht_bd + 0.5)].copy()

#     roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
#     top_rois = roi_counts.iloc[:6]
#     top_rois_ls.append(top_rois)

#     # cumulative_counts = roi_counts.cumsum()
#     # cumulative_pct = (cumulative_counts / len(df_plot)) * 100
#     # top_rois = roi_counts[cumulative_pct <= threshold]
#     # # Include one more ROI to cross the threshold
#     # if len(top_rois) < len(cumulative_pct):
#     #     top_rois = roi_counts.iloc[:len(top_rois) + 1]
#     # if len(top_rois) > 5:
#     #     top_rois = top_rois.iloc[:5]


#     # Horizontal histogram of counts in top ROIs only
#     fig, ax = plt.subplots(figsize=(3, 3))

#     # Create horizontal bar plot
#     y_positions = range(len(top_rois))
#     ax.barh(y_positions, top_rois.values, color='black', edgecolor='black')

#     # Set y-axis labels
#     ax.set_yticks(y_positions)
#     ax.set_yticklabels(top_rois.index, fontsize=10)
    
#     # # Reverse x-axis so 0 is on the right
#     # ax.invert_xaxis()
#     # # Move y-axis to the right side
#     # ax.yaxis.tick_right()
#     # # ax.yaxis.set_ticks_position('right')
#     # ax.yaxis.set_label_position('right')
    
#     # # remove x ticks
#     # ax.set_xticks([])

#     # # Labels and title
#     # ax.set_xlabel('Count', fontsize=12)
#     # ax.set_ylabel('ROI', fontsize=12)
#     # ax.set_title(f'Top ROIs by Count (ht_bd={ht_bd})', fontsize=14)
#     # # Add grid for better readability
#     # ax.grid(axis='x', alpha=0.3)
    
#     plt.suptitle(f'Top ROIs (ht_bd={ht_bd})', fontsize=14, fontweight='bold')
#     plt.tight_layout()

#     # Save the figure
#     # fig.savefig(plot_dir / f'top_roi_count_ht{ht_bd}.png', dpi=300, bbox_inches='tight')
#     fig.savefig(plot_dir / f'top_roi_count_ht{ht_bd}_left.png', dpi=300, bbox_inches='tight')
#     # fig.savefig(plot_dir / f'top_roi_count_ht{ht_bd}.pdf', bbox_inches='tight')
#     # fig.savefig(plot_dir / f'top_roi_count_ht{ht_bd}_left.pdf', bbox_inches='tight')
#     # Close the figure to prevent display
#     plt.close(fig)

# %% [markdown]
# ### all 5 channels, left side

# %%
# load vp_cb

lr = 'l'
meta_cb_vpn = pd.read_pickle(Path(cache_dir, f'vp_cb_hit_vic_{lr}.p'))
print(len(meta_cb_vpn))

thr_vic = 5e-4
# median
inst = meta_cb_vpn.groupby('instance').agg({'VIC':'median'}).reset_index()
inst = inst[inst.VIC > thr_vic]['instance'].unique()
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.instance.isin(inst)]
print(len(meta_cb_vpn))
# # indiv
# meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.VIC > thr_vic]
# print(len(meta_cb_vpn))

# change col names
meta_cb_vpn.rename(columns={'VIC':'vision', 'hitting_time':'ht'}, inplace=True)

# %%
# use ht
syn = pd.merge(syn0, 
               meta_cb_vpn[['instance', 'bodyId', 'vision', 'ht', 'main_groups']], 
               on='bodyId', how='inner')
print(syn.shape)

# %%
# by ht, separate vp and cb

gps = ['OL output', 'nonOL']
top_rois_ls = []
tbar_counts = []

for gp in gps:
    if gp == 'OL output':
        ht_div = [1, 5]
        vmax_ls = [0.26]
        for i in range(len(ht_div)-1):
            df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
            df_plot['val'] = df_plot['vision']
            roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
            top_rois = roi_counts.iloc[:6]
            top_rois_ls.append(top_rois)
            tbar_counts.append(len(df_plot))
            percentiles = np.nanpercentile(df_plot['val'], [5, 99])
            if np.isscalar(percentiles):
                vmin = vmax = percentiles
            else:
                vmin, vmax = percentiles
            print(vmin, vmax)
            fig, _, _ = plot_extreme_projection_xy(
                df_plot,
                outlines_bkgd,
                xrange = [20000, 80000],
                yrange = [000, 50000],
                xnbins = 120 *2, 
                ynbins = 100 *2, #50000 / 100 *8
                agg = 'largest',
                im_norm = 'linear',
                vmin = 0,
                vmax = vmax_ls[i],
                # vmax = vmax,
                agg_frac = 0.1,
            )
            plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_div[i]}_{ht_div[i+1]}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
            plt.savefig(Path(plot_dir, f'vic_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}_{lr}.pdf'), bbox_inches='tight')
            plt.close(fig)
    elif gp == 'nonOL':
        ht_div = [2, 3.5, 4, 5]
        vmax_ls = [0.11, 0.06, 0.02]
        for i in range(len(ht_div)-1):
            df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
            df_plot['val'] = df_plot['vision']
            roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
            top_rois = roi_counts.iloc[:6]
            top_rois_ls.append(top_rois)
            tbar_counts.append(len(df_plot))
            percentiles = np.nanpercentile(df_plot['val'], [5, 99])
            if np.isscalar(percentiles):
                vmin = vmax = percentiles
            else:
                vmin, vmax = percentiles
            print(vmin, vmax)
            fig, _, _ = plot_extreme_projection_xy(
                df_plot,
                outlines_bkgd,
                xrange = [20000, 80000],
                yrange = [000, 50000],
                xnbins = 120 *2, 
                ynbins = 100 *2, #50000 / 100 *8
                agg = 'largest',
                im_norm = 'linear',
                vmin = 0,
                vmax = vmax_ls[i],
                # vmax = vmax,
                agg_frac = 0.1,
            )
            plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_div[i]}_{ht_div[i+1]}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
            plt.savefig(Path(plot_dir, f'vic_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}_{lr}.pdf'), bbox_inches='tight')
            plt.close(fig)

# %% [markdown]
# ### sex dim 

# %%
cypher = """
MATCH (n:Neuron)
RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance,
       n.dimorphism AS dimorphism
ORDER BY n.type, n.instance
"""
df = c.fetch_custom(cypher) 

# %%
df = df[~df['dimorphism'].str.contains('potentially', na=True)]
df.dimorphism.value_counts(dropna=False)

# %%
# load vp_cb
lr = 'r'
meta_cb_vpn = pd.read_pickle(Path(DATA_DIR, f'vp_cb_hit_vic_{lr}.p'))
print(len(meta_cb_vpn))

thr_vic = 5e-4
# median
inst = meta_cb_vpn.groupby('instance').agg({'VIC':'median'}).reset_index()
inst = inst[inst.VIC > thr_vic]['instance'].unique()
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.instance.isin(inst)]
print(len(meta_cb_vpn))
# # indiv
# meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.VIC > thr_vic]
# print(len(meta_cb_vpn))

# change col names
meta_cb_vpn.rename(columns={'VIC':'vision', 'hitting_time':'ht'}, inplace=True)

# %%
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.instance.isin(df.instance)]
meta_cb_vpn.shape

# %%
syn = pd.merge(syn0, 
               meta_cb_vpn[['instance', 'bodyId', 'vision', 'ht', 'main_groups']], 
               on='bodyId', how='inner')
print(syn.shape)

# %%

vmax_ls = [0.26]
df_plot = syn.copy()
df_plot['val'] = df_plot['vision']
roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
top_rois = roi_counts.iloc[:6]
top_rois_ls.append(top_rois)
tbar_counts.append(len(df_plot))
percentiles = np.nanpercentile(df_plot['val'], [5, 99])
if np.isscalar(percentiles):
    vmin = vmax = percentiles
else:
    vmin, vmax = percentiles
print(vmin, vmax)
fig, _, _ = plot_extreme_projection_xy(
    df_plot,
    outlines_bkgd,
    xrange = [20000, 80000],
    yrange = [000, 50000],
    xnbins = 120 *2, 
    ynbins = 100 *2, #50000 / 100 *8
    agg = 'largest',
    im_norm = 'linear',
    vmin = 0,
    # vmax = vmax_ls[i],
    vmax = vmax,
    agg_frac = 0.1,
)
plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
plt.savefig(Path(plot_dir, f'sexdim_vic_ht_{lr}.pdf'), bbox_inches='tight')
plt.close(fig)

# %% [markdown]
# ### obs. Find top ROIs with most rows and create density plots

# %%
ht_bd = 3
df_plot = syn.loc[(syn['ht'] >= ht_bd - 0.5) & (syn['ht'] < ht_bd + 0.5)].copy()

# %%
# Find top ROIs that account for at least 80% of total rows
threshold = 80

total_rows = len(df_plot)
roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
cumulative_counts = roi_counts.cumsum()
# Calculate cumulative percentage
cumulative_pct = (cumulative_counts / total_rows) * 100

top_rois = cumulative_pct[cumulative_pct <= threshold]
# Include one more ROI to cross the threshold
if len(top_rois) < len(cumulative_pct):
    top_rois = cumulative_pct.iloc[:len(top_rois) + 1]
if len(top_rois) > 5:
    top_rois = top_rois.iloc[:5]

top_rois = top_rois.index.tolist()
top_rois

# %%
# Create horizontal half violin plots with swapped axes
# x-axis: vision, y-axis: ROI names
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Use the same data and calculations from before
from scipy import stats

# Collect all vision values to determine global y-axis range
all_vision_vals = []
roi_data_dict = {}
for roi in top_rois:
    roi_data = df_plot[df_plot['roi'] == roi]['vision'].values
    roi_data_dict[roi] = roi_data
    all_vision_vals.extend(roi_data)

# Set global x-axis limits
global_min = np.min(all_vision_vals)
global_max = np.max(all_vision_vals)

# Position for each ROI on y-axis
y_positions = np.arange(len(top_rois))

# Calculate all densities first to find global max density for consistent scaling
all_densities_h = []
x_range = np.linspace(global_min, global_max, 200)

for roi in top_rois:
    roi_data = roi_data_dict[roi]
    kde = stats.gaussian_kde(roi_data)
    density = kde(x_range)
    all_densities_h.append(density)

# Find the global maximum density across all ROIs
global_max_density_h = max([d.max() for d in all_densities_h])

# Plot each ROI as a horizontal half violin
for idx, roi in enumerate(top_rois):
    roi_data = roi_data_dict[roi]
    density = all_densities_h[idx]
    
    # Normalize density using GLOBAL max density for consistent scaling
    density_normalized = density / global_max_density_h * 0.5  # Scale to 0.35 height
    
    # Plot only the upper half (horizontal orientation)
    y_pos = y_positions[idx]
    ax.fill_between(x_range, y_pos, y_pos + density_normalized, 
                     alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1.5)
    
    # # Add quartile lines
    # q1, median, q3 = np.percentile(roi_data, [25, 50, 75])
    # ax.vlines(median, y_pos, y_pos + 0.35, colors='white', linewidth=2, zorder=3)
    # ax.vlines([q1, q3], y_pos, y_pos + 0.35, colors='white', linewidth=1, linestyles='dashed', zorder=3)

# Set labels and formatting
ax.set_xlabel('Vision', fontsize=14, fontweight='bold')
ax.set_ylabel('ROI', fontsize=14, fontweight='bold')
ax.set_yticks(y_positions + 0.175)  # Center the labels
ax.set_yticklabels([f'{roi}\n(n={len(roi_data_dict[roi]):,})' for roi in top_rois], fontsize=12)
ax.set_xlim([global_min, global_max])
ax.set_ylim([-0.1, len(top_rois) - 0.5])
ax.grid(True, alpha=0.3, axis='x')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.title('Vision Distribution by ROI (Top 5)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
# plt.savefig(Path(plot_dir, 'top5_roi_vision_distribution_horizontal.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### vpn clusters

# %%
# VPN cluster
from utils import ol_data
FRAC = 0.19
clusters  = ol_data.get_ol_clusters(frac_thre=FRAC)


# vpn_clu = pd.read_csv(CACHE_DIR / f'{DATASET}_OL_{SIDE_CHAR}_in_out_clusters.csv')
# meta_cb_vpn_clu = pd.merge(meta_cb_vpn, vpn_clu, on='instance', how='left')

# # clu_comp = pd.read_csv(Path(result_dir, 'vpn_cluster_comparison.csv'))
# # meta_cb_vpn_clu = pd.merge(meta_cb_vpn, clu_comp, on='instance', how='left')

# meta_cb_vpn_clu = meta_cb_vpn_clu[meta_cb_vpn_clu['cluster'].notna()]

# meta_cb_vpn_clu.shape

# %%
clusters.cluster.value_counts(dropna=False)

# %%
# use ht
syn = pd.merge(syn0, 
               meta_cb_vpn_clu[['bodyId', 'vision', 'ht', 'cluster', 'label_8']], 
               on='bodyId', how='inner')
print(syn.shape)

# %%
# by cluster
n_clu = ['label_8']
for n_cluster in n_clu:
    clu_list = syn[n_cluster].dropna().unique()
    for clu in clu_list:
        df_plot = syn.loc[ (syn[n_cluster] == clu)].copy()
        df_plot['val'] = df_plot['vision']
        percentiles = np.nanpercentile(df_plot['val'], [5, 99])
        if np.isscalar(percentiles):
            vmin = vmax = percentiles
        else:
            vmin, vmax = percentiles
        fig, _, _ = plot_extreme_projection_xy(
            df_plot,
            outlines_bkgd,
            xrange = [20000, 80000],
            yrange = [000, 50000],
            agg = 'largest',
            im_norm = 'linear',
            vmin = 0,
            vmax = vmax,
            agg_frac = 0.1,
        )
        # plt.savefig(Path(plot_dir, f'vic_vpn_{n_cluster}_clu_{int(clu)}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(Path(plot_dir, f'vic_vpn_{n_cluster}_clu_{int(clu)}.pdf'),  bbox_inches='tight')
        plt.close(fig)

# %%
# by cluster
for clu in meta_cb_vpn_clu.cluster.unique():
    df_plot = syn.loc[syn['cluster'] == clu].copy()
    df_plot['val'] = df_plot['vision']
    percentiles = np.nanpercentile(df_plot['val'], [5, 99])
    if np.isscalar(percentiles):
        vmin = vmax = percentiles
    else:
        vmin, vmax = percentiles
    fig, _, _ = plot_extreme_projection_xy(
        df_plot,
        outlines_bkgd,
        xrange = [20000, 80000],
        yrange = [000, 50000],
        agg = 'largest',
        im_norm = 'linear',
        vmin = 0,
        vmax = vmax,
        agg_frac = 0.1,
    )
    # plt.savefig(Path(plot_dir, f'vic_vpn_clu_{int(clu)}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(Path(plot_dir, f'vic_vpn_clu_{int(clu)}.pdf'),  bbox_inches='tight')
    plt.close(fig)

# %% [markdown]
# ### cb clusters

# %%
# VCBN cluster
from utils import cb_data
FRAC_CB = 0.17
cl_r = cb_data.get_cb_clusters(side_char='r', frac_thre=FRAC_CB, force=True)

# %%
cl_r.cluster.value_counts()

# %%
# load vp_cb
lr = 'r'
meta_cb_vpn = pd.read_pickle(Path(DATA_DIR, f'vp_cb_hit_vic_{lr}.p'))
print(len(meta_cb_vpn))

thr_vic = 5e-4
# median
inst = meta_cb_vpn.groupby('instance').agg({'VIC':'median'}).reset_index()
inst = inst[inst.VIC > thr_vic]['instance'].unique()
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.instance.isin(inst)]
print(len(meta_cb_vpn))
# # indiv
# meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.VIC > thr_vic]
# print(len(meta_cb_vpn))

# change col names
meta_cb_vpn.rename(columns={'VIC':'vision', 'hitting_time':'ht'}, inplace=True)
meta_cb_vpn.shape

# %%
# cluster
meta_cb_vpn_clu = pd.merge(meta_cb_vpn, cl_r, on='instance', how='left')
meta_cb_vpn_clu = meta_cb_vpn_clu[meta_cb_vpn_clu['cluster'].notna()] # basically drop OL cells
meta_cb_vpn_clu['cluster'] = meta_cb_vpn_clu['cluster'].astype(int)

meta_cb_vpn_clu.shape

# %%
# use ht
syn = pd.merge(syn0, 
               meta_cb_vpn_clu[['bodyId', 'instance', 'main_groups', 'vision', 'ht', 'cluster']], 
               on='bodyId', how='inner')
print(syn.shape)

# %%
# by ht, separate vp and cb

gps = ['nonOL']
top_rois_ls = []
tbar_counts = []

for clu in range(1, cl_r.cluster.max()+1):
    for gp in gps:
        ht_div = [2, 3.5, 4, 5]
        # vmax_ls = [0.11, 0.06, 0.02]
        for i in range(len(ht_div)-1):
            df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1]) & (syn['cluster'] == clu)].copy()
            df_plot['val'] = df_plot['vision']
            roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
            top_rois = roi_counts.iloc[:6]
            top_rois_ls.append(top_rois)
            tbar_counts.append(len(df_plot))
            percentiles = np.nanpercentile(df_plot['val'], [5, 99])
            if np.isscalar(percentiles):
                vmin = vmax = percentiles
            else:
                vmin, vmax = percentiles
            print(vmin, vmax)
            fig, _, _ = plot_extreme_projection_xy(
                df_plot,
                outlines_bkgd,
                xrange = [20000, 80000],
                yrange = [000, 50000],
                xnbins = 120 *2, 
                ynbins = 100 *2, #50000 / 100 *8
                agg = 'largest',
                im_norm = 'linear',
                vmin = 0,
                # vmax = vmax_ls[i],
                vmax = vmax,
                agg_frac = 0.1,
            )
            plt.text(0.5, 0.95, f'n_inst={df_plot.instance.nunique()},n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_div[i]}_{ht_div[i+1]}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
            plt.savefig(Path(plot_dir, f'vic_clu_{clu}_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}.pdf'), bbox_inches='tight')
            plt.close(fig)

# %%


# %% [markdown]
# ## resolution

# %% [markdown]
# ### visual inputs
# 

# %%
# # from Judith, 
# lr = '' if SIDE == 'right' else '_left'

# vp_cb_vic = pd.read_pickle(CACHE_DIR / f'vp_cb{lr}_vic_w_hit_df.p')
# # already median filtered
# fit_rf = pd.read_pickle(CACHE_DIR / f'{DATASET}{lr}_w_hit_fit_rf.p')

# meta_cb_vpn = pd.merge(
#     fit_rf[['bodyId','instance','size','r2', 'hitting_time', 'main_groups']],
#     vp_cb_vic[['bodyId','VIC']],
#     on='bodyId', how='left'
# )


lr = 'r'
vp_cb_vic = pd.read_pickle(Path(DATA_DIR, f'vp_cb_hit_vic_{lr}.p'))

# meta_ahull = pd.read_pickle(Path(cache_dir, 'meta_ahull_scan.pkl'))
meta_ahull = pd.read_pickle(Path(DATA_DIR, 'meta_VP_CB_ahull_cumsum60.pkl'))
# remove nan in ahull_size
meta_ahull = meta_ahull[meta_ahull['ahull_size'].notna()]
# filter by ratio of area to count
meta_ahull['area_to_count'] = meta_ahull['ahull_size'] / meta_ahull['ahull_kept_points']
meta_ahull = meta_ahull[meta_ahull['area_to_count'] < 5]
print(meta_ahull.shape)

meta_cb_vpn = pd.merge(
    # meta_ahull[['bodyId', 'instance', 'ahull_cumsum_0.6', 'ht']],
    meta_ahull[['bodyId', 'instance', 'ahull_size']],
    vp_cb_vic[['bodyId', 'VIC', 'hitting_time', 'main_groups']],
    on='bodyId', how='inner'
)

meta_cb_vpn.rename(columns={
    'VIC':'vision', 'ahull_size':'area_fit', 'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape)

# %%
# get syn
syn = pd.merge(syn0, 
               meta_cb_vpn[['bodyId', 'instance', 'vision', 'ht', 'area_fit', 'main_groups']], 
               on='bodyId', how='inner')
print(syn.shape)

# %%
# by ht, separate vp and cb
gps = ['OL output', 'nonOL']
top_rois_ls = []
tbar_counts = []
vmin = 5
vmax = 400 # used by this cell and the next

for gp in gps:
    if gp == 'OL output':
        ht_div = [1, 5]
        for i in range(len(ht_div)-1):
            df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
            df_plot['val'] = df_plot['area_fit']
            roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
            top_rois = roi_counts.iloc[:6].index.tolist()
            tbar_counts.append(len(df_plot))

            percentiles = np.nanpercentile(df_plot['val'], [5, 99])
            # if np.isscalar(percentiles):
            #     vmin = vmax = percentiles
            # else:
            #     vmin, vmax = percentiles
            print(percentiles)
            fig, _, _ = plot_extreme_projection_xy(
                df_plot,
                outlines_bkgd,
                xrange = [20000, 80000],
                yrange = [000, 50000],
                xnbins = 120 *2, 
                ynbins = 100 *2, #50000 / 100 *8
                agg = 'smallest',
                agg_frac = 0.1,
                im_norm = 'log',
                vmin=vmin,
                vmax=vmax,
            )
            plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_div[i]}_{ht_div[i+1]}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
            plt.savefig(Path(plot_dir, f'res_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}_{lr}.pdf'), bbox_inches='tight')
            plt.close(fig)
            
    elif gp == 'nonOL':
        ht_div = [2, 3.5, 4, 5]
        for i in range(len(ht_div)-1):
            df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
            df_plot['val'] = df_plot['area_fit']
            roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
            top_rois = roi_counts.iloc[:6].index.tolist()
            top_rois_ls.append(top_rois)
            tbar_counts.append(len(df_plot))
            
            percentiles = np.nanpercentile(df_plot['val'], [5, 99])
            # if np.isscalar(percentiles):
            #     vmin = vmax = percentiles
            # else:
            #     vmin, vmax = percentiles
            print(percentiles)
            fig, _, _ = plot_extreme_projection_xy(
                df_plot,
                outlines_bkgd,
                xrange = [20000, 80000],
                yrange = [000, 50000],
                xnbins = 120*2, 
                ynbins = 100*2, #50000 / 100 *8
                agg = 'smallest',
                agg_frac = 0.1,
                im_norm = 'log',
                vmin=vmin,
                vmax=vmax,
            )
            plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_div[i]}_{ht_div[i+1]}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
            plt.savefig(Path(plot_dir, f'res_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}_{lr}.pdf'), bbox_inches='tight')
            plt.close(fig)

# %%


# %% [markdown]
# ### cumul count vs area_fit for each roi, per layer

# %%
roi_col

# %%
# pick colors for top rois
gps = ['OL output', 'nonOL']
top_rois_ls = []
tbar_counts = []
N = 6 

for gp in gps:
    if gp == 'OL output':
        ht_div = [1, 5]
        for i in range(len(ht_div)-1):
            df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
            tbar_counts.append(len(df_plot))
            # tbar count per roi in this layer 
            tbar_count_roi_se = df_plot.groupby('roi').size().sort_values(ascending=False)
            tbar_count_roi = tbar_count_roi_se.reset_index(name='tbar_count') 
            top_rois = tbar_count_roi_se.iloc[:N]
            top_rois_ls.append(top_rois)
            
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            for idx, roi in enumerate(top_rois.index.tolist()):        
                # Get dob values for this ROI and sort in descending order
                sorted_val = sorted(df_plot[df_plot['roi'] == roi]['area_fit'].dropna().unique(), reverse=False)
                # compute cumulative counts
                syn_roi = df_plot[df_plot['roi'] == roi]['area_fit'].values
                vals = []
                cumulative_counts = []
                for val in sorted_val:
                    count = (syn_roi <= val).sum()
                    vals.append(val)
                    cumulative_counts.append(count)
                
                # Get total tbar count for this roi_base
                tbar_count = tbar_count_roi.loc[tbar_count_roi['roi'] == roi, 'tbar_count'].sum()
                
                # Calculate normalized cumulative counts
                cumulative_counts = np.array(cumulative_counts) / tbar_count
                
                # Plot cumulative distribution with color and label
                # roi_base = roi.replace('(R)', '').replace('(L)', '')  # remove ending '(R)' and '(L)'
                roi_base = roi.replace('(R)', '')  # remove ending '(R)'
                ax.plot(vals, cumulative_counts, 
                        linewidth=2, color=roi_col[roi_col.ROI == roi_base]['rgba'].values[0], 
                        label=f'{roi}_n={tbar_count / 1e3:.0f}k')
                ax.plot(vals[-1], cumulative_counts[-1], 'o', color='black', markersize=2)

            # Formatting
            ax.set_xlabel('Area Fit', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cumulative Fraction', fontsize=12, fontweight='bold')
            ax.set_xlim(0, vmax)  # Reversed x-axis
            ax.set_ylim(0, 1.0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='best', frameon=True)

            plt.tight_layout()
            plt.savefig(Path(plot_dir, f'res_cum_top{N}_roi_{ht_div[i]}_{ht_div[i+1]}.pdf'), transparent=True, bbox_inches='tight')
            # plt.show()
            plt.close()

    elif gp == 'nonOL':
        ht_div = [2, 3.5, 4, 5]
        vmax_ls = [0.1, 0.05, 0.015]
        for i in range(len(ht_div)-1):
            df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
            # tbar count per roi in this layer 
            tbar_count_roi_se = df_plot.groupby('roi').size().sort_values(ascending=False)
            tbar_count_roi = tbar_count_roi_se.reset_index(name='tbar_count') 
            top_rois = tbar_count_roi_se.iloc[:N]
            top_rois_ls.append(top_rois)
            tbar_counts.append(len(df_plot))

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            for idx, roi in enumerate(top_rois.index.tolist()):        
                # Get values for this ROI and sort in descending order
                sorted_val = sorted(df_plot[df_plot['roi'] == roi]['area_fit'].dropna().unique(), reverse=False)
                # compute cumulative counts
                syn_roi = df_plot[df_plot['roi'] == roi]['area_fit'].values
                vals = []
                cumulative_counts = []
                for val in sorted_val:
                    count = (syn_roi <= val).sum()
                    vals.append(val)
                    cumulative_counts.append(count)
                
                # Get total tbar count for this roi_base
                tbar_count = tbar_count_roi.loc[tbar_count_roi['roi'] == roi, 'tbar_count'].sum()
                
                # Calculate normalized cumulative counts
                cumulative_counts = np.array(cumulative_counts) / tbar_count
                
                # Plot cumulative distribution with color and label
                roi_base = roi.replace('(R)', '')  # remove ending '(R)'
                ax.plot(vals, cumulative_counts, 
                        linewidth=2, color=roi_col[roi_col.ROI == roi_base]['rgba'].values[0], 
                        label=f'{roi}_n={tbar_count / 1e3:.0f}k')
                ax.plot(vals[-1], cumulative_counts[-1], 'o', color='black', markersize=2)

            # Formatting
            ax.set_xlabel('Area Fit', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cumulative Fraction', fontsize=12, fontweight='bold')
            ax.set_xlim(0, vmax)  # Reversed x-axis
            ax.set_ylim(0, 1.0)
            ax.spines['top'].set_visible(False) 
            ax.spines['right'].set_visible(False)
            ax.legend(loc='best', frameon=True)

            plt.tight_layout()
            plt.savefig(Path(plot_dir, f'res_cum_top{N}_roi_{ht_div[i]}_{ht_div[i+1]}.pdf'), transparent=True, bbox_inches='tight')
            # plt.show()
            plt.close()
            
# top_rois_ls = list(set([roi for sublist in top_rois_ls for roi in sublist]))
# cmap = plt.get_cmap('tab20')
# roi_color = {roi: cmap(i) for i, roi in enumerate(top_rois_ls)}   

# %%
# top-N accounting for perc of tbars in plot
top_rois_df = pd.DataFrame(top_rois_ls).fillna(0)
top_rois_df.sum(axis=1) / tbar_counts

# %% [markdown]
# ### DEBUG

# %%
syn_r = syn.copy()

# %%
syn_l = syn.copy()

# %%
# DEBUG
ht_bd = 2
# - res
df_plot = syn_l.loc[(syn_l['ht'] >= ht_bd - 0.5) & (syn_l['ht'] < ht_bd + 0.5)].copy()
df_l = df_plot.drop_duplicates(subset=['bodyId']).groupby(['instance']).agg(
    vision=('vision', 'median'),
    ht=('ht', 'median'),
    area_fit=('area_fit', 'median'),
    r2=('r2', 'median'),
    ).reset_index()
df_l['type'] = df_l['instance'].str.replace('_L$', '', regex=True)

df_plot = syn_r.loc[(syn_r['ht'] >= ht_bd - 0.5) & (syn_r['ht'] < ht_bd + 0.5)].copy()
df_r = df_plot.drop_duplicates(subset=['bodyId']).groupby(['instance']).agg(
    vision=('vision', 'median'),
    ht=('ht', 'median'),
    area_fit=('area_fit', 'median'),
    r2=('r2', 'median'),
    ).reset_index()
df_r['type'] = df_r['instance'].str.replace('_R$', '', regex=True)

# merge
df_merge = pd.merge(
    df_l,
    df_r,
    on='type',
    suffixes=('_l', '_r'),
    how='outer'
)
# df_merge['area_fit'] = (df_merge['area_fit_l'] + df_merge['area_fit_r']) /2
# smallest of l and r
df_merge['area_fit'] = df_merge[['area_fit_l', 'area_fit_r']].min(axis=1)
df_merge.sort_values('area_fit', ascending=True).head(20)

# %%
# meta_cb_vpn = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum40_left.csv'), dtype={'bodyId': int})
meta_cb_vpn = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum40.csv'), dtype={'bodyId': int})
thr_vic = 5e-4
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['vision'] > thr_vic]
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['r2'] > 0]

# %%
meta_cb_vpn[meta_cb_vpn.instance.isin(['LPC1_R'])]

# %% [markdown]
# ### vpn clusters

# %%
plot_dir = FIG_DIR / 'quan_propagation' / 'cumsum40_vic1em4'
plot_dir.mkdir(parents=True, exist_ok=True)

meta_cb_vpn = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum40.csv'), dtype={'bodyId': int})
# meta_cb_vpn = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum40_left.csv'), dtype={'bodyId': int})
thr_vic = 1e-4
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['vision'] > thr_vic]
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['r2'] > 0]

# VPN cluster
vpn_clu = pd.read_csv(CACHE_DIR / f'{DATASET}_in_out_clusters.csv')

clu_comp = pd.read_csv(Path(result_dir, 'vpn_cluster_comparison.csv'))

meta_cb_vpn = pd.merge(meta_cb_vpn, clu_comp, on='instance', how='left')
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['cluster'].notna()]

meta_cb_vpn.shape

# %%
# use ht
syn = pd.merge(syn0, 
               meta_cb_vpn[['bodyId', 'vision', 'ht', 'area_fit', 'r2', 'cluster', 'label_8']], 
               on='bodyId', how='inner')
print(syn.shape)

# %%
# by cluster
for clu in meta_cb_vpn.cluster.unique():
    df_plot = syn.loc[syn['cluster'] == clu].copy()
    df_plot['val'] = df_plot['area_fit']
    percentiles = np.nanpercentile(df_plot['val'], [5, 99])
    if np.isscalar(percentiles):
        vmin = vmax = percentiles
    else:
        vmin, vmax = percentiles
    fig, _, _ = plot_extreme_projection_xy(
        df_plot,
        outlines_bkgd,
        xrange = [20000, 80000],
        yrange = [000, 50000],
        agg = 'smallest',
        agg_frac = 0.1,
        im_norm = 'log',
        vmin=1,
        vmax=vmax,
    )
    plt.title(f'area_fit, n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, VIC clu={clu}', fontsize=6, fontweight='bold', pad=20)
    plt.savefig(Path(plot_dir, f'res_vpn_clu_{int(clu)}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# %%
# by cluster
n_clu = ['label_8']
for n_cluster in n_clu:
    clu_list = syn[n_cluster].dropna().unique()
    for clu in clu_list:
        df_plot = syn.loc[ (syn[n_cluster] == clu)].copy()
        df_plot['val'] = df_plot['area_fit']
        percentiles = np.nanpercentile(df_plot['val'], [5, 99])
        if np.isscalar(percentiles):
            vmin = vmax = percentiles
        else:
            vmin, vmax = percentiles
        fig, _, _ = plot_extreme_projection_xy(
            df_plot,
            outlines_bkgd,
            xrange = [20000, 80000],
            yrange = [000, 50000],
            agg = 'smallest',
            agg_frac = 0.1,
            im_norm = 'log',
            vmin=1,
            vmax=vmax,
        )
        plt.title(f'area_fit, n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, collab clu={clu}', fontsize=6, fontweight='bold', pad=20)
        plt.savefig(Path(plot_dir, f'res_vpn_{n_cluster}_clu_{int(clu)}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

# %% [markdown]
# ### cb clusters

# %%
# cluster
cb_clu = pd.read_csv(CACHE_DIR / f'{DATASET}_cb_in_out_clusters.csv', dtype={'bodyId': int}, index_col=0)

meta_cb_vpn = pd.merge(meta_cb_vpn, cb_clu, on='instance', how='left')
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['cluster'].notna()]
meta_cb_vpn['cluster'] = meta_cb_vpn['cluster'].astype(int)

meta_cb_vpn.shape

# %%
# use ht
syn = pd.merge(syn0, 
               meta_cb_vpn[['bodyId', 'vision', 'ht', 'area_fit', 'r2', 'cluster']], 
               on='bodyId', how='inner')
print(syn.shape)

# %%
# by cluster
for clu in meta_cb_vpn.cluster.unique():
    df_plot = syn.loc[syn['cluster'] == clu].copy()
    df_plot['val'] = df_plot['area_fit']
    percentiles = np.nanpercentile(df_plot['val'], [5, 99])
    if np.isscalar(percentiles):
        vmin = vmax = percentiles
    else:
        vmin, vmax = percentiles
    fig, _, _ = plot_extreme_projection_xy(
        df_plot,
        outlines_bkgd,
        xrange = [20000, 80000],
        yrange = [000, 50000],
        agg = 'smallest',
        agg_frac = 0.1,
        im_norm = 'log',
        vmin=1,
        vmax=vmax,
    )
    plt.title(f'area_fit, n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, VIC clu={clu}', fontsize=6, fontweight='bold', pad=20)
    plt.savefig(Path(plot_dir, f'res_CB_clu_{int(clu)}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# %% [markdown]
# ### load prop and meta for plotting rf

# %%
stepsn = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_{SIDE_CHAR}_lat_flow_sum.npz')
stepsn.shape

# %%
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

# %% [markdown]
# ### cont. plot rf individual
# 
# 3 groups, also see following sections "eg, ht > 4" etc

# %%
# # get tab10 color
# cmap = mpl.colormaps.get_cmap('tab10')
# neu_colors = [cmap(i) for i in range(3)]

# pal = ['Oranges', 'Purples', 'Greens', 'Blues']
# cmap = [mpl.colormaps.get_cmap(s) for s in pal]
# # Get the darkest color (highest value in the colormap, typically at position 0.9-1.0)
# neu_colors = [cmap[i](0.9) for i in range(3)] 

neu_colors = [mpl.colormaps.get_cmap('Dark2')(i) for i in range(3)]
#  convert to this format [1, "rgb(0, 20, 200)"]
neu_colors_rgb = []
for color in neu_colors:
    r = int(color[0] * 255)
    g = int(color[1] * 255)
    b = int(color[2] * 255)
    neu_colors_rgb.append(f"rgb({r}, {g}, {b})")

# %%
# one per plot
ids = [87239, 12282, 25289] # ht > 4
# ids = [53437, 49437, 152122] # ht > 3.5 
# ids = [28227, 80342, 17594] # ht > 1
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R'])] 
for i, bid in enumerate(ids):
    outidx = meta.idx[meta.bodyId == bid] 
    df = ci.result_summary(stepsn, inidx, outidx,
                        inidx_map = idx_to_coords, 
                        outidx_map = idx_to_bodyId,
                        display_threshold = 0,
                        display_output= False
                        )
    # set smallest to 0
    # df[df < df.max() * 0.1] = 0
    fig = ci.hex_heatmap(df, global_min=0, 
                         custom_colorscale= [[0, "rgb(255, 255,255)"], [1, neu_colors_rgb[i]]] )
    fig.show()
    # save plotly
    fig.write_image(Path(plot_eg_dir, f'rf_{bid}.pdf'))

# %%
meta_cb_vpn[meta_cb_vpn.bodyId.isin([28227, 80342, 17594])]

# %% [markdown]
# ### eg, ht > 4

# %%
gp = 'nonOL'
ht_div = [1, 3.5, 4, 5]
i = 2
mm = meta_cb_vpn[(meta_cb_vpn['main_groups'] == gp) & (meta_cb_vpn['ht'] > ht_div[i]) & (meta_cb_vpn['ht'] <= ht_div[i+1])].copy()
df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].drop_duplicates(subset=['instance']).copy()

# %%
# SAVE for M.R.
# df_plot.to_csv(Path(plot_eg_dir, f'bodyid_area_ht_{ht_div[i]}_{ht_div[i+1]}.csv'), index=False)
# df_plot.to_csv(Path(plot_eg_dir, f'bodyid_area_vpn.csv'), index=False) # vpn

# %%
# meta.sort_values(['vision'], ascending=False).head(10)
# meta.sort_values(['area_fit'], ascending=True).head(10)

# df_plot.sort_values(['area_fit'], ascending=True).head(10)
df_plot[df_plot.roi == 'EB']

# %%
inst = 'LHPD3a2_a_R'; bodyId = 87239 # good
inst = 'VES023_L'; bodyId = 25289 # good
# inst = 'CB4087_R' # low wt
# inst = 'LAL304m_R'; bodyId = 11924
inst = 'AVLP714m_R'; bodyId = 12282
# gap in the ring -> 'ER4d_R' ER3m_R 

# %%
meta_cb_vpn[meta_cb_vpn.bodyId.isin([87239, 12282, 25289])]

# %%
n_mesh = neu.fetch_mesh_neuron([87239, 12282, 25289])
# n_ske = neu.fetch_skeletons([87239, 12282, 25289])

fig = plot_neuron_with_outlines(n_mesh, outlines_bkgd, neuron_color=neu_colors)
fig.show()
# SAVE
fig.write_image(Path(plot_eg_dir, f'ht_{i}.png'), scale=10)

# %% [markdown]
# ### eg, ht > 3.5

# %%
gp = 'nonOL'
ht_div = [1, 3.5, 4, 5]
i = 1
mm = meta_cb_vpn[(meta_cb_vpn['main_groups'] == gp) & (meta_cb_vpn['ht'] > ht_div[i]) & (meta_cb_vpn['ht'] <= ht_div[i+1])].copy()
df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].drop_duplicates(subset=['bodyId','instance']).copy()

# %%
df_plot.sort_values('area_fit').head(20)

# %%
# inst = 'ER2_a_R'; bodyId = 20664
inst = 'PLP156_L'; bodyId = 53437  # old 'CB2896_L'
# inst = 'PLP156_L'; bodyId = 102049
inst = 'CB1897_R'; bodyId = 49437
inst = 'PS335_L'; bodyId = 152122

n_mesh = neu.fetch_mesh_neuron(bodyId)
# n_ske = neu.fetch_skeletons(bodyId)

# %%
n_mesh = neu.fetch_mesh_neuron([49437])
fig = plot_neuron_with_outlines(n_mesh, outlines_bkgd)
fig.show()

# %%
n_mesh = neu.fetch_mesh_neuron([53437, 49437, 152122])

fig = plot_neuron_with_outlines(n_mesh, outlines_bkgd, neuron_color=neu_colors)
fig.show()
# SAVE
# fig.write_image(Path(plot_eg_dir, f'ht_{i}.png'), scale=10)

# %% [markdown]
# ### eg, ht > 1

# %%
gp = 'nonOL'
ht_div = [1, 3.5, 4, 5]
i = 0
# mm = meta_cb_vpn[(meta_cb_vpn['main_groups'] == gp) & (meta_cb_vpn['ht'] > ht_div[i]) & (meta_cb_vpn['ht'] <= ht_div[i+1])].copy()
df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].drop_duplicates(subset=['bodyId','instance']).copy()

# %%
df_plot.sort_values('area_fit').head(30)

# %%
mm.sort_values(['vision'], ascending=False).head(30)

# %%
inst = 'AOTU002_a_R'; bodyId = 28227
inst = 'TuBu09_SBU_R'; bodyId = 80342
inst = 'SLP075_R'; bodyId = 17594
# inst = 'PLP158_R'; bodyId = 43903

# %%
n_mesh = neu.fetch_mesh_neuron([28227, 80342, 17594])

fig = plot_neuron_with_outlines(n_mesh, outlines_bkgd, neuron_color=neu_colors)
fig.show()
# SAVE
fig.write_image(Path(plot_eg_dir, f'ht_{i}.png'), scale=10)
fig.close()

# %% [markdown]
# ### eg, vpn

# %%
gp = 'OL output'
ht_div = [1, 5]
i = 0
meta = meta_cb_vpn[(meta_cb_vpn['main_groups'] == gp) & (meta_cb_vpn['ht'] > ht_div[i]) & (meta_cb_vpn['ht'] <= ht_div[i+1])].copy()
df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].drop_duplicates(subset=['bodyId','instance']).copy()

# %%
df_plot.groupby('instance').agg(
    most_numerous_roi=('roi', lambda x: x.value_counts().index[0]),
    area_fit_median=('area_fit', 'median'),
    vision_median=('vision', 'median'),
).reset_index().sort_values('area_fit_median').head(50)

# %%
meta.sort_values(['area_fit'], ascending=True).drop_duplicates(subset=['instance']).head(30)

# %%
# inst = 'MeTu3c_R'; bodyId = 61351
# inst = 'LC12_R'; bodyId = 54497
inst = 'MeVP11_R', bodyId = 59191
# inst = 'LPLC4_R'; bodyId = 20175
inst = 'LoVP18_R'; bodyId = 19233

# %%
outlines_bkgd_olr = get_roi_outline(['ME(R)'], 0.0002)

# %%
n_mesh = neu.fetch_mesh_neuron([61351, 59191, 19233])

# get tab10 color
cmap = mpl.colormaps.get_cmap('tab10')
neu_colors = [cmap(i) for i in range(3)]

fig = plot_neuron_with_outlines(n_mesh, outlines_bkgd, neuron_color=neu_colors)
fig.show()
# SAVE
# fig.write_image(Path(plot_eg_dir, f'ht_{i}.png'), scale=1)

# %%


# %% [markdown]
# ## retinotopy

# %%
## ## from Judith, 
lr = '' if SIDE == 'right' else '_left'

vp_cb_vic = pd.read_pickle(CACHE_DIR / f'vp_cb{lr}_vic_w_hit_df.p')
fit_rf = pd.read_pickle(CACHE_DIR / f'{DATASET}{lr}_w_hit_fit_rf.p')

meta_cb_vpn = pd.merge(
    fit_rf[['bodyId','instance','size','r2', 'hitting_time', 'main_groups']],
    vp_cb_vic[['bodyId','VIC']],
    on='bodyId', how='left'
)
meta_cb_vpn.rename(columns={'VIC':'vision', 'size':'area_fit', 'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape)
meta_cb_vpn.shape

# %% [markdown]
# ### RI for numerous types

# %%
retino_metric = pd.read_csv(Path(result_dir, 'retino_metric_vpn.csv'))
# retino_metric = pd.read_csv(Path(result_dir, 'retino_metric_cb.csv'))

# %%
# use ht
syn = pd.merge(syn0, 
               meta_cb_vpn[['bodyId', 'instance']], 
               on='bodyId', how='inner')
syn = pd.merge(syn, 
               retino_metric, 
               on=['instance','roi'], how='inner')
print(syn.shape)

# %%
df_plot = syn.copy()
df_plot['val'] = df_plot['RI'].abs()
percentiles = np.nanpercentile(df_plot['val'], [5, 99])
if np.isscalar(percentiles):
    vmin = vmax = percentiles
else:
    vmin, vmax = percentiles
print(vmin, vmax)
fig, _, _ = plot_extreme_projection_xy(
    df_plot,
    outlines_bkgd,
    xrange = [20000, 80000],
    yrange = [000, 50000],
    agg = 'largest',
    agg_frac = 0.1,
    # im_norm = 'log',
    vmin=0,
    vmax=0.7,
)
plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
plt.savefig(Path(plot_dir, f'ri_vpn{lr}.pdf'), bbox_inches='tight')
# plt.savefig(Path(plot_dir, f'ri_cb{lr}.pdf'), bbox_inches='tight')
plt.close(fig)

# %% [markdown]
# ### eg

# %%
retino_metric['RI'] = retino_metric['RI'].astype(float).round(2)
retino_metric['TP'] = retino_metric['TP'].astype(float).round(2)
retino_metric['abs_RI'] = retino_metric['RI'].abs()

# %%
syn = pd.merge(syn0, 
               meta_cb_vpn[['bodyId', 'instance']], 
               on='bodyId', how='inner')
print(syn.shape)

# %%
syn[syn.roi == 'BU(R)'].groupby('instance').agg(
    tbar_count = ('bodyId', 'count'),
    id_count = ('bodyId', 'nunique')
).reset_index().sort_values('id_count', ascending=False).head(20)

# %%
mm = meta_cb_vpn[(meta_cb_vpn['main_groups'] == gp) & (meta_cb_vpn['ht'] > ht_div[i]) & (meta_cb_vpn['ht'] <= ht_div[i+1])].copy()
df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].drop_duplicates(subset=['instance']).copy()

# %%
# meta.sort_values(['vision'], ascending=False).head(10)
# meta.sort_values(['area_fit'], ascending=True).head(10)

# df_plot.sort_values(['area_fit'], ascending=True).head(10)
df_plot[df_plot.roi == 'EB']

# %%
CB1464_R, SAD200m_R, AMMC002_R, SMP461_R, PS335_L, PVLP203m_R (good), CB1705_R, PS095_L, LAL061_R (interesting rf), DNg02_a_L, WED096_R,

weak rf but good: PLP037_L, DNge114_L

CX cells: TuBu05_SBU_R, TuBu08_SBU_R, TuBu10_SBU_R,  ER3d_a_R, ER2_a_R, ER3w_a_R, ER3a_a_R, ER4d_R, ER3m_R, TuBu09_SBU_R, ER2_c_R

mid-strip, not good tho:  SCL001m_R, AVLP709m_R

# %%
bb = meta_cb_vpn[meta_cb_vpn.instance.isin([
    'CB1464_R', 'SAD200m_R', 'AMMC002_R', 'SMP461_R', 'PS335_L', 'PVLP203m_R', 'CB1705_R', 'PS095_L', 'LAL061_R', 'DNg02_a_L', 'WED096_R', 
    'PLP037_L', 'DNge114_L', 'TuBu05_SBU_R', 'TuBu08_SBU_R', 'TuBu10_SBU_R',  'ER3d_a_R', 'ER2_a_R', 'ER3w_a_R', 'ER3a_a_R', 'ER4d_R', 'ER3m_R', 
    'TuBu09_SBU_R', 'ER2_c_R', 'SCL001m_R', 'AVLP709m_R'
])]

# %%
n_mesh = neu.fetch_mesh_neuron([87239, 12282, 25289])
# n_ske = neu.fetch_skeletons([87239, 12282, 25289])

fig = plot_neuron_with_outlines(n_mesh, outlines_bkgd, neuron_color=neu_colors)
fig.show()
# SAVE
fig.write_image(Path(plot_eg_dir, f'ht_{i}.png'), scale=10)

# %% [markdown]
# # extre value proj, binocular

# %%
plot_dir = FIG_DIR / 'quan_propagation' / 'v1'
plot_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### outlines

# %%
# load outline
outline_1 = pd.read_csv(Path(result_dir, f'outline_cb_1.csv'), index_col=False)
outline_2 = pd.read_csv(Path(result_dir, f'outline_cb_2.csv'), index_col=False)
outlines_bkgd = [outline_1,  outline_2]

# %% [markdown]
# ## load data

# %%
# load vic and ht
thr_vic = 5e-4

vp_cb_r = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_r.p'))
meta_cb_vpn = vp_cb_r.copy()
inst = meta_cb_vpn.groupby('instance').agg({'VIC':'median'}).reset_index()
inst = inst[inst.VIC > thr_vic]['instance'].unique()
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.instance.isin(inst)]
meta_cb_vpn.rename(columns={'VIC':'vision', 'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape)
meta_cb_vpn_r = meta_cb_vpn.copy()

vp_cb_l = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_l.p'))
meta_cb_vpn = vp_cb_l.copy()
inst = meta_cb_vpn.groupby('instance').agg({'VIC':'median'}).reset_index()
inst = inst[inst.VIC > thr_vic]['instance'].unique()
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.instance.isin(inst)]
meta_cb_vpn.rename(columns={'VIC':'vision', 'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape)
meta_cb_vpn_l = meta_cb_vpn.copy()

# %% [markdown]
# ## combine left and right

# %%
meta_cb_vpn = pd.merge(
    meta_cb_vpn_r[['bodyId', 'instance', 'main_groups', 'ht', 'vision']], 
    meta_cb_vpn_l[['bodyId', 'instance', 'main_groups','ht', 'vision']], 
    on=['bodyId', 'instance'], how='outer', suffixes=('_r', '_l'))
meta_cb_vpn['cell_type'] = meta_cb_vpn['instance'].str.replace('_L$', '', regex=True).str.replace('_R$', '', regex=True)
print(meta_cb_vpn.shape)

# only CB
meta_cb_vpn = meta_cb_vpn[(meta_cb_vpn['main_groups_r'] == 'nonOL')
                        | (meta_cb_vpn['main_groups_l'] == 'nonOL')]
print(meta_cb_vpn.shape)

meta_inst = meta_cb_vpn.groupby(['instance'])[['vision_r','vision_l', 'ht_l', 'ht_r']].median().reset_index()
print(meta_inst.shape)

# %% [markdown]
# ## dob and VIC

# %%
# sum of vic
meta_cb_vpn['vision'] = meta_cb_vpn[['vision_r', 'vision_l']].sum(axis=1, skipna=True)
# dob, min / max
meta_cb_vpn['dob'] = meta_cb_vpn[['vision_l', 'vision_r']].min(axis=1, skipna=False) / meta_cb_vpn[['vision_l', 'vision_r']].max(axis=1,skipna=False)

# fill NaN in dob with 0
meta_cb_vpn['dob'] = meta_cb_vpn['dob'].fillna(0)

# ht as the earliest of the two sides, i.e. min
meta_cb_vpn['ht'] = meta_cb_vpn[['ht_l', 'ht_r']].min(axis=1, skipna=True)

# main_groups as the one that's not NaN
meta_cb_vpn['main_groups'] = meta_cb_vpn['main_groups_r'].combine_first(meta_cb_vpn['main_groups_l'])

# %%
syn0 = pd.read_pickle(Path(cache_dir, 'tbar_vp_cb.pkl'))

syn = pd.merge(syn0, 
               meta_cb_vpn[['instance', 'bodyId', 'vision', 'ht', 'dob', 'main_groups']], 
               on='bodyId', how='inner')

# %%
# DEBUG
# meta_cb_vpn[meta_cb_vpn.dob > 0.5].shape, meta_cb_vpn.shape
# meta_cb_vpn[~meta_cb_vpn.bodyId.isin(syn.bodyId.unique())]
# meta_cb_vpn[(meta_cb_vpn.main_groups == 'nonOL') & (meta_cb_vpn.ht > 5)]

# %%
# SAVE for M.R,
# meta_cb_vpn[meta_cb_vpn.dob > 0.5].to_csv(Path(plot_dir, f'bodyid_dob_larger_than_05.csv'), index=False)

# %%
# DOB
gp = 'nonOL'
ht_div = [2, 3.5, 4, 5]
top_rois_ls = []
for i in range(len(ht_div)-1):
    df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
    df_plot['val'] = df_plot['dob']
    roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
    top_rois = roi_counts#.iloc[:20]
    top_rois_ls.append(top_rois)
    fig, _, _ = plot_extreme_projection_xy(
        df_plot,
        outlines_bkgd,
        xrange = [20000, 80000],
        yrange = [000, 50000],
        xnbins = 120 *2, 
        ynbins = 100 *2,
        agg = 'largest',
        cmap_div = True,
        im_norm = 'linear',
        vmin = 0,
        vmax = 1,
        agg_frac = 0.1,
    )
    plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_div[i]}_{ht_div[i+1]}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
    plt.savefig(Path(plot_dir, f'dob_union_max_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}.pdf'), bbox_inches='tight')
    plt.close(fig)

# %%
# # VIC
# gp = 'nonOL'
# ht_div = [1, 3.5, 4, 5]
# top_rois_ls = []
# for i in range(len(ht_div)-1):
#     df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
#     df_plot['val'] = df_plot['vision']
#     roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
#     top_rois = roi_counts.iloc[:6]
#     top_rois_ls.append(top_rois)
#     percentiles = np.nanpercentile(df_plot['val'], [5, 99])
#     if np.isscalar(percentiles):
#         vmin = vmax = percentiles
#     else:
#         vmin, vmax = percentiles
#     print(vmin, vmax)
#     fig, _, _ = plot_extreme_projection_xy(
#         df_plot,
#         outlines_bkgd,
#         xrange = [20000, 80000],
#         yrange = [000, 50000],
#         xnbins = 120 *2, 
#         ynbins = 100 *2,
#         agg = 'largest',
#         im_norm = 'linear',
#         vmin = 0,
#         vmax = vmax,
#         agg_frac = 0.1,
#     )
#     plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_div[i]}_{ht_div[i+1]}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
#     plt.savefig(Path(plot_dir, f'vic_union_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}.pdf'), bbox_inches='tight')
#     plt.close(fig)

# %% [markdown]
# ### eg

# %%
gp = 'nonOL'
ht_div = [1, 3.5, 4, 5]
i = 0
mm = meta_cb_vpn[(meta_cb_vpn['main_groups'] == gp) & (meta_cb_vpn['ht'] > ht_div[i]) & (meta_cb_vpn['ht'] <= ht_div[i+1])].copy()
df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].drop_duplicates(subset=['instance']).copy()

# %%
mm.shape

# %%
aa = mm[(mm.dob > 0.8) & (mm.vision > 0.01)].sort_values('instance')

# %%
# layer 2-3.5
AOTU046, 517788, 12855
CB1805, 74616, 52306
CB1836, 129981, 549558
CB3171, 91635, 38888
CB4069, 48784, 49325
CB4097, 51911, 51911
CL225,103805, 542876
CL355, 173445, 165063
DNp41, 15502, 15028
ER5, 521535, 521535
PS077, 62156, 153412
PS282, 46200, 36929
PS324, 25800, 19402
PS341, 58706, 40102
PS357, 139432, 121954
SLP322, 219774, 61388
# KCg-d, 111197, 51679

# layer 3.5-4
CB1547, 27298, 38233
CB1834, 37526, 48220
CB2037, 53906, 36547
CB2084, 45284, 31398
CB3010, 38297, 63416
CB3044, 531063, 61226
CB3074, 64709, 516883
CB4169, 61560, 29481
DNb03, 16829, 513296
DNpe012_a, 29934, 32166
LAL094, 151629, 72181
PS141, 23153, 21058
PVLP046, 17939, 20434
# CB2094, 68199, 24243
# GNG638, 12115, 34725
# PVLP005, 521734, 145878

# layer 4-5
AMMC002, [543768, 87162]
DNp72, 24752, 21031
PS081, 70779, 16254
WED096, 36479, 109171
WED132, 32268, 26525
# CB1544, 25970, 25255
# DNbe005, 12133, 11916

# %%
meta_cb_vpn[meta_cb_vpn.bodyId.isin([87162])]

# %%
neu_colors = [mpl.colormaps.get_cmap('Dark2')(i) for i in range(10)]

# neu_colors = [mpl.colormaps.get_cmap('tab20')(i) for i in range(20)]

# %%
# n_mesh = neu.fetch_mesh_neuron([219774, 60990, 151629, 72181, 543768, 87162])
n_mesh = neu.fetch_mesh_neuron([219774, 60990, 72181,  87162])
# neu_colors = [mpl.colormaps.get_cmap('Dark2')(i) for i in range(4)]

fig = plot_neuron_with_outlines(n_mesh, outlines_bkgd, neuron_color=neu_colors[:len(n_mesh)])
fig.show()
# SAVE
# fig.write_image(Path(plot_eg_dir, 'dob_eg.png'), scale=10)

# %% [markdown]
# ### cont. cumul count vs DOB for each roi, per layer

# %%
# per layer
gp = 'nonOL'
ht_div = [2, 3.5, 4, 5]
N = 6 # top N rois
perc_shown = []
# # Get colormap
# cmap = plt.get_cmap('tab10')
# # colors = [cmap(i) for i in range(N)]
# # assign each roi in top_rois_ls a color
# roi_base_color = {roi: cmap(i) for i, roi in enumerate(top_rois_ls)}   

for i in range(len(ht_div)-1):
    df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
    
    # choosetop roi and roi_base by tbar (dob>0.5) count
    top_rois_df = df_plot[df_plot.dob > 0.5].groupby(['roi']).size().reset_index(name='count').sort_values(by='count', ascending=False) 
    tbar_roi_df = df_plot.groupby(['roi']).size().reset_index(name='count')
    
    top_rois_df['roi_base'] = top_rois_df['roi'].apply(lambda roi: roi.replace('(R)', '').replace('(L)', ''))
    tbar_roi_df['roi_base'] = tbar_roi_df['roi'].apply(lambda roi: roi.replace('(R)', '').replace('(L)', ''))
    # top_rois_df['roi_base'] = top_rois_df['roi'].apply(lambda roi: roi.replace('(R)', ''))
    # tbar_roi_df['roi_base'] = tbar_roi_df['roi'].apply(lambda roi: roi.replace('(R)', ''))
    # top N ROIs by count
    top_n_rois = top_rois_df.groupby('roi_base')['count'].sum().sort_values(ascending=False).index[:N]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    count_sum = []
    for idx, roi_base in enumerate(top_n_rois):
        roi = tbar_roi_df[tbar_roi_df['roi_base'] == roi_base]['roi'].values
        
        # Get dob values for this ROI and sort in descending order
        sorted_dob = sorted(df_plot[df_plot['roi'].isin(roi)]['dob'].dropna().unique(), reverse=True)
        if sorted_dob[-1] == 0:
            sorted_dob = sorted_dob[:-1]

        # compute cumulative counts
        syn_roi = df_plot[df_plot['roi'].isin(roi)]['dob']
        count_sum.append(len(syn_roi))
        dob_values = []
        cumulative_counts = []
        for val in sorted_dob:
            count = (syn_roi >= val).sum()
            dob_values.append(val)
            cumulative_counts.append(count)
        # count_sum.append(cumulative_counts[-1])

        # Get total tbar count for this roi_base
        tbar_count = tbar_roi_df.loc[tbar_roi_df['roi'].isin(roi), 'count'].sum()
        
        # Calculate normalized cumulative counts
        cumulative_counts = np.array(cumulative_counts) / tbar_count
        
        # Plot cumulative distribution with color and label
        ax.plot(dob_values, cumulative_counts, 
                linewidth=2, color=roi_col[roi_col.ROI == roi_base]['rgba'].values[0], 
                label=f'{roi_base}_n={tbar_count / 1e3:.0f}k')
        ax.plot(dob_values[-1], cumulative_counts[-1], 'o', color='black', markersize=2)

    # print(count_sum)
    perc_shown.append(np.sum(count_sum) /tbar_roi_df['count'].sum() )

    # Formatting
    ax.set_xlabel('DOB', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Fraction', fontsize=12, fontweight='bold')
    ax.set_xlim(1, 0)  # Reversed x-axis
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True)

    plt.tight_layout()
    plt.savefig(Path(plot_dir, f'dob_cum_top{N}_roi_{ht_div[i]}_{ht_div[i+1]}.pdf'), transparent=True, bbox_inches='tight')
    # plt.show()
    plt.close()



# %%
perc_shown

# %% [markdown]
# ## obs, DOB vs 

# %%
# sum of vic
meta_inst['vision'] = meta_inst[['vision_r', 'vision_l']].sum(axis=1, skipna=True)
meta_inst['vision_max'] = meta_inst[['vision_l', 'vision_r']].max(axis=1,skipna=True)
meta_inst['vision_min'] = meta_inst[['vision_l', 'vision_r']].min(axis=1, skipna=False)
# dob, min / max
meta_inst['dob'] = meta_inst[['vision_l', 'vision_r']].min(axis=1, skipna=False) / meta_inst[['vision_l', 'vision_r']].max(axis=1,skipna=False)

# fill NaN in dob with 0
meta_inst['dob'] = meta_inst['dob'].fillna(0)

# ht as the earliest
meta_inst['ht'] = meta_inst[['ht_l', 'ht_r']].min(axis=1, skipna=True)

# %%
meta_inst

# %%
# plot dob vs vic
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sc = ax.scatter(
    meta_inst.dropna(subset=['vision_min'])['vision_min'],
    meta_inst.dropna(subset=['vision_min'])['vision_max'],
    s=5,
    c='black',
    alpha=0.5,
    edgecolors='none'
)
# ax.set_xscale('log')
# set asp = 1
ax.set_aspect('equal', adjustable='box')
# add y = x
ax.plot([meta_inst['vision_min'].min(), meta_inst['vision_max'].max()], 
        [meta_inst['vision_min'].min(), meta_inst['vision_max'].max()], 
        color='red', linestyle='--', linewidth=1)
plt.tight_layout()
# plt.savefig(Path(plot_dir, f'dob_vs_vic_union_cb.pdf'), bbox_inches='tight')
plt.show()

# %% [markdown]
# ## obs. by instance, density plot, left vs right vic

# %%
# sum of vic
meta_inst['vision'] = meta_inst[['vision_r', 'vision_l']].sum(axis=1, skipna=True)
# dob, min / max
meta_inst['dob'] = meta_inst[['vision_l', 'vision_r']].min(axis=1, skipna=False) / meta_inst[['vision_l', 'vision_r']].max(axis=1,skipna=False)

# set NaN to  for plotting
meta_inst.fillna({'vision_r': 3e-4, 'vision_l': 3e-4}, inplace=True)

# %%
# scatter plot
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# Separate data with and without NaN
mask_valid = meta_inst['dob'].notna()
data_valid = meta_inst[mask_valid]
data_nan = meta_inst[~mask_valid]

# Plot NaN values as black
if len(data_nan) > 0:
    ax.scatter(
        data_nan['vision_l'], 
        data_nan['vision_r'], 
        c='black',
        s=5, 
        alpha=0.7, 
        edgecolors='k',
        label='NaN'
    )
# Plot valid values with colormap
if len(data_valid) > 0:
    sc = ax.scatter(
        data_valid['vision_l'], 
        data_valid['vision_r'], 
        c=data_valid['dob'], 
        cmap='viridis', 
        norm=plt.Normalize(vmin=0, vmax=1), 
        s=10, 
        alpha=0.7, 
        edgecolors='k'
    )
    plt.colorbar(sc, label='DOB (min/max)')
ax.set_xscale('log')
ax.set_yscale('log')
# asp =1 
ax.set_aspect('equal', adjustable='box')
# plt.plot([1e-4, 1e2], [1e-4, 1e2], 'r--', linewidth=1)
fig.show()

# %%
# Histogram of dob, dropping NaN
fig, ax = plt.subplots(figsize=(6, 4))

# Drop NaN values
dob_valid = meta_inst['dob'].dropna()
# Create histogram
ax.hist(dob_valid, bins=np.arange(0, 1.05, 0.05), alpha=0.7, edgecolor='black', facecolor='#64a0d1')

# Formatting
ax.set_xlabel('Degree of Binocularity (DOB)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, axis='y')

# Add text with count
ax.text(0.98, 0.98, f'n_inst = {len(dob_valid):,}', 
        fontsize=10, fontweight='bold', 
        ha='right', va='top', 
        transform=ax.transAxes)

plt.tight_layout()
# plt.savefig(Path(plot_dir, f'dob_histogram.pdf'), bbox_inches='tight')
plt.show()

# %%
# 2D contour plot
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# Separate data with and without NaN
mask_valid = meta_inst['dob'].notna()
data_valid = meta_inst[mask_valid]
data_nan = meta_inst[~mask_valid]

# Plot NaN values as black
if len(data_nan) > 0:
    ax.scatter(
        data_nan['vision_l'], 
        data_nan['vision_r'], 
        c='black',
        s=5, 
        alpha=0.7, 
        edgecolors='k',
        label='NaN'
    )

# Plot valid values with contour
if len(data_valid) > 0:
    # Get log-scaled data
    x = np.log10(data_valid['vision_l'])
    y = np.log10(data_valid['vision_r'])
    
    # Calculate kernel density estimate
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=0.1)
    
    # Create grid for contour plot
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    
    # Plot contours
    contour = ax.contour(10**Xi, 10**Yi, zi, levels=8, cmap='viridis', linewidths=1.5)
    ax.contourf(10**Xi, 10**Yi, zi, levels=8, cmap='viridis', alpha=0.6)
    contourf = ax.contourf(10**Xi, 10**Yi, zi, levels=8, cmap='viridis', alpha=0.6)
    plt.colorbar(contourf, label='Density')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Vision Left', fontsize=12, fontweight='bold')
ax.set_ylabel('Vision Right', fontsize=12, fontweight='bold')
if len(data_nan) > 0:
    ax.legend()
fig.show()

# %%
# 2D histogram - Vision Left vs Vision Right, separating NaN dob
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# Separate data based on dob
mask_valid_dob = meta_inst['dob'].notna()
data_with_dob = meta_inst[mask_valid_dob]

h1 = ax.hist2d(
    data_with_dob['vision_l'], 
    data_with_dob['vision_r'],
    bins=[np.logspace(-4, 0, 25), np.logspace(-4, 0, 25)],
    cmap='viridis',
    # norm=mpl.colors.LogNorm()
)
plt.colorbar(h1[3], ax=ax, label='Count')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Vision Left', fontsize=12, fontweight='bold')
ax.set_ylabel('Vision Right', fontsize=12, fontweight='bold')
ax.set_title(f'Binocular (n={len(data_with_dob)})', fontsize=14, fontweight='bold')
ax.plot([1e-4, 1], [1e-4, 1], 'r--', linewidth=2, label='Unity')
ax.legend()

plt.suptitle('2D Histogram: Vision Left vs Right', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
# plt.savefig(Path(plot_dir, 'vision_l_vs_r_hist2d_separated.pdf'), bbox_inches='tight')
plt.show()

# %%
# linear scale, 2D histogram - Vision Left vs Vision Right, separating NaN dob
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Separate data based on dob
mask_valid_dob = meta_inst['dob'].notna()
data_with_dob = meta_inst[mask_valid_dob]
data_without_dob = meta_inst[~mask_valid_dob]

# Plot 1: Data WITH dob (binocular)
if len(data_with_dob) > 0:
    h1 = axes[0].hist2d(
        data_with_dob['vision_l'], 
        data_with_dob['vision_r'],
        bins=[np.linspace(0, 0.05, 50), np.linspace(0, 0.05, 50)],
        cmap='viridis'
    )
    plt.colorbar(h1[3], ax=axes[0], label='Count')
    # axes[0].set_xscale('log')
    # axes[0].set_yscale('log')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlabel('Vision Left', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Vision Right', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Binocular (n={len(data_with_dob)})', fontsize=14, fontweight='bold')
    axes[0].plot([1e-4, 1], [1e-4, 1], 'r--', linewidth=2, label='Unity')
    axes[0].legend()
plt.suptitle('2D Histogram: Vision Left vs Right', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
# plt.savefig(Path(plot_dir, 'vision_l_vs_r_hist2d_separated.pdf'), bbox_inches='tight')
plt.show()

# %%
# 2D contour plot showing density
from scipy.stats import gaussian_kde

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# Get all data points (ignoring dob)
x = np.log10(meta_inst['vision_l'])
y = np.log10(meta_inst['vision_r'])

# Calculate kernel density estimate
xy = np.vstack([x, y])
kde = gaussian_kde(xy)

# Create grid for contour plot
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
Xi, Yi = np.meshgrid(xi, yi)
zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

# Plot contours
ax.contour(10**Xi, 10**Yi, zi, levels=8, cmap='viridis', linewidths=1.5)
contourf = ax.contourf(10**Xi, 10**Yi, zi, levels=8, cmap='viridis', alpha=0.6)
plt.colorbar(contourf, label='Density')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Vision Left', fontsize=12, fontweight='bold')
ax.set_ylabel('Vision Right', fontsize=12, fontweight='bold')
fig.show()

# %% [markdown]
# # obs. 2D extreme values proj, binocular, vic thresholding

# %% [markdown]
# ### outlines

# %%
# load outline
outline_1 = pd.read_csv(Path(result_dir, f'outline_cb_1.csv'), index_col=False)
outline_2 = pd.read_csv(Path(result_dir, f'outline_cb_2.csv'), index_col=False)
outlines_bkgd = [outline_1,  outline_2]

# %% [markdown]
# ## load data

# %%
plot_dir = FIG_DIR / 'quan_propagation' / 'judy_1101_bi'
plot_dir.mkdir(parents=True, exist_ok=True)

# %%
# meta_cb_vpn_r = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum40.csv'), dtype={'bodyId': int})
# meta_cb_vpn_l = pd.read_csv(Path(cache_dir, 'meta_cb_vpn_rfsize_csum40_left.csv'), dtype={'bodyId': int})

# thr_vic = 1e-4

# meta_cb_vpn_r = meta_cb_vpn_r[meta_cb_vpn_r['vision'] > thr_vic]
# meta_cb_vpn_l = meta_cb_vpn_l[meta_cb_vpn_l['vision'] > thr_vic]

# %%
# from Judith, for VIC
thr_vic = 5e-4

vp_cb_vic_r = pd.read_pickle(CACHE_DIR / 'vp_cb_vic_w_hit_df.p')
meta_cb_vpn_r = vp_cb_vic_r[vp_cb_vic_r['VIC'] > thr_vic].copy()
meta_cb_vpn_r.rename(columns={'VIC':'vision', 'size':'area_fit', 'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn_r.shape)

vp_cb_vic_l = pd.read_pickle(CACHE_DIR / 'vp_cb_left_vic_w_hit_df.p')
meta_cb_vpn_l = vp_cb_vic_l[vp_cb_vic_l['VIC'] > thr_vic].copy()
meta_cb_vpn_l.rename(columns={'VIC':'vision', 'size':'area_fit', 'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn_l.shape)

# %%

meta_cb_vpn = pd.merge(
    meta_cb_vpn_r[['bodyId', 'instance', 'main_groups', 'ht', 'vision']], 
    meta_cb_vpn_l[['bodyId', 'instance', 'ht', 'vision']], 
    on=['bodyId', 'instance'], how='inner', suffixes=('_r', '_l'))
meta_cb_vpn['cell_type'] = meta_cb_vpn['instance'].str.replace('_L$', '', regex=True).str.replace('_R$', '', regex=True)
meta_cb_vpn.shape

# %%
# tbars
syn0 = pd.read_pickle(Path(cache_dir, 'tbar_vp_cb.pkl'))

# %% [markdown]
# ### DEBUG left vs right

# %%
# for rf size
meta_type_r = meta_cb_vpn_r.groupby('instance').agg(
    bodyId_count = ('bodyId', 'count'),
    area_fit = ('area_fit', 'median'),
    r2 = ('r2', 'median'),
    vision = ('vision', 'median'),
    ht = ('ht', 'median'),  
    main_groups = ('main_groups', 'first')
).reset_index()

meta_type_l = meta_cb_vpn_l.groupby('instance').agg(
    bodyId_count = ('bodyId', 'count'),
    area_fit = ('area_fit', 'median'),
    r2 = ('r2', 'median'),
    vision = ('vision', 'median'),
    ht = ('ht', 'median'),  
    main_groups = ('main_groups', 'first')
).reset_index()

meta_type_r.shape, meta_type_l.shape

# %%
# for VIC
meta_type_r = vp_cb_vic_r.groupby('instance').agg(
    bodyId_count = ('bodyId', 'count'),
    VIC = ('VIC', 'median'),
    hitting_time = ('hitting_time', 'median'),  
    main_groups = ('main_groups', 'first')
).reset_index()
meta_type_r.rename(columns={'VIC':'vision', 'hitting_time':'ht'}, inplace=True)

meta_type_l = vp_cb_vic_l.groupby('instance').agg(
    bodyId_count = ('bodyId', 'count'),
    VIC = ('VIC', 'median'),
    hitting_time = ('hitting_time', 'median'),  
    main_groups = ('main_groups', 'first')
).reset_index()
meta_type_l.rename(columns={'VIC':'vision', 'hitting_time':'ht'}, inplace=True)

meta_type_r.shape, meta_type_l.shape

# %%
aa = pd.merge(
    meta_type_r,
    meta_type_l,
    on='instance',
    suffixes=('_r', '_l'),
    how='outer'
).sort_values(['instance'])

# %%
aa = pd.merge(
    meta_type_r[(4.5 < meta_type_r.ht) & (meta_type_r.ht < 5.6) & (meta_type_r.vision > 0.002)],
    meta_type_l[(4.5 < meta_type_l.ht) & (meta_type_l.ht < 5.6) & (meta_type_l.vision > 0.002)],
    on='instance',    suffixes=('_r', '_l'), how='outer').sort_values(['instance'])

# %%
vp_cb_vic_r[vp_cb_vic_r.instance.isin(['ER4d_L'])]

# %%
vp_cb_vic_l[vp_cb_vic_l.instance.isin(['ER3p_a_R'])]

# %%
fit_rf_r[fit_rf_r.instance.isin(['ER3p_a_L'])]

# %%
fit_rf_l[fit_rf_l.instance.isin(['ER3p_a_R'])]

# %%
meta_cb_vpn_r[meta_cb_vpn_r.instance.isin(['EPG(PB08)_L1'])]

# %%
meta_type_l[meta_type_l.main_groups == 'OL output']

# %% [markdown]
# ### some notes on DEBUG
# 
# hotspot:
# 
# 229653, R8p -> SLP250_R

# %%


# %% [markdown]
# ## left vs right, VIC and ht

# %%
meta_cb_vpn.head()

# %%
# Plot vision_r vs vision_l
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(meta_cb_vpn['ht_r'], meta_cb_vpn['ht_l'], alpha=0.5, s=10)
ax.set_xlabel('Vision Right', fontsize=12, fontweight='bold')
ax.set_ylabel('Vision Left', fontsize=12, fontweight='bold')
ax.set_title(f'Vision Right vs Left (n={len(meta_cb_vpn)})', fontsize=14, fontweight='bold')
# Add unity line
ax.plot([0, 5.5], [0, 5.5], 'r--', linewidth=1, label='Unity')
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
# plt.savefig(Path(plot_dir, 'vision_r_vs_vision_l.pdf'), bbox_inches='tight')
plt.show()

# %% [markdown]
# ## VIC and dob, intersection
# 
# union doens't make sense

# %%
# keep those with binocular input
meta_cb_vpn = meta_cb_vpn[(meta_cb_vpn['ht_l'].notna()) & (meta_cb_vpn['ht_r'].notna())]

# keep symmetric instances
meta_m = meta_cb_vpn[~meta_cb_vpn['instance'].str.contains('_(R|L)$', regex=True, na=False)].copy()

meta_lr = meta_cb_vpn[meta_cb_vpn['instance'].str.contains('_(R|L)$', regex=True, na=False)].copy()
# for each cell_type, keep only those that have both _L and _R instances
def has_both_sides(group):
    instances = group['instance'].values
    has_L = any('_L' in inst for inst in instances)
    has_R = any('_R' in inst for inst in instances)
    return has_L and has_R
meta_lr = meta_lr.groupby('cell_type').filter(has_both_sides)

# re-combine
meta_cb_vpn = pd.concat([meta_m, meta_lr], axis=0)

print(meta_cb_vpn.shape)

# %%
# Wrong filter
# meta_lr = meta_lr.groupby('cell_type').filter(lambda x: len(x) == 2)

# %%
meta_cb_vpn['ht'] = meta_cb_vpn[['ht_r', 'ht_l']].min(axis=1)
# have the same ht for _R and _L
ht = meta_cb_vpn.groupby('cell_type').agg({'ht': 'mean'}).reset_index()
meta_cb_vpn = pd.merge(meta_cb_vpn.drop(columns=['ht']), ht, on='cell_type', how='left')
print(meta_cb_vpn.shape)
# sum up vision
meta_cb_vpn['vision'] = meta_cb_vpn[['vision_r', 'vision_l']].sum(axis=1, skipna=True)
# # choose the smaller area_fit
# meta_cb_vpn['area_fit'] = meta_cb_vpn[['area_fit_r', 'area_fit_l']].min(axis=1, skipna=True)

# deg of binocularity, min of vision_r and vision_l devided by the sum
# meta_cb_vpn['dob'] = meta_cb_vpn[['vision_l', 'vision_r']].min(axis=1) / meta_cb_vpn[['vision_l', 'vision_r']].sum(axis=1)
# meta_cb_vpn['dob'] = meta_cb_vpn['dob']*2

# dob, min / max
meta_cb_vpn['dob'] = meta_cb_vpn[['vision_l', 'vision_r']].min(axis=1) / meta_cb_vpn[['vision_l', 'vision_r']].max(axis=1)

# %%
# # save
# meta_cb_vpn.to_csv(Path(cache_dir, 'meta_cb_vpn_bi_forJudi.csv'), index=False)

# %%
syn = pd.merge(syn0, 
               meta_cb_vpn[['bodyId', 'vision', 'ht', 'dob', 'main_groups']], 
               on='bodyId', how='left')

# %%
# # by ht
# ht_div = [1, 2, 3, 4, 5]

# for ht_bd in ht_div:
#     df_plot = syn.loc[(syn['ht'] >= ht_bd - 0.5) & (syn['ht'] < ht_bd + 0.5)].copy()
#     ## ##
#     # df_plot['val'] = df_plot['vision']
#     df_plot['val'] = df_plot['dob']
#     percentiles = np.nanpercentile(df_plot['val'], [5, 99])
#     if np.isscalar(percentiles):
#         vmin = vmax = percentiles
#     else:
#         vmin, vmax = percentiles
#     print(vmin, vmax)
#     fig, _, _ = plot_extreme_projection_xy(
#         df_plot,
#         outlines_bkgd,
#         xrange = [20000, 80000],
#         yrange = [000, 50000],
#         agg = 'largest',
#         im_norm = 'linear',
#         vmin = 0,
#         vmax = vmax,
#         agg_frac = 0.1,
#     )
#     plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_bd}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)

#     # plt.savefig(Path(plot_dir, f'vic_ht_{ht_bd}_bi.png'), dpi=300, bbox_inches='tight')
#     # plt.savefig(Path(plot_dir, f'vic_ht_{ht_bd}_bi.pdf'), bbox_inches='tight')
#     plt.savefig(Path(plot_dir, f'dob_ht_{ht_bd}_bi.pdf'), bbox_inches='tight')
#     plt.close(fig)

# %%
# by ht, separated vp and cb
gp = 'nonOL'
ht_div = [1, 3.5, 4, 5]
vmax_ls = [1, 1, 1]
for i in range(len(ht_div)-1):
    df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
    ## ##
    # df_plot['val'] = df_plot['vision']
    df_plot['val'] = df_plot['dob']
    roi_counts = df_plot.groupby('roi').size().sort_values(ascending=False)
    top_rois = roi_counts.iloc[:6]
    top_rois_ls.append(top_rois)
    percentiles = np.nanpercentile(df_plot['val'], [5, 99])
    if np.isscalar(percentiles):
        vmin = vmax = percentiles
    else:
        vmin, vmax = percentiles
    print(vmin, vmax)
    fig, _, _ = plot_extreme_projection_xy(
        df_plot,
        outlines_bkgd,
        xrange = [20000, 80000],
        yrange = [000, 50000],
        # agg = 'largest',
        agg = 'smallest',
        cmap_div = True,
        im_norm = 'linear',
        vmin = 0,
        # vmax = vmax,
        vmax = 1,
        agg_frac = 0.1,
    )
    plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_div[i]}_{ht_div[i+1]}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
    ## ##
    # plt.savefig(Path(plot_dir, f'vic_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}.pdf'), bbox_inches='tight')
    # plt.savefig(Path(plot_dir, f'dob_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}.pdf'), bbox_inches='tight')
    plt.savefig(Path(plot_dir, f'dob_min_ht_{gp}_{ht_div[i]}_{ht_div[i+1]}.pdf'), bbox_inches='tight')
    plt.close(fig)

# %%
# all in one, cb
gp = 'nonOL'
ht_div = [1, 5]
vmax_ls = [1]
i = 0
df_plot = syn[(syn['main_groups'] == gp) & (syn['ht'] > ht_div[i]) & (syn['ht'] <= ht_div[i+1])].copy()
df_plot['val'] = df_plot['dob']
fig, _, _ = plot_extreme_projection_xy(
    df_plot,
    outlines_bkgd,
    xrange = [20000, 80000],
    yrange = [000, 50000],
    agg = 'largest',
    # agg = 'smallest',
    cmap_div = True,
    im_norm = 'linear',
    vmin = 0,
    # vmax = vmax,
    vmax = 1,
    agg_frac = 0.1,
)
plt.text(0.5, 0.95, f'n_n={df_plot.bodyId.nunique()}, n_syn={len(df_plot)}, ht={ht_div[i]}_{ht_div[i+1]}', fontsize=10, fontweight='bold', ha='center', va='top', transform=plt.gca().transAxes)
plt.savefig(Path(plot_dir, f'dob_max.pdf'), bbox_inches='tight')
# plt.savefig(Path(plot_dir, f'dob_min.pdf'), bbox_inches='tight')
plt.close(fig)

# %%
# plot meta_cb_vpn.dob vs meta_cb_vpn.ht
plt.figure(figsize=(4,4))
plt.scatter(meta_cb_vpn['ht'], meta_cb_vpn['dob'], alpha=0.5)
plt.xlabel('hitting time (s)', fontsize=14, fontweight='bold')
plt.ylabel('death onset (s)', fontsize=14, fontweight='bold')
plt.title('Degree of Binocularity vs Hitting Time', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
# save
# plt.savefig(Path(plot_dir, f'dob_vs_ht_bi.png'), dpi=300, bbox_inches='tight')
plt.show()

# %%
meta_cb_vpn[meta_cb_vpn.cell_type == 'PS008_a2']

# %% [markdown]
# ### DEBUG

# %%
ht_bd = 5
df_plot = syn.loc[(syn['ht'] >= ht_bd - 0.5) & (syn['ht'] < ht_bd + 0.5)].copy()
df_plot['val'] = df_plot['vision']
# df_plot['val'] = df_plot['dob']
percentiles = np.nanpercentile(df_plot['val'], [5, 99])
if np.isscalar(percentiles):
    vmin = vmax = percentiles
else:
    vmin, vmax = percentiles

fig, _, _ = plot_extreme_projection_xy(
    df_plot,
    outlines_bkgd,
    xrange = [20000, 80000],
    yrange = [000, 50000],
    agg = 'largest',
    im_norm = 'linear',
    vmin = 0,
    vmax = vmax,
    agg_frac = 0.1,
)
# # plt.savefig(Path(plot_dir, f'vic_ht_{ht_bd}_bi.png'), dpi=300, bbox_inches='tight')
# plt.savefig(Path(plot_dir, f'dob_ht_{ht_bd}.png'), dpi=300, bbox_inches='tight')
# plt.close(fig)
fig.show()

# %%
df_plot.groupby('bodyId').agg(vision=('vision', 'median')).reset_index().merge(meta_cb_vpn[['bodyId','instance']], on='bodyId', how='left').sort_values('vision', ascending=False).head()

# %%
meta_cb_vpn[meta_cb_vpn.bodyId.isin([17682, 801562, 109328, 103156])]

# %%
meta_cb_vpn[meta_cb_vpn.cell_type.isin(['PS352','IN07B059','DNge088','AN06B048'])].sort_values(['instance'])

# %% [markdown]
# ## example

# %%
meta_cb_vpn[meta_cb_vpn.ht > 4.5].sort_values(['vision', 'instance'])

# %% [markdown]
# ## left vs right size

# %%
# from Judith, for rf size
thr_vic = 5e-4
thr_r2 = 0.05

fit_rf_r = pd.read_pickle(CACHE_DIR / f'{DATASET}_fit_rf.p')
vp_cb_vic_r = pd.read_pickle(CACHE_DIR / 'vp_cb_vic_w_hit_df.p')
meta_cb_vpn = pd.merge(
    fit_rf_r[['bodyId','instance','size','r2', 'hitting_time']],
    vp_cb_vic_r[['bodyId','VIC', 'main_groups']],
    on='bodyId', how='left'
)
meta_cb_vpn.rename(columns={'VIC':'vision', 'size':'area_fit', 'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape)
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['vision'] > thr_vic]
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['r2'] > thr_r2]
print(meta_cb_vpn.shape)
meta_cb_vpn_r = meta_cb_vpn.copy()

fit_rf_l = pd.read_pickle(CACHE_DIR / f'{DATASET}_left_fit_rf.p')
vp_cb_vic_l = pd.read_pickle(CACHE_DIR / 'vp_cb_left_vic_w_hit_df.p')
meta_cb_vpn = pd.merge(
    fit_rf_l[['bodyId','instance','size','r2', 'hitting_time']],
    vp_cb_vic_l[['bodyId','VIC', 'main_groups']],
    on='bodyId', how='left'
)
meta_cb_vpn.rename(columns={'VIC':'vision', 'size':'area_fit', 'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape)
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['vision'] > thr_vic]
meta_cb_vpn = meta_cb_vpn[meta_cb_vpn['r2'] > thr_r2]
print(meta_cb_vpn.shape)
meta_cb_vpn_l = meta_cb_vpn.copy()

# %%
# # keep those with binocular input
# print(meta_cb_vpn.shape)
# meta_cb_vpn = meta_cb_vpn[(meta_cb_vpn['r2_r'] > 0) & (meta_cb_vpn['r2_l'] > 0)]
# meta_cb_vpn = meta_cb_vpn[(meta_cb_vpn['area_fit_l'].notna()) & (meta_cb_vpn['area_fit_r'].notna())]

# # keep symmetric instances
# meta_m = meta_cb_vpn[~meta_cb_vpn['instance'].str.contains('_(R|L)$', regex=True, na=False)].copy()

# meta_lr = meta_cb_vpn[meta_cb_vpn['instance'].str.contains('_(R|L)$', regex=True, na=False)].copy()
# meta_lr = meta_lr.groupby('cell_type').filter(lambda x: len(x) == 2)
# # combine
# meta_cb_vpn = pd.concat([meta_m, meta_lr], axis=0)

# print(meta_cb_vpn.shape)

# %%
# scatter plot area_fit_l vs area_fit_r, colored by (ht_r+ht_l)/2
plt.figure(figsize=(6, 6))
sc = plt.scatter(
    meta_cb_vpn['area_fit_l'],
    meta_cb_vpn['area_fit_r'],
    c=(meta_cb_vpn['ht_l'] + meta_cb_vpn['ht_r']) / 2,
    cmap='viridis',
    alpha=0.7,
    edgecolors='w',
    linewidths=0.5
)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Area Fit Left')
plt.ylabel('Area Fit Right')
plt.title(f'Area Left vs Right, n_cell= {len(meta_cb_vpn)}')
cbar = plt.colorbar(sc)
cbar.set_label('Average HT')
plt.plot([1, 1e5], [1, 1e5], 'r--')  # unity line
plt.xlim(1, 1e3)
plt.ylim(1, 1e3)
plt.grid(True, which='both', ls='-', alpha=0.5)
# asp = 1
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
# plt.savefig(Path(plot_dir, 'area_fit_left_vs_right.png'), dpi=300, bbox_inches='tight')
# plt.close()
plt.show()

# %% [markdown]
# # ? obs. 2D extreme values proj, binocular, ghop

# %%
# roi background
outlines_ol = get_roi_outline(['ME(R)', 'ME(L)'], 0.0002)
outlines_roi = get_roi_outline(['AOTU(R)', 'PVLP(R)', 'PLP(L)', 'WED(R)', 'GOR(L)', 'SPS(R)', 'BU(R)'], 0.0002)
# outlines_roi = get_roi_outline(['AOTU(L)', 'PVLP(L)', 'PLP(R)', 'WED(L)', 'GOR(R)', 'SPS(L)', 'BU(L)'], 0.0002)

# combine
outlines_bkgd = outlines_ol + outlines_roi

# %% [markdown]
# ## load data

# %%
## ## tbars
# right
syn0 = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_tbar_cb.pkl'))
syn0 = syn0[syn0['type'] == 'pre']

# left
syn0_left = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_tbar_cb_left.pkl'))

# %%
# combine tbars
syn0 = pd.concat([syn0, syn0_left], ignore_index=True)
syn0 = syn0.drop_duplicates(subset=['bodyId', 'x', 'y', 'z'], keep='first')
syn0.shape

# %%
# VPN cluster
# vpn_clu = pd.read_csv(CACHE_DIR / f'{DATASET}_in_out_clusters.csv')

# %% [markdown]
# ## VIC and dob, T1 or T2, union or intersection

# %%
plot_dir = FIG_DIR / 'quan_propagation' / 'cumsum40_041010'
plot_dir.mkdir(parents=True, exist_ok=True)

# %%
#  load vic
# meta_cb_vpn = pd.merge(meta_cb_vpn, vpn_clu, on='instance', how='left')

meta_cb_vpn_r = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_cumsum40_ghop_vic_041010.pkl'))
meta_cb_vpn_l = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_cumsum40_ghop_vic_041010_left.pkl'))

# %%
## ## union
meta_cb_vpn = pd.merge(
    meta_cb_vpn_r.loc[(meta_cb_vpn_r['ghop'].isin(['T0','T1','T2'])), ['bodyId', 'instance', 'cell_type', 'main_groups', 'ghop', 'ht', 'vision', 'col_count', 'area_fit']], 
    meta_cb_vpn_l.loc[(meta_cb_vpn_l['ghop'].isin(['T0','T1','T2'])), ['bodyId', 'instance', 'cell_type', 'main_groups', 'ghop', 'ht', 'vision', 'col_count', 'area_fit']], 
    on=['bodyId', 'instance', 'cell_type'], how='outer', suffixes=('_r', '_l'))
meta_cb_vpn.shape

# %%
## ## intersection
meta_cb_vpn = pd.merge(
    meta_cb_vpn_r.loc[(meta_cb_vpn_r['ghop'].isin(['T0','T1','T2'])), ['bodyId', 'instance', 'cell_type', 'main_groups', 'ghop', 'ht', 'vision', 'col_count', 'area_fit']], 
    meta_cb_vpn_l.loc[(meta_cb_vpn_l['ghop'].isin(['T0','T1','T2'])), ['bodyId', 'instance', 'cell_type', 'main_groups', 'ghop', 'ht', 'vision', 'col_count', 'area_fit']], 
    on=['bodyId', 'instance', 'cell_type'], how='inner', suffixes=('_r', '_l'))
meta_cb_vpn.shape

# %%
# sum up vision
meta_cb_vpn['vision'] = meta_cb_vpn[['vision_r', 'vision_l']].sum(axis=1, skipna=True)
# choose the smaller area_fit
meta_cb_vpn['area_fit'] = meta_cb_vpn[['area_fit_r', 'area_fit_l']].min(axis=1, skipna=True)
# deg of binocularity, max of vision_r and vision_l devided by the sum
meta_cb_vpn['dob'] = meta_cb_vpn[['vision_l', 'vision_r']].max(axis=1) / meta_cb_vpn[['vision_l', 'vision_r']].sum(axis=1)
meta_cb_vpn['dob'] = 2 - meta_cb_vpn['dob']*2

# %%
# DEBUG
meta_cb_vpn[(meta_cb_vpn['vision_r'].notna()) & (meta_cb_vpn['vision_l'].notna())]
# meta_cb_vpn[meta_cb_vpn['area_fit_r'].notna() & meta_cb_vpn['area_fit_l'].notna()]
# meta_cb_vpn[(meta_cb_vpn.ghop_r == 'T0') | (meta_cb_vpn.ghop_l == 'T0')]
# meta_cb_vpn

# %%
# redefine T1 and T2 bodyIds 

# T1
meta = meta_cb_vpn[(meta_cb_vpn['ghop_l'] == 'T1') | (meta_cb_vpn['ghop_r'] == 'T1')].copy()
print(meta.shape)
meta = meta.groupby('cell_type').filter(lambda x: len(x) == 2)
print(meta.shape)
# all rows with the same cell_type have to have the same set of ghop_l and ghop_r
meta = meta.groupby('cell_type').filter(lambda x: set(x['ghop_l'].values) == set(x['ghop_r'].values))
print(meta.shape)
# only keep those with both T1 and T2 -> not good
# meta = meta.groupby('cell_type').filter(lambda x: (x['ghop_l'].values == x['ghop_r'].values).all())

# meta.sort_values('cell_type').head(10)

# DEBUG
# # this is ok, we'll include T1+T2 as T1, and keep T2 only as T2
# meta[(meta['ghop_l'] == 'T2') | (meta['ghop_r'] == 'T2')]

ids_t1 = meta['bodyId'].unique().tolist()

# T2
meta = meta_cb_vpn[(meta_cb_vpn['ghop_l'] == 'T2') | (meta_cb_vpn['ghop_r'] == 'T2')].copy()
print(meta.shape)
meta = meta.groupby('cell_type').filter(lambda x: len(x) == 2)
print(meta.shape)
# all rows with the same cell_type have to have the same set of ghop_l and ghop_r
meta = meta.groupby('cell_type').filter(lambda x: set(x['ghop_l'].values) == set(x['ghop_r'].values))
print(meta.shape)
# remove cell_type with T1 
meta = meta.groupby('cell_type').filter(lambda x: 'T1' not in x['ghop_l'].values and 'T1' not in x['ghop_r'].values)
print(meta.shape)

ids_t2 = meta['bodyId'].unique().tolist()

# T0
meta = meta_cb_vpn[(meta_cb_vpn['ghop_l'] == 'T0') | (meta_cb_vpn['ghop_r'] == 'T0')].copy()
print(meta.shape)

ids_t0 = meta['bodyId'].unique().tolist()

# group into list
ids_t0t1t2 = [ids_t0, ids_t1, ids_t2]

# %%
syn = pd.merge(syn0, 
               meta_cb_vpn[['bodyId', 'vision', 'ghop_l', 'ghop_r', 'area_fit', 'dob']], 
               on='bodyId', how='left')

# %%
# color bar limits
val = syn.loc[(syn['ghop_l'].isin(['T0','T1','T2'])) | syn['ghop_r'].isin(['T0','T1','T2']),'dob'].values
print(np.nanpercentile(val, [0,1, 5, 95,99,100]) )
vmin_fixed = 0; vmax_fixed = 0.4

# %%
ghops = ['T0','T1','T2']
for ids, ghop in zip(ids_t0t1t2, ghops):
    df_plot = syn.loc[syn['bodyId'].isin(ids)].copy()
    df_plot['val'] = df_plot['dob']
    # percentiles = np.nanpercentile(df_plot['val'], [5, 95])
    # if np.isscalar(percentiles):
    #     vmin = vmax = percentiles
    # else:
    #     vmin, vmax = percentiles
    fig, _, _ = plot_extreme_projection_xy(
        df_plot,
        outlines_bkgd,
        xrange,
        yrange,
        agg = 'largest',
        im_norm = 'linear',
        vmin = 0,
        vmax = 1,
        agg_frac = 0.1,
    )
    ## ##
    plt.savefig(Path(plot_dir, f'dob_bi_union_{ghop}.png'), dpi=300, bbox_inches='tight')
    # plt.savefig(Path(plot_dir, f'vic_bi_union_{ghop}.png'), dpi=300, bbox_inches='tight')
    # plt.savefig(Path(plot_dir, f'vic_bi_intersection_{ghop}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# %% [markdown]
# ## resolution, T1 or T2, union or intersection

# %%
# load vic
plot_dir = FIG_DIR / 'quan_propagation' / 'cumsum40_041010'
plot_dir.mkdir(parents=True, exist_ok=True)

meta_cb_vpn_r = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_cumsum40_ghop_res_041010_r2.pkl'))
meta_cb_vpn_l = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_cumsum40_ghop_res_041010_r2_left.pkl'))

# %%
## ## union
meta_cb_vpn = pd.merge(
    meta_cb_vpn_r.loc[(meta_cb_vpn_r['ghop'].isin(['T0','T1','T2'])), ['bodyId', 'instance', 'cell_type', 'main_groups', 'ghop', 'ht', 'vision', 'col_count', 'area_fit']], 
    meta_cb_vpn_l.loc[(meta_cb_vpn_l['ghop'].isin(['T0','T1','T2'])), ['bodyId', 'instance', 'cell_type', 'main_groups', 'ghop', 'ht', 'vision', 'col_count', 'area_fit']], 
    on=['bodyId', 'instance', 'cell_type'], how='outer', suffixes=('_r', '_l'))

# sum up vision
meta_cb_vpn['vision'] = meta_cb_vpn[['vision_r', 'vision_l']].sum(axis=1, skipna=True)
# choose the smaller area_fit
meta_cb_vpn['area_fit'] = meta_cb_vpn[['area_fit_r', 'area_fit_l']].min(axis=1, skipna=True)
meta_cb_vpn.shape

# %%
## ## intersection
meta_cb_vpn = pd.merge(
    meta_cb_vpn_r.loc[(meta_cb_vpn_r['ghop'].isin(['T0','T1','T2'])), ['bodyId', 'instance', 'cell_type', 'main_groups', 'ghop', 'ht', 'vision', 'col_count', 'area_fit']], 
    meta_cb_vpn_l.loc[(meta_cb_vpn_l['ghop'].isin(['T0','T1','T2'])), ['bodyId', 'instance', 'cell_type', 'main_groups', 'ghop', 'ht', 'vision', 'col_count', 'area_fit']], 
    on=['bodyId', 'instance', 'cell_type'], how='inner', suffixes=('_r', '_l'))

# sum up vision
meta_cb_vpn['vision'] = meta_cb_vpn[['vision_r', 'vision_l']].sum(axis=1, skipna=True)
# choose the smaller area_fit
meta_cb_vpn['area_fit'] = meta_cb_vpn[['area_fit_r', 'area_fit_l']].min(axis=1, skipna=True)
meta_cb_vpn.shape

# %%
# redefine T1 and T2 bodyIds 

# T1
meta = meta_cb_vpn[(meta_cb_vpn['ghop_l'] == 'T1') | (meta_cb_vpn['ghop_r'] == 'T1')].copy()
print(meta.shape)
meta = meta.groupby('cell_type').filter(lambda x: len(x) == 2)
print(meta.shape)
# all rows with the same cell_type have to have the same set of ghop_l and ghop_r
meta = meta.groupby('cell_type').filter(lambda x: set(x['ghop_l'].values) == set(x['ghop_r'].values))
print(f'T1 - {meta.shape}')
# only keep those with both T1 and T2 -> not good
# meta = meta.groupby('cell_type').filter(lambda x: (x['ghop_l'].values == x['ghop_r'].values).all())

# meta.sort_values('cell_type').head(10)

# DEBUG
# # this is ok, we'll include T1+T2 as T1, and keep T2 only as T2
# meta[(meta['ghop_l'] == 'T2') | (meta['ghop_r'] == 'T2')]

ids_t1 = meta['bodyId'].unique().tolist()

# T2
meta = meta_cb_vpn[(meta_cb_vpn['ghop_l'] == 'T2') | (meta_cb_vpn['ghop_r'] == 'T2')].copy()
print(meta.shape)
meta = meta.groupby('cell_type').filter(lambda x: len(x) == 2)
print(meta.shape)
# all rows with the same cell_type have to have the same set of ghop_l and ghop_r
meta = meta.groupby('cell_type').filter(lambda x: set(x['ghop_l'].values) == set(x['ghop_r'].values))
print(meta.shape)
# remove cell_type with T1 
meta = meta.groupby('cell_type').filter(lambda x: 'T1' not in x['ghop_l'].values and 'T1' not in x['ghop_r'].values)
print(f'T2 - {meta.shape}')

ids_t2 = meta['bodyId'].unique().tolist()

# T0
meta = meta_cb_vpn[(meta_cb_vpn['ghop_l'] == 'T0') | (meta_cb_vpn['ghop_r'] == 'T0')].copy()
print(meta.shape)

ids_t0 = meta['bodyId'].unique().tolist()

# group into list
ids_t0t1t2 = [ids_t0, ids_t1, ids_t2]

# %%
syn = pd.merge(syn0, 
               meta_cb_vpn[['bodyId', 'area_fit', 'ghop_l', 'ghop_r']],
               on='bodyId', how='left')

# %%
# color bar limits
val = syn.loc[(syn['ghop_l'].isin(['T0','T1','T2'])) | syn['ghop_r'].isin(['T0','T1','T2']),'area_fit'].values
print(np.nanpercentile(val, [0,1, 5, 95,99,100]) )
vmin_fixed = 1; vmax_fixed = 200

# %%
ghops = ['T0','T1','T2']
for ids, ghop in zip(ids_t0t1t2, ghops):
    df_plot = syn.loc[syn['bodyId'].isin(ids)].copy()
    df_plot['val'] = df_plot['area_fit']

    percentiles = np.nanpercentile(df_plot['val'], [5, 95])
    if np.isscalar(percentiles):
        vmin = vmax = percentiles
    else:
        vmin, vmax = percentiles

    fig, _, _ = plot_extreme_projection_xy(
        df_plot,
        outlines_bkgd,
        xrange,
        yrange,
        agg = 'smallest',
        im_norm = 'log',
        vmin = 1,
        vmax = vmax,
        agg_frac = 0.1,
    )
    ## ##
    # plt.savefig(Path(plot_dir, f'res_bi_union_{ghop}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(Path(plot_dir, f'res_bi_intersection_{ghop}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# %% [markdown]
# # obs. 2D extreme values proj, call func, vnc

# %%
# roi background
# outlines_ol = get_roi_outline(['ME(R)', 'ME(L)'], 0.0002)
# outlines_roi = get_roi_outline(['AOTU(R)', 'PVLP(R)', 'PLP(L)', 'WED(R)', 'GOR(L)', 'SPS(R)', 'BU(R)'], 0.0002)

# vnc
outlines_bkgd_vnc = get_roi_outline(['VNC'], 0.0002, lop='y')

# combine
outlines_bkgd = outlines_bkgd_vnc

# %%
## ## tbars
# right
syn0 = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_tbar_vnc.pkl'))

# left
# syn0 = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_tbar_left.pkl'))

# psd for ratio
# syn0 = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_psd_for_ratio.pkl'))

# %%
# VPN cluster
vpn_clu = pd.read_csv(CACHE_DIR / f'{DATASET}_in_out_clusters.csv')

# %% [markdown]
# ## VIC

# %%
## ## load
plot_dir = FIG_DIR / 'quan_propagation' / 'cumsum40_001010'
plot_dir.mkdir(parents=True, exist_ok=True)

meta_cb_vpn = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_cumsum40_ghop_vic_001010.pkl'))

meta_cb_vpn = pd.merge(meta_cb_vpn, vpn_clu, on='instance', how='left')

# %%
syn = pd.merge(syn0, 
               meta_cb_vpn[['bodyId', 'vision', 'ghop', 'cluster']], 
               on='bodyId', how='left')

syn = syn[syn['ghop'].isin(['T0','T1','T2'])]

# %%
# color bar limits
val = syn.loc[syn['ghop'].isin(['T0','T1','T2']),'vision'].values
print(np.nanpercentile(val, [0,1, 5, 95,99,100]) )
vmin_fixed = 0; vmax_fixed = 0.1

# %%
ghops = ['T0','T1','T2']
for ghop in ghops:
    df_plot = syn.loc[syn['ghop'] == ghop].copy()
    df_plot['val'] = df_plot['vision']

    fig, _, _ = plot_extreme_projection_xz(
        df_plot,
        outlines_bkgd,
        xrange = xrange_vnc,
        zrange = zrange_vnc,
        agg = 'largest',
        im_norm = 'linear',
        vmin = vmin_fixed,
        vmax = vmax_fixed,
        agg_frac = 0.1,
    )

    if fig is not None:
        plt.savefig(Path(plot_dir, f'vic_{ghop}_vnc.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

# %% [markdown]
# ## resolution

# %%
## ## load
meta_cb_vpn = pd.read_pickle(Path(cache_dir, 'meta_cb_vpn_cumsum40_ghop_res_001010_r2.pkl'))

meta_cb_vpn = pd.merge(meta_cb_vpn, vpn_clu, on='instance', how='left')

# %%
syn = pd.merge(syn0, 
               meta_cb_vpn[['bodyId', 'area_fit', 'area_fit_input', 'vision', 'col_count', 'col_count_input', 'ghop', 'cluster']], 
               on='bodyId', how='left')

syn = syn[syn['ghop'].isin(['T0','T1','T2'])]

syn['ratio_area'] = syn['area_fit'] / syn['area_fit_input']
syn['ratio_col'] = syn['col_count'] / syn['col_count_input']

# %%
## ## color bar limits
val = syn.loc[syn['ghop'].isin(['T0','T1','T2']),'area_fit'].values
# val = syn.loc[syn['ghop'].isin(['T0','T1','T2']),'col_count'].values
print(np.nanpercentile(val, [0,1, 5, 95,99,100]) )
vmin_fixed = 1; vmax_fixed = 300

# ## ## ratio
# val = syn.loc[syn['ghop'].isin(['T0','T1','T2']),['ratio_area']]
# val = syn.loc[syn['ghop'].isin(['T0','T1','T2']),['ratio_col']]
# print(np.nanpercentile(np.log2(val), [0,1, 5, 95,99,100]) )
# # vmin_fixed = 0.2; vmax_fixed = 8
# vmin_fixed = -2; vmax_fixed = 3

# %%
ghops = ['T0','T1','T2']
vals = ['area_fit', 'col_count']
agg = 'smallest'
im_norm = 'log'
vmin_fixed = 70
vmax_fixed = 300

for val in vals:
    for ghop in ghops:
        df_plot = syn.loc[syn['ghop'] == ghop].copy()
        df_plot['val'] = df_plot[val]

        fig, _, _ = plot_extreme_projection_xz(
            df_plot,
            outlines_bkgd,
            agg = agg,
            agg_frac = 0.1,
            im_norm = im_norm,
            vmin=vmin_fixed,
            vmax=vmax_fixed,
        )
        plt.savefig(Path(plot_dir, f'{val}_{ghop}_vnc.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

# %% [markdown]
# # end

# %%




