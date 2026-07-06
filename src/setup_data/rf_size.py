# compute ellipse and ahull sizes

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
import scipy

from utils.rf_metrics import count_col_ahull
from utils.ol_rf import compute_rf
from utils.config import (CACHE_DIR, DATA_DIR, DATASET, PARAMS_DIR, SIDE_CHAR)

# %%
cache_dir = CACHE_DIR / 'quan_propagation'
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Some setups

# %% [markdown]
# ## Get meta

# %%
# Load meta, this with AN's R7/8, 
meta = pd.read_csv(DATA_DIR / f'{DATASET}_meta.csv')
print(meta.shape)  

# %%
# make dictionaries that map indices to meta info
idx_to_bodyId = dict(zip(meta.idx, meta.bodyId))
idx_to_coords = dict(zip(meta.idx, meta.coords))
bodyId_to_idx = dict(zip(meta.bodyId, meta.idx))

# %% [markdown]
# ## ol types

# %%
# cell type table xlsx
oltypes = pd.read_excel(PARAMS_DIR / 'Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx')
oltypes.rename(columns={'cell type':'cell_type', 'main groups':'main_groups'}, inplace=True)
print(oltypes.shape)

# only OL intrinsic types
oltypes_on = oltypes[oltypes.main_groups.isin(['ONIN','ONCN'])].copy()
print(oltypes_on.shape)

# %% [markdown]
# ## rois

# %%
# neuprint.fetch_primary_rois()
# rois_dict = neuprint.queries.fetch_roi_hierarchy(include_subprimary=False)
# rois_dict['CNS'].keys()

# %%
from utils.query_roi import get_primary_rois
rois_cb = get_primary_rois('CentralBrain')
len(rois_cb)

# %% [markdown]
# ## load prop matrix sum

# %%
stepsn = scipy.sparse.load_npz(Path(DATA_DIR, f'{DATASET}_{SIDE_CHAR}_lat_flow_sum.npz'))
print(stepsn.shape)
print(Path(DATA_DIR, f'{DATASET}_{SIDE_CHAR}_lat_flow_sum.npz'))

# %% [markdown]
# # Ellipse size from Gaussian fit, OL types

# %%
from utils.ol_rf import _compute_rf_params, _clean_rf_params

# %% [markdown]
# ### all cells with VIC > thr

# %%
# ~15 min
rf_fit = _compute_rf_params(
    meta, stepsn, 
    in_instances=['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R7d_R', 'R8_R', 'R8d_R'],
    out_instances = oltypes.instance.unique().tolist(),
    cumsum_thre = 0.7,
)
rf_fit = _clean_rf_params(rf_fit)

print(rf_fit.shape)

# save
rf_fit.to_pickle(Path(cache_dir, 'rf_fit_thr07.pkl'))

# %% [markdown]
# # ahull size
# 
# volume under a 2D Gaussian within a radius of 1 std is about 39%

# %% [markdown]
# ## vp + cb, right-side

# %%
# right-size meta for vpn and vcbn with ht and vic
vp_cb_r = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_r.p'))
print(vp_cb_r.shape)

# %%
# loop, ~60 min

meta_ahull = vp_cb_r.copy()

# filter by VIC, per instance 
thr_vic = 5e-4
inst = meta_ahull.groupby('instance').agg({'VIC':'median'}).reset_index()
inst = inst[inst.VIC > thr_vic]['instance']
meta_ahull = meta_ahull[meta_ahull.instance.isin(inst)]

thr_mode = 'cumsum'
remove_frac = 0.6
min_rmfrac = 0.1
meta_ahull['ahull_size'] = np.nan
meta_ahull['ahull_com_x'] = np.nan
meta_ahull['ahull_com_y'] = np.nan
meta_ahull['ahull_kept_points'] = np.nan
meta_ahull['ahull_rmfrac'] = np.nan
meta_ahull['ahull_alpha'] = np.nan
skipped_idx = []

inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R', 'R7d_R', 'R8d_R'])]

for i, row in meta_ahull.iterrows():
    try:
        print(bodyId_to_idx[row['bodyId']])
        rf = compute_rf(meta, stepsn, inidx=inidx, outidx=bodyId_to_idx[row['bodyId']])
        df = rf.set_index('coords')[['effective weight']]
        edge_points, edges, area, com, kept_points, used_frac, used_alpha = count_col_ahull(
            df, thr_mode=thr_mode, remove_frac = remove_frac, min_remove_frac = min_rmfrac
            )
        if edges is not None and len(edge_points) == 1:
            meta_ahull.at[i, 'ahull_size'] = area
            meta_ahull.at[i, 'ahull_com_x'] = com[0]
            meta_ahull.at[i, 'ahull_com_y'] = com[1]
            meta_ahull.at[i, 'ahull_kept_points'] = len(kept_points)
            meta_ahull.at[i, 'ahull_rmfrac'] = used_frac
            meta_ahull.at[i, 'ahull_alpha'] = used_alpha
        else:
            skipped_idx.append(i)
    except ValueError as e:
        print(f"skipping bodyId {row['bodyId']}: {e}")
        skipped_idx.append(i)
        continue

meta_skipped = meta_ahull.loc[skipped_idx]

# save
meta_ahull.to_pickle(Path(DATA_DIR, 'meta_VP_CB_ahull_cumsum60.pkl'))


# %% [markdown]
# ## ol intrinsic, right-side

# %%
# loop, ~70 min
meta_ahull = meta[meta['instance'].isin(oltypes_on['instance'])]

thr_mode = 'cumsum'
remove_frac = 0.6
meta_ahull['ahull_size'] = np.nan
meta_ahull['ahull_com_x'] = np.nan
meta_ahull['ahull_com_y'] = np.nan
meta_ahull['ahull_kept_points'] = np.nan
meta_ahull['ahull_rmfrac'] = np.nan
meta_ahull['ahull_alpha'] = np.nan
skipped_idx = []

inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R', 'R7d_R', 'R8d_R'])]

for i, row in meta_ahull.iterrows():
    try:
        print(bodyId_to_idx[row['bodyId']])
        rf = compute_rf(meta, stepsn, inidx=inidx, outidx=bodyId_to_idx[row['bodyId']])
        df = rf.set_index('coords')[['effective weight']]
        edge_points, edges, area, com, kept_points, used_frac, used_alpha = count_col_ahull(
            df, thr_mode=thr_mode, remove_frac = remove_frac, min_remove_frac = min_rmfrac
            )
        if edges is not None and len(edge_points) == 1:
            meta_ahull.at[i, 'ahull_size'] = area
            meta_ahull.at[i, 'ahull_com_x'] = com[0]
            meta_ahull.at[i, 'ahull_com_y'] = com[1]
            meta_ahull.at[i, 'ahull_kept_points'] = len(kept_points)
            meta_ahull.at[i, 'ahull_rmfrac'] = used_frac
            meta_ahull.at[i, 'ahull_alpha'] = used_alpha
        else:
            skipped_idx.append(i)
    except ValueError as e:
        print(f"skipping bodyId {row['bodyId']}: {e}")
        skipped_idx.append(i)
        continue

meta_skipped = meta_ahull.loc[skipped_idx]

# save
meta_ahull.to_pickle(Path(DATA_DIR, 'meta_ON_ahull_cumsum60.pkl'))

# %% [markdown]
# # End


