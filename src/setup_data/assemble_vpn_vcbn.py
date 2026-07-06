# run in this order: this -> effwt_matrix -> rf_size -> contrib_plot_mip + gallery

# combine meta with vic and ht

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

# %%
from utils import olc_client
c = olc_client.connect(verbose=False)

# %%
import pandas as pd

from utils.config import (DATA_DIR, DATASET, HIT_THRE, N_FLOW, PARAMS_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Some setups

# %% [markdown]
# ## ol types

# %%
# cell type table xlsx
oltypes = pd.read_excel(PARAMS_DIR / 'Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx')
oltypes.rename(columns={'cell type':'cell_type', 'main groups':'main_groups'}, inplace=True)
print(oltypes.shape)

oltypes_vpn = oltypes[oltypes['main_groups'].str.contains('VPN')]

# %% [markdown]
# ## Get meta

# %%
# Load meta, this with AN's R7/8, 
meta = pd.read_csv(DATA_DIR / f'{DATASET}_meta.csv')
print(meta.shape)

# %% [markdown]
# # vpn and vcbn, ht + vic

# %%
from utils.vic import get_vcbn_types

# right-side vpn and vcbn
vcbn = get_vcbn_types(side_char='r')
vp_cb_r = meta[
    meta['instance'].isin(vcbn.instance) |
    meta['instance'].isin(oltypes[oltypes['main_groups']=='VPN']['instance']) 
]

# left-side vpn and vcbn
vcbn = get_vcbn_types(side_char='l')
vp_cb_l = meta[
    meta['instance'].isin(vcbn.instance) |
    (
        meta['cell_type'].isin(oltypes[oltypes['main_groups']=='VPN']['cell_type']) &
        ~meta['instance'].isin(oltypes[oltypes['main_groups']=='VPN']['instance'])
    )
]

vp_cb_r.shape, vp_cb_l.shape

# %%
# add ht and vic

# load hitting time
ht_r = pd.read_csv(DATA_DIR / f'{DATASET}_r_flow_{N_FLOW}step_{HIT_THRE}thre_hit.csv')
ht_l = pd.read_csv(DATA_DIR / f'{DATASET}_l_flow_{N_FLOW}step_{HIT_THRE}thre_hit.csv')
print(ht_r.shape, ht_l.shape)

from utils.vic import get_ol_cb_vic_raw

# load vic
vic_r = get_ol_cb_vic_raw(side_char='r')
vic_l = get_ol_cb_vic_raw(side_char='l')
print(vic_r.shape, vic_l.shape)

# combine
vp_cb_r = pd.merge(vp_cb_r[['bodyId','cell_type','idx']], ht_r, on='idx', how='inner')
vp_cb_l = pd.merge(vp_cb_l[['bodyId','cell_type','idx']], ht_l, on='idx', how='inner')
print(vp_cb_r.shape, vp_cb_l.shape)

vp_cb_r = pd.merge(vp_cb_r, vic_r[['bodyId','VIC', 'main_groups', 'region']], on='bodyId', how='inner')
vp_cb_l = pd.merge(vp_cb_l, vic_l[['bodyId','VIC', 'main_groups', 'region']], on='bodyId', how='inner')
print(vp_cb_r.shape, vp_cb_l.shape)

# %%
# save to cache
vp_cb_r.to_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_r.p'))
vp_cb_l.to_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_l.p'))

# # load
vp_cb_r = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_r.p'))
vp_cb_l = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_l.p'))

# %% [markdown]
# # End
