# compute effwt matrix for visual channels, right and left side, and save to cache
# this is used for quick determining VIC

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
import numpy as np
import pandas as pd
import scipy
import gc

import connectome_interpreter as ci

# %%
from utils.config import (DATA_DIR, DATASET)

cache_dir = Path(PROJECT_ROOT, 'cache', 'quan_propagation')
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load meta

# %%
# Load meta, this with AN's R7/8, 
meta = pd.read_csv(DATA_DIR / f'{DATASET}_meta.csv')
print(meta.shape)  

# %%
# make dictionaries that map indices to meta info
idx_to_bodyId = dict(zip(meta.idx, meta.bodyId))
idx_to_coords = dict(zip(meta.idx, meta.coords))

# %%
# output by removing all visual inputs
outidx = meta[
    ~meta['cell_type'].isin(['L1', 'L2', 'L3', 'R7', 'R8', 'R7d', 'R8d','HBeyelet', 'R7R8_unclear', 'R7_unclear', 'R8_unclear'])
    ]['idx']

# %% [markdown]
# # effwt, right side

# %% [markdown]
# ## load stepsn

# %%
stepsn = scipy.sparse.load_npz(Path(DATA_DIR, f'{DATASET}_r_lat_flow_sum.npz'))
stepsn.shape

# %% [markdown]
# ## effwt matrix from all 7 channels
# `effwt_visr[str(bodyId)].sum` gives similar result to "malecns_v1.0_r_ol_cb_vic_raw.parquet" but without HB eyelet

# %%
inidx = meta.idx[meta.instance.isin(['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R', 'R7d_R', 'R8d_R'])] 

effwt = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )
print(effwt.shape)

# save effwt_visr
effwt.to_pickle(Path(cache_dir, 'effwt_visr_7.pkl'))

# load effwt_visr from pickle
# effwt_visr = pd.read_pickle(Path(cache_dir, 'effwt_visr_7.pkl'))


# %% [markdown]
# ## effwt matrix from individual channels

# %%
channel_list = ['L1_R', 'L2_R', 'L3_R', 'R7_R', 'R8_R', 'R7d_R', 'R8d_R', 'HBeyelet_R']

for ch in channel_list:
    inidx = meta.idx[meta.instance.isin([ch])] 
    effwt = ci.result_summary(stepsn, inidx, outidx,
                        inidx_map = idx_to_coords, 
                        outidx_map = idx_to_bodyId,
                        display_threshold = 0,
                        display_output= False
                        )
    print(effwt.shape)

    # save effwt_visr
    effwt.to_pickle(Path(cache_dir, f'effwt_visr_{ch}.pkl'))

# %%
del stepsn
gc.collect()

# %% [markdown]
# # effwt, left side

# %% [markdown]
# ## load stepsn

# %%
stepsn = scipy.sparse.load_npz(Path(DATA_DIR, f'{DATASET}_l_lat_flow_sum.npz'))
stepsn.shape

# %% [markdown]
# ## effwt matrix from all 7 channels

# %%
inidx = meta.idx[meta.instance.isin(['L1_L', 'L2_L', 'L3_L', 'R7_L', 'R8_L',  'R7d_L', 'R8d_L'])] 
effwt = ci.result_summary(stepsn, inidx, outidx,
                    inidx_map = idx_to_coords, 
                    outidx_map = idx_to_bodyId,
                    display_threshold = 0,
                    display_output= False
                    )
print(effwt.shape)

# save effwt_visl
effwt.to_pickle(Path(cache_dir, 'effwt_visl_7.pkl'))

# load effwt_visl
# effwt_visl = pd.read_pickle(Path(cache_dir, 'effwt_visl_7.pkl'))

# %%
del stepsn
gc.collect()

# %% [markdown]
# # End


