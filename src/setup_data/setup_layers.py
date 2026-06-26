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

# %% [markdown]
# # Compute flow hitting times
#
# Inputs:
# - `{dataset}_meta.csv`
# - `{dataset}_inprop.npz`
#
# Outputs:
# - `{dataset}_{side_char}_flow_{N_FLOW}step_{HIT_THRE}thre_hit.csv` (per-idx)
# - `{dataset}_{side_char}_flow_{N_FLOW}step_{HIT_THRE}thre_hit_per_group.csv` (per-instance median)
#
# Runtime: < 2 hours.
#

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().resolve().parent if Path.cwd().name == 'setup_data' else Path.cwd().resolve()))

import pandas as pd
import scipy.sparse as sp
from dotenv import load_dotenv

from utils.config import DATA_DIR, DATASET, HIT_THRE, N_FLOW, PROJECT_ROOT, SIDE_CHAR
import setup_data as sd

load_dotenv(PROJECT_ROOT / '.env')
TAG = DATASET.split('_', 1)[-1]  # e.g. 'v1.0'
print(f'dataset: {DATASET}; side: {SIDE_CHAR}; data: {DATA_DIR}')


# %% [markdown]
# ## Load prerequisites

# %%
meta = pd.read_csv(DATA_DIR / f'malecns_{TAG}_meta.csv')
inprop = sp.load_npz(DATA_DIR / f'malecns_{TAG}_inprop.npz')
print(f'meta rows: {len(meta):,}; inprop shape: {inprop.shape} nnz={inprop.nnz:,}')


# %% [markdown]
# ## Run flow

# %%
idx_to_instance = dict(zip(meta.idx, meta.instance))
seed_groups = sd.R_FLOW_SEEDS if SIDE_CHAR == 'r' else sd.L_FLOW_SEEDS
save_prefix = f'malecns_{TAG}_{SIDE_CHAR}_flow_'

flow_df = sd.fetch_instance_flow(
    inprop, idx_to_instance, seed_groups, DATA_DIR, save_prefix,
    steps=N_FLOW, thre=HIT_THRE,
)
print(f'flow rows: {len(flow_df):,}')
print(f'wrote {DATA_DIR}/{save_prefix}{N_FLOW}step_{HIT_THRE}thre_hit.csv'
      f' and {save_prefix}{N_FLOW}step_{HIT_THRE}thre_hit_per_group.csv')

