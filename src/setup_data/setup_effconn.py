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
# # Compute lateral flow (forward-directed connectivity)
#
# Inputs:
# - `{dataset}_meta.csv`
# - `{dataset}_inprop.npz`
# - `{dataset}_{side_char}_flow_{N_FLOW}step_{HIT_THRE}thre_hit_per_group.csv`
#
# Outputs:
# - `{dataset}_{side_char}_lat_flow_0.npz`
# - `{dataset}_{side_char}_lat_flow_sum.npz`
#
# Runtime: 8× A100 ≈ 15 min; CPU-only (~24 cores) ≈ 2–3 hours.
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
TAG = DATASET.split('_', 1)[-1]
print(f'dataset: {DATASET}; side: {SIDE_CHAR}; data: {DATA_DIR}')


# %% [markdown]
# ## Load prerequisites

# %%
meta = pd.read_csv(DATA_DIR / f'malecns_{TAG}_meta.csv')
inprop = sp.load_npz(DATA_DIR / f'malecns_{TAG}_inprop.npz')
flow_df = pd.read_csv(
    DATA_DIR / f'malecns_{TAG}_{SIDE_CHAR}_flow_{N_FLOW}step_{HIT_THRE}thre_hit_per_group.csv'
)
idx_to_instance = dict(zip(meta.idx, meta.instance))
print(f'meta rows: {len(meta):,}; inprop shape: {inprop.shape} nnz={inprop.nnz:,}')
print(f'flow rows: {len(flow_df):,}')


# %% [markdown]
# ## Trim & save

# %%
trimmed_inprop = sd.trim_inprop_by_flow(
    inprop, idx_to_instance, flow_df, flow_diff_min=0.0,
)
print(f'trimmed_inprop nnz: {trimmed_inprop.nnz:,}')

lat_prefix = f'malecns_{TAG}_{SIDE_CHAR}_lat_flow_'
sd.save_lat_flow_0(trimmed_inprop, DATA_DIR, lat_prefix)

summed = sd.compute_lat_flow_sum(trimmed_inprop)
sd.save_lat_flow_sum(summed, DATA_DIR, lat_prefix)
print(f'summed nnz: {summed.nnz:,}')
print(f'wrote {DATA_DIR}/{lat_prefix}0.npz and {DATA_DIR}/{lat_prefix}sum.npz')

