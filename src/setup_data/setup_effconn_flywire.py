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
# # Compute lateral flow on the flywire-fafb-v783b dataset (LEFT flow)
#
# **Temporary replica of `setup_effconn.ipynb`** adapted to the flywire dataset,
# which does not follow the `malecns_` filename convention and ships only a
# per-neuron flow file (no `_per_group` sidecar).
#
# Differences vs `setup_effconn.ipynb`:
# - filenames have no `malecns_` prefix (`TAG = 'flywire-fafb-v783b'`)
# - side is left (`SIDE_CHAR = 'l'`)
# - the per-instance flow is derived here by taking the **median** hitting time
#   over all neurons sharing an `instance`, since flywire has no `_per_group` file
#
# Inputs:
# - `flywire-fafb-v783b_meta.csv`
# - `flywire-fafb-v783b_inprop.npz`
# - `flywire-fafb-v783b_l_flow_20step_0.1thre_hit.csv`  (per-neuron)
#
# Outputs:
# - `flywire-fafb-v783b_l_flow_20step_0.1thre_hit_per_group.csv`  (derived, median)
# - `flywire-fafb-v783b_l_lat_flow_0.npz`
# - `flywire-fafb-v783b_l_lat_flow_sum.npz`

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().resolve().parent if Path.cwd().name == 'setup_data' else Path.cwd().resolve()))

import pandas as pd
import scipy.sparse as sp
from dotenv import load_dotenv

from utils.config import DATA_DIR, HIT_THRE, N_FLOW, PROJECT_ROOT
import setup_data as sd

load_dotenv(PROJECT_ROOT / '.env')

# --- flywire-specific overrides (temporary replica of setup_effconn) ---
TAG = 'flywire-fafb-v783b'   # this dataset has no 'malecns_' prefix
SIDE_CHAR = 'l'              # left flow
print(f'dataset: {TAG}; side: {SIDE_CHAR}; data: {DATA_DIR}')


# %% [markdown]
# ## Load prerequisites
#
# flywire ships only the per-neuron `_hit.csv`. Collapse it to one hitting time
# per instance via the **median** over neurons sharing that instance, and save a
# `_per_group.csv` sidecar matching the malecns schema (`instance`, `hitting_time`).

# %%
meta = pd.read_csv(DATA_DIR / f'{TAG}_meta.csv')
inprop = sp.load_npz(DATA_DIR / f'{TAG}_inprop.npz')

flow_hit = pd.read_csv(
    DATA_DIR / f'{TAG}_{SIDE_CHAR}_flow_{N_FLOW}step_{HIT_THRE}thre_hit.csv'
)
flow_df = flow_hit.groupby('instance', as_index=False)['hitting_time'].median()

per_group_path = DATA_DIR / f'{TAG}_{SIDE_CHAR}_flow_{N_FLOW}step_{HIT_THRE}thre_hit_per_group.csv'
flow_df.to_csv(per_group_path, index=False)

idx_to_instance = dict(zip(meta.idx, meta.instance))
print(f'meta rows: {len(meta):,}; inprop shape: {inprop.shape} nnz={inprop.nnz:,}')
print(f'flow rows (per-neuron): {len(flow_hit):,}; per-instance (median): {len(flow_df):,}')
print(f'wrote {per_group_path}')


# %% [markdown]
# ## Trim & save

# %%
trimmed_inprop = sd.trim_inprop_by_flow(
    inprop, idx_to_instance, flow_df, flow_diff_min=0.0,
)
print(f'trimmed_inprop nnz: {trimmed_inprop.nnz:,}')

lat_prefix = f'{TAG}_{SIDE_CHAR}_lat_flow_'
sd.save_lat_flow_0(trimmed_inprop, DATA_DIR, lat_prefix)

summed = sd.compute_lat_flow_sum(trimmed_inprop)
sd.save_lat_flow_sum(summed, DATA_DIR, lat_prefix)
print(f'summed nnz: {summed.nnz:,}')
print(f'wrote {DATA_DIR}/{lat_prefix}0.npz and {DATA_DIR}/{lat_prefix}sum.npz')
