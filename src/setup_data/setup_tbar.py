# Query and cache the presynaptic T-bar (tbar) synapse locations for the
# VPN + VCBN neurons, plus the per-ROI central-brain tbar counts. These are the
# heavy, run-once neuprint queries that feed the visual-input contribution /
# resolution / coverage projection panels (Main Fig. 5g/6a/6b, SI Fig. 11/12)
# rendered in make_figures/paper_figures.ipynb, and the per-instance galleries.
#
# Extracted from the old quan_propagation/contrib_plot_mip + pokemon_plot
# notebooks (the two cells marked "run once, ~100 min" and "run once, ~9 hours").
# Run AFTER assemble_vpn_vcbn.py (needs vp_cb_hit_vic_{r,l}.p).

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
import neuprint
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC

from utils.config import (CACHE_DIR, DATA_DIR, FIG_DIR)
from utils.query_roi import get_primary_rois

cache_dir = CACHE_DIR / 'quan_propagation'
cache_dir.mkdir(parents=True, exist_ok=True)
result_dir = FIG_DIR / 'quan_propagation'
result_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Central-brain ROIs

# %%
rois_cb = get_primary_rois('CentralBrain')
len(rois_cb)

# %% [markdown]
# # Load vpn + vcbn (from assemble_vpn_vcbn.py)

# %%
vp_cb_r = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_r.p'))
vp_cb_l = pd.read_pickle(Path(DATA_DIR, 'vp_cb_hit_vic_l.p'))

ids_lr = list(set(vp_cb_r['bodyId']).union(set(vp_cb_l['bodyId'])))
len(ids_lr)

# %% [markdown]
# # Query tbar in cb + vpn, run once, ~ 100 min

# %%
syn = neuprint.fetch_synapses(NC(bodyId=ids_lr), SC(rois=rois_cb, primary_only=True, type='pre'))

# save pickle
syn.to_pickle(Path(cache_dir, 'tbar_vp_cb.pkl'))

# %% [markdown]
# # Sample tbar points for the R alpha-shape outline step

# %%
# 50k-point sample written to csv -> feeds the (offline, R) alpha-shape that
# produces params/outline_cb_{1,2}.csv used as the projection backgrounds.
xy = syn[['x', 'y']].sample(n=50000, random_state=42)
xy.to_csv(Path(cache_dir, 'tbar_cb_50k.csv'), index=False)

# %% [markdown]
# # CB ROIs tbar counts, run once, ~ 9 hours

# %%
# get syn count for each roi and save into a df
tbar_count_roi_cb = pd.DataFrame(columns=['roi', 'tbar_count'])
for roi in rois_cb:
    ss = neuprint.fetch_synapses(NC(), SC(rois=[roi], primary_only=True, type='pre'))
    tbar_count_roi_cb = pd.concat(
        [tbar_count_roi_cb, pd.DataFrame({'roi': [roi], 'tbar_count': [len(ss)]})],
        ignore_index=True,
    )

# save tbar_count_roi_cb
tbar_count_roi_cb.to_csv(Path(result_dir, 'tbar_count_roi_cb.csv'), index=False)

# %% [markdown]
# # End
