# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: ol-c-kernel
#     language: python
#     name: ol-c-kernel
# ---

# %%
from pathlib import Path
import os
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))
data_path = PROJECT_ROOT / 'results' / 'input_distribution'

from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
from utils.input_distr_functions import store_input_of_ol

# %%
#number of inputs to consider
limit = 7
#fraction in one neuropil
thre = 0.4

# %%
store_input_of_ol(limit, thre, i0=280)

# %%

# %%
