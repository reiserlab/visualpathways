# %%

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
import pandas as pd
import numpy as np
import re
import pickle
import alphashape
from shapely.geometry import LineString
import matplotlib.pyplot as plt

# %%
result_dir = PROJECT_ROOT / 'results' / 'eyemap'
result_dir.mkdir(parents=True, exist_ok=True)

# %%
# edge column
from utils.hex_geometry import fl_get_edge_ids
from utils.hex_hex import all_hex_df

# center = [hex1, hex2] = [18, 19]
hex_offset = [18, 19]

all_hex = all_hex_df()
all_hex['q'] = all_hex['hex1_id'] - hex_offset[0]
all_hex['p'] = all_hex['hex2_id'] - hex_offset[1]

all_hex['h'] = all_hex['q'] - all_hex['p']
all_hex['v'] = all_hex['p'] + all_hex['q']


# %%
# plot the edge ids
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(all_hex['h'], all_hex['v'], 'o', markersize=1)
ax.set_aspect('equal')
plt.show()

# %%
# save csv
all_hex.to_csv(Path(result_dir, 'hexcoord_right.csv'), index=False)
