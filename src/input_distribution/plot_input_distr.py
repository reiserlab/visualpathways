# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
import numpy as np
import plotly.graph_objects as go

from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses, fetch_synapse_connections, fetch_neurons
from utils.hex_geometry import find_depth, find_hex_ids, find_neuron_hex_ids, load_depth_bins
from utils.query_functions import fetch_top_input_instances
from utils.input_distr_functions import find_1d_input_distrs, fetch_n_find_rel_hex, find_rel_hex
from utils.plotting_functions import plot_1d_polar, plot_1d_distr, plot_col_heatmap

# %%
instance_name = 'T4c'+'_R'
neuropil = 'ME(R)'
hex0=45

# %% [markdown]
# ### home column for post-synapses

# %%
syn_df = fetch_synapses(NC(instance=instance_name), SC(rois=[neuropil], type='post'))

hex_df = find_neuron_hex_ids(syn_df, roi_str=neuropil, method='majority')#, method='majority') #, method='COM')
hex_red_df = hex_df[['bodyId','hex1_id','hex2_id']]\
    .rename(columns={'bodyId': 'bodyId_post'})


# %%
n_out = hex_red_df['bodyId_post'].nunique()
print('Number of %s neurons: %d'%(instance_name, n_out))

# %% [markdown]
# ### distribution of all input connections

# %%
conn_control_df = fetch_synapse_connections(None, \
                                    NC(instance=instance_name) ,\
                                    SC(rois=[neuropil]))
syn_control_df = conn_control_df[['bodyId_pre','x_post','y_post','z_post','bodyId_post']]\
    .rename(columns={'x_post': 'x', 'y_post': 'y', 'z_post': 'z'})

hex_control_df = find_rel_hex(hex_red_df, syn_control_df, neuropil)
hex_control_df['hex_names'] = 100*(hex_control_df['hex1_id']+hex0) + hex_control_df['hex2_id']+hex0

# %%
hex_bid_control_df = hex_control_df.groupby(['bodyId_post','hex_names'])['bodyId_pre'].count().reset_index(name='count')
hex_bid_control_df['frac'] = hex_bid_control_df.groupby('bodyId_post')['count'].transform(lambda x: x/x.sum())
print('Mean number of inputs: %.1f'%(hex_bid_control_df.groupby('bodyId_post')['count'].sum().mean()))

hex_count_control_df = hex_bid_control_df.groupby('hex_names')['frac'].sum().reset_index(name='frac')
hex_count_control_df['frac'] = hex_count_control_df['frac']/n_out
hex_count_control_df['hex1_id'] = (hex_count_control_df['hex_names'].values/100).astype(int)-hex0
hex_count_control_df['hex2_id'] = hex_count_control_df['hex_names']-100*(hex_count_control_df['hex1_id']+hex0)-hex0

# %%
hex_count_control_df['frac'].sum()

# %%
hex_area = (hex_count_control_df['hex1_id'].max()-hex_count_control_df['hex1_id'].min())*(hex_count_control_df['hex2_id'].max()-hex_count_control_df['hex2_id'].min())
print('Hex area: %d'%hex_area)

if hex_area>300:
    hex_size = 'large'
elif (hex_area<=300)&(hex_area>50):
    hex_size = 'medium'
else:
    hex_size = 'small'

# %%
hex_size = 'small'

# %%
cmax=np.max([0.1,hex_count_control_df['frac'].max()])
cmax

# %%
fig = plot_col_heatmap(hex_count_control_df, 'frac', hex_size, cmax=cmax)
# fig.update_layout(title=dict(text=f'Mean of all connections into {instance_name[:-2]}', x=0.5, y=0.9))
fig.update_layout(title=dict(text=f'Mean of all connections into {instance_name[:-2]}', x=0.5, y=0.9))

fig.show()
fig.write_image(data_path / f'{instance_name}_{neuropil[:-3]}_from_all_2d.pdf')
fig.write_image(data_path / f'{instance_name}_{neuropil[:-3]}_from_all_2d.png')

# %% [markdown]
# ### distribution of input connections from one cell type

# %%
input_instance = 'Mi4'+'_R'

# %%
hex_instance_df = fetch_n_find_rel_hex(hex_red_df, input_instance, instance_name, neuropil, batch_size=10)
hex_instance_df['hex_names'] = 100*(hex_instance_df['hex1_id']+hex0) + hex_instance_df['hex2_id']+hex0

# %%
hex_bid_instance_df = hex_instance_df.groupby(['bodyId_post','hex_names'])['bodyId_pre'].count().reset_index(name='count')
hex_bid_instance_df['frac'] = hex_bid_instance_df.groupby('bodyId_post')['count'].transform(lambda x: x/x.sum())
print('Mean number of inputs: %.1f'%(hex_bid_instance_df.groupby('bodyId_post')['count'].sum().mean()))

hex_count_instance_df = hex_bid_instance_df.groupby('hex_names')['frac'].sum().reset_index(name='frac')
hex_count_instance_df['frac'] = hex_count_instance_df['frac']/n_out
hex_count_instance_df['hex1_id'] = (hex_count_instance_df['hex_names'].values/100).astype(int)-hex0
hex_count_instance_df['hex2_id'] = hex_count_instance_df['hex_names']-100*(hex_count_instance_df['hex1_id']+hex0)-hex0

# %%
fig = plot_col_heatmap(hex_count_instance_df, 'frac', hex_size, cmax=cmax)
fig.update_layout(title=dict(text=f'Mean of {input_instance[:-2]} connections into {instance_name[:-2]}', x=0.5, y=0.9))

fig.show()
fig.write_image(data_path / f'{instance_name}_{neuropil[:-3]}_from_{input_instance[:-2]}_2d.pdf')
fig.write_image(data_path / f'{instance_name}_{neuropil[:-3]}_from_{input_instance[:-2]}_2d.png')

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### radial and angular distributions

# %%
# rad_dist_control_df, angle_dist_control_df = find_1d_input_distrs(hex_control_df)
rad_dist_df, angle_dist_df = find_1d_input_distrs(hex_instance_df)

# %%
fig = plot_1d_distr(rad_dist_df, 'radius', 'frac')

fig.show()
fig.write_image(data_path / f'{instance_name}_{neuropil[:-3]}_from_{input_instance[:-2]}_rad.pdf')
fig.write_image(data_path / f'{instance_name}_{neuropil[:-3]}_from_{input_instance[:-2]}_rad.png')

# %%
fig2 = plot_1d_polar(angle_dist_df, 'angle', 'frac')

fig2.show()
fig.write_image(data_path / f'{instance_name}_{neuropil[:-3]}_from_{input_instance[:-2]}_ang.pdf')
fig.write_image(data_path / f'{instance_name}_{neuropil[:-3]}_from_{input_instance[:-2]}_ang.png')

# %%

# %%
