# Standalone script to query data

# %% Project setup
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
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
import pickle
import numpy as np
import pandas as pd

# %% Import EM libraries
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC
from neuprint import fetch_adjacencies, fetch_synapses, fetch_neurons

# This library wasn't installed before, you might need to rerun library installation
import navis
import navis.interfaces.neuprint as neu

# %%
from queries.completeness import fetch_ol_types

# %%
from utils.config import DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %%
# check source
import inspect

x = inspect.getsource(neu.fetch_mesh_neuron)
print(x)

# %%
# roi
# neu.fetch_all_rois()
# neu.fetch_primary_rois()

neu.fetch_roi('ME(R)')
neu.fetch_roi('vnc-shell')


# %%
# get neuropil meshes from neuprint
from utils.roi import ROI

vols = []
for name in ['OL(R)', 'ME(L)', 'FB']:
    v = ROI(name).get_volume()
    v.color=(1, .95, .95, 0.3) # rgb and alpha
    vols.append(v)

# %%
# get roi mesh from neuroglancer / cloudvolume

import cloudvolume as cv

vol = cv.CloudVolume('precomputed://gs://flyem-cns-roi-7c971aa681da83f9a074a1f0e8ef60f4/fullbrain-major-shells', use_https=True, progress=False)

m = vol.mesh.get([1,2])

print(np.min(m.vertices, axis=0)/8)
print(np.max(m.vertices, axis=0)/8)

# m.vertices = m.vertices / 8 # nm -> voxels

# %%
# get roi mesh from neuroglancer / cloudvolume

import cloudvolume as cv
import os

vol = cv.CloudVolume(os.environ['SHELL_SOURCE'], use_https=True, progress=False)

m = vol.mesh.get([1])

print(np.min(m.vertices, axis=0)/8)
print(np.max(m.vertices, axis=0)/8)

# m.vertices = m.vertices / 8 # nm -> voxels

# %%
print(os.environ['SHELL_SOURCE'])

# %% [markdown]
# NG link  
# https://clio-ng.janelia.org/ with source  
# precomputed://gs://flyem-cns-roi-7c971aa681da83f9a074a1f0e8ef60f4/brain-shell-v2.2

# %%
# from dvid,  not working ?? 

import dvid 
dvid.setup('http://emdata6.int.janelia.org:9000/', ':master', 'zhaoa')
all = dvid.get_roi('all_brain', form="MESH")

# %%
# get neuron mesh using cv
import os
vol_n = cv.CloudVolume(os.environ['SEGMENTATION_SOURCE'], use_https=True, progress=False)
m_n = vol_n.mesh.get([36034])



# %% [markdown]
# ### Query all OL types and instances

# %%
from utils.ol_types import OLTypes

olt = OLTypes(include_tbd=True, include_placeholder=True)
oltypes = olt.get_neuron_list(primary_classification=['OL', 'non-OL'], side='both')

# combine type and hemisphere
oltypes['instance'] = oltypes['type'] + '_' + oltypes['hemisphere']

# keep only type, instance, and main_groups
oltypes = oltypes[['type', 'instance', 'main_groups']]

# add R7_unclear, R8_unclear, and R7R8_unclear to type
df_add = pd.DataFrame({'type': ['R7_unclear', 'R8_unclear', 'R7R8_unclear'],
                       'instance': ['R7_unclear_R', 'R8_unclear_R', 'R7R8_unclear_R'],
                        'main_groups': ['OL_intrinsic', 'OL_intrinsic', 'OL_intrinsic']
                    })

oltypes = pd.concat([oltypes, df_add], ignore_index=True)

# oltypes.main_groups.value_counts(dropna=False)

# check duplicates
# oltypes[oltypes.duplicated(subset=['instance'], keep=False)]

# %% [markdown]
# ### Fetch neuron, connection, and synapse info

# %%
neuron_df, conn_roi_df = fetch_neurons(NC(instance = oltypes['instance'].tolist()))

# add main_groups
neuron_df = pd.merge(
    neuron_df[['bodyId','instance', 'type', 'pre','post','downstream','upstream','consensusNt','predictedNt','celltypePredictedNt']],
    oltypes[['instance', 'main_groups']], 
    on = 'instance', how = 'left'
)

# connections
_, adj_df = fetch_adjacencies(
    sources= NC(instance=oltypes['instance']),
    targets= NC(instance=oltypes['instance']), 
    rois=['OL(R)'], include_nonprimary=True,
    batch_size=1000)

# synapses
syn = fetch_synapses(NC(bodyId=neuron_df['bodyId']), SC(type='', primary_only=True))
# syn = fetch_synapses(NC(bodyId=neuron_df['bodyId']), SC(type='', primary_only=True), nt=False) #with modified function

# # check duplicates
# neuron_df[neuron_df.duplicated(subset=['bodyId'], keep=False)]

# save and load
# oltypes.to_pickle(Path(DATA_DIR, 'oltypes.pkl'))
# conn_roi_df.to_pickle(Path(DATA_DIR, 'conn_roi.pkl'))
# neuron_df.to_pickle(Path(DATA_DIR, 'neuron_info.pkl'))
# adj_df.to_pickle(Path(DATA_DIR, 'edgelist_ol.pkl'))
# syn.to_pickle(Path(DATA_DIR, 'syn.pkl'))

# oltypes = pd.read_pickle(Path(DATA_DIR, 'oltypes.pkl'))
# conn_roi_df = pd.read_pickle(Path(DATA_DIR, 'conn_roi.pkl'))
# neuron_df = pd.read_pickle(Path(DATA_DIR, 'neuron_info.pkl'))
# adj_df = pd.read_pickle(Path(DATA_DIR, 'edgelist_ol.pkl'))
# syn = pd.read_pickle(Path(DATA_DIR, 'syn.pkl'))

# %% [markdown]
# ### Query synapses with cipher

# %%
# one cell
body_ids = [126435]

cql = f"""
        WITH {list(body_ids)} as bodyIds
        MATCH
            (n:Neuron)-[:Contains]->(ss:SynapseSet),
            (ss)-[:Contains]->(s:Synapse)

        WHERE
            n.bodyId in bodyIds
            AND s.type = 'pre'
        
        // De-duplicate 's' because 'pre' synapses can appear in more than one SynapseSet
        WITH DISTINCT n, s

        RETURN n.bodyId as bodyId,
                s.type as type,
                s.location.x as x,
                s.location.y as y,
                s.location.z as z,
                s as props, // All properties, as json
                //s['OL(R)'] as OL,
                s.ntAcetylcholineProb as ntAcetylcholineProb,
                s.ntDopamineProb as ntDopamineProb,
                s.ntGabaProb as ntGabaProb,
                s.ntGlutamateProb as ntGlutamateProb,
                s.ntSerotoninProb as ntSerotoninProb,
                s.ntHistamineProb as ntHistamineProb,
                s.ntOctopamineProb as ntOctopamineProb

        ORDER BY bodyId
        """

nt_syn = c.fetch_custom(cql)


# %%
# plot a bar graph for the prob in the first row of nt_syn 
import matplotlib.pyplot as plt

nt_syn.iloc[0, [6,9,8,11,7,12,10]].plot(kind='bar')

# rename xticks to be the neurotransmitter names
plt.xticks(np.arange(7), ['ACh', 'Glu', 'GABA', 'His', 'Dop', 'OA', '5HT'], rotation=0)


# %%
# convert the previous cell into a function
from textwrap import dedent

def fetch_nt_synapses(client, body_ids):
    cql = dedent(f"""
            WITH {list(body_ids)} as bodyIds
            MATCH
                (n:Neuron)-[:Contains]->(ss:SynapseSet),
                (ss)-[:Contains]->(s:Synapse)

            WHERE
                n.bodyId in bodyIds
                AND s.type = 'pre'
            
            // De-duplicate 's' because 'pre' synapses can appear in more than one SynapseSet
            WITH DISTINCT n, s

            RETURN n.bodyId as bodyId,
                    s.type as type,
                    s.location.x as x,
                    s.location.y as y,
                    s.location.z as z,
                    //s as props // All properties, as json
                    s.ntAcetylcholineProb as ntAcetylcholineProb,
                    s.ntGlutamateProb as ntGlutamateProb,
                    s.ntGabaProb as ntGabaProb,
                    s.ntHistamineProb as ntHistamineProb,
                    s.ntDopamineProb as ntDopamineProb,
                    s.ntOctopamineProb as ntOctopamineProb,
                    s.ntSerotoninProb as ntSerotoninProb

            ORDER BY bodyId
            """)
    data = client.fetch_custom(cql, format='json')['data']
    # data = client.fetch_custom(cql, format='json')

    return pd.DataFrame(
        data,
        columns=['bodyId', 'type', 'x', 'y', 'z', 'ACh', 'Glu', 'GABA', 'His', 'Dop', 'OA', '5HT']
        )   
    

# divide a list into chunks, the last chuck include all that's left, and iterate over them
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


# %%
# query in batches
batch_dfs = []
for batch_bodies in chunker(neuron_df['bodyId'], 10):
# for batch_bodies in chunker(neurons['bodyId'], 10):
    batch_df = fetch_nt_synapses(c, batch_bodies)
    batch_dfs.append( batch_df )

syn = pd.concat( batch_dfs, ignore_index=True )



# %% [markdown]
# ### Parallelize fetch, NOT WORKING

# %%
import multiprocessing as mp

# By using the global fetch_custom instead of c.fetch_custom(),
# We allow neuprint-python to automatically create/select a
# separate Client object for each process.
# (Otherwise, there can be issues when sharing a Client across processes.)
from neuprint import fetch_custom

def fetch_syn(body):
    cql = f"""
        MATCH
            (n:Neuron)-[:Contains]->(ss:SynapseSet),
            (ss)-[:Contains]->(s:Synapse)

        WHERE
            n.bodyId = {body}
            AND s.type = 'pre'

        // De-duplicate 's' because 'pre' synapses can appear in more than one SynapseSet
        WITH DISTINCT n, s

        RETURN n.bodyId as bodyId,
                s.type as type,
                s.location.x as x,
                s.location.y as y,
                s.location.z as z,
                s as props // All properties, as json

        ORDER BY bodyId
        """
    return fetch_custom(cql)


# %%
from tqdm import tqdm
body_ids = neuron_df.sample(10).bodyId

# pool = mp.Pool(16)
pool = mp.Pool(4)
with pool:
    results = list(tqdm(pool.imap_unordered(fetch_syn, body_ids), total=len(body_ids)))
df = pd.concat(results, ignore_index=True)