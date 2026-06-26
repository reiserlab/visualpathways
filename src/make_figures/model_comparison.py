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
#     display_name: Python (venv)
#     language: python
#     name: python-venv
# ---

# %% [markdown]
# # Model comparison
#
# Compares the default model (`config.DATASET` on `config.SIDE_CHAR`) against an alternative model across five quantities:
#
# 1. per-instance layer (hitting time)
# 2. VPN clusters — ARI + optimal matching
# 3. OL propagated RF size
# 4. VIC for VPN and CB neurons
# 5. VCBN clusters — ARI + optimal matching
#
# Alt files live under `data/model_comparison/<ALT>/` and share the same basenames as the default files being compared. Set `ALT` below.
#

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().resolve().parent if Path.cwd().name == 'make_figures' else Path.cwd().resolve()))

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix

from utils import config
from utils.palettes import load_colors
from utils import cb_data, core_data, ol_data, ol_rf, vic
from utils import plotting as plot
from utils.ol_rf import _clean_rf_params

ALT = 'flow2'
ALT_DIR = config.DATA_DIR / 'model_comparison' / ALT
SAVE_DIR = config.FIG_DIR / 'model_comparison' / ALT
colored_region, colored_main_groups, _, colored_seed = load_colors()

meta_ol = core_data.get_ol_meta()

def _cluster_match(cl_a, cl_b, tick_prefix='c'):
    """Merge two cluster frames on `instance`; return ARI + confusion + Hungarian assignment.
    Shape matches `ol_data.get_ol_lr_cluster_match` so `plot_lr_cluster_confusion` can consume it."""
    m = cl_a[['instance', 'cluster']].merge(
        cl_b[['instance', 'cluster']], on='instance', how='inner', suffixes=('_a', '_b'),
    )
    ari = adjusted_rand_score(m['cluster_a'].values, m['cluster_b'].values)
    conf = confusion_matrix(m['cluster_a'].values, m['cluster_b'].values)
    row_ind, col_ind = linear_sum_assignment(-conf)
    n_a = int(m['cluster_a'].max())
    n_b = int(m['cluster_b'].max())
    n = max(n_a, n_b)
    return {
        'ari': ari, 'n_common': len(m),
        'confusion': conf, 'row_ind': row_ind, 'col_ind': col_ind,
        'n_clu_L': n_a, 'n_clu_R': n_b,
        'tick_labels': [f'{tick_prefix}{i}' for i in range(1, n + 1)],
    }


def _relabel_alt_to_match(cl_alt, match, max_def):
    """Rename alt clusters so Hungarian-matched pairs share the default's label.
    Unmatched alt clusters get labels `max_def+1, max_def+2, ...`."""
    alt_to_def = {c + 1: r + 1 for r, c in zip(match['row_ind'], match['col_ind'])}
    extra = sorted(set(cl_alt['cluster'].unique()) - set(alt_to_def))
    for i, c in enumerate(extra, start=max_def + 1):
        alt_to_def[c] = i
    out = cl_alt.copy()
    out['cluster'] = out['cluster'].map(alt_to_def).astype(int)
    return out



# %% [markdown]
# ## 1. Layer (hitting time)
#

# %%
flow_name = f'{config.DATASET}_{config.SIDE_CHAR}_flow_{config.N_FLOW}step_{config.HIT_THRE}thre_hit_per_group.csv'
flow_def = core_data.get_flow_per_group()
flow_alt = pd.read_csv(ALT_DIR / flow_name)

df = (
    flow_def.merge(flow_alt, on='instance', suffixes=('_def', '_alt'))
            .merge(meta_ol[['instance', 'main_groups']].drop_duplicates(),
                   on='instance', how='inner')
)

plot.plot_comparison_scatter(
    df, 'hitting_time_def', 'hitting_time_alt',
    'main_groups', colored_main_groups,
    xaxis_title=f'{config.DATASET} layer', yaxis_title=f'{ALT} layer',
    diag_range=[0.9, 5], axis_range=[0.9, 5],
    filename_stem=f'{config.FIG_DATASET}_vs_{ALT}_hitting',
    save_dir=SAVE_DIR,
)


# %% [markdown]
# ## 2. VPN clusters
#

# %%
cl_def = ol_data.get_ol_clusters(frac_thre=0.19)
cl_alt = pd.read_csv(ALT_DIR / f'{config.DATASET}_in_out_clusters.csv')

match = _cluster_match(cl_def, cl_alt, tick_prefix='c')
print(f'Number of consistent VPNs: {match["n_common"]}')
print(f'ARI: {match["ari"]:.2f}')

plot.plot_lr_cluster_confusion(
    match,
    dataset=f'{config.FIG_DATASET}_vs_{ALT}',
    filename_stem='vpn_confusion',
    xaxis_title=f'{ALT} cluster', yaxis_title=f'{config.DATASET} cluster',
    save_dir=SAVE_DIR,
)

print(f'Optimal matching (n_def={match["n_clu_L"]}, n_alt={match["n_clu_R"]}):')
for r, c in zip(match['row_ind'], match['col_ind']):
    print(f'  default c{r + 1}  ->  alt c{c + 1}')

# Stacked bar of the alt's per-cluster pathway features, re-numbered to match default
feat_alt = pd.read_csv(ALT_DIR / f'{config.DATASET}_in_out_features.csv', index_col=0)
feat_alt.index.name = 'instance'
cl_alt_matched = _relabel_alt_to_match(cl_alt, match, max_def=match['n_clu_L'])
cluster_num_alt = cl_alt_matched['cluster'].value_counts().sort_index()

plot.plot_clusters_pathway_features(
    feat_alt, list(feat_alt.columns), cl_alt_matched, cluster_num_alt, colored_seed,
    dataset=f'{config.FIG_DATASET}_vs_{ALT}', save_dir=SAVE_DIR,
)


# %% [markdown]
# ## 3. OL propagated RF size
#

# %%
rf_def = ol_rf.get_rf_type_ol().groupby('instance')['size'].median().reset_index()
rf_alt = (
    _clean_rf_params(pd.read_csv(ALT_DIR / f'{config.DATASET}_types_fit_rf.csv'))
    .groupby('instance')['size'].median().reset_index()
)

df = (
    rf_def.merge(rf_alt, on='instance', suffixes=('_def', '_alt'))
          .merge(meta_ol[['instance', 'main_groups']].drop_duplicates(),
                 on='instance', how='inner')
)

plot.plot_comparison_scatter(
    df, 'size_def', 'size_alt',
    'main_groups', colored_main_groups,
    xaxis_title=f'{config.DATASET} RF size (col)', yaxis_title=f'{ALT} RF size (col)',
    diag_range=[0.5, 1000], axis_range=[np.log10(0.5), np.log10(1000)],
    log_axes=True,
    filename_stem=f'{config.FIG_DATASET}_vs_{ALT}_rf_size',
    save_dir=SAVE_DIR,
)


# %% [markdown]
# ## 4. VIC (VPN + CB)
#

# %%
vic_def = vic.get_ol_cb_vic_type()
vic_alt = pd.read_csv(ALT_DIR / 'vp_cb_vic.csv')

df = vic_def.merge(vic_alt[['instance', 'VIC']], on='instance', suffixes=('_def', '_alt'))

plot.plot_comparison_scatter(
    df, 'VIC_def', 'VIC_alt',
    'region', colored_region,
    xaxis_title=f'{config.DATASET} VIC', yaxis_title=f'{ALT} VIC',
    diag_range=[1e-5, 1], log_axes=True,
    filename_stem=f'{config.FIG_DATASET}_vs_{ALT}_vic',
    save_dir=SAVE_DIR, size=3,
)


# %% [markdown]
# ## 5. VCBN clusters
#

# %%
cl_def = cb_data.get_cb_clusters(frac_thre=0.19)
cl_alt = pd.read_csv(ALT_DIR / f'{config.DATASET}_cb_in_out_clusters.csv')

match = _cluster_match(cl_def, cl_alt, tick_prefix='d')
print(f'Number of consistent VCBNs: {match["n_common"]}')
print(f'ARI: {match["ari"]:.2f}')

plot.plot_cb_lr_cluster_confusion(
    match,
    dataset=f'{config.FIG_DATASET}_vs_{ALT}',
    filename_stem='vcbn_confusion',
    xaxis_title=f'{ALT} cluster', yaxis_title=f'{config.DATASET} cluster',
    save_dir=SAVE_DIR,
)

print(f'Optimal matching (n_def={match["n_clu_L"]}, n_alt={match["n_clu_R"]}):')
for r, c in zip(match['row_ind'], match['col_ind']):
    print(f'  default d{r + 1}  ->  alt d{c + 1}')

# Stacked bar of the alt's per-cluster pathway features, re-numbered to match default
feat_alt = pd.read_csv(ALT_DIR / f'{config.DATASET}_cb_in_out_features.csv', index_col=0)
feat_alt.index.name = 'instance'
cl_alt_matched = _relabel_alt_to_match(cl_alt, match, max_def=match['n_clu_L'])
cluster_num_alt = cl_alt_matched['cluster'].value_counts().sort_index()

plot.plot_cb_clusters_pathway_features(
    feat_alt, list(feat_alt.columns), cl_alt_matched, cluster_num_alt, colored_seed,
    dataset=f'{config.FIG_DATASET}_vs_{ALT}', save_dir=SAVE_DIR,
)

