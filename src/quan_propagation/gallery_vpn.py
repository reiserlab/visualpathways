# %%
%load_ext autoreload
%autoreload 2

# %% [markdown]
# ### make gallery for VPNs

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
import numpy as np
import scipy.sparse

import navis
from neuprint import fetch_adjacencies
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages


import connectome_interpreter as ci

# %%
from quan_propagation.func import count_col_ahull, plot_extreme_projection_xy
from utils.ol_rf import compute_rf
from utils.plotting_functions import plot_gaussian_params
from utils.plotter import plot_cns, save_figure_anterior

# # Test if count_col is imported
# print(count_col)

# %%
from utils.config import (DATA_DIR, DATASET)

result_dir = Path(PROJECT_ROOT, 'results', 'quan_propagation')
result_dir.mkdir(parents=True, exist_ok=True)

cache_dir = Path(PROJECT_ROOT, 'cache', 'quan_propagation')
cache_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load data

# %% [markdown]
# ## Get meta

# %%
# LOAD Judith meta, this has AN's R7/8, 
meta = pd.read_csv(DATA_DIR / f'{DATASET}_meta.csv')
        
#  add superclass and class
meta.loc[meta['cell_type'].str.contains('^R7|^R8'), 'superclass'] = 'ol_sensory'
meta.loc[meta['cell_type'].str.contains('^R7|^R8'), 'class'] = 'visual'

print(meta.shape)  

# %%
# make L1 excitatory so it's an ON cell 
meta.loc[meta.cell_type == 'L1', 'sign'] = 1

# %%
# make dictionaries that map indices to meta info
idx_to_bodyId = dict(zip(meta.idx, meta.bodyId))
idx_to_coords = dict(zip(meta.idx, meta.coords))
bodyId_to_idx = dict(zip(meta.bodyId, meta.idx))

# %% [markdown]
# ## brain outline 

# %%
# load outline
outline_1 = pd.read_csv(Path(result_dir, f'outline_cb_1.csv'), index_col=False)
outline_2 = pd.read_csv(Path(result_dir, f'outline_cb_2.csv'), index_col=False)

outlines_bkgd = [outline_1,  outline_2]

# %% [markdown]
# ## syn

# %%
syn0 = pd.read_pickle(Path(cache_dir, 'tbar_vp_cb.pkl'))

# %% [markdown]
# ## ht, vic, size
# 

# %%
lr = 'r'
vp_cb_vic = pd.read_pickle(Path(DATA_DIR, f'vp_cb_hit_vic_{lr}.p'))

# meta_ahull = pd.read_pickle(Path(cache_dir, 'meta_ahull_scan.pkl'))
meta_ahull = pd.read_pickle(Path(DATA_DIR, 'meta_VP_CB_ahull_cumsum60.pkl'))

meta_cb_vpn = pd.merge(
    meta_ahull[['bodyId', 'instance', 'ahull_size']],
    vp_cb_vic[['bodyId', 'VIC', 'hitting_time', 'main_groups']],
    on='bodyId', how='inner'
)
print(meta_cb_vpn.shape)

# filter by VIC, per instance 
thr_vic = 5e-4
inst = meta_ahull.groupby('instance').agg({'VIC':'median'}).reset_index()
inst = inst[inst.VIC > thr_vic]['instance']
meta_ahull = meta_ahull[meta_ahull.instance.isin(inst)]

meta_cb_vpn.rename(columns={'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape)

# %%
# load size, ellipse fit and ahull
rf_fit = pd.read_pickle(Path(cache_dir, 'rf_fit_thr07.pkl'))
print(rf_fit.shape)
# meta_ahull = pd.read_pickle(Path(cache_dir, 'meta_ahull_scan.pkl'))

rf_size = pd.merge(
    rf_fit[['instance','bodyId', 'size', 'r2']],
    # meta_ahull[['bodyId', 'ahull_max_0.4', 'ahull_cumsum_0.6']],
    meta_ahull[['bodyId', 'ahull_size']],
    on='bodyId',
    how='inner'
)
print(rf_fit.shape)

# %%
# compute instance-level median
rf_size_median = rf_size.groupby('instance').agg({
    'size': 'median',
    'r2': 'median',
    'ahull_size': 'median',
}).reset_index()

# ht_median = ht.groupby(['instance'])['ht'].median().reset_index()
# vic_median = vic.groupby(['instance', 'main_groups'])['VIC'].median().reset_index()
ht_median = meta_cb_vpn.groupby(['instance', 'main_groups'])['ht'].median().reset_index()
vic_median = meta_cb_vpn.groupby(['instance', 'main_groups'])['VIC'].median().reset_index()

# %%


# %% [markdown]
# ## Load inprop

# %%
step0 = scipy.sparse.load_npz(DATA_DIR / f'{DATASET}_r_lat_flow_0.npz').tocsr()

# %% [markdown]
# ## load effwt

# %%
effwt_visr_5 = pd.read_pickle(Path(cache_dir, 'effwt_visr_5.pkl'))

# %% [markdown]
# ## ol types
# 
# R-dominant ?

# %%
# sort
import re
def natural_sort_key(s):
    """Convert string to list of strings and integers for natural sorting"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', str(s))]

# # reorder rf_size_median by 'instance' value (case-insensitive, natural number sorting)
# rf_size_median = rf_size_median.sort_values(by='instance', key=lambda x: x.map(natural_sort_key)).reset_index(drop=True)

# %%
# cell type table
# oltypes0 = pd.read_excel(Path(PROJECT_ROOT, 'params', 'Sup_Table_1_Cell-types_and_counts_final.xlsx'))
oltypes0 = pd.read_excel(Path(PROJECT_ROOT, 'params', 'Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx'))
oltypes0.rename(columns={'cell type':'cell_type', 'main groups':'main_groups'}, inplace=True)
print(oltypes0.shape)

# oltypes_nonvpn = oltypes0[~oltypes0['main_groups'].str.contains('VPN')]
oltypes_vpn_0 = oltypes0[oltypes0['main_groups'].str.contains('VPN')]
# oltypes_vcn = oltypes0[oltypes0['main_groups'].str.contains('VCN')]
# oltypes_ol = oltypes0[oltypes0['main_groups'].str.contains('^ON')]

# oltypes = pd.read_csv(Path(cache_data_dir, 'fromJudith', 'TableS1_OLR_in_out_summary.csv'))
# oltypes_vpn = oltypes[oltypes['main_groups'].str.contains('VPN')]
# oltypes_vpn = pd.merge(oltypes_vpn, oltypes_vpn_0[['cell_type', 'instance', 'bodyId in figures']], on='instance', how='left')

oltypes = oltypes0.sort_values(by='instance', key=lambda x: x.map(natural_sort_key)).reset_index(drop=True)
oltypes_vpn = oltypes_vpn_0.sort_values(by='instance', key=lambda x: x.map(natural_sort_key)).reset_index(drop=True)
oltypes_vpn.shape

# %%
# inst for gallery
inst_vp = pd.merge(oltypes_vpn, rf_size_median, on='instance', how='left')
print(inst_vp.shape)
inst_vp = pd.merge(inst_vp, ht_median[['instance','ht']], on='instance', how='left')
print(inst_vp.shape)
inst_vp = pd.merge(inst_vp, vic_median[['instance','VIC']], on='instance', how='left')
print(inst_vp.shape)

# %%
# # meta by type
# meta_type_cb_vpn = meta_cb_vpn.groupby('instance', group_keys=True).apply(
#     lambda g: g.sort_values('area_fit').iloc[[len(g) // 2]],
#     include_groups=False
# # ).reset_index(level=0).reset_index(drop=True)[['bodyId', 'instance', 'area_fit', 'r2', 'ht', 'main_groups', 'vision']]
# ).reset_index(level=0).reset_index(drop=True)[['bodyId', 'instance', 'area_fit', 'r2',  'vision']]

# meta_type_cb_vpn['bodyId_count'] = meta_cb_vpn.groupby('instance')['bodyId'].count().loc[meta_type_cb_vpn['instance'].values].values

# # # reorder to match original column style
# # meta_type_cb_vpn = meta_type_cb_vpn[['bodyId_count', 'bodyId', 'instance', 'area_fit', 'r2', 'vision', 'ht', 'main_groups']]

# %% [markdown]
# ## VP cluster

# %%
from utils import ol_data
FRAC = 0.19
clusters  = ol_data.get_ol_clusters(frac_thre=FRAC)

# %% [markdown]
# ## VCBN ids

# %%
# central-brain (non-OL) candidate target bodyIds
cb_ids = meta_cb_vpn.loc[meta_cb_vpn['main_groups'] == 'nonOL', 'bodyId'].unique()

# %%


# %% [markdown]
# # Main loop

# %%
import io
from PIL import Image
import fitz  # PyMuPDF

# NOTE: mpl_to_pdf_bytes (the heatmap-rasterising matplotlib->PDF exporter) is
# defined in the CONFIG cell ("## make pdf") so it always reloads with the dpi
# setting. The other export helpers live here.

def plotly_to_pdf_bytes(fig):
    """Plotly Figure -> vector PDF bytes (needs kaleido; same dep your PNG path used)."""
    return fig.to_image(format='pdf')

def place(page, pdf_bytes, rect, title=None, subtext=None):
    """Tile a single-page PDF into `rect` (keeps vectors, preserves aspect)."""
    src = fitz.open("pdf", pdf_bytes)
    page.show_pdf_page(rect, src, 0)          # keep_proportion=True by default
    src.close()
    if title:
        page.insert_text((rect.x0 + 4, rect.y0 + 11), title, fontsize=9)
    if subtext:
        lines = subtext.split('\n') if isinstance(subtext, str) else list(subtext)
        fs, line_h = 6, 6 * 1.4
        # draw bottom-up: last line sits ~4 pt above rect.y1, earlier lines stack above
        for k, line in enumerate(reversed(lines)):
            page.insert_text((rect.x0 + 4, rect.y1 - 50 - k * line_h), line, fontsize=fs)

def place_image(page, img_path, rect, title=None):
    """Place a raster image (PNG) into `rect`, preserving aspect ratio."""
    page.insert_image(rect, filename=str(img_path), keep_proportion=True)
    if title:
        page.insert_text((rect.x0 + 4, rect.y0 + 11), title, fontsize=9)


def _fmt(v, fmt="{:.2f}"):
    """Format a possibly-NaN value for the stats table."""
    if pd.isna(v):
        return "n/a"
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)

def make_stats_table_fig(row, figsize=(3.2, 2.6)):
    """Build a matplotlib table of per-instance stats (vector, for `place`)."""
    rows = [
        ("instance",           str(row['instance'])),
        ("cell count",         _fmt(row['no. of cells'], "{:.0f}")),
        ("pred. NT",           str(row['predicted neurotransmitter'])),
        ("median hitting time",    _fmt(row['ht'])),
        ("median VIC",         _fmt(row['VIC'], "{:.4f}")),
        ("ellipse size (#col)",    _fmt(row['size'], "{:.1f}")),
        ("median r²",     _fmt(row['r2'])),
        ("α-hull size (#col)",  _fmt(row['ahull_size'], "{:.1f}")),
    ]
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=['metric', 'value'],
                   cellLoc='left', colLoc='left', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('0.85')
        if r == 0:                       # header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('0.92')
    fig.tight_layout(pad=0.2)
    return fig

def make_vic_hop_figs(
    inst, *, step0, syn0, meta_cb_vpn, cb_ids,
    bodyId_to_idx, idx_to_bodyId, outlines_bkgd,
    conn_thr=100, min_weight=1, top_n_max=5,
):
    """Build three central-brain VIC max-projection heatmaps for one instance.

    Panels (left -> right on the gallery's bottom row):
      * T0 : the VPN population itself, colored by per-cell VIC.
      * T1 : its direct (1-hop) central-brain targets.
      * T2 : the next hop -- targets of the *curated* T1 population.

    Both hops reuse a single propagation matrix `step0` (the 1-hop in-proportion
    matrix A): T1 is seeded from T0, and T2 is seeded from the T1 result and has
    `step0` applied once more. This chains the populations actually shown in the
    panels and avoids materializing A @ A. Consequently T2 only counts 2-hop
    paths through the displayed T1 set (central-brain, multi-synapse, past the
    >conn_thr connection gate); paths relaying through OL or pruned neurons are
    intentionally excluded.

    PERF: up to 2 live neuprint `fetch_adjacencies` calls per instance (to drop
    single-synapse partners and enforce the >conn_thr connection gate). For large
    galleries this dominates runtime; caching partner weights keyed on
    (sources, targets) is where to optimize later.

    Heatmap spatial resolution is HEATMAP_XNBINS x HEATMAP_YNBINS (CONFIG cell).

    Returns a dict:
      t0/t1/t2        : matplotlib Figures (or None if that hop has no data)
      n_t0/n_t1/n_t2  : cell counts for the panel titles
      top_inst_t1/t2  : up to `top_n_max` "instance: median-value" strings
    """
    out = dict(t0=None, t1=None, t2=None, n_t0=0, n_t1=None, n_t2=None,
               top_inst_t1=[], top_inst_t2=[])

    # bodyIds of the VPN instance itself (the T0 seed population)
    ids_t0 = meta_cb_vpn.loc[meta_cb_vpn.instance == inst, 'bodyId'].unique()
    out['n_t0'] = len(ids_t0)
    if len(ids_t0) == 0:
        return out

    def _project(df_plot, pct_hi, label):
        """Render one max-projection heatmap; `pct_hi` sets the upper color clip
        (color floor is fixed at 0)."""
        df_plot['val'] = df_plot['value'] if 'value' in df_plot else df_plot['VIC']
        vmax = np.nanpercentile(df_plot['val'], pct_hi) if len(df_plot) else 0
        fig, _, _ = plot_extreme_projection_xy(
            df_plot, outlines_bkgd,
            xrange=[20000, 80000], yrange=[0, 50000],
            xnbins=HEATMAP_XNBINS, ynbins=HEATMAP_YNBINS,
            agg='largest', agg_frac=0.1, im_norm='linear',
            vmin=0, vmax=vmax, colorbar_label=label)
        return fig

    def _downstream(seed_ids, seed_weights, exclude_ids):
        sidx = [bodyId_to_idx[i] for i in seed_ids]
        summed = seed_weights @ step0[sidx, :]      # row-vec × matrix -> 1-D ndarray
        dn = summed.nonzero()[0]
        d = pd.DataFrame({'bodyId': [idx_to_bodyId[i] for i in dn],
                          'value': summed[dn]})
        d = d[d['bodyId'].isin(cb_ids) & ~d['bodyId'].isin(exclude_ids)]
        return d.sort_values('value', ascending=False)

    def get_adj_with_thr(sources, targets):
        """Live neuprint check: keep only targets whose summed synaptic weight
        from `sources` exceeds `min_weight` (drops single-synapse partners)."""
        _, c = fetch_adjacencies(sources=sources, targets=targets,
                                 include_nonprimary=True, rois=['CentralBrain'])
        c = c.groupby('bodyId_post').agg({'weight': 'sum'}).reset_index()
        return c[c['weight'] > min_weight]

    def _add_instance(df):
        """Attach each target's instance and build the 'instance: value' list
        (ranked by median propagated value) used for the panel annotation."""
        df = df.copy()
        df['instance'] = df['bodyId'].map(
            lambda x: meta_cb_vpn.loc[meta_cb_vpn.bodyId == x, 'instance'].values[0])
        df_inst = (df.groupby('instance').agg({'value': 'median'})
                   .reset_index().sort_values('value', ascending=False))
        top = [f"{r['instance']}: {r['value']:.2g}"
               for _, r in df_inst.head(min(top_n_max, df_inst.shape[0])).iterrows()]
        return df, top

    # --- T0: the VPN population, colored by per-cell VIC ---
    df_plot = pd.merge(
        syn0,
        meta_cb_vpn.loc[meta_cb_vpn.instance == inst, ['instance', 'bodyId', 'VIC']],
        on='bodyId', how='inner')
    out['t0'] = _project(df_plot, 95, 'VIC')

    # first-hop seed weights = per-cell VIC of the VPN population
    vision_values = (meta_cb_vpn[meta_cb_vpn.bodyId.isin(ids_t0)]
                     .set_index('bodyId')['VIC'].to_dict())
    w_t0 = np.array([vision_values[b] for b in ids_t0], dtype=np.float32)

    # --- T1: 1-hop downstream of T0 ---
    df = _downstream(ids_t0, w_t0, ids_t0)
    c1 = get_adj_with_thr(ids_t0, df['bodyId'].unique()) if df.shape[0] > 0 \
        else pd.DataFrame(columns=['bodyId_post', 'weight'])

    if c1['weight'].sum() > conn_thr:                    # gate: skip weak projections
        df = df[df['bodyId'].isin(c1['bodyId_post'].unique())]
        if df.shape[0] > 0:
            df, out['top_inst_t1'] = _add_instance(df)
            out['n_t1'] = df.shape[0]
            out['t1'] = _project(pd.merge(syn0, df, on='bodyId', how='inner'),
                                 95, 'VIC')

            # --- T2: reseed from the curated T1 set, propagate one more hop ---
            ids_t1 = df['bodyId'].values
            w_t1 = df['value'].values.astype(np.float32)   # T1 propagated VIC as weights
            df = _downstream(ids_t1, w_t1, np.concatenate([ids_t0, ids_t1]))
            if df.shape[0] > 0:
                c1 = get_adj_with_thr(ids_t1, df['bodyId'].unique())
                df = df[df['bodyId'].isin(c1['bodyId_post'].unique())]
                if df.shape[0] > 0:
                    df, out['top_inst_t2'] = _add_instance(df)
                    out['n_t2'] = df.shape[0]
                    out['t2'] = _project(pd.merge(syn0, df, on='bodyId', how='inner'),
                                         95, 'VIC')
    return out

# %%
# 2D gaussian, prob mass (cumsum) within one std contour, is ~0.39
# PDF height (max) at one std is ~0.61

# check ahull area calculation using bodyid = 28862.
# check edge_points using 531732
# thr_2dGaussian = 0.4

# pathway plot
from utils import ol_data, core_data
from utils import plotting as plot
from utils.palettes import load_colors
ol_flow = core_data.get_ol_flow_type()
ol_meta     = core_data.get_ol_meta()
colored_region, colored_main_groups, colored_sign, colored_seed = load_colors()
type_to_sign = dict(zip(ol_meta.cell_type, ol_meta.sign))
sign_to_color = colored_sign['color'].to_dict()
type_to_mg = dict(zip(ol_meta.cell_type, colored_main_groups.loc[ol_meta.main_groups, 'color']))

# sampled_meta = meta_type_cb_vpn_sorted.sample(n=500, random_state=43)
# sampled_meta = oltypes[oltypes['cell_type'].isin(['Mi1', 'Mi15', 'T4c', 'T5a','MeTu2a','MeTu3a'])].reset_index(drop=True)
# sampled_meta = sampled_meta[sampled_meta['instance'].isin(['LC4_R', 'LC6_R', 'VS_R','MeTu2a_R','MeTu3a_R'])].reset_index(drop=True)
sampled_meta = inst_vp[inst_vp['instance'].isin(['LC6_R', 'VS_R', 'LC41_R'])].reset_index(drop=True)
# sampled_meta = inst_vp[inst_vp['instance'].isin(oltypes_vpn['instance'].sample(n=3, random_state=42).tolist())].reset_index(drop=True)


# %% [markdown]
# ## make pdf

# %%
# =======================================================================
#  GALLERY CONFIG  --  every tunable knob in one place.
#  Edit values here, then re-run THIS cell -> the build cell -> the index
#  cell. (Re-running this cell is enough; no kernel restart needed.)
# =======================================================================
import io
import matplotlib as mpl

# --- A. FILE SIZE  (size <-> quality trade-offs) -----------------------
# The 3 VIC heatmaps (bottom row) are rasterised; their bytes are the main
# size driver and are set by the savefig DPI -- NOT by the bin count (sec. C).
# Approx full-gallery (~352-page) file size, with the RF hex panel kept vector:
#     dpi 72 -> ~27 MB | 60 -> ~24 | 55 -> ~22 | 50 -> ~20.5
#     dpi 48 -> ~21.5  | 45 -> ~21 | 42 -> ~20 | 40 -> ~19
HEATMAP_DPI      = 48        # lower = smaller file, blockier heatmaps

# Anterior-view (top-left) render size in px, before the white-margin crop.
# The panel is only ~342 pt wide, so ~720 px already ~= 150 dpi. Halving the
# px roughly halves the anterior's ~3.5 MB share (-~1.7 MB).
ANTERIOR_WIDTH   = 720
ANTERIOR_HEIGHT  = 403

# Palette size for the colour-indexing in the size-shrink cell (after the build).
# 256 = full colormap fidelity; 128 ~= -1.5 MB with faint heatmap-gradient banding.
PALETTE_COLORS   = 256

# --- B. PAGE LAYOUT  (points; 72 pt = 1 inch) --------------------------
W, H         = 1080, 620     # page width / height
M, TITLE_H   = 12, 28        # panel margin / top title-bar height
NCOLS, NROWS = 3, 2          # panel grid: TL TM TR / BL BM BR

# --- C. CONTENT / ANALYSIS  (change what's shown, not the file size) ----
HEATMAP_XNBINS, HEATMAP_YNBINS = 240, 200  # VIC heatmap spatial bins (granularity)
CONN_THR       = 100         # min summed synaptic weight for the T0->T1 / T1->T2 hop gates
PATHWAY_TOP_N  = 5           # number of strongest upstream paths drawn (top-middle)
MESH_LOD       = 2           # anterior neuron mesh level-of-detail (higher = coarser/faster)
RF_THR_MODE    = 'cumsum'    # RF alpha-hull threshold mode (top-right)
RF_REMOVE_FRAC = 0.6         # RF alpha-hull cumulative-mass cutoff

# --- apply settings ----------------------------------------------------
mpl.rcParams['savefig.dpi'] = HEATMAP_DPI

# --- internal helper (no need to edit) ---------------------------------
# Rasterise the heatmap imshow before PDF export so the dense VIC heatmaps stay
# compact rasters (at HEATMAP_DPI) instead of thousands of vector cells. Defined
# in this always-re-run config cell so the fix loads even if the cell-42 helper
# cell isn't re-executed. (The RF hex panel is a separate plotly figure and is
# intentionally kept as vector.)
def mpl_to_pdf_bytes(fig):
    """Matplotlib Figure -> vector PDF bytes (imshow rasterised), and close it."""
    for ax in fig.axes:
        for im in ax.get_images():        # AxesImages = the heatmap imshow(s)
            im.set_rasterized(True)
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

# %%
# Create a PDF file to save all plots. Geometry / sizes / thresholds are all set
# in the CONFIG cell above; this cell just consumes them.
import kaleido

pdf_path = Path(result_dir, 'gallery_vpn.pdf')

def grid_cells(ncols=NCOLS, nrows=NROWS):
    """Return a list of fitz.Rect tiles in row-major order (TL, TM, TR, BL, BM, BR).
    Uses the page-layout knobs (W, H, M, TITLE_H, NCOLS, NROWS) from the CONFIG cell."""
    cells = []
    col_w = W / ncols
    row_h = (H - TITLE_H) / nrows
    for r in range(nrows):
        for col in range(ncols):
            x0 = col * col_w       + (M if col == 0          else M / 2)
            x1 = (col + 1) * col_w - (M if col == ncols - 1  else M / 2)
            y0 = TITLE_H + r * row_h       + (M if r == 0         else M / 2)
            y1 = TITLE_H + (r + 1) * row_h - (M if r == nrows - 1 else M / 2)
            cells.append(fitz.Rect(x0, y0, x1, y1))
    return cells


# --- build the multi-page PDF ---------------------------------------------
# Persistent Kaleido server: one Chrome reused across every fig.to_image()
# (anterior + RF panels), so each export is ~0.3s instead of paying ~5s Chrome
# cold-start per call. stop-then-start clears any stale server from an aborted run.
try:
    kaleido.stop_sync_server()
except Exception:
    pass
kaleido.start_sync_server()

out = fitz.open()                              # ONE doc, opened before the loop

for i, row in inst_vp.iterrows():
# for i, row in sampled_meta.iterrows():
    print(row['instance'])
    bodyId    = row['bodyId in figures']
    inst      = row['instance']
    cell_type = row['cell_type']
    clu = clusters[clusters.instance == inst]['cluster'].values
    if len(clu) == 0:
        continue

    # check if this inst exist in vic_median, if not skip
    if inst not in vic_median['instance'].values:
        print(f"instance {inst} not in vic_median, skipping")
        continue

    if str(bodyId) not in effwt_visr_5.columns:
        print(f"bodyId {bodyId} ({inst}) not in effwt_visr_5, skipping")
        continue

    page = out.new_page(width=W, height=H)     # ONE page per instance
    tl, tm, tr, bl, bm, br = grid_cells()       # row-major: TL TM TR / BL BM BR
    # page.insert_text((M, 18), f"instance: {inst}  -  bodyId: {bodyId}", fontsize=13)

    # --- per-panel titles built from the (now removed) stats table ---------
    title_tl = (f"Anterior view - {inst} - cluster {clu[0]} - "
                # f"{_fmt(row['no. of cells'], '{:.0f}')} cells - "
                f"pred. NT: {row['predicted neurotransmitter']}")
    title_tm = (f"top {PATHWAY_TOP_N} upstream paths - median VIC: {_fmt(row['VIC'], '{:.4f}')}, "
                f"layer: {_fmt(row['ht'])}")
    title_tr = (f"RF - median ellipse: {_fmt(row['size'], '{:.1f}')} col, "
                f"median ahull: {_fmt(row['ahull_size'], '{:.1f}')} col")
    
    # top-left: anterior view — 

    # already a PDF, place directly (stays vector)
    # pdf_image_path = Path(
    #     "C:/Users/zhaoa/HHMI Dropbox/Arthur Zhao/OL_CONNECTOME/results/images/VPN",
    #     f"{cell_type}_anterior-view.pdf")
    # if pdf_image_path.exists():
    #     place(page, pdf_image_path.read_bytes(), tl, title=title_tl)

    # render the example neuron (coarse mesh, MESH_LOD) over the CNS *outline* (line traces).
    # benchmark: opaque silhouette mesh -> ~23s Kaleido/WebGL export; outline -> ~6s.
    # the silhouette's big mesh dominated render time (constant across neurons).
    fig_ant = plot_cns(
        int(bodyId), cell_type,
        show_meshes=True, show_outline=True, mesh_lod=MESH_LOD,
        color=(0, 0, 0, 1), annotate=False,   # title_tl already labels the panel
    )
    # ANTERIOR_WIDTH/HEIGHT (CONFIG) set the render px before the ~0.7 content crop.
    img_fn = save_figure_anterior(fig_ant, name=f"tmp_anterior", path=cache_dir,
                                  width=ANTERIOR_WIDTH, height=ANTERIOR_HEIGHT)
    place_image(page, img_fn, tl, title=title_tl)


    # top-middle: pathway
    _layer_cumsum = {inst: 0.38}
    frac, fig_path = plot.plot_pathway(
        ol_data.get_paths_to_instance(inst), inst, ol_flow,
        # thre_cumsum=_layer_cumsum[inst],
        top_n=PATHWAY_TOP_N,
        neuron_to_color=type_to_mg,
        neuron_to_sign=type_to_sign, 
        sign_color_map=sign_to_color,
        save_path=None,
        show=False
    )
    place(page, mpl_to_pdf_bytes(fig_path), tm, title=title_tm)

    # top-right:
    # base effective-weight hex heatmap (Plotly) -- kept as VECTOR (not rasterised)
    df = effwt_visr_5[str(bodyId)].copy()
    fig_rf0 = ci.hex_heatmap(
        df,
        custom_colorscale=[[0, "rgb(255, 255, 255)"], [1, "rgb(200, 20, 0)"]],
        global_min=0,
    )
    # params_single_df = rf_fit.loc[rf_fit.bodyId == bodyId]
    # if params_single_df.empty:
    params_single_df = rf_fit.loc[rf_fit.instance == inst]
    fig1 = plot_gaussian_params(params_single_df, example_bid=bodyId, fac=np.sqrt(3))
    for trace in fig1.data[1:]:
        fig_rf0.add_trace(trace)

    # fit-area / r2 annotation strings
    if len(params_single_df) > 0 and 'size' in params_single_df.columns:
        fit_area_str = f"{params_single_df['size'].median():.2f}"
        r2_str = f"r2: {params_single_df['r2'].values[0]:.2f}"
    else:
        fit_area_val = params_single_df['a'].values * params_single_df['b'].values * np.pi
        fit_area_str = f"{fit_area_val[0]:.2f}"
        r2_str = f"r2: {params_single_df['r2'].values[0]:.2f}"

    # ahull scan (RF_THR_MODE / RF_REMOVE_FRAC from CONFIG)
    thr_mode, remove_frac = RF_THR_MODE, RF_REMOVE_FRAC
    edge_points, edges, area, com, kept_points = count_col_ahull(
        df, thr_mode=thr_mode, remove_frac=remove_frac)
    if edges is not None and len(edge_points) == 1:
        fig_rf = go.Figure(fig_rf0)
        edge_pt = edge_points[0]
        for j in range(len(edge_pt)):
            p1, p2 = edge_pt[j], edge_pt[(j + 1) % len(edge_pt)]
            fig_rf.add_trace(go.Scatter(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode='lines',
                line=dict(color='blue', width=1), showlegend=False, hoverinfo='skip'))
        # fig_rf.add_trace(go.Scatter(
        #     x=kept_points.values[:, 0], y=kept_points.values[:, 1], mode='markers',
        #     marker=dict(color='gray', size=5), showlegend=False, hoverinfo='skip'))
        fig_rf.add_trace(go.Scatter(
            x=[com[0]], y=[com[1]], mode='markers',
            marker=dict(color='blue', size=10, symbol='cross'),
            name='center of mass', hoverinfo='skip'))
        # fig_rf.add_annotation(
        #     xref="paper", yref="paper", x=0.01, y=0.9, showarrow=False, 
        #     xanchor="left",  align="left", font=dict(size=12),
        #     text=(f"bodyId: {bodyId}<br>mode: {thr_mode}<br>remove frac: {remove_frac:.1f}<br>"
        #           f"ahull: {area[0]:.2f} col<br>ellipse: {fit_area_str} col<br>{r2_str}"))
        place(page, plotly_to_pdf_bytes(fig_rf), tr, title=title_tr)
    else:
        print(f"multiple ahulls for bodyId {bodyId} ({thr_mode}, {remove_frac}), skipping panel")


    # --- bottom row: central-brain VIC max-projection heatmaps (T0/T1/T2) ---
    conn_thr = CONN_THR
    figs = make_vic_hop_figs(
        inst, step0=step0, syn0=syn0,
        meta_cb_vpn=meta_cb_vpn, cb_ids=cb_ids,
        bodyId_to_idx=bodyId_to_idx, idx_to_bodyId=idx_to_bodyId,
        outlines_bkgd=outlines_bkgd, conn_thr=conn_thr)
    if figs['t0'] is not None:
        place(page, mpl_to_pdf_bytes(figs['t0']), bl,
              title=f"VIC max-proj - {figs['n_t0']} cells")
    if figs['t1'] is not None:
        place(page, mpl_to_pdf_bytes(figs['t1']), bm,
              title=f"1-hop downstream - {figs['n_t1']} cells ( > {conn_thr} connections)",
              subtext=["top inst: VIC"] + figs['top_inst_t1'])
    if figs['t2'] is not None:
        place(page, mpl_to_pdf_bytes(figs['t2']), br,
              title=f"2-hop downstream - {figs['n_t2']} cells ( > {conn_thr} connections)",
              subtext=["top inst: VIC"] + figs['top_inst_t2'])

    print(f"Added page for instance {inst} (bodyId {bodyId})")

kaleido.stop_sync_server()                     # release the persistent Chrome
# deflate_images Flate-compresses the (otherwise uncompressed) anterior-view
# rasters that dominate file size; lossless, ~13x smaller. garbage/clean prune
# orphaned objects left by the build.
out.save(str(pdf_path), garbage=4, deflate=True, deflate_images=True, deflate_fonts=True, clean=True)
print(f"Saved {out.page_count} pages to {pdf_path}")

# %%
# --- shrink file size: rewrite embedded RGB rasters to 8-bit /Indexed colour -----
# The heatmaps come from a <=256-colour matplotlib colormap and the anterior render
# is near 2-tone, so palette-indexing is visually lossless yet ~2x smaller than RGB.
# Palette size = PALETTE_COLORS (CONFIG cell). All vector content (pathway, RF hexes,
# colorbar/outline/axis text) is left untouched.
import os
import re
import zlib
from PIL import Image

def _expected_len(d, x):
    """Bytes a strict viewer expects in the decoded image stream, from its dims."""
    cs = d.xref_get_key(x, 'ColorSpace')[1]
    w = int(d.xref_get_key(x, 'Width')[1]); h = int(d.xref_get_key(x, 'Height')[1])
    bpc = int(d.xref_get_key(x, 'BitsPerComponent')[1] or 8)
    ncomp = 1 if ('Indexed' in cs or 'Gray' in cs) else (3 if 'RGB' in cs else 4)
    return (((w * ncomp * bpc + 7) // 8) * h), cs

def _image_issues(d):
    """List (xref, kind) for /Image objects strict viewers (Acrobat/Ghostscript)
    reject: 'palette' = lookup length != (hival+1)*3; 'data' = decoded stream length
    != Width*Height. Both have been real bugs here (hival mismatch; and passing
    pre-compressed bytes to update_stream, which double-compresses -> short data)."""
    issues = []
    for x in range(1, d.xref_length()):
        if d.xref_get_key(x, 'Subtype')[1] != '/Image':
            continue
        expect, cs = _expected_len(d, x)
        m = re.search(r'/Indexed\s*/\w+\s*(\d+)\s*<([0-9A-Fa-f]*)>', cs)
        if m and len(m.group(2)) // 2 != (int(m.group(1)) + 1) * 3:
            issues.append((x, 'palette')); continue
        try:
            got = len(d.xref_stream(x))
        except Exception:
            got = -1
        if got != expect:
            issues.append((x, 'data'))
    return issues

def _repair_images(d):
    """Auto-fix the two known defect classes in place. Returns count fixed."""
    fixed = 0
    for x, kind in _image_issues(d):
        if kind == 'palette':                       # hival declared too high
            cs = d.xref_get_key(x, 'ColorSpace')[1]
            m = re.search(r'(/Indexed\s*/\w+)\s*\d+\s*<([0-9A-Fa-f]*)>', cs)
            d.xref_set_key(x, 'ColorSpace', f'[{m.group(1)} {len(m.group(2))//2//3 - 1} <{m.group(2)}>]')
            fixed += 1
        elif kind == 'data':                        # stream was double-compressed
            expect, _ = _expected_len(d, x)
            try:
                raw = zlib.decompress(d.xref_stream(x))
            except Exception:
                raw = b''
            if len(raw) == expect:
                d.update_stream(x, raw)             # re-store with a SINGLE Flate pass
                fixed += 1
    return fixed

def index_pdf_images(pdf_file, colors=256):
    """Re-encode opaque RGB images in `pdf_file` to /Indexed colour, in place."""
    d = fitz.open(str(pdf_file))
    n = 0
    for x in range(1, d.xref_length()):
        if d.xref_get_key(x, 'Subtype')[1] != '/Image':
            continue
        try:
            pix = fitz.Pixmap(d, x)
        except Exception:
            continue
        if pix.alpha or (pix.n - pix.alpha) < 3:        # only opaque RGB
            continue
        im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        p = im.quantize(colors=colors, method=Image.MEDIANCUT)
        # Pillow >=10 returns ONLY the used palette entries (the near-2-tone anterior
        # needs ~30); hival MUST match the bytes we write or strict viewers reject it.
        pal = bytes(p.getpalette('RGB'))
        ncol = len(pal) // 3
        # Pass RAW bytes: update_stream defaults to compress=True and Flate-encodes ONCE.
        # Passing pre-zlib'd bytes here double-compresses -> "Insufficient data for an image".
        d.update_stream(x, p.tobytes())
        d.xref_set_key(x, 'ColorSpace', f'[/Indexed /DeviceRGB {ncol - 1} <{pal.hex()}>]')
        d.xref_set_key(x, 'BitsPerComponent', '8')
        for k in ('DecodeParms', 'SMask', 'Decode'):
            d.xref_set_key(x, k, 'null')
        n += 1

    # Self-heal any palette/data defects (e.g. from a stale kernel running older code).
    nfix = _repair_images(d)
    if nfix:
        print(f"  self-check repaired {nfix} image(s)")

    tmp = str(pdf_file) + '.tmp'
    d.save(tmp, garbage=4, deflate=True, clean=True)   # fitz can't overwrite the open file
    d.close()
    os.replace(tmp, str(pdf_file))

    # Hard gate: re-open and confirm no image would be rejected by a strict viewer.
    chk = fitz.open(str(pdf_file))
    left = _image_issues(chk)
    chk.close()
    if left:
        raise RuntimeError(f"{len(left)} image(s) still malformed (strict viewers will "
                           f"error/blank): {left[:5]}")
    return n

_mb = lambda f: round(Path(f).stat().st_size / 1e6, 2)
_before = _mb(pdf_path)
_n = index_pdf_images(pdf_path, colors=PALETTE_COLORS)
print(f"indexed {_n} images: {_before} MB -> {_mb(pdf_path)} MB  (image integrity check passed)")

# %%


# %% [markdown]
# # End


