# %%
%load_ext autoreload
%autoreload 2

# %% [markdown]
# ### make gallery for VCBNs

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
# print(f"Project root directory: {PROJECT_ROOT}")

# %%
from utils import olc_client
c = olc_client.connect(verbose=True)

# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import connectome_interpreter as ci

from quan_propagation.func import count_col_ahull
from utils.plotting_functions import plot_gaussian_params
from utils.plotter import plot_cns, save_figure_anterior

from utils.config import (DATA_DIR, DATASET)

# %%
result_dir = Path(PROJECT_ROOT, 'results', 'quan_propagation')
result_dir.mkdir(parents=True, exist_ok=True)

cache_dir = Path(PROJECT_ROOT, 'cache', 'quan_propagation')
cache_dir.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# # load data

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

meta_ahull = pd.read_pickle(Path(DATA_DIR, 'meta_VP_CB_ahull_cumsum60.pkl'))

meta_cb_vpn = pd.merge(
    meta_ahull[['bodyId', 'instance', 'ahull_size']],
    vp_cb_vic[['bodyId', 'VIC', 'hitting_time', 'main_groups']],
    on='bodyId', how='inner'
)
print(meta_cb_vpn.shape)

meta_cb_vpn.rename(columns={
    'hitting_time':'ht'}, inplace=True)
print(meta_cb_vpn.shape)

# %%
# load size, ellipse fit and ahull
rf_fit = pd.read_pickle(Path(cache_dir, 'rf_fit_thr07.pkl'))
print(rf_fit.shape)
rf_size = pd.merge(
    rf_fit[['instance','bodyId', 'size', 'r2']],
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

# %% [markdown]
# ## load effwt

# %%
effwt_visr_5 = pd.read_pickle(Path(cache_dir, 'effwt_visr_5.pkl'))

# %% [markdown]
# ## vcbn types

# %%
# sort
import re
def natural_sort_key(s):
    """Convert string to list of strings and integers for natural sorting"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', str(s))]

# %%
# LOAD meta, this has AN's R7/8, 
meta = pd.read_csv(DATA_DIR / f'{DATASET}_meta.csv')
    
#  add superclass and class
meta.loc[meta['cell_type'].str.contains('^R7|^R8'), 'superclass'] = 'ol_sensory'
meta.loc[meta['cell_type'].str.contains('^R7|^R8'), 'class'] = 'visual'

print(meta.shape)  

# %%
# oltypes
oltypes = pd.read_excel(Path(PROJECT_ROOT, 'params', 'Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx'))
oltypes.rename(columns={'cell type':'cell_type', 'main groups':'main_groups'}, inplace=True)
print(oltypes.shape)

# %%
# inst for gallery
# For each (instance, cell_type), keep the bodyId with the highest VIC
inst_cb = (
    vp_cb_vic.loc[
        vp_cb_vic.groupby(['instance', 'cell_type'])['VIC'].idxmax(),
        ['instance', 'cell_type', 'bodyId']
    ]
    .reset_index(drop=True)
)

# Add bodyId count (unique neurons) per instance
inst_cb = inst_cb.merge(
    vp_cb_vic.groupby('instance')['bodyId'].nunique().rename('bodyId_count').reset_index(),
    on='instance',
    how='left'
)

# add nt
inst_cb = inst_cb.merge(
    meta.groupby('instance')['nt'].first().reset_index(),
    on='instance',
    how='left'
)

# remove oltypes
inst_cb = inst_cb[~inst_cb['cell_type'].isin(oltypes['cell_type'])]
print(inst_cb.shape)
inst_cb = pd.merge(inst_cb, rf_size_median, on='instance', how='inner')
print(inst_cb.shape)
inst_cb = pd.merge(inst_cb, ht_median[['instance','ht']], on='instance', how='inner')
print(inst_cb.shape)
inst_cb = pd.merge(inst_cb, vic_median[['instance','VIC']], on='instance', how='inner')
print(inst_cb.shape)

# %%
# sort by ht
inst_cb = inst_cb.sort_values(by='ht').reset_index(drop=True)


# %% [markdown]
# ## VP cluster

# %%
# VCBN cluster
from utils import cb_data
FRAC_CB = 0.17
cl_r = cb_data.get_cb_clusters(side_char='r', frac_thre=FRAC_CB, force=True)

# %% [markdown]
# # main loop

# %%
import io
from PIL import Image
import fitz  # PyMuPDF

def mpl_to_pdf_bytes(fig):
    """Matplotlib Figure -> vector PDF bytes (and close it)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

def mpl_to_rasterized_pdf_bytes(fig):
    """Matplotlib Figure -> vector PDF bytes (imshow rasterised), and close it."""
    for ax in fig.axes:
        for im in ax.get_images():        # AxesImages = the heatmap imshow(s)
            im.set_rasterized(True)
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

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

def make_stats_table_fig(row, cl, figsize=(3.2, 2.6)):
    """Build a matplotlib table of per-instance stats (vector, for `place`)."""
    rows = [
        ("instance",           str(row['instance'])),
        ("cell count",         _fmt(row['bodyId_count'], "{:.0f}")),
        ("pred. NT",           str(row['nt'])),
        ("VCBN cluster",            str(cl)),
        ("median layer",    _fmt(row['ht'])),
        ("median VIC",         _fmt(row['VIC'], "{:.4f}")),
        ("median α-hull size",  f"{_fmt(row['ahull_size'], "{:.1f}")} columns"),
        ("median ellipse size",    f"{_fmt(row['size'], "{:.1f}")} columns"),
        ("median r²",     _fmt(row['r2'])),
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

# %%
# 2D gaussian, prob mass (cumsum) within one std contour, is ~0.39
# PDF height (max) at one std is ~0.61

# check ahull area calculation using bodyid = 28862.
# check edge_points using 531732
# thr_2dGaussian = 0.4

# pathway plot
from utils import core_data, cb_data
from utils import plotting as plot
from utils.palettes import load_colors
# cb_flow = core_data.get_cb_flow_type()
cb_flow = core_data.get_flow_per_group()
cb_meta     = core_data.get_meta()
colored_region, colored_main_groups, colored_sign, colored_seed = load_colors()
type_to_sign = dict(zip(cb_meta.cell_type, cb_meta.sign))
sign_to_color = colored_sign['color'].to_dict()
# type_to_mg = dict(zip(cb_meta.cell_type, colored_main_groups.loc[cb_meta.main_groups, 'color']))
mg = cb_meta.main_groups.where(cb_meta.main_groups.isin(colored_main_groups.index), 'other')
type_to_mg = dict(zip(cb_meta.cell_type, colored_main_groups.loc[mg, 'color']))


# %%
# take every 10th row in inst_cb as
sampled_meta = inst_cb.iloc[::10, :].reset_index(drop=True)
sampled_meta.shape

# %%
# take 10 samples from sampled_meta
# sampled_meta = inst_cb[inst_cb['instance'].isin(inst_cb['instance'].sample(n=10, random_state=42).tolist())].reset_index(drop=True)

# sampled_meta = inst_cb[
#     (inst_cb['ht'] > 4) & (inst_cb['ahull_size'] < 20)
# ]

# sampled_meta = inst_cb[inst_cb['cell_type'].isin(
#     ['PVLP010', 'PS126', 'PS074','VES103','PS101','PS112','LC22']
# )]

# print(sampled_meta.shape)

# %% [markdown]
# ## make pdf

# %%
# Palette size for the colour-indexing in the size-shrink cell (after the build).
# 256 = full colormap fidelity; 128 ~= -1.5 MB with faint heatmap-gradient banding.
PALETTE_COLORS   = 256

# %%
# Create a PDF file to save all plots
import kaleido
PATHWAY_TOP_N  = 5           # number of strongest upstream paths drawn (top-middle)
RF_THR_MODE    = 'cumsum'    # RF alpha-hull threshold mode (top-right)
RF_REMOVE_FRAC = 0.6         # RF alpha-hull cumulative-mass cutoff

pdf_path = Path(result_dir, 'gallery_vcbn.pdf')

# --- page geometry (points; 72 pt = 1 in) ---------------------------------
W, H = 720, 620          # page width and height in points
M, TITLE_H = 12, 28      # margin and title bar height in points
gx, gy = W / 2, (H - TITLE_H) / 2  # grid center x; grid cell height (below title)
def quadrants():
    tl = fitz.Rect(M,         TITLE_H + M,        gx - M/2, TITLE_H + gy - M/2)
    tr = fitz.Rect(gx + M/2,  TITLE_H + M,        W - M,    TITLE_H + gy - M/2)
    bl = fitz.Rect(M,         TITLE_H + gy + M/2, gx - M/2, H - M)
    br = fitz.Rect(gx + M/2,  TITLE_H + gy + M/2, W - M,    H - M)
    return tl, tr, bl, br

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

for i, row in sampled_meta.iterrows():
    print(row['instance'])
    # bodyId    = row['bodyId in figures']
    bodyId = row['bodyId']
    inst      = row['instance']
    cell_type = row['cell_type']
    cl = cl_r.loc[cl_r['instance'] == inst, 'cluster'].values
    cl = cl[0] if len(cl) > 0 else 'n/a'

    # check if this inst exist in vic_median, if not skip
    if inst not in vic_median['instance'].values:
        print(f"instance {inst} not in vic_median, skipping")
        continue

    if str(bodyId) not in effwt_visr_5.columns:
        print(f"bodyId {bodyId} ({inst}) not in effwt_visr_5, skipping")
        continue

    page = out.new_page(width=W, height=H)     # ONE page per instance
    tl, tr, bl, br = quadrants()
    # page.insert_text((M, 18), f"instance: {inst}  -  bodyId: {bodyId}", fontsize=13)

    # top-left: anterior view — already a PDF, place directly 
    fig_ant = plot_cns(
        int(bodyId), cell_type,
        show_meshes=True, show_outline=True, mesh_lod=2,
        color=(0, 0, 0, 1), annotate=False,   
    )
    img_fn = save_figure_anterior(fig_ant, name=f"tmp_anterior", path=cache_dir)
    title_tl = (f"Anterior view - {inst}"
                # f" - {_fmt(row['bodyId_count'], '{:.0f}')} cells"
                # f" - pred. NT: {row['nt']}"
                )
    place_image(page, img_fn, tl, title=title_tl)

    # top-right: pathway
    _layer_cumsum = {inst: 0.38}
    frac, fig_path = plot.plot_pathway(
        cb_data.get_cb_paths_to_instance(inst), 
        inst, cb_flow,  top_n=PATHWAY_TOP_N,
        neuron_to_color=type_to_mg,
        neuron_to_sign=type_to_sign, 
        sign_color_map=sign_to_color,
        save_path=None,
        show=False,
        show_frac=False
    )
    place(page, mpl_to_pdf_bytes(fig_path), tr, title=f"top {PATHWAY_TOP_N} upstream paths - {frac:.0%} of total path weight")

    # bottom-left: stats table
    place(page, mpl_to_pdf_bytes(make_stats_table_fig(row, cl)), bl, title="")

    # bottom-right: 
    # base effective-weight hex heatmap (Plotly)
    df = effwt_visr_5[str(bodyId)].copy()
    fig_rf0 = ci.hex_heatmap(
        df,
        custom_colorscale=[[0, "rgb(255, 255, 255)"], [1, "rgb(200, 20, 0)"]],
        global_min=0,
    )
    # -- ellipse
    # params_single_df = rf_fit.loc[rf_fit.bodyId == bodyId]
    params_single_df = rf_fit.loc[rf_fit.instance == inst]
    fig1 = plot_gaussian_params(params_single_df, example_bid=bodyId, fac=np.sqrt(3))
    for trace in fig1.data[1:]:
        fig_rf0.add_trace(trace)
    # # fit-area / r2 annotation strings
    # if len(params_single_df) > 0 and 'size' in params_single_df.columns:
    #     fit_area_str = f"{params_single_df['size'].median():.2f}"
    #     r2_str = f"r2: {params_single_df['r2'].values[0]:.2f}"
    # else:
    #     fit_area_val = params_single_df['a'].values * params_single_df['b'].values * np.pi
    #     fit_area_str = f"{fit_area_val[0]:.2f}"
    #     r2_str = f"r2: {params_single_df['r2'].values[0]:.2f}"

    # -- ahull scan (cumsum 0.6)
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
        #     xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False, 
        #     xanchor="left", align="left", font=dict(size=12),
        #     text=(f"mode: {thr_mode}<br>remove frac: {remove_frac:.1f}<br>"
        #           f"ahull area: {area[0]:.2f}"))
        place(page, plotly_to_pdf_bytes(fig_rf), br, title="")
    else:
        print(f"multiple ahulls for bodyId {bodyId} ({thr_mode}, {remove_frac}), skipping panel")

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


# %%
# # Create a PDF file to save all plots
# pdf_path = Path(result_dir, 'gallery_area_cb.pdf')

# # with PdfPages(pdf_path) as pdf:
#     # for i, row in oltypes_vpn.loc[oltypes_vpn.instance.str.contains('^LC.*')].iterrows():
#     for i, row in meta_type_cb_vpn[:5].iterrows():
#         inst = row['instance']
#         bodyId = meta_cb_vpn[meta_cb_vpn['instance'] == inst].bodyId.values[0]
        
        

#         # - Generate the RF hex plot
#         df = effwt_visr[str(bodyId)].copy()
#         fig_rf = ci.hex_heatmap(df,
#                         custom_colorscale=[[0, "rgb(255, 255, 255)"], [1, "rgb(200, 20, 0)"]],
#                         global_min=0
#                         )
                        
#         params_single_df = fit_rf.loc[fit_rf.bodyId== bodyId]
#         fig1 = plot_gaussian_params(params_single_df, example_bid=bodyId, fac=np.sqrt(3))

#         # ahull
#         edge_points, edges, area, com, kept_points = count_col_ahull(df, thr_max = 0.1)
#         for edge in edges:
#             # point1 = xy_coord_pos.values[edge[0]]
#             # point2 = xy_coord_pos.values[edge[1]]
#             point1 = kept_points.values[edge[0]]
#             point2 = kept_points.values[edge[1]]
#             fig_rf.add_trace(go.Scatter(
#                 x=[point1[0], point2[0]],
#                 y=[point1[1], point2[1]],
#                 mode='lines',
#                 line=dict(color='gray', width=2),
#                 showlegend=False,
#                 hoverinfo='skip'
#             ))
#         # Plot xy_coord_pos points
#         fig_rf.add_trace(go.Scatter(
#             x=kept_points.values[:, 0],
#             y=kept_points.values[:, 1],
#             mode='markers',
#             marker=dict(color='gray', size=5),
#             showlegend=False,
#             hoverinfo='skip'
#         ))

#         # Plot center of mass
#         fig_rf.add_trace(go.Scatter(
#             x=[com[0]],
#             y=[com[1]],
#             mode='markers',
#             marker=dict(color='black', size=10, symbol='cross', line=dict(width=4,  color='black')),
#             name='center of mass',
#             hoverinfo='skip'
#         ))

#         for trace in fig1.data[1:]:
#             fig_rf.add_trace(trace)
#         # fig_rf = ci.hex_heatmap(df, custom_colorscale=[[0, "rgb(255, 255, 255)"], [1, "rgb(200, 20, 0)"]], global_min=0)
        
#         # - Create a new figure with two subplots side by side
#         fig_combined, axs = plt.subplots(2, 2, figsize=(8, 8))
        
#         # # Add the anterior view image if available
#         # if fig_ant is not None:
#         #     axs[0, 0].imshow(fig_ant)
#         #     axs[0, 0].axis('off')
#         #     axs[0, 0].set_title(f'Anterior view - instance: {inst}', fontsize=12)
#         # # Add the OL view image if available
#         # if fig_OL is not None:
#         #     axs[1, 0].imshow(fig_OL)
#         #     axs[1, 0].axis('off')
#         #     axs[1, 0].set_title(f'OL view, bodyId={bodyId}', fontsize=12)

#         # - RF heatmap (convert Plotly to image)
#         img_bytes = fig_rf.to_image(format="png", width=800, height=800, scale=2)
#         img = Image.open(io.BytesIO(img_bytes))
#         axs[0, 1].imshow(img)
#         axs[0, 1].axis('off')
        
#         vic_str = f'{row.vision:.1e}'
#         ht_str = f'{np.round(row.ht, 1)}'
#         area_fit_str = f'{np.round(row.area_fit, 1)}' if row.area_fit else 'N/A'
#         r2_str = f'{np.round(row.r2, 2)}' if row.r2 else 'N/A'
#         area_ahull_str = '' if len(area) != 1 else f'{np.round(area[0]*thr_2dGaussian, 1)}'

#         axs[0, 1].set_title(
#             f'{bodyId} - vic: {vic_str} - ht: {ht_str} - area_fit: {area_fit_str} - r2: {r2_str} - col_ahull: {area_ahull_str}',
#             fontsize=12
#         )
    
#         # - coverage
#         ids = meta_cb_vpn[meta_cb_vpn.instance==inst].bodyId.values
#         df = effwt_visr[[str(id) for id in ids]].copy()
#         df = df.sum(axis=1)
#         fig_cover = ci.hex_heatmap(df,
#                 custom_colorscale=[[0, "rgb(255, 255, 255)"], [1, "rgb(200, 20, 0)"]],
#                 global_min=0
#                 )
#         # add com from fit
#         params_df = fit_rf.loc[fit_rf.bodyId.isin(ids)]
#         fig_cover.add_trace(go.Scatter(
#             x=params_df['x0'].values,
#             y=params_df['y0'].values *np.sqrt(3),
#             mode='markers',
#             marker=dict(color='black', size=20, symbol='circle'),
#             showlegend=False,
#         ))    
#         img_bytes = fig_cover.to_image(format="png", width=800, height=800, scale=2)
#         img = Image.open(io.BytesIO(img_bytes))
#         axs[1, 1].imshow(img)
#         axs[1, 1].axis('off')
#         area_fit_str = f'{np.round(row.area_fit, 1)}' if row.area_fit else 'N/A'
#         r2_str = f'{np.round(row.r2, 2)}' if row.r2 else 'N/A'

#         axs[1, 1].set_title(
#             f'instance: {inst} - count :{row.bodyId_count} ',   
#             fontsize=12)

#         plt.tight_layout()
        
#         # Save the combined figure to the PDF
#         pdf.savefig(fig_combined, dpi=300, bbox_inches='tight')
#         # plt.close(fig_vic)        
#         plt.close(fig_combined)

# print(f"Saved all plots to {pdf_path}")

# %%


# %% [markdown]
# # End


