# %% [markdown]
# Explore input into (right-side) VCN cell types, decomposed by presynaptic
# category, mirroring the Main Fig. 6a Sankey decomposition.
#
# Postsynaptic: right-side VCN cell types (main_groups == "VCN" and projecting
# into the right OL, i.e. region == "right OL"; some _L-named neurons that
# arborize in the right OL belong here intentionally).
#
# Presynaptic categories (same neuron categorization that builds main_6a, via
# get_visual_system_sankey_connectivity): VIN, VPN clusters oc1..oc9, VCBN
# clusters cc1..cc10, and non-visual CB. Colours match the 6a Sankey.
#
# Two outputs:
#   - pie chart of the total number of synaptic connections (raw counts from
#     malecns_v1.0_prop.npz) into right VCNs, split by presynaptic category
#     (saved as PDF + HTML).
#   - table (results/tables/) of per-instance input contributions using inprop
#     (input-normalized weights): median over each instance's right-OL neurons.

# %%
import sys
from pathlib import Path

from dotenv import find_dotenv

PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.sparse as sp

from utils import config, core_data, cb_data
from utils.palettes import load_colors
from utils.plotting import _CLUSTER_BLUES, _VCBN_CC_TO_OC

FRAC_OL = 0.19  # VPN (OL) clustering threshold used in paper_figures.ipynb
FRAC_CB = 0.17  # VCBN (CB) clustering threshold used in paper_figures.ipynb

# %%
# --- Load data ---
meta = core_data.get_meta()
prop = core_data.get_full_prop()  # int32 synapse-count matrix, rows=pre, cols=post
inprop = sp.load_npz(config.DATA_DIR / f"{config.DATASET}_inprop.npz").tocsc()
assert int(meta["idx"].max()) == prop.shape[0] - 1, "meta.idx must index prop rows/cols"

# Presynaptic categorization identical to main_6a (VIN / VPN-as-oc / VCBN-as-cc / CB)
sankey = cb_data.get_visual_system_sankey_connectivity(
    ol_frac_thre=FRAC_OL, cb_frac_thre=FRAC_CB,
    weight_thre=1000, min_plot_neurons=120, output_normalized=False,
)
ms = sankey["meta_sub"].copy()
n_ol_clu, n_cb_clu = sankey["n_ol_clu"], sankey["n_cb_clu"]


def _column(cat, clu):
    if cat == "VIN":
        return "VIN"
    if cat == "VPN":
        return f"oc{int(clu)}"
    if cat == "VCBN":
        return f"cc{int(clu)}"
    if cat == "CB":
        return "non-visual CB"
    return None


ms["col"] = [_column(c, k) for c, k in zip(ms["category"], ms["cluster"])]
ms = ms.dropna(subset=["col"])
col_order = (
    ["VIN"]
    + [f"oc{i}" for i in range(1, n_ol_clu + 1)]
    + [f"cc{i}" for i in range(1, n_cb_clu + 1)]
    + ["non-visual CB"]
)
col_to_row = {c: i for i, c in enumerate(col_order)}

# (n_cat x N) indicator of presynaptic category membership
S = sp.coo_matrix(
    (np.ones(len(ms)), (ms["col"].map(col_to_row).values, ms["idx"].values)),
    shape=(len(col_order), prop.shape[0]),
).tocsr()

# --- Postsynaptic: right-side VCN cell types ---
vcn_post = meta[(meta["main_groups"] == "VCN") & (meta["region"] == "right OL")]
post_idx = vcn_post["idx"].values

# %%
# --- Colours matched to Main Fig. 6a (plot_visual_system_sankey) ---
colored_region, colored_main_groups, _, _ = load_colors()
_oc_colors = {f"oc{i + 1}": c for i, c in enumerate(_CLUSTER_BLUES)}
# Use the single authoritative VCBN(cc) -> VPN(oc) colour key from plotting.py so
# this pie always matches the updated Main Fig. 6a Sankey colours.
cmap = {"VIN": colored_main_groups.loc["VIN", "color"]}
cmap.update({f"oc{i}": _oc_colors[f"oc{i}"] for i in range(1, n_ol_clu + 1)})
cmap.update({cc: _oc_colors[oc] for cc, oc in _VCBN_CC_TO_OC.items()})
cmap["non-visual CB"] = colored_region.loc["CB", "color"]
assert set(cmap) == set(col_order), "colour map must cover every category"

# %%
# --- Pie: total synapses (prop) into right VCNs by presynaptic category ---
syn = np.asarray((S @ prop[:, post_idx]).sum(axis=1)).ravel()
pie_df = pd.DataFrame({"category": col_order, "synapses": syn})
pie_df["pct"] = 100 * pie_df["synapses"] / pie_df["synapses"].sum()
print(f"Right-side VCN targets: {len(vcn_post)} neurons, {vcn_post['instance'].nunique()} instances")
print(f"Total synapses into right VCNs from these categories: {pie_df['synapses'].sum():,.0f}\n")
print(pie_df.to_string(index=False, formatters={"synapses": "{:,.0f}".format, "pct": "{:.1f}%".format}))

# Drop categories below 1% of the total from the pie (table keeps all columns).
PIE_MIN_PCT = 1.0
pie_plot = pie_df[pie_df["pct"] >= PIE_MIN_PCT].copy()
plot_order = [c for c in col_order if c in set(pie_plot["category"])]
pie_plot["pct_shown"] = 100 * pie_plot["synapses"] / pie_plot["synapses"].sum()
_dropped = pie_df[pie_df["pct"] < PIE_MIN_PCT]
print(
    f"\nOmitting {len(_dropped)} categories below {PIE_MIN_PCT:.0f}% of total "
    f"({_dropped['pct'].sum():.1f}% of synapses): {', '.join(_dropped['category'])}"
    "\n(remaining slices renormalized to 100%)"
)

fig = px.pie(
    pie_plot,
    values="synapses",
    names="category",
    color="category",
    color_discrete_map=cmap,
    category_orders={"category": plot_order},
    title="Synapses into right VCNs by presynaptic category (>=1%)",
)
fig.update_traces(sort=False, textposition="inside", texttemplate="%{percent:.0%}")
fig.update_layout(
    width=720,
    height=520,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="arial", size=14),
    legend=dict(font=dict(size=12)),
)

pdf_dir = config.FIG_DIR / "CB_pathways"
pdf_dir.mkdir(parents=True, exist_ok=True)
out_pdf = pdf_dir / f"{config.DATASET}_vcn_r_input_pie.pdf"
# Render the PDF with matplotlib (kaleido executable is unavailable in this env,
# so plotly write_image can't export). Same data and 6a colours as the plotly pie.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig_m, ax = plt.subplots(figsize=(7.6, 5.2))
wedges, _ = ax.pie(
    pie_plot["synapses"].values,
    colors=[cmap[c] for c in pie_plot["category"]],
    startangle=90, counterclock=False,
    wedgeprops=dict(edgecolor="white", linewidth=0.6),
)
ax.legend(
    wedges,
    [f"{c}  ({p:.0f}%)" for c, p in zip(pie_plot["category"], pie_plot["pct_shown"])],
    title="presynaptic category", loc="center left",
    bbox_to_anchor=(1.0, 0.5), fontsize=9, title_fontsize=10, frameon=False,
)
ax.set_title("Synapses into right VCNs by presynaptic category (≥1%)", fontsize=12)
ax.set_aspect("equal")
try:
    fig_m.savefig(out_pdf, bbox_inches="tight")
    print(f"\nSaved pie PDF -> {out_pdf}")
except PermissionError:
    print(f"\n{out_pdf.name} is locked (open in a viewer?); kept existing PDF.")
finally:
    plt.close(fig_m)

out_html = config.HTML_FIG_DIR / f"{config.DATASET}_vcn_r_input_pie.html"
config.HTML_FIG_DIR.mkdir(parents=True, exist_ok=True)
try:
    fig.write_html(out_html)
    print(f"Saved pie HTML -> {out_html}")
except PermissionError:
    print(f"{out_html.name} is locked (open in a viewer?); skipped HTML.")

# Drop the superseded 11-category pie HTML if present (best effort).
_old_html = config.HTML_FIG_DIR / f"{config.DATASET}_cbn_to_vcn_r_pie.html"
try:
    _old_html.unlink(missing_ok=True)
except (PermissionError, OSError):
    pass

# %%
# --- Table: per right-VCN-instance input contributions, using inprop
# (input-normalized weights), same presynaptic categories as the pie. Each
# value is the MEDIAN over the instance's right-OL neurons of the fraction of
# that neuron's total input coming from the category. (visual-input / left-OL /
# VNC / VCN presynaptic input has no column, so rows need not sum to 1.) ---
contrib = (S @ inprop[:, post_idx]).toarray().T  # (n_post x n_col)
contrib_df = pd.DataFrame(contrib, columns=col_order)
contrib_df.insert(0, "instance", vcn_post["instance"].values)
table = contrib_df.groupby("instance", as_index=False)[col_order].median()

tables_dir = config.FIG_DIR / "tables"
tables_dir.mkdir(parents=True, exist_ok=True)
out_csv = tables_dir / f"{config.DATASET}_vcn_r_input_contributions.csv"
table.to_csv(out_csv, index=False)
print(f"\nSaved table ({len(table)} VCN instances x {len(col_order)} categories) -> {out_csv}")
print(table.round(4).head(8).to_string(index=False))

# %%
