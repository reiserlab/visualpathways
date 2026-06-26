# Code carried over from the upstream repo

Parts of this folder originated in the companion repository
**reiserlab/male-drosophila-visual-system-connectome-code** (`src/utils/`).
License: upstream is GPL v3 (same as this repo). This notice preserves
attribution.

Comparison performed against upstream `main` on 2026-04-23. Classification
reflects the state of this folder at that time.

## Verbatim (vendored unchanged)

These files are byte-identical (or effectively so) to their upstream
counterparts:

- `ng_view.py` — `NG_View` enum of camera view presets.

## Vendored with a local patch

- `plotter.py` — plotly/navis rendering pipeline (`group_plotter`, `plot_cns`,
  `get_skeleton`, `get_mesh`, `get_roi`, `show_figure`, `save_figure`, ...).
  Single modification: rewrote the two `madvisc.utils.*` imports to point at
  siblings in this folder — `from utils.ng_view import NG_View` and
  `from utils.helper import slugify`.

## Near-identical (trivially diverged)

- `geometry.py` — local has one extra stray import (`from cmath import cos`)
  on top of the upstream file; otherwise identical.

## Forked (same starting point, locally diverged)

These files share their skeleton / many functions with upstream but have
accumulated substantive local edits. They should be treated as local code
that derives from upstream rather than verbatim vendored files.

- `helper.py` — `slugify`, `num_expand`, and related helpers.
- `olc_client.py` — NeuPrint client connection helper.
- `ROI_calculus.py` — ROI intersection / volume math.
- `ROI_layers.py` — layer-based ROI analysis.
- `ROI_voxels.py` — voxel-level ROI data.
- `plotting_functions.py` — hex heatmaps, Mollweide/Mercator, PyVis helpers, etc.

## Local-only (no upstream equivalent, or unrelated despite shared name)

- `ROI_columns.py` — the local file is a small original module and does not
  correspond to the (much larger) upstream `ROI_columns.py`.
- `_unused_functions.py`, `align_mi1_t4.py`, `cave_client.py`,
  `celltype_conn_by_roi.py`, `clustering.py`, `connectivity.py`,
  `graph_utils.py`, `helper_jh.py`, `hex_hex.py`, `input_distr_functions.py`,
  `module_analysis.py`, `neuron_bag_jh.py`, `ol_color.py`, `ol_rf.py`,
  `ol_types.py`, `prop_by_adj.py`, `query.py`, `ROI_plots.py`,
  `summary_plot_preprocessor.py`, `weight_prop.py` — local to this repo; no
  upstream counterpart was found under `src/utils/`.
