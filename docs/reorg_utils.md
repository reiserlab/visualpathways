# Utils Reorganization: Unused Function Cleanup

## Summary

Audited all 26 Python files in `src/utils/` to identify functions and classes not referenced
anywhere in the codebase. Unused functions were extracted from their source files and preserved
in `src/utils/_unused_functions.py`. Entirely unused files were left in place but flagged below.

**Result:** 1,637 lines of dead code removed across 7 files.

---

## Functions Removed from Partially-Used Files

### `ROI_calculus.py` (2 functions removed)

| Function | Reason |
|---|---|
| `syn_per_col_count` | No external callers |
| `find_per_columnbin_coverage` | No external callers |

*Note: `trim_by_col_count`, `find_per_columnbin_spanned_no_cols`, `count_hex_loc`, and `find_layers`
were initially flagged as having no direct external callers but are internal helpers for
`find_size_distr` and `find_mesh_layers` (both used externally), so they were kept.*

### `ROI_columns.py` (10 functions removed)

| Function | Reason |
|---|---|
| `find_holes` | No callers |
| `smooth_center_columns_w_median` | No callers |
| `create_center_column_pins` | No callers |
| `load_roi_pin_params` | No callers |
| `trim_syn_by_pc` | No callers |
| `find_anchors` | No callers |
| `find_neighbor_avg` | No callers |
| `find_shortest_path` | No callers |
| `find_shortest_path_dist` | No callers |
| `find_uniform_interpolation` | No callers |

Only `load_hexed_body_ids` was retained (used in `ROI_plots.py` and `weight_prop.py`).
Unused imports (`numpy`, `networkx`, `alphashape`, `trimesh`, `sklearn`, `scipy`, `navis`, `neuprint`) were also cleaned up.

### `graph_utils.py` (2 functions removed)

| Function | Reason |
|---|---|
| `fetch_shortest_paths_from_edgelist` | No callers; BFS implementation superseded by `nx.all_shortest_paths` in `collect_shortest_paths` |
| `collect_fixedlength_paths` | No callers |

Cleaned up now-unused imports: `collections.defaultdict`, `collections.deque`, `neuprint.fetch_paths`.

### `helper.py` (1 function removed)

| Function | Reason |
|---|---|
| `add_color_group` | No callers |

Also removed unused `import pandas as pd`.

### `plotting_functions.py` (2 functions removed)

| Function | Reason |
|---|---|
| `plot_heatmap` | No callers |
| `plot_flip_syn_hist` | No callers |

*Note: `calc_node_sizes_Angel` was initially flagged but is an internal helper called by
`plot_pyvis_Angel` (which IS used externally), so it was kept.*

### `prop_by_adj.py` (4 functions removed)

| Function | Reason |
|---|---|
| `solve_column_chunk` | No external callers; helper for `parallel_sparse_inverse` |
| `parallel_sparse_inverse` | No external callers |
| `monitor_memory` | No external callers; helper for `parallel_sparse_inverse_safe` |
| `parallel_sparse_inverse_safe` | No external callers |

Cleaned up now-unused imports: `scipy`, `multiprocessing`, `psutil`, `time`, `gc`, `traceback`.

### `ol_rf.py` (1 function removed)

| Function | Reason |
|---|---|
| `pqw_LO` | No callers; superseded by `hexw_columnar` |

---

## Entirely Unused Files (left in place)

These files have **no** functions or classes referenced anywhere in the codebase.
They were left in their original location but should be considered for deletion.

| File | Contents | Notes |
|---|---|---|
| `helper_jh.py` | `slugify`, `num_expand` | Exact duplicates of functions in `helper.py` |
| `module_analysis.py` | `find_la_modules`, `find_infomap`, `find_consensus`, `find_consensus_und`, `find_agreement`, `dummyvar` | Community detection utilities; no imports found |
| `ROI_plots.py` | `plot_mi1_t4_alignment`, `plot_pin_assignment`, `plot_all_syn` | Visualization functions; no imports found |
| `ROI_voxels.py` | `region_boxes`, `fetch_brain_voxels`, `voxelize_col_and_lay` | Voxelization pipeline; no imports found |
| `neuron_bag_jh.py` | `NeuronBag` class | Neuron collection class; no imports found |
| `summary_plot_preprocessor.py` | `SummaryPlotPreprocessor` class | Summary plot preprocessing; no imports found |

---

## Files Not Changed (all functions used)

The following files in `src/utils/` had all their functions used either directly or through
internal dependency chains:

- `ROI_layers.py` - all 11 functions are used (3 directly, 8 as internal helpers)
- `connectivity.py` - all 4 functions used
- `geometry.py` - all 5 functions used
- `hex_hex.py` - both functions used
- `input_distr_functions.py` - all 5 functions used (4 directly + 1 internal helper)
- `celltype_conn_by_roi.py` - `CelltypeConnByRoi` class used
- `ol_color.py` - `OL_COLOR` enum used
- `ol_types.py` - `OLTypes` class used
- `olc_client.py` - `connect` function used (51 callers)
- `cave_client.py` - `connect` function used
- `weight_prop.py` - all 4 functions used (3 directly + 1 internal helper)
- `query.py` - all 3 functions used

---

## Backup File

All removed functions are preserved in `src/utils/_unused_functions.py`, organized by
source file with section headers. This file is not meant to be imported; it serves as a
reference in case any removed function needs to be restored.
