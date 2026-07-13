# CLAUDE.md - Optic Lobe Connectome Analysis Repository

This file helps Claude Code understand the context, structure, and conventions of this repository.

**See also:** [`SKILLS.md`](SKILLS.md) — reusable, verified how-tos (e.g. authoring tracked changes in `.docx` files).

## Project Overview

This is a neuroscience research repository for analyzing the **optic lobe (OL) connectome** in the Drosophila (fruit fly) male brain. The codebase processes and visualizes electron microscopy (EM) connectome data from Janelia FlyEM's Male Brain Dataset (MaleCNS), with FlyWire (FAFB) comparisons, accessed via the NeuPrint API.

**Maintainer:** Reiser Lab at Janelia Research Campus
**License:** GNU General Public License v3
**Intended Users:** Internal/invited collaborators (neuroscience researchers)
**Project Type:** Research analysis codebase that renders the figures for the OL connectome paper

### Main Use Cases
- Query and analyze neural connectivity patterns in the optic lobe
- Compute signal propagation through neural circuits
- Generate receptive field (RF) maps for visual system neurons
- Cluster neurons by connectivity and analyze pathways
- Compare male vs. female (sexual dimorphism) connectivity
- Render the paper's figure panels and assemble them into a browsable viewer

## Repository Structure

> **Note (2026-06, branch `sankey_improve`):** the source tree was pruned to only the code that
> generates the paper figures. ~22k lines / 10 directories were removed (`network/`, `test/`, `rf/`,
> `pathways/`, `queries/`, `input_distribution/`, `plot_neuron/`, `eyemap/`, `ng_export/`, `judith/`)
> plus 18 `utils/` modules. The four module dirs below are what remains.
>
> **Note (2026-07):** `src/quan_propagation/` was dissolved. Its cache-building
> scripts moved to `setup_data/` as the flat numbered stages `s4_assemble_vpn_vcbn`
> (which also folds in the former `rf_size` step) and `s5_tbar` (the heavy tbar
> queries); the former `effwt_matrix` step was dropped — its per-channel table now
> lives in `utils.vic`. Its helper modules moved to `utils/` (`gallery_common.py`);
> the two gallery notebooks moved to `make_figures/` (standalone); and the
> `contrib_plot_mip` + `pokemon_plot` analyses were merged into
> `make_figures/paper_figures.ipynb` (the "AZ panels" section).
>
> **Note (2026-07-05):** `src/dimorphism/` does not exist on `main` or this
> branch — it was deleted upstream in commit `2352732` ("aggressive clean
> up") but this file wasn't updated to match. Tracked in
> [issue #42](https://github.com/reiserlab/ol-connectome-analysis/issues/42);
> the module-dir count and section below reflect its current absence.
> Also, six `make_figures/*.ipynb` entry points documented below
> (`exp_aspect_comp.ipynb`, `flywire_main_2d_4d_polarity.ipynb`,
> `model_comparison.ipynb`, `polarity_signed.ipynb`,
> `propagated_arf_examples.ipynb`, `ring_neurons.ipynb`) were deleted in the
> same commit and are removed from the list below. See
> [`docs/run_order.md`](docs/run_order.md) for the current, verified
> setup→figures run order and known gaps.
>
> **Note (2026-07-09):** the panel-manifest/distributor module was renamed and
> moved up out of `lib/`: `make_figures/lib/paper_figs.py` →
> `make_figures/figures.py` (the `make_figures/lib/` subfolder is now gone).
> Import it as `from make_figures import figures as pf`. It derives its
> `results/` path from `utils.config.FIG_DIR` (no longer from `__file__`, which
> had silently pointed `distribute()` at `src/results/`). There is **no**
> `figures.ipynb`; the main figure notebook is `paper_figures.ipynb`.

```
├── src/                      # Main source code — 3 active module dirs (dimorphism/ currently absent, see note above)
│   ├── utils/                # Core utilities: cached-data loaders + the plotting stack
│   ├── make_figures/         # Paper-figure orchestrator notebooks + the panel-manifest helper
│   └── setup_data/           # MaleCNS/FlyWire data download + preprocessing pipeline
├── docs/                     # Documentation (setup guides, git workflow, methods notes, run_order.md)
├── cache/                     # Gitignored cache: cache/data/ (raw, setup_data-written) + cache/data_proc/ (derived, lazily built) + cache/model_comparison/ (SI 5/6 flow variants)
├── params/                   # Parameter files — incl. paper_figs.xlsx (the figure→panel manifest)
│                             #   and TableS2_experimental_data.xlsx (experimental data source)
├── results/                  # Analysis outputs, incl. html_figures/ (interactive-panel viewer)
├── pyproject.toml            # Python dependencies (the real manifest; requirements.txt is a thin `pip install -e .` shim)
├── requirements-lock.txt     # Pinned versions for an exactly-reproducible env
├── .env_sample               # Configuration template (copy to .env)
├── README.md                 # Brief project description
└── LICENSE                   # GNU GPL v3
```

**Where the data comes from:** connectome data is fetched via the **`connectome_interpreter`** package
(wrapped inside `utils/core_data.py`), *not* a local query layer. The old `src/queries/` NeuPrint
wrapper was figure-unused and has been removed.

## Core Concepts

### 1. OL Neuron Types and Instances
- **Types:** classes of neurons (e.g., "T4a", "Mi1", "Tm3")
- **Instances:** individual neurons of a type (e.g., "T4a_001")
- Loaded from cached metadata via `utils/core_data.py` and `utils/ol_data.py`.

### 2. Region of Interest (ROI) Analysis
- **ROIs:** anatomical brain regions (e.g., "ME", "LO", "LOP")
- **Layers:** subdivisions within ROIs (e.g., "ME(R7)", "ME(R8)")
- Geometry/hex-grid/layer helpers: `utils/geometry.py`, `utils/hex_hex.py`; layer assignment is precomputed by `setup_data/s2_layers.py`.

### 3. Signal Propagation
- Track how signals flow through multi-layer neural circuits; compute effective weights, contributions, influence.
- The "AZ"-marked paper panels (visual-input contribution / resolution / coverage projections) are rendered by the "AZ panels" section of `make_figures/paper_figures.ipynb`; their heavy cache is built by `setup_data/` (`s4_assemble_vpn_vcbn`, `s5_tbar`).

### 4. Receptive Fields (RF)
- Map visual-space locations that drive neuronal responses (Gaussian fitting, retinotopic mapping).
- Code: `utils/ol_rf.py`, `utils/external_rf.py`; the input syn-per-column data is precomputed by `setup_data/s1_malecns_data.ipynb` (§6), and RF fits are computed on demand by `ol_rf`.

### 5. Clustering & Pathways
- Cluster neurons by connectivity (Leiden, via `igraph`'s built-in implementation) and analyze pathways / cluster graphs.
- Code: `utils/ol_data.py`, `utils/cb_data.py` (clustering), `utils/path_filtering.py`, and the cluster-graph plotters in `utils/plotting.py`.

## Code Organization by Module

### `src/utils/` — Core Utilities (Most Important!)
**Before implementing new functionality, check this directory first.** Two layers live here: cached-data loaders and the plotting stack. `utils/plotting.py` is the central save helper — it writes every per-topic results folder.

- **`config.py`** — central config: paths (incl. `DATA_DIR`, `CACHE_PROC_DIR`, `MODEL_COMPARISON_DIR`, `FIG_DIR`), dataset selection (`FIG_DATASET`), shared color palettes
- **`core_data.py`** — foundational cached-data loaders shared across every downstream module (wraps `connectome_interpreter`)
- **`ol_data.py`** — OL-side connectivity, clustering, pathways, and related cached data
- **`cb_data.py`** — CB-side connectivity, clustering, and OL↔CB joint analyses
- **`vic.py`** — OL↔CB and binocular vicinity (VIC) cached data
- **`cache.py`** — cache helpers for preprocessing outputs
- **`ol_rf.py`** — receptive-field utilities for OL neurons (incl. `count_col_ahull`, the hex-grid alpha-hull column-extent metric)
- **`external_rf.py`** — external/grid receptive-field fitting helpers
- **`polarity.py`** — ON/OFF polarity cross-checks against the fly-literature spreadsheet
- **`geometry.py`**, **`hex_hex.py`** — geometry / hex-grid / depth utilities
- **`plotting.py`** — the main plotting + save module (panels grouped into sections 1..11 by figure group)
- **`figure_helpers.py`** — reusable helpers extracted from `paper_figures.ipynb` (signed polarity-bar family, per-instance effective-#-inputs, label wrapping)
- **`plotting_functions.py`** — mollweide / polar / pyvis plotting family
- **`projection_plots.py`** — 2D extreme-value projection / neuron-with-outline plots (AZ contribution panels)
- **`pdf_report.py`**, **`plotter.py`**, **`gallery_index.py`**, **`gallery_common.py`** — gallery/PDF-report figure builders (used by the `make_figures/gallery_*.ipynb` notebooks)
- **`alpha_shape.py`**, **`ng_view.py`** — alpha-hull geometry + misc helpers
- **`palettes.py`** — hardcoded colour palettes for paper-figure renderers
- **`path_filtering.py`** — path-level filters for pathway example diagrams
- **`ol_color.py`** — colour-space helpers
- **`query_roi.py`** — primary-ROI lookup helper (`get_primary_rois`), used by the AZ-panel scripts
- **`olc_client.py`** — NeuPrint/OLC client setup
- **`config.py`/`olc_client.py`** provide the project/auth bootstrap used across modules

### `src/quan_propagation/` — removed (dissolved 2026-07)
This directory no longer exists. Its contents were redistributed:
- **Cache-building scripts → `setup_data/`:** the flat numbered stages `s4_assemble_vpn_vcbn` (VPN/VCBN assembly + the combined RF caches, folding in the former `rf_size` step) and `s5_tbar` (the ~100 min tbar-location query + ~9 h per-ROI tbar counts). The former `effwt_matrix` step was dropped — its per-channel effective-weight table now lives in `utils.vic`.
- **Helper modules → `utils/`:** `gallery_common.py` (and the earlier-ported `projection_plots.py`, `pdf_report.py`, `plotter.py`, etc.).
- **Galleries → `make_figures/`:** `gallery_vpn.ipynb`, `gallery_vcbn.ipynb` (standalone; import `from utils.gallery_common import make_bar_fig`).
- **Panel analyses → `make_figures/paper_figures.ipynb`:** the `contrib_plot_mip` (Main 5g/6a/7c, SI 11/12) and `pokemon_plot` (coverage, Main 6b) cells now live in the "AZ panels" section near the end of the notebook.

### `src/make_figures/` — Paper-Figure Rendering
- **Plotting entry points (notebooks):** `paper_figures.ipynb` (main; includes the "AZ panels" section), `gallery_vpn.ipynb`, `gallery_vcbn.ipynb` (standalone per-instance galleries). Seven other entry points once documented here — `figures.ipynb` (previously labelled "main"), `exp_aspect_comp.ipynb`, `flywire_main_2d_4d_polarity.ipynb`, `model_comparison.ipynb`, `polarity_signed.ipynb`, `propagated_arf_examples.ipynb`, `ring_neurons.ipynb` — no longer exist.
- **`figures.py`** — maps result PDFs/sidecars produced by the plotters onto per-panel filenames in `results/<topic>/`; `distribute()` + `build_index()` rebuild `results/html_figures/`. Only rows whose `results file` cell is a literal `Blender`/`Neuroglancer`/`custom`/`AZ` tag (the `EXTERNAL` set) are excluded from `distribute()` — those have no auto-distributable file. The AZ visual-projection panels themselves point at real `quan_propagation/v1/...` paths, so they **are** distributed into the numbered folders by the normal `distribute('main_5'/…)` calls; only their `html_figures/` viewer entry is added by hand (AZ panels are PDF-only, no `.html` sidecar).
- **`.py` files:** jupytext pairs of the notebooks above (`paper_figures.py`, `gallery_vpn.py`, `gallery_vcbn.py`) plus the `figures.py` manifest helper. (The former standalone side scripts `compare_effective_n_vpn_vcbn.py` / `explore_cbn_to_vcn.py` were deleted.)

### `src/setup_data/` — Data Prep Pipeline
Precomputes the cached data + input ARFs the figure code reads. **Kept even though figure code never imports it** — it's a separate entry point you run to (re)build the cache. Flat layout: two library modules (`malecns.py`, `flow.py`) plus five jupytext-paired numbered stages (`s1`–`s5`); there is **no** `lib/`, `base_cache/`, or `cb_cache/` subfolder.
- **`malecns.py`** — MaleCNS meta + sparse-matrix assembly functions (library; imported as `sd` by `s1_malecns_data`)
- **`flow.py`** — dataset-agnostic lateral-flow matrix builders (library; imported as `sd` by `s3_effconn`)
- **`s1_malecns_data`**, **`s2_layers`**, **`s3_effconn`** — dataset/layer/effective-connectivity builders, run in that order (`s3_effconn` also builds the FlyWire lateral flow via the override cell in its "Dataset selection" section; `s2_layers` also holds the item-5 model-comparison flow-variant cells that write `cache/model_comparison/`)
- **Input-RF caches** — no separate stage; the four `utils.ol_rf` input-RF caches build lazily on first figure call. Force a rebuild with `ol_rf.rebuild_input_rf_caches(force=True)` after re-running `s1_malecns_data` §6 "Input RF raw data".
- **`s4_assemble_vpn_vcbn`**, **`s5_tbar`** — build the AZ-panel cache, run in that order (see the `quan_propagation` dissolution note above). `s4` assembles the VPN/VCBN tables and the combined RF caches (ellipse + alpha hull, the former standalone `rf_size` step); `s5` runs the heavy tbar queries. The former `effwt_matrix` step was removed: its per-channel effective-weight table now lives in `utils.vic.get_ol_cb_effweight_by_channel` (backing `get_effweight_channel`), and the galleries compute the per-neuron RF on demand via `utils.core_data.grouped_effweight`.

### `src/dimorphism/` — Sexual Dimorphism (currently absent — see issue #42)
This directory does not exist on `main` or this branch; it was deleted in commit
`2352732` ("aggressive clean up") without a corresponding CLAUDE.md update. When
present it held:
- **`dimorphism_analysis.py`/`.ipynb`** — male vs. female dimorphism analysis (table + figure panels)
- **`flywire_dimorphism_panel.py`** — FlyWire (CB visual-fraction) dimorphism panel; dimorphism flag comes from neuprint `flywire-fafb:v783b`, now a column in the flywire meta CSV

Orphaned outputs (`results/neurons_dimorphism.csv`,
`results/tables/malecns_v1.0_dimorphism.csv`) and the Main 7c-left panel in
`params/paper_figs.xlsx` still reference this analysis with no current
generating code — see
[GitHub issue #42](https://github.com/reiserlab/ol-connectome-analysis/issues/42)
for the open question of whether this was an intentional removal (source is
still recoverable via `git show 2352732^:src/dimorphism/`).

## Key Technologies & Dependencies

### Neuroscience Libraries
- **`neuprint-python`** (forked): NeuPrint connectome API
- **`navis`** (forked): Neuroanatomy visualization
- **`connectome_interpreter`**: connectivity/effective-weight analysis (primary data layer)
- **`caveclient`**, **`fafbseg`**, **`flybrains`**: FlyWire/CAVE tooling and brain atlases

### Scientific Stack
- **numpy**, **scipy**, **pandas**, **scikit-learn**: data processing
- **networkx**, **python-igraph**: graph analysis & clustering (Leiden clustering uses `igraph`'s own built-in implementation, not the separate `leidenalg` package — that package isn't a dependency)
- **alphashape**, **trimesh**, **Shapely**: geometric computations

### Visualization
- **matplotlib**, **plotly**, **pyvis**: 2D/3D visualization
- **colormath**: color space conversions

### Infrastructure
- **cloud_volume**: 3D volumetric data access
- **python-dotenv**: environment configuration
- **requests**, **tqdm**: HTTP client and progress bars

**Note:** Two dependencies are custom forks at `@master` branch (see "Known Issues" below).

## Development Guidelines

### Environment Setup
1. **Copy `.env_sample` to `.env`** and fill in credentials:
   ```bash
   NEUPRINT_SERVER_URL=neuprint-cns.janelia.org
   NEUPRINT_DATASET_NAME=cns
   NEUPRINT_APPLICATION_CREDENTIALS=<your_token>
   ```
   Get your token from https://neuprint-cns.janelia.org/account

2. **Install dependencies:**
   ```bash
   pip install -e .                        # pyproject.toml is the real manifest
   # or, for an exactly-reproducible pinned env:
   pip install -r requirements-lock.txt
   ```
   `requirements.txt` still works (`pip install -r requirements.txt`) but is just a
   thin `-e .` + `mypy` shim over `pyproject.toml` — don't edit dependencies there.
   The project's conda env is `ol-connectome` (Jupyter kernel `ol-c-kernel`); use that interpreter to run repo code.

3. **Project setup boilerplate** (used in most scripts):
   ```python
   from pathlib import Path
   import sys
   from dotenv import load_dotenv, find_dotenv
   PROJECT_ROOT = Path(find_dotenv()).parent
   sys.path.append(str(PROJECT_ROOT.joinpath('src')))
   ```
   **Note:** This boilerplate is temporary; planned reorganization will make the project a proper package.

### Import Conventions
Since `src/` lacks `__init__.py` files, imports look like:
```python
from utils.core_data import get_meta, get_ol_meta
from utils import plotting as plot
from make_figures import figures as pf
```

### Jupyter Notebook Workflow
- Most analysis is done in Jupyter notebooks (the `make_figures/*.ipynb` entry points).
- Notebook `.py` jupytext pairs exist alongside the `.ipynb` files.
- Use `NAVIS_JUPYTER_PLOT3D_BACKEND=plotly` for 3D plotting in notebooks.

### Git Workflow
- PR-based workflow (see `docs/git-getting-started.md`).
- Branch naming: descriptive names (e.g., `feature/vpn-clustering`).

## Common Tasks & Patterns

### Render paper figures
Open the relevant `src/make_figures/*.ipynb` (start with `paper_figures.ipynb`), which pulls cached data through
`utils/core_data.py` / `utils/ol_data.py` and plots via `utils/plotting.py`. Saving is centralized in
`utils/plotting.py`, which writes a PDF and an interactive HTML sidecar into `results/<topic>/`.

### Distribute panels into the viewer
After plots are generated, `src/make_figures/figures.py` maps result files onto per-panel filenames using
the manifest `params/paper_figs.xlsx`:
```python
from make_figures import figures as pf
pf.distribute()      # rename/copy topic-folder sidecars into results/html_figures/
pf.build_index()     # regenerate results/html_figures/index.html
```

### (Re)build cached data
Run the `src/setup_data/` numbered stages (e.g. `s1_malecns_data`, `s2_layers`, `s3_effconn`) to rebuild
the meta/prop/inprop matrices and input RFs that the figure code consumes. This hits NeuPrint and is slow.

### Render the AZ panels
The "AZ panels" section near the end of `make_figures/paper_figures.ipynb` renders the AZ-marked panels
(visual-input contribution, resolution, coverage). It reads cache built by `setup_data/s5_tbar`
(+ `s4_assemble_vpn_vcbn`). The per-instance galleries are separate
standalone notebooks: `make_figures/gallery_vpn.ipynb`, `make_figures/gallery_vcbn.ipynb`.

## Important Notes for Claude Code

### DO:
✅ **Check `src/utils/` first** before implementing new utilities — extensive reusable functions exist
✅ **Fetch data through `utils/core_data.py`** (`connectome_interpreter`), not a hand-rolled NeuPrint query
✅ **Respect research context** — this is scientific analysis code, prioritize correctness
✅ **Keep `src/setup_data/`** — it's the data-prep pipeline even though figure code never imports it
✅ **Follow PR workflow** for changes (see `docs/git-getting-started.md`)
✅ **Be careful with data files** — `cache/data/`, `cache/data_proc/`, and the `params/*.xlsx` manifests are research inputs; the user edits the xlsx files live
✅ **Use project setup boilerplate** at the top of standalone scripts (see "Environment Setup")

### DON'T:
❌ **Don't reintroduce the removed modules** — `queries/`, `network/`, `rf/`, `pathways/`, `input_distribution/`, `quan_propagation/`, `dimorphism/` (currently absent, see issue #42), etc. were deleted as figure-unused (or pending a restore decision)
❌ **Don't modify data files** without explicit permission
❌ **Don't implement functionality that likely exists** in `src/utils/`
❌ **Don't break NeuPrint authentication** — the `.env` file is critical
❌ **Don't make breaking changes** to widely-imported modules (`utils/core_data`, `utils/plotting`, `utils/config`)
❌ **Don't assume package imports work** — project lacks `__init__.py` files (see "Known Issues")
❌ **Don't rely on `write_image` (PDF/PNG export)** in the CLI env — kaleido is broken there; HTML sidecars via `write_html` work

### Code Quality Considerations
- **Large files exist:** `utils/plotting.py`, `utils/cb_data.py`, and `make_figures/paper_figures.ipynb` are large (mixed concerns).
- **Debug code:** DEBUG comments remain (especially in the ported `utils/projection_plots.py` / the AZ-panel cells).
- **Stale docs vs. code:** much historical code has been removed; verify a file/symbol exists before relying on it.

## Known Issues & Technical Debt

From `docs/repo_reorg.md`, the following issues are known:

1. **Not a proper Python package:** No `__init__.py` files, requires `sys.path.append` boilerplate
2. **Git dependencies unpinned:** `navis` and `neuprint-python` forks track `@master` (non-reproducible)
3. **No linting/CI:** No automated code quality checks
4. **No test framework:** there is no test suite (the old `src/test/` scripts were removed)
5. **Large monolithic files:** some modules are 3,000+ lines (mixed concerns)
6. **No Git LFS:** large CSV files (~50 MB) tracked directly in git
7. **Broken PDF export:** kaleido in the CLI env can't `write_image`; figures are rendered in the user's own notebook

## Future Improvements (Planned)

Priority improvements documented in `docs/repo_reorg.md`:

1. **Convert to proper package** - Add `pyproject.toml`, `__init__.py` files, make pip-installable
2. **Pin dependencies** - Lock `navis` and `neuprint-python` to specific commit SHAs
3. **Add linting/CI** - Set up ruff and GitHub Actions workflow
4. **Set up pytest** - Add a proper test suite
5. **Expand README** - Add architecture overview, setup guide, data pipeline explanation
6. **Refactor large files** - Break up 3,000+ line modules into focused components
7. **Use Git LFS** - Migrate large data files to LFS

## Additional Resources

- **Documentation:** See `docs/` directory for setup guides, git workflow, plotting tips
- **Original codebase:** Some code carried over from [male-drosophila-visual-system-connectome-code](https://github.com/reiserlab/male-drosophila-visual-system-connectome-code)
- **NeuPrint API:** https://neuprint-cns.janelia.org/
- **Janelia FlyEM:** https://www.janelia.org/project-team/flyem
- **Reiser Lab:** https://www.janelia.org/lab/reiser-lab/

---

**Last Updated:** 2026-07-11
**Repository Stats:** 3 active module directories under `src/` (`utils/`, `make_figures/`, `setup_data/`) — `dimorphism/` is documented above but currently absent, see issue #42 — GNU GPL v3 license
