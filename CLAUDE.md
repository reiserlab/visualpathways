# CLAUDE.md - Optic Lobe Connectome Analysis Repository

This file helps Claude Code understand the context, structure, and conventions of this repository.

## Project Overview

This is a neuroscience research repository for analyzing the **optic lobe (OL) connectome** in the Drosophila (fruit fly) male brain. The codebase processes and visualizes electron microscopy (EM) connectome data from Janelia FlyEM's Male Brain Dataset, accessed via the NeuPrint API.

**Maintainer:** Reiser Lab at Janelia Research Campus
**License:** GNU General Public License v3
**Intended Users:** Internal/invited collaborators (neuroscience researchers)
**Project Type:** Research analysis codebase with Python scripts and Jupyter notebooks

### Main Use Cases
- Query and analyze neural connectivity patterns in the optic lobe
- Compute signal propagation through neural circuits
- Generate receptive field maps for visual system neurons
- Visualize connectome data in 2D/3D
- Perform network/graph analysis of neural pathways
- Cluster neurons by connectivity patterns

## Repository Structure

```
├── src/                      # Main source code (63 Python files, ~34,558 lines)
│   ├── utils/                # Core utilities (28 files, 10,350+ lines)
│   ├── queries/              # NeuPrint data querying (4 files)
│   ├── quan_propagation/     # Quantitative signal propagation (18 files)
│   ├── network/              # Network/graph analysis (3 files)
│   ├── pathways/             # Pathway clustering (1 file)
│   ├── rf/                   # Receptive field analysis (7 files)
│   ├── input_distribution/   # Input distribution analysis (2 files)
│   ├── eyemap/               # Eye mapping utilities (1 file)
│   ├── weight_propagation/   # Weight propagation notebooks
│   ├── ng_export/            # Neuroglancer export (1 notebook)
│   ├── setup_data/           # MaleCNS data download + preprocessing pipeline (6 files)
│   ├── make_figures/         # Paper-figure rendering: orchestrators, side analyses, manifest helper (5 files)
│   └── test/                 # Test scripts (6 files)
├── docs/                     # Documentation (setup guides, git workflow, etc.)
├── data/                     # Data files (eyemap/, flywire/, collaborative data)
├── params/                   # Parameter files for analysis
├── results/                  # Output directory for analysis results
├── requirements.txt          # Python dependencies
├── .env_sample               # Configuration template (copy to .env)
├── README.md                 # Brief project description
└── LICENSE                   # GNU GPL v3
```

## Core Concepts

### 1. Optic Lobe (OL) Neuron Types and Instances
- **Types:** Classes of neurons (e.g., "T4a", "Mi1", "Tm3")
- **Instances:** Individual neurons of a type (e.g., "T4a_001", "T4a_002")
- Accessed via `src/utils/ol_types.py` (class-based interface)

### 2. Region of Interest (ROI) Analysis
- **ROIs:** Anatomical brain regions (e.g., "ME", "LO", "LOP")
- **Layers:** Subdivisions within ROIs (e.g., "ME(R7)", "ME(R8)")
- Key files: `ROI_calculus.py`, `ROI_plots.py`, `ROI_layers.py`, `ROI_voxels.py`

### 3. Signal Propagation
- Track how signals flow through multi-layer neural circuits
- Compute effective weights, contributions, and influence
- Main module: `src/quan_propagation/`

### 4. Receptive Fields (RF)
- Map visual space locations that drive neuronal responses
- Gaussian fitting and retinotopic mapping
- Module: `src/rf/`

### 5. Network Analysis
- Graph-based circuit analysis with networkx/igraph
- Flow level assignment using Bayesian traversal models
- Module: `src/network/`

## Code Organization by Module

### `src/utils/` - Core Utilities (Most Important!)
**Before implementing any new functionality, check this directory for existing utilities.**

Key files:
- **`ol_types.py`** (256 lines): Cell type definitions and access
- **`connectivity.py`** (341 lines): Instance connectivity filtering
- **`ROI_calculus.py`** (785 lines): ROI intersection and volume calculations
- **`ROI_plots.py`** (519 lines): ROI visualization functions
- **`ROI_layers.py`** (580 lines): Layer-based ROI analysis
- **`plotting_functions.py`** (628 lines): Matplotlib/Plotly plotting utilities
- **`celltype_conn_by_roi.py`** (1,109 lines): Cell type connectivity by region
- **`query.py`** (345 lines): Standalone query utilities with project setup
- **`ol_rf.py`**: Receptive field utilities for optic lobe neurons
- **`weight_prop.py`**: Weight propagation helpers
- **`input_distr_functions.py`**: Input distribution analysis
- **`_unused_functions.py`** (2,458 lines): Recently removed dead code (commit 281ac66)

### `src/queries/` - Data Querying
Interfaces with NeuPrint database and remote data sources:
- **`completeness.py`** (13,144 lines): Fetch neuron types/instances from NeuPrint
- **`query_data.py`** (11,308 lines): General data retrieval
- **`types_and_instances.py`** (9,243 lines): Cell type/instance management
- **`ol_neuron.py`** (2,881 lines): OL-specific neuron queries

### `src/quan_propagation/` - Signal Propagation (Main Analysis Module)
18 files for tracking signal flow through connectomes:
- **`func.py`**: ROI outline generation, shape analysis
- **`computing_functions.py`**: Effective weight calculation, path traversal
- **`external_rf.py`**: Receptive field fitting (Gaussian)
- **`binocular_cb.py`**, **`retinotopy_cb.py`**, **`resolution_cb.py`**: Central brain variants
- **`contrib_*.py`**: Contribution/influence calculations
- **`gallery_*.py`**: Visualization galleries
- **`vpn_collab.py`**: VPN (visual projection neuron) collaboration

### `src/network/` - Network Analysis
- **`flow_level_traversalModels.py`**: Bayesian flow level assignment
- **`flow_level_draw_digraph.py`**: Network visualization
- **`prop_by_adj.py`**: Adjacency-based propagation

### `src/pathways/` - Pathway Analysis
- **`clustering_vpn.py`**: Cluster VPNs by OL inputs

### `src/rf/` - Receptive Field Analysis
- **`example_make_RF_malecns.ipynb/py`**: Tutorial for generating RF maps
- **`find_col_input_malecns.ipynb/py`**: Find columnar inputs
- **`rf_malecns_Yijie.ipynb`** (5.3MB): Extensive RF analysis notebook

### `src/input_distribution/` - Input Distribution
- **`find_input_distr.py`**: Find distribution patterns
- **`plot_input_distr.py`**: Visualize distributions

## Key Technologies & Dependencies

### Neuroscience Libraries
- **`neuprint-python`** (forked): NeuPrint connectome API
- **`navis`** (forked): Neuroanatomy visualization
- **`caveclient`**: Connectome Annotation Versioning Engine
- **`fafbseg`**: FlyWire segmentation tools
- **`flybrains`**: Drosophila brain atlases

### Scientific Stack
- **numpy**, **scipy**, **pandas**, **scikit-learn**: Data processing
- **networkx**, **python-igraph**, **leidenalg**: Graph analysis
- **alphashape**, **trimesh**, **Shapely**: Geometric computations

### Visualization
- **matplotlib**, **plotly**, **pyvis**: 2D/3D visualization
- **colormath**: Color space conversions

### Infrastructure
- **cloud_volume**: 3D volumetric data access
- **python-dotenv**: Environment configuration
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
   pip install -r requirements.txt
   ```

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
from utils.ol_types import OLTypes
from queries.completeness import fetch_neuron_types
from quan_propagation.func import compute_effective_weights
```

### Jupyter Notebook Workflow
- Most analysis is done in Jupyter notebooks
- Notebooks are located in module directories (e.g., `src/rf/*.ipynb`)
- Use `NAVIS_JUPYTER_PLOT3D_BACKEND=plotly` for 3D plotting in notebooks

### Git Workflow
- PR-based workflow (see `docs/git_workflow.md`)
- Branch naming: descriptive names (e.g., `feature/vpn-clustering`)
- Recent clean-up: commit 281ac66 removed 1,637 lines of unused code

### Testing
- Test scripts in `src/test/` (not yet a proper test suite)
- Current tests: color conversions, propagation, visualization (Plotly, PyVis)
- **Planned:** Migrate to pytest framework (see "Future Improvements")

## Common Tasks & Patterns

### 1. Query Neuron Types and Instances
```python
from utils.ol_types import OLTypes
from queries.completeness import fetch_neuron_types

ol = OLTypes()
t4_neurons = ol.get_type('T4')
# Query instances from NeuPrint
instances = fetch_neuron_types('T4*')
```

### 2. Compute Connectivity Patterns
```python
from utils.connectivity import filter_connections
from utils.celltype_conn_by_roi import get_connectivity_by_roi

# Filter connections by weight threshold
strong_conn = filter_connections(conn_df, weight_threshold=5)
# Get connectivity grouped by ROI
roi_conn = get_connectivity_by_roi(neuron_type='T4', roi='ME')
```

### 3. Run Signal Propagation Analysis
```python
from quan_propagation.computing_functions import compute_effective_weights
from quan_propagation.func import generate_roi_outlines

# Compute signal flow through circuit
weights = compute_effective_weights(connections, input_signals)
# Generate ROI outlines for visualization
outlines = generate_roi_outlines(roi_name='ME')
```

### 4. Generate Receptive Field Maps
```python
from rf.example_make_RF_malecns import generate_rf_map
from utils.ol_rf import compute_rf_center

# Generate RF map for neuron type
rf_map = generate_rf_map(neuron_type='T4a')
# Compute RF center coordinates
center = compute_rf_center(rf_map)
```

### 5. Create Visualizations
```python
from utils.plotting_functions import plot_connectivity_matrix
from utils.ROI_plots import plot_roi_layers

# Plot connectivity matrix
plot_connectivity_matrix(conn_df, save_path='results/conn_matrix.png')
# Plot ROI layers
plot_roi_layers(roi='ME', layers=['R7', 'R8'])
```

## Important Notes for Claude Code

### DO:
✅ **Check `src/utils/` first** before implementing new utilities - extensive reusable functions exist
✅ **Use existing patterns** for ROI calculations, plotting, and connectivity analysis
✅ **Respect research context** - this is scientific analysis code, prioritize correctness
✅ **Test with notebooks** when possible (most modules have example notebooks)
✅ **Follow PR workflow** for changes (see `docs/git_workflow.md`)
✅ **Read existing documentation** in `docs/` directory before major changes
✅ **Be careful with data files** - `data/` contains research data, don't modify without permission
✅ **Use project setup boilerplate** at the top of standalone scripts (see "Environment Setup")
✅ **Check for TODOs** - 54+ TODO/FIXME comments exist, may provide context

### DON'T:
❌ **Don't create new files unnecessarily** - prefer editing existing utilities
❌ **Don't modify data files** without explicit permission
❌ **Don't implement functionality that likely exists** in `src/utils/`
❌ **Don't ignore existing conventions** (imports, plotting styles, etc.)
❌ **Don't break NeuPrint authentication** - `.env` file is critical
❌ **Don't delete debug code** without understanding context (37+ DEBUG comments exist)
❌ **Don't make breaking changes** to widely-imported modules (`utils/`, `queries/`)
❌ **Don't assume package imports work** - project lacks `__init__.py` files (see "Known Issues")

### Code Quality Considerations
- **Large files exist:** Some modules are 3,000+ lines (see `docs/repo_reorg.md`)
- **Known bugs:** Potential infinite recursion in `ol_color.py:160`, unused assignments in `helper.py`
- **Typos:** "Parmeters" in `ol_types.py:49`
- **Debug code:** 37+ DEBUG comments (especially in `quan_propagation/`)
- **Dead code:** Recently cleaned up 1,637 lines, but more may exist

## Known Issues & Technical Debt

From `docs/repo_reorg.md`, the following issues are known:

1. **Not a proper Python package:** No `__init__.py` files, requires `sys.path.append` boilerplate
2. **Git dependencies unpinned:** `navis` and `neuprint-python` forks track `@master` (non-reproducible)
3. **No linting/CI:** No automated code quality checks
4. **No test framework:** `src/test/` contains scripts, not pytest suites
5. **Large monolithic files:** Some files are 3,000+ lines (mixed concerns)
6. **No Git LFS:** Large CSV files (~50 MB) tracked directly in git

## Future Improvements (Planned)

Priority improvements documented in `docs/repo_reorg.md`:

1. **Convert to proper package** - Add `pyproject.toml`, `__init__.py` files, make pip-installable
2. **Pin dependencies** - Lock `navis` and `neuprint-python` to specific commit SHAs
3. **Add linting/CI** - Set up ruff and GitHub Actions workflow
4. **Set up pytest** - Migrate test scripts to proper test framework
5. **Expand README** - Add architecture overview, setup guide, data pipeline explanation
6. **Refactor large files** - Break up 3,000+ line modules into focused components
7. **Use Git LFS** - Migrate large data files to LFS
8. **Clean up tech debt** - Triage 54+ TODOs, fix bugs, add type hints

## Additional Resources

- **Documentation:** See `docs/` directory for setup guides, git workflow, plotting tips
- **Original codebase:** Some code carried over from [male-drosophila-visual-system-connectome-code](https://github.com/reiserlab/male-drosophila-visual-system-connectome-code)
- **NeuPrint API:** https://neuprint-cns.janelia.org/
- **Janelia FlyEM:** https://www.janelia.org/project-team/flyem
- **Reiser Lab:** https://www.janelia.org/lab/reiser-lab/

---

**Last Updated:** 2026-02-06
**Repository Stats:** 63 Python files, ~34,558 lines of code, GNU GPL v3 license
