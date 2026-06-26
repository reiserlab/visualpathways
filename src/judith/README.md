# connectome_interpreter_janelia

Analysis pipeline for interpreting the Janelia male Central Nervous System (maleCNS) connectome
using effective-connectivity methods from Yijie Yin's
[`connectome_interpreter`](https://github.com/YijieYin/connectome_interpreter). The repository
produces every pipeline figure of the accompanying paper from neuprint-downloaded connectivity,
via a flat preprocessing / plotting layer and two orchestrator notebooks.

> Status: internal use only.

## Layout

```
src/
  make_figures/
    figures.ipynb             orchestrator: render every paper/SI panel (per-section)
    paper_figures.ipynb       single-pass driver for every panel in paper_figs.xlsx
    model_comparison.ipynb    cross-dataset model comparison (maleCNS vs. Flywire / VNR)
    polarity_signed.ipynb     one-off polarity experiment (signed E−I bars)
    paper_figs.py             paper_figs.xlsx coverage check + panel distribution

  setup_data/
    setup_meta_malecns.ipynb  neuprint pull: meta, prop, inprop, ROI, flow, lateral flow
    preprocessing.ipynb       orchestrator: download + cache all preprocessing artefacts
    setup_data.py             neuprint fetch + meta/prop/inprop/ROI/flow builders
    _run_input_rfs.py         one-off rerun of the input-RF subpipeline + Main Fig 3c/d/e
    _verify_setup.py          one-off cross-check vs. _orig data files

  utils/
    config.py                 paths, DATASET / SIDE / N_FLOW / HIT_THRE, colour palettes
    cache.py                  cached_parquet / cached_npz / cached_pickle
    path_filtering.py         path-level filters used by plot.plot_pathway
    olc_client.py             neuprint auth helper
    external_rf.py            Gaussian RF fitting
    core_data.py              foundational loaders (get_meta, get_flow, get_ol_meta, …)
    ol_data.py                OL connectivity / clustering / pathways
    cb_data.py                CB connectivity / clustering / OL↔CB joint analyses
    vic.py                    OL↔CB and binocular VIC
    ol_rf.py                  RF fitting + cached RF data + neuPrint hexw_columnar
    polarity.py               ON/OFF polarity cross-checks
    plotting.py               paper-figure plot_* renderers (was judith/plotting.py)
    plotting_functions.py     low-level plot primitives (mollweide, hist, scatter, …)

data/            maleCNS meta CSVs, prop/inprop sparse matrices, ROI pickles, flow CSVs
results/         generated PDFs, organised by figure group (main_*, si_*, rf_fits, …)
fig_comp/        comparison PDFs across dataset versions (v0.9 vs. v0.12 etc.)
params/paper_figs.xlsx  panel → generator mapping (157 / 157 panels covered)
```

Imports use the nested form: `from utils.config import DATA_DIR`,
`from utils.cache import cached_parquet`, etc. Notebooks bootstrap `src/`
onto `sys.path` automatically, so they work whether launched from `src/`
or from `src/judith/`.

## Setup

```bash
python -m venv /home/node/venv
source /home/node/venv/bin/activate
pip install -e .
```

Dependencies are declared in `pyproject.toml`. `connectome_interpreter` is pinned to
GitHub `main` (not PyPI) because several required functions (`find_paths_of_length`,
`plot_paths`, `filter_paths`, `group_paths`) have not yet been released.

Copy `env-sample.txt` to `.env` and fill in at least:

- `NEUPRINT_SERVER_URL`, `NEUPRINT_DATASET_NAME`, `NEUPRINT_APPLICATION_CREDENTIALS`
- `SEGMENTATION_SOURCE`, `SHELL_SOURCE`, `FULL_SHELL_SOURCE`, `BRAIN_VOLUME_URL`
- Optional: `NEUVID_PATH`, `BLENDER_PATH` for movie rendering

Install the Jupyter kernel so the notebooks can find the venv:

```bash
python -m ipykernel install --user --name venv --display-name "Python (venv)"
```

## Running

1. `src/setup_data/setup_meta_malecns.ipynb` — pulls from neuprint and writes
   `data/malecns_{tag}_*` (meta, prop, inprop, ROI pickles, flow CSV, lateral flow).
   Slow: the flow step is multi-hour per side.
2. `src/setup_data/preprocessing.ipynb` — populates `data/cache/` with derived artefacts
   (clusters, RF fits, pathways, hitting-time tables, …).
3. `src/make_figures/figures.ipynb` — renders every panel into `results/<group>/`.

The dataset / side / flow parameters live in `src/utils/config.py`
(`DATASET`, `SIDE`, `N_FLOW`, `HIT_THRE`). `paper_figs.xlsx` maps each paper panel
to the generator call that produces it.

## Conventions

- Cached preprocessing helpers live in `src/utils/{core_data,ol_data,cb_data,
  vic,ol_rf,polarity}.py`; `plotting.py` (in `src/utils/`) is a flat module
  with `=== N. <topic> ===` section banners.
- `<module>.get_*` for cached preprocessing outputs (e.g.
  `core_data.get_meta()`, `ol_data.get_ol_clusters()`), `plot.plot_*` for
  figures; names are content-oriented, not figure-numbered.
- Cache keys bake in parameters that change the output (e.g. `frac_thre=0.19 → "0p190"`).
- OL-subset sidecars (`data/{dataset}_OL_{side}_*`) are built lazily from full-brain
  files by `core_data._ensure_ol_sidecars`.
- CB RF plots go to `results/CB_rf_fits/` (parallel to `rf_fits/`).

## Provenance

Most of the code started as a clone of
[`connectome_interpreter`](https://github.com/YijieYin/connectome_interpreter),
stripped of heavy dependencies (torch, navis) and wired to the Janelia CNS dataset.
CNS-specific data prep borrows from
[`connectome_data_prep`](https://github.com/YijieYin/connectome_data_prep)
and the Reiser lab
[male drosophila visual system connectome](https://github.com/reiserlab/male-drosophila-visual-system-connectome-code).

## License

GPL v3 — see the repository-root `LICENSE` file.
