# Judith file classifications

Per-file destination decisions for the 14 files brought into `src/judith/` by the Judith-clean merge (PR #29, commit `9273907`). Each row records where the file landed and what was rewritten / updated as part of the move.

## Decision table

| # | Bucket | File | Decision | Notes |
|---|--------|------|----------|-------|
| 1 | A | `config.py` | **Moved to `src/utils/config.py`** + dedup `load_colors` ✅ | Fixes the `utils/cache.py → judith.config` backwards dep. All importers updated. `load_colors()` was duplicated verbatim with [utils/loading_functions.py:181](../utils/loading_functions.py); deleted from `config.py` (and dropped the now-unused `pandas` import) and switched all 5 notebooks + `_run_input_rfs.py` to `from utils.loading_functions import load_colors`. `NT_COLORS` constant remains but has zero refs — flag for Stage 4. |
| 2 | B | `setup_data.py` | **Moved to fresh `src/setup_data/setup_data.py`** ✅ | Deleted old `src/setup_data/` (5 files: `setup_cns.ipynb`, `setup_flywire.ipynb`, `setup_flywire_edgelist.py`, `add_r7r8.ipynb`, `comparison_cns_flywire.ipynb`) — judith's pipeline supersedes. **Overrides** `merge_plan.md` §7.2 keep-list. Then created fresh `src/setup_data/` and moved 3 files there: `setup_data.py`, `setup_meta_malecns.ipynb`, `_verify_setup.py`. Imports updated: `from judith import setup_data as sd` → `from setup_data import setup_data as sd` in `_run_input_rfs.py` and `setup_meta_malecns.ipynb`. Notebook `sys.path` heuristic updated (`'judith'` → `'setup_data'`). Docstring at `_preprocessing.py:1678` updated. `CLAUDE.md` and `src/judith/README.md` layout sections updated. |
| 3 | B | `preprocessing.py` | **Renamed + moved to `src/setup_data/_preprocessing.py`** ✅ | Underscore prefix marks it as a setup-pipeline internal module. 7 callers updated via sed (`from judith import preprocessing as pre` → `from setup_data import _preprocessing as pre`; `from judith.preprocessing import _compute_rf` → `from setup_data._preprocessing import _compute_rf`). README + CLAUDE.md updated. |
| 4 | B | `plotting.py` | **Moved to `src/utils/plotting.py`** ✅ | Co-locates with `utils/plotting_functions.py` (low-level primitives). 6 callers updated via sed (`from judith import plotting as plot` → `from utils import plotting as plot`; `from judith.plotting import _POLARITY_DISPLAY_RENAMES` → `from utils.plotting import _POLARITY_DISPLAY_RENAMES`). README + CLAUDE.md updated. |
| 5 | B | `paper_figs.py` | **Moved to `src/make_figures/paper_figs.py`** ✅ | Co-locates with its only caller `paper_figures.ipynb`. Single import updated via sed: `from judith import paper_figs as pf` → `from make_figures import paper_figs as pf`. `XLSX = ROOT / "docs" / "paper_figs.xlsx"` path resolution unchanged. |
| 6 | C | `setup_meta_malecns.ipynb` | **Moved to `src/setup_data/`** ✅ | Bundled with `setup_data.py` move (row 2). |
| 7 | C | `preprocessing.ipynb` | **Moved to `src/setup_data/preprocessing.ipynb`** ✅ | Bundled with `_preprocessing.py` move (row 3). cell-1 updated: `sys.path` heuristic now `'setup_data'`; import line now `from setup_data import _preprocessing as pre`. |
| 8 | C | `figures.ipynb` | **Moved to `src/make_figures/figures.ipynb`** ✅ | Co-locates with `paper_figures.ipynb`. cell-1 `sys.path` heuristic updated (`'judith'` → `'make_figures'`); other imports unchanged. README "Running" step 3 updated. |
| 9 | C | `paper_figures.ipynb` | **Moved to fresh `src/make_figures/paper_figures.ipynb`** ✅ | Created new top-level folder `src/make_figures/`. cell-1 updated: `sys.path` heuristic now `'make_figures'`; imports point at `utils.plotting`, `setup_data._preprocessing`, `make_figures.paper_figs`. CLAUDE.md tree updated. |
| 10 | D | `model_comparison.ipynb` | **Moved to `src/make_figures/`** ✅ | Co-locates with the other figure-producing notebooks. cell-1 `sys.path` heuristic updated; other imports unchanged. Outputs go to `results/model_comparison/<ALT>/` regardless of notebook location. |
| 11 | D | `polarity_signed.ipynb` | **Moved to `src/make_figures/`** ✅ | Co-locates with `figures.ipynb` (signed-bar variant of polarity panels in `figures.ipynb` sec 8/10). cell-1 `sys.path` heuristic updated; other imports unchanged. |
| 12 | E | `_run_input_rfs.py` | **Moved to `src/setup_data/`** ✅ | One-off rerun of input-RF subpipeline (steps 6-7 of `setup_meta_malecns.ipynb` + Main 3c/d/e renders). Topical fit with `_verify_setup.py` and `_preprocessing.py`. Zero importers; path resolution unchanged. Hardcoded `/tmp/input_rfs.log`, `EXAMPLE_BID = 30134`, `SIDE_CHAR = 'r'` kept as-is. CLAUDE.md (5 → 6 files) and README updated. |
| 13 | E | `_verify_setup.py` | **Moved to `src/setup_data/`** ✅ | Bundled with `setup_data.py` move (row 2). One-off cross-check vs `_orig` files; hardcoded `/workspace/data` path; will likely be deleted later. |
| 14 | F | `README.md` | _TBD_ | Decided last; what's in `src/judith/` after the dust settles? |

## What still needs to be done

### Open decision

- **Row 14 — `src/judith/README.md`**: now that the pipeline is split across `src/utils/`, `src/setup_data/`, and `src/make_figures/`, where does the document describing the pipeline live? Options:
  - Stay in `src/judith/` (keeps a doc-only `judith/` folder; semantically odd since no judith code lives there anymore)
  - Move to repo root (rename to `PREPRINT.md` or similar) — surfaces the pipeline doc to anyone browsing the repo
  - Merge into the root `README.md` — fold the pipeline section into the main repo doc
  - Move to `docs/preprint_pipeline.md` — co-locate with `docs/paper_figs.xlsx` and other docs
  - Delete `src/judith/` entirely after deciding the README's home

### Smoke test (run on the cluster — local `python` is forbidden by user CLAUDE.md)

```bash
python -m py_compile $(find src -name "*.py")
cd src && python -c "from utils import config, cache, loading_functions, plotting, plotting_functions; from setup_data import setup_data, _preprocessing; from make_figures import paper_figs"
```

Then open each retained notebook and run the imports cell only.

### Stage 4 dedup followups (deferred to a separate PR)

- **`NT_COLORS` in `utils/config.py`** — zero references in the repo; dead code candidate.
- **`utils/plotting.py` ships ~14 private `_*` helpers** (`_pie`, `_box`, `_scatter`, `_heatmatrix`, `_stacked_bars`, …) that duplicate public functions in `utils/plotting_functions.py`. Could shave ~600–800 lines from `plotting.py` by importing them instead. Signatures differ slightly (kw-only `*,` separators); per-fn diff needed before swap.
- **`figures.ipynb` and `paper_figures.ipynb`** are largely overlapping orchestrators (topic-first vs. panel-first organization over the same plot calls). Candidates for dedup.
- **`utils/loading_functions.py`** has zero importers in the current tree apart from `load_colors` — likely most of the file is dead.
- **`_verify_setup.py`** is hardcoded to `/workspace/data` and `_orig` reference files that may no longer exist — likely deletable.
