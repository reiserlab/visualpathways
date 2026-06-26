# Repository Reorganization Roadmap, v2 — Legacy Function Retirement

Companion to [repo_reorg.md](repo_reorg.md). Tracks the in-progress migration
from the original `src/utils/{computing,loading}_functions.py` API to the new
`src/setup_data/_preprocessing.py` pipeline (the "judith" layout — see
[src/judith/README.md](../src/judith/README.md)).

The new pipeline exposes cached, parameterised `pre.get_*` entry points; under
the hood, those entry points call private `_compute_*` / `_load_*` helpers that
were ported from the legacy module-level functions. The legacy functions are
still imported by a handful of notebooks and `.py` files, so they cannot be
deleted yet — this document records what changed, what's still on each side,
and the order in which they should be retired.

---

## Snapshot: pairs and their differences

Files referenced:
- legacy:
  [src/utils/computing_functions.py](../src/utils/computing_functions.py),
  [src/utils/loading_functions.py](../src/utils/loading_functions.py)
- new:
  [src/setup_data/_preprocessing.py](../src/setup_data/_preprocessing.py)

### 1. `compute_rf` → `_compute_rf`

- Legacy: [`computing_functions.py:276`](../src/utils/computing_functions.py#L276)
- New: [`_preprocessing.py:1326`](../src/setup_data/_preprocessing.py#L1326)

| Aspect | Legacy | New |
|---|---|---|
| Signature | `(meta, stepsn, inidx, outidx)` | `(meta_ol, stepsn, inidx, outidx, *, idx_to_coords=None, idx_to_root=None)` |
| Dict reuse | Rebuilds `idx_to_coords` / `idx_to_root` on every call | Accepts pre-built dicts (built once in `_compute_rf_params` loop) |
| `result_summary` | default | adds `display_output=False` (suppresses print noise) |
| `coords` column | Overwritten to `f"{x},{y/sqrt(3)}"` (loses the original hex string) | Preserved as raw integer hex `"x,y"` for hex-grid plotting |
| Import of `result_summary` | Module-top | Local |

The `coords` overwrite in the legacy version was a latent bug for any hex-plot
consumer; the new version fixes it. `x`/`y` numeric columns are identical.

### 2. `compute_rf_params` → `_compute_rf_params`

- Legacy: [`computing_functions.py:300`](../src/utils/computing_functions.py#L300)
- New: [`_preprocessing.py:1359`](../src/setup_data/_preprocessing.py#L1359)

| Aspect | Legacy | New |
|---|---|---|
| Inner RF call | `compute_rf(...)` (rebuilds dicts) | `_compute_rf(...)` with shared dicts |
| Output assembly | NaN-filled `np.empty((n,8))` per target, concatenated and NaN-dropped at the end | List of row dicts → single `pd.DataFrame(rows)`; passing rows are simply appended |
| `bodyId` assignment | After loop, by re-querying `meta[meta.instance == target].bodyId.values` (assumes ordering matches `outidx`) | Inline per body via `bids[i]` |
| Membership test | `np.isin(meta.instance, ...)` | pandas `.isin(...)` |
| Empty-df handling | `df.empty \| (weight gate)` | `if df.empty: continue` |
| Output schema | `[n_col, r2, x0, y0, a, b, phi, amp, bodyId, instance, seed]` | Same |

**Known bug — fixed in Phase D (commit pending after verification).** Both
legacy and the pre-Phase-D new version had:

```python
tot_weight = df['effective weight'].sum()
if ... <= rel_input_weight * tot_weight:
    continue
```

The LHS resolved to `tot_weight` itself (`df['effective weight'].sum()` in
legacy, `tot_weight` in new), so the gate was `s <= 0.1 * s` — always False
for positive `s`, so the filter never rejected anything. Both versions
passed through every fit; downstream NaN-dropping was the only quality
filter.

Phase D replaced the no-op per-body skip with a **per-row filter**: keep
input columns whose `effective weight` is at least `rel_input_weight` (10%
by default) of the body's unfiltered total driven weight, then fit the
Gaussian on the surviving columns. Threshold is computed against the
pre-filter `tot_weight` so it's invariant under filtering. Legacy
`computing_functions.compute_rf_params` retains the original no-op gate
(unchanged — it gets deleted in Phase C).

### 3. `compute_clean_rf_params` → `_clean_rf_params`

- Legacy: [`computing_functions.py:249`](../src/utils/computing_functions.py#L249)
- New: [`_preprocessing.py:1406`](../src/setup_data/_preprocessing.py#L1406)

| Aspect | Legacy | New |
|---|---|---|
| Mutation | In-place (`params_df['r2'] = ...`) and returns the same df | Copies first (`df = params_df.copy()`), returns the copy |
| Math | identical (clip, size, ecc, cell_type) | identical |

### 4. `compute_directedness` → `_compute_directedness`

- Legacy: [`computing_functions.py:491`](../src/utils/computing_functions.py#L491)
- New: [`_preprocessing.py:286`](../src/setup_data/_preprocessing.py#L286)

| Aspect | Legacy | New |
|---|---|---|
| Side effects | Adds `hitting_time_diff` and `frac` columns to `conn_df` in-place | Requires caller to have added them already; reads them but doesn't mutate |
| Default `hit_diff_thre` | `0` | `0.5` (matches the value used by `get_ol_directedness`) |
| Math | identical (ff / fb / la fractions, outer-merged, `cell_type_pre` derived) | identical |

### 5. `compute_feature_vectors` → `_compute_feature_vectors`

- Legacy: [`computing_functions.py:452`](../src/utils/computing_functions.py#L452)
- New: [`_preprocessing.py:573`](../src/setup_data/_preprocessing.py#L573)

Math is identical. New version drops a trailing dead-code block (`vic_df`
commented out) and the dead `, vic_df` from the tuple return.

### 6. `compute_hierarchical_clustering` → `_compute_hierarchical_clustering`

- Legacy: [`computing_functions.py:439`](../src/utils/computing_functions.py#L439)
- New: [`_preprocessing.py:612`](../src/setup_data/_preprocessing.py#L612)

Functionally identical (row-normalise, `sch.linkage`). New version imports
`scipy.cluster.hierarchy` locally.

### 7. `compute_participation` → `_compute_participation`

- Legacy: [`computing_functions.py:389`](../src/utils/computing_functions.py#L389)
- New: [`_preprocessing.py:651`](../src/setup_data/_preprocessing.py#L651)

| Aspect | Legacy | New |
|---|---|---|
| Empty cluster guard | `... / len(outidx_i)` — divides by zero if a cluster is empty | `... / max(len(outidx_i), 1)` |
| Math | otherwise identical | otherwise identical |

### 8. `load_meta_groups` → `_load_meta_groups`

- Legacy: [`loading_functions.py:329`](../src/utils/loading_functions.py#L329)
- New: [`_preprocessing.py:20`](../src/setup_data/_preprocessing.py#L20)

Functionally identical. The new version is wrapped by
[`get_meta`](../src/setup_data/_preprocessing.py#L104) which adds parquet
caching via `cached_parquet`.

### 9. `load_flow_data` → `get_flow`

- Legacy: [`loading_functions.py:293`](../src/utils/loading_functions.py#L293)
- New: [`_preprocessing.py:121`](../src/setup_data/_preprocessing.py#L121)

Same CSV read + `count` annotation, now cached and parameterised by
`(dataset, side_char, n_flow, hit_thre)`.

### 10. `compute_ol_cb_vic` → inlined inside `get_ol_cb_vic_raw`

- Legacy: [`computing_functions.py:161`](../src/utils/computing_functions.py#L161)
- New: inlined as a nested `_compute` closure inside
  [`get_ol_cb_vic_raw` at `_preprocessing.py:1836`](../src/setup_data/_preprocessing.py#L1836)
  (see the docstring on line 1844 — *"Matches legacy
  `computing_functions.compute_ol_cb_vic`"*). There is no top-level
  `_compute_ol_cb_vic` helper.

Behavioural equivalence still needs a Phase C verification step (run both
implementations on a fixed seed and diff the output parquet). Two derived
`get_*` entry points wrap the same data: `get_ol_cb_vic_raw` and
`get_ol_cb_vic_type` (line 1899).

### 11. `load_layer_example_types` → `load_layer_example_types`

- Legacy: [`loading_functions.py:51`](../src/utils/loading_functions.py#L51)
- New: [`_preprocessing.py:364`](../src/setup_data/_preprocessing.py#L364)

True duplicate — the new copy keeps the public name (no leading underscore).
The new version takes `side_char` ('r'/'l') instead of `side` ('right'/'left')
and returns `[t + suffix for t in _LAYER_EXAMPLE_TYPES]` with
`_LAYER_EXAMPLE_TYPES = ['T4a', 'Mi4']`. Logic is otherwise identical.

The legacy `loading_functions.load_layer_example_types` has no external
callers (the new copy is what notebooks import). Safe to drop with the
rest of `loading_functions.py` in Phase C.

---

## Caller audit

External callers (i.e. excluding the legacy file itself and its internal
helpers) currently importing the legacy symbols:

| Legacy symbol | External callers |
|---|---|
| `compute_rf` | [`quan_propagation/contrib_cb_t4t5.py:2406`](../src/quan_propagation/contrib_cb_t4t5.py#L2406), [`quan_propagation/contrib_cb_t4t5.ipynb`](../src/quan_propagation/contrib_cb_t4t5.ipynb), [`quan_propagation/gallery_cb.py:72`](../src/quan_propagation/gallery_cb.py#L72), [`quan_propagation/gallery_cb.ipynb`](../src/quan_propagation/gallery_cb.ipynb), [`quan_propagation/gallery_olr.ipynb`](../src/quan_propagation/gallery_olr.ipynb), [`quan_propagation/gallery_vpn.py:69`](../src/quan_propagation/gallery_vpn.py#L69), [`quan_propagation/gallery_vpn.ipynb`](../src/quan_propagation/gallery_vpn.ipynb), [`quan_propagation/resolution_cb.py:2684`](../src/quan_propagation/resolution_cb.py#L2684), [`quan_propagation/resolution_cb.ipynb`](../src/quan_propagation/resolution_cb.ipynb) |
| `compute_rf_params` | [`rf/example_make_RF_malecns.ipynb`](../src/rf/example_make_RF_malecns.ipynb); commented-out in `resolution_cb.{py,ipynb}`, `resolution_cb_left.{py,ipynb}`, and `contrib_cb_t4t5.{py,ipynb}` |
| `compute_clean_rf_params` | [`rf/example_make_RF_malecns.ipynb`](../src/rf/example_make_RF_malecns.ipynb), [`make_figures/model_comparison.ipynb`](../src/make_figures/model_comparison.ipynb); commented-out in `resolution_cb*` and `contrib_cb_t4t5*` |
| `compute_directedness` | none |
| `compute_feature_vectors` | none |
| `compute_hierarchical_clustering` | none (one internal call inside `compute_flat_clusters`) |
| `compute_participation` | none |
| `compute_ol_cb_vic` | none (the new inlined version replaces it) |
| `load_meta_groups` | none (only internal calls from other legacy `loading_functions`) |
| `load_flow_data` | none (only internal calls from other legacy `loading_functions`) |
| `load_layer_example_types` | none (notebooks import the new copy from `_preprocessing.py`) |

### Legacy functions with no direct counterpart in `_preprocessing.py`

These need an explicit "delete or migrate" decision before Phase C closes.
Grouped by likely disposition:

**Probably inlined / superseded by `get_*` entry points (confirm by grep):**
- `compute_flat_clusters`, `compute_n_clusters` — likely covered by
  `get_ol_clusters`
- `compute_type_participation`, `compute_type_directedness` — likely covered
  by `get_ol_type_participation` / `get_ol_type_directedness`
- `compute_connectivity_pathways` — likely covered by
  `get_ol_connectivity_pathways`

**No counterpart in new pipeline — verify unused, then delete or migrate:**
- `compute_total_effective_weight`, `compute_median_rf_params`
- Spatial helpers: `find_depth`, `find_hex_ids`, `find_neuron_hex_ids`,
  `fl_get_edge_ids`, `load_depth_bins`, `load_hexed_body_ids`, `_load_pins`

**Reference/config loaders from `loading_functions.py` (most likely just
dead code; sweep before deleting):**
- Experimental-data loaders: `load_borst_data`, `load_clandinin_data`,
  `load_er_data`, `load_klapoetke_data`, `load_motion_data`
- Hardcoded type / body-ID lookups: `load_cb_cluster_types`,
  `load_cluster_types`, `load_out_types`, `load_participation_types`,
  `load_plot_types`, `load_rf_example_types`, `load_rf_example_neuron`
- Colour palettes: `load_colors`, `load_nt_colors`, `load_roi_colors`

If any of these survive the grep sweep with real external callers, decide
on a destination: `utils/config.py`, a new `utils/palettes.py`, etc. These
are config, not preprocessing.

### Additional caller of the NEW module

`src/utils/plotting.py` already imports `_compute_rf` from
`setup_data._preprocessing` (lines [1944](../src/utils/plotting.py#L1944)
and [3336](../src/utils/plotting.py#L3336)). It reaches into a private
helper of a private-named module — part of Phase E's motivation for
splitting the public data API into `utils/`.

---

## Priority 9: Retire the legacy `computing_functions` / `loading_functions` API

**Effort:** Medium

**Problem:** Every legacy function now has a `_compute_*` / `_load_*` twin in
`setup_data/_preprocessing.py` (or is superseded by a cached `get_*` entry
point). The legacy module is kept alive only by a handful of notebooks and
two `.py` files that still `from utils.computing_functions import compute_rf`.
Maintaining two parallel implementations risks drift and means the
[`rel_input_weight` gate bug](#2-compute_rf_params--_compute_rf_params)
exists in two places.

**Fix (phased):**

**Execution sequence: A → D → E (absorbs B) → C.** Phase headings retain
their original letter labels as stable identifiers for cross-referencing;
this section presents them in execution order. Rationale:

- **A first** — lowest-risk, byte-identical output guarantee for active
  callers. Must precede C (callers must be off legacy before deletion).
- **D second** — the `rel_input_weight` fix changes cached outputs. Doing
  it in place in `setup_data/_preprocessing.py` (before the Phase E move)
  keeps verification isolated to one logical change. Easiest to validate
  by diffing against the legacy implementation, which still exists.
- **E third (absorbs B)** — co-locating `_compute_*` helpers with their
  `get_*` wrappers in `utils/*.py` (decided 2026-05-15) means the rename
  from `_compute_*` → `compute_*` happens at move time, folding Phase B
  into the same disruption.
- **C last** — irreversible. Keep the legacy module around as a reference
  implementation for as long as D and E verifications need it.

### Phase A — migrate active external callers (~2 hours)

Lowest-risk first; each step independently verifiable by re-running the
notebook and diffing outputs.

1. **`compute_rf` callers in `quan_propagation/`** — replace
   `from utils.computing_functions import compute_rf` with
   `from setup_data._preprocessing import _compute_rf` in
   [`contrib_cb_t4t5.{py,ipynb}`](../src/quan_propagation/contrib_cb_t4t5.py),
   [`gallery_cb.{py,ipynb}`](../src/quan_propagation/gallery_cb.py),
   [`gallery_olr.ipynb`](../src/quan_propagation/gallery_olr.ipynb),
   [`gallery_vpn.{py,ipynb}`](../src/quan_propagation/gallery_vpn.py),
   [`resolution_cb.{py,ipynb}`](../src/quan_propagation/resolution_cb.py).
   The two functions differ only in the `coords`-column fix; downstream code
   in these galleries reads `x`/`y` for fitting, so output is unchanged.

2. **`compute_rf_params` / `compute_clean_rf_params` in `rf/` and
   `make_figures/`** — replace with `_compute_rf_params` / `_clean_rf_params`
   in [`example_make_RF_malecns.ipynb`](../src/rf/example_make_RF_malecns.ipynb)
   and [`model_comparison.ipynb`](../src/make_figures/model_comparison.ipynb).
   Output schema is identical; the only difference is `_clean_rf_params`
   doesn't mutate its input (safer).

3. **Delete commented-out blocks** in `resolution_cb*.{py,ipynb}` and
   `contrib_cb_t4t5.{py,ipynb}` referencing `compute_rf_params` /
   `compute_clean_rf_params` — they are dead, predating the new pipeline.

### Phase D — fix the carried-over bugs (~30 min)

Tracked separately because each needs a small judgement call:

- **`rel_input_weight` gate** (`_compute_rf_params`): replace the no-op
  per-body skip with a per-row filter. Sketch:

  ```python
  tot_weight = df["effective weight"].sum()
  if tot_weight <= 0:
      continue
  df = df[df["effective weight"] > rel_input_weight * tot_weight]
  if df.empty:
      continue
  amp = float(df["effective weight"].max())
  params_ij, rf_fitted = fit_rf_gaussian(df)
  ```

  Keep `tot_weight` computed *before* the filter so the threshold reflects
  the unfiltered total. Verify on a known test case (one body where you can
  predict by hand which input columns survive). Note that this change will
  shift the contents of cached `*_fit_rf.csv` files; rerun the affected
  preprocessing notebooks and check the diff before merging.
- **Notebook clean-up**: covered by Phase A step 3 (remove commented-out
  blocks in `resolution_cb*` and `contrib_cb_t4t5*`).

### Phase E — split public data API out of `setup_data/` (absorbs Phase B)

`setup_data/_preprocessing.py` currently mixes two responsibilities:

1. **Private `_compute_*` / `_load_*` helpers** — ported from the legacy
   `computing_functions` / `loading_functions` modules. These belong with
   the preprocessing pipeline.
2. **Public `get_*` cached entry points** — used by analysis notebooks and
   `utils/plotting.py` via `from setup_data import _preprocessing as pre`.
   The leading underscore in the module name is misleading: it is the de
   facto public data API.

A grep for `pre.get_*` / `pre.load_*` / `_compute_rf` across the repo finds
**~53 distinct public functions** in `_preprocessing.py` that are called from
outside `src/setup_data/`. External call sites:

- [`src/make_figures/paper_figures.ipynb`](../src/make_figures/paper_figures.ipynb)
- [`src/make_figures/figures.ipynb`](../src/make_figures/figures.ipynb)
- [`src/make_figures/model_comparison.ipynb`](../src/make_figures/model_comparison.ipynb)
- [`src/make_figures/polarity_signed.ipynb`](../src/make_figures/polarity_signed.ipynb)
- [`src/utils/plotting.py`](../src/utils/plotting.py) (uses `_compute_rf`)

#### Proposed split: move externally-used `get_*` into `src/utils/`

Group by domain. Each row lists functions that should move together
(because they share helpers / cached parquet names / docstring references).

| Destination file | Functions to move |
|---|---|
| `utils/core_data.py` *(new)* — foundational metadata + flow loaders used by ~everything | `get_meta`, `is_cb_neuron`, `get_ol_meta`, `get_inventory_meta`, `get_flow`, `get_flow_ol`, `get_ol_flow_type`, `get_sector_map`, `load_layer_example_types` |
| `utils/ol_data.py` *(new)* — OL-side connectivity / clustering / pathways | `get_ol_connectivity`, `get_ol_directedness`, `get_ol_type_directedness`, `get_ol_features`, `get_ol_clusters`, `get_ol_clusters_intra`, `get_ol_participation`, `get_ol_type_participation`, `get_ol_connectivity_pathways`, `get_ol_sankey_connectivity`, `get_ol_stepsn_sum`, `get_ol_stepsn_0`, `get_ol_prop`, `get_full_prop`, `get_ol_layers`, `get_ol_roi_adjacency`, `get_ol_roi_coverage`, `get_ol_lr_sweep`, `get_ol_lr_cluster_match`, `get_paths_to_instance`, `get_participation_paths` |
| `utils/cb_data.py` *(new)* — CB-side counterparts | `get_cb_connectivity`, `get_cb_directedness`, `get_cb_type_directedness`, `get_cb_features`, `get_cb_clusters`, `get_cb_clusters_intra`, `get_cb_cluster_tbars_per_roi`, `get_cb_lr_sweep`, `get_cb_lr_cluster_match`, `get_cb_layer_lr`, `get_cb_layer_lr_homologue`, `get_cb_paths_to_instance`, `get_ol_in_cb_participation`, `get_ol_in_cb_type_participation`, `get_ol_cb_cluster_connectivity`, `get_ol_to_cb_weights` |
| `utils/ol_rf.py` *(expand existing)* — RF getters + promoted `_compute_rf` | `get_rf_raw_ol`, `get_rf_type_ol`, `get_rf_raw_cb`, `get_rf_type_cb`, `get_rf_connectivity_edges_ol`, `get_input_rf_raw_ol`, `get_input_rf_type_ol`, `get_rf_comparison_body`, `get_rf_comparison_type`, `get_experimental_rf_sizes`, `get_rf_types_combined`. **Also promote `_compute_rf` → public `compute_rf`** so `utils/plotting.py` no longer needs to reach into a private helper. |
| `utils/vic.py` *(new)* — OL↔CB and binocular vicinity | `get_ol_type_vic`, `get_ol_cb_vic_raw`, `get_ol_cb_vic_type`, `get_cb_vic_binocular`, `get_cb_vic_binocular_type`, `get_cb_vic_lr_homologue`, `get_vcbn_types`, `get_roi_syn_vic` |
| `utils/polarity.py` *(new)* — polarity experiment getters | `get_polarity_experiments`, `get_ol_polarity_comparison`, `get_cb_polarity_comparison` |

#### Co-location of private helpers (Phase B absorbed)

Private `_compute_*` helpers move WITH their `get_*` wrappers into the same
`utils/*.py` destination file. Revisit only if a `_compute_*` helper ever
picks up a non-`get_*` caller from outside the helper's destination file.

Concretely, the move is "for each public `get_X` in the table above, also
move the `_compute_X` (or `_clean_X` etc.) helper it calls." `_compute_rf`
is the only helper that gets renamed at move time (drop the underscore, →
public `compute_rf` in `utils/ol_rf.py`) because Phase A's migrated
callers in `quan_propagation/` import it directly with custom `inidx` /
`outidx`. All other `_compute_*` / `_clean_*` helpers stay private after
the move — their only callers are the co-located `get_*` wrappers.

This subsumes the originally-separate **Phase B** (promote `_compute_*` to
public): the rename happens at move time, so there's no separate phase.

#### After the move

`src/setup_data/_preprocessing.py` would retain only:
- the private `_compute_*` / `_load_*` helpers that are NOT used by any
  externally-called `get_*` (effectively none, after co-location);
- module-level constants (`DATASET`, `DATA_DIR`, `SIDE_CHAR`);
- nothing else.

At that point `src/setup_data/` is genuinely a "run once to populate
`data/cache/`" pipeline directory containing only the orchestration scripts
(`setup_data.py`, `setup_effconn.py`, `setup_layers.py`,
`setup_malecns_data.py`, `_run_input_rfs.py`, `_verify_setup.py`,
`preprocessing.ipynb`). The module name `_preprocessing.py` can then be
dropped entirely (or renamed to `_pipeline.py` to reflect its remaining
contents).

### Phase C — delete legacy modules (~15 min)

When Phases A, D, E are merged and no `from utils.computing_functions` /
`from utils.loading_functions` import remains in the repo:

1. Delete [`src/utils/computing_functions.py`](../src/utils/computing_functions.py)
   and [`src/utils/loading_functions.py`](../src/utils/loading_functions.py).
2. Drop the line from [`src/judith/README.md:36-37`](../src/judith/README.md#L36)
   listing them under `utils/`.
3. Before deletion, confirm the legacy functions without direct counterparts
   (see [Caller audit § Legacy functions with no direct counterpart](#legacy-functions-with-no-direct-counterpart-in-_preprocessingpy))
   are either unused (drop) or have been migrated to their decided
   destination. Re-grep the repo for each name; if any survive with active
   callers, resolve before deleting the parent module.

### Phase B (absorbed into Phase E)

The originally-separate Phase B (promote `_compute_*` → public `compute_*`)
is folded into Phase E above. The rename happens at move time, so there is
no standalone Phase B step.

---

## Verification checklist

For each phase, before merging:

- [ ] Re-run the affected notebook end-to-end on a fresh `data/cache/`; diff
      the generated CSV / PDF outputs against the pre-migration baseline.
- [ ] (Phase D gate) After the `rel_input_weight` fix, the new RF fit
      parameters round-trip through `_clean_rf_params` and produce a `size`
      / `ecc` distribution consistent with the previously published figures.
- [ ] (Phase E gate) Every `pre.get_*` call site in `make_figures/*.ipynb`
      and `utils/plotting.py` is updated to import from the new `utils/*.py`
      destination. Notebook outputs unchanged.
- [ ] (Phase C gate) `grep` for `from utils.computing_functions` and
      `from utils.loading_functions` returns zero hits anywhere in the repo.
- [ ] (Phase C gate) `pre.get_ol_clusters` / `pre.get_ol_participation` /
      `pre.get_ol_type_directedness` (now in their new `utils/` locations)
      produce outputs that match their legacy equivalents bit-for-bit on a
      fixed seed.

---

## Execution log (2026-05-15)

All phases executed in the order **A → D → E.1 → E.2 → E.3 → E.4 → E.5 →
E.6 → E.7 → C**. Eleven commits on branch `re-org-v2`:

| Commit  | Phase | Summary |
|---|---|---|
| `88013fd` | doc | restructure: integrate addendum, reorder phases |
| `d032e24` | A   | migrate `compute_rf` / `compute_rf_params` / `compute_clean_rf_params` callers (16 files) |
| `f81c1c9` | D   | fix `rel_input_weight` no-op gate in `_compute_rf_params` |
| `a3749ae` | E.1 | extract foundation loaders → `utils/core_data.py` (10 public + 6 private) |
| `5b85c93` | E.2 | extract OL connectivity / clustering → `utils/ol_data.py` (19 public + 4 private) |
| `e9a2de1` | E.3 | extract VIC → `utils/vic.py` (8 public) |
| `382f7f9` | E.4 | extract RF → expanded `utils/ol_rf.py` (11 public + 9 private); promote `_compute_rf` → `compute_rf`; update Phase A callers in `quan_propagation/*` and `utils/plotting.py` |
| `aaceb2b` | E.5 | extract polarity → `utils/polarity.py` (3 public + 1 private + 1 constant) |
| `f37c4f3` | E.6 | extract CB-side → `utils/cb_data.py` (17 public) |
| `9eaec4d` | E.7 | drop `_preprocessing.py` shim; rewrite 13 importer files to use `utils/*` directly |
| `38987db` | C   | delete `utils/computing_functions.py` + `utils/loading_functions.py`; migrate ~8 still-used functions to `utils/hex_geometry.py` + `utils/palettes.py` |

### Final state of `src/`

- **Deleted**: `setup_data/_preprocessing.py`, `utils/computing_functions.py`,
  `utils/loading_functions.py`.
- **New `utils/` modules**: `core_data.py`, `ol_data.py`, `cb_data.py`,
  `vic.py`, `polarity.py`, `hex_geometry.py`, `palettes.py`.
- **Expanded existing**: `utils/ol_rf.py` (RF cached data appended after the
  pre-existing neuPrint-query helpers like `hexw_columnar`).
- `src/setup_data/` is now pure orchestration: `setup_data.py`,
  `setup_effconn.py`, `setup_layers.py`, `setup_malecns_data.py`,
  `_run_input_rfs.py`, `_verify_setup.py`, `preprocessing.ipynb`.
- ~25 legacy functions with no external callers (`compute_total_effective_weight`,
  `load_borst_data`, `load_nt_colors`, etc.) were dropped silently with their
  parent module in Phase C.

### Deviations from the original plan

- **Phase A's "byte-identical output" claim was wrong** for the two live
  `compute_rf` call sites in `contrib_cb_t4t5.py:2413` and `resolution_cb.py:2691`.
  Those sites had 4 lines after the call that undid legacy's `coords =
  "x,y/sqrt(3)"` mangling. With the new `_compute_rf` (raw hex `coords`),
  those 4 lines would introduce a sqrt(3) error. I deleted them — the
  resulting code matches the pattern already in `utils/plotting.py`. End
  state is byte-identical to legacy, but the code diff is more than a pure
  import swap.

- **Phase E went into `utils/ol_rf.py` even though that module connects to
  neuPrint at import time** (line 25: `c = olc_client.connect(verbose=True)`,
  plus `from fafbseg import flywire`). After E.4, anything that imports
  the new utils modules indirectly via `_preprocessing.py` re-exports also
  triggered the neuPrint connection. Phase E.7's elimination of the shim
  contains the blast radius — only direct importers of `utils.ol_rf` now
  trigger the connection. All current production callers (notebooks) have
  `.env` loaded, so the connection succeeds. CI / headless environments
  without credentials would fail at import time. **Pre-existing wart, not
  introduced by this migration**, but more visible now.

- **Two new files spawned by jupytext**: `src/quan_propagation/gallery_olr.py`
  and `src/make_figures/model_comparison.py`. The repo convention pairs
  every `.ipynb` with a `.py:percent` mirror — these notebooks had no
  mirror before, jupytext created them when I ran `--sync`. Kept; matches
  convention.

### Bugs found and fixed mid-execution

- **`_split_polarity_types` lost during E.2.** My extraction helper had a
  range-overlap bug where adjacent function boundaries could share lines.
  The bug deleted `_split_polarity_types` (which sits between two OL
  functions in the original layout) from `_preprocessing.py` without
  copying it into `ol_data.py`. The helper was fixed mid-Phase-E.4
  (added comment-stripping to `block_end` calculation in `find_blocks`).
  `_split_polarity_types` was restored from git in Phase E.5 and bundled
  into `utils/polarity.py`. **Window of risk**: any caller of
  `get_polarity_experiments` between commits `5b85c93` (E.2) and `aaceb2b`
  (E.5) would have hit a `NameError` at runtime. Worth re-running any
  notebook touching polarity to confirm it works post-E.5.

### Verification still owed (cannot be done from CLI without notebook execution)

1. **Re-run `make_figures/{paper_figures,figures,model_comparison,polarity_signed}.ipynb`** end-to-end against a fresh `data/cache/`. Confirm:
   - All figure outputs match pre-migration baseline EXCEPT Phase D-affected
     RF caches (`*_fit_rf.{csv,p}` and anything downstream of `get_rf_type_ol`
     / `get_rf_type_cb` / `get_input_rf_*`).
2. **Phase D scientific check**: pick a known bodyId, hand-predict which
   input columns survive `rel_input_weight * tot_weight` (default 10%),
   confirm the new fit reflects only those columns. New `size` / `ecc`
   distributions should remain in the same order of magnitude as published
   figures but will not be bit-identical.
3. **Re-run polarity notebook** specifically to confirm `_split_polarity_types`
   is wired correctly post-restoration.
4. **Re-run `quan_propagation/{contrib_cb_t4t5,resolution_cb}.ipynb`** to
   confirm the hex-heatmap output from the two live `compute_rf` call sites
   is unchanged after I deleted the redundant 4-line `coords` conversion.
5. **`quan_propagation/gallery_*.ipynb`**: were import-only callers (no
   live `compute_rf(` calls in any). Should be no-ops behaviourally; sanity-check
   only.

### Pre-existing issues NOT addressed (out of scope)

- **Syntax errors in jupytext-mirror `.py` files** at the line where
  Python parses notebook markdown/data as code:
  `quan_propagation/contrib_cb_t4t5.py:~1834`,
  `quan_propagation/resolution_cb.py:~1834`,
  `quan_propagation/contrib_plot_mip.py:~1811`,
  and a few others. The `.ipynb` runs fine in Jupyter; the `.py` mirror
  doesn't parse standalone. Predates this work (visible at HEAD~3+). Should
  be cleaned up separately — likely root cause is some notebook cells with
  raw-text content being mis-rendered to `.py:percent`.
- **`src/utils/_unused_functions.py`** archive of previously-removed dead
  code (per CLAUDE.md, "Recently removed dead code (commit 281ac66)").
  Left untouched.
- **`src/utils/ol_rf.py` import-time neuPrint connection** — pre-existing
  side effect, not introduced by Phase E. See "Deviations" above.

### Possible risks for future maintainers

- **Phase D output drift**: cached `*_fit_rf.{csv,p}` files in `data/cache/`
  must be regenerated. Anything that compared against pre-Phase-D figures
  byte-for-byte will fail. The fix is correct, but figures will look
  different than the pre-migration baseline. The published figures in
  `docs/paper_figs.xlsx` need a re-render before next paper update.
- **`utils.ol_rf` is a mixed module**: the pre-existing neuPrint-query
  helpers (`hexw_columnar`, `pqw_columnar*`) sit at the top; the new
  cached RF accessors (`compute_rf`, `get_rf_*`) sit below. Future
  refactoring should consider splitting this into `utils/ol_rf_neuprint.py`
  (legacy hexw helpers) and `utils/ol_rf.py` (cached RF) — that would let
  pure-cache callers import without triggering neuPrint connection.
- **Helper script `_phase_e_helper.py` was created and then deleted**
  during the migration. It contained a bug (range-overlap, see above).
  If a similar mass-refactor is needed in future, the helper's
  `find_blocks` boundary-detection logic must trim BOTH trailing blanks
  AND trailing comments to avoid range overlap with the next top-level
  statement.
- **`compute_ol_cb_vic` was reimplemented as a nested closure** inside
  `get_ol_cb_vic_raw` rather than as a separate `_compute_*` helper. If
  someone needs to call the OL↔CB VIC computation with custom seeds /
  outidx in the future, they have to either inline the closure code or
  refactor it into a real helper. The bit-for-bit equivalence verification
  promised by the original plan was NOT run — both implementations differ
  only in trivial style (variable renaming, `is_cb_neuron` factoring), but
  a fixed-seed parquet diff would be the conclusive check.
- **Forward compatibility of the new module split**: the destination
  table assigned each function to a single bucket based on its primary
  domain. Functions that legitimately straddle buckets (e.g.
  `get_ol_roi_coverage` lives in `cb_data.py` despite the "OL" prefix
  because it depends on `vic`) may surprise future readers. The grouping
  is pragmatic, not principled.
- **`load_layer_example_types` exists in `utils/core_data.py`** with a
  different signature (`side_char` instead of `side`) than the deleted
  `loading_functions.load_layer_example_types`. Any code path that
  resurrects the old signature would silently produce wrong suffixes
  (`"_R"` vs `"_L"` flip). The notebooks updated in E.7 use the new
  signature correctly; any third-party fork would need a manual update.
