
"""
This cell does the initial project setup.
If you start a new script or notebook, make sure to copy & paste this part.

A script with this code uses the location of the `.env` file as the anchor for
the whole project (= PROJECT_ROOT). Afterwards, code inside the `src` directory
are available for import.
"""
from pathlib import Path
import sys
from dotenv import find_dotenv
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath('src')))

from utils.config import (
    DATA_DIR, DATASET, EYEMAP_DIR, PARAMS_DIR, SIDE, SIDE_CHAR,
)

import pandas as pd
import numpy as np
import re
import scipy.sparse as sp
from scipy.spatial import distance

from utils import olc_client
c = olc_client.connect(verbose=False)

from neuprint import fetch_neurons, fetch_synapses, fetch_adjacencies
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC
from neuprint.queries import fetch_all_rois, fetch_roi_hierarchy

import navis
import navis.interfaces.neuprint as neu
# from fafbseg import flywire

# Phase E.4: cached RF functions appended below. They need cached_parquet
# plus several `get_*` functions from sibling utils modules.
from utils.cache import cached_parquet
from utils.core_data import _SIDE_WORD, get_flow, get_flow_per_group, get_meta, get_ol_flow_type, get_ol_meta
from utils.ol_data import get_ol_connectivity, get_ol_stepsn_sum, get_ol_type_directedness
from utils.vic import get_ol_cb_vic_type, get_ol_type_vic, get_vcbn_types


def hexw_columnar(
    bodyId:int,
    thr_qt:float=0.0,
    syntype:str='post',
    roi_str:str='ME(R)',
) -> pd.DataFrame:
    """
    get RF for columnar neurons using upstream connections in column rois

    Args
    ------
    bodyId : int
        bodyId of the neuron
    thr_qt : float
        quantile threshold for synapse count
    syntype : str
        default as post synapse, where to receive input
    roi_str : str
        neuprint ROI, can only be ME(R), LO(R), LOP(R)

    Returns
    -------
    df : pd.DataFrame
        data frame with columns ['bodyId', 'type', 'instance']
    """

    assert roi_str in ['ME(R)', 'LO(R)', 'LOP(R)'],\
            f"ROI must be one of 'ME(R)', 'LO(R)', 'LOP(R)', but is actually '{roi_str}'"
    
    # remove (R) from roi_str
    roi = roi_str.replace('(R)', '')

    # query synapse data
    _, roi_counts_df = fetch_neurons(NC(bodyId=bodyId))
    df = roi_counts_df[roi_counts_df['roi'].str.contains(f'{roi}_R_col_')] # keep col rois 
    # thresholding synapse count
    thr_post = df[syntype].quantile(thr_qt)
    df = df[(df[syntype] >= thr_post)]

    # return the input col numbers
    hex = np.array([int(s) for t in df['roi'] for s in re.findall(r'\d+', t)]).reshape(-1,2)
    # add weights
    hexw = pd.DataFrame(
        np.concatenate((hex, df[syntype].to_numpy().reshape(-1,1)), axis=1), 
        columns=['hex1', 'hex2', 'wt']
        )
    hexw.sort_values(by='wt', ascending=False, inplace=True)
    hexw.reset_index(drop=True, inplace=True)

    return hexw


# def pqw_columnar_flywire(
#     bodyId:np.int64,
#     thr_qt:float=0.0,
#     thr_dist:float=10000,
#     syntype:str='post',
#     layer_str:str='M5', #only work with M1/5/10 for now
# ) -> pd.DataFrame:
#     """
#     get RF for columnar neuorns using column rois

#     Args
#     ------
#     bodyId : int
#         bodyId of the neuron
#     thr_qt : float
#         quantile threshold for synapse count
#     thr_dist : float
#         distance threshold between syn and col
#     syntype : str
#         default as post synapse, where to receive input
#     roi_str : str
#         neuprint ROI, can only be ME(R), LO(R), LOP(R)

#     Returns
#     -------
#     df : pd.DataFrame
#         data frame with columns ['bodyId', 'type', 'instance']
#     """

#     assert layer_str in ['M1', 'M5', 'M10'],\
#             f"ROI must be one of 'M1, M5, M10', but is actually '{layer_str}'"
#     assert syntype in ['pre', 'post'],\
#             f"syntype must be one of 'pre, post', but is actually '{syntype}'"
    
#     # load eyemap
#     pqxyztpind = pd.read_excel(EYEMAP_DIR / '783_20240513' / f'pqxyztpid_layercol_{SIDE}.xlsx')
#     pqxyztpind.reset_index(drop=True, inplace=True)
#     pqxyztpind.astype({'rootid': 'Int64'})

#     pre_bool = syntype == 'pre'
#     post_bool = syntype == 'post'
#     # query synapse data
#     syn = flywire.get_synapses(bodyId, pre=pre_bool, post=post_bool)
#     synxyz = syn[[f'{syntype}_x', f'{syntype}_y', f'{syntype}_z']].values
#     # col xyz
#     colxyz = pqxyztpind[[f'x_{layer_str}',f'y_{layer_str}',f'z_{layer_str}']].values
    
#     # compute the distance matrix
#     dist = distance.cdist(colxyz, synxyz)
#     # find the minimum distance for each row
#     min_dist = np.min(dist, axis=1)
#     # index of columns that are smaller than the threshold
#     col_idx = np.where(min_dist < thr_dist)[0]
    
#     # assign weights to each col
#     # find which col_xyz is cloest to each synxyz
#     dist = dist[col_idx,:]
#     if len(dist) == 0:
#         return pd.DataFrame(columns=['p', 'q', 'wt'])
#     # find the minimum distance for each colum
#     min_dist = np.min(dist, axis=0)
#     # remove columns with min_dist > thr
#     dist = dist[:, min_dist < thr_dist]
#     # find the index of the minimum distance for each column
#     min_idx = np.argmin(dist, axis=0)
#     # value count min_idx, sort by index
#     wts = pd.Series(min_idx).value_counts().sort_index()
#     # make an array with wts, wts.index as row numbers, fill missing rows with 0
#     wt_arr = np.zeros(len(col_idx))
#     wt_arr[wts.index] = wts.values

#     # combine pqxyztpind[['p','q']] and wt_arr
#     pqw = np.concatenate((pqxyztpind[['p','q']].values[col_idx], wt_arr[:,None]), axis=1)
#     # convert to df and change type to int, rename columns
#     pqw = pd.DataFrame(pqw, columns=['p','q','wt'])
#     pqw = pqw.astype({'p': int, 'q': int, 'wt': int})
    
#     # thresholding synapse count
#     thr_post = pqw['wt'].quantile(thr_qt)
#     pqw = pqw[(pqw['wt'] >= thr_post)]

#     pqw.sort_values(by='wt', ascending=False, inplace=True)
#     pqw.reset_index(drop=True, inplace=True)
        
#     return pqw
# _VISUAL_INPUT_STEMS = ["L1", "L2", "L3", "R7", "R8"]


def compute_rf(meta_ol, stepsn, inidx, outidx, *, idx_to_coords=None, idx_to_root=None):
    """Single-target RF from the OL-subset `lat_flow_sum` (stepsn).

    Returns a long df (coords, x, y, bodyId, effective weight). Matches legacy
    `computing_functions.compute_rf`. `idx_to_coords` / `idx_to_root` can be
    passed in to avoid rebuilding the OL-wide dicts on every call.
    """
    from connectome_interpreter.compress_paths import result_summary

    if idx_to_coords is None:
        idx_to_coords = dict(zip(meta_ol.idx, meta_ol.coords))
    if idx_to_root is None:
        idx_to_root = dict(zip(meta_ol.idx, meta_ol.bodyId))
    df = result_summary(
        stepsn, inidx, outidx,
        inidx_map=idx_to_coords, outidx_map=idx_to_root,
        display_threshold=0, display_output=False,
    )
    df = df[(df.index != "nan") & (df.index != "None") & (~df.index.isnull())]
    if df.empty:
        return df
    df = (
        df.iloc[:, 0].to_frame().stack().reset_index()
        .rename(columns={"level_0": "coords", "level_1": "bodyId", 0: "effective weight"})
    )
    xy = df["coords"].str.split(",", expand=True).astype(int).values
    df["x"] = xy[:, 0]
    # `y` is scaled by 1/sqrt(3) so the hex grid is isotropic for the 2D
    # Gaussian fit. The raw `coords` string is preserved for hex plotting.
    df["y"] = xy[:, 1] / np.sqrt(3)
    return df


def _compute_rf_params(meta_ol, stepsn, in_instances, out_instances, cumsum_thre=0.7):
    """Fit a 2D Gaussian per target bodyId (across every `out_instances`).

    Returns a long df with columns [n_col, r2, x0, y0, a, b, phi, amp, bodyId,
    instance, seed]. Skips bodies whose total effective input weight is
    non-positive or whose Gaussian fit yields NaN. Dicts are built once and
    reused for every body.
    """
    from utils.external_rf import fit_rf_gaussian

    inidx = meta_ol[meta_ol.instance.isin(in_instances)].idx.values
    in_string = "_".join([s[:-2] for s in in_instances])
    idx_to_coords = dict(zip(meta_ol.idx, meta_ol.coords))
    idx_to_root = dict(zip(meta_ol.idx, meta_ol.bodyId))

    rows = []
    for target in out_instances:
        sub = meta_ol[meta_ol.instance == target]
        outidx_arr = sub.idx.values
        bids = sub.bodyId.values
        for i, bid in enumerate(bids):
            df = compute_rf(
                meta_ol, stepsn, inidx, outidx_arr[i:i + 1],
                idx_to_coords=idx_to_coords, idx_to_root=idx_to_root,
            )
            if df.empty:
                continue
            tot_weight = df["effective weight"].sum()
            if tot_weight <= 0:
                continue
            amp = float(df["effective weight"].max())
            params_ij, rf_fitted = fit_rf_gaussian(df, cumsum_thre=cumsum_thre)
            rows.append({
                "n_col": int(rf_fitted.shape[0]), "r2": params_ij[0],
                "x0": params_ij[1], "y0": params_ij[2],
                "a": params_ij[3], "b": params_ij[4], "phi": params_ij[5],
                "amp": amp, "bodyId": int(bid),
                "instance": target, "seed": in_string,
            })

    params = pd.DataFrame(rows)
    if params.empty:
        return params
    params = params.dropna().reset_index(drop=True)
    return params


def _clean_rf_params(params_df):
    """Apply the cleaning step from legacy `compute_clean_rf_params`:
    clip r2 to [-1, 1], clip a/b to [0.5, inf), derive size / ecc / cell_type.
    """
    df = params_df.copy()
    df["r2"] = df["r2"].clip(lower=-1, upper=1)
    df["a"] = df["a"].clip(lower=0.5)
    df["b"] = df["b"].clip(lower=0.5)
    df["size"] = df["a"] * df["b"] * np.pi * np.sqrt(3) / 2
    df["ecc"] = np.sqrt(1 - (df["b"] / df["a"]) ** 2)
    df["cell_type"] = [s[:-2] for s in df["instance"]]
    return df


_VISUAL_INPUT_STEMS = ["L1", "L2", "L3", "R7", "R8", "R7d", "R8d", "HBeyelet"]
_PHOTORECEPTOR_TYPES = {"R7", "R8", "R7d", "R8d", "HBeyelet"}


def _default_visual_input_instances(side_char: str) -> list[str]:
    suffix = "_R" if side_char == "r" else "_L"
    return [s + suffix for s in _VISUAL_INPUT_STEMS]


def get_rf_raw_ol(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    in_instances: list[str] | None = None, force: bool = False,
) -> pd.DataFrame:
    """Per-body OL receptive-field fits.

    Fits a 2D Gaussian on the effective-weight RF from visual inputs
    (L1/L2/L3/R7/R8 by default) onto every non-visual-input OL instance on
    the selected side, via the OL-subset `lat_flow_sum`. Params are cleaned
    (`_clean_rf_params`) and merged with per-instance flow (hitting_time,
    main_groups, sign).
    """
    def _compute():
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_sum(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        flow_type = get_ol_flow_type(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )

        ins = in_instances if in_instances is not None else _default_visual_input_instances(side_char)
        out_inst = ol_meta[ol_meta.main_groups != "visual input"].instance.unique()

        params = _compute_rf_params(ol_meta, stepsn, ins, out_inst)
        if params.empty:
            return params
        params = _clean_rf_params(params)
        params = params.merge(
            flow_type[["cell_type", "main_groups", "sign", "hitting_time"]].drop_duplicates("cell_type"),
            on="cell_type", how="left",
        )
        return params

    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_raw", _compute, force=force,
    )


def get_rf_type_ol(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-OL-instance median RF params + merged directedness + VIC + flow.

    Phi is aggregated via circular mean (period π); other fit params via
    median. `n_neurons` is the count of fitted bodies per (instance, seed).
    """
    from utils.external_rf import circular_mean

    def _compute():
        raw = get_rf_raw_ol(dataset=dataset, data_dir=data_dir, side_char=side_char)
        dir_type = get_ol_type_directedness(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        vic = get_ol_type_vic(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        flow_type = get_ol_flow_type(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )

        agg = (
            raw.groupby(["instance", "seed"])[["n_col", "r2", "a", "b", "phi", "amp"]]
            .median().reset_index()
        )
        agg["phi"] = (
            raw.groupby(["instance", "seed"])["phi"]
            .apply(lambda s: circular_mean(s.values)).values
        )
        agg["n_neurons"] = (
            raw.groupby(["instance", "seed"])["bodyId"].count().values
        )
        agg = _clean_rf_params(agg)
        agg = agg.merge(
            flow_type[["cell_type", "main_groups", "sign", "hitting_time"]].drop_duplicates("cell_type"),
            on="cell_type", how="left",
        )
        agg = agg.merge(
            dir_type[["instance_pre", "frac_ff", "frac_la", "frac_fb"]],
            left_on="instance", right_on="instance_pre", how="left",
        ).drop(columns=["instance_pre"])
        agg = agg.merge(vic, on="instance", how="left")
        return agg

    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_type", _compute, force=force,
    )


def get_rf_connectivity_edges_ol(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, include_cb_post: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Long-form OL edge table with per-type `size_diff` and `layer_diff`.

    Edges come from `get_ol_connectivity`. Post-side sizes are pulled from
    `get_rf_types_combined` (OL + CB-VCBN), so CB-post edges are retained
    when `include_cb_post=True`. Set it to False to match the OL-only subset
    from Step 6a.
    """
    def _compute():
        conn = get_ol_connectivity(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        if include_cb_post:
            rf_type = get_rf_types_combined(
                dataset=dataset, data_dir=data_dir, side_char=side_char,
                vic_thre=vic_thre,
            )
        else:
            rf_type = get_rf_type_ol(
                dataset=dataset, data_dir=data_dir, side_char=side_char,
            )
        sizes = rf_type[["instance", "size"]].drop_duplicates("instance")
        conn = conn.merge(
            sizes.rename(columns={"instance": "instance_pre", "size": "size_pre"}),
            on="instance_pre", how="left",
        )
        conn = conn.merge(
            sizes.rename(columns={"instance": "instance_post", "size": "size_post"}),
            on="instance_post", how="left",
        )
        conn = conn.dropna(subset=["hitting_time_pre", "hitting_time_post", "size_pre", "size_post"])
        conn["layer_diff"] = (
            conn["hitting_time_post"].astype(float) - conn["hitting_time_pre"].astype(float)
        )
        conn["size_diff"] = (
            (conn["size_post"].astype(float) - conn["size_pre"].astype(float))
            / conn["size_pre"].astype(float)
        )
        return conn.reset_index(drop=True)

    tag = "cb" if include_cb_post else "ol"
    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_edges_{tag}", _compute, force=force,
    )


def _rf_type_sx_sy(df):
    """Rotate (a, b) back to horizontal / vertical semi-axes based on phi.
    Scale factor `sqrt(sqrt(3)/2)` converts hex-column units to column-radius
    units (legacy convention). Returns (sx, sy) as numpy arrays.
    """
    fac = np.sqrt(np.sqrt(3) / 2)
    horizontal = np.abs(df["phi"]) < np.pi / 4
    sx = np.where(horizontal, fac * df["a"], fac * df["b"])
    sy = np.where(horizontal, fac * df["b"], fac * df["a"])
    return sx, sy


def _synthetic_visual_input_prop(side_char: str) -> pd.DataFrame:
    """Synthetic propagated-RF rows for visual-input seeds (L1/L2/L3) using
    the legacy convention: `size=1` column, `sx=sy=1/sqrt(pi)` (unit-area
    circular RF), `phi=0`. Used by `get_rf_comparison_type` so RF-size
    comparisons can include visual inputs as a lower-bound reference."""
    suffix = "_R" if side_char == "r" else "_L"
    r = 1.0 / np.sqrt(np.pi)
    return pd.DataFrame({
        "instance": [f"L1{suffix}", f"L2{suffix}", f"L3{suffix}"],
        "cell_type": ["L1", "L2", "L3"],
        "main_groups": ["visual input"] * 3,
        "sign": [-1.0, 1.0, 1.0],
        "size": [1.0, 1.0, 1.0],
        "sx": [r, r, r],
        "sy": [r, r, r],
        "phi": [0.0, 0.0, 0.0],
    })


def _ol_instances(
    dataset: str, data_dir, side_char: str,
) -> set[str]:
    """OL-side instances, matching the filter in `_ensure_ol_sidecars`.

    Read directly from `get_meta` so callers (e.g. ROI-count / ROI-adj
    downloads) don't need the OL sidecar — which can't be built until the
    full-brain flow and lat_flow files also exist.
    """
    meta = get_meta(dataset=dataset, data_dir=data_dir)
    side_word = _SIDE_WORD[side_char]
    sub = meta[
        (meta.region == f"{side_word} OL")
        & (meta.main_groups != "other")
        & (meta.instance != "R1-R6_R")
    ]
    return set(sub.instance.unique())


def _ol_non_visual_input_instances(
    dataset: str, data_dir, side_char: str,
) -> set[str]:
    """`_ol_instances` minus photoreceptors (R7/R8/R7d/R8d/HBeyelet).

    Excludes only true photoreceptors, not lamina VINs (L1–L5), so direct
    RF fits are computed for L1–L5 alongside OL-internal/output neurons.
    """
    meta = get_meta(dataset=dataset, data_dir=data_dir)
    return _ol_instances(dataset, data_dir, side_char) - set(
        meta[meta.cell_type.isin(_PHOTORECEPTOR_TYPES)].instance.unique()
    )


def get_input_rf_raw_ol(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    source_stem: str | None = None, force: bool = False,
) -> pd.DataFrame:
    """Per-body direct input RF fits, computed from raw neuprint synapse data.

    Reads `data/{source_stem}_input_syn_per_col.parquet` (default
    `{dataset}_OL_{side_char}`), restricts to OL instances excluding
    photoreceptors (R7/R8/R7d/R8d/HBeyelet) and R1-R6 — L1–L5 are included.
    Fits a 2D
    Gaussian per (instance, roi, bodyId) on per-column `synapse_frac` as
    `effective weight`, with `amp = sum(effective weight)`. Cleaned via
    `_clean_rf_params`, cached as parquet.

    The raw parquet is produced by `setup_data.setup_data.fetch_input_syn_per_col`
    (called from `setup_malecns_data.ipynb` section 6). If you rerun the
    download, pass `force=True` to invalidate this cache.
    """
    from utils.external_rf import fit_rf_gaussian
    stem = source_stem or f"{dataset}_OL_{side_char}"

    def _compute():
        raw = pd.read_parquet(data_dir / f"{stem}_input_syn_per_col.parquet")
        instances = _ol_non_visual_input_instances(dataset, data_dir, side_char)
        raw = raw[raw["instance"].isin(instances)].copy()
        raw["effective weight"] = raw["synapse_frac"]

        rows = []
        for (inst, roi, bid), sub in raw.groupby(["instance", "roi", "bodyId"]):
            sub_xy = sub[["x", "y", "effective weight"]]
            amp = float(sub_xy["effective weight"].sum())
            params_ij, rf_fitted = fit_rf_gaussian(sub_xy)
            rows.append({
                "n_col": int(rf_fitted.shape[0]),
                "r2": params_ij[0], "x0": params_ij[1], "y0": params_ij[2],
                "a": params_ij[3], "b": params_ij[4], "phi": params_ij[5],
                "amp": amp, "bodyId": int(bid),
                "instance": inst, "roi": roi,
            })
        params = pd.DataFrame(rows)
        if params.empty:
            return params
        params = params.dropna().reset_index(drop=True)
        return _clean_rf_params(params)

    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_input_raw", _compute, force=force,
    )


def get_input_rf_type_ol(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    source_stem: str | None = None, force: bool = False,
) -> pd.DataFrame:
    """Per-(instance, roi) median direct input RF fits.

    Aggregates `get_input_rf_raw_ol` by `(instance, roi)`: median on
    `[n_col, r2, a, b, amp]`, `circular_mean` on `phi`, count on `bodyId` ->
    `n_neurons`. Cleaned via `_clean_rf_params`, cached as parquet.
    """
    from utils.external_rf import circular_mean

    def _compute():
        raw = get_input_rf_raw_ol(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            source_stem=source_stem,
        )
        if raw.empty:
            return raw
        numeric = (
            raw.groupby(["instance", "roi"])[["n_col", "r2", "a", "b", "amp"]]
            .median().reset_index()
        )
        phi = (
            raw.groupby(["instance", "roi"])["phi"]
            .apply(lambda x: circular_mean(x.values)).reset_index()
        )
        n_neurons = (
            raw.groupby(["instance", "roi"])["bodyId"].count().reset_index()
            .rename(columns={"bodyId": "n_neurons"})
        )
        out = numeric.merge(phi, on=["instance", "roi"])
        out = out.merge(n_neurons, on=["instance", "roi"])
        return _clean_rf_params(out)

    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_input_type", _compute, force=force,
    )


def get_rf_comparison_body(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Merge propagated + direct per-body RFs on (bodyId, instance) with suffixes
    `_prop` / `_dir`."""
    def _compute():
        prop = get_rf_raw_ol(dataset=dataset, data_dir=data_dir, side_char=side_char)
        direct = get_input_rf_raw_ol(dataset=dataset, data_dir=data_dir, side_char=side_char)
        common = set(prop.columns) & set(direct.columns) - {"bodyId", "instance"}
        prop = prop.rename(columns={c: f"{c}_prop" for c in common})
        direct = direct.rename(columns={c: f"{c}_dir" for c in common})
        return prop.merge(direct, on=["bodyId", "instance"], how="inner")

    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_comparison_body", _compute, force=force,
    )


def get_rf_comparison_type(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Merge propagated + direct per-instance RFs with sx/sy added on each side.

    Visual-input seeds (L1/L2/L3) are appended to the propagated side via
    `_synthetic_visual_input_prop` so they appear in downstream RF-size
    comparisons even though they aren't fitted as outputs in `get_rf_raw_ol`.
    An inner-join is used so only instances present in both the propagated and
    direct-fit tables are kept; L1/L2/L3 appear after their direct-fit cache
    is rebuilt with `get_input_rf_raw_ol(force=True)`.
    """
    def _compute():
        prop = get_rf_type_ol(dataset=dataset, data_dir=data_dir, side_char=side_char).copy()
        direct = get_input_rf_type_ol(dataset=dataset, data_dir=data_dir, side_char=side_char).copy()
        prop["sx"], prop["sy"] = _rf_type_sx_sy(prop)
        direct["sx"], direct["sy"] = _rf_type_sx_sy(direct)
        prop = pd.concat(
            [prop, _synthetic_visual_input_prop(side_char)], ignore_index=True,
        )
        common = set(prop.columns) & set(direct.columns) - {"instance"}
        prop = prop.rename(columns={c: f"{c}_prop" for c in common})
        direct = direct.rename(columns={c: f"{c}_dir" for c in common})
        return prop.merge(direct, on="instance", how="inner")

    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_comparison_type", _compute, force=force,
    )


def get_experimental_rf_sizes(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    source_file: str = "fly_literature_experiments.xlsx",
    sheet: str = "RF size", force: bool = False,
) -> pd.DataFrame:
    """Experimental RF sizes from a fly-literature spreadsheet sheet.

    Adds `instance = "Neuron type" + '_R'`. Keeps the sheet's native columns
    (`size`, `sx`, `sy`, `FWHMx`, `FWHMy`, `Reference` for sheet 'RF size';
    `size_center`, `size_ellipse`, ... for sheet 'RF size CC').
    """
    def _compute():
        df = pd.read_excel(PARAMS_DIR / source_file, sheet_name=sheet)
        df = df[~df["Neuron type"].isna()].copy()
        df["instance"] = df["Neuron type"].astype(str) + "_R"
        df = df.rename(columns={"Neuron type": "cell_type"})
        return df.reset_index(drop=True)

    tag = sheet.replace(" ", "_")
    return cached_parquet(f"{dataset}_exp_{tag}", _compute, force=force)


def get_rf_raw_cb(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, in_instances: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Per-body CB receptive-field fits for VCBN instances (VIC > vic_thre).

    Uses the full `lat_flow_sum.npz` (not OL-only). Heavy first-run compute.
    Output instances come from `get_vcbn_types`.
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        stepsn = sp.load_npz(
            data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz"
        )
        vcbn = get_vcbn_types(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            vic_thre=vic_thre,
        )
        ins = in_instances if in_instances is not None else _default_visual_input_instances(side_char)
        out_inst = vcbn["instance"].values
        params = _compute_rf_params(meta, stepsn, ins, out_inst)
        if params.empty:
            return params
        params = _clean_rf_params(params)
        # merge region and flow
        flow_type = meta.groupby("instance")["region"].first().reset_index()
        params = params.merge(flow_type, on="instance", how="left")
        vic_type = get_ol_cb_vic_type(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        params = params.merge(
            vic_type[["instance", "VIC"]], on="instance", how="left",
        )
        ht = get_flow_per_group(dataset=dataset, data_dir=data_dir, side_char=side_char)
        params = params.merge(ht, on="instance", how="left")
        sign = meta.groupby("instance")["sign"].first().reset_index()
        params = params.merge(sign, on="instance", how="left")
        mg = meta.groupby("instance")["main_groups"].first().reset_index()
        params = params.merge(mg, on="instance", how="left")
        return params.reset_index(drop=True)

    tag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_cb_raw_{tag}", _compute, force=force,
    )


def get_rf_type_cb(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Per-instance median CB RF fits with circular-mean phi and merged
    region/VIC/layer/sign/main_groups."""
    from utils.external_rf import circular_mean

    def _compute():
        raw = get_rf_raw_cb(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            vic_thre=vic_thre,
        )
        if raw.empty:
            return raw
        agg = (
            raw.groupby(["instance", "seed"])[["n_col", "r2", "a", "b", "phi", "amp"]]
               .median().reset_index()
        )
        agg["phi"] = (
            raw.groupby(["instance", "seed"])["phi"]
               .apply(lambda s: circular_mean(s.values)).values
        )
        agg["n_neurons"] = raw.groupby(["instance", "seed"])["bodyId"].count().values
        agg = _clean_rf_params(agg)
        side_cols = ["region", "VIC", "hitting_time", "sign", "main_groups"]
        meta_type = (
            raw.groupby("instance")[side_cols].first().reset_index()
        )
        return agg.merge(meta_type, on="instance", how="left")

    tag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_cb_type_{tag}", _compute, force=force,
    )


def get_rf_types_combined(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Concatenated OL + CB per-instance RF fits. Output has a `region`
    column distinguishing `right OL` vs `CB`. Used by CB-vs-OL plots."""
    def _compute():
        ol = get_rf_type_ol(dataset=dataset, data_dir=data_dir, side_char=side_char).copy()
        ol["region"] = f"{_SIDE_WORD[side_char]} OL"
        cb = get_rf_type_cb(
            dataset=dataset, data_dir=data_dir, side_char=side_char, vic_thre=vic_thre,
        ).copy()
        return pd.concat([ol, cb], axis=0, ignore_index=True)

    tag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_rf_types_combined_{tag}", _compute, force=force,
    )


