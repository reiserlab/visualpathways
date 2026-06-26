"""Preprocessing functions grouped by legacy notebook (1..11).

Each section hosts the compute-and-cache helpers for the corresponding figure
group. Functions consume `cache/data/` inputs and write derived artefacts to
`cache/data_proc/` via helpers in `cache.py`. Plotting code should only
consume the cached outputs.
"""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.cache import cached_parquet, cached_pickle
from utils.config import DATA_DIR, DATASET, HIT_THRE, N_FLOW, PARAMS_DIR, SIDE_CHAR


# === 0. Shared loaders ===

from utils.vic import (
    get_cb_vic_binocular,
    get_cb_vic_binocular_type,
    get_cb_vic_lr_homologue,
    get_ol_cb_vic_raw,
    get_ol_cb_vic_type,
    get_ol_type_vic,
    get_roi_syn_vic,
    get_vcbn_types,
)


def _load_meta_groups(dataset: str, data_dir) -> pd.DataFrame:
    """Load meta CSV and annotate with main_groups / region.

    Migrated from legacy `loading_functions.load_meta_groups`.
    """
    meta = pd.read_csv(data_dir / f"{dataset}_meta.csv")
    meta["cell_type"] = meta["cell_type"].fillna("unknown")
    meta["instance"] = meta["instance"].fillna("unknown")

    # sign=0 unless neurotransmitter is acetylcholine / gaba / glutamate
    idx_0 = meta[~meta.nt.isin(["acetylcholine", "gaba", "glutamate"])].index
    meta.loc[idx_0, "sign"] = 0

    # main group for each cell type (from Nern et al. SuppTable01)
    seed_types = ["L1", "L2", "L3", "R8", "R7", "R8d", "R7d", "HBeyelet"]
    table_df = pd.read_excel(PARAMS_DIR / "Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx")
    table_df["main_groups"] = table_df["main groups"].replace(
        {"ONCN": "OL internal", "ONIN": "OL internal", "VPN": "OL output", "VCN": "CB input"}
    )
    table_df.loc[
        table_df[table_df["cell type"].isin(seed_types)].index, "main_groups"
    ] = "visual input"

    # if cell type exists on both sides, keep the right-side row
    both_sides = (
        table_df.assign(side=table_df["instance"].str[-2:])
        .groupby("cell type")["side"]
        .apply(lambda x: set(x) >= {"_L", "_R"})
    )
    table_df = table_df[
        ~(
            table_df["cell type"].isin(both_sides[both_sides].index)
            & table_df["instance"].str.endswith("_L")
        )
    ]
    meta = meta.merge(
        table_df[["cell type", "main_groups"]],
        how="left",
        left_on="cell_type",
        right_on="cell type",
    ).drop(columns=["cell type"])

    # R7/R8 are split into p/y in table_df — label their parents directly
    meta.loc[meta[meta.cell_type == "R7"].index, "main_groups"] = "visual input"
    meta.loc[meta[meta.cell_type == "R8"].index, "main_groups"] = "visual input"
    meta["main_groups"] = meta["main_groups"].fillna("nonOL")

    # region
    meta["region"] = "CB"
    meta.loc[
        meta[
            meta.superclass.isin(
                ["cb_intrinsic", "cb_motor", "cb_sensory", "sensory_descending", "descending_neuron"]
            )
        ].index,
        "region",
    ] = "CB"
    meta.loc[
        meta[
            meta.superclass.isin(
                [
                    "vnc_intrinsic", "vnc_motor", "vnc_sensory", "vnc_efferent",
                    "sensory_ascending", "efferent_ascending", "ascending_neuron",
                ]
            )
        ].index,
        "region",
    ] = "VNC"
    right_instances = list(table_df.instance.values) + ["R7_R", "R8_R"]
    meta.loc[meta[meta.instance.isin(right_instances)].index, "region"] = "right OL"
    left_instances = [
        re.sub("_TMP$", "_R", re.sub("_R$", "_L", re.sub("_L$", "_TMP", s)))
        for s in right_instances
    ]
    meta.loc[meta[meta.instance.isin(left_instances)].index, "region"] = "left OL"

    # collapse residual "other" main_groups to nonOL/CB
    idx_other = meta[meta.main_groups == "other"].index
    meta.loc[idx_other, "main_groups"] = "nonOL"
    meta.loc[idx_other, "region"] = "CB"

    return meta


def get_meta(*, dataset: str = DATASET, data_dir=DATA_DIR, force: bool = False) -> pd.DataFrame:
    """Cached meta with `main_groups`, `region`, cleaned `cell_type`/`instance`."""
    return cached_parquet(
        f"{dataset}_meta",
        lambda: _load_meta_groups(dataset, data_dir),
        force=force,
    )


def is_cb_neuron(meta: pd.DataFrame) -> pd.Series:
    """CB-analysis mask: anatomical CB-region neurons"""
    return meta["region"] == "CB"
    

def get_flow(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    side_char: str = SIDE_CHAR, n_flow: int = N_FLOW, hit_thre: float = HIT_THRE,
    force: bool = False,
) -> pd.DataFrame:
    """Per-neuron hitting times; adds per-instance `count`.

    Migrated from legacy `loading_functions.load_flow_data`.
    """
    def _compute():
        fp = data_dir / f"{dataset}_{side_char}_flow_{n_flow}step_{hit_thre}thre_hit.csv"
        flow = pd.read_csv(fp)
        flow["count"] = flow.groupby("instance")["instance"].transform("count")
        return flow

    return cached_parquet(f"{dataset}_{side_char}_flow", _compute, force=force)


def get_flow_per_group(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    side_char: str = SIDE_CHAR, n_flow: int = N_FLOW, hit_thre: float = HIT_THRE,
    force: bool = False,
) -> pd.DataFrame:
    """Per-instance median hitting times. Reads `..._hit_per_group.csv`
    written by `fetch_instance_flow`."""
    def _compute():
        fp = data_dir / f"{dataset}_{side_char}_flow_{n_flow}step_{hit_thre}thre_hit_per_group.csv"
        return pd.read_csv(fp)

    return cached_parquet(f"{dataset}_{side_char}_flow_per_group", _compute, force=force)


_SIDE_WORD = {"r": "right", "l": "left"}


def _ensure_ol_sidecars(
    dataset: str, data_dir, side_char: str,
    n_flow: int = N_FLOW, hit_thre: float = HIT_THRE,
):
    """Materialise `OL_{side_char}_*` sidecar files in `cache/data/` if missing.

    Builds from the full-brain files. Mirrors legacy `load_OL_data`: takes
    region == '{side} OL' & main_groups != 'other' & instance != 'R1-R6_R',
    reindexes to local 0..N-1 `idx`, keeps original full-brain idx as `idx_full`,
    and slices prop / lat_flow_0 / lat_flow_sum accordingly.
    """
    ol_meta_f = data_dir / f"{dataset}_OL_{side_char}_meta.csv"
    ol_flow_f = data_dir / f"{dataset}_OL_{side_char}_flow_{n_flow}step_{hit_thre}thre_hit.csv"
    ol_prop_f = data_dir / f"{dataset}_OL_{side_char}_prop.npz"
    ol_sum_f  = data_dir / f"{dataset}_OL_{side_char}_lat_flow_sum.npz"
    ol_0_f    = data_dir / f"{dataset}_OL_{side_char}_lat_flow_0.npz"

    targets = [ol_meta_f, ol_flow_f, ol_prop_f, ol_sum_f, ol_0_f]
    if all(p.exists() for p in targets):
        return

    side_word = _SIDE_WORD[side_char]
    meta = get_meta(dataset=dataset, data_dir=data_dir)
    idx_sub = meta[
        (meta.region == f"{side_word} OL")
        & (meta.main_groups != "other")
        & (meta.instance != "R1-R6_R")
    ].index.values
    idx_sub = np.sort(idx_sub)

    if not ol_meta_f.exists():
        meta_ol = (
            meta.set_index("idx").loc[idx_sub]
            .reset_index(names="idx_full")
            .reset_index(names="idx")
        )
        meta_ol.to_csv(ol_meta_f, index=False)

    if not ol_flow_f.exists():
        flow = pd.read_csv(
            data_dir / f"{dataset}_{side_char}_flow_{n_flow}step_{hit_thre}thre_hit.csv"
        )
        flow = flow.merge(
            meta[["idx", "sign", "main_groups", "cell_type"]], on="idx", how="right"
        )
        flow_sub = flow.set_index("idx").loc[idx_sub].reset_index()
        flow_ol = (
            flow_sub.groupby("instance")
            .agg({
                "cell_type": "first", "hitting_time": "median", "sign": "first",
                "main_groups": "first", "idx": "count",
            })
            .reset_index()
            .rename(columns={"idx": "count"})
        )
        flow_ol.to_csv(ol_flow_f, index=False)

    if not ol_prop_f.exists():
        prop = sp.load_npz(data_dir / f"{dataset}_prop.npz")
        sp.save_npz(ol_prop_f, prop[idx_sub][:, idx_sub])

    if not ol_0_f.exists():
        steps_0 = sp.load_npz(data_dir / f"{dataset}_{side_char}_lat_flow_0.npz")
        sp.save_npz(ol_0_f, steps_0[idx_sub][:, idx_sub])

    if not ol_sum_f.exists():
        steps_sum = sp.load_npz(data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz")
        sp.save_npz(ol_sum_f, steps_sum[idx_sub][:, idx_sub])


def get_ol_meta(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """OL-subset meta (sidecar in `cache/data/`, built from full-brain files if missing)."""
    def _compute():
        _ensure_ol_sidecars(dataset, data_dir, side_char)
        return pd.read_csv(data_dir / f"{dataset}_OL_{side_char}_meta.csv")

    return cached_parquet(f"{dataset}_OL_{side_char}_meta", _compute, force=force)


def get_ol_flow_type(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    n_flow: int = N_FLOW, hit_thre: float = HIT_THRE, force: bool = False,
) -> pd.DataFrame:
    """Per-OL-instance median hitting time (sidecar in `cache/data/`, built if missing)."""
    def _compute():
        _ensure_ol_sidecars(dataset, data_dir, side_char, n_flow=n_flow, hit_thre=hit_thre)
        return pd.read_csv(
            data_dir / f"{dataset}_OL_{side_char}_flow_{n_flow}step_{hit_thre}thre_hit.csv"
        )

    return cached_parquet(f"{dataset}_OL_{side_char}_flow_type", _compute, force=force)


def get_ol_connectivity(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Long-format connectivity for pre-neurons in OL, with pre/post meta
    and per-instance median hitting times joined.
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        ol_instances = set(ol_meta.instance.unique())

        prop = sp.load_npz(data_dir / f"{dataset}_prop.npz").tocoo()
        conn = pd.DataFrame({"idx_pre": prop.row, "idx_post": prop.col, "weight": prop.data})

        meta_small = meta[["idx", "bodyId", "instance", "region", "main_groups", "sign"]]
        conn = conn.merge(meta_small, left_on="idx_pre", right_on="idx", how="left").rename(
            columns={
                "bodyId": "bodyId_pre", "instance": "instance_pre", "region": "region_pre",
                "main_groups": "main_groups_pre", "sign": "sign_pre",
            }
        ).drop(columns="idx")
        conn = conn[conn["instance_pre"].isin(ol_instances)]

        conn = conn.merge(meta_small, left_on="idx_post", right_on="idx", how="left").rename(
            columns={
                "bodyId": "bodyId_post", "instance": "instance_post", "region": "region_post",
                "main_groups": "main_groups_post", "sign": "sign_post",
            }
        ).drop(columns="idx")

        flow_type = get_flow_per_group(dataset=dataset, data_dir=data_dir)
        conn = conn.merge(
            flow_type, left_on="instance_pre", right_on="instance", how="left"
        ).rename(columns={"hitting_time": "hitting_time_pre"}).drop(columns="instance")
        conn = conn.merge(
            flow_type, left_on="instance_post", right_on="instance", how="left"
        ).rename(columns={"hitting_time": "hitting_time_post"}).drop(columns="instance")

        conn = conn[~pd.isna(conn["hitting_time_pre"]) & ~pd.isna(conn["hitting_time_post"])]
        conn["hitting_time_diff"] = conn["hitting_time_post"] - conn["hitting_time_pre"]
        conn["frac"] = conn["weight"] / conn.groupby("idx_pre")["weight"].transform("sum")
        return conn.reset_index(drop=True)

    return cached_parquet(f"{dataset}_OL_{side_char}_connectivity", _compute, force=force)


def _compute_directedness(conn_df: pd.DataFrame, hit_diff_thre: float = 0.5) -> pd.DataFrame:
    """Per-pre-neuron fractions of feedforward / feedback / lateral weight.

    Migrated from legacy `computing_functions.compute_directedness`.
    Requires `conn_df` to already have `hitting_time_diff` and `frac` columns.
    """
    dir_ff = (
        conn_df[conn_df["hitting_time_diff"] > hit_diff_thre]
        .groupby("idx_pre").agg({"frac": "sum", "instance_pre": "first"})
        .reset_index().rename(columns={"frac": "frac_ff"})
    )
    dir_fb = (
        conn_df[conn_df["hitting_time_diff"] < -hit_diff_thre]
        .groupby("idx_pre").agg({"frac": "sum", "instance_pre": "first"})
        .reset_index().rename(columns={"frac": "frac_fb"})
    )
    dir_la = (
        conn_df[np.abs(conn_df["hitting_time_diff"]) <= hit_diff_thre]
        .groupby("idx_pre").agg({"frac": "sum", "instance_pre": "first"})
        .reset_index().rename(columns={"frac": "frac_la"})
    )
    dir_df = (
        dir_ff.merge(dir_fb, on=["idx_pre", "instance_pre"], how="outer")
        .merge(dir_la, on=["idx_pre", "instance_pre"], how="outer")
        .fillna(0)
    )
    dir_df["cell_type_pre"] = [s[:-2] for s in dir_df["instance_pre"].values]
    return dir_df


def get_ol_directedness(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    hit_diff_thre: float = 0.5, force: bool = False,
) -> pd.DataFrame:
    """Per-neuron feedforward / feedback / lateral fractions for OL-pre connections."""
    def _compute():
        conn_ol = get_ol_connectivity(dataset=dataset, data_dir=data_dir, side_char=side_char)
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        dir_df = _compute_directedness(conn_ol, hit_diff_thre=hit_diff_thre)
        dir_df = dir_df.merge(
            ol_meta[["instance", "main_groups", "sign"]].drop_duplicates(),
            left_on="instance_pre", right_on="instance", how="left",
        ).rename(columns={"main_groups": "main_groups_pre", "sign": "sign_pre"}).drop(columns="instance")
        return dir_df

    tag = f"thre{hit_diff_thre}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_directedness_{tag}", _compute, force=force
    )


def get_ol_prop(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
):
    """Sparse OL-subset connectivity matrix (sidecar in `cache/data/`, built if missing)."""
    _ensure_ol_sidecars(dataset, data_dir, side_char)
    return sp.load_npz(data_dir / f"{dataset}_OL_{side_char}_prop.npz")


def get_ol_stepsn_sum(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
):
    """Sparse OL-subset summed lateral-flow matrix (sidecar in `cache/data/`, built if missing)."""
    _ensure_ol_sidecars(dataset, data_dir, side_char)
    return sp.load_npz(data_dir / f"{dataset}_OL_{side_char}_lat_flow_sum.npz")


def get_ol_stepsn_0(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
):
    """Sparse OL-subset `lat_flow_0` matrix (sidecar in `cache/data/`, built if missing)."""
    _ensure_ol_sidecars(dataset, data_dir, side_char)
    return sp.load_npz(data_dir / f"{dataset}_OL_{side_char}_lat_flow_0.npz")


_LAYER_EXAMPLE_TYPES = ['T4a']


def load_layer_example_types(side_char: str = SIDE_CHAR) -> list:
    """Cell types used as "layer example" pathway subjects (legacy helper)."""
    suffix = '_R' if side_char == 'r' else '_L'
    return [t + suffix for t in _LAYER_EXAMPLE_TYPES]


def _grouped_paths(stepsn, inidx, outidx, n_interm: int, idx_to_instance: dict,
                   outprop: bool = False) -> list:
    """Collect per-length `find_paths_of_length` results, grouped by instance.

    `outprop=False` (default) gives input-normalised edges (fraction of post's
    input coming from pre-type). `outprop=True` gives output-normalised edges
    (fraction of pre's output going to post-type).
    """
    from connectome_interpreter.path_finding import find_paths_of_length, group_paths
    all_paths = []
    for k in range(1, n_interm + 1):
        paths = find_paths_of_length(stepsn, inidx, outidx, k)
        if paths is None or paths.empty:
            continue
        paths = group_paths(
            paths, pre_group=idx_to_instance, post_group=idx_to_instance,
            combining_method='mean', outprop=outprop,
        )
        all_paths.append(paths)
    return all_paths


def get_full_prop(*, dataset: str = DATASET, data_dir=DATA_DIR):
    """Full sparse connectivity matrix."""
    return sp.load_npz(data_dir / f"{dataset}_prop.npz")


def get_ol_type_directedness(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    hit_diff_thre: float = 0.5, force: bool = False,
) -> pd.DataFrame:
    """Per-instance (median) directedness, renormalised to sum=1."""
    def _compute():
        d = get_ol_directedness(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            hit_diff_thre=hit_diff_thre,
        )
        t = (
            d.groupby(["instance_pre", "cell_type_pre", "main_groups_pre", "sign_pre"])
            [["frac_ff", "frac_fb", "frac_la"]].median().reset_index()
        )
        t[["frac_ff", "frac_fb", "frac_la"]] = (
            t[["frac_ff", "frac_fb", "frac_la"]]
            / t[["frac_ff", "frac_fb", "frac_la"]].sum(1).values[:, np.newaxis]
        )
        return t.sort_values("main_groups_pre").reset_index(drop=True)

    tag = f"thre{hit_diff_thre}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_type_directedness_{tag}", _compute, force=force
    )


# === 1. Inventory ===


def get_inventory_meta(
    *, dataset: str = DATASET, data_dir=DATA_DIR, force: bool = False
) -> pd.DataFrame:
    """Meta filtered and `type` canonicalised for inventory pies.

    Drops left OL and 'other' regions, drops `*_unclear` and `R1-R6` rows, and
    collapses R7p/R7y/R8p/R8y variants into R7/R8.
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir).copy()
        meta["type"] = meta["cell_type"]
        # meta = meta[(meta.region != "other") & (meta.region != "left OL")]
        meta = meta[(~meta.type.str.contains("unclear")) & (meta.type != "R1-R6")]
        meta["type"] = meta["type"].replace(
            {"R7p": "R7", "R7y": "R7", "R8p": "R8", "R8y": "R8"}
        )
        return meta

    return cached_parquet(f"{dataset}_inventory_meta", _compute, force=force)


# === 2. Flow ===


def get_flow_ol(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-neuron flow restricted to OL instances, with per-instance median joined."""
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        flow = get_flow(dataset=dataset, data_dir=data_dir, side_char=side_char)
        flow = flow.merge(meta[["idx", "main_groups", "sign", "bodyId"]], on="idx", how="right")
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        flow = flow[flow.instance.isin(ol_meta.instance.unique())]
        flow_type = get_ol_flow_type(dataset=dataset, data_dir=data_dir, side_char=side_char)
        flow = flow.merge(
            flow_type[["instance", "hitting_time"]], on="instance", how="left", suffixes=("", "_median")
        )
        return flow

    return cached_parquet(f"{dataset}_OL_{side_char}_flow_neurons", _compute, force=force)


def get_sector_map(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Assign each visual-column coord to one of 5 retinotopic sectors."""
    def _compute():
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        df = ol_meta[ol_meta.instance.str.startswith("L2")][["coords"]].drop_duplicates().reset_index(drop=True)
        df = df[~pd.isna(df["coords"])].copy()
        xy = df["coords"].str.split(",", expand=True).astype(int).values
        df["hex1_id"] = (xy[:, 1] - xy[:, 0]) / 2
        df["hex2_id"] = (xy[:, 0] + xy[:, 1]) / 2
        xy2 = xy.astype(float).copy()
        xy2[:, 0] = xy2[:, 0] - 1
        xy2[:, 1] = (xy2[:, 1] - 39) / np.sqrt(3) / 1.3

        df["sector"] = 0
        df.loc[df[(df["hex1_id"] > 21) & (df["hex2_id"] > 24)].index, "sector"] = 2
        df.loc[df[(df["hex1_id"] > 15) & (df["hex2_id"] <= 24)].index, "sector"] = 1
        df.loc[df[(df["hex1_id"] <= 21) & (df["hex2_id"] > 16)].index, "sector"] = 3
        df.loc[df[(df["hex1_id"] <= 15) & (df["hex2_id"] <= 16)].index, "sector"] = 4
        df.loc[df[xy2[:, 0] ** 2 + xy2[:, 1] ** 2 < 40].index, "sector"] = 5

        manual_5 = [
            "-5,47", "-5,45", "-5,33", "-5,31", "7,47", "7,45", "7,33", "7,31",
            "-3,51", "-1,53", "1,55", "3,53", "5,51",
            "-3,27", "-1,25", "1,23", "3,25", "5,27",
        ]
        df.loc[df.coords.isin(manual_5), "sector"] = 5
        return df[["coords", "sector"]].reset_index(drop=True)

    return cached_parquet(f"{dataset}_coords_to_sectors", _compute, force=force)


def get_ol_layers(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-synapse (bodyId, roi, hitting_time) rows for OL anatomical layers.

    Keeps `ME_{S}_layer_*`, `LO_{S}_layer_*`, `LOP_{S}_layer_*`, and `AME({S})`
    (S = `side_char.upper()`). Each (bodyId, roi) row is expanded by the
    presynapse count so each synapse contributes one sample to violins.
    Hitting time is joined from the side-specific flow CSV.
    """
    def _compute():
        side_upper = side_char.upper()
        roi_counts = pd.read_pickle(
            data_dir / f"{dataset}_OL_{side_char}_roi_counts.pkl"
        )
        prefixes = (
            f"ME_{side_upper}_layer_",
            f"LO_{side_upper}_layer_",
            f"LOP_{side_upper}_layer_",
        )
        mask = (
            roi_counts["roi"].str.startswith(prefixes)
            | (roi_counts["roi"] == f"AME({side_upper})")
        )
        sub = roi_counts.loc[mask, ["bodyId", "roi", "pre"]]
        expanded = sub.loc[sub.index.repeat(sub["pre"])].drop(columns="pre")
        flow = get_flow(dataset=dataset, data_dir=data_dir, side_char=side_char)
        expanded = expanded.merge(
            flow[["bodyId", "hitting_time"]], on="bodyId", how="left"
        )
        return expanded.reset_index(drop=True)

    return cached_parquet(f"{dataset}_OL_{side_char}_layers", _compute, force=force)


def get_ol_roi_adjacency(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-(pre, post, roi) adjacency weights joined with pre/post hitting times.

    Loads `{dataset}_OL_{side_char}_roi_adj.pkl` (columns bodyId_pre,
    bodyId_post, roi, weight) and merges hitting_time from the full flow
    table onto both pre and post bodyIds. Adds `hit_diff = hitting_time_post
    - hitting_time_pre`. Rows with missing pre or post hitting time are
    dropped.
    """
    def _compute():
        adj = pd.read_pickle(data_dir / f"{dataset}_OL_{side_char}_roi_adj.pkl")
        flow = get_flow(dataset=dataset, data_dir=data_dir, side_char=side_char)
        hit = flow[["bodyId", "hitting_time"]]
        adj = adj.merge(
            hit.rename(columns={"bodyId": "bodyId_pre", "hitting_time": "hitting_time_pre"}),
            on="bodyId_pre", how="left",
        ).merge(
            hit.rename(columns={"bodyId": "bodyId_post", "hitting_time": "hitting_time_post"}),
            on="bodyId_post", how="left",
        )
        adj = adj.dropna(subset=["hitting_time_pre", "hitting_time_post"])
        adj["hit_diff"] = adj["hitting_time_post"] - adj["hitting_time_pre"]
        return adj.reset_index(drop=True)

    return cached_parquet(f"{dataset}_OL_{side_char}_roi_adjacency", _compute, force=force)


# === 3. Clusters ===


def _compute_feature_vectors(stepsn, inidx, outidx, inidx_map, outidx_map) -> pd.DataFrame:
    """Migrated from legacy `computing_functions.compute_feature_vectors`.

    For each output neuron, the feature is the median (across same-instance
    output neurons) of the row-summed effective input weight from each input
    instance. Result: rows = output instances, cols = input instances.
    """
    in_out_df = pd.DataFrame(
        data=stepsn[:, outidx][inidx, :].toarray(),
        index=[str(inidx_map[k]) for k in inidx],
        columns=outidx,
    )
    in_out_df = in_out_df.groupby(in_out_df.index).sum()
    in_out_df.columns = [str(outidx_map[k]) for k in in_out_df.columns]
    in_out_df = in_out_df.T.groupby(level=0).median().T
    return in_out_df.T


def get_ol_features(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Visual-input → OL-output feature vectors (heavy compute via lat_flow_sum)."""
    def _compute():
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_sum(dataset=dataset, data_dir=data_dir, side_char=side_char)
        idx_to_instance = dict(zip(ol_meta.idx, ol_meta.instance))
        in_inst = ol_meta[ol_meta.main_groups == "visual input"].instance.unique()
        out_inst = ol_meta[ol_meta.main_groups == "OL output"].instance.unique()
        inidx = ol_meta[ol_meta.instance.isin(in_inst)].idx.values
        outidx = ol_meta[ol_meta.instance.isin(out_inst)].idx.values
        feat = _compute_feature_vectors(stepsn, inidx, outidx, idx_to_instance, idx_to_instance)
        feat.index.name = "instance"
        return feat.reset_index()

    df = cached_parquet(f"{dataset}_OL_{side_char}_features", _compute, force=force)
    return df.set_index("instance")


def _compute_hierarchical_clustering(feature_df, method="ward", metric="euclidean"):
    import scipy.cluster.hierarchy as sch
    feature_vec = feature_df.values
    feature_vec = feature_vec / feature_vec.sum(1)[:, np.newaxis]
    return sch.linkage(feature_vec, method=method, metric=metric)


def get_ol_clusters(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.21, method: str = "ward", metric: str = "euclidean",
    force: bool = False,
) -> pd.DataFrame:
    """Flat clusters of OL-output instances (cached). frac_thre is a fraction of `Z[:,2].max()`."""
    import scipy.cluster.hierarchy as sch

    def _compute():
        feat = get_ol_features(dataset=dataset, data_dir=data_dir, side_char=side_char)
        Z = _compute_hierarchical_clustering(feat, method=method, metric=metric)
        thre = frac_thre * Z[:, 2].max()
        cluster_idx = sch.fcluster(Z, thre, criterion="distance")
        # Reverse cluster numbering so c1 corresponds to scipy's last leaf
        # (matches the convention used by get_ol_clusters_intra). Without this,
        # the intra heatmap and other cluster plots would be mirror images.
        n_clu = int(cluster_idx.max())
        cluster_idx = n_clu + 1 - cluster_idx
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        df = pd.DataFrame({"instance": feat.index.values, "cluster": cluster_idx})
        df = df.merge(
            ol_meta.groupby("instance")["bodyId"].count().reset_index(name="count"),
            on="instance", how="left",
        )
        return df

    tag = f"{frac_thre:.3f}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_clusters_{tag}", _compute, force=force,
    )


def _compute_participation(meta_ol, stepsn, cluster_df):
    n_clu = int(cluster_df["cluster"].max())
    in_out_mat = np.zeros((stepsn.shape[0], n_clu))
    for i in range(n_clu):
        out_inst_i = cluster_df[cluster_df.cluster == i + 1]["instance"].values
        outidx_i = meta_ol[meta_ol.instance.isin(out_inst_i)].index.values
        in_out_mat[:, i] = stepsn[:, outidx_i].toarray().sum(1) / max(len(outidx_i), 1)
    for i in range(n_clu):
        out_inst_i = cluster_df[cluster_df.cluster == i + 1]["instance"].values
        outidx_i = meta_ol[meta_ol.instance.isin(out_inst_i)].index.values
        in_out_mat[outidx_i, :] = 0
        in_out_mat[outidx_i, i] = 1
    in_out_mat = in_out_mat / in_out_mat.sum(1)[:, np.newaxis]
    cluster_names = [f"c{i + 1}" for i in range(n_clu)]
    df = pd.DataFrame(in_out_mat, columns=cluster_names, index=meta_ol.index)
    return pd.concat([df, meta_ol], axis=1)


def get_ol_participation(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.21, force: bool = False,
) -> pd.DataFrame:
    """Per-neuron participation in each OL-output cluster (cached)."""
    def _compute():
        meta_ol = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_sum(dataset=dataset, data_dir=data_dir, side_char=side_char)
        clu = get_ol_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char, frac_thre=frac_thre,
        )
        return _compute_participation(meta_ol, stepsn, clu)

    tag = f"{frac_thre:.3f}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_participation_{tag}", _compute, force=force,
    )


def get_ol_type_participation(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.21, force: bool = False,
) -> pd.DataFrame:
    """Per-instance (median) participation, normalised, with idxmax / max / n_clusters."""
    def _compute():
        part = get_ol_participation(
            dataset=dataset, data_dir=data_dir, side_char=side_char, frac_thre=frac_thre,
        )
        cluster_names = list(part.columns[part.columns.str.contains(r"^c\d+$")])
        df = part.groupby("instance")[cluster_names].median().reset_index()
        vals = df[cluster_names].values
        df[cluster_names] = vals / vals.sum(1)[:, np.newaxis]
        df = df[~pd.isna(df[cluster_names[0]])]
        df["max"] = np.sort(df[cluster_names].values, axis=1)[:, -1:].sum(1)
        df["idxmax"] = df[cluster_names].idxmax(1)
        df["n_clusters"] = (df[cluster_names] >= 1 / len(cluster_names)).sum(1)
        return df

    tag = f"{frac_thre:.3f}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_OL_{side_char}_type_participation_{tag}", _compute, force=force,
    )


def get_ol_connectivity_pathways(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.21, force: bool = False,
):
    """Pathway-level connectivity matrices (full + e/i breakdowns) and their diagonals.

    Returns a dict {'conn': list[5][N+2,N+2], 'diag': list[5][N+2]} (cached as pickle).
    Migrated from legacy `computing_functions.compute_connectivity_pathways`.
    """
    from utils.cache import cached_pickle
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        prop = get_full_prop(dataset=dataset, data_dir=data_dir)
        cluster_df = get_ol_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char, frac_thre=frac_thre,
        )
        idx_to_instance = dict(zip(meta.idx, meta.instance))
        n_clu = int(cluster_df["cluster"].max())
        instance_list = np.zeros(n_clu + 2, dtype=object)
        instance_list[0] = meta[
            (meta.main_groups == "visual input") & (meta.region == "right OL")
        ].instance.values
        instance_list[-1] = meta[is_cb_neuron(meta)].instance.values
        for i in range(n_clu):
            instance_list[i + 1] = cluster_df[cluster_df.cluster == i + 1]["instance"].values

        pre_signs = [[-1, 0, 1], [1], [1], [-1], [-1]]
        post_signs = [[-1, 0, 1], [1], [-1], [1], [-1]]
        n_combi = len(pre_signs)
        conn_list = [np.zeros((n_clu + 2, n_clu + 2)) for _ in range(n_combi)]
        diag_list = [np.zeros(n_clu + 2) for _ in range(n_combi)]

        for i in range(n_combi):
            for k in range(n_clu + 2):
                pre_idx = meta[
                    meta.instance.isin(instance_list[k]) & meta.sign.isin(pre_signs[i])
                ].index.values
                for l in range(n_clu + 2):
                    post_idx = meta[
                        meta.instance.isin(instance_list[l]) & meta.sign.isin(post_signs[i])
                    ].index.values
                    conn_list[i][k, l] = prop[pre_idx][:, post_idx].sum()
                    if (k == l) and (k < n_clu + 1) and len(pre_idx) and len(post_idx):
                        self_conn = result_summary(
                            prop, pre_idx, post_idx,
                            inidx_map=idx_to_instance, outidx_map=idx_to_instance,
                            display_threshold=0, combining_method="sum",
                            display_output=False,
                        )
                        common = list(set(self_conn.index.values) & set(self_conn.columns.values))
                        diag_list[i][k] = self_conn.loc[common, common].values.diagonal().sum()
        return {"conn": conn_list, "diag": diag_list}

    tag = f"{frac_thre:.3f}".replace(".", "p")
    return cached_pickle(
        f"{dataset}_OL_{side_char}_connectivity_pathways_{tag}", _compute, force=force,
    )


def get_ol_clusters_intra(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.15, force: bool = False,
):
    """Cluster-cluster connectivity matrix (output-normalised) and total weight.

    Used by section 3's intra-cluster connectivity heatmap.
    Returns dict {'matrix': ndarray, 'total_weight': float, 'n_clu': int}.
    """
    from utils.cache import cached_pickle
    from connectome_interpreter.compress_paths import result_summary
    import scipy.cluster.hierarchy as sch

    def _compute():
        feat = get_ol_features(dataset=dataset, data_dir=data_dir, side_char=side_char)
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        prop_ol = get_ol_prop(dataset=dataset, data_dir=data_dir, side_char=side_char)
        idx_to_instance = dict(zip(ol_meta.idx, ol_meta.instance))

        Z = _compute_hierarchical_clustering(feat)
        thre = frac_thre * Z[:, 2].max()
        cluster_idx = sch.fcluster(Z, thre, criterion="distance")
        clu = pd.DataFrame({"instance": feat.index.values, "cluster": cluster_idx})
        n_clu5 = int(clu["cluster"].max())
        clu["cluster"] = n_clu5 + 1 - clu["cluster"]

        outidx = ol_meta[ol_meta.instance.isin(clu["instance"].values)].idx.values
        instance_to_cluster = dict(zip(clu["instance"], clu["cluster"]))
        idx_to_cluster = {
            idx: instance_to_cluster[inst]
            for idx, inst in idx_to_instance.items() if inst in instance_to_cluster
        }
        m = result_summary(
            prop_ol, outidx, outidx, idx_to_cluster, idx_to_cluster,
            combining_method="sum", display_output=False,
        )
        m.index = [int(i) for i in m.index]
        m.columns = [int(i) for i in m.columns]
        m = m.loc[range(1, n_clu5 + 1), range(1, n_clu5 + 1)].fillna(0).values
        total = m.sum()
        m = m / m.sum(axis=1)[:, np.newaxis]
        return {"matrix": m, "total_weight": float(total), "n_clu": n_clu5}

    tag = f"{frac_thre:.3f}".replace(".", "p")
    return cached_pickle(
        f"{dataset}_OL_{side_char}_clusters_intra_{tag}", _compute, force=force,
    )


def get_ol_sankey_connectivity(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.19,
    in_instances=(
        "L1_R", "L2_R", "L3_R", "R7_R", "R8_R",
        "R7d_R", "R8d_R", "HBeyelet_R",
    ),
    weight_thre: float = 5.0,
    force: bool = False,
):
    """Pre-build the diagram-level connectivity matrix used by the Sankey plot.

    Bins per-instance median hitting times into layers 0..3 (clamp ≥3.5 to 3)
    for OL **and** CB / left OL neurons, builds diagram labels (input cell
    types, OLI.{i}, OLO{j}.{i}, plus 'CB' / 'left OL'), and aggregates inprop
    weight via `result_summary`. The bin-pair forward loop (pre_bin < post_bin)
    enforces `layer_diff > 0` and drops self-loops and backward edges; rows
    for terminals (CB / left OL) are zeroed post-aggregation so they remain
    pure sinks. Returns a dict with `conn_diagram` and `meta_sub`.
    """
    from utils.cache import cached_pickle
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        clusters = get_ol_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char, frac_thre=frac_thre,
        )
        prop = get_full_prop(dataset=dataset, data_dir=data_dir)

        # Per-instance median hitting time across the broad flow table — covers
        # OL, CB, and left-OL instances so terminals get real bins. Clamp at 6
        # since CB / left-OL neurons can have hit_times beyond the OL range.
        flow_b = get_flow_per_group(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        ).copy()
        flow_b.loc[flow_b.hitting_time >= 6.5, "hitting_time"] = 6
        bins = np.arange(-0.5, 6 + 0.5 + 1, 1)
        labels = (bins[1:] + bins[:-1]) / 2
        flow_b["hit_bin"] = pd.cut(flow_b["hitting_time"], bins=bins, labels=labels).astype(float)

        ol_inst = ol_meta[
            ol_meta.main_groups.isin(["visual input", "OL internal", "OL output"])
        ].instance.unique()
        cb_inst = meta[is_cb_neuron(meta) | (meta.region == "left OL")].instance.unique()
        meta_sub = meta[meta.instance.isin(ol_inst) | meta.instance.isin(cb_inst)].reset_index(drop=True).copy()
        meta_sub = meta_sub.merge(clusters[["instance", "cluster"]], how="left", on="instance")
        meta_sub = meta_sub.merge(flow_b[["instance", "hit_bin"]], how="left", on="instance")
        meta_sub = meta_sub[meta_sub.region != "other"].reset_index(drop=True).copy()

        cb_mask = is_cb_neuron(meta_sub)
        loff_mask = (meta_sub.region == "left OL") & ~cb_mask
        terminal_mask = cb_mask | loff_mask

        # OL neurons display in bins 1..3 only — clamp any OL hit_bin > 3 to 3
        # so OLI.4+ never appears (layer 4 is reserved for the terminal box).
        # CB / left-OL keep their real bin (up to 6) so backward-edge detection
        # against deeper OL neurons stays correct.
        ol_clamp = (~terminal_mask) & (meta_sub.hit_bin > 3)
        meta_sub.loc[ol_clamp, "hit_bin"] = 3

        n_clu = int(clusters["cluster"].max())
        n_bin = int(labels.max())  # 6 → max bin in filter loop (CB / left-OL range)
        meta_sub["diagram"] = "other"
        in_idx = meta_sub[meta_sub.instance.isin(in_instances)].index
        meta_sub.loc[in_idx, "diagram"] = meta_sub.loc[in_idx, "cell_type"]
        for i in range(3):
            oli_idx = meta_sub[
                (meta_sub.main_groups == "OL internal") & (meta_sub.hit_bin == i + 1)
            ].index
            meta_sub.loc[oli_idx, "diagram"] = f"OLI.{i + 1}"
            for j in range(n_clu):
                olo_idx = meta_sub[
                    (meta_sub.cluster == j + 1) & (meta_sub.hit_bin == i + 1)
                ].index
                if len(olo_idx) > 0:
                    meta_sub.loc[olo_idx, "diagram"] = f"OLO{j + 1}.{i + 1}"

        # CB and left OL are two separate terminal boxes, both at layer 4.
        meta_sub.loc[cb_mask, "diagram"] = "CB"
        meta_sub.loc[loff_mask, "diagram"] = "left OL"
        meta_sub = meta_sub[meta_sub.diagram != "other"].copy()
        meta_sub = meta_sub.dropna(subset=["hit_bin"]).copy()

        idx_to_diagram = dict(zip(meta_sub.idx, meta_sub.diagram))

        # output-normalized connectivity (= inprop)
        total_post = prop.sum(axis=1).A1.astype(float)
        inv = np.zeros_like(total_post)
        np.reciprocal(total_post, where=total_post != 0, out=inv)
        inprop = prop.multiply(inv.reshape((-1, 1))).astype(np.float32)

        diagram_names = list(meta_sub["diagram"].unique())
        conn_diagram = pd.DataFrame(
            np.zeros((len(diagram_names), len(diagram_names))),
            index=diagram_names, columns=diagram_names,
        )

        # Forward bin-pair loop: pre_bin = 0..n_bin-1 → post_bin > pre_bin.
        # Enforces layer_diff > 0 at the (post-binning) integer level, so
        # within-bin self-loops and backward edges never contribute.
        for pre_bin in range(n_bin):
            inidx = meta_sub[meta_sub.hit_bin == pre_bin].idx.values
            outidx = meta_sub[meta_sub.hit_bin > pre_bin].idx.values
            if len(inidx) == 0 or len(outidx) == 0:
                continue
            conn_i = result_summary(
                inprop, inidx, outidx, idx_to_diagram, idx_to_diagram,
                combining_method="sum", outprop=False, display_output=False,
            )
            conn_diagram.loc[conn_i.index, conn_i.columns] += conn_i

        # Terminal boxes are sinks-only: zero outgoing edges and self-loops on
        # the diagonal (CB → CB and left OL → left OL would otherwise
        # accumulate from intra-terminal forward bin pairs across bins 0..6).
        for term in ("CB", "left OL"):
            if term in conn_diagram.index:
                conn_diagram.loc[term, :] = 0.0
        for d in conn_diagram.index:
            if d in conn_diagram.columns:
                conn_diagram.loc[d, d] = 0.0

        return {
            "conn_diagram": conn_diagram,
            "meta_sub": meta_sub[["instance", "cell_type", "main_groups", "region",
                                  "cluster", "hit_bin", "diagram"]].copy(),
            "n_clu": n_clu,
            "weight_thre": weight_thre,
        }

    tag = f"{frac_thre:.3f}".replace(".", "p")
    return cached_pickle(
        f"{dataset}_OL_{side_char}_sankey_{tag}", _compute, force=force,
    )


def get_paths_to_instance(
    instance: str, *, dataset: str = DATASET, data_dir=DATA_DIR,
    side_char: str = SIDE_CHAR, in_main_group: str = 'visual input',
    force: bool = False,
) -> dict:
    """Effective-weight paths from `in_main_group` → `instance` in OL, for each
    path length 1..ceil(hit_time[instance]).

    Returns `{'paths': list[pd.DataFrame], 'hit_inst': float}`, where `paths` is
    the grouped (per-instance) output of `find_paths_of_length` for each length.
    Filtering / `conn_paths_frac` is deferred to plot time.
    """
    def _compute():
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_0(dataset=dataset, data_dir=data_dir, side_char=side_char)
        flow_type = get_ol_flow_type(dataset=dataset, data_dir=data_dir, side_char=side_char)
        idx_to_instance = dict(zip(ol_meta.idx, ol_meta.instance))
        in_instances = ol_meta[ol_meta.main_groups == in_main_group].instance.unique()
        inidx = ol_meta[ol_meta.instance.isin(in_instances)].idx.values
        outidx = ol_meta[ol_meta.instance == instance].idx.values
        hit_inst = float(flow_type[flow_type.instance == instance].hitting_time.values[0])
        n_interm = int(np.ceil(hit_inst))
        all_paths = _grouped_paths(stepsn, inidx, outidx, n_interm, idx_to_instance)
        return {'paths': all_paths, 'hit_inst': hit_inst}

    tag = instance.replace('/', '_')
    return cached_pickle(
        f"{dataset}_OL_{side_char}_paths_to_{tag}", _compute, force=force,
    )


def get_participation_paths(
    instance: str, *, dataset: str = DATASET, data_dir=DATA_DIR,
    side_char: str = SIDE_CHAR, out_main_group: str = 'OL output',
    outprop: bool = False, force: bool = False,
) -> dict:
    """Effective-weight paths from `instance` → `out_main_group` in OL, for each
    path length 1..ceil(max(out_hit) - hit[instance]).

    `outprop=False` (default, matches legacy) — input-normalised edges.
    `outprop=True` — output-normalised edges.
    """
    def _compute():
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_0(dataset=dataset, data_dir=data_dir, side_char=side_char)
        flow_type = get_ol_flow_type(dataset=dataset, data_dir=data_dir, side_char=side_char)
        idx_to_instance = dict(zip(ol_meta.idx, ol_meta.instance))
        out_instances = ol_meta[ol_meta.main_groups == out_main_group].instance.unique()
        inidx = ol_meta[ol_meta.instance == instance].idx.values
        outidx = ol_meta[ol_meta.instance.isin(out_instances)].idx.values
        hit_max = float(flow_type[flow_type.instance.isin(out_instances)].hitting_time.max())
        hit_self = float(flow_type[flow_type.instance == instance].hitting_time.values[0])
        hit_inst = hit_max - hit_self
        n_interm = int(np.ceil(hit_inst))
        all_paths = _grouped_paths(
            stepsn, inidx, outidx, n_interm, idx_to_instance, outprop=outprop,
        )
        return {'paths': all_paths, 'hit_inst': hit_inst}

    tag = instance.replace('/', '_')
    suffix = '_out' if outprop else ''
    return cached_pickle(
        f"{dataset}_OL_{side_char}_participation_paths_{tag}{suffix}",
        _compute, force=force,
    )


# === 4. LR clustering ===


def _swap_lr_suffix(s: str) -> str:
    """Flip trailing `_L`↔`_R` suffix; leave anything else alone."""
    return re.sub("_TMP$", "_R", re.sub("_R$", "_L", re.sub("_L$", "_TMP", s)))


def get_ol_lr_sweep(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    n_points: int = 48, frac_log2_min: float = -5.0, frac_log2_max: float = -0.1,
    method: str = "ward", metric: str = "euclidean",
    force: bool = False,
) -> pd.DataFrame:
    """Sweep clustering threshold on both sides; per-frac compare L/R.

    Each row: frac, n_clu_R, n_clu_L, ari_LR (cell-type-matched),
    ari_fromR (L clusters vs. centroid-assignment from R), diag (mean
    diagonal of output-normalised R cluster-cluster connectivity),
    silhouette / ch / db on R features.
    """
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import cdist
    from sklearn.metrics import (
        adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
        silhouette_score,
    )
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        feat_R = get_ol_features(dataset=dataset, data_dir=data_dir, side_char="r")
        feat_L = get_ol_features(dataset=dataset, data_dir=data_dir, side_char="l")
        meta_R = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char="r")
        prop_R = get_ol_prop(dataset=dataset, data_dir=data_dir, side_char="r")
        idx_to_instance_R = dict(zip(meta_R.idx, meta_R.instance))

        feat_R = feat_R[~feat_R.iloc[:, 0].isna()]
        feat_L = feat_L[~feat_L.iloc[:, 0].isna()]

        Z_R = _compute_hierarchical_clustering(feat_R, method=method, metric=metric)
        Z_L = _compute_hierarchical_clustering(feat_L, method=method, metric=metric)

        outidx_R = meta_R[meta_R.instance.isin(feat_R.index)].idx.values

        frac_list = 2.0 ** np.linspace(frac_log2_min, frac_log2_max, n_points)
        rows = []
        for frac in frac_list:
            cl_R = sch.fcluster(Z_R, frac * Z_R[:, 2].max(), criterion="distance")
            cl_L = sch.fcluster(Z_L, frac * Z_L[:, 2].max(), criterion="distance")
            n_R = int(cl_R.max())
            n_L = int(cl_L.max())

            df_R = pd.DataFrame({"instance": feat_R.index.values, "cluster": cl_R})
            df_L = pd.DataFrame({"instance": feat_L.index.values, "cluster": cl_L})
            df_R["cell_type"] = [s[:-2] for s in df_R["instance"]]
            df_L["cell_type"] = [s[:-2] for s in df_L["instance"]]
            compare = df_R.merge(df_L, on="cell_type", how="inner", suffixes=("_R", "_L"))
            ari_lr = adjusted_rand_score(compare["cluster_R"].values, compare["cluster_L"].values)

            instance_to_cluster_R = dict(zip(df_R["instance"], df_R["cluster"]))
            idx_to_cluster_R = {
                i: instance_to_cluster_R[inst]
                for i, inst in idx_to_instance_R.items()
                if inst in instance_to_cluster_R
            }
            m = result_summary(
                prop_R, outidx_R, outidx_R, idx_to_cluster_R, idx_to_cluster_R,
                combining_method="sum", display_output=False,
            )
            m.index = [int(i) for i in m.index]
            m.columns = [int(i) for i in m.columns]
            m = m.loc[range(1, n_R + 1), range(1, n_R + 1)].fillna(0).values
            m = m / m.sum(axis=1, keepdims=True)
            diag_mean = float(np.diag(m).sum() / n_R)

            silh = silhouette_score(feat_R.values, cl_R)
            ch = calinski_harabasz_score(feat_R.values, cl_R)
            db = davies_bouldin_score(feat_R.values, cl_R)

            centroids_R = np.vstack([
                feat_R.values[cl_R == c].mean(axis=0) for c in range(1, n_R + 1)
            ])
            labels_L_from_R = cdist(feat_L.values, centroids_R, metric="euclidean").argmin(axis=1) + 1
            ari_from_r = adjusted_rand_score(cl_L, labels_L_from_R)

            rows.append({
                "frac": float(frac),
                "n_clu_R": n_R, "n_clu_L": n_L,
                "ari_LR": float(ari_lr), "ari_fromR": float(ari_from_r),
                "diag": diag_mean,
                "silhouette": float(silh),
                "ch": float(ch), "db": float(db),
            })
        return pd.DataFrame(rows)

    tag = f"n{n_points}_{frac_log2_min}_{frac_log2_max}".replace("-", "m").replace(".", "p")
    return cached_parquet(f"{dataset}_OL_lr_sweep_{tag}", _compute, force=force)


def get_ol_lr_cluster_match(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    frac_thre: float = 0.19, force: bool = False,
):
    """L↔R cluster confusion matrix + Hungarian assignment at fixed frac_thre.

    L instance names are flipped `_L`→`_R` before merging on `instance`,
    mirroring the legacy approach. Returns a dict
    `{confusion, row_ind, col_ind, n_clu_L, n_clu_R, tick_labels}`.
    """
    from utils.cache import cached_pickle
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    def _compute():
        cl_R = get_ol_clusters(dataset=dataset, data_dir=data_dir,
                               side_char="r", frac_thre=frac_thre)
        cl_L = get_ol_clusters(dataset=dataset, data_dir=data_dir,
                               side_char="l", frac_thre=frac_thre)
        cl_L = cl_L.copy()
        cl_L["instance"] = [_swap_lr_suffix(s) for s in cl_L["instance"].values]
        merged = cl_R.merge(cl_L, on="instance", how="inner", suffixes=("_R", "_L"))
        conf = confusion_matrix(merged["cluster_L"].values, merged["cluster_R"].values)
        row_ind, col_ind = linear_sum_assignment(-conf)
        n_L = int(merged["cluster_L"].max())
        n_R = int(merged["cluster_R"].max())
        n = max(n_L, n_R)
        tick_labels = [f"c{i}" for i in range(1, n + 1)]
        return {
            "confusion": conf,
            "row_ind": row_ind, "col_ind": col_ind,
            "n_clu_L": n_L, "n_clu_R": n_R,
            "tick_labels": tick_labels,
        }

    tag = f"{frac_thre:.3f}".replace(".", "p")
    return cached_pickle(f"{dataset}_OL_lr_cluster_match_{tag}", _compute, force=force)


# === 5. Polarity experimental comparisons ===


def _split_polarity_types(cell) -> list[str]:
    """Split a 'Neuron type' cell from VCBNexperiments, dropping parts whose
    own name ends in '?' (e.g. `ER4d, ER3?` → `['ER4d']`)."""
    if pd.isna(cell):
        return []
    parts = [p.strip() for p in str(cell).split(",")]
    return [p for p in parts if p and not p.endswith("?")]


_CB_POLARITY_PREFIXES = ["DN", "PLP", "PVLP", "TuBu", "ER4d", "LNd_b"]


def _cb_polarity_cat(group_label: str) -> int:
    """Sort key: category rank for CB polarity bar ordering."""
    for i, prefix in enumerate(_CB_POLARITY_PREFIXES):
        if group_label.startswith(prefix):
            return i
    return len(_CB_POLARITY_PREFIXES)


def _abbreviate_group_label(parts: list[str]) -> str:
    """Abbreviate a group to 'PrefixFirst-Last' when all types share a prefix
    and have consecutive numeric or alphabetic suffixes. Falls back to
    comma-joining otherwise.

    Examples:
    - ['TuBu02','TuBu03','TuBu04','TuBu05'] -> 'TuBu02-05'
    - ['DNg02_a','DNg02_b',...,'DNg02_f']   -> 'DNg02_a-f'
    - ['PVLP008_a1',...,'PVLP008_a4']       -> 'PVLP008_a1-4'
    """
    if len(parts) <= 1:
        return parts[0] if parts else ""
    # Numeric suffix with letter-only prefix: TuBu02-05
    matches = [re.fullmatch(r"([A-Za-z_]+)(\d+)", p) for p in parts]
    if all(matches):
        prefixes = [m.group(1) for m in matches]
        suffixes = [m.group(2) for m in matches]
        if len(set(prefixes)) == 1:
            nums = [int(s) for s in suffixes]
            if nums == list(range(nums[0], nums[0] + len(nums))):
                return f"{prefixes[0]}{suffixes[0]}-{suffixes[-1]}"
    # Alphabetic single-char suffix: DNg02_a through DNg02_f
    matches_alpha = [re.fullmatch(r"(.+)_([a-z])$", p) for p in parts]
    if all(matches_alpha):
        prefixes = [m.group(1) for m in matches_alpha]
        suffixes = [m.group(2) for m in matches_alpha]
        if len(set(prefixes)) == 1:
            codes = [ord(c) for c in suffixes]
            if codes == list(range(codes[0], codes[0] + len(codes))):
                return f"{prefixes[0]}_{suffixes[0]}-{suffixes[-1]}"
    # Numeric suffix with any prefix: PVLP008_a1 through PVLP008_a4
    matches_any = [re.fullmatch(r"(.+?)(\d+)$", p) for p in parts]
    if all(matches_any):
        prefixes = [m.group(1) for m in matches_any]
        suffixes = [m.group(2) for m in matches_any]
        if len(set(prefixes)) == 1:
            nums = [int(s) for s in suffixes]
            if nums == list(range(nums[0], nums[0] + len(nums))):
                return f"{prefixes[0]}{suffixes[0]}-{suffixes[-1]}"
    return ", ".join(parts)


_POLARITY_INSTANCE_ALIASES: dict[str, str] = {}


def get_polarity_experiments(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    source_file: str = "fly_literature_experiments.xlsx",
    sheet: str = "input sensitivity",
    force: bool = False,
) -> pd.DataFrame:
    """Curated ON/OFF polarity labels from the fly-literature experiments
    spreadsheet.

    Keeps rows whose `polarity` is exactly `'ON'` or `'OFF'` (drops NaN / any
    value with `'?'`). Splits comma-separated multi-type cells into one row
    per type; drops parts that themselves end in `'?'`. Adds `instance` =
    `type + '_R'` (or `'_L'` if `side_char='l'`) and `in_ol` based on
    `get_ol_meta` membership. Applies `_POLARITY_INSTANCE_ALIASES` for types
    whose sheet name maps to a sub-variant in the dataset's meta.
    """
    def _compute():
        suffix = "_R" if side_char == "r" else "_L"
        df = pd.read_excel(PARAMS_DIR / source_file, sheet_name=sheet)
        df = df[df["polarity"].isin(["ON", "OFF"])].copy()
        df["_parts"] = df["Neuron type"].apply(_split_polarity_types)
        # group_label captures the original multi-type entry before exploding,
        # abbreviated when types share a prefix with consecutive numeric suffixes
        df["group_label"] = df["_parts"].apply(_abbreviate_group_label)
        df = df.explode("_parts").dropna(subset=["_parts"])
        df = df.rename(columns={"_parts": "type"})
        df["instance"] = df["type"].astype(str) + suffix
        df["instance"] = df["instance"].replace(_POLARITY_INSTANCE_ALIASES)
        ol_instances = set(
            get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
            .instance.dropna().unique()
        )
        df["in_ol"] = df["instance"].isin(ol_instances)
        return df[["type", "polarity", "instance", "in_ol", "group_label"]].reset_index(drop=True)

    return cached_parquet(
        f"{dataset}_OL_{side_char}_polarity_labels", _compute, force=force,
    )


def get_ol_polarity_comparison(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-OL-instance effective-weight feature vectors from visual-input seeds
    plus predicted polarity (`'ON' if L1 > L2 else 'OFF'`) and match flag.

    Feature vectors are normalised per row (rows that sum to zero stay zero).
    Seed inputs are every instance with `hitting_time == 0` in
    `get_ol_flow_type` — legacy behaviour (includes L1/L2/L3, R7/R7p/R7y,
    R8/R8p/R8y, R7d/R8d, HBeyelet). Uses OL `lat_flow_sum` (fast, OL-only).
    """
    def _compute():
        labels = get_polarity_experiments(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        labels = labels[labels["in_ol"]].reset_index(drop=True)

        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_sum(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        flow_type = get_ol_flow_type(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        in_instances = flow_type[flow_type.hitting_time == 0].instance.unique()
        inidx = ol_meta[ol_meta.instance.isin(in_instances)].idx.values
        outidx = ol_meta[ol_meta.instance.isin(labels["instance"].values)].idx.values

        idx_to_instance = dict(zip(ol_meta.idx, ol_meta.instance))
        feat = _compute_feature_vectors(
            stepsn, inidx, outidx, idx_to_instance, idx_to_instance,
        )
        feat = feat.loc[feat.index.isin(labels["instance"].values)]
        row_sum = feat.sum(axis=1)
        feat_norm = feat.div(row_sum.where(row_sum > 0, 1), axis=0)
        feat_norm = feat_norm.reset_index(names="instance")

        merged = labels.merge(feat_norm, on="instance", how="inner")
        merged = merged.merge(
            ol_meta[["instance", "main_groups"]].drop_duplicates("instance"),
            on="instance", how="left",
        )
        suffix = "_R" if side_char == "r" else "_L"
        l1_col, l2_col = f"L1{suffix}", f"L2{suffix}"
        l1 = merged[l1_col] if l1_col in merged.columns else 0.0
        l2 = merged[l2_col] if l2_col in merged.columns else 0.0
        merged["predicted"] = np.where(l1 > l2, "ON", "OFF")
        merged["match"] = merged["predicted"] == merged["polarity"]
        # Drop visual-input seeds themselves — they sit on the input side of the
        # flow so their own feature vectors collapse to self-loops, and L1/L2
        # are the rule's own predictors (comparing them is tautological).
        merged = merged[merged["main_groups"] != "visual input"].reset_index(drop=True)
        return merged

    return cached_parquet(
        f"{dataset}_OL_{side_char}_polarity_comparison", _compute, force=force,
    )


def get_cb_polarity_comparison(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-CB-instance effective-weight feature vectors from visual-input
    seeds plus predicted polarity (`'ON' if L1 > L2 else 'OFF'`) and match
    flag.

    Uses the labels rows where `in_ol == False` (24 CB-only types) and
    propagates on the full `lat_flow_sum.npz` (not OL-subset). Seeds are the
    side's visual-input instances from the full meta.
    """
    def _compute():
        labels = get_polarity_experiments(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        labels = labels[~labels["in_ol"]].reset_index(drop=True)

        meta = get_meta(dataset=dataset, data_dir=data_dir)
        stepsn = sp.load_npz(
            data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz"
        )
        side_word = _SIDE_WORD[side_char]
        in_instances = meta[
            (meta.main_groups == "visual input") & (meta.side == side_word)
        ].instance.unique()
        inidx = meta[meta.instance.isin(in_instances)].idx.values
        outidx = meta[meta.instance.isin(labels["instance"].values)].idx.values
        idx_to_instance = dict(zip(meta.idx, meta.instance))

        # Step 1: normalize per output body (compute input-weight ratios per body)
        raw = pd.DataFrame(
            stepsn[:, outidx][inidx, :].toarray(),
            index=[idx_to_instance[k] for k in inidx],
            columns=outidx,
        )
        raw = raw.groupby(raw.index).sum()
        raw.columns = [idx_to_instance[k] for k in raw.columns]
        raw = raw.T  # rows=output bodies, cols=input instances
        body_sum = raw.sum(axis=1)
        raw_norm = raw.div(body_sum.where(body_sum > 0, 1), axis=0)

        # Step 2: mean over bodies of the same cell type
        type_norm = raw_norm.groupby(level=0).mean()
        type_norm = type_norm.reset_index(names="instance")

        merged = labels.merge(type_norm, on="instance", how="inner")
        merged = merged.merge(
            meta[["instance", "main_groups"]].drop_duplicates("instance"),
            on="instance", how="left",
        )

        # Step 3: mean over cell types sharing the same spreadsheet entry
        feature_cols = [c for c in type_norm.columns if c != "instance"]
        grouped = (
            merged.groupby(["group_label", "polarity"])[feature_cols]
            .mean()
            .reset_index()
        )
        suffix = "_R" if side_char == "r" else "_L"
        grouped["instance"] = grouped["group_label"] + suffix
        grouped["main_groups"] = (
            merged[["group_label", "main_groups"]]
            .drop_duplicates("group_label")
            .set_index("group_label")["main_groups"]
            .reindex(grouped["group_label"].values)
            .values
        )
        l1_col, l2_col = f"L1{suffix}", f"L2{suffix}"
        l1 = grouped[l1_col] if l1_col in grouped.columns else 0.0
        l2 = grouped[l2_col] if l2_col in grouped.columns else 0.0
        grouped["predicted"] = np.where(l1 > l2, "ON", "OFF")
        grouped["match"] = grouped["predicted"] == grouped["polarity"]
        grouped["_cat"] = grouped["group_label"].apply(_cb_polarity_cat)
        grouped = grouped.sort_values(["_cat", "group_label"]).drop(columns="_cat").reset_index(drop=True)
        return grouped

    return cached_parquet(
        f"{dataset}_{side_char}_cb_polarity_comparison", _compute, force=force,
    )


# === 6. Propagated RFs ===


_VISUAL_INPUT_STEMS = ["L1", "L2", "L3", "R7", "R8"]


def _compute_rf(meta_ol, stepsn, inidx, outidx, *, idx_to_coords=None, idx_to_root=None):
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
    df = df[(df.index != "nan") & (~df.index.isnull())]
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


def _compute_rf_params(meta_ol, stepsn, in_instances, out_instances):
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
            df = _compute_rf(
                meta_ol, stepsn, inidx, outidx_arr[i:i + 1],
                idx_to_coords=idx_to_coords, idx_to_root=idx_to_root,
            )
            if df.empty:
                continue
            tot_weight = df["effective weight"].sum()
            if tot_weight <= 0:
                continue
            amp = float(df["effective weight"].max())
            params_ij, rf_fitted = fit_rf_gaussian(df)
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


# === 7. RF size experimental comparisons ===


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
    """`_ol_instances` minus `main_groups == 'visual input'` (R7/R8/HBeyelet).

    Matches the filter in `_ensure_ol_sidecars` + the non-visual-input filter
    in `get_rf_raw_ol`.
    """
    meta = get_meta(dataset=dataset, data_dir=data_dir)
    return _ol_instances(dataset, data_dir, side_char) - set(
        meta[meta.main_groups == "visual input"].instance.unique()
    )


def get_input_rf_raw_ol(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    source_stem: str | None = None, force: bool = False,
) -> pd.DataFrame:
    """Per-body direct input RF fits, computed from raw neuprint synapse data.

    Reads `cache/data/{source_stem}_input_syn_per_col.parquet` (default
    `{dataset}_OL_{side_char}`), restricts to the same instance set as
    `get_rf_raw_ol` (OL-side non-visual-input, excluding R1-R6), fits a 2D
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
    comparisons even though they aren't fitted as outputs in
    `get_rf_raw_ol`.
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


def get_cb_connectivity(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Long-format connectivity for pre-neurons in the VCBN set (CB side).

    Mirrors `get_ol_connectivity` but filters the full `prop.npz` to rows whose
    `instance_pre` is a VCBN type (`get_vcbn_types`). Adds per-instance median
    hitting times and the `frac` = weight / Σ(weight) per pre-neuron.
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        vcbn = get_vcbn_types(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            vic_thre=vic_thre,
        )
        cb_instances = set(vcbn.instance.unique())

        prop = sp.load_npz(data_dir / f"{dataset}_prop.npz").tocoo()
        conn = pd.DataFrame({"idx_pre": prop.row, "idx_post": prop.col, "weight": prop.data})

        meta_small = meta[["idx", "bodyId", "instance", "region", "main_groups", "sign"]]
        conn = conn.merge(meta_small, left_on="idx_pre", right_on="idx", how="left").rename(
            columns={
                "bodyId": "bodyId_pre", "instance": "instance_pre", "region": "region_pre",
                "main_groups": "main_groups_pre", "sign": "sign_pre",
            }
        ).drop(columns="idx")
        conn = conn[conn["instance_pre"].isin(cb_instances)]

        conn = conn.merge(meta_small, left_on="idx_post", right_on="idx", how="left").rename(
            columns={
                "bodyId": "bodyId_post", "instance": "instance_post", "region": "region_post",
                "main_groups": "main_groups_post", "sign": "sign_post",
            }
        ).drop(columns="idx")

        flow_type = get_flow_per_group(dataset=dataset, data_dir=data_dir, side_char=side_char)
        conn = conn.merge(
            flow_type, left_on="instance_pre", right_on="instance", how="left"
        ).rename(columns={"hitting_time": "hitting_time_pre"}).drop(columns="instance")
        conn = conn.merge(
            flow_type, left_on="instance_post", right_on="instance", how="left"
        ).rename(columns={"hitting_time": "hitting_time_post"}).drop(columns="instance")

        conn = conn[~pd.isna(conn["hitting_time_pre"]) & ~pd.isna(conn["hitting_time_post"])]
        conn["hitting_time_diff"] = conn["hitting_time_post"] - conn["hitting_time_pre"]
        conn["frac"] = conn["weight"] / conn.groupby("idx_pre")["weight"].transform("sum")
        return conn.reset_index(drop=True)

    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_cb_{side_char}_connectivity_{vtag}", _compute, force=force,
    )


def get_cb_directedness(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, hit_diff_thre: float = 0.5, force: bool = False,
) -> pd.DataFrame:
    """Per-neuron feedforward / feedback / lateral fractions for VCBN pre connections.

    Legacy nb 8 cell 26. `main_groups_pre` is overridden to `'nonOL'` to match
    legacy cell 27.
    """
    def _compute():
        conn_cb = get_cb_connectivity(
            dataset=dataset, data_dir=data_dir, side_char=side_char, vic_thre=vic_thre,
        )
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        dir_df = _compute_directedness(conn_cb, hit_diff_thre=hit_diff_thre)
        dir_df = dir_df.merge(
            meta[["instance", "sign"]].drop_duplicates(),
            left_on="instance_pre", right_on="instance", how="left",
        ).rename(columns={"sign": "sign_pre"}).drop(columns="instance")
        dir_df["main_groups_pre"] = "nonOL"
        return dir_df

    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    tag = f"thre{hit_diff_thre}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_cb_{side_char}_directedness_{vtag}_{tag}", _compute, force=force,
    )


def get_cb_type_directedness(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, hit_diff_thre: float = 0.5, force: bool = False,
) -> pd.DataFrame:
    """Per-VCBN-instance (median) directedness, renormalised to sum=1."""
    def _compute():
        d = get_cb_directedness(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            vic_thre=vic_thre, hit_diff_thre=hit_diff_thre,
        )
        t = (
            d.groupby(["instance_pre", "cell_type_pre", "main_groups_pre", "sign_pre"])
            [["frac_ff", "frac_fb", "frac_la"]].median().reset_index()
        )
        t[["frac_ff", "frac_fb", "frac_la"]] = (
            t[["frac_ff", "frac_fb", "frac_la"]]
            / t[["frac_ff", "frac_fb", "frac_la"]].sum(1).values[:, np.newaxis]
        )
        return t.sort_values("main_groups_pre").reset_index(drop=True)

    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    tag = f"thre{hit_diff_thre}".replace(".", "p")
    return cached_parquet(
        f"{dataset}_cb_{side_char}_type_directedness_{vtag}_{tag}", _compute, force=force,
    )


def get_cb_features(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Visual-input → VCBN feature vectors on one side.

    Rows are VCBN instance names (from `get_vcbn_types(side_char)`); cols are
    visual-input instances on that side. Computed on the full
    `lat_flow_sum.npz` for the side. Heavy compute on first run.
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
        side_word = _SIDE_WORD[side_char]
        in_inst = meta[
            (meta.main_groups == "visual input") & (meta.side == side_word)
        ].instance.unique()
        out_inst = vcbn["instance"].values
        inidx = meta[meta.instance.isin(in_inst)].idx.values
        outidx = meta[meta.instance.isin(out_inst)].idx.values
        idx_to_instance = dict(zip(meta.idx, meta.instance))
        feat = _compute_feature_vectors(
            stepsn, inidx, outidx, idx_to_instance, idx_to_instance,
        )
        # Drop VCBN rows whose leading-seed column is NaN (no propagation)
        seed_col = f"L1_R" if side_char == "r" else f"L1_L"
        if seed_col in feat.columns:
            feat = feat[~feat[seed_col].isna()]
        feat.index.name = "instance"
        return feat.reset_index()

    tag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    df = cached_parquet(
        f"{dataset}_cb_{side_char}_features_{tag}", _compute, force=force,
    )
    return df.set_index("instance")


def get_cb_clusters(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.19, n_clusters: int | None = None,
    method: str = "ward", metric: str = "euclidean",
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Flat clusters of VCBN instances (cached).

    If `n_clusters` is given, uses `criterion='maxclust'` to force exactly that
    many clusters (useful when a distance cutoff skips the desired count due
    to ties). Otherwise uses `frac_thre * max_distance` as a distance cutoff.
    """
    import scipy.cluster.hierarchy as sch

    def _compute():
        feat = get_cb_features(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            vic_thre=vic_thre,
        )
        Z = _compute_hierarchical_clustering(feat, method=method, metric=metric)
        if n_clusters is not None:
            cluster_idx = sch.fcluster(Z, n_clusters, criterion="maxclust")
        else:
            thre = frac_thre * Z[:, 2].max()
            cluster_idx = sch.fcluster(Z, thre, criterion="distance")
        # Reverse cluster numbering so c1 corresponds to scipy's last leaf
        # (matches the convention used by get_cb_clusters_intra and the OL
        # cluster numbering). Without this, downstream plots would be mirror
        # images of the intra heatmap.
        n_clu = int(cluster_idx.max())
        cluster_idx = n_clu + 1 - cluster_idx
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        df = pd.DataFrame({"instance": feat.index.values, "cluster": cluster_idx})
        df = df.merge(
            meta.groupby("instance")["bodyId"].count().reset_index(name="count"),
            on="instance", how="left",
        )
        return df

    tag = f"n{n_clusters}" if n_clusters is not None else f"{frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_cb_{side_char}_clusters_{tag}_{vtag}",
        _compute, force=force,
    )


def get_cb_lr_sweep(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    n_points: int = 48, frac_log2_min: float = -5.0, frac_log2_max: float = -0.1,
    method: str = "ward", metric: str = "euclidean",
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Sweep CB clustering threshold on both sides and compare L/R.

    Each row: frac, n_clu_R, n_clu_L, ari_LR (cell-type-matched via
    stripping `_R`/`_L` suffix), ari_fromR (L clusters vs. centroid
    assignment from R).
    """
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import cdist
    from sklearn.metrics import adjusted_rand_score

    def _compute():
        feat_R = get_cb_features(
            dataset=dataset, data_dir=data_dir, side_char="r", vic_thre=vic_thre,
        )
        feat_L = get_cb_features(
            dataset=dataset, data_dir=data_dir, side_char="l", vic_thre=vic_thre,
        )

        Z_R = _compute_hierarchical_clustering(feat_R, method=method, metric=metric)
        Z_L = _compute_hierarchical_clustering(feat_L, method=method, metric=metric)

        frac_list = 2.0 ** np.linspace(frac_log2_min, frac_log2_max, n_points)
        rows = []
        for frac in frac_list:
            cl_R = sch.fcluster(Z_R, frac * Z_R[:, 2].max(), criterion="distance")
            cl_L = sch.fcluster(Z_L, frac * Z_L[:, 2].max(), criterion="distance")
            n_R = int(cl_R.max())
            n_L = int(cl_L.max())

            df_R = pd.DataFrame({"instance": feat_R.index.values, "cluster": cl_R})
            df_L = pd.DataFrame({"instance": feat_L.index.values, "cluster": cl_L})
            df_R["cell_type"] = [s[:-2] for s in df_R["instance"]]
            df_L["cell_type"] = [s[:-2] for s in df_L["instance"]]
            compare = df_R.merge(df_L, on="cell_type", how="inner", suffixes=("_R", "_L"))
            ari_lr = adjusted_rand_score(
                compare["cluster_R"].values, compare["cluster_L"].values,
            ) if len(compare) else float("nan")

            centroids_R = np.vstack([
                feat_R.values[cl_R == c].mean(axis=0) for c in range(1, n_R + 1)
            ])
            labels_L_from_R = cdist(
                feat_L.values, centroids_R, metric="euclidean",
            ).argmin(axis=1) + 1
            ari_from_r = adjusted_rand_score(cl_L, labels_L_from_R)

            rows.append({
                "frac": float(frac),
                "n_clu_R": n_R, "n_clu_L": n_L,
                "ari_LR": float(ari_lr), "ari_fromR": float(ari_from_r),
            })
        return pd.DataFrame(rows)

    tag = f"n{n_points}_{frac_log2_min}_{frac_log2_max}".replace("-", "m").replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_cb_lr_sweep_{tag}_{vtag}", _compute, force=force,
    )


def get_cb_lr_cluster_match(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    frac_thre: float = 0.19, n_clusters: int | None = None,
    vic_thre: float = 5e-4, force: bool = False,
):
    """L↔R CB-cluster confusion matrix + Hungarian assignment.

    If `n_clusters` is set, forces both sides to that many clusters via
    `criterion='maxclust'` (useful when the L-side distance cutoff skips the
    desired count due to ties). Otherwise uses `frac_thre` on both sides.

    Left instance names are flipped `_L`↔`_R` before merging on instance,
    mirroring the legacy approach. Returns
    `{confusion, row_ind, col_ind, n_clu_L, n_clu_R, tick_labels}`.
    """
    from utils.cache import cached_pickle
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    def _compute():
        kw = dict(n_clusters=n_clusters) if n_clusters is not None else dict(frac_thre=frac_thre)
        cl_R = get_cb_clusters(
            dataset=dataset, data_dir=data_dir, side_char="r",
            vic_thre=vic_thre, **kw,
        )
        cl_L = get_cb_clusters(
            dataset=dataset, data_dir=data_dir, side_char="l",
            vic_thre=vic_thre, **kw,
        )
        cl_L = cl_L.copy()
        cl_L["instance"] = [_swap_lr_suffix(s) for s in cl_L["instance"].values]
        merged = cl_R.merge(cl_L, on="instance", how="inner", suffixes=("_R", "_L"))
        conf = confusion_matrix(merged["cluster_L"].values, merged["cluster_R"].values)
        row_ind, col_ind = linear_sum_assignment(-conf)
        n_L = int(merged["cluster_L"].max())
        n_R = int(merged["cluster_R"].max())
        n = max(n_L, n_R)
        tick_labels = [f"d{i}" for i in range(1, n + 1)]
        return {
            "confusion": conf,
            "row_ind": row_ind, "col_ind": col_ind,
            "n_clu_L": n_L, "n_clu_R": n_R,
            "tick_labels": tick_labels,
        }

    tag = f"n{n_clusters}" if n_clusters is not None else f"{frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_pickle(
        f"{dataset}_cb_lr_cluster_match_{tag}_{vtag}", _compute, force=force,
    )


def get_ol_in_cb_participation(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    cb_frac_thre: float = 0.19,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Per-neuron participation (OL outputs + CB VCBN) in each CB cluster.

    Mirrors `_compute_participation` but restricts neurons to OL outputs on
    the chosen side + CB VCBN instances, and the target clusters are CB
    clusters. Legacy nb 8 cell 23.
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        stepsn = sp.load_npz(
            data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz"
        )
        cb_clu = get_cb_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            frac_thre=cb_frac_thre, vic_thre=vic_thre,
        )
        side_word = _SIDE_WORD[side_char]
        ol_out_mask = (meta.region == f"{side_word} OL") & (meta.main_groups == "OL output")
        cb_mask = meta.instance.isin(cb_clu.instance.values)
        meta_sub = meta[ol_out_mask | cb_mask].copy()

        idx_sub = np.sort(meta_sub.index.values)
        meta_sub = meta_sub.set_index("idx").loc[idx_sub].reset_index(
            names="idx_full"
        ).reset_index(names="idx")
        stepsn_sub = stepsn[idx_sub][:, idx_sub]

        df = _compute_participation(meta_sub, stepsn_sub, cb_clu)
        return df

    ftag = f"{cb_frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_{side_char}_ol_in_cb_participation_{ftag}_{vtag}",
        _compute, force=force,
    )


def get_ol_in_cb_type_participation(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    cb_frac_thre: float = 0.19,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Per-instance median participation in CB clusters + max / idxmax /
    n_clusters summary + instance-level region / main_groups / sign /
    hitting_time."""
    def _compute():
        part = get_ol_in_cb_participation(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            cb_frac_thre=cb_frac_thre, vic_thre=vic_thre,
        )
        cluster_names = list(part.columns[part.columns.str.contains(r"^c\d+$")])
        df = part.groupby("instance")[cluster_names].median().reset_index()
        vals = df[cluster_names].values
        df[cluster_names] = vals / vals.sum(1)[:, np.newaxis]
        df = df[~pd.isna(df[cluster_names[0]])]
        df["max"] = np.sort(df[cluster_names].values, axis=1)[:, -1:].sum(1)
        df["idxmax"] = df[cluster_names].idxmax(1)
        df["n_clusters"] = (df[cluster_names] >= 1 / len(cluster_names)).sum(1)
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        ht = get_flow_per_group(dataset=dataset, data_dir=data_dir, side_char=side_char)
        df = df.merge(ht, on="instance", how="left")
        df = df.merge(
            meta[["instance", "region", "main_groups", "sign"]].drop_duplicates("instance"),
            on="instance", how="left",
        )
        return df

    ftag = f"{cb_frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_{side_char}_ol_in_cb_type_participation_{ftag}_{vtag}",
        _compute, force=force,
    )


def get_cb_cluster_tbars_per_roi(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    cb_frac_thre: float = 0.19,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """Per-(CB-cluster, ROI) tbar count wide df (clusters × ROI) for the
    `cb_clusters_rois` heatmap. Merges `tbar_VIC_forJudith.pkl` with CB
    cluster assignments via bodyId → instance → cluster. Legacy nb 8 c 103."""
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        cb_clu = get_cb_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            frac_thre=cb_frac_thre, vic_thre=vic_thre,
        )
        tbar = pd.read_pickle(data_dir / "tbar_VIC_forJudith.pkl")
        tbar = tbar.merge(
            meta[["bodyId", "instance"]], on="bodyId", how="left",
        ).merge(
            cb_clu[["instance", "cluster"]], on="instance", how="left",
        )
        tbar = tbar[~tbar["cluster"].isna()].copy()
        tbar["cluster"] = tbar["cluster"].astype(int)
        counts = tbar.groupby(["cluster", "roi"])["bodyId"].count().unstack(fill_value=0)
        return counts

    from utils.cache import cached_pickle
    ftag = f"{cb_frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_pickle(
        f"{dataset}_{side_char}_cb_cluster_tbars_per_roi_{ftag}_{vtag}",
        _compute, force=force,
    )


def get_cb_clusters_intra(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    frac_thre: float = 0.19,
    vic_thre: float = 5e-4, force: bool = False,
):
    """CB-cluster × CB-cluster connectivity matrix (output-normalised).

    Mirrors `get_ol_clusters_intra` but uses full `prop.npz` and the CB
    clustering. Returns `{'matrix': ndarray, 'total_weight': float,
    'n_clu': int}`.
    """
    from utils.cache import cached_pickle
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        prop = get_full_prop(dataset=dataset, data_dir=data_dir)
        cb_clu = get_cb_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            frac_thre=frac_thre, vic_thre=vic_thre,
        )
        n = int(cb_clu["cluster"].max())
        outidx = meta[meta.instance.isin(cb_clu.instance.values)].idx.values
        instance_to_cluster = dict(zip(cb_clu["instance"], cb_clu["cluster"]))
        idx_to_instance = dict(zip(meta.idx, meta.instance))
        idx_to_cluster = {
            i: instance_to_cluster[inst]
            for i, inst in idx_to_instance.items()
            if inst in instance_to_cluster
        }
        m = result_summary(
            prop, outidx, outidx, idx_to_cluster, idx_to_cluster,
            combining_method="sum", display_output=False,
        )
        m.index = [int(i) for i in m.index]
        m.columns = [int(i) for i in m.columns]
        m = m.reindex(index=range(1, n + 1), columns=range(1, n + 1)).fillna(0).values
        total = m.sum()
        m = m / m.sum(axis=1)[:, np.newaxis]
        return {"matrix": m, "total_weight": float(total), "n_clu": n}

    ftag = f"{frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_pickle(
        f"{dataset}_cb_{side_char}_clusters_intra_{ftag}_{vtag}",
        _compute, force=force,
    )


def get_ol_cb_cluster_connectivity(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    ol_frac_thre: float = 0.19, cb_frac_thre: float = 0.19,
    vic_thre: float = 5e-4, force: bool = False,
):
    """OL-cluster × CB-cluster connectivity via full `prop.npz`.

    Aggregates per-body effective weights by cluster membership on both
    sides, returning a dense `(n_ol_clu, n_cb_clu)` DataFrame indexed/colored
    by cluster id. Legacy nb 8 cell 22.
    """
    from utils.cache import cached_pickle
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        prop = get_full_prop(dataset=dataset, data_dir=data_dir)
        ol_clu = get_ol_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            frac_thre=ol_frac_thre,
        )
        cb_clu = get_cb_clusters(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            frac_thre=cb_frac_thre, vic_thre=vic_thre,
        )
        n_ol = int(ol_clu["cluster"].max())
        n_cb = int(cb_clu["cluster"].max())

        inidx = meta[meta.instance.isin(ol_clu.instance.values)].idx.values
        outidx = meta[meta.instance.isin(cb_clu.instance.values)].idx.values

        idx_to_instance = dict(zip(meta.idx, meta.instance))
        instance_to_ol = dict(zip(ol_clu["instance"], ol_clu["cluster"]))
        instance_to_cb = dict(zip(cb_clu["instance"], cb_clu["cluster"]))
        idx_to_ol = {
            i: instance_to_ol[inst]
            for i, inst in idx_to_instance.items()
            if inst in instance_to_ol
        }
        idx_to_cb = {
            i: instance_to_cb[inst]
            for i, inst in idx_to_instance.items()
            if inst in instance_to_cb
        }
        conn = result_summary(
            prop, inidx, outidx, idx_to_ol, idx_to_cb,
            combining_method="sum", display_output=False,
        )
        conn.index = [int(i) for i in conn.index]
        conn.columns = [int(i) for i in conn.columns]
        conn = conn.reindex(index=range(1, n_ol + 1), columns=range(1, n_cb + 1)).fillna(0)
        return conn

    otag = f"{ol_frac_thre:.3f}".replace(".", "p")
    ctag = f"{cb_frac_thre:.3f}".replace(".", "p")
    vtag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_pickle(
        f"{dataset}_{side_char}_ol_cb_cluster_conn_{otag}_{ctag}_{vtag}",
        _compute, force=force,
    )


def get_cb_layer_lr_homologue(
    *, dataset: str = DATASET, data_dir=DATA_DIR, force: bool = False,
) -> pd.DataFrame:
    """Cell-type-matched right + left layer (hitting_time) comparison.

    Per side: median hitting_time per instance, strip `_R`/`_L` suffix to
    cell_type, keep the variant with the **min** hitting_time (earliest
    layer). Inner-merge on cell_type gives one row per bilateral homologue.
    Legacy nb 9 cells 6 + 20.
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        def _side(side_char):
            per_inst = get_flow_per_group(
                dataset=dataset, data_dir=data_dir, side_char=side_char,
            ).copy()
            per_inst["cell_type"] = [
                s[:-2] if s[-2:] in ("_R", "_L") else s
                for s in per_inst["instance"]
            ]
            idx = per_inst.groupby("cell_type")["hitting_time"].idxmin()
            return per_inst.loc[idx].reset_index(drop=True)

        r = _side("r").rename(columns={"instance": "instance_r", "hitting_time": "hitting_time_r"})
        l = _side("l").rename(columns={"instance": "instance_l", "hitting_time": "hitting_time_l"})
        df = r.merge(l, on="cell_type", how="inner")
        df = df.merge(
            meta[["cell_type", "region", "main_groups"]].drop_duplicates("cell_type"),
            on="cell_type", how="left",
        )
        return df

    return cached_parquet(
        f"{dataset}_cb_layer_lr_homologue", _compute, force=force,
    )


def get_cb_layer_lr(
    *, dataset: str = DATASET, data_dir=DATA_DIR, force: bool = False,
) -> pd.DataFrame:
    """Per-instance right + left median hitting_time joined with region + main_groups."""
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        ht_r = get_flow_per_group(
            dataset=dataset, data_dir=data_dir, side_char="r",
        ).rename(columns={"hitting_time": "hitting_time_r"})
        ht_l = get_flow_per_group(
            dataset=dataset, data_dir=data_dir, side_char="l",
        ).rename(columns={"hitting_time": "hitting_time_l"})
        df = ht_r.merge(ht_l, on="instance", how="inner")
        df = df.merge(
            meta[["instance", "region", "main_groups"]].drop_duplicates("instance"),
            on="instance", how="left",
        )
        return df

    return cached_parquet(
        f"{dataset}_cb_layer_lr", _compute, force=force,
    )


def get_cb_paths_to_instance(
    instance: str, *, dataset: str = DATASET, data_dir=DATA_DIR,
    side_char: str = SIDE_CHAR, in_main_group: str = 'visual input',
    n_interm_max: int = 4, force: bool = False,
) -> dict:
    """Effective-weight paths from OL `in_main_group` → CB `instance` (full-brain
    `lat_flow_0`), for each path length 1..min(ceil(hit_diff), n_interm_max).
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        stepsn = sp.load_npz(data_dir / f"{dataset}_{side_char}_lat_flow_0.npz")
        flow_type = get_flow_per_group(dataset=dataset, data_dir=data_dir, side_char=side_char)
        idx_to_instance = dict(zip(meta.idx, meta.instance))
        side_word = _SIDE_WORD[side_char]
        in_instances = meta[
            (meta.main_groups == in_main_group) & (meta.region == f'{side_word} OL')
        ].instance.unique()
        inidx = meta[meta.instance.isin(in_instances)].idx.values
        outidx = meta[meta.instance == instance].idx.values
        hit_self = float(flow_type[flow_type.instance == instance].hitting_time.values[0])
        hit_min = float(flow_type[flow_type.instance.isin(in_instances)].hitting_time.min())
        hit_inst = hit_self - hit_min
        n_interm = min(int(np.ceil(hit_inst)), n_interm_max)
        all_paths = _grouped_paths(stepsn, inidx, outidx, n_interm, idx_to_instance)
        return {'paths': all_paths, 'hit_inst': hit_inst}

    tag = instance.replace('/', '_')
    return cached_pickle(
        f"{dataset}_{side_char}_cb_paths_to_{tag}", _compute, force=force,
    )


# === 10. CB RFs ===


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


def get_ol_to_cb_weights(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, r2_thre: float = 0.05, force: bool = False,
) -> pd.DataFrame:
    """Normalised effective-weight matrix OL-output × CB-type.

    Uses the full `lat_flow_sum.npz` with instance-level aggregation.
    Restricted to CB types with `r2 > r2_thre` (from `get_rf_type_cb`) and
    sorted by each side's median RF `size`. Column-normalised (each CB type
    column sums to 1 after taking absolute values).
    """
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        stepsn = sp.load_npz(
            data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz"
        )
        combined = get_rf_types_combined(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
            vic_thre=vic_thre,
        )
        side_word = _SIDE_WORD[side_char]
        ol_types = combined[
            (combined["region"] == f"{side_word} OL")
            & (combined["main_groups"] == "OL output")
            & (combined["r2"] > r2_thre)
        ].sort_values("size")["instance"].values
        cb_types = combined[
            (combined["region"] == "CB") & (combined["r2"] > r2_thre)
        ].sort_values("size")["instance"].values

        inidx = meta[
            (meta.region == f"{side_word} OL") & (meta.main_groups == "OL output")
        ].idx.values
        outidx = meta[meta.instance.isin(cb_types)].idx.values
        idx_to_instance = dict(zip(meta.idx, meta.instance))

        m = result_summary(
            stepsn, inidx, outidx,
            inidx_map=idx_to_instance, outidx_map=idx_to_instance,
            display_threshold=0, display_output=False,
        )
        m = m.div(np.abs(m).sum(axis=0), axis=1)
        # keep only rows/cols that actually appear in the matrix
        ol_rows = [i for i in ol_types if i in m.index]
        cb_cols = [c for c in cb_types if c in m.columns]
        return m.loc[ol_rows, cb_cols]

    from utils.cache import cached_pickle
    tag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_pickle(
        f"{dataset}_OL_{side_char}_ol_to_cb_weights_{tag}", _compute, force=force,
    )


def get_ol_roi_coverage(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    n_perm: int = 10000, cov0: float = 0.2, seed: int | None = None,
    force: bool = False,
):
    """Per-ROI retinotopic coverage + permutation p-value.

    For each ROI in `get_roi_syn_vic`, computes the synapse-weighted
    effective input per visual column from right-OL visual inputs, projects
    onto the 5 retinotopic sectors from `get_sector_map`, and tests
    uniformity (`cov0` per sector) via a permutation of per-column weights.

    Returns `{'summary': df, 'per_roi': {roi: coverage_df}}` (pickled).
    `summary` has columns `roi, n_neu, n_syn, sec1..sec5, pval`, sorted by
    synapse count descending. `per_roi[roi]` has columns `coords,
    effective weight` (normalised to sum=1).
    """
    from utils.cache import cached_pickle
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = sp.load_npz(
            data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz"
        )
        sectors = get_sector_map(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        syn = get_roi_syn_vic(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )

        side_word = _SIDE_WORD[side_char]
        in_inst = ol_meta[
            (ol_meta.main_groups == "visual input") & (ol_meta.side == side_word)
        ].instance.unique()
        inidx = meta[meta.instance.isin(in_inst)].idx.values

        idx_to_coords = dict(zip(meta.idx, meta.coords))
        idx_to_bodyId = dict(zip(meta.idx, meta.bodyId))

        sector_arr = sectors["sector"].values
        G = np.zeros((len(sectors), 5), dtype=int)
        for i in range(5):
            G[sector_arr == i + 1, i] = 1

        rng = np.random.default_rng(seed)
        rois = syn["roi"].unique()
        summary_cols = ["n_neu", "n_syn", "sec1", "sec2", "sec3", "sec4", "sec5", "pval"]
        summary = pd.DataFrame(0.0, index=rois, columns=summary_cols)
        per_roi = {}

        for roi in rois:
            syn_roi = syn[syn.roi == roi]
            outidx = meta[meta.bodyId.isin(syn_roi.bodyId.unique())].idx.values
            if len(outidx) == 0:
                continue
            summary.loc[roi, "n_neu"] = float(len(outidx))

            df = result_summary(
                stepsn, inidx, outidx,
                inidx_map=idx_to_coords, outidx_map=idx_to_bodyId,
                combining_method="sum", display_threshold=0, display_output=False,
            )
            df = df[(df.index != "nan") & (~df.index.isnull())]
            if df.empty:
                continue

            # weight by per-bodyId synapse count, sum across bodyIds
            syn_count = syn_roi.groupby("bodyId").size()
            syn_count.index = syn_count.index.astype(str)
            common = [c for c in df.columns if c in syn_count.index]
            if not common:
                continue
            weighted = df[common] * syn_count.loc[common].values
            weights = weighted.sum(axis=1).to_frame("effective weight")
            total = float(weights["effective weight"].sum())
            if total <= 0:
                continue
            weights["effective weight"] = weights["effective weight"] / total
            summary.loc[roi, "n_syn"] = float(syn_count.loc[common].sum())
            per_roi[roi] = weights.reset_index().rename(columns={"index": "coords"})

            w = (
                weights.reindex(sectors["coords"].values)
                .fillna(0)["effective weight"].values
            )
            per_sector = w @ G
            summary.loc[roi, ["sec1", "sec2", "sec3", "sec4", "sec5"]] = per_sector

            # permutation: argsort of random matrix = row-wise permutation
            rand = rng.random((n_perm, len(w)))
            perm_idx = rand.argsort(axis=1)
            T_perm = ((w[perm_idx] @ G - cov0) ** 2).sum(axis=1)
            T_obs = float(((per_sector - cov0) ** 2).sum())
            summary.loc[roi, "pval"] = (np.sum(T_perm >= T_obs) + 1) / (n_perm + 1)

        summary = (
            summary.sort_values("n_syn", ascending=False)
            .reset_index().rename(columns={"index": "roi"})
        )
        return {"summary": summary, "per_roi": per_roi}

    tag = f"n{n_perm}_cov{cov0:.2f}".replace(".", "p")
    if seed is not None:
        tag += f"_seed{seed}"
    return cached_pickle(
        f"{dataset}_OL_{side_char}_roi_coverage_{tag}", _compute, force=force,
    )


# === 11. Model comparisons ===
