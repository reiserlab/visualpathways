"""OL↔CB and binocular vicinity (VIC) cached data.

Hosts visual-information-content (VIC) loaders that measure how much of a CB
neuron's input traces back to the OL. Covers:

- OL → CB VIC: `get_ol_cb_vic_raw`, `get_ol_cb_vic_type`, `get_ol_type_vic`,
  `get_vcbn_types`
- Binocular / left-right CB VIC: `get_cb_vic_binocular`,
  `get_cb_vic_binocular_type`, `get_cb_vic_lr_homologue`
- Per-ROI synaptic VIC: `get_roi_syn_vic`

Moved out of `setup_data/_preprocessing.py` in Phase E.3.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.cache import cached_parquet
from utils.config import DATA_DIR, DATASET, SIDE_CHAR
from utils.core_data import (
    _SIDE_WORD,
    get_meta,
    get_ol_meta,
    is_cb_neuron,
)
from utils.ol_data import get_ol_stepsn_sum


def get_ol_type_vic(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-OL-instance median VIC: summed effective weight from visual inputs
    via `lat_flow_sum`, aggregated first per bodyId then median per instance.
    """
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_sum(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        side_word = _SIDE_WORD[side_char]
        inidx = ol_meta[
            (ol_meta.main_groups == "visual input") & (ol_meta.side == side_word)
        ].idx.values
        outidx = ol_meta.idx.values
        idx_to_zero = {i: 0 for i in ol_meta.idx}
        idx_to_bid = dict(zip(ol_meta.idx, ol_meta.bodyId))

        vic = result_summary(
            stepsn, inidx, outidx,
            inidx_map=idx_to_zero, outidx_map=idx_to_bid,
            display_threshold=0, display_output=False,
        )
        vic = vic.T
        vic.columns = ["VIC"]
        vic = vic.reset_index(names="bodyId")
        vic["bodyId"] = vic["bodyId"].astype(int)
        vic = vic.merge(
            ol_meta[["bodyId", "instance"]].drop_duplicates(),
            on="bodyId", how="left",
        )
        return vic.groupby("instance")["VIC"].median().reset_index()

    return cached_parquet(
        f"{dataset}_OL_{side_char}_type_vic", _compute, force=force,
    )


def get_ol_cb_vic_raw(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-body summed effective input weight from visual inputs (VIC) across
    OL outputs + CB neurons. Uses the full `lat_flow_sum.npz`.

    Returns a long df with columns `bodyId, VIC, region, instance,
    main_groups`. Matches legacy `computing_functions.compute_ol_cb_vic`
    followed by the instance/main_groups merge from nb 8 cell 7.
    """
    from connectome_interpreter.compress_paths import result_summary

    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        stepsn = sp.load_npz(
            data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz"
        )
        side_word = _SIDE_WORD[side_char]
        idx_to_zero = {i: 0 for i in meta.index}
        idx_to_bid = dict(zip(meta.index, meta.bodyId))

        inidx = meta[
            (meta.main_groups == "visual input") & (meta.side == side_word)
        ].idx.values

        # OL outputs on this side
        outidx_ol = meta[
            (meta.region == f"{side_word} OL") & (meta.main_groups == "OL output")
        ].index.values
        vic_ol = result_summary(
            stepsn, inidx, outidx_ol,
            inidx_map=idx_to_zero, outidx_map=idx_to_bid,
            display_threshold=0, display_output=False,
        ).T.reset_index()
        vic_ol.columns = ["bodyId", "VIC"]
        vic_ol["region"] = f"{side_word} OL"

        # CB neurons (includes VCN / CB-input whose anatomical region is right/left OL)
        outidx_cb = meta[is_cb_neuron(meta)].index.values
        vic_cb = result_summary(
            stepsn, inidx, outidx_cb,
            inidx_map=idx_to_zero, outidx_map=idx_to_bid,
            display_threshold=0, display_output=False,
        ).T.reset_index()
        vic_cb.columns = ["bodyId", "VIC"]
        vic_cb["region"] = "CB"

        vic = pd.concat([vic_ol, vic_cb], axis=0).reset_index(drop=True)
        vic["bodyId"] = vic["bodyId"].astype(int)
        vic = vic.merge(
            meta[["bodyId", "instance", "main_groups"]].drop_duplicates(),
            on="bodyId", how="left",
        )
        vic = vic[~pd.isna(vic["instance"])]
        vic = vic[~vic["instance"].str.contains("unclear")]
        return vic.reset_index(drop=True)

    return cached_parquet(
        f"{dataset}_{side_char}_ol_cb_vic_raw", _compute, force=force,
    )


def get_ol_cb_vic_type(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-(instance, region) median VIC for OL-output + CB neurons."""
    def _compute():
        vic = get_ol_cb_vic_raw(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        return (
            vic.groupby(["instance", "region"])["VIC"].median()
               .reset_index()
               .sort_values("VIC", ascending=False)
               .reset_index(drop=True)
        )

    return cached_parquet(
        f"{dataset}_{side_char}_ol_cb_vic_type", _compute, force=force,
    )


def get_vcbn_types(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """CB instances with median VIC above `vic_thre` — the VCBN set."""
    def _compute():
        type_vic = get_ol_cb_vic_type(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        return type_vic[
            (type_vic["region"] == "CB") & (type_vic["VIC"] > vic_thre)
        ].reset_index(drop=True)

    tag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_{side_char}_vcbn_types_{tag}", _compute, force=force,
    )


def get_cb_vic_binocular(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    fillna: float = 1e-6, force: bool = False,
) -> pd.DataFrame:
    """Merge per-body right + left VIC for CB neurons.

    Uses `get_ol_cb_vic_raw` on both sides, restricted to `region == 'CB'`,
    merges on bodyId (outer), fills missing VIC with `fillna` so log-scale
    plots stay finite. Returns per-body df with `VIC_r, VIC_l`.
    """
    def _compute():
        right = get_ol_cb_vic_raw(
            dataset=dataset, data_dir=data_dir, side_char="r",
        )
        left = get_ol_cb_vic_raw(
            dataset=dataset, data_dir=data_dir, side_char="l",
        )
        r_cb = right[right["region"] == "CB"].rename(columns={"VIC": "VIC_r"})
        l_cb = left[left["region"] == "CB"].rename(columns={"VIC": "VIC_l"})
        merged = r_cb.merge(
            l_cb[["bodyId", "VIC_l"]], on="bodyId", how="outer",
        )
        merged["VIC_r"] = merged["VIC_r"].fillna(fillna)
        merged["VIC_l"] = merged["VIC_l"].fillna(fillna)
        return merged.reset_index(drop=True)

    return cached_parquet(
        f"{dataset}_cb_vic_binocular", _compute, force=force,
    )


def get_cb_vic_binocular_type(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    fillna: float = 1e-6, force: bool = False,
) -> pd.DataFrame:
    """Per-instance median VIC_r / VIC_l for CB neurons."""
    def _compute():
        raw = get_cb_vic_binocular(
            dataset=dataset, data_dir=data_dir, fillna=fillna,
        )
        return raw.groupby("instance")[["VIC_r", "VIC_l"]].median().reset_index()

    return cached_parquet(
        f"{dataset}_cb_vic_binocular_type", _compute, force=force,
    )


def get_cb_vic_lr_homologue(
    *, dataset: str = DATASET, data_dir=DATA_DIR, force: bool = False,
) -> pd.DataFrame:
    """Cell-type-matched right + left CB VIC (homologue comparison).

    For each side, propagates side-local visual inputs via the side's
    `lat_flow_sum.npz` onto all CB bodies, aggregates per instance via
    `get_ol_cb_vic_raw`, then collapses hemisphere variants by stripping
    `_R`/`_L` from `instance` and taking the **max** VIC per cell_type.
    Inner-merge on cell_type gives one row per bilateral homologue.
    Legacy nb 9 cells 8, 24-25.
    """
    def _compute():
        def _side_cell_type_vic(side_char, col):
            raw = get_ol_cb_vic_raw(
                dataset=dataset, data_dir=data_dir, side_char=side_char,
            )
            cb = raw[raw["region"] == "CB"].copy()
            per_inst = cb.groupby("instance")["VIC"].median().reset_index()
            per_inst["cell_type"] = [
                s[:-2] if s[-2:] in ("_R", "_L") else s
                for s in per_inst["instance"]
            ]
            idx = per_inst.groupby("cell_type")["VIC"].idxmax()
            out = per_inst.loc[idx, ["cell_type", "instance", "VIC"]].rename(
                columns={"instance": f"instance_{side_char}", "VIC": col},
            )
            return out.reset_index(drop=True)

        r = _side_cell_type_vic("r", "VIC_r")
        l = _side_cell_type_vic("l", "VIC_l")
        return r.merge(l, on="cell_type", how="inner")

    return cached_parquet(
        f"{dataset}_cb_vic_lr_homologue", _compute, force=force,
    )


def get_roi_syn_vic(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-synapse (bodyId, roi, VIC) records used by the coverage analysis.

    Combines `tbar_VIC_forJudith.pkl` (continuous VIC on non-OL synapses)
    with the four OL-side ROIs (`ME({S})`, `AME({S})`, `LO({S})`, `LOP({S})`)
    drawn from `{dataset}_OL_{side_char}_roi_counts.pkl`, expanded to
    per-synapse rows with VIC=1 (inside-OL is trivially fully visual).
    """
    def _compute():
        side_upper = side_char.upper()
        tbar = pd.read_pickle(data_dir / "tbar_VIC_forJudith.pkl").rename(
            columns={"vision": "VIC"}
        )

        roi_counts = pd.read_pickle(
            data_dir / f"{dataset}_OL_{side_char}_roi_counts.pkl"
        )
        ol_rois = [
            f"ME({side_upper})", f"AME({side_upper})",
            f"LO({side_upper})", f"LOP({side_upper})",
        ]
        ol_sub = roi_counts[roi_counts.roi.isin(ol_rois)].copy()
        ol_expanded = ol_sub.loc[ol_sub.index.repeat(ol_sub["pre"])].copy()
        ol_expanded["VIC"] = 1.0

        return pd.concat(
            [tbar[["bodyId", "roi", "VIC"]],
             ol_expanded[["bodyId", "roi", "VIC"]]],
            ignore_index=True,
        )

    return cached_parquet(
        f"{dataset}_OL_{side_char}_roi_syn_vic", _compute, force=force,
    )


