"""Central config: paths, dataset selection, and shared color palettes."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "cache" / "data"
PARAMS_DIR = PROJECT_ROOT / "params"
EYEMAP_DIR = PARAMS_DIR / "eyemap"
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_PROC_DIR = PROJECT_ROOT / "cache" / "data_proc"
FIG_DIR = PROJECT_ROOT / "results"
HTML_FIG_DIR = FIG_DIR / "html_figures"

DATASET = "malecns_v1.0"
FIG_DATASET = "malecns_v0.9"
SIDE = "right"
N_FLOW = 20
HIT_THRE = 0.1

SIDE_CHAR = SIDE[0]
DATASET_DIR = PROJECT_ROOT / "cache" / "data" / DATASET


def load_colors():
    """Return (region, main_groups, sign, seed) color DataFrames."""
    colored_main_groups_df = pd.DataFrame(
        ["#90f0a6", "#ffc4e2", "#64a0d1", "#cc9e4f", "#616161"],
        index=["visual input", "OL internal", "OL output", "CB input", "other"],
        columns=["color"],
    )
    colored_region_df = pd.DataFrame(
        ["#B6B6B6", "#868686", "#616161", "#383838"],
        index=["right OL", "left OL", "CB", "VNC"],
        columns=["color"],
    )
    colored_sign_df = pd.DataFrame(
        ["#EE672D", "#979DA5", "#1F4695"], index=[1, 0, -1], columns=["color"],
    )
    colored_seed_df = pd.DataFrame.from_dict(
        {
            "L1": "#b4e5cb", "L2": "#029356", "L3": "#8fce0a",
            "R7": "#8e56f5", "R7d": "#4220B4", "R8": "#51bfc7", "R8d": "#099C9C",
            "HBeyelet": "#151515",
        },
        orient="index", columns=["color"],
    )
    return colored_region_df, colored_main_groups_df, colored_sign_df, colored_seed_df


def load_roi_colors():
    """ROI palette as a DataFrame with `ROI`, `Hex`, `rgba` columns."""
    import matplotlib.colors as mcolors
    roi_hex = {
        "SLP":  "#5dc9e1",
        "SMP":  "#4597d2",
        "ICL":  "#652d91",
        "IB":   "#8351a1",
        "EB":   "#f36f40",
        "PB":   "#fbab3d",
        "BU":   "#f49cc3",
        "LAL":  "#c6197e",
        "PVLP": "#1cb24b",
        "WED":  "#079647",
        "PLP":  "#0d783d",
        "AOTU": "#1c401d",
        "IPS":  "#9dd6c9",
        "SPS":  "#05af8e",
        "GNG":  "#21366b",
    }
    df = pd.DataFrame({"ROI": list(roi_hex.keys()), "Hex": list(roi_hex.values())})
    df["rgba"] = df["Hex"].apply(mcolors.to_rgba)
    return df


def load_nt_colors():
    """Neurotransmitter → hex color mapping."""
    return {
        "acetylcholine": "#EE672D",
        "glutamate":     "#09A64D",
        "gaba":          "#1F4695",
        "dopamine":      "#979DA5",
        "histamine":     "#979DA5",
        "octopamine":    "#979DA5",
        "serotonin":     "#979DA5",
        "other":         "#979DA5",
        "unclear":       "#979DA5",
    }
