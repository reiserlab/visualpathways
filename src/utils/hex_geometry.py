"""Hex-grid / depth utilities shared across eyemap, input-distribution, and
retinotopy analyses.

Hosts:
- `find_depth`: normalised depth from 3D coords inside a layer ROI
- `load_depth_bins`: bin edges/centres for `find_depth` outputs
- `find_hex_ids`: assign 3D points to hex columns
- `find_neuron_hex_ids`: per-neuron hex assignment (majority or COM)
- `load_hexed_body_ids`: hex-id → bodyId map for ME / LOP (Mi1→T4 alignment)
- `fl_get_edge_ids`: edge columns of the hex grid (< 4 neighbours)
- `_load_pins`: shared helper that reads the per-ROI column-pin pickle.

Moved out of `utils/computing_functions.py` in Phase C as the last step
before that legacy module is deleted.
"""
import os

import numpy as np
import pandas as pd
from scipy import spatial

from utils.config import PARAMS_DIR
from utils.hex_hex import all_hex_df, get_hex_df


_ROI_OPTIONS = ['ME(R)', 'LO(R)', 'LOP(R)', 'ME(L)', 'LO(L)', 'LOP(L)']


def _load_pins(roi_str: str = 'ME(R)', suffix: str = ''):
    """Load columns/pins from `params/{roi_str[:-3]}{_L}_col_center_pins{suffix}.pickle`.

    Returns (col_ids, n_bins, pins). `pins` has shape (n_cols * n_bins, 3).
    """
    assert roi_str in _ROI_OPTIONS, (
        f"ROI must be one of {_ROI_OPTIONS}, but is actually '{roi_str}'"
    )
    if roi_str[-2] == 'R':
        col_df = pd.read_pickle(PARAMS_DIR / f"{roi_str[:-3]}_col_center_pins{suffix}.pickle")
    else:
        col_df = pd.read_pickle(PARAMS_DIR / f"{roi_str[:-3]}_L_col_center_pins{suffix}.pickle")

    col_df = col_df.dropna()
    col_ids = col_df.index.values
    n_bins = int((col_df.shape[1] - 3) / 3)
    pins = col_df.iloc[:, 3:].values.reshape((-1, 3))
    return col_ids, n_bins, pins


def find_depth(xyz_df, roi_str='ME(R)', samp=2) -> pd.DataFrame:
    """Find normalized depth (0=top, 1=bottom) and depth bin for each 3D point in xyz_df."""
    assert roi_str in _ROI_OPTIONS, (
        f"ROI must be one of {_ROI_OPTIONS}, but is actually '{roi_str}'"
    )
    _, n_bins, pins = _load_pins(roi_str=roi_str)

    tree = spatial.KDTree(pins)
    _, minid = tree.query(xyz_df[['x', 'y', 'z']].values)

    depth_bins = np.mod(minid, n_bins)
    n_bins = int(np.floor((n_bins - 1) / samp)) + 1
    depth_bins = np.asarray(np.floor(depth_bins / samp), dtype='int')

    return pd.DataFrame.from_dict({
        'depth': (n_bins - 1 - depth_bins) / (n_bins - 1),
        'bin': n_bins - 1 - depth_bins,
    })


def load_depth_bins(roi_str: str = 'ME(R)', samp: int = 2):
    """Return (bin_edges, bin_centers) for depth in `roi_str`."""
    assert roi_str in _ROI_OPTIONS, (
        f"ROI must be one of {_ROI_OPTIONS}, but is actually '{roi_str}'"
    )
    _, n_bins, _ = _load_pins(roi_str=roi_str)
    n_bins_samp = int(np.floor((n_bins - 1) / samp)) + 1
    bin_edges = np.linspace(
        0 - 1 / (n_bins_samp - 1) / 2,
        1 + 1 / (n_bins_samp - 1) / 2,
        n_bins_samp + 1,
    )
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    return bin_edges, bin_centers


def find_hex_ids(xyz_df, roi_str='ME(R)') -> pd.DataFrame:
    """Assign 3D points (assumed in `roi_str`) to columns. Returns col_id, hex1_id, hex2_id."""
    assert roi_str in _ROI_OPTIONS, (
        f"ROI must be one of {_ROI_OPTIONS}, but is actually '{roi_str}'"
    )
    col_ids, n_bins, pins = _load_pins(roi_str=roi_str)

    tree = spatial.KDTree(pins)
    _, minid = tree.query(xyz_df[['x', 'y', 'z']].values)

    col_df = all_hex_df()
    col_df.index.name = 'col_id'

    result_df = pd.DataFrame(
        col_ids[np.floor(minid / n_bins).astype(int)], columns=['col_id']
    )
    return result_df.merge(col_df, 'left', on='col_id')


def find_neuron_hex_ids(syn_df, roi_str='ME(R)', method='majority') -> pd.DataFrame:
    """Assign a single hex coordinate to each neuron (majority of synapses, or COM)."""
    assert roi_str in _ROI_OPTIONS, (
        f"ROI must be one of {_ROI_OPTIONS}, but is actually '{roi_str}'"
    )
    if method == 'majority':
        hex_df = find_hex_ids(syn_df, roi_str=roi_str)
        syn_df['col_id'] = hex_df['col_id'].values
        target_df = pd.DataFrame(
            syn_df.groupby('bodyId')[['col_id']].agg(lambda x: pd.Series.mode(x)[0])
        )
    elif method == 'COM':
        target_df = pd.DataFrame(syn_df.groupby('bodyId')[['x', 'y', 'z']].mean())
        hex_df = find_hex_ids(target_df, roi_str=roi_str)
        target_df['col_id'] = hex_df['col_id'].values
    else:
        raise ValueError(f"method must be 'majority' or 'COM', got {method!r}")

    col_df = all_hex_df()
    col_df.index.name = 'col_id'

    target_df.sort_values('col_id', inplace=True)
    target_df.reset_index(inplace=True)
    return target_df.merge(col_df, 'left', on='col_id')


def load_hexed_body_ids(roi_str='ME(R)') -> pd.DataFrame:
    """Load hex_ids with assigned body_ids: Kit's manual ME assignment plus Mi1→T4 (LOP)."""
    assert roi_str in _ROI_OPTIONS, (
        f"ROI must be one of {_ROI_OPTIONS}, but is actually '{roi_str}'"
    )
    column_df = get_hex_df(neuropil=f'ME{roi_str[-3:]}')

    if roi_str[:-3] == 'LOP':
        if os.environ.get('NEUPRINT_DATASET_NAME') == 'flywire':
            alignment_file = PARAMS_DIR / 'mi1_t4_alignment_flywire.xlsx'
            column_df2 = pd.read_excel(alignment_file, dtype=str, engine='openpyxl')
            column_df2 = column_df2.astype({'valid_group': int})
        else:
            if roi_str[-2] == 'R':
                alignment_file = PARAMS_DIR / 'mi1_t4_alignment.xlsx'
            else:
                alignment_file = PARAMS_DIR / 'mi1_t4_alignment_L.xlsx'
            column_df2 = pd.read_excel(alignment_file).convert_dtypes()
        column_df2 = column_df2[column_df2['valid_group'] == 1]
        column_df2 = column_df2.rename(columns={
            'mi1_bid': 'Mi1',
            't4a_bid': 'T4a', 't4b_bid': 'T4b', 't4c_bid': 'T4c', 't4d_bid': 'T4d',
        })
        column_df = column_df.merge(column_df2, how='left', on='Mi1')
        column_df = column_df.loc[:, ['hex1_id', 'hex2_id', 'Mi1', 'T4a', 'T4b', 'T4c', 'T4d']]
        column_df.drop_duplicates(inplace=True)

    return column_df


def fl_get_edge_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows of `df[['hex1_id', 'hex2_id']]` that are on the hex grid edge (<4 neighbors)."""
    df = df.drop_duplicates(['hex1_id', 'hex2_id'])[['hex1_id', 'hex2_id']].astype(int)
    df['neighbors'] = df.apply(
        lambda r: np.where(
            (np.abs(df['hex1_id'] - r.hex1_id) == 1)
            & (np.abs(df['hex2_id'] - r.hex2_id) == 1)
        )[0],
        axis=1,
    )
    df['is_edge'] = df.apply(lambda r: len(r['neighbors']) < 4, axis=1)
    return df[df['is_edge']][['hex1_id', 'hex2_id']]
