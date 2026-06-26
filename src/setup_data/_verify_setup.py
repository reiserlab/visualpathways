"""Temporary verification: compare newly-generated {DATASET}_{meta,prop,inprop}
files with the pre-existing `_orig` versions in DATA_DIR.

Run AFTER executing setup_malecns_data.ipynb end-to-end. Delete when done.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.config import DATA_DIR, DATASET

DATA = DATA_DIR
TAG = DATASET.split('_', 1)[-1]  # e.g. 'v1.0'

new_meta = DATA / f"malecns_{TAG}_meta.csv"
orig_meta = DATA / f"malecns_{TAG}_meta_orig.csv"
new_prop = DATA / f"malecns_{TAG}_prop.npz"
orig_prop = DATA / f"malecns_{TAG}_prop_orig.npz"
new_inprop = DATA / f"malecns_{TAG}_inprop.npz"
orig_inprop = DATA / f"malecns_{TAG}_inprop_orig.npz"

for p in (new_meta, orig_meta, new_prop, orig_prop, new_inprop, orig_inprop):
    if not p.exists():
        print(f"MISSING: {p}")
        sys.exit(1)

print("=== META ===")
m_new = pd.read_csv(new_meta)
m_orig = pd.read_csv(orig_meta)
print(f"rows: new={len(m_new):,}  orig={len(m_orig):,}")
new_ids = set(m_new.bodyId)
orig_ids = set(m_orig.bodyId)
print(f"bodyId only-new:  {len(new_ids - orig_ids):,}")
print(f"bodyId only-orig: {len(orig_ids - new_ids):,}")
print(f"bodyId common:    {len(new_ids & orig_ids):,}")

# Per-column diff on the common bodyIds
common = new_ids & orig_ids
m_new_c = m_new[m_new.bodyId.isin(common)].set_index("bodyId").sort_index()
m_orig_c = m_orig[m_orig.bodyId.isin(common)].set_index("bodyId").sort_index()
print("\nColumn-wise diff counts (common bodyIds):")
for col in ("cell_type", "instance", "nt", "class", "superclass", "subclass", "side", "coords", "sign"):
    if col in m_new_c.columns and col in m_orig_c.columns:
        a = m_new_c[col].fillna("__NA__").astype(str)
        b = m_orig_c[col].fillna("__NA__").astype(str)
        diffs = (a != b).sum()
        print(f"  {col:12s}: {diffs:>8,} differ ({100*diffs/len(a):.2f}%)")

# Coords filled-in count per instance (spot-check)
print("\nCoords-non-null instance counts (new, top 20):")
print(m_new[~m_new.coords.isna()].instance.value_counts().head(20).to_string())

print("\n=== PROP (raw weight) ===")
p_new = sp.load_npz(new_prop)
p_orig = sp.load_npz(orig_prop)
print(f"shape:  new={p_new.shape}  orig={p_orig.shape}")
print(f"nnz:    new={p_new.nnz:,}  orig={p_orig.nnz:,}")
print(f"sum:    new={p_new.sum():,}  orig={p_orig.sum():,}")
if p_new.shape == p_orig.shape:
    d = (p_new - p_orig)
    d.eliminate_zeros()
    print(f"diff nnz: {d.nnz:,}")
    if d.nnz:
        print(f"diff abs max: {np.abs(d.data).max()}")
        print(f"diff abs mean: {np.abs(d.data).mean():.4f}")

print("\n=== INPROP (normalized; expected to differ — new uses tot_weight, orig used upstream) ===")
i_new = sp.load_npz(new_inprop)
i_orig = sp.load_npz(orig_inprop)
print(f"shape:  new={i_new.shape}  orig={i_orig.shape}")
print(f"nnz:    new={i_new.nnz:,}  orig={i_orig.nnz:,}")
print(f"sum:    new={i_new.sum():.2f}  orig={i_orig.sum():.2f}")
