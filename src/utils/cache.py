"""Cache helpers for preprocessing outputs.

`cached_*(name, compute_fn, force=False, **kwargs)` returns the cached artefact
if `data/cache/<name>.<ext>` exists and `force=False`, otherwise calls
`compute_fn(**kwargs)`, writes the result, and returns it. Module name is
`cache` (not `io`) to avoid the stdlib collision.
"""
from pathlib import Path
import pickle

import pandas as pd
import scipy.sparse as sp

from utils.config import CACHE_PROC_DIR


def _path(name: str, ext: str) -> Path:
    CACHE_PROC_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_PROC_DIR / f"{name}.{ext}"


def cached_parquet(name, compute_fn, *, force=False, verbose=False, **kwargs):
    p = _path(name, "parquet")
    if p.exists() and not force:
        if verbose:
            print(f"[cache] loading from cache: {p}")
        return pd.read_parquet(p)
    if verbose:
        reason = "force re-compute" if p.exists() else "no cache found"
        print(f"[cache] computing ({reason}), will write: {p}")
    df = compute_fn(**kwargs)
    df.to_parquet(p, index=False)
    return df


def cached_npz(name, compute_fn, *, force=False, **kwargs):
    p = _path(name, "npz")
    if p.exists() and not force:
        return sp.load_npz(p)
    m = compute_fn(**kwargs)
    sp.save_npz(p, m)
    return m


def cached_pickle(name, compute_fn, *, force=False, **kwargs):
    p = _path(name, "pkl")
    if p.exists() and not force:
        with open(p, "rb") as f:
            return pickle.load(f)
    obj = compute_fn(**kwargs)
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    return obj
