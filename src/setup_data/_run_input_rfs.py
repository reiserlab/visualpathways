# %%
"""Temporary: fetch input syn-per-col parquet, then fit + render figures."""
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# %%
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv

# %%
from utils import olc_client
import setup_data as sd
from utils import ol_rf

# %%
from utils.config import DATA_DIR, DATASET, PROJECT_ROOT, SIDE_CHAR
load_dotenv(PROJECT_ROOT / ".env")
TAG = DATASET.split('_', 1)[-1]  # e.g. 'v1.0'
LOG = Path("/tmp/input_rfs.log")
LOG.write_text("")


# %%
def log(msg):
    s = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    with open(LOG, "a") as f:
        f.write(s + "\n")
        f.flush()
        os.fsync(f.fileno())


# %%
try:
    log("start")
    c = olc_client.connect(verbose=False)
    log("connected")

    stem = f"malecns_{TAG}_OL_{SIDE_CHAR}"

    instances = sorted(
        ol_rf._ol_non_visual_input_instances(f"malecns_{TAG}", DATA_DIR, SIDE_CHAR)
    )
    log(f"instances: {len(instances)}")

    syn_df = sd.fetch_input_syn_per_col(instances, client=c, rel_input_weight=0.4)
    log(f"fetch done: rows={len(syn_df):,}")
    path = sd.save_input_syn_per_col(syn_df, DATA_DIR, stem)
    log(f"saved -> {path.name}")

    log("fitting input RFs (per body)...")
    raw = ol_rf.get_input_rf_raw_ol(force=True)
    log(f"per-body rows: {len(raw):,}")

    log("fitting input RFs (per type)...")
    types = ol_rf.get_input_rf_type_ol(force=True)
    log(f"per-(instance,roi) rows: {len(types):,}")

    log("regenerating downstream caches...")
    cmp_body = ol_rf.get_rf_comparison_body(force=True)
    log(f"cmp_body rows: {len(cmp_body):,}")
    cmp_type = ol_rf.get_rf_comparison_type(force=True)
    log(f"cmp_type rows: {len(cmp_type):,}")

    log("rendering figures...")
    import pandas as pd
    from utils import plotting as plot
    from utils import config
    from utils.palettes import load_colors

    colored_region, colored_main_groups, colored_sign, _ = load_colors()
    EXAMPLE_BID = 30134

    roi_counts = pd.read_pickle(
        DATA_DIR / f"malecns_{TAG}_OL_{SIDE_CHAR}_roi_counts.pkl"
    )
    log("plot_input_rf_example (Main Fig 3c)")
    plot.plot_input_rf_example(roi_counts, raw, EXAMPLE_BID, "LC6_R")

    type_ol = ol_rf.get_rf_type_ol()
    exp_sz = ol_rf.get_experimental_rf_sizes()
    exp_cc = ol_rf.get_experimental_rf_sizes(sheet="RF size CC")

    log("plot_rf_size_example_comparison")
    plot.plot_rf_size_example_comparison(cmp_body, star_instance="LC6_R", example_bid=EXAMPLE_BID)
    log("plot_rf_size_type_scatters")
    plot.plot_rf_size_type_scatters(cmp_type, colored_main_groups)
    log("plot_rf_size_vs_experiment")
    plot.plot_rf_size_vs_experiment(cmp_type, exp_sz, colored_main_groups, colored_region)

    log("DONE")
except Exception:
    log("EXCEPTION:\n" + traceback.format_exc())
    raise
