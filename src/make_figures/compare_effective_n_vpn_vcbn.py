# %% [markdown]
# Compare the effective number of visual inputs (N_eff) between VPN and VCBN
# instances, and test whether N_eff is smaller for VCBNs than for VPNs.
#
# N_eff per instance = exp(Shannon entropy of the sum-1-normalised 8-dim
# effective visual-input feature vector). Identical to `_instance_effective_n`
# in figures.ipynb (section 5). See docs/methods_cluster_input_mixing.md.

# %%
import sys
from pathlib import Path

from dotenv import find_dotenv

PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from utils import ol_data, cb_data

FRAC_OL = 0.19  # VPN (OL) clustering threshold (paper_figures.ipynb)
FRAC_CB = 0.17  # VCBN (CB) clustering threshold (paper_figures.ipynb)


# %%
def instance_effective_n(feature_df, cluster_df):
    """Per-instance effective # of visual inputs, joined to cluster.

    Replicates figures.ipynb `_instance_effective_n`.
    Returns (df[instance, eff_n, cluster], n_inputs).
    """
    in_cols = [c for c in feature_df.columns if c.endswith(("_L", "_R"))]
    vals = feature_df[in_cols].fillna(0.0)
    row_sum = vals.sum(1)
    valid = row_sum > 0
    p = vals[valid].div(row_sum[valid], axis=0)
    entropy = -(p * np.log(p.where(p > 0))).fillna(0.0).sum(1)
    df = pd.DataFrame({"instance": p.index, "eff_n": np.exp(entropy).values})
    return df.merge(cluster_df[["instance", "cluster"]], on="instance"), len(in_cols)


# %%
# --- VPN (optic-lobe output) ---
feat_vpn = ol_data.get_ol_features()                      # side_char='r' default
cl_vpn = ol_data.get_ol_clusters(frac_thre=FRAC_OL)
vpn_df, vpn_n_in = instance_effective_n(feat_vpn, cl_vpn)

# --- VCBN (central-brain output) ---
feat_cb = cb_data.get_cb_features(side_char="r")
cl_cb = cb_data.get_cb_clusters(side_char="r", frac_thre=FRAC_CB)
vcbn_df, vcbn_n_in = instance_effective_n(feat_cb, cl_cb)

assert vpn_n_in == vcbn_n_in, (
    f"input-channel ceiling differs: VPN={vpn_n_in}, VCBN={vcbn_n_in}; "
    "N_eff would not be on a comparable scale"
)
print(f"input-channel ceiling (max N_eff): {vpn_n_in} for both groups\n")

vpn = vpn_df["eff_n"].to_numpy()
vcbn = vcbn_df["eff_n"].to_numpy()


# %%
def describe(name, x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    print(
        f"{name:5s}  n={len(x):4d}  mean={x.mean():.3f}  median={med:.3f}  "
        f"IQR=[{q1:.3f}, {q3:.3f}]  range=[{x.min():.3f}, {x.max():.3f}]"
    )


print("Per-instance effective number of visual inputs (N_eff):")
describe("VPN", vpn)
describe("VCBN", vcbn)

# %%
# One-sided Mann-Whitney U: is N_eff stochastically smaller for VCBN than VPN?
# scipy returns U1, the statistic for the FIRST sample (vcbn): U1 = #(vcbn>vpn) pairs.
U, p_mw = stats.mannwhitneyu(vcbn, vpn, alternative="less")
n1, n2 = len(vcbn), len(vpn)
# Cliff's delta = P(vcbn>vpn) - P(vcbn<vpn) = (2*U1 - n1*n2)/(n1*n2).
# Negative => VCBN typically smaller than VPN.
cliffs_delta = 2 * U / (n1 * n2) - 1

# Also a two-sided test for completeness.
U2, p_two = stats.mannwhitneyu(vcbn, vpn, alternative="two-sided")

print("\nMann-Whitney U test (VCBN N_eff < VPN N_eff):")
print(f"  U (VCBN<VPN)      = {U:.1f}")
print(f"  one-sided p       = {p_mw:.3e}")
print(f"  two-sided p       = {p_two:.3e}")
print(f"  Cliff's delta     = {cliffs_delta:+.3f}  "
      "(negative => VCBN typically smaller)")
print(f"  median difference = {np.median(vcbn) - np.median(vpn):+.3f} "
      f"(VCBN {np.median(vcbn):.3f} - VPN {np.median(vpn):.3f})")

# %%
# Save the per-instance values for reuse / plotting.
out = pd.concat(
    [vpn_df.assign(group="VPN"), vcbn_df.assign(group="VCBN")],
    ignore_index=True,
)
out_path = PROJECT_ROOT / "results" / "tables" / "effective_n_vpn_vs_vcbn.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)
print(f"\nwrote {out_path}")
