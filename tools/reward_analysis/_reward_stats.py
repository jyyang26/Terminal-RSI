#!/usr/bin/env python3
"""Stage 2: compute all statistics for the three deliverables -> _reward_stats.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
df = pd.read_pickle(BASE / "_reward_rows.pkl")

RUN1 = "Run 1 (Discrete Progress)"
RUN2 = "Run 2 (Success-Smoothed)"

BUCKETS = [(-1, 3, "≤3"), (3, 8, "4–8"), (8, 15, "9–15"), (15, 10**9, ">15")]


def bucket_label(n):
    for lo, hi, lab in BUCKETS:
        if lo < n <= hi:
            return lab
    return ">15"


df["vbucket"] = df["total_validators"].apply(bucket_label)


def corr_block(sub):
    x = sub["total_validators"].to_numpy(dtype=float)
    y = sub["reward"].to_numpy(dtype=float)
    out = {}
    if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0:
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        # log validator count correlation (relationship is saturating/curved)
        lr, lp = stats.pearsonr(np.log1p(x), y)
        out = {
            "pearson": float(pr), "pearson_p": float(pp),
            "spearman": float(sr), "spearman_p": float(sp),
            "pearson_logNv": float(lr), "pearson_logNv_p": float(lp),
        }
    return out


result = {"runs": {RUN1: {}, RUN2: {}}, "run_labels": {"run1": RUN1, "run2": RUN2}}

for run in (RUN1, RUN2):
    sub = df[df.run == run]
    r = sub["reward"].to_numpy(dtype=float)
    result["runs"][run]["overall"] = {
        "n": int(len(sub)),
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=1)),
        "cv": float(np.std(r, ddof=1) / np.mean(r)) if np.mean(r) else None,
        "p25": float(np.percentile(r, 25)),
        "median": float(np.percentile(r, 50)),
        "p75": float(np.percentile(r, 75)),
        "min": float(np.min(r)),
        "max": float(np.max(r)),
        "success_rate": float(sub["terminal_success"].mean()),
    }
    result["runs"][run]["corr"] = corr_block(sub)

# Validator-bucket breakdown
bucket_rows = []
for lo, hi, lab in BUCKETS:
    row = {"bucket": lab}
    for run, tag in ((RUN1, "run1"), (RUN2, "run2")):
        s = df[(df.run == run) & (df.vbucket == lab)]
        r = s["reward"].to_numpy(dtype=float)
        row[f"{tag}_n"] = int(len(s))
        row[f"{tag}_mean"] = float(np.mean(r)) if len(r) else 0.0
        row[f"{tag}_std"] = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        row[f"{tag}_median"] = float(np.median(r)) if len(r) else 0.0
    bucket_rows.append(row)
result["validator_buckets"] = bucket_rows

# Per-family breakdown (joined across runs)
fam_rows = []
families = sorted(df["family"].unique())
for fam in families:
    row = {"family": fam}
    nv_all = df[df.family == fam]["total_validators"].to_numpy(dtype=float)
    row["n_traj"] = int(len(df[(df.family == fam) & (df.run == RUN1)]))
    row["mean_validators"] = float(np.mean(nv_all)) if len(nv_all) else 0.0
    for run, tag in ((RUN1, "run1"), (RUN2, "run2")):
        s = df[(df.run == run) & (df.family == fam)]
        r = s["reward"].to_numpy(dtype=float)
        row[f"{tag}_mean"] = float(np.mean(r)) if len(r) else 0.0
        row[f"{tag}_std"] = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        row[f"{tag}_p25"] = float(np.percentile(r, 25)) if len(r) else 0.0
        row[f"{tag}_median"] = float(np.percentile(r, 50)) if len(r) else 0.0
        row[f"{tag}_p75"] = float(np.percentile(r, 75)) if len(r) else 0.0
        row[f"{tag}_success"] = float(s["terminal_success"].mean()) if len(s) else 0.0
    fam_rows.append(row)
# sort by mean validators desc for display
fam_rows.sort(key=lambda d: d["mean_validators"], reverse=True)
result["families"] = fam_rows

# Between-family dispersion of family-mean reward (how much reward tracks family identity)
f1 = np.array([r["run1_mean"] for r in fam_rows])
f2 = np.array([r["run2_mean"] for r in fam_rows])
result["family_mean_dispersion"] = {
    "run1_std_of_family_means": float(np.std(f1, ddof=1)),
    "run2_std_of_family_means": float(np.std(f2, ddof=1)),
}

# Variance-vs-validator relationship: correlate per-family std with per-family mean validators
mv = np.array([r["mean_validators"] for r in fam_rows])
s1 = np.array([r["run1_std"] for r in fam_rows])
s2 = np.array([r["run2_std"] for r in fam_rows])
mask = mv > 0
result["std_vs_validators"] = {
    "run1_pearson": float(stats.pearsonr(mv[mask], s1[mask])[0]),
    "run2_pearson": float(stats.pearsonr(mv[mask], s2[mask])[0]),
}

(BASE / "_reward_stats.json").write_text(json.dumps(result, indent=2))

# ---- console summary ----
print("=" * 78)
for run in (RUN1, RUN2):
    o = result["runs"][run]["overall"]
    c = result["runs"][run]["corr"]
    print(f"\n{run}")
    print(f"  reward  mean={o['mean']:.4f}  std={o['std']:.4f}  CV={o['cv']:.3f}  "
          f"median={o['median']:.4f}  max={o['max']:.3f}  success={o['success_rate']:.3%}")
    print(f"  corr(N_v,R)  Pearson={c['pearson']:.4f} (p={c['pearson_p']:.1e})  "
          f"Spearman={c['spearman']:.4f}  Pearson(logN_v)={c['pearson_logNv']:.4f}")

print("\nValidator buckets (mean reward / std):")
for row in bucket_rows:
    print(f"  {row['bucket']:>5}  n1={row['run1_n']:4d}  "
          f"R1={row['run1_mean']:.3f}±{row['run1_std']:.3f}   "
          f"R2={row['run2_mean']:.3f}±{row['run2_std']:.3f}")

print("\nFamily-mean reward dispersion (std across families):")
print(f"  Run1={result['family_mean_dispersion']['run1_std_of_family_means']:.4f}  "
      f"Run2={result['family_mean_dispersion']['run2_std_of_family_means']:.4f}")
print("std(reward) vs mean validators correlation:")
print(f"  Run1={result['std_vs_validators']['run1_pearson']:.4f}  "
      f"Run2={result['std_vs_validators']['run2_pearson']:.4f}")
print(f"\nFamilies analyzed: {len(fam_rows)}")
print("Saved -> _reward_stats.json")
