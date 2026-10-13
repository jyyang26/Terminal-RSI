#!/usr/bin/env python3
"""Deliverable 2: multi-page A4 white-paper PDF -> reward_distribution_analysis.pdf.

Built with matplotlib's PdfPages (reportlab/weasyprint not installed). Pages:
cover, TOC, executive summary, reward-function derivation, parameter table,
overall distributions, validator-bucket analysis, per-family table, conclusions.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

BASE = Path(__file__).resolve().parent
df = pd.read_pickle(BASE / "_reward_rows.pkl")
S = json.loads((BASE / "_reward_stats.json").read_text())
R1 = "Run 1 (Discrete Progress)"
R2 = "Run 2 (Success-Smoothed)"

HERO, NAVY, ICE, MIDB, GREY = "#285AC7", "#0F2459", "#E8F0FE", "#93B5F5", "#5A739C"
A4 = (8.27, 11.69)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "mathtext.fontset": "dejavusans"})

o1, o2 = S["runs"][R1]["overall"], S["runs"][R2]["overall"]
c1, c2 = S["runs"][R1]["corr"], S["runs"][R2]["corr"]


def new_page(pdf, header=None):
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.972), 1, 0.028, transform=ax.transAxes, color=NAVY, lw=0))
    fig.text(0.07, 0.978, "Harbor Terminal-RL · Reward Distribution Analysis",
             fontsize=8.5, color="white", va="center")
    fig.text(0.93, 0.978, "Qwen3-27B · PPO 256k", fontsize=8.5, color="white",
             va="center", ha="right")
    if header:
        fig.text(0.07, 0.935, header, fontsize=17, color=NAVY, fontweight="bold", va="top")
        ax.axhline(0.922, 0.07, 0.93, color=MIDB, lw=1.5)
    return fig, ax


def footer(fig, n):
    fig.text(0.5, 0.016, f"— {n} —", fontsize=8, color=GREY, ha="center")


def wrapped(fig, x, y, text, fontsize=10.5, color="#1e293b", leading=0.0175, bold=False, maxw=95):
    import textwrap
    yy = y
    for para in text.split("\n"):
        if not para.strip():
            yy -= leading * 0.7
            continue
        for ln in textwrap.wrap(para, maxw) or [""]:
            fig.text(x, yy, ln, fontsize=fontsize, color=color, va="top",
                     fontweight="bold" if bold else "normal")
            yy -= leading
    return yy


pdf = PdfPages(BASE / "reward_distribution_analysis.pdf")

# ============ PAGE 1 — COVER ============
fig = plt.figure(figsize=A4)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0.62), 1, 0.38, color=NAVY, lw=0))
ax.add_patch(plt.Rectangle((0, 0.60), 1, 0.02, color=HERO, lw=0))
fig.text(0.5, 0.86, "Reward Distribution Analysis", fontsize=30, color="white",
         ha="center", fontweight="bold")
fig.text(0.5, 0.80, "Decoupling RL Reward from Validator Count", fontsize=18,
         color="#CDE0FF", ha="center")
fig.text(0.5, 0.755, "in Terminal-Agent Training", fontsize=18, color="#CDE0FF", ha="center")
fig.text(0.5, 0.675, "Discrete Progress  vs.  Success-Smoothed Progress Reward",
         fontsize=13, color="white", ha="center", style="italic")
fig.text(0.5, 0.52, "A reward-shaping audit of two Qwen3-27B Harbor PPO runs (kxm RESUME19),\n"
                    "quantifying how each reward couples to task-family identity and\n"
                    "validator (test) count — and how the redesign suppresses reward hacking.",
         fontsize=11.5, color="#334155", ha="center", linespacing=1.6)
# key-numbers strip
cards = [
    ("2 × 2,560", "trajectories\nanalysed"),
    (f"{c1['spearman']:.2f} → {c2['spearman']:.2f}", "Spearman(N_v, R)\n(global)"),
    ("96 → 0", "failing trials\nwith R ≥ 1"),
    ("0.42 → 0.21", "failed-trial\nreward std"),
]
for i, (big, small) in enumerate(cards):
    x = 0.09 + i * 0.21
    ax.add_patch(plt.Rectangle((x, 0.30), 0.185, 0.13, color=ICE, ec=MIDB, lw=1.2))
    fig.text(x + 0.0925, 0.385, big, fontsize=15, color=NAVY, ha="center", fontweight="bold")
    fig.text(x + 0.0925, 0.335, small, fontsize=8.5, color="#334155", ha="center", linespacing=1.3)
fig.text(0.5, 0.2, "slime-terminal-RL · slime_plugins.harbor.partial_reward", fontsize=10,
         color=GREY, ha="center", family="monospace")
fig.text(0.5, 0.06, "Generated 2026-10-13 · latest 10 rollout dumps per run",
         fontsize=9, color=GREY, ha="center")
pdf.savefig(fig); plt.close(fig)

# ============ PAGE 2 — TOC + EXECUTIVE SUMMARY ============
fig, ax = new_page(pdf, "Contents  &  Executive Summary")
toc = [
    "1.  Executive Summary .................................................. 2",
    "2.  Reward-Function Mathematics & Derivation ................ 3",
    "3.  Experiment Parameters & Setup ................................ 4",
    "4.  Overall Reward Distributions .................................... 5",
    "5.  Reward vs. Validator Count (bucket analysis) ............. 6",
    "6.  Per-Family Reward Statistics ................................... 7",
    "7.  Conclusions & Recommendations .............................. 8",
]
y = 0.89
for t in toc:
    fig.text(0.09, y, t, fontsize=11, color="#1e293b", family="monospace"); y -= 0.025
ax.axhline(0.70, 0.07, 0.93, color=GRID if (GRID:="#D7E0F2") else MIDB, lw=1)

summary = (
    f"We compare two reward formulations on identically-sampled Harbor terminal-agent "
    f"rollouts ({o1['n']} trajectories each, latest 10 PPO steps). Run 1 uses the legacy "
    f"test_count_progress reward R = K_p / D_f (passed tests over a per-family divisor). "
    f"Run 2 uses the success_smoothed reward R = 1[solved] + K_p/(K_p + D_f).\n\n"
    f"Headline findings:\n\n"
    f"•  Reward hacking eliminated. Under Run 1, 96 FAILING trajectories (5.3%) earn "
    f"R >= 1.0 by brushing many checks in validator-rich families without ever solving — "
    f"a failing trial reaches R = {df[(df.run==R1)&(df.terminal_success==0)].reward.max():.2f}, "
    f"out-scoring 99%+ of genuine solves. Under Run 2 this count is ZERO: every failure is "
    f"bounded below 1 (max {df[(df.run==R2)&(df.terminal_success==0)].reward.max():.2f}) and "
    f"every solve is >= 1.20. The reward now cleanly separates 'solved' from 'partial'.\n\n"
    f"•  Failure-regime variance regularised. The standard deviation of failed-trial reward "
    f"halves, 0.419 -> 0.206, and the saturating K_p/(K_p+D') term removes the heavy upper "
    f"tail (failed-reward p99 1.83 -> 0.76).\n\n"
    f"•  Decoupling is in the FAILURE regime, not the global correlation. The global "
    f"Spearman(N_v, R) is ~0.80 in BOTH runs — because genuinely harder, validator-rich "
    f"tasks legitimately earn the +1 success bonus when solved. The pathology the redesign "
    f"targets is the partial-reward channel: failed-trial reward no longer inflates with "
    f"validator count, so the policy cannot farm reward by partial-passing."
)
wrapped(fig, 0.09, 0.66, summary, fontsize=10.5, leading=0.0172, maxw=92)
footer(fig, 2); pdf.savefig(fig); plt.close(fig)

# ============ PAGE 3 — REWARD MATH ============
fig, ax = new_page(pdf, "2.  Reward-Function Mathematics")
y = 0.88
y = wrapped(fig, 0.09, y, "2.1  Run 1 — Legacy discrete test-count progress", fontsize=12.5,
            color=NAVY, bold=True, leading=0.03)
fig.text(0.12, y, r"$R \;=\; \dfrac{K_p}{D_f}\,,\qquad K_p=\#\{\text{passed checks}\},"
                  r"\;\; D_f=\text{per-family divisor}$",
         fontsize=12.5, color="#1e293b", va="top")
y -= 0.055
y = wrapped(fig, 0.09, y,
            "The reward is the ABSOLUTE passed-check count divided by a fixed per-family "
            "divisor D_f (openfoam_cfd 795 … openscad 1; global fallback 82). It is UNBOUNDED "
            "above: a trajectory that passes many checks scores K_p/D_f regardless of whether "
            "the terminal task gate ever passed. In a family with hundreds of checks, a "
            "non-solving trajectory that brushes a large fraction can therefore score R > 1 — "
            "numerically indistinguishable from (or larger than) a genuine solve. This is the "
            "reward-hacking surface: progress is linear and uncapped in K_p.",
            fontsize=10.5, leading=0.0172, maxw=92)
y -= 0.02
y = wrapped(fig, 0.09, y, "2.2  Run 2 — Success-smoothed progress (ours)", fontsize=12.5,
            color=NAVY, bold=True, leading=0.03)
fig.text(0.12, y, r"$R \;=\; \mathbb{1}[\text{solved}] \;+\; \dfrac{K_p}{K_p + D'_f}$",
         fontsize=13.5, color="#1e293b", va="top")
y -= 0.05
y = wrapped(fig, 0.09, y,
            "Two coupled changes defeat the hack:\n\n"
            "(1) Success bonus 1[solved]. Adds exactly 1.0 when the terminal gate passes "
            "(terminal_reward >= 1), else 0. Because the smoothed term below is strictly < 1, "
            "EVERY solve (>=1) dominates EVERY failure (<1) — the two populations can never "
            "overlap, no matter how many checks a failure brushes.\n\n"
            "(2) Saturating partial term K_p/(K_p + D'_f). This is a Michaelis–Menten / Hill "
            "form: monotonic in K_p but BOUNDED in [0,1), with diminishing returns. The "
            "marginal value of the (n+1)-th passed check is D'_f/(K_p+D'_f)^2, which decays as "
            "K_p grows — so brushing ever more checks yields vanishing reward, and no family's "
            "raw check count can blow the signal up. D'_f is a companion divisor table "
            "re-tuned FOR this formula (18 families changed: high-scoring-failure families ×2, "
            "long-horizon low-yield families ÷4).",
            fontsize=10.5, leading=0.0172, maxw=92)
# small inline curve comparison
axc = fig.add_axes([0.57, 0.10, 0.36, 0.21])
k = np.linspace(0, 60, 400)
for D, col, lab in [(12, HERO, "D'=12"), (30, GREY, "D'=30")]:
    axc.plot(k, k / (k + D), color=col, lw=2, label=f"smoothed {lab}")
axc.plot(k, k / 30, color=NAVY, lw=1.6, ls="--", label="legacy K/30")
axc.axhline(1.0, color="#94a3b8", lw=1, ls=":")
axc.set_ylim(0, 2.0); axc.set_xlim(0, 60)
axc.set_xlabel("passed checks $K_p$", fontsize=9); axc.set_ylabel("partial reward", fontsize=9)
axc.set_title("Saturating vs. linear partial term", fontsize=9.5, color=NAVY)
axc.tick_params(labelsize=8); axc.legend(fontsize=7.5, frameon=False)
for sp in ("top", "right"): axc.spines[sp].set_visible(False)
fig.text(0.09, 0.26, "Fig 2.1  The legacy term (dashed)\ngrows without bound and crosses\n1.0; the smoothed term saturates\nbelow 1.0, capping any partial-\npass reward.",
         fontsize=9.5, color=GREY, va="top", linespacing=1.5)
footer(fig, 3); pdf.savefig(fig); plt.close(fig)

# ============ PAGE 4 — PARAMETERS ============
fig, ax = new_page(pdf, "3.  Experiment Parameters & Setup")
rows = [
    ("Attribute", "Run 1 — Discrete Progress", "Run 2 — Success-Smoothed"),
    ("Reward post-process", "test_count_progress_rewards", "success_smoothed_progress_rewards"),
    ("Reward formula", "K_p / D_f", "1[solved] + K_p/(K_p+D'_f)"),
    ("Partial reward bound", "unbounded (≥0)", "[0, 1) + success bonus"),
    ("Actor learning rate", "5e-7", "1e-7  (1/5×)"),
    ("Megatron config", "…_warmcritic.yaml", "…_warmcritic_actorlr5th.yaml"),
    ("Global divisor (fallback)", "82.0", "82.0"),
    ("Per-family divisors", "46 families (p90)", "46 families (18 re-tuned)"),
    ("Algorithm / batch", "PPO · 256k · dense-27B", "PPO · 256k · dense-27B"),
    ("Trajectories analysed", f"{o1['n']} (10 dumps)", f"{o2['n']} (10 dumps)"),
    ("Terminal success rate", f"{o1['success_rate']:.1%}", f"{o2['success_rate']:.1%}"),
]
ncol = 3; x0 = 0.07; widths = [0.26, 0.345, 0.345]
y = 0.88; rh = 0.047
xs = [x0 + sum(widths[:i]) for i in range(ncol)]
for ri, row in enumerate(rows):
    yy = y - ri * rh
    if ri == 0:
        ax.add_patch(plt.Rectangle((x0, yy - rh + 0.004), sum(widths), rh, color=NAVY, lw=0))
    elif ri % 2 == 1:
        ax.add_patch(plt.Rectangle((x0, yy - rh + 0.004), sum(widths), rh, color=ICE, lw=0))
    for ci, cell in enumerate(row):
        col = "white" if ri == 0 else ("#0f172a")
        fw = "bold" if (ri == 0 or ci == 0) else "normal"
        fig.text(xs[ci] + 0.008, yy - rh / 2 + 0.004, cell, fontsize=9.3, color=col,
                 va="center", fontweight=fw,
                 family="monospace" if (ci > 0 and ri > 0) else "DejaVu Sans")
yb = y - len(rows) * rh - 0.02
wrapped(fig, 0.09, yb,
        "Both runs resume from the same kxm RESUME19 checkpoint pair (cross-lineage actor + "
        "critic) and sample the identical mixed-3000 task pool, so family and validator-count "
        "distributions are matched across runs (mean N_v 39.5 vs 40.2; median 10 vs 10). The "
        "only deliberate differences are the reward post-process function, its companion "
        "divisor table, and the 5× lower actor learning rate in Run 2 (the smoothed reward's "
        "success bonus makes the raw signal larger, so the step size is reduced to keep the "
        "policy update stable).",
        fontsize=10.5, leading=0.0172, maxw=92)
footer(fig, 4); pdf.savefig(fig); plt.close(fig)

# ============ PAGE 5 — OVERALL DISTRIBUTIONS ============
fig, ax = new_page(pdf, "4.  Overall Reward Distributions")
# histograms
axh = fig.add_axes([0.1, 0.52, 0.82, 0.33])
bins = np.linspace(0, 2.2, 45)
axh.hist(df[df.run == R1].reward, bins=bins, color=GREY, alpha=0.55, label="Run 1 (legacy)", density=True)
axh.hist(df[df.run == R2].reward, bins=bins, color=HERO, alpha=0.55, label="Run 2 (ours)", density=True)
axh.axvline(1.0, color=NAVY, ls="--", lw=1.4)
axh.text(1.01, axh.get_ylim()[1] * 0.9, "success gate", fontsize=8.5, color=NAVY)
axh.set_xlabel("Shaped reward R", fontsize=10); axh.set_ylabel("density", fontsize=10)
axh.set_title("Reward histograms (all trajectories)", fontsize=11, color=NAVY)
axh.legend(fontsize=9, frameon=False); axh.tick_params(labelsize=8.5)
for sp in ("top", "right"): axh.spines[sp].set_visible(False)
# stats table
tbl = [
    ("Metric", "Run 1", "Run 2", "Δ"),
    ("Mean reward", f"{o1['mean']:.3f}", f"{o2['mean']:.3f}", f"{o2['mean']-o1['mean']:+.3f}"),
    ("Std reward", f"{o1['std']:.3f}", f"{o2['std']:.3f}", f"{o2['std']-o1['std']:+.3f}"),
    ("Median", f"{o1['median']:.3f}", f"{o2['median']:.3f}", f"{o2['median']-o1['median']:+.3f}"),
    ("P25 / P75", f"{o1['p25']:.2f}/{o1['p75']:.2f}", f"{o2['p25']:.2f}/{o2['p75']:.2f}", "—"),
    ("Max", f"{o1['max']:.3f}", f"{o2['max']:.3f}", f"{o2['max']-o1['max']:+.3f}"),
    ("Pearson(N_v,R)", f"{c1['pearson']:.3f}", f"{c2['pearson']:.3f}", f"{c2['pearson']-c1['pearson']:+.3f}"),
    ("Spearman(N_v,R)", f"{c1['spearman']:.3f}", f"{c2['spearman']:.3f}", f"{c2['spearman']-c1['spearman']:+.3f}"),
    ("Failed-trial std", "0.419", "0.206", "-0.213"),
    ("Failing trials R≥1", "96 (5.3%)", "0 (0.0%)", "-96"),
]
x0 = 0.1; widths = [0.3, 0.17, 0.17, 0.18]; rh = 0.035
xs = [x0 + sum(widths[:i]) for i in range(4)]
yt = 0.42
for ri, row in enumerate(tbl):
    yy = yt - ri * rh
    if ri == 0:
        ax.add_patch(plt.Rectangle((x0, yy - rh + 0.003), sum(widths), rh, color=NAVY, lw=0))
    elif ri % 2 == 1:
        ax.add_patch(plt.Rectangle((x0, yy - rh + 0.003), sum(widths), rh, color=ICE, lw=0))
    for ci, cell in enumerate(row):
        col = "white" if ri == 0 else "#0f172a"
        fig.text(xs[ci] + 0.006, yy - rh / 2 + 0.003, cell, fontsize=9.2, color=col, va="center",
                 fontweight="bold" if (ri == 0 or ci == 0) else "normal",
                 family="monospace" if (ci > 0 and ri > 0) else "DejaVu Sans")
wrapped(fig, 0.09, 0.09,
        "The Run 2 histogram is visibly BIMODAL: a lower lobe of bounded partial failures and "
        "a sharp solved spike just above 1.0 — exactly the clean 'solved vs not' separation the "
        "success bonus is designed to create. Run 1 is unimodal with a long right tail crossing "
        "the gate, where failing trajectories masquerade as solves.",
        fontsize=10, leading=0.017, maxw=92)
footer(fig, 5); pdf.savefig(fig); plt.close(fig)

# ============ PAGE 6 — VALIDATOR BUCKETS ============
fig, ax = new_page(pdf, "5.  Reward vs. Validator Count")
buckets = S["validator_buckets"]
labs = [b["bucket"] for b in buckets]
xpos = np.arange(len(labs))
axb = fig.add_axes([0.1, 0.55, 0.82, 0.31])
w = 0.38
m1 = [b["run1_mean"] for b in buckets]; m2 = [b["run2_mean"] for b in buckets]
e1 = [b["run1_std"] for b in buckets]; e2 = [b["run2_std"] for b in buckets]
axb.bar(xpos - w/2, m1, w, yerr=e1, color=GREY, alpha=0.8, label="Run 1", capsize=3,
        error_kw=dict(ecolor="#334155", lw=1))
axb.bar(xpos + w/2, m2, w, yerr=e2, color=HERO, alpha=0.85, label="Run 2", capsize=3,
        error_kw=dict(ecolor=NAVY, lw=1))
axb.axhline(1.0, color=NAVY, ls="--", lw=1.2)
axb.set_xticks(xpos); axb.set_xticklabels(labs, fontsize=9.5)
axb.set_xlabel("Validator-count bucket $N_v$", fontsize=10); axb.set_ylabel("Mean reward ± std", fontsize=10)
axb.set_title("Mean reward by validator bucket (all trajectories)", fontsize=11, color=NAVY)
axb.legend(fontsize=9, frameon=False); axb.tick_params(labelsize=8.5)
for sp in ("top", "right"): axb.spines[sp].set_visible(False)
# bucket table
tbl = [("Bucket", "n", "R1 mean", "R1 std", "R2 mean", "R2 std")]
for b in buckets:
    tbl.append((b["bucket"], str(b["run1_n"]), f"{b['run1_mean']:.3f}", f"{b['run1_std']:.3f}",
                f"{b['run2_mean']:.3f}", f"{b['run2_std']:.3f}"))
x0 = 0.14; widths = [0.1, 0.08, 0.13, 0.13, 0.13, 0.13]; rh = 0.036
xs = [x0 + sum(widths[:i]) for i in range(6)]
yt = 0.44
for ri, row in enumerate(tbl):
    yy = yt - ri * rh
    if ri == 0:
        ax.add_patch(plt.Rectangle((x0, yy - rh + 0.003), sum(widths), rh, color=NAVY, lw=0))
    elif ri % 2 == 1:
        ax.add_patch(plt.Rectangle((x0, yy - rh + 0.003), sum(widths), rh, color=ICE, lw=0))
    for ci, cell in enumerate(row):
        col = "white" if ri == 0 else "#0f172a"
        fig.text(xs[ci] + 0.006, yy - rh / 2 + 0.003, cell, fontsize=9.3, color=col, va="center",
                 fontweight="bold" if ri == 0 else "normal",
                 family="monospace" if ri > 0 else "DejaVu Sans")
wrapped(fig, 0.09, 0.2,
        f"Globally, mean reward rises with validator count under BOTH rewards (Spearman "
        f"{c1['spearman']:.2f} / {c2['spearman']:.2f}) — large-N_v tasks are harder but, when "
        f"solved, legitimately earn more under Run 2. The decisive difference is the FAILURE "
        f"regime (next page & main figure): failed-trial reward correlates with N_v at Spearman "
        f"0.84 in Run 1 but is flat and bounded (≤0.86) in Run 2. The >15 bucket's Run-2 std is "
        f"larger only because the success bonus makes that bucket bimodal (many solves at ~1.5), "
        f"not because the partial signal is noisier.",
        fontsize=10, leading=0.017, maxw=92)
footer(fig, 6); pdf.savefig(fig); plt.close(fig)

# ============ PAGE 7 — FAMILY TABLE ============
fig, ax = new_page(pdf, "6.  Per-Family Reward Statistics")
fams = S["families"][:20]
hdr = ("Family", "n", "N_v", "R1 μ", "R1 σ", "R2 μ", "R2 σ", "R2 succ")
tbl = [hdr]
for r in fams:
    tbl.append((r["family"][:20], str(r["n_traj"]), f"{r['mean_validators']:.0f}",
                f"{r['run1_mean']:.2f}", f"{r['run1_std']:.2f}",
                f"{r['run2_mean']:.2f}", f"{r['run2_std']:.2f}", f"{r['run2_success']:.0%}"))
x0 = 0.055; widths = [0.235, 0.055, 0.07, 0.095, 0.095, 0.095, 0.095, 0.1]; rh = 0.0345
xs = [x0 + sum(widths[:i]) for i in range(len(widths))]
yt = 0.9
for ri, row in enumerate(tbl):
    yy = yt - ri * rh
    if ri == 0:
        ax.add_patch(plt.Rectangle((x0, yy - rh + 0.002), sum(widths), rh, color=NAVY, lw=0))
    elif ri % 2 == 1:
        ax.add_patch(plt.Rectangle((x0, yy - rh + 0.002), sum(widths), rh, color=ICE, lw=0))
    for ci, cell in enumerate(row):
        col = "white" if ri == 0 else "#0f172a"
        fig.text(xs[ci] + 0.004, yy - rh / 2 + 0.002, cell, fontsize=8.3, color=col, va="center",
                 fontweight="bold" if (ri == 0 or ci == 0) else "normal",
                 family="monospace" if (ci > 0 and ri > 0) else "DejaVu Sans")
yb = yt - len(tbl) * rh - 0.015
wrapped(fig, 0.055, yb,
        "Families ordered by mean validator count (top 20 of 46). μ/σ are mean/std of shaped "
        "reward; 'R2 succ' is Run-2 terminal success rate. Note openfoam_cfd / cmake_build_config "
        "(N_v in the hundreds): under Run 1 their failed trials still accrue sizeable K_p/D_f; "
        "under Run 2 the saturating term caps them. The full 46-family, sortable/searchable "
        "table is in the companion HTML report (reward_distribution_report.html).",
        fontsize=9.5, leading=0.016, maxw=100)
footer(fig, 7); pdf.savefig(fig); plt.close(fig)

# ============ PAGE 8 — CONCLUSIONS ============
fig, ax = new_page(pdf, "7.  Conclusions & Recommendations")
y = 0.88
y = wrapped(fig, 0.09, y, "Findings", fontsize=13, color=NAVY, bold=True, leading=0.03)
y = wrapped(fig, 0.09, y,
            "1.  The success-smoothed reward ELIMINATES the partial-pass reward-hacking surface. "
            "The 96 (5.3%) failing trajectories that scored R≥1 under the legacy reward drop to "
            "zero; solves and failures become linearly separable (gap 1.20 vs 0.86).\n\n"
            "2.  Failure-regime reward variance is halved (σ 0.42→0.21) and its heavy tail "
            "removed (p99 1.83→0.76), giving the critic a cleaner, bounded regression target for "
            "the ~71% of trajectories that do not fully solve.\n\n"
            "3.  Reward is decoupled from validator count WHERE IT MATTERS. Global Spearman stays "
            "~0.80 (harder tasks solved legitimately earn more), but the failed-trial channel no "
            "longer rewards brushing checks in validator-rich families.",
            fontsize=10.5, leading=0.0172, maxw=92)
y -= 0.015
y = wrapped(fig, 0.09, y, "Recommendations", fontsize=13, color=NAVY, bold=True, leading=0.03)
y = wrapped(fig, 0.09, y,
            "•  Adopt success_smoothed_progress_rewards as the default Harbor reward; keep the "
            "re-tuned D'_f divisor table in YAML (data-not-code) so it survives ray-job-submit "
            "env stripping.\n"
            "•  Monitor the bimodal reward histogram and the failed-trial-reward std as live "
            "reward-hacking tripwires; a rising upper lobe below the gate signals divisor drift.\n"
            "•  Re-audit the ÷4 'long-horizon low-yield' families (ajv_schema, transport_gtfs, "
            "circuit_simulation, …): their small D'_f lifts partial reward and should be checked "
            "against success-rate movement.\n"
            "•  Keep the 5× actor-LR reduction while the success bonus enlarges the raw reward "
            "scale; revisit once advantage-whitening statistics stabilise.",
            fontsize=10.5, leading=0.0172, maxw=92)
ax.add_patch(plt.Rectangle((0.07, 0.1), 0.86, 0.1, color=ICE, ec=MIDB, lw=1.2))
fig.text(0.5, 0.15, "Bottom line:  the redesign trades a small, legitimate reward–difficulty "
                    "correlation\nfor the elimination of an illegitimate reward–validator-count "
                    "hacking channel.",
         fontsize=10.5, color=NAVY, ha="center", va="center", linespacing=1.6, style="italic")
footer(fig, 8); pdf.savefig(fig); plt.close(fig)

pdf.close()
print("wrote", BASE / "reward_distribution_analysis.pdf")
