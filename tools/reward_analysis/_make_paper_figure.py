#!/usr/bin/env python3
"""Deliverable 3: academic paper main figure (16:9) -> reward_comparison_paper.{png,pdf}.

Left (a) Run 1 legacy discrete-progress reward; right (b) Run 2 success-smoothed.
Per-family reward distributions (boxplots), outcome-colored scatter, a mean-reward
trend across families ordered by validator count, the success gate line, and the
success band -- with academic callouts for the two key findings.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
df = pd.read_pickle(BASE / "_reward_rows.pkl")
stats = json.loads((BASE / "_reward_stats.json").read_text())

R1 = "Run 1 (Discrete Progress)"
R2 = "Run 2 (Success-Smoothed)"

# ---- blue academic palette ----
HERO = "#285AC7"      # hero blue (success scatter / primary)
NAVY = "#0F2459"      # deep navy (titles, trend, fitting)
ICE = "#E8F0FE"       # soft ice blue (box fill)
MIDB = "#93B5F5"      # mid blue (fill accents / success band)
GREY = "#5A739C"      # contrast grey-blue (Run-1 baseline / failures)
GRID = "#D7E0F2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 1.1,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,  # editable text in vector PDF
    "ps.fonttype": 42,
})

# families spanning the validator range, ordered by mean validator count (ascending)
FAMILIES = [
    "typography_font_shaping", "network_security_pcap", "rf_signal_processing",
    "structural_biology_pdb", "point_cloud_lidar", "power_grid_simulation",
    "sqlite_analytics", "hydrology_raster_modeling", "blender_scene_graph",
    "nginx_routing", "kicad_pcb", "web_ui_state_machine", "verilog_rtl", "openfoam_cfd",
]
SHORT = {
    "typography_font_shaping": "typography", "network_security_pcap": "net_pcap",
    "rf_signal_processing": "rf_signal", "structural_biology_pdb": "struct_bio",
    "point_cloud_lidar": "pointcloud", "power_grid_simulation": "power_grid",
    "sqlite_analytics": "sqlite", "hydrology_raster_modeling": "hydrology",
    "blender_scene_graph": "blender", "nginx_routing": "nginx", "kicad_pcb": "kicad",
    "web_ui_state_machine": "web_ui", "verilog_rtl": "verilog", "openfoam_cfd": "openfoam",
}
fammeta = {r["family"]: r for r in stats["families"]}
mv = {f: fammeta[f]["mean_validators"] for f in FAMILIES}

YMAX = 2.15


def panel(ax, run, tag, title):
    sub = df[df.run == run]
    xs = np.arange(len(FAMILIES))
    fail_box = []
    fail_means = []
    for f in FAMILIES:
        rf = sub[(sub.family == f) & (sub.terminal_success == 0)]["reward"].to_numpy(float)
        fail_box.append(rf if len(rf) else np.array([np.nan]))
        fail_means.append(np.mean(rf) if len(rf) else np.nan)

    # success band (reward range of fully-solved trajectories)
    succ = sub[sub.terminal_success == 1]["reward"].to_numpy(float)
    if len(succ):
        lo, hi = np.percentile(succ, 5), np.percentile(succ, 95)
        ax.axhspan(lo, hi, color=MIDB, alpha=0.18, zorder=0)
        ax.text(len(FAMILIES) - 0.45, (lo + hi) / 2, "solved\nband", fontsize=11.5,
                color=NAVY, ha="right", va="center", alpha=0.8, style="italic")

    # success gate reference line
    ax.axhline(1.0, color=GREY, ls="--", lw=1.6, zorder=1)
    ax.text(-0.4, 1.03, "success gate R=1.0", fontsize=11.5, color=GREY, va="bottom")

    # boxplots of FAILED-trial reward (the reward-hacking regime)
    valid_idx = [i for i, d in enumerate(fail_box) if not np.isnan(d).all()]
    bp = ax.boxplot([fail_box[i] for i in valid_idx], positions=[xs[i] for i in valid_idx],
                    widths=0.62, patch_artist=True, showfliers=False, zorder=3,
                    medianprops=dict(color=NAVY, lw=2.0),
                    whiskerprops=dict(color=GREY, lw=1.2),
                    capprops=dict(color=GREY, lw=1.2),
                    boxprops=dict(lw=1.3))
    for b in bp["boxes"]:
        b.set(facecolor=ICE, edgecolor=GREY, alpha=0.95)

    # scatter: faint failed trials + bold solved trials
    for i, f in enumerate(FAMILIES):
        s = sub[sub.family == f]
        fpts = s[s.terminal_success == 0]["reward"].to_numpy(float)
        if len(fpts):
            jit = np.random.default_rng(i * 7 + 1).uniform(-0.2, 0.2, len(fpts))
            ax.scatter(np.full(len(fpts), i) + jit, np.clip(fpts, 0, YMAX),
                       s=8, color=GREY, alpha=0.28, linewidths=0, zorder=4)
        spts = s[s.terminal_success == 1]["reward"].to_numpy(float)
        if len(spts):
            jit = np.random.default_rng(i * 7 + 2).uniform(-0.2, 0.2, len(spts))
            ax.scatter(np.full(len(spts), i) + jit, np.clip(spts, 0, YMAX),
                       s=13, color=HERO, alpha=0.5, linewidths=0, zorder=5)

    # failed-trial mean trend across families (ordered by validator count)
    fm = np.array(fail_means)
    ax.plot(xs, np.clip(fm, 0, YMAX), color=NAVY, lw=2.6, marker="o", ms=6,
            markerfacecolor="white", markeredgecolor=NAVY, markeredgewidth=1.8, zorder=6)

    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[f] for f in FAMILIES], rotation=42, ha="right", fontsize=12)
    ax.set_ylim(0, YMAX)
    ax.set_xlim(-0.8, len(FAMILIES) - 0.2)
    ax.set_title(title, fontsize=16, color=NAVY, fontweight="bold", pad=6)
    ax.set_xlabel("Task family  (ordered by mean validator count  $N_v$  →)", fontsize=14.5, color="#1e293b")
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return fail_means


fig = plt.figure(figsize=(16, 9), dpi=300)
fig.subplots_adjust(left=0.058, right=0.985, top=0.80, bottom=0.165, wspace=0.09)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)
plt.setp(ax2.get_yticklabels(), visible=False)

m1 = panel(ax1, R1, "run1", "(a)  Run 1 — Discrete Progress\n$R = K_p / D_f$")
m2 = panel(ax2, R2, "run2", "(b)  Run 2 — Success-Smoothed (Ours)\n$R = \\mathbb{1}[\\mathrm{solved}] + K_p/(K_p{+}D_f)$")
ax1.set_ylabel("Shaped reward  $R$", fontsize=15, color="#1e293b")

for a in (ax1, ax2):
    a.tick_params(axis="y", labelsize=12.5)

# ---- figure-level title ----
fig.suptitle("Decoupling RL Reward from Validator Count in Terminal-Agent Training",
             fontsize=21, color=NAVY, fontweight="bold", x=0.5, y=0.975)
fig.text(0.5, 0.930,
         "Qwen3-27B Harbor PPO · 2,560 trajectories / run · reward distribution across task families vs. validator count",
         fontsize=13, color="#475569", ha="center")

# ---- legend ----
handles = [
    plt.Rectangle((0, 0), 1, 1, fc=ICE, ec=GREY, lw=1.3, label="Failed-trial reward (box: IQR)"),
    Line2D([0], [0], marker="o", ls="", mfc=HERO, mec=HERO, ms=8, alpha=0.7, label="Solved trajectory"),
    Line2D([0], [0], color=NAVY, lw=2.6, marker="o", mfc="white", ms=6, label="Per-family failed-trial mean"),
    Line2D([0], [0], color=GREY, ls="--", lw=1.6, label="Success gate (R=1.0)"),
]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.895),
           ncol=4, frameon=False, fontsize=13, handletextpad=0.5, columnspacing=1.8)

# ---- callouts ----
c1 = stats["runs"][R1]["corr"]
c2 = stats["runs"][R2]["corr"]


def callout(ax, xy, xytext, text, fc):
    ax.annotate(
        text, xy=xy, xytext=xytext, fontsize=13, color=NAVY, va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec=NAVY, lw=1.4, alpha=0.96),
        arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.8,
                        connectionstyle="arc3,rad=-0.25"), zorder=20,
    )


# Run 1: reward inflation + hacking overlap on the high-validator families
callout(
    ax1, xy=(4.15, 1.35), xytext=(5.7, 1.72),
    text=("Run 1: failed-trial reward\n"
          f"inflates with validator count ($r_s={c1['spearman']:.2f}$).\n"
          "96 failing trials (5.3%) score $R\\geq1$,\n"
          "overlapping genuine solves\n→ prone to reward hacking"),
    fc="#FBFCFE",
)
# Run 2: clean separation / regularized failures
callout(
    ax2, xy=(8.0, 0.40), xytext=(1.9, 1.66),
    text=("Run 2: failed-trial reward stays\n"
          "flat & bounded ($R\\leq0.86$, var halved\n"
          "0.42→0.21); solves lift into a tight\n"
          "band ($R\\geq1.20$). Clean gap →\n"
          "reward decoupled from $N_v$."),
    fc="#EAF1FF",
)

out_png = BASE / "reward_comparison_paper.png"
out_pdf = BASE / "reward_comparison_paper.pdf"
# No bbox_inches='tight': preserve the exact 16:9 (figsize 16x9) aspect ratio.
fig.savefig(out_png, dpi=320, facecolor="white")
fig.savefig(out_pdf, facecolor="white")
print("wrote", out_png)
print("wrote", out_pdf)
