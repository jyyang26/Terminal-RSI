#!/usr/bin/env python3
"""Radar deliverables over all 46 families.

Normalization (理论计算), documented on every artifact:
  V* = 100 * ln(1+V) / ln(1+Vmax)      log scale (validator counts span ~1000x)
  R* = 100 * R / Rmax                  shared Rmax over BOTH runs -> R1,R2 comparable

Outputs:
  reward_radar_report.html   combined 3-series radar + two side-by-side per-run radars
  reward_radar_combined.pdf  A4 landscape, 3-series radar (V*, Run1 R*, Run2 R*)
  reward_radar_per_run.pdf   A4 landscape, two radars (each: V* + that run's R*)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
S = json.loads((BASE / "_reward_stats.json").read_text())

HERO, NAVY, ICE, MIDB, GREY = "#285AC7", "#0F2459", "#E8F0FE", "#93B5F5", "#5A739C"
VCOL = "#0F2459"   # validator series (deep navy, reference shape)
R1COL = "#5A739C"  # Run 1 (grey-blue)
R2COL = "#285AC7"  # Run 2 (hero blue)

SHORT = {
    "openfoam_cfd": "openfoam", "cmake_build_config": "cmake_build", "tree_sitter_grammar": "tree_sitter",
    "openssl_pki": "openssl_pki", "verilog_rtl": "verilog", "web_ui_state_machine": "web_ui_sm",
    "kicad_pcb": "kicad", "duckdb_reporting": "duckdb", "blender_scene_graph": "blender",
    "color_imaging_pipeline": "color_img", "hydrology_raster_modeling": "hydrology",
    "opa_rego_policy": "opa_rego", "sqlite_analytics": "sqlite", "power_grid_simulation": "power_grid",
    "finite_element_mechanics": "finite_elem", "z3_constraint_system": "z3_constr",
    "robotics_control": "robotics", "gdal_geospatial": "gdal_geo", "materials_crystallography": "materials",
    "seismology_waveform_processing": "seismology", "point_cloud_lidar": "pointcloud",
    "container_supply_chain": "container", "phylogenetics_tree_reconstruction": "phylo_tree",
    "structural_biology_pdb": "struct_bio", "cartography_map_rendering": "cartography",
    "music_score_engraving": "music_score", "ajv_schema": "ajv_schema", "video_broadcast_qc": "video_qc",
    "geology_well_log_analysis": "geology_log", "molecular_chemistry": "mol_chem",
    "audio_signal_processing": "audio_sig", "proteomics_mass_spectrometry": "proteomics",
    "rf_signal_processing": "rf_signal", "transport_gtfs_schedule": "gtfs",
    "microscopy_cell_analysis": "microscopy", "document_ocr_publishing": "doc_ocr",
    "astronomy_fits_photometry": "astronomy", "network_security_pcap": "net_pcap",
    "medical_imaging": "medical_img", "circuit_simulation": "circuit_sim",
    "typography_font_shaping": "typography", "climate_netcdf_analysis": "climate",
    "acoustics_room_simulation": "acoustics", "openscad_parametric_cad": "openscad",
    "samtools_bio_qc": "samtools", "nginx_routing": "nginx",
}

# order families by validator count ASCENDING -> validator shape spirals outward
fams = sorted(S["families"], key=lambda d: d["mean_validators"])
names = [SHORT.get(f["family"], f["family"][:12]) for f in fams]
full = [f["family"] for f in fams]
V = np.array([f["mean_validators"] for f in fams])
R1 = np.array([f["run1_mean"] for f in fams])
R2 = np.array([f["run2_mean"] for f in fams])
Vmax, Rmax = V.max(), max(R1.max(), R2.max())
Vstar = 100 * np.log1p(V) / np.log1p(Vmax)
R1star = 100 * R1 / Rmax
R2star = 100 * R2 / Rmax
N = len(fams)

# ============================================================= HTML
payload = {
    "names": names, "full": full,
    "V": [round(float(x), 1) for x in V], "R1": [round(float(x), 3) for x in R1],
    "R2": [round(float(x), 3) for x in R2],
    "Vstar": [round(float(x), 1) for x in Vstar], "R1star": [round(float(x), 1) for x in R1star],
    "R2star": [round(float(x), 1) for x in R2star],
    "Vmax": round(float(Vmax), 1), "Rmax": round(float(Rmax), 3),
}
DATA = json.dumps(payload)

HTML = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Reward × Validator Radar · 46 Families</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
 :root{{--hero:#285AC7;--navy:#0F2459;--ice:#E8F0FE;--mid:#93B5F5;--grey:#5A739C;--ink:#1e293b;--muted:#64748b;--line:#dbe3f3;--bg:#f6f8fd}}
 *{{box-sizing:border-box}} body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;color:var(--ink);background:var(--bg)}}
 header{{background:linear-gradient(135deg,var(--navy),var(--hero));color:#fff;padding:34px 42px 26px}}
 header .kick{{font-size:12.5px;letter-spacing:.14em;text-transform:uppercase;color:#bcd0f7;margin-bottom:7px}}
 header h1{{margin:0;font-size:26px;font-weight:800}} header p{{margin:9px 0 0;font-size:14px;color:#dbe6ff;max-width:1000px}}
 .wrap{{max-width:1280px;margin:0 auto;padding:26px 22px 60px}}
 section{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px 24px;margin:20px 0;box-shadow:0 1px 3px rgba(15,36,89,.05)}}
 h2{{font-size:19px;color:var(--navy);margin:0 0 4px;font-weight:750}} h2 .num{{color:var(--mid);font-weight:800;margin-right:9px}}
 .sub{{color:var(--muted);font-size:13.5px;margin:0 0 14px}}
 .chart{{width:100%;height:620px}} .half{{height:560px}}
 .two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
 .formula{{background:var(--ice);border-left:4px solid var(--hero);padding:12px 16px;border-radius:8px;font-size:13.5px;color:#263859;margin-top:12px}}
 .formula code{{background:#fff;padding:1px 7px;border-radius:5px;color:var(--navy);font-size:13px}}
 @media(max-width:900px){{.two{{grid-template-columns:1fr}}}}
 footer{{color:var(--muted);font-size:12.5px;text-align:center;padding:22px}}
</style></head><body>
<header>
  <div class="kick">Harbor Terminal-RL · Reward × Validator Radar</div>
  <h1>Per-Family Radar: Validator Count vs. Reward (46 families)</h1>
  <p>Each spoke is one task family (ordered by validator count). The radar overlays the family's
     <b>validator count</b> against the <b>mean reward</b> earned by each reward function during training —
     revealing whether reward tracks validator quantity across the domain mix.</p>
</header>
<div class="wrap">

<section>
  <h2><span class="num">01</span>Combined Radar — Validator vs. Run 1 vs. Run 2</h2>
  <p class="sub">Three series per spoke: validator count (V*), Run 1 reward (R1*), Run 2 reward (R2*). Hover a series to read it.</p>
  <div id="radarAll" class="chart"></div>
  <div class="formula">
    <b>Normalization (理论计算).</b> Axes are 0–100 so mixed scales are comparable:
    &nbsp;<code>V* = 100·ln(1+V)/ln(1+V<sub>max</sub>)</code> (log — validator counts span ~1000×, V<sub>max</sub>={payload['Vmax']}),
    &nbsp;<code>R* = 100·R/R<sub>max</sub></code> (shared R<sub>max</sub>={payload['Rmax']} over both runs, so R1 and R2 are directly comparable).
    Raw values appear in the tooltip.
  </div>
</section>

<section>
  <h2><span class="num">02</span>Per-Reward-Function Radars (side by side)</h2>
  <p class="sub">Each radar has two series per spoke: the family's validator count (V*) and that run's reward (R*).
     Where the reward shape hugs the validator shape, reward is coupled to validator quantity; where it detaches, it is decoupled.</p>
  <div class="two">
    <div><div id="radarR1" class="chart half"></div></div>
    <div><div id="radarR2" class="chart half"></div></div>
  </div>
</section>

<footer>Generated 2026-10-13 · 46 families · companion: reward_radar_combined.pdf, reward_radar_per_run.pdf · exact per-family numbers in reward_distribution_report.html</footer>
</div>
<script>
const D = {DATA};
const HERO="#285AC7", NAVY="#0F2459", GREY="#5A739C";
const indicators = D.names.map(n=>({{name:n, max:100}}));
function tip(raw){{
  return function(p){{
    // p.value is the normalized array for the hovered series; show raw values compactly
    const arr = (p.seriesName.indexOf('Validator')>=0)? D.V
              : (p.seriesName.indexOf('Run 1')>=0)? D.R1 : D.R2;
    let rows = D.names.map((n,i)=>`${{n}}: <b>${{arr[i]}}</b>`);
    return `<div style="max-height:320px;overflow:auto;font-size:11px;line-height:1.5">`
         + `<b style="color:${{NAVY}}">${{p.seriesName}}</b> (raw)<br/>` + rows.join('<br/>') + `</div>`;
  }};
}}
const nameStyle = {{color:'#334155', fontSize:10}};

echarts.init(document.getElementById('radarAll')).setOption({{
  tooltip:{{confine:true, formatter:tip()}},
  legend:{{data:['Validator count V*','Run 1 reward R1*','Run 2 reward R2*'],top:0,textStyle:{{color:'#334155'}}}},
  radar:{{indicator:indicators, radius:'68%', center:['50%','55%'], splitNumber:4,
    axisName:nameStyle, splitLine:{{lineStyle:{{color:'#e3e9f6'}}}},
    splitArea:{{areaStyle:{{color:['#ffffff','#f4f7fe']}}}}, axisLine:{{lineStyle:{{color:'#cdd8ee'}}}}}},
  series:[{{type:'radar', symbol:'none', data:[
     {{value:D.Vstar, name:'Validator count V*', lineStyle:{{color:NAVY,width:2.4,type:'dashed'}}, areaStyle:{{opacity:0}}, itemStyle:{{color:NAVY}}}},
     {{value:D.R1star, name:'Run 1 reward R1*', lineStyle:{{color:GREY,width:2}}, areaStyle:{{color:GREY,opacity:0.12}}, itemStyle:{{color:GREY}}}},
     {{value:D.R2star, name:'Run 2 reward R2*', lineStyle:{{color:HERO,width:2.2}}, areaStyle:{{color:HERO,opacity:0.14}}, itemStyle:{{color:HERO}}}}
  ]}}]
}});

function perRun(id, star, rcol, label){{
  echarts.init(document.getElementById(id)).setOption({{
    tooltip:{{confine:true, formatter:tip()}},
    title:{{text:label,left:'center',top:4,textStyle:{{color:NAVY,fontSize:15}}}},
    legend:{{data:['Validator count V*', label+' reward R*'],top:30,textStyle:{{color:'#334155'}}}},
    radar:{{indicator:indicators, radius:'64%', center:['50%','57%'], splitNumber:4,
      axisName:{{color:'#334155',fontSize:9}}, splitLine:{{lineStyle:{{color:'#e3e9f6'}}}},
      splitArea:{{areaStyle:{{color:['#ffffff','#f4f7fe']}}}}, axisLine:{{lineStyle:{{color:'#cdd8ee'}}}}}},
    series:[{{type:'radar', symbol:'none', data:[
      {{value:D.Vstar, name:'Validator count V*', lineStyle:{{color:NAVY,width:2.2,type:'dashed'}}, areaStyle:{{opacity:0}}, itemStyle:{{color:NAVY}}}},
      {{value:star, name:label+' reward R*', lineStyle:{{color:rcol,width:2.3}}, areaStyle:{{color:rcol,opacity:0.16}}, itemStyle:{{color:rcol}}}}
    ]}}]
  }});
}}
perRun('radarR1', D.R1star, GREY, 'Run 1');
perRun('radarR2', D.R2star, HERO, 'Run 2');
window.addEventListener('resize',()=>['radarAll','radarR1','radarR2'].forEach(id=>echarts.getInstanceByDom(document.getElementById(id))?.resize()));
</script></body></html>"""
(BASE / "reward_radar_report.html").write_text(HTML, encoding="utf-8")
print("wrote reward_radar_report.html")

# ============================================================= PDF helpers
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
A4L = (11.69, 8.27)  # landscape
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
closed = np.concatenate([angles, angles[:1]])


def draw_radar(ax, series, labelsize=8.0, rlab=110, rlabels=True, rlabelsize=7):
    """series: list of (values[N], color, label, dashed, fill_alpha, lw)."""
    for vals, col, lab, dashed, fa, lw in series:
        v = np.concatenate([vals, vals[:1]])
        ax.plot(closed, v, color=col, lw=lw, ls="--" if dashed else "-", label=lab, zorder=5)
        if fa > 0:
            ax.fill(closed, v, color=col, alpha=fa, zorder=2)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"] if rlabels else [], fontsize=rlabelsize, color="#94a3b8")
    ax.set_xticks(angles)
    ax.set_xticklabels([])  # draw rotated labels manually for readability
    ax.grid(color="#d7e0f2", lw=0.8)
    ax.spines["polar"].set_color("#cdd8ee")
    # manual rotated spoke labels
    for ang, lab in zip(angles, names):
        deg = np.degrees(ang)
        rot = 90 - deg
        if rot < -90:
            rot += 180
        elif rot > 90:
            rot -= 180
        ha = "center"
        ax.text(ang, rlab, lab, rotation=rot, rotation_mode="anchor",
                ha=ha, va="center", fontsize=labelsize, color="#334155", fontweight="bold")


NOTE = (r"Normalization:  $V^{*}=100\cdot\ln(1{+}V)/\ln(1{+}V_{max})$  (log; $V_{max}=%.0f$)"
        r"      $R^{*}=100\cdot R/R_{max}$  (shared $R_{max}=%.2f$ over both runs)"
        % (Vmax, Rmax))

RINGS = [25, 50, 75, 100]
VTICKS = [np.expm1(r / 100 * np.log1p(Vmax)) for r in RINGS]   # real verifier counts
RTICKS = [r / 100 * Rmax for r in RINGS]                        # real reward values


def annotate_dual_scale(ax, reward_color, fs=11):
    """Label the radial rings in TWO real dimensions: verifier count (navy, the dashed
    line) and reward value (reward_color, the solid line). The two scales sit on two
    well-separated radial spokes in the empty top-right sector so they never overlap
    each other or the data."""
    bb = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82)
    for r, vv, rv in zip(RINGS, VTICKS, RTICKS):
        tr = r if r < 100 else 93   # pull the outermost tick in, clear of perimeter labels
        ax.text(0.07, tr, f"{vv:.0f}", color=NAVY, fontsize=fs, ha="center", va="center",
                fontweight="bold", zorder=12, bbox=bb)
        ax.text(0.46, tr, f"{rv:.2f}", color=reward_color, fontsize=fs, ha="center", va="center",
                fontweight="bold", zorder=12, bbox=bb)


# ---------- PDF 1: combined ----------
pdf1 = PdfPages(BASE / "reward_radar_combined.pdf")
fig = plt.figure(figsize=A4L)
fig.suptitle("Per-Family Radar — Validator Count vs. Run 1 & Run 2 Reward",
             fontsize=19, color=NAVY, fontweight="bold", y=0.975)
fig.text(0.5, 0.925, "46 task families (ordered by validator count) · Qwen3-27B Harbor PPO",
         fontsize=11.5, color="#475569", ha="center")
ax = fig.add_axes([0.015, 0.11, 0.52, 0.72], polar=True)
draw_radar(ax, [
    (Vstar, VCOL, "Validator count  $V^{*}$", True, 0.0, 2.3),
    (R1star, R1COL, "Run 1 reward  $R_1^{*}$", False, 0.12, 2.0),
    (R2star, R2COL, "Run 2 reward  $R_2^{*}$", False, 0.14, 2.2),
], labelsize=8.0, rlab=107)
ax.legend(loc="upper left", bbox_to_anchor=(1.08, 1.02), fontsize=12, frameon=False)
# readable side explainer
fig.text(0.605, 0.63,
         "How to read\n", fontsize=13, color=NAVY, fontweight="bold", va="top")
fig.text(0.605, 0.585,
         "• The dashed navy outline is each\n  family's validator count.\n"
         "• Where a reward series (grey = Run 1,\n  blue = Run 2) bulges OUT toward the\n"
         "  validator outline, that family's reward\n  rides its validator count.\n"
         "• Where it stays small while the validator\n  outline is large, reward is decoupled\n  from validator quantity.\n\n"
         "Run 2's blue shape is smoother across\nspokes and detaches from the navy\n"
         "outline on the high-validator families\n(cmake, openfoam, tree_sitter), i.e. the\n"
         "success-smoothed reward no longer\ninflates with validator count.",
         fontsize=10.5, color="#1e293b", va="top", linespacing=1.5)
fig.text(0.5, 0.045, NOTE, fontsize=9.5, color="#475569", ha="center")
pdf1.savefig(fig); plt.close(fig); pdf1.close()
print("wrote reward_radar_combined.pdf")

# ---------- PDF 2: two per-run radars side by side ----------
pdf2 = PdfPages(BASE / "reward_radar_per_run.pdf")
fig = plt.figure(figsize=A4L)
fig.suptitle("Reward Dynamics", fontsize=22, color=NAVY, fontweight="bold", y=0.965)
# radars sized so 46 radial labels clear BOTH the page edges and the centre gap
axL = fig.add_axes([0.082, 0.205, 0.333, 0.471], polar=True)
axR = fig.add_axes([0.585, 0.205, 0.333, 0.471], polar=True)
draw_radar(axL, [
    (Vstar, VCOL, "Verifier count", True, 0.0, 2.2),
    (R1star, R1COL, "RPR-1 reward", False, 0.16, 2.3),
], labelsize=7.8, rlab=112, rlabels=False)
draw_radar(axR, [
    (Vstar, VCOL, "Verifier count", True, 0.0, 2.2),
    (R2star, R2COL, "RPR-2 reward", False, 0.16, 2.3),
], labelsize=7.8, rlab=112, rlabels=False)
annotate_dual_scale(axL, R1COL, fs=10.5)
annotate_dual_scale(axR, R2COL, fs=10.5)
fig.text(0.249, 0.80, "(a)  RPR-1", fontsize=17, color=NAVY, ha="center", fontweight="bold")
fig.text(0.751, 0.80, "(b)  RPR-2", fontsize=17, color=NAVY, ha="center", fontweight="bold")

# single merged legend row for the whole figure (all bold)
leg_handles = [
    Line2D([0], [0], color=VCOL, lw=2.6, ls="--", label="Verifier count (dashed)"),
    Line2D([0], [0], color=R1COL, lw=2.8, label="RPR-1 reward"),
    Line2D([0], [0], color=R2COL, lw=2.8, label="RPR-2 reward"),
]
fig.legend(handles=leg_handles, loc="lower center", bbox_to_anchor=(0.5, 0.035),
           ncol=3, frameon=False, columnspacing=2.6, handletextpad=0.7,
           prop={"weight": "bold", "size": 13.5})
pdf2.savefig(fig); plt.close(fig); pdf2.close()
print("wrote reward_radar_per_run.pdf")
